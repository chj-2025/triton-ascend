//===- SimtAnchorAnalysis.cpp - Materializable SIMT anchors --------------===//

#include "AscendModel/Analysis/SimtAnchorAnalysis.h"
#include "AscendModel/CostModelTrace.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <numeric>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static int64_t getStaticNumElements(RankedTensorType type) {
  if (!type || !type.hasStaticShape())
    return 0;
  return std::accumulate(type.getShape().begin(), type.getShape().end(),
                         int64_t{1}, std::multiplies<int64_t>());
}

static std::string getScalarTypeName(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (auto integer = dyn_cast<IntegerType>(type))
    return ("i" + std::to_string(integer.getWidth()));
  if (type.isIndex())
    return "index";
  return "unknown";
}

struct PlainCumsumLegality {
  int64_t axisExtent;
  std::string elementType;
};

static std::optional<PlainCumsumLegality>
analyzePlainOneDimensionalCumsum(Operation *op) {
  if (!op || op->getName().getStringRef() != "tt.scan" ||
      op->getNumOperands() != 1 || op->getNumResults() != 1)
    return std::nullopt;

  auto axis = op->getAttrOfType<IntegerAttr>("axis");
  auto sourceType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!axis || !sourceType || !sourceType.hasStaticShape())
    return std::nullopt;

  int64_t axisValue = axis.getInt();
  if (axisValue < 0 || axisValue >= sourceType.getRank())
    return std::nullopt;
  for (auto [index, extent] : llvm::enumerate(sourceType.getShape()))
    if (static_cast<int64_t>(index) != axisValue && extent != 1)
      return std::nullopt;

  int64_t realCombineOps = 0;
  bool isAdd = false;
  if (op->getNumRegions() != 1 || op->getRegion(0).empty())
    return std::nullopt;
  Block &body = op->getRegion(0).front();
  Operation *terminator = body.empty() ? nullptr : &body.back();
  for (Operation &nested : body) {
    if (&nested == terminator)
      continue;
    llvm::StringRef name = nested.getName().getStringRef();
    if (name == "arith.extf" || name == "arith.truncf" ||
        name == "arith.bitcast")
      continue;
    ++realCombineOps;
    isAdd = name == "arith.addf" || name == "arith.addi";
  }
  if (realCombineOps != 1 || !isAdd || !terminator ||
      terminator->getName().getStringRef() != "tt.scan.return")
    return std::nullopt;

  return PlainCumsumLegality{sourceType.getShape()[axisValue],
                             getScalarTypeName(sourceType.getElementType())};
}

static bool hasTensorPointerOperand(Operation *op) {
  if (!op || op->getNumOperands() == 0)
    return false;
  auto type = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  return type && type.hasStaticShape() && type.getRank() <= 5;
}

static bool pointerDependsOnLoadedIndex(Operation *memoryOp) {
  if (!memoryOp || memoryOp->getNumOperands() == 0)
    return false;
  llvm::SmallVector<Value> worklist{memoryOp->getOperand(0)};
  llvm::DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;

    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Operation *parent = argument.getOwner()->getParentOp();
      unsigned number = argument.getArgNumber();
      if (parent && parent->getName().getStringRef() == "scf.for" &&
          number > 0 && number + 2 < parent->getNumOperands()) {
        worklist.push_back(parent->getOperand(number + 2));
        Operation *yield = argument.getOwner()->getTerminator();
        if (yield && number - 1 < yield->getNumOperands())
          worklist.push_back(yield->getOperand(number - 1));
      }
      continue;
    }

    Operation *producer = value.getDefiningOp();
    if (!producer)
      continue;
    llvm::StringRef name = producer->getName().getStringRef();
    if (name == "tt.load" || name == "tt.gather")
      return true;
    llvm::append_range(worklist, producer->getOperands());
  }
  return false;
}

static Value findPointerOffset(Value pointer) {
  llvm::SmallVector<Value> worklist{pointer};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    Operation *producer = value.getDefiningOp();
    if (!producer)
      continue;
    if (producer->getName().getStringRef() == "tt.addptr" &&
        producer->getNumOperands() > 1)
      return producer->getOperand(1);
    llvm::append_range(worklist, producer->getOperands());
  }
  return {};
}

static std::string getAtomicOperation(Operation *op) {
  if (op->getName().getStringRef() == "tt.atomic_cas")
    return "cas";
  Attribute attribute = op->getAttr("atomic_rmw_op");
  if (!attribute)
    return "unknown";
  if (auto integer = dyn_cast<IntegerAttr>(attribute)) {
    switch (integer.getInt()) {
    case 1:
      return "and";
    case 2:
      return "or";
    case 3:
      return "xor";
    case 4:
      return "add";
    case 5:
      return "fadd";
    case 6:
      return "max";
    case 7:
      return "min";
    case 8:
      return "umax";
    case 9:
      return "umin";
    case 10:
      return "xchg";
    default:
      return "unknown";
    }
  }
  std::string text;
  llvm::raw_string_ostream os(text);
  attribute.print(os);
  os.flush();
  for (char &character : text)
    character = llvm::toLower(character);
  for (llvm::StringRef name : {"fadd", "umax", "umin", "xchg", "exch", "and",
                               "xor", "add", "max", "min", "or"})
    if (llvm::StringRef(text).contains(name))
      return name == "exch" ? "xchg" : name.str();
  return "unknown";
}

static bool isSupportedCumsumType(llvm::StringRef type) {
  return type == "i8" || type == "i16" || type == "i32" || type == "f16" ||
         type == "bf16" || type == "f32";
}

static bool isSupportedAtomicType(llvm::StringRef type,
                                  llvm::StringRef operation) {
  if (operation == "and" || operation == "or" || operation == "xor")
    return type == "i32" || type == "i64";
  if (operation == "xchg" || operation == "cas")
    return type == "i32" || type == "i64" || type == "f32";
  if (operation == "add" || operation == "fadd" || operation == "max" ||
      operation == "min" || operation == "umax" || operation == "umin")
    return type == "i32" || type == "i64" || type == "f16" || type == "bf16" ||
           type == "f32";
  return false;
}

static bool isTriangularSolveLoop(Operation *op) {
  if (!op || op->getName().getStringRef() != "scf.for" ||
      op->getNumRegions() != 1 || op->getRegion(0).empty())
    return false;

  bool hasVectorLoad = false;
  bool hasAxisZeroReduce = false;
  bool hasMaskedUpdate = false;
  Block &body = op->getRegion(0).front();
  body.walk([&](Operation *nested) {
    llvm::StringRef name = nested->getName().getStringRef();
    if (name == "tt.load" && nested->getNumResults() == 1) {
      if (auto type =
              dyn_cast<RankedTensorType>(nested->getResult(0).getType()))
        hasVectorLoad |= type.hasStaticShape() && type.getRank() == 1 &&
                         type.getShape()[0] == 16;
    } else if (name == "tt.reduce") {
      auto axis = nested->getAttrOfType<IntegerAttr>("axis");
      hasAxisZeroReduce |= axis && axis.getInt() == 0;
    } else if (name == "arith.select") {
      hasMaskedUpdate = true;
    }
  });
  if (!hasVectorLoad || !hasAxisZeroReduce || !hasMaskedUpdate)
    return false;

  // The production solve_tril kernel has one block-update loop.  Its
  // iter_args carry the 16x16 triangular state; requiring that fixed state
  // shape keeps this single-loop form distinct from a generic vector reduce.
  bool hasTriangularState = false;
  for (BlockArgument argument : body.getArguments()) {
    if (auto type = dyn_cast<RankedTensorType>(argument.getType()))
      hasTriangularState |= type.hasStaticShape() && type.getRank() == 2 &&
                            type.getShape()[0] == 16 &&
                            type.getShape()[1] == 16;
  }

  // Multi-loop lowering variants use two or more sibling block-update loops;
  // retain that form for kernels whose state shape is not explicit in the
  // loop arguments.
  Block *parentBlock = op->getBlock();
  if (!parentBlock)
    return false;
  unsigned siblingMatches = 0;
  for (Operation &sibling : *parentBlock) {
    if (&sibling == op || sibling.getName().getStringRef() != "scf.for")
      continue;
    bool siblingLoad = false;
    bool siblingReduce = false;
    bool siblingSelect = false;
    sibling.walk([&](Operation *nested) {
      llvm::StringRef name = nested->getName().getStringRef();
      if (name == "tt.load" && nested->getNumResults() == 1)
        if (auto type =
                dyn_cast<RankedTensorType>(nested->getResult(0).getType()))
          siblingLoad |= type.hasStaticShape() && type.getRank() == 1 &&
                         type.getShape()[0] == 16;
      siblingReduce |= name == "tt.reduce";
      siblingSelect |= name == "arith.select";
    });
    siblingMatches += siblingLoad && siblingReduce && siblingSelect;
  }
  return hasTriangularState || siblingMatches >= 1;
}

static TriangularSolveFacts analyzeTriangularSolveFacts(Operation *anchor) {
  TriangularSolveFacts facts;
  if (!anchor || !anchor->getBlock())
    return facts;

  auto recordStateType = [&](Type type) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || !tensor.hasStaticShape() || tensor.getRank() != 2)
      return;
    if (facts.blockRows == 0 && facts.blockColumns == 0) {
      facts.blockRows = tensor.getShape()[0];
      facts.blockColumns = tensor.getShape()[1];
      facts.accumulatorType = getScalarTypeName(tensor.getElementType());
    }
  };
  if (anchor->getNumRegions() == 1 && !anchor->getRegion(0).empty())
    for (BlockArgument argument : anchor->getRegion(0).front().getArguments())
      recordStateType(argument.getType());
  for (Type type : anchor->getResultTypes())
    recordStateType(type);

  Operation *lastRecurrence = nullptr;
  int64_t recurrenceLoopOps = 0;
  for (Operation &candidate : *anchor->getBlock()) {
    if (!isTriangularSolveLoop(&candidate))
      continue;
    ++recurrenceLoopOps;
    lastRecurrence = &candidate;
  }

  // The recognized 16x16 recurrence updates rows [2, 16), hence 14 modeled
  // trips per full block.  Keep the start row explicit so structural matching
  // and reports do not hide that structural assumption in a magic trip count.
  facts.recurrenceStartRow = 2;
  // recurrenceLoopCount is consumed by StageCostModel as N_iter.  It must be
  // the dynamic body execution count, not the number of sibling scf.for ops.
  // A full 16x16 block updates rows [2, 16), hence 14 iterations.
  facts.recurrenceLoopCount =
      facts.blockRows > facts.recurrenceStartRow
          ? recurrenceLoopOps * (facts.blockRows - facts.recurrenceStartRow)
          : recurrenceLoopOps;

  for (Operation *candidate = lastRecurrence ? lastRecurrence->getNextNode()
                                             : nullptr;
       candidate; candidate = candidate->getNextNode()) {
    candidate->walk([&](Operation *nested) {
      facts.denseDotTailOps +=
          nested->getName().getStringRef() == "tt.dot" ? 1 : 0;
    });
  }
  facts.requiresCubeTailPartition = facts.denseDotTailOps > 0;
  return facts;
}

static bool isMovableTriangularTensorSetup(Operation *op) {
  if (!op || op->getNumResults() == 0 || op->getNumRegions() != 0)
    return false;
  if (!llvm::all_of(op->getResultTypes(),
                    [](Type type) { return isa<RankedTensorType>(type); }))
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  // Triton hoists both integer offset splats and floating-point zero splats
  // out of a hand-written scope. Keep all arith.constant ops as captures so
  // the automatically materialized scope has the same boundary contract.
  return name == "tt.make_range" || name == "tt.expand_dims" ||
         name == "tt.broadcast" || name == "tt.splat" || name == "arith.cmpi" ||
         name == "arith.cmpf" || name == "arith.andi" || name == "arith.ori" ||
         name == "arith.xori" || name == "arith.addi" || name == "arith.subi" ||
         name == "arith.muli" || name == "arith.index_cast";
}

static Operation *getTopLevelOwnerInBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

/// Return the exact TTIR operations represented by the hand-written
/// triangular-solve scope. The four 16x16 input loads stay outside. Pure
/// ranked-tensor mask setup is moved after those loads, then the sibling
/// recurrences and final diagonal updates execute inside the scope. The dense
/// tt.dot tail remains outside. Keeping this in analysis makes the plan
/// authoritative; the materializer must not rediscover a different set.
static llvm::SmallVector<Operation *, 0>
collectTriangularSolveScopeOperations(Operation *anchor,
                                      Operation *&insertionPoint) {
  llvm::SmallVector<Operation *, 0> result;
  insertionPoint = nullptr;
  if (!isTriangularSolveLoop(anchor) || !anchor->getBlock())
    return result;

  Block *block = anchor->getBlock();
  Operation *firstLoop = nullptr;
  Operation *lastLoop = nullptr;
  for (Operation &nested : *block) {
    if (!isTriangularSolveLoop(&nested))
      continue;
    if (!firstLoop)
      firstLoop = &nested;
    lastLoop = &nested;
  }
  if (!firstLoop || !lastLoop)
    return result;

  Operation *lastInputLoad = nullptr;
  Operation *cursor = firstLoop->getPrevNode();
  while (cursor && cursor->getName().getStringRef() != "tt.load")
    cursor = cursor->getPrevNode();
  if (cursor)
    lastInputLoad = cursor;
  Operation *start = lastInputLoad ? lastInputLoad->getNextNode() : firstLoop;
  insertionPoint = start;

  Operation *end = lastLoop;
  cursor = lastLoop->getNextNode();
  while (cursor) {
    llvm::StringRef name = cursor->getName().getStringRef();
    if (name != "arith.uitofp" && name != "arith.addf" &&
        name != "arith.select")
      break;
    end = cursor;
    cursor = cursor->getNextNode();
  }

  llvm::SmallVector<Operation *, 0> recurrenceRange;
  llvm::DenseSet<Operation *> selected;
  for (Operation *op = start; op; op = op->getNextNode()) {
    recurrenceRange.push_back(op);
    selected.insert(op);
    if (op == end)
      break;
  }
  if (recurrenceRange.empty() || recurrenceRange.back() != end)
    return {};

  // In the unscope source, o_i/m_A/m_I are constructed before the initial
  // block loads. The hand-written scope constructs them after the loads.
  // Pull in only a side-effect-free ranked-tensor backward slice whose uses
  // are entirely inside the selected scope; scalar/pointer setup and all
  // memory operations remain captures outside.
  llvm::DenseSet<Operation *> setupCandidates;
  llvm::SmallVector<Value, 16> worklist;
  for (Operation *op : recurrenceRange) {
    llvm::append_range(worklist, op->getOperands());
    op->walk([&](Operation *nested) {
      if (nested != op)
        llvm::append_range(worklist, nested->getOperands());
    });
  }
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    Operation *definingOp = value.getDefiningOp();
    if (!definingOp || definingOp->getBlock() != block ||
        definingOp == insertionPoint || !lastInputLoad ||
        !definingOp->isBeforeInBlock(lastInputLoad) ||
        !isMovableTriangularTensorSetup(definingOp) ||
        !setupCandidates.insert(definingOp).second)
      continue;
    llvm::append_range(worklist, definingOp->getOperands());
  }

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation *candidate : llvm::make_early_inc_range(setupCandidates)) {
      bool hasOutsideUse =
          llvm::any_of(candidate->getResults(), [&](Value resultValue) {
            return llvm::any_of(resultValue.getUses(), [&](OpOperand &use) {
              Operation *owner = getTopLevelOwnerInBlock(use.getOwner(), block);
              return !selected.contains(owner) &&
                     !setupCandidates.contains(owner);
            });
          });
      if (!hasOutsideUse)
        continue;
      setupCandidates.erase(candidate);
      changed = true;
    }
  }

  for (Operation &op : *block)
    if (setupCandidates.contains(&op))
      result.push_back(&op);
  llvm::append_range(result, recurrenceRange);
  return result;
}

static std::optional<SimtAnchorDescriptor> analyzeAnchor(Operation *op,
                                                         bool compileOn91095) {
  if (!op)
    return std::nullopt;
  SimtAnchorDescriptor descriptor;
  descriptor.operation = op;
  llvm::StringRef name = op->getName().getStringRef();

  if (name == "tt.gather") {
    descriptor.kind = SimtAnchorKind::DirectGather;
  } else if (name == "tt.histogram") {
    descriptor.kind = SimtAnchorKind::Histogram;
    auto input = op->getNumOperands() > 0
                     ? dyn_cast<RankedTensorType>(op->getOperand(0).getType())
                     : RankedTensorType();
    auto result = op->getNumResults() > 0
                      ? dyn_cast<RankedTensorType>(op->getResult(0).getType())
                      : RankedTensorType();
    const int64_t inputElements = getStaticNumElements(input);
    const int64_t numBins =
        result && result.hasStaticShape() && result.getRank() == 1
            ? result.getShape()[0]
            : 0;
    const std::string inputType = input ? getScalarTypeName(input) : "unknown";
    const std::string resultType =
        result ? getScalarTypeName(result) : "unknown";
    descriptor.lowerability.allSimd = false;
    descriptor.lowerability.allSimtOnly = false;
    bool supported = input && result && input.hasStaticShape() &&
                     result.hasStaticShape() && input.getRank() == 1 &&
                     result.getRank() == 1 &&
                     (inputType == "i8" || inputType == "i16" ||
                      inputType == "i32" || inputType == "i64") &&
                     resultType == "i32" && inputElements > 0 && numBins > 0;
    if (!supported)
      descriptor.lowerability.mixed = false;
  } else if (name == "tt.scan") {
    auto facts = analyzePlainOneDimensionalCumsum(op);
    if (!facts)
      return std::nullopt;
    descriptor.kind = SimtAnchorKind::PlainOneDimensionalCumsum;
    descriptor.lowerability.allSimd = false;
    if (facts->axisExtent <= 0 || !isSupportedCumsumType(facts->elementType))
      descriptor.lowerability.mixed = false;
  } else if (name == "tt.atomic_rmw" || name == "tt.atomic_cas") {
    descriptor.kind = SimtAnchorKind::TensorAtomic;
    auto result = op->getNumResults() > 0
                      ? dyn_cast<RankedTensorType>(op->getResult(0).getType())
                      : RankedTensorType();
    const int64_t updateElements = getStaticNumElements(result);
    const std::string valueType =
        result ? getScalarTypeName(result) : "unknown";
    const std::string operation = getAtomicOperation(op);
    std::string offsetType = "unknown";
    if (op->getNumOperands() > 0) {
      Value offset = findPointerOffset(op->getOperand(0));
      if (offset)
        offsetType = getScalarTypeName(offset.getType());
    }
    bool supported = result && result.hasStaticShape() && updateElements > 0 &&
                     isSupportedAtomicType(valueType, operation) &&
                     (offsetType == "i32" || offsetType == "i64");
    const bool resultUsed =
        op->getNumResults() > 0 && !op->getResult(0).use_empty();
    if (resultUsed && (valueType == "f16" || valueType == "bf16"))
      supported = false;
    if (!supported)
      descriptor.lowerability.mixed = false;
  } else if (isTriangularSolveLoop(op)) {
    descriptor.kind = SimtAnchorKind::TriangularSolveLoop;
    descriptor.triangularSolve = analyzeTriangularSolveFacts(op);
  } else if (isLoadedIndexDependentMemoryOp(op)) {
    descriptor.kind = SimtAnchorKind::LoadedIndexDependentMemory;
  } else {
    return std::nullopt;
  }

  descriptor.materializable = compileOn91095 && descriptor.lowerability.mixed;
  return descriptor;
}

static bool operationTreeContains(Operation *root, llvm::StringRef name) {
  if (!root)
    return false;
  if (root->getName().getStringRef() == name)
    return true;
  bool found = false;
  root->walk([&](Operation *nested) {
    found |= nested->getName().getStringRef() == name;
  });
  return found;
}

static bool isPointerLikeType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << type;
  stream.flush();
  return llvm::StringRef(text).contains("!tt.ptr");
}

/// Pure shape/constant/index/pointer construction carries no compute
/// substance; a generic SIMT scope around it would only pay switching cost.
/// The list mirrors isMovableTriangularTensorSetup's notion of tensor setup
/// plus the pointer-advance ops from isAddressOnlyLoopValue.
static bool isPureTensorSetupOp(Operation *op) {
  const llvm::StringRef name = op->getName().getStringRef();
  return name == "arith.constant" || name == "tt.make_range" ||
         name == "tt.expand_dims" || name == "tt.broadcast" ||
         name == "tt.splat" || name == "tt.addptr" ||
         name == "tt.advance" || name == "arith.cmpi" ||
         name == "arith.cmpf" || name == "arith.andi" ||
         name == "arith.ori" || name == "arith.xori" ||
         name == "arith.addi" || name == "arith.subi" ||
         name == "arith.muli" || name == "arith.index_cast";
}

/// A run of roots only anchors a generic compute region when it contains at
/// least one tensor-valued op that is real compute (not pure setup).  This
/// keeps e.g. scalar induction loops and constant/mask construction from
/// fabricating pointless scope evidence.
static bool runHasTensorComputeSubstance(llvm::ArrayRef<Operation *> runRoots) {
  for (Operation *root : runRoots) {
    if (!root)
      continue;
    bool substance = false;
    root->walk([&](Operation *nested) {
      if (substance || isPureTensorSetupOp(nested))
        return;
      for (Value result : nested->getResults())
        if (isa<RankedTensorType>(result.getType())) {
          substance = true;
          return;
        }
    });
    if (substance)
      return true;
  }
  return false;
}

/// Mirror the exact SSA contract that wrapAnchorRange (materializer) and
/// deriveLocalSimtScopeTraffic (stage analysis) enforce on a planned scope:
/// all roots in one block, no terminator/isolated-from-above/scope ops, and
/// no pointer-like value returned by the span.  A single-root span returns
/// every result; a multi-root span returns only values with an outside user.
static bool spanRootsCanBeLocalScope(llvm::ArrayRef<Operation *> spanRoots) {
  if (spanRoots.empty())
    return false;
  Block *block = spanRoots.front()->getBlock();
  if (!block)
    return false;
  for (Operation *root : spanRoots) {
    if (!root || root->getBlock() != block ||
        root->hasTrait<OpTrait::IsTerminator>() ||
        root->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return false;
    const llvm::StringRef name = root->getName().getStringRef();
    if (name == "scope.scope" || name == "scope.return")
      return false;
  }

  llvm::DenseSet<Operation *> inside;
  for (Operation *root : spanRoots) {
    inside.insert(root);
    root->walk([&](Operation *nested) { inside.insert(nested); });
  }
  const bool isRange = spanRoots.size() > 1;
  for (Operation *root : spanRoots) {
    for (Value result : root->getResults()) {
      const bool escapes = llvm::any_of(result.getUsers(), [&](Operation *user) {
        return !inside.contains(user);
      });
      if (!isRange || escapes)
        if (isPointerLikeType(result.getType()))
          return false;
    }
  }
  return true;
}

/// Synthesize one GenericComputeRegion anchor per function for kernels with
/// no materializable specialized anchor.  The anchor is the maximal span of
/// the entry block between the first and the last "transform run": maximal
/// contiguous runs of top-level roots whose operation trees contain no
/// tt.load/tt.store and no specialized anchor op, and that carry tensor
/// compute substance.  Roots between two anchored runs (loads, stores,
/// pointer setup) join the span because the phase machinery and
/// mergeSimtStageAnchors turn all anchor roots of one stage into a single
/// lexical scope interval.  The span is validated up front so the
/// materialized scope is exactly the validated one.
static void synthesizeGenericComputeRegionAnchors(ModuleOp module,
                                                  SimtAnchorPlan &plan) {
  COSTMODEL_TRACE_DEBUG("synthesizeGenericComputeRegionAnchors");
  llvm::DenseSet<Operation *> specializedOps;
  for (const SimtAnchorDescriptor &anchor : plan.anchors) {
    if (anchor.operation)
      specializedOps.insert(anchor.operation);
    for (Operation *op : anchor.scopeOperations)
      if (op)
        specializedOps.insert(op);
  }

  for (Operation &function : module.getBody()->getOperations()) {
    const llvm::StringRef functionName = function.getName().getStringRef();
    if (functionName != "tt.func" && functionName != "func.func")
      continue;
    if (function.getNumRegions() == 0 || function.getRegion(0).empty())
      continue;
    Block &entry = function.getRegion(0).front();

    llvm::SmallVector<Operation *> roots;
    llvm::SmallVector<bool> isTransformRoot;
    for (Operation &op : entry.getOperations()) {
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          op.hasAttr("ta.auto_blockify_v1.schedule"))
        continue;
      roots.push_back(&op);
      isTransformRoot.push_back(
          !specializedOps.contains(&op) &&
          !operationTreeContains(&op, "tt.load") &&
          !operationTreeContains(&op, "tt.store"));
    }
    costModelDebug() << "generic compute-region synthesis: function "
                     << "(function-like op) roots=" << roots.size()
                     << " transformRoots="
                     << llvm::count(isTransformRoot, true) << "\n";

    // Maximal contiguous transform-root runs; only runs with tensor compute
    // substance anchor the span.
    std::optional<size_t> spanBegin;
    std::optional<size_t> spanEnd;
    size_t runBegin = 0;
    bool inRun = false;
    for (size_t index = 0; index <= roots.size(); ++index) {
      const bool transform =
          index < roots.size() && isTransformRoot[index];
      if (transform && !inRun) {
        inRun = true;
        runBegin = index;
      } else if (!transform && inRun) {
        inRun = false;
        llvm::ArrayRef<Operation *> run(roots.data() + runBegin,
                                        index - runBegin);
        if (runHasTensorComputeSubstance(run)) {
          if (!spanBegin)
            spanBegin = runBegin;
          spanEnd = index - 1;
          costModelDebug() << "substance run roots=[" << runBegin << ","
                           << (index - 1) << "] ops=" << run.size() << "\n";
        }
      }
    }
    if (!spanBegin || !spanEnd)
      continue;

    llvm::SmallVector<Operation *, 16> spanRoots(roots.begin() + *spanBegin,
                                                 roots.begin() + *spanEnd + 1);
    costModelDebug() << "candidate span roots=[" << *spanBegin << ","
                     << *spanEnd << "] ops=" << spanRoots.size() << "\n";
    if (!spanRootsCanBeLocalScope(spanRoots)) {
      costModelDebug() << "rejected: span cannot be wrapped as a local scope "
                          "(pointer escape or illegal op)\n";
      continue;
    }

    SimtAnchorDescriptor descriptor;
    descriptor.operation = spanRoots.front();
    descriptor.kind = SimtAnchorKind::GenericComputeRegion;
    descriptor.scopeOperations =
        llvm::SmallVector<Operation *, 0>(spanRoots.begin(), spanRoots.end());
    descriptor.scopeInsertionPoint = spanRoots.front();
    descriptor.lowerability.allSimd = true;
    descriptor.lowerability.allSimtOnly = true;
    descriptor.lowerability.mixed = true;
    descriptor.materializable = true;
    plan.anchors.push_back(std::move(descriptor));
    costModelDebug() << "synthesized GenericComputeRegion anchor: ops="
                     << spanRoots.size() << "\n";
  }
}

} // namespace

llvm::StringRef mlir::ascend::stringifySimtAnchorKind(SimtAnchorKind kind) {
  switch (kind) {
  case SimtAnchorKind::DirectGather:
    return "direct_gather";
  case SimtAnchorKind::LoadedIndexDependentMemory:
    return "loaded_index_dependent_memory";
  case SimtAnchorKind::Histogram:
    return "histogram";
  case SimtAnchorKind::PlainOneDimensionalCumsum:
    return "plain_1d_cumsum";
  case SimtAnchorKind::TensorAtomic:
    return "tensor_atomic";
  case SimtAnchorKind::TriangularSolveLoop:
    return "triangular_solve_loop";
  case SimtAnchorKind::GenericComputeRegion:
    return "generic_compute_region";
  }
  llvm_unreachable("unknown SIMT anchor kind");
}

llvm::SmallVector<Operation *> SimtAnchorPlan::materializableRoots() const {
  llvm::SmallVector<Operation *> result;
  result.reserve(anchors.size());
  for (const SimtAnchorDescriptor &anchor : anchors)
    if (anchor.materializable)
      result.push_back(anchor.operation);
  return result;
}

std::optional<SimtAnchorDescriptor>
mlir::ascend::mergeSimtStageAnchors(const SimtAnchorPlan &plan,
                                    llvm::ArrayRef<unsigned> anchorIndices) {
  llvm::SmallVector<const SimtAnchorDescriptor *> anchors;
  llvm::DenseSet<unsigned> seen;
  for (unsigned index : anchorIndices) {
    if (index >= plan.anchors.size() || !seen.insert(index).second)
      continue;
    const SimtAnchorDescriptor &anchor = plan.anchors[index];
    if (!anchor.materializable || !anchor.operation)
      return std::nullopt;
    anchors.push_back(&anchor);
  }
  if (anchors.empty())
    return std::nullopt;
  if (anchors.size() == 1)
    return *anchors.front();

  Block *block = nullptr;
  llvm::DenseSet<Operation *> selected;
  for (const SimtAnchorDescriptor *anchor : anchors) {
    llvm::ArrayRef<Operation *> operations = anchor->scopeOperations;
    if (operations.empty())
      operations = llvm::ArrayRef(anchor->operation);
    for (Operation *operation : operations) {
      if (!operation || !operation->getBlock() ||
          (block && block != operation->getBlock()))
        return std::nullopt;
      block = operation->getBlock();
      selected.insert(operation);
    }
  }
  if (!block)
    return std::nullopt;

  Operation *first = nullptr;
  Operation *last = nullptr;
  for (Operation &operation : *block) {
    if (!selected.contains(&operation))
      continue;
    if (!first)
      first = &operation;
    last = &operation;
  }
  if (!first || !last)
    return std::nullopt;

  SimtAnchorDescriptor merged = *anchors.front();
  merged.scopeOperations.clear();
  bool inRange = false;
  for (Operation &operation : *block) {
    if (&operation == first)
      inRange = true;
    if (inRange) {
      if (operation.hasTrait<OpTrait::IsTerminator>() ||
          operation.hasTrait<OpTrait::IsIsolatedFromAbove>() ||
          operation.getName().getStringRef() == "scope.scope" ||
          operation.getName().getStringRef() == "scope.return")
        return std::nullopt;
      merged.scopeOperations.push_back(&operation);
    }
    if (&operation == last)
      break;
  }
  merged.scopeInsertionPoint = first;
  return merged;
}

bool mlir::ascend::isLoadedIndexDependentMemoryOp(Operation *op) {
  if (!op)
    return false;
  llvm::StringRef name = op->getName().getStringRef();
  return (name == "tt.load" || name == "tt.store") &&
         hasTensorPointerOperand(op) && pointerDependsOnLoadedIndex(op);
}

SimtAnchorPlan mlir::ascend::buildMixedSimtAnchorPlan(ModuleOp module,
                                                      bool compileOn91095) {
  COSTMODEL_TRACE("buildMixedSimtAnchorPlan");
  SimtAnchorPlan plan;
  llvm::DenseSet<Operation *> operationsInPlannedScope;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (operationsInPlannedScope.contains(op))
      return WalkResult::skip();
    std::optional<SimtAnchorDescriptor> descriptor =
        analyzeAnchor(op, compileOn91095);
    if (!descriptor)
      return WalkResult::advance();

    if (descriptor->kind == SimtAnchorKind::TriangularSolveLoop) {
      descriptor->scopeOperations = collectTriangularSolveScopeOperations(
          op, descriptor->scopeInsertionPoint);
      if (descriptor->scopeOperations.empty()) {
        descriptor->materializable = false;
        descriptor->lowerability.mixed = false;
      }
    } else {
      descriptor->scopeOperations.push_back(op);
      descriptor->scopeInsertionPoint = op;
    }
    for (Operation *scopeOperation : descriptor->scopeOperations)
      operationsInPlannedScope.insert(scopeOperation);
    plan.anchors.push_back(std::move(*descriptor));
    return WalkResult::skip();
  });

  bool anyMixed = false;
  bool mixedBlocked = false;
  for (const SimtAnchorDescriptor &anchor : plan.anchors) {
    plan.kernelLowerability.allSimd &= anchor.lowerability.allSimd;
    plan.kernelLowerability.allSimtOnly &= anchor.lowerability.allSimtOnly;
    anyMixed |= anchor.lowerability.mixed;
    mixedBlocked |= !anchor.lowerability.mixed && !anchor.lowerability.allSimd;
  }

  // Mixed-route blind-spot coverage: when no specialized materializable
  // anchor exists (and no specialized anchor blocks the mixed route), any
  // kernel with an ordinary compute region would previously fall to
  // backend_default for the mixed candidate -- even when that region is
  // faster on the scalar units.  Synthesize one GenericComputeRegion anchor
  // per function from validated transform-root spans so the route model can
  // score a local SIMT scope and pay its switching cost.
  const bool hasMaterializableSpecialized = llvm::any_of(
      plan.anchors,
      [](const SimtAnchorDescriptor &anchor) { return anchor.materializable; });
  if (compileOn91095 && !hasMaterializableSpecialized && !mixedBlocked) {
    costModelLog() << "no materializable specialized anchor; trying "
                      "GenericComputeRegion synthesis\n";
    synthesizeGenericComputeRegionAnchors(module, plan);
    for (const SimtAnchorDescriptor &anchor : plan.anchors) {
      if (anchor.kind != SimtAnchorKind::GenericComputeRegion)
        continue;
      // Generic regions are ordinary compute: they lower on both paths and
      // keep the mixed route legal.
      anyMixed |= anchor.lowerability.mixed;
    }
  }
  plan.kernelLowerability.mixed = anyMixed && !mixedBlocked;
  // Print every matched anchor operation so the anchor <-> op correspondence
  // is directly visible (IR-dump style after the "op:" marker).
  for (size_t i = 0; i < plan.anchors.size(); ++i) {
    const SimtAnchorDescriptor &anchor = plan.anchors[i];
    costModelLog() << "anchor[" << i << "] kind="
                   << stringifySimtAnchorKind(anchor.kind)
                   << " materializable="
                   << (anchor.materializable ? "true" : "false") << "\n";
    costModelLog() << "  op: ";
    anchor.operation->print(llvm::errs());
    llvm::errs() << "\n";
  }
  costModelLog() << "output: anchors=" << plan.anchors.size()
                 << " kernelLowerability allSimd="
                 << (plan.kernelLowerability.allSimd ? "true" : "false")
                 << " allSimtOnly="
                 << (plan.kernelLowerability.allSimtOnly ? "true" : "false")
                 << " mixed="
                 << (plan.kernelLowerability.mixed ? "true" : "false") << "\n";
  return plan;
}
