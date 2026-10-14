//===- SimtRegisterPressureAnalysis.cpp - Whole-VF live values ------------===//

#include "AscendModel/Analysis/SimtRegisterPressureAnalysis.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <system_error>

using namespace mlir;
using namespace mlir::ascend;

namespace {

// Shape-only aliases are counted once. This is not a claim that arbitrary
// transpose/convert-layout operations are free or use no temporary registers.
static Value storageRoot(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    StringRef name = op->getName().getStringRef();
    if (op->getNumOperands() != 1 || op->getNumResults() != 1 ||
        (name != "tt.expand_dims" && name != "tt.reshape" &&
         name != "tt.splat"))
      break;
    value = op->getOperand(0);
  }
  return value;
}

static std::optional<int64_t> valueUnits(Value value, int64_t threads) {
  Type type = value.getType();
  int64_t elements = 1;
  bool tensor = false;
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (!shaped.hasStaticShape())
      return std::nullopt;
    elements = shaped.getNumElements();
    type = shaped.getElementType();
    tensor = true;
  }
  int64_t bits;
  if (isa<IntegerType, FloatType>(type))
    bits = type.getIntOrFloatBitWidth();
  else if (isa<IndexType, triton::PointerType>(type))
    bits = 64;
  else
    return std::nullopt;
  // Do not assume f16/bool packing, or divide a scalar by the lane count.
  const int64_t perThread =
      tensor ? elements / threads + (elements % threads != 0) : 1;
  const int64_t words = (bits + 31) / 32;
  if (words <= 0 || perThread > std::numeric_limits<int64_t>::max() / words)
    return std::nullopt;
  return perThread * words;
}

static std::optional<int64_t> predicateUnits(Value value, int64_t threads) {
  Type type = value.getType();
  int64_t elements = 1;
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (!shaped.hasStaticShape())
      return std::nullopt;
    elements = shaped.getNumElements();
    type = shaped.getElementType();
  }
  auto integer = dyn_cast<IntegerType>(type);
  if (!integer || integer.getWidth() != 1)
    return std::nullopt;
  return elements / threads + (elements % threads != 0);
}

static bool crossesIntoLoop(Value value, Operation *user) {
  Operation *definition = value.getDefiningOp();
  for (Operation *parent = user->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto loop = dyn_cast<scf::ForOp>(parent);
    if (!loop)
      continue;
    if (!definition || !loop->isProperAncestor(definition))
      return true;
  }
  return false;
}

static bool rematerializable(Value value) {
  Operation *op = value.getDefiningOp();
  return op && (op->getName().getStringRef() == "arith.constant" ||
                op->getName().getStringRef() == "tt.make_range");
}

enum class PredicateClass { IntegerEquality, IntegerOrdered, Shape, Other };

static PredicateClass predicateClass(Operation *op) {
  auto compare = dyn_cast<arith::CmpIOp>(op);
  if (compare) {
    switch (compare.getPredicate()) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::ne:
      return PredicateClass::IntegerEquality;
    default:
      return PredicateClass::IntegerOrdered;
    }
  }
  StringRef name = op->getName().getStringRef();
  if (name == "tt.expand_dims" || name == "tt.broadcast" ||
      name == "tt.reshape" || name == "tt.splat")
    return PredicateClass::Shape;
  return PredicateClass::Other;
}

static bool isPredicateBooleanConsumer(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "arith.andi" || name == "arith.ori" || name == "arith.xori";
}

static std::optional<std::array<int64_t, 4>> dotGeometry(Operation *op) {
  if (!op || op->getName().getStringRef() != "tt.dot" ||
      op->getNumOperands() != 3 || op->getNumResults() != 1)
    return std::nullopt;
  auto a = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto b = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
  auto result = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!a || !b || !result || a.getRank() != 2 || b.getRank() != 2 ||
      result.getRank() != 2 || !a.hasStaticShape() || !b.hasStaticShape() ||
      !result.hasStaticShape() || a.getDimSize(1) != b.getDimSize(0) ||
      result.getDimSize(0) != a.getDimSize(0) ||
      result.getDimSize(1) != b.getDimSize(1) ||
      !isa<IntegerType, FloatType>(a.getElementType()) ||
      !isa<IntegerType, FloatType>(b.getElementType()) ||
      !isa<IntegerType, FloatType>(result.getElementType()))
    return std::nullopt;
  const int64_t bits = a.getElementType().getIntOrFloatBitWidth();
  if (b.getElementType().getIntOrFloatBitWidth() != bits ||
      result.getElementType().getIntOrFloatBitWidth() != bits)
    return std::nullopt;
  return std::array<int64_t, 4>{a.getDimSize(0), b.getDimSize(1),
                                a.getDimSize(1), bits};
}

static void collectNearestDotProducers(Value value,
                                       llvm::DenseSet<Operation *> &producers,
                                       llvm::DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return;
  if (definition->getName().getStringRef() == "tt.dot") {
    producers.insert(definition);
    return;
  }
  if (!isMemoryEffectFree(definition))
    return;
  for (Value operand : definition->getOperands())
    collectNearestDotProducers(operand, producers, visited);
}

static void collectDotConsumerContext(
    Operation *dot,
    const llvm::DenseMap<Operation *, int64_t> &operationOrdinal,
    SimtDotDependencySite &site) {
  llvm::SmallVector<Value> pending(dot->getResults().begin(),
                                   dot->getResults().end());
  llvm::DenseSet<Value> visitedValues;
  llvm::DenseSet<Operation *> pureConsumers;
  llvm::DenseSet<Operation *> sideEffectFrontier;
  const int64_t definitionOrdinal = operationOrdinal.lookup(dot);
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    if (!visitedValues.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (user->getName().getStringRef() == "tt.dot")
        continue;
      const int64_t span = std::max<int64_t>(0, operationOrdinal.lookup(user) -
                                                    definitionOrdinal);
      if (!isMemoryEffectFree(user)) {
        if (sideEffectFrontier.insert(user).second)
          site.maxSideEffectUseSpan = std::max(site.maxSideEffectUseSpan, span);
        continue;
      }
      if (!pureConsumers.insert(user).second)
        continue;
      site.maxNonDotPureUseSpan = std::max(site.maxNonDotPureUseSpan, span);
      pending.append(user->getResults().begin(), user->getResults().end());
    }
  }
  site.nonDotPureConsumerCount = static_cast<int64_t>(pureConsumers.size());
  site.sideEffectFrontierCount =
      static_cast<int64_t>(sideEffectFrontier.size());
}

} // namespace

llvm::Error SimtRegisterPressureAnalysis::analyze(ModuleOp module,
                                                  StagePartition &partition,
                                                  int64_t baseWarpCount) const {
  if (!module || baseWarpCount <= 0 || baseWarpCount > 64)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMT pressure requires a module and 1..64 base warps");
  llvm::DenseMap<Operation *, size_t> roots;
  for (auto entry : llvm::enumerate(partition.stages)) {
    size_t index = entry.index();
    LogicalStage &stage = entry.value();
    stage.wholeSimtRegisterPressure.reset();
    for (Operation *root : stage.operations)
      roots.try_emplace(root, index);
  }
  if (roots.empty())
    return llvm::Error::success();
  // AutoBlockify's shell and its body have distinct Stage owners. Choose the
  // closest explicitly owned ancestor, never let the shell swallow its body.
  llvm::DenseMap<Operation *, size_t> owner;
  module.walk([&](Operation *op) {
    for (Operation *root = op; root; root = root->getParentOp()) {
      auto found = roots.find(root);
      if (found != roots.end()) {
        owner.try_emplace(op, found->second);
        break;
      }
    }
  });
  Liveness liveness(module);
  llvm::DenseMap<Operation *, int64_t> dotPredecessorCount;
  llvm::DenseMap<Operation *, int64_t> dotSuccessorCount;
  llvm::DenseMap<Operation *, int64_t> dotDepth;
  llvm::DenseMap<Operation *, int64_t> dotOrdinal;
  llvm::DenseMap<Operation *, int64_t> operationOrdinal;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> dotPredecessors;
  int64_t nextOperationOrdinal = 0;
  module.walk(
      [&](Operation *op) { operationOrdinal[op] = nextOperationOrdinal++; });
  int64_t nextDotOrdinal = 0;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "tt.dot")
      return;
    llvm::DenseSet<Operation *> predecessors;
    llvm::DenseSet<Value> visitedValues;
    for (Value operand : op->getOperands())
      collectNearestDotProducers(operand, predecessors, visitedValues);
    SmallVector<Operation *> orderedPredecessors(predecessors.begin(),
                                                 predecessors.end());
    llvm::sort(orderedPredecessors, [&](Operation *lhs, Operation *rhs) {
      return dotOrdinal.lookup(lhs) < dotOrdinal.lookup(rhs);
    });
    int64_t depth = 1;
    for (Operation *predecessor : orderedPredecessors)
      depth = std::max(depth, dotDepth.lookup(predecessor) + 1);
    dotOrdinal[op] = nextDotOrdinal++;
    dotPredecessors[op] = orderedPredecessors;
    dotPredecessorCount[op] = static_cast<int64_t>(orderedPredecessors.size());
    dotSuccessorCount.try_emplace(op, 0);
    for (Operation *predecessor : orderedPredecessors)
      ++dotSuccessorCount[predecessor];
    dotDepth[op] = depth;
  });
  llvm::DenseMap<Operation *, SmallVector<Operation *>> dotNeighbors;
  for (const auto &entry : dotOrdinal)
    dotNeighbors.try_emplace(entry.first);
  for (const auto &entry : dotPredecessors)
    for (Operation *predecessor : entry.second) {
      dotNeighbors[entry.first].push_back(predecessor);
      dotNeighbors[predecessor].push_back(entry.first);
    }
  int64_t maximumWeakComponentNodes = 0;
  bool maximumWeakComponentGeometryKnown = false;
  std::array<int64_t, 4> maximumWeakComponentGeometry{};
  llvm::DenseSet<Operation *> componentVisited;
  for (const auto &entry : dotOrdinal) {
    Operation *root = entry.first;
    if (!componentVisited.insert(root).second)
      continue;
    SmallVector<Operation *> pending{root};
    int64_t nodes = 0;
    bool geometryKnown = true;
    std::optional<std::array<int64_t, 4>> geometry;
    while (!pending.empty()) {
      Operation *dot = pending.pop_back_val();
      ++nodes;
      auto current = dotGeometry(dot);
      if (!current)
        geometryKnown = false;
      else if (!geometry)
        geometry = *current;
      else if (*geometry != *current)
        geometryKnown = false;
      for (Operation *neighbor : dotNeighbors.lookup(dot))
        if (componentVisited.insert(neighbor).second)
          pending.push_back(neighbor);
    }
    geometryKnown = geometryKnown && geometry.has_value();
    if (nodes > maximumWeakComponentNodes) {
      maximumWeakComponentNodes = nodes;
      maximumWeakComponentGeometryKnown = geometryKnown;
      if (geometry)
        maximumWeakComponentGeometry = *geometry;
    } else if (nodes == maximumWeakComponentNodes &&
               (!geometryKnown || !maximumWeakComponentGeometryKnown ||
                *geometry != maximumWeakComponentGeometry)) {
      maximumWeakComponentGeometryKnown = false;
    }
  }
  // Lookup is deterministic by Stage/operation order; unordered live sets
  // contribute integer sums only, never order-dependent allocation decisions.
  for (auto entry : llvm::enumerate(partition.stages)) {
    size_t index = entry.index();
    LogicalStage &stage = entry.value();
    SimtRegisterPressure facts;
    facts.baseWarpCount = baseWarpCount;
    facts.ttirMaximumWeakDotComponentNodes = maximumWeakComponentNodes;
    facts.ttirMaximumWeakDotComponentGeometryKnown =
        maximumWeakComponentGeometryKnown;
    if (maximumWeakComponentGeometryKnown) {
      facts.ttirMaximumWeakDotComponentM = maximumWeakComponentGeometry[0];
      facts.ttirMaximumWeakDotComponentN = maximumWeakComponentGeometry[1];
      facts.ttirMaximumWeakDotComponentK = maximumWeakComponentGeometry[2];
      facts.ttirMaximumWeakDotComponentElementBits =
          maximumWeakComponentGeometry[3];
    }
    llvm::DenseSet<Value> unknown;
    llvm::DenseSet<Value> predicateDefinitions;
    llvm::DenseMap<Value, int64_t> predicateDepth;
    llvm::DenseSet<Value> loopInvariantPredicates;
    llvm::DenseSet<Operation *> visited;
    for (Operation *root : stage.operations) {
      root->walk([&](Operation *op) {
        if (owner.lookup(op) != index || !op->getBlock() ||
            !visited.insert(op).second)
          return;
        // These are optional lowering-side labels.
        if (auto attr = op->getAttrOfType<IntegerAttr>(
                "ascend.simt.lowered_peak_live_register_units"))
          facts.loweredPeakLiveUnitsPerThread = std::max<int64_t>(
              facts.loweredPeakLiveUnitsPerThread, attr.getInt());
        if (auto attr =
                op->getAttrOfType<IntegerAttr>("ascend.simt.unroll_factor"))
          facts.backendUnrollFactor =
              std::max<int64_t>(facts.backendUnrollFactor, attr.getInt());
        if (op->getName().getStringRef() == "tt.dot") {
          ++facts.ttirDotDefinitionCount;
          const int64_t predecessors = dotPredecessorCount.lookup(op);
          facts.ttirDotDependencyEdgeCount += predecessors;
          if (predecessors == 0)
            ++facts.ttirDotDependencySourceCount;
          if (dotSuccessorCount.lookup(op) == 0)
            ++facts.ttirDotDependencySinkCount;
          facts.ttirMaxDotDependencyDepth =
              std::max(facts.ttirMaxDotDependencyDepth, dotDepth.lookup(op));
          SimtDotDependencySite dependency;
          dependency.ordinal = dotOrdinal.lookup(op);
          dependency.operationOrdinal = operationOrdinal.lookup(op);
          for (Operation *predecessor : dotPredecessors.lookup(op))
            dependency.predecessorOrdinals.push_back(
                dotOrdinal.lookup(predecessor));
          dependency.depth = dotDepth.lookup(op);
          dependency.successorCount = dotSuccessorCount.lookup(op);
          collectDotConsumerContext(op, operationOrdinal, dependency);
          facts.dotDependencySites.push_back(std::move(dependency));
        }
        for (Value result : op->getResults()) {
          auto units = predicateUnits(result, baseWarpCount * 32);
          if (!units || !predicateDefinitions.insert(result).second)
            continue;
          facts.predicateDefinitionUnitsPerThread += *units;
          switch (predicateClass(op)) {
          case PredicateClass::IntegerEquality:
            facts.ttirIntegerEqualityPredicateDefinitionUnitsPerThread +=
                *units;
            break;
          case PredicateClass::IntegerOrdered:
            facts.ttirIntegerOrderedPredicateDefinitionUnitsPerThread += *units;
            break;
          case PredicateClass::Shape:
            facts.ttirPredicateShapeDefinitionUnitsPerThread += *units;
            break;
          case PredicateClass::Other:
            facts.ttirOtherPredicateDefinitionUnitsPerThread += *units;
            break;
          }
          int64_t depth = 1;
          for (Value operand : op->getOperands()) {
            if (!predicateUnits(operand, baseWarpCount * 32))
              continue;
            facts.ttirPredicateDagEdgeUnitsPerThread += *units;
            depth = std::max(depth, predicateDepth.lookup(operand) + 1);
          }
          predicateDepth[result] = depth;
          facts.ttirMaxPredicateDagDepth =
              std::max(facts.ttirMaxPredicateDagDepth, depth);
          const int64_t fanout = static_cast<int64_t>(
              std::distance(result.use_begin(), result.use_end()));
          facts.maxPredicateFanout = std::max(facts.maxPredicateFanout, fanout);
          facts.predicateUseUnitsPerThread += *units * fanout;
          for (OpOperand &use : result.getUses()) {
            Operation *user = use.getOwner();
            if (isa<arith::SelectOp>(user) && use.getOperandNumber() == 0)
              facts.ttirSelectPredicateUseUnitsPerThread += *units;
            if (isPredicateBooleanConsumer(user))
              facts.ttirBooleanPredicateUseUnitsPerThread += *units;
            if (!crossesIntoLoop(result, use.getOwner()))
              continue;
            facts.loopInvariantPredicateUseUnitsPerThread += *units;
            if (loopInvariantPredicates.insert(result).second)
              facts.loopInvariantPredicateUnitsPerThread += *units;
          }
        }
        if (op->getNumRegions() != 0)
          return;
        const auto *info = liveness.getLiveness(op->getBlock());
        if (!info)
          return;
        auto live = info->currentlyLiveValues(op);
        for (Operation *parent = op->getParentOp();
             parent && parent->getBlock() &&
             !parent->hasTrait<OpTrait::IsIsolatedFromAbove>();
             parent = parent->getParentOp()) {
          if (auto loop = dyn_cast<scf::ForOp>(parent)) {
            live.insert(loop.getUpperBound());
            live.insert(loop.getStep());
            live.insert(loop.getInductionVar());
          }
          const auto *outer = liveness.getLiveness(parent->getBlock());
          if (!outer)
            continue;
          for (Value value : outer->currentlyLiveValues(parent)) {
            // Parent results do not exist yet while its region is executing.
            if (value.getDefiningOp() != parent &&
                !liveness.isDeadAfter(value, parent))
              live.insert(value);
          }
        }
        llvm::DenseSet<Value> counted;
        int64_t tensor = 0, scalar = 0, cross = 0, carried = 0, remat = 0;
        int64_t predicates = 0;
        int64_t pointers = 0, blockPointers = 0, blockPointerRanks = 0;
        for (Value original : live) {
          Value value = storageRoot(original);
          if (!counted.insert(value).second)
            continue;
          auto units = valueUnits(value, baseWarpCount * 32);
          if (!units) {
            unknown.insert(value);
            continue;
          }
          if (rematerializable(value)) {
            remat += *units;
            continue;
          }
          if (auto units = predicateUnits(value, baseWarpCount * 32))
            predicates += *units;
          (isa<ShapedType>(value.getType()) ? tensor : scalar) += *units;
          Type element = value.getType();
          if (auto shaped = dyn_cast<ShapedType>(element))
            element = shaped.getElementType();
          if (auto pointer = dyn_cast<triton::PointerType>(element)) {
            pointers += *units;
            if (auto pointee =
                    dyn_cast<RankedTensorType>(pointer.getPointeeType())) {
              ++blockPointers;
              blockPointerRanks += pointee.getRank();
            }
          }
          Operation *definition = value.getDefiningOp();
          if (definition && owner.count(definition) &&
              owner.lookup(definition) != index)
            cross += *units;
          if (auto argument = dyn_cast<BlockArgument>(value))
            if (isa<scf::ForOp>(argument.getOwner()->getParentOp()) &&
                argument.getArgNumber() != 0)
              carried += *units;
        }
        facts.peakLiveUnitsPerThread =
            std::max(facts.peakLiveUnitsPerThread, tensor + scalar);
        facts.peakTensorUnitsPerThread =
            std::max(facts.peakTensorUnitsPerThread, tensor);
        facts.peakScalarUnitsPerThread =
            std::max(facts.peakScalarUnitsPerThread, scalar);
        facts.peakCrossStageUnitsPerThread =
            std::max(facts.peakCrossStageUnitsPerThread, cross);
        facts.peakLoopCarriedUnitsPerThread =
            std::max(facts.peakLoopCarriedUnitsPerThread, carried);
        facts.peakRematerializableUnitsPerThread =
            std::max(facts.peakRematerializableUnitsPerThread, remat);
        facts.peakLivePredicateUnitsPerThread =
            std::max(facts.peakLivePredicateUnitsPerThread, predicates);
        facts.peakPointerUnitsPerThread =
            std::max(facts.peakPointerUnitsPerThread, pointers);
        facts.peakBlockPointerCount =
            std::max(facts.peakBlockPointerCount, blockPointers);
        facts.peakBlockPointerRankSum =
            std::max(facts.peakBlockPointerRankSum, blockPointerRanks);
        // Retained state must be sampled at the actual dot. Combining a
        // pointer peak elsewhere with a tensor peak here invents liveness.
        if (op->getName().getStringRef() == "tt.dot" &&
            op->getNumOperands() == 3 && op->getNumResults() == 1) {
          auto a = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
          auto b = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
          if (!a || !b || a.getRank() != 2 || b.getRank() != 2 ||
              !a.hasStaticShape() || !b.hasStaticShape() ||
              a.getDimSize(1) != b.getDimSize(0) ||
              !isa<IntegerType, FloatType>(a.getElementType()))
            return;
          SimtDotPressureSite site;
          site.m = a.getDimSize(0);
          site.n = b.getDimSize(1);
          site.k = a.getDimSize(1);
          site.elementBits = a.getElementType().getIntOrFloatBitWidth();
          llvm::DenseSet<Value> operands;
          for (Value original : op->getOperands()) {
            Value value = storageRoot(original);
            if (operands.insert(value).second && !rematerializable(value))
              if (auto units = valueUnits(value, baseWarpCount * 32))
                site.operandUnitsPerThread += *units;
          }
          for (Value value : counted) {
            if (operands.contains(value) || value.getDefiningOp() == op ||
                rematerializable(value))
              continue;
            if (auto units = valueUnits(value, baseWarpCount * 32))
              (isa<ShapedType>(value.getType())
                   ? site.retainedTensorUnitsPerThread
                   : site.retainedScalarUnitsPerThread) += *units;
          }
          facts.dotSites.push_back(site);
        }
      });
    }
    facts.unknownValueCount = unknown.size();
    if (!stage.operations.empty())
      stage.wholeSimtRegisterPressure = facts;
  }
  SmallVector<const SimtDotDependencySite *> functionDotSites;
  for (const LogicalStage &stage : partition.stages)
    if (stage.wholeSimtRegisterPressure)
      for (const auto &site :
           stage.wholeSimtRegisterPressure->dotDependencySites)
        functionDotSites.push_back(&site);
  llvm::sort(functionDotSites, [](const auto *lhs, const auto *rhs) {
    return lhs->ordinal < rhs->ordinal;
  });
  int64_t minimumPositiveSideEffectSpan = 0;
  for (const SimtDotDependencySite *site : functionDotSites) {
    if (site->sideEffectFrontierCount == 0 || site->maxSideEffectUseSpan <= 0)
      continue;
    if (minimumPositiveSideEffectSpan == 0 ||
        site->maxSideEffectUseSpan < minimumPositiveSideEffectSpan)
      minimumPositiveSideEffectSpan = site->maxSideEffectUseSpan;
  }
  std::string functionDotUseSignature;
  for (const SimtDotDependencySite *site : functionDotSites) {
    if (!functionDotUseSignature.empty())
      functionDotUseSignature += ";";
    functionDotUseSignature += std::to_string(site->ordinal) + ":";
    for (auto predecessor : llvm::enumerate(site->predecessorOrdinals)) {
      if (predecessor.index() != 0)
        functionDotUseSignature += ",";
      functionDotUseSignature += std::to_string(predecessor.value());
    }
    const int64_t normalizedSideEffectUseSpan =
        site->sideEffectFrontierCount > 0
            ? std::max<int64_t>(0, site->maxSideEffectUseSpan -
                                       minimumPositiveSideEffectSpan)
            : 0;
    functionDotUseSignature +=
        ":" + std::to_string(site->depth) + ":" +
        std::to_string(site->successorCount) + ":" +
        std::to_string(site->nonDotPureConsumerCount) + ":" +
        std::to_string(site->sideEffectFrontierCount) + ":" +
        std::to_string(site->maxNonDotPureUseSpan) + ":" +
        std::to_string(normalizedSideEffectUseSpan);
  }
  for (LogicalStage &stage : partition.stages)
    if (stage.wholeSimtRegisterPressure)
      stage.wholeSimtRegisterPressure->ttirFunctionDotUseSignature =
          functionDotUseSignature;
  // The floor describes one static whole-function stream.
  auto chooseOwner = [&](size_t begin, size_t end) -> std::optional<size_t> {
    std::optional<size_t> selected;
    int64_t selectedPeak = -1;
    for (size_t index = begin; index < end; ++index) {
      const LogicalStage &stage = partition.stages[index];
      if (!stage.wholeSimtRegisterPressure || stage.iterationCount != 1 ||
          stage.scheduleKind != StageScheduleKind::StraightLine)
        continue;
      const int64_t peak =
          stage.wholeSimtRegisterPressure->peakLiveUnitsPerThread;
      if (!selected || peak > selectedPeak) {
        selected = index;
        selectedPeak = peak;
      }
    }
    return selected;
  };
  std::optional<size_t> recurrence;
  std::optional<size_t> firstDot;
  std::optional<size_t> lastDot;
  for (auto entry : llvm::enumerate(partition.stages)) {
    const LogicalStage &stage = entry.value();
    if (!recurrence &&
        stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence)
      recurrence = entry.index();
    if (stage.wholeSimtRegisterPressure &&
        stage.wholeSimtRegisterPressure->ttirDotDefinitionCount > 0) {
      if (!firstDot)
        firstDot = entry.index();
      lastDot = entry.index();
    }
  }
  const size_t stageCount = partition.stages.size();
  bool split = false;
  if (maximumWeakComponentNodes > 0 && recurrence && firstDot && lastDot &&
      *recurrence < *firstDot) {
    auto before = chooseOwner(0, *recurrence);
    auto dots = chooseOwner(*recurrence + 1, *lastDot + 1);
    auto after = chooseOwner(*lastDot + 1, stageCount);
    if (before && dots && after) {
      partition.stages[*before]
          .wholeSimtRegisterPressure->dotComponentPrivateStackChargeRegion =
          SimtPrivateStackChargeRegion::BeforeRecurrence;
      partition.stages[*dots]
          .wholeSimtRegisterPressure->dotComponentPrivateStackChargeRegion =
          SimtPrivateStackChargeRegion::DotRegion;
      partition.stages[*after]
          .wholeSimtRegisterPressure->dotComponentPrivateStackChargeRegion =
          SimtPrivateStackChargeRegion::AfterDot;
      split = true;
    }
  }
  if (maximumWeakComponentNodes > 0 && !split)
    if (auto owner = chooseOwner(0, stageCount))
      partition.stages[*owner]
          .wholeSimtRegisterPressure->dotComponentPrivateStackChargeRegion =
          SimtPrivateStackChargeRegion::WholeFunction;
  return llvm::Error::success();
}
