//===- StagePartitioner.cpp - Build semantic Phase/Stage IR -------------===//

#include "AscendModel/Analysis/StagePartitioner.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <system_error>
#include <utility>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static void recomputeIssueElements(StageWorkload &work) {
  double elements = work.scalarOperations + work.predicateElements;
  for (const auto &entry : work.operationElements)
    elements += entry.second;
  elements += 32.0 * (work.loadWarpInstructions + work.storeWarpInstructions);
  work.issueElements = elements;
}

static double getTypeElementCount(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    if (!shaped.hasStaticShape())
      return 1.0;
    return static_cast<double>(std::max<int64_t>(1, shaped.getNumElements()));
  }
  return 1.0;
}

static Type getScalarElementType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  return type;
}

static std::string typeToString(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << type;
  stream.flush();
  return text;
}

static bool isPointerLikeType(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    type = tensor.getElementType();
  return llvm::StringRef(typeToString(type)).contains("!tt.ptr");
}

/// True when a loop argument only participates in address induction.  Such a
/// value is an implementation recurrence that later pointer canonicalization
/// can eliminate; it is not an algorithmic loop-carried dependency and must
/// not disable the SIMD independent-loop roofline model.
static bool isAddressOnlyLoopValue(Value root) {
  llvm::SmallVector<Value, 8> worklist{root};
  llvm::DenseSet<Value> visited;
  bool reachesAddressUse = false;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      const llvm::StringRef name = user->getName().getStringRef();
      if (name == "scf.yield" || name == "scf.condition")
        continue;
      if ((name == "tt.load" || name == "tt.store" ||
           name.starts_with("tt.atomic")) &&
          use.getOperandNumber() == 0) {
        reachesAddressUse = true;
        continue;
      }
      const bool addressForwarding =
          name == "tt.addptr" || name == "tt.advance" || name == "tt.splat" ||
          name == "tt.broadcast" || name == "tt.expand_dims" ||
          name == "arith.addi" || name == "arith.subi" ||
          name == "arith.muli" || name == "arith.index_cast";
      if (!addressForwarding)
        return false;
      if (name == "tt.addptr" || name == "tt.advance")
        reachesAddressUse = true;
      llvm::append_range(worklist, user->getResults());
    }
  }
  return reachesAddressUse;
}

static int64_t getScalarBitWidth(Type type) {
  type = getScalarElementType(type);
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (auto floating = dyn_cast<FloatType>(type))
    return floating.getWidth();
  if (isa<IndexType>(type))
    return 64;
  return 0;
}

static double getValueBytes(Value value) {
  const int64_t bits = getScalarBitWidth(value.getType());
  return bits > 0 ? getTypeElementCount(value.getType()) *
                        static_cast<double>(bits) / 8.0
                  : 0.0;
}

static double getOperationElements(Operation *operation) {
  double elements = 1.0;
  for (Type type : operation->getResultTypes())
    elements = std::max(elements, getTypeElementCount(type));
  if (operation->getNumResults() == 0)
    for (Value value : operation->getOperands())
      elements = std::max(elements, getTypeElementCount(value.getType()));
  return elements;
}

static bool hasTensorResult(Operation *operation) {
  return llvm::any_of(operation->getResultTypes(),
                      [](Type type) { return isa<ShapedType>(type); });
}

static llvm::StringRef getProfileOperationName(Operation *operation) {
  const llvm::StringRef name = operation->getName().getStringRef();
  return llvm::StringSwitch<llvm::StringRef>(name)
      .Cases("arith.addf", "tt.add", "f32.add")
      .Case("arith.subf", "f32.sub")
      .Case("arith.mulf", "f32.mul")
      .Case("arith.divf", "f32.div")
      .Cases("arith.maximumf", "arith.maxnumf", "f32.max")
      .Cases("math.absf", "tt.abs", "f32.abs")
      .Cases("math.exp", "tt.exp", "f32.exp")
      .Cases("math.log", "tt.log", "f32.log")
      .Cases("arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp",
             "convert.cast")
      .Cases("arith.fptosi", "arith.fptoui", "convert.cast")
      .Default("generic.issue");
}

static void accumulateDotWorkload(Operation *operation, StageWorkload &work) {
  if (operation->getNumOperands() < 2)
    return;
  auto lhs = dyn_cast<ShapedType>(operation->getOperand(0).getType());
  auto rhs = dyn_cast<ShapedType>(operation->getOperand(1).getType());
  if (!lhs || !rhs || !lhs.hasStaticShape() || !rhs.hasStaticShape() ||
      lhs.getRank() < 2 || rhs.getRank() < 2)
    return;
  const int64_t m = lhs.getShape()[lhs.getRank() - 2];
  const int64_t k = lhs.getShape()[lhs.getRank() - 1];
  const int64_t n = rhs.getShape()[rhs.getRank() - 1];
  if (m > 0 && n > 0 && k > 0)
    work.dotFlops += 2.0 * static_cast<double>(m) * static_cast<double>(n) *
                     static_cast<double>(k);
}

static void accumulateReductionWorkload(Operation *operation,
                                        StageWorkload &work) {
  if (operation->getNumOperands() == 0)
    return;
  auto input = dyn_cast<ShapedType>(operation->getOperand(0).getType());
  auto axis = operation->getAttrOfType<IntegerAttr>("axis");
  if (!input || !input.hasStaticShape() || !axis || input.getRank() == 0)
    return;
  int64_t dimension = axis.getInt();
  if (dimension < 0)
    dimension += input.getRank();
  if (dimension < 0 || dimension >= input.getRank())
    return;
  const int64_t extent = input.getShape()[dimension];
  if (extent <= 1)
    return;
  const double depth = std::ceil(std::log2(static_cast<double>(extent)));
  work.shuffleLaneSteps += getTypeElementCount(input) * depth;
}

static void accumulateOneOperation(Operation *operation, StageWorkload &work) {
  if (!operation || operation->hasTrait<OpTrait::IsTerminator>())
    return;
  const llvm::StringRef name = operation->getName().getStringRef();
  const double elements = getOperationElements(operation);

  if ((name == "tt.load" || name == "tt.gather") &&
      operation->getNumResults() > 0) {
    Value result = operation->getResult(0);
    work.loadBytes += getValueBytes(result);
    work.loadWarpInstructions += std::ceil(elements / 32.0);
    return;
  }
  if ((name == "tt.store" || name.starts_with("tt.atomic")) &&
      operation->getNumOperands() > 1) {
    Value value = operation->getOperand(1);
    work.storeBytes += getValueBytes(value);
    work.storeWarpInstructions +=
        std::ceil(getTypeElementCount(value.getType()) / 32.0);
    return;
  }
  if (name == "tt.dot") {
    accumulateDotWorkload(operation, work);
    return;
  }
  if (name == "tt.reduce" || name == "tt.scan")
    accumulateReductionWorkload(operation, work);
  if (name == "arith.cmpi" || name == "arith.cmpf") {
    work.predicateElements += elements;
    return;
  }
  if (name == "scf.for" || name == "scf.if" || name == "scf.while")
    return;

  if (!hasTensorResult(operation)) {
    work.scalarOperations += 1.0;
    return;
  }
  work.operationElements[getProfileOperationName(operation)] += elements;
}

static void mergeWorkload(StageWorkload &into, StageWorkload from);

static void scaleWorkload(StageWorkload &work, double scale) {
  work.scalarOperations *= scale;
  work.loadBytes *= scale;
  work.storeBytes *= scale;
  work.loadWarpInstructions *= scale;
  work.storeWarpInstructions *= scale;
  work.predicateElements *= scale;
  work.shuffleLaneSteps *= scale;
  work.dotFlops *= scale;
  work.estimatedSpillTransactions *= scale;
  for (auto &entry : work.operationElements)
    entry.second *= scale;
  recomputeIssueElements(work);
}

static std::optional<int64_t> getConstantInteger(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return std::nullopt;
  auto attribute = definition->getAttrOfType<IntegerAttr>("value");
  if (!attribute)
    return std::nullopt;
  return attribute.getInt();
}

static int64_t getLoopTripCount(Operation *operation,
                                int64_t stageIterationCount) {
  const llvm::StringRef name = operation->getName().getStringRef();
  if (name == "scf.for" && operation->getNumOperands() >= 3) {
    const std::optional<int64_t> lower =
        getConstantInteger(operation->getOperand(0));
    const std::optional<int64_t> upper =
        getConstantInteger(operation->getOperand(1));
    const std::optional<int64_t> step =
        getConstantInteger(operation->getOperand(2));
    if (lower && upper && step && *step > 0 && *upper > *lower)
      return (*upper - *lower + *step - 1) / *step;
  }
  if (name == "scf.for" || name == "scf.while")
    return std::max<int64_t>(1, stageIterationCount);
  return 1;
}

/// Accumulate dynamic work, not merely the number of syntactic TTIR ops.
/// A loop body appears once in TTIR but executes `tripCount` times.  The
/// resulting total dynamic work is normalized to one Stage iteration by
/// makePerIteration().  Consequently N_iter * C_body accounts for every
/// loop iteration instead of accidentally counting the body once.
/// AutoBlockify V1 is the exception: its loop is a scheduling shell and its
/// direct body operations are already separate semantic roots.
static void accumulateDynamicOperationTree(Operation *operation,
                                           StageWorkload &work,
                                           double multiplicity,
                                           int64_t fallbackLoopTripCount) {
  if (!operation)
    return;
  StageWorkload local;
  accumulateOneOperation(operation, local);
  scaleWorkload(local, multiplicity);
  mergeWorkload(work, std::move(local));

  if (operation->hasAttr("ta.auto_blockify_v1.loop"))
    return;
  const double childMultiplicity =
      multiplicity *
      static_cast<double>(getLoopTripCount(operation, fallbackLoopTripCount));
  for (Region &region : operation->getRegions())
    for (Block &block : region)
      for (Operation &nested : block.getOperations())
        accumulateDynamicOperationTree(&nested, work, childMultiplicity,
                                       fallbackLoopTripCount);
}

static int64_t countAlgorithmLoops(const LogicalStage &stage) {
  int64_t count = 0;
  for (Operation *root : stage.operations) {
    if (!root || root->hasAttr("ta.auto_blockify_v1.loop"))
      continue;
    root->walk([&](Operation *operation) {
      const llvm::StringRef name = operation->getName().getStringRef();
      if ((name == "scf.for" || name == "scf.while") &&
          !operation->hasAttr("ta.auto_blockify_v1.loop"))
        ++count;
    });
  }
  return count;
}

static void mergeWorkload(StageWorkload &into, StageWorkload from) {
  into.scalarOperations += from.scalarOperations;
  into.loadBytes += from.loadBytes;
  into.storeBytes += from.storeBytes;
  into.loadWarpInstructions += from.loadWarpInstructions;
  into.storeWarpInstructions += from.storeWarpInstructions;
  into.predicateElements += from.predicateElements;
  into.shuffleLaneSteps += from.shuffleLaneSteps;
  into.dotFlops += from.dotFlops;
  into.estimatedSpillTransactions += from.estimatedSpillTransactions;
  for (const auto &[name, elements] : from.operationElements)
    into.operationElements[name] += elements;
  recomputeIssueElements(into);
}

static void makePerIteration(LogicalStage &stage) {
  const double count =
      static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
  StageWorkload &work = stage.workload;
  work.scalarOperations /= count;
  work.loadBytes /= count;
  work.storeBytes /= count;
  work.loadWarpInstructions /= count;
  work.storeWarpInstructions /= count;
  work.predicateElements /= count;
  work.shuffleLaneSteps /= count;
  work.dotFlops /= count;
  work.estimatedSpillTransactions /= count;
  for (auto &entry : work.operationElements)
    entry.second /= count;
  recomputeIssueElements(work);
}

static LogicalStage makeStage(llvm::StringRef id, StageCostModelKind kind,
                              StageScheduleKind schedule, int64_t iterations,
                              StageWorkload workload) {
  LogicalStage stage;
  stage.id = id.str();
  stage.costModelKind = kind;
  stage.scheduleKind = schedule;
  stage.iterationCount = std::max<int64_t>(1, iterations);
  stage.workload = std::move(workload);
  makePerIteration(stage);
  return stage;
}

static LogicalStage withControl(LogicalStage stage, int64_t branches,
                                int64_t divergent, double activeLaneRatio) {
  stage.features.conditionalBranchCount = std::max<int64_t>(0, branches);
  stage.features.divergentBranchCount = std::max<int64_t>(0, divergent);
  stage.features.activeLaneRatio = std::clamp(activeLaneRatio, 0.0, 1.0);
  return stage;
}

static LogicalStage asLocalSIMT(LogicalStage stage) {
  stage.localSimtMaterializable = true;
  // StageModeLegalityAnalysis opens F2/F4 only when backend integration has
  // the AutoBlockify V1 Scope SuperBlock wrapper available.
  stage.localSimtFactors = {1};
  return stage;
}

static void addPhase(StagePartition &partition, llvm::StringRef id,
                     LogicalStage stage) {
  LogicalPhase phase;
  phase.id = id.str();
  phase.stages.push_back(std::move(stage));
  partition.phases.push_back(std::move(phase));
}

static bool operationTreeContainsName(Operation *root, llvm::StringRef name);

static bool hasPhase(const PhaseBoundaryPlan *plan, llvm::StringRef id) {
  return plan && llvm::is_contained(plan->rootPhaseIds, id);
}

static void prependAutoBlockifyStages(StagePartition &partition,
                                      const SimdSimtFeatureSummary &features,
                                      const PhaseBoundaryPlan *plan) {
  if (!features.autoBlockifyV1Applied &&
      !hasPhase(plan, "auto_blockify_dispatch"))
    return;
  LogicalPhase phase;
  phase.id = "auto_blockify_dispatch";
  phase.stages.push_back(makeStage("physical_program_dispatch",
                                   StageCostModelKind::AutoBlockifyDispatch,
                                   StageScheduleKind::StraightLine, 1, {}));
  if (features.autoBlockifyV1LoopCount > 0 ||
      hasPhase(plan, "auto_blockify_dispatch"))
    phase.stages.push_back(
        makeStage("logical_program_loop", StageCostModelKind::AutoBlockifyLoop,
                  StageScheduleKind::IndependentPipelined,
                  std::max<int64_t>(1, features.autoBlockifyV1LoopCount), {}));
  partition.phases.push_back(std::move(phase));
}

static StagePartition
partitionTriangular(const SimdSimtFeatureSummary &features,
                    const TriangularSolveFacts &facts,
                    const PhaseBoundaryPlan *plan) {
  StagePartition partition;
  partition.domain = "triangular_recurrence";
  prependAutoBlockifyStages(partition, features, plan);

  LogicalStage head = withControl(
      makeStage("head_index_mask", StageCostModelKind::PredicateMask,
                StageScheduleKind::StraightLine, 1, {}),
      features.conditionalBranchCount -
          features.simtAnchors.conditionalBranchCount,
      features.divergentBranchCount - features.simtAnchors.divergentBranchCount,
      features.activeLaneRatio);
  head.workload.paysKernelSetup = true;
  addPhase(partition, "head", std::move(head));

  if (hasPhase(plan, "diagonal_load"))
    addPhase(partition, "diagonal_load",
             makeStage("load_diagonal_tiles",
                       StageCostModelKind::ContinuousTileMemory,
                       StageScheduleKind::IndependentPipelined, 1, {}));

  const int64_t iterations = std::max<int64_t>(1, facts.recurrenceLoopCount);
  LogicalStage recurrence = asLocalSIMT(withControl(
      makeStage("diagonal_inverse_recurrence",
                StageCostModelKind::LoopCarriedRecurrence,
                StageScheduleKind::LoopCarriedSerial, iterations, {}),
      features.simtAnchors.conditionalBranchCount,
      features.simtAnchors.divergentBranchCount,
      features.simtAnchors.activeLaneRatio));
  const int64_t rows =
      std::max<int64_t>(1, facts.blockRows - facts.recurrenceStartRow);
  recurrence.features.parallelRecurrenceGroupCount =
      std::max<int64_t>(1, (iterations + rows - 1) / rows);
  addPhase(partition, "diagonal_inverse", std::move(recurrence));

  if (hasPhase(plan, "merge_store")) {
    bool hasDot = false;
    bool hasStore = false;
    for (auto root : llvm::enumerate(plan->rootOperations)) {
      if (plan->rootPhaseIds[root.index()] != "merge_store")
        continue;
      hasDot |= operationTreeContainsName(root.value(), "tt.dot");
      hasStore |= operationTreeContainsName(root.value(), "tt.store");
    }
    LogicalPhase phase;
    phase.id = "merge_store";
    if (hasDot)
      phase.stages.push_back(
          makeStage("dense_dot_tail", StageCostModelKind::CubeRoofline,
                    StageScheduleKind::IndependentPipelined,
                    std::max<int64_t>(1, facts.denseDotTailOps), {}));
    if (hasStore)
      phase.stages.push_back(makeStage("store_inverse_tile",
                                       StageCostModelKind::ContinuousTileStore,
                                       StageScheduleKind::StraightLine, 1, {}));
    if (!phase.stages.empty())
      partition.phases.push_back(std::move(phase));
  }
  return partition;
}

static StagePartition partitionRowwise(const SimdSimtFeatureSummary &features,
                                       const PhaseBoundaryPlan *plan) {
  StagePartition partition;
  partition.domain = "loaded_index_rowwise_reduction";
  prependAutoBlockifyStages(partition, features, plan);
  addPhase(partition, "row_dispatch",
           makeStage("row_index_generation",
                     StageCostModelKind::IndexGeneration,
                     StageScheduleKind::StraightLine, 1, {}));
  addPhase(partition, "row_load",
           asLocalSIMT(makeStage(
               "indirect_row_gather", StageCostModelKind::IndirectGatherMemory,
               StageScheduleKind::PartiallyDependent, 1, {})));
  const int64_t iterations =
      std::max<int64_t>(1, features.staticLoopTripCountMax);
  addPhase(partition, "row_reduction",
           makeStage("rowwise_reduction", StageCostModelKind::RowwiseReduction,
                     StageScheduleKind::PartiallyDependent, iterations, {}));
  addPhase(partition, "convert_store",
           makeStage("conversion_pack_store",
                     StageCostModelKind::ConversionPack,
                     StageScheduleKind::IndependentPipelined, iterations, {}));
  return partition;
}

static StagePartition
partitionIndirectDot(const SimdSimtFeatureSummary &features,
                     const PhaseBoundaryPlan *plan) {
  StagePartition partition;
  partition.domain = "indirect_underfilled_dot";
  prependAutoBlockifyStages(partition, features, plan);
  addPhase(partition, "index_setup",
           makeStage("index_generation", StageCostModelKind::IndexGeneration,
                     StageScheduleKind::StraightLine, 1, {}));
  addPhase(partition, "gather_tiles",
           asLocalSIMT(makeStage(
               "indirect_tile_gather", StageCostModelKind::IndirectGatherMemory,
               StageScheduleKind::PartiallyDependent, 1, {})));
  addPhase(partition, "dot",
           makeStage("tiny_cube_dot", StageCostModelKind::TinyCubeRoofline,
                     StageScheduleKind::IndependentPipelined, 1, {}));
  addPhase(partition, "output_store",
           makeStage("store_dot_result",
                     StageCostModelKind::ContinuousTileStore,
                     StageScheduleKind::StraightLine, 1, {}));
  return partition;
}

static bool anchorMatchesStage(const SimtAnchorDescriptor &anchor,
                               const LogicalStage &stage) {
  if (!anchor.materializable || !stage.localSimtMaterializable)
    return false;
  if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence)
    return anchor.kind == SimtAnchorKind::TriangularSolveLoop;
  if (stage.costModelKind == StageCostModelKind::IndirectGatherMemory ||
      stage.costModelKind == StageCostModelKind::IndirectScalarMemory)
    return anchor.kind == SimtAnchorKind::DirectGather ||
           anchor.kind == SimtAnchorKind::LoadedIndexDependentMemory;
  return false;
}

static Operation *getTopLevelSemanticRoot(Operation *operation);

static bool stageOwnsAnchor(const LogicalStage &stage,
                            const SimtAnchorDescriptor &anchor) {
  if (!anchorMatchesStage(anchor, stage))
    return false;
  auto owns = [&](Operation *operation) {
    Operation *root = getTopLevelSemanticRoot(operation);
    return root && llvm::is_contained(stage.operations, root);
  };
  if (anchor.scopeOperations.empty())
    return owns(anchor.operation);
  return llvm::all_of(anchor.scopeOperations, owns);
}

/// Associate exact anchor descriptors with their owning Stage after complete
/// operation ownership has been established.  This is deliberately separate
/// from operation assignment: anchors are materialization evidence, not a
/// second source of Stage boundaries.
static void attachExactAnchorOwnership(StagePartition &partition,
                                       const SimtAnchorPlan &anchorPlan) {
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      if (!stage.localSimtMaterializable)
        continue;
      stage.simtAnchorIndices.clear();
      for (auto indexedAnchor : llvm::enumerate(anchorPlan.anchors)) {
        const SimtAnchorDescriptor &anchor = indexedAnchor.value();
        if (anchor.materializable && stageOwnsAnchor(stage, anchor))
          stage.simtAnchorIndices.push_back(
              static_cast<unsigned>(indexedAnchor.index()));
      }
      stage.localSimtMaterializable = !stage.simtAnchorIndices.empty();
      if (!stage.localSimtMaterializable)
        stage.localSimtFactors.clear();
    }
  }
}

static bool isFunctionLikeTTIROp(Operation *operation) {
  if (!operation)
    return false;
  const llvm::StringRef name = operation->getName().getStringRef();
  return name == "tt.func" || name == "func.func";
}

static Operation *getTopLevelSemanticRoot(Operation *operation) {
  if (!operation)
    return nullptr;
  Operation *root = operation;
  while (Operation *parent = root->getParentOp()) {
    if (isFunctionLikeTTIROp(parent) ||
        parent->hasAttr("ta.auto_blockify_v1.loop"))
      return root;
    root = parent;
  }
  return nullptr;
}

static std::vector<Operation *> collectTopLevelSemanticRoots(ModuleOp module) {
  std::vector<Operation *> result;
  auto appendBlock = [&](Block &block) {
    for (Operation &nested : block.getOperations()) {
      if (nested.hasTrait<OpTrait::IsTerminator>())
        continue;
      result.push_back(&nested);
      // AutoBlockify V1's scf.for is a scheduling shell.  Own the shell as
      // loop control, then expose its direct body operations as semantic
      // roots.  Other structured operations remain atomic roots so their
      // nested recurrence/reduction work is not double-owned.
      if (!nested.hasAttr("ta.auto_blockify_v1.loop") ||
          nested.getNumRegions() == 0)
        continue;
      for (Block &body : nested.getRegion(0))
        for (Operation &bodyOperation : body.getOperations())
          if (!bodyOperation.hasTrait<OpTrait::IsTerminator>())
            result.push_back(&bodyOperation);
    }
  };
  for (Operation &operation : module.getBody()->getOperations()) {
    if (!isFunctionLikeTTIROp(&operation) || operation.getNumRegions() == 0)
      continue;
    for (Block &block : operation.getRegion(0))
      appendBlock(block);
  }
  return result;
}

static bool operationTreeContainsName(Operation *root, llvm::StringRef name) {
  bool found = root && root->getName().getStringRef() == name;
  if (!root || found)
    return found;
  root->walk([&](Operation *nested) {
    found |= nested->getName().getStringRef() == name;
  });
  return found;
}

static bool operationTreeContainsLoadedIndexMemory(Operation *root) {
  bool found = root && isLoadedIndexDependentMemoryOp(root);
  if (!root || found)
    return found;
  root->walk([&](Operation *nested) {
    if (!found)
      found = isLoadedIndexDependentMemoryOp(nested);
  });
  return found;
}

/// PhaseBoundaryAnalysis owns the algorithm-level serial cut.  Each root is
/// assigned exactly one Phase id in execution order.  The state machines are
/// monotone: after a boundary is crossed, a later root cannot move back to an
/// earlier Phase.  Cost and candidate mode are intentionally absent here.
static llvm::Error assignRootPhaseIds(PhaseBoundaryPlan &plan) {
  llvm::DenseSet<Operation *> anchorRoots(plan.localSimtAnchorRoots.begin(),
                                          plan.localSimtAnchorRoots.end());
  std::optional<size_t> firstAnchorIndex;
  std::optional<size_t> lastAnchorIndex;
  for (auto indexedRoot : llvm::enumerate(plan.rootOperations)) {
    if (!anchorRoots.contains(indexedRoot.value()))
      continue;
    if (!firstAnchorIndex)
      firstAnchorIndex = indexedRoot.index();
    lastAnchorIndex = indexedRoot.index();
  }
  if (plan.domain == PhaseBoundaryDomain::TriangularRecurrence &&
      !firstAnchorIndex)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "triangular PhaseBoundaryAnalysis requires an exact anchor region");
  if (plan.domain == PhaseBoundaryDomain::TriangularRecurrence &&
      firstAnchorIndex && lastAnchorIndex) {
    for (size_t index = *firstAnchorIndex; index <= *lastAnchorIndex; ++index)
      if (!anchorRoots.contains(plan.rootOperations[index]))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "triangular PhaseBoundaryAnalysis requires a contiguous planned "
            "scope region");
  }

  plan.rootPhaseIds.clear();
  plan.rootPhaseIds.reserve(plan.rootOperations.size());
  llvm::StringRef current;
  for (auto indexedRoot : llvm::enumerate(plan.rootOperations)) {
    Operation *root = indexedRoot.value();
    if (!root)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "PhaseBoundaryAnalysis received a null semantic root");
    if (root->hasAttr("ta.auto_blockify_v1.loop") ||
        root->hasAttr("ta.auto_blockify_v1.schedule")) {
      plan.rootPhaseIds.push_back("auto_blockify_dispatch");
      continue;
    }

    switch (plan.domain) {
    case PhaseBoundaryDomain::TriangularRecurrence:
      if (firstAnchorIndex && indexedRoot.index() >= *firstAnchorIndex &&
          indexedRoot.index() <= *lastAnchorIndex)
        current = "diagonal_inverse";
      else if (lastAnchorIndex && indexedRoot.index() > *lastAnchorIndex)
        current = "merge_store";
      else if (operationTreeContainsName(root, "tt.dot") ||
               operationTreeContainsName(root, "tt.store"))
        current = "merge_store";
      else if ((current.empty() || current == "head") &&
               operationTreeContainsName(root, "tt.load"))
        current = "diagonal_load";
      else if (current.empty())
        current = "head";
      break;
    case PhaseBoundaryDomain::LoadedIndexRowwiseReduction:
      if (operationTreeContainsName(root, "tt.reduce"))
        current = "row_reduction";
      else if (current == "row_reduction" ||
               operationTreeContainsName(root, "tt.store"))
        current = "convert_store";
      else if ((current.empty() || current == "row_dispatch") &&
               operationTreeContainsLoadedIndexMemory(root))
        current = "row_load";
      else if (current.empty())
        current = "row_dispatch";
      break;
    case PhaseBoundaryDomain::IndirectUnderfilledDot:
      if (operationTreeContainsName(root, "tt.dot"))
        current = "dot";
      else if (current == "dot" || operationTreeContainsName(root, "tt.store"))
        current = "output_store";
      else if ((current.empty() || current == "index_setup") &&
               operationTreeContainsLoadedIndexMemory(root))
        current = "gather_tiles";
      else if (current.empty())
        current = "index_setup";
      break;
    }
    plan.rootPhaseIds.push_back(current.str());
  }
  if (plan.rootPhaseIds.size() != plan.rootOperations.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "PhaseBoundaryAnalysis did not own every semantic root");
  llvm::StringSet<> closedPhases;
  llvm::StringRef currentPhase;
  for (const std::string &phaseId : plan.rootPhaseIds) {
    if (phaseId == currentPhase)
      continue;
    if (!currentPhase.empty())
      closedPhases.insert(currentPhase);
    if (closedPhases.contains(phaseId))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "PhaseBoundaryAnalysis produced a non-contiguous Phase '%s'",
          phaseId.c_str());
    currentPhase = phaseId;
  }
  return llvm::Error::success();
}

static LogicalStage *findStage(StagePartition &partition, llvm::StringRef id) {
  for (LogicalPhase &phase : partition.phases)
    for (LogicalStage &stage : phase.stages)
      if (stage.id == id)
        return &stage;
  return nullptr;
}

static llvm::StringRef stageIdForPhase(PhaseBoundaryDomain domain,
                                       llvm::StringRef phaseId) {
  if (domain == PhaseBoundaryDomain::TriangularRecurrence)
    return llvm::StringSwitch<llvm::StringRef>(phaseId)
        .Case("head", "head_index_mask")
        .Case("diagonal_load", "load_diagonal_tiles")
        .Case("diagonal_inverse", "diagonal_inverse_recurrence")
        .Default({});
  if (domain == PhaseBoundaryDomain::LoadedIndexRowwiseReduction)
    return llvm::StringSwitch<llvm::StringRef>(phaseId)
        .Case("row_dispatch", "row_index_generation")
        .Case("row_load", "indirect_row_gather")
        .Case("row_reduction", "rowwise_reduction")
        .Case("convert_store", "conversion_pack_store")
        .Default({});
  return llvm::StringSwitch<llvm::StringRef>(phaseId)
      .Case("index_setup", "index_generation")
      .Case("gather_tiles", "indirect_tile_gather")
      .Case("dot", "tiny_cube_dot")
      .Case("output_store", "store_dot_result")
      .Default({});
}

static int64_t getStageOrdinal(const StagePartition &partition,
                               const LogicalStage *target) {
  int64_t ordinal = 0;
  for (const LogicalPhase &phase : partition.phases) {
    for (const LogicalStage &stage : phase.stages) {
      if (&stage == target)
        return ordinal;
      ++ordinal;
    }
  }
  return -1;
}

static llvm::Error assignOperation(LogicalStage *stage, Operation *operation,
                                   llvm::DenseSet<Operation *> &owned) {
  if (!stage || !operation)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageBoundaryAnalysis could not map a TTIR operation");
  if (!owned.insert(operation).second)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageBoundaryAnalysis assigned a TTIR operation more than once");
  stage->operations.push_back(operation);
  return llvm::Error::success();
}

static llvm::Error
attachCompleteOperationOwnership(StagePartition &partition,
                                 const PhaseBoundaryPlan &plan) {
  if (!plan.hasOperationGraph())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageBoundaryAnalysis requires complete Phase root ownership");
  llvm::DenseSet<Operation *> owned;
  int64_t lastStageOrdinal = -1;
  bool mergeStoreReached = findStage(partition, "dense_dot_tail") == nullptr;

  for (auto indexedRoot : llvm::enumerate(plan.rootOperations)) {
    Operation *root = indexedRoot.value();
    const llvm::StringRef phaseId = plan.rootPhaseIds[indexedRoot.index()];
    LogicalStage *target = nullptr;
    if (phaseId == "auto_blockify_dispatch") {
      if (root->hasAttr("ta.auto_blockify_v1.loop"))
        target = findStage(partition, "logical_program_loop");
      else if (!root->hasAttr("ta.auto_blockify_v1.schedule"))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "AutoBlockify Phase contains a root without V1 provenance");
      Operation *parent = root->getParentOp();
      if (!target)
        target = findStage(partition,
                           parent && parent->hasAttr("ta.auto_blockify_v1.loop")
                               ? "logical_program_loop"
                               : "physical_program_dispatch");
    }

    if (!target && plan.domain == PhaseBoundaryDomain::TriangularRecurrence &&
        phaseId == "merge_store") {
      mergeStoreReached |= operationTreeContainsName(root, "tt.store");
      target = findStage(partition, mergeStoreReached ? "store_inverse_tile"
                                                      : "dense_dot_tail");
    }
    if (!target)
      target = findStage(partition, stageIdForPhase(plan.domain, phaseId));
    const int64_t ordinal = getStageOrdinal(partition, target);
    if (ordinal < 0)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "StageBoundaryAnalysis selected a missing Stage for Phase '%s' "
          "and root '%s'",
          phaseId.str().c_str(), root->getName().getStringRef().str().c_str());
    if (ordinal < lastStageOrdinal)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "StageBoundaryAnalysis produced non-contiguous Stage ownership");
    lastStageOrdinal = ordinal;
    if (llvm::Error error = assignOperation(target, root, owned))
      return error;
  }

  if (owned.size() != plan.rootOperations.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageBoundaryAnalysis did not conserve TTIR operation ownership");
  partition.operationOwnershipComplete = true;
  partition.modeledOperationCount =
      static_cast<int64_t>(plan.rootOperations.size());
  return llvm::Error::success();
}

static void collectOwnedOperationTree(Operation *root,
                                      llvm::DenseSet<Operation *> &owned) {
  if (!root)
    return;
  owned.insert(root);
  // The AutoBlockify loop is intentionally split into a scheduling shell and
  // direct semantic body roots.  Treating the shell as the owner of its body
  // would double-own every algorithm operation.
  if (root->hasAttr("ta.auto_blockify_v1.loop"))
    return;
  root->walk([&](Operation *nested) {
    if (nested != root)
      owned.insert(nested);
  });
}

static bool isValueDefinedInside(Value value,
                                 const llvm::DenseSet<Operation *> &owned) {
  if (Operation *definition = value.getDefiningOp())
    return owned.contains(definition);
  auto argument = dyn_cast<BlockArgument>(value);
  Operation *parent = argument ? argument.getOwner()->getParentOp() : nullptr;
  return parent && owned.contains(parent);
}

static int64_t staticTensorBytes(Value value) {
  auto shaped = dyn_cast<ShapedType>(value.getType());
  if (!shaped || !shaped.hasStaticShape())
    return 0;
  Type elementType = shaped.getElementType();
  if (!isa<IntegerType, FloatType>(elementType))
    return 0;
  const int64_t elements = shaped.getNumElements();
  const int64_t bits = elementType.getIntOrFloatBitWidth();
  if (elements <= 0 || bits <= 0)
    return 0;
  constexpr int64_t maximum = std::numeric_limits<int64_t>::max();
  if (elements > (maximum - 7) / bits)
    return maximum;
  return (elements * bits + 7) / 8;
}

static int64_t staticTensorBytes(llvm::ArrayRef<Value> values) {
  int64_t total = 0;
  for (Value value : values) {
    const int64_t bytes = staticTensorBytes(value);
    if (bytes > std::numeric_limits<int64_t>::max() - total)
      return std::numeric_limits<int64_t>::max();
    total += bytes;
  }
  return total;
}

/// Derive the exact SSA contract of every Stage from operation ownership.
/// Values defined outside and consumed inside are live-ins; values defined
/// inside and consumed by any operation outside are live-outs.
static void deriveStageLiveValues(StagePartition &partition) {
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      llvm::DenseSet<Operation *> owned;
      for (Operation *root : stage.operations)
        collectOwnedOperationTree(root, owned);

      llvm::SetVector<Value> liveIns;
      llvm::SetVector<Value> liveOuts;
      for (Operation *operation : owned) {
        for (Value operand : operation->getOperands())
          if (!isValueDefinedInside(operand, owned))
            liveIns.insert(operand);
        for (Value result : operation->getResults())
          if (llvm::any_of(result.getUsers(), [&](Operation *user) {
                return !owned.contains(user);
              }))
            liveOuts.insert(result);
      }
      stage.liveIns.assign(liveIns.begin(), liveIns.end());
      stage.liveOuts.assign(liveOuts.begin(), liveOuts.end());
      stage.liveInBytes = staticTensorBytes(stage.liveIns);
      stage.liveOutBytes = staticTensorBytes(stage.liveOuts);
    }
  }
}

/// Derive the physical tensor traffic of the exact local SIMT scopes that the
/// materializer will create.  Stage live values are intentionally not used:
/// a Stage can own SIMD operations around a much smaller local scope, and
/// charging its complete live-out footprint would invent UB traffic.
static void deriveLocalSimtScopeTraffic(StagePartition &partition,
                                        const SimtAnchorPlan &anchorPlan) {
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      stage.localSimtScopeCount = 0;
      stage.scopeInputTensorBytes = 0;
      stage.scopeOutputTensorBytes = 0;
      auto merged = mergeSimtStageAnchors(anchorPlan, stage.simtAnchorIndices);
      if (!merged)
        continue;
      {
        const SimtAnchorDescriptor &anchor = *merged;
        llvm::SmallVector<Operation *> roots;
        const bool isRange = anchor.scopeOperations.size() > 1;
        if (isRange)
          llvm::append_range(roots, anchor.scopeOperations);
        else
          roots.push_back(anchor.operation);

        llvm::DenseSet<Operation *> inside;
        for (Operation *root : roots) {
          if (!root)
            continue;
          inside.insert(root);
          root->walk([&](Operation *nested) { inside.insert(nested); });
        }

        llvm::SetVector<Value> captured;
        for (Operation *operation : inside)
          for (Value operand : operation->getOperands())
            if (!isValueDefinedInside(operand, inside))
              captured.insert(operand);

        llvm::SetVector<Value> returned;
        for (Operation *root : roots) {
          if (!root)
            continue;
          for (Value result : root->getResults()) {
            // A single-op scope returns every result.  A range scope mirrors
            // wrapAnchorRange and returns only values with an outside user.
            if (!isRange ||
                llvm::any_of(result.getUsers(), [&](Operation *user) {
                  return !inside.contains(user);
                }))
              returned.insert(result);
          }
        }

        // TritonToUnstructure cannot reconstruct offset information for a
        // tensor-of-pointer returned by scope.scope.  Capturing pointers is
        // legal (the scope is not isolated from above), but returning pointer
        // state would make this local Mixed implementation fail after route
        // selection.  Reject that implementation before it is scored; the
        // same Stage remains legal in a whole-kernel pure-SIMT route.
        if (llvm::any_of(returned, [](Value value) {
              return isPointerLikeType(value.getType());
            })) {
          stage.localSimtMaterializable = false;
          stage.localSimtFactors.clear();
          stage.simtAnchorIndices.clear();
          continue;
        }

        ++stage.localSimtScopeCount;
        stage.scopeInputTensorBytes +=
            staticTensorBytes(captured.getArrayRef());
        stage.scopeOutputTensorBytes +=
            staticTensorBytes(returned.getArrayRef());
      }
    }
  }
}

} // namespace

llvm::Expected<ProgramStructure>
ProgramStructureAnalysis::analyze(ModuleOp module,
                                  const SimtAnchorPlan &anchorPlan) const {
  if (!module)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ProgramStructureAnalysis requires ModuleOp");
  ProgramStructure structure;
  structure.rootOperations = collectTopLevelSemanticRoots(module);
  if (structure.rootOperations.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "ProgramStructureAnalysis found no top-level TTIR operation");

  // Compound scopes may legally move pure tensor setup across input loads.
  // Stage boundaries must describe the program that the selected route will
  // materialize, rather than treating the pre-materialization textual order
  // as immutable.  Normalize each compound anchor to its planned insertion
  // point before PhaseBoundaryAnalysis performs a serial cut.  The operation
  // objects are not mutated here; only the analysis view is reordered.
  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (!anchor.materializable || anchor.scopeOperations.size() < 2 ||
        !anchor.scopeInsertionPoint)
      continue;

    llvm::SmallVector<Operation *, 8> scopeRoots;
    for (Operation *operation : anchor.scopeOperations) {
      Operation *root = getTopLevelSemanticRoot(operation);
      if (root && llvm::is_contained(structure.rootOperations, root) &&
          !llvm::is_contained(scopeRoots, root))
        scopeRoots.push_back(root);
    }
    Operation *insertionRoot =
        getTopLevelSemanticRoot(anchor.scopeInsertionPoint);
    auto insertionIt = llvm::find(structure.rootOperations, insertionRoot);
    if (scopeRoots.empty() || insertionIt == structure.rootOperations.end())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "ProgramStructureAnalysis cannot normalize a compound SIMT scope");

    const size_t insertionPosition =
        static_cast<size_t>(insertionIt - structure.rootOperations.begin());
    std::vector<Operation *> reordered;
    reordered.reserve(structure.rootOperations.size());
    size_t normalizedInsertionPosition = 0;
    for (auto indexedRoot : llvm::enumerate(structure.rootOperations)) {
      if (indexedRoot.index() < insertionPosition &&
          !llvm::is_contained(scopeRoots, indexedRoot.value()))
        ++normalizedInsertionPosition;
      if (!llvm::is_contained(scopeRoots, indexedRoot.value()))
        reordered.push_back(indexedRoot.value());
    }
    reordered.insert(reordered.begin() + normalizedInsertionPosition,
                     scopeRoots.begin(), scopeRoots.end());
    structure.rootOperations = std::move(reordered);
  }

  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (!anchor.materializable)
      continue;
    auto addRoot = [&](Operation *operation) {
      Operation *root = getTopLevelSemanticRoot(operation);
      if (root && llvm::is_contained(structure.rootOperations, root) &&
          !llvm::is_contained(structure.localSimtAnchorRoots, root))
        structure.localSimtAnchorRoots.push_back(root);
    };
    if (anchor.scopeOperations.empty())
      addRoot(anchor.operation);
    else
      for (Operation *operation : anchor.scopeOperations)
        addRoot(operation);
  }
  return structure;
}

static std::optional<PhaseBoundaryPlan>
identifyPhaseBoundary(const SimdSimtFeatureSummary &features,
                      const StagePartitionerOptions &options) {
  if (features.simtAnchors.triangularSolves.size() == 1 &&
      features.simtAnchors.count > 0) {
    PhaseBoundaryPlan plan{PhaseBoundaryDomain::TriangularRecurrence,
                           "triangular_recurrence",
                           features.simtAnchors.triangularSolves.front()};
    return plan;
  }
  if (features.dotOps == 0 && features.reduceOps > 0 &&
      features.loadedIndexDependentMemoryOps > 0 && features.loadOps > 0 &&
      features.storeOps > 0) {
    PhaseBoundaryPlan plan{PhaseBoundaryDomain::LoadedIndexRowwiseReduction,
                           "loaded_index_rowwise_reduction", std::nullopt};
    return plan;
  }
  if (features.dotOps > 0 && features.reduceOps == 0 &&
      features.loadedIndexDependentMemoryOps > 0 &&
      features.dotFlops <= options.tinyDotFlopsMax && features.loadOps > 0 &&
      features.storeOps > 0) {
    PhaseBoundaryPlan plan{PhaseBoundaryDomain::IndirectUnderfilledDot,
                           "indirect_underfilled_dot", std::nullopt};
    return plan;
  }
  return std::optional<PhaseBoundaryPlan>{};
}

llvm::Expected<std::optional<PhaseBoundaryPlan>>
PhaseBoundaryAnalysis::analyze(ModuleOp module,
                               const SimtAnchorPlan &anchorPlan,
                               const SimdSimtFeatureSummary &features,
                               const StagePartitionerOptions &options) const {
  auto plan = identifyPhaseBoundary(features, options);
  if (!plan)
    return std::optional<PhaseBoundaryPlan>{};
  auto structure = ProgramStructureAnalysis().analyze(module, anchorPlan);
  if (!structure)
    return structure.takeError();
  plan->rootOperations = std::move(structure->rootOperations);
  plan->localSimtAnchorRoots = std::move(structure->localSimtAnchorRoots);
  if (llvm::Error error = assignRootPhaseIds(*plan))
    return std::move(error);
  return std::optional<PhaseBoundaryPlan>{std::move(*plan)};
}

llvm::Expected<StagePartition>
StageBoundaryAnalysis::analyze(const PhaseBoundaryPlan &phasePlan,
                               const SimdSimtFeatureSummary &features,
                               const SimtAnchorPlan *anchorPlan) const {
  if (!phasePlan.hasOperationGraph() || !anchorPlan)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageBoundaryAnalysis requires PreparedTTIR ownership");
  StagePartition partition;
  switch (phasePlan.domain) {
  case PhaseBoundaryDomain::TriangularRecurrence:
    if (!phasePlan.triangularSolve)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "triangular PhaseBoundaryPlan has no recurrence facts");
    partition =
        partitionTriangular(features, *phasePlan.triangularSolve, &phasePlan);
    break;
  case PhaseBoundaryDomain::LoadedIndexRowwiseReduction:
    partition = partitionRowwise(features, &phasePlan);
    break;
  case PhaseBoundaryDomain::IndirectUnderfilledDot:
    partition = partitionIndirectDot(features, &phasePlan);
    break;
  }
  partition.domain = phasePlan.domainName;
  if (llvm::Error error =
          attachCompleteOperationOwnership(partition, phasePlan))
    return std::move(error);
  attachExactAnchorOwnership(partition, *anchorPlan);
  deriveStageLiveValues(partition);
  deriveLocalSimtScopeTraffic(partition, *anchorPlan);
  return partition;
}

llvm::Error StageFeatureAnalysis::analyze(StagePartition &partition) const {
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      StageModelFeatures &facts = stage.features;
      const double activeLaneRatio = facts.activeLaneRatio;
      facts = StageModelFeatures{};
      facts.activeLaneRatio = activeLaneRatio;
      llvm::DenseSet<Operation *> owned;
      for (Operation *root : stage.operations)
        collectOwnedOperationTree(root, owned);
      bool hasMemory = false;
      int64_t algorithmLoopCount = 0;
      for (Operation *operation : owned) {
        const llvm::StringRef name = operation->getName().getStringRef();
        if (name == "scf.for" || name == "scf.while") {
          facts.hasLoop = true;
          ++facts.loopBackedgeCount;
          if (!operation->hasAttr("ta.auto_blockify_v1.loop"))
            ++algorithmLoopCount;
          if (!operation->hasAttr("ta.auto_blockify_v1.loop") &&
              operation->getNumRegions() > 0 &&
              !operation->getRegion(0).empty()) {
            Block &body = operation->getRegion(0).front();
            const unsigned firstCarriedArgument = name == "scf.for" ? 1 : 0;
            for (unsigned argumentIndex = firstCarriedArgument;
                 argumentIndex < body.getNumArguments(); ++argumentIndex) {
              BlockArgument argument = body.getArgument(argumentIndex);
              if (argument.use_empty())
                continue;
              if (isPointerLikeType(argument.getType()) ||
                  isAddressOnlyLoopValue(argument))
                facts.hasPointerInduction = true;
              else
                facts.hasLoopCarriedDataDependency = true;
            }
            if (name == "scf.for" && body.getNumArguments() > 0 &&
                isAddressOnlyLoopValue(body.getArgument(0)))
              facts.hasPointerInduction = true;
          }
        }
        if (name == "scf.if" || name == "cf.cond_br") {
          ++facts.conditionalBranchCount;
          ++facts.divergentBranchCount;
        }
        if (name.contains("barrier") || name.contains("sync"))
          ++facts.synchronizationCount;
        if (name == "tt.load" || name == "tt.store" || name == "tt.gather" ||
            name.starts_with("tt.atomic")) {
          hasMemory = true;
          facts.hasIndirectMemory |=
              isLoadedIndexDependentMemoryOp(operation) ||
              name == "tt.gather" || name.starts_with("tt.atomic");
        }
        facts.hasReduction |=
            name == "tt.reduce" || name == "tt.scan" || name == "linalg.reduce";
        facts.hasDot |= name == "tt.dot" || name.contains("matmul") ||
                        name.contains("mmad");
        facts.hasConversionPack |=
            name == "arith.extf" || name == "arith.truncf" ||
            name == "arith.fptosi" || name == "arith.fptoui" ||
            name == "arith.sitofp" || name == "arith.uitofp" ||
            name == "tt.fp_to_fp" || name.contains("convert") ||
            name.contains("pack") || name.contains("unpack");
      }
      facts.hasContiguousMemory = hasMemory && !facts.hasIndirectMemory;
      if (algorithmLoopCount > 0 && stage.iterationCount > 1) {
        if (facts.hasLoopCarriedDataDependency)
          facts.parallelRecurrenceGroupCount = algorithmLoopCount;
        facts.loopBackedgeCount = 1;
        facts.conditionalBranchCount = std::max<int64_t>(
            facts.conditionalBranchCount > 0 ? 1 : 0,
            facts.conditionalBranchCount / algorithmLoopCount);
        facts.divergentBranchCount =
            std::max<int64_t>(facts.divergentBranchCount > 0 ? 1 : 0,
                              facts.divergentBranchCount / algorithmLoopCount);
      }
      if (!facts.isValid())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' has invalid features",
                                       stage.id.c_str());
    }
  }
  return llvm::Error::success();
}

llvm::Error StageKindClassifier::analyze(StagePartition &partition,
                                         int64_t tinyDotFlopsMax) const {
  if (!partition.operationOwnershipComplete)
    return llvm::Error::success();
  auto compatible = [](StageCostModelKind kind,
                       const StageModelFeatures &facts) {
    switch (kind) {
    case StageCostModelKind::LoopCarriedRecurrence:
      return facts.hasLoopCarriedDataDependency;
    case StageCostModelKind::IndependentPipelinedLoop:
      return facts.hasLoop && !facts.hasLoopCarriedDataDependency;
    case StageCostModelKind::RowwiseReduction:
      return facts.hasReduction;
    case StageCostModelKind::CubeRoofline:
    case StageCostModelKind::TinyCubeRoofline:
      return facts.hasDot;
    case StageCostModelKind::IndirectScalarMemory:
    case StageCostModelKind::IndirectGatherMemory:
      return facts.hasIndirectMemory;
    case StageCostModelKind::ContinuousTileMemory:
    case StageCostModelKind::ContinuousTileStore:
    case StageCostModelKind::ContinuousShortLoad:
    case StageCostModelKind::CachePolicyStore:
      return facts.hasContiguousMemory;
    case StageCostModelKind::ConversionPack:
      return facts.hasConversionPack;
    default:
      return true;
    }
  };
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      const StageModelFeatures &facts = stage.features;
      if (facts.hasDot && (facts.hasReduction || facts.hasIndirectMemory ||
                           facts.hasLoopCarriedDataDependency))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "requires_split: Stage '%s' owns incompatible dominant structures",
            stage.id.c_str());

      auto derive = [&]() {
        if (facts.hasDot)
          return stage.workload.dotFlops * stage.iterationCount <=
                         static_cast<double>(
                             std::max<int64_t>(1, tinyDotFlopsMax))
                     ? StageCostModelKind::TinyCubeRoofline
                     : StageCostModelKind::CubeRoofline;
        if (facts.hasReduction)
          return StageCostModelKind::RowwiseReduction;
        if (facts.hasConversionPack)
          return StageCostModelKind::ConversionPack;
        if (facts.hasLoop)
          return facts.hasLoopCarriedDataDependency
                     ? StageCostModelKind::LoopCarriedRecurrence
                     : StageCostModelKind::IndependentPipelinedLoop;
        if (facts.hasIndirectMemory)
          return StageCostModelKind::IndirectGatherMemory;
        if (facts.hasContiguousMemory)
          return stage.workload.storeBytes > 0.0 &&
                         stage.workload.loadBytes == 0.0
                     ? StageCostModelKind::ContinuousTileStore
                     : StageCostModelKind::ContinuousTileMemory;
        return StageCostModelKind::ScalarIssue;
      };
      if (!compatible(stage.costModelKind, facts))
        stage.costModelKind = derive();
      if (!compatible(stage.costModelKind, facts) ||
          (stage.costModelKind == StageCostModelKind::TinyCubeRoofline &&
           stage.workload.dotFlops * stage.iterationCount >
               static_cast<double>(std::max<int64_t>(1, tinyDotFlopsMax))))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage '%s' operation graph does not match StageCostModelKind '%s'",
            stage.id.c_str(),
            stringifyStageCostModel(stage.costModelKind).str().c_str());
      if (stage.costModelKind == StageCostModelKind::IndependentPipelinedLoop)
        stage.scheduleKind = StageScheduleKind::IndependentPipelined;
      else if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence)
        stage.scheduleKind = StageScheduleKind::LoopCarriedSerial;
    }
  }
  return llvm::Error::success();
}

llvm::Error StageWorkloadAnalysis::analyze(StagePartition &partition) const {
  if (!partition.operationOwnershipComplete)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StageWorkloadAnalysis requires complete operation ownership");
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      StageWorkload work;
      work.paysKernelSetup = stage.workload.paysKernelSetup;
      const int64_t loopCount = countAlgorithmLoops(stage);
      const int64_t fallbackLoopTripCount =
          loopCount > 0 ? std::max<int64_t>(1, stage.iterationCount / loopCount)
                        : 1;
      for (Operation *root : stage.operations)
        accumulateDynamicOperationTree(root, work, 1.0, fallbackLoopTripCount);
      recomputeIssueElements(work);
      stage.workload = std::move(work);
      makePerIteration(stage);
      if (!stage.workload.isFiniteAndNonNegative())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage '%s' has invalid operation-derived workload",
            stage.id.c_str());
    }
  }
  return llvm::Error::success();
}

llvm::Error
StagePartitionVerifier::verify(const StagePartition &partition) const {
  if (partition.phases.empty())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "StagePartition has no Phase");
  llvm::StringSet<> phaseIds;
  llvm::StringSet<> stageIds;
  llvm::DenseSet<Operation *> ownedOperations;
  llvm::DenseSet<unsigned> ownedAnchors;
  for (const LogicalPhase &phase : partition.phases) {
    if (phase.id.empty() || !phaseIds.insert(phase.id).second)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "StagePartition has duplicate Phase id");
    if (phase.stages.empty())
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Phase '%s' has no Stage",
                                     phase.id.c_str());
    for (const LogicalStage &stage : phase.stages) {
      if (stage.id.empty() || !stageIds.insert(stage.id).second)
        return llvm::createStringError(std::errc::invalid_argument,
                                       "StagePartition has duplicate Stage id");
      if (stage.iterationCount < 1)
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' has invalid iteration count",
                                       stage.id.c_str());
      if (stage.localSimtMaterializable &&
          partition.operationOwnershipComplete && stage.operations.empty())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "materializable Stage '%s' has no operation ownership",
            stage.id.c_str());
      if (stage.localSimtMaterializable &&
          partition.operationOwnershipComplete &&
          stage.simtAnchorIndices.empty())
        return llvm::createStringError(
            std::errc::invalid_argument,
            "materializable Stage '%s' has no exact SIMT anchor ownership",
            stage.id.c_str());
      if (partition.operationOwnershipComplete)
        for (unsigned index : stage.simtAnchorIndices)
          if (!ownedAnchors.insert(index).second)
            return llvm::createStringError(
                std::errc::invalid_argument,
                "StagePartition SIMT anchor ownership overlaps");
      if (partition.operationOwnershipComplete)
        for (Operation *operation : stage.operations)
          if (!operation || !ownedOperations.insert(operation).second)
            return llvm::createStringError(
                std::errc::invalid_argument,
                "StagePartition operation ownership overlaps");
    }
  }
  if (partition.operationOwnershipComplete &&
      static_cast<int64_t>(ownedOperations.size()) !=
          partition.modeledOperationCount)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition operation ownership is incomplete");
  if (!partition.operationOwnershipComplete)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition requires complete TTIR operation ownership");
  return llvm::Error::success();
}

llvm::Error
StageModeLegalityAnalysis::analyze(StagePartition &partition,
                                   int64_t maximumSuperblockFactor,
                                   bool scopeSuperblockMaterializable) const {
  const int64_t maximum = std::clamp<int64_t>(maximumSuperblockFactor, 1, 4);
  // Local and whole-kernel SuperBlock candidates consume the same SIMT warp
  // resources.  Do not regenerate F4 here after evaluateStageModel has
  // already reduced the target maximum to F2 for num_warps=32 (or to F1 for
  // a smaller runtime grid).
  const int64_t localMaximum = scopeSuperblockMaterializable ? maximum : 1;
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      stage.simdLegal = true;
      stage.simtLegal = true;
      stage.legalSimtFactors = {1};
      // A pure-SIMT SuperBlock factor is a whole-kernel schedule, not a
      // recurrence-only annotation.  Every SIMT Stage must therefore expose
      // the same factor candidates; KernelRouteSolver keeps the chosen factor
      // uniform.  Local mixed scopes stay restricted by localSimtFactors
      // (F1 unless Scope SuperBlock materialization is explicitly available).
      if (maximum >= 2)
        stage.legalSimtFactors.push_back(2);
      if (maximum >= 4)
        stage.legalSimtFactors.push_back(4);
      if (stage.localSimtMaterializable) {
        // The ABI-v2 scope materializer batches complete logical programs
        // around this Stage.  F2/F4 therefore does not require multiple
        // recurrence groups inside one logical program; that older W2/W4
        // interpretation was only warp widening, not a SuperBlock.
        stage.localSimtFactors = {1};
        if (scopeSuperblockMaterializable)
          for (int64_t factor : {2, 4})
            if (factor <= localMaximum)
              stage.localSimtFactors.push_back(factor);
      }
      if (stage.localSimtMaterializable &&
          (stage.localSimtFactors.empty() ||
           llvm::any_of(stage.localSimtFactors, [&](int64_t factor) {
             return factor < 1 || factor > localMaximum ||
                    (factor != 1 && factor != 2 && factor != 4);
           })))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "local SIMT factors are invalid for Stage '%s'", stage.id.c_str());
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::optional<StagePartition>>
StagePartitioner::partition(ModuleOp module, const SimtAnchorPlan &anchorPlan,
                            const SimdSimtFeatureSummary &features,
                            const StagePartitionerOptions &options) const {
  auto phasePlan =
      PhaseBoundaryAnalysis().analyze(module, anchorPlan, features, options);
  if (!phasePlan)
    return phasePlan.takeError();
  if (!*phasePlan)
    return std::optional<StagePartition>{};
  auto result =
      StageBoundaryAnalysis().analyze(**phasePlan, features, &anchorPlan);
  if (!result)
    return result.takeError();
  StageWorkloadAnalysis workloadAnalysis;
  if (llvm::Error error = workloadAnalysis.analyze(*result))
    return std::move(error);
  StageFeatureAnalysis featureAnalysis;
  if (llvm::Error error = featureAnalysis.analyze(*result))
    return std::move(error);
  if (llvm::Error error =
          StageKindClassifier().analyze(*result, options.tinyDotFlopsMax))
    return std::move(error);
  StageModeLegalityAnalysis legalityAnalysis;
  if (llvm::Error error =
          legalityAnalysis.analyze(*result, options.maximumSuperblockFactor,
                                   options.scopeSuperblockMaterializable))
    return std::move(error);
  if (llvm::Error error = StagePartitionVerifier().verify(*result))
    return std::move(error);
  return std::optional<StagePartition>{std::move(*result)};
}
