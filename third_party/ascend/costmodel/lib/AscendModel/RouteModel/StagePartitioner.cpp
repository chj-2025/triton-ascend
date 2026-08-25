//===- StagePartitioner.cpp - Build semantic Phase/Stage IR -------------===//

#include "AscendModel/RouteModel/StagePartitioner.h"

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

static double mapValue(const llvm::StringMap<int64_t> &map,
                       llvm::StringRef name) {
  auto iterator = map.find(name);
  return iterator == map.end()
             ? 0.0
             : static_cast<double>(std::max<int64_t>(0, iterator->second));
}

static StageWorkload buildWorkload(const SimtAnchorFeatureSummary &features) {
  StageWorkload work;
  const std::pair<llvm::StringLiteral, llvm::StringLiteral> names[] = {
      {"f32.add", "add"},     {"f32.sub", "sub"}, {"f32.mul", "mul"},
      {"f32.div", "div"},     {"f32.max", "max"}, {"f32.abs", "abs"},
      {"f32.exp", "exp"},     {"f32.log", "log"}, {"convert.cast", "cast"},
      {"f32.clamp", "clamp"},
  };
  for (const auto &[profileName, featureName] : names) {
    const double elements = mapValue(features.opElements, featureName);
    if (elements > 0.0)
      work.operationElements[profileName] = elements;
  }
  work.scalarOperations = mapValue(features.weightedOps, "scalar");
  work.loadBytes = features.loadBytes;
  work.storeBytes = features.storeBytes;
  work.loadWarpInstructions =
      static_cast<double>(features.loadWarpInstructions);
  work.storeWarpInstructions =
      static_cast<double>(features.storeWarpInstructions);
  work.predicateElements = static_cast<double>(std::max<int64_t>(
      features.predicateLaneEvaluations, features.predicateElements));
  work.shuffleLaneSteps = static_cast<double>(features.shuffleLaneSteps);
  work.dotFlops = static_cast<double>(features.dotFlops);
  return work;
}

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

static double consume(double &remaining, double requested) {
  const double value =
      std::min(std::max(0.0, remaining), std::max(0.0, requested));
  remaining -= value;
  return value;
}

static StageWorkload consumeExact(StageWorkload &remaining,
                                  const StageWorkload &requested) {
  StageWorkload result;
  result.scalarOperations =
      consume(remaining.scalarOperations, requested.scalarOperations);
  result.loadBytes = consume(remaining.loadBytes, requested.loadBytes);
  result.storeBytes = consume(remaining.storeBytes, requested.storeBytes);
  result.loadWarpInstructions =
      consume(remaining.loadWarpInstructions, requested.loadWarpInstructions);
  result.storeWarpInstructions =
      consume(remaining.storeWarpInstructions, requested.storeWarpInstructions);
  result.predicateElements =
      consume(remaining.predicateElements, requested.predicateElements);
  result.shuffleLaneSteps =
      consume(remaining.shuffleLaneSteps, requested.shuffleLaneSteps);
  result.dotFlops = consume(remaining.dotFlops, requested.dotFlops);
  result.estimatedSpillTransactions =
      consume(remaining.estimatedSpillTransactions,
              requested.estimatedSpillTransactions);
  for (const auto &[name, elements] : requested.operationElements) {
    double &available = remaining.operationElements[name];
    const double owned = consume(available, elements);
    if (owned > 0.0)
      result.operationElements[name] = owned;
  }
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
}

static StageWorkload takeScalarAndPredicate(StageWorkload &remaining) {
  StageWorkload result;
  result.scalarOperations = std::exchange(remaining.scalarOperations, 0.0);
  result.predicateElements = std::exchange(remaining.predicateElements, 0.0);
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
}

static StageWorkload takeLoads(StageWorkload &remaining) {
  StageWorkload result;
  result.loadBytes = std::exchange(remaining.loadBytes, 0.0);
  result.loadWarpInstructions =
      std::exchange(remaining.loadWarpInstructions, 0.0);
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
}

static StageWorkload takeStores(StageWorkload &remaining) {
  StageWorkload result;
  result.storeBytes = std::exchange(remaining.storeBytes, 0.0);
  result.storeWarpInstructions =
      std::exchange(remaining.storeWarpInstructions, 0.0);
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
}

static StageWorkload takeDot(StageWorkload &remaining) {
  StageWorkload result;
  result.dotFlops = std::exchange(remaining.dotFlops, 0.0);
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
}

static void moveOperation(StageWorkload &from, StageWorkload &to,
                          llvm::StringRef name) {
  auto iterator = from.operationElements.find(name);
  if (iterator == from.operationElements.end())
    return;
  to.operationElements[name] += iterator->second;
  from.operationElements.erase(iterator);
}

static StageWorkload takeAllOperations(StageWorkload &remaining) {
  StageWorkload result;
  result.operationElements = std::move(remaining.operationElements);
  remaining.operationElements.clear();
  recomputeIssueElements(result);
  recomputeIssueElements(remaining);
  return result;
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

static LogicalStage makeStage(llvm::StringRef id, llvm::StringRef description,
                              StageCostModelKind kind,
                              StageScheduleKind schedule, int64_t iterations,
                              StageWorkload workload) {
  LogicalStage stage;
  stage.id = id.str();
  stage.description = description.str();
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
                     llvm::StringRef description, LogicalStage stage) {
  LogicalPhase phase;
  phase.id = id.str();
  phase.description = description.str();
  phase.stages.push_back(std::move(stage));
  partition.phases.push_back(std::move(phase));
}

static bool operationTreeContainsName(Operation *root, llvm::StringRef name);

static bool hasWork(const StageWorkload &work) {
  if (work.scalarOperations > 0.0 || work.loadBytes > 0.0 ||
      work.storeBytes > 0.0 || work.predicateElements > 0.0 ||
      work.shuffleLaneSteps > 0.0 || work.dotFlops > 0.0 ||
      work.estimatedSpillTransactions > 0.0)
    return true;
  return llvm::any_of(work.operationElements,
                      [](const auto &entry) { return entry.second > 0.0; });
}

static void prependAutoBlockifyStages(StagePartition &partition,
                                      StageWorkload &remaining,
                                      const SimdSimtFeatureSummary &features,
                                      bool operationGraphHasAutoBlockify) {
  if (!features.autoBlockifyV1Applied && !operationGraphHasAutoBlockify)
    return;
  StageWorkload dispatch;
  dispatch.scalarOperations =
      consume(remaining.scalarOperations,
              static_cast<double>(features.autoBlockifyV1ScheduleOpCount));
  recomputeIssueElements(dispatch);
  LogicalPhase phase;
  phase.id = "auto_blockify_dispatch";
  phase.description = "AutoBlockify V1 physical/logical program dispatch";
  phase.stages.push_back(makeStage(
      "physical_program_dispatch", "Map physical PID to logical block range",
      StageCostModelKind::AutoBlockifyDispatch, StageScheduleKind::StraightLine,
      1, std::move(dispatch)));
  if (features.autoBlockifyV1LoopCount > 0 || operationGraphHasAutoBlockify) {
    StageWorkload loop;
    loop.scalarOperations =
        consume(remaining.scalarOperations,
                static_cast<double>(features.autoBlockifyV1LoopCount));
    recomputeIssueElements(loop);
    phase.stages.push_back(makeStage(
        "logical_program_loop", "Iterate AutoBlockify V1 logical programs",
        StageCostModelKind::AutoBlockifyLoop,
        StageScheduleKind::IndependentPipelined,
        std::max<int64_t>(1, features.autoBlockifyV1LoopCount),
        std::move(loop)));
  }
  partition.phases.push_back(std::move(phase));
}

static StagePartition
partitionTriangular(const SimdSimtFeatureSummary &features,
                    const TriangularSolveFacts &facts,
                    const PhaseBoundaryPlan *operationGraphPlan) {
  StagePartition partition;
  partition.domain = "triangular_recurrence";
  StageWorkload remaining = buildKernelStageWorkload(features);
  StageWorkload anchor = buildWorkload(features.simtAnchors);
  anchor = consumeExact(remaining, anchor);

  const bool graphHasAutoBlockify =
      operationGraphPlan && llvm::is_contained(operationGraphPlan->rootPhaseIds,
                                               "auto_blockify_dispatch");
  prependAutoBlockifyStages(partition, remaining, features,
                            graphHasAutoBlockify);

  StageWorkload head = takeScalarAndPredicate(remaining);
  mergeWorkload(head, takeAllOperations(remaining));
  head.paysKernelSetup = true;
  addPhase(partition, "head", "Tile offset and triangular mask setup",
           withControl(makeStage("head_index_mask",
                                 "Scalar indices and triangular masks",
                                 StageCostModelKind::PredicateMask,
                                 StageScheduleKind::StraightLine, 1,
                                 std::move(head)),
                       features.conditionalBranchCount -
                           features.simtAnchors.conditionalBranchCount,
                       features.divergentBranchCount -
                           features.simtAnchors.divergentBranchCount,
                       features.activeLaneRatio));

  StageWorkload loads = takeLoads(remaining);
  const bool graphHasDiagonalLoad =
      operationGraphPlan &&
      llvm::is_contained(operationGraphPlan->rootPhaseIds, "diagonal_load");
  if (hasWork(loads) || graphHasDiagonalLoad)
    addPhase(partition, "diagonal_load", "Load diagonal tile data",
             makeStage("load_diagonal_tiles", "Continuous diagonal tile loads",
                       StageCostModelKind::ContinuousTileMemory,
                       StageScheduleKind::IndependentPipelined, 1,
                       std::move(loads)));

  const int64_t recurrenceIterations =
      std::max<int64_t>(1, facts.recurrenceLoopCount);
  // The feature-summary fallback has no owned operation graph from which to
  // count sibling loops.  Derive the number of independent 16x16 recurrence
  // groups from the structural triangular facts.  Production operation-graph
  // analysis below recomputes the same field directly from owned loops.
  LogicalStage recurrence = asLocalSIMT(withControl(
      makeStage("diagonal_inverse_recurrence",
                "Predicate, short reduction and recurrent state update",
                StageCostModelKind::LoopCarriedRecurrence,
                StageScheduleKind::LoopCarriedSerial, recurrenceIterations,
                std::move(anchor)),
      features.simtAnchors.conditionalBranchCount,
      features.simtAnchors.divergentBranchCount,
      features.simtAnchors.activeLaneRatio));
  const int64_t iterationsPerGroup =
      std::max<int64_t>(1, facts.blockRows - facts.recurrenceStartRow);
  recurrence.features.parallelRecurrenceGroupCount = std::max<int64_t>(
      1, (recurrenceIterations + iterationsPerGroup - 1) / iterationsPerGroup);
  addPhase(partition, "diagonal_inverse",
           "Loop-carried blockwise triangular recurrence",
           std::move(recurrence));

  StageWorkload dot = takeDot(remaining);
  StageWorkload stores = takeStores(remaining);
  mergeWorkload(stores, std::move(remaining));
  bool graphHasDot = false;
  bool graphHasStore = false;
  if (operationGraphPlan)
    for (auto indexedRoot :
         llvm::enumerate(operationGraphPlan->rootOperations)) {
      if (operationGraphPlan->rootPhaseIds[indexedRoot.index()] !=
          "merge_store")
        continue;
      graphHasDot |= operationTreeContainsName(indexedRoot.value(), "tt.dot");
      graphHasStore |=
          operationTreeContainsName(indexedRoot.value(), "tt.store");
    }
  if (hasWork(dot) || hasWork(stores) || graphHasDot || graphHasStore) {
    LogicalPhase mergeStore;
    mergeStore.id = "merge_store";
    mergeStore.description = "Dense off-diagonal merge and result store";
    if (hasWork(dot) || graphHasDot)
      mergeStore.stages.push_back(makeStage(
          "dense_dot_tail", "Dense dot tail preserved for Cube",
          StageCostModelKind::CubeRoofline,
          StageScheduleKind::IndependentPipelined,
          std::max<int64_t>(1, facts.denseDotTailOps), std::move(dot)));
    if (hasWork(stores) || graphHasStore)
      mergeStore.stages.push_back(
          makeStage("store_inverse_tile", "Continuous result tile store",
                    StageCostModelKind::ContinuousTileStore,
                    StageScheduleKind::StraightLine, 1, std::move(stores)));
    partition.phases.push_back(std::move(mergeStore));
  }
  return partition;
}

static StagePartition
partitionRowwise(const SimdSimtFeatureSummary &features,
                 const PhaseBoundaryPlan *operationGraphPlan) {
  StagePartition partition;
  partition.domain = "loaded_index_rowwise_reduction";
  StageWorkload remaining = buildKernelStageWorkload(features);
  const bool graphHasAutoBlockify =
      operationGraphPlan && llvm::is_contained(operationGraphPlan->rootPhaseIds,
                                               "auto_blockify_dispatch");
  prependAutoBlockifyStages(partition, remaining, features,
                            graphHasAutoBlockify);

  StageWorkload index = takeScalarAndPredicate(remaining);
  index.paysKernelSetup = true;
  addPhase(partition, "row_dispatch", "Resolve token and row indices",
           withControl(
               makeStage("row_index_generation", "Scalar row index and masks",
                         StageCostModelKind::IndexGeneration,
                         StageScheduleKind::StraightLine, 1, std::move(index)),
               features.conditionalBranchCount, features.divergentBranchCount,
               features.activeLaneRatio));

  addPhase(partition, "row_load", "Loaded-index-dependent row access",
           asLocalSIMT(makeStage("indirect_row_gather",
                                 "Gather the selected row tile",
                                 StageCostModelKind::IndirectGatherMemory,
                                 StageScheduleKind::PartiallyDependent, 1,
                                 takeLoads(remaining))));

  StageWorkload reduction;
  reduction.shuffleLaneSteps = std::exchange(remaining.shuffleLaneSteps, 0.0);
  moveOperation(remaining, reduction, "f32.max");
  recomputeIssueElements(reduction);
  const int64_t reductionIterations =
      std::max<int64_t>(1, features.staticLoopTripCountMax);
  addPhase(partition, "row_reduction", "Reduce each selected row",
           makeStage("rowwise_reduction", "Row-local reduction tree",
                     StageCostModelKind::RowwiseReduction,
                     StageScheduleKind::PartiallyDependent, reductionIterations,
                     std::move(reduction)));

  StageWorkload convert = takeStores(remaining);
  mergeWorkload(convert, takeAllOperations(remaining));
  mergeWorkload(convert, std::move(remaining));
  addPhase(partition, "convert_store", "Scale, convert, pack and store",
           makeStage("conversion_pack_store", "Conversion and packed output",
                     StageCostModelKind::ConversionPack,
                     StageScheduleKind::IndependentPipelined,
                     reductionIterations, std::move(convert)));
  return partition;
}

static StagePartition
partitionIndirectDot(const SimdSimtFeatureSummary &features,
                     const PhaseBoundaryPlan *operationGraphPlan) {
  StagePartition partition;
  partition.domain = "indirect_underfilled_dot";
  StageWorkload remaining = buildKernelStageWorkload(features);
  const bool graphHasAutoBlockify =
      operationGraphPlan && llvm::is_contained(operationGraphPlan->rootPhaseIds,
                                               "auto_blockify_dispatch");
  prependAutoBlockifyStages(partition, remaining, features,
                            graphHasAutoBlockify);

  StageWorkload index = takeScalarAndPredicate(remaining);
  mergeWorkload(index, takeAllOperations(remaining));
  index.paysKernelSetup = true;
  addPhase(partition, "index_setup", "Generate gather indices and masks",
           withControl(
               makeStage("index_generation", "Index and predicate generation",
                         StageCostModelKind::IndexGeneration,
                         StageScheduleKind::StraightLine, 1, std::move(index)),
               features.conditionalBranchCount, features.divergentBranchCount,
               features.activeLaneRatio));

  addPhase(partition, "gather_tiles", "Gather dot input tiles",
           asLocalSIMT(makeStage("indirect_tile_gather",
                                 "Loaded-index-dependent operand gathers",
                                 StageCostModelKind::IndirectGatherMemory,
                                 StageScheduleKind::PartiallyDependent, 1,
                                 takeLoads(remaining))));

  addPhase(partition, "dot", "Under-filled matrix product",
           makeStage("tiny_cube_dot", "Small dot with Cube underfill",
                     StageCostModelKind::TinyCubeRoofline,
                     StageScheduleKind::IndependentPipelined, 1,
                     takeDot(remaining)));

  StageWorkload store = takeStores(remaining);
  mergeWorkload(store, std::move(remaining));
  addPhase(partition, "output_store", "Write dot result",
           makeStage("store_dot_result", "Continuous result store",
                     StageCostModelKind::ContinuousTileStore,
                     StageScheduleKind::StraightLine, 1, std::move(store)));
  return partition;
}

namespace {

/// Semantic stage kinds used by the unified state machine.  These are derived
/// purely from operation semantics (op names, loaded-index dependencies,
/// loop-carried recurrence) and are intentionally independent of any
/// business-scenario domain such as triangular/rowwise/indirect-dot.
enum class SemanticStage {
  AutoBlockifyDispatch,
  IndexSetup,   // 准备索引: scalar/index/predicate generation
  Gather,       // 取料: loaded-index-dependent or plain tile load
  Recurrence,   // 递推加工: loop-carried triangular recurrence
  Reduction,    // 归约加工: rowwise reduction
  Dot,          // 矩阵加工: matrix product
  Histogram,    // 直方图统计: tt.histogram
  Atomic,       // 原子操作: tt.atomic_rmw / tt.atomic_cas
  Scatter,      // 散列写回: indirect/scatter store
  ConvertStore, // 出货: conversion + pack + regular store
};

static int semanticStagePriority(SemanticStage stage) {
  switch (stage) {
  case SemanticStage::AutoBlockifyDispatch:
    return -1;
  case SemanticStage::IndexSetup:
    return 0;
  case SemanticStage::Gather:
    return 1;
  case SemanticStage::Recurrence:
    return 2;
  case SemanticStage::Reduction:
    return 3;
  case SemanticStage::Dot:
    return 4;
  case SemanticStage::Histogram:
    return 5;
  case SemanticStage::Atomic:
    return 6;
  case SemanticStage::Scatter:
    return 7;
  case SemanticStage::ConvertStore:
    return 8;
  }
  return 0;
}

/// Whether a stage is a storage/output stage.  Storage stages (scatter,
/// convert_store) represent the kernel's output phase and must not regress
/// back to a compute stage once entered.
static bool isStorageStage(SemanticStage stage) {
  return stage == SemanticStage::Scatter ||
         stage == SemanticStage::ConvertStore;
}

/// Whether a stage is a computation stage.  Computation stages (gather,
/// recurrence, reduction, dot, histogram, atomic) may appear in any order
/// relative to each other — the order is determined by the IR, not by a
/// hardcoded priority.
static bool isComputeStage(SemanticStage stage) {
  return stage == SemanticStage::Gather ||
         stage == SemanticStage::Recurrence ||
         stage == SemanticStage::Reduction ||
         stage == SemanticStage::Dot ||
         stage == SemanticStage::Histogram ||
         stage == SemanticStage::Atomic;
}

/// Decide whether the state machine should advance from `current` to
/// `classified`.  Unlike the old fixed-priority approach, this allows
/// computation stages to appear in any order (e.g., histogram before atomic,
/// or gather → histogram → gather).  Two hard constraints:
///   1. storage stages must not regress back to computation;
///   2. IndexSetup (preparation) cannot be re-entered once left — subsequent
///      arith/type ops belong to the current stage, not a new preparation phase.
static bool shouldAdvanceStage(SemanticStage current, SemanticStage classified) {
  if (classified == current)
    return false; // same semantic — stay in current stage
  if (isStorageStage(current) && !isStorageStage(classified))
    return false; // storage phase cannot regress to computation
  // IndexSetup is a preparation phase (index/predicate generation).  Once
  // the kernel enters a computation or storage stage, subsequent arith/type
  // conversion ops are part of the current stage, not a new preparation
  // phase.  Allowing re-entry would produce non-contiguous stage ownership.
  if (current != SemanticStage::IndexSetup &&
      classified == SemanticStage::IndexSetup)
    return false; // cannot regress back to preparation after leaving it
  return true; // semantic changed and no regression — advance
}

static llvm::StringRef semanticStageId(SemanticStage stage) {
  switch (stage) {
  case SemanticStage::AutoBlockifyDispatch:
    return "auto_blockify_dispatch";
  case SemanticStage::IndexSetup:
    return "index_setup";
  case SemanticStage::Gather:
    return "gather";
  case SemanticStage::Recurrence:
    return "recurrence";
  case SemanticStage::Reduction:
    return "reduction";
  case SemanticStage::Dot:
    return "dot";
  case SemanticStage::Histogram:
    return "histogram";
  case SemanticStage::Atomic:
    return "atomic";
  case SemanticStage::Scatter:
    return "scatter";
  case SemanticStage::ConvertStore:
    return "convert_store";
  }
  return "index_setup";
}

static llvm::StringRef semanticStageDescription(llvm::StringRef stageId) {
  return llvm::StringSwitch<llvm::StringRef>(stageId)
      .Case("index_setup", "Scalar index and predicate generation")
      .Case("gather", "Loaded-index-dependent or tile data gather")
      .Case("recurrence", "Loop-carried blockwise recurrence")
      .Case("reduction", "Row-local reduction tree")
      .Case("dot", "Matrix product")
      .Case("histogram", "Histogram bin counting")
      .Case("atomic", "Atomic read-modify-write")
      .Case("scatter", "Indirect or scatter store")
      .Case("convert_store", "Conversion, pack and regular store")
      .Default("Semantic stage");
}

static StageCostModelKind semanticStageCostModelKind(llvm::StringRef stageId) {
  return llvm::StringSwitch<StageCostModelKind>(stageId)
      .Case("index_setup", StageCostModelKind::IndexGeneration)
      .Case("gather", StageCostModelKind::IndirectGatherMemory)
      .Case("recurrence", StageCostModelKind::LoopCarriedRecurrence)
      .Case("reduction", StageCostModelKind::RowwiseReduction)
      .Case("dot", StageCostModelKind::CubeRoofline)
      .Case("histogram", StageCostModelKind::IndirectGatherMemory)
      .Case("atomic", StageCostModelKind::IndirectGatherMemory)
      .Case("scatter", StageCostModelKind::ContinuousTileStore)
      .Case("convert_store", StageCostModelKind::ContinuousTileStore)
      .Default(StageCostModelKind::ScalarIssue);
}

static StageScheduleKind semanticStageSchedule(llvm::StringRef stageId) {
  return llvm::StringSwitch<StageScheduleKind>(stageId)
      .Case("gather", StageScheduleKind::PartiallyDependent)
      .Case("recurrence", StageScheduleKind::LoopCarriedSerial)
      .Case("dot", StageScheduleKind::IndependentPipelined)
      .Case("reduction", StageScheduleKind::PartiallyDependent)
      .Case("histogram", StageScheduleKind::PartiallyDependent)
      .Case("atomic", StageScheduleKind::PartiallyDependent)
      .Case("scatter", StageScheduleKind::PartiallyDependent)
      .Case("convert_store", StageScheduleKind::IndependentPipelined)
      .Default(StageScheduleKind::StraightLine);
}

} // namespace

/// Unified semantic Stage partitioner.  Unlike the domain-specific templates
/// above, this derives Stage boundaries purely from operation semantics via
/// assignRootStageIdsBySemantics.  When an operation graph is available the
/// Stage workload/costModelKind/scheduleKind are filled by downstream
/// analyses (StageWorkloadAnalysis, StageKindClassifier) from exact owned
/// operations.  The feature-summary fallback path synthesizes a best-effort
/// Stage skeleton from aggregate feature counts so that unmatched kernels
/// still receive a route decision instead of falling through to legacy.
static StagePartition
partitionBySemantics(const SimdSimtFeatureSummary &features,
                     const PhaseBoundaryPlan *operationGraphPlan) {
  StagePartition partition;
  partition.domain = "semantic";
  StageWorkload remaining = buildKernelStageWorkload(features);

  const bool graphHasAutoBlockify =
      operationGraphPlan && llvm::is_contained(operationGraphPlan->rootPhaseIds,
                                               "auto_blockify_dispatch");
  prependAutoBlockifyStages(partition, remaining, features,
                            graphHasAutoBlockify);

  if (operationGraphPlan && operationGraphPlan->hasOperationGraph()) {
    // Operation-graph path: collect unique Stage ids in execution order and
    // create one Phase per Stage.  Workload is intentionally empty here;
    // StageWorkloadAnalysis accumulates it from exact owned operations.
    std::vector<std::string> stageIds;
    llvm::StringSet<> seen;
    for (const std::string &phaseId : operationGraphPlan->rootPhaseIds) {
      if (phaseId == "auto_blockify_dispatch")
        continue;
      if (seen.insert(phaseId).second)
        stageIds.push_back(phaseId);
    }

    const int64_t iterationCount =
        std::max<int64_t>(1, features.staticLoopTripCountMax);
    bool firstRealStage = true;
    for (const std::string &stageId : stageIds) {
      LogicalStage stage;
      stage.id = stageId;
      stage.description = semanticStageDescription(stageId).str();
      stage.costModelKind = semanticStageCostModelKind(stageId);
      stage.scheduleKind = semanticStageSchedule(stageId);
      stage.iterationCount = iterationCount;
      if (firstRealStage) {
        stage.workload.paysKernelSetup = true;
        firstRealStage = false;
      }
      addPhase(partition, stageId, stage.description, std::move(stage));
    }
    return partition;
  }

  // Feature-summary fallback: synthesize a best-effort Stage skeleton from
  // aggregate feature counts.  This keeps the costmodel applicable to kernels
  // that have no operation graph without falling through to legacy.
  const int64_t iterationCount =
      std::max<int64_t>(1, features.staticLoopTripCountMax);
  bool firstRealStage = true;

  // Index/predicate/scalar setup is always present (kernel setup work).
  StageWorkload index = takeScalarAndPredicate(remaining);
  mergeWorkload(index, takeAllOperations(remaining));
  index.paysKernelSetup = true;
  LogicalStage indexStage = makeStage("index_setup",
                                       semanticStageDescription("index_setup"),
                                       StageCostModelKind::IndexGeneration,
                                       StageScheduleKind::StraightLine, 1,
                                       std::move(index));
  indexStage.features.conditionalBranchCount =
      features.conditionalBranchCount;
  indexStage.features.divergentBranchCount = features.divergentBranchCount;
  indexStage.features.activeLaneRatio = features.activeLaneRatio;
  addPhase(partition, "index_setup", indexStage.description,
           std::move(indexStage));
  firstRealStage = false;

  if (features.loadedIndexDependentMemoryOps > 0 || features.loadOps > 0) {
    LogicalStage gather = asLocalSIMT(makeStage(
        "gather", semanticStageDescription("gather"),
        StageCostModelKind::IndirectGatherMemory,
        StageScheduleKind::PartiallyDependent, 1, takeLoads(remaining)));
    addPhase(partition, "gather", gather.description, std::move(gather));
  }

  if (!features.simtAnchors.triangularSolves.empty()) {
    const TriangularSolveFacts &facts =
        features.simtAnchors.triangularSolves.front();
    StageWorkload anchor = buildWorkload(features.simtAnchors);
    anchor = consumeExact(remaining, anchor);
    const int64_t recurrenceIterations =
        std::max<int64_t>(1, facts.recurrenceLoopCount);
    LogicalStage recurrence = asLocalSIMT(withControl(
        makeStage("recurrence", semanticStageDescription("recurrence"),
                  StageCostModelKind::LoopCarriedRecurrence,
                  StageScheduleKind::LoopCarriedSerial, recurrenceIterations,
                  std::move(anchor)),
        features.simtAnchors.conditionalBranchCount,
        features.simtAnchors.divergentBranchCount,
        features.simtAnchors.activeLaneRatio));
    const int64_t iterationsPerGroup = std::max<int64_t>(
        1, facts.blockRows - facts.recurrenceStartRow);
    recurrence.features.parallelRecurrenceGroupCount = std::max<int64_t>(
        1, (recurrenceIterations + iterationsPerGroup - 1) / iterationsPerGroup);
    addPhase(partition, "recurrence", recurrence.description,
             std::move(recurrence));
  }

  if (features.reduceOps > 0) {
    StageWorkload reduction;
    reduction.shuffleLaneSteps = std::exchange(remaining.shuffleLaneSteps, 0.0);
    moveOperation(remaining, reduction, "f32.max");
    recomputeIssueElements(reduction);
    addPhase(partition, "reduction",
             semanticStageDescription("reduction").str(),
             makeStage("reduction", semanticStageDescription("reduction"),
                       StageCostModelKind::RowwiseReduction,
                       StageScheduleKind::PartiallyDependent, iterationCount,
                       std::move(reduction)));
  }

  if (features.dotOps > 0) {
    addPhase(partition, "dot", semanticStageDescription("dot").str(),
             makeStage("dot", semanticStageDescription("dot"),
                       StageCostModelKind::CubeRoofline,
                       StageScheduleKind::IndependentPipelined, 1,
                       takeDot(remaining)));
  }

  if (features.histogramOps > 0) {
    LogicalStage histogram = asLocalSIMT(makeStage(
        "histogram", semanticStageDescription("histogram"),
        semanticStageCostModelKind("histogram"),
        semanticStageSchedule("histogram"), 1, takeAllOperations(remaining)));
    addPhase(partition, "histogram", histogram.description,
             std::move(histogram));
    llvm::errs() << "[COSTMODEL]   partitionBySemantics(fallback): added"
                 << " 'histogram' stage (histogramOps=" << features.histogramOps
                 << ")\n";
  }

  if (features.atomicOps > 0) {
    LogicalStage atomic = asLocalSIMT(makeStage(
        "atomic", semanticStageDescription("atomic"),
        semanticStageCostModelKind("atomic"),
        semanticStageSchedule("atomic"), 1, takeAllOperations(remaining)));
    addPhase(partition, "atomic", atomic.description, std::move(atomic));
    llvm::errs() << "[COSTMODEL]   partitionBySemantics(fallback): added"
                 << " 'atomic' stage (atomicOps=" << features.atomicOps
                 << ")\n";
  }

  StageWorkload convert = takeStores(remaining);
  mergeWorkload(convert, takeAllOperations(remaining));
  mergeWorkload(convert, std::move(remaining));
  if (hasWork(convert)) {
    addPhase(partition, "convert_store",
             semanticStageDescription("convert_store").str(),
             makeStage("convert_store", semanticStageDescription("convert_store"),
                       StageCostModelKind::ContinuousTileStore,
                       StageScheduleKind::IndependentPipelined, iterationCount,
                       std::move(convert)));
  }
  return partition;
}

static bool anchorMatchesStage(const SimtAnchorDescriptor &anchor,
                               const LogicalStage &stage) {
  if (!anchor.materializable || !stage.localSimtMaterializable)
    return false;
  if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence)
    return anchor.kind == SimtAnchorKind::TriangularSolveLoop;
  if (stage.costModelKind == StageCostModelKind::IndirectGatherMemory ||
      stage.costModelKind == StageCostModelKind::IndirectScalarMemory) {
    // Gather stage matches gather/loaded-index/histogram/atomic anchors.
    // Histogram and atomic are irregular-memory SIMT mechanisms that share
    // the indirect-gather cost model.
    return anchor.kind == SimtAnchorKind::DirectGather ||
           anchor.kind == SimtAnchorKind::LoadedIndexDependentMemory ||
           anchor.kind == SimtAnchorKind::Histogram ||
           anchor.kind == SimtAnchorKind::TensorAtomic;
  }
  // Scatter and ConvertStore stages use ContinuousTileStore cost model.
  // Atomic anchors can also match a scatter stage (scatter-store semantics).
  if (stage.costModelKind == StageCostModelKind::ContinuousTileStore)
    return anchor.kind == SimtAnchorKind::TensorAtomic;
  return false;
}

static Operation *getTopLevelSemanticRoot(Operation *operation);

static void attachAnchorOperationOwnership(StagePartition &partition,
                                           const SimtAnchorPlan &anchorPlan) {
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      if (!stage.localSimtMaterializable)
        continue;
      for (auto indexedAnchor : llvm::enumerate(anchorPlan.anchors)) {
        const SimtAnchorDescriptor &anchor = indexedAnchor.value();
        if (!anchorMatchesStage(anchor, stage))
          continue;
        stage.simtAnchorIndices.push_back(
            static_cast<unsigned>(indexedAnchor.index()));
        if (anchor.scopeOperations.empty()) {
          if (anchor.operation &&
              !llvm::is_contained(stage.operations, anchor.operation))
            stage.operations.push_back(anchor.operation);
          continue;
        }
        for (Operation *operation : anchor.scopeOperations)
          if (operation && !llvm::is_contained(stage.operations, operation))
            stage.operations.push_back(operation);
      }
      // A production mixed Stage must be backed by the exact operations that
      // the materializer will consume.  Synthetic feature-only tests use the
      // overload without an anchor plan and retain fallback behavior.
      stage.localSimtMaterializable = !stage.operations.empty();
      if (!stage.localSimtMaterializable)
        stage.localSimtFactors.clear();
    }
  }
}

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
      // Try to match natural SIMT anchors to this stage.
      stage.simtAnchorIndices.clear();
      if (stage.localSimtMaterializable) {
        for (auto indexedAnchor : llvm::enumerate(anchorPlan.anchors)) {
          const SimtAnchorDescriptor &anchor = indexedAnchor.value();
          if (anchor.materializable && stageOwnsAnchor(stage, anchor))
            stage.simtAnchorIndices.push_back(
                static_cast<unsigned>(indexedAnchor.index()));
        }
        stage.localSimtMaterializable = !stage.simtAnchorIndices.empty();
        llvm::errs() << "[COSTMODEL]   attachExactAnchorOwnership: stage='"
                     << stage.id << "' naturalAnchors="
                     << stage.simtAnchorIndices.size()
                     << " localSimtMaterializable="
                     << (stage.localSimtMaterializable ? "true" : "false")
                     << "\n";
      }
      // A stage with operations but no natural SIMT anchor can still be
      // materialized as a stage-level SIMT scope: the materializer wraps
      // every owned operation in one scope.scope { vector_mode="simt" }.
      if (!stage.localSimtMaterializable && !stage.operations.empty()) {
        stage.localSimtMaterializable = true;
        if (stage.localSimtFactors.empty())
          stage.localSimtFactors = {1};
        llvm::errs() << "[COSTMODEL]   attachExactAnchorOwnership: stage='"
                     << stage.id << "' has no natural anchor but owns "
                     << stage.operations.size()
                     << " ops → marked materializable as WholeStageSimt\n";
      }
      if (!stage.localSimtMaterializable)
        stage.localSimtFactors.clear();
    }
  }
}

/// Split ``WholeStageSimt`` stages by materialization constraints.
///
/// A ``WholeStageSimt`` stage has no natural SIMT anchor and is marked
/// materializable as a single scope.  However, the materializer must exclude
/// memory-allocating ops (``tt.load``, ``tt.alloc``, …) and pointer-producing
/// ops (``tt.addptr``, ``tt.splat`` of pointer, ``tt.bitcast`` of pointer)
/// from the SIMT scope — they stay in SIMD.  If such an excluded op sits
/// between two in-scope ops, the materializer would move the in-scope ops
/// into a scope body, leaving the excluded op behind and breaking SSA
/// dominance.
///
/// To keep the cost model's evaluation granularity aligned with the
/// post-materialization structure, we split each ``WholeStageSimt`` stage
/// at every excluded op into multiple sub-stages:
///   - Sub-stages consisting solely of excluded ops become pure-SIMD stages.
///   - Sub-stages consisting solely of in-scope ops remain SIMT-materializable.
///
/// After splitting, ``deriveStageLiveValues`` computes accurate live-in /
/// live-out for each sub-stage, and ``deriveLocalSimtScopeTraffic`` charges
/// only the real scope traffic because each SIMT sub-stage owns exactly the
/// ops that will enter its scope.
static void splitWholeStageSimtByMaterializationConstraints(
    StagePartition &partition) {
  for (LogicalPhase &phase : partition.phases) {
    SmallVector<LogicalStage, 4> newStages;
    for (LogicalStage &stage : phase.stages) {
      const bool hasNaturalAnchor = !stage.simtAnchorIndices.empty();
      if (hasNaturalAnchor || !stage.localSimtMaterializable ||
          stage.operations.empty()) {
        newStages.push_back(std::move(stage));
        continue;
      }
      const std::string baseId = stage.id;
      llvm::errs() << "[COSTMODEL]   splitWholeStageSimt: stage='"
                   << baseId << "' ops=" << stage.operations.size() << "\n";

      // Partition operations into contiguous runs of in-scope / excluded ops.
      struct Run {
        SmallVector<Operation *> ops;
        bool inScope;
      };
      SmallVector<Run> runs;
      for (Operation *op : stage.operations) {
        const bool excluded =
            isMemoryAllocatingOp(op) || producesScopeOpaqueValue(op);
        const bool inScope = !excluded;
        if (runs.empty() || runs.back().inScope != inScope)
          runs.push_back({{}, inScope});
        runs.back().ops.push_back(op);
      }

      if (runs.size() <= 1) {
        newStages.push_back(std::move(stage));
        continue;
      }

      int simtRunIndex = 0;
      int simdRunIndex = 0;
      for (const Run &run : runs) {
        LogicalStage sub;
        if (run.inScope) {
          sub.id = baseId + "_simt" + Twine(simtRunIndex++).str();
          sub.costModelKind = stage.costModelKind;
          sub.scheduleKind = stage.scheduleKind;
          sub.iterationCount = stage.iterationCount;
          sub.features = stage.features;
          sub.simdLegal = false;
          sub.simtLegal = stage.simtLegal;
          sub.localSimtMaterializable = true;
          sub.legalSimtFactors = stage.legalSimtFactors;
          sub.localSimtFactors = stage.localSimtFactors;
          sub.description =
              baseId + " (SIMT scope run #" + Twine(simtRunIndex).str() + ")";
        } else {
          sub.id = baseId + "_simd" + Twine(simdRunIndex++).str();
          sub.costModelKind = StageCostModelKind::ScalarIssue;
          sub.scheduleKind = StageScheduleKind::StraightLine;
          sub.iterationCount = stage.iterationCount;
          sub.simdLegal = true;
          sub.simtLegal = false;
          sub.localSimtMaterializable = false;
          sub.description =
              baseId + " (SIMD run #" + Twine(simdRunIndex).str() + ")";
        }
        llvm::append_range(sub.operations, run.ops);
        llvm::errs() << "[COSTMODEL]     sub-stage '" << sub.id << "' mode="
                     << (run.inScope ? "SIMT" : "SIMD")
                     << " ops=" << run.ops.size() << "\n";
        newStages.push_back(std::move(sub));
      }
    }
    phase.stages.assign(newStages.begin(), newStages.end());
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

/// Count occurrences of each operation kind in the subtree of `root` and
/// classify the root into the SemanticStage whose operation kind has the
/// highest count.  This replaces the previous first-match if-else chain which
/// could misclassify a root containing both tt.load and tt.store as
/// ConvertStore just because tt.store was checked first.
///
/// The mapping from op name to SemanticStage is:
///   tt.histogram           → Histogram
///   tt.atomic_rmw/cas      → Atomic
///   tt.dot                  → Dot
///   tt.reduce              → Reduction
///   tt.load (loaded-index) → Gather  (weighted higher than plain load)
///   tt.load (plain)        → Gather
///   tt.store (scatter)     → Scatter  (indirect store, detected via
///                                     loaded-index-dependent pointer)
///   tt.store (regular)     → ConvertStore
///   everything else        → IndexSetup (weight 0, fallback)
struct SemanticOpCounters {
  int64_t histogram = 0;
  int64_t atomic = 0;
  int64_t dot = 0;
  int64_t reduce = 0;
  int64_t gather = 0;
  int64_t scatter = 0;
  int64_t convertStore = 0;
  int64_t indexSetup = 0;
};

static void countSemanticOps(Operation *root, SemanticOpCounters &counts) {
  if (!root)
    return;
  auto classify = [](Operation *op) {
    const llvm::StringRef name = op->getName().getStringRef();
    if (name == "tt.histogram")
      return &SemanticOpCounters::histogram;
    if (name == "tt.atomic_rmw" || name == "tt.atomic_cas")
      return &SemanticOpCounters::atomic;
    if (name == "tt.dot")
      return &SemanticOpCounters::dot;
    if (name == "tt.reduce" || name == "tt.scan")
      return &SemanticOpCounters::reduce;
    if (name == "tt.load")
      return isLoadedIndexDependentMemoryOp(op) ? &SemanticOpCounters::gather
                                                : &SemanticOpCounters::gather;
    if (name == "tt.store")
      return isLoadedIndexDependentMemoryOp(op) ? &SemanticOpCounters::scatter
                                                : &SemanticOpCounters::convertStore;
    if (name == "tt.gather")
      return &SemanticOpCounters::gather;
    if (name == "tt.scatter")
      return &SemanticOpCounters::scatter;
    return &SemanticOpCounters::indexSetup;
  };

  // Count the root operation itself.  For high-level semantic ops with
  // regions (tt.scan/tt.reduce/tt.dot), the root's own classification is
  // the authoritative one — their region body is an implementation detail
  // (overflow checks, combinator lambda) and is skipped below.
  int64_t SemanticOpCounters::*bucket = classify(root);
  ++(counts.*bucket);

  // If the root itself is a high-level semantic op with a region, its
  // region body is an implementation detail (the reduction/scan combinator
  // lambda).  Don't walk into it — those inner arith ops (overflow checks,
  // etc.) would inflate indexSetup and distort the classification.
  const llvm::StringRef rootName = root->getName().getStringRef();
  if (rootName == "tt.scan" || rootName == "tt.reduce" ||
      rootName == "tt.dot")
    return;

  // Walk descendants but skip region bodies of high-level semantic ops.
  // Their regions are implementation details (e.g., reduce/scan lambda with
  // arith ops for overflow checking), not separate semantic stages.
  root->walk([&](Operation *nested) -> mlir::WalkResult {
    int64_t SemanticOpCounters::*b = classify(nested);
    ++(counts.*b);

    // Don't descend into region bodies of semantic ops with lambdas.
    const llvm::StringRef name = nested->getName().getStringRef();
    if (name == "tt.scan" || name == "tt.reduce" || name == "tt.dot")
      return mlir::WalkResult::skip();
    return mlir::WalkResult::advance();
  });
}

static SemanticStage classifyRootSemantic(Operation *root) {
  SemanticOpCounters counts;
  countSemanticOps(root, counts);

  llvm::errs() << "[COSTMODEL]   classifyRootSemantic: ";
  if (root)
    root->print(llvm::errs());
  else
    llvm::errs() << "<null>";
  llvm::errs() << "\n";
  llvm::errs() << "[COSTMODEL]     op counts: indexSetup=" << counts.indexSetup
               << " gather=" << counts.gather
               << " dot=" << counts.dot
               << " reduce=" << counts.reduce
               << " histogram=" << counts.histogram
               << " atomic=" << counts.atomic
               << " scatter=" << counts.scatter
               << " convertStore=" << counts.convertStore << "\n";

  // Select the stage with the highest count.  Ties are broken by priority
  // order (later stages win) to keep the monotone state machine moving
  // forward.
  struct Candidate {
    int64_t count;
    SemanticStage stage;
  };
  const Candidate candidates[] = {
      {counts.indexSetup,    SemanticStage::IndexSetup},
      {counts.gather,        SemanticStage::Gather},
      {counts.dot,           SemanticStage::Dot},
      {counts.reduce,        SemanticStage::Reduction},
      {counts.histogram,     SemanticStage::Histogram},
      {counts.atomic,        SemanticStage::Atomic},
      {counts.scatter,       SemanticStage::Scatter},
      {counts.convertStore,  SemanticStage::ConvertStore},
  };
  SemanticStage best = SemanticStage::IndexSetup;
  int64_t bestCount = 0;
  for (const Candidate &c : candidates) {
    if (c.count > bestCount ||
        (c.count > 0 && c.count == bestCount &&
         semanticStagePriority(c.stage) > semanticStagePriority(best))) {
      best = c.stage;
      bestCount = c.count;
    }
  }
  llvm::errs() << "[COSTMODEL]     → classified as: " << semanticStageId(best)
               << " (count=" << bestCount << ")\n";
  return best;
}

/// PhaseBoundaryAnalysis owns the algorithm-level serial cut.  Each root is
/// assigned exactly one semantic Stage id in execution order via a unified
/// monotone state machine.  The machine is driven purely by operation
/// semantics (op names, loaded-index dependencies, recurrence loops) and is
/// independent of any business-scenario domain.  After a boundary is crossed,
/// a later root cannot move back to an earlier Stage (monotone progress).
static llvm::Error assignRootStageIdsBySemantics(PhaseBoundaryPlan &plan) {
  plan.rootPhaseIds.clear();
  plan.rootPhaseIds.reserve(plan.rootOperations.size());

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

  // Determine whether the anchor range is a recurrence region (contains a
  // loop) or a gather region.  This mirrors the original triangular
  // (recurrence) vs rowwise/indirect-dot (gather) distinction without
  // referencing any domain enum.
  bool anchorRangeIsRecurrence = false;
  if (firstAnchorIndex && lastAnchorIndex) {
    for (size_t index = *firstAnchorIndex; index <= *lastAnchorIndex; ++index) {
      if (operationTreeContainsName(plan.rootOperations[index], "scf.for")) {
        anchorRangeIsRecurrence = true;
        break;
      }
    }
  }

  SemanticStage current = SemanticStage::IndexSetup;
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

    const bool inAnchorRange =
        firstAnchorIndex && lastAnchorIndex &&
        indexedRoot.index() >= *firstAnchorIndex &&
        indexedRoot.index() <= *lastAnchorIndex;

    SemanticStage classified = SemanticStage::IndexSetup;
    if (inAnchorRange) {
      classified = anchorRangeIsRecurrence ? SemanticStage::Recurrence
                                          : SemanticStage::Gather;
    } else {
      classified = classifyRootSemantic(root);
    }

    if (shouldAdvanceStage(current, classified)) {
      llvm::errs() << "[COSTMODEL]     stage advance: "
                   << semanticStageId(current) << " → "
                   << semanticStageId(classified)
                   << " (semantic change";
      if (isStorageStage(current))
        llvm::errs() << ", storage→storage";
      llvm::errs() << ")\n";
      current = classified;
    } else if (classified != current) {
      llvm::errs() << "[COSTMODEL]     stage hold: stayed at "
                   << semanticStageId(current)
                   << " (classified=" << semanticStageId(classified)
                   << ", ";
      if (isStorageStage(current) && !isStorageStage(classified))
        llvm::errs() << "storage regression blocked";
      else if (classified == SemanticStage::IndexSetup)
        llvm::errs() << "index_setup re-entry blocked";
      else
        llvm::errs() << "regression blocked";
      llvm::errs() << ")\n";
    }
    const std::string stageId = semanticStageId(current).str();
    llvm::errs() << "[COSTMODEL]   assignRootStageIdsBySemantics: root["
                 << indexedRoot.index() << "] → stage='" << stageId << "'";
    if (inAnchorRange)
      llvm::errs() << " (inAnchorRange)";
    llvm::errs() << "\n";
    plan.rootPhaseIds.push_back(stageId);
  }

  if (plan.rootPhaseIds.size() != plan.rootOperations.size())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "PhaseBoundaryAnalysis did not own every semantic root");

  // Contiguity check: computation stages may alternate freely (e.g.
  // gather → histogram → gather is legal — the IR defines the order, not
  // a hardcoded priority).  Two hard constraints enforced by
  // shouldAdvanceStage above (this check is a safety net):
  //   1. storage stages must not regress to computation;
  //   2. index_setup (preparation) must not reappear once left.
  llvm::StringRef currentPhase;
  bool enteredStoragePhase = false;
  bool leftIndexSetup = false;
  for (const std::string &phaseId : plan.rootPhaseIds) {
    if (phaseId == currentPhase)
      continue;
    if (phaseId == "index_setup") {
      if (leftIndexSetup)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "PhaseBoundaryAnalysis regressed back to index_setup after "
            "leaving the preparation phase");
    } else if (phaseId != "auto_blockify_dispatch") {
      leftIndexSetup = true;
    }
    const bool isStorage = phaseId == "scatter" || phaseId == "convert_store";
    if (isStorage)
      enteredStoragePhase = true;
    if (enteredStoragePhase && !isStorage)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "PhaseBoundaryAnalysis regressed from storage to computation: "
          "'%s' after a storage stage",
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
    } else {
      // Semantic Stage id directly names the Stage; no domain-specific
      // phaseId→stageId translation is needed.
      target = findStage(partition, phaseId);
    }
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
  partition.boundarySource = "operation_graph";
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

/// Pretty-print a Value for debugging: result number + type + bytes (if known).
static std::string valueDebugString(Value value) {
  std::string buf;
  llvm::raw_string_ostream os(buf);
  if (auto result = dyn_cast<OpResult>(value))
    os << "%" << result.getResultNumber();
  else if (auto arg = dyn_cast<BlockArgument>(value))
    os << "%arg" << arg.getArgNumber();
  else
    os << "<unknown>";
  os << " : " << value.getType();
  const int64_t bytes = staticTensorBytes(value);
  if (bytes > 0)
    os << " (" << bytes << "B)";
  os.flush();
  return buf;
}

/// Print a list of values with their types and bytes, one per line, indented.
static void dumpValueList(llvm::raw_ostream &os, llvm::ArrayRef<Value> values,
                          llvm::StringRef label, int64_t totalBytes) {
  os << "[COSTMODEL]       " << label << " (count=" << values.size()
     << ", bytes=" << totalBytes << "):\n";
  for (Value value : values)
    os << "[COSTMODEL]         " << valueDebugString(value) << "\n";
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

      llvm::errs() << "[COSTMODEL]   deriveStageLiveValues: stage='"
                   << stage.id << "' ops=" << stage.operations.size()
                   << " owned=" << owned.size()
                   << " liveIn=" << stage.liveIns.size()
                   << " liveOut=" << stage.liveOuts.size()
                   << " liveInBytes=" << stage.liveInBytes
                   << " liveOutBytes=" << stage.liveOutBytes << "\n";
      dumpValueList(llvm::errs(), stage.liveIns, "live-in",
                    stage.liveInBytes);
      dumpValueList(llvm::errs(), stage.liveOuts, "live-out",
                    stage.liveOutBytes);
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
      llvm::DenseSet<Operation *> coveredByRange;

      for (unsigned anchorIndex : stage.simtAnchorIndices) {
        if (anchorIndex >= anchorPlan.anchors.size())
          continue;
        const SimtAnchorDescriptor &anchor = anchorPlan.anchors[anchorIndex];
        if (!anchor.materializable || !anchor.operation ||
            coveredByRange.contains(anchor.operation))
          continue;

        llvm::SmallVector<Operation *> roots;
        const bool isRange = anchor.scopeOperations.size() > 1;
        if (isRange) {
          llvm::append_range(roots, anchor.scopeOperations);
          for (Operation *operation : roots)
            coveredByRange.insert(operation);
        } else {
          roots.push_back(anchor.operation);
        }

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

        ++stage.localSimtScopeCount;
        const int64_t inputBytes = staticTensorBytes(captured.getArrayRef());
        const int64_t outputBytes = staticTensorBytes(returned.getArrayRef());
        stage.scopeInputTensorBytes += inputBytes;
        stage.scopeOutputTensorBytes += outputBytes;

        llvm::errs() << "[COSTMODEL]   deriveLocalSimtScopeTraffic: stage='"
                     << stage.id << "' naturalAnchor #" << anchorIndex
                     << " isRange=" << (isRange ? "true" : "false")
                     << " inside=" << inside.size()
                     << " captured=" << captured.size()
                     << " returned=" << returned.size()
                     << " inputBytes=" << inputBytes
                     << " outputBytes=" << outputBytes << "\n";
        dumpValueList(llvm::errs(), captured.getArrayRef(),
                      "scope-captured (live-in)", inputBytes);
        dumpValueList(llvm::errs(), returned.getArrayRef(),
                      "scope-returned (live-out)", outputBytes);
      }

      // A stage with no natural anchors but with operations is materialized
      // as one stage-level SIMT scope.  Its traffic is the stage's full
      // live-in/live-out, already computed by deriveStageLiveValues.
      if (stage.simtAnchorIndices.empty() &&
          stage.localSimtMaterializable && !stage.operations.empty()) {
        stage.localSimtScopeCount = 1;
        stage.scopeInputTensorBytes = stage.liveInBytes;
        stage.scopeOutputTensorBytes = stage.liveOutBytes;
        llvm::errs() << "[COSTMODEL]   deriveLocalSimtScopeTraffic: stage='"
                     << stage.id
                     << "' WholeStageSimt scopeCount=1"
                     << " scopeInputBytes=" << stage.scopeInputTensorBytes
                     << " scopeOutputBytes=" << stage.scopeOutputTensorBytes
                     << " (liveInBytes=" << stage.liveInBytes
                     << " liveOutBytes=" << stage.liveOutBytes << ")\n";
        dumpValueList(llvm::errs(), stage.liveIns,
                      "scope-captured (live-in)", stage.liveInBytes);
        dumpValueList(llvm::errs(), stage.liveOuts,
                      "scope-returned (live-out)", stage.liveOutBytes);
      }
    }
  }
}

static double totalOperationElements(const StageWorkload &work) {
  double result = 0.0;
  for (const auto &entry : work.operationElements)
    result += entry.second;
  return result;
}

static StageWorkload accumulatePartition(const StagePartition &partition) {
  StageWorkload total;
  for (const LogicalPhase &phase : partition.phases) {
    for (const LogicalStage &stage : phase.stages) {
      StageWorkload work = stage.workload;
      const double count =
          static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
      work.scalarOperations *= count;
      work.loadBytes *= count;
      work.storeBytes *= count;
      work.loadWarpInstructions *= count;
      work.storeWarpInstructions *= count;
      work.predicateElements *= count;
      work.shuffleLaneSteps *= count;
      work.dotFlops *= count;
      work.estimatedSpillTransactions *= count;
      for (auto &entry : work.operationElements)
        entry.second *= count;
      mergeWorkload(total, std::move(work));
    }
  }
  return total;
}

static bool near(double lhs, double rhs) {
  return std::abs(lhs - rhs) <= 1.0e-6 * std::max({1.0, lhs, rhs});
}

} // namespace

StageWorkload
mlir::ascend::buildKernelStageWorkload(const SimdSimtFeatureSummary &features) {
  SimtAnchorFeatureSummary kernel;
  kernel.opElements = features.opElements;
  kernel.weightedOps = features.weightedOps;
  kernel.loadBytes = features.loadBytes;
  kernel.storeBytes = features.storeBytes;
  kernel.loadWarpInstructions = features.loadWarpInstructions;
  kernel.storeWarpInstructions = features.storeWarpInstructions;
  kernel.predicateElements = features.predicateElements;
  kernel.predicateLaneEvaluations = features.predicateLaneEvaluations;
  kernel.shuffleLaneSteps = features.shuffleLaneSteps;
  kernel.dotFlops = features.dotFlops;
  StageWorkload work = buildWorkload(kernel);
  work.scalarOperations = static_cast<double>(features.scalarOps);
  recomputeIssueElements(work);
  return work;
}

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

llvm::Expected<std::optional<PhaseBoundaryPlan>>
PhaseBoundaryAnalysis::analyze(const SimdSimtFeatureSummary &features,
                               const StagePartitionerOptions &options) const {
  (void)options;
  llvm::errs() << "[COSTMODEL] --- PhaseBoundaryAnalysis::analyze START ---\n";
  // Semantic-driven: no domain matching.  Every kernel receives a semantic
  // Stage plan so that the costmodel is applicable without per-scenario
  // domain templates.  The feature-summary overload has no operation graph,
  // so Stage boundaries are synthesized from aggregate feature counts by
  // partitionBySemantics.
  PhaseBoundaryPlan plan;
  plan.domainName = "semantic";
  llvm::errs() << "[COSTMODEL]   Semantic plan (no domain matching): dotOps="
               << features.dotOps << " reduceOps=" << features.reduceOps
               << " loadedIndexDepMem=" << features.loadedIndexDependentMemoryOps
               << " loadOps=" << features.loadOps
               << " storeOps=" << features.storeOps
               << " dotFlops=" << features.dotFlops << "\n";
  llvm::errs() << "[COSTMODEL] --- PhaseBoundaryAnalysis::analyze END ---\n";
  return std::optional<PhaseBoundaryPlan>{std::move(plan)};
}

llvm::Expected<std::optional<PhaseBoundaryPlan>>
PhaseBoundaryAnalysis::analyze(ModuleOp module,
                               const SimtAnchorPlan &anchorPlan,
                               const SimdSimtFeatureSummary &features,
                               const StagePartitionerOptions &options) const {
  auto plan = analyze(features, options);
  if (!plan)
    return plan.takeError();
  if (!*plan)
    return plan;
  auto structure = ProgramStructureAnalysis().analyze(module, anchorPlan);
  if (!structure)
    return structure.takeError();
  (*plan)->rootOperations = std::move(structure->rootOperations);
  (*plan)->localSimtAnchorRoots = std::move(structure->localSimtAnchorRoots);
  if (llvm::Error error = assignRootStageIdsBySemantics(**plan))
    return std::move(error);
  return plan;
}

llvm::Expected<StagePartition>
StageBoundaryAnalysis::analyze(const PhaseBoundaryPlan &phasePlan,
                               const SimdSimtFeatureSummary &features,
                               const SimtAnchorPlan *anchorPlan) const {
  StagePartition partition = partitionBySemantics(
      features, phasePlan.hasOperationGraph() ? &phasePlan : nullptr);
  partition.domain = phasePlan.domainName;
  if (phasePlan.hasOperationGraph()) {
    if (llvm::Error error =
            attachCompleteOperationOwnership(partition, phasePlan))
      return std::move(error);
    if (anchorPlan)
      attachExactAnchorOwnership(partition, *anchorPlan);
    splitWholeStageSimtByMaterializationConstraints(partition);
    deriveStageLiveValues(partition);
    if (anchorPlan)
      deriveLocalSimtScopeTraffic(partition, *anchorPlan);
  } else if (anchorPlan) {
    attachAnchorOperationOwnership(partition, *anchorPlan);
  }
  return partition;
}

llvm::Error StageFeatureAnalysis::analyze(StagePartition &partition) const {
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      StageModelFeatures &facts = stage.features;
      if (partition.operationOwnershipComplete) {
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
              // The scf.for induction variable itself commonly feeds pointer
              // arithmetic.  Record that work without treating it as a data
              // recurrence.
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
          facts.hasReduction |= name == "tt.reduce" || name == "tt.scan" ||
                                name == "linalg.reduce";
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
        // Counts consumed by StageCostEvaluator are per logical iteration.
        // Multiple sibling recurrence loops are flattened into one Stage
        // iteration space, so identical control events must be normalized by
        // the number of sibling loops rather than charged once per loop on
        // every flattened iteration.
        if (algorithmLoopCount > 0 && stage.iterationCount > 1) {
          if (facts.hasLoopCarriedDataDependency)
            facts.parallelRecurrenceGroupCount = algorithmLoopCount;
          facts.loopBackedgeCount = 1;
          facts.conditionalBranchCount = std::max<int64_t>(
              facts.conditionalBranchCount > 0 ? 1 : 0,
              facts.conditionalBranchCount / algorithmLoopCount);
          facts.divergentBranchCount = std::max<int64_t>(
              facts.divergentBranchCount > 0 ? 1 : 0,
              facts.divergentBranchCount / algorithmLoopCount);
        }
        facts.source =
            "exact post-layout/post-AutoBlockify-V1 TTIR operation graph";
        if (!facts.isValid())
          return llvm::createStringError(std::errc::invalid_argument,
                                         "Stage '%s' has invalid features",
                                         stage.id.c_str());
        continue;
      }
      facts.hasLoop =
          stage.iterationCount > 1 ||
          stage.costModelKind == StageCostModelKind::AutoBlockifyLoop ||
          stage.costModelKind == StageCostModelKind::IndependentPipelinedLoop ||
          stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence;
      facts.loopBackedgeCount = facts.hasLoop ? 1 : 0;
      facts.hasLoopCarriedDataDependency =
          stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence;
      facts.hasPointerInduction =
          facts.hasLoop && !facts.hasLoopCarriedDataDependency;
      facts.hasContiguousMemory =
          llvm::is_contained({StageCostModelKind::ContinuousTileMemory,
                              StageCostModelKind::ContinuousTileStore,
                              StageCostModelKind::ContinuousShortLoad,
                              StageCostModelKind::CachePolicyStore},
                             stage.costModelKind);
      facts.hasIndirectMemory =
          llvm::is_contained({StageCostModelKind::IndirectScalarMemory,
                              StageCostModelKind::IndirectGatherMemory},
                             stage.costModelKind);
      facts.hasReduction =
          llvm::is_contained({StageCostModelKind::LoopCarriedRecurrence,
                              StageCostModelKind::RowwiseReduction},
                             stage.costModelKind);
      facts.hasDot = llvm::is_contained({StageCostModelKind::CubeRoofline,
                                         StageCostModelKind::TinyCubeRoofline},
                                        stage.costModelKind);
      facts.hasConversionPack =
          stage.costModelKind == StageCostModelKind::ConversionPack;
      facts.source = "feature-summary fallback structural facts";
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
  for (LogicalPhase &phase : partition.phases) {
    for (LogicalStage &stage : phase.stages) {
      const StageModelFeatures &facts = stage.features;
      // Auxiliary scalar/index/predicate and memory work may live in a
      // specialized Stage, but two independent dominant formulas may not.
      // Reaching this check means StageBoundaryAnalysis missed a structural
      // cut.  Silently selecting the first matching Kind would hide or
      // double-count work, so report a stable boundary diagnostic instead.
      // Conversion operations are not a split condition by themselves:
      // predicate-to-float and accumulator casts are often auxiliary work of
      // a recurrence/reduction/dot Stage.  ConversionPack becomes dominant
      // only when no stronger structure owns the Stage (see deriveKind()).
      const bool requiresSplit =
          facts.hasDot && (facts.hasReduction || facts.hasIndirectMemory ||
                           facts.hasLoopCarriedDataDependency);
      if (requiresSplit)
        return llvm::createStringError(
            std::errc::invalid_argument,
            "requires_split: Stage '%s' owns incompatible dominant "
            "structures (carried=%d, indirect=%d, reduction=%d, dot=%d, "
            "conversion=%d, roots=%zu)",
            stage.id.c_str(), facts.hasLoopCarriedDataDependency,
            facts.hasIndirectMemory, facts.hasReduction, facts.hasDot,
            facts.hasConversionPack, stage.operations.size());
      auto kindMatchesFacts = [&](StageCostModelKind kind) {
        switch (kind) {
        case StageCostModelKind::AutoBlockifyDispatch:
        case StageCostModelKind::AutoBlockifyLoop:
        case StageCostModelKind::ScalarIssue:
        case StageCostModelKind::ScalarControl:
        case StageCostModelKind::ScalarMath:
        case StageCostModelKind::IndexGeneration:
        case StageCostModelKind::PredicateMask:
        case StageCostModelKind::LoopPredicate:
          return true;
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
        }
        return false;
      };
      auto deriveKind = [&]() {
        // Preserve semantic specializations when their defining evidence is
        // present.  Otherwise classify from the exact Stage operation graph;
        // no workload name or experiment identity participates here.
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
      if (!kindMatchesFacts(stage.costModelKind)) {
        stage.costModelKind = deriveKind();
        if (stage.costModelKind == StageCostModelKind::IndependentPipelinedLoop)
          stage.scheduleKind = StageScheduleKind::IndependentPipelined;
        else if (stage.costModelKind ==
                 StageCostModelKind::LoopCarriedRecurrence)
          stage.scheduleKind = StageScheduleKind::LoopCarriedSerial;
      }
      const StageCostModelKind kind = stage.costModelKind;
      auto mismatch = [&]() {
        return llvm::createStringError(
            std::errc::invalid_argument,
            "Stage '%s' operation graph does not match StageCostModelKind "
            "'%s' (loop=%d, carried=%d, contiguous=%d, indirect=%d, "
            "reduction=%d, dot=%d, conversion=%d, roots=%zu)",
            stage.id.c_str(), stringifyStageCostModel(kind).str().c_str(),
            facts.hasLoop, facts.hasLoopCarriedDataDependency,
            facts.hasContiguousMemory, facts.hasIndirectMemory,
            facts.hasReduction, facts.hasDot, facts.hasConversionPack,
            stage.operations.size());
      };
      switch (kind) {
      case StageCostModelKind::AutoBlockifyDispatch:
      case StageCostModelKind::AutoBlockifyLoop:
        break;
      case StageCostModelKind::LoopCarriedRecurrence:
        if (!facts.hasLoopCarriedDataDependency)
          return mismatch();
        break;
      case StageCostModelKind::IndependentPipelinedLoop:
        if (!facts.hasLoop || facts.hasLoopCarriedDataDependency)
          return mismatch();
        break;
      case StageCostModelKind::RowwiseReduction:
        if (!facts.hasReduction)
          return mismatch();
        break;
      case StageCostModelKind::CubeRoofline:
        if (!facts.hasDot)
          return mismatch();
        break;
      case StageCostModelKind::TinyCubeRoofline:
        if (!facts.hasDot ||
            stage.workload.dotFlops * stage.iterationCount >
                static_cast<double>(std::max<int64_t>(1, tinyDotFlopsMax)))
          return mismatch();
        break;
      case StageCostModelKind::IndirectScalarMemory:
      case StageCostModelKind::IndirectGatherMemory:
        if (!facts.hasIndirectMemory)
          return mismatch();
        break;
      case StageCostModelKind::ContinuousTileMemory:
      case StageCostModelKind::ContinuousTileStore:
      case StageCostModelKind::ContinuousShortLoad:
      case StageCostModelKind::CachePolicyStore:
        if (!facts.hasContiguousMemory)
          return mismatch();
        break;
      case StageCostModelKind::ConversionPack:
        if (!facts.hasConversionPack)
          return mismatch();
        break;
      case StageCostModelKind::ScalarIssue:
      case StageCostModelKind::ScalarControl:
      case StageCostModelKind::ScalarMath:
      case StageCostModelKind::IndexGeneration:
      case StageCostModelKind::PredicateMask:
      case StageCostModelKind::LoopPredicate:
        // These kinds may legitimately contain auxiliary address, predicate,
        // control, or short metadata memory work.  Their dominant semantics
        // is established by the contiguous boundary plan.
        break;
      }
    }
  }
  return llvm::Error::success();
}

llvm::Error
StageWorkloadAnalysis::verify(const StagePartition &partition,
                              const StageWorkload &kernelWorkload) const {
  if (!kernelWorkload.isFiniteAndNonNegative())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "kernel StageWorkload is invalid");
  const StageWorkload owned = accumulatePartition(partition);
  if (!near(owned.scalarOperations, kernelWorkload.scalarOperations) ||
      !near(owned.loadBytes, kernelWorkload.loadBytes) ||
      !near(owned.storeBytes, kernelWorkload.storeBytes) ||
      !near(owned.predicateElements, kernelWorkload.predicateElements) ||
      !near(owned.shuffleLaneSteps, kernelWorkload.shuffleLaneSteps) ||
      !near(owned.dotFlops, kernelWorkload.dotFlops) ||
      !near(totalOperationElements(owned),
            totalOperationElements(kernelWorkload)))
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition does not conserve post-transform TTIR workload");
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
StagePartitionVerifier::verify(const StagePartition &partition,
                               const StageWorkload &kernelWorkload) const {
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
          stage.simtAnchorIndices.empty() &&
          stage.operations.empty())
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
  if (partition.operationOwnershipComplete)
    return llvm::Error::success();
  return StageWorkloadAnalysis().verify(partition, kernelWorkload);
}

llvm::Error
StageModeLegalityAnalysis::analyze(StagePartition &partition,
                                   int64_t maximumSuperblockFactor,
                                   bool scopeSuperblockMaterializable) const {
  const int64_t maximum = std::clamp<int64_t>(maximumSuperblockFactor, 1, 4);
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
        // A mixed F2/F4 implementation is legal only when backend integration
        // can wrap the complete logical-program body with AutoBlockify V1.
        // The local scope itself remains a single-mode Stage; factor changes
        // its implementation, not the Stage boundary.
        stage.localSimtFactors = scopeSuperblockMaterializable
                                     ? stage.legalSimtFactors
                                     : std::vector<int64_t>{1};
      }
      if (stage.localSimtMaterializable &&
          (stage.localSimtFactors.empty() ||
           llvm::any_of(stage.localSimtFactors, [&](int64_t factor) {
             return !llvm::is_contained(stage.legalSimtFactors, factor);
           })))
        return llvm::createStringError(
            std::errc::invalid_argument,
            "local SIMT factors are invalid for Stage '%s'", stage.id.c_str());
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::optional<StagePartition>>
StagePartitioner::partition(const SimdSimtFeatureSummary &features,
                            const StagePartitionerOptions &options) const {
  auto phasePlan = PhaseBoundaryAnalysis().analyze(features, options);
  if (!phasePlan)
    return phasePlan.takeError();
  if (!*phasePlan)
    return std::optional<StagePartition>{};
  auto result = StageBoundaryAnalysis().analyze(**phasePlan, features);
  if (!result)
    return result.takeError();

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
  if (llvm::Error error = StagePartitionVerifier().verify(
          *result, buildKernelStageWorkload(features)))
    return std::move(error);
  return std::optional<StagePartition>{std::move(*result)};
}

llvm::Expected<std::optional<StagePartition>>
StagePartitioner::partition(ModuleOp module, const SimtAnchorPlan &anchorPlan,
                            const SimdSimtFeatureSummary &features,
                            const StagePartitionerOptions &options) const {
  llvm::errs() << "[COSTMODEL] --- StagePartitioner::partition(module) START ---\n";
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
  llvm::errs() << "[COSTMODEL]   Phase→Stage partition: phases=" << result->phases.size() << "\n";
  for (size_t pi = 0; pi < result->phases.size(); ++pi) {
    llvm::errs() << "[COSTMODEL]     Phase[" << pi << "] id='" << result->phases[pi].id
                 << "' stages=" << result->phases[pi].stages.size() << "\n";
    for (size_t si = 0; si < result->phases[pi].stages.size(); ++si) {
      const LogicalStage &s = result->phases[pi].stages[si];
      llvm::errs() << "[COSTMODEL]       Stage[" << pi << "." << si << "] id='" << s.id
                   << "' kind=" << stringifyStageCostModel(s.costModelKind)
                   << " simdLegal=" << (s.simdLegal ? "true" : "false")
                   << " simtLegal=" << (s.simtLegal ? "true" : "false")
                   << " ops=" << s.operations.size() << "\n";
      if (!s.operations.empty()) {
        llvm::errs() << "[COSTMODEL]         --- IR of Stage '" << s.id << "' ---\n";
        for (Operation *op : s.operations) {
          if (op) {
            llvm::errs() << "[COSTMODEL]           ";
            op->print(llvm::errs());
            llvm::errs() << "\n";
          }
        }
      }
    }
  }
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
  if (llvm::Error error = StagePartitionVerifier().verify(
          *result, buildKernelStageWorkload(features)))
    return std::move(error);
  llvm::errs() << "[COSTMODEL]   Partition verification passed\n";
  llvm::errs() << "[COSTMODEL] --- StagePartitioner::partition(module) END ---\n";
  return std::optional<StagePartition>{std::move(*result)};
}

llvm::Expected<std::optional<StagePartition>>
StagePartitioner::partition(const SimdSimtFeatureSummary &features,
                            const StagePartitionerOptions &options,
                            const SimtAnchorPlan &anchorPlan) const {
  auto phasePlan = PhaseBoundaryAnalysis().analyze(features, options);
  if (!phasePlan)
    return phasePlan.takeError();
  if (!*phasePlan)
    return std::optional<StagePartition>{};
  auto result =
      StageBoundaryAnalysis().analyze(**phasePlan, features, &anchorPlan);
  if (!result)
    return result.takeError();
  llvm::errs() << "[COSTMODEL]   Phase→Stage partition (no-module): phases=" << result->phases.size() << "\n";
  for (size_t pi = 0; pi < result->phases.size(); ++pi) {
    llvm::errs() << "[COSTMODEL]     Phase[" << pi << "] id='" << result->phases[pi].id
                 << "' stages=" << result->phases[pi].stages.size() << "\n";
    for (size_t si = 0; si < result->phases[pi].stages.size(); ++si) {
      const LogicalStage &s = result->phases[pi].stages[si];
      llvm::errs() << "[COSTMODEL]       Stage[" << pi << "." << si << "] id='" << s.id
                   << "' kind=" << stringifyStageCostModel(s.costModelKind)
                   << " simdLegal=" << (s.simdLegal ? "true" : "false")
                   << " simtLegal=" << (s.simtLegal ? "true" : "false")
                   << " ops=" << s.operations.size() << "\n";
      if (!s.operations.empty()) {
        llvm::errs() << "[COSTMODEL]         --- IR of Stage '" << s.id << "' ---\n";
        for (Operation *op : s.operations) {
          if (op) {
            llvm::errs() << "[COSTMODEL]           ";
            op->print(llvm::errs());
            llvm::errs() << "\n";
          }
        }
      }
    }
  }
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
  if (llvm::Error error = StagePartitionVerifier().verify(
          *result, buildKernelStageWorkload(features)))
    return std::move(error);
  return std::optional<StagePartition>{std::move(*result)};
}
