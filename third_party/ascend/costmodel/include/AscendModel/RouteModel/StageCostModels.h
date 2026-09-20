//===- StageCostModels.h - Per-stage analytical models --------*- C++ -*-===//
//
// StagePartitioner, StageCostEvaluator, and KernelRouteSolver are separate
// components.  This file defines the immutable data passed between them and
// the mode-specific StageCostModel tree used by StageCostEvaluator.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_STAGECOSTMODELS_H
#define ASCENDMODEL_ROUTEMODEL_STAGECOSTMODELS_H

#include "AscendModel/RouteModel/StageRouteCostModel.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::ascend {

enum class StageCostModelKind {
  AutoBlockifyDispatch,
  AutoBlockifyLoop,
  ScalarIssue,
  ScalarControl,
  ScalarMath,
  /// Elementwise tensor/vector computation, not scalar dispatch. The mode
  /// selects SIMD vector throughput or SIMT lane throughput for its workload.
  VectorIssue,
  IndexGeneration,
  PredicateMask,
  LoopPredicate,
  ContinuousTileMemory,
  ContinuousTileStore,
  ContinuousShortLoad,
  CachePolicyStore,
  IndirectScalarMemory,
  IndirectGatherMemory,
  AtomicMemory,
  IndependentPipelinedLoop,
  LoopCarriedRecurrence,
  RowwiseReduction,
  PrefixScan,
  CubeRoofline,
  TinyCubeRoofline,
  ConversionPack,
};

llvm::StringRef stringifyStageCostModel(StageCostModelKind kind);

struct LogicalStage {
  std::string id;
  StageCostModelKind costModelKind = StageCostModelKind::ScalarIssue;
  StageScheduleKind scheduleKind = StageScheduleKind::StraightLine;
  int64_t iterationCount = 1;
  StageModelFeatures features;
  StageWorkload workload;
  /// Exact TTIR ownership when StagePartition was built from an operation
  /// graph.  Feature-summary fallback partitions deliberately leave this
  /// empty and must not be treated as materialization evidence.
  std::vector<Operation *> operations;
  /// SSA values crossing the Stage boundary.  These are derived from the
  /// same exact operation ownership as `operations`; they are the contract
  /// consumed by legality checks and the scope materializer.
  std::vector<Value> liveIns;
  std::vector<Value> liveOuts;
  int64_t liveInBytes = 0;
  int64_t liveOutBytes = 0;
  /// Exact tensor traffic at the local scope boundary.  Unlike Stage
  /// live-in/live-out, these fields mirror the SSA values captured by and
  /// returned from the single compound scope.scope region produced by
  /// mergeSimtStageAnchors.  If the anchors cannot form one scope, the local
  /// SIMT implementation is illegal rather than being scored approximately.
  int64_t scopeInputTensorBytes = 0;
  int64_t scopeOutputTensorBytes = 0;
  /// Indices into the immutable SimtAnchorPlan.  A mixed route may
  /// materialize only anchors owned by Stages that the solver selected as
  /// SIMT; consuming every materializable anchor would violate the route.
  std::vector<unsigned> simtAnchorIndices;
  bool simdLegal = false;
  bool simtLegal = false;
  /// True when this Stage has exact operation ownership/live-in/live-out and
  /// can therefore become a local SIMT scope inside a mixed kernel.
  bool localSimtMaterializable = false;
  /// True when the one selected local scope will be a direct operation of an
  /// AutoBlockify V1 loop body.  NPUIR's current scope-SuperBlock ABI requires
  /// this stronger condition for F2/F4; nested scopes remain legal at F1.
  bool localSuperblockMaterializable = false;
  std::vector<int64_t> legalSimtFactors;
  std::vector<int64_t> localSimtFactors;
};

struct StagePartition {
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::vector<LogicalStage> stages;
};

struct StageOperationRate {
  double throughput = 0.0;
  double factor = 1.0;
  /// Aggregate throughput of one SIMT VF indexed by its launched warp count.
  /// Each concrete Stage implementation selects its own point using
  /// baseWarpCount * SuperBlockFactor; throughput is the fallback for profiles
  /// that do not yet provide an occupancy curve.
  std::map<int64_t, double> throughputByWarpCount;
};

/// Completed group fit after subtracting a matched nonempty control VF:
/// C(n>0)=first+(n-1)*increment. No empty-harness cost is charged to an access.
/// The key includes full logical geometry and concrete VF warp count/factor.
/// Populating it requires a matching lowering/probe, not shape alone.
struct StageMemoryLayoutCost {
  double firstCycles = 0.0;
  double incrementalCycles = 0.0;
  int64_t minCount = 1;
  int64_t maxCount = 0;
  std::string evidence;
  bool isValid() const;
  std::optional<double> evaluate(double count) const;
};

struct StageAtomicRate {
  double logicalElementsPerCycle = 0.0;
  double operationStartupCycles = 0.0;
  double resultDependencyCycles = 0.0;
  /// Relative penalty used when address collision is a runtime property.  It
  /// is a versioned selection-score policy, not a claim that absolute latency
  /// grows by this factor on every workload.
  double unknownContentionMultiplier = 1.0;

  bool isValid() const;
};

struct StageModeProfile {
  /// Resolved by the evaluator; not an independently editable profile knob.
  int64_t activeWarpCount = 0;
  double setupCycles = 0.0;
  /// Completion-inclusive setup measurements indexed by the launched SIMT
  /// warp count.  The scalar setupCycles remains the documented fallback.
  std::map<int64_t, double> setupCyclesByWarpCount;
  int64_t vectorWidth = 1;
  int64_t issueWidth = 1;
  llvm::StringMap<StageOperationRate> operationRates;
  double loadBytesPerCycle = 0.0;
  double storeBytesPerCycle = 0.0;
  /// Startup cost of one logical load/store operation.  For a continuous
  /// group curve this is used when no shape-matched group-startup curve is
  /// present; otherwise it applies only to uncovered operations.
  double loadOperationSetupCycles = 0.0;
  double storeOperationSetupCycles = 0.0;
  double loadWarpInstructionsPerCycle = 0.0;
  double storeWarpInstructionsPerCycle = 0.0;
  std::map<int64_t, double> loadWarpInstructionsPerCycleByWarpCount;
  std::map<int64_t, double> storeWarpInstructionsPerCycleByWarpCount;
  std::map<std::string, StageMemoryLayoutCost> loadLayoutCosts;
  std::map<std::string, StageMemoryLayoutCost> storeLayoutCosts;
  /// Optimistic full-tile service estimate, not first-round completion or
  /// barrier/stack cost. The unit is fitted address coverage, not an ISA claim.
  int64_t storeFootprintUnitBytes = 0;
  double storeCyclesPerFootprintUnit = 0.0;
  std::map<int64_t, double> storeIssueFloorByWarpCount;
  double predicateOperationsPerCycle = 0.0;
  double shuffleLanesPerCycle = 0.0;
  std::map<int64_t, double> shuffleLanesPerCycleByWarpCount;
  /// Latency of one dependent scan step. For SIMD this is one dependent
  /// vector add; for SIMT it is one dependent shuffle/add/control step.
  double prefixScanStepLatencyCycles = 0.0;
  double dotSetupCycles = 0.0;
  double dotFlopsPerCycle = 0.0;
  double issueOperationsPerCycle = 0.0;
  /// Completion-exclusive incremental cost of one loaded-index load, indexed
  /// by the load's logical warp-instruction count.  Loaded-index work without
  /// a classified load shape is unsupported instead of receiving an invented
  /// generic transaction rate or dependency latency.
  std::map<int64_t, double> indirectLoadSystemCyclesByWarpInstructions;
  /// One-time loaded-index execution envelope for a Stage, indexed by the
  /// same logical warp-instruction shape.  It is the intercept of the same
  /// count sweep as the incremental curve and is paid once, not per load.
  std::map<int64_t, double> indirectLoadStartupSystemCyclesByWarpInstructions;
  /// Measured cost of one physical SuperBlock group's recurrence iteration.
  /// The outer key is the SuperBlock factor; the inner key is the semantic
  /// reduction lane-step count of one logical program.  F logical programs
  /// share this group cost, while route composition accounts for waves once.
  int64_t recurrenceGroupBaseWarpCount = 0;
  std::map<int64_t, std::map<int64_t, double>>
      recurrenceReductionGroupCyclesByFactorAndLaneSteps;
  /// Steady incremental cost of one ordinary FP32 load/store operation in a
  /// physical group. Outer key is SuperBlock factor; inner key is bytes per
  /// logical-program operation. The measurement already includes all F
  /// logical programs and must not be scaled by F again.
  int64_t continuousMemoryGroupBaseWarpCount = 0;
  std::map<int64_t, std::map<int64_t, double>>
      continuousLoadGroupCyclesByFactorAndBytes;
  std::map<int64_t, std::map<int64_t, double>>
      continuousStoreGroupCyclesByFactorAndBytes;
  /// Optional one-time Stage-direction startup fitted independently from the
  /// steady operation slope. It is paid once for all covered shapes, not once
  /// per operation; profiles omit it when n=0/1 evidence is not wave-stable.
  std::map<int64_t, std::map<int64_t, double>>
      continuousLoadGroupStartupCyclesByFactorAndBytes;
  std::map<int64_t, std::map<int64_t, double>>
      continuousStoreGroupStartupCyclesByFactorAndBytes;
  bool isValid(StageMode mode) const;
};

struct HardwareProfile {
  std::string profileVersion;
  std::string target;
  /// Base warp count of one logical SIMT program. This is a compile option,
  /// not a hardware constant. A SuperBlock-F implementation launches
  /// baseSimtWarpCount * F active warps in the physical VF.
  int64_t baseSimtWarpCount = 1;
  StageModeProfile simd;
  StageModeProfile simt;
  ScopeHandoffCost scopeHandoff;

  bool isValid() const;
};

class StageCostEvaluator {
public:
  llvm::Expected<StageCostTable> evaluate(const StagePartition &partition,
                                          const HardwareProfile &profile) const;
};

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_STAGECOSTMODELS_H
