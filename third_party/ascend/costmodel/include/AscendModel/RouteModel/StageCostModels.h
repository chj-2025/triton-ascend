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
  ScalarLoad,
  ScalarStore,
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

struct StageControlFlowRates {
  double loopBackedgeCycles = 0.0;
  double conditionalBranchCycles = 0.0;
  double divergentBranchPenaltyCycles = 0.0;
  double synchronizationCycles = 0.0;

  bool isFiniteAndNonNegative() const;
};

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
  std::optional<SimtRegisterPressure> wholeSimtRegisterPressure;
  /// Conservative pre-lowering UB footprint of one logical SIMT program.
  /// For a known dot geometry this is the simultaneous lhs/rhs/accumulator
  /// tensor footprint.  SuperBlock legality multiplies it by F and compares
  /// it with the target's usable UB capacity; this is a structural resource
  /// bound, never a workload or operator-name exception.
  int64_t simtUbWorkingSetBytesPerProgram = 0;
  int64_t simtUsableUbBytes = 0;
  std::vector<int64_t> simtFactorsRejectedByUb;
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
  /// Optional completion curve for one semantic operation instance:
  ///   startup + max(0, work_units - free_work_units) / throughput * factor.
  /// The defaults preserve the ordinary throughput-only formula.
  double startupCycles = 0.0;
  double freeWorkUnits = 0.0;
  /// Minimum logical element count for charging startupCycles. This keeps a
  /// completion curve measured from tensor lowering from leaking into scalar
  /// region operations that share the same arithmetic instruction class.
  int64_t minimumStartupElements = 1;
  /// Optional one-time startup of this operation's dynamic grid-stride
  /// lowering. It is kept separate from startupCycles because route
  /// composition pays it once per kernel instead of once per source program.
  double gridStrideKernelStartupCycles = 0.0;
  /// True when the measured semantic completion already contains the fixed
  /// enter/exit envelope of a local SIMT scope. Boundary tensor traffic stays
  /// explicit because it depends on the surrounding Stage ABI.
  bool includesLocalScopeHandoff = false;
  /// Aggregate throughput of one SIMT VF indexed by its launched warp count.
  /// Each concrete Stage implementation selects its own point using
  /// baseWarpCount * SuperBlockFactor; throughput is the fallback for profiles
  /// that do not yet provide an occupancy curve.
  std::map<int64_t, double> throughputByWarpCount;
};

/// Completion-inclusive cost of one straight-line prefix-scan helper,
/// selected by the TTIR combine class. Memory remains owned by its load/store
/// Stages. The helper itself is expressed only through semantic scan geometry:
///   one sequence: interpolate(C_single(axisExtent))
///   multiple sequences: startup + elements / elementsPerCycle
///                     + (axisExtent - 1) * dependencyStepCycles.
struct StraightLinePrefixScanCost {
  double startupCycles = 0.0;
  double elementsPerCycle = 0.0;
  double dependencyStepCycles = 0.0;
  /// A single independent sequence is lowered through a distinct backend
  /// helper whose power-of-two extent changes its implementation regime.
  /// This structural curve takes precedence over the multi-sequence
  /// throughput/dependency formula when independentSequences == 1.
  std::map<int64_t, double> singleSequenceCompletionCyclesByExtent;
  std::string evidence;
  /// A non-unit scan axis lowers through a different backend helper.  Its
  /// completion is governed by total semantic elements within a measured TTIR
  /// geometry rectangle.  Zero fields disable this optional curve.
  double nonUnitStrideStartupCycles = 0.0;
  double nonUnitStrideElementsPerCycle = 0.0;
  int64_t nonUnitStrideMinimumAxisExtent = 0;
  int64_t nonUnitStrideMaximumAxisExtent = 0;
  int64_t nonUnitStrideMinimumAxisStride = 0;
  int64_t nonUnitStrideMaximumAxisStride = 0;
  int64_t nonUnitStrideMinimumSemanticElements = 0;
  int64_t nonUnitStrideMaximumSemanticElements = 0;
  std::string nonUnitStrideEvidence;

  bool isValid() const;
};

/// Offline fit from TTIR-visible source facts to one generated SIMT vector
/// stream.  `directInstructionWeight` applies to the semantic warp
/// instruction count; `peakLiveRegisterWeight` applies to the allocator
/// excess (peak live units minus the selected register budget).  The fit is
/// deliberately optional because TTIR alone cannot prove a physical stream.
struct SimtSourceFit {
  bool enabled = false;
  double intercept = 0.0;
  double directInstructionWeight = 0.0;
  double segmentWeight = 0.0;
  double strideRatioWeight = 0.0;
  double maskedWeight = 0.0;
  double peakLiveRegisterWeight = 0.0;
  /// Weight for the effective backend unroll excess (unroll factor - 1).
  /// This is used only when a lowering supplies explicit unroll evidence.
  double unrollFactorWeight = 0.0;

  bool isValid() const;
  double evaluate(double directInstructions, double segments,
                  double strideRatio, double maskedAccesses,
                  double peakLiveExcess, double unrollFactor = 1.0) const;
};

/// Conditional one-time private-stack floor for a function-wide dot DAG.
/// The instruction floors and complete-group prices are frozen together;
/// matching is structural and never uses an operator or Stage name.
struct DotComponentPrivateStackFloor {
  bool enabled = false;
  int64_t activeWarpCount = 0;
  int64_t superblockFactor = 0;
  int64_t minimumComponentNodes = 0;
  int64_t maximumComponentNodes = 0;
  int64_t m = 0, n = 0, k = 0;
  int64_t elementBits = 0;
  std::string dotUseSignature;
  int64_t loadInstructionFloor = 0;
  int64_t storeInstructionFloor = 0;
  /// End-to-end price of the load/store group in one compiler schedule.  This
  /// is a floor on the combined stream, not the sum of isolated opcode prices.
  double combinedGroupCycles = 0.0;
  /// Structural attribution of the whole-function group envelope.  The
  /// fractions sum to one. They allow phase diagnostics to place the charge
  /// before a serial recurrence, in the following dot region, and after the
  /// last dot without changing the route-level total.
  double preRecurrenceFraction = 0.0;
  double dotRegionFraction = 0.0;
  double postDotFraction = 0.0;
  std::string evidence;

  bool isValid() const;
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
  bool includesSharedLayout = false;
  bool isValid() const;
  std::optional<double> evaluate(double count) const;
};

struct PipelinedStridedStoreGroupCost {
  double kernelStartupCycles = 0.0;
  double firstStoreCycles = 0.0;
  double incrementalStoreCycles = 0.0;
  double extraLaneCycles = 0.0;
  int64_t minimumRows = 0;
  int64_t maximumRows = 0;
  int64_t minimumStores = 0;
  int64_t maximumStores = 0;
  std::string evidence;

  bool isValid() const;
};

/// Completion floor for the common masked copy shape with one rank-one FP32
/// contiguous access and one rank-one FP32 strided access.  The two directions
/// lower differently: a strided destination uses the scatter store service,
/// while a strided source uses the compacting load service.  The fit is
/// bounded by source-visible row count and static loop trip count.  It never
/// matches an operator or an exact kernel shape.
struct MaskedStridedCopyCompletionCost {
  double baseSystemCycles = 0.0;
  double additionalRowSystemCycles = 0.0;
  int64_t minimumRows = 0;
  int64_t maximumRows = 0;
  int64_t minimumIterations = 0;
  int64_t maximumIterations = 0;
  std::string evidence;

  bool isValid() const;
  std::optional<double> evaluate(int64_t rows, int64_t iterations) const;
};

struct StageAtomicRate {
  double logicalElementsPerCycle = 0.0;
  double operationStartupCycles = 0.0;
  double resultDependencyCycles = 0.0;
  /// Relative penalty used when address collision is a runtime property.  It
  /// is a versioned selection-score policy, not a claim that absolute latency
  /// grows by this factor on every workload.
  double unknownContentionMultiplier = 1.0;
  /// Optional rate for the one-physical-program hardware regime. Zero keeps
  /// the ordinary rate. Route composition selects it only for an all-SIMD
  /// launch with exactly one physical program.
  double singlePhysicalProgramLogicalElementsPerCycle = 0.0;

  bool isValid() const;
};

struct StageModeProfile {
  /// Resolved by the evaluator; not an independently editable profile knob.
  int64_t activeWarpCount = 0;
  double setupCycles = 0.0;
  /// Completion-inclusive setup measurements indexed by the launched SIMT
  /// warp count.  The scalar setupCycles remains the documented fallback.
  std::map<int64_t, double> setupCyclesByWarpCount;
  /// Physical SIMD instruction width. Keep bits so FP16 and FP32 segmented
  /// operations are rounded against the same hardware vector instruction.
  /// Physical SIMD instruction width.  Unlike vectorWidth, this preserves
  /// bytes/bits so FP16 and FP32 short-axis segments are priced correctly.
  int64_t vectorWidthBits = 1;
  int64_t vectorWidth = 1;
  int64_t issueWidth = 1;
  /// Scalar white-box terms in SYS_CNT cycles. These coefficients describe
  /// the target memory pipelines and are independent of any operator.
  double mainScalarLoadPrepCycles = 0.0;
  double mainScalarLoadFillCycles = 0.0;
  double mainScalarLoadIssueCycles = 0.0;
  double mainScalarLoadOutstandingLines = 0.0;
  double mainScalarLoadExtraLineLowCycles = 0.0;
  double mainScalarLoadExtraLineHighCycles = 0.0;
  double mainScalarLoadExtraLineHighThreshold = 0.0;
  double simtUniformLoadPrepCycles = 0.0;
  double simtUniformLoadFillCycles = 0.0;
  double simtUniformLoadDiffLineIssueCycles = 0.0;
  double mte3StorePrepCycles = 0.0;
  double mte3StoreFillCycles = 0.0;
  double simtUniformStoreBaseCycles = 0.0;
  /// Zero means no declared limit. A positive value is a compiler/target
  /// legality bound for reverse scans in this mode.
  int64_t maximumReversePrefixScanExtent = 0;
  llvm::StringMap<StageOperationRate> operationRates;
  double loadBytesPerCycle = 0.0;
  double storeBytesPerCycle = 0.0;
  /// Startup cost of one logical load/store operation.  For a continuous
  /// group curve this is used when no shape-matched group-startup curve is
  /// present; otherwise it applies only to uncovered operations.
  double loadOperationSetupCycles = 0.0;
  double storeOperationSetupCycles = 0.0;
  /// SIMD independent-loop MTE2 envelope for two or more uninterrupted FP32
  /// load streams. Startup is paid once per Stage, then the grouped byte rate
  /// prices one dynamic loop iteration.
  double independentLoadGroupKernelStartupCycles = 0.0;
  double independentLoadGroupIterationStartupCycles = 0.0;
  double independentLoadGroupBytesPerCycle = 0.0;
  int64_t independentLoadGroupMinimumOperations = 0;
  int64_t independentLoadGroupMaximumOperations = 0;
  /// Completion cost for a SIMD load whose tensor address is computed by a
  /// non-affine floating-point DAG and therefore lowers to scalar lane
  /// service. Zero disables the optional measured contract.
  int64_t computedAddressScalarizedLoadMinimumElements = 0;
  int64_t computedAddressScalarizedLoadMaximumElements = 0;
  double computedAddressScalarizedLoadCyclesPerElement = 0.0;
  std::string computedAddressScalarizedLoadEvidence;
  /// Residual SIMD completion curve for a rank-one FP32 load whose tensor
  /// address is selected by scalar control flow and lowered lane by lane.
  std::map<int64_t, double>
      controlFlowAddressScalarizedLoadResidualCyclesByElements;
  std::string controlFlowAddressScalarizedLoadEvidence;
  /// Residual completion of the rank-two FP16 SIMD masked load/store group.
  /// Outer key is the physical row width in bytes; the bounded inner curve is
  /// indexed by statically proven active rows. No value is extrapolated.
  std::map<int64_t, std::map<int64_t, double>>
      maskedLoadStoreRmwResidualCyclesByRowBytesAndActiveRows;
  std::string maskedLoadStoreRmwEvidence;
  /// SIMD scalar-row store groups indexed by row stride in bytes. The curve
  /// separates the first store, additional stores, and lanes beyond the
  /// measured 32-lane baseline.
  std::map<int64_t, PipelinedStridedStoreGroupCost>
      pipelinedStridedStoreGroupsByRowStrideBytes;
  /// Directional completion floors for masked FP32 strided copies.  The key
  /// is the physical byte stride of the strided side.
  std::map<int64_t, MaskedStridedCopyCompletionCost>
      maskedStridedLoadCopyCompletionByStrideBytes;
  std::map<int64_t, MaskedStridedCopyCompletionCost>
      maskedStridedStoreCopyCompletionByStrideBytes;
  /// Generic SIMD completion floor for statically padded rectangular tiles.
  /// It is derived from physical and provably active elements, never from an
  /// operator name or exact shape key.
  bool irregularTileCompletionEnabled = false;
  int64_t irregularTilePhysicalThresholdElements = 0;
  double irregularTilePhysicalExcessCyclesPerElement = 0.0;
  double irregularTileActiveCyclesPerElement = 0.0;
  double irregularTileTailCyclesPerInactiveElement = 0.0;
  int64_t irregularTileRowGroupSize = 0;
  int64_t irregularTileSmallRowMaximumRows = 0;
  double irregularTileLargeRowMaskedFloorScale = 1.0;
  /// Fraction of the physical masked-tile materialization that remains
  /// visible when the tile contains exactly two complete architectural row
  /// groups.  The remainder is already owned by the rowwise completion
  /// envelope; this is a structural geometry coefficient, not an operator
  /// correction.
  double irregularTileAlignedTwoRowGroupPhysicalMaterializationScale = 0.0;
  /// Above this complete-kernel read/write footprint, the resident partial
  /// materialization ownership no longer applies and the rowwise Stage pays
  /// the full physical masked-tile materialization.
  int64_t irregularTileLargeWorkingSetFullPhysicalMinimumBytes = 0;
  double irregularTileAlignedFirstRowGroupCyclesPerElement = 0.0;
  double irregularTileAlignedAdditionalRowGroupCyclesPerElement = 0.0;
  std::string irregularTileCompletionEvidence;
  double loadWarpInstructionsPerCycle = 0.0;
  double storeWarpInstructionsPerCycle = 0.0;
  /// Optional rates for post-TTIR generated vector streams.  A zero rate
  /// falls back to the ordinary SIMT load/store rate.
  double sharedLocalLoadInstructionsPerCycle = 0.0;
  double sharedLocalStoreInstructionsPerCycle = 0.0;
  double privateStackLoadInstructionsPerCycle = 0.0;
  double privateStackStoreInstructionsPerCycle = 0.0;
  /// Register budget in 32-bit units/thread.  Zero derives the tested
  /// launch-bound curve min(127, floor(1024 / activeWarps)).
  double registerBudgetPerThread = 0.0;
  double spillTransactionCoverage = 1.0;
  SimtSourceFit globalLoadSourceFit;
  SimtSourceFit globalStoreSourceFit;
  SimtSourceFit sharedLoadSourceFit;
  SimtSourceFit sharedStoreSourceFit;
  SimtSourceFit privateLoadSourceFit;
  SimtSourceFit privateStoreSourceFit;
  /// Conditional whole-VF floors. Multiple independently calibrated dot/use
  /// families may share one active-warp count; exact structural matching
  /// selects one non-overlapping envelope. Each entry also carries its
  /// SuperBlock factor, so a 4/8/16-warp measurement cannot silently price
  /// another launch shape.
  std::vector<DotComponentPrivateStackFloor> dotComponentPrivateStackFloors;
  /// Whole-kernel entry convergence and final-drain completion curves.  They
  /// are consumed after runtime group replay, because their independent
  /// variable is the per-slot physical-group count rather than one Stage
  /// iteration.
  std::vector<WholeSimdCompletionCurve> wholeSimdCompletionCurves;
  std::vector<WholeSimtCompletionCurve> wholeSimtCompletionCurves;
  std::map<int64_t, double> loadWarpInstructionsPerCycleByWarpCount;
  std::map<int64_t, double> storeWarpInstructionsPerCycleByWarpCount;
  std::map<std::string, StageMemoryLayoutCost> loadLayoutCosts;
  std::map<std::string, StageMemoryLayoutCost> storeLayoutCosts;
  /// FMA accumulator -> row-major store layout exchange. Keys describe the
  /// measured dtype/shape/base-warps/F domain, not an operator or Stage name.
  std::map<std::string, StageMemoryLayoutCost> dotStoreSharedLayoutCosts;
  /// Composite whole-SIMT FMA-dot paths, including operand GM/shared traffic.
  /// Key includes TTIR geometry, base warp count and SuperBlock factor.
  std::map<std::string, StageMemoryLayoutCost> dotOperandPathCosts;
  /// Optimistic full-tile service estimate, not first-round completion or
  /// barrier/stack cost. The unit is fitted address coverage, not an ISA claim.
  int64_t storeFootprintUnitBytes = 0;
  double storeCyclesPerFootprintUnit = 0.0;
  std::map<int64_t, double> storeIssueFloorByWarpCount;
  double predicateOperationsPerCycle = 0.0;
  /// Source-level row-reduction envelope.  SIMD consumes the measured
  /// operation startup and lane-step rate; SIMT falls back to its measured
  /// warp-shuffle rate because no separate SIMT reduction fit is claimed.
  double reductionOperationSetupCycles = 0.0;
  double reductionLaneStepsPerCycle = 0.0;
  /// Current SIMD lowering uses a serialized scalar UB loop for multi-result
  /// reductions above a structural extent threshold.  These parameters price
  /// that compiler template from TTIR geometry; zero disables the surface.
  int64_t scalarizedMultiResultReductionMinimumExtent = 0;
  int64_t scalarizedMultiResultReductionReferenceResults = 0;
  double scalarizedMultiResultReductionSetupCycles = 0.0;
  double scalarizedMultiResultReductionElementsPerCycle = 0.0;
  std::string scalarizedMultiResultReductionEvidence;
  /// Completion service of ascend.sort/hivm.hir.vsort in its measured
  /// geometry domain.  The backend primitive is row-serviced for extents in
  /// this interval, so rows are the throughput variable.
  int64_t sortMinimumExtent = 0;
  int64_t sortMaximumExtent = 0;
  int64_t sortElementBits = 0;
  double sortSetupCycles = 0.0;
  double sortRowsPerCycle = 0.0;
  std::string sortEvidence;
  /// One-time completion envelope for a reduction of a physically wide,
  /// zero-filled accumulator whose exact active prefix is proven from a
  /// one-trip TTIR loop. This is paid once by the owning Stage setup.
  double maskedAccumulatorReductionCompletionSetupCycles = 0.0;
  /// Completion-inclusive cost of one physical SIMT group executing a
  /// structurally matched row-reduction body.  The dimensions are
  /// (rows, contiguous reduction extent); the outer map is SuperBlock F.
  /// Curves are bounded to their measured geometry and contain no operator or
  /// kernel identity.  The reference operation count prevents a two-reduction
  /// envelope from silently pricing a different fused reduction graph.
  int64_t rowwiseReductionGroupBaseWarpCount = 0;
  int64_t rowwiseReductionGroupReferenceOperations = 0;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      rowwiseReductionGroupCyclesByFactorRowsAndExtent;
  /// Additional completion surfaces keyed first by the exact number of TTIR
  /// reduction operations.  This lets one- and two-reduction graphs share the
  /// same structural formula without applying either surface to the other.
  std::map<int64_t,
           std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>>
      rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent;
  std::string rowwiseReductionGroupEvidence;
  /// Whole-SIMT completion body for one multi-result tt.reduce.  The backend
  /// lowers a padded reduction axis through distinct templates selected by
  /// the exact tail alignment class.  Each class owns a bounded row curve;
  /// the online selector uses only TTIR rows/extent/result count/element bits.
  int64_t multiResultReductionGroupBaseWarpCount = 0;
  int64_t multiResultReductionResultCount = 0;
  int64_t multiResultReductionElementBits = 0;
  int64_t multiResultReductionPaddedAxisExtent = 0;
  std::map<std::string, std::map<int64_t, double>>
      multiResultReductionGroupCyclesByTailClassAndRows;
  std::string multiResultReductionGroupEvidence;
  /// Route-level completion for the structural two-pass pattern containing
  /// one row reduction and two segmented loaded-index row reads.  SIMT may
  /// transfer ownership of the isolated segmented-read envelopes to the
  /// complete composition; SIMD uses a bounded (extent,Q) residual surface.
  /// Neither mechanism assumes cross-Stage pipeline overlap.
  bool twoPassRowwiseOwnsSegmentedIndirectLoads = false;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      twoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot;
  std::string twoPassRowwiseCompletionEvidence;
  double shuffleLanesPerCycle = 0.0;
  std::map<int64_t, double> shuffleLanesPerCycleByWarpCount;
  /// Latency of one dependent scan step. For SIMD this is one dependent
  /// vector add; for SIMT it is one dependent shuffle/add/control step.
  double prefixScanStepLatencyCycles = 0.0;
  /// Completion-inclusive throughput of the backend external helper used by
  /// a straight-line SIMT scan.  The semantic element count is derived from
  /// TTIR scan geometry.  Zero keeps the decomposed analytical formula.
  double straightLinePrefixScanElementsPerCycle = 0.0;
  std::string straightLinePrefixScanCompletionEvidence;
  /// SIMD helper completion curves keyed by structural combine class. These
  /// are used only for straight-line scans; dynamic grid-stride scans retain
  /// their independently measured Q/extent curves.
  std::map<std::string, StraightLinePrefixScanCost>
      straightLinePrefixScanCostsByCombine;
  /// Dynamic source-loop prefix-scan body/startup curves, indexed by semantic
  /// scan extent and kept separate by direction.
  std::map<int64_t, double> gridStrideForwardScanCyclesByExtent;
  std::map<int64_t, double> gridStrideForwardScanStartupCyclesByExtent;
  std::map<int64_t, double> gridStrideReverseScanCyclesByExtent;
  std::map<int64_t, double> gridStrideReverseScanStartupCyclesByExtent;
  /// Factor-specific source-loop scan service.  Source iterations remain
  /// serialized at Q=ceil(P/C); these curves capture the measured active-warp
  /// resource rate of the selected physical SuperBlock without dividing Q.
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideForwardScanCyclesByFactorAndExtent;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideForwardScanStartupCyclesByFactorAndExtent;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideReverseScanCyclesByFactorAndExtent;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideReverseScanStartupCyclesByFactorAndExtent;
  /// Steady-state direct-memory rates for a source grid-stride loop whose
  /// complete read/write footprint remains in the measured resident domain.
  /// The route solver owns this decision because only it knows the total
  /// logical-program count.  Zero disables the optional resident curve.
  int64_t gridStrideResidentMemoryMaximumBytes = 0;
  double gridStrideResidentLoadWarpInstructionsPerCycle = 0.0;
  double gridStrideResidentStoreWarpInstructionsPerCycle = 0.0;
  std::string gridStrideResidentMemoryEvidence;
  double prefixScanDependencyFactor = 1.0;
  double dotSetupCycles = 0.0;
  double dotFlopsPerCycle = 0.0;
  /// Aggregate dependent-dot throughput selected by the physical active-warp
  /// count.  Each point comes from an F sweep of the same structural dot
  /// microbenchmark; it is a hardware occupancy curve, not an operator fit.
  std::map<int64_t, double> dotFlopsPerCycleByWarpCount;
  /// Relative price applied to the generic residual when a whole-SIMT Stage
  /// matches at least one exact dot-operand path but leaves other dot work
  /// unmatched.  The generic setup/rate above already includes the full
  /// completion scale, so this ratio converts it to the independently fitted
  /// partial-composite residual scale.  One preserves the generic price.
  double dotPartialCompositeResidualRelativeScale = 1.0;
  double issueOperationsPerCycle = 0.0;
  /// Target UB capacity available to one physical SIMT group.  Zero means the
  /// profile does not provide a pre-lowering UB legality bound.
  int64_t usableUbBytes = 0;
  /// Completion-exclusive incremental cost of one loaded-index load, indexed
  /// by the load's logical warp-instruction count.  Loaded-index work without
  /// a classified load shape is unsupported instead of receiving an invented
  /// generic transaction rate or dependency latency.
  std::map<int64_t, double> indirectLoadSystemCyclesByWarpInstructions;
  /// SIMD scalarizes a loaded-index tensor into lane transactions.  This
  /// curve preserves sub-warp active-element information that ceil(N/32)
  /// discards; SIMT continues to use the warp-instruction curves below.
  std::map<int64_t, double> indirectLoadSystemCyclesByElements;
  /// One-time loaded-index execution envelope for a Stage, indexed by the
  /// same logical warp-instruction shape.  It is the intercept of the same
  /// count sweep as the incremental curve and is paid once, not per load.
  std::map<int64_t, double> indirectLoadStartupSystemCyclesByWarpInstructions;
  /// Completion-exclusive scatter service indexed by the number of distinct
  /// destination segments in one TTIR store.  A contiguous inner run is one
  /// segment; a lane-wise scatter contributes one segment per element.
  std::map<int64_t, double> indirectStoreSystemCyclesBySegments;
  /// Physical-group loaded-index curves measured independently for each
  /// SuperBlock factor.  These supersede the legacy F1 curve when present.
  /// Startup is a kernel-wide envelope and is composed once by the route,
  /// while the incremental curve is charged for every executed group.
  std::map<int64_t, std::map<int64_t, double>>
      indirectLoadGroupCyclesByFactorAndWarpInstructions;
  std::map<int64_t, std::map<int64_t, double>>
      indirectLoadGroupStartupCyclesByFactorAndWarpInstructions;
  /// Completion-exclusive segmented loaded-index service.  Dimensions are
  /// SuperBlock factor, distinct loaded rows/segments, and proven contiguous
  /// elements in each segment.
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      segmentedIndirectLoadGroupCyclesByFactorSegmentsAndContiguousElements;
  /// FP32 counterpart of the segmented loaded-index surface.  Keeping the
  /// element width explicit avoids applying FP16 row-copy evidence to an FP32
  /// gather with the same logical geometry.
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      segmentedIndirectLoadFp32GroupCyclesByFactorSegmentsAndContiguousElements;
  /// Dynamic grid-stride loops serialize successive source-loop bodies on
  /// one resident program.  These isolated curves are separate from the
  /// static-range group curves, whose unrolled operations can overlap.
  std::map<int64_t, double> gridStrideIndirectLoadCyclesByWarpInstructions;
  std::map<int64_t, double>
      gridStrideIndirectLoadStartupCyclesByWarpInstructions;
  /// Factor-specific dynamic source-loop service.  Unlike the static-range
  /// group curves above, these values describe one serialized source-loop
  /// iteration at the selected active-warp configuration.
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideIndirectLoadCyclesByFactorAndWarpInstructions;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideIndirectLoadStartupCyclesByFactorAndWarpInstructions;
  /// Completion-inclusive dot cost for one dynamic grid-stride body,
  /// indexed by semantic FLOPs of one dot, plus its route-shared startup.
  std::map<int64_t, double> gridStrideDotCyclesByFlops;
  std::map<int64_t, double> gridStrideDotStartupCyclesByFlops;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideDotCyclesByFactorAndFlops;
  std::map<int64_t, std::map<int64_t, double>>
      gridStrideDotStartupCyclesByFactorAndFlops;
  /// Residual of composing a loaded-index body and a dot body in the same
  /// dynamic source loop after inclusion-exclusion removes their shared copy
  /// baseline. Dimensions are (factor, indirect warp instructions, dot FLOPs).
  /// This is serialized completion evidence and assumes no pipeline overlap.
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      gridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      gridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry;
  std::string gridStrideIndirectDotCompositionEvidence;
  /// Measured cost of one physical SuperBlock group's recurrence iteration.
  /// The outer key is the SuperBlock factor; the inner key is the semantic
  /// reduction lane-step count of one logical program.  F logical programs
  /// share this group cost, while route composition accounts for waves once.
  int64_t recurrenceGroupBaseWarpCount = 0;
  std::map<int64_t, std::map<int64_t, double>>
      recurrenceReductionGroupCyclesByFactorAndLaneSteps;
  /// One-time startup of the same measured recurrence group.  The outer and
  /// inner keys match the iteration curve above.  This intercept is paid once
  /// per physical runtime group, independently of the recurrence trip count.
  std::map<int64_t, std::map<int64_t, double>>
      recurrenceReductionGroupStartupCyclesByFactorAndLaneSteps;
  /// Completion-inclusive SIMD cost for a loop-carried recurrence whose one
  /// dynamic iteration performs exactly one single-row, single-result FP32
  /// reduction. This lowering is structurally different from the matrix
  /// recurrence above even when aggregate shuffle lane-steps are similar.
  /// Every geometry/program/iteration dimension is bounded by measurements.
  bool singleResultRowReductionRecurrenceCompletionEnabled = false;
  int64_t singleResultRowReductionRecurrenceMinimumExtent = 0;
  int64_t singleResultRowReductionRecurrenceMaximumExtent = 0;
  int64_t singleResultRowReductionRecurrenceMinimumIterations = 0;
  int64_t singleResultRowReductionRecurrenceMaximumIterations = 0;
  int64_t singleResultRowReductionRecurrenceMinimumPrograms = 0;
  int64_t singleResultRowReductionRecurrenceMaximumPrograms = 0;
  double singleResultRowReductionRecurrenceIterationCycles = 0.0;
  double singleResultRowReductionRecurrenceStartupInterceptCycles = 0.0;
  double singleResultRowReductionRecurrenceStartupSqrtProgramCycles = 0.0;
  std::string singleResultRowReductionRecurrenceEvidence;
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
  /// High-wave completion service for the SIMD parent of a Mixed route.  It
  /// scales with compiler-derived parent memory bytes and AIC loop count.
  bool mixedParentMemoryCompletionEnabled = false;
  int64_t mixedParentMemoryMinimumIterations = 0;
  int64_t mixedParentMemoryReferenceBytesPerProgram = 0;
  double mixedParentMemoryCyclesPerExcessIteration = 0.0;
  std::string mixedParentMemoryCompletionEvidence;
  double spillTransactionsPerCycle = 0.0;
  /// Loaded-index memory cannot use the continuous MTE/LSU throughput model.
  /// These rates operate on logical warp/transaction counts and include one
  /// uncovered dependency latency per Stage iteration.
  double indirectLoadTransactionsPerCycle = 0.0;
  double indirectStoreTransactionsPerCycle = 0.0;
  double indirectDependencyLatencyCycles = 0.0;
  llvm::StringMap<StageAtomicRate> atomicRates;
  StageControlFlowRates controlFlow;

  bool isValid(StageMode mode) const;
};

struct HardwareProfile {
  std::string profileVersion;
  std::string target;
  /// Base warp count of one logical SIMT program. This is a compile option,
  /// not a hardware constant. A SuperBlock-F implementation launches
  /// baseSimtWarpCount * F active warps in the physical VF.
  int64_t baseSimtWarpCount = 1;
  /// CV1:2 work partition, applied before local SuperBlock grouping. This is
  /// not a divisor of Q or of the launch warp count.
  int64_t cvAivSubblocks = 1;
  /// Resource/grid bound for analytical estimates, independent of anchors and
  /// executable candidate enablement. Does not grant materialization legality.
  int64_t maximumSimtEstimationFactor = 4;
  /// Logical warp groups available to one SIMT program.  This is a compile
  /// option, not a hardware constant, and bounds cross-group interleaving in
  /// recurrence Stage models.
  int64_t logicalWarpGroupCount = 1;
  /// Long-lived recurrence state consumes finite register/stack bandwidth.
  /// The byte rate is shared by the SIMD recurrence-state term and the extra
  /// pressure created when a SIMT SuperBlock replicates that state; neither
  /// formula depends on a workload name.
  /// Largest factor that still gives proportional latency-hiding benefit.
  int64_t superblockUsefulFactorLimit = 1;
  /// Largest factor that may replicate loop-carried live state without an
  /// explicit persistent-state pressure charge.  This is intentionally
  /// independent from the latency-hiding limit: straight-line kernels may
  /// benefit through F4 while recurrence state becomes expensive above F2.
  int64_t superblockPersistentStatePressureFreeFactor = 1;
  double superblockPersistentStateBytesPerCycle = 1.0;
  StageModeProfile simd;
  StageModeProfile simt;
  ScopeHandoffCost scopeHandoff;

  bool isValid() const;
};

class StageCostEvaluator {
public:
  llvm::Expected<StageCostTable>
  evaluate(const StagePartition &partition, const HardwareProfile &profile,
           bool collectCalibrationEstimates = true) const;
};

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_STAGECOSTMODELS_H
