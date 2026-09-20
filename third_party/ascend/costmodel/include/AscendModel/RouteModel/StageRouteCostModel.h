//===- StageRouteCostModel.h - Logical-stage route model -------*- C++ -*-===//
//
// A kernel is represented as serial algorithm stages.  Every Stage is
// implemented entirely by SIMD or entirely by SIMT.  A mixed kernel is a
// route containing both modes; there is deliberately no mixed Stage.
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_ROUTEMODEL_STAGEROUTECOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_STAGEROUTECOSTMODEL_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::ascend {

enum class StageMode { SIMD, SIMT };
enum class StageKernelRouteKind { AllSIMD, AllSIMT, Mixed };
enum class StageScheduleKind {
  StraightLine,
  IndependentPipelined,
  LoopCarriedSerial,
  PartiallyDependent,
};

llvm::StringRef stringifyStageMode(StageMode mode);

struct StageImplementation {
  StageMode mode = StageMode::SIMD;
  /// SIMD always uses factor=1.  For a whole-kernel SIMT implementation this
  /// is the AutoBlockify V1 factor.  A local SIMT implementation identifies a
  /// mixed-kernel candidate whose selected Stage is materialized as a scope.
  /// The current backend still applies factor>1 through the surrounding V1
  /// kernel schedule; it is not an independently widened scope VF.
  int64_t superblockFactor = 1;
  bool localScope = false;

  bool isValid() const;
  llvm::json::Object toJSON() const;
};

/// Structural facts owned by one logical Stage.  Pointer induction is kept
/// separate from a true loop-carried data dependency because later address
/// lowering can remove it without serializing the Stage payload.
struct StageModelFeatures {
  bool hasLoop = false;
  bool hasLoopCarriedDataDependency = false;
  bool hasPointerInduction = false;
  bool hasContiguousMemory = false;
  bool hasIndirectMemory = false;
  bool hasAtomicMemory = false;
  bool hasReduction = false;
  bool hasPrefixScan = false;
  bool hasDot = false;
  bool hasConversionPack = false;
  /// The Stage is inside the AutoBlockify V1 scheduling loop, possibly below
  /// a nested structured region. Route composition uses this execution-domain
  /// fact to distinguish one-time dispatch from work repeated for each
  /// per-slot group.
  bool insideAutoBlockifyV1Loop = false;
  int64_t conditionalBranchCount = 0;
  int64_t divergentBranchCount = 0;
  int64_t loopBackedgeCount = 0;
  // Workload estimates, not legality gates. A bounded loop may execute fewer
  // iterations at runtime (including zero); an unknown loop uses a fallback.
  int64_t upperBoundLoopCount = 0;
  int64_t unknownTripCountLoopCount = 0;
  int64_t synchronizationCount = 0;
  /// Number of mutually independent loop-carried recurrence groups owned by
  /// this Stage.  Each group is serial internally, but SIMT may interleave
  /// different groups on independent warp groups.  Non-recurrence Stages and
  /// a single recurrence use one group.
  int64_t parallelRecurrenceGroupCount = 1;
  double activeLaneRatio = 1.0;

  bool isValid() const;
  bool permitsSimdRoofline() const;
  llvm::json::Object toJSON() const;
};

/// Logical GM access, before SIMD sub-block or SIMT lane mapping. Unknown
/// strides are not zero and a unit inner stride does not imply a packed tile.
struct StageMemoryAccess {
  /// Logical address geometry, not proof of physical instructions or lanes.
  enum class LayoutClass {
    Unknown,
    ContiguousShort,
    ContiguousWide,
    StridedShort,
    StridedWide,
  };
  std::vector<int64_t> shape;
  std::vector<std::optional<int64_t>> strides;
  int64_t elementBits = 0;
  double count = 1.0;
  bool store = false;
  bool masked = false;

  int64_t effectiveRank() const;
  int64_t contiguousAxisBytes() const;
  int64_t rowBytes() const;
  int64_t rowStrideBytes() const;
  int64_t rowCount() const;
  /// Conditional piece count before dimension folding; not a DMA selection.
  int64_t descriptorCount(unsigned maxRank) const;
  LayoutClass layoutClass() const;
  std::string layoutKey() const;
  bool hasStaticLayout() const;
  bool hasUnitInnerStride() const;
  bool hasNonUnitInnerStride() const;
  bool isFlatContiguous() const;
  bool isValid() const;
  llvm::json::Object toJSON() const;
};

llvm::StringRef
stringifyStageMemoryLayout(StageMemoryAccess::LayoutClass layout);

/// Mode-independent work owned exactly once by one Stage.  Values are
/// logical elements/bytes, not mode-specific instructions or cycles.
struct StageWorkload {
  // Element count of one operation -> dynamic occurrences per Stage iteration.
  // Keep operation boundaries until mode-specific lane rounding is applied.
  llvm::StringMap<std::map<int64_t, double>> operationCounts;
  double scalarOperations = 0.0;
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  /// Number of logical load/store operations. Operation boundaries are kept
  /// because bytes alone cannot model short-transfer and pipeline behavior.
  double loadOperations = 0.0;
  double storeOperations = 0.0;
  double loadWarpInstructions = 0.0;
  double storeWarpInstructions = 0.0;
  std::vector<StageMemoryAccess> memoryAccesses;
  /// (bytes, logical elements) in one ordinary contiguous
  /// load/store -> dynamic occurrences. Both dimensions are retained so an
  /// FP16 and FP32 operation with the same byte count cannot silently consume
  /// the same measured curve. Preserve elements before warp rounding, which
  /// otherwise loses dtype information for short operations.
  /// Non-packed/unknown-stride block pointers are
  /// excluded from these curves, but retain their bytes and operation counts
  /// above for analytical fallback. Tensor-pointer continuity is not proven
  /// by this block-pointer check.
  std::map<std::pair<int64_t, int64_t>, double> continuousLoadCounts;
  std::map<std::pair<int64_t, int64_t>, double> continuousStoreCounts;
  /// (bytes, logical elements) in one loaded-index load -> dynamic load
  /// occurrences. Preserve both the data width and operation boundary because
  /// a measured FP16 gather curve is not evidence for an FP32 gather with the
  /// same number of warp instructions.
  std::map<std::pair<int64_t, int64_t>, double> indirectLoadCounts;
  std::map<int64_t, double> predicateCounts;
  /// (scan-axis extent, number of independent sequences) -> dynamic scan
  /// occurrences per Stage iteration. Prefix-scan dependence depth and
  /// available warp/vector parallelism are different dimensions and cannot
  /// be reconstructed from their product in shuffleLaneSteps.
  std::map<std::pair<int64_t, int64_t>, double> prefixScanCounts;
  double shuffleLaneSteps = 0.0;
  /// Dynamic dot matrix instances per Stage iteration (including batch).
  /// Startup is paid per instance; aggregate FLOPs alone cannot represent a
  /// chain of tiny dots.
  double dotOperations = 0.0;
  /// Per-matrix FLOP shape -> dynamic matrix instances.  Tiny-dot legality is
  /// a per matrix property; using aggregate `dotFlops` would incorrectly turn
  /// a chain of small dots into one large-dot Stage.
  std::map<int64_t, double> dotFlopCounts;
  double dotFlops = 0.0;
  double issueElements = 0.0;
  // Dynamic events normalized by iterationCount, like the other workload.
  double loopBackedges = 0.0;
  double conditionalBranches = 0.0;
  double divergentBranches = 0.0;
  double synchronizations = 0.0;
  bool paysKernelSetup = false;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

/// Resource costs for one iteration after raw Stage workload has been mapped
/// through the selected immutable hardware profile. Setup is paid once; all
/// other fields are per iteration.
struct StageResourceCycles {
  double setup = 0.0;
  double scalar = 0.0;
  double load = 0.0;
  double store = 0.0;
  double atomic = 0.0;
  double compute = 0.0;
  double predicate = 0.0;
  double shuffle = 0.0;
  /// Cycles for the tt.scan-contributed portion of shuffle at the ideal rate.
  double scanShuffle = 0.0;
  double dot = 0.0;
  double issue = 0.0;
  /// Prefix scan has two independent lower bounds: aggregate instruction
  /// issue across all sequences and the longest dependent scan chain.
  double prefixScanIssue = 0.0;
  double prefixScanCriticalPath = 0.0;
  /// Candidate-specific measured lower bound for one physical SuperBlock
  /// group's recurrence body iteration.  Unlike the other per-iteration
  /// resources, this already includes the candidate's F logical programs.
  double recurrenceGroupCriticalPath = 0.0;
  /// Candidate-specific measured incremental memory cost for one physical
  /// group. These fields cover only operation shapes present in the profile;
  /// load/store retain any unclassified fallback work.
  double continuousLoadGroup = 0.0;
  double continuousStoreGroup = 0.0;
  /// Full-tile store service floor divided by F, like the analytical store
  /// resource. SuperBlock applies F once. Excludes first-round/sync costs.
  double storeLayoutService = 0.0;
  double criticalPath = 0.0;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

struct StageImplementationCost {
  StageImplementation implementation;
  double totalCycles = 0.0;
  StageResourceCycles resources;
  /// Describes the strongest formula evidence used for this candidate.  It
  /// is report provenance, not a route multiplier.
  std::string formulaEvidence;

  bool isValid() const;
  llvm::json::Object toJSON() const;
};

struct LogicalStageCost {
  std::string id;
  std::string model;
  StageScheduleKind schedule = StageScheduleKind::StraightLine;
  int64_t iterationCount = 1;
  StageModelFeatures features;
  StageWorkload workload;
  int64_t ownedOperationCount = 0;
  /// Unique source locations of the TTIR operations owned by this Stage.
  /// These are calibration provenance only: they let a debug-line-enabled
  /// CaModel artifact map binary PCs back to the immutable StagePartition
  /// without adding marker operations or attributes to production IR.
  std::vector<std::string> sourceLocations;
  int64_t liveInCount = 0;
  int64_t liveOutCount = 0;
  /// Static tensor footprint crossing the Stage boundary.  Counts alone are
  /// insufficient for a mixed route: returning tensor<8xf16> and
  /// tensor<8x1024xf16> are both one SSA value but have very different
  /// register/stack hand-off costs.
  int64_t liveInBytes = 0;
  int64_t liveOutBytes = 0;
  /// Exact tensor footprint crossing the one compound local SIMT scope.
  int64_t scopeInputTensorBytes = 0;
  int64_t scopeOutputTensorBytes = 0;
  std::vector<unsigned> simtAnchorIndices;
  bool localSimtMaterializable = false;
  bool localSuperblockMaterializable = false;
  /// Factors legal for a whole-kernel pure-SIMT schedule.
  std::vector<int64_t> legalSimtFactors;
  /// Factors legal when this Stage alone is materialized as a local scope.
  std::vector<int64_t> localSimtFactors;
  std::vector<StageImplementationCost> implementations;

  llvm::json::Object toJSON() const;
};

struct StageCostTable {
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::string profileVersion;
  int64_t logicalProgramCountHint = 0;
  /// AIV/Vector-core count used by pure-AIV routes.  A Mixed CV executable
  /// owns its V1 logical-program loop in the unsplit MIX/AIC parent and uses
  /// physicalAiCoreCountHint instead; this value remains the execution width
  /// of the local AIV sub-blocks.
  int64_t physicalCoreCountHint = 0;
  /// AIC core count used by Cube-bearing routes.  In particular, a Mixed CV
  /// route uses this count for the shared parent logical-program loop; the
  /// AIV peer receives the same loop bounds after SplitMixKernel.
  int64_t physicalAiCoreCountHint = 0;
  std::vector<LogicalStageCost> stages;
};

struct ScopeHandoffCost {
  /// Completion-inclusive cost of entering and leaving one empty local SIMT
  /// scope.  The two directions are deliberately not exposed as independent
  /// parameters: adjacent SIMD/SIMT pipelines overlap, so directional costs
  /// cannot be recovered by subtracting standalone timings.
  double fixedScopeCycles = 0.0;
  /// Incremental aggregate boundary-copy rates measured with the exact scope
  /// lowering.  They include both register-file sides of the UB handoff and
  /// therefore are not per-thread or raw UB bandwidth claims.
  double inputHandoffBytesPerCycle = 1.0;
  double outputHandoffBytesPerCycle = 1.0;

  bool isValid() const;
  /// Data movement caused by the scope boundary.  This is Stage body work:
  /// the tensor is copied through the boundary, but it is not a mode-switch
  /// instruction.  The factor accounts for all logical programs in one
  /// physical SuperBlock group.
  double estimateScopeBoundaryData(int64_t inputBytes, int64_t outputBytes,
                                   int64_t superblockFactor) const;
  /// Backward-compatible combined query: fixed scope envelope plus boundary
  /// tensor data.  Route solving uses the two terms separately so reports do
  /// not classify data movement as a mode-switch cost.
  double estimateScopeHandoff(int64_t inputBytes, int64_t outputBytes,
                              int64_t superblockFactor) const;
  llvm::json::Object toJSON() const;
};

struct StageRoutePlan {
  StageKernelRouteKind candidate = StageKernelRouteKind::AllSIMD;
  bool legal = false;
  std::vector<StageImplementation> implementations;
  std::vector<double> scopeHandoffCycles;
  std::vector<double> logicalStageCycles;
  int64_t routeSuperblockFactor = 1;
  /// Number of launched physical blocks on the selected route. A Cube-bearing
  /// Mixed kernel launches on AIC blocks; its AIV peer is a sub-block of the
  /// same parent schedule rather than a second logical-program grid.
  int64_t runtimePhysicalProgramCount = 0;
  /// Number of execution slots used to divide logical-program ownership.
  /// Cube-bearing Mixed and all-SIMD routes use physicalAiCoreCountHint;
  /// pure-AIV routes use physicalCoreCountHint.
  int64_t runtimeSchedulingSlotCount = 0;
  int64_t runtimeLogicalProgramsPerSchedulingSlot = 0;
  int64_t runtimeFullGroupCount = 0;
  int64_t runtimeTailProgramCount = 0;
  /// Per-Stage execution-domain schedule. A Mixed kernel can have an AIC
  /// physical entry plus V1/AIV logical-program ownership and a local SIMT
  /// scope, so one kernel-wide Q is not sufficient to explain every Stage.
  /// These vectors are parallel to
  /// `implementations` and `logicalStageCycles`.
  std::vector<int64_t> runtimeStageSchedulingSlotCounts;
  std::vector<int64_t> runtimeStageLogicalProgramsPerSchedulingSlot;
  std::vector<int64_t> runtimeStageFullGroupCounts;
  std::vector<int64_t> runtimeStageTailProgramCounts;
  std::vector<int64_t> runtimeStageLoopIterationCounts;
  /// Execution-domain labels parallel to the per-stage runtime vectors.
  /// These are report provenance, not additional multipliers: they make it
  /// explicit whether a Mixed stage is replayed by the parent driver or by
  /// the local AIV scope. Without this distinction a route JSON can look
  /// as if one kernel-wide Q was applied to every stage.
  std::vector<std::string> runtimeStageExecutionDomains;
  /// Tail schedule labels parallel to the per-stage runtime vectors.  A
  /// local scope with F>1 has a factor-F main group and, when Q%F != 0,
  /// separate factor-1 tail iterations; this label records which case was
  /// used by the cost calculation.
  std::vector<std::string> runtimeStageTailModes;
  /// Number of serialized execution units on the most-loaded physical core:
  /// logical programs for all-SIMD, F-wide groups for pure SIMT, and
  /// full-groups plus individual F1 tails for Mixed.
  int64_t runtimeLoopIterationCount = 1;
  double totalCycles = 0.0;

  llvm::json::Object toJSON() const;
};

struct StageCostModelSummary {
  bool applied = false;
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::string profileVersion;
  std::vector<LogicalStageCost> stages;
  ScopeHandoffCost scopeHandoff;
  StageRoutePlan allSimd;
  StageRoutePlan allSimt;
  StageRoutePlan mixed;

  llvm::json::Object toJSON() const;
};

llvm::Expected<StageCostModelSummary>
solveStageRoutes(const StageCostTable &costTable,
                 const ScopeHandoffCost &scopeHandoff);

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_STAGEROUTECOSTMODEL_H
