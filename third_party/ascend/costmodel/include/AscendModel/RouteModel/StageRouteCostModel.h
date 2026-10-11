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

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <tuple>
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

/// Completion-inclusive correction for a structurally matched whole-SIMT
/// dot schedule.  Runtime scheduling supplies the number of physical
/// factor-F groups on the most-loaded AIV.  The curve is keyed by the
/// function-wide dot/use topology and launch shape, never by an operator,
/// source location, or B/T/H whitelist.
struct WholeSimtCompletionCurve {
  bool enabled = false;
  int64_t activeWarpCount = 0;
  int64_t superblockFactor = 0;
  int64_t minimumComponentNodes = 0;
  int64_t maximumComponentNodes = 0;
  int64_t m = 0, n = 0, k = 0;
  int64_t elementBits = 0;
  std::string dotUseSignature;
  int64_t referenceRowStrideBytes = 0;
  double referencePreRecurrenceScalarOperations = 0.0;

  /// Optional factor-specific corrections for the measured recurrence and
  /// post-recurrence dot body.  They are applied only after the exact
  /// function-wide dot/use topology above has matched.
  double recurrenceScale = 1.0;
  double dotRegionScale = 1.0;

  /// Missing entry/convergence service per physical group:
  /// g*(c0+c1*L+c2*R+c3*L*R+c4*S+c5*L*S)+c6+c7*L,
  /// where L=log2(stride/ref), S is the pre-recurrence scalar-operation
  /// excess and R=g^power/(g^power+midpoint^power).
  double entryGroupMidpoint = 0.0;
  double entryGroupPower = 0.0;
  std::array<double, 8> entryAdditiveCoefficients = {};

  /// Completion-inclusive final-drain service using the same eight-term
  /// basis.  The ramp can be exponential or Hill.  A final partial scheduling
  /// wave adds c_tail*-log2(active_slot_fraction).  The route model raises the
  /// final two Stage total to this floor; it never subtracts work.
  bool exitRampExponential = true;
  double exitGroupTimeConstant = 0.0;
  double exitGroupMidpoint = 0.0;
  double exitGroupPower = 0.0;
  std::array<double, 8> exitAbsoluteCoefficients = {};
  double exitTailLog2ActiveFractionCoefficient = 0.0;

  /// Scalar prologue/epilogue work outside the four instrumented body phases.
  /// This contributes to the whole-kernel total and is reported separately.
  double kernelBoundarySystemCycles = 0.0;
  std::string evidence;

  bool isValid() const;
};

/// Completion curve for a structurally matched whole-SIMD dot schedule.
/// It corrects four effects which isolated opcode rates cannot own safely:
/// entry convergence, recurrence pre/post work, a branched dot DAG, and final
/// store drain.  The match is bounded by compiler-derived structure, row
/// stride and runtime group count; it never uses an operator or shape name.
struct WholeSimdCompletionCurve {
  bool enabled = false;
  int64_t minimumComponentNodes = 0;
  int64_t maximumComponentNodes = 0;
  int64_t m = 0, n = 0, k = 0;
  int64_t elementBits = 0;
  std::string dotUseSignature;
  int64_t minimumRowStrideBytes = 0;
  int64_t maximumRowStrideBytes = 0;
  int64_t minimumGroupCount = 0;
  int64_t maximumGroupCount = 0;

  double entryBaseScale = 1.0;
  double entrySaturatedScale = 1.0;
  /// Large multi-wave schedules have different convergence for power-of-two
  /// and non-power-of-two row strides.  The aligned midpoint is the ordinary
  /// entryGroupMidpoint; this second midpoint owns the unaligned branch.
  double entryUnalignedGroupMidpoint = 0.0;
  double entryGroupMidpoint = 0.0;
  double entryGroupPower = 0.0;
  /// One-wave completion floor as a Hill curve of active scheduling slots.
  double singleWaveEntryBaseSystemCycles = 0.0;
  double singleWaveEntrySaturatedSystemCycles = 0.0;
  double singleWaveEntrySlotMidpoint = 0.0;
  double singleWaveEntrySlotPower = 0.0;
  /// Two-wave completion floor as a function of the last-wave active-slot
  /// fraction.  The Gaussian term owns the half-full-wave peak, the Hill term
  /// owns the high-tail reduction, and the final term owns a full second wave.
  double twoWaveEntryBaseSystemCycles = 0.0;
  double twoWaveEntryGaussianSystemCycles = 0.0;
  double twoWaveEntryTailCenter = 0.0;
  double twoWaveEntryTailWidth = 0.0;
  double twoWaveEntryHighTailPenaltySystemCycles = 0.0;
  double twoWaveEntryHighTailMidpoint = 0.0;
  double twoWaveEntryHighTailPower = 0.0;
  double twoWaveEntryFullTailSystemCycles = 0.0;
  double recurrenceSetupSystemCyclesPerGroup = 0.0;
  double dotRegionScale = 1.0;
  double singleWaveDotBaseSystemCycles = 0.0;
  double singleWaveDotSaturatedSystemCycles = 0.0;
  double singleWaveDotSlotMidpoint = 0.0;
  double singleWaveDotSlotPower = 0.0;
  double twoWaveDotTargetSystemCycles = 0.0;
  double exitBaseScale = 1.0;
  double exitSaturatedScale = 1.0;
  double exitUnalignedGroupMidpoint = 0.0;
  double exitGroupMidpoint = 0.0;
  double exitGroupPower = 0.0;
  double singleWaveExitBaseSystemCycles = 0.0;
  double singleWaveExitSaturatedSystemCycles = 0.0;
  double singleWaveExitSlotMidpoint = 0.0;
  double singleWaveExitSlotPower = 0.0;
  double twoWaveExitBaseSystemCycles = 0.0;
  double twoWaveExitGaussianSystemCycles = 0.0;
  double twoWaveExitTailCenter = 0.0;
  double twoWaveExitTailWidth = 0.0;
  double kernelBoundarySystemCycles = 0.0;
  std::string evidence;

  bool isValid() const;
};

llvm::StringRef stringifyStageMode(StageMode mode);

struct StageImplementation {
  StageMode mode = StageMode::SIMD;
  /// SIMD always uses factor=1.  For a whole-kernel SIMT implementation this
  /// is the AutoBlockify V1 factor.  A local SIMT implementation identifies a
  /// mixed-kernel candidate whose selected Stage is materialized as a scope.
  /// The local materializer batches the surrounding V1 loop in groups of F,
  /// widens the scope VF, and keeps SIMD/Cube peers on that same parent loop.
  int64_t superblockFactor = 1;
  bool localScope = false;
  /// AIVs sharing ONE logical program in a CV parent, not scheduling slots.
  /// Standalone SIMT uses one; local CV implementations may use two.
  int64_t aivSubblocks = 1;

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
  /// A source-level grid-stride loop has lower bound tt.get_program_id and
  /// step tt.get_num_programs.  Its body is replayed Q=ceil(P/C) times by the
  /// kernel itself; SuperBlock groups the C resident program ids and must not
  /// divide that source-loop trip count by F.
  bool isGridStrideLoopShell = false;
  bool insideGridStrideLoopBody = false;
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
  /// True when at least one address component is produced by a memory load.
  /// This records TTIR provenance only; address geometry below determines
  /// whether the access is elementwise-indirect or contains contiguous runs.
  bool addressDependsOnLoadedIndex = false;
  /// True when the tensor address DAG contains non-affine floating-point
  /// index computation (for example scale/floor/clamp before fptosi) and no
  /// affine lane-stride proof exists. Some SIMD lowerings serialize this
  /// pattern lane by lane; profiles may attach a completion term without
  /// relying on an operator name.
  bool addressHasNonAffineTensorComputation = false;
  /// True when a tensor address is selected by scalar control flow (currently
  /// an scf.if result). The SIMD backend may lower the resulting GM access to
  /// a scalar per-lane loop even though the selected alternatives are affine.
  bool addressHasTensorControlFlowSelection = false;
  /// True when this is a masked store whose value DAG contains a masked load
  /// guarded by the exact same predicate. The current SIMD backend lowers a
  /// bounded rank-two FP16 instance to load-source, load-destination, merge,
  /// store. This records only the TTIR structure; a profile still gates the
  /// measured completion domain.
  bool maskedStoreHasMatchingLoadDependency = false;
  /// Fraction of statically provable active elements in the memory mask.
  /// Runtime-only bounds remain one. This is source geometry, not a claim
  /// about predicated instruction issue or transferred GM bytes.
  double activeElementRatio = 1.0;
  /// TTIR provenance only, not an unconditional layout/ISA claim. A measured
  /// lowering contract must additionally match shape, dtype and launch shape.
  bool fromDotAccumulator = false;
  /// Nonempty only when every use is a direct dot input in the same block.
  /// A composite dot profile may then own this load instead of the GM model.
  std::vector<std::string> dotConsumerPaths;

  int64_t effectiveRank() const;
  int64_t contiguousAxisBytes() const;
  int64_t rowBytes() const;
  int64_t rowStrideBytes() const;
  int64_t rowCount() const;
  /// Number of elements in a contiguous run that can be proven even when
  /// another address dimension is runtime-indirect. Returns one when no
  /// multi-element run is provable.
  int64_t provenContiguousElements() const;
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

/// Route-independent semantic description of one dynamically executed TTIR
/// atomic operation. Physical transactions remain a mode/profile concern.
struct AtomicWorkload {
  std::string kind;
  std::string dataType;
  std::string memorySemantic;
  std::string memoryScope;
  double logicalElements = 0.0;
  double logicalOperationInstances = 0.0;
  double provenContiguousRunWidth = 0.0;
  bool resultUsed = false;
  bool addressDependsOnLoadedIndex = false;
  bool contentionUnknown = true;

  std::string profileKey() const;
  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

/// A group of tensor operations that share the same projected pointwise
/// lowering geometry. Broadcast boundaries retain distinct segments so short
/// rows cannot be merged into one full-width SIMD instruction.
struct TensorOperationWorkload {
  std::string operation;
  int64_t elementBitWidth = 0;
  double logicalElements = 0.0;
  double segmentCount = 0.0;
  int64_t contiguousElementsPerSegment = 0;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
};

/// Mode-independent work owned exactly once by one Stage.  Values are
/// logical elements/bytes, not mode-specific instructions or cycles.
struct StageWorkload {
  // Element count of one operation -> dynamic occurrences per Stage iteration.
  // Keep operation boundaries until mode-specific lane rounding is applied.
  llvm::StringMap<std::map<int64_t, double>> operationCounts;
  std::vector<TensorOperationWorkload> tensorOperationWorkloads;
  double scalarOperations = 0.0;
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  /// Number of logical load/store operations. Operation boundaries are kept
  /// because bytes alone cannot model short-transfer and pipeline behavior.
  double loadOperations = 0.0;
  double storeOperations = 0.0;
  double loadWarpInstructions = 0.0;
  double storeWarpInstructions = 0.0;
  /// Lowering-provided generated vector traffic.  These are instruction
  /// counts per Stage iteration, not semantic TTIR bytes.  They are optional:
  /// an absent count means that the lowering did not expose the stream.
  double sharedLocalLoadInstructions = 0.0;
  double sharedLocalStoreInstructions = 0.0;
  double privateStackLoadInstructions = 0.0;
  double privateStackStoreInstructions = 0.0;
  std::vector<StageMemoryAccess> memoryAccesses;
  std::vector<AtomicWorkload> atomicWorkloads;
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
  /// (bytes, logical elements, proven contiguous elements per segment) in one
  /// loaded-index load -> dynamic occurrences. These loads retain indirect
  /// outer addressing but must not consume a scalar element-gather curve.
  std::map<std::tuple<int64_t, int64_t, int64_t>, double>
      segmentedIndirectLoadCounts;
  /// Loaded-index stores retain the same distinction as loads: a scalar
  /// scatter has one destination segment per logical element, while an
  /// index_put-like row scatter has a proven contiguous inner segment.
  std::map<std::pair<int64_t, int64_t>, double> indirectStoreCounts;
  std::map<std::tuple<int64_t, int64_t, int64_t>, double>
      segmentedIndirectStoreCounts;
  std::map<int64_t, double> predicateCounts;
  /// (scan-axis extent, number of independent sequences, scan-axis element
  /// stride, result count, reverse, returned combine class) -> scan
  /// occurrences per Stage iteration. Prefix-scan dependence depth, sequence
  /// parallelism, physical axis stride, result arity and the lowered combine
  /// helper are different dimensions and cannot be reconstructed from their
  /// product in shuffleLaneSteps. The combine class is a structural TTIR
  /// property (add, mul, minmax or other), never an operator/kernel name.
  std::map<std::tuple<int64_t, int64_t, int64_t, int64_t, bool, std::string>,
           double>
      prefixScanCounts;
  /// Dynamic row-reduction operations per Stage iteration.  Keep this
  /// separate from lane-steps because the measured SIMD reduction envelope
  /// has a per-operation startup as well as an asymptotic lane-step rate.
  double reductionOperations = 0.0;
  double shuffleLaneSteps = 0.0;
  /// (independent rows, reduction-axis extent) -> reduction occurrences.
  /// Keep the geometry at the reduction op because a Stage boundary may
  /// separate the producer load from the reduction itself.
  std::map<std::tuple<int64_t, int64_t, int64_t>, double>
      reductionGeometryCounts;
  /// Associative operation returned by a tt.reduce combine region. Keep this
  /// beside reduction geometry because add and min/max lower to distinct SIMD
  /// completion templates even for identical rows and extents.
  std::map<std::string, double> reductionCombineClassCounts;
  /// Multi-result reductions that the current SIMD lowering serializes in
  /// scalar UB loops.  The key is (independent rows, reduction extent,
  /// result count, element bit width).  Keep this separate from the ordinary
  /// vector reduction surface: a pair-valued arg-reduction has a different
  /// lowering template even when its source geometry is identical.
  std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, double>
      scalarizedMultiResultReductionCounts;
  /// Native ascend.sort invocations keyed by (independent rows, sort-axis
  /// extent, element bit width).  The measured backend primitive is row
  /// serviced, so aggregate tensor elements alone are insufficient.
  std::map<std::tuple<int64_t, int64_t, int64_t>, double> sortCounts;
  /// True when a reduction consumes the result of a provably one-trip loop
  /// whose zero-initialized accumulator is updated only from zero-filled,
  /// prefix-masked loads.  In that case the runtime loop bound is an exact
  /// active reduction extent even when the accumulator tensor is physically
  /// wider.  This is TTIR structure, not an operator or kernel identity.
  std::map<std::pair<int64_t, int64_t>, double>
      oneTripMaskedAccumulatorReductionExtents;
  /// Dynamic dot matrix instances per Stage iteration (including batch).
  /// Startup is paid per instance; aggregate FLOPs alone cannot represent a
  /// chain of tiny dots.
  double dotOperations = 0.0;
  /// Per-matrix FLOP shape -> dynamic matrix instances.  Tiny-dot legality is
  /// a per matrix property; using aggregate `dotFlops` would incorrectly turn
  /// a chain of small dots into one large-dot Stage.
  std::map<int64_t, double> dotFlopCounts;
  /// TTIR operand provenance/layout key -> (FLOPs per dot, occurrences).
  /// Does not assert a physical shared-memory instruction count.
  std::map<std::string, std::pair<int64_t, double>> dotOperandPaths;
  double dotFlops = 0.0;
  double issueElements = 0.0;
  // Dynamic events normalized by iterationCount, like the other workload.
  double loopBackedges = 0.0;
  double conditionalBranches = 0.0;
  double divergentBranches = 0.0;
  double synchronizations = 0.0;
  /// Scalar GM operations owned by this Stage per iteration.
  double scalarLoadCount = 0.0;
  double scalarStoreCount = 0.0;
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
  /// True when same-layout rowwise input streams retain one resident address
  /// program: the primary load startup remains per iteration and additional
  /// stream startups are owned once by Stage setup.
  bool rowwiseAdditionalLoadSetupsOwnedOnce = false;
  /// Generated-stream attribution.  These cycles are already included in
  /// `load`/`store`; they are diagnostic decomposition, not extra terms.
  double sharedLocalLoad = 0.0;
  double sharedLocalStore = 0.0;
  double privateStackLoad = 0.0;
  double privateStackStore = 0.0;
  /// Residual needed to raise already attributed private-stack work to a
  /// measured combined load/store group floor.
  double privateStackCombinedFloor = 0.0;
  /// SIMD backend completion for a tensor load whose non-affine address DAG
  /// lowers to a scalar per-lane service loop. Already included in `load`.
  double computedAddressScalarizedLoad = 0.0;
  /// SIMD backend completion for a load whose tensor address is selected by
  /// scalar control flow. Already included in `load`.
  double controlFlowAddressScalarizedLoad = 0.0;
  /// Residual SIMD completion for the measured masked load/store group whose
  /// store lowering reads the destination before merging its mask. Already
  /// included in `store`.
  double maskedLoadStoreReadModifyWrite = 0.0;
  bool dotComponentPrivateStackFloorApplied = false;
  /// Shared-memory layout exchange, including its internal barriers. Stored
  /// per logical program; factor-specific measurements are divided by F here
  /// and multiplied once when forming the physical SuperBlock group.
  double sharedLayout = 0.0;
  double atomic = 0.0;
  /// Alternative atomic service for the one-physical-program SIMD regime.
  /// It is complete for all atomics in this Stage and replaces `atomic` only
  /// when the route has exactly one launched physical program.
  double singlePhysicalProgramAtomic = 0.0;
  bool hasSinglePhysicalProgramAtomicRate = false;
  double compute = 0.0;
  /// Serialized backend completion paid once per semantic tensor-operation
  /// instance. Unlike throughput work, this cannot disappear behind a
  /// load/store resource maximum.
  double operationCompletion = 0.0;
  double predicate = 0.0;
  /// Row-reduction execution envelope.  This remains separate from prefix
  /// scan shuffle work because SIMD reductions are measured as source-level
  /// reduction operations rather than one shuffle per tree edge.
  double reduction = 0.0;
  /// SIMD scalar-loop service selected for structurally matched multi-result
  /// reductions.  It replaces the optimistic vector-reduction envelope.
  double scalarizedMultiResultReduction = 0.0;
  /// Completion-exclusive service of the native backend row-sort primitive.
  double sort = 0.0;
  /// One-time SIMD completion setup for a structurally proven one-trip,
  /// zero-filled masked accumulator reduction. Included in `setup`; exposed
  /// separately so reports show its ownership.
  double maskedAccumulatorReductionCompletionSetup = 0.0;
  /// Completion-inclusive row-reduction body for one physical SIMT group.
  /// This replaces the decomposed memory/reduction estimate only for the
  /// bounded TTIR geometry described by the profile.
  double rowwiseReductionGroupBody = 0.0;
  /// Completion-inclusive body of a structurally matched whole-SIMT
  /// multi-result reduction lowering.  It replaces the optimistic decomposed
  /// reduction/shuffle estimate without taking ownership of adjacent Stages.
  double multiResultReductionGroupBody = 0.0;
  double shuffle = 0.0;
  /// Portion of shuffle attributed to tt.scan dependency levels.
  double scanShuffle = 0.0;
  double dot = 0.0;
  /// Diagnostic attribution only: already included in dot, not additive.
  double dotOperandGroup = 0.0;
  double dotInputLoadsCovered = 0.0;
  double loopControl = 0.0;
  double branchControl = 0.0;
  double divergence = 0.0;
  double synchronization = 0.0;
  double spill = 0.0;
  double issue = 0.0;
  /// SIMD-only completion bounds for rectangular tiles. The masked floor is a
  /// complete per-iteration competitor; the other two terms are diagnostic
  /// parts used to extend the vector service bound.
  double irregularMaskedTileFloor = 0.0;
  double irregularPhysicalMaterialization = 0.0;
  /// Full physical materialization before resident row-group ownership. This
  /// is diagnostic at Stage level and lets route composition restore the
  /// complete serialized cost after a hardware working-set boundary.
  double irregularFullPhysicalMaterialization = 0.0;
  double irregularTailMaterialization = 0.0;
  /// Prefix scan has two independent lower bounds: aggregate instruction
  /// issue across all sequences and the longest dependent scan chain.
  double prefixScanIssue = 0.0;
  double prefixScanCriticalPath = 0.0;
  /// Completion-inclusive body cost from an isolated dynamic grid-stride
  /// prefix-scan sweep. It replaces the decomposed scan/load/store estimate.
  double gridStridePrefixScanBody = 0.0;
  /// The selected dynamic source-loop resource curve was measured for this
  /// concrete SuperBlock factor.  Stage scoring forms an F-wide total; route
  /// composition divides it by F once while retaining the serial source Q.
  bool gridStrideFactorSpecificResourceBody = false;
  /// Candidate-specific measured lower bound for one physical SuperBlock
  /// group's recurrence body iteration.  Unlike the other per-iteration
  /// resources, this already includes the candidate's F logical programs.
  double recurrenceGroupCriticalPath = 0.0;
  /// One-time intercept of the matching physical-group recurrence sweep.
  /// Route composition pays it once for each executed physical group; the
  /// Stage trip count must not multiply it.
  double recurrenceGroupStartup = 0.0;
  /// Candidate-specific measured incremental memory cost for one physical
  /// group. These fields cover only operation shapes present in the profile;
  /// load/store retain any unclassified fallback work.
  double continuousLoadGroup = 0.0;
  double continuousStoreGroup = 0.0;
  /// Measured loaded-index envelope for one physical execution group. The
  /// source probes launch one program on every AIV, so route composition pays
  /// this once per runtime wave; SuperBlock must not multiply it by F again.
  /// Uncovered indirect traffic remains in `load` and scales per program.
  double indirectLoadGroup = 0.0;
  /// Kernel-wide intercept of the loaded-index group curve.  This does not
  /// belong to a Stage iteration: route composition pays the maximum matched
  /// envelope once across all indirect stages.
  double indirectLoadGroupStartup = 0.0;
  /// Startup selected from isolated dynamic grid-stride resource sweeps.
  /// Route composition pays the maximum matched resource envelope once.
  double gridStrideStartup = 0.0;
  /// Full-tile store service floor divided by F, like the analytical store
  /// resource. SuperBlock applies F once. Excludes first-round/sync costs.
  double storeLayoutService = 0.0;
  double criticalPath = 0.0;
  /// True when indirect memory had no dtype/shape-matched composite curve and
  /// used the mode's measured generic operation/byte or warp-instruction
  /// rates. This is a diagnostic evidence marker, not an added cost.
  bool indirectMemoryAnalyticalFallback = false;

  bool isFiniteAndNonNegative() const;
  llvm::json::Object toJSON() const;
  /// Preserve every non-zero resource while omitting default-valued
  /// diagnostic fields from the online compiler report.
  llvm::json::Object toCompactJSON() const;
};

struct StageImplementationCost {
  StageImplementation implementation;
  double totalCycles = 0.0;
  /// A profiled semantic operation in this implementation already includes
  /// the fixed local-SIMT scope enter/exit completion envelope.
  bool localScopeCompletionInclusive = false;
  /// Diagnostics for a physical CV1:N payload partition.  The ideal value is
  /// the analytical perfectly balanced result.  The correction interpolates
  /// toward the unpartitioned result using a target-wide efficiency measured
  /// by structural CV probes.
  double cvPartitionIdealTotalCycles = 0.0;
  double cvPartitionPayloadEfficiency = 1.0;
  double cvPartitionInefficiencyCorrectionCycles = 0.0;
  StageResourceCycles resources;
  /// Describes the strongest formula evidence used for this candidate.  It
  /// is report provenance, not a route multiplier.
  std::string formulaEvidence;
  /// Per-AIV work before SuperBlock replication. LogicalStageCost::workload
  /// remains the original, mode-independent logical-program workload.
  std::optional<StageWorkload> perAivWorkload;

  bool isValid() const;
  llvm::json::Object toJSON() const;
  llvm::json::Object toCompactJSON() const;
};

/// A dot's operands and the distinct values retained across that operation.
struct SimtDotPressureSite {
  int64_t m = 0, n = 0, k = 0;
  int64_t elementBits = 0;
  int64_t operandUnitsPerThread = 0;
  int64_t retainedTensorUnitsPerThread = 0;
  int64_t retainedScalarUnitsPerThread = 0;

  llvm::json::Object toJSON() const;
};

/// One source-ordered TTIR dot node. Dot ordinals preserve relative dot order;
/// operationOrdinal is the full module walk position. Predecessors are the
/// nearest upstream dots through pure SSA operations. The downstream context
/// stops at another dot or a side-effecting operation, so stores and unknown
/// effects are observations, not transparent data dependencies.
struct SimtDotDependencySite {
  int64_t ordinal = 0;
  int64_t operationOrdinal = 0;
  std::vector<int64_t> predecessorOrdinals;
  int64_t depth = 0;
  int64_t successorCount = 0;
  int64_t nonDotPureConsumerCount = 0;
  int64_t sideEffectFrontierCount = 0;
  int64_t maxNonDotPureUseSpan = 0;
  int64_t maxSideEffectUseSpan = 0;

  llvm::json::Object toJSON() const;
};

/// Simultaneous TTIR values, in 32-bit units per thread of one logical
/// program. Balanced element distribution is an explicit approximation:
/// register classes, layout replication, rematerialization and backend
/// temporaries are not resolved by this analysis. These are NOT spill counts.
enum class SimtPrivateStackChargeRegion {
  None,
  WholeFunction,
  BeforeRecurrence,
  DotRegion,
  AfterDot,
};

struct SimtRegisterPressure {
  int64_t baseWarpCount = 0;
  int64_t peakLiveUnitsPerThread = 0;
  int64_t peakTensorUnitsPerThread = 0;
  int64_t peakScalarUnitsPerThread = 0;
  int64_t peakCrossStageUnitsPerThread = 0;
  int64_t peakLoopCarriedUnitsPerThread = 0;
  int64_t peakRematerializableUnitsPerThread = 0;
  /// Predicate structure remains separate from 32-bit GPR pressure.  CCEC
  /// allocates P and R banks separately, and one predicate with multiple
  /// consumers can produce more reloads than saves.
  int64_t peakLivePredicateUnitsPerThread = 0;
  int64_t predicateDefinitionUnitsPerThread = 0;
  int64_t predicateUseUnitsPerThread = 0;
  int64_t maxPredicateFanout = 0;
  /// TTIR predicate topology, kept separate from machine P-register claims.
  /// Integer comparisons are split by semantic class. Predicate reshape,
  /// expand, broadcast, and splat results have their own source class;
  /// generated masks and floating predicates remain in `other`. DAG edges
  /// are weighted by the result's per-thread predicate units.
  int64_t ttirIntegerEqualityPredicateDefinitionUnitsPerThread = 0;
  int64_t ttirIntegerOrderedPredicateDefinitionUnitsPerThread = 0;
  int64_t ttirPredicateShapeDefinitionUnitsPerThread = 0;
  int64_t ttirOtherPredicateDefinitionUnitsPerThread = 0;
  int64_t ttirPredicateDagEdgeUnitsPerThread = 0;
  int64_t ttirMaxPredicateDagDepth = 0;
  int64_t ttirSelectPredicateUseUnitsPerThread = 0;
  int64_t ttirBooleanPredicateUseUnitsPerThread = 0;
  /// Predicate values defined outside an scf.for and consumed inside it.
  /// These are source facts for save-once/reload-per-iteration calibration;
  /// they are not themselves spill counts.
  int64_t loopInvariantPredicateUnitsPerThread = 0;
  int64_t loopInvariantPredicateUseUnitsPerThread = 0;
  /// Separate address features for calibrating lowering expansion. A TTIR
  /// block pointer is not proof of one physical 64-bit address register.
  /// Each peak is independent; these fields must not be added to peakLive.
  int64_t peakPointerUnitsPerThread = 0;
  int64_t peakBlockPointerCount = 0;
  int64_t peakBlockPointerRankSum = 0;
  int64_t unknownValueCount = 0;
  /// Optional backend evidence.  TTIR analysis cannot derive these exactly;
  /// a lowering may attach the measured peak register units and effective
  /// unroll factor to the owning operation for calibration.
  int64_t loweredPeakLiveUnitsPerThread = 0;
  int64_t backendUnrollFactor = 1;
  /// TTIR dot dependency topology.  Edges connect a dot result to a later dot
  /// through any number of intervening pure SSA operations.  These fields are
  /// source observations only; they do not claim a backend schedule or spill
  /// bundle.
  int64_t ttirDotDefinitionCount = 0;
  int64_t ttirDotDependencyEdgeCount = 0;
  int64_t ttirDotDependencySourceCount = 0;
  int64_t ttirDotDependencySinkCount = 0;
  int64_t ttirMaxDotDependencyDepth = 0;
  /// Function-wide weak connectivity of the dot DAG. Geometry is reported
  /// only when every maximum-size component has the same known dot geometry.
  /// These are source facts; they do not imply that a spill was generated.
  int64_t ttirMaximumWeakDotComponentNodes = 0;
  bool ttirMaximumWeakDotComponentGeometryKnown = false;
  int64_t ttirMaximumWeakDotComponentM = 0;
  int64_t ttirMaximumWeakDotComponentN = 0;
  int64_t ttirMaximumWeakDotComponentK = 0;
  int64_t ttirMaximumWeakDotComponentElementBits = 0;
  /// Canonical function-wide dot topology plus observable use context. This
  /// excludes operation names and source locations. It keeps a calibrated
  /// private-stack envelope from leaking into a different dot/use schedule
  /// whose aggregate component size happens to match.
  std::string ttirFunctionDotUseSignature;
  /// A whole-VF stack group is either owned once, or split across three
  /// structurally identified straight-line regions around a serial recurrence
  /// and its following dot DAG. It is never charged once per dot Stage.
  SimtPrivateStackChargeRegion dotComponentPrivateStackChargeRegion =
      SimtPrivateStackChargeRegion::None;
  std::vector<SimtDotDependencySite> dotDependencySites;
  /// Simultaneous context at each dot, not a sum of unrelated peak values.
  /// Retained values exclude dot operands/results and rematerializable values.
  /// Offline calibration labels may describe backend unrolling; no actual
  /// CCEC unroll factor or physical-register count is claimed at TTIR level.
  std::vector<SimtDotPressureSite> dotSites;

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
  std::optional<SimtRegisterPressure> wholeSimtRegisterPressure;
  int64_t simtUbWorkingSetBytesPerProgram = 0;
  int64_t simtUsableUbBytes = 0;
  std::vector<int64_t> simtFactorsRejectedByUb;
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
  /// Analytical local-SIMT body estimates, including Stages with no scope
  /// anchor. They share the executable implementations' workload/formulas but
  /// are NOT route candidates. Scope boundary traffic and handoff are excluded.
  std::vector<StageImplementationCost> simtCalibrationEstimates;
  std::vector<StageImplementationCost> implementations;

  llvm::json::Object toJSON(bool includeDiagnosticImplementations = true) const;
};

struct StageCostTable {
  bool operationOwnershipComplete = false;
  int64_t modeledOperationCount = 0;
  std::string profileVersion;
  int64_t logicalProgramCountHint = 0;
  /// Launch count before backend row coalescing.  Route scheduling uses the
  /// transformed logicalProgramCountHint; structurally matched completion
  /// formulas may use this source count for per-program work retained inside
  /// a coalesced physical program.
  int64_t sourceLogicalProgramCountHint = 0;
  /// AIV/Vector-core count used by pure-AIV routes.  A Mixed CV executable
  /// owns its V1 logical-program loop in the unsplit MIX/AIC parent and uses
  /// physicalAiCoreCountHint instead; this value remains the execution width
  /// of the local AIV sub-blocks.
  int64_t physicalCoreCountHint = 0;
  /// AIC core count used by Cube-bearing routes.  In particular, a Mixed CV
  /// route uses this count for the shared parent logical-program loop; the
  /// AIV peer receives the same loop bounds after SplitMixKernel.
  int64_t physicalAiCoreCountHint = 0;
  int64_t cvAivSubblocks = 1;
  int64_t baseSimtWarpCount = 1;
  /// Per-transfer MTE2 program establishment cost from the SIMD hardware
  /// profile.  Route composition uses this only to transfer ownership when
  /// several adjacent, pure direct-load Stages lower into one straight-line
  /// MTE program; byte service remains owned by each Stage.
  double simdLoadOperationStartupCycles = 0.0;
  /// One-time kernel establishment cost for a group of independent SIMD load
  /// streams. Route composition also uses this for a short multi-memory
  /// kernel whose loads/stores remain separate serialized Stages.
  double simdIndependentLoadGroupKernelStartupCycles = 0.0;
  /// Completion-inclusive backend-helper rate for a straight-line SIMT scan.
  /// Route composition uses its presence to transfer a pure generic
  /// index/address prologue to the same helper once at P=1.
  double simtStraightLinePrefixScanElementsPerCycle = 0.0;
  std::string simtStraightLinePrefixScanCompletionEvidence;
  std::vector<WholeSimdCompletionCurve> wholeSimdCompletionCurves;
  std::vector<WholeSimtCompletionCurve> wholeSimtCompletionCurves;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simdGridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simdGridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry;
  std::string simdGridStrideIndirectDotCompositionEvidence;
  int64_t simdIrregularTileLargeWorkingSetFullPhysicalMinimumBytes = 0;
  std::string simdIrregularTileCompletionEvidence;
  int64_t simtGridStrideResidentMemoryMaximumBytes = 0;
  double simtGridStrideResidentLoadWarpInstructionsPerCycle = 0.0;
  double simtGridStrideResidentStoreWarpInstructionsPerCycle = 0.0;
  std::string simtGridStrideResidentMemoryEvidence;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simtGridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simtGridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry;
  std::string simtGridStrideIndirectDotCompositionEvidence;
  bool simtTwoPassRowwiseOwnsSegmentedIndirectLoads = false;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simtTwoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot;
  std::string simtTwoPassRowwiseCompletionEvidence;
  std::map<int64_t, std::map<int64_t, std::map<int64_t, double>>>
      simdTwoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot;
  std::string simdTwoPassRowwiseCompletionEvidence;
  bool simdSingleResultRowReductionRecurrenceCompletionEnabled = false;
  int64_t simdSingleResultRowReductionRecurrenceMinimumExtent = 0;
  int64_t simdSingleResultRowReductionRecurrenceMaximumExtent = 0;
  int64_t simdSingleResultRowReductionRecurrenceMinimumIterations = 0;
  int64_t simdSingleResultRowReductionRecurrenceMaximumIterations = 0;
  int64_t simdSingleResultRowReductionRecurrenceMinimumPrograms = 0;
  int64_t simdSingleResultRowReductionRecurrenceMaximumPrograms = 0;
  double simdSingleResultRowReductionRecurrenceIterationCycles = 0.0;
  double simdSingleResultRowReductionRecurrenceStartupInterceptCycles = 0.0;
  double simdSingleResultRowReductionRecurrenceStartupSqrtProgramCycles = 0.0;
  std::string simdSingleResultRowReductionRecurrenceEvidence;
  bool simtSingleResultRowReductionRecurrenceCompletionEnabled = false;
  int64_t simtSingleResultRowReductionRecurrenceMinimumExtent = 0;
  int64_t simtSingleResultRowReductionRecurrenceMaximumExtent = 0;
  int64_t simtSingleResultRowReductionRecurrenceMinimumIterations = 0;
  int64_t simtSingleResultRowReductionRecurrenceMaximumIterations = 0;
  int64_t simtSingleResultRowReductionRecurrenceMinimumPrograms = 0;
  int64_t simtSingleResultRowReductionRecurrenceMaximumPrograms = 0;
  double simtSingleResultRowReductionRecurrenceIterationCycles = 0.0;
  double simtSingleResultRowReductionRecurrenceStartupInterceptCycles = 0.0;
  double simtSingleResultRowReductionRecurrenceStartupSqrtProgramCycles = 0.0;
  std::string simtSingleResultRowReductionRecurrenceEvidence;
  /// Completion-inclusive elapsed formula for a bounded multi-result prefix
  /// scan lowering. The route score subtracts the generic fixed kernel
  /// boundary supplied with the same profile, so elapsed prediction adds that
  /// boundary exactly once.
  bool simdMultiResultPrefixScanCompletionEnabled = false;
  int64_t simdMultiResultPrefixScanMinimumPrograms = 0;
  int64_t simdMultiResultPrefixScanMaximumPrograms = 0;
  int64_t simdMultiResultPrefixScanMinimumPaddedAxis = 0;
  int64_t simdMultiResultPrefixScanMaximumPaddedAxis = 0;
  int64_t simdMultiResultPrefixScanRequiredResultCount = 0;
  int64_t simdMultiResultPrefixScanRequiredIndependentSequences = 0;
  double simdMultiResultPrefixScanMinimumActiveRatio = 0.0;
  double simdMultiResultPrefixScanMaximumActiveRatio = 0.0;
  double simdMultiResultPrefixScanElapsedInterceptCycles = 0.0;
  double simdMultiResultPrefixScanElapsedSqrtProgramCycles = 0.0;
  double simdMultiResultPrefixScanElapsedInactiveRatioCycles = 0.0;
  double simdMultiResultPrefixScanElapsedPaddedAxisCycles = 0.0;
  double simdMultiResultPrefixScanFixedBoundaryCycles = 0.0;
  std::string simdMultiResultPrefixScanEvidence;
  /// Completion-inclusive route-resource formula for a straight-line SIMT
  /// pairwise selection network. The matcher uses the rank-2 input/output
  /// memory geometry and the repeated extent-2 reduction topology.
  bool simtPairwiseSelectionNetworkCompletionEnabled = false;
  int64_t simtPairwiseSelectionNetworkMinimumInputElements = 0;
  int64_t simtPairwiseSelectionNetworkMaximumInputElements = 0;
  int64_t simtPairwiseSelectionNetworkMinimumOutputColumns = 0;
  int64_t simtPairwiseSelectionNetworkMaximumOutputColumns = 0;
  int64_t simtPairwiseSelectionNetworkMinimumReductionStages = 0;
  double simtPairwiseSelectionNetworkInterceptCycles = 0.0;
  double simtPairwiseSelectionNetworkInputElementCycles = 0.0;
  double simtPairwiseSelectionNetworkLog2OutputColumnCycles = 0.0;
  std::string simtPairwiseSelectionNetworkEvidence;
  /// The SIMD min/max scan helper has an additional completion tail for
  /// large sets of independent sequences.  This is a bounded route residual;
  /// the Stage scan and memory resources remain independently auditable.
  bool simdLargeSequenceMinmaxScanCompletionEnabled = false;
  int64_t simdLargeSequenceMinmaxScanMinimumIndependentSequences = 0;
  int64_t simdLargeSequenceMinmaxScanMaximumIndependentSequences = 0;
  int64_t simdLargeSequenceMinmaxScanMinimumExtent = 0;
  int64_t simdLargeSequenceMinmaxScanMaximumExtent = 0;
  double simdLargeSequenceMinmaxScanPerIndependentSequenceCycles = 0.0;
  std::string simdLargeSequenceMinmaxScanEvidence;
  /// The straight-line SIMT prefix-scan helper shares issue/completion work
  /// across independent min/max sequences. The analytical Stage sum prices
  /// every sequence independently, so subtract the bounded shared envelope.
  bool simtMultiSequenceMinmaxScanCompletionEnabled = false;
  int64_t simtMultiSequenceMinmaxScanMinimumIndependentSequences = 0;
  int64_t simtMultiSequenceMinmaxScanMaximumIndependentSequences = 0;
  int64_t simtMultiSequenceMinmaxScanMinimumExtent = 0;
  int64_t simtMultiSequenceMinmaxScanMaximumExtent = 0;
  double simtMultiSequenceMinmaxScanBaseReductionCycles = 0.0;
  double simtMultiSequenceMinmaxScanPerIndependentSequenceReductionCycles = 0.0;
  double simtMultiSequenceMinmaxScanMaximumReductionCycles = 0.0;
  int64_t simtMultiSequenceMinmaxScanLargeMinimumIndependentSequences = 0;
  int64_t simtMultiSequenceMinmaxScanLargeMaximumIndependentSequences = 0;
  int64_t simtMultiSequenceMinmaxScanLargeMinimumExtent = 0;
  int64_t simtMultiSequenceMinmaxScanLargeMaximumExtent = 0;
  double simtMultiSequenceMinmaxScanLargeBaseReductionCycles = 0.0;
  double simtMultiSequenceMinmaxScanLargePerIndependentSequenceReductionCycles =
      0.0;
  double simtMultiSequenceMinmaxScanLargeMaximumReductionCycles = 0.0;
  std::string simtMultiSequenceMinmaxScanEvidence;
  bool simdStraightLineAddReductionCompletionEnabled = false;
  int64_t simdStraightLineAddReductionMinimumRows = 0;
  int64_t simdStraightLineAddReductionMaximumRows = 0;
  int64_t simdStraightLineAddReductionMinimumExtent = 0;
  int64_t simdStraightLineAddReductionMaximumExtent = 0;
  double simdStraightLineAddReductionSharedCompletionCycles = 0.0;
  std::string simdStraightLineAddReductionEvidence;
  bool simdTwoLoopSingleReductionCompletionEnabled = false;
  int64_t simdTwoLoopSingleReductionMinimumSourcePrograms = 0;
  int64_t simdTwoLoopSingleReductionMaximumSourcePrograms = 0;
  int64_t simdTwoLoopSingleReductionMinimumRuntimeIterations = 0;
  int64_t simdTwoLoopSingleReductionMaximumRuntimeIterations = 0;
  int64_t simdTwoLoopSingleReductionRequiredExtent = 0;
  double simdTwoLoopSingleReductionFirstIterationCycles = 0.0;
  double simdTwoLoopSingleReductionAdditionalIterationCycles = 0.0;
  std::string simdTwoLoopSingleReductionEvidence;
  bool mixedParentMemoryCompletionEnabled = false;
  int64_t mixedParentMemoryMinimumIterations = 0;
  int64_t mixedParentMemoryReferenceBytesPerProgram = 0;
  double mixedParentMemoryCyclesPerExcessIteration = 0.0;
  std::string mixedParentMemoryCompletionEvidence;
  std::vector<LogicalStageCost> stages;
};

struct ScopeHandoffCost {
  /// Completion-inclusive cost of entering and leaving one empty local SIMT
  /// scope.  The two directions are deliberately not exposed as independent
  /// parameters: adjacent SIMD/SIMT pipelines overlap, so directional costs
  /// cannot be recovered by subtracting standalone timings.
  double fixedScopeCycles = 0.0;
  /// CV lowering inserts AIC/AIV coordination around a local SIMT scope in
  /// addition to the ordinary AIV scope enter/leave envelope.  This term is
  /// paid only when the selected implementation is physically split across
  /// multiple AIV subblocks.
  double cvCrossCoreSynchronizationCycles = 0.0;
  /// Fraction of the ideal CV payload speedup that is realized.  One keeps
  /// the analytical 1:N split; zero conservatively retains the unpartitioned
  /// payload cost.  This is calibrated across structural probes, never from
  /// an operator or exact tensor shape.
  double cvPartitionPayloadEfficiency = 1.0;
  /// Incremental aggregate boundary-copy rates measured with the exact scope
  /// lowering.  They include both register-file sides of the UB handoff and
  /// therefore are not per-thread or raw UB bandwidth claims.
  double inputHandoffBytesPerCycle = 1.0;
  double outputHandoffBytesPerCycle = 1.0;

  bool isValid() const;
  double fixedCyclesForAivSubblocks(int64_t aivSubblocks) const;
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
  /// A backend-lowerability gate may make a route unselectable even though
  /// every Stage has a costable implementation.  Preserve that analytical
  /// cost for forced-route diagnostics without admitting it to Auto choice.
  bool diagnosticCostAvailable = false;
  /// Structural backend-lowerability reason when this candidate is rejected
  /// before cost comparison. Empty for ordinary missing-implementation cases.
  std::string ineligibilityReason;
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
  bool singlePhysicalProgramAtomicRateApplied = false;
  double singlePhysicalProgramAtomicAdjustmentCycles = 0.0;
  bool wholeSimdCompletionApplied = false;
  int64_t wholeSimdCompletionRowStrideBytes = 0;
  int64_t wholeSimdCompletionGroupCount = 0;
  bool wholeSimdRowStridePowerOfTwo = false;
  double wholeSimdLastWaveActiveSlotFraction = 1.0;
  double wholeSimdEntryScale = 1.0;
  double wholeSimdEntryCompletionAddedCycles = 0.0;
  double wholeSimdEntryCompletionTargetCycles = 0.0;
  double wholeSimdRecurrenceSetupAddedCycles = 0.0;
  double wholeSimdDotRegionScale = 1.0;
  double wholeSimdDotCompletionAddedCycles = 0.0;
  double wholeSimdDotCompletionTargetCycles = 0.0;
  double wholeSimdExitScale = 1.0;
  double wholeSimdExitCompletionAddedCycles = 0.0;
  double wholeSimdExitCompletionTargetCycles = 0.0;
  double wholeSimdKernelBoundaryCycles = 0.0;
  std::string wholeSimdCompletionEvidence;
  bool wholeSimtCompletionApplied = false;
  int64_t wholeSimtCompletionRowStrideBytes = 0;
  int64_t wholeSimtCompletionGroupCount = 0;
  double wholeSimtEntryCompletionAddedCycles = 0.0;
  double wholeSimtExitCompletionAddedCycles = 0.0;
  double wholeSimtExitCompletionTargetCycles = 0.0;
  double wholeSimtPreRecurrenceScalarOperations = 0.0;
  double wholeSimtLastWaveActiveSlotFraction = 1.0;
  double wholeSimtRecurrenceScale = 1.0;
  double wholeSimtDotRegionScale = 1.0;
  double wholeSimtKernelBoundaryCycles = 0.0;
  std::string wholeSimtCompletionEvidence;
  bool mixedParentMemoryCompletionApplied = false;
  int64_t mixedParentMemoryParentIterations = 0;
  double mixedParentMemoryBytesPerProgram = 0.0;
  double mixedParentMemoryScale = 0.0;
  double mixedParentMemoryAddedCycles = 0.0;
  std::string mixedParentMemoryCompletionEvidence;
  double indirectLoadGroupStartupCycles = 0.0;
  double gridStrideStartupCycles = 0.0;
  bool gridStrideFirstIndirectServiceOwnedByBoundary = false;
  double gridStrideFirstIndirectServiceAdjustmentCycles = 0.0;
  std::string gridStrideFirstIndirectServiceEvidence;
  bool gridStrideFactorOneBodyApplied = false;
  /// A factor-specific scan curve describes the scan engine body, while GM
  /// transfers owned by the same merged Stage remain one source-iteration of
  /// work. Record the replacement of factor-F transfer service with F1
  /// transfer service so the route remains invariant to Stage fusion.
  bool gridStrideFactorOneDirectMemoryApplied = false;
  double gridStrideFactorOneDirectMemoryAdjustmentCycles = 0.0;
  bool gridStrideIndirectDotCompositionResidualApplied = false;
  int64_t gridStrideCompositionIndirectWarpInstructions = 0;
  int64_t gridStrideCompositionDotFlops = 0;
  double gridStrideCompositionBodyResidualCycles = 0.0;
  double gridStrideCompositionStartupResidualCycles = 0.0;
  double gridStrideCompositionTotalResidualCycles = 0.0;
  std::string gridStrideCompositionEvidence;
  /// A completion-inclusive SIMD scan curve owns the direct load/store
  /// stages in the same source grid-stride body.  These fields record the
  /// route-level ownership transfer so route.json exposes the removed
  /// duplicate cost.
  bool gridStrideSimdScanMemoryOwnershipApplied = false;
  double gridStrideSimdScanMemoryOwnershipCycles = 0.0;
  /// Adjacent straight-line SIMD load Stages can be artifacts of TTIR Stage
  /// partitioning even though lowering establishes one MTE2 program for the
  /// uninterrupted load run.  Record the removed duplicate startup only;
  /// all operation bytes remain serialized and fully charged.
  bool straightLineSimdLoadStartupOwnershipApplied = false;
  int64_t straightLineSimdLoadStartupOwnedOperationCount = 0;
  double straightLineSimdLoadStartupAdjustmentCycles = 0.0;
  std::string straightLineSimdLoadStartupEvidence;
  bool shortSimdMultiMemoryStartupApplied = false;
  int64_t shortSimdMultiMemoryLoadStageCount = 0;
  int64_t shortSimdMultiMemoryStoreStageCount = 0;
  double shortSimdMultiMemoryStartupCycles = 0.0;
  std::string shortSimdMultiMemoryStartupEvidence;
  /// The complete independent-load group startup was measured at a full
  /// vector-core launch. At sub-grid occupancy the generic kernel-boundary
  /// envelope owns that one-time interval only when the group covers every
  /// load. A grouped prefix followed by residual loads retains its distinct
  /// MTE-program startup.
  bool lowOccupancySimdLoadGroupStartupOwnershipApplied = false;
  double lowOccupancySimdLoadGroupStartupAdjustmentCycles = 0.0;
  std::string lowOccupancySimdLoadGroupStartupEvidence;
  bool straightLineSimtScanIndexOwnershipApplied = false;
  double straightLineSimtScanIndexOwnershipCycles = 0.0;
  std::string straightLineSimtScanCompletionEvidence;
  bool gridStrideResidentMemoryApplied = false;
  double gridStrideResidentMemoryWorkingSetBytes = 0.0;
  double gridStrideResidentMemoryAdjustmentCycles = 0.0;
  std::string gridStrideResidentMemoryEvidence;
  bool largeSimdIrregularTileWorkingSetApplied = false;
  double largeSimdIrregularTileWorkingSetBytes = 0.0;
  double largeSimdIrregularTileAdjustmentCycles = 0.0;
  std::string largeSimdIrregularTileEvidence;
  bool twoPassRowwiseCompletionApplied = false;
  int64_t twoPassRowwiseContiguousExtent = 0;
  double twoPassRowwiseSegmentedLoadOwnershipCycles = 0.0;
  double twoPassRowwiseResidualCycles = 0.0;
  std::string twoPassRowwiseCompletionEvidence;
  bool simdSingleResultRowReductionRecurrenceCompletionApplied = false;
  int64_t simdSingleResultRowReductionRecurrenceExtent = 0;
  int64_t simdSingleResultRowReductionRecurrenceIterations = 0;
  int64_t simdSingleResultRowReductionRecurrencePrograms = 0;
  double simdSingleResultRowReductionRecurrenceTargetCycles = 0.0;
  double simdSingleResultRowReductionRecurrenceAdjustmentCycles = 0.0;
  std::string simdSingleResultRowReductionRecurrenceEvidence;
  bool simtSingleResultRowReductionRecurrenceCompletionApplied = false;
  int64_t simtSingleResultRowReductionRecurrenceExtent = 0;
  int64_t simtSingleResultRowReductionRecurrenceIterations = 0;
  int64_t simtSingleResultRowReductionRecurrencePrograms = 0;
  double simtSingleResultRowReductionRecurrenceTargetCycles = 0.0;
  double simtSingleResultRowReductionRecurrenceAdjustmentCycles = 0.0;
  std::string simtSingleResultRowReductionRecurrenceEvidence;
  bool simdMultiResultPrefixScanCompletionApplied = false;
  int64_t simdMultiResultPrefixScanSourcePrograms = 0;
  int64_t simdMultiResultPrefixScanPaddedAxis = 0;
  int64_t simdMultiResultPrefixScanActiveAxis = 0;
  int64_t simdMultiResultPrefixScanResultCount = 0;
  int64_t simdMultiResultPrefixScanIndependentSequences = 0;
  double simdMultiResultPrefixScanActiveRatio = 0.0;
  double simdMultiResultPrefixScanAbsoluteTargetCycles = 0.0;
  double simdMultiResultPrefixScanRouteTargetCycles = 0.0;
  double simdMultiResultPrefixScanAdjustmentCycles = 0.0;
  std::string simdMultiResultPrefixScanEvidence;
  bool simtPairwiseSelectionNetworkCompletionApplied = false;
  int64_t simtPairwiseSelectionNetworkInputRows = 0;
  int64_t simtPairwiseSelectionNetworkInputColumns = 0;
  int64_t simtPairwiseSelectionNetworkOutputColumns = 0;
  int64_t simtPairwiseSelectionNetworkReductionStages = 0;
  double simtPairwiseSelectionNetworkTargetCycles = 0.0;
  double simtPairwiseSelectionNetworkAdjustmentCycles = 0.0;
  std::string simtPairwiseSelectionNetworkEvidence;
  bool simdLargeSequenceMinmaxScanCompletionApplied = false;
  int64_t simdLargeSequenceMinmaxScanIndependentSequences = 0;
  int64_t simdLargeSequenceMinmaxScanExtent = 0;
  double simdLargeSequenceMinmaxScanAdjustmentCycles = 0.0;
  std::string simdLargeSequenceMinmaxScanEvidence;
  bool simdRank2StridedDirectLoadCompletionApplied = false;
  int64_t simdRank2StridedDirectLoadElements = 0;
  double simdRank2StridedDirectLoadAdjustmentCycles = 0.0;
  std::string simdRank2StridedDirectLoadEvidence;
  bool simtRank2StridedDirectLoadCompletionApplied = false;
  int64_t simtRank2StridedDirectLoadElements = 0;
  double simtRank2StridedDirectLoadTargetCycles = 0.0;
  double simtRank2StridedDirectLoadAdjustmentCycles = 0.0;
  std::string simtRank2StridedDirectLoadEvidence;
  bool rank2InterleavedPrefixScanCompletionApplied = false;
  int64_t rank2InterleavedPrefixScanExtent = 0;
  int64_t rank2InterleavedPrefixScanIndependentSequences = 0;
  double rank2InterleavedPrefixScanTargetCycles = 0.0;
  double rank2InterleavedPrefixScanAdjustmentCycles = 0.0;
  std::string rank2InterleavedPrefixScanEvidence;
  bool simtStaticLoopDirectMemoryCompletionApplied = false;
  int64_t simtStaticLoopDirectMemoryIterations = 0;
  int64_t simtStaticLoopDirectMemoryLoadCount = 0;
  double simtStaticLoopDirectMemoryElapsedTargetCycles = 0.0;
  double simtStaticLoopDirectMemoryAdjustmentCycles = 0.0;
  std::string simtStaticLoopDirectMemoryEvidence;
  bool simtMultiSequenceMinmaxScanCompletionApplied = false;
  int64_t simtMultiSequenceMinmaxScanIndependentSequences = 0;
  int64_t simtMultiSequenceMinmaxScanExtent = 0;
  double simtMultiSequenceMinmaxScanReductionCycles = 0.0;
  double simtMultiSequenceMinmaxScanAdjustmentCycles = 0.0;
  std::string simtMultiSequenceMinmaxScanEvidence;
  bool simdStraightLineAddReductionCompletionApplied = false;
  int64_t simdStraightLineAddReductionRows = 0;
  int64_t simdStraightLineAddReductionExtent = 0;
  double simdStraightLineAddReductionAdjustmentCycles = 0.0;
  std::string simdStraightLineAddReductionEvidence;
  bool simdTwoLoopSingleReductionCompletionApplied = false;
  int64_t simdTwoLoopSingleReductionRuntimeIterations = 0;
  double simdTwoLoopSingleReductionAdjustmentCycles = 0.0;
  std::string simdTwoLoopSingleReductionEvidence;
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
  /// Auditable pure-SIMT plans for each legal whole-kernel factor. Route
  /// selection still uses `allSimt`; these plans expose the alternatives
  /// needed to compare F1/F2/F4 costs without recompiling the analysis.
  std::map<int64_t, StageRoutePlan> allSimtFactorPlans;
  StageRoutePlan mixed;
  /// Fixed parent context for comparing individual SIMT Stage bodies. This
  /// does not describe an all-SIMT executable (which may have a different Q).
  int64_t calibrationLogicalPrograms = 0;
  int64_t calibrationSchedulingSlots = 0;
  int64_t calibrationAivPartitions = 1;

  llvm::json::Object toJSON(bool includeDiagnosticImplementations = true) const;
};

llvm::Expected<StageCostModelSummary>
solveStageRoutes(const StageCostTable &costTable,
                 const ScopeHandoffCost &scopeHandoff);

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_STAGEROUTECOSTMODEL_H
