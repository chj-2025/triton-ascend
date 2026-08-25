//===- StagePartitioner.h - Build semantic Phase/Stage IR ----*- C++ -*-===//

#ifndef ASCENDMODEL_ROUTEMODEL_STAGEPARTITIONER_H
#define ASCENDMODEL_ROUTEMODEL_STAGEPARTITIONER_H

#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/RouteModel/StageCostModels.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::ascend {

struct StagePartitionerOptions {
  int64_t tinyDotFlopsMax = 16384;
  int64_t maximumSuperblockFactor = 1;
  bool scopeSuperblockMaterializable = false;
};

/// Result of PhaseBoundaryAnalysis.  This is structural boundary evidence,
/// not a route or a cost-model decision.  Stage boundaries are derived purely
/// from operation semantics, independent of any business-scenario domain.
struct PhaseBoundaryPlan {
  std::string domainName = "semantic";
  /// Top-level semantic TTIR operations in execution order.  Nested region
  /// operations are owned transitively by their top-level root.
  std::vector<Operation *> rootOperations;
  /// Algorithm Phase ownership parallel to rootOperations.  This is produced
  /// by PhaseBoundaryAnalysis and is immutable input to StageBoundaryAnalysis;
  /// Stage partitioning must not move a root across this boundary.
  std::vector<std::string> rootPhaseIds;
  std::vector<Operation *> localSimtAnchorRoots;

  bool hasOperationGraph() const {
    return !rootOperations.empty() &&
           rootOperations.size() == rootPhaseIds.size();
  }
};

/// Ordered post-transform TTIR roots plus the exact roots covered by each
/// materializable SIMT anchor.  AutoBlockify V1's outer loop is represented
/// as a scheduling shell while its direct body operations remain semantic
/// roots; this prevents double ownership.
struct ProgramStructure {
  std::vector<Operation *> rootOperations;
  std::vector<Operation *> localSimtAnchorRoots;
};

class ProgramStructureAnalysis {
public:
  llvm::Expected<ProgramStructure>
  analyze(ModuleOp module, const SimtAnchorPlan &anchorPlan) const;
};

/// Recognizes algorithm-level serial regions.  The current feature-summary
/// overload is an explicit fallback; the operation-graph overload is the
/// target implementation for production Stage ownership.
class PhaseBoundaryAnalysis {
public:
  llvm::Expected<std::optional<PhaseBoundaryPlan>>
  analyze(const SimdSimtFeatureSummary &features,
          const StagePartitionerOptions &options) const;

  llvm::Expected<std::optional<PhaseBoundaryPlan>>
  analyze(ModuleOp module, const SimtAnchorPlan &anchorPlan,
          const SimdSimtFeatureSummary &features,
          const StagePartitionerOptions &options) const;
};

/// Splits each Phase into single-kind Stages.  It does not evaluate cycles or
/// choose SIMD/SIMT.  The optional anchor plan is used only as exact ownership
/// evidence for materializable local SIMT Stages.
class StageBoundaryAnalysis {
public:
  llvm::Expected<StagePartition>
  analyze(const PhaseBoundaryPlan &phasePlan,
          const SimdSimtFeatureSummary &features,
          const SimtAnchorPlan *anchorPlan = nullptr) const;
};

/// Derives structural facts for every already-owned Stage.  It never chooses
/// a route and never reads a hardware throughput profile.
class StageFeatureAnalysis {
public:
  llvm::Error analyze(StagePartition &partition) const;
};

/// Verifies/classifies the one dominant resource semantics of every Stage.
/// Strong structures (recurrence, reduction, dot, indirect/continuous
/// memory, conversion) are inferred from owned operations and must agree
/// with the boundary plan.  Scalar sub-kinds remain boundary semantics.
class StageKindClassifier {
public:
  llvm::Error analyze(StagePartition &partition, int64_t tinyDotFlopsMax) const;
};

/// Verifies that operation/resource work is owned exactly once.  Unlike the
/// retired implementation, this component never distributes a kernel total
/// using per-kind weights.
class StageWorkloadAnalysis {
public:
  llvm::Error analyze(StagePartition &partition) const;
  llvm::Error verify(const StagePartition &partition,
                     const StageWorkload &kernelWorkload) const;
};

/// Derives legal SIMD/SIMT implementations from structural Stage facts.
class StageModeLegalityAnalysis {
public:
  llvm::Error analyze(StagePartition &partition,
                      int64_t maximumSuperblockFactor = 4,
                      bool scopeSuperblockMaterializable = false) const;
};

class StagePartitionVerifier {
public:
  llvm::Error verify(const StagePartition &partition,
                     const StageWorkload &kernelWorkload) const;
};

/// Partitions post-layout/post-AutoBlockify-V1 TTIR facts into serial Phases
/// and single-mode Stages.  A Stage may later be evaluated as SIMD or SIMT,
/// but it is never internally mixed.
class StagePartitioner {
public:
  llvm::Expected<std::optional<StagePartition>>
  partition(const SimdSimtFeatureSummary &features,
            const StagePartitionerOptions &options) const;

  llvm::Expected<std::optional<StagePartition>>
  partition(const SimdSimtFeatureSummary &features,
            const StagePartitionerOptions &options,
            const SimtAnchorPlan &anchorPlan) const;

  llvm::Expected<std::optional<StagePartition>>
  partition(ModuleOp module, const SimtAnchorPlan &anchorPlan,
            const SimdSimtFeatureSummary &features,
            const StagePartitionerOptions &options) const;
};

/// Builds the mode-independent kernel workload used only for ownership
/// conservation checks.  It contains logical elements/bytes, never cycles.
StageWorkload buildKernelStageWorkload(const SimdSimtFeatureSummary &features);

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_STAGEPARTITIONER_H
