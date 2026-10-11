//===- SimtRegisterPressureAnalysis.h - Whole-VF live values --*- C++ -*-===//

#ifndef ASCENDMODEL_ANALYSIS_SIMTREGISTERPRESSUREANALYSIS_H
#define ASCENDMODEL_ANALYSIS_SIMTREGISTERPRESSUREANALYSIS_H

#include "AscendModel/RouteModel/StageCostModels.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::ascend {

/// Analyze liveness over the whole prospective pure-SIMT function, not each
/// Stage in isolation. The result is a TTIR pressure proxy, NOT a register
/// allocation prediction or a proven lower bound. No performance is charged.
/// baseWarpCount maps one logical program; SuperBlock replication must not
/// multiply its per-thread live state. Local scopes need separate analysis.
class SimtRegisterPressureAnalysis {
public:
  llvm::Error analyze(ModuleOp module, StagePartition &partition,
                      int64_t baseWarpCount) const;
};

} // namespace mlir::ascend

#endif
