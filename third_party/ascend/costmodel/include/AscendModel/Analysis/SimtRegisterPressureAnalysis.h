//===- SimtRegisterPressureAnalysis.h - Whole-VF live values --*- C++ -*-===//

#ifndef ASCENDMODEL_ANALYSIS_SIMTREGISTERPRESSUREANALYSIS_H
#define ASCENDMODEL_ANALYSIS_SIMTREGISTERPRESSUREANALYSIS_H

#include "AscendModel/RouteModel/StageCostModels.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::ascend {

/// Analyze liveness over the whole prospective pure-SIMT function, not each
/// Stage in isolation.
class SimtRegisterPressureAnalysis {
public:
  llvm::Error analyze(ModuleOp module, StagePartition &partition,
                      int64_t baseWarpCount) const;
};

} // namespace mlir::ascend

#endif
