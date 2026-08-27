#ifndef ASCENDMODEL_COSTMODELTRACE_H
#define ASCENDMODEL_COSTMODELTRACE_H

#include "mlir/IR/Operation.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>

namespace mlir::ascend {

// Runtime log verbosity, resolved once from $COSTMODEL_LOG_LEVEL:
//   0 | "off"     : silent
//   1 | "info"    : interface call graph + per-step inputs/outputs + IR dumps
//                  (default)
//   2 | "verbose" : additionally every per-stage / per-implementation detail
inline unsigned costModelTraceDepth = 0;

namespace detail {
inline int costModelLogLevelFromEnv() {
  if (const char *env = std::getenv("COSTMODEL_LOG_LEVEL")) {
    const llvm::StringRef value = llvm::StringRef(env).trim();
    if (value == "0" || value.equals_insensitive("off"))
      return 0;
    if (value == "2" || value.equals_insensitive("verbose") ||
        value.equals_insensitive("debug"))
      return 2;
  }
  return 1;
}
} // namespace detail

inline int costModelLogLevel = detail::costModelLogLevelFromEnv();

// Level >= 1: interface call relations and per-step inputs/outputs.
inline llvm::raw_ostream &costModelLog() {
  if (costModelLogLevel < 1)
    return llvm::nulls();
  llvm::raw_ostream &os = llvm::errs() << "[COSTMODEL] ";
  for (unsigned i = 0; i < costModelTraceDepth; ++i)
    os << "  ";
  return os;
}

// Level >= 2: internal detail (per-stage features/workloads/implementation
// costs, ownership bookkeeping, ...).
inline llvm::raw_ostream &costModelDebug() {
  if (costModelLogLevel < 2)
    return llvm::nulls();
  return costModelLog();
}

// Dump an IR snapshot (costmodel input module, analysis module, materialized
// module) at level >= 1.  The IR body is printed without trace indentation so
// it stays directly copy-pasteable into test cases.
inline void costModelDumpIR(const char *label, Operation *op) {
  if (!op || costModelLogLevel < 1)
    return;
  llvm::errs() << "[COSTMODEL] ===== IR dump: " << label << " =====\n";
  op->print(llvm::errs());
  llvm::errs() << "\n[COSTMODEL] ===== IR dump end =====\n";
}

class CostModelTraceScope {
public:
  explicit CostModelTraceScope(const char *name, int level = 1)
      : name(name), level(level) {
    if (costModelLogLevel >= level) {
      active = true;
      costModelLog() << ">>> " << name << "\n";
      ++costModelTraceDepth;
    }
  }
  ~CostModelTraceScope() {
    if (active) {
      --costModelTraceDepth;
      costModelLog() << "<<< " << name << "\n";
    }
  }
  CostModelTraceScope(const CostModelTraceScope &) = delete;
  CostModelTraceScope &operator=(const CostModelTraceScope &) = delete;

private:
  const char *name;
  int level;
  bool active = false;
};

} // namespace mlir::ascend

// Interface-level trace: printed at level >= 1.
#define COSTMODEL_TRACE(FunctionName)                                          \
  ::mlir::ascend::CostModelTraceScope costModelTraceScope(FunctionName, 1)

// Internal-helper trace: printed only at level >= 2.
#define COSTMODEL_TRACE_DEBUG(FunctionName)                                    \
  ::mlir::ascend::CostModelTraceScope costModelTraceScope(FunctionName, 2)

#endif // ASCENDMODEL_COSTMODELTRACE_H
