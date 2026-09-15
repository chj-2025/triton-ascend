//===- MaterializeSimtScopes.cpp - Materialize local SIMT scopes --------===//
//
// Materialization consumes the immutable SimtAnchorPlan produced by the Route
// Model and creates SSA-safe scope.scope regions.  No per-operation selection
// marker is written or read.  The compatibility pass below only validates that
// an admitted mixed decision already carries its local scope contract.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/Analysis/SimtAnchorAnalysis.h"
#include "AscendModel/Support/CostModelLogger.h"
#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/Transforms/SimtSelection.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <utility>

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_MATERIALIZESIMTSCOPESPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

using namespace simt_selection;

static bool isMaterializable(Operation *op) {
  return op->getBlock() && !isa<ModuleOp>(op) &&
         !op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         !op->hasTrait<OpTrait::IsTerminator>() &&
         op->getName().getStringRef() != "scope.scope" &&
         op->getName().getStringRef() != "scope.return";
}

/// Wrap one anchor operation and thread all of its SSA results through
/// scope.return.
///
/// Scope regions are not isolated from above, so operands remain legal
/// captures.  Moving only the planned operation keeps SIMD producers and
/// consumers outside the SIMT region.
inline constexpr llvm::StringLiteral kScopeSuperblockFactorAttr =
    "ascend.scope_superblock.factor";

static void setScopeExecutionAttrs(Operation *scopeOp, OpBuilder &builder,
                                   int64_t superblockFactor) {
  scopeOp->setAttr(kVectorModeAttr, builder.getStringAttr("simt"));
  scopeOp->setAttr(kScopeSuperblockFactorAttr,
                   builder.getI64IntegerAttr(superblockFactor));
}

static LogicalResult wrapAnchorOperation(Operation *op,
                                         int64_t superblockFactor) {
  OpBuilder builder(op);
  OperationState scopeState(op->getLoc(), "scope.scope");
  scopeState.addTypes(op->getResultTypes());
  scopeState.addRegion();
  Operation *scopeOp = builder.create(scopeState);
  setScopeExecutionAttrs(scopeOp, builder, superblockFactor);

  Region &scopeRegion = scopeOp->getRegion(0);
  auto *scopeBody = new Block();
  scopeRegion.push_back(scopeBody);

  SmallVector<Value> originalResults(op->getResults());
  op->moveBefore(scopeBody, scopeBody->end());

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(scopeBody);
  OperationState returnState(op->getLoc(), "scope.return");
  returnState.addOperands(originalResults);
  Operation *returnOp = bodyBuilder.create(returnState);

  if (scopeOp->getNumResults() != originalResults.size())
    return op->emitError("SIMT scope result count does not match anchor op");

  for (auto [original, replacement] :
       llvm::zip_equal(originalResults, scopeOp->getResults())) {
    original.replaceAllUsesExcept(replacement, returnOp);
  }
  return success();
}

/// Wrap an exact planned operation set and thread only values that escape it.
/// `insertionPoint` lets solve_tril move pure mask setup across the initial
/// loads while keeping those loads outside, matching the hand-written scope.
static LogicalResult wrapAnchorRange(ArrayRef<Operation *> ops,
                                     Operation *insertionPoint,
                                     int64_t superblockFactor) {
  if (ops.empty())
    return success();
  Block *parent = insertionPoint ? insertionPoint->getBlock() : nullptr;
  if (!parent)
    return failure();

  DenseSet<Operation *> planned;
  for (Operation *op : ops) {
    if (!op || op->getBlock() != parent || !isMaterializable(op))
      return failure();
    planned.insert(op);
  }

  auto isInsideRange = [&](Operation *user) {
    for (Operation *owner = user; owner; owner = owner->getParentOp())
      if (planned.contains(owner))
        return true;
    return false;
  };

  SmallVector<Value> escaping;
  DenseSet<Value> seen;
  for (Operation *op : ops)
    for (Value result : op->getResults())
      for (OpOperand &use : result.getUses())
        if (!isInsideRange(use.getOwner()) && seen.insert(result).second) {
          escaping.push_back(result);
          break;
        }

  OpBuilder builder(insertionPoint);
  OperationState scopeState(insertionPoint->getLoc(), "scope.scope");
  SmallVector<Type> escapingTypes;
  escapingTypes.reserve(escaping.size());
  for (Value value : escaping)
    escapingTypes.push_back(value.getType());
  scopeState.addTypes(escapingTypes);
  scopeState.addRegion();
  Operation *scopeOp = builder.create(scopeState);
  setScopeExecutionAttrs(scopeOp, builder, superblockFactor);

  Region &scopeRegion = scopeOp->getRegion(0);
  auto *scopeBody = new Block();
  scopeRegion.push_back(scopeBody);
  for (Operation *op : ops)
    op->moveBefore(scopeBody, scopeBody->end());

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(scopeBody);
  OperationState returnState(insertionPoint->getLoc(), "scope.return");
  returnState.addOperands(escaping);
  Operation *returnOp = bodyBuilder.create(returnState);

  if (scopeOp->getNumResults() != escaping.size())
    return failure();
  for (auto [original, replacement] :
       llvm::zip_equal(escaping, scopeOp->getResults())) {
    for (OpOperand &use : llvm::make_early_inc_range(original.getUses()))
      if (use.getOwner() != returnOp && !isInsideRange(use.getOwner()))
        use.set(replacement);
  }
  return success();
}

} // namespace

LogicalResult materializeSimtAnchorPlan(ModuleOp module,
                                        const SimtAnchorPlan &plan,
                                        int64_t superblockFactor) {
  COSTMODEL_TRACE("materializeSimtAnchorPlan");
  costModelLog() << "input: module anchors=" << plan.anchors.size()
                 << " superblock_factor=" << superblockFactor << "\n";
  if (superblockFactor <= 0 || (superblockFactor & (superblockFactor - 1)) != 0)
    return module.emitError(
        "SIMT scope superblock factor must be a positive power of two");
  struct PlannedRange {
    SmallVector<Operation *> operations;
    Operation *insertionPoint = nullptr;
  };
  SmallVector<Operation *> anchorOps;
  SmallVector<PlannedRange> anchorRanges;
  DenseSet<Operation *> coveredByRange;

  for (auto indexedAnchor : llvm::enumerate(plan.anchors)) {
    const size_t index = indexedAnchor.index();
    const SimtAnchorDescriptor &anchor = indexedAnchor.value();
    Operation *op = anchor.operation;
    auto logAnchorDecision = [&](llvm::StringRef action) {
      llvm::raw_ostream &os = costModelDebug();
      os << "anchor[" << index << "]: action=" << action
         << " kind=" << stringifySimtAnchorKind(anchor.kind)
         << " materializable=" << anchor.materializable;
      if (op)
        os << " op=" << op->getName().getStringRef() << " " << op->getLoc();
      os << " scopeOperations=" << anchor.scopeOperations.size() << "\n";
    };
    if (!anchor.materializable || !op) {
      logAnchorDecision(!op ? "skip_null_operation"
                            : "skip_not_materializable");
      continue;
    }
    if (coveredByRange.contains(op)) {
      logAnchorDecision("skip_covered_by_compound_range");
      continue;
    }
    if (hasEnclosingVectorMode(op, "simt")) {
      logAnchorDecision("skip_already_inside_simt_scope");
      continue;
    }

    if (anchor.scopeOperations.size() > 1) {
      for (Operation *rangeOp : anchor.scopeOperations)
        coveredByRange.insert(rangeOp);
      PlannedRange range;
      llvm::append_range(range.operations, anchor.scopeOperations);
      range.insertionPoint = anchor.scopeInsertionPoint;
      anchorRanges.push_back(std::move(range));
      logAnchorDecision("plan_compound_scope");
      continue;
    }

    if (!isMaterializable(op)) {
      logAnchorDecision("error_not_materializable");
      return op->emitError(
          "SIMT anchor is not materializable as a local scope");
    }
    anchorOps.push_back(op);
    logAnchorDecision("plan_single_operation_scope");
  }

  int64_t materialized = 0;
  for (const PlannedRange &range : anchorRanges) {
    if (failed(wrapAnchorRange(range.operations, range.insertionPoint,
                               superblockFactor)))
      return failure();
    ++materialized;
  }
  for (Operation *op : anchorOps) {
    if (failed(wrapAnchorOperation(op, superblockFactor)))
      return failure();
    ++materialized;
  }

  if (materialized == 0)
    return module.emitError(
        "mixed_simd_simt has no materializable local SIMT scope");
  costModelLog() << "output: materialized scopes=" << materialized << "\n";
  return success();
}

namespace {

static bool containsLocalSimtScope(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "scope.scope")
      return WalkResult::advance();
    auto mode = getVectorMode(op);
    if (!mode || mode.getValue() != "simt")
      return WalkResult::advance();
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

struct MaterializeSimtScopesPass
    : public impl::MaterializeSimtScopesPassBase<MaterializeSimtScopesPass> {
  using MaterializeSimtScopesPassBase::MaterializeSimtScopesPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!isMixedModelDecision(module))
      return;
    if (containsLocalSimtScope(module))
      return;
    module.emitError(
        "mixed_simd_simt requires a materialized scope.scope<simt> contract");
    signalPassFailure();
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
