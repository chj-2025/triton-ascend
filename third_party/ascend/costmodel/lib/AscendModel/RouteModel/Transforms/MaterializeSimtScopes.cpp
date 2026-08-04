//===- MaterializeSimtScopes.cpp - Materialize local SIMT scopes --------===//
//
// Materialization consumes the immutable SimtAnchorPlan produced by the Route
// Model and creates SSA-safe scope.scope regions.  No per-operation selection
// marker is written or read.  The compatibility pass below only validates that
// an admitted mixed decision already carries its local scope contract.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/RouteModel/SimtSelection.h"
#include "AscendModel/RouteModel/SimtAnchorAnalysis.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

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
static LogicalResult wrapAnchorOperation(Operation *op) {
  OpBuilder builder(op);
  OperationState scopeState(op->getLoc(), "scope.scope");
  scopeState.addTypes(op->getResultTypes());
  scopeState.addAttribute(kVectorModeAttr, builder.getStringAttr("simt"));
  scopeState.addRegion();
  Operation *scopeOp = builder.create(scopeState);

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

/// Find the contiguous setup/recurrence/final-update range represented by a
/// triangular-solve loop anchor.  The source TTIR has the four input loads
/// immediately before the setup and the dense dot tail immediately after the
/// final update; those operations deliberately stay outside the range.
static SmallVector<Operation *>
collectTriangularSolveRange(Operation *anchor) {
  SmallVector<Operation *> result;
  auto anchorKind = anchor ? classifyMixedSimtAnchor(anchor) : std::nullopt;
  if (!anchor || !anchorKind ||
      *anchorKind != SimtAnchorKind::TriangularSolveLoop ||
      !anchor->getBlock())
    return result;

  Block *block = anchor->getBlock();
  Operation *firstLoop = nullptr;
  Operation *lastLoop = nullptr;
  for (Operation &nested : *block) {
    auto kind = classifyMixedSimtAnchor(&nested);
    if (!kind || *kind != SimtAnchorKind::TriangularSolveLoop)
      continue;
    if (!firstLoop)
      firstLoop = &nested;
    lastLoop = &nested;
  }
  if (!firstLoop || !lastLoop)
    return result;

  Operation *lastInputLoad = nullptr;
  Operation *cursor = firstLoop->getPrevNode();
  while (cursor && cursor->getName().getStringRef() != "tt.load")
    cursor = cursor->getPrevNode();
  if (cursor)
    lastInputLoad = cursor;
  Operation *start = lastInputLoad ? lastInputLoad->getNextNode() : firstLoop;

  Operation *end = lastLoop;
  cursor = lastLoop->getNextNode();
  while (cursor) {
    llvm::StringRef name = cursor->getName().getStringRef();
    if (name != "arith.uitofp" && name != "arith.addf" &&
        name != "arith.select")
      break;
    end = cursor;
    cursor = cursor->getNextNode();
  }

  for (Operation *op = start; op; op = op->getNextNode()) {
    result.push_back(op);
    if (op == end)
      break;
  }
  if (result.empty() || result.back() != end)
    result.clear();
  return result;
}

/// Wrap a contiguous range and thread only values that escape the range.  The
/// existing single-op wrapper is intentionally kept for all other anchors.
static LogicalResult wrapAnchorRange(ArrayRef<Operation *> ops) {
  if (ops.empty())
    return success();
  Block *parent = ops.front()->getBlock();
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

  OpBuilder builder(ops.front());
  OperationState scopeState(ops.front()->getLoc(), "scope.scope");
  SmallVector<Type> escapingTypes;
  escapingTypes.reserve(escaping.size());
  for (Value value : escaping)
    escapingTypes.push_back(value.getType());
  scopeState.addTypes(escapingTypes);
  scopeState.addAttribute(kVectorModeAttr,
                          builder.getStringAttr("simt"));
  scopeState.addRegion();
  Operation *scopeOp = builder.create(scopeState);

  Region &scopeRegion = scopeOp->getRegion(0);
  auto *scopeBody = new Block();
  scopeRegion.push_back(scopeBody);
  for (Operation *op : ops)
    op->moveBefore(scopeBody, scopeBody->end());

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(scopeBody);
  OperationState returnState(ops.front()->getLoc(), "scope.return");
  returnState.addOperands(escaping);
  Operation *returnOp = bodyBuilder.create(returnState);

  if (scopeOp->getNumResults() != escaping.size())
    return failure();
  for (auto [original, replacement] :
       llvm::zip_equal(escaping, scopeOp->getResults())) {
    for (OpOperand &use : llvm::make_early_inc_range(original.getUses()))
      if (use.getOwner() != returnOp &&
          !isInsideRange(use.getOwner()))
        use.set(replacement);
  }
  return success();
}

} // namespace

LogicalResult materializeSimtAnchorPlan(ModuleOp module,
                                        const SimtAnchorPlan &plan) {
  SmallVector<Operation *> anchorOps;
  SmallVector<SmallVector<Operation *>> anchorRanges;
  DenseSet<Operation *> coveredByRange;

  for (const SimtAnchorDescriptor &anchor : plan.anchors) {
    Operation *op = anchor.operation;
    if (!anchor.materializable || !op || coveredByRange.contains(op))
      continue;
    if (hasEnclosingVectorMode(op, "simt"))
      continue;

    if (anchor.kind == SimtAnchorKind::TriangularSolveLoop) {
      SmallVector<Operation *> range = collectTriangularSolveRange(op);
      if (range.empty())
        return op->emitError(
            "triangular SIMT anchor has no materializable contiguous range");
      for (Operation *rangeOp : range)
        coveredByRange.insert(rangeOp);
      anchorRanges.push_back(std::move(range));
      continue;
    }

    if (!isMaterializable(op))
      return op->emitError(
          "SIMT anchor is not materializable as a local scope");
    anchorOps.push_back(op);
  }

  int64_t materialized = 0;
  for (const SmallVector<Operation *> &range : anchorRanges) {
    if (failed(wrapAnchorRange(range)))
      return failure();
    ++materialized;
  }
  for (Operation *op : anchorOps) {
    if (failed(wrapAnchorOperation(op)))
      return failure();
    ++materialized;
  }

  if (materialized == 0)
    return module.emitError(
        "mixed_simd_simt has no materializable local SIMT scope");
  return success();
}

namespace {

static bool containsLocalSimtScope(ModuleOp module) {
  bool found = false;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "scope.scope")
      return WalkResult::advance();
    auto mode = op->getAttrOfType<StringAttr>(kVectorModeAttr);
    if (!mode || mode.getValue() != "simt")
      return WalkResult::advance();
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

struct MaterializeSimtScopesPass
    : public impl::MaterializeSimtScopesPassBase<
          MaterializeSimtScopesPass> {
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
