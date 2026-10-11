//===- AutoBlockifyV1.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Triton-Adapter port of the NPUIR AutoBlockify V1 transform.
//
// V1 preserves the tile tensor shapes of one logical Triton program.  It caps
// the physical launch to one vector-core wave and executes the original
// logical programs from an outer loop:
//
//   chunk = ceildiv(logicalGridSize, physicalVectorCoreCount)
//   for linear in [hwBlockId * chunk,
//                  min((hwBlockId + 1) * chunk, logicalGridSize)):
//     linear += (warp_id % factor)  // when factor > 1, with step=factor
//     (pidX, pidY, pidZ) = unflatten(linear, gridX, gridY)
//
// AutoBlockify V2 lives in ascend/lib/AutoBlockify.  It changes tensor tile
// shapes and is deliberately kept independent from this pass.
//
//===----------------------------------------------------------------------===//

#include "ascend/include/AutoBlockifyV1/AutoBlockifyV1.h"
#include "ascend/include/Utils/SuperBlockFactor.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "ta-auto-blockify-v1"

namespace mlir::triton {
#define GEN_PASS_DEF_TAAUTOBLOCKIFYV1
#define GEN_PASS_DEF_TAREFINEAUTOBLOCKIFYV1SUPERBLOCK
#include "ascend/include/AutoBlockifyV1/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace mlir::triton {
namespace {

static constexpr llvm::StringLiteral autoBlockifyV1Attr = "ta.auto_blockify_v1";
static constexpr llvm::StringLiteral autoBlockifyV1LoopAttr =
    "ta.auto_blockify_v1.loop";
static constexpr llvm::StringLiteral autoBlockifyV1ScheduleAttr =
    "ta.auto_blockify_v1.schedule";
static constexpr llvm::StringLiteral autoBlockifyV1SuperBlockFactorAttr =
    "ta.auto_blockify_v1.superblock_factor";

struct GridValues {
  Value x;
  Value y;
  Value z;
};

struct LogicalProgramIds {
  Value x;
  Value y;
  Value z;
};

static GridValues buildGridValues(OpBuilder &builder, Location loc) {
  return {
      builder.create<GetNumProgramsOp>(loc, ProgramIDDim::X),
      builder.create<GetNumProgramsOp>(loc, ProgramIDDim::Y),
      builder.create<GetNumProgramsOp>(loc, ProgramIDDim::Z),
  };
}

static LogicalProgramIds buildLogicalProgramIds(OpBuilder &builder,
                                                Location loc,
                                                Value linearProgramId,
                                                GridValues grid) {
  Value divX = builder.create<arith::DivUIOp>(loc, linearProgramId, grid.x);
  return {
      builder.create<arith::RemUIOp>(loc, linearProgramId, grid.x),
      builder.create<arith::RemUIOp>(loc, divX, grid.y),
      builder.create<arith::DivUIOp>(loc, divX, grid.y),
  };
}

class TAAutoBlockifyV1Pass
    : public impl::TAAutoBlockifyV1Base<TAAutoBlockifyV1Pass> {
public:
  using Base = impl::TAAutoBlockifyV1Base<TAAutoBlockifyV1Pass>;
  using Base::Base;

  explicit TAAutoBlockifyV1Pass(const TAAutoBlockifyV1Options &options)
      : Base(options) {}

  void runOnOperation() override {
    FuncOp ttFunc = getOperation();

    unsigned physicalCoreCount = physicalVectorCoreCount;
    if (physicalCoreCount == 0) {
      ttFunc.emitError("physical-vector-core-count must be greater than zero");
      return signalPassFailure();
    }

    unsigned factor = superBlockFactor;
    if (!ascend::isSupportedSuperBlockFactor(factor)) {
      ttFunc.emitError("superblock-factor must be one of ")
          << ascend::kSupportedSuperBlockFactorsDescription << ", got "
          << factor;
      return signalPassFailure();
    }

    // Only transform entry kernels.  Helper functions retain their original
    // program-id semantics.
    if (!ttFunc.isPublic() || !ttFunc.getResultTypes().empty())
      return;

    // Preserve operation identity across the move into the generated loop so
    // the cost model can separate V1 scheduling IR from the original
    // algorithm IR.  Newly created operations are tagged at the end.
    llvm::DenseSet<Operation *> originalOperations;
    ttFunc.walk([&](Operation *op) { originalOperations.insert(op); });

    SmallVector<GetProgramIdOp> programIdOps;
    ttFunc.walk([&](GetProgramIdOp op) { programIdOps.push_back(op); });
    if (programIdOps.empty())
      return;

    bool originalFuncHasOneBlock = ttFunc.getBody().hasOneBlock();
    Block *entryBlock = &ttFunc.getBody().front();
    auto *bodyBlock = entryBlock->splitBlock(entryBlock->begin());

    Location loc = ttFunc.getLoc();
    OpBuilder builder(entryBlock, entryBlock->begin());

    GridValues grid = buildGridValues(builder, loc);
    Value yz = builder.create<arith::MulIOp>(loc, grid.y, grid.z);
    Value logicalBlockCount = builder.create<arith::MulIOp>(loc, grid.x, yz);
    // Triton-Adapter's V1 contract represents the physical launch id with
    // tt.get_program_id x.  The NPUIR lowering consumes this value as the
    // flattened physical-block index; there is no gpu.linear_block_id
    // operation in the Triton dialect used by this pass.  Keep the original
    // tt.get_program_id operations in `programIdOps`; they are replaced below
    // with the unflattened logical ids.
    Value blockIdx = builder.create<GetProgramIdOp>(loc, ProgramIDDim::X);
    Value physicalBlockCount = builder.create<arith::ConstantIntOp>(
        loc, physicalCoreCount, /*width=*/32);
    // Match NPUIR's Triton-level AutoBlockify V1 schedule: each physical
    // block owns one contiguous chunk of logical programs.  The factor only
    // changes how that chunk is consumed by the SuperBlock; it does not turn
    // the chunk into a global strided loop or change the tile shape.
    Value chunk = builder.create<arith::CeilDivUIOp>(loc, logicalBlockCount,
                                                     physicalBlockCount);
    Value lowerBound = builder.create<arith::MulIOp>(loc, blockIdx, chunk);
    Value end = builder.create<arith::AddIOp>(loc, lowerBound, chunk);
    Value upperBound =
        builder.create<arith::MinUIOp>(loc, end, logicalBlockCount);
    Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);

    auto forOp = builder.create<scf::ForOp>(loc, lowerBound, upperBound, one);
    forOp->setAttr(autoBlockifyV1LoopAttr, builder.getUnitAttr());
    // Keep the selected factor on the scheduling loop as well as on the
    // function.  NPUIR exposes this loop-local fact and downstream scope
    // materialization uses it to distinguish an F1 V1 loop from a refined
    // SuperBlock loop.  The function attribute remains for route metadata.
    forOp->setAttr(autoBlockifyV1SuperBlockFactorAttr,
                   builder.getI32IntegerAttr(factor));
    builder.create<ReturnOp>(loc);

    if (originalFuncHasOneBlock) {
      assert(std::next(Region::iterator(bodyBlock)) == ttFunc.getBody().end() &&
             "the original single body block must be the final block");
      Block *newBodyBlock = forOp.getBody();
      newBodyBlock->getOperations().splice(newBodyBlock->begin(),
                                           bodyBlock->getOperations());
      bodyBlock->erase();
    } else {
      OpBuilder bodyBuilder(forOp.getBody(), forOp.getBody()->begin());
      auto executeRegion =
          bodyBuilder.create<scf::ExecuteRegionOp>(forOp.getLoc(), TypeRange{});
      Region &newBodyRegion = executeRegion.getRegion();
      Region &bodyRegion = ttFunc.getBody();
      newBodyRegion.getBlocks().splice(
          newBodyRegion.begin(), bodyRegion.getBlocks(),
          Region::iterator(bodyBlock), bodyRegion.end());
    }

    Value linearProgramId = forOp.getInductionVar();
    builder.setInsertionPointAfterValue(linearProgramId);

    if (factor > 1) {
      Value iv = forOp.getInductionVar();
      builder.setInsertionPoint(forOp);
      Value factorValue =
          builder.create<arith::ConstantIntOp>(loc, iv.getType(), factor);
      // Keep the warp-size constant outside the loop body, as in NPUIR.  It
      // is a schedule parameter, not per-logical-program work.
      Value warpSize =
          builder.create<arith::ConstantIntOp>(loc, iv.getType(), 32);
      Block *loopBody = forOp.getBody();
      forOp.setStep(factorValue);
      OpBuilder bodyBuilder(loopBody, loopBody->begin());
      Value tidIndex =
          bodyBuilder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
      Value tid =
          bodyBuilder.create<arith::IndexCastOp>(loc, iv.getType(), tidIndex);
      Value warpId = bodyBuilder.create<arith::DivUIOp>(loc, tid, warpSize);
      linearProgramId = bodyBuilder.create<arith::AddIOp>(
          loc, iv,
          bodyBuilder.create<arith::RemUIOp>(loc, warpId, factorValue));

      Value inBounds = bodyBuilder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, linearProgramId,
          forOp.getUpperBound());
      auto ifOp = bodyBuilder.create<scf::IfOp>(loc, inBounds);
      Block *thenBlock = ifOp.thenBlock();
      thenBlock->getOperations().splice(
          thenBlock->begin(), loopBody->getOperations(),
          std::next(Block::iterator(ifOp)), std::prev(loopBody->end()));
      builder.setInsertionPointToStart(thenBlock);
    }

    LogicalProgramIds logicalPids =
        buildLogicalProgramIds(builder, loc, linearProgramId, grid);
    for (GetProgramIdOp op : programIdOps) {
      Value replacement;
      switch (op.getAxis()) {
      case ProgramIDDim::X:
        replacement = logicalPids.x;
        break;
      case ProgramIDDim::Y:
        replacement = logicalPids.y;
        break;
      case ProgramIDDim::Z:
        replacement = logicalPids.z;
        break;
      }
      op.replaceAllUsesWith(replacement);
      op.erase();
    }

    SmallVector<ReturnOp> returnOps;
    forOp.walk([&](ReturnOp op) { returnOps.push_back(op); });
    for (ReturnOp op : returnOps) {
      if (!originalFuncHasOneBlock) {
        OpBuilder returnBuilder(op);
        returnBuilder.create<scf::YieldOp>(op.getLoc(), op.getOperands());
      }
      op.erase();
    }

    ttFunc.walk([&](Operation *op) {
      if (!originalOperations.contains(op))
        op->setAttr(autoBlockifyV1ScheduleAttr, builder.getUnitAttr());
    });
    ttFunc->setAttr(autoBlockifyV1Attr, builder.getUnitAttr());
    ttFunc->setAttr(autoBlockifyV1SuperBlockFactorAttr,
                    builder.getI32IntegerAttr(factor));
    ttFunc->getParentOfType<ModuleOp>()->setAttr(
        "ta.auto_blockify_v1.materialized", builder.getI32IntegerAttr(1));
  }
};

class TARefineAutoBlockifyV1SuperBlockPass
    : public impl::TARefineAutoBlockifyV1SuperBlockBase<
          TARefineAutoBlockifyV1SuperBlockPass> {
public:
  using Base = impl::TARefineAutoBlockifyV1SuperBlockBase<
      TARefineAutoBlockifyV1SuperBlockPass>;
  using Base::Base;

  explicit TARefineAutoBlockifyV1SuperBlockPass(
      const TARefineAutoBlockifyV1SuperBlockOptions &options)
      : Base(options) {}

  void runOnOperation() override {
    FuncOp ttFunc = getOperation();
    const unsigned factor = superBlockFactor;
    if (!ascend::isSupportedSuperBlockFactor(factor)) {
      ttFunc.emitError("superblock-factor must be one of ")
          << ascend::kSupportedSuperBlockFactorsDescription << ", got "
          << factor;
      return signalPassFailure();
    }
    if (factor == 1)
      return;
    if (!ttFunc->hasAttr(autoBlockifyV1Attr)) {
      ttFunc.emitError(
          "SuperBlock refinement requires an existing AutoBlockify V1 loop");
      return signalPassFailure();
    }
    if (auto current = ttFunc->getAttrOfType<IntegerAttr>(
            autoBlockifyV1SuperBlockFactorAttr)) {
      if (current.getInt() == factor)
        return;
      if (current.getInt() != 1) {
        ttFunc.emitError("cannot refine an AutoBlockify V1 loop from factor ")
            << current.getInt() << " to " << factor;
        return signalPassFailure();
      }
    }

    SmallVector<scf::ForOp> schedulingLoops;
    ttFunc.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(autoBlockifyV1LoopAttr))
        schedulingLoops.push_back(loop);
    });
    if (schedulingLoops.size() != 1) {
      ttFunc.emitError("expected exactly one AutoBlockify V1 scheduling loop, "
                       "found ")
          << schedulingLoops.size();
      return signalPassFailure();
    }

    scf::ForOp forOp = schedulingLoops.front();
    Block *loopBody = forOp.getBody();
    SmallVector<Operation *> existingOperations;
    llvm::DenseSet<Operation *> existingSet;
    for (Operation &op : loopBody->without_terminator()) {
      existingOperations.push_back(&op);
      existingSet.insert(&op);
    }

    Value iv = forOp.getInductionVar();
    Value oldUpper = forOp.getUpperBound();
    Location loc = forOp.getLoc();
    OpBuilder beforeLoop(forOp);
    Value factorValue =
        beforeLoop.create<arith::ConstantIntOp>(loc, iv.getType(), factor);
    // The existing factor-1 loop already has NPUIR's contiguous per-core
    // chunk bounds. Refinement only changes the consumption stride within
    // that chunk; it must not rewrite the chunk bounds or physical schedule.
    forOp.setStep(factorValue);

    OpBuilder bodyBuilder(loopBody, loopBody->begin());
    Value warpSize =
        bodyBuilder.create<arith::ConstantIntOp>(loc, iv.getType(), 32);
    Value tidIndex =
        bodyBuilder.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    Value tid =
        bodyBuilder.create<arith::IndexCastOp>(loc, iv.getType(), tidIndex);
    Value warpId = bodyBuilder.create<arith::DivUIOp>(loc, tid, warpSize);
    Value taskId = bodyBuilder.create<arith::RemUIOp>(loc, warpId, factorValue);
    Value logicalProgramId = bodyBuilder.create<arith::AddIOp>(loc, iv, taskId);
    Value inBounds = bodyBuilder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, logicalProgramId, oldUpper);
    auto ifOp = bodyBuilder.create<scf::IfOp>(loc, inBounds);

    for (OpOperand &use : llvm::make_early_inc_range(iv.getUses()))
      if (existingSet.contains(use.getOwner()))
        use.set(logicalProgramId);

    Block *thenBlock = ifOp.thenBlock();
    Operation *thenYield = thenBlock->getTerminator();
    for (Operation *op : existingOperations)
      op->moveBefore(thenYield);

    OpBuilder attrBuilder(ttFunc.getContext());
    for (Operation *op :
         {factorValue.getDefiningOp(), warpSize.getDefiningOp(),
          tidIndex.getDefiningOp(), tid.getDefiningOp(), warpId.getDefiningOp(),
          taskId.getDefiningOp(), logicalProgramId.getDefiningOp(),
          inBounds.getDefiningOp(), ifOp.getOperation()})
      op->setAttr(autoBlockifyV1ScheduleAttr, attrBuilder.getUnitAttr());
    ttFunc->setAttr(autoBlockifyV1SuperBlockFactorAttr,
                    attrBuilder.getI32IntegerAttr(factor));
    forOp->setAttr(autoBlockifyV1SuperBlockFactorAttr,
                   attrBuilder.getI32IntegerAttr(factor));
    ttFunc->getParentOfType<ModuleOp>()->setAttr(
        "ta.auto_blockify_v1.superblock_refined",
        attrBuilder.getI32IntegerAttr(factor));
  }
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>>
createTAAutoBlockifyV1Pass(const TAAutoBlockifyV1Options &options) {
  return std::make_unique<TAAutoBlockifyV1Pass>(options);
}

std::unique_ptr<OperationPass<FuncOp>>
createTARefineAutoBlockifyV1SuperBlockPass(
    const TARefineAutoBlockifyV1SuperBlockOptions &options) {
  return std::make_unique<TARefineAutoBlockifyV1SuperBlockPass>(options);
}

} // namespace mlir::triton
