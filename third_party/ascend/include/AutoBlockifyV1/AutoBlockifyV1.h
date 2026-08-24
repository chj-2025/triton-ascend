//===- AutoBlockifyV1.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ADAPTER_AUTO_BLOCKIFY_V1_H
#define TRITON_ADAPTER_AUTO_BLOCKIFY_V1_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_DECL_TASIMTAUTOBLOCKIFYV1
#define GEN_PASS_DECL_TAREFINESIMTAUTOBLOCKIFYV1SUPERBLOCK
#include "ascend/include/AutoBlockifyV1/Passes.h.inc"

#define GEN_PASS_DEF_TASIMTAUTOBLOCKIFYV1
#define GEN_PASS_DEF_TAREFINESIMTAUTOBLOCKIFYV1SUPERBLOCK
#include "ascend/include/AutoBlockifyV1/Passes.h.inc"

namespace mlir::triton {

std::unique_ptr<OperationPass<FuncOp>>
createTASIMTAutoBlockifyV1Pass(const TASIMTAutoBlockifyV1Options &options = {});

std::unique_ptr<OperationPass<FuncOp>>
createTARefineSIMTAutoBlockifyV1SuperBlockPass(
    const TARefineSIMTAutoBlockifyV1SuperBlockOptions &options = {});

} // namespace mlir::triton

#endif // TRITON_ADAPTER_AUTO_BLOCKIFY_V1_H
