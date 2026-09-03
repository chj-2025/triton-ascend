//===- TTIRLayoutMergePass.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ADAPTER_TTIR_LAYOUT_MERGE_PASS_H
#define TRITON_ADAPTER_TTIR_LAYOUT_MERGE_PASS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DEF_TTIRLAYOUTMERGE
#include "ascend/include/TTIRLayoutMerge/Passes.h.inc"

namespace mlir::triton {

std::unique_ptr<OperationPass<mlir::ModuleOp>> createTTIRLayoutMergePass();

} // namespace mlir::triton

#endif // TRITON_ADAPTER_TTIR_LAYOUT_MERGE_PASS_H
