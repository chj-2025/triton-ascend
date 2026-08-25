//===- Passes.h - TTIR layout-merge passes -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_TTIR_LAYOUT_MERGE_PASSES_H
#define TRITON_ASCEND_TTIR_LAYOUT_MERGE_PASSES_H

#include "TTIRLayoutMergePass.h"

namespace mlir::triton {

#define GEN_PASS_REGISTRATION
#include "ascend/include/TTIRLayoutMerge/Passes.h.inc"

} // namespace mlir::triton

#endif // TRITON_ASCEND_TTIR_LAYOUT_MERGE_PASSES_H
