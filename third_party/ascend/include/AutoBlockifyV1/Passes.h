//===- Passes.h - AutoBlockify V1 pass registration ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ADAPTER_AUTO_BLOCKIFY_V1_PASSES_H
#define TRITON_ADAPTER_AUTO_BLOCKIFY_V1_PASSES_H

#include "AutoBlockifyV1.h"

namespace mlir::triton {

#define GEN_PASS_REGISTRATION
#include "ascend/include/AutoBlockifyV1/Passes.h.inc"

} // namespace mlir::triton

#endif // TRITON_ADAPTER_AUTO_BLOCKIFY_V1_PASSES_H
