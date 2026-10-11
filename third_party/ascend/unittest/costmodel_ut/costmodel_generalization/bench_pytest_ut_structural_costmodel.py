#!/usr/bin/env python3
"""Structural cost-model holdouts copied from Ascend pytest_ut.

Each case preserves the source kernel's block shape and launch geometry.  The
only additions are route forcing, correctness validation and device profiling.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as extension
from triton.runtime import driver

from profile_utils import profile_kernel

SOURCE_ROOT = "third_party/ascend/unittest/pytest_ut"


@triton.jit
def discrete_mask_4way_kernel(inp, out, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    mask = ((rows[:, None] < M) & (cols[None, :] < N)
            & ((rows * 2)[:, None] < BLOCK_M)
            & ((cols * 2)[None, :] < BLOCK_N))
    ptr = rows[:, None] * BLOCK_N + cols[None, :]
    value = tl.load(inp + ptr, mask=mask, other=0.0)
    tl.store(out + ptr, value + 1.0, mask=mask)


@triton.jit
def reduce_sum_epilogue_kernel(out, inp, bias, xnumel, rnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr,
                               RBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    for sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + sub + tl.arange(0, XBLOCK_SUB)
        xmask = xindex[:, None] < xnumel
        value = tl.load(inp + rindex + RBLOCK * xindex[:, None], mask=xmask & rmask)
        value = tl.reshape(value, [XBLOCK_SUB, RBLOCK])
        b = tl.load(bias + xindex, mask=xindex < xnumel)
        tl.store(out + xindex, tl.sum(value, axis=1) + b)


@triton.jit
def max_dim0_kernel(inp, out, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    mi = tl.arange(0, MNUMEL)
    ni = tl.arange(0, NNUMEL)
    mask = (mi[:, None] < M) & (ni[None, :] < N)
    value = tl.load(inp + mi[:, None] * N + ni[None, :], mask=mask, other=-float("inf"))
    tl.store(out + ni, tl.max(value, axis=0), mask=ni < N)


@triton.jit
def argmax_2d_kernel(inp, out, DIM: tl.constexpr, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr,
                     NNUMEL: tl.constexpr):
    mi = tl.arange(0, MNUMEL)
    ni = tl.arange(0, NNUMEL)
    mask = (mi[:, None] < M) & (ni[None, :] < N)
    value = tl.load(inp + mi[:, None] * N + ni[None, :], mask=mask, other=-float("inf"))
    index = tl.argmax(value, DIM)
    if DIM == 0:
        tl.store(out + tl.arange(0, N), index)
    else:
        tl.store(out + tl.arange(0, M), index)


@triton.jit
def gather_after_reduce_kernel(logits, topk_ids, out, logits_stride, vocab_size, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    max_value = -float("inf")
    for start in range(0, vocab_size, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        values = tl.load(logits + row * logits_stride + offsets, mask=offsets < vocab_size, other=-float("inf"))
        max_value = tl.maximum(max_value, tl.max(values))
    topk_id = tl.load(topk_ids + row + tl.arange(0, 1))
    selected = tl.load(logits + row * logits_stride + topk_id)
    tl.store(out + row + tl.arange(0, 1), selected - max_value)


@triton.jit
def view_3d_kernel(out, inp, XB: tl.constexpr, YB: tl.constexpr, ZB: tl.constexpr):
    x = tl.arange(0, XB)
    y = tl.arange(0, YB)
    z = tl.arange(0, ZB)
    index = x[:, None, None] * YB * ZB + y[None, :, None] * ZB + z[None, None, :]
    value = tl.load(inp + index)
    viewed = tl.view(value, (ZB, XB * YB))
    out_index = tl.arange(0, ZB)[:, None] * XB * YB + tl.arange(0, XB * YB)[None, :]
    tl.store(out + out_index, viewed)


@triton.jit
def permute_reshape_fused_kernel(a, b, c, out, BLOCK1: tl.constexpr, BLOCK1_SUB: tl.constexpr, BLOCK2: tl.constexpr,
                                 S: tl.constexpr, N: tl.constexpr, D: tl.constexpr):
    offset: tl.constexpr = tl.program_id(0) * BLOCK1
    sbase = tl.arange(0, BLOCK1_SUB)
    dbase = tl.arange(0, BLOCK2)
    for loop1 in range(BLOCK1 // BLOCK1_SUB):
        s = offset + loop1 * BLOCK1_SUB + sbase[:, None]
        for n in range(N):
            for loop2 in range(D // BLOCK2):
                d = loop2 * BLOCK2 + dbase[None, :]
                x = tl.load(a + 32768 * n + 8 * s + d)
                y = tl.load(b + 32768 * n + 8 * s + d)
                z = tl.load(c + 8 * n + d + 96 * s)
                tl.store(out + 8 * n + d + 96 * s, x + y + z)


@triton.jit
def nearest_resize_kernel(src, dst, SRC_COLS: tl.constexpr, DST_COLS: tl.constexpr, RR_H: tl.constexpr,
                          RR_W: tl.constexpr, STRIDE_IN_H: tl.constexpr, STRIDE_IN_W: tl.constexpr,
                          STRIDE_IN_C: tl.constexpr, STRIDE_OUT_H: tl.constexpr, STRIDE_OUT_W: tl.constexpr,
                          STRIDE_OUT_C: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    channel = tl.program_id(0)
    row = tl.program_id(1)
    dst_cols = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dst_offsets = channel * STRIDE_OUT_C + row * STRIDE_OUT_H + dst_cols * STRIDE_OUT_W
    src_col = tl.floor(dst_cols * RR_W)
    src_row = row * RR_H
    src_offsets = (channel * STRIDE_IN_C + src_row.to(tl.int32) * STRIDE_IN_H +
                   tl.clamp(src_col, 0, SRC_COLS - 1).to(tl.int32) * STRIDE_IN_W)
    value = tl.load(src + src_offsets)
    tl.store(dst + dst_offsets, value, mask=dst_cols < DST_COLS)


@triton.jit
def sort_2d_kernel(inp, out, N: tl.constexpr, M: tl.constexpr, DESCENDING: tl.constexpr):
    cols = tl.arange(0, M)
    rows = tl.arange(0, N) * M
    index = cols[None, :] + rows[:, None]
    value = tl.load(inp + index)
    value = extension.sort(value, descending=DESCENDING, dim=1)
    tl.store(out + index, value)


@triton.jit
def topk_kernel(inp, stride_in, out, stride_out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    value = tl.load(inp + rows[:, None] * stride_in + cols[None, :])
    selected = tl.topk(value, K)
    tl.store(out + rows[:, None] * stride_out + tl.arange(0, K)[None, :], selected)


@triton.jit
def scan_pair_add(av, ai, bv, bi):
    return av + bv, ai + bi


@triton.jit
def multi_input_scan_kernel(values, indices, out_values, out_indices, axis_size, total_slices, BLOCK_SIZE: tl.constexpr,
                            REVERSE: tl.constexpr):
    slice_id = tl.program_id(0)
    if slice_id >= total_slices:
        return
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < axis_size
    base = slice_id * axis_size
    value = tl.load(values + base + offsets, mask=mask, other=0.0)
    index = tl.load(indices + base + offsets, mask=mask, other=0)
    result_value, result_index = tl.associative_scan((value, index), axis=0, combine_fn=scan_pair_add, reverse=REVERSE)
    tl.store(out_values + base + offsets, result_value, mask=mask)
    tl.store(out_indices + base + offsets, result_index, mask=mask)


@triton.jit
def scan_min_combine(a, b):
    return tl.minimum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def cummin_3d_kernel(out, inp, DIM: tl.constexpr, REVERSE: tl.constexpr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr,
                     ZBLOCK: tl.constexpr):
    x = tl.arange(0, XBLOCK)
    r = tl.arange(0, RBLOCK)
    z = tl.arange(0, ZBLOCK)
    index = (x[:, None, None] * RBLOCK * ZBLOCK + r[None, :, None] * ZBLOCK + z[None, None, :])
    value = tl.load(inp + index)
    result = tl.associative_scan(value, axis=DIM, combine_fn=scan_min_combine, reverse=REVERSE)
    tl.store(out + index, result)


@triton.jit
def histogram_kernel(inp, out, M: tl.constexpr, N: tl.constexpr):
    value = tl.load(inp + tl.arange(0, M))
    bins = tl.histogram(value, N)
    tl.store(out + tl.arange(0, N), bins)


@triton.jit
def scalar_reduce_store_kernel(inp, out, M, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    value = tl.load(inp + offsets, mask=offsets < M).to(tl.float32)
    tl.store(out + pid, tl.sum(value))


@triton.jit
def conditional_scalar_load_kernel(inp, out, XBLOCK: tl.constexpr):
    source = tl.arange(0, XBLOCK)
    target = source
    if tl.program_id(0) == 0:
        source = source * 1
    else:
        source = source * 2
    value = tl.load(inp + source, mask=source < XBLOCK, other=0.0)
    tl.store(out + target, value)


@triton.jit
def dot_bf16_kernel(out, lhs, rhs, B: tl.constexpr, C: tl.constexpr, D: tl.constexpr):
    bi = tl.arange(0, B)
    ci = tl.arange(0, C)
    di = tl.arange(0, D)
    a = tl.load(lhs + bi[:, None] * C + ci[None, :])
    b = tl.load(rhs + ci[:, None] * D + di[None, :])
    value = tl.dot(a, b, input_precision="hf32")
    tl.store(out + bi[:, None] * D + di[None, :], value)


@triton.jit
def fused_softmax_kernel(out, inp, input_stride, output_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row in tl.range(row_start, n_rows, row_step):
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        value = tl.load(inp + row * input_stride + cols, mask=mask, other=-float("inf"))
        numerator = tl.exp(value - tl.max(value, axis=0))
        result = numerator / tl.sum(numerator, axis=0)
        tl.store(out + row * output_stride + cols, result, mask=mask)


def route_kwargs(args: argparse.Namespace, logical_programs: int, num_warps: int = 4) -> dict:
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    return {
        "num_warps": num_warps,
        "compile_on_910_95": True,
        "compile_mode": "simd_simt" if args.route == "auto" else args.route,
        "superblock_factor": args.factor,
        "logical_program_count_hint": logical_programs,
        "physical_vector_core_count_hint": int(props["num_vectorcore"]),
        "auto_simt_scope_mode": "auto" if args.route == "auto" else "off",
        "auto_simt_scope_dump": str(args.report) if args.route == "auto" else "",
    }


def make_result(source, launch, actual, expected, kernel, category, metadata, rtol=1e-4, atol=1e-4):
    return (f"{SOURCE_ROOT}/{source}", launch, actual, expected, kernel, category, metadata, rtol, atol)


def build_case(args: argparse.Namespace):
    torch.manual_seed(17)
    name = args.case

    if name == "discrete_mask_4way_f16_8x8":
        inp = torch.ones((8, 8), dtype=torch.float16, device="npu")
        out = torch.zeros_like(inp)
        expected = torch.zeros_like(inp)
        expected[:4, :4] = 2

        def launch():
            discrete_mask_4way_kernel[(1, )](inp, out, M=6, N=6, BLOCK_M=8, BLOCK_N=8, **route_kwargs(args, 1))

        return make_result("test_discrete_mask_loadstore.py", launch, out, expected, "discrete_mask_4way_kernel",
                           "discrete_memory", {"shape": [8, 8], "valid": [6, 6], "grid": 1})

    if name == "reduce_sum_epilogue_f32_8x8x512":
        inp = torch.randn((8, 8, 512), dtype=torch.float32, device="npu")
        bias = torch.randn((64, ), dtype=torch.float32, device="npu")
        out = torch.empty_like(bias)
        expected = inp.sum(dim=-1).reshape(-1) + bias

        def launch():
            reduce_sum_epilogue_kernel[(8, )](out, inp, bias, 64, 512, XBLOCK=8, XBLOCK_SUB=2, RBLOCK=512,
                                              **route_kwargs(args, 8))

        return make_result("test_reduce_sum.py", launch, out, expected, "reduce_sum_epilogue_kernel", "reduction",
                           {"shape": [8, 8, 512], "grid": 8}, 1e-3, 1e-3)

    if name == "max_dim0_f32_64x8":
        inp = torch.randn((64, 8), dtype=torch.float32, device="npu")
        out = torch.empty((8, ), dtype=torch.float32, device="npu")
        expected = inp.max(dim=0).values

        def launch():
            max_dim0_kernel[(1, )](inp, out, M=64, N=8, MNUMEL=64, NNUMEL=32, **route_kwargs(args, 1))

        return make_result("test_max_dim0.py", launch, out, expected, "max_dim0_kernel", "reduction",
                           {"shape": [64, 8], "padded": [64, 32], "grid": 1})

    if name == "argmax_f32_37x125_dim1":
        inp = torch.randn((37, 125), dtype=torch.float32, device="npu")
        inp += torch.arange(125, device="npu", dtype=torch.float32) * 1e-5
        out = torch.empty((37, ), dtype=torch.int32, device="npu")
        expected = inp.argmax(dim=1).to(torch.int32)

        def launch():
            argmax_2d_kernel[(1, 1)](inp, out, DIM=1, M=37, N=125, MNUMEL=37, NNUMEL=125, **route_kwargs(args, 1))

        return make_result("test_argmax.py", launch, out, expected, "argmax_2d_kernel", "reduction",
                           {"shape": [37, 125], "dim": 1, "grid": [1, 1]}, 0, 0)

    if name == "gather_after_reduce_f32_16x1024":
        inp = torch.randn((16, 1024), dtype=torch.float32, device="npu")
        ids = torch.randint(0, 1024, (16, ), dtype=torch.int64, device="npu")
        out = torch.empty((16, ), dtype=torch.float32, device="npu")
        expected = inp.gather(1, ids[:, None]).squeeze(1) - inp.max(1).values

        def launch():
            gather_after_reduce_kernel[(16, )](inp, ids, out, 1024, 1024, BLOCK=128, **route_kwargs(args, 16))

        return make_result("test_indirect_scalar_load_offset.py", launch, out, expected, "gather_after_reduce_kernel",
                           "discrete_memory+reduction+scalar_memory", {"shape": [16, 1024], "BLOCK": 128, "grid": 16},
                           1e-4, 1e-4)

    if name == "view_f32_2x256x16":
        inp = torch.randn((2, 256, 16), dtype=torch.float32, device="npu")
        out = torch.empty((16, 512), dtype=torch.float32, device="npu")
        expected = inp.view(16, 512)

        def launch():
            view_3d_kernel[(1, )](out, inp, XB=2, YB=256, ZB=16, **route_kwargs(args, 1))

        return make_result("test_view.py", launch, out, expected, "view_3d_kernel", "view_layout",
                           {"input_shape": [2, 256, 16], "output_shape": [16, 512], "grid": 1})

    if name == "permute_reshape_fused_f32":
        shape = (1, 12, 4096, 8)
        a = torch.randn(shape, dtype=torch.float32, device="npu")
        b = torch.randn_like(a)
        c = torch.randn((4096, 1, 96), dtype=torch.float32, device="npu")
        out = torch.empty_like(c)
        expected = c + (a + b).permute(2, 0, 1, 3).reshape_as(c)

        def launch():
            permute_reshape_fused_kernel[(32, )](a, b, c, out, BLOCK1=128, BLOCK1_SUB=64, BLOCK2=8, S=4096, N=12, D=8,
                                                 **route_kwargs(args, 32))

        return make_result("test_permute_reshape.py", launch, out, expected, "permute_reshape_fused_kernel",
                           "view_layout+fusion", {"shape": list(shape), "BLOCK1": 128, "BLOCK1_SUB": 64, "grid": 32})

    if name == "nearest_resize_i32_1x4x64x64":
        inp = torch.randint(0, 255, (1, 4, 64, 64), dtype=torch.int32, device="npu")
        out = torch.empty((1, 4, 32, 32), dtype=torch.int32, device="npu")
        expected = inp[:, :, ::2, ::2].contiguous()

        def launch():
            nearest_resize_kernel[(4, 32, 1)](inp, out, SRC_COLS=64, DST_COLS=32, RR_H=2.0, RR_W=2.0, STRIDE_IN_H=64,
                                              STRIDE_IN_W=1, STRIDE_IN_C=4096, STRIDE_OUT_H=32, STRIDE_OUT_W=1,
                                              STRIDE_OUT_C=1024, BLOCK_SIZE=32, **route_kwargs(args, 128))

        return make_result("test_resize_performance.py", launch, out, expected, "nearest_resize_kernel",
                           "cv+discrete_memory",
                           {"input_shape": [1, 4, 64, 64], "output_shape": [1, 4, 32, 32], "grid": [4, 32, 1]}, 0, 0)

    if name == "sort_f32_256x16":
        inp = torch.randn((256, 16), dtype=torch.float32, device="npu")
        out = torch.empty_like(inp)
        expected = torch.sort(inp, dim=1).values

        def launch():
            sort_2d_kernel[(1, )](inp, out, N=256, M=16, DESCENDING=False, **route_kwargs(args, 1))

        return make_result("test_sort.py", launch, out, expected, "sort_2d_kernel", "sorting",
                           {"shape": [256, 16], "grid": 1})

    if name == "topk_f32_16x64_k4":
        inp = torch.arange(1024, dtype=torch.float32, device="npu").reshape(16, 64) / 10
        out = torch.empty((16, 4), dtype=torch.float32, device="npu")
        expected = torch.topk(inp, k=4, dim=-1).values

        def launch():
            topk_kernel[(1, )](inp, inp.stride(0), out, out.stride(0), M=16, N=64, K=4,
                               **route_kwargs(args, 1, num_warps=8))

        return make_result("test_topk.py", launch, out, expected, "topk_kernel", "sorting",
                           {"shape": [16, 64], "k": 4, "grid": 1})

    if name == "multi_input_scan_f32_2x5x10":
        values = torch.randn((2, 5, 10), dtype=torch.float32, device="npu")
        indices = torch.arange(100, dtype=torch.int32, device="npu").reshape(2, 5, 10)
        out_values = torch.empty_like(values)
        out_indices = torch.empty_like(indices)
        expected_values = values.cumsum(dim=-1)
        expected_indices = indices.cumsum(dim=-1).to(torch.int32)

        def launch():
            multi_input_scan_kernel[(10, )](values, indices, out_values, out_indices, 10, 10, BLOCK_SIZE=16,
                                            REVERSE=False, **route_kwargs(args, 10))

        return make_result("test_associative_scan_multi_input.py", launch, (out_values, out_indices),
                           (expected_values, expected_indices), "multi_input_scan_kernel", "scan",
                           {"permuted_shape": [2, 5, 10], "axis_size": 10, "grid": 10})

    if name == "cummin_f32_3x5x7_dim1":
        inp = torch.randn((3, 5, 7), dtype=torch.float32, device="npu")
        out = torch.empty_like(inp)
        expected = torch.cummin(inp, dim=1).values

        def launch():
            cummin_3d_kernel[(1, )](out, inp, DIM=1, REVERSE=False, XBLOCK=3, RBLOCK=5, ZBLOCK=7,
                                    **route_kwargs(args, 1))

        return make_result("test_cummin.py", launch, out, expected, "cummin_3d_kernel", "scan",
                           {"shape": [3, 5, 7], "dim": 1, "grid": 1})

    if name == "histogram_i32_2048_bins2":
        inp = torch.randint(0, 2, (2048, ), dtype=torch.int32, device="npu")
        out = torch.empty((2, ), dtype=torch.int32, device="npu")
        expected = torch.bincount(inp.to(torch.int64), minlength=2).to(torch.int32)

        def launch():
            histogram_kernel[(1, )](inp, out, M=2048, N=2, **route_kwargs(args, 1))

        return make_result("test_histogram.py", launch, out, expected, "histogram_kernel", "histogram",
                           {"M": 2048, "bins": 2, "grid": 1}, 0, 0)

    if name == "scalar_reduce_store_f32_n16":
        inp = torch.ones((16, ), dtype=torch.float32, device="npu")
        out = torch.empty((4, ), dtype=torch.float32, device="npu")
        expected = torch.full((4, ), 4.0, dtype=torch.float32, device="npu")

        def launch():
            scalar_reduce_store_kernel[(4, )](inp, out, 16, BLOCK_SIZE=4, **route_kwargs(args, 4))

        return make_result("test_store_scalar.py", launch, out, expected, "scalar_reduce_store_kernel",
                           "scalar_memory+reduction", {"shape": [16], "BLOCK_SIZE": 4, "grid": 4})

    if name == "conditional_load_f32_n32":
        inp = torch.randn((32, ), dtype=torch.float32, device="npu")
        out = torch.empty_like(inp)
        expected = inp.clone()

        def launch():
            conditional_scalar_load_kernel[(1, )](inp, out, XBLOCK=32, **route_kwargs(args, 1))

        return make_result("test_if_load.py", launch, out, expected, "conditional_scalar_load_kernel",
                           "scalar_control+memory", {"shape": [32], "grid": 1})

    if name == "dot_bf16_16x32x16":
        lhs = torch.randn((16, 32), dtype=torch.bfloat16, device="npu")
        rhs = torch.randn((32, 16), dtype=torch.bfloat16, device="npu")
        out = torch.empty((16, 16), dtype=torch.float32, device="npu")
        expected = lhs.float() @ rhs.float()

        def launch():
            dot_bf16_kernel[(1, 1, 1)](out, lhs, rhs, B=16, C=32, D=16, **route_kwargs(args, 1))

        return make_result("test_dot.py", launch, out, expected, "dot_bf16_kernel", "dot",
                           {"B": 16, "C": 32, "D": 16, "grid": [1, 1, 1]}, 2e-2, 2e-2)

    if name == "fused_softmax_f32_128x257":
        inp = torch.randn((128, 257), dtype=torch.float32, device="npu")
        out = torch.empty_like(inp)
        expected = torch.softmax(inp, dim=1)

        def launch():
            fused_softmax_kernel[(32, 1, 1)](out, inp, inp.stride(0), out.stride(0), 128, 257, BLOCK_SIZE=512,
                                             **route_kwargs(args, 32))

        return make_result("test_02_fused_softmax.py", launch, out, expected, "fused_softmax_kernel",
                           "fusion+reduction", {"shape": [128, 257], "BLOCK_SIZE": 512, "grid": [32, 1, 1]}, 2e-4, 2e-4)

    raise ValueError(name)


CASES = (
    "discrete_mask_4way_f16_8x8",
    "reduce_sum_epilogue_f32_8x8x512",
    "max_dim0_f32_64x8",
    "argmax_f32_37x125_dim1",
    "gather_after_reduce_f32_16x1024",
    "view_f32_2x256x16",
    "permute_reshape_fused_f32",
    "nearest_resize_i32_1x4x64x64",
    "sort_f32_256x16",
    "topk_f32_16x64_k4",
    "multi_input_scan_f32_2x5x10",
    "cummin_f32_3x5x7_dim1",
    "histogram_i32_2048_bins2",
    "scalar_reduce_store_f32_n16",
    "conditional_load_f32_n32",
    "dot_bf16_16x32x16",
    "fused_softmax_f32_128x257",
)


def assert_close(actual, expected, rtol, atol):
    if isinstance(actual, tuple):
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=rtol, atol=atol)
        return max(float((a.float() - e.float()).abs().max()) for a, e in zip(actual, expected))
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    return float((actual.float() - expected.float()).abs().max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=CASES)
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), default="auto")
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()

    (source, launch, actual, expected, kernel_name, category, metadata, rtol, atol) = build_case(args)
    launch()
    torch.npu.synchronize()
    max_abs = assert_close(actual, expected, rtol, atol)
    if args.compile_only:
        print(
            json.dumps(
                {
                    "case": args.case, "source_pytest": source, "category": category, "correct": True, "max_abs":
                    max_abs, "compile_only": True, **metadata
                }, indent=2))
        return

    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()
    samples = []
    for _ in range(args.repeats):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(args.batch):
            launch()
        end.record()
        torch.npu.synchronize()
        samples.append(float(begin.elapsed_time(end)) * 1000.0 / args.batch)
    samples.sort()
    profiler = profile_kernel(launch, args.profile_dir, kernel_name) if args.profile_dir else None
    print(
        json.dumps(
            {
                "case": args.case,
                "source_pytest": source,
                "category": category,
                "route_requested": args.route,
                "compile_mode": "simd_simt" if args.route == "auto" else args.route,
                "factor": args.factor,
                "correct": True,
                "max_abs": max_abs,
                "elapsed_us": {
                    "median": statistics.median(samples), "minimum": samples[0], "maximum": samples[-1], "samples":
                    samples
                },
                "measurement_target": "profiler.average_us",
                "report": str(args.report),
                "profiler": profiler,
                **metadata,
            }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
