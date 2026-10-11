#!/usr/bin/env python3
"""Additional cost-model benchmarks sourced from Ascend ``pytest_ut``.

The kernels and launch geometry below mirror the cited unit tests.  This file
only adds the common correctness, route-report and profiler interface used by
the generalization matrix.
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
import triton.language.extra.cann.libdevice as libdevice
from triton.runtime import driver

from profile_utils import profile_kernel

SOURCE_ROOT = "third_party/ascend/unittest/pytest_ut"


@triton.jit
def exp_kernel(in_ptr, out_ptr, n, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    for loop in range((XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB):
        index = offset + loop * XBLOCK_SUB + base
        value = tl.load(in_ptr + index, mask=index < n)
        tl.store(out_ptr + index, tl.exp(value), mask=index < n)


@triton.jit
def log_kernel(in_ptr, out_ptr, N: tl.constexpr, NUMEL: tl.constexpr):
    index = tl.arange(0, NUMEL)
    value = tl.load(in_ptr + index, mask=index < N)
    tl.store(out_ptr + index, tl.math.log(value), mask=index < N)


@triton.jit
def relu_kernel(in0, in1, out, n, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    for sub in range(0, XBLOCK, XBLOCK_SUB):
        index = offset + sub + tl.arange(0, XBLOCK_SUB)
        mask = index < n
        x0 = tl.load(in0 + index, mask=mask)
        x1 = tl.load(in1 + index, mask=mask)
        tl.store(out + index, x0 + libdevice.relu(x1), mask=mask)


@triton.jit
def where_kernel(in0, in1, out, n, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    for sub in range(0, XBLOCK, XBLOCK_SUB):
        index = offset + sub + tl.arange(0, XBLOCK_SUB)
        mask = index < n
        x0 = tl.load(in0 + index, mask=mask)
        x1 = tl.load(in1 + index, mask=mask)
        tl.store(out + index, tl.where(x0 < x1, x0, x1), mask=mask)


@triton.jit
def broadcast_kernel(in_ptr, out_ptr, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    l = tl.arange(0, L)
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    in_index = tl.arange(0, 1)[:, None, None] * N * M + m[None, :, None] * N + n[None, None, :]
    out_index = l[:, None, None] * N * M + m[None, :, None] * N + n[None, None, :]
    value = tl.load(in_ptr + in_index)
    old = tl.load(out_ptr + out_index)
    result, _ = tl.broadcast(value, old)
    tl.store(out_ptr + out_index, result)


@triton.jit
def max_dim1_kernel(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    m = tl.arange(0, MNUMEL)
    n = tl.arange(0, NNUMEL)
    mask = (m[:, None] < M) & (n[None, :] < N)
    value = tl.load(in_ptr + m[:, None] * N + n[None, :], mask=mask, other=-float("inf"))
    tl.store(out_ptr + m, tl.max(value, 1), mask=m < M)


@triton.jit
def mean_dim1_kernel(in_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr, MNUMEL: tl.constexpr, NNUMEL: tl.constexpr):
    m = tl.arange(0, MNUMEL)
    n = tl.arange(0, NNUMEL)
    mask = (m[:, None] < M) & (n[None, :] < N)
    value = tl.load(in_ptr + m[:, None] * N + n[None, :], mask=mask, other=0.0)
    result = tl.sum(value.to(tl.float32), 1) / N
    tl.store(out_ptr + m, result, mask=m < M)


@triton.jit
def minimum(a, b):
    return tl.minimum(a, b, tl.PropagateNan.ALL)


@triton.jit
def mul_reduce_kernel(in0, in1, out, L: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      Z: tl.constexpr):
    l = tl.arange(0, L)
    m = tl.arange(0, M)
    n = tl.arange(0, N)
    k = tl.arange(0, K)
    z = tl.arange(0, Z)
    index = (l[:, None, None, None, None] * Z * K * N * M + m[None, :, None, None, None] * Z * K * N +
             n[None, None, :, None, None] * Z * K + k[None, None, None, :, None] * Z + z[None, None, None, None, :])
    value = tl.load(in0 + index) * tl.load(in1 + index)
    result = tl.reduce(value, 4, minimum, keep_dims=True)
    out_index = (l[:, None, None, None, None] * K * N * M + m[None, :, None, None, None] * K * N +
                 n[None, None, :, None, None] * K + k[None, None, None, :, None])
    tl.store(out + out_index, result)


@triton.jit
def cumprod_kernel(out_ptr, in_ptr, XBLOCK: tl.constexpr):
    index = tl.arange(0, XBLOCK)
    value = tl.load(in_ptr + index)
    tl.store(out_ptr + index, tl.cumprod(value, axis=0, reverse=False))


@triton.jit
def cummax_combine(a, b):
    return tl.maximum(a, b, propagate_nan=tl.PropagateNan.ALL)


@triton.jit
def cummax_kernel(out_ptr, in_ptr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr):
    x = tl.arange(0, XBLOCK)
    r = tl.arange(0, RBLOCK)
    index = x[:, None] * RBLOCK + r[None, :]
    value = tl.load(in_ptr + index)
    result = tl.associative_scan(value, axis=1, combine_fn=cummax_combine, reverse=False)
    tl.store(out_ptr + index, result)


@triton.jit
def atomic_max_kernel(in_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)
    yindex = tl.arange(0, BLOCK_SIZE)
    value = tl.load(in_ptr + xindex)
    tl.atomic_max(out_ptr + yindex, value, mask=xindex < n)


def route_kwargs(args: argparse.Namespace, grid: int, num_warps: int) -> dict:
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    return {
        "num_warps": num_warps,
        "compile_on_910_95": True,
        "compile_mode": "simd_simt" if args.route == "auto" else args.route,
        "superblock_factor": args.factor,
        "logical_program_count_hint": grid,
        "physical_vector_core_count_hint": vector_cores,
        "auto_simt_scope_mode": "auto" if args.route == "auto" else "off",
        "auto_simt_scope_dump": str(args.report) if args.route == "auto" else "",
    }


def build_case(args: argparse.Namespace):
    torch.manual_seed(0)
    case = args.case

    if case == "exp_f32_2x4096x8":
        source = f"{SOURCE_ROOT}/test_exp.py"
        shape, grid, xblock, sub = (2, 4096, 8), 2, 32768, 1024
        x = torch.randn(shape, dtype=torch.float32, device="npu")
        out = torch.empty_like(x)
        expected = torch.exp(x)

        def launch():
            exp_kernel[(grid, )](x, out, x.numel(), XBLOCK=xblock, XBLOCK_SUB=sub, **route_kwargs(args, grid, 4))

        return source, launch, out, expected, "exp_kernel", {
            "shape": shape, "grid": grid, "XBLOCK": xblock, "XBLOCK_SUB": sub
        }, 1e-4, 1e-4

    if case == "log_f32_n781":
        source = f"{SOURCE_ROOT}/test_log.py"
        n, block = 781, 1024
        x = torch.rand((n, ), dtype=torch.float32, device="npu") + 0.1
        out = torch.empty_like(x)
        expected = torch.log(x)

        def launch():
            log_kernel[(1, )](x, out, N=n, NUMEL=block, **route_kwargs(args, 1, 4))

        return source, launch, out, expected, "log_kernel", {"N": n, "NUMEL": block, "grid": 1}, 1e-4, 1e-4

    if case == "relu_f32_2x4096x8":
        source = f"{SOURCE_ROOT}/test_relu.py"
        shape, grid, xblock, sub = (2, 4096, 8), 2, 32768, 1024
        x0 = torch.randn(shape, dtype=torch.float32, device="npu")
        x1 = torch.randn(shape, dtype=torch.float32, device="npu")
        out = torch.empty_like(x0)
        expected = x0 + torch.relu(x1)

        def launch():
            relu_kernel[(grid, )](x0, x1, out, x0.numel(), XBLOCK=xblock, XBLOCK_SUB=sub, **route_kwargs(args, grid, 4))

        return source, launch, out, expected, "relu_kernel", {
            "shape": shape, "grid": grid, "XBLOCK": xblock, "XBLOCK_SUB": sub
        }, 1e-4, 1e-4

    if case == "where_f32_2x1024x8":
        source = f"{SOURCE_ROOT}/test_where_mask.py"
        shape, grid, xblock, sub = (2, 1024, 8), 2, 8192, 1024
        x0 = torch.randn(shape, dtype=torch.float32, device="npu")
        x1 = torch.randn(shape, dtype=torch.float32, device="npu")
        out = torch.empty_like(x0)
        expected = torch.where(x0 < x1, x0, x1)

        def launch():
            where_kernel[(grid, )](x0, x1, out, x0.numel(), XBLOCK=xblock, XBLOCK_SUB=sub,
                                   **route_kwargs(args, grid, 4))

        return source, launch, out, expected, "where_kernel", {
            "shape": shape, "grid": grid, "XBLOCK": xblock, "XBLOCK_SUB": sub
        }, 1e-5, 1e-5

    if case == "broadcast_bf16_2x4x8":
        source = f"{SOURCE_ROOT}/test_broadcast.py"
        shape = (2, 4, 8)
        x = torch.randn((1, 4, 8), dtype=torch.bfloat16, device="npu")
        out = torch.zeros(shape, dtype=torch.bfloat16, device="npu")
        expected = x.repeat(2, 1, 1)

        def launch():
            broadcast_kernel[(1, )](x, out, L=2, M=4, N=8, **route_kwargs(args, 1, 4))

        return source, launch, out, expected, "broadcast_kernel", {"shape": shape, "grid": 1}, 1e-3, 1e-3

    if case == "max_dim1_f32_m64_n8":
        source = f"{SOURCE_ROOT}/test_max_dim1.py"
        m, n, mblock, nblock = 64, 8, 64, 32
        x = torch.randn((m, n), dtype=torch.float32, device="npu")
        out = torch.empty((m, ), dtype=torch.float32, device="npu")
        expected = torch.max(x, dim=1).values

        def launch():
            max_dim1_kernel[(1, )](x, out, M=m, N=n, MNUMEL=mblock, NNUMEL=nblock, **route_kwargs(args, 1, 4))

        return source, launch, out, expected, "max_dim1_kernel", {
            "M": m, "N": n, "MNUMEL": mblock, "NNUMEL": nblock, "grid": 1
        }, 1e-5, 1e-5

    if case == "mean_dim1_f32_m57_n3":
        source = f"{SOURCE_ROOT}/test_mean_dim1.py"
        m, n, mblock, nblock = 57, 3, 64, 16
        x = torch.randn((m, n), dtype=torch.float32, device="npu")
        out = torch.empty((m, ), dtype=torch.float32, device="npu")
        expected = torch.mean(x, dim=1)

        def launch():
            mean_dim1_kernel[(1, )](x, out, M=m, N=n, MNUMEL=mblock, NNUMEL=nblock, **route_kwargs(args, 1, 4))

        return source, launch, out, expected, "mean_dim1_kernel", {
            "M": m, "N": n, "MNUMEL": mblock, "NNUMEL": nblock, "grid": 1
        }, 1e-5, 1e-5

    if case == "mul_reduce_f16_16x1x1x1x1":
        source = f"{SOURCE_ROOT}/test_mul_reduce.py"
        shape = (16, 1, 1, 1, 1)
        x0 = torch.randn(shape, dtype=torch.float16, device="npu")
        x1 = torch.randn(shape, dtype=torch.float16, device="npu")
        out = torch.empty(shape, dtype=torch.float16, device="npu")
        expected = x0 * x1

        def launch():
            mul_reduce_kernel[(1, )](x0, x1, out, L=16, M=1, N=1, K=1, Z=1, **route_kwargs(args, 1, 4))

        return source, launch, out, expected, "mul_reduce_kernel", {"shape": shape, "grid": 1}, 1e-3, 1e-3

    if case == "cumprod_f32_n1024":
        source = f"{SOURCE_ROOT}/test_cumprod.py"
        n = 1024
        x = (0.995 + 0.01 * torch.rand((n, ), dtype=torch.float32, device="npu"))
        out = torch.empty_like(x)
        expected = torch.cumprod(x, dim=0)

        def launch():
            cumprod_kernel[(1, )](out, x, XBLOCK=n, **route_kwargs(args, 1, 4))

        return source, launch, out, expected, "cumprod_kernel", {"N": n, "grid": 1}, 1e-3, 1e-3

    if case == "cummax_f32_16x32_dim1":
        source = f"{SOURCE_ROOT}/test_cummax.py"
        shape = (16, 32)
        x = torch.randn(shape, dtype=torch.float32, device="npu")
        out = torch.empty_like(x)
        expected = torch.cummax(x, dim=1).values

        def launch():
            cummax_kernel[(1, )](out, x, XBLOCK=16, RBLOCK=32, **route_kwargs(args, 1, 4))

        return source, launch, out, expected, "cummax_kernel", {"shape": shape, "axis": 1, "grid": 1}, 1e-5, 1e-5

    if case == "atomic_max_i32_32x32_c2":
        source = f"{SOURCE_ROOT}/test_atomic_max.py"
        rows, cols, grid = 32, 32, 2
        block = rows * cols // grid
        x = torch.arange(rows * cols, dtype=torch.int32, device="npu").reshape(rows, cols)
        out = torch.full((rows // grid, cols), -1, dtype=torch.int32, device="npu")
        expected = torch.maximum(x[:rows // grid], x[rows // grid:])

        def launch():
            out.fill_(-1)
            atomic_max_kernel[(grid, )](x, out, x.numel(), BLOCK_SIZE=block, **route_kwargs(args, grid, 4))

        return source, launch, out, expected, "atomic_max_kernel", {
            "shape": (rows, cols), "grid": grid, "BLOCK_SIZE": block
        }, 0.0, 0.0

    raise AssertionError(case)


CASES = (
    "exp_f32_2x4096x8",
    "log_f32_n781",
    "relu_f32_2x4096x8",
    "where_f32_2x1024x8",
    "broadcast_bf16_2x4x8",
    "max_dim1_f32_m64_n8",
    "mean_dim1_f32_m57_n3",
    "mul_reduce_f16_16x1x1x1x1",
    "cumprod_f32_n1024",
    "cummax_f32_16x32_dim1",
    "atomic_max_i32_32x32_c2",
)


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

    source, launch, actual, expected, kernel_name, metadata, rtol, atol = build_case(args)
    launch()
    torch.npu.synchronize()
    max_abs = float((actual.float() - expected.float()).abs().max())
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    if args.compile_only:
        print(
            json.dumps(
                {
                    "case": args.case, "source_pytest": source, "correct": True, "max_abs": max_abs, "compile_only":
                    True, "report": str(args.report), **metadata
                }, indent=2, sort_keys=True))
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
                "route_requested": args.route,
                "compile_mode": "simd_simt" if args.route == "auto" else args.route,
                "factor": args.factor,
                "correct": True,
                "max_abs": max_abs,
                "timing_batch": args.batch,
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
