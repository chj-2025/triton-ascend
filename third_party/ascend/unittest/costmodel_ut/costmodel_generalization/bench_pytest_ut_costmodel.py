#!/usr/bin/env python3
"""Cost-model benchmarks copied from Ascend pytest_ut operator tests.

Each case keeps the source test's kernel geometry and correctness semantics,
while adding the common route-report, profiler, and fresh-process interface
used by the generalization matrix.  Source provenance is emitted in the
benchmark JSON and is never consumed as a cost-model feature.
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
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


@triton.jit
def permute_2d_kernel(output_ptr, input_ptr, X: tl.constexpr, Y: tl.constexpr):
    index = tl.arange(0, X * Y)
    value = tl.load(input_ptr + index)
    value = value.reshape(X, Y).trans().reshape(X * Y)
    tl.store(output_ptr + index, value)


@triton.jit
def sigmoid_kernel(input0, input1, output, xnumel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + tl.arange(0, XBLOCK_SUB)
        mask = xindex < xnumel
        x0 = tl.load(input0 + xindex, mask=mask)
        x1 = tl.load(input1 + xindex, mask=mask)
        tl.store(output + xindex, x0 + tl.sigmoid(x1), mask=mask)


@triton.jit
def sum_low_kernel(
    input0,
    input1,
    input2,
    output,
    xnumel,
    ynumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    x0 = xindex[:, None]
    rbase = tl.arange(0, RBLOCK)
    accumulator = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, ynumel, RBLOCK):
        rindex = roffset + rbase
        r1 = rindex[None, :]
        value = tl.load(input0 + r1 + ynumel * x0)
        value += tl.load(input1 + r1 + ynumel * x0)
        value += tl.load(input2 + r1 + ynumel * x0)
        accumulator += value
    tl.store(output + xindex, tl.sum(accumulator, axis=1))


@triton.jit
def sum_high_kernel(
    input0,
    input1,
    input2,
    output,
    rnumel,
    xnumel,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xbase = tl.arange(0, XBLOCK_SUB)
    rbase = tl.arange(0, RBLOCK)
    for xoffset_sub in range(0, XBLOCK, XBLOCK_SUB):
        xindex = xoffset + xoffset_sub + xbase
        x0 = xindex[None, :]
        accumulator = tl.full([RBLOCK, XBLOCK_SUB], 0, tl.float32)
        for roffset in range(0, rnumel, RBLOCK):
            rindex = roffset + rbase
            r1 = rindex[:, None]
            value = tl.load(input0 + xnumel * r1 + x0)
            value += tl.load(input1 + xnumel * r1 + x0)
            value += tl.load(input2 + xnumel * r1 + x0)
            accumulator += value
        tl.store(output + xindex, tl.sum(accumulator, axis=0))


@triton.jit
def rms_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    variance_vector = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        variance_vector += x * x
    variance = tl.sum(variance_vector, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        weight = tl.load(weight_ptr + cols, mask=mask).to(tl.float32)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + cols, (x * rstd * weight).to(tl.float16), mask=mask)


@triton.jit
def layer_norm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    mean_ptr,
    rstd_ptr,
    stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr += row * stride
    y_ptr += row * stride
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / n_cols
    centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(centered * centered, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(variance + eps)
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    weight = tl.load(weight_ptr + cols, mask=mask)
    bias = tl.load(bias_ptr + cols, mask=mask)
    tl.store(y_ptr + cols, centered * rstd * weight + bias, mask=mask)


@triton.jit
def insert_slice_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    SLICE_OFFSET: tl.constexpr,
    SLICE_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.load(output_ptr + offsets, mask=mask)
    x_sub = extension.extract_slice(x, [SLICE_OFFSET], [SLICE_SIZE], [1])
    y_sub = extension.extract_slice(y, [SLICE_OFFSET], [SLICE_SIZE], [1])
    output_sub = x_sub + y_sub
    output = extension.insert_slice(output, output_sub, [SLICE_OFFSET], [SLICE_SIZE], [1])
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def atomic_add_kernel(input_ptr, accumulator_ptr, old_value_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    xoffset = tl.program_id(0) * BLOCK_SIZE
    xindex = xoffset + tl.arange(0, BLOCK_SIZE)
    out_index = tl.arange(0, BLOCK_SIZE)
    mask = xindex < n_elements
    value = tl.load(input_ptr + xindex, mask=mask)
    old = tl.atomic_add(accumulator_ptr + out_index, value, mask=mask)
    tl.store(old_value_ptr + out_index, old, mask=mask)


def route_kwargs(args: argparse.Namespace, grid: int, num_warps: int) -> dict:
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    return {
        "num_warps": num_warps,
        # A5 is a 910_95-class target.  Keep this explicit because importing
        # extension intrinsics can query the lazy ACL target predicate before
        # the first NPU context has finished initializing.
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
    if case == "vector_add_n98432":
        source = f"{SOURCE_ROOT}/test_01_vector_add.py"
        n = 98432
        x = torch.rand(n, device="npu")
        y = torch.rand(n, device="npu")
        output = torch.empty_like(x)
        grid = triton.cdiv(n, 1024)
        expected = x + y

        def launch():
            vector_add_kernel[(grid, )](x, y, output, n, BLOCK_SIZE=1024, **route_kwargs(args, grid, 4))

        return source, launch, output, expected, "vector_add_kernel", {"n": n, "grid": grid}, 1e-5, 1e-5

    if case == "permute_x64_y32":
        source = f"{SOURCE_ROOT}/test_2d_permute.py"
        x_size, y_size = 64, 32
        x = torch.randn((x_size, y_size), device="npu")
        output = torch.empty((y_size, x_size), dtype=x.dtype, device="npu")
        expected = x.t().contiguous()

        def launch():
            permute_2d_kernel[(1, )](output, x, X=x_size, Y=y_size, **route_kwargs(args, 1, 4))

        return source, launch, output, expected, "permute_2d_kernel", {"X": x_size, "Y": y_size, "grid": 1}, 1e-3, 1e-3

    if case == "sigmoid_f32_2x4096x8":
        source = f"{SOURCE_ROOT}/test_sigmoid.py"
        shape = (2, 4096, 8)
        x0 = torch.randn(shape, dtype=torch.float32, device="npu")
        x1 = torch.randn(shape, dtype=torch.float32, device="npu")
        output = torch.empty_like(x0)
        expected = x0 + torch.sigmoid(x1)
        grid, xblock, sub = 2, 32768, 1024

        def launch():
            sigmoid_kernel[(grid, )](x0, x1, output, x0.numel(), XBLOCK=xblock, XBLOCK_SUB=sub,
                                     **route_kwargs(args, grid, 4))

        return source, launch, output, expected, "sigmoid_kernel", {
            "shape": shape, "grid": grid, "XBLOCK": xblock, "XBLOCK_SUB": sub
        }, 1e-4, 1e-4

    if case in {"sum_low_f32_64x8192", "sum_high_f32_64x8192"}:
        source = f"{SOURCE_ROOT}/test_sum.py"
        shape = (64, 8192)
        a = torch.randn(shape, dtype=torch.float32, device="npu")
        b = torch.randn(shape, dtype=torch.float32, device="npu")
        c = torch.randn(shape, dtype=torch.float32, device="npu")
        if case.startswith("sum_low"):
            output = torch.empty((shape[0], ), dtype=torch.float32, device="npu")
            expected = (a + b + c).sum(dim=1)
            grid = 1

            def launch():
                sum_low_kernel[(grid, )](a, b, c, output, shape[0], shape[1], XBLOCK=64, RBLOCK=32,
                                         **route_kwargs(args, grid, 4))

            metadata = {"shape": shape, "grid": grid, "XBLOCK": 64, "RBLOCK": 32}
            kernel_name = "sum_low_kernel"
        else:
            output = torch.empty((shape[1], ), dtype=torch.float32, device="npu")
            expected = (a + b + c).sum(dim=0)
            grid = 32

            def launch():
                sum_high_kernel[(grid, )](a, b, c, output, shape[0], shape[1], XBLOCK=256, XBLOCK_SUB=64, RBLOCK=64,
                                          **route_kwargs(args, grid, 4))

            metadata = {"shape": shape, "grid": grid, "XBLOCK": 256, "XBLOCK_SUB": 64, "RBLOCK": 64}
            kernel_name = "sum_high_kernel"
        return source, launch, output, expected, kernel_name, metadata, 1e-3, 1e-3

    if case == "rms_norm_f16_m128_n128":
        source = f"{SOURCE_ROOT}/test_rms_norm.py"
        rows, cols, block = args.rms_rows, args.rms_cols, args.rms_block_size
        if block < cols or block & (block - 1):
            raise ValueError("RMS BLOCK_SIZE must be a power of two >= n_cols")
        x = torch.randn((rows, cols), dtype=torch.float16, device="npu")
        weight = torch.randn((cols, ), dtype=torch.float16, device="npu")
        output = torch.empty_like(x, dtype=torch.float16)
        x32 = x.float()
        expected = (x32 / torch.sqrt(x32.square().mean(-1, keepdim=True) + 1e-5) * weight.float()).half()

        def launch():
            rms_norm_kernel[(rows, )](x, output, weight, x.stride(0), cols, 1e-5, BLOCK_SIZE=block,
                                      **route_kwargs(args, rows, 8))

        return source, launch, output, expected, "rms_norm_kernel", {
            "M": rows, "N": cols, "BLOCK_SIZE": block, "grid": rows
        }, 1e-3, 1e-3

    if case == "layer_norm_f16_m128_n128":
        source = f"{SOURCE_ROOT}/test_layernorm.py"
        rows, cols, block = 128, 128, 128
        x = -2.3 + 0.5 * torch.randn((rows, cols), dtype=torch.float16, device="npu")
        weight = torch.rand((cols, ), dtype=torch.float16, device="npu")
        bias = torch.rand((cols, ), dtype=torch.float16, device="npu")
        output = torch.empty_like(x)
        mean = torch.empty((rows, ), dtype=torch.float32, device="npu")
        rstd = torch.empty((rows, ), dtype=torch.float32, device="npu")
        expected = torch.nn.functional.layer_norm(x, (cols, ), weight, bias, 1e-5)

        def launch():
            layer_norm_kernel[(rows, )](x, output, weight, bias, mean, rstd, x.stride(0), cols, 1e-5, BLOCK_SIZE=block,
                                        **route_kwargs(args, rows, 1))

        return source, launch, output, expected, "layer_norm_kernel", {
            "M": rows, "N": cols, "BLOCK_SIZE": block, "grid": rows
        }, 0.0, 1e-2

    if case == "insert_slice_n1024_s32":
        source = f"{SOURCE_ROOT}/test_insert_slice.py"
        n, slice_size = 1024, 32
        x = torch.rand(n, device="npu")
        y = torch.rand(n, device="npu")
        output = torch.zeros_like(x)
        expected = torch.zeros_like(x)
        expected[:slice_size] = x[:slice_size] + y[:slice_size]

        def launch():
            output.zero_()
            insert_slice_kernel[(1, )](x, y, output, n, BLOCK_SIZE=1024, SLICE_OFFSET=0, SLICE_SIZE=slice_size,
                                       **route_kwargs(args, 1, 4))

        return source, launch, output, expected, "insert_slice_kernel", {
            "n": n, "slice_offset": 0, "slice_size": slice_size, "grid": 1
        }, 1e-5, 1e-5

    if case == "atomic_add_f32_32x32_c2":
        source = f"{SOURCE_ROOT}/test_atomic_add.py"
        rows, cols, grid = 32, 32, 2
        block = rows * cols // grid
        x = torch.full((rows, cols), 3.0, dtype=torch.float32, device="npu")
        accumulator = torch.full((rows // grid, cols), 2.0, dtype=torch.float32, device="npu")
        old_values = torch.empty_like(accumulator)
        expected = torch.full_like(accumulator, 8.0)

        def launch():
            accumulator.fill_(2.0)
            atomic_add_kernel[(grid, )](x, accumulator, old_values, x.numel(), BLOCK_SIZE=block,
                                        **route_kwargs(args, grid, 4))

        return source, launch, accumulator, expected, "atomic_add_kernel", {
            "shape": (rows, cols), "grid": grid, "BLOCK_SIZE": block
        }, 1e-5, 1e-5

    raise AssertionError(case)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", required=True, choices=(
            "vector_add_n98432",
            "permute_x64_y32",
            "sigmoid_f32_2x4096x8",
            "sum_low_f32_64x8192",
            "sum_high_f32_64x8192",
            "rms_norm_f16_m128_n128",
            "layer_norm_f16_m128_n128",
            "insert_slice_n1024_s32",
            "atomic_add_f32_32x32_c2",
        ))
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), default="auto")
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--rms-rows", type=int, default=128)
    parser.add_argument("--rms-cols", type=int, default=128)
    parser.add_argument("--rms-block-size", type=int, default=8192)
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
