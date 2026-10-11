#!/usr/bin/env python3
"""Cost-model holdouts copied from Ascend pytest_ut kernels.

The launch geometry is deliberately kept identical to the cited tests.  This
wrapper adds route forcing, correctness checks and profiler measurement only.
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
from triton.runtime import driver

from profile_utils import profile_kernel

SOURCE_ROOT = "third_party/ascend/unittest/pytest_ut"


@triton.jit
def binary_loop_kernel(a, b, out, n, OP: tl.constexpr, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base = tl.arange(0, XBLOCK_SUB)
    for loop in range((XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB):
        index = offset + loop * XBLOCK_SUB + base
        mask = index < n
        x = tl.load(a + index, mask=mask)
        y = tl.load(b + index, mask=mask)
        if OP == 0:
            z = x + y
        elif OP == 1:
            z = x * y
        elif OP == 2:
            z = x / y
        else:
            z = tl.maximum(x, y)
        tl.store(out + index, z, mask=mask)


@triton.jit
def unary_kernel(inp, out, N: tl.constexpr, NUMEL: tl.constexpr, OP: tl.constexpr):
    index = tl.arange(0, NUMEL)
    x = tl.load(inp + index, mask=index < N)
    if OP == 0:
        y = tl.math.sqrt(x)
    else:
        y = tl.cos(x)
    tl.store(out + index, y, mask=index < N)


@triton.jit
def f2i32_kernel(inp, out, N: tl.constexpr, NUMEL: tl.constexpr):
    index = tl.arange(0, NUMEL)
    mask = index < N
    x = tl.load(inp + index, mask=mask)
    tl.store(out + index, tl.cast(x, tl.int32), mask=mask)


@triton.jit
def strided_compaction_kernel(inp, out, in_numel, out_numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr,
                              STRIDE: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    for sub in range(0, XBLOCK, XBLOCK_SUB):
        index = offset + sub + tl.arange(0, XBLOCK_SUB)
        src_index = index * STRIDE
        value = tl.load(inp + src_index, mask=src_index < in_numel)
        tl.store(out + index, value, mask=index < out_numel)


@triton.jit
def strided_scatter_kernel(inp, out, in_numel, out_numel, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr,
                           STRIDE: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    for sub in range(0, XBLOCK, XBLOCK_SUB):
        index = offset + sub + tl.arange(0, XBLOCK_SUB)
        dst_index = index * STRIDE
        value = tl.load(inp + index, mask=index < in_numel)
        tl.store(out + dst_index, value, mask=dst_index < out_numel)


@triton.jit
def block_ptr_strided_kernel(inp, out, src_shape_m, src_shape_n, src_stride_m, src_stride_n, BLOCK_M: tl.constexpr,
                             BLOCK_N: tl.constexpr):
    in_block = tl.make_block_ptr(base=inp, shape=(src_shape_m, src_shape_n), strides=(src_stride_m, src_stride_n),
                                 offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    out_block = tl.make_block_ptr(base=out, shape=(BLOCK_M, BLOCK_N), strides=(BLOCK_N, 1), offsets=(0, 0),
                                  block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(out_block, tl.load(in_block))


@triton.jit
def scan_max_combine(a, b):
    return tl.maximum(a, b)


@triton.jit
def scan3d_kernel(out, inp, DIM: tl.constexpr, REVERSE: tl.constexpr, XBLOCK: tl.constexpr, RBLOCK: tl.constexpr,
                  ZBLOCK: tl.constexpr):
    x = tl.arange(0, XBLOCK)
    r = tl.arange(0, RBLOCK)
    z = tl.arange(0, ZBLOCK)
    index = (x[:, None, None] * RBLOCK * ZBLOCK + r[None, :, None] * ZBLOCK + z[None, None, :])
    value = tl.load(inp + index)
    result = tl.associative_scan(value, axis=DIM, reverse=REVERSE, combine_fn=scan_max_combine)
    tl.store(out + index, result)


@triton.jit
def chunk_cumsum_kernel(inp, out, T, H: tl.constexpr, BT: tl.constexpr):
    tile, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    base = batch * T
    src = tl.make_block_ptr(inp + base * H + head, (T, ), (H, ), (tile * BT, ), (BT, ), (0, ))
    dst = tl.make_block_ptr(out + base * H + head, (T, ), (H, ), (tile * BT, ), (BT, ), (0, ))
    value = tl.load(src, boundary_check=(0, )).to(tl.float32)
    tl.store(dst, tl.cumsum(value, axis=0), boundary_check=(0, ))


def route_kwargs(args: argparse.Namespace, logical_programs: int) -> dict:
    props = driver.active.utils.get_device_properties(torch.npu.current_device())
    return {
        "num_warps": 4,
        "compile_on_910_95": True,
        "compile_mode": "simd_simt" if args.route == "auto" else args.route,
        "superblock_factor": args.factor,
        "logical_program_count_hint": logical_programs,
        "physical_vector_core_count_hint": int(props["num_vectorcore"]),
        "auto_simt_scope_mode": "auto" if args.route == "auto" else "off",
        "auto_simt_scope_dump": str(args.report) if args.route == "auto" else "",
    }


def prefix_max(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.cummax(x, dim=dim).values


def build_case(args: argparse.Namespace):
    torch.manual_seed(0)
    name = args.case
    binary = {
        "add_f32_2x4096x8": (0, "test_add.py", torch.add),
        "mul_f32_2x4096x8": (1, "test_mul.py", torch.mul),
        "div_f32_2x4096x8": (2, "test_div.py", torch.div),
        "maximum_f32_2x4096x8": (3, "test_maximum.py", torch.maximum),
    }
    if name in binary:
        op, source_file, reference = binary[name]
        shape, grid, xblock, sub = (2, 4096, 8), 2, 32768, 1024
        a = torch.randn(shape, dtype=torch.float32, device="npu")
        b = torch.randn(shape, dtype=torch.float32, device="npu")
        if op == 2:
            b = b + torch.where(b >= 0, torch.ones_like(b), -torch.ones_like(b))
        out = torch.empty_like(a)
        expected = reference(a, b)

        def launch():
            binary_loop_kernel[(grid, )](a, b, out, a.numel(), OP=op, XBLOCK=xblock, XBLOCK_SUB=sub,
                                         **route_kwargs(args, grid))

        return (f"{SOURCE_ROOT}/{source_file}", launch, out, expected, "binary_loop_kernel",
                {"shape": shape, "grid": grid, "XBLOCK": xblock, "XBLOCK_SUB": sub}, 2e-4, 2e-4)

    if name in ("sqrt_f32_n781", "cos_f32_n781"):
        op = 0 if name.startswith("sqrt") else 1
        source_file = "test_sqrt.py" if op == 0 else "test_cos.py"
        n, block = 781, 1024
        inp = torch.rand((n, ), dtype=torch.float32, device="npu")
        out = torch.empty_like(inp)
        expected = torch.sqrt(inp) if op == 0 else torch.cos(inp)

        def launch():
            unary_kernel[(1, )](inp, out, N=n, NUMEL=block, OP=op, **route_kwargs(args, 1))

        return (f"{SOURCE_ROOT}/{source_file}", launch, out, expected, "unary_kernel",
                {"N": n, "NUMEL": block, "grid": 1}, 2e-4, 2e-4)

    if name == "f2i32_f32_n3":
        n, block = 3, 32
        inp = torch.tensor([-3.75, 0.5, 7.25], dtype=torch.float32, device="npu")
        out = torch.empty((n, ), dtype=torch.int32, device="npu")
        expected = inp.to(torch.int32)

        def launch():
            f2i32_kernel[(1, )](inp, out, N=n, NUMEL=block, **route_kwargs(args, 1))

        return (f"{SOURCE_ROOT}/test_elementwise_f2i.py", launch, out, expected, "f2i32_kernel",
                {"N": n, "NUMEL": block, "grid": 1}, 0, 0)

    if name == "strided_compaction_f32_s16":
        in_numel, stride, grid, xblock, sub = 4096 * 16, 16, 2, 2048, 256
        out_numel = grid * xblock
        inp = torch.randn((in_numel, ), dtype=torch.float32, device="npu")
        out = torch.empty((out_numel, ), dtype=torch.float32, device="npu")
        expected = inp[::stride][:out_numel].contiguous()

        def launch():
            strided_compaction_kernel[(grid, )](inp, out, in_numel, out_numel, XBLOCK=xblock, XBLOCK_SUB=sub,
                                                STRIDE=stride, **route_kwargs(args, grid))

        return (f"{SOURCE_ROOT}/test_load_strided.py", launch, out, expected, "strided_compaction_kernel",
                {"in_numel": in_numel, "out_numel": out_numel, "stride": stride, "grid": grid}, 1e-4, 1e-4)

    if name == "strided_scatter_f32_s16":
        in_numel, stride, grid, xblock, sub = 4096, 16, 2, 2048, 256
        out_numel = in_numel * stride
        inp = torch.randn((in_numel, ), dtype=torch.float32, device="npu")
        out = torch.zeros((out_numel, ), dtype=torch.float32, device="npu")
        expected = torch.zeros_like(out)
        expected[::stride] = inp

        def launch():
            strided_scatter_kernel[(grid, )](inp, out, in_numel, out_numel, XBLOCK=xblock, XBLOCK_SUB=sub,
                                             STRIDE=stride, **route_kwargs(args, grid))

        return (f"{SOURCE_ROOT}/test_load_strided.py", launch, out, expected, "strided_scatter_kernel",
                {"in_numel": in_numel, "out_numel": out_numel, "stride": stride, "grid": grid}, 1e-4, 1e-4)

    if name == "block_ptr_strided_f32_4x8_s32x4":
        bm, bn, sm, sn = 4, 8, 32, 4
        max_offset = (bm - 1) * sm + (bn - 1) * sn + 1
        inp = torch.randn((max_offset, ), dtype=torch.float32, device="npu")
        out = torch.empty((bm, bn), dtype=torch.float32, device="npu")
        rows = torch.arange(bm, device="npu")[:, None]
        cols = torch.arange(bn, device="npu")[None, :]
        expected = inp[rows * sm + cols * sn]

        def launch():
            block_ptr_strided_kernel[(1, )](inp, out, bm * sm, sn * bn, sm, sn, BLOCK_M=bm, BLOCK_N=bn,
                                            **route_kwargs(args, 1))

        return (f"{SOURCE_ROOT}/test_load_strided.py", launch, out, expected, "block_ptr_strided_kernel",
                {"block": [bm, bn], "strides": [sm, sn], "grid": 1}, 1e-4, 1e-4)

    if name == "associative_scan_max_f32_128x4x16_dim2":
        shape, dim = (128, 4, 16), 2
        inp = torch.randn(shape, dtype=torch.float32, device="npu")
        out = torch.empty_like(inp)
        expected = prefix_max(inp, dim)

        def launch():
            scan3d_kernel[(1, )](out, inp, DIM=dim, REVERSE=False, XBLOCK=128, RBLOCK=4, ZBLOCK=16,
                                 **route_kwargs(args, 1))

        return (f"{SOURCE_ROOT}/test_associative_scan.py", launch, out, expected, "scan3d_kernel",
                {"shape": shape, "dim": dim, "grid": 1}, 1e-4, 1e-4)

    if name == "chunk_cumsum_f32_b4_t2048_h8_bt128":
        batch, time, heads, block = 4, 2048, 8, 128
        inp = torch.randn((batch, time, heads), dtype=torch.float32, device="npu") * 0.1
        out = torch.empty_like(inp)
        expected = inp.reshape(batch, time // block, block, heads).cumsum(2).reshape_as(inp)
        nt, logical = time // block, (time // block) * batch * heads

        def launch():
            chunk_cumsum_kernel[(nt, batch * heads)](inp, out, time, H=heads, BT=block, **route_kwargs(args, logical))

        return (f"{SOURCE_ROOT}/test_load_strided.py", launch, out, expected, "chunk_cumsum_kernel",
                {"shape": [batch, time, heads], "BT": block, "grid": [nt, batch * heads]}, 1e-3, 1e-3)

    raise ValueError(name)


CASES = (
    "add_f32_2x4096x8",
    "mul_f32_2x4096x8",
    "div_f32_2x4096x8",
    "maximum_f32_2x4096x8",
    "sqrt_f32_n781",
    "cos_f32_n781",
    "f2i32_f32_n3",
    "strided_compaction_f32_s16",
    "strided_scatter_f32_s16",
    "block_ptr_strided_f32_4x8_s32x4",
    "associative_scan_max_f32_128x4x16_dim2",
    "chunk_cumsum_f32_b4_t2048_h8_bt128",
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
                    True, **metadata
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
