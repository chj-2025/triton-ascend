#!/usr/bin/env python3
"""Cross-kernel StageCostModel benchmark for the existing softmax case."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.runtime import driver

from profile_utils import profile_kernel


@triton.jit
def triton_softmax_kernel(
    in_ptr,
    out_ptr,
    rows,
    cols: tl.constexpr,
    X0BLOCK: tl.constexpr,
    X0BLOCK_SUB: tl.constexpr,
    R1BLOCK_SUB: tl.constexpr,
):
    x0_offset = tl.program_id(0) * X0BLOCK
    base_x0 = tl.arange(0, X0BLOCK_SUB)
    loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
    base_r1 = tl.arange(0, R1BLOCK_SUB)
    for loop_x0 in range(loops_x0):
        x0 = x0_offset + loop_x0 * X0BLOCK_SUB + base_x0[:, None]
        x0_mask = x0 < min(X0BLOCK + x0_offset, rows)
        r1 = base_r1[None, :]
        mask = x0_mask & (r1 < cols)
        value = tl.load(in_ptr + r1 + cols * x0, mask=mask, other=-float("inf"))
        maximum = tl.max(value, 1).reshape(X0BLOCK_SUB, 1)
        numerator = tl.exp(value - maximum)
        denominator = tl.sum(numerator, 1).reshape(X0BLOCK_SUB, 1)
        tl.store(out_ptr + r1 + cols * x0, numerator / denominator, mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), required=True)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--rows", type=int, default=76800)
    parser.add_argument("--grid", type=int, default=56)
    parser.add_argument("--layout", choices=("original", "power2"), default="original")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()

    if args.factor not in (1, 2, 4):
        raise ValueError("--factor must be 1, 2, or 4")
    torch.manual_seed(0)
    cols = 150
    x0block_sub = 80 if args.layout == "original" else 64
    r1block_sub = 150 if args.layout == "original" else 256
    x0block = triton.cdiv(args.rows, args.grid)
    source = torch.randn((args.rows, cols), dtype=torch.float32, device="npu")
    output = torch.empty_like(source)
    compile_mode = "simd_simt" if args.route == "auto" else args.route
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])

    def launch() -> None:
        triton_softmax_kernel[(args.grid, )](
            source,
            output,
            args.rows,
            cols,
            X0BLOCK=x0block,
            X0BLOCK_SUB=x0block_sub,
            R1BLOCK_SUB=r1block_sub,
            multibuffer=True,
            enable_vf_fusion=True,
            num_warps=4,
            compile_mode=compile_mode,
            superblock_factor=args.factor,
            logical_program_count_hint=args.grid,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.report else "",
        )

    launch()
    torch.npu.synchronize()
    expected = torch.softmax(source, dim=1)
    max_abs = float((output - expected).abs().max())
    torch.testing.assert_close(output, expected, rtol=1.0e-4, atol=1.0e-5)
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
    profiler = (profile_kernel(launch, args.profile_dir, "triton_softmax_kernel") if args.profile_dir else None)
    print(
        json.dumps(
            {
                "case": f"softmax_76800x150_{args.layout}",
                "route_requested": args.route,
                "compile_mode": compile_mode,
                "factor": args.factor,
                "rows": args.rows,
                "cols": cols,
                "grid": args.grid,
                "logical_programs": args.grid,
                "vector_cores": vector_cores,
                "timing_batch": args.batch,
                "correct": True,
                "max_abs": max_abs,
                "elapsed_us": {
                    "median": statistics.median(samples),
                    "minimum": samples[0],
                    "maximum": samples[-1],
                    "samples": samples,
                },
                "report": str(args.report) if args.report else None,
                "profiler": profiler,
            }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
