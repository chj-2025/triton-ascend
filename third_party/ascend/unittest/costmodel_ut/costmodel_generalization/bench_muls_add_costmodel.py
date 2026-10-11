#!/usr/bin/env python3
"""Cross-kernel StageCostModel benchmark for the existing muls_add case."""

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
def muls_add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    scale,
    n_elements,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    for block_id in range(pid, n_blocks, num_programs):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, x * scale + y, mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), required=True)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--n-elements", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()

    if args.factor not in (1, 2, 4):
        raise ValueError("--factor must be 1, 2, or 4")
    torch.manual_seed(0)
    x = torch.randn(args.n_elements, dtype=torch.float32, device="npu")
    y = torch.randn_like(x)
    output = torch.empty_like(x)
    n_blocks = triton.cdiv(args.n_elements, args.block_size)
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    grid = min(n_blocks, vector_cores)
    compile_mode = "simd_simt" if args.route == "auto" else args.route

    def launch() -> None:
        muls_add_kernel[(grid, )](
            x,
            y,
            output,
            args.scale,
            args.n_elements,
            n_blocks,
            BLOCK_SIZE=args.block_size,
            num_warps=4,
            compile_mode=compile_mode,
            superblock_factor=args.factor,
            # Each loop body consumes one logical block.  The physical grid is
            # capped by the vector-core count and strip-mines these work items.
            logical_program_count_hint=n_blocks,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.report else "",
        )

    launch()
    torch.npu.synchronize()
    expected = x * args.scale + y
    max_abs = float((output - expected).abs().max())
    torch.testing.assert_close(output, expected, rtol=1.0e-5, atol=1.0e-5)
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
    profiler = (profile_kernel(launch, args.profile_dir, "muls_add_kernel") if args.profile_dir else None)
    print(
        json.dumps(
            {
                "case": f"muls_add_n{args.n_elements}",
                "route_requested": args.route,
                "compile_mode": compile_mode,
                "factor": args.factor,
                "n_elements": args.n_elements,
                "block_size": args.block_size,
                "n_blocks": n_blocks,
                "grid": grid,
                "logical_programs": n_blocks,
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
