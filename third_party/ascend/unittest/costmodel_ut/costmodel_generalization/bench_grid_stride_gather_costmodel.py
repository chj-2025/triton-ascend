#!/usr/bin/env python3
"""Measure a dynamic grid-stride indirect-gather loop on A5."""

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
def grid_stride_gather(source, indices, output, logical_programs, source_width: tl.constexpr, block: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    row_lane = tl.arange(0, block)
    column = tl.arange(0, block)
    gathered_column = tl.load(indices + column)
    for program in range(pid, logical_programs, programs):
        row = program * block + row_lane
        values = tl.load(source + row[:, None] * source_width + gathered_column[None, :])
        tl.store(output + row[:, None] * block + column[None, :], values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logical-programs", type=int, required=True)
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), default="auto")
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()

    torch.manual_seed(29)
    block = 16
    source_width = 256
    rows = args.logical_programs * block
    source = torch.randn((rows, source_width), dtype=torch.float16, device="npu")
    indices = torch.tensor(
        [10, 25, 100, 200, 5, 50, 150, 255, 1, 2, 3, 4, 6, 7, 8, 9],
        dtype=torch.int32,
        device="npu",
    )
    output = torch.empty((rows, block), dtype=torch.float16, device="npu")
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    grid = min(args.logical_programs, vector_cores)
    compile_mode = "simd_simt" if args.route == "auto" else args.route

    def launch() -> None:
        grid_stride_gather[(grid, )](
            source,
            indices,
            output,
            args.logical_programs,
            source_width,
            block,
            compile_mode=compile_mode,
            num_warps=4,
            superblock_factor=args.factor,
            logical_program_count_hint=args.logical_programs,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.route == "auto" else "",
        )

    launch()
    torch.npu.synchronize()
    torch.testing.assert_close(output, source[:, indices])
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
    profile = (profile_kernel(launch, args.profile_dir, "grid_stride_gather") if args.profile_dir else None)
    print(
        json.dumps(
            {
                "case": "grid_stride_gather",
                "logical_programs": args.logical_programs,
                "grid": grid,
                "route_requested": args.route,
                "factor": args.factor,
                "correct": True,
                "elapsed_us": {
                    "median": statistics.median(samples),
                    "minimum": samples[0],
                    "maximum": samples[-1],
                    "samples": samples,
                },
                "profiler": profile,
            }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
