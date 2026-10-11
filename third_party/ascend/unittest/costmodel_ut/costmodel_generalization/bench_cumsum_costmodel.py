#!/usr/bin/env python3
"""Cross-operator prefix-scan benchmark for the Stage cost model."""

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
def cumsum_kernel(
    input_ptr,
    output_ptr,
    rows,
    WIDTH: tl.constexpr,
    REVERSE: tl.constexpr,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    column = tl.arange(0, WIDTH)
    for row in range(pid, rows, programs):
        value = tl.load(input_ptr + row * WIDTH + column)
        result = tl.cumsum(value, axis=0, reverse=REVERSE)
        tl.store(output_ptr + row * WIDTH + column, result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), required=True)
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--rows", type=int, default=1792)
    parser.add_argument("--width", type=int, choices=(128, 256, 512), default=256)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()

    torch.manual_seed(37)
    source = torch.randint(-1024, 1024, (args.rows, args.width), dtype=torch.int32, device="npu")
    output = torch.empty_like(source)
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    grid = min(args.rows, vector_cores)
    compile_mode = "simd_simt" if args.route == "auto" else args.route

    def launch() -> None:
        cumsum_kernel[(grid, )](
            source,
            output,
            args.rows,
            WIDTH=args.width,
            REVERSE=args.reverse,
            num_warps=4,
            compile_mode=compile_mode,
            superblock_factor=args.factor,
            logical_program_count_hint=args.rows,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.report else "",
        )

    launch()
    torch.npu.synchronize()
    expected = source.flip(1) if args.reverse else source
    expected = expected.cpu().cumsum(1).to(torch.int32)
    if args.reverse:
        expected = expected.flip(1)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
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
    profiler = (profile_kernel(launch, args.profile_dir, "cumsum_kernel") if args.profile_dir else None)
    print(
        json.dumps(
            {
                "case": "cumsum",
                "route_requested": args.route,
                "compile_mode": compile_mode,
                "factor": args.factor,
                "rows": args.rows,
                "width": args.width,
                "reverse": args.reverse,
                "grid": grid,
                "logical_programs": args.rows,
                "correct": True,
                "elapsed_us": {
                    "median": statistics.median(samples),
                    "minimum": samples[0],
                    "maximum": samples[-1],
                    "samples": samples,
                },
                "report": str(args.report) if args.report else None,
                "profiler": profiler,
            },
            indent=2,
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
