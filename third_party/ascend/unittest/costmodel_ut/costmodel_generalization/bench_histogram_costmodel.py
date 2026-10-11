#!/usr/bin/env python3
"""Generalization benchmark adapted from triton_cases/historgram."""

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
def histogram_kernel(input_ptr, output_ptr, WIDTH: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, WIDTH)
    values = tl.load(input_ptr + row * WIDTH + column)
    counts = tl.histogram(values, WIDTH)
    tl.store(output_ptr + row * WIDTH + column, counts)


def histogram_input(rows: int, width: int, distribution: str) -> torch.Tensor:
    torch.manual_seed(1)
    if distribution == "uniform":
        return torch.randint(0, width, (rows, width), dtype=torch.int32)
    if distribution == "single_bin":
        return torch.zeros((rows, width), dtype=torch.int32)
    scores = torch.randn((rows, width), dtype=torch.float32)
    bits = scores.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    keys = torch.where(scores < 0, (~bits) & 0xFFFFFFFF, bits | 0x80000000)
    return ((keys >> 24) & (width - 1)).to(torch.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", choices=("uniform", "single_bin", "radix"), required=True)
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), required=True)
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--width", type=int, choices=(64, 128, 256), default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if args.distribution == "radix" and args.width & (args.width - 1):
        raise ValueError("radix distribution requires a power-of-two width")

    host_input = histogram_input(args.rows, args.width, args.distribution)
    expected = torch.zeros_like(host_input)
    expected.scatter_add_(1, host_input.to(torch.int64), torch.ones_like(host_input))
    source = host_input.npu()
    output = torch.empty_like(source)
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    compile_mode = "simd_simt" if args.route == "auto" else args.route

    def launch() -> None:
        histogram_kernel[(args.rows, )](
            source,
            output,
            WIDTH=args.width,
            num_warps=4,
            compile_on_910_95=True,
            compile_mode=compile_mode,
            superblock_factor=args.factor,
            logical_program_count_hint=args.rows,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.report else "",
        )

    launch()
    torch.npu.synchronize()
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
    profiler = (profile_kernel(launch, args.profile_dir, "histogram_kernel") if args.profile_dir else None)
    print(
        json.dumps(
            {
                "case": "histogram",
                "distribution": args.distribution,
                "route_requested": args.route,
                "compile_mode": compile_mode,
                "factor": args.factor,
                "rows": args.rows,
                "width": args.width,
                "grid": args.rows,
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
