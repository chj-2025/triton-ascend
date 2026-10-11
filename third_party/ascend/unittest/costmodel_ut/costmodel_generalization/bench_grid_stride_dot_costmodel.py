#!/usr/bin/env python3
"""Measure a dynamic grid-stride dot loop on A5."""

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
def grid_stride_dot(a_ptr, b_ptr, output_ptr, logical_programs, stride_am, stride_ak, stride_bk, stride_bn, stride_om,
                    stride_on, block: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    lane = tl.arange(0, block)
    b = tl.load(b_ptr + lane[:, None] * stride_bk + lane[None, :] * stride_bn)
    for program in range(pid, logical_programs, programs):
        row = program * block + lane
        a = tl.load(a_ptr + row[:, None] * stride_am + lane[None, :] * stride_ak)
        result = tl.dot(a, b)
        tl.store(output_ptr + row[:, None] * stride_om + lane[None, :] * stride_on, result)


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

    torch.manual_seed(31)
    block = 16
    rows = args.logical_programs * block
    a = torch.randn((rows, block), dtype=torch.float16, device="npu")
    b = torch.randn((block, block), dtype=torch.float16, device="npu")
    output = torch.empty((rows, block), dtype=torch.float32, device="npu")
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    grid = min(args.logical_programs, vector_cores)
    compile_mode = "simd_simt" if args.route == "auto" else args.route

    def launch() -> None:
        grid_stride_dot[(grid, )](
            a,
            b,
            output,
            args.logical_programs,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            output.stride(0),
            output.stride(1),
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
    expected = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
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
    profile = (profile_kernel(launch, args.profile_dir, "grid_stride_dot") if args.profile_dir else None)
    print(
        json.dumps(
            {
                "case": "grid_stride_dot",
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
