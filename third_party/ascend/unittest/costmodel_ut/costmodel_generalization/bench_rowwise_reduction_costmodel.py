#!/usr/bin/env python3
"""Operator-independent rowwise-reduction probe for StageCostModel."""

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
def rowwise_reduction_kernel(
    input_ptr,
    output_ptr,
    rows,
    ROWS_PER_PROGRAM: tl.constexpr,
    COLS: tl.constexpr,
    REPEATS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_lane = tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    col = tl.arange(0, COLS)[None, :]
    for repeat in range(REPEATS):
        row = (pid * REPEATS + repeat) * ROWS_PER_PROGRAM + row_lane
        mask = row < rows
        value = tl.load(input_ptr + row * COLS + col, mask=mask, other=0.0)
        maximum = tl.max(value, axis=1).reshape(ROWS_PER_PROGRAM, 1)
        total = tl.sum(value, axis=1).reshape(ROWS_PER_PROGRAM, 1)
        tl.store(output_ptr + row * 2, maximum, mask=mask)
        tl.store(output_ptr + row * 2 + 1, total, mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), default="auto")
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--rows-per-program", type=int, choices=(16, 32, 64), required=True)
    parser.add_argument("--cols", type=int, choices=(64, 128, 256), required=True)
    parser.add_argument("--repeats-per-program", type=int, default=16)
    parser.add_argument("--grid", type=int, default=56)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()

    rows = args.grid * args.repeats_per_program * args.rows_per_program
    torch.manual_seed(0)
    source = torch.randn((rows, args.cols), dtype=torch.float32, device="npu")
    output = torch.empty((rows, 2), dtype=torch.float32, device="npu")
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    compile_mode = "simd_simt" if args.route == "auto" else args.route

    def launch() -> None:
        rowwise_reduction_kernel[(args.grid, )](
            source,
            output,
            rows,
            ROWS_PER_PROGRAM=args.rows_per_program,
            COLS=args.cols,
            REPEATS=args.repeats_per_program,
            num_warps=4,
            compile_mode=compile_mode,
            superblock_factor=args.factor,
            logical_program_count_hint=args.grid,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.route == "auto" else "",
            multibuffer=True,
            enable_vf_fusion=True,
        )

    launch()
    torch.npu.synchronize()
    expected = torch.stack((source.max(dim=1).values, source.sum(dim=1)), dim=1)
    max_abs = float((output - expected).abs().max())
    torch.testing.assert_close(output, expected, rtol=1.0e-4, atol=1.0e-4)
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
    profiler = (profile_kernel(launch, args.profile_dir, "rowwise_reduction_kernel") if args.profile_dir else None)
    lane_steps = (2 * args.rows_per_program * args.cols * (args.cols.bit_length() - 1) * args.repeats_per_program)
    print(
        json.dumps(
            {
                "case": "rowwise_reduction_probe",
                "route_requested": args.route,
                "compile_mode": compile_mode,
                "factor": args.factor,
                "rows_per_program": args.rows_per_program,
                "cols": args.cols,
                "repeats_per_program": args.repeats_per_program,
                "grid": args.grid,
                "rows": rows,
                "semantic_reduction_lane_steps_per_program": lane_steps,
                "correct": True,
                "timing_batch": args.batch,
                "max_abs": max_abs,
                "elapsed_us": {
                    "median": statistics.median(samples),
                    "minimum": samples[0],
                    "maximum": samples[-1],
                    "samples": samples,
                },
                "report": str(args.report),
                "profiler": profiler,
            }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
