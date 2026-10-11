#!/usr/bin/env python3
"""Structural masked-layout probe for generic SIMD/SIMT memory costing.

The probe varies logical row width, compiled vector width, physical row stride,
rows per program, and loop count independently.  Copy, reduction, and softmax
variants share the same access geometry so memory-layout residuals can be
separated from arithmetic and reduction work without using an operator name or
an exact production shape as a model feature.
"""

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
def masked_layout_copy_kernel(
    input_ptr,
    output_ptr,
    rows,
    ROWS_PER_PROGRAM: tl.constexpr,
    LOGICAL_COLS: tl.constexpr,
    VECTOR_COLS: tl.constexpr,
    STRIDE_COLS: tl.constexpr,
    LOOPS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_lane = tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    col = tl.arange(0, VECTOR_COLS)[None, :]
    for repeat in range(LOOPS):
        row = (pid * LOOPS + repeat) * ROWS_PER_PROGRAM + row_lane
        mask = (row < rows) & (col < LOGICAL_COLS)
        offsets = row * STRIDE_COLS + col
        value = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        tl.store(output_ptr + offsets, value + 1.0, mask=mask)


@triton.jit
def masked_layout_reduction_kernel(
    input_ptr,
    output_ptr,
    rows,
    ROWS_PER_PROGRAM: tl.constexpr,
    LOGICAL_COLS: tl.constexpr,
    VECTOR_COLS: tl.constexpr,
    STRIDE_COLS: tl.constexpr,
    LOOPS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_lane = tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    col = tl.arange(0, VECTOR_COLS)[None, :]
    for repeat in range(LOOPS):
        row = (pid * LOOPS + repeat) * ROWS_PER_PROGRAM + row_lane
        row_mask = row < rows
        mask = row_mask & (col < LOGICAL_COLS)
        value = tl.load(input_ptr + row * STRIDE_COLS + col, mask=mask, other=0.0)
        maximum = tl.max(value, axis=1).reshape(ROWS_PER_PROGRAM, 1)
        total = tl.sum(value, axis=1).reshape(ROWS_PER_PROGRAM, 1)
        tl.store(output_ptr + row * 2, maximum, mask=row_mask)
        tl.store(output_ptr + row * 2 + 1, total, mask=row_mask)


@triton.jit
def masked_layout_softmax_kernel(
    input_ptr,
    output_ptr,
    rows,
    ROWS_PER_PROGRAM: tl.constexpr,
    LOGICAL_COLS: tl.constexpr,
    VECTOR_COLS: tl.constexpr,
    STRIDE_COLS: tl.constexpr,
    LOOPS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_lane = tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    col = tl.arange(0, VECTOR_COLS)[None, :]
    for repeat in range(LOOPS):
        row = (pid * LOOPS + repeat) * ROWS_PER_PROGRAM + row_lane
        row_mask = row < rows
        mask = row_mask & (col < LOGICAL_COLS)
        offsets = row * STRIDE_COLS + col
        value = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
        maximum = tl.max(value, axis=1).reshape(ROWS_PER_PROGRAM, 1)
        numerator = tl.exp(value - maximum)
        denominator = tl.sum(numerator, axis=1).reshape(ROWS_PER_PROGRAM, 1)
        tl.store(output_ptr + offsets, numerator / denominator, mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("copy", "reduction", "softmax"), required=True)
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), default="auto")
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--rows-per-program", type=int, required=True)
    parser.add_argument("--logical-cols", type=int, required=True)
    parser.add_argument("--vector-cols", type=int, required=True)
    parser.add_argument("--stride-cols", type=int, required=True)
    parser.add_argument("--loops", type=int, required=True)
    parser.add_argument("--grid", type=int, default=56)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--vf-fusion-mode", choices=("ub-aware-op", "max-parallel"), default=None)
    parser.add_argument("--vf-merge-level", type=int, default=1)
    parser.add_argument("--multiple-consumer-fusion", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()

    for name in (
            "rows_per_program",
            "logical_cols",
            "vector_cols",
            "stride_cols",
            "loops",
            "grid",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.logical_cols > args.vector_cols:
        parser.error("--logical-cols must not exceed --vector-cols")
    if args.logical_cols > args.stride_cols:
        parser.error("--logical-cols must not exceed --stride-cols")

    rows = args.grid * args.loops * args.rows_per_program
    torch.manual_seed(0)
    source = torch.randn((rows, args.stride_cols), dtype=torch.float32, device="npu")
    if args.kind == "reduction":
        output = torch.empty((rows, 2), dtype=torch.float32, device="npu")
        kernel = masked_layout_reduction_kernel
        kernel_name = "masked_layout_reduction_kernel"
    else:
        output = torch.full_like(source, float("nan"))
        kernel = (masked_layout_copy_kernel if args.kind == "copy" else masked_layout_softmax_kernel)
        kernel_name = ("masked_layout_copy_kernel" if args.kind == "copy" else "masked_layout_softmax_kernel")

    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    compile_mode = "simd_simt" if args.route == "auto" else args.route

    def launch() -> None:
        kernel[(args.grid, )](
            source,
            output,
            rows,
            ROWS_PER_PROGRAM=args.rows_per_program,
            LOGICAL_COLS=args.logical_cols,
            VECTOR_COLS=args.vector_cols,
            STRIDE_COLS=args.stride_cols,
            LOOPS=args.loops,
            num_warps=4,
            compile_mode=compile_mode,
            superblock_factor=args.factor,
            logical_program_count_hint=args.grid,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.route == "auto" else "",
            multibuffer=True,
            enable_vf_fusion=True,
            vf_fusion_mode=args.vf_fusion_mode,
            vf_merge_level=args.vf_merge_level,
            hfusion_enable_multiple_consumer_fusion=args.multiple_consumer_fusion,
        )

    launch()
    torch.npu.synchronize()
    active = source[:, :args.logical_cols]
    if args.kind == "copy":
        expected = active + 1.0
        actual = output[:, :args.logical_cols]
        tolerance = {"rtol": 1.0e-5, "atol": 1.0e-5}
    elif args.kind == "reduction":
        expected = torch.stack((active.max(dim=1).values, active.sum(dim=1)), dim=1)
        actual = output
        tolerance = {"rtol": 1.0e-4, "atol": 1.0e-4}
    else:
        expected = torch.softmax(active, dim=1)
        actual = output[:, :args.logical_cols]
        tolerance = {"rtol": 1.0e-4, "atol": 1.0e-5}
    max_abs = float((actual - expected).abs().max())
    torch.testing.assert_close(actual, expected, **tolerance)

    if args.compile_only:
        print(
            json.dumps(
                {
                    "case": "masked_layout_structural_probe",
                    "kind": args.kind,
                    "compile_only": True,
                    "correct": True,
                    "max_abs": max_abs,
                    "report": str(args.report),
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
    profiler = (profile_kernel(launch, args.profile_dir, kernel_name) if args.profile_dir else None)
    row_bytes = args.vector_cols * 4
    row_stride_bytes = args.stride_cols * 4
    source_data_ptr = source.data_ptr()
    output_data_ptr = output.data_ptr()
    print(
        json.dumps(
            {
                "case": "masked_layout_structural_probe",
                "kind": args.kind,
                "route_requested": args.route,
                "compile_mode": compile_mode,
                "factor": args.factor,
                "rows_per_program": args.rows_per_program,
                "logical_cols": args.logical_cols,
                "vector_cols": args.vector_cols,
                "stride_cols": args.stride_cols,
                "row_bytes": row_bytes,
                "row_stride_bytes": row_stride_bytes,
                "row_to_stride_ratio": row_bytes / row_stride_bytes,
                "active_lane_ratio": args.logical_cols / args.vector_cols,
                "loops": args.loops,
                "grid": args.grid,
                "rows": rows,
                "source_data_ptr": source_data_ptr,
                "output_data_ptr": output_data_ptr,
                "source_alignment_2m": source_data_ptr % (2 * 1024 * 1024),
                "output_alignment_2m": output_data_ptr % (2 * 1024 * 1024),
                "correct": True,
                "max_abs": max_abs,
                "timing_batch": args.batch,
                "elapsed_us": {
                    "median": statistics.median(samples),
                    "minimum": samples[0],
                    "maximum": samples[-1],
                    "samples": samples,
                },
                "measurement_target": "profiler.average_us",
                "vf_fusion_mode": args.vf_fusion_mode,
                "vf_merge_level": args.vf_merge_level,
                "multiple_consumer_fusion": args.multiple_consumer_fusion,
                "report": str(args.report),
                "profiler": profiler,
            }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
