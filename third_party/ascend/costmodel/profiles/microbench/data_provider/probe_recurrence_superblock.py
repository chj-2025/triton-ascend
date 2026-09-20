#!/usr/bin/env python3
"""Measure loop-carried recurrence scaling for SIMD and SIMT F1/F2/F4."""

import argparse
import csv
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


@triton.jit
def row_prefix(row, columns: tl.constexpr):
    if columns == 16:
        return row
    else:
        first, _ = tl.split(tl.trans(tl.reshape(row, (2, 8))))
        return first


@triton.jit(do_not_specialize=["trips"])
def recurrence_superblock_probe(source, output, trips,
                                recurrence_groups: tl.constexpr,
                                columns: tl.constexpr = 16,
                                row_stride: tl.constexpr = 16,
                                grid_x: tl.constexpr = 0,
                                diagonal_stride: tl.constexpr = 0):
    pid = tl.program_id(0)
    lane = tl.arange(0, 16)
    column = tl.arange(0, columns)
    source_base = pid * 64 * row_stride
    if grid_x:
        source_base = (pid % grid_x) * 64 * row_stride + (pid // grid_x) * 64
    state0 = tl.zeros((16, columns), tl.float32)
    state1 = tl.zeros((16, columns), tl.float32)
    state2 = tl.zeros((16, columns), tl.float32)
    state3 = tl.zeros((16, columns), tl.float32)

    for step in range(1, trips + 1):
        row = -tl.load(source + source_base + step * row_stride + lane)
        row = tl.where(lane < step, row, 0.0)
        update = row_prefix(row, columns) + tl.sum(row[:, None] * state0, axis=0)
        state0 = tl.where((lane == step)[:, None], update, state0)
    if recurrence_groups >= 2:
        for step in range(1, trips + 1):
            row = -tl.load(source + source_base + (16 + step) * row_stride + diagonal_stride + lane)
            row = tl.where(lane < step, row, 0.0)
            update = row_prefix(row, columns) + tl.sum(row[:, None] * state1, axis=0)
            state1 = tl.where((lane == step)[:, None], update, state1)
    if recurrence_groups >= 3:
        for step in range(1, trips + 1):
            row = -tl.load(source + source_base + (32 + step) * row_stride + 2 * diagonal_stride + lane)
            row = tl.where(lane < step, row, 0.0)
            update = row_prefix(row, columns) + tl.sum(row[:, None] * state2, axis=0)
            state2 = tl.where((lane == step)[:, None], update, state2)
    if recurrence_groups >= 4:
        for step in range(1, trips + 1):
            row = -tl.load(source + source_base + (48 + step) * row_stride + 3 * diagonal_stride + lane)
            row = tl.where(lane < step, row, 0.0)
            update = row_prefix(row, columns) + tl.sum(row[:, None] * state3, axis=0)
            state3 = tl.where((lane == step)[:, None], update, state3)

    result = tl.sum(state0, axis=0)
    if recurrence_groups >= 2:
        result += tl.sum(state1, axis=0)
    if recurrence_groups >= 3:
        result += tl.sum(state2, axis=0)
    if recurrence_groups >= 4:
        result += tl.sum(state3, axis=0)
    tl.store(output + pid * columns + column, result)


def reference(source: torch.Tensor, trips: int, groups: int, columns: int = 16,
              diagonal_stride: int = 0) -> torch.Tensor:
    source = source.float()
    result = torch.zeros((source.shape[0], columns), dtype=torch.float32)
    lane = torch.arange(16)
    for group in range(groups):
        state = torch.zeros((source.shape[0], 16, columns), dtype=torch.float32)
        for step in range(1, trips + 1):
            start = group * diagonal_stride
            row = -source[:, group * 16 + step, start:start + 16]
            row = torch.where(lane < step, row, 0.0)
            row = row[:, :columns] + torch.einsum("bi,bij->bj", row, state)
            state[:, step, :] = row
        result += state.sum(dim=1)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("simd", "simt_only"),
                        default="simt_only")
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--trips", type=int, choices=(0, 1, 2, 4, 8, 14),
                        required=True)
    parser.add_argument("--recurrence-groups", type=int, choices=(1, 4),
                        default=4)
    parser.add_argument("--columns", type=int, choices=(8, 16), default=16)
    parser.add_argument("--row-stride", type=int, choices=(16, 2048), default=16)
    parser.add_argument("--grid-x", type=int, default=0,
                        help="Zero: disjoint program tiles. Positive: x-fastest logical IDs with 64-column tiles interleaved in each row.")
    parser.add_argument("--diagonal-stride", type=int, choices=(0, 16), default=0)
    parser.add_argument(
        "--logical-programs", type=int, default=0,
        help="Logical grid size; zero uses four waves before SuperBlock grouping.")
    args = parser.parse_args()
    if args.mode == "simd" and args.factor != 1:
        parser.error("SIMD recurrence measurements require --factor=1")
    args.output.mkdir(parents=True, exist_ok=False)

    vector_cores = int(driver.active.utils.get_device_properties(
        torch.npu.current_device())["num_vectorcore"])
    logical_programs = args.logical_programs or vector_cores * 4
    if logical_programs < 1:
        raise ValueError("logical-programs must be positive")
    if args.grid_x < 0 or (args.grid_x and logical_programs % args.grid_x):
        parser.error("grid-x must be zero or a positive divisor of logical-programs")
    needed_columns = (logical_programs // args.grid_x) * 64 if args.grid_x else 16 + 3 * args.diagonal_stride
    if needed_columns > args.row_stride:
        parser.error("row-stride is too small for the requested nonoverlapping tiles")
    torch.manual_seed(19)
    source_cpu = torch.randn((args.grid_x or logical_programs, 64, args.row_stride), dtype=torch.float32) / 64
    reference_source = source_cpu
    if args.grid_x:
        reference_source = torch.stack([
            source_cpu[pid % args.grid_x, :, (pid // args.grid_x) * 64:(pid // args.grid_x + 1) * 64]
            for pid in range(logical_programs)
        ])
    source = source_cpu.npu()
    output = torch.empty((logical_programs, args.columns), dtype=torch.float32,
                         device="npu")

    def launch():
        options = {"compile_mode": args.mode, "num_warps": 4}
        if args.mode == "simt_only":
            options["superblock_factor"] = args.factor
        return recurrence_superblock_probe[(logical_programs,)](
            source, output, args.trips, args.recurrence_groups,
            args.columns, args.row_stride, args.grid_x, args.diagonal_stride, **options)

    binary = launch()
    torch.testing.assert_close(
        output.cpu(), reference(reference_source, args.trips, args.recurrence_groups, args.columns, args.diagonal_stride),
        rtol=2e-3, atol=2e-3)
    for key in ("ttir", "ttadapter", "npuir", "bcmlir", "npubin"):
        value = binary.asm.get(key)
        if isinstance(value, str):
            (args.output / ("kernel." + key)).write_text(value)
        elif isinstance(value, bytes):
            (args.output / ("kernel." + key)).write_bytes(value)

    for _ in range(20):
        launch()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(args.output / "profile")),
    ) as profiler:
        for _ in range(20):
            launch()
        torch.npu.synchronize()
    torch.npu.synchronize()

    samples = []
    for path in args.output.rglob("kernel_details.csv"):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if ("recurrence_superblock_probe" in str(row) and row.get("Duration(us)")):
                    samples.append(float(row["Duration(us)"]))
    if len(samples) != 20:
        raise RuntimeError(f"expected 20 samples, got {len(samples)}")
    torch.testing.assert_close(
        output.cpu(), reference(reference_source, args.trips, args.recurrence_groups, args.columns, args.diagonal_stride),
        rtol=2e-3, atol=2e-3)
    result = {
        "mode": args.mode,
        "factor": args.factor,
        "trips": args.trips,
        "recurrence_groups": args.recurrence_groups,
        "columns": args.columns,
        "row_stride_elements": args.row_stride,
        "grid_x": args.grid_x,
        "diagonal_stride_elements": args.diagonal_stride,
        "source_allocation_bytes": source_cpu.numel() * source_cpu.element_size(),
        "warmup": 20,
        "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "binary_metadata": dict(binary.metadata._asdict()),
        "npubin_sha256": hashlib.sha256((args.output / "kernel.npubin").read_bytes()).hexdigest(),
        "environment": {key: os.environ.get(key) for key in (
            "ASCEND_RT_VISIBLE_DEVICES", "TRITON_DEBUG",
            "TRITON_DISABLE_LINE_INFO", "TRITON_CACHE_DIR")},
        "interpretation": "Whole-kernel shape probe, not an isolated Mixed scope. Full 16-element row loads and reduction depth remain unchanged; slopes require verified group scheduling.",
        "logical_programs": logical_programs,
        "physical_programs":
            (logical_programs + args.factor - 1) // args.factor,
        "vector_cores": vector_cores,
        "expected_waves":
            ((logical_programs + args.factor - 1) // args.factor +
             vector_cores - 1) // vector_cores,
        "sample_count": len(samples),
        "median_us": statistics.median(samples),
        "samples_us": samples,
        "correct": True,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
