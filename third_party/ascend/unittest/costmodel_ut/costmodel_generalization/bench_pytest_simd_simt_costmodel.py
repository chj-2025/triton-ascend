#!/usr/bin/env python3
"""Fresh profiler benchmarks sourced from pytest_ut/test_simd_simt.py."""

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

SOURCE_PYTEST = "third_party/ascend/unittest/pytest_ut/test_simd_simt.py"


@triton.jit
def indirect_load_add_kernel(src, indices, output, add_value, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    index = tl.load(indices + lane)
    value = tl.load(src + index)
    tl.store(output + lane, value + add_value)


@triton.jit
def indirect_index_div_kernel(src, add_src, indices, output, offset1, offset2, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    index = tl.load(indices + lane)
    off1 = tl.load(offset1 + lane)
    off2 = tl.load(offset2 + lane)
    source_index = index + (off2 / off1).to(tl.int64) * 2
    tl.store(output + lane, tl.load(src + source_index) + tl.load(add_src + lane))


@triton.jit
def indirect_index_and_kernel(src, add_src, indices, output, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    index = tl.load(indices + lane)
    tl.store(output + lane, tl.load(src + (index & 63)) + tl.load(add_src + lane))


@triton.jit
def indirect_index_maxmin_kernel(src, add_src, indices, output, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    index = tl.load(indices + lane)
    source_index = tl.minimum(tl.maximum(index, 3), 64)
    tl.store(output + lane, tl.load(src + source_index) + tl.load(add_src + lane))


@triton.jit
def indirect_load_2d_min_kernel(src, indices, output, n_cols, M: tl.constexpr, K: tl.constexpr):
    row = tl.arange(0, M)[:, None]
    col = tl.arange(0, K)[None, :]
    gathered = tl.load(indices + row * K + col)
    tl.store(output + row * K + col, tl.load(src + row * n_cols + gathered))


@triton.jit
def indirect_load_2d_grid_kernel(
    src,
    indices,
    output,
    n_cols,
    gather_cols,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    row2d = row[:, None]
    col2d = col[None, :]
    gathered = tl.load(indices + row2d * gather_cols + col2d)
    tl.store(
        output + row2d * gather_cols + col2d,
        tl.load(src + row2d * n_cols + gathered),
    )


@triton.jit
def index_select_kernel(src, indices, output, D: tl.constexpr, BLOCK_M: tl.constexpr):
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, D)
    gathered_row = tl.load(indices + row)
    value = tl.load(src + gathered_row[:, None] * D + col[None, :])
    tl.store(output + row[:, None] * D + col[None, :], value)


@triton.jit
def index_put_kernel(values, indices, output, D: tl.constexpr, BLOCK_M: tl.constexpr):
    row = tl.arange(0, BLOCK_M)
    col = tl.arange(0, D)
    destination_row = tl.load(indices + row)
    value = tl.load(values + row[:, None] * D + col[None, :])
    tl.store(output + destination_row[:, None] * D + col[None, :], value)


@triton.jit
def indirect_store_kernel(output, indices, values, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    index = tl.load(indices + lane)
    value = tl.load(values + lane)
    tl.store(output + index, value)


def route_kwargs(args: argparse.Namespace, logical_programs: int) -> dict:
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    return {
        "num_warps": 4,
        "compile_on_910_95": True,
        "compile_mode": "simd_simt" if args.route == "auto" else args.route,
        "superblock_factor": args.factor,
        "logical_program_count_hint": logical_programs,
        "physical_vector_core_count_hint": vector_cores,
        "auto_simt_scope_mode": "auto" if args.route == "auto" else "off",
        "auto_simt_scope_dump": str(args.report) if args.route == "auto" else "",
    }


def common_1d_inputs():
    src = torch.arange(280, dtype=torch.float32, device="npu")
    add_src = torch.arange(8, dtype=torch.float32, device="npu")
    indices = torch.tensor(
        [10, 25, 100, 200, 5, 50, 150, 255],
        dtype=torch.int64,
        device="npu",
    )
    output = torch.zeros(8, dtype=torch.float32, device="npu")
    return src, add_src, indices, output


def gather_2d_inputs():
    rows, source_cols, gather_cols = 4, 16, 8
    src = torch.arange(rows * source_cols, dtype=torch.float32, device="npu").reshape(rows, source_cols).contiguous()
    indices = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [15, 14, 13, 12, 11, 10, 9, 8],
            [1, 3, 5, 7, 9, 11, 13, 15],
            [2, 4, 6, 8, 10, 12, 14, 0],
        ],
        dtype=torch.int64,
        device="npu",
    ).contiguous()
    output = torch.empty((rows, gather_cols), dtype=torch.float32, device="npu")
    expected = torch.gather(src, dim=1, index=indices)
    return rows, source_cols, gather_cols, src, indices, output, expected


def build_case(args: argparse.Namespace):
    case = args.case
    if case == "indirect_load_add_b8":
        src, _, indices, output = common_1d_inputs()
        expected = src[indices] + 3.0

        def launch():
            indirect_load_add_kernel[(1, )](src, indices, output, 3.0, BLOCK=8, **route_kwargs(args, 1))

        return launch, output, expected, "indirect_load_add_kernel", {"BLOCK": 8, "grid": 1}

    if case == "indirect_index_div_b8":
        src, add_src, indices, output = common_1d_inputs()
        offset1 = torch.arange(1, 9, dtype=torch.int64, device="npu")
        offset2 = torch.full((8, ), 3, dtype=torch.int64, device="npu")
        expected = src[indices + (offset2 / offset1).to(torch.int64) * 2] + add_src

        def launch():
            indirect_index_div_kernel[(1, )](src, add_src, indices, output, offset1, offset2, BLOCK=8,
                                             **route_kwargs(args, 1))

        return launch, output, expected, "indirect_index_div_kernel", {"BLOCK": 8, "grid": 1}

    if case == "indirect_index_and_b8":
        src, add_src, indices, output = common_1d_inputs()
        expected = src[indices & 63] + add_src

        def launch():
            indirect_index_and_kernel[(1, )](src, add_src, indices, output, BLOCK=8, **route_kwargs(args, 1))

        return launch, output, expected, "indirect_index_and_kernel", {"BLOCK": 8, "grid": 1}

    if case == "indirect_index_maxmin_b8":
        src, add_src, indices, output = common_1d_inputs()
        expected = src[torch.clamp(indices, min=3, max=64)] + add_src

        def launch():
            indirect_index_maxmin_kernel[(1, )](src, add_src, indices, output, BLOCK=8, **route_kwargs(args, 1))

        return launch, output, expected, "indirect_index_maxmin_kernel", {"BLOCK": 8, "grid": 1}

    if case in {"indirect_load_2d_min_m4_n16_k8", "indirect_load_2d_grid_m4_n16_k8"}:
        rows, source_cols, gather_cols, src, indices, output, expected = gather_2d_inputs()
        if case.startswith("indirect_load_2d_min"):

            def launch():
                indirect_load_2d_min_kernel[(1, )](src, indices, output, source_cols, M=rows, K=gather_cols,
                                                   **route_kwargs(args, 1))

            kernel_name = "indirect_load_2d_min_kernel"
            metadata = {"M": rows, "N": source_cols, "K": gather_cols, "grid": 1}
        else:

            def launch():
                indirect_load_2d_grid_kernel[(2, 2)](src, indices, output, source_cols, gather_cols, BLOCK_M=2,
                                                     BLOCK_K=4, **route_kwargs(args, 4))

            kernel_name = "indirect_load_2d_grid_kernel"
            metadata = {"M": rows, "N": source_cols, "K": gather_cols, "grid": [2, 2]}
        return launch, output, expected, kernel_name, metadata

    if case in {"index_select_n64_d32_m8", "index_put_n64_d32_m8"}:
        rows, cols, selected = 64, 32, 8
        indices = torch.tensor([3, 10, 0, 55, 7, 20, 63, 1], dtype=torch.int64, device="npu")
        if case.startswith("index_select"):
            src = torch.arange(rows * cols, dtype=torch.float32, device="npu").reshape(rows, cols)
            output = torch.zeros((selected, cols), dtype=torch.float32, device="npu")
            expected = src[indices]

            def launch():
                index_select_kernel[(1, )](src, indices, output, D=cols, BLOCK_M=selected, **route_kwargs(args, 1))

            kernel_name = "index_select_kernel"
        else:
            values = torch.arange(selected * cols, dtype=torch.float32, device="npu").reshape(selected, cols)
            output = torch.zeros((rows, cols), dtype=torch.float32, device="npu")
            expected = torch.zeros_like(output)
            expected[indices] = values

            def launch():
                output.zero_()
                index_put_kernel[(1, )](values, indices, output, D=cols, BLOCK_M=selected, **route_kwargs(args, 1))

            kernel_name = "index_put_kernel"
        return launch, output, expected, kernel_name, {"N": rows, "D": cols, "M": selected, "grid": 1}

    if case == "indirect_store_b8":
        output = torch.zeros(256, dtype=torch.float32, device="npu")
        indices = torch.tensor(
            [10, 25, 100, 200, 5, 50, 150, 255],
            dtype=torch.int64,
            device="npu",
        )
        values = torch.arange(1, 9, dtype=torch.float32, device="npu")
        expected = torch.zeros_like(output)
        expected[indices] = values

        def launch():
            output.zero_()
            indirect_store_kernel[(1, )](output, indices, values, BLOCK=8, **route_kwargs(args, 1))

        return launch, output, expected, "indirect_store_kernel", {"BLOCK": 8, "grid": 1}
    raise AssertionError(case)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case", required=True, choices=(
            "indirect_load_add_b8",
            "indirect_index_div_b8",
            "indirect_index_and_b8",
            "indirect_index_maxmin_b8",
            "indirect_load_2d_min_m4_n16_k8",
            "indirect_load_2d_grid_m4_n16_k8",
            "index_select_n64_d32_m8",
            "index_put_n64_d32_m8",
            "indirect_store_b8",
        ))
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), default="auto")
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch", type=int, default=20)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()

    launch, actual, expected, kernel_name, metadata = build_case(args)
    launch()
    torch.npu.synchronize()
    max_abs = float((actual - expected).abs().max())
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    if args.compile_only:
        print(
            json.dumps(
                {
                    "case": args.case,
                    "source_pytest": SOURCE_PYTEST,
                    "correct": True,
                    "max_abs": max_abs,
                    "compile_only": True,
                    "report": str(args.report),
                    **metadata,
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
    print(
        json.dumps(
            {
                "case": args.case,
                "source_pytest": SOURCE_PYTEST,
                "route_requested": args.route,
                "compile_mode": "simd_simt" if args.route == "auto" else args.route,
                "factor": args.factor,
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
                "report": str(args.report),
                "profiler": profiler,
                **metadata,
            }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
