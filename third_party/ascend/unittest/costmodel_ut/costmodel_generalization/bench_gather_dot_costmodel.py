#!/usr/bin/env python3
"""Indirect-gather plus dot benchmark for cross-operator model holdout."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu
import acl

# Initialize the ACL SoC query before Triton's lazy target predicate can cache
# a pre-context False value in a fresh benchmark process.
ACL_SOC_NAME = acl.get_soc_name()

import triton
import triton.language as tl
from triton.backends.ascend.utils import is_compile_on_910_95
from triton.runtime import driver

from profile_utils import profile_kernel


@triton.jit
def gather_dot_kernel(
    a_ptr,
    b_ptr,
    indices_ptr,
    output_ptr,
    logical_programs,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    row_lane = tl.arange(0, BLOCK)
    column = tl.arange(0, BLOCK)
    gather_k = tl.load(indices_ptr + column)
    b = tl.load(b_ptr + gather_k[:, None] * stride_bk + column[None, :] * stride_bn)
    for program in range(pid, logical_programs, programs):
        row = program * BLOCK + row_lane
        a = tl.load(a_ptr + row[:, None] * stride_am + gather_k[None, :] * stride_ak)
        result = tl.dot(a, b)
        tl.store(
            output_ptr + row[:, None] * stride_om + column[None, :] * stride_on,
            result,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), required=True)
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--logical-programs", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if not is_compile_on_910_95():
        raise RuntimeError(f"gather-dot costmodel benchmark requires A5, got {ACL_SOC_NAME}")

    torch.manual_seed(19)
    block = 16
    source_k = 256
    rows = args.logical_programs * block
    a = torch.randn((rows, source_k), dtype=torch.float16, device="npu")
    b = torch.randn((source_k, block), dtype=torch.float16, device="npu")
    indices = torch.tensor(
        [10, 25, 100, 200, 5, 50, 150, 255, 1, 2, 3, 4, 6, 7, 8, 9],
        dtype=torch.int32,
        device="npu",
    )
    output = torch.empty((rows, block), dtype=torch.float32, device="npu")
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    grid = min(args.logical_programs, vector_cores)
    compile_mode = "simd_simt" if args.route == "auto" else args.route

    def launch() -> None:
        gather_dot_kernel[(grid, )](
            a,
            b,
            indices,
            output,
            args.logical_programs,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK=block,
            num_warps=4,
            compile_mode=compile_mode,
            superblock_factor=args.factor,
            logical_program_count_hint=args.logical_programs,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.report else "",
        )

    launch()
    torch.npu.synchronize()
    expected = torch.matmul(a[:, indices].float(), b[indices, :].float())
    maximum = float((output - expected).abs().max())
    torch.testing.assert_close(output, expected, rtol=1.0e-2, atol=1.0e-2)
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
    profiler = (profile_kernel(launch, args.profile_dir, "gather_dot_kernel") if args.profile_dir else None)
    print(
        json.dumps(
            {
                "case": "gather_dot",
                "route_requested": args.route,
                "compile_mode": compile_mode,
                "factor": args.factor,
                "logical_programs": args.logical_programs,
                "grid": grid,
                "correct": True,
                "max_abs": maximum,
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
