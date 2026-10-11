#!/usr/bin/env python3
"""Generic contiguous-memory benchmark for SIMD and SIMT route formulas."""

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
def memory_stream_kernel(
    x0,
    x1,
    x2,
    x3,
    output,
    n_elements,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
    NLOADS: tl.constexpr,
):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    for block in range(pid, n_blocks, programs):
        offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        value = tl.load(x0 + offsets, mask=mask, other=0.0)
        if NLOADS >= 2:
            value += tl.load(x1 + offsets, mask=mask, other=0.0)
        if NLOADS >= 3:
            value += tl.load(x2 + offsets, mask=mask, other=0.0)
        if NLOADS >= 4:
            value += tl.load(x3 + offsets, mask=mask, other=0.0)
        tl.store(output + offsets, value, mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), required=True)
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--loads", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--n-elements", type=int, default=2097152)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()

    torch.manual_seed(0)
    inputs = [torch.randn(args.n_elements, dtype=torch.float32, device="npu") for _ in range(4)]
    output = torch.empty_like(inputs[0])
    n_blocks = triton.cdiv(args.n_elements, args.block_size)
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    grid = min(n_blocks, vector_cores)
    compile_mode = "simd_simt" if args.route == "auto" else args.route

    def launch() -> None:
        memory_stream_kernel[(grid, )](
            *inputs,
            output,
            args.n_elements,
            n_blocks,
            BLOCK_SIZE=args.block_size,
            NLOADS=args.loads,
            num_warps=4,
            compile_mode=compile_mode,
            superblock_factor=args.factor,
            logical_program_count_hint=n_blocks,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.report else "",
            multibuffer=True,
            enable_vf_fusion=True,
        )

    launch()
    torch.npu.synchronize()
    expected = sum(inputs[:args.loads])
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
    profiler = (profile_kernel(launch, args.profile_dir, "memory_stream_kernel") if args.profile_dir else None)
    print(
        json.dumps(
            {
                "case": f"memory_stream_l{args.loads}_n{args.n_elements}",
                "route_requested": args.route,
                "compile_mode": compile_mode,
                "factor": args.factor,
                "loads": args.loads,
                "stores": 1,
                "n_elements": args.n_elements,
                "block_size": args.block_size,
                "logical_programs": n_blocks,
                "grid": grid,
                "vector_cores": vector_cores,
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
            },
            indent=2,
            sort_keys=True,
        ))


if __name__ == "__main__":
    main()
