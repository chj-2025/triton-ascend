#!/usr/bin/env python3
"""Measure contiguous and loaded-index atomic address streams."""

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
def atomic_address_kernel(
    input_ptr,
    index_ptr,
    output_ptr,
    elements,
    BLOCK: tl.constexpr,
    INDIRECT: tl.constexpr,
):
    program = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    offset = program * BLOCK + lane
    mask = offset < elements
    value = tl.load(input_ptr + offset, mask=mask, other=0)
    destination = tl.load(index_ptr + offset, mask=mask, other=0) if INDIRECT else offset
    tl.atomic_add(output_ptr + destination, value, mask=mask)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", choices=("continuous", "discrete"), required=True)
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), required=True)
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--grid", type=int, default=56)
    parser.add_argument("--block", type=int, choices=(64, 128, 256, 512), default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()

    elements = args.grid * args.block
    host_input = (torch.arange(elements, dtype=torch.int32) % 17) + 1
    # An affine permutation is bijective, so the discrete case changes only
    # address provenance/locality and has deterministic race-free validation.
    host_index = (torch.arange(elements, dtype=torch.int64) * 8191 + 37) % elements
    source = host_input.npu()
    indices = host_index.to(torch.int32).npu()
    output = torch.zeros(elements, dtype=torch.int32, device="npu")
    expected = torch.empty_like(host_input)
    if args.address == "discrete":
        expected[host_index] = host_input
    else:
        expected.copy_(host_input)

    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    compile_mode = "simd_simt" if args.route == "auto" else args.route

    def launch() -> None:
        atomic_address_kernel[(args.grid, )](
            source,
            indices,
            output,
            elements,
            BLOCK=args.block,
            INDIRECT=args.address == "discrete",
            num_warps=4,
            compile_on_910_95=True,
            compile_mode=compile_mode,
            superblock_factor=args.factor,
            logical_program_count_hint=args.grid,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.report else "",
        )

    output.zero_()
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
    profiler = (profile_kernel(launch, args.profile_dir, "atomic_address_kernel") if args.profile_dir else None)
    print(
        json.dumps(
            {
                "case": "atomic_address",
                "address": args.address,
                "route_requested": args.route,
                "compile_mode": compile_mode,
                "factor": args.factor,
                "grid": args.grid,
                "block": args.block,
                "logical_programs": args.grid,
                "elements": elements,
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
