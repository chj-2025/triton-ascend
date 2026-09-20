#!/usr/bin/env python3
"""Measure continuous load/store scaling for physical SIMD/SIMT groups."""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
import triton.runtime.driver as driver


@triton.jit
def contiguous_superblock_probe(source, output, block: tl.constexpr,
                                operation_count: tl.constexpr,
                                direction: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, block)
    if direction == 0:
        value = tl.zeros((block,), tl.float32)
        for op in tl.static_range(operation_count):
            address = (pid * operation_count + op) * block + offsets
            value += tl.load(source + address)
        # A fixed sink keeps every load observable. It disappears when the
        # operation-count slope is taken.
        tl.store(output + pid * block + offsets, value)
    else:
        value = tl.load(source + pid * block + offsets)
        for op in tl.static_range(operation_count):
            address = (pid * operation_count + op) * block + offsets
            tl.store(output + address, value + op)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("simd", "simt_only"), required=True)
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--direction", choices=("load", "store"), required=True)
    parser.add_argument("--block", type=int, choices=(64, 256, 1024), required=True)
    parser.add_argument("--operation-count", type=int,
                        choices=(0, 1, 2, 4, 8, 16, 32),
                        required=True)
    parser.add_argument("--logical-programs", type=int, default=0)
    parser.add_argument("--timer", choices=("profiler", "event"),
                        default="profiler")
    args = parser.parse_args()
    if args.mode == "simd" and args.factor != 1:
        parser.error("SIMD measurements require --factor=1")
    args.output.mkdir(parents=True, exist_ok=False)

    vector_cores = int(driver.active.utils.get_device_properties(
        torch.npu.current_device())["num_vectorcore"])
    logical_programs = args.logical_programs or vector_cores * 4
    if args.direction == "load":
        source_elements = logical_programs * max(1, args.operation_count) * args.block
        output_elements = logical_programs * args.block
    else:
        source_elements = logical_programs * args.block
        output_elements = (logical_programs * max(1, args.operation_count) *
                           args.block)
    torch.manual_seed(23)
    source = torch.randn(source_elements, dtype=torch.float32, device="npu")
    output = torch.empty(output_elements, dtype=torch.float32, device="npu")
    direction = 0 if args.direction == "load" else 1

    def launch():
        options = {"compile_mode": args.mode, "num_warps": 4}
        if args.mode == "simt_only":
            options["superblock_factor"] = args.factor
        return contiguous_superblock_probe[(logical_programs,)](
            source, output, args.block, args.operation_count, direction,
            **options)

    binary = launch()
    torch.npu.synchronize()
    if args.operation_count:
        if args.direction == "load":
            expected = source.reshape(logical_programs, args.operation_count,
                                      args.block).sum(dim=1)
            torch.testing.assert_close(
                output.reshape(logical_programs, args.block), expected)
        else:
            expected = (source.reshape(logical_programs, 1, args.block) +
                        torch.arange(args.operation_count, device="npu").reshape(
                            1, args.operation_count, 1))
            torch.testing.assert_close(
                output.reshape(logical_programs, args.operation_count, args.block),
                expected)
    for key in ("ttir", "ttadapter", "npuir"):
        value = binary.asm.get(key)
        if isinstance(value, str):
            (args.output / ("kernel." + key)).write_text(value)

    for _ in range(20):
        launch()
    torch.npu.synchronize()
    samples = []
    if args.timer == "profiler":
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=3, active=20),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(args.output / "profile")),
        ) as profiler:
            for _ in range(24):
                launch()
                profiler.step()
        torch.npu.synchronize()
        for path in args.output.rglob("kernel_details.csv"):
            with path.open(newline="") as stream:
                samples.extend(
                    float(row["Duration(us)"])
                    for row in csv.DictReader(stream)
                    if row.get("Duration(us)"))
    else:
        interface = driver.active.get_device_interface()
        starts = [interface.Event(enable_timing=True) for _ in range(20)]
        ends = [interface.Event(enable_timing=True) for _ in range(20)]
        for start, end in zip(starts, ends):
            start.record()
            launch()
            end.record()
        interface.synchronize()
        samples = [start.elapsed_time(end) * 1000.0
                   for start, end in zip(starts, ends)]
    samples = samples[:20]
    if len(samples) < 10:
        raise RuntimeError(f"expected at least 10 samples, got {len(samples)}")
    physical_programs = (logical_programs + args.factor - 1) // args.factor
    result = {
        "mode": args.mode,
        "factor": args.factor,
        "direction": args.direction,
        "block_elements": args.block,
        "bytes_per_operation_per_logical_program": args.block * 4,
        "operation_count": args.operation_count,
        "logical_programs": logical_programs,
        "physical_programs": physical_programs,
        "vector_cores": vector_cores,
        "expected_waves": (physical_programs + vector_cores - 1) // vector_cores,
        "timer": args.timer,
        "sample_count": len(samples),
        "median_us": statistics.median(samples),
        "samples_us": samples,
        "correct": True,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
