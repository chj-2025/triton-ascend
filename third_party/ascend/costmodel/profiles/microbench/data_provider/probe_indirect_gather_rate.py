#!/usr/bin/env python3
"""Measure loaded-index FP16 gather slopes with a single final store."""

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
def indirect_gather_probe(source, indices, output, baseline, rows: tl.constexpr,
                          source_width: tl.constexpr, count: tl.constexpr):
    pid = tl.program_id(0)
    row = pid * rows + tl.arange(0, rows)
    col = tl.arange(0, 16)
    gathered_col = tl.load(indices + col)
    # A runtime zero preserves a common scalar baseline and final store across
    # the measured count>0 variants used by the incremental-slope fit.
    acc = tl.zeros((rows, 16), tl.float32) + tl.load(baseline)
    plane = tl.num_programs(0) * rows * source_width
    for repeat in tl.static_range(count):
        offsets = repeat * plane + row[:, None] * source_width + gathered_col[None, :]
        acc += tl.load(source + offsets).to(tl.float32)
    tl.store(output + row[:, None] * 16 + col[None, :], acc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("simd", "simt_only"), required=True)
    parser.add_argument("--rows", type=int, choices=(16, 32), required=True)
    parser.add_argument("--count", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--index-pattern", choices=("sequential", "scattered"),
                        default="scattered")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    programs = int(driver.active.utils.get_device_properties(
        torch.npu.current_device())["num_vectorcore"])
    width = 256
    torch.manual_seed(7)
    source = torch.randn((max(1, args.count), programs * args.rows, width),
                         dtype=torch.float16, device="npu")
    index_values = (list(range(16)) if args.index_pattern == "sequential" else
                    [10, 25, 100, 200, 5, 50, 150, 255,
                     1, 2, 3, 4, 6, 7, 8, 9])
    indices = torch.tensor(index_values, dtype=torch.int32, device="npu")
    output = torch.empty((programs * args.rows, 16), dtype=torch.float32,
                         device="npu")
    baseline = torch.zeros((1,), dtype=torch.float32, device="npu")

    def launch():
        return indirect_gather_probe[(programs,)](
            source, indices, output, baseline, args.rows, width, args.count,
            compile_mode=args.mode, num_warps=4)

    binary = launch()
    expected = source[:, :, indices].float().sum(dim=0)
    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)
    for key in ("ttir", "ttadapter", "npuir"):
        value = binary.asm.get(key)
        if isinstance(value, str):
            (args.output / ("kernel." + key)).write_text(value)
    for _ in range(20):
        launch()
    torch.npu.synchronize()
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
    samples = []
    for path in args.output.rglob("kernel_details.csv"):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("Duration(us)"):
                    samples.append(float(row["Duration(us)"]))
    if len(samples) not in (20, 21):
        raise RuntimeError(f"expected 20 samples, got {len(samples)}")
    result = dict(mode=args.mode, rows=args.rows, count=args.count,
                  index_pattern=args.index_pattern,
                  num_warps=4, logical_programs=programs,
                  warp_instructions_per_gather=args.rows * 16 // 32,
                  median_us=statistics.median(samples[:20]),
                  samples_us=samples[:20], correct=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
