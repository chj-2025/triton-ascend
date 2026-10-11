#!/usr/bin/env python3
"""Measure source grid-stride resource composition and lowered spill traffic.

The development kernels isolate copy, loaded-index gather, and dot bodies.  A
gather+dot kernel is a validation point: its measurements are not used to fit
the component curves.  Q and the backend SuperBlock factor vary independently
so a source loop iteration cannot be mistaken for an AutoBlockify group.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import statistics
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit
def copy_loop(source, output, logical_programs: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    rows = tl.arange(0, 16)
    columns = tl.arange(0, 16)
    for logical in range(pid, logical_programs, programs):
        offsets = (logical * 16 + rows[:, None]) * 16 + columns[None, :]
        tl.store(output + offsets, tl.load(source + offsets))


@triton.jit
def gather_loop(source, indices, output, logical_programs: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    rows = tl.arange(0, 16)
    columns = tl.arange(0, 16)
    gathered_columns = tl.load(indices + columns)
    for logical in range(pid, logical_programs, programs):
        global_rows = logical * 16 + rows
        values = tl.load(source + global_rows[:, None] * 256 + gathered_columns[None, :])
        tl.store(output + global_rows[:, None] * 16 + columns[None, :], values)


@triton.jit
def dot_loop(a, b, output, logical_programs: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    rows = tl.arange(0, 16)
    columns = tl.arange(0, 16)
    invariant_b = tl.load(b + rows[:, None] * 16 + columns[None, :])
    for logical in range(pid, logical_programs, programs):
        global_rows = logical * 16 + rows
        lhs = tl.load(a + global_rows[:, None] * 16 + columns[None, :])
        result = tl.dot(lhs, invariant_b)
        tl.store(output + global_rows[:, None] * 16 + columns[None, :], result)


@triton.jit
def gather_dot_loop(a, b, indices, output, logical_programs: tl.constexpr):
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    rows = tl.arange(0, 16)
    columns = tl.arange(0, 16)
    gathered_k = tl.load(indices + columns)
    invariant_b = tl.load(b + gathered_k[:, None] * 16 + columns[None, :])
    for logical in range(pid, logical_programs, programs):
        global_rows = logical * 16 + rows
        lhs = tl.load(a + global_rows[:, None] * 256 + gathered_k[None, :])
        result = tl.dot(lhs, invariant_b)
        tl.store(output + global_rows[:, None] * 16 + columns[None, :], result)


def profile_kernel(launch, output: Path, kernel_name: str) -> list[float]:
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=3, active=20),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output)),
    ) as profiler:
        for _ in range(24):
            launch()
            profiler.step()
    torch.npu.synchronize()
    samples = []
    for path in output.rglob("kernel_details.csv"):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("Name") == kernel_name and row.get("Duration(us)"):
                    samples.append(float(row["Duration(us)"]))
    if len(samples) < 20:
        raise RuntimeError(f"expected at least 20 profiler samples, got {len(samples)}")
    return samples[:20]


def event_kernel(launch, batch: int = 50) -> list[float]:
    """Measure batched launches with device events and return us/launch."""
    interface = driver.active.get_device_interface()
    starts = [interface.Event(enable_timing=True) for _ in range(20)]
    ends = [interface.Event(enable_timing=True) for _ in range(20)]
    for start, end in zip(starts, ends):
        start.record()
        for _ in range(batch):
            launch()
        end.record()
    interface.synchronize()
    return [start.elapsed_time(end) * 1000.0 / batch for start, end in zip(starts, ends)]


def lowered_counts(npuir: str) -> dict[str, int]:
    upper = npuir.upper()
    return {
        "ldk": len(re.findall(r"\b(?:SIMT_)?LDK\b", upper)),
        "stk": len(re.findall(r"\b(?:SIMT_)?STK\b", upper)),
        "lds": len(re.findall(r"\b(?:SIMT_)?LDS\b", upper)),
        "sts": len(re.findall(r"\b(?:SIMT_)?STS\b", upper)),
        "sync_block": len(re.findall(r"\bSYNC_BLOCK\b", upper)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=("copy", "gather", "dot", "gather_dot"), required=True)
    parser.add_argument("--q", type=int, choices=(1, 2, 3, 4, 5, 8), required=True)
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--compile-mode", choices=("simt_only", "simd_simt"), default="simt_only")
    parser.add_argument("--timer", choices=("profiler", "event", "route_only"), default="profiler")
    parser.add_argument("--profile-root", type=Path, help="Optional profiler scratch root (for example /dev/shm).")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    logical_programs = vector_cores * args.q
    torch.manual_seed(29)
    rows = logical_programs * 16
    source = torch.randn((rows, 256), dtype=torch.float16, device="npu")
    dense = torch.randn((rows, 16), dtype=torch.float16, device="npu")
    rhs = torch.randn((256, 16), dtype=torch.float16, device="npu")
    indices = torch.tensor([10, 25, 100, 200, 5, 50, 150, 255, 1, 2, 3, 4, 6, 7, 8, 9], dtype=torch.int32, device="npu")
    output = torch.empty((rows, 16), dtype=torch.float32, device="npu")
    route_report = args.output / "route.json"

    launch_options = dict(
        num_warps=4,
        compile_mode=args.compile_mode,
        superblock_factor=args.factor,
        enable_auto_blockify=True,
        logical_program_count_hint=logical_programs,
        physical_vector_core_count_hint=vector_cores,
        auto_simt_scope_mode="auto" if args.compile_mode == "simd_simt" else "off",
        auto_simt_scope_dump=str(route_report) if args.compile_mode == "simd_simt" else "",
    )

    if args.variant == "copy":

        def launch():
            return copy_loop[(vector_cores, )](dense, output, logical_programs, **launch_options)

        expected = dense.float()
        kernel_name = "copy_loop"
    elif args.variant == "gather":

        def launch():
            return gather_loop[(vector_cores, )](source, indices, output, logical_programs, **launch_options)

        expected = source[:, indices].float()
        kernel_name = "gather_loop"
    elif args.variant == "dot":

        def launch():
            return dot_loop[(vector_cores, )](dense, rhs[:16], output, logical_programs, **launch_options)

        expected = torch.matmul(dense.float(), rhs[:16].float())
        kernel_name = "dot_loop"
    else:

        def launch():
            return gather_dot_loop[(vector_cores, )](source, rhs, indices, output, logical_programs, **launch_options)

        expected = torch.matmul(source[:, indices].float(), rhs[indices].float())
        kernel_name = "gather_dot_loop"

    binary = launch()
    torch.npu.synchronize()
    maximum_error = float((output - expected).abs().max())
    torch.testing.assert_close(output, expected, rtol=1.0e-2, atol=1.0e-2)
    for key in ("ttir", "ttadapter", "npuir"):
        text = binary.asm.get(key)
        if isinstance(text, str):
            (args.output / f"kernel.{key}").write_text(text)
    if args.timer == "route_only":
        if not route_report.exists():
            raise RuntimeError("route-only probe did not emit route.json")
        result = {
            "variant":
            args.variant,
            "q":
            args.q,
            "requested_factor":
            args.factor,
            "compile_mode":
            args.compile_mode,
            "timer":
            args.timer,
            "physical_programs":
            vector_cores,
            "logical_programs":
            logical_programs,
            "maximum_error":
            maximum_error,
            "lowered_instruction_counts":
            lowered_counts(binary.asm.get("npuir") if isinstance(binary.asm.get("npuir"), str) else ""),
            "route_report":
            str(route_report),
            "correct":
            True,
        }
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    for _ in range(20):
        launch()
    torch.npu.synchronize()
    profile_output = ((args.profile_root / args.output.name) if args.profile_root else (args.output / "profile"))
    if args.timer == "profiler":
        profile_output.mkdir(parents=True, exist_ok=False)
        samples = profile_kernel(launch, profile_output, kernel_name)
        details = next(profile_output.rglob("kernel_details.csv"), None)
        if details is not None:
            shutil.copy2(details, args.output / "kernel_details.csv")
    else:
        samples = event_kernel(launch)
    npuir = binary.asm.get("npuir") if isinstance(binary.asm.get("npuir"), str) else ""
    result = {
        "variant": args.variant,
        "q": args.q,
        "requested_factor": args.factor,
        "compile_mode": args.compile_mode,
        "timer": args.timer,
        "profile_output": str(profile_output) if args.timer == "profiler" else None,
        "physical_programs": vector_cores,
        "logical_programs": logical_programs,
        "average_us": statistics.fmean(samples),
        "median_us": statistics.median(samples),
        "samples_us": samples,
        "maximum_error": maximum_error,
        "lowered_instruction_counts": lowered_counts(npuir),
        "route_report": str(route_report) if route_report.exists() else None,
        "correct": True,
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
