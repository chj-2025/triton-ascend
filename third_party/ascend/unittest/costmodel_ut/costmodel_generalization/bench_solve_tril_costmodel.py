#!/usr/bin/env python3
"""Held-out triangular-recurrence benchmark for the generic route model.

The benchmark launches the unannotated production kernel directly so auto
mode must derive its route from TTIR structure.  The operator name and tensor
shape are evidence labels only; neither is exposed as a model feature.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import sys

import torch
import triton
from triton.runtime import driver

from profile_utils import profile_kernel

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "simd_simt"))
import run_solve_tril_costmodel as solve_driver  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("auto", "simd", "simt_only", "mixed"), required=True)
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--bt", type=int, choices=(16, 32, 64), default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--timing-batch", type=int, default=1)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if args.tokens % args.bt:
        raise ValueError("--tokens must be divisible by --bt")

    torch.manual_seed(0)
    chunks = args.tokens // args.bt
    lower = torch.tril(torch.randn((args.bt, args.bt), dtype=torch.float32), diagonal=-1) * 0.01
    identity = torch.eye(args.bt, dtype=lower.dtype)
    expected = torch.linalg.inv(identity + lower).to("npu")
    source = torch.empty(
        (args.batch, args.tokens, args.heads, args.bt),
        dtype=torch.float32,
        device="npu",
    )
    source.view(args.batch, chunks, args.bt, args.heads,
                args.bt).copy_(lower.reshape(1, 1, args.bt, 1, args.bt).to("npu"))
    output = torch.empty_like(source)

    case = solve_driver._load_case("solve_tril.py")
    kernels = {
        16: case.solve_tril_16x16_kernel,
        32: case.merge_16x16_to_32x32_inverse_kernel,
        64: case.merge_16x16_to_64x64_inverse_kernel,
    }
    kernel = kernels[args.bt]
    if args.route == "mixed":
        if args.bt != 64:
            raise ValueError("the held-out manual Mixed reference is defined for BT=64")
        kernel = case.merge_16x16_to_64x64_inverse_kernel_mix
    kernel_name = kernel.fn.__name__ if hasattr(kernel, "fn") else kernel.__name__
    nt = chunks
    logical_programs = nt * args.batch * args.heads
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])
    compile_mode = "simd_simt" if args.route in ("auto", "mixed") else args.route

    def launch() -> None:
        kernel[(nt, args.batch * args.heads)](
            A=source,
            Ai=output,
            cu_seqlens=None,
            chunk_indices=None,
            T=args.tokens,
            H=args.heads,
            BT=args.bt,
            USE_TMA=False,
            DOT_PRECISION="ieee",
            # Keep the production solve_tril launch geometry: F1/F2/F4 map to
            # 4/8/16 active SIMT warps on the physical VF.
            num_warps=4,
            compile_mode=compile_mode,
            superblock_factor=args.factor,
            logical_program_count_hint=logical_programs,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto" if args.route == "auto" else "off",
            auto_simt_scope_dump=str(args.report) if args.report else "",
        )

    launch()
    torch.npu.synchronize()
    output_blocks = output.view(args.batch, chunks, args.bt, args.heads, args.bt)
    probes = {
        (0, 0, 0),
        (args.batch // 2, chunks // 2, args.heads // 2),
        (args.batch - 1, chunks - 1, args.heads - 1),
    }
    max_abs = 0.0
    for batch_id, chunk_id, head_id in probes:
        actual = output_blocks[batch_id, chunk_id, :, head_id, :]
        max_abs = max(max_abs, float((actual - expected).abs().max()))
        torch.testing.assert_close(actual, expected, atol=3.0e-2, rtol=3.0e-2)
    for _ in range(args.warmup):
        launch()
    torch.npu.synchronize()
    samples = []
    for _ in range(args.repeats):
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        begin.record()
        for _ in range(args.timing_batch):
            launch()
        end.record()
        torch.npu.synchronize()
        samples.append(float(begin.elapsed_time(end)) * 1000.0 / args.timing_batch)
    samples.sort()
    profiler = (profile_kernel(launch, args.profile_dir, kernel_name) if args.profile_dir else None)
    print(
        json.dumps(
            {
                "case": f"triangular_recurrence_b{args.batch}_t{args.tokens}_h{args.heads}_bt{args.bt}",
                "route_requested": args.route,
                "compile_mode": compile_mode,
                "factor": args.factor,
                "shape": {
                    "batch": args.batch,
                    "tokens": args.tokens,
                    "heads": args.heads,
                    "bt": args.bt,
                },
                "logical_programs": logical_programs,
                "grid": [nt, args.batch * args.heads],
                "vector_cores": vector_cores,
                "correct": True,
                "max_abs": max_abs,
                "correctness_probes": sorted(probes),
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
