#!/usr/bin/env python3
"""A5 benchmark for pytest_ut's FBGEMM rowwise FP8 quantization kernel."""

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

SOURCE_PYTEST = ("third_party/ascend/unittest/pytest_ut/"
                 "test_simd_simt_costmodel_cases.py")


@triton.jit
def fbgemm_rowwise_quant_kernel(
    output_ptr,
    output_scale_ptr,
    input_ptr,
    token_indices_ptr,
    expert_indices_ptr,
    scores_ptr,
    scale_ub_ptr,
    stride_t,
    stride_e,
    valid_token_count,
    D: tl.constexpr,
    TL_FP8_DTYPE: tl.constexpr,
    MAX_FP8: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    tl.static_assert(D % BLOCK_D == 0, "D must be a multiple of BLOCK_D")
    output_token = tl.program_id(0)
    valid_count = tl.load(valid_token_count, None, eviction_policy="evict_last")
    if output_token >= valid_count:
        return
    input_token = tl.load(token_indices_ptr + output_token)
    expert = tl.load(expert_indices_ptr + output_token)
    score = tl.load(scores_ptr + input_token * stride_t + expert * stride_e).to(tl.float32)
    offsets = tl.arange(0, BLOCK_D)
    input_block = input_ptr + input_token.to(tl.int64) * D + offsets
    row_max = 0.0
    for _ in range(0, D, BLOCK_D):
        values = (tl.load(input_block, eviction_policy="evict_last").to(tl.float32) * score)
        row_max = tl.maximum(tl.max(tl.abs(values)), row_max)
        input_block += BLOCK_D
    if CLAMP_MAX:
        row_max = tl.clamp(row_max, EPS, tl.load(scale_ub_ptr))
    else:
        row_max = tl.maximum(row_max, EPS)
    scale = MAX_FP8 / row_max
    tl.store(output_scale_ptr + output_token, 1.0 / scale)
    input_block = input_ptr + input_token.to(tl.int64) * D + offsets
    output_block = output_ptr + output_token.to(tl.int64) * D + offsets
    for _ in range(0, D, BLOCK_D):
        values = (tl.load(input_block, eviction_policy="evict_first").to(tl.float32) * score)
        quantized = tl.clamp(values * scale, -MAX_FP8, MAX_FP8).to(TL_FP8_DTYPE)
        tl.store(output_block, quantized, cache_modifier=".cg")
        input_block += BLOCK_D
        output_block += BLOCK_D


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", choices=("auto", "simd", "simt_only"), default="auto")
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--valid", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path)
    args = parser.parse_args()
    if not is_compile_on_910_95():
        raise RuntimeError(f"FBGEMM costmodel benchmark requires A5, got {ACL_SOC_NAME}")

    torch.manual_seed(2)
    input_tensor = torch.randn((args.tokens, args.width), dtype=torch.float16, device="npu")
    token_indices = (torch.arange(args.valid, dtype=torch.int32, device="npu") % args.tokens)
    expert_indices = (torch.arange(args.valid, dtype=torch.int32, device="npu") % args.experts)
    scores = torch.randn((args.tokens, args.experts), dtype=torch.float16, device="npu")
    valid_count = torch.tensor([args.valid], dtype=torch.int32, device="npu")
    scale_ub = torch.tensor([448.0], dtype=torch.float32, device="npu")
    output = torch.empty((args.valid, args.width), dtype=torch.float8_e4m3fn, device="npu")
    output_scale = torch.empty((args.valid, ), dtype=torch.float32, device="npu")
    properties = driver.active.utils.get_device_properties(torch.npu.current_device())
    compile_mode = "simd_simt" if args.route == "auto" else args.route
    launch_options = {
        "num_warps": 4,
        "compile_mode": compile_mode,
        "logical_program_count_hint": args.valid,
        "physical_vector_core_count_hint": int(properties["num_vectorcore"]),
        "physical_aicore_count_hint": int(properties["num_aicore"]),
        "auto_simt_scope_mode": "auto" if args.route == "auto" else "off",
        "auto_simt_scope_dump": str(args.report) if args.route == "auto" else "",
    }
    # Match the pytest guard: Auto materializes the selected F4 route.
    if args.route != "auto":
        launch_options["superblock_factor"] = args.factor

    def launch() -> None:
        fbgemm_rowwise_quant_kernel[(args.valid, )](
            output,
            output_scale,
            input_tensor,
            token_indices,
            expert_indices,
            scores,
            scale_ub,
            scores.stride(0),
            scores.stride(1),
            valid_count,
            D=args.width,
            TL_FP8_DTYPE=tl.float8e4nv,
            MAX_FP8=448.0,
            EPS=1.0e-12,
            CLAMP_MAX=False,
            BLOCK_D=args.width,
            **launch_options,
        )

    launch()
    torch.npu.synchronize()
    gathered = (input_tensor[token_indices].float() * scores[token_indices, expert_indices].float()[:, None])
    row_max = torch.clamp(torch.amax(torch.abs(gathered), dim=1), min=1.0e-12)
    quant_scale = 448.0 / row_max
    expected_scale = 1.0 / quant_scale
    expected = torch.clamp(gathered * quant_scale[:, None], -448.0, 448.0).to(torch.float8_e4m3fn)
    torch.testing.assert_close(output_scale, expected_scale, rtol=2e-3, atol=2e-3)
    output_f32 = output.float()
    expected_f32 = expected.float()
    difference = torch.abs(output_f32 - expected_f32)
    magnitude = torch.abs(expected_f32)
    normal_ulp = torch.pow(
        2.0,
        torch.floor(torch.log2(torch.clamp(magnitude, min=2**-6))) - 3,
    )
    fp8_ulp = torch.where(
        magnitude < 2**-6,
        torch.full_like(magnitude, 2**-9),
        normal_ulp,
    )
    max_ulp_error = float((difference / fp8_ulp).max())
    assert torch.all(difference <= fp8_ulp), max_ulp_error

    if args.compile_only:
        print(
            json.dumps(
                {
                    "case": "pytest_fbgemm_rowwise_quant",
                    "source_pytest": SOURCE_PYTEST,
                    "correct": True,
                    "max_ulp_error": max_ulp_error,
                    "compile_only": True,
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
    profiler = (profile_kernel(launch, args.profile_dir, "fbgemm_rowwise_quant_kernel") if args.profile_dir else None)
    print(
        json.dumps(
            {
                "case": "pytest_fbgemm_rowwise_quant",
                "source_pytest": SOURCE_PYTEST,
                "route_requested": args.route,
                "compile_mode": compile_mode,
                "factor": args.factor,
                "tokens": args.tokens,
                "width": args.width,
                "experts": args.experts,
                "valid": args.valid,
                "correct": True,
                "max_ulp_error": max_ulp_error,
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
            }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
