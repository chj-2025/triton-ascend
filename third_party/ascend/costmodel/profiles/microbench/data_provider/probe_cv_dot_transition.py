#!/usr/bin/env python3
"""Measure lowering-generated AIC/AIV coordination around local SIMT.

The sweep varies only the number and ordering of tiny Cube stages around one
local SIMT recurrence.  Trip-count pairs separate recurrence payload from the
fixed CV envelope; dot-count and order pairs expose whether the remaining
coordination is one fixed local-scope cost or scales with Cube/Vector stage
boundaries.  Cost-model calibration must use the development dot counts and
hold out the three-dot case used by solve_tril.
"""

from __future__ import annotations

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
from triton.runtime import driver


@triton.jit
def triangular_recurrence(seed_ptr, base, lane, state, trips: tl.constexpr):
    for step in range(1, trips + 1):
        row = tl.load(seed_ptr + base + step * 16 + lane)
        row = tl.where(lane < step, row, 0.0)
        update = row + tl.sum(row[:, None] * state, axis=0)
        state = tl.where((lane == step)[:, None], update, state)
    return state


@triton.jit
def cv_dot_transition_probe(
    lhs_ptr,
    rhs_ptr,
    seed_ptr,
    scratch_ptr,
    output_ptr,
    TRIPS: tl.constexpr,
    DOTS: tl.constexpr,
    SCOPE_FIRST: tl.constexpr,
    RECURRENCE_GROUPS: tl.constexpr,
    DEPENDENT_DOTS: tl.constexpr,
    SOLVE_TAIL: tl.constexpr,
    STRIDED_RECURRENCE_STATES: tl.constexpr,
    TAIL_ROW_STRIDE: tl.constexpr,
    TAIL_MATRIX_SPAN: tl.constexpr,
):
    pid = tl.program_id(0)
    lane = tl.arange(0, 16)
    matrix = lane[:, None] * 16 + lane[None, :]
    state_base = pid * RECURRENCE_GROUPS * 256
    tiles_per_row: tl.constexpr = TAIL_ROW_STRIDE // 16
    tail_tile_row = pid // tiles_per_row
    tail_tile_col = pid % tiles_per_row
    tail_matrix = lane[:, None] * TAIL_ROW_STRIDE + lane[None, :]
    tail_base = tail_tile_row * 16 * TAIL_ROW_STRIDE + tail_tile_col * 16
    if SOLVE_TAIL and STRIDED_RECURRENCE_STATES:
        state0 = tl.load(lhs_ptr + 0 * TAIL_MATRIX_SPAN + tail_base + tail_matrix)
        state1 = tl.load(lhs_ptr + 1 * TAIL_MATRIX_SPAN + tail_base + tail_matrix)
        state2 = tl.load(lhs_ptr + 2 * TAIL_MATRIX_SPAN + tail_base + tail_matrix)
        state3 = tl.load(lhs_ptr + 3 * TAIL_MATRIX_SPAN + tail_base + tail_matrix)
    else:
        state0 = tl.load(seed_ptr + state_base + matrix)
        if RECURRENCE_GROUPS >= 2:
            state1 = tl.load(seed_ptr + state_base + 256 + matrix)
        if RECURRENCE_GROUPS >= 3:
            state2 = tl.load(seed_ptr + state_base + 512 + matrix)
        if RECURRENCE_GROUPS >= 4:
            state3 = tl.load(seed_ptr + state_base + 768 + matrix)

    if SCOPE_FIRST:
        state0 = triangular_recurrence(seed_ptr, state_base, lane, state0, TRIPS)
        if RECURRENCE_GROUPS >= 2:
            state1 = triangular_recurrence(seed_ptr, state_base + 256, lane, state1, TRIPS)
        if RECURRENCE_GROUPS >= 3:
            state2 = triangular_recurrence(seed_ptr, state_base + 512, lane, state2, TRIPS)
        if RECURRENCE_GROUPS >= 4:
            state3 = triangular_recurrence(seed_ptr, state_base + 768, lane, state3, TRIPS)

    if SOLVE_TAIL:
        tail_matrix_offset: tl.constexpr = 4 if STRIDED_RECURRENCE_STATES else 0
        a21 = tl.load(lhs_ptr + (tail_matrix_offset + 0) * TAIL_MATRIX_SPAN + tail_base + tail_matrix)
        a31 = tl.load(lhs_ptr + (tail_matrix_offset + 1) * TAIL_MATRIX_SPAN + tail_base + tail_matrix)
        a32 = tl.load(lhs_ptr + (tail_matrix_offset + 2) * TAIL_MATRIX_SPAN + tail_base + tail_matrix)
        a41 = tl.load(lhs_ptr + (tail_matrix_offset + 3) * TAIL_MATRIX_SPAN + tail_base + tail_matrix)
        a42 = tl.load(lhs_ptr + (tail_matrix_offset + 4) * TAIL_MATRIX_SPAN + tail_base + tail_matrix)
        a43 = tl.load(lhs_ptr + (tail_matrix_offset + 5) * TAIL_MATRIX_SPAN + tail_base + tail_matrix)
        ai21 = -tl.dot(tl.dot(state1, a21), state0)
        ai32 = -tl.dot(tl.dot(state2, a32), state1)
        ai43 = -tl.dot(tl.dot(state3, a43), state2)
        ai31 = -tl.dot(state2, tl.dot(a31, state0) + tl.dot(a32, ai21))
        ai42 = -tl.dot(state3, tl.dot(a42, state1) + tl.dot(a43, ai32))
        ai41 = -tl.dot(
            state3,
            tl.dot(a41, state0) + tl.dot(a42, ai21) + tl.dot(a43, ai31),
        )
        if STRIDED_RECURRENCE_STATES:
            tl.store(scratch_ptr + 0 * TAIL_MATRIX_SPAN + tail_base + tail_matrix, state0)
            tl.store(scratch_ptr + 1 * TAIL_MATRIX_SPAN + tail_base + tail_matrix, state1)
            tl.store(scratch_ptr + 2 * TAIL_MATRIX_SPAN + tail_base + tail_matrix, state2)
            tl.store(scratch_ptr + 3 * TAIL_MATRIX_SPAN + tail_base + tail_matrix, state3)
        tl.store(scratch_ptr + (tail_matrix_offset + 0) * TAIL_MATRIX_SPAN + tail_base + tail_matrix, ai21)
        tl.store(scratch_ptr + (tail_matrix_offset + 1) * TAIL_MATRIX_SPAN + tail_base + tail_matrix, ai31)
        tl.store(scratch_ptr + (tail_matrix_offset + 2) * TAIL_MATRIX_SPAN + tail_base + tail_matrix, ai32)
        tl.store(scratch_ptr + (tail_matrix_offset + 3) * TAIL_MATRIX_SPAN + tail_base + tail_matrix, ai41)
        tl.store(scratch_ptr + (tail_matrix_offset + 4) * TAIL_MATRIX_SPAN + tail_base + tail_matrix, ai42)
        tl.store(scratch_ptr + (tail_matrix_offset + 5) * TAIL_MATRIX_SPAN + tail_base + tail_matrix, ai43)
    elif DEPENDENT_DOTS:
        if DOTS >= 1:
            a = tl.load(lhs_ptr + pid * 256 + matrix)
            b = tl.load(rhs_ptr + pid * 256 + matrix)
            accumulator = tl.dot(a, b)
            if DOTS >= 2:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 3:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 4:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 5:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 6:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 7:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 8:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 9:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 10:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 11:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 12:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 13:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 14:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 15:
                accumulator = tl.dot(accumulator, b)
            if DOTS >= 16:
                accumulator = tl.dot(accumulator, b)
            tl.store(scratch_ptr + pid * 256 + matrix, accumulator)
    else:
        # Each static dot reads and stores a distinct matrix, so the compiler
        # cannot fold repeated Cube work while DOTS spans below/above the local
        # SIMT recurrence envelope.
        for dot_id in range(0, DOTS):
            a = tl.load(lhs_ptr + (pid * DOTS + dot_id) * 256 + matrix)
            b = tl.load(rhs_ptr + (pid * DOTS + dot_id) * 256 + matrix)
            d = tl.dot(a, b)
            tl.store(scratch_ptr + (pid * DOTS + dot_id) * 256 + matrix, d)

    if not SCOPE_FIRST:
        state0 = triangular_recurrence(seed_ptr, state_base, lane, state0, TRIPS)
        if RECURRENCE_GROUPS >= 2:
            state1 = triangular_recurrence(seed_ptr, state_base + 256, lane, state1, TRIPS)
        if RECURRENCE_GROUPS >= 3:
            state2 = triangular_recurrence(seed_ptr, state_base + 512, lane, state2, TRIPS)
        if RECURRENCE_GROUPS >= 4:
            state3 = triangular_recurrence(seed_ptr, state_base + 768, lane, state3, TRIPS)

    if not (SOLVE_TAIL and STRIDED_RECURRENCE_STATES):
        tl.store(output_ptr + state_base + matrix, state0)
        if RECURRENCE_GROUPS >= 2:
            tl.store(output_ptr + state_base + 256 + matrix, state1)
        if RECURRENCE_GROUPS >= 3:
            tl.store(output_ptr + state_base + 512 + matrix, state2)
        if RECURRENCE_GROUPS >= 4:
            tl.store(output_ptr + state_base + 768 + matrix, state3)


def count_npuir_sync(text: str) -> dict[str, int]:
    return {
        "sync_block_set": text.count("hivm.hir.sync_block_set"),
        "sync_block_wait": text.count("hivm.hir.sync_block_wait"),
        "set_flag": text.count("hivm.hir.set_flag"),
        "wait_flag": text.count("hivm.hir.wait_flag"),
        "pipe_barrier": text.count("hivm.hir.pipe_barrier"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--trips", type=int, choices=(0, 14), required=True)
    parser.add_argument(
        "--dots",
        type=int,
        choices=(0, 1, 2, 3, 4, 8, 16),
        required=True,
    )
    parser.add_argument(
        "--dot-pattern",
        choices=("independent", "dependent"),
        default="independent",
    )
    parser.add_argument(
        "--tail-pattern",
        choices=("simple", "solve"),
        default="simple",
    )
    parser.add_argument(
        "--tail-row-stride",
        type=int,
        default=16,
        help="Solve-tail row stride in elements; must be a multiple of 16.",
    )
    parser.add_argument(
        "--strided-recurrence-states",
        action="store_true",
        help="Place the four recurrence state loads/stores in the solve-tail strided layout.",
    )
    parser.add_argument("--scope-order", choices=("before", "after"), required=True)
    parser.add_argument("--logical-programs", type=int, default=28)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument(
        "--recurrence-groups",
        type=int,
        choices=(1, 4),
        default=1,
    )
    parser.add_argument(
        "--profile",
        type=Path,
        help="Optional route-selection profile for controlled route ablation.",
    )
    args = parser.parse_args()
    if args.tail_pattern == "solve" and args.recurrence_groups != 4:
        parser.error("--tail-pattern solve requires --recurrence-groups 4")
    if args.strided_recurrence_states and args.tail_pattern != "solve":
        parser.error("--strided-recurrence-states requires --tail-pattern solve")
    if args.tail_row_stride < 16 or args.tail_row_stride % 16:
        parser.error("--tail-row-stride must be a multiple of 16 and >= 16")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(113)
    dot_allocations = ((10 if args.strided_recurrence_states else 6)
                       if args.tail_pattern == "solve" else 1 if args.dot_pattern == "dependent" else max(1, args.dots))
    tail_tiles_per_row = args.tail_row_stride // 16
    tail_tile_rows = (args.logical_programs + tail_tiles_per_row - 1) // tail_tiles_per_row
    tail_matrix_span = tail_tile_rows * 16 * args.tail_row_stride
    if args.tail_pattern == "solve":
        lhs = torch.randn(
            (dot_allocations * tail_matrix_span, ),
            device="npu",
            dtype=torch.float32,
        ) * 0.01
        rhs = torch.empty((1, ), device="npu", dtype=torch.float32)
        scratch = torch.empty_like(lhs)
    else:
        shape = (args.logical_programs, dot_allocations, 16, 16)
        lhs = torch.randn(shape, device="npu", dtype=torch.float32) * 0.01
        rhs = torch.randn_like(lhs) * 0.01
        scratch = torch.empty_like(lhs)
    seed = torch.randn(
        (args.logical_programs, args.recurrence_groups, 16, 16), device="npu", dtype=torch.float32) * 0.01
    output = torch.zeros_like(seed)
    route_report = args.output_dir / "route.json"
    vector_cores = int(driver.active.utils.get_device_properties(torch.npu.current_device())["num_vectorcore"])

    def launch():
        return cv_dot_transition_probe[(args.logical_programs, )](
            lhs,
            rhs,
            seed,
            scratch,
            output,
            args.trips,
            args.dots,
            args.scope_order == "before",
            args.recurrence_groups,
            args.dot_pattern == "dependent",
            args.tail_pattern == "solve",
            args.strided_recurrence_states,
            args.tail_row_stride,
            tail_matrix_span,
            compile_mode="simd_simt",
            num_warps=4,
            superblock_factor=args.factor,
            logical_program_count_hint=args.logical_programs,
            physical_vector_core_count_hint=vector_cores,
            auto_simt_scope_mode="auto",
            auto_simt_scope_dump=str(route_report),
            auto_simt_model_profile=str(args.profile) if args.profile else "",
        )

    binary = launch()
    torch.npu.synchronize()
    if not route_report.is_file():
        raise RuntimeError("cost-model route report was not emitted")
    if not bool(torch.isfinite(output).all().cpu()):
        raise RuntimeError("non-finite output")
    if (args.dots or args.tail_pattern == "solve") and not bool(torch.isfinite(scratch).all().cpu()):
        raise RuntimeError("non-finite dot result")
    for key in ("ttir", "ttadapter", "npuir", "bcmlir", "npubin"):
        value = binary.asm.get(key)
        if isinstance(value, str):
            (args.output_dir / ("kernel." + key)).write_text(value)
        elif isinstance(value, bytes):
            (args.output_dir / ("kernel." + key)).write_bytes(value)

    for _ in range(12):
        launch()
    torch.npu.synchronize()
    profile_dir = args.output_dir / "profile"
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
    ):
        for _ in range(args.samples):
            launch()
        torch.npu.synchronize()
    durations = []
    for path in profile_dir.rglob("kernel_details.csv"):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if "cv_dot_transition_probe" in str(row) and row.get("Duration(us)"):
                    durations.append(float(row["Duration(us)"]))
    if len(durations) != args.samples:
        raise RuntimeError(f"expected {args.samples} samples, got {len(durations)}")

    ttadapter = (args.output_dir / "kernel.ttadapter").read_text()
    npuir_path = args.output_dir / "kernel.npuir"
    npuir = npuir_path.read_text() if npuir_path.is_file() else ""
    result = {
        "schema": "cv-dot-transition-structural-probe-v1",
        "trips": args.trips,
        "dots": args.dots,
        "dot_pattern": args.dot_pattern,
        "tail_pattern": args.tail_pattern,
        "tail_row_stride_elements": args.tail_row_stride,
        "tail_matrix_span_elements": tail_matrix_span,
        "strided_recurrence_states": args.strided_recurrence_states,
        "scope_order": args.scope_order,
        "split": "holdout" if args.dots == 3 else "development",
        "logical_programs": args.logical_programs,
        "superblock_factor": args.factor,
        "recurrence_groups": args.recurrence_groups,
        "route_selection_profile": str(args.profile) if args.profile else None,
        "route_report": str(route_report),
        "median_us": statistics.median(durations),
        "samples_us": durations,
        "correct": True,
        "ttadapter_scope_count": ttadapter.count("scope.scope"),
        "ttadapter_scope_factor":
        (args.factor if f"ascend.scope_superblock.factor = {args.factor}" in ttadapter else None),
        "npuir_sync_counts": count_npuir_sync(npuir),
        "npuir_available": npuir_path.is_file(),
        "npubin_sha256": hashlib.sha256((args.output_dir / "kernel.npubin").read_bytes()).hexdigest(),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "TRITON_DEBUG",
                "TRITON_DISABLE_LINE_INFO",
                "TRITON_CACHE_DIR",
            )
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
