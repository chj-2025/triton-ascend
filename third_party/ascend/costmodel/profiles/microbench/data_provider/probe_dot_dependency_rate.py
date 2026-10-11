#!/usr/bin/env python3
"""Compare dot input-feedback and accumulator-feedback with one final store."""

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["rhs_scale", "matrix_bound"])
def dot_dependency_probe(a_ptr, b_ptr, out_ptr, size: tl.constexpr, count: tl.constexpr,
                         accumulate: tl.constexpr = False, fresh_rhs: tl.constexpr = False,
                         row_stride: tl.constexpr = 0, fresh_lhs: tl.constexpr = False,
                         retain_results: tl.constexpr = False, block_pointers: tl.constexpr = False,
                         computed_rhs: tl.constexpr = False, rhs_scale=1.0, runtime_bounds: tl.constexpr = False,
                         matrix_bound=16, program_column_stride: tl.constexpr = 0,
                         column_major_programs: tl.constexpr = False, program_rows: tl.constexpr = 1):
    x = tl.arange(0, size)
    pid = tl.program_id(0)
    bound = matrix_bound if runtime_bounds else size
    offsets = pid * size * size + x[:, None] * size + x[None, :]
    if block_pointers:
        a_block = tl.make_block_ptr(a_ptr + pid * size * size, (bound, bound), (size, 1), (0, 0), (size, size), (1, 0))
        a = tl.load(a_block, boundary_check=(0, 1), padding_option="zero")
    else:
        a = tl.load(a_ptr + offsets)
    stride: tl.constexpr = size if row_stride == 0 else row_stride
    streamed: tl.constexpr = fresh_rhs or fresh_lhs
    slots: tl.constexpr = max(1, count) if streamed else 1
    if program_column_stride:
        columns: tl.constexpr = stride // program_column_stride
        if column_major_programs:
            b_base = (pid % program_rows) * slots * size * stride + (pid // program_rows) * program_column_stride
        else:
            b_base = (pid // columns) * slots * size * stride + (pid % columns) * program_column_stride
    else:
        b_base = pid * slots * size * stride
    b_offsets = b_base + x[:, None] * stride + x[None, :]
    if block_pointers:
        b_block = tl.make_block_ptr(b_ptr + b_base, (bound, bound), (stride, 1), (0, 0), (size, size), (1, 0))
    if not streamed:
        if block_pointers:
            b = tl.load(b_block, boundary_check=(0, 1), padding_option="zero")
        else:
            b = tl.load(b_ptr + b_offsets)
        if computed_rhs:
            # A runtime multiply prevents treating this operand as a direct
            # GM load. The independent reference applies the same multiplier.
            b = b * rhs_scale
    if accumulate:
        acc = tl.full((size, size), 0, tl.float32)
        for step in tl.static_range(count):
            if streamed:
                if block_pointers:
                    current = tl.make_block_ptr(b_ptr + b_base + step * size * stride, (bound, bound), (stride, 1),
                                                (0, 0), (size, size), (1, 0))
                    b = tl.load(current, boundary_check=(0, 1), padding_option="zero")
                else:
                    b = tl.load(b_ptr + b_offsets + step * size * stride)
                if computed_rhs:
                    b = b * rhs_scale
            if fresh_lhs:
                acc = tl.dot(b, a, acc, input_precision="ieee")
            else:
                acc = tl.dot(a, b, acc, input_precision="ieee")
        tl.store(out_ptr + offsets, acc)
    else:
        results = ()
        for step in tl.static_range(count):
            if streamed:
                if block_pointers:
                    current = tl.make_block_ptr(b_ptr + b_base + step * size * stride, (bound, bound), (stride, 1),
                                                (0, 0), (size, size), (1, 0))
                    b = tl.load(current, boundary_check=(0, 1), padding_option="zero")
                else:
                    b = tl.load(b_ptr + b_offsets + step * size * stride)
                if computed_rhs:
                    b = b * rhs_scale
            if fresh_lhs:
                a = tl.dot(b, a, input_precision="ieee")
            else:
                a = tl.dot(a, b, input_precision="ieee")
            if retain_results:
                results += (a, )
        if retain_results:
            # All intermediate results are externally observable. Inspect the
            # lowered schedule before attributing any change to live ranges:
            # the compiler may move these stores earlier when aliasing permits.
            for step in tl.static_range(count):
                result_offsets = (pid * count + step) * size * size + x[:, None] * size + x[None, :]
                tl.store(out_ptr + result_offsets, results[step])
        else:
            tl.store(out_ptr + offsets, a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("simd", "simt_only"), required=True)
    parser.add_argument("--size", type=int, choices=(16, 32), required=True)
    parser.add_argument("--count", type=int, choices=(0, 1, 2, 4, 8, 16, 32), required=True)
    parser.add_argument("--path", choices=("chain", "accumulate"), default="chain")
    parser.add_argument("--factor", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--logical-programs", type=int, default=1)
    parser.add_argument("--fresh-rhs", action="store_true",
                        help="Each dot consumes a different GM tile instead of reusing the preloaded RHS")
    parser.add_argument("--fresh-lhs", action="store_true",
                        help="Stream the left operand from GM; feed back the right operand in the chain")
    parser.add_argument("--row-stride", type=int, default=0)
    parser.add_argument("--dump-ir", action="store_true",
                        help="Save the actual NPUIR command and after-pass IR without enabling debug assertions")
    parser.add_argument("--block-pointers", action="store_true",
                        help="Use block-pointer dot inputs, which may take a different K-tiling lowering")
    parser.add_argument("--computed-rhs", action="store_true",
                        help="Make the loaded operand a computed tensor; inspect its shared-memory staging")
    parser.add_argument("--rhs-scale", type=float, default=1.0,
                        help="Runtime multiplier used with --computed-rhs (not a compile-time constant)")
    parser.add_argument("--runtime-bounds", action="store_true",
                        help="Retain runtime row/column boundary predicates on block-pointer loads")
    parser.add_argument("--matrix-bound", type=int,
                        help="Runtime valid extent, defaults to size; smaller values test masked zero padding")
    parser.add_argument(
        "--program-column-stride", type=int, default=0,
        help="Pack independent programs across columns of a shared strided row; zero keeps separate allocations")
    parser.add_argument("--column-major-programs", action="store_true",
                        help="Vary the program row first, matching grid(row, column) flattening")
    parser.add_argument("--distinct-program-inputs", action="store_true",
                        help="Give every program a distinct RHS scale to validate packed address mapping")
    parser.add_argument("--capture-only", action="store_true",
                        help="One checked launch for CaModel; no timing or performance claim")
    parser.add_argument(
        "--retain-results", action="store_true",
        help="Store every input-feedback dot result after the chain; diagnostic live-value pressure sweep")
    args = parser.parse_args()
    if args.logical_programs < 1 or (args.mode != "simt_only" and args.factor != 1):
        parser.error("Positive logical-programs required; factor>1 only applies to SIMT")
    if args.row_stride and args.row_stride < args.size:
        parser.error("row-stride must be zero or at least size")
    if args.fresh_lhs and args.fresh_rhs:
        parser.error("Select exactly one streamed operand")
    if args.retain_results and (args.path != "chain" or args.count == 0):
        parser.error("retain-results requires a nonempty input-feedback chain")
    if args.runtime_bounds and not args.block_pointers:
        parser.error("runtime-bounds requires block-pointers")
    if args.matrix_bound is not None and (not args.runtime_bounds or not 0 < args.matrix_bound <= args.size):
        parser.error("matrix-bound requires runtime-bounds and must be within [1, size]")
    stride = args.row_stride or args.size
    if args.program_column_stride and (args.program_column_stride < args.size or stride % args.program_column_stride):
        parser.error("program-column-stride must be >= size and divide row-stride")
    if args.column_major_programs and not args.program_column_stride:
        parser.error("column-major-programs requires packed program-column-stride")
    args.output.mkdir(parents=True, exist_ok=False)

    torch.manual_seed(123)
    a_cpu = torch.randn(args.size, args.size)
    b_cpu = torch.randn(args.size, args.size) / args.size**0.5
    a = a_cpu.expand(args.logical_programs, -1, -1).contiguous().npu()
    slots = max(1, args.count) if args.fresh_rhs or args.fresh_lhs else 1
    columns = stride // args.program_column_stride if args.program_column_stride else 1
    groups = triton.cdiv(args.logical_programs, columns)
    b = torch.zeros((groups, slots, args.size, stride), device="npu")
    column_stride = args.program_column_stride or stride
    program_scales = torch.ones(groups * columns)
    if args.distinct_program_inputs:
        program_scales += torch.arange(groups * columns) / (2.0 * groups * columns)
    scale_grid = (program_scales.reshape(columns, groups).T if args.column_major_programs else program_scales.reshape(
        groups, columns))
    b.view(groups, slots, args.size, columns,
           column_stride).permute(0, 3, 1, 2, 4)[..., :args.size] = (b_cpu * scale_grid[:, :, None, None, None]).npu()
    out = (torch.empty((args.logical_programs, args.count, args.size,
                        args.size), device="npu") if args.retain_results else torch.empty_like(a))

    def launch():
        return dot_dependency_probe[(args.logical_programs, )](
            a, b, out, args.size, args.count, args.path == "accumulate", args.fresh_rhs, args.row_stride,
            args.fresh_lhs, args.retain_results, args.block_pointers, args.computed_rhs, args.rhs_scale,
            args.runtime_bounds, args.matrix_bound or args.size, args.program_column_stride, args.column_major_programs,
            groups, compile_mode=args.mode, num_warps=4, superblock_factor=args.factor)

    original_run = subprocess.run

    def trace_compile(command, *positional, **keywords):
        if (isinstance(command, (list, tuple)) and "-o" in command and Path(command[0]).name == "bishengir-compile"):
            command = [*command, "--mlir-print-ir-after-all"]
            (args.output / "compile_command.json").write_text(json.dumps(command, indent=2))
            result = original_run(command, *positional, **keywords)
            (args.output / "compiler_after_all.log").write_bytes((result.stdout or b"") + (result.stderr or b""))
            return result
        return original_run(command, *positional, **keywords)

    if args.dump_ir:
        subprocess.run = trace_compile
    try:
        binary = launch()
    finally:
        subprocess.run = original_run
    if args.runtime_bounds:
        bound = args.matrix_bound or args.size
        a_cpu[bound:, :] = 0
        a_cpu[:, bound:] = 0
        b_cpu[bound:, :] = 0
        b_cpu[:, bound:] = 0
    if args.computed_rhs:
        b_cpu = b_cpu * args.rhs_scale
    if args.path == "accumulate":
        expected = args.count * (b_cpu @ a_cpu if args.fresh_lhs else a_cpu @ b_cpu)
    else:
        expected = a_cpu
        results = []
        for _ in range(args.count):
            expected = b_cpu @ expected if args.fresh_lhs else expected @ b_cpu
            results.append(expected)
        if args.retain_results:
            expected = torch.stack(results)
    expected = expected.expand(args.logical_programs, *expected.shape)
    if args.retain_results:
        multipliers = program_scales[:args.logical_programs, None]**torch.arange(1, args.count + 1)
    else:
        power = args.count if args.path == "chain" else 1
        multipliers = program_scales[:args.logical_programs]**power
    expected = expected * multipliers[..., None, None]
    torch.testing.assert_close(out.cpu(), expected, rtol=2e-3, atol=2e-4)
    for key in ("ttir", "ttadapter", "npuir", "bcmlir", "npubin"):
        value = binary.asm.get(key)
        if isinstance(value, str):
            (args.output / ("kernel." + key)).write_text(value)
        elif isinstance(value, bytes):
            (args.output / ("kernel." + key)).write_bytes(value)

    if args.capture_only:
        capture = {
            "correct": True,
            "capture_only": True,
            "mode": args.mode,
            "size": args.size,
            "count": args.count,
            "path": args.path,
            "grid": [args.logical_programs],
            "factor": args.factor,
            "block_pointers": args.block_pointers,
            "fresh_lhs": args.fresh_lhs,
            "fresh_rhs": args.fresh_rhs,
            "retain_results": args.retain_results,
            "computed_rhs": args.computed_rhs,
            "rhs_scale": args.rhs_scale,
            "runtime_bounds": args.runtime_bounds,
            "matrix_bound": args.matrix_bound or args.size,
            "program_column_stride": args.program_column_stride,
            "column_major_programs": args.column_major_programs,
            "distinct_program_inputs": args.distinct_program_inputs,
            "environment": {
                key: os.environ.get(key)
                for key in ("ASCEND_RT_VISIBLE_DEVICES", "NPU_DEVICE_LIMIT", "TRITON_DEBUG", "TRITON_DISABLE_LINE_INFO",
                            "TRITON_CACHE_DIR")
            },
            "npubin_sha256": hashlib.sha256((args.output / "kernel.npubin").read_bytes()).hexdigest(),
            "binary_metadata": dict(binary.metadata._asdict()),
        }
        (args.output / "summary.json").write_text(json.dumps(capture, indent=2, default=str))
        print("DOT_CAPTURE_FULL_OUTPUT_PASS", flush=True)
        return

    for _ in range(20):
        launch()
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output / "profile")),
    ) as profiler:
        for _ in range(20):
            launch()
        torch.npu.synchronize()
    torch.npu.synchronize()

    samples = []
    for path in args.output.rglob("kernel_details.csv"):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if ("dot_dependency_probe" in str(row) and row.get("Duration(us)")):
                    samples.append(float(row["Duration(us)"]))
    if len(samples) != 20:
        raise RuntimeError(f"expected 20 samples, got {len(samples)}")
    torch.testing.assert_close(out.cpu(), expected, rtol=2e-3, atol=2e-4)
    result = {
        "column_major_programs":
        args.column_major_programs,
        "distinct_program_inputs":
        args.distinct_program_inputs,
        "computed_rhs":
        args.computed_rhs,
        "rhs_scale":
        args.rhs_scale,
        "runtime_bounds":
        args.runtime_bounds,
        "matrix_bound":
        args.matrix_bound or args.size,
        "program_column_stride":
        args.program_column_stride,
        "mode":
        args.mode,
        "size":
        args.size,
        "count":
        args.count,
        "path":
        args.path,
        "num_warps":
        4,
        "median_us":
        statistics.median(samples),
        "samples_us":
        samples,
        "correct":
        True,
        "warmup":
        20,
        "profiler_active_samples":
        20,
        "grid": [args.logical_programs],
        "requested_factor":
        args.factor,
        "fresh_rhs":
        args.fresh_rhs,
        "fresh_lhs":
        args.fresh_lhs,
        "retain_results":
        args.retain_results,
        "block_pointers":
        args.block_pointers,
        "rhs_row_stride_elements":
        stride,
        "source_rhs_allocation_bytes":
        b.numel() * b.element_size(),
        "environment": {
            key: os.environ.get(key)
            for key in ("ASCEND_RT_VISIBLE_DEVICES", "TRITON_DEBUG", "TRITON_DISABLE_LINE_INFO", "TRITON_CACHE_DIR")
        },
        "binary_metadata":
        dict(binary.metadata._asdict()),
        "npubin_sha256":
        hashlib.sha256((args.output / "kernel.npubin").read_bytes()).hexdigest(),
        "interpretation":
        "Whole-kernel samples; incremental slopes require an IR check of retained MMA/FixPipe calls. Not isolated hardware startup.",
    }
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
