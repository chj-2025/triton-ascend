#!/usr/bin/env python3
"""Run one cost-model case in an isolated cache and archive compiler evidence.

This collector deliberately knows nothing about operator names or model
features.  It records the same observable fields for every Triton case:
benchmark output, route report, TTIR dumps, backend binary metadata and static
LDK/STK instructions.  NPU-profiler kernel Duration is the cost-model target.
Device-event time, static assembly counts, and profiler pipe counters remain
diagnostics rather than online inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

RESOURCE_RE = re.compile(r"(?:SIMT VF )?Function properties for (?P<function>[^:]+): "
                         r"Stack size: (?P<stack>\d+) bytes, Used register number: (?P<registers>\d+)")
INSTRUCTION_RE = re.compile(r"\b(?P<opcode>LDK|STK)(?:\.[A-Za-z0-9_]+)*\b", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_last_json(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    # Benchmark programs may print progress followed by one pretty-printed,
    # nested JSON object.  Decoding every opening brace and keeping the last
    # success returns the last *inner* object (often ``elapsed_us``), not the
    # benchmark record.  Prefer the object that ends latest in the stream and,
    # for equal end positions, spans the most text.  This also handles several
    # complete JSON records by selecting the final one.
    candidates: list[tuple[int, int, dict]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((index + consumed, consumed, value))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def disassemble(binary: Path, tool: Path, output: Path) -> dict:
    completed = subprocess.run(
        [str(tool), "-d", str(binary)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output.write_text(completed.stdout, encoding="utf-8")
    counts = {"LDK": 0, "STK": 0}
    for match in INSTRUCTION_RE.finditer(completed.stdout):
        counts[match.group("opcode").upper()] += 1
    return {
        "returncode": completed.returncode,
        "sha256": sha256(binary),
        "bytes": binary.stat().st_size,
        "static_instruction_counts": counts,
        "disassembly": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--route-report", type=Path)
    parser.add_argument("--measure-in-fresh-process", action="store_true")
    parser.add_argument(
        "--prepare-without-profiler",
        action="store_true",
        help="compile/cache once without paying a profiler pass before measurements",
    )
    parser.add_argument("--prepare-append", action="append", default=[])
    parser.add_argument("--measurement-processes", type=int, default=1)
    parser.add_argument(
        "--measurement-selection",
        choices=("last", "minimum-profiler", "median-profiler"),
        default="last",
    )
    parser.add_argument(
        "--objdump",
        type=Path,
        default=Path("/data/kaixin/triton_cases/llvm-objdump"),
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a benchmark command is required after --")
    if args.measurement_processes <= 0:
        parser.error("--measurement-processes must be positive")
    if args.measurement_processes > 1 and not args.measure_in_fresh_process:
        parser.error("multiple measurement processes require --measure-in-fresh-process")

    case_dir = args.output_root / args.name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    cache_dir = case_dir / "cache"
    dump_dir = case_dir / "dump"
    cache_dir.mkdir(parents=True)
    dump_dir.mkdir(parents=True)

    env = os.environ.copy()
    env.update({
        "TRITON_CACHE_DIR": str(cache_dir),
        "TRITON_DUMP_DIR": str(dump_dir),
        "TRITON_KERNEL_DUMP": "1",
        "TRITON_ALWAYS_COMPILE": "1",
    })
    started = time.time()
    started_utc = datetime.now(timezone.utc).isoformat()
    prepare_command = list(command)
    if args.prepare_without_profiler and "--profile-dir" in prepare_command:
        profile_index = prepare_command.index("--profile-dir")
        del prepare_command[profile_index:profile_index + 2]
    prepare_command += args.prepare_append
    prepare_completed = None
    if args.measure_in_fresh_process:
        prepare_completed = subprocess.run(
            prepare_command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        (case_dir / "compile_stdout.log").write_text(prepare_completed.stdout, encoding="utf-8")
        (case_dir / "compile_stderr.log").write_text(prepare_completed.stderr, encoding="utf-8")
    measurement_env = env.copy()
    if prepare_completed is not None and prepare_completed.returncode == 0:
        measurement_env["TRITON_ALWAYS_COMPILE"] = "0"
    measurement_runs = []
    if prepare_completed is not None and prepare_completed.returncode != 0:
        completed = prepare_completed
        selected_measurement_index = None
    else:
        for index in range(args.measurement_processes):
            run = subprocess.run(
                command,
                env=measurement_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            benchmark = parse_last_json(run.stdout)
            measurement_runs.append((index, run, benchmark))
            (case_dir / f"measurement_{index:02d}_stdout.log").write_text(run.stdout, encoding="utf-8")
            (case_dir / f"measurement_{index:02d}_stderr.log").write_text(run.stderr, encoding="utf-8")
            if run.returncode:
                break

        def profiler_us(item: tuple[int, subprocess.CompletedProcess, dict | None]) -> float:
            benchmark = item[2] or {}
            value = (benchmark.get("profiler") or {}).get("average_us")
            return float(value) if value is not None else float("inf")

        selectable = [item for item in measurement_runs if item[1].returncode == 0]
        if not selectable:
            selected = measurement_runs[-1]
        elif args.measurement_selection == "minimum-profiler":
            selected = min(selectable, key=profiler_us)
        elif args.measurement_selection == "median-profiler":
            finite = sorted(selectable, key=profiler_us)
            selected = finite[len(finite) // 2]
        else:
            selected = selectable[-1]
        selected_measurement_index, completed, _ = selected
    (case_dir / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(completed.stderr, encoding="utf-8")

    resources = []
    for match in RESOURCE_RE.finditer(completed.stdout + "\n" + completed.stderr):
        resources.append({
            "function": match.group("function"),
            "stack_bytes_per_thread": int(match.group("stack")),
            "used_registers_per_thread": int(match.group("registers")),
        })

    binaries = []
    if args.objdump.is_file():
        for index, binary in enumerate(sorted(cache_dir.rglob("*.npubin"))):
            binaries.append({
                "path": str(binary),
                **disassemble(
                    binary,
                    args.objdump,
                    case_dir / f"binary_{index:02d}.disassembly.txt",
                ),
            })

    route = None
    if args.route_report and args.route_report.is_file():
        route = json.loads(args.route_report.read_text(encoding="utf-8"))
        shutil.copy2(args.route_report, case_dir / "route.json")

    selected_benchmark = parse_last_json(completed.stdout)
    profiler_us = ((selected_benchmark or {}).get("profiler") or {}).get("average_us")
    event_us = (selected_benchmark or {}).get("elapsed_us", {}).get("median")
    event_gap_us = None
    if profiler_us is not None and event_us is not None:
        event_gap_us = float(event_us) - float(profiler_us)

    result = {
        "schema":
        "triton-costmodel-backend-evidence-v1",
        "case":
        args.name,
        "command":
        command,
        "prepare_command":
        prepare_command if prepare_completed is not None else None,
        "prepare_returncode": (prepare_completed.returncode if prepare_completed is not None else None),
        "measurement_in_fresh_process":
        args.measure_in_fresh_process,
        "measurement_process_count":
        len(measurement_runs),
        "measurement_selection":
        args.measurement_selection,
        "selected_measurement_index":
        selected_measurement_index,
        "measurement_runs": [{
            "index": index,
            "returncode": run.returncode,
            "benchmark": benchmark,
        } for index, run, benchmark in measurement_runs],
        "returncode":
        completed.returncode,
        "started_utc":
        started_utc,
        "elapsed_wall_seconds":
        time.time() - started,
        "environment": {
            key: measurement_env.get(key)
            for key in (
                "ASCEND_RT_VISIBLE_DEVICES",
                "TRITON_DEBUG",
                "TRITON_DISABLE_LINE_INFO",
                "TRITON_ALWAYS_COMPILE",
                "TRITON_KERNEL_DUMP",
                "TRITON_CACHE_DIR",
                "TRITON_DUMP_DIR",
            )
        },
        "benchmark":
        selected_benchmark,
        "measurement_metrics": {
            "costmodel_target": ("npu_profiler_kernel_duration_average_us" if profiler_us is not None else
                                 "device_event_median_us_diagnostic_only_profiler_disabled"),
            "profiler_kernel_duration_average_us":
            profiler_us,
            "device_event_median_us":
            event_us,
            "device_event_minus_profiler_us":
            event_gap_us,
        },
        "route_report_present":
        route is not None,
        "route_summary":
        None if route is None else {
            "profile_version": route.get("profile_version"),
            "candidate_costs": route.get("candidate_costs"),
            "effective_decision_kind": route.get("effective_decision_kind"),
            "effective_runtime_factor": route.get("effective_runtime_factor"),
            "selected_route_elapsed_prediction": route.get("selected_route_elapsed_prediction"),
            "profile_target": route.get("profile_target"),
            "actual_target": route.get("actual_target"),
        },
        "backend_resources":
        resources,
        "binaries":
        binaries,
    }
    (case_dir / "evidence.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
