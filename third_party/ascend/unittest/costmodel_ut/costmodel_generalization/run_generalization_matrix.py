#!/usr/bin/env python3
"""Run the cross-operator cost-model matrix with isolated compiler evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def capture(command: list[str], cwd: Path | None = None) -> dict:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {"command": command, "returncode": completed.returncode, "output": completed.stdout}


def git_state(path: Path) -> dict:
    if not path.is_dir():
        return {"root": str(path), "sha": None, "status_porcelain": None}
    revision = capture(["git", "rev-parse", "HEAD"], cwd=path)
    status = capture(["git", "status", "--porcelain=v1"], cwd=path)
    diff = capture(["git", "diff", "--binary", "HEAD", "--"], cwd=path)
    revision_text = revision["output"].strip() if revision["returncode"] == 0 else None
    status_lines = (status["output"].splitlines() if status["returncode"] == 0 else None)
    state_material = json.dumps(
        {
            "revision": revision_text,
            "status": status_lines,
            "diff": diff["output"] if diff["returncode"] == 0 else None,
        },
        sort_keys=True,
    ).encode("utf-8")
    return {
        "root": str(path),
        "sha": revision_text,
        "status_porcelain": status_lines,
        "diff_sha256": hashlib.sha256((diff["output"] if diff["returncode"] == 0 else "").encode("utf-8")).hexdigest(),
        "source_state_sha256": hashlib.sha256(state_material).hexdigest(),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compiler_state(expected: Path) -> dict:
    resolved_expected = expected.resolve()
    selected_text = shutil.which("bishengir-compile")
    if selected_text is None:
        raise RuntimeError("bishengir-compile is not on PATH")
    selected = Path(selected_text).resolve()
    if selected != resolved_expected:
        raise RuntimeError("wrong bishengir-compile selected: "
                           f"expected {resolved_expected}, got {selected}; "
                           "source /data/kaixin/set_env.sh before running the matrix")
    version = capture([str(selected), "--version"])
    if version["returncode"] != 0:
        raise RuntimeError(f"failed to query {selected}: {version['output']}")
    return {
        "path": str(selected),
        "version": version["output"].strip(),
        "size_bytes": selected.stat().st_size,
        "mtime_ns": selected.stat().st_mtime_ns,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=HERE / "generalization_matrix.json")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="required for execution; omitted only with --list-cases",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="list every case in manifest order, then exit without requiring an NPU",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="CASE_NAME",
        help="run only this exact manifest case; repeat to select multiple cases",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        help=("optional profiler scratch root; compilation evidence and caches "
              "remain under --output-root"),
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("/data/kaixin/triton-ascend/third_party/ascend/costmodel/profiles/"
                     "simd_simt/ascend_950pr_simd_simt_v1.json"),
    )
    parser.add_argument(
        "--expected-bishengir",
        type=Path,
        default=Path("/data/kaixin/bishengir-install/bin/bishengir-compile"),
        help="fail before compiling if PATH resolves a different compiler",
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--measurement-processes",
        type=int,
        default=3,
        help="fresh processes used for the median profiler measurement",
    )
    parser.add_argument(
        "--measurement-selection",
        choices=("minimum-profiler", "median-profiler"),
        default="median-profiler",
    )
    parser.add_argument(
        "--disable-profiler",
        action="store_true",
        help="diagnostic escape hatch; profiler Duration is the validation target",
    )
    parser.add_argument("--include-explicit", action="store_true")
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        metavar="ROUTE[:FACTOR]",
        help=("run only the requested route/factor pair; repeat for multiple pairs "
              "(for example --mode simd or --mode simt_only:4)"),
    )
    parser.add_argument("--only-family", action="append", default=[])
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        parser.error("require 0 <= --shard-index < --shard-count")
    if args.measurement_processes <= 0:
        parser.error("--measurement-processes must be positive")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases_by_name = {case["name"]: case for case in manifest["cases"]}
    if len(cases_by_name) != len(manifest["cases"]):
        parser.error("manifest contains duplicate case names")
    unknown_cases = sorted(set(args.case) - set(cases_by_name))
    if unknown_cases:
        parser.error(f"unknown --case value(s): {', '.join(unknown_cases)}")
    if args.list_cases:
        for index, case in enumerate(manifest["cases"]):
            print(f"{index:03d}\t{case['name']}\t{case['family']}\t"
                  f"{case['split']}\t{case['driver']}")
        return
    if args.output_root is None:
        parser.error("--output-root is required unless --list-cases is used")
    selected_cases = set(args.case)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.profile_root:
        args.profile_root.mkdir(parents=True, exist_ok=True)
    if not args.profile.is_file():
        raise FileNotFoundError(args.profile)
    profile_snapshot = args.output_root / "costmodel_profile.json"
    shutil.copy2(args.profile, profile_snapshot)
    manifest_snapshot = args.output_root / "generalization_matrix.json"
    shutil.copy2(args.manifest, manifest_snapshot)
    # Freeze and record the two debug-affecting values once for all routes.
    os.environ.setdefault("TRITON_DEBUG", "0")
    os.environ.setdefault("TRITON_DISABLE_LINE_INFO", "0")
    run_started_utc = datetime.now(timezone.utc).isoformat()
    toolchain = {
        "triton_ascend": git_state(Path("/data/kaixin/triton-ascend")),
        "npuir": git_state(Path("/data/kaixin/AscendNPU-IR")),
        "bishengir_compile": compiler_state(args.expected_bishengir),
    }
    device_snapshot = capture(["npu-smi", "info"])

    modes = []
    for mode_spec in args.mode:
        route, separator, factor_text = mode_spec.partition(":")
        if route not in {"auto", "simd", "simt_only", "mixed"}:
            parser.error(f"unsupported route in --mode: {route}")
        try:
            factor = int(factor_text) if separator else 1
        except ValueError:
            parser.error(f"invalid factor in --mode: {mode_spec}")
        if factor <= 0:
            parser.error(f"factor must be positive in --mode: {mode_spec}")
        modes.append((route, factor))
    if not modes:
        modes = [("auto", 1)]
    if args.include_explicit and not args.mode:
        modes += [("simd", 1), ("simt_only", 1), ("simt_only", 2), ("simt_only", 4)]

    records = []
    summary_context = {
        "schema": "triton-costmodel-generalization-run-v1",
        "started_utc": run_started_utc,
        "device": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "environment": {
            "TRITON_DEBUG": os.environ.get("TRITON_DEBUG"),
            "TRITON_DISABLE_LINE_INFO": os.environ.get("TRITON_DISABLE_LINE_INFO"),
        },
        "toolchain": toolchain,
        "device_snapshot": device_snapshot,
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "manifest": str(args.manifest),
        "manifest_snapshot": str(manifest_snapshot.resolve()),
        "manifest_sha256": sha256(manifest_snapshot),
        "selected_cases": list(args.case),
        "profile_snapshot": str(profile_snapshot.resolve()),
        "profile_sha256": sha256(profile_snapshot),
    }
    for case_index, case in enumerate(manifest["cases"]):
        if selected_cases and case["name"] not in selected_cases:
            continue
        if args.only_family and case["family"] not in args.only_family:
            continue
        if case_index % args.shard_count != args.shard_index:
            continue
        case_modes = list(modes)
        if args.include_explicit and not args.mode:
            case_modes += [(entry["route"], int(entry.get("factor", 1))) for entry in case.get("extra_modes", [])]
        for route, factor in case_modes:
            name = f"{case['name']}__{route}_f{factor}"
            report = args.output_root / f"{name}.route.json"
            report.unlink(missing_ok=True)
            driver = (HERE / case["driver"]).resolve()
            factor_flag = case.get("factor_argument", "--factor")
            command = [
                sys.executable,
                str(driver),
                "--route",
                route,
                factor_flag,
                str(factor),
                "--warmup",
                str(args.warmup),
                "--repeats",
                str(args.repeats),
                "--report",
                str(report),
                *case.get("arguments", []),
            ]
            if not args.disable_profiler and case.get("profiler", True):
                profile_parent = args.profile_root or args.output_root
                command += [
                    "--profile-dir",
                    str(profile_parent / name / "profiler"),
                ]
            collector = [
                sys.executable,
                str(HERE / "collect_backend_evidence.py"),
                "--name",
                name,
                "--output-root",
                str(args.output_root),
                "--route-report",
                str(report),
                "--measure-in-fresh-process",
                "--prepare-without-profiler",
                "--measurement-processes",
                str(args.measurement_processes),
                "--measurement-selection",
                args.measurement_selection,
            ]
            completed = subprocess.run(collector + ["--", *command], check=False)
            # The content-addressed compiler cache is reproducible derived
            # state, not gate evidence. Remove it after each case so a full
            # serialized matrix does not exhaust the evidence filesystem.
            shutil.rmtree(args.output_root / name / "cache", ignore_errors=True)
            records.append({
                "name": name,
                "family": case["family"],
                "split": case["split"],
                "route": route,
                "factor": factor,
                "returncode": completed.returncode,
                "evidence_dir": str((args.output_root / name).resolve()),
            })
            (args.output_root / "matrix_summary.partial.json").write_text(
                json.dumps(
                    {**summary_context, "records": records},
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            if completed.returncode:
                break
    toolchain_end = {
        "triton_ascend": git_state(Path("/data/kaixin/triton-ascend")),
        "npuir": git_state(Path("/data/kaixin/AscendNPU-IR")),
        "bishengir_compile": compiler_state(args.expected_bishengir),
    }
    source_stable = all(toolchain[name].get("source_state_sha256") == toolchain_end[name].get("source_state_sha256")
                        for name in ("triton_ascend", "npuir"))
    compiler_stable = (toolchain["bishengir_compile"] == toolchain_end["bishengir_compile"])
    summary = {
        **summary_context,
        "toolchain_end": toolchain_end,
        "source_stable_during_run": source_stable,
        "compiler_stable_during_run": compiler_stable,
        "records": records,
    }
    (args.output_root / "matrix_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True),
                                                          encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(1 if any(row["returncode"] for row in records) or not source_stable or not compiler_stable else 0)


if __name__ == "__main__":
    main()
