#!/usr/bin/env python3
"""Compare generic SIMD/SIMT predictions with the cross-operator matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STANDARD_MODES = (
    ("auto", 1),
    ("simd", 1),
    ("simt_only", 1),
    ("simt_only", 2),
    ("simt_only", 4),
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def measured_us(evidence: dict) -> float | None:
    benchmark = evidence.get("benchmark") or {}
    if not benchmark.get("correct", False):
        return None
    profiler = benchmark.get("profiler") or {}
    if profiler.get("average_us") is not None:
        return float(profiler["average_us"])
    elapsed = benchmark.get("elapsed_us") or {}
    return float(elapsed["median"]) if elapsed.get("median") is not None else None


def boundary_ticks(profile: dict, physical_programs: int) -> float:
    timing = profile["kernel_timing"]
    points = sorted((int(key), float(value))
                    for key, value in timing["boundary_residual_system_cycles_by_physical_program_count"].items())
    if physical_programs <= points[0][0]:
        return points[0][1]
    if physical_programs >= points[-1][0]:
        return points[-1][1]
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if physical_programs <= right_x:
            ratio = (physical_programs - left_x) / (right_x - left_x)
            return left_y + ratio * (right_y - left_y)
    raise AssertionError("unreachable boundary interpolation")


def predicted_us(route: dict, mode: str, factor: int, profile: dict) -> float | None:
    if mode == "auto":
        prediction = route.get("selected_route_elapsed_prediction") or {}
        return (float(prediction["predicted_microseconds"]) if prediction.get("applied") else None)
    if mode == "simd":
        candidate = (route["stage_model"].get("routes") or {}).get("all_simd")
    elif mode == "mixed":
        candidate = (route["stage_model"].get("routes") or {}).get("mixed_simd_simt")
    else:
        candidate = (route["stage_model"].get("all_simt_factor_routes") or {}).get(str(factor))
    if not candidate or not candidate.get("legal", False):
        return None
    ticks = float(candidate["total_system_cycles"]) + boundary_ticks(profile,
                                                                     int(candidate["runtime_physical_program_count"]))
    frequency = float(profile["kernel_timing"]["system_counter_frequency_mhz"])
    return ticks / frequency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "generalization_matrix.json")
    parser.add_argument("--maximum-error", type=float, default=0.10)
    args = parser.parse_args()

    summary = load(args.evidence_root / "matrix_summary.json")
    profile = load(args.profile)
    manifest = load(args.manifest)
    expected = {}
    for case in manifest["cases"]:
        modes = list(STANDARD_MODES)
        modes.extend((entry["route"], int(entry.get("factor", 1))) for entry in case.get("extra_modes", []))
        for mode, factor in modes:
            key = (case["name"], mode, factor)
            if key in expected:
                raise ValueError(f"duplicate expected matrix record {key}")
            expected[key] = {"family": case["family"], "split": case["split"]}
    metadata = {}
    locations = {}
    records = {}
    for record in summary["records"]:
        base = record["name"].split("__", 1)[0]
        metadata[base] = {
            "family": record["family"],
            "split": record["split"],
        }
        key = (base, record["route"], int(record["factor"]))
        if key in records:
            raise ValueError(f"duplicate matrix record {key}")
        if key not in expected:
            raise ValueError(f"unexpected matrix record {key}")
        if metadata[base] != expected[key]:
            raise ValueError(f"matrix metadata differs from manifest for {key}")
        records[key] = record
        locations[key] = Path(record.get("evidence_dir", args.evidence_root / record["name"]))
    auto_routes = {}
    auto_failures = {}
    for base in metadata:
        auto_dir = locations.get((base, "auto", 1))
        if auto_dir is None:
            auto_failures[base] = "missing_auto_record"
        elif not (auto_dir / "route.json").is_file():
            auto_failures[base] = "missing_auto_route_report"
        else:
            auto_routes[base] = load(auto_dir / "route.json")
    results = []
    for (base, mode, factor), record in sorted(records.items()):
        failures = []
        if int(record.get("returncode", 1)) != 0:
            failures.append(f"runner_returncode={record.get('returncode')}")
        case_dir = locations[(base, mode, factor)]
        evidence_path = case_dir / "evidence.json"
        evidence = None
        if not evidence_path.is_file():
            failures.append("missing_evidence")
        else:
            evidence = load(evidence_path)
            if int(evidence.get("returncode", 1)) != 0:
                failures.append(f"benchmark_returncode={evidence.get('returncode')}")
        actual = measured_us(evidence) if evidence is not None else None
        if actual is None:
            failures.append("incorrect_or_missing_timing")
        route = auto_routes.get(base)
        if route is None:
            failures.append(auto_failures.get(base, "missing_auto_route_report"))
        predicted = (predicted_us(route, mode, factor, profile) if route is not None else None)
        if predicted is None:
            failures.append("missing_legal_prediction")
        error = (predicted / actual - 1.0 if predicted is not None and actual is not None and actual > 0.0 else None)
        passed = not failures and abs(error) <= args.maximum_error
        results.append({
            "case": base,
            **metadata[base],
            "mode": mode,
            "factor": factor,
            "predicted_us": predicted,
            "measured_us": actual,
            "relative_error": error,
            "failure_reasons": failures,
            "pass": passed,
        })
    for (base, mode, factor), meta in sorted(expected.items()):
        if (base, mode, factor) in records:
            continue
        results.append({
            "case": base,
            **meta,
            "mode": mode,
            "factor": factor,
            "predicted_us": None,
            "measured_us": None,
            "relative_error": None,
            "failure_reasons": ["missing_matrix_record"],
            "pass": False,
        })
    results.sort(key=lambda row: (row["case"], row["mode"], row["factor"]))
    report = {
        "schema": "triton-costmodel-generalization-audit-v1",
        "maximum_absolute_relative_error": args.maximum_error,
        "expected_record_count": len(expected),
        "matrix_record_count": len(summary["records"]),
        "audited_record_count": len(results),
        "all_pass": (bool(results) and len(results) == len(expected) and all(row["pass"] for row in results)),
        "results": results,
    }
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Cost model cross-operator audit",
        "",
        "| Case | Family | Split | Route | Factor | Predicted (us) | Measured (us) | Error | Gate |",
        "|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in results:
        predicted_text = (f"{row['predicted_us']:.3f}" if row["predicted_us"] is not None else "N/A")
        measured_text = (f"{row['measured_us']:.3f}" if row["measured_us"] is not None else "N/A")
        error_text = (f"{row['relative_error']:+.2%}" if row["relative_error"] is not None else "N/A")
        gate = "PASS" if row["pass"] else "FAIL"
        if row["failure_reasons"]:
            gate += ": " + ", ".join(row["failure_reasons"])
        lines.append(f"| {row['case']} | {row['family']} | {row['split']} | "
                     f"{row['mode']} | {row['factor']} | {predicted_text} | "
                     f"{measured_text} | {error_text} | {gate} |")
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
