#!/usr/bin/env python3
"""Audit a CV dot/scope sweep without fitting an operator identity."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def line_fit(points: list[tuple[float, float]]) -> tuple[float, float]:
    if len(points) < 2:
        raise ValueError("at least two development points are required")
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    denominator = sum((x - mean_x)**2 for x, _ in points)
    if denominator == 0.0:
        raise ValueError("development dot counts are not distinct")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator
    return mean_y - slope * mean_x, slope


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-error", type=float, default=0.10)
    args = parser.parse_args()

    records = []
    for summary_path in sorted(args.root.glob("*/summary.json")):
        summary = load(summary_path)
        if summary.get("schema") != "cv-dot-transition-structural-probe-v1":
            continue
        route_path = summary_path.parent / "route.json"
        if not route_path.is_file():
            raise FileNotFoundError(route_path)
        route = load(route_path)
        prediction = route.get("selected_route_elapsed_prediction") or {}
        if not prediction.get("applied"):
            raise ValueError(f"missing route prediction in {route_path}")
        predicted = float(prediction["predicted_microseconds"])
        measured = float(summary["median_us"])
        sync = summary["npuir_sync_counts"]
        records.append({
            "directory": str(summary_path.parent),
            "factor": int(summary["superblock_factor"]),
            "scope_order": summary["scope_order"],
            "dots": int(summary["dots"]),
            "trips": int(summary["trips"]),
            "split": summary["split"],
            "predicted_us": predicted,
            "measured_us": measured,
            "raw_relative_error": predicted / measured - 1.0,
            "residual_us": measured - predicted,
            "sync_block_total": int(sync["sync_block_set"]) + int(sync["sync_block_wait"]),
            "set_wait_flag_total": int(sync["set_flag"]) + int(sync["wait_flag"]),
            "profile_version": route.get("profile_version"),
        })

    expected = {(factor, order, dots, trips)
                for factor in (1, 2, 4)
                for order in ("before", "after")
                for dots in (1, 2, 3, 4)
                for trips in (0, 14)}
    observed = {(row["factor"], row["scope_order"], row["dots"], row["trips"]) for row in records}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)

    fits = []
    holdouts = []
    grouped: defaultdict[tuple[int, str, int], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["factor"], row["scope_order"], row["trips"])].append(row)
    for (factor, order, trips), rows in sorted(grouped.items()):
        development = [row for row in rows if row["dots"] in (1, 2, 4)]
        intercept, slope = line_fit([(row["dots"], row["residual_us"]) for row in development])
        fits.append({
            "factor": factor,
            "scope_order": order,
            "trips": trips,
            "residual_intercept_us": intercept,
            "residual_per_dot_stage_us": slope,
            "development_dot_counts": [1, 2, 4],
        })
        for row in rows:
            if row["dots"] != 3:
                continue
            corrected = row["predicted_us"] + intercept + slope * row["dots"]
            error = corrected / row["measured_us"] - 1.0
            holdouts.append({
                **row,
                "corrected_predicted_us": corrected,
                "corrected_relative_error": error,
                "pass": abs(error) <= args.maximum_error,
            })

    trip_slopes = []
    by_pair = {(row["factor"], row["scope_order"], row["dots"], row["trips"]): row for row in records}
    for factor in (1, 2, 4):
        for order in ("before", "after"):
            for dots in (1, 2, 3, 4):
                zero = by_pair.get((factor, order, dots, 0))
                work = by_pair.get((factor, order, dots, 14))
                if not zero or not work:
                    continue
                trip_slopes.append({
                    "factor": factor,
                    "scope_order": order,
                    "dots": dots,
                    "split": "holdout" if dots == 3 else "development",
                    "measured_us_per_trip": (work["measured_us"] - zero["measured_us"]) / 14.0,
                    "predicted_us_per_trip": (work["predicted_us"] - zero["predicted_us"]) / 14.0,
                })

    complete = not missing and not unexpected and len(records) == len(expected)
    report = {
        "schema": "cv-dot-transition-structural-audit-v1",
        "measurement_target": "torch_npu_profiler_kernel_duration_median_us",
        "operator_identity_used": False,
        "development_dot_counts": [1, 2, 4],
        "holdout_dot_count": 3,
        "maximum_absolute_relative_error": args.maximum_error,
        "coverage_complete": complete,
        "missing": missing,
        "unexpected": unexpected,
        "fits": fits,
        "trip_slopes": trip_slopes,
        "holdouts": holdouts,
        "all_holdouts_pass": (complete and bool(holdouts) and all(row["pass"] for row in holdouts)),
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["all_holdouts_pass"] else 1)


if __name__ == "__main__":
    main()
