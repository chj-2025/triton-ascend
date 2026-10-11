#!/usr/bin/env python3
"""Fit grid-stride components and validate gather+dot composition holdouts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEVELOPMENT_Q = {1, 3, 5, 8}
VALIDATION_Q = {2, 4}


def linear_fit(points: list[tuple[int, float]]) -> tuple[float, float]:
    count = len(points)
    sx = sum(x for x, _ in points)
    sy = sum(y for _, y in points)
    sxx = sum(x * x for x, _ in points)
    sxy = sum(x * y for x, y in points)
    denominator = count * sxx - sx * sx
    if count < 2 or denominator == 0:
        raise ValueError("at least two distinct development Q values are required")
    slope = (count * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / count
    return intercept, slope


def measured_average_us(record: dict) -> float:
    """Accept the current timer-neutral summary and the legacy profiler key."""
    value = record.get("average_us", record.get("profiler_average_us"))
    if value is None:
        raise ValueError(f"record has no average measurement: {record['path']}")
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--route-input", type=Path,
                        help="Route-only structural probes from the uncorrected cost model.")
    args = parser.parse_args()

    records = []
    for path in sorted(args.input.glob("*/summary.json")):
        record = json.loads(path.read_text())
        record["path"] = str(path)
        records.append(record)
    grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["requested_factor"], record["variant"])].append(record)

    route_predictions: dict[tuple[int, int], float] = {}
    if args.route_input:
        for path in sorted(args.route_input.glob("*/route.json")):
            route = json.loads(path.read_text())
            q = int(route["logical_programs_per_scheduling_slot"])
            elapsed = route["selected_route_elapsed_prediction"]
            fixed_cycles = float(elapsed["fixed_system_cycles"])
            frequency_mhz = float(elapsed["system_counter_frequency_mhz"])
            factor_routes = route["stage_model"]["all_simt_factor_routes"]
            for factor_text, factor_route in factor_routes.items():
                route_predictions[(int(factor_text),
                                   q)] = (float(factor_route["total_system_cycles"]) + fixed_cycles) / frequency_mhz

    fits = {}
    validations = []
    for factor in (1, 2, 4):
        component_fits = {}
        for variant in ("copy", "gather", "dot"):
            points = [(r["q"], measured_average_us(r)) for r in grouped[(factor, variant)] if r["q"] in DEVELOPMENT_Q]
            intercept, slope = linear_fit(points)
            component_fits[variant] = {"intercept_us": intercept, "body_us_per_q": slope}
        fits[str(factor)] = component_fits

        copy_fit = component_fits["copy"]
        gather_fit = component_fits["gather"]
        dot_fit = component_fits["dot"]
        composition_points = [(r["q"], measured_average_us(r))
                              for r in grouped[(factor, "gather_dot")]
                              if r["q"] in DEVELOPMENT_Q]
        composition_fit = None
        if len(composition_points) >= 2:
            composition_intercept, composition_slope = linear_fit(composition_points)
            independent_intercept = (gather_fit["intercept_us"] + dot_fit["intercept_us"] - copy_fit["intercept_us"])
            independent_slope = (gather_fit["body_us_per_q"] + dot_fit["body_us_per_q"] - copy_fit["body_us_per_q"])
            composition_fit = {
                "intercept_us": composition_intercept,
                "body_us_per_q": composition_slope,
                "independent_component_intercept_us": independent_intercept,
                "independent_component_body_us_per_q": independent_slope,
                "residual_startup_us": max(0.0, composition_intercept - independent_intercept),
                "residual_body_us_per_q": max(0.0, composition_slope - independent_slope),
            }
            component_fits["gather_dot_composition"] = composition_fit
        model_residual_fit = None
        model_residual_points = []
        if route_predictions:
            for record in grouped[(factor, "gather_dot")]:
                q = int(record["q"])
                if q not in DEVELOPMENT_Q or (factor, q) not in route_predictions:
                    continue
                model_residual_points.append((q, measured_average_us(record) - route_predictions[(factor, q)]))
            if len(model_residual_points) >= 2:
                residual_startup, residual_body = linear_fit(model_residual_points)
                model_residual_fit = {
                    "startup_adjustment_us":
                    residual_startup,
                    "body_adjustment_us_per_q":
                    residual_body,
                    "fit_points":
                    [{"q": q, "actual_minus_uncorrected_model_us": value} for q, value in model_residual_points],
                }
                component_fits["uncorrected_model_composition_residual"] = (model_residual_fit)
        for record in grouped[(factor, "gather_dot")]:
            if record["q"] not in VALIDATION_Q:
                continue
            q = record["q"]
            # Inclusion-exclusion removes the duplicated loop/control and
            # store baseline.  No gather+dot measurement enters this formula.
            predicted = ((gather_fit["intercept_us"] + q * gather_fit["body_us_per_q"]) +
                         (dot_fit["intercept_us"] + q * dot_fit["body_us_per_q"]) -
                         (copy_fit["intercept_us"] + q * copy_fit["body_us_per_q"]))
            actual = measured_average_us(record)
            validation = {
                "factor": factor,
                "q": q,
                "predicted_us_from_components": predicted,
                "actual_gather_dot_us": actual,
                "relative_error": predicted / actual - 1.0,
                "lowered_instruction_counts": record["lowered_instruction_counts"],
            }
            if composition_fit is not None:
                applied = (
                    composition_fit["independent_component_intercept_us"] + composition_fit["residual_startup_us"] + q *
                    (composition_fit["independent_component_body_us_per_q"] + composition_fit["residual_body_us_per_q"])
                )
                validation["predicted_us_with_nonnegative_residual"] = applied
                validation["composition_relative_error"] = applied / actual - 1.0
            if model_residual_fit is not None and (factor, q) in route_predictions:
                uncorrected = route_predictions[(factor, q)]
                corrected = (uncorrected + model_residual_fit["startup_adjustment_us"] +
                             q * model_residual_fit["body_adjustment_us_per_q"])
                validation["uncorrected_model_prediction_us"] = uncorrected
                validation["uncorrected_model_relative_error"] = (uncorrected / actual - 1.0)
                validation["corrected_model_prediction_us"] = corrected
                validation["corrected_model_relative_error"] = (corrected / actual - 1.0)
            validations.append(validation)

    output = {
        "measurement_target": "torch_npu_profiler_kernel_average_us",
        "development_q": sorted(DEVELOPMENT_Q),
        "validation_q": sorted(VALIDATION_Q),
        "component_fits": fits,
        "gather_dot_validations": validations,
        "record_count": len(records),
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
