#!/usr/bin/env python3
"""Fit and audit a generic SIMD or whole-SIMT residual.

The regression target is the part of measured elapsed time not already owned
by the analytical Stage resources and the measured kernel-boundary residual.
Inputs are a fixed allowlist of numeric TTIR/Stage facts.  Case names, source
paths and exact tensor shapes never enter the feature vector.

This script is an offline promotion gate.  It does not edit the production
profile.  A fit is eligible for review only when every held-out operator family
stays within the requested whole-kernel error bound.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

PRESSURE_FEATURES = (
    "peak_live_32bit_units_per_thread",
    "peak_tensor_32bit_units_per_thread",
    "peak_scalar_32bit_units_per_thread",
    "peak_cross_stage_32bit_units_per_thread",
    "peak_loop_carried_32bit_units_per_thread",
    "peak_rematerializable_32bit_units_per_thread",
    "peak_live_predicate_units_per_thread",
    "predicate_definition_units_per_thread",
    "predicate_use_units_per_thread",
    "max_predicate_fanout",
    "ttir_integer_equality_predicate_definition_units_per_thread",
    "ttir_integer_ordered_predicate_definition_units_per_thread",
    "ttir_predicate_shape_definition_units_per_thread",
    "ttir_other_predicate_definition_units_per_thread",
    "ttir_predicate_dag_edge_units_per_thread",
    "ttir_max_predicate_dag_depth",
    "ttir_select_predicate_use_units_per_thread",
    "ttir_boolean_predicate_use_units_per_thread",
    "loop_invariant_predicate_units_per_thread",
    "loop_invariant_predicate_use_units_per_thread",
    "peak_pointer_32bit_units_per_thread",
    "peak_block_pointer_count",
    "peak_block_pointer_rank_sum",
    "unknown_value_count",
    "ttir_dot_definition_count",
    "ttir_dot_dependency_edge_count",
    "ttir_dot_dependency_source_count",
    "ttir_dot_dependency_sink_count",
    "ttir_max_dot_dependency_depth",
    "ttir_maximum_weak_dot_component_nodes",
)

WORKLOAD_FEATURES = (
    "load_bytes_per_iteration",
    "store_bytes_per_iteration",
    "load_operations_per_iteration",
    "store_operations_per_iteration",
    "load_warp_instructions_per_iteration",
    "store_warp_instructions_per_iteration",
    "scalar_operations_per_iteration",
    "predicate_elements_per_iteration",
    "shuffle_lane_steps_per_iteration",
    "dot_flops_per_iteration",
    "dot_operations_per_iteration",
    "loop_backedges_per_iteration",
    "conditional_branches_per_iteration",
    "divergent_branches_per_iteration",
    "synchronizations_per_iteration",
    "issue_elements_per_iteration",
)

# Deliberately small and fixed.  These are architecture/IR quantities with a
# direct physical interpretation; no dynamic one-hot Stage or case identity is
# allowed into the production candidate.  A fixed list also prevents a new
# benchmark from silently expanding the model dimension.
MODEL_FEATURES = (
    "active_warps",
    "base_group_system_cycles",
    "logical_stage_count",
    "derived.peak_live_units",
    "derived.peak_cross_stage_units",
    "derived.peak_loop_carried_units",
    "derived.peak_pointer_units",
    "derived.predicate_complexity",
    "derived.dot_definitions",
    "derived.dot_dependency_complexity",
    "derived.dot_live_context_units",
    "derived.memory_warp_instructions",
    "derived.scalar_operations",
    "derived.issue_elements",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def number(mapping: dict, key: str) -> float:
    value = mapping.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


def candidate_route(route: dict, mode: str, factor: int) -> dict:
    if mode == "simd":
        if factor != 1:
            raise ValueError("SIMD has no SuperBlock factor")
        return route["stage_model"]["routes"]["all_simd"]
    return route["stage_model"]["all_simt_factor_routes"][str(factor)]


def structural_features(route: dict, mode: str, factor: int) -> dict[str, float]:
    factor_route = candidate_route(route, mode, factor)
    runtime_iterations = float(factor_route["runtime_loop_iteration_count"])
    base_warps = (max(
        (number(
            stage.get("whole_simt_register_pressure") or {},
            "base_warp_count",
        ) for stage in route["stage_model"]["logical_stages"]),
        default=1.0,
    ) if mode == "simt" else 0.0)
    features: dict[str, float] = {
        "superblock_factor": float(factor),
        "active_warps": base_warps * factor,
        "runtime_loop_iterations": runtime_iterations,
        "logical_programs_per_slot": float(factor_route["runtime_logical_programs_per_scheduling_slot"]),
        "tail_program_count": float(factor_route["runtime_tail_program_count"]),
        "base_group_system_cycles": float(factor_route["total_system_cycles"]) / max(1.0, runtime_iterations),
        "logical_stage_count": float(len(route["stage_model"]["logical_stages"])),
    }
    model_counts: defaultdict[str, float] = defaultdict(float)
    for stage in route["stage_model"]["logical_stages"]:
        iterations = float(stage.get("iteration_count", 1.0))
        model_counts[stage.get("model", "unknown")] += iterations
        # These fields describe the prospective whole-SIMT function.  They are
        # not valid SIMD register-pressure facts, so keep them out of the SIMD
        # feature vector instead of transferring a SIMT spill proxy by name.
        pressure = (stage.get("whole_simt_register_pressure") or {} if mode == "simt" else {})
        for key in PRESSURE_FEATURES:
            value = number(pressure, key)
            features[f"pressure_max.{key}"] = max(features.get(f"pressure_max.{key}", 0.0), value)
            features[f"pressure_sum.{key}"] = (features.get(f"pressure_sum.{key}", 0.0) + value)
        dot_sites = pressure.get("dot_live_contexts") or []
        for site in dot_sites:
            for key in (
                    "operand_32bit_units_per_thread",
                    "retained_tensor_32bit_units_per_thread",
                    "retained_scalar_32bit_units_per_thread",
            ):
                value = number(site, key)
                features[f"dot_site_max.{key}"] = max(features.get(f"dot_site_max.{key}", 0.0), value)
                features[f"dot_site_sum.{key}"] = (features.get(f"dot_site_sum.{key}", 0.0) + value)
        workload = stage.get("workload") or {}
        for key in WORKLOAD_FEATURES:
            features[f"workload.{key}"] = (features.get(f"workload.{key}", 0.0) + number(workload, key) * iterations)
    for key, value in model_counts.items():
        # The model kind is a generic Stage class, not a kernel/operator name.
        features[f"stage_model.{key}"] = value
    features["derived.peak_live_units"] = features.get("pressure_max.peak_live_32bit_units_per_thread", 0.0)
    features["derived.peak_cross_stage_units"] = features.get("pressure_max.peak_cross_stage_32bit_units_per_thread",
                                                              0.0)
    features["derived.peak_loop_carried_units"] = features.get("pressure_max.peak_loop_carried_32bit_units_per_thread",
                                                               0.0)
    features["derived.peak_pointer_units"] = features.get("pressure_max.peak_pointer_32bit_units_per_thread", 0.0)
    features["derived.predicate_complexity"] = sum(
        features.get(f"pressure_sum.{key}", 0.0) for key in (
            "predicate_definition_units_per_thread",
            "predicate_use_units_per_thread",
            "ttir_predicate_dag_edge_units_per_thread",
        ))
    features["derived.dot_definitions"] = features.get("pressure_sum.ttir_dot_definition_count", 0.0)
    dot_edges = features.get("pressure_sum.ttir_dot_dependency_edge_count", 0.0)
    dot_depth = features.get("pressure_max.ttir_max_dot_dependency_depth", 0.0)
    features["derived.dot_dependency_complexity"] = dot_edges * max(1.0, dot_depth)
    features["derived.dot_live_context_units"] = sum(
        features.get(f"dot_site_sum.{key}", 0.0) for key in (
            "operand_32bit_units_per_thread",
            "retained_tensor_32bit_units_per_thread",
            "retained_scalar_32bit_units_per_thread",
        ))
    features["derived.memory_warp_instructions"] = sum(
        features.get(f"workload.{key}", 0.0) for key in (
            "load_warp_instructions_per_iteration",
            "store_warp_instructions_per_iteration",
        ))
    features["derived.scalar_operations"] = features.get("workload.scalar_operations_per_iteration", 0.0)
    features["derived.issue_elements"] = features.get("workload.issue_elements_per_iteration", 0.0)
    return features


def boundary_ticks(profile: dict, physical_programs: int) -> float:
    points = sorted(
        (int(key), float(value))
        for key, value in profile["kernel_timing"]["boundary_residual_system_cycles_by_physical_program_count"].items())
    if physical_programs <= points[0][0]:
        return points[0][1]
    if physical_programs >= points[-1][0]:
        return points[-1][1]
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if physical_programs <= right_x:
            ratio = (physical_programs - left_x) / (right_x - left_x)
            return left_y + ratio * (right_y - left_y)
    raise AssertionError("unreachable boundary interpolation")


def collect(root: Path, profile: dict, maximum_error: float, mode: str) -> tuple[list[dict], dict]:
    summary = load(root / "matrix_summary.json")
    metadata = {}
    locations = {}
    for row in summary["records"]:
        base = row["name"].split("__", 1)[0]
        metadata[base] = {"family": row["family"], "split": row["split"]}
        locations[(base, row["route"], int(row["factor"]))] = Path(row.get("evidence_dir", root / row["name"]))
    samples = []
    for base, meta in sorted(metadata.items()):
        auto_dir = locations.get((base, "auto", 1))
        if auto_dir is None or not (auto_dir / "route.json").is_file():
            continue
        route = load(auto_dir / "route.json")
        factors = (1, ) if mode == "simd" else (1, 2, 4)
        measured_route = "simd" if mode == "simd" else "simt_only"
        factor_routes = route["stage_model"].get("all_simt_factor_routes", {})
        for factor in factors:
            factor_dir = locations.get((base, measured_route, factor))
            if factor_dir is None:
                continue
            evidence_path = factor_dir / "evidence.json"
            if not evidence_path.is_file():
                continue
            if mode == "simt" and str(factor) not in factor_routes:
                continue
            evidence = load(evidence_path)
            benchmark = evidence.get("benchmark") or {}
            # StageCostModel predicts in-kernel execution time. Batched device
            # events include the launch/queue interval and have a visible floor
            # for tiny kernels, so they are diagnostics and must not train a
            # route residual.
            elapsed = (benchmark.get("profiler") or {}).get("average_us")
            if elapsed is None or not benchmark.get("correct", False):
                continue
            factor_route = candidate_route(route, mode, factor)
            if not factor_route.get("legal", False):
                continue
            base_ticks = float(factor_route["total_system_cycles"]) + boundary_ticks(
                profile, int(factor_route["runtime_physical_program_count"]))
            measured_ticks = float(elapsed) * 1000.0
            runtime_iterations = max(1.0, float(factor_route["runtime_loop_iteration_count"]))
            samples.append({
                "case": base,
                **meta,
                "factor": factor,
                "mode": mode,
                "features": structural_features(route, mode, factor),
                "measured_ticks": measured_ticks,
                "base_ticks": base_ticks,
                "target_group_residual_ticks": max(0.0, (measured_ticks - base_ticks) / runtime_iterations),
                "runtime_iterations": runtime_iterations,
                "maximum_error": maximum_error,
            })
    return samples, summary


def design(samples: list[dict], names: list[str]) -> np.ndarray:
    return np.asarray(
        [[sample["features"].get(name, 0.0) for name in names] for sample in samples],
        dtype=np.float64,
    )


def fit_ridge(training: list[dict], names: list[str], alpha: float) -> dict:
    raw = design(training, names)
    log_x = np.log1p(np.maximum(raw, 0.0))
    # Do not center: with non-negative coefficients this preserves monotonicity
    # in every structural pressure/work feature.
    scale = np.sqrt(np.mean(log_x * log_x, axis=0))
    scale[scale < 1.0e-9] = 1.0
    x = np.c_[np.ones(len(training)), log_x / scale]
    y = np.asarray([row["target_group_residual_ticks"] for row in training])
    regularizer = np.eye(x.shape[1], dtype=np.float64) * alpha
    regularizer[0, 0] = 0.0
    gram = x.T @ x + regularizer
    rhs = x.T @ y
    # Projected gradient solves the convex non-negative ridge problem.  The
    # resulting correction cannot decrease when pressure/work increases.
    lipschitz = max(1.0e-12, float(np.linalg.eigvalsh(gram).max()))
    coefficients = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(20000):
        updated = np.maximum(0.0, coefficients - (gram @ coefficients - rhs) / lipschitz)
        if np.linalg.norm(updated - coefficients) <= 1.0e-11 * max(1.0, np.linalg.norm(coefficients)):
            coefficients = updated
            break
        coefficients = updated
    return {
        "feature_names": names,
        "log_scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "alpha": alpha,
        "coefficient_constraint": "non_negative_monotonic",
    }


def predict(sample: dict, model: dict) -> float:
    raw = design([sample], model["feature_names"])[0]
    x = np.log1p(np.maximum(raw, 0.0)) / model["log_scale"]
    group = max(0.0, float(model["coefficients"][0] + x @ model["coefficients"][1:]))
    return sample["base_ticks"] + group * sample["runtime_iterations"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=HERE / "generalization_matrix.json")
    parser.add_argument("--mode", choices=("simd", "simt"), default="simt")
    parser.add_argument("--maximum-error", type=float, default=0.10)
    args = parser.parse_args()
    profile = load(args.profile)
    manifest = load(args.manifest)
    samples, matrix = collect(args.evidence_root, profile, args.maximum_error, args.mode)
    training = [row for row in samples if row["split"] == "train"]
    if len(training) < 6:
        raise RuntimeError(f"need at least six training points, got {len(training)}")
    names = list(MODEL_FEATURES)

    case_metadata = {row["name"]: (row["family"], row["split"]) for row in manifest["cases"]}
    training_cases_by_family: defaultdict[str, set[str]] = defaultdict(set)
    heldout_cases_by_family: defaultdict[str, set[str]] = defaultdict(set)
    for case, (family, split) in case_metadata.items():
        target = training_cases_by_family if split == "train" else heldout_cases_by_family
        target[family].add(case)
    factors = (1, ) if args.mode == "simd" else (1, 2, 4)
    expected_sample_keys = {(case, factor) for case in case_metadata for factor in factors}
    actual_sample_keys = {(row["case"], row["factor"]) for row in samples}
    missing_samples = sorted(expected_sample_keys - actual_sample_keys)
    unexpected_samples = sorted(actual_sample_keys - expected_sample_keys)
    sample_coverage_ok = not missing_samples and not unexpected_samples
    coverage_ok = (len(training_cases_by_family) >= 3 and len(heldout_cases_by_family) >= 5
                   and all(len(cases) >= 2 for cases in training_cases_by_family.values()) and sample_coverage_ok)

    # Select regularization with leave-one-operator-family-out validation.  A
    # family is never split across train and validation folds.
    families = sorted({row["family"] for row in training})
    candidates = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    scores = []
    for alpha in candidates:
        errors = []
        for family in families:
            fold_train = [row for row in training if row["family"] != family]
            fold_test = [row for row in training if row["family"] == family]
            if len(fold_train) < 2 or not fold_test:
                continue
            model = fit_ridge(fold_train, names, alpha)
            for row in fold_test:
                errors.append(abs(predict(row, model) / row["measured_ticks"] - 1.0))
        scores.append({
            "alpha": alpha,
            "family_holdout_max_absolute_error": max(errors, default=float("inf")),
            "family_holdout_mean_absolute_error": float(np.mean(errors)) if errors else float("inf"),
        })
    selected = min(
        scores,
        key=lambda row: (
            row["family_holdout_max_absolute_error"],
            row["family_holdout_mean_absolute_error"],
        ),
    )
    model = fit_ridge(training, names, selected["alpha"])
    results = []
    for row in samples:
        predicted = predict(row, model)
        error = predicted / row["measured_ticks"] - 1.0
        results.append({
            "case": row["case"],
            "family": row["family"],
            "split": row["split"],
            "factor": row["factor"],
            "base_us": row["base_ticks"] / 1000.0,
            "predicted_us": predicted / 1000.0,
            "measured_us": row["measured_ticks"] / 1000.0,
            "relative_error": error,
            "pass": abs(error) <= args.maximum_error,
        })
    heldout = [row for row in results if row["split"] == "holdout"]
    training_results = [row for row in results if row["split"] == "train"]
    family_holdout_pass = (selected["family_holdout_max_absolute_error"] <= args.maximum_error)
    final_training_pass = bool(training_results) and all(row["pass"] for row in training_results)
    production_gate = (coverage_ok and family_holdout_pass and final_training_pass and bool(heldout)
                       and all(row["pass"] for row in heldout))
    report = {
        "schema": "generic-route-residual-fit-v1",
        "measurement_target": "torch_npu_profiler_kernel_average_us",
        "mode": args.mode,
        "status": "eligible_for_production_review" if production_gate else "research_not_promoted",
        "forbidden_features_enforced": [
            "operator_name",
            "kernel_name",
            "source_path",
            "exact_shape_signature",
        ],
        "split_unit": "operator_family",
        "maximum_absolute_relative_error": args.maximum_error,
        "regularization_search": scores,
        "coverage": {
            "training_cases_by_family": {key: sorted(value)
                                         for key, value in training_cases_by_family.items()},
            "heldout_cases_by_family": {key: sorted(value)
                                        for key, value in heldout_cases_by_family.items()},
            "minimum_training_families": 3,
            "minimum_heldout_families": 5,
            "minimum_training_cases_per_family": 2,
            "expected_mode_samples": len(expected_sample_keys),
            "actual_mode_samples": len(actual_sample_keys),
            "missing_mode_samples": [{"case": case, "factor": factor} for case, factor in missing_samples],
            "unexpected_mode_samples": [{"case": case, "factor": factor} for case, factor in unexpected_samples],
            "pass": coverage_ok,
        },
        "training_family_holdout_pass": family_holdout_pass,
        "final_training_pass": final_training_pass,
        "model": model,
        "results": results,
        "heldout_all_pass": bool(heldout) and all(row["pass"] for row in heldout),
        "matrix_policy": manifest.get("policy"),
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
