#!/usr/bin/env python3
"""Fail closed when the cross-operator model contract is weakened."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path

FORBIDDEN_ONLINE_FEATURES = {
    "operator_name",
    "kernel_name",
    "source_path",
    "exact_shape_signature",
}
LEGACY_CASE_FIT_KEYS = {
    "whole_simd_completion_curves",
    "whole_simt_completion_curves",
    "dot_component_private_stack_floors",
}
STANDARD_ROUTE_KEYS = {
    ("auto", 1),
    ("simd", 1),
    ("simt_only", 1),
    ("simt_only", 2),
    ("simt_only", 4),
}


def literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{path} has no literal assignment named {name}")


def walk_keys(value, path=""):
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            yield child_path, key, child
            yield from walk_keys(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_keys(child, f"{path}[{index}]")


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=here / "generalization_matrix.json")
    parser.add_argument("--fit-script", type=Path, default=here / "fit_general_route_residual.py")
    parser.add_argument("--profile", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy = manifest["policy"]
    if set(policy["forbidden_online_features"]) != FORBIDDEN_ONLINE_FEATURES:
        raise ValueError("manifest forbidden_online_features changed")
    if policy.get("split_unit") != "operator_family":
        raise ValueError("generalization split must remain operator_family")
    if float(policy.get("maximum_absolute_relative_error", 1.0)) > 0.10:
        raise ValueError("generalization error gate may not exceed 10%")

    names = set()
    expected_measurements = 0
    families_by_split: defaultdict[str, set[str]] = defaultdict(set)
    cases_by_family: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for case in manifest["cases"]:
        if case["name"] in names:
            raise ValueError(f"duplicate case name {case['name']}")
        names.add(case["name"])
        split = case["split"]
        if split not in ("train", "holdout"):
            raise ValueError(f"invalid split {split}")
        family = case["family"]
        families_by_split[split].add(family)
        cases_by_family[(split, family)].add(case["name"])
        route_keys = set(STANDARD_ROUTE_KEYS)
        for entry in case.get("extra_modes", []):
            key = (entry["route"], int(entry.get("factor", 1)))
            if key in route_keys:
                raise ValueError(f"duplicate route {key} for {case['name']}")
            route_keys.add(key)
        expected_measurements += len(route_keys)
        driver = (args.manifest.parent / case["driver"]).resolve()
        if not driver.is_file():
            raise FileNotFoundError(driver)
    overlap = families_by_split["train"] & families_by_split["holdout"]
    if overlap:
        raise ValueError(f"operator-family leakage across splits: {sorted(overlap)}")
    if len(families_by_split["train"]) < 3:
        raise ValueError("need at least three training operator families")
    if len(families_by_split["holdout"]) < 5:
        raise ValueError("need at least five held-out operator families")
    for family in families_by_split["train"]:
        if len(cases_by_family[("train", family)]) < 2:
            raise ValueError(f"training family {family} needs two independent cases")

    model_features = tuple(literal_assignment(args.fit_script, "MODEL_FEATURES"))
    if len(model_features) > 16 or len(model_features) != len(set(model_features)):
        raise ValueError("MODEL_FEATURES must stay small and unique")
    leaked = FORBIDDEN_ONLINE_FEATURES & set(model_features)
    if leaked:
        raise ValueError(f"forbidden online features: {sorted(leaked)}")

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if int(profile.get("schema_version", 0)) != 19:
        raise ValueError("production candidate must use generic schema 19")
    for path, key, value in walk_keys(profile):
        if key in FORBIDDEN_ONLINE_FEATURES:
            raise ValueError(f"forbidden profile key at {path}")
        if key in LEGACY_CASE_FIT_KEYS and value:
            raise ValueError(f"legacy case-fit data enabled at {path}")
    timing = profile.get("kernel_timing") or {}
    boundary = timing.get("boundary_residual_system_cycles_by_physical_program_count")
    if not isinstance(boundary, dict) or len(boundary) < 4:
        raise ValueError("generic physical-program kernel boundary curve is missing")

    result = {
        "schema": "costmodel-generalization-contract-audit-v1",
        "status": "PASS",
        "profile_version": profile.get("profile_version"),
        "training_families": sorted(families_by_split["train"]),
        "heldout_families": sorted(families_by_split["holdout"]),
        "model_feature_count": len(model_features),
        "model_features": model_features,
        "forbidden_online_features": sorted(FORBIDDEN_ONLINE_FEATURES),
        "legacy_case_fit_entries_enabled": False,
        "case_count": len(manifest["cases"]),
        "expected_measurement_count": expected_measurements,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
