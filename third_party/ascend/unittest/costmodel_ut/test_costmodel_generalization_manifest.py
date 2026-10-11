from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

ROOT = Path(__file__).with_name("costmodel_generalization")
MAIN = ROOT / "generalization_matrix.json"
SNAPSHOT = ROOT / "generalization_matrix_v140_all_100.json"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_canonical_matrix_contains_both_historical_holdouts() -> None:
    main = load(MAIN)
    first = load(ROOT / "pytest_ut_structural_first_12_manifest.json")
    second = load(ROOT / "pytest_ut_structural_17_manifest.json")

    cases = main["cases"]
    names = [case["name"] for case in cases]
    first_names = {case["name"] for case in first["cases"]}
    second_names = {case["name"] for case in second["cases"]}

    assert len(cases) == 100
    assert len(names) == len(set(names))
    assert len(first_names) == 12
    assert len(second_names) == 17
    assert first_names.isdisjoint(second_names)
    assert first_names | second_names <= set(names)
    assert MAIN.read_bytes() == SNAPSHOT.read_bytes()

    regression_sets = {entry["name"]: entry["case_count"] for entry in main["regression_sets"]}
    assert regression_sets == {
        "historical_main_v105": 71,
        "pytest_ut_structural_second_holdout": 17,
        "pytest_ut_structural_first_holdout": 12,
    }


def test_every_case_has_a_repository_local_manual_driver() -> None:
    cases = load(MAIN)["cases"]
    for case in cases:
        driver_text = case["driver"]
        driver_path = PurePosixPath(driver_text)
        assert not driver_path.is_absolute()
        assert ".." not in driver_path.parts

        driver = ROOT.joinpath(*driver_path.parts)
        assert driver.is_file(), f"missing driver for {case['name']}: {driver}"
        source = driver.read_text(encoding="utf-8")
        factor_argument = case.get("factor_argument", "--factor")
        for flag in ("--route", factor_argument, "--warmup", "--repeats", "--report"):
            assert flag in source, f"{driver} does not expose {flag}"


def test_manual_runner_lists_all_cases_without_an_npu() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "run_generalization_matrix.py"), "--list-cases"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    listed = [line for line in completed.stdout.splitlines() if line]
    assert len(listed) == 100
    assert listed[0].startswith("000\t")
    assert listed[-1].startswith("099\t")
