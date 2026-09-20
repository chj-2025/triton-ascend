from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[2]
    / "costmodel"
    / "profiles"
    / "microbench"
    / "data_provider"
    / "camodel"
    / "validate_stage_calibration.py"
)
SPEC = importlib.util.spec_from_file_location("validate_stage_calibration", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def _metadata() -> dict[str, object]:
    return {
        "same_binary": True,
        "target": "Ascend950PR_9579",
        "triton_ascend_sha": "ta-sha",
        "ascend_npu_ir_sha": "npuir-sha",
    }


def test_calibration_gate_accepts_phase_and_kernel_within_threshold() -> None:
    result = VALIDATOR.validate(
        {
            "metadata": _metadata(),
            "phases": [
                {
                    "id": "P1",
                    "measurement_kind": "phase_wall",
                    "predicted_system_cycles": 1000.0,
                    "measured_system_cycles": 950.0,
                }
            ],
            "kernel": {
                "id": "kernel",
                "measurement_kind": "kernel_wall",
                "predicted_system_cycles": 4000.0,
                "measured_system_cycles": 4200.0,
            },
        },
        0.10,
        None,
    )
    assert result["passed"] is True
    assert result["failed_records"] == []


def test_calibration_gate_rejects_service_window() -> None:
    payload = {
        "metadata": _metadata(),
        "phases": [
            {
                "id": "P2",
                "measurement_kind": "service_window",
                "predicted_system_cycles": 1000.0,
                "measured_system_cycles": 1000.0,
            }
        ],
    }
    with pytest.raises(ValueError, match="wall-time measurement"):
        VALIDATOR.validate(payload, 0.10, None)


def test_calibration_gate_requires_explicit_syscnt_conversion() -> None:
    payload = {
        "metadata": _metadata(),
        "phases": [
            {
                "id": "P3",
                "measurement_kind": "phase_wall",
                "predicted_system_cycles": 1000.0,
                "measured_syscnt_ticks": 1000.0,
            }
        ],
    }
    with pytest.raises(ValueError, match="syscnt_ticks_per_system_cycle"):
        VALIDATOR.validate(payload, 0.10, None)
