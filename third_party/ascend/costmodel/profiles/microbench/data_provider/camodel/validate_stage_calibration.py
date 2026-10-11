#!/usr/bin/env python3
"""Validate Stage/Phase and kernel cost calibration evidence.

The validator deliberately does not fit a correction factor.  It compares
predicted and measured values from the same binary and fails closed when the
measurement is a service window, when units are implicit, or when provenance
is incomplete.  A passing result therefore means that the supplied evidence
supports the <=10% acceptance rule; it does not make an unmeasured stage
appear calibrated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

EXIT_MISMATCH = 1
EXIT_EVIDENCE = 2


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _measurement(item: dict[str, Any], prefix: str, ticks_per_cycle: float | None) -> float:
    direct = f"{prefix}_system_cycles"
    if direct in item:
        return _finite_number(item[direct], direct)

    ticks_name = f"{prefix}_syscnt_ticks"
    if ticks_name not in item:
        raise ValueError(f"record {item.get('id', '<unnamed>')} needs {direct} or "
                         f"{ticks_name}")
    if ticks_per_cycle is None:
        raise ValueError(f"{ticks_name} requires metadata.syscnt_ticks_per_system_cycle "
                         "or --syscnt-ticks-per-system-cycle")
    ticks = _finite_number(item[ticks_name], ticks_name)
    if ticks < 0.0 or ticks_per_cycle <= 0.0:
        raise ValueError(f"{ticks_name} and tick frequency must be positive")
    return ticks / ticks_per_cycle


def _relative_error(predicted: float, measured: float) -> tuple[float, float]:
    absolute = abs(predicted - measured)
    if measured == 0.0:
        return absolute, 0.0 if absolute == 0.0 else math.inf
    return absolute, absolute / abs(measured)


def _validate_record(item: Any, ticks_per_cycle: float | None) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("each phase record must be an object")
    record_id = item.get("id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError("each phase record needs a non-empty id")
    kind = item.get("measurement_kind")
    if kind not in {"phase_wall", "wall", "kernel_wall"}:
        raise ValueError(f"{record_id}: measurement_kind must be phase_wall, wall, or "
                         f"kernel_wall; {kind!r} is not a wall-time measurement")
    predicted = _finite_number(
        item.get("predicted_system_cycles"),
        f"{record_id}.predicted_system_cycles",
    )
    measured = _measurement(item, "measured", ticks_per_cycle)
    if predicted < 0.0 or measured < 0.0:
        raise ValueError(f"{record_id}: costs must be non-negative")
    absolute, relative = _relative_error(predicted, measured)
    return {
        "id": record_id,
        "measurement_kind": kind,
        "predicted_system_cycles": predicted,
        "measured_system_cycles": measured,
        "absolute_error_system_cycles": absolute,
        "relative_error": relative,
    }


def validate(payload: Any, threshold: float, override_ticks: float | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata is required")
    if metadata.get("same_binary") is not True:
        raise ValueError("metadata.same_binary must be true")
    for key in ("target", "triton_ascend_sha", "ascend_npu_ir_sha"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"metadata.{key} is required")

    ticks_value = override_ticks
    if ticks_value is None and "syscnt_ticks_per_system_cycle" in metadata:
        ticks_value = _finite_number(
            metadata["syscnt_ticks_per_system_cycle"],
            "metadata.syscnt_ticks_per_system_cycle",
        )
    if ticks_value is not None and ticks_value <= 0.0:
        raise ValueError("syscnt tick conversion must be positive")

    phases = payload.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("phases must be a non-empty array")
    records = [_validate_record(item, ticks_value) for item in phases]
    kernel = payload.get("kernel")
    if kernel is not None:
        records.append(_validate_record(kernel, ticks_value))

    for record in records:
        record["pass"] = record["relative_error"] <= threshold
    failed = [record["id"] for record in records if not record["pass"]]
    return {
        "schema_version": 1,
        "threshold": threshold,
        "metadata": metadata,
        "records": records,
        "passed": not failed,
        "failed_records": failed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--syscnt-ticks-per-system-cycle", type=float)
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or args.threshold < 0.0:
        parser.error("--threshold must be finite and non-negative")
    try:
        payload = json.loads(args.evidence.read_text(encoding="utf-8"))
        result = validate(payload, args.threshold, args.syscnt_ticks_per_system_cycle)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"calibration evidence error: {exc}", file=sys.stderr)
        return EXIT_EVIDENCE

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        try:
            args.output.write_text(encoded, encoding="utf-8")
        except OSError as exc:
            print(f"cannot write {args.output}: {exc}", file=sys.stderr)
            return EXIT_EVIDENCE
    else:
        print(encoded, end="")
    if result["passed"]:
        return 0
    print(
        "calibration gate failed: "
        f"{len(result['failed_records'])} record(s) exceed "
        f"{args.threshold:.1%}",
        file=sys.stderr,
    )
    return EXIT_MISMATCH


if __name__ == "__main__":
    raise SystemExit(main())
