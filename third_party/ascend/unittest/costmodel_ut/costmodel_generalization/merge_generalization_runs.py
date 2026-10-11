#!/usr/bin/env python3
"""Merge independently executed device shards without copying their dumps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    records = []
    shards = []
    seen = set()
    manifests = set()
    profile_hashes = set()
    profile_snapshots = []
    manifest_hashes = set()
    manifest_snapshots = []
    toolchain_states = set()
    for root in args.input_root:
        path = root / "matrix_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary.get("source_stable_during_run", False):
            raise ValueError(f"source changed or was not audited during shard {root}")
        manifests.add(summary.get("manifest"))
        profile_hashes.add(summary.get("profile_sha256"))
        profile_snapshots.append(Path(summary["profile_snapshot"]))
        manifest_hashes.add(summary.get("manifest_sha256"))
        manifest_snapshots.append(Path(summary["manifest_snapshot"]))
        toolchain_states.add(
            tuple(
                sorted((
                    name,
                    state.get("sha"),
                    state.get("source_state_sha256"),
                ) for name, state in (summary.get("toolchain") or {}).items())))
        ending_state = tuple(
            sorted((
                name,
                state.get("sha"),
                state.get("source_state_sha256"),
            ) for name, state in (summary.get("toolchain_end") or {}).items()))
        if ending_state != next(iter(toolchain_states)):
            raise ValueError(f"toolchain state changed during shard {root}")
        shards.append({
            "root": str(root.resolve()),
            "device": summary.get("device"),
            "environment": summary.get("environment"),
            "toolchain": summary.get("toolchain"),
            "device_snapshot": summary.get("device_snapshot"),
            "shard": summary.get("shard"),
        })
        for record in summary["records"]:
            key = record["name"]
            if key in seen:
                raise ValueError(f"duplicate measurement {key}")
            seen.add(key)
            copied = dict(record)
            copied.setdefault("evidence_dir", str((root / key).resolve()))
            records.append(copied)
    if len(manifests) != 1:
        raise ValueError(f"shards used different manifests: {sorted(manifests)}")
    if len(profile_hashes) != 1 or None in profile_hashes:
        rendered_hashes = sorted("<missing>" if value is None else value for value in profile_hashes)
        raise ValueError(f"shards used different profiles: {rendered_hashes}")
    if len(manifest_hashes) != 1 or None in manifest_hashes:
        rendered_hashes = sorted("<missing>" if value is None else value for value in manifest_hashes)
        raise ValueError(f"shards used different manifests: {rendered_hashes}")
    if len(toolchain_states) != 1 or not next(iter(toolchain_states), ()):
        raise ValueError("shards used different or missing toolchain source states")

    records.sort(key=lambda row: row["name"])
    result = {
        "schema": "triton-costmodel-generalization-merged-run-v1",
        "merged_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": manifests.pop(),
        "records": records,
        "shards": shards,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    merged_profile = args.output_root / "costmodel_profile.json"
    shutil.copy2(profile_snapshots[0], merged_profile)
    merged_manifest = args.output_root / "generalization_matrix.json"
    shutil.copy2(manifest_snapshots[0], merged_manifest)
    result["profile_snapshot"] = str(merged_profile.resolve())
    result["profile_sha256"] = profile_hashes.pop()
    result["manifest"] = str(merged_manifest.resolve())
    result["manifest_snapshot"] = str(merged_manifest.resolve())
    result["manifest_sha256"] = manifest_hashes.pop()
    result["toolchain_state"] = [{"name": name, "sha": sha, "source_state_sha256": state_sha}
                                 for name, sha, state_sha in next(iter(toolchain_states))]
    (args.output_root / "matrix_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True),
                                                          encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(1 if any(row["returncode"] for row in records) else 0)


if __name__ == "__main__":
    main()
