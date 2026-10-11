#!/usr/bin/env python3
"""Count lowering-generated synchronization by NPUIR function/core domain.

This is a provenance tool, not a latency model.  It records structural counts
needed to compare independent CV probes with operator holdouts; it never maps a
count to cycles without paired device measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

FUNCTION = re.compile(r"(?m)^\s*func\.func\s+@([^\s(]+)")
CORE = re.compile(r"hivm\.func_core_type\s*=\s*#hivm\.func_core_type<([^>]+)>")


def counts(text: str) -> dict[str, int]:
    result = {
        "sync_block_set_vector": text.count("sync_block_set[<VECTOR>"),
        "sync_block_wait_vector": text.count("sync_block_wait[<VECTOR>"),
        "sync_block_set_cube": text.count("sync_block_set[<CUBE>"),
        "sync_block_wait_cube": text.count("sync_block_wait[<CUBE>"),
        "set_flag": text.count("hivm.hir.set_flag"),
        "wait_flag": text.count("hivm.hir.wait_flag"),
        "pipe_barrier": text.count("hivm.hir.pipe_barrier"),
        "get_sub_block_idx": text.count("get_sub_block_idx"),
        "scope_calls": text.count("scope_call"),
    }
    result["sync_block_total"] = sum(value for key, value in result.items() if key.startswith("sync_block_"))
    return result


def analyze(path: Path) -> dict:
    text = path.read_text(errors="replace")
    matches = list(FUNCTION.finditer(text))
    functions = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.start():end]
        core = CORE.search(body[:min(len(body), 4096)])
        item = {
            "name": match.group(1),
            "core": core.group(1) if core else "unknown",
            **counts(body),
        }
        if item["sync_block_total"] or item["get_sub_block_idx"]:
            functions.append(item)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "whole_module": counts(text),
        "synchronizing_functions": functions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {
        "schema": "npuir-cross-core-sync-structure-v1",
        "latency_claim": None,
        "modules": [analyze(path) for path in args.inputs],
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
