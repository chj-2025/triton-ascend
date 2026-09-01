#!/usr/bin/env python3
"""Parse raw CAModel output into normalized per-unit instruction counts."""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SIMT_OP_RE = re.compile(r"\[(\d+)\].*?\b(SIMT_[A-Z0-9_]+)\b")
CORE_RE = re.compile(r"(core\d+)\.(veccore\d+)\.rvec\.simt\.(.+)\.dump$")

PRIMARY_SUFFIXES = (
    "lsu",
    "dvg0",
    "dvg1",
    "dvg2",
    "dvg3",
    "exu0",
    "exu1",
    "exu2",
    "exu3",
)

OP_GROUPS = {
    "memory": ("SIMT_LDG", "SIMT_STG", "SIMT_LDS", "SIMT_STS"),
    "shuffle": ("SIMT_SHFL", ),
    "predicate": ("SIMT_ISETP", "SIMT_ISETP_I", "SIMT_PLOP3"),
    "control": ("SIMT_BRANCH", "SIMT_END"),
    "float_alu": ("SIMT_FADD", "SIMT_FMUL", "SIMT_FMNMX", "SIMT_FMNMX_I"),
    "int_alu": ("SIMT_IADD", "SIMT_IADD_I", "SIMT_IADD_X", "SIMT_IADD_X_I", "SIMT_IMUL", "SIMT_SHFI", "SIMT_LOP3"),
    "move": ("SIMT_MOV", ),
}


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iter_primary_simt_dumps(dump_dir):
    for path in sorted(Path(dump_dir).glob("core*.veccore*.rvec.simt.*.dump")):
        match = CORE_RE.search(path.name)
        if not match:
            continue
        suffix = match.group(3)
        if suffix in PRIMARY_SUFFIXES:
            yield path, match.group(1), match.group(2), suffix


def _scan_simt_dump(path):
    counts = Counter()
    first_ts = {}
    last_ts = {}
    examples = {}
    with path.open(errors="ignore") as file:
        for line in file:
            match = SIMT_OP_RE.search(line)
            if not match:
                continue
            timestamp = int(match.group(1))
            op = match.group(2)
            counts[op] += 1
            first_ts.setdefault(op, timestamp)
            last_ts[op] = timestamp
            examples.setdefault(op, line.strip()[:180])
    return counts, first_ts, last_ts, examples


def _merge_counter(dst, src):
    for key, value in src.items():
        dst[key] += value


def _group_counts(op_counts):
    groups = Counter()
    matched = set()
    for group, prefixes in OP_GROUPS.items():
        for op, count in op_counts.items():
            if any(op == prefix or op.startswith(prefix + "_") for prefix in prefixes):
                groups[group] += count
                matched.add(op)
    for op, count in op_counts.items():
        if op not in matched:
            groups["other"] += count
    return groups


def _span_from_first_last(first, last):
    if not first or not last:
        return None
    return {
        "first": min(first.values()),
        "last": max(last.values()),
        "delta": max(last.values()) - min(first.values()),
    }


def _parse_instr_exe(root):
    result = {
        "files": [],
        "pipe_cycles": Counter(),
        "instr_cycles": Counter(),
        "instr_calls": Counter(),
    }
    for path in sorted(Path(root).glob("**/*instr_exe*.csv")):
        with path.open(newline="", errors="ignore") as file:
            rows = list(csv.DictReader(file))
        pipe_cycles = Counter()
        instr_cycles = Counter()
        instr_calls = Counter()
        for row in rows:
            instr = row.get("instr") or row.get("Instr") or row.get("instruction") or "<unknown>"
            pipe = row.get("pipe") or row.get("Pipe") or "<unknown>"
            cycles = _as_float(row.get("cycles") or row.get("Cycles") or row.get("cycle"))
            calls = _as_float(row.get("call_count") or row.get("Call Count") or row.get("count") or 1)
            pipe_cycles[pipe] += cycles
            instr_cycles[instr] += cycles
            instr_calls[instr] += calls
        result["files"].append({
            "path": str(path),
            "rows": len(rows),
            "pipe_cycles": dict(pipe_cycles.most_common()),
            "instr_cycles": dict(instr_cycles.most_common(50)),
        })
        _merge_counter(result["pipe_cycles"], pipe_cycles)
        _merge_counter(result["instr_cycles"], instr_cycles)
        _merge_counter(result["instr_calls"], instr_calls)
    result["pipe_cycles"] = dict(result["pipe_cycles"].most_common())
    result["instr_cycles"] = dict(result["instr_cycles"].most_common(80))
    result["instr_calls"] = dict(result["instr_calls"].most_common(80))
    return result


def _make_seed(op_counts, group_counts, span):
    total_ops = sum(op_counts.values())
    duration = span["delta"] if span else 0
    seed = {
        "duration_cycles": duration,
        "total_primary_simt_ops": total_ops,
        "naive_cycles_per_primary_op": duration / total_ops if total_ops else None,
        "groups": {},
    }
    for group, count in group_counts.items():
        seed["groups"][group] = {
            "ops": count,
            "ops_per_cycle": count / duration if duration else None,
            "naive_cycles_per_op": duration / count if count else None,
        }
    return seed


def extract(root):
    root = Path(root)
    dump_dir = root / "dump" if (root / "dump").is_dir() else root
    aggregate_counts = Counter()
    aggregate_first = {}
    aggregate_last = {}
    aggregate_examples = {}
    per_unit = defaultdict(lambda: {"counts": Counter(), "first_ts": {}, "last_ts": {}, "files": []})

    for path, core, veccore, _suffix in _iter_primary_simt_dumps(dump_dir):
        counts, first_ts, last_ts, examples = _scan_simt_dump(path)
        unit_key = f"{core}.{veccore}"
        per_unit[unit_key]["files"].append(path.name)
        _merge_counter(per_unit[unit_key]["counts"], counts)
        _merge_counter(aggregate_counts, counts)
        for op, timestamp in first_ts.items():
            per_unit[unit_key]["first_ts"].setdefault(op, timestamp)
            aggregate_first.setdefault(op, timestamp)
            aggregate_examples.setdefault(op, examples.get(op, ""))
        for op, timestamp in last_ts.items():
            per_unit[unit_key]["last_ts"][op] = max(per_unit[unit_key]["last_ts"].get(op, timestamp), timestamp)
            aggregate_last[op] = max(aggregate_last.get(op, timestamp), timestamp)

    per_unit_output = {}
    for unit, data in sorted(per_unit.items()):
        counts = data["counts"]
        groups = _group_counts(counts)
        span = _span_from_first_last(data["first_ts"], data["last_ts"])
        per_unit_output[unit] = {
            "files": data["files"],
            "op_counts": dict(counts.most_common()),
            "group_counts": dict(groups.most_common()),
            "span": span,
            "seed": _make_seed(counts, groups, span),
        }

    aggregate_groups = _group_counts(aggregate_counts)
    aggregate_span = _span_from_first_last(aggregate_first, aggregate_last)
    return {
        "source": str(root),
        "dump_dir": str(dump_dir),
        "primary_suffixes": list(PRIMARY_SUFFIXES),
        "aggregate": {
            "op_counts": dict(aggregate_counts.most_common()),
            "group_counts": dict(aggregate_groups.most_common()),
            "span": aggregate_span,
            "examples": {op: aggregate_examples[op]
                         for op in aggregate_counts
                         if op in aggregate_examples},
            "seed": _make_seed(aggregate_counts, aggregate_groups, aggregate_span),
        },
        "per_unit": per_unit_output,
        "instr_exe": _parse_instr_exe(root),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="OPPROF directory or CAModel dump directory")
    parser.add_argument("-o", "--output", type=Path, help="output normalized JSON path")
    args = parser.parse_args()

    result = extract(args.root)
    if not result["per_unit"]:
        parser.error(f"no supported primary SIMT dumps found under {result['dump_dir']}")
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
