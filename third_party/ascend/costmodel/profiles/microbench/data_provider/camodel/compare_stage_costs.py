"""Compare selected StageCostModel terms with CaModel instruction evidence.

The tool deliberately separates observation from attribution.  An instruction
CSV can always provide a whole-kernel envelope and resource-family totals, but
it can only provide per-Stage measurements when a compiler-produced PC range
map is supplied.  Missing attribution is reported as unobservable instead of
being filled with a residual or a workload-specific ratio.
"""

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

RESOURCE_KEYS = (
    "setup",
    "load_per_iteration",
    "store_per_iteration",
    "compute_per_iteration",
    "dot_per_iteration",
    "scalar_per_iteration",
    "predicate_per_iteration",
    "shuffle_per_iteration",
    "branch_control_per_iteration",
    "loop_control_per_iteration",
    "divergence_per_iteration",
    "synchronization_per_iteration",
    "spill_per_iteration",
    "issue_per_iteration",
    "critical_path_per_iteration",
    "epilogue",
)

FAMILY_RESOURCE_KEYS = {
    "load": ("load_per_iteration", ),
    "store": ("store_per_iteration", ),
    "compute": ("compute_per_iteration", ),
    "dot": ("dot_per_iteration", ),
    "scalar": ("scalar_per_iteration", ),
    "predicate": ("predicate_per_iteration", ),
    "shuffle": ("shuffle_per_iteration", ),
    "control": ("branch_control_per_iteration", "loop_control_per_iteration", "divergence_per_iteration"),
    "synchronization": ("synchronization_per_iteration", ),
    "spill": ("spill_per_iteration", ),
}


def parse_int(value):
    return int(str(value), 0)


def classify_instruction(name, pipe):
    upper_name = name.upper()
    upper_pipe = pipe.upper()
    if upper_name == "VF_SIMT":
        return "envelope"
    if "LD" in upper_pipe or upper_name.startswith(("SIMT_LD", "LD_")):
        return "load"
    if "ST" in upper_pipe or upper_name.startswith(("SIMT_ST", "ST_")):
        return "store"
    if "DOT" in upper_name or "MMAD" in upper_name or "CUBE" in upper_pipe:
        return "dot"
    if any(token in upper_name for token in ("SHFL", "SHUFFLE", "P2R", "R2P")):
        return "shuffle"
    if any(token in upper_name for token in ("SETP", "ISETP", "PRED")):
        return "predicate"
    if any(token in upper_name for token in ("BRANCH", "DVG", "LOOP")):
        return "control"
    if any(token in upper_name for token in ("SYNC", "SET_FLAG", "WAIT_FLAG")):
        return "synchronization"
    if upper_name.endswith(("STK", "LDK")):
        return "spill"
    if "SCALAR" in upper_pipe:
        return "scalar"
    return "compute"


def load_pc_map(path):
    if path is None:
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    ranges = data.get("stages", data)
    normalized = []
    for stage in ranges:
        normalized.append({
            "id": stage["id"],
            "begin": parse_int(stage["pc_begin"]),
            "end": parse_int(stage["pc_end"]),
        })
    return normalized


def stage_for_pc(pc, ranges):
    matches = [entry["id"] for entry in ranges if entry["begin"] <= pc < entry["end"]]
    if len(matches) > 1:
        raise RuntimeError(f"overlapping Stage PC ranges for {pc:#x}: {matches}")
    return matches[0] if matches else None


def read_camodel_csv(path, ranges):
    aggregate = defaultdict(lambda: {"static_instructions": 0, "call_count": 0, "cycles": 0.0, "running_time_us": 0.0})
    per_stage = defaultdict(lambda: defaultdict(lambda: {
        "static_instructions": 0,
        "call_count": 0,
        "cycles": 0.0,
        "running_time_us": 0.0,
    }))
    unmapped = []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            family = classify_instruction(row["instr"], row.get("pipe", ""))
            values = {
                "static_instructions": 1,
                "call_count": int(row["call_count"]),
                "cycles": float(row["cycles"]),
                "running_time_us": float(row["running_time(us)"]),
            }
            for key, value in values.items():
                aggregate[family][key] += value
            if not ranges or family == "envelope":
                continue
            pc = parse_int(row["addr"])
            stage_id = stage_for_pc(pc, ranges)
            if stage_id is None:
                unmapped.append(f"{pc:#x}")
                continue
            for key, value in values.items():
                per_stage[stage_id][family][key] += value
    return aggregate, per_stage, sorted(set(unmapped))


def addr2line(binary, addresses, load_bias, executable):
    relative = [address - load_bias for address in addresses]
    if any(address < 0 for address in relative):
        raise RuntimeError("CaModel instruction address is below the supplied load bias")
    command = [str(executable), "-a", "-i", "-e", str(binary), *(hex(address) for address in relative)]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stacks = defaultdict(list)
    current = None
    for line in result.stdout.splitlines():
        if re.fullmatch(r"0x[0-9a-fA-F]+", line):
            current = int(line, 16) + load_bias
        elif current is not None:
            stacks[current].append(line)
    locations = {}
    for address in addresses:
        stack = stacks.get(address, [])
        # Inline stacks often start in a CCE intrinsic or library helper.  The
        # Triton source frame is the evidence required by Stage provenance.
        locations[address] = next((line for line in stack if ".py:" in line), stack[0] if stack else "??:0")
    return locations


def family_is_predicted(stage, family):
    keys = FAMILY_RESOURCE_KEYS.get(family, ())
    resources = stage["predicted_resource_system_cycles"]
    return any(float(resources.get(key, 0.0)) > 0.0 for key in keys)


def read_instruction_source_attribution(path, predicted, binary, load_bias, executable):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    addresses = sorted({parse_int(row["addr"]) for row in rows})
    locations = addr2line(binary, addresses, load_bias, executable)
    source_owners = build_source_owners(predicted)
    stage_by_id = {stage["id"]: stage for stage in predicted}
    per_stage = defaultdict(lambda: defaultdict(lambda: {
        "static_instructions": 0,
        "call_count": 0,
        "cycles": 0.0,
        "running_time_us": 0.0,
    }))
    ambiguous = defaultdict(set)
    unmatched = set()
    for row in rows:
        family = classify_instruction(row["instr"], row.get("pipe", ""))
        if family == "envelope":
            continue
        key = source_line_key(locations[parse_int(row["addr"])])
        if key is None:
            continue
        candidates = {
            stage_id
            for stage_id in source_owners.get(key, set())
            if family_is_predicted(stage_by_id[stage_id], family)
        }
        evidence = f"{key}:{family}"
        if not candidates:
            unmatched.add(evidence)
            continue
        if len(candidates) != 1:
            ambiguous[evidence].update(candidates)
            continue
        target = per_stage[next(iter(candidates))][family]
        target["static_instructions"] += 1
        target["call_count"] += int(row["call_count"])
        target["cycles"] += float(row["cycles"])
        target["running_time_us"] += float(row["running_time(us)"])
    return per_stage, {key: sorted(value) for key, value in ambiguous.items()}, sorted(unmatched)


def source_line_key(value):
    match = re.search(r'([^/\\"()]+\.py)"?:(\d+)', value)
    if not match:
        return None
    return f"{match.group(1)}:{match.group(2)}"


def build_source_owners(predicted):
    owners = defaultdict(set)
    for stage in predicted:
        for location in stage.get("source_locations", []):
            key = source_line_key(location)
            if key:
                owners[key].add(stage["id"])
    return owners


def read_code_correlation(path, predicted):
    owners = build_source_owners(predicted)
    per_stage = defaultdict(lambda: {
        "rows": 0,
        "call_count": 0,
        "cycles": 0.0,
        "running_time_us": 0.0,
        "source_lines": [],
    })
    ambiguous = defaultdict(set)
    unmatched = set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = source_line_key(row.get("code", ""))
            if key is None:
                continue
            stage_ids = owners.get(key, set())
            if not stage_ids:
                unmatched.add(key)
                continue
            if len(stage_ids) != 1:
                ambiguous[key].update(stage_ids)
                continue
            stage_id = next(iter(stage_ids))
            target = per_stage[stage_id]
            target["rows"] += 1
            target["call_count"] += int(float(row.get("call_count") or 0))
            target["cycles"] += float(row.get("cycles") or 0)
            target["running_time_us"] += float(row.get("running_time(us)") or 0)
            target["source_lines"].append(key)
    for value in per_stage.values():
        value["source_lines"] = sorted(set(value["source_lines"]))
    return per_stage, {key: sorted(value) for key, value in ambiguous.items()}, sorted(unmatched)


def find_stage_model(report):
    model = report.get("stage_model") or report.get("stage_cost_model")
    if model is None:
        raise RuntimeError("report does not contain stage_model")
    return model


def flatten_stages(model):
    if "logical_stages" not in model:
        raise RuntimeError("stage_model does not contain logical_stages")
    return model["logical_stages"]


def selected_route(model, route_name):
    routes = model["routes"]
    aliases = {
        "all_simd": "all_simd",
        "all_simt": "all_simt_only",
        "mixed": "mixed_simd_simt",
    }
    route = routes[aliases[route_name]]
    if not route.get("legal", False):
        raise RuntimeError(f"requested route {route_name} is illegal")
    return route


def implementation_for(stage, selection):
    wanted = selection["implementation"]
    for implementation in stage["implementations"]:
        actual = implementation["implementation"]
        if (actual["mode"] == wanted["mode"]
                and int(actual.get("superblock_factor", 1)) == int(wanted.get("superblock_factor", 1))
                and bool(actual.get("local_scope", False)) == bool(wanted.get("local_scope", False))):
            return implementation
    raise RuntimeError(f"selected implementation missing from Stage {stage['id']}")


def predicted_stage_rows(model, route_name):
    stages = flatten_stages(model)
    route = selected_route(model, route_name)
    selections = route["stages"]
    if len(stages) != len(selections):
        raise RuntimeError("route selection count does not match logical Stage count")
    rows = []
    for stage, selection in zip(stages, selections):
        implementation = implementation_for(stage, selection)
        resources = implementation["resource_system_cycles"]
        rows.append({
            "id": stage["id"],
            "kind": stage.get("model"),
            "source_locations": stage.get("source_locations", []),
            "implementation": selection["implementation"],
            "iteration_count": stage.get("iteration_count", 1),
            "predicted_total_system_cycles": selection["logical_stage_system_cycles"],
            "predicted_entry_transition_system_cycles": selection.get("entry_transition_system_cycles", 0.0),
            "predicted_resource_system_cycles": {key: resources.get(key, 0.0)
                                                 for key in RESOURCE_KEYS},
        })
    return rows, route


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("instruction_csv", type=Path)
    parser.add_argument("--route", choices=("all_simd", "all_simt", "mixed"), required=True)
    parser.add_argument("--stage-pc-map", type=Path)
    parser.add_argument("--code-correlation-csv", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--load-bias", type=parse_int)
    parser.add_argument("--addr2line", type=Path, default=Path("llvm-addr2line"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    model = find_stage_model(report)
    predicted, route = predicted_stage_rows(model, args.route)
    ranges = load_pc_map(args.stage_pc_map)
    aggregate, per_stage, unmapped = read_camodel_csv(args.instruction_csv, ranges)
    source_observations = {}
    ambiguous_source_lines = {}
    unmatched_source_lines = []
    if args.code_correlation_csv:
        source_observations, ambiguous_source_lines, unmatched_source_lines = read_code_correlation(
            args.code_correlation_csv, predicted)
    instruction_source_observations = {}
    ambiguous_instruction_sources = {}
    unmatched_instruction_sources = []
    if args.binary:
        if args.load_bias is None:
            parser.error("--binary requires --load-bias")
        (instruction_source_observations, ambiguous_instruction_sources,
         unmatched_instruction_sources) = read_instruction_source_attribution(args.instruction_csv, predicted,
                                                                              args.binary, args.load_bias,
                                                                              args.addr2line)

    for row in predicted:
        if ranges:
            row["camodel_observation"] = {
                "status": "observed" if row["id"] in per_stage else "no_matching_pc",
                "resource_families": per_stage.get(row["id"], {}),
            }
        else:
            row["camodel_observation"] = {
                "status": "unobservable_without_stage_pc_map",
                "resource_families": {},
            }
        if args.code_correlation_csv:
            row["camodel_source_observation"] = {
                "status": "observed" if row["id"] in source_observations else "not_uniquely_attributed",
                "totals": source_observations.get(row["id"], {}),
            }
        if args.binary:
            row["camodel_instruction_source_observation"] = {
                "status": "observed" if row["id"] in instruction_source_observations else "not_uniquely_attributed",
                "resource_families": instruction_source_observations.get(row["id"], {}),
            }

    output = {
        "schema_version":
        1,
        "unit_contract": {
            "predicted": "system_cycle_selection_score",
            "camodel_cycles": "simulator_cycle_aggregate_from_instruction_csv",
            "camodel_running_time": "microseconds_reported_by_camodel",
            "direct_numeric_comparison_valid": False,
        },
        "route":
        args.route,
        "route_factor":
        route.get("route_superblock_factor", 1),
        "stage_assignment":
        "pc_range" if ranges else "unobservable",
        "stage_pc_map":
        str(args.stage_pc_map) if args.stage_pc_map else None,
        "code_correlation_csv":
        str(args.code_correlation_csv) if args.code_correlation_csv else None,
        "ambiguous_source_lines":
        ambiguous_source_lines,
        "unmatched_source_lines":
        unmatched_source_lines,
        "binary":
        str(args.binary) if args.binary else None,
        "load_bias":
        args.load_bias,
        "ambiguous_instruction_sources":
        ambiguous_instruction_sources,
        "unmatched_instruction_sources":
        unmatched_instruction_sources,
        "unmapped_instruction_pcs":
        unmapped,
        "camodel_kernel_resource_families":
        aggregate,
        "stages":
        predicted,
        "next_required_evidence": ([] if ranges else [
            "compiler-emitted non-overlapping Stage PC ranges, or uniquely attributable debug-line correlation, for the same binary",
            "SYS_CNT conversion metadata before comparing absolute cycle values",
        ]),
    }
    rendered = json.dumps(output, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
