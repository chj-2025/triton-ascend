# CAModel data source workflow

This directory documents how to generate CAModel simulator data and how to
promote the parsed result into the microbenchmark profile data source used by
`../../ascend_davidv100_v1.json`.

## Directory layout

```text
data_provider/
  *.cce / *_host.cpp                 # runnable microbenchmark sources
  build_and_run.sh                   # build + on-board run helper
  build_commands.md                  # normal build/run commands
  camodel/
    README.md                        # this workflow
    camodel_experiment_matrix.json   # planned CAModel experiment coverage
    extract_camodel_system_cycle_profile.py
    compare_stage_costs.py           # Stage prediction / CaModel evidence join
```

## 1. Build the probe

Run on the Ascend board from `data_provider/`:

```bash
cd /data/kaixin/triton-ascend/third_party/ascend/costmodel/profiles/microbench/data_provider
bash -lc './build_and_run.sh simt_memory'
```

`build_and_run.sh` expands to:

```bash
ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 \
  -I"$INC" "$name.cce" -o "$name.o"

g++ -O2 "${name}_host.cpp" -o "${name}_host" \
  -I"$ASCEND_TOOLKIT_HOME/x86_64-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/include" \
  -L"$ASCEND_TOOLKIT_HOME/lib64" \
  -lruntime -lascendcl
```

If the template headers are not in the default path:

```bash
INC=/data/kaixin/AscendNPU-IR/bishengir/lib/Template/include \
bash -lc './build_and_run.sh simt_memory'
```

## 2. Generate CAModel simulator output

After the host binary is built, run CAModel through `msopprof simulator`:

```bash
msopprof simulator --soc-version=Ascend950PR ./simt_memory_host
```

If the local CANN package uses davinci naming, use:

```bash
msopprof simulator --soc-version=dav-c310 ./simt_memory_host
```

The simulator creates an output directory similar to:

```text
OPPROF_YYYYMMDDHHMMSS_xxx/
  device0/
    ...
```

Keep the raw `OPPROF_*` directory as the primary evidence artifact.  The
profile JSON should not point only to an opaque number; it should point to the
source probe, host launcher, raw CAModel output, parser, and derived result.

## 3. Parse CAModel counts into SYS_CNT-domain rates

Run the following commands from `data_provider/`. The first parser accepts
either the `OPPROF_*` root directory or its `dump/` subdirectory. A supported
input contains primary SIMT issue/receive dumps such as:

```text
OPPROF_YYYYMMDDHHMMSS_xxx/
  dump/
    core0.veccore0.rvec.simt.lsu.dump
    core0.veccore0.rvec.simt.exu0.dump
    core0.veccore0.rvec.simt.dvg0.dump
    ...
```

First normalize the raw CAModel dumps into per-unit instruction counts:

```bash
cd third_party/ascend/costmodel/profiles/microbench/data_provider

python3 camodel/parse_camodel_counts.py \
  /path/to/OPPROF_YYYYMMDDHHMMSS_xxx \
  -o parsed_camodel_counts.json
```

Passing the `dump/` directory directly is equivalent:

```bash
python3 camodel/parse_camodel_counts.py \
  /path/to/OPPROF_YYYYMMDDHHMMSS_xxx/dump \
  -o parsed_camodel_counts.json
```

Verify that at least one active unit was parsed before converting the rates:

```bash
python3 -c \
  'import json; d=json.load(open("parsed_camodel_counts.json")); print(d["per_unit"].keys())'
```

The command should print entries such as `core0.veccore0`. An empty mapping
means that the selected directory does not contain a supported primary SIMT
dump; do not continue with an empty file.

The generated JSON count file has this shape:

```json
{
  "per_unit": {
    "aiv0": {
      "span": {"delta": 1000},
      "group_counts": {
        "memory": 100,
        "float_alu": 200,
        "shuffle": 0,
        "predicate": 10,
        "control": 20,
        "int_alu": 30
      }
    }
  }
}
```

Then convert the normalized counts from simulator cycles into SYS_CNT-domain
effective rates and save the result:

```bash
python3 camodel/extract_camodel_system_cycle_profile.py parsed_camodel_counts.json \
  --simulator-clock-mhz 1650.0 \
  --sys-cnt-mhz 988.9 \
  --scope simt_memory \
  > parsed_simt_memory.json
```

The complete data flow is:

```text
OPPROF_* raw dumps
  -> parse_camodel_counts.py
  -> parsed_camodel_counts.json
  -> extract_camodel_system_cycle_profile.py
  -> parsed_simt_memory.json
```

The parser emits JSON like:

```json
{
  "unit": "system_cycles",
  "scope": "simt_memory",
  "rates": {
    "memory": {
      "warp_instructions_per_system_cycle": 0.5
    }
  },
  "confidence": "low"
}
```

Important: these are workload-effective CAModel rates.  They are useful as
calibration evidence, but they are not bare hardware peak throughput or
isolated instruction latency.

## 4. Promote the result into the microbenchmark profile

When a CAModel-derived number is used by `ascend_davidv100_v1.json`, record the
complete provenance in the measurement `source` field.

Recommended source format:

```json
"source": "data_provider/simt_memory.cce; data_provider/simt_memory_host.cpp; data_provider/camodel/OPPROF_xxx; data_provider/camodel/parsed_simt_memory.json; data_provider/camodel/extract_camodel_system_cycle_profile.py; data_provider/build_commands.md"
```

The measurement should also state:

- `source_kind`: `camodel_simulator` or `isolated_microbenchmark_with_camodel`
- `cycle_domain`: normally `SYS_CNT` after parser conversion
- `confidence`: usually `low` or `medium`, unless independently validated
- `description`: whether the value is workload-effective or isolated

## 5. Validation checklist

Before treating a CAModel result as a data source:

1. Source `.cce` and `_host.cpp` are checked into `data_provider/`.
2. Exact build command is covered by `build_commands.md`.
3. Raw CAModel `OPPROF_*` artifact or a documented extracted subset is saved.
4. Parser command and output JSON are saved under `data_provider/camodel/`.
5. `ascend_davidv100_v1.json` `source` points to all relevant artifacts.
6. The profile description does not claim hardware peak if the measurement is
   only workload-effective.

## 6. Compare a selected route with Stage-level evidence

`compare_stage_costs.py` joins the selected Stage implementations in a Route
Model report with an instruction-summary CSV from the same CaModel binary:

```bash
python3 camodel/compare_stage_costs.py route_report.json \
  OPPROF_xxx/device0/core0.veccore1_instr_exe_xxx.csv \
  --route mixed --output stage_comparison.json
```

Without `--stage-pc-map`, the output deliberately keeps every Stage's CaModel
observation as `unobservable_without_stage_pc_map`; only the whole-kernel
resource-family totals are reported.  This prevents whole-kernel residuals
from being presented as measured Stage cycles.

For actual Stage attribution, the PC map must come from the compiler build of
the same binary and contain non-overlapping half-open ranges:

```json
{
  "stages": [
    {"id": "head_index_mask", "pc_begin": "0x1000", "pc_end": "0x1100"},
    {"id": "diagonal_recurrence", "pc_begin": "0x1100", "pc_end": "0x1800"}
  ]
}
```

Then pass `--stage-pc-map stage_pc_map.json`.  CaModel simulator cycles remain
in their original clock domain; absolute comparison with Route Model system
cycles is invalid until the output also records the measured simulator/SYS_CNT
clock conversion.

When the binary was built with `TRITON_DISABLE_LINE_INFO=false`, the simulator
also emits `*_code_exe.csv`. Pass it with
`--code-correlation-csv path/to/core0.veccore0_code_exe.csv`; the tool derives
source-line ownership from each Stage's `source_locations` report field. A
line shared by multiple Stages is reported in `ambiguous_source_lines` and is
not divided between them. This replaces the old workload-specific scripts
that hard-coded solve_tril line-number ranges.

For finer evidence, the instruction CSV can be correlated directly with the
same debug binary. The load bias must be taken from that CaModel launch (it is
the runtime address corresponding to binary text address zero), not guessed
from a different run:

```bash
python3 camodel/compare_stage_costs.py route_report.json \
  core0.veccore0_instr_exe.csv --route all_simt \
  --binary OPPROF_xxx/dump/aicore_binary.o \
  --load-bias 0x10d0d000 \
  --addr2line /path/to/llvm-addr2line \
  --output stage_instruction_comparison.json
```

The tool follows inline frames back to the Triton Python line, classifies the
instruction resource family, and attributes it only when that line and family
have exactly one predicted Stage owner. Ambiguous and unmatched evidence is
reported explicitly. Debug-line attribution is supplementary: lowering may
move a load to the line of a later dot/store, so compiler-emitted Stage PC
ranges remain the required evidence for complete Stage calibration.
