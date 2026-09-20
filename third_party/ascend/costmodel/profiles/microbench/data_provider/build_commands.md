# Microbenchmark data provider

This directory contains the source CCE probes and host launchers used as
evidence for the target-specific 950PR and 950DT microbenchmark profiles.

Run on the Ascend board, from this directory:

```bash
cd /data/kaixin/triton-ascend/third_party/ascend/costmodel/profiles/microbench/data_provider
bash -lc './build_and_run.sh tput concur2 meas simt_memory simt_gm_memory simt_shuffle'
```

If the template headers are not under the default path, override `INC`:

```bash
INC=/data/kaixin/AscendNPU-IR/bishengir/lib/Template/include \
bash -lc './build_and_run.sh tput concur2 meas simt_memory simt_gm_memory simt_shuffle'
```

The device build command used by `build_and_run.sh` is:

```bash
ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 \
  -I"$INC" "$name.cce" -o "$name.o"
```

The host build command used by `build_and_run.sh` is:

```bash
g++ -O2 "${name}_host.cpp" -o "${name}_host" \
  -I"$ASCEND_TOOLKIT_HOME/$(uname -m)-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/$(uname -m)-linux/include" \
  -L"$ASCEND_TOOLKIT_HOME/$(uname -m)-linux/lib64" \
  -Wl,-rpath,"$ASCEND_TOOLKIT_HOME/$(uname -m)-linux/lib64" \
  -lruntime -lascendcl
```

## Source mapping

### SIMT internal synchronization probe

`run_control_rates.sh` now implements SIMT kind 2 as `__syncthreads()`.
Earlier versions skipped this kind: those runs are **not** evidence for SIMT
barrier latency. Kind 6 is a cross-warp UB exchange (one store, two thread-block
barriers, one load); it is available only in SIMT mode. Neither is a SIMD/SIMT
mode transition.

```bash
PROBE_MODES=SIMT PROBE_KINDS='0 2 6' PROBE_DEVICE=4 \
  bash run_control_rates.sh sync_rates.csv
```

The driver keeps the raw nine short/long sample pairs and checks every output
element for kinds 0/2/6. Each row is the SYS_CNT difference divided by
`repeats * (long_iterations - short_iterations)`: **ticks per whole VF loop
iteration**, not per thread, warp, or individual barrier. Subtracting kind 0
is a measured empty-loop differential, not proof that all components can be
added independently. Do not add this differential to a dot/recurrence curve
that already contains the same synchronization.

Recheck independent iteration bounds with the saved specialized binary, for
example for kind 2 and eight launched warps:

```bash
cp control_rates_simt_kind2.o control_rates.o
ASCEND_RT_VISIBLE_DEVICES=4 ./control_rates_host 1 2 8 64 256 20
```

The generated LLVM IR should have one `llvm.hivm.sync.workitems` call in the
kind-2 SIMT loop and two in kind 6. Keep that IR, the specialized `.o`, the
CSV, device/environment information and source hashes with the result.

| JSON evidence source | Local files | Related measurements |
|---|---|---|
| `triton_cases/SIMT_Test/tput.cce` | `tput.cce`, `tput_host.cpp` | `simt.f32.add.throughput` |
| `triton_cases/SIMT_Test/tput.cce; concur2.cce` | `tput.cce`, `tput_host.cpp`, `concur2.cce`, `concur2_host.cpp` | `simd.f32.add.throughput`, `simd.f32.add.dependent_latency` |
| `triton_cases/SIMT_Test/meas.cce` | `meas.cce`, `meas_host.cpp` | `simt.setup.empty`, `simt.setup.empty_with_barrier` |
| `triton_cases/SIMT_Test/simt_memory_david_v100_20260725.csv` | `simt_memory.cce`, `simt_memory_host.cpp` | SIMT UB load/store throughput and bandwidth |
| `triton_cases/SIMT_Test/simt_gm_memory_david_v100_20260725.csv` | `simt_gm_memory.cce`, `simt_gm_memory_host.cpp` | SIMT GM load/store throughput and bandwidth |
| `triton_cases/SIMT_Test/simt_shuffle_david_v100_20260725.csv` | `simt_shuffle.cce`, `simt_shuffle_host.cpp` | SIMT shuffle throughput and dependent latency |
| `simt_transition_microbench_tail16_barrier_20260713.txt` | `transition.cce`, `transition_host.cpp`, `run_transition.remote.sh` | SIMT transition harness setup proxies |
| `950dt_setup_with_simd_{1,2}_20260913.csv` | `setup_probe.cce`, `setup_probe_host.cpp` | Serialized minimal SIMD/SIMT Stage proxies and barrier-only control |
| `950dt_simd_memory{,_repeat}_20260913.log` | `simd_memory.cce`, `simd_memory_host.cpp` | SIMD continuous GM↔UB rate plus per-operation startup |
| `950dt_control_rates_pair_20260913.csv`; `950dt_control_warp_sweep_20260913.csv` | `control_rates.cce`, `control_rates_host.cpp` | Scalar/control formula audit; these do not yet define a single loop-backedge parameter |
| `950dt_scan_rates_{1,2}_20260913.csv` | `scan_rates.cce`, `scan_rates_host.cpp` | Correctness-checked SIMD/SIMT prefix-scan sweep over scan extent and independent columns |
| `950dt_transition_pair_{1,2}_20260913.csv` | `transition_pair.cce`, `transition_pair_host.cpp` | Grouped/alternating schedule diagnostic with identical operation and barrier counts; negative residual is evidence against additive transition extraction, not a cost parameter |

`build_and_run.sh` sources `/data/kaixin/set_env.sh` or `/home/kaixin/set_env.sh`,
then builds `<name>.cce -> <name>.o` and `<name>_host.cpp -> <name>_host`.

On the 950DT machine, preserve the already activated `/home/kaixin2`
environment and select the physical device explicitly:

```bash
cd /home/kaixin2/<fresh-measurement-directory>
source /home/kaixin2/set_env.sh
conda activate kaixin2
export ASCEND_RT_VISIBLE_DEVICES=3
export INC=/home/kaixin2/AscendNPU-IR/bishengir/lib/Template/include
bash build_and_run.sh scan_rates
# Physical device 3 is remapped to runtime device 0 by visibility.
./scan_rates_host 0 >950dt_scan_rates_20260913.csv
```

Every accepted scan row validates all `rows × columns` outputs. The timed
region contains only repeated UB-resident scan implementations and their
completion barrier; initialization and GM readback are outside it.

## CAModel / msopprof simulator

The full workflow for generating CAModel data and promoting it into
`../ascend_950pr_v1.json` as a data source is documented in
`camodel/README.md`.

The simulator command follows the Ascend devkit documentation pattern:

```bash
msopprof simulator --soc-version=Ascend950PR ./${name}_host
```

If the local CANN package uses the davinci-style SOC name, use the matching
target name instead, for example:

```bash
msopprof simulator --soc-version=dav-c310 ./${name}_host
```

The output directory is usually named `OPPROF_*`.  Per-instruction/per-unit
simulator data is under:

```text
OPPROF_*/device0/
```

The current parser for converted CAModel counts is:

```bash
python3 camodel/extract_camodel_system_cycle_profile.py parsed_camodel_counts.json \
  --simulator-clock-mhz 1650.0 \
  --sys-cnt-mhz 988.9 \
  --scope <experiment-name>
```

Files:

| Purpose | File |
|---|---|
| CAModel experiment plan | `camodel/camodel_experiment_matrix.json` |
| CAModel count-to-SYS_CNT parser | `camodel/extract_camodel_system_cycle_profile.py` |

## SIMT dot operand-path diagnostics

`probe_dot_dependency_rate.py` distinguishes a GM operand from a computed
operand which the SIMT lowering stages through shared memory. Use separate
output directories and an empty compilation cache for a new compiler build:

```bash
source /data/kaixin/set_env.sh
export TRITON_DEBUG=0 TRITON_DISABLE_LINE_INFO=1
export ASCEND_RT_VISIBLE_DEVICES=4 TRITON_ALL_BLOCKS_PARALLEL=1
unset NPU_DEVICE_LIMIT
export TRITON_CACHE_DIR=$(mktemp -d /data/kaixin/dot_probe_cache_XXXXXX)
python probe_dot_dependency_rate.py /data/kaixin/dot_rhs_count4 \
  --mode simt_only --size 16 --count 4 --path chain --factor 2 \
  --logical-programs 512 --row-stride 2048 --block-pointers \
  --runtime-bounds --program-column-stride 64 --column-major-programs \
  --distinct-program-inputs --fresh-rhs --dump-ir
```

Replace `--fresh-rhs` with `--fresh-lhs` for a GM left operand, or with
`--computed-rhs` for a runtime-scaled right operand. In the last variant the
repeated input-feedback dots have two computed operands; inspect the dump to
verify shared-memory staging. `--rhs-scale` is a runtime scalar, not a constant
folding workaround in the original operator. `--matrix-bound 13` checks masked
zero padding without changing the 16x16 compile-time tile. Independent programs
can be packed across a strided row; `--column-major-programs` makes program rows
vary first. Distinct per-program inputs check the packed address mapping.

Measure counts 2 and 4 to obtain an incremental slope, then use count 8 only as
a holdout. Convert a kernel slope to a **physical-group** slope with the actual
AutoBlockify group count; do not multiply by F a second time. The first dot may
have a different operand path, so verify the *difference* in lowered dot counts,
layouts, K-tiling and memory paths. `--dump-ir` preserves the actual compiler
command, after-pass IR, binary and metadata; each run checks the complete output.

These are diagnostic data, not automatically accepted production profile
parameters. A linear chain does not reproduce a larger DAG's simultaneous live
tensors, private/stack traffic or operand reuse. A composite dot slope includes
its internal transfers and synchronization: adding the same transfers again as
an independent load Stage double-counts them. Do not fit any unexplained
whole-kernel difference into transition cost.
