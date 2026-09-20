# Ascend 950DT profile bootstrap

These are independent **draft templates**, not runnable/calibrated profiles:

| File | Purpose |
|---|---|
| `../configs/ascend_950dt.json` | Hardware topology, capacities and absolute-model inputs |
| `microbench/ascend_950dt_v1.json` | Target-specific measurements and architecture facts |
| `simd_simt/ascend_950dt_simd_simt_v1.json` | SIMD/SIMT Stage model parameters |

The field structure follows the PR files, but PR numeric rates, frequencies,
core counts and fitted penalties were **not** imported. `null` means unknown,
not zero, a disabled cost, or an instruction to fall back to PR.

The files parse as JSON, but the Stage selection profile intentionally does not
yet satisfy every production numeric requirement. Do not use it for route
selection until the remaining entries are measured and schema/loader tests
pass. Backend target selection and loader target matching have been
implemented locally: \`Ascend950DT*\` resolves the DT profile and
\`Ascend950PR_*\` resolves the PR profile. Those changes still require the
existing build-tree tests before they are accepted.

## Model-specific configuration verified on 2026-09-12

The physical TD machine's installed `Ascend950DT_9582.ini` confirms 32 AIC,
64 AIV, nominal HBM 96 GiB, L2 128 MiB, L1 512 KiB, L0A/B 64 KiB each,
L0C 256 KiB and platform-usable UB 248 KiB. These values now populate the
DT config. Configured compute frequency is 1650 MHz; the vector register
width is 256 bytes. The two corresponding shared-profile entries are
configuration facts, not fresh throughput measurements. Independent SYS_CNT
measurement gives 1000 MHz (bounded by 999.923515 and 1000.122079 MHz).
Runtime allocatable HBM is distinct from nominal capacity.

The architecture-only withdrawal below describes the preceding state;
exact-model configuration evidence supersedes it for these fields only.

## Earlier architecture-only reference

Local `asc_devkit/docs/zh/asc_950_feature_guide.md` associates PR/DT with NPU
architecture 3510. Its architecture specification
`docs/zh/guide/编程指南/高级编程/硬件实现/架构规格/NPU架构版本3510.md`
documents the following nominal capacities:

| Resource | Initial value | Meaning |
|---|---:|---|
| SIMD vector width | 256 bytes / 2048 bits | Width, not measured operation throughput |
| UB | 256 KiB | Nominal capacity; runtime usable space/reservations must be queried |
| L1 | 512 KiB | Per-core buffer, not device L2 |
| L0A / L0B | 64 KiB each | Matrix input buffers |
| L0C | 256 KiB | Matrix accumulator buffer |

These were initially architecture-only references. The exact-model INI and
direct ACL queries have now superseded them. In particular, ACL reports
216 KiB runtime UB, while the platform INI reports 248 KiB compiler-usable UB;
these are different interfaces and must not be presented as the same capacity.

## Measured DT entries

| Entry | Value | Evidence / boundary |
|---|---:|---|
| SYS_CNT | 1000 MHz | `microbench/data_provider/950dt_syscnt_20260912.json` |
| SIMD FP32 ADD | 3.30 vector instructions/tick | Correctness-checked ILP sweep; not dependent latency |
| SIMT FP32 ADD | 105.6 scalar operations/tick | 32 warps, ILP8; not per-thread throughput |
| Minimal VF enqueue | 123.625 ticks/call | 32 warps, steady-state slope, no per-call barrier |
| Minimal VF serialized | 235.6075 ticks/call | Mean of two fresh 32-warp 200→400 slopes, including PIPE_ALL |
| Minimal SIMD Stage proxy | 44.8475 ticks/call | Mean of two fresh one-vdup/one-vsts 200→400 slopes, including PIPE_ALL; not an isolated instruction latency |
| SIMT shuffle | 0.815417 warp instructions/tick | 32 warps, ILP4, effective loop rate; assembly review pending |
| SIMT UB load / store | 125.753856 / 103.085475 bytes/tick | Composite loop rate, including arithmetic/control; not peak bandwidth |
| SIMT GM load / store | 31.690013 / 75.377274 bytes/tick | Active footprint 32 to 128 MiB; not cold-HBM bandwidth |
| SIMD operation sweep | add 3.300160, mul 1.650165, div 0.471365 vector instructions/tick | Compile-time-specialized binaries; optimized-IR instruction-count audit passed |
| SIMT operation sweep | add 105.597458, sub 105.606392, mul 105.603584, div 52.803430 scalar operations/tick | Final compile-time-specialized binaries with runtime-dependent seeds; optimized-IR audit passed |
| SIMT max / abs | 39.293961 / 79.979156 semantic operations/tick | Four independent cyclic value pairs prevent idempotent-chain folding |
| SIMD exp / log | 0.470588 / 0.413439 semantic operations/tick | Two physical runs; one finite-range source operation per chain |
| SIMT exp / log | 64.025010 / 19.225538 semantic operations/tick | Two physical runs through public AscendC SIMT API; one finite-range operation per chain |
| SIMD / SIMT predicate compare | factors 0.999974 / 1.523398 relative to ADD | Derived from repeated, audited compare+select minus select-only probes |
| SIMT select / cast | 90.712979 / 87.491544 scalar operations/tick | Optimized-IR audit passed; used only as relative factors against the same specialized add binary |
| SIMT clamp primitives | 78.589877 lower/upper-bound primitives/tick | One semantic clamp is the pair, so its factor counts two primitives |
| SIMD continuous GM→UB | `105.984801 + bytes / 167.608226` ticks | Joint fit of two independent 4/32/128 KiB sweeps; startup is per logical load |
| SIMD continuous UB→GM | `91.298767 + bytes / 213.142804` ticks | Joint fit of two independent 4/32/128 KiB sweeps; startup is per logical store |
| SIMD dependent scalar-fadd pair delta | 2.424625 ticks/additional fadd | Two fresh pair runs; optimized LLVM IR proves one versus two surviving fadds |
| SIMT loop + one fadd | 13.940594 to 38.7855 ticks/iteration | 1 to 32 warp sweep; not an isolated backedge latency and not linear in warp count |
| SIMD prefix scan, one vector group | `46.067690 + (axis_extent - 1) * 3.030274` ticks | Two complete 4/8/16/32-row sweeps; 8/16/32 independent columns give the same cost while they remain in one vector group |
| SIMT prefix scan | 278.6535 ticks at 4x8; 1447.76 ticks at 32x32 | Two complete correctness-checked sweeps; the large shape dependence disproves one target-wide scan dependency factor |

ADD raw results are in `microbench/data_provider/950dt_add_checked_20260912.json`.
Setup raw results and all six warp configurations are in
`microbench/data_provider/950dt_setup_20260912.json` (336 correctness-checked samples).
The minimal VF writes one float to keep its work observable. These setup
measurements do not establish the Mixed local-scope handoff cost. No PR residual
has been imported to fill the remaining unknowns.

The shared profile has all 25 original entries populated (configuration facts
and qualified workload measurements combined). Dependent ADD and shuffle
latencies are 1.81821 and 10.2846 SYS_CNT ticks respectively, each from two
independent physical runs; all 252 output checks passed.
Every dependent run checks chain lengths 4/16/64 and loop counts 128/256/512.
Optimized LLVM IR verifies all six intended chains with
`check_dependent_latency_ir.py`; its five audit tests pass. See
`microbench/data_provider/dependent_latency_README.md`.

The wider operation-rate sweep is fail-closed. Each operation is compiled into
a separate binary, and `check_operation_rates_ir.py` compares its optimized
LLVM IR with an identity baseline before the binary may contribute a factor.
SIMD max/abs/clamp pass because the vendor vector intrinsics remain explicit.
SIMT max/abs/clamp use cyclic runtime-dependent value pairs, and their final IR
also passes. Exp/log are run with one finite-range operation per chain, not a
hundreds-of-iterations recurrence that quickly reaches exceptional values.
SIMT exp/log use the public `AscendC::Simt` API. The final Exp IR
contains eight `llvm.exp` calls; Log has eight semantic calls represented by
sixteen static `llvm.log` occurrences because its normal/subnormal paths are
guarded. Exactly one Log path executes for each semantic call. Evidence:
`950dt_operation_rates_checked_20260913.csv`,
`950dt_operation_rates_checked_ir_audit_20260913.log`, and
`950dt_operation_rates_simt_final_20260913.csv`, plus the repeated
`950dt_operation_rates_final_missing*_20260913` CSV and IR-audit files.

The control-flow pair experiment deliberately does **not** populate
`loop_backedge_system_cycles`. With one dependent fadd per lane, the measured
per-iteration cost is 13.940594/13.939875/15.152219/17.575719/29.090969/38.7855
ticks at 1/2/4/8/16/32 warps. Adding a second dependent fadd changes those
values by 0.604937/1.210875/1.211719/0.606781/-0.001031/-0.000531 ticks.
Optimized LLVM IR retains both fadds. Therefore a single constant backedge
term or `N × scalar_latency` cannot represent the interaction between the
per-warp dependency path and shared issue bandwidth. Evidence:
`950dt_control_rates_pair_20260913.csv`,
`950dt_control_rates_pair_ir_audit_20260913.log`, and
`950dt_control_warp_sweep_20260913.csv`. This remains a formula gap; no fitted
number is promoted to the selection profile.

CaModel attempt (2026-09-13): msopprof accepts `Ascend950DT_9582`, but the child
fails at rtSetDevice with 507033 (driver device-open failure), before any kernel
executes. Repeating with visible device 0 also fails. Outer msopprof exits zero
despite the child failure and empty profiling results. Neither attempt is valid
calibration evidence. Do not infer simulator success from outer exit code alone.
This count does not imply the hardware config or Stage profile is complete.
Memory instruction rates are byte rates divided by 32 lanes * 4 bytes and
are not additional independent measurements. GM/UB/shuffle raw samples are in
`microbench/data_provider/950dt_{gm,ub,shuffle}_checked_20260912.json`.

The six transition-harness entries come from
`microbench/data_provider/950dt_setup_net_20260912.json`: serialized minimal VF
minus barrier-only slopes in the same binary, with two complete 504-sample runs.
They are approximately 209 ticks for 1-16 warps and 244.095 ticks at 32 warps.
They are **not Mixed transition constants**. Adding the baseline mode changed
the 32-warp serialized slope from 230.205 to 248.945 ticks; repeating the same
new binary gives 248.960. This cross-binary sensitivity still needs explanation.
Do not replace the Stage transition model with these values.

A fresh DT run of \`transition.cce\` also disproves additive extraction of a
directional transition constant. Subtracting separately measured SIMD and SIMT
segments from the combined schedule gives negative residuals: approximately
-23 to -66 cycles for SIMT-to-SIMD and -117 to -147 cycles for SIMD-to-SIMT
without a middle barrier. The segments overlap, so this arithmetic is invalid;
the old 668-cycle PR residual must not be copied into DT. Raw output is
\`microbench/data_provider/950dt_transition_harness_20260913.log\`. A true
directional handoff probe must first hold Stage work constant and vary only the
boundary and live tensor bytes.

The stricter grouped-versus-alternating probe holds the number of SIMD writes,
SIMT writes and `PIPE_ALL` barriers constant. Across 36 combinations (4/8/16
warps, 16/64 elements, 128/512/2048 operations, shared and separate UB
buffers), two runs report an extremely stable **negative** residual of
-38.280 to -38.114 SYS_CNT ticks per additional alternation pair. Alternation
is faster because the generated grouped and alternating loops have different
issue schedules; this is proof that a positive fixed directional latency
cannot be obtained by subtracting independently scheduled loops. It is not a
negative scope-handoff delta and is therefore not written into the selection
profile. Raw evidence:
`microbench/data_provider/950dt_transition_pair_{1,2}_20260913.csv`.

The prefix-scan sweep also found a formula issue rather than merely a missing
constant. For SIMD, a scan of axis extent `R` and `S` independent sequences
fits `46.067690 + (R-1)*ceil(S/64)*3.030274` ticks for the measured
`S<=32` range. For SIMT, the lower bound has two independent dimensions:

```text
C_scan_SIMT = C_setup(active_warps)
              + depth * max(C_shuffle_dependency,
                            shuffle_warp_instructions / shuffle_issue_rate)
              + C_active_add + C_store + C_control
```

Here `depth=ceil(log2(R))`; `active_warps` and the number of independent scan
sequences cannot be reconstructed from the current scalar
`shuffleLaneSteps`. `StageWorkload` now records scan extent and independent
sequence count explicitly; `StageCostEvaluator` still needs to consume those
dimensions with the formula above. Consequently
`prefix_scan.dependency_factor` remains `null` instead of preserving the old
one-factor shortcut. The raw 24-point runs are
`950dt_scan_rates_{1,2}_20260913.csv`; all values agree within 0.03% and all
outputs passed.

To reproduce the probes in the already initialized TD environment:

```bash
cd /home/kaixin2/td_syscnt_20260912_XW7Fcp
export INC=/home/kaixin2/AscendNPU-IR/bishengir/lib/Template/include
# Copy the checked-in probe sources and build_and_run.sh here first.
bash build_and_run.sh setup_probe simt_shuffle simt_memory simt_gm_memory
PROBE_DEVICE=3 bash run_operation_rates.sh operation_rates_checked.csv
```

The build script preserves the activated Conda/CANN environment, selects host
headers by architecture and uses dav-c310-vec for these probes. Device selection
must be explicit before running; these measurements used physical device 3.

## Completion order

1. On TD, use `/home/kaixin2` and Conda `kaixin2`. Record source revisions,
   installed compiler, exact target, device health and active device.
2. Query topology/capacities and independently measure SYS_CNT frequency.
   Keep device-compute cycles distinct from SYS_CNT ticks.
3. Fill shared microbench entries from DT sweeps, retaining inputs, repetition
   counts, units, scope, binary identity and raw results. Null entries are not
   evidence. Verify generated instructions and launch bounds.
4. Calibrate Stage formulas with those measurements. Do not inherit PR residual
   penalties or assign unexplained kernel error to scope-handoff costs.
5. Validate all three production schemas and measurement references. Implement
   explicit DT profile selection/version support with tests. In particular,
   review the PR profile's existing broad `Ascend950*` compatibility pattern;
   it must not silently select PR data for DT.
6. Compare SIMD / SIMT F1/F2/F4 and materializable Mixed implementations, then
   run the full guard suite. Keep DT results separate from PR results.

The checked device-query validator is
`microbench/data_provider/validate_950dt_physical_config.py`. It verifies the
product identity, ACL architecture number, 32/64 core topology, runtime HBM,
L2, runtime-reported UB, and the independent SYS_CNT clock domain. The archived
device evidence contains only the installed INI path and hash, not its body, so
the nominal HBM/L0/L1/UB values and configured 1.65 GHz clock cannot yet be
replayed from repository evidence. Recollect the full INI before declaring the
physical profile complete.

Status: model-specific configuration partially filled; DT measurement, runtime loading, target selection and
regression acceptance are **not complete**. After removing dead selection
multiplier fields and replacing the old five-field directional transition
proxy with three scope-handoff parameters, the Stage route profile still has
**13 explicit `null` fields**
and must remain non-runnable. Run
`microbench/data_provider/audit_active_simd_simt_profiles.py` to guard this
state and all active measurement references.

The 13 remaining fields are deliberately explicit rather than filled from
950PR or inferred from an end-to-end residual:

- model-domain boundary: `selection_calibration.tiny_dot_flops_max`;
- SIMD: dot startup/rate, generic issue rate, and generic prefix-scan step;
- SIMT: dot startup/rate, generic issue rate, generic prefix-scan step, and
  `superblock.useful_factor_limit`;
- Mixed scope: fixed envelope, input handoff rate, and output handoff rate.

`950dt_dot_scaling_20260913.json` is not sufficient to fill the four dot
fields: its 1/2/4-dot deltas also contain output store, synchronization and
scheduling changes, and the evidence itself records
`absolute_stage_calibration=false`. Likewise, the measured concrete scan
points cannot be collapsed into one target-wide step latency. These fields
stay `null` until an identifiable probe supplies the exact numerator and
execution scope required by the production formula.
