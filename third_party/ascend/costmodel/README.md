# Ascend Cost Model architecture

This directory contains a TTIR performance model, a SIMD/SIMT route model,
and one shared evidence layer.

```text
profiles/microbench
        |
        v
AscendModelProfile
   |             |
   v             v
AscendModelAnalysis      AscendModelRouteModel
(TTIR performance)       (SIMD/SIMT routing)
          \               /
           v             v
           AscendModelTransforms / backend integration
```

The two models share measurements, not objectives or scoring formulas:

- `configs/`, `include/AscendModel/Analysis`, `lib/AscendModel/Analysis`,
  `IR`, and the original transforms implement the TTIR performance model used
  by the in-process autotune path.
- `profiles/microbench`, `include/AscendModel/Profile`, and
  `lib/AscendModel/Profile` own model-neutral measurements plus their loader,
  units, clock domains, target checks, and provenance.
- `profiles/simd_simt`, `include/AscendModel/RouteModel`, and
  `lib/AscendModel/RouteModel` own SIMD/SIMT feature extraction, Coverage,
  conditional candidate scoring, post-score checks, selection reporting, and
  scope materialization.

`third_party/ascend/backend/compiler.py` is an integration layer: it schedules
the native passes and locates installed assets. `python/setup.py` copies
`profiles/` to `triton/backends/ascend/costmodel_profiles/` while building a
Python package. That installed directory is generated runtime data, not a
source-of-truth Cost Model directory.

## Coverage boundary

Coverage is checked before candidate scoring.  An unknown `scf.for` trip count
is rejected by default; the only bounded exception is a recognized
`triangular_solve_loop` anchor group.  That exception still requires the
small-tensor limit, one to four materializable anchors, and the mask/reduction
limits from the SIMD/SIMT profile.  Removing the triangular mechanism evidence
therefore returns `unknown_loop_trip_count` rather than silently admitting a
generic dynamic loop.  The regression tests for both paths are in
`unittest/costmodel_ut/SimdSimtCostModelTest.cpp`.

## Generic SIMT Stage spill model

The route model does not recognize a particular operator to predict register
spill.  `StageWorkloadAnalysis` computes the peak SSA live set of each owned
Stage and records it as `peak_live_register_units_per_logical_program`, where
one unit is one 32-bit register word.  A value is live from its definition (or
first Stage use for a live-in) through its last use; an escaping value remains
live to the Stage boundary.

For a SIMT candidate the profile then applies the generic rule

```text
threads = logical_warp_group_count * simt.issue_width
excess_per_program = max(0, ceil(peak_units / threads)
                            - simt.register_budget_per_thread)
spill_transactions = excess_per_program * threads
                     * simt.spill_transaction_coverage
                     * superblock_factor
spill_cycles = spill_transactions / simt.spill_transactions_per_system_cycle
```

The resulting `register_pressure_per_iteration` is emitted separately from
explicit lowering-provided `spill_per_iteration`, so a report shows whether a
cost came from static SSA pressure or an already observed `LDK/STK` stream.
`register_budget_per_thread` and `spill_transaction_coverage` are optional
stage-resource profile fields with conservative defaults of 32 and 1.0;
target-specific values must be backed by allocator/CaModel evidence.

The 10,240 B P4 value used in earlier notes is a grouped Stage traffic value;
it must not be treated as the peak live set of one logical program.  The
online model only accepts the SSA-derived
`peak_live_register_units_per_logical_program`; the F2 loop grouping then
multiplies per-program spill work, rather than multiplying the per-thread
register footprint by two.

## TTIR memory traffic and generated vector-pipe traffic

The memory part of a SIMT Stage is modeled on the two physical vector
resources, not on four independent instruction classes:

```text
RVECLD = LDG + LDS + LDK
RVECST = STG + STS + STK
```

For a SIMT Stage the corresponding cycle terms are accumulated as

```text
C_RVECLD = W_LDG / rho_LDG + W_LDS / rho_LDS + W_LDK / rho_LDK
C_RVECST = W_STG / rho_STG + W_STS / rho_STS + W_STK / rho_STK
```

where `W_*` are per-iteration instruction counts and `rho_*` are measured
instructions per system cycle.  The current profile has one aggregate
`rho_LDG`/`rho_STG` fallback; separate LDS/STS/LDK/STK rates are optional and
must come from isolated evidence.  `C_RVECLD` and `C_RVECST` are then the
single `load` and `store` resource terms in the Stage envelope, not four
serial additions on top of one another.

`tt.load`/`tt.gather` and `tt.store`/atomics provide semantic global traffic
(`LDG`/`STG`) directly from TTIR.  LDS/STS staging and LDK/STK private stack
operations are introduced by later layout conversion, scope lowering, and
register allocation.  They cannot be reconstructed exactly from unannotated
TTIR because allocator decisions, live ranges, and generated address/layout
instructions are not present there.

When post-layout TTIR still contains `ttg.local_load`/`ttg.local_store` (or the
equivalent `hivm.hir.local_load`/`hivm.hir.local_store` extension), the analysis
counts those operations directly as LDS/STS instruction work.  This is the
reliable IR-visible case; `ttg.convert_layout` by itself is not treated as a
memory operation because some conversions stay in registers.

When a lowering pass has that information, it attaches non-negative numeric
counts to the operation owning the generated stream.  The accepted attribute
names are:

```text
ascend.simt.shared_local_load_instructions
ascend.simt.shared_local_store_instructions
ascend.simt.private_stack_load_instructions
ascend.simt.private_stack_store_instructions
```

If a target has separate microbenchmarks, its SIMD/SIMT profile may provide
optional rates under `stage_resources.generated_vector_traffic`:
`shared_local_*_instructions_per_system_cycle` and
`private_stack_*_instructions_per_system_cycle`.  A missing or zero rate
falls back to the aggregate `load_warp_instructions_per_system_cycle` or
`store_warp_instructions_per_system_cycle`; this keeps older profiles valid
without pretending that a stack rate has been measured.

The `*_warp_instructions` and `*_stack_*` spellings are accepted as aliases
for compatibility with an in-flight lowering.  Counts are per Stage
iteration.  For SIMT, the evaluator folds them into the RVECLD/RVECST load and
store envelope and emits their separate cycle contributions in `route.json`;
the same JSON also exposes the direct `global_load`/`global_store` portions.
For SIMD they are ignored because they are SIMT-lowering facts, not semantic
TTIR bytes.  Generated traffic also enters the SIMT issue lower bound once,
so it is not hidden by the load/store terms.  Until the lowering emits these
attributes, the model reports zero generated traffic and retains the static
SSA-pressure estimate as a lower-bound diagnostic; it does not infer exact
LDK/STK counts from `SIMT_WARP_STACK_SIZE` or a stack byte size.
When exact private-stack counts are present, the static peak-live stack proxy
is suppressed to avoid charging the allocator traffic twice.

### TTIR source-to-generated-traffic fit (implemented)

`StageWorkloadAnalysis` now records `simt_memory_access_facts` for every
`tt.load`/`tt.gather` and `tt.store`/atomic operation: operation count, row
segments, contiguous row bytes, stride bytes, masked count, and whether a
static stride vector was available.  A representative pattern is retained
only when all accesses in the Stage agree; mixed or unknown layouts do not
match a layout curve.

The profile loader consumes two optional kinds of offline evidence:

* `stage_resources.layout_memory.load/store` contains exact canonical layout
  curves (`first_system_cycles + (n-1)*incremental_system_cycles`) keyed by
  rank, row bytes, stride, rows, element width, mask, shape, strides, active
  warps, and SuperBlock factor.  The evaluator uses an exact key match and
  otherwise keeps the aggregate warp-rate fallback.  The same section may be
  present in the SIMD profile with `vector2d:` keys (rank/row/stride/rows,
  dtype, mask, shape, and strides only).  SIMD has no SuperBlock warp factor;
  its curve is the measured 2-D MTE schedule and route wave scaling is applied
  once after the per-program envelope.  For example, the 950PR profile uses
  the `simd_memory_2d` drain=2 fit for the 16x16 FP32, 64-byte-row,
    8192-byte-stride and 16384-byte-stride solve_tril tiles, so a flat
    `bytes / rate` estimate is not used for S6/S22 when either measured key is
    available.  The 16384-byte curve comes from the 2026-09-20 A5 probe and
    covers the H64 shape.
* `stage_resources.*_source_fit` contains non-negative coefficients for the
  generic TTIR-to-work fit:

  ```text
  W_fit = intercept + a_direct W_direct + a_segments S
          + a_stride N max(stride/contiguous - 1, 0)
          + a_masked M + a_peak_live P
  ```

  `W_direct` is the existing semantic warp-instruction count for GM traffic,
  or the explicit LDS/STS/LDK/STK count for generated traffic.  Fits are
  disabled unless the profile sets `enabled=true`; therefore profiles without
  offline lowering evidence retain the previous conservative formula.  The
  resulting `global_*`, `shared_local_*`, and `private_stack_*` contributions
  are emitted separately in `route.json` and folded into the single
RVECLD/RVECST resource envelope exactly once.
`route.json` now labels the selected decomposition with `load_model` and
`store_model` (`aggregate_warp_rate`, `source_fit`, or `layout_curve`).

This is the online boundary: assembly/CaModel/trace are used offline to fit
the profile coefficients and canonical curves; the route model itself consumes
only TTIR facts plus the versioned profile.  No dump file or simulator is
required while compiling a kernel.  `fit_simt_source_memory.py` writes a fit
artifact, and `apply_simt_source_fit.py` is the explicit promotion step that
copies its coefficients into a new profile while retaining evidence hashes and
held-out error metadata; an unpromoted fit cannot affect route selection.
