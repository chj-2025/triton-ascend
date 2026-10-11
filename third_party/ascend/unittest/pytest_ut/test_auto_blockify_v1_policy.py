# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from triton.backends.ascend.compiler import (
    _append_pure_simt_auto_blockify_options,
    _bisheng_options_with_resource_usage,
    _build_costmodel_analysis_ttir,
    _can_materialize_scope_superblock,
    _default_auto_simt_profile_name,
    _metadata_for_backend_cross_route_rerun,
    _backend_private_stack_load_charge,
    _parse_simt_private_stack_disassembly,
    _preserve_all_simd_auto_blockify_contract,
    _publish_route_transform_capability,
    _publish_backend_resource_usage,
    _resolve_auto_blockify_v1_policy,
    _select_backend_resource_aware_factor,
    _selected_npuir_superblock_factor,
)

pytestmark = pytest.mark.backend("cpu")

SAFE_TTIR = "module { tt.func public @safe() { tt.return } }"
ATOMIC_TTIR = "module { tt.func public @atomic() { %0 = tt.atomic_rmw add } }"
CACHE_MODIFIER_TTIR = (
    'module { tt.func public @cached() { tt.store %ptr, %value {cacheModifier = 1 : i32} : tensor<8x!tt.ptr<f16>> } }')


@pytest.mark.parametrize(
    "arch,expected",
    [
        ("Ascend950DT_9582", "simd_simt/ascend_950dt_simd_simt_v1.json"),
        ("ascend950dt_9589", "simd_simt/ascend_950dt_simd_simt_v1.json"),
        ("Ascend950PR_9579", "simd_simt/ascend_950pr_simd_simt_v1.json"),
        # An architecture alias is not evidence for the PR product profile.
        ("dav-c310", "simd_simt/david_v100_simd_simt_v1.json"),
    ],
)
def test_default_auto_simt_profile_is_product_specific(arch, expected):
    assert _default_auto_simt_profile_name(arch) == expected


@pytest.mark.parametrize(
    "env_enabled,option,expected,source",
    [
        (True, None, True, "TRITON_ALL_BLOCKS_PARALLEL"),
        (False, None, False, "TRITON_ALL_BLOCKS_PARALLEL"),
        (True, False, False, "option"),
        (False, True, True, "option"),
    ],
)
def test_auto_blockify_v1_enable_policy(env_enabled, option, expected, source):
    metadata = {}
    opt = SimpleNamespace(enable_auto_blockify=option)
    with patch("triton.backends.ascend.compiler._is_auto_map_parallel_blocks_enabled", return_value=env_enabled):
        assert _resolve_auto_blockify_v1_policy(SAFE_TTIR, metadata, opt) is expected
    assert metadata["auto_blockify_v1_enabled"] is expected
    assert metadata["auto_blockify_v1_selection_source"] == source


@pytest.mark.parametrize("resolved", [False, True])
def test_all_simd_preserves_resolved_auto_blockify_contract(resolved):
    metadata = {
        "auto_simt_effective_kind": "all_simd",
        "auto_blockify_v1_enabled": resolved,
        # A stale value must be replaced by the resolved policy.
        "auto_blockify_v1_runtime_cap": not resolved,
    }
    _preserve_all_simd_auto_blockify_contract(metadata)
    assert metadata["auto_blockify_v1_enabled"] is resolved
    assert metadata["auto_blockify_v1_runtime_cap"] is resolved


def test_auto_blockify_v1_blacklist_disables_both_implementations():
    metadata = {}
    opt = SimpleNamespace(enable_auto_blockify=None)
    with patch("triton.backends.ascend.compiler._is_auto_map_parallel_blocks_enabled", return_value=True), \
         patch("triton.backends.ascend.compiler._warn_auto_blockify_disabled") as warn:
        assert not _resolve_auto_blockify_v1_policy(ATOMIC_TTIR, metadata, opt)
    assert metadata["has_auto_blockify_blacklist_op"]
    warn.assert_called_once()


def test_auto_blockify_v1_explicit_blacklist_override_is_preserved():
    metadata = {"has_auto_blockify_blacklist_op": False}
    opt = SimpleNamespace(enable_auto_blockify=True)
    with patch("triton.backends.ascend.compiler._is_auto_map_parallel_blocks_enabled", return_value=False):
        assert _resolve_auto_blockify_v1_policy(ATOMIC_TTIR, metadata, opt)


def test_cache_modifier_does_not_disable_auto_blockify_v1():
    metadata = {}
    opt = SimpleNamespace(enable_auto_blockify=True)
    with patch("triton.backends.ascend.compiler._is_auto_map_parallel_blocks_enabled", return_value=False):
        assert _resolve_auto_blockify_v1_policy(CACHE_MODIFIER_TTIR, metadata, opt)
    assert not metadata["has_auto_blockify_blacklist_op"]


def test_route_transform_capability_is_single_resolved_fact():
    metadata = {
        "ttir_layout_merge_applied": True,
        "ttir_layout_coalesce_factor": 8,
        "ttir_layout_coalesce_axis": 0,
        "auto_blockify_v1_requested": True,
        "auto_blockify_v1_enabled": True,
        "auto_blockify_v1_disable_reasons": [],
    }
    opt = SimpleNamespace(
        compile_on_910_95=True,
        num_warps=4,
        logical_program_count_hint=9,
        physical_vector_core_count_hint=4,
        physical_aicore_count_hint=2,
    )
    capability = __import__("json").loads(_publish_route_transform_capability(metadata, opt))
    assert capability["layout_coalescing_applied"]
    assert capability["layout_coalescing_factor"] == 8
    assert capability["auto_blockify_v1_materializable"]
    assert capability["schema_version"] == 2
    assert capability["physical_vector_core_count_hint"] == 4
    assert capability["physical_aicore_count_hint"] == 2
    assert capability["cv_aiv_subblock_count"] == 2
    assert "runtime_schedules" not in capability
    assert capability["whole_kernel_superblock_factors"] == [1, 2, 4, 8, 16]
    assert capability["scope_superblock_factors"] == [1, 2, 4]
    assert capability["source_logical_program_count_hint"] == 9
    assert capability["logical_program_count_hint"] == 2
    metadata["backend_rejected_whole_kernel_factors"] = [4]
    metadata["backend_forced_route_candidate"] = "all_simd"
    rerouted = __import__("json").loads(_publish_route_transform_capability(metadata, opt))
    assert rerouted["whole_kernel_superblock_factors"] == [1, 2, 8, 16]
    assert rerouted["scope_superblock_factors"] == [1, 2, 4]
    assert rerouted["backend_forced_route_candidate"] == "all_simd"
    opt.enable_auto_bind_sub_block = False
    disabled = __import__("json").loads(_publish_route_transform_capability(metadata, opt))
    assert disabled["cv_aiv_subblock_count"] == 1
    assert disabled["logical_program_count_hint"] == capability["logical_program_count_hint"]


@pytest.mark.parametrize(
    "num_warps,expected",
    [
        (1, [1, 2, 4, 8, 16, 32, 64]),
        (2, [1, 2, 4, 8, 16, 32]),
        (4, [1, 2, 4, 8, 16]),
        (8, [1, 2, 4, 8]),
        (16, [1, 2, 4]),
        (32, [1, 2]),
        (64, [1]),
    ],
)
def test_route_transform_capability_respects_warp_limit(num_warps, expected):
    metadata = {
        "auto_blockify_v1_enabled": True,
        "auto_blockify_v1_disable_reasons": [],
    }
    opt = SimpleNamespace(
        compile_on_910_95=True,
        num_warps=num_warps,
        logical_program_count_hint=0,
    )
    capability = __import__("json").loads(_publish_route_transform_capability(metadata, opt))
    assert capability["whole_kernel_superblock_factors"] == expected
    assert capability["scope_superblock_factors"] == [factor for factor in expected if factor <= 4]


@pytest.mark.parametrize(
    "compile_on_910_95,num_warps,v1_materializable,expected",
    [
        (True, 1, True, True),
        (True, 4, True, True),
        (True, 4, False, False),
        (False, 4, True, False),
        (True, 0, True, False),
    ],
)
def test_scope_superblock_uses_npuir_abi_v2(compile_on_910_95, num_warps, v1_materializable, expected):
    opt = SimpleNamespace(compile_on_910_95=compile_on_910_95, num_warps=num_warps)
    assert _can_materialize_scope_superblock({}, opt, v1_materializable) is expected


@pytest.mark.parametrize(
    "metadata,option_factor,expected",
    [
        ({}, 2, 2),
        ({"auto_simt_scope_superblock_factor": 1}, 4, 4),
        ({
            "auto_simt_effective_kind": "mixed_simd_simt",
            "auto_simt_scope_superblock_factor": 2,
        }, 4, 1),
        ({
            "compile_mode": "simd_simt",
            "auto_simt_scope_superblock_factor": 4,
        }, 4, 1),
        ({
            "auto_simt_effective_kind": "all_simt_only",
            "auto_simt_superblock_factor": 4,
        }, 2, 4),
    ],
)
def test_selected_npuir_superblock_factor_respects_route_owner(metadata, option_factor, expected):
    opt = SimpleNamespace(superblock_factor=option_factor)
    assert _selected_npuir_superblock_factor(metadata, opt) == expected


@pytest.mark.parametrize("selected_factor", [32, 64])
@pytest.mark.parametrize(
    "ta_materializes,runtime_cap,expected_prefix",
    [
        (False, True, ["--enable-auto-blockify-loop"]),
        (True, True, []),
        (True, False, None),
    ],
)
def test_pure_simt_passes_selected_factor_to_bishengir(selected_factor, ta_materializes, runtime_cap, expected_prefix):
    options = []
    metadata = {
        "auto_blockify_v1_enabled": True,
        "auto_blockify_v1_runtime_cap": runtime_cap,
        "auto_simt_effective_kind": "all_simt_only",
        "auto_simt_superblock_factor": selected_factor,
    }
    opt = SimpleNamespace(
        enable_ta_auto_blockify_v1=ta_materializes,
        superblock_factor=1,
    )
    _append_pure_simt_auto_blockify_options(options, metadata, opt)
    expected = ([] if expected_prefix is None else [*expected_prefix, f"--super-block-factor={selected_factor}"])
    assert options == expected


def test_costmodel_analysis_view_materializes_v1_only_on_clone():
    metadata = {"auto_blockify_v1_enabled": True}
    original = SimpleNamespace(context=object())
    analysis = SimpleNamespace()
    with patch("triton.backends.ascend.compiler._parse_ttir_text", return_value=analysis) as parse, \
         patch("triton.backends.ascend.compiler._run_ta_auto_blockify_v1", return_value=True) as run_v1:
        result = _build_costmodel_analysis_ttir(original, metadata, SimpleNamespace())
    parse.assert_called_once()
    run_v1.assert_called_once()
    assert run_v1.call_args.args[0] is analysis
    assert run_v1.call_args.kwargs["super_block_factor"] == 1
    assert metadata["auto_simt_costmodel_analysis_v1_materialized"]
    assert metadata["auto_simt_costmodel_analysis_ir"] == "post_auto_blockify_v1_f1_ttir"
    assert result == str(analysis)


def test_backend_resource_usage_is_parsed_without_inventing_spill_cost():
    metadata = {}
    stderr = (b"remark: <unknown>:0:0: Greedy Register Allocator: Function: "
              b"kernel_vf_simt_simt_entry: MI Instruction count changed from 100 to 141; "
              b"Delta: 41 [-Rpass-analysis=size-info]\n"
              b"remark: <unknown>:0:0: Greedy Register Allocator: Function: "
              b"kernel_vf_simt_simt_entry: MI Instruction count changed from 141 to 244; "
              b"Delta: 103 [-Rpass-analysis=size-info]\n"
              b"[BISHENG] SIMT VF Function properties for kernel_vf_simt_simt_entry: "
              b"Stack size: 600 bytes, Used register number: 64\n"
              b"[BISHENG] Function properties for helper: Stack size: 16 bytes, "
              b"Used register number: 12\n")
    resources = _publish_backend_resource_usage(b"", stderr, metadata)
    assert resources == [
        {
            "function": "kernel_vf_simt_simt_entry",
            "execution_model": "simt_vf",
            "stack_size_bytes": 600,
            "used_registers": 64,
            "regalloc_generated_instructions": 144,
        },
        {
            "function": "helper",
            "execution_model": "cce",
            "stack_size_bytes": 16,
            "used_registers": 12,
        },
    ]
    assert metadata["backend_max_stack_size_bytes"] == 600
    assert metadata["backend_min_used_registers"] == 12
    assert "spill" not in " ".join(metadata)


def test_simt_private_stack_disassembly_separates_persistent_loop_body():
    disassembly = """
0000000000000010 <kernel_vf_simt_simt_entry>:
      10:  0001 STK.b32 R1, [R2+0]
      18:  0002 STK.b32 R3, [R2+1]
      20:  0003 LDK.b32 R4, [R2+0]
      28:  0004 STK.b32 R4, [R2+2]
      30:  0005 BRANCH 4294967280, !P0 <kernel_vf_simt_simt_entry+0x10>
      38:  0006 LDK.b32 R5, [R2+1]
      40:  0007 BRANCH 4294967264 <kernel_vf_simt_simt_entry+0x10>
      48:  0008 LDK.b32 R6, [R2+2]
0000000000000050 <helper>:
      50:  0009 END
"""
    facts = _parse_simt_private_stack_disassembly(disassembly)
    assert facts["persistent_loop_start_offset"] == 0x10
    assert facts["persistent_loop_backedge_offset"] == 0x30
    assert facts["static_private_stack_instructions"] == {
        "once_before": {"load": 0, "store": 2},
        "per_persistent_iteration": {"load": 2, "store": 1},
        "once_after": {"load": 1, "store": 0},
    }
    assert facts["static_private_stack_totals"] == {"load": 3, "store": 3}


def test_backend_private_stack_charge_uses_actual_ldk_loop_placement():
    resource = {
        "simt_private_stack_disassembly": {
            "static_private_stack_instructions": {
                "once_before": {"load": 5, "store": 59},
                "per_persistent_iteration": {"load": 454, "store": 136},
                "once_after": {"load": 0, "store": 0},
            }
        }
    }
    charge = _backend_private_stack_load_charge(resource, {"runtime_loop_iteration_count": 3}, factor=4, num_warps=4,
                                                group_cycles=27.26605902775)
    assert charge["active_warp_count"] == 16
    assert charge["static_load_instructions_per_warp"] == 1367
    assert charge["system_cycles"] == pytest.approx(1367 * 2 * 27.26605902775)


def test_cce_resource_usage_option_is_idempotent():
    base = "-cce-link-aicore-ll-module /tmp/libdevice.bc"
    once = _bisheng_options_with_resource_usage(base)
    assert once == base + " --cce-res-usage"
    assert _bisheng_options_with_resource_usage(once) == once
    assert _bisheng_options_with_resource_usage(None) == "--cce-res-usage"


def _factor_report(scores, mixed_score=400000.0):
    return json.dumps({
        "application_reason": "minimum_cost_candidate",
        "candidate_costs": {
            "all_simd": 320000.0,
            "all_simt_only": min(scores.values()),
            "mixed_simd_simt": mixed_score,
        },
        "selectable_candidates": ["all_simd", "all_simt_only", "mixed_simd_simt"],
        "selected_superblock_factor": min(scores, key=scores.get),
        "selected_route_elapsed_prediction": {
            "candidate": "all_simt_only",
            "base_selection_score": min(scores.values()),
            "fixed_system_cycles": 1700.0,
            "system_counter_frequency_mhz": 1000.0,
        },
        "stage_model": {
            "all_simt_factor_routes": {
                str(factor): {
                    "legal": True,
                    "total_system_cycles": score,
                    "runtime_full_group_count": 4,
                    "runtime_tail_program_count": 0,
                    "runtime_loop_iteration_count": 4,
                    "runtime_physical_program_count": 56,
                    "runtime_scheduling_slot_count": 56,
                    "runtime_logical_programs_per_scheduling_slot": 16,
                }
                for factor, score in scores.items()
            }
        },
    })


def test_backend_resource_feedback_rejects_relative_allocator_cliff_without_inventing_spill_cost():
    report = _factor_report({1: 242000.0, 2: 201000.0, 4: 164000.0})
    resources = {
        1: {"stack_size_bytes": 208, "used_registers": 127, "regalloc_generated_instructions": 194},
        2: {"stack_size_bytes": 320, "used_registers": 127, "regalloc_generated_instructions": 293},
        4: {"stack_size_bytes": 600, "used_registers": 64, "regalloc_generated_instructions": 598},
    }
    factor, feedback, updated = _select_backend_resource_aware_factor(report, resources, 4)
    updated = json.loads(updated)
    assert factor == 2
    assert feedback["rejected_factors"] == [4]
    assert updated["selected_superblock_factor"] == 2
    assert updated["candidate_costs"]["all_simt_only"] == 201000.0
    assert updated["selected_route_elapsed_prediction"]["predicted_microseconds"] == 202.7
    assert not feedback["absolute_spill_cost_applied"]
    assert feedback["absolute_spill_cost_reason"] == ("actual_ldk_count_and_loop_placement_unavailable")


def test_backend_resource_feedback_recompares_all_legal_route_candidates():
    report = _factor_report(
        {1: 241987.0, 2: 201505.0, 4: 163670.0},
        mixed_score=164083.0,
    )
    resources = {
        1: {"stack_size_bytes": 208, "used_registers": 127},
        2: {"stack_size_bytes": 320, "used_registers": 127},
        4: {"stack_size_bytes": 600, "used_registers": 64},
    }
    factor, feedback, updated = _select_backend_resource_aware_factor(report, resources, 4)
    updated = json.loads(updated)
    assert factor == 2
    assert feedback["post_backend_candidate_scores"]["all_simt_only"] == 201505.0
    assert feedback["post_backend_selected_candidate"] == "mixed_simd_simt"
    assert updated["backend_recommended_decision_kind"] == "mixed_simd_simt"
    assert updated["candidate_costs"]["mixed_simd_simt"] == 164083.0


def test_backend_cross_route_rerun_restores_mixed_runtime_memory_contract():
    metadata = {
        "compile_mode": "simt_only",
        "parallel_mode": "simt",
        "shared_mem_dynamic_size": 122880,
        "force_simt_only": True,
        "auto_simt_effective_kind": "all_simt_only",
    }
    opt = SimpleNamespace(parallel_mode="mix_simd_simt", shared_mem_dynamic_size=221184)

    rerouted = _metadata_for_backend_cross_route_rerun(metadata, [4], opt, "mixed_simd_simt")

    assert rerouted["compile_mode"] == "simd_simt"
    assert rerouted["parallel_mode"] == "mix_simd_simt"
    assert rerouted["shared_mem_dynamic_size"] == 221184
    assert rerouted["force_simt_only"] is False
    assert rerouted["backend_rejected_whole_kernel_factors"] == [4]
    assert rerouted["backend_forced_route_candidate"] == "mixed_simd_simt"
    assert "auto_simt_effective_kind" not in rerouted


def test_backend_resource_feedback_prices_actual_private_loads_for_every_factor():
    report = _factor_report({1: 242000.0, 2: 201000.0, 4: 164000.0})

    def resource(factor, body_loads):
        return {
            "stack_size_bytes": 64,
            "used_registers": 96,
            "simt_private_stack_disassembly": {
                "static_private_stack_instructions": {
                    "once_before": {"load": 0, "store": 0},
                    "per_persistent_iteration": {"load": body_loads, "store": 0},
                    "once_after": {"load": 0, "store": 0},
                }
            },
        }

    resources = {1: resource(1, 10), 2: resource(2, 20), 4: resource(4, 454)}
    factor, feedback, updated = _select_backend_resource_aware_factor(report, resources, 4, num_warps=4,
                                                                      private_load_group_cycles=27.26605902775)
    updated = json.loads(updated)
    assert factor == 2
    assert feedback["absolute_spill_cost_applied"]
    assert feedback["absolute_spill_cost_reason"] == ("actual_static_ldk_count_with_persistent_loop_placement")
    assert feedback["backend_adjusted_scores"]["4"] > feedback["backend_adjusted_scores"]["2"]
    assert updated["selected_route_elapsed_prediction"]["backend_private_stack_load_system_cycles"] == (pytest.approx(
        20 * 4 * (4 * 2) / 8 * 27.26605902775))


def test_backend_resource_feedback_keeps_resource_stable_minimum():
    report = _factor_report({1: 300.0, 2: 200.0, 4: 100.0})
    resources = {
        1: {"stack_size_bytes": 8, "used_registers": 32},
        2: {"stack_size_bytes": 4, "used_registers": 34},
        4: {"stack_size_bytes": 8, "used_registers": 34},
    }
    factor, feedback, _ = _select_backend_resource_aware_factor(report, resources, 4)
    assert factor == 4
    assert feedback["rejected_factors"] == []
