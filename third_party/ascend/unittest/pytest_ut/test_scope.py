#!/usr/bin/env python3
import os

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import pytest
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.backends.ascend.compiler import (
    _analyze_auto_simt_scope_features,
    _estimate_auto_simt_scope_decision,
    _is_whole_body_void_simt_scope,
    _wrap_whole_body_void_simt_scope,
)


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel to MLIR."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, Options(), {}, {})
    return str(module)


# ============== Kernel definitions ==============


@triton.jit
def kernel_nested_scope(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test nested scopes."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        with al.scope(core_mode="vector"):
            with al.scope(core_mode="cube"):
                x = tl.load(x_ptr + i, mask=i < n)
                y = tl.load(y_ptr + i, mask=i < n)
                result = x + y
                tl.store(out_ptr + i, result, mask=i < n)


@triton.jit
def kernel_scope_escape(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test variable defined inside scope, used outside."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=i < n)
    # Use x outside of the scope
    a = x + 1.0
    tl.store(out_ptr + i, a, mask=i < n)


@triton.jit
def kernel_scope_cube(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test cube core mode."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="cube"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


@triton.jit
def kernel_scope_vector(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test vector core mode."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


@triton.jit
def kernel_scope_disable_auto_sync(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test disable auto sync."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(core_mode="vector", disable_auto_sync=True):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


@triton.jit
def kernel_scope_vector_mode_simt(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Test vector_mode SIMT annotation."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    with al.scope(vector_mode="simt"):
        x = tl.load(x_ptr + i, mask=i < n)
        y = tl.load(y_ptr + i, mask=i < n)
        result = x + y
        tl.store(out_ptr + i, result, mask=i < n)


# ============== Pytest tests ==============


def test_nested_scope():
    """Test nested scopes compile successfully."""
    mlir = compile_kernel(
        kernel_nested_scope, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256}
    )
    assert "scope.scope" in mlir
    assert len(mlir) > 0


def test_scope_escape():
    """Test variable escaping from scope."""
    mlir = compile_kernel(kernel_scope_escape, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256})
    assert "scope.scope" in mlir
    assert len(mlir) > 0


def test_scope_cube_mode():
    """Test cube core mode generates correct attributes."""
    mlir = compile_kernel(
        kernel_scope_cube, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256}
    )
    assert "scope.scope" in mlir
    # Check for cube core type attribute
    assert "hivm.tcore_type" in mlir or "CUBE" in mlir.upper()


def test_scope_vector_mode():
    """Test vector core mode generates correct attributes."""
    mlir = compile_kernel(
        kernel_scope_vector, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256}
    )
    assert "scope.scope" in mlir
    # Check for vector core type attribute
    assert "hivm.tcore_type" in mlir or "VECTOR" in mlir.upper()


def test_scope_disable_auto_sync():
    """Test disable auto sync generates correct attributes."""
    mlir = compile_kernel(
        kernel_scope_disable_auto_sync,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert "scope.scope" in mlir
    # Check for disable auto sync attribute
    assert "hivm.disable_auto_sync" in mlir


def test_scope_vector_mode_simt():
    """Test vector_mode='simt' generates the canonical vec_mode attr."""
    mlir = compile_kernel(
        kernel_scope_vector_mode_simt,
        {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"},
        {"BLOCK": 256},
    )
    assert "scope.scope" in mlir
    assert "vec_mode" in mlir
    assert "simt" in mlir


def test_auto_simt_scope_cost_model_selects_rank2_gather_reduce():
    ttir = """
module {
  tt.func public @rank2_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) {
    %c0 = arith.constant 0 : i32
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %2 = tt.broadcast %1 : tensor<8x1xi32> -> tensor<8x8xi32>
    %3 = tt.addptr %arg0, %2 : !tt.ptr<f32>, tensor<8x8xi32>
    %4 = tt.load %3 : tensor<8x8x!tt.ptr<f32>>
    %5 = arith.cmpf olt, %4, %4 : tensor<8x8xf32>
    %6 = tt.broadcast %5 : tensor<8x8xi1> -> tensor<8x8xi1>
    %7 = "tt.reduce"(%4) ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<8x8xf32>) -> tensor<8xf32>
    %8 = tt.addptr %arg2, %0 : !tt.ptr<f32>, tensor<8xi32>
    tt.store %8, %7 : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
"""
    features = _analyze_auto_simt_scope_features(ttir)
    decision = _estimate_auto_simt_scope_decision(features)
    assert decision["eligible"]
    assert decision["choose_simt"]


def test_auto_simt_scope_cost_model_keeps_rank1_vector_kernel():
    ttir = """
module {
  tt.func public @rank1_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.addptr %arg0, %0 : !tt.ptr<f32>, tensor<128xi32>
    %2 = tt.load %1 : tensor<128x!tt.ptr<f32>>
    %3 = arith.addf %2, %2 : tensor<128xf32>
    %4 = tt.addptr %arg1, %0 : !tt.ptr<f32>, tensor<128xi32>
    tt.store %4, %3 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
"""
    features = _analyze_auto_simt_scope_features(ttir)
    decision = _estimate_auto_simt_scope_decision(features)
    assert not decision["choose_simt"]
    assert decision["reason"] == "rank1_or_scalar_kernel"


def test_auto_simt_scope_wraps_whole_body_as_void_simt_scope():
    ttir = """
module {
  tt.func public @wrap_kernel(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    tt.return
  }
}
"""
    scoped = _wrap_whole_body_void_simt_scope(ttir)
    assert "scope.scope : () -> ()" in scoped
    assert 'vec_mode = "simt"' in scoped
    assert _is_whole_body_void_simt_scope(scoped)


# ============== Main for manual testing ==============

if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Nested Scopes")
    print("=" * 60)
    mlir = compile_kernel(
        kernel_nested_scope, {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256}
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)

    print("\n" + "=" * 60)
    print("Test 2: Scope Escape")
    print("=" * 60)
    mlir = compile_kernel(kernel_scope_escape, {"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 256})
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)
