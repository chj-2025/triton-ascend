// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// A canonical SIMT scope must select the mixed BiShengIR pipeline and the
// SIMT-aware runtime launch path. The scope is an input contract and may be
// consumed by outlining/lowering, so this test guards the surviving function
// contract rather than requiring the source scope operation in output IR.
// CHECK-LABEL: func.func @simt_scope
// CHECK-SAME: parallel_mode = "mix_simd_simt"
// CHECK-NOT: parallel_mode = "mix_simd"
tt.func public @simt_scope(%arg0: !tt.ptr<f32>) {
  %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
  %ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
  scope.scope : () -> () {
    tt.store %ptrs, %zero : tensor<16x!tt.ptr<f32>>
    scope.return
  } {vector_mode = "simt"}
  tt.return
}
