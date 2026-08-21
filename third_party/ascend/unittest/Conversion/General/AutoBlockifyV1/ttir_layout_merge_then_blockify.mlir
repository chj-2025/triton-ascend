// RUN: triton-opt %s --ttir-layout-merge \
// RUN:   --ta-simt-auto-blockify-v1="physical-vector-core-count=64 superblock-factor=1" \
// RUN:   | FileCheck %s

// The route model consumes this post-transform TTIR shape: layout coalescing
// first expands one 16-element logical tile into 16 contiguous tiles, then V1
// wraps the resulting logical programs in its physical-core loop.  Neither
// transform lowers TTIR to Linalg/HIVM.

// CHECK-LABEL: module attributes {
// CHECK-SAME: hacc.coalesce_axis = 0 : i32
// CHECK-SAME: hacc.coalesce_factor = 16 : i32
// CHECK-SAME: ta.auto_blockify_v1.materialized = 1 : i32
// CHECK-SAME: ta.ttir_layout_merge.applied
// CHECK-LABEL: tt.func public @layout_then_blockify
// CHECK-SAME: attributes {ta.auto_blockify_v1, ta.auto_blockify_v1.superblock_factor = 1 : i32}
// CHECK: tt.get_program_id x {{.*}} : i32
// CHECK: scf.for
// CHECK-NOT: tt.get_program_id
// CHECK: tt.load {{.*}} : tensor<16x16x!tt.ptr<f32>>
// CHECK: tt.store
// CHECK: } {ta.auto_blockify_v1.loop, ta.auto_blockify_v1.schedule}
// CHECK-NOT: linalg.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @layout_then_blockify(%src: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                                       %dst: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %c256 = arith.constant dense<256> : tensor<16xi32>
    %zero = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %base = arith.muli %pid, %c16 : i32
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %base_splat = tt.splat %base : i32 -> tensor<16xi32>
    %offsets = arith.addi %base_splat, %range : tensor<16xi32>
    %mask = arith.cmpi slt, %offsets, %c256 : tensor<16xi32>
    %src_splat = tt.splat %src : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %src_ptrs = tt.addptr %src_splat, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    %value = tt.load %src_ptrs, %mask, %zero : tensor<16x!tt.ptr<f32>>
    %dst_splat = tt.splat %dst : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %dst_ptrs = tt.addptr %dst_splat, %offsets : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    tt.store %dst_ptrs, %value, %mask : tensor<16x!tt.ptr<f32>>
    tt.return
  }
}
