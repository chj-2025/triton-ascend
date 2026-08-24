// RUN: triton-opt --ta-simt-auto-blockify-v1="physical-vector-core-count=64 superblock-factor=1" %s | FileCheck %s

// V1 must silently skip helper functions and bodies without a logical program
// id.  In particular, it must not introduce a loop or claim materialization.

// CHECK-NOT: ta.auto_blockify_v1.materialized
// CHECK-NOT: ta.auto_blockify_v1.loop
// CHECK-NOT: scf.for

module {
  // CHECK-LABEL: tt.func private @private_helper
  // CHECK: tt.get_program_id x
  // CHECK: tt.return
  tt.func private @private_helper() {
    %pid = tt.get_program_id x : i32
    tt.return
  }

  // CHECK-LABEL: tt.func public @no_program_id
  // CHECK: arith.constant 0 : i32
  // CHECK: tt.return
  tt.func public @no_program_id() {
    %zero = arith.constant 0 : i32
    tt.return
  }
}
