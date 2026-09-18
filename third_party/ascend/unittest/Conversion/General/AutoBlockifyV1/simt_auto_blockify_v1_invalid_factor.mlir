// RUN: not triton-opt --ta-simt-auto-blockify-v1="physical-vector-core-count=64 superblock-factor=3" %s 2>&1 | FileCheck %s --check-prefix=F3
// RUN: not triton-opt --ta-simt-auto-blockify-v1="physical-vector-core-count=64 superblock-factor=128" %s 2>&1 | FileCheck %s --check-prefix=F128

// F3: superblock-factor must be one of 1, 2, 4, 8, 16, 32 or 64, got 3
// F128: superblock-factor must be one of 1, 2, 4, 8, 16, 32 or 64, got 128

module {
  tt.func public @invalid_factor(%arg0: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %ptr = tt.addptr %arg0, %pid : !tt.ptr<f32>, i32
    %zero = arith.constant 0.0 : f32
    tt.store %ptr, %zero : !tt.ptr<f32>
    tt.return
  }
}
