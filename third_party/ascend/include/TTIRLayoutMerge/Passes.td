#ifndef TTIR_LAYOUT_MERGE_PASSES
#define TTIR_LAYOUT_MERGE_PASSES

include "mlir/Pass/PassBase.td"

def TTIRLayoutMerge : Pass<"ttir-layout-merge", "mlir::ModuleOp"> {
    let summary = "Materialize TTIR layout merges before route cost analysis";
    let description = [{
      Applies implicit-permute normalization, strided-axis coalescing,
      contiguous tile-chunk coalescing, and independent row coalescing while
      the IR is still TTIR.  The SIMD/SIMT Route Model is scheduled after this
      pass so it observes the actual merged tensor shapes and memory-access
      structure.
    }];
    let constructor = "triton::createTTIRLayoutMergePass()";
}

#endif // TTIR_LAYOUT_MERGE_PASSES
