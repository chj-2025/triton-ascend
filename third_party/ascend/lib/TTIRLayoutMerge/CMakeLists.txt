add_triton_library(TTIRLayoutMerge
  RowCoalescing.cpp
  TTIRLayoutMergePass.cpp

  DEPENDS
  TTIRLayoutMergePassIncGen

  LINK_LIBS PUBLIC
  TritonToLinalg
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  TritonIR
)
