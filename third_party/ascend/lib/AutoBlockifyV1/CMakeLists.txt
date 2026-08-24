add_triton_library(AutoBlockifyV1
  SIMTAutoBlockifyV1.cpp

  DEPENDS
  AutoBlockifyV1PassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  TritonIR
)
