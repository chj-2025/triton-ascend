add_mlir_library(AscendModelAnalysis
  PARTIAL_SOURCES_INTENDED
  HardwareConfig.cpp
  PipelineAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/include/AscendModel/Analysis

  DEPENDS
  AscendModelOpsIncGen

  LINK_LIBS PUBLIC
  AscendModelProfile
  AscendModelIR
  MLIRIR
  MLIRSupport
  LLVMSupport
)
