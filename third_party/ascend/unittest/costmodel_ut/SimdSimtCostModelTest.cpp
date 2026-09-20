#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/StageCostModels.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

using mlir::ascend::HardwareProfile;
using mlir::ascend::LogicalStage;
using mlir::ascend::LogicalStageCost;
using mlir::ascend::ScopeHandoffCost;
using mlir::ascend::SimdSimtFeatureSummary;
using mlir::ascend::solveStageRoutes;
using mlir::ascend::StageCostEvaluator;
using mlir::ascend::StageCostModelKind;
using mlir::ascend::StageCostTable;
using mlir::ascend::StageFeatureAnalysis;
using mlir::ascend::StageImplementationCost;
using mlir::ascend::StageMemoryAccess;
using mlir::ascend::StageMemoryLayoutCost;
using mlir::ascend::StageMode;
using mlir::ascend::StageModeLegalityAnalysis;
using mlir::ascend::StagePartition;
using mlir::ascend::StagePartitioner;
using mlir::ascend::StagePartitionerOptions;
using mlir::ascend::StageScheduleKind;
using mlir::ascend::StageWorkload;
using mlir::ascend::StageWorkloadAnalysis;
using mlir::ascend::TensorOperationWorkload;
using mlir::ascend::TriangularSolveFacts;

namespace {

SimdSimtFeatureSummary triangularBt16StageFeatures() {
  SimdSimtFeatureSummary f;
  f.simtAnchors.count = 1;
  TriangularSolveFacts triangular;
  triangular.blockRows = 16;
  triangular.blockColumns = 16;
  triangular.accumulatorType = "f32";
  triangular.recurrenceStartRow = 2;
  triangular.recurrenceLoopCount = 14;
  f.simtAnchors.triangularSolves.push_back(triangular);
  return f;
}

} // namespace

namespace {

HardwareProfile hardwareProfile(ScopeHandoffCost scopeHandoff = {}) {
  HardwareProfile profile;
  profile.profileVersion = "unit-test-profile-v1";
  profile.target = "Ascend950PR_9579";
  auto fill = [](auto &mode) {
    mode.setupCycles = 10.0;
    mode.vectorWidthBits = 2048;
    mode.vectorWidth = 64;
    mode.issueWidth = 64;
    mode.operationRates["f32.add"] = {1.0, 1.0};
    mode.operationRates["f32.mul"] = {1.0, 1.0};
    mode.operationRates["f32.max"] = {1.0, 1.0};
    mode.operationRates["convert.cast"] = {1.0, 1.0};
    mode.loadBytesPerCycle = 32.0;
    mode.storeBytesPerCycle = 16.0;
    mode.loadWarpInstructionsPerCycle = 1.0;
    mode.storeWarpInstructionsPerCycle = 1.0;
    mode.predicateOperationsPerCycle = 1.0;
    mode.shuffleLanesPerCycle = 32.0;
    mode.dotSetupCycles = 8.0;
    mode.dotFlopsPerCycle = 64.0;
    mode.issueOperationsPerCycle = 4.0;
    mode.prefixScanStepLatencyCycles = 2.0;
  };
  fill(profile.simd);
  fill(profile.simt);
  profile.simt.vectorWidth = 1;
  profile.simt.vectorWidthBits = 1;
  profile.simt.issueWidth = 32;
  profile.scopeHandoff = std::move(scopeHandoff);
  return profile;
}

LogicalStage
logicalStage(llvm::StringRef id, StageCostModelKind kind,
             StageScheduleKind schedule = StageScheduleKind::StraightLine,
             int64_t iterations = 1) {
  LogicalStage stage;
  stage.id = id.str();
  stage.costModelKind = kind;
  stage.scheduleKind = schedule;
  stage.iterationCount = iterations;
  stage.simdLegal = true;
  stage.simtLegal = true;
  stage.legalSimtFactors = {1};
  stage.workload.paysKernelSetup = true;
  stage.workload.operationCounts["f32.add"][64] = 1.0;
  stage.workload.issueElements = 4.0;
  return stage;
}

llvm::Expected<StageCostTable>
evaluateOneStage(LogicalStage stage,
                 HardwareProfile profile = hardwareProfile()) {
  StagePartition partition;
  partition.stages.push_back(std::move(stage));
  return StageCostEvaluator().evaluate(partition, profile);
}

} // namespace

TEST(SimdSimtCostModelTest, CompleteUnitProfileIsValid) {
  EXPECT_TRUE(hardwareProfile().isValid());
}

TEST(SimdSimtCostModelTest, StoreServiceProfileRejectsPartialAndInvalidData) {
  auto profile = hardwareProfile();
  profile.simt.storeFootprintUnitBytes = 128;
  EXPECT_FALSE(profile.isValid());
  profile.simt.storeCyclesPerFootprintUnit = 2;
  profile.simt.storeIssueFloorByWarpCount = {{8, 30}};
  EXPECT_TRUE(profile.isValid());
  profile.simt.storeIssueFloorByWarpCount[8] = -1;
  EXPECT_FALSE(profile.isValid());
}

TEST(SimdSimtCostModelTest, SuperBlockSelectsRatesAtTotalLaunchedWarpCount) {
  LogicalStage stage =
      logicalStage("simt_payload", StageCostModelKind::ScalarIssue);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 4};
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["f32.add"][32] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.setupCyclesByWarpCount = {{4, 20.0}, {16, 80.0}};
  auto &add = profile.simt.operationRates["f32.add"];
  add.throughputByWarpCount = {{4, 2.0}, {16, 16.0}};

  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 2u);
  EXPECT_EQ(costs[0].implementation.superblockFactor, 1);
  EXPECT_EQ(costs[1].implementation.superblockFactor, 4);
  EXPECT_DOUBLE_EQ(costs[0].resources.setup, 20.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.setup, 80.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.compute, 16.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.compute, 2.0);
}

TEST(SimdSimtCostModelTest, StageHasOnlySimdOrSimtImplementations) {
  LogicalStage stage = logicalStage("scalar", StageCostModelKind::ScalarIssue);
  auto table = evaluateOneStage(std::move(stage));
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  ASSERT_EQ(table->stages.front().implementations.size(), 2u);
  EXPECT_EQ(table->stages.front().implementations[0].implementation.mode,
            StageMode::SIMD);
  EXPECT_EQ(table->stages.front().implementations[1].implementation.mode,
            StageMode::SIMT);
}

TEST(SimdSimtCostModelTest, SimdPricesShortAxesPerSegmentAndElementWidth) {
  auto simdCost = [](int64_t elementBits, int64_t contiguousElements,
                     double segmentCount) {
    LogicalStage stage =
        logicalStage("short-axis", StageCostModelKind::ScalarMath);
    stage.workload.operationElements["f32.add"] =
        static_cast<double>(contiguousElements) * segmentCount;
    TensorOperationWorkload tensor;
    tensor.operation = "f32.add";
    tensor.elementBitWidth = elementBits;
    tensor.logicalElements =
        static_cast<double>(contiguousElements) * segmentCount;
    tensor.segmentCount = segmentCount;
    tensor.contiguousElementsPerSegment = contiguousElements;
    stage.workload.tensorOperationWorkloads.push_back(std::move(tensor));
    auto table = evaluateOneStage(std::move(stage));
    if (!table) {
      ADD_FAILURE() << llvm::toString(table.takeError());
      return mlir::ascend::StageResourceCycles{};
    }
    return table->stages.front().implementations.front().resources;
  };

  // These are already projected segments (e.g. broadcast-constrained rows),
  // not initial TTIR shapes. A short segment still uses a vector instruction.
  auto fourBySixteen = simdCost(32, 16, 4.0);
  auto eightByEight = simdCost(32, 8, 8.0);
  auto sixteenByFour = simdCost(32, 4, 16.0);
  EXPECT_DOUBLE_EQ(fourBySixteen.compute, 4.0);
  EXPECT_DOUBLE_EQ(eightByEight.compute, 8.0);
  EXPECT_DOUBLE_EQ(sixteenByFour.compute, 16.0);
  EXPECT_DOUBLE_EQ(sixteenByFour.scalar, 0.0);
  EXPECT_DOUBLE_EQ(fourBySixteen.issue, 1.0);
  EXPECT_DOUBLE_EQ(eightByEight.issue, 2.0);
  EXPECT_DOUBLE_EQ(sixteenByFour.issue, 4.0);

  // FP16 uses twice as many elements per 256-byte instruction; width is not
  // silently inherited from the old FP32-only vectorWidth field.
  auto fp16Rows = simdCost(16, 128, 2.0);
  EXPECT_DOUBLE_EQ(fp16Rows.compute, 2.0);
}

TEST(SimdSimtCostModelTest,
     WorkloadDoesNotInferSegmentsOrScalarFallbackFromLogicalRows) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%a4x16: tensor<4x16xf32>,
                        %b4x16: tensor<4x16xf32>,
                        %a8x8: tensor<8x8xf32>,
                        %b8x8: tensor<8x8xf32>,
                        %a16x4: tensor<16x4xf32>,
                        %b16x4: tensor<16x4xf32>) {
        %four_by_sixteen = arith.addf %a4x16, %b4x16 : tensor<4x16xf32>
        %eight_by_eight = arith.addf %a8x8, %b8x8 : tensor<8x8xf32>
        %sixteen_by_four = arith.addf %a16x4, %b16x4 : tensor<16x4xf32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("short_axis_geometry", StageCostModelKind::ScalarMath);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  ASSERT_TRUE(function);
  for (mlir::Operation &operation :
       function.getBody().front().without_terminator())
    stage.operations.push_back(&operation);
  partition.stages.push_back(std::move(stage));

  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  const auto &workloads =
      partition.stages.front().workload.tensorOperationWorkloads;
  ASSERT_EQ(workloads.size(), 1u);
  EXPECT_EQ(workloads.front().contiguousElementsPerSegment, 64);
  EXPECT_EQ(workloads.front().elementBitWidth, 32);
  EXPECT_DOUBLE_EQ(workloads.front().logicalElements, 192.0);
  EXPECT_DOUBLE_EQ(workloads.front().segmentCount, 3.0);
}

TEST(SimdSimtCostModelTest,
     WorkloadRetainsRowBroadcastAndComparisonInputWidth) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%a: tensor<8x8xf32>, %row: tensor<8x1xf32>,
                        %a3: tensor<2x4x8xf32>, %row3: tensor<2x4x1xf32>) {
        %b = "tt.broadcast"(%row) : (tensor<8x1xf32>) -> tensor<8x8xf32>
        %sum = arith.addf %a, %b : tensor<8x8xf32>
        %mask = arith.cmpf olt, %a, %sum : tensor<8x8xf32>
        %b3 = "tt.broadcast"(%row3) : (tensor<2x4x1xf32>) -> tensor<2x4x8xf32>
        %sum3 = arith.addf %a3, %b3 : tensor<2x4x8xf32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("broadcast", StageCostModelKind::ScalarMath);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  for (mlir::Operation &operation :
       function.getBody().front().without_terminator())
    if (mlir::isa<mlir::arith::AddFOp, mlir::arith::CmpFOp>(operation))
      stage.operations.push_back(&operation);
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  const auto &workloads =
      partition.stages.front().workload.tensorOperationWorkloads;
  ASSERT_EQ(workloads.size(), 2u);
  for (const auto &workload : workloads) {
    EXPECT_EQ(workload.elementBitWidth, 32);
    if (workload.operation == "f32.add") {
      EXPECT_EQ(workload.contiguousElementsPerSegment, 8);
      EXPECT_DOUBLE_EQ(workload.segmentCount, 16);
    } else {
      EXPECT_EQ(workload.operation, "predicate.cmp");
      EXPECT_EQ(workload.contiguousElementsPerSegment, 64);
      EXPECT_DOUBLE_EQ(workload.segmentCount, 1);
    }
  }
}

TEST(SimdSimtCostModelTest, WorkloadUsesPostFlattenBroadcastSuffix) {
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%x: tensor<2x4x8xf32>,
                        %leading: tensor<1x4x8xf32>,
                        %middle: tensor<2x1x8xf32>,
                        %x4: tensor<2x4x8x4xf32>,
                        %leading4: tensor<1x4x8x4xf32>,
                        %unit: tensor<2x4x8x1xf32>,
                        %row: tensor<2x4x1x1xf32>,
                        %ix: tensor<2x4x8xi32>, %iy: tensor<2x1x8xi32>) {
        %a = "tt.broadcast"(%leading) : (tensor<1x4x8xf32>) -> tensor<2x4x8xf32>
        %s0 = arith.addf %x, %a : tensor<2x4x8xf32>
        %b = "tt.broadcast"(%middle) : (tensor<2x1x8xf32>) -> tensor<2x4x8xf32>
        %s1 = arith.addf %x, %b : tensor<2x4x8xf32>
        %c = "tt.broadcast"(%leading4) : (tensor<1x4x8x4xf32>) -> tensor<2x4x8x4xf32>
        %s2 = arith.addf %x4, %c : tensor<2x4x8x4xf32>
        %d = "tt.broadcast"(%row) : (tensor<2x4x1x1xf32>) -> tensor<2x4x8x1xf32>
        %s3 = arith.addf %unit, %d : tensor<2x4x8x1xf32>
        %e = "tt.broadcast"(%iy) : (tensor<2x1x8xi32>) -> tensor<2x4x8xi32>
        %s4 = arith.addi %ix, %e : tensor<2x4x8xi32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  for (mlir::Operation &operation :
       function.getBody().front().without_terminator()) {
    if (!mlir::isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(operation))
      continue;
    LogicalStage stage =
        logicalStage("flatten", StageCostModelKind::ScalarMath);
    stage.operations.push_back(&operation);
    partition.stages.push_back(std::move(stage));
  }
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  ASSERT_EQ(partition.stages.size(), 5u);
  const int64_t widths[] = {32, 8, 128, 8, 8};
  const double segments[] = {2, 8, 2, 8, 8};
  for (size_t i = 0; i < partition.stages.size(); ++i) {
    const auto &groups = partition.stages[i].workload.tensorOperationWorkloads;
    ASSERT_EQ(groups.size(), 1u);
    EXPECT_EQ(groups[0].contiguousElementsPerSegment, widths[i]);
    EXPECT_DOUBLE_EQ(groups[0].segmentCount, segments[i]);
  }
}

TEST(SimdSimtCostModelTest,
     ScopeSuperBlockLegalityRequiresBackendAndResourceMaximum) {
  auto makePartition = []() {
    StagePartition partition;
    LogicalStage stage =
        logicalStage("payload", StageCostModelKind::LoopCarriedRecurrence,
                     StageScheduleKind::LoopCarriedSerial, /*iterations=*/16);
    stage.features.hasLoop = true;
    stage.features.hasLoopCarriedDataDependency = true;
    stage.localSimtMaterializable = true;
    stage.localSuperblockMaterializable = true;
    stage.localSimtFactors = {1};
    partition.stages.push_back(std::move(stage));
    return partition;
  };

  StagePartition f1Only = makePartition();
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(f1Only, 4, false))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(f1Only.stages[0].localSimtFactors, (std::vector<int64_t>{1}));

  StagePartition scopeSuperblock = makePartition();
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(scopeSuperblock, 4, true, 4))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(scopeSuperblock.stages[0].localSimtFactors,
            (std::vector<int64_t>{1, 2, 4}));

  // ABI-v2 creates an F1 V1 scheduling loop and refines only the selected
  // scope after bufferization, but local and whole-kernel factors still share
  // the same target/runtime warp-resource maximum.
  StagePartition mixedOnly = makePartition();
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(mixedOnly, 1, true, 1))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(mixedOnly.stages[0].legalSimtFactors, (std::vector<int64_t>{1}));
  EXPECT_EQ(mixedOnly.stages[0].localSimtFactors, (std::vector<int64_t>{1}));
  auto mixedOnlyCosts = evaluateOneStage(mixedOnly.stages[0]);
  if (!mixedOnlyCosts)
    FAIL() << llvm::toString(mixedOnlyCosts.takeError());
  ASSERT_EQ(mixedOnlyCosts->stages[0].implementations.size(), 3u);
  EXPECT_EQ(mixedOnlyCosts->stages[0].legalSimtFactors,
            (std::vector<int64_t>{1}));
  EXPECT_EQ(mixedOnlyCosts->stages[0].localSimtFactors,
            (std::vector<int64_t>{1}));
}

TEST(SimdSimtCostModelTest, LocalScopeFactorsHonorKernelResourceMaximum) {
  StagePartition partition;
  LogicalStage stage;
  stage.id = "indirect_tile_gather";
  stage.costModelKind = StageCostModelKind::IndirectGatherMemory;
  stage.scheduleKind = StageScheduleKind::PartiallyDependent;
  stage.iterationCount = 1;
  stage.localSimtMaterializable = true;
  stage.localSuperblockMaterializable = true;
  partition.stages.push_back(std::move(stage));

  ASSERT_FALSE(StageModeLegalityAnalysis().analyze(
      partition, /*maximumSuperblockFactor=*/2,
      /*scopeSuperblockMaterializable=*/true,
      /*maximumScopeSuperblockFactor=*/2));
  const LogicalStage &result = partition.stages.front();
  EXPECT_EQ(result.legalSimtFactors, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(result.localSimtFactors, (std::vector<int64_t>{1, 2}));
}

TEST(SimdSimtCostModelTest, KernelMixedRouteComesFromAdjacentStageModes) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto addStage = [&](llvm::StringRef id, double simd, double simt) {
    mlir::ascend::LogicalStageCost stage;
    stage.id = id.str();
    stage.localSimtMaterializable = true;
    stage.localSimtFactors = {1};
    auto cost = [&](StageMode mode, double cycles, bool localScope = false) {
      mlir::ascend::StageImplementationCost result;
      result.implementation = {mode, 1, localScope};
      result.totalCycles = cycles;
      return result;
    };
    stage.implementations = {cost(StageMode::SIMD, simd),
                             cost(StageMode::SIMT, simt),
                             cost(StageMode::SIMT, simt, true)};
    table.stages.push_back(stage);
  };
  addStage("head", 10.0, 20.0);
  addStage("payload", 100.0, 50.0);
  addStage("store", 30.0, 45.0);

  ScopeHandoffCost scopeHandoff;
  scopeHandoff.fixedScopeCycles = 12.0;
  auto result = solveStageRoutes(table, scopeHandoff);
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  EXPECT_DOUBLE_EQ(result->allSimd.totalCycles, 140.0);
  EXPECT_DOUBLE_EQ(result->allSimt.totalCycles, 115.0);
  EXPECT_DOUBLE_EQ(result->mixed.totalCycles, 102.0);
  ASSERT_EQ(result->mixed.implementations.size(), 3u);
  EXPECT_EQ(result->mixed.implementations[0].mode, StageMode::SIMD);
  EXPECT_EQ(result->mixed.implementations[1].mode, StageMode::SIMT);
  EXPECT_EQ(result->mixed.implementations[2].mode, StageMode::SIMD);
  ASSERT_EQ(result->mixed.scopeHandoffCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(result->mixed.scopeHandoffCycles[1], 12.0);
}

TEST(SimdSimtCostModelTest, MixedScopePaysExactBidirectionalUbHandoffCost) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles, bool localScope = false) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1, localScope};
    cost.totalCycles = cycles;
    return cost;
  };
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.implementations = {makeCost(StageMode::SIMD, 10.0),
                          makeCost(StageMode::SIMT, 20.0)};
  mlir::ascend::LogicalStageCost payload;
  payload.id = "large_result_payload";
  payload.localSimtMaterializable = true;
  payload.localSimtFactors = {1};
  payload.scopeInputTensorBytes = 4096;
  payload.scopeOutputTensorBytes = 16384;
  payload.implementations = {makeCost(StageMode::SIMD, 100.0),
                             makeCost(StageMode::SIMT, 10.0),
                             makeCost(StageMode::SIMT, 10.0, true)};
  mlir::ascend::LogicalStageCost tail = head;
  tail.id = "tail";
  table.stages = {head, payload, tail};

  ScopeHandoffCost scopeHandoff;
  scopeHandoff.inputHandoffBytesPerCycle = 4096.0 / 48.0;
  scopeHandoff.outputHandoffBytesPerCycle = 16384.0 / 160.0;
  auto routes = solveStageRoutes(table, scopeHandoff);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  // Input: 4096/256 + 4096/(4*32) = 48 cycles.
  // Output: 16384/(4*32) + 16384/512 = 160 cycles.
  // Head/payload/tail: 10 + (10 + 208) + 10 = 238 cycles.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 238.0);
  ASSERT_EQ(routes->mixed.scopeHandoffCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[0], 0.0);
  // Boundary tensor traffic is Stage body work.  With no fixed envelope in
  // this test, the mode-switch handoff itself is zero.
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[1], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[2], 0.0);
  EXPECT_GT(routes->mixed.totalCycles, routes->allSimd.totalCycles);
}

TEST(SimdSimtCostModelTest,
     MixedScopeSuperBlockCountsOneTransitionAndEveryProgramHandoff) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, int64_t factor, double cycles,
                      bool localScope = false) {
    StageImplementationCost cost;
    cost.implementation = {mode, factor, localScope};
    cost.totalCycles = cycles;
    return cost;
  };

  LogicalStageCost head;
  head.id = "head";
  head.implementations = {makeCost(StageMode::SIMD, 1, 10.0),
                          makeCost(StageMode::SIMT, 1, 100.0)};

  LogicalStageCost payload;
  payload.id = "payload";
  payload.localSimtMaterializable = true;
  payload.localSimtFactors = {1, 2, 4};
  payload.scopeInputTensorBytes = 4096;
  payload.scopeOutputTensorBytes = 4096;
  payload.implementations = {makeCost(StageMode::SIMD, 1, 1000.0),
                             makeCost(StageMode::SIMT, 1, 1000.0),
                             makeCost(StageMode::SIMT, 1, 1000.0, true),
                             makeCost(StageMode::SIMT, 2, 300.0, true),
                             makeCost(StageMode::SIMT, 4, 100.0, true)};

  LogicalStageCost tail = head;
  tail.id = "tail";
  table.stages = {head, payload, tail};

  ScopeHandoffCost scopeHandoff;
  scopeHandoff.fixedScopeCycles = 80.0;
  scopeHandoff.inputHandoffBytesPerCycle = 4096.0 / 48.0;
  scopeHandoff.outputHandoffBytesPerCycle = 4096.0 / 40.0;
  auto routes = solveStageRoutes(table, scopeHandoff);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());

  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  // The combined F4 scope pays one 80-cycle fixed pair. Each of the four
  // logical programs transfers 48 input cycles and 40 output cycles as Stage
  // body work, not as a mode-switch handoff.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles,
                   10.0 + 100.0 + 80.0 + 4.0 * (48.0 + 40.0) + 10.0);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[1], 80.0);
}

TEST(SimdSimtCostModelTest, ReportsMeasuredAndFallbackSuperBlockEvidence) {
  LogicalStage measured =
      logicalStage("measured_memory", StageCostModelKind::ContinuousTileMemory);
  measured.legalSimtFactors = {1, 2};
  measured.workload.continuousLoadCounts[{1024, 256}] = 1.0;
  measured.workload.loadBytes = 1024.0;
  measured.workload.loadWarpInstructions = 8.0;
  HardwareProfile profile = hardwareProfile();
  profile.simt.continuousMemoryGroupBaseWarpCount = 1;
  profile.simt.continuousLoadGroupCyclesByFactorAndBytes = {
      {1, {{1024, 100.0}}}, {2, {{1024, 130.0}}}};
  auto measuredTable = evaluateOneStage(measured, profile);
  if (!measuredTable)
    FAIL() << llvm::toString(measuredTable.takeError());
  ASSERT_EQ(measuredTable->stages.front().implementations.size(), 3u);
  EXPECT_EQ(measuredTable->stages.front().implementations[2].formulaEvidence,
            "factor_specific_measured_group_curve");

  HardwareProfile mismatchedWarpProfile = profile;
  mismatchedWarpProfile.baseSimtWarpCount = 2;
  auto mismatchedTable = evaluateOneStage(measured, mismatchedWarpProfile);
  if (!mismatchedTable)
    FAIL() << llvm::toString(mismatchedTable.takeError());
  ASSERT_EQ(mismatchedTable->stages.front().implementations.size(), 3u);
  EXPECT_EQ(mismatchedTable->stages.front().implementations[2].formulaEvidence,
            "uncalibrated_superblock_fallback");

  LogicalStage fallback =
      logicalStage("fallback", StageCostModelKind::ScalarIssue);
  fallback.legalSimtFactors = {1, 2};
  fallback.workload.scalarOperations = 16.0;
  auto fallbackTable = evaluateOneStage(fallback, profile);
  if (!fallbackTable)
    FAIL() << llvm::toString(fallbackTable.takeError());
  ASSERT_EQ(fallbackTable->stages.front().implementations.size(), 3u);
  EXPECT_EQ(fallbackTable->stages.front().implementations[2].formulaEvidence,
            "uncalibrated_superblock_fallback+unclassified_scalar_issue_floor");
}

TEST(SimdSimtCostModelTest, PrefixScanUsesShapeAwareIssueAndDependencyBounds) {
  LogicalStage stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.features.hasReduction = true;
  stage.features.hasPrefixScan = true;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 0.0;
  stage.workload.shuffleLaneSteps = 0.0;
  stage.workload.prefixScanCounts[{32, 8}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.prefixScanStepLatencyCycles = 2.0;
  profile.simt.prefixScanStepLatencyCycles = 10.0;
  auto table = evaluateOneStage(std::move(stage), profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());

  const auto &implementations = table->stages.front().implementations;
  ASSERT_EQ(implementations.size(), 2u);
  EXPECT_EQ(implementations[0].implementation.mode, StageMode::SIMD);
  EXPECT_EQ(implementations[1].implementation.mode, StageMode::SIMT);
  EXPECT_DOUBLE_EQ(implementations[0].resources.prefixScanCriticalPath, 62.0);
  EXPECT_DOUBLE_EQ(implementations[1].resources.prefixScanCriticalPath, 50.0);
  EXPECT_DOUBLE_EQ(implementations[0].resources.shuffle, 0.0);
  EXPECT_DOUBLE_EQ(implementations[1].resources.shuffle, 40.0);
  EXPECT_DOUBLE_EQ(implementations[0].resources.prefixScanIssue, 31.0 / 4.0);
  // R=32 has five levels and S=8 independent sequences.  Each of the 40
  // warp-steps issues one shuffle, one predicate and one add; short active
  // lane sets from different steps must not be combined before rounding.
  EXPECT_DOUBLE_EQ(implementations[1].resources.prefixScanIssue,
                   3.0 * 5.0 * 8.0 / 4.0);
}

TEST(SimdSimtCostModelTest, PrefixScanIssuePreservesEveryWarpStep) {
  LogicalStage stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.features.hasReduction = true;
  stage.features.hasPrefixScan = true;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 0.0;
  stage.workload.shuffleLaneSteps = 0.0;
  stage.workload.prefixScanCounts[{4, 8}] = 1.0;

  auto table = evaluateOneStage(std::move(stage));
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &simt = table->stages.front().implementations[1];
  // ceil(log2(4))=2, so 8 sequences have 16 warp-steps.  The issue workload
  // is exactly 16 * (shuffle + predicate + add), independent of active-lane
  // density at the two levels.
  EXPECT_DOUBLE_EQ(simt.resources.prefixScanIssue, 16.0 * 3.0 / 4.0);
}

TEST(SimdSimtCostModelTest, ScalarIssueIsNotPackedAsVectorLanes) {
  for (double scalars : {0.0, 1.0, 33.0}) {
    auto stage = logicalStage("scalar_issue", StageCostModelKind::ScalarIssue);
    stage.workload.operationCounts.clear();
    stage.workload.operationCounts["f32.add"][64] = 1;
    stage.workload.scalarOperations = scalars;
    stage.workload.issueElements = scalars + 64.0;
    auto table = evaluateOneStage(std::move(stage));
    if (!table)
      FAIL() << llvm::toString(table.takeError());
    const auto &costs = table->stages.front().implementations;
    ASSERT_EQ(costs.size(), 2u);
    // 64 element-wise operations represent one SIMD or two SIMT warp
    // instructions. The scalar instruction count is unchanged in both modes.
    EXPECT_DOUBLE_EQ(costs[0].resources.issue, (scalars + 1.0) / 4.0);
    EXPECT_DOUBLE_EQ(costs[1].resources.issue, (scalars + 2.0) / 4.0);
  }
}

TEST(SimdSimtCostModelTest, VectorIssueUsesOperationThroughputAndVectorWidth) {
  for (int64_t elements : {1, 16, 64, 65, 128, 256}) {
    auto stage = logicalStage("vector", StageCostModelKind::VectorIssue);
    stage.workload.paysKernelSetup = false;
    stage.workload.operationCounts.clear();
    stage.workload.operationCounts["f32.sub"][elements] = 3.0;
    stage.workload.issueElements = 3.0 * elements;
    auto profile = hardwareProfile();
    profile.simd.operationRates["f32.sub"] = {2.0, 1.0};
    profile.simt.operationRates["f32.sub"] = {32.0, 1.0};
    auto table = evaluateOneStage(stage, profile);
    if (!table)
      FAIL() << llvm::toString(table.takeError());
    EXPECT_EQ(table->stages.front().model, "vector_issue");
    const auto &costs = table->stages.front().implementations;
    ASSERT_EQ(costs.size(), 2u);
    const double simdInstructions = 3.0 * ((elements + 63) / 64);
    EXPECT_DOUBLE_EQ(costs[0].resources.compute, simdInstructions / 2.0);
    EXPECT_DOUBLE_EQ(costs[0].resources.issue, simdInstructions / 4.0);
    EXPECT_DOUBLE_EQ(costs[0].resources.scalar, 0.0);
    EXPECT_DOUBLE_EQ(costs[0].totalCycles, simdInstructions / 2.0);
    EXPECT_DOUBLE_EQ(costs[1].resources.compute, 3.0 * elements / 32.0);
    EXPECT_NE(costs[0].formulaEvidence.find("elementwise_workload"),
              std::string::npos);
  }
}

TEST(SimdSimtCostModelTest, UnprofiledVectorExecutionIsExplicitInReport) {
  auto stage = logicalStage("vector", StageCostModelKind::VectorIssue);
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["generic.issue"][64] = 1.0;
  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  for (const auto &cost : table->stages.front().implementations)
    EXPECT_NE(cost.formulaEvidence.find("unprofiled_vector_execution"),
              std::string::npos);
}

TEST(SimdSimtCostModelTest, IssueIsAStageFloorNotAnAdditiveResource) {
  LogicalStage stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["generic.issue"][400] = 1.0;
  stage.workload.scalarOperations = 4.0;
  stage.workload.issueElements = 400.0;
  stage.workload.prefixScanCounts[{2, 1}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.issueOperationsPerCycle = 1.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  ASSERT_GT(cost.resources.issue,
            cost.resources.scalar + cost.resources.compute +
                cost.resources.predicate + cost.resources.shuffle);
  EXPECT_DOUBLE_EQ(cost.totalCycles, cost.resources.issue);
}

TEST(SimdSimtCostModelTest, IssueCountingPreservesOperationBoundaries) {
  LogicalStage stage =
      logicalStage("short_ops", StageCostModelKind::ScalarIssue);
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["generic.issue"][1] = 2.0;
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 2.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.issueOperationsPerCycle = 1.0;
  profile.simt.issueOperationsPerCycle = 1.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  ASSERT_EQ(table->stages.front().implementations.size(), 2u);
  EXPECT_DOUBLE_EQ(table->stages.front().implementations[0].resources.issue,
                   2.0);
  EXPECT_DOUBLE_EQ(table->stages.front().implementations[1].resources.issue,
                   2.0);
}

TEST(SimdSimtCostModelTest, ControlInstructionsContributeToIssueFloor) {
  LogicalStage stage =
      logicalStage("control", StageCostModelKind::ScalarControl);
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loopBackedges = 3.0;
  stage.workload.conditionalBranches = 2.0;
  stage.workload.synchronizations = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.issueOperationsPerCycle = 1.0;
  profile.simt.issueOperationsPerCycle = 1.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_DOUBLE_EQ(table->stages.front().implementations[0].resources.issue,
                   6.0);
  EXPECT_DOUBLE_EQ(table->stages.front().implementations[1].resources.issue,
                   6.0);
}

TEST(SimdSimtCostModelTest, LoopCarriedRecurrenceAppliesScanDependencyFactor) {
  auto buildStage = [] {
    LogicalStage stage = logicalStage(
        "recurrence_scan", StageCostModelKind::LoopCarriedRecurrence,
        StageScheduleKind::LoopCarriedSerial, /*iterations=*/4);
    stage.features.hasLoop = true;
    stage.features.hasLoopCarriedDataDependency = true;
    stage.features.hasPrefixScan = true;
    stage.workload.operationElements.clear();
    stage.workload.scalarOperations = 0.0;
    stage.workload.issueElements = 64.0;
    // Mixed shuffle pool: 320 scan-class steps (tt.scan) + 320 reduce-class
    // steps (tt.reduce).
    stage.workload.shuffleLaneSteps = 640.0;
    stage.workload.scanShuffleLaneSteps = 320.0;
    return stage;
  };

  // Identity factors keep the legacy recurrence scoring.
  HardwareProfile baselineProfile = hardwareProfile();
  baselineProfile.simd.prefixScanDependencyFactor = 1.0;
  baselineProfile.simt.prefixScanDependencyFactor = 1.0;
  auto baseline = evaluateOneStage(buildStage(), baselineProfile);
  if (!baseline)
    FAIL() << llvm::toString(baseline.takeError());
  ASSERT_EQ(baseline->stages.front().implementations.size(), 2u);

  // Scan factors: SIMD 2.5 / SIMT 1.0 (production profile shape).
  HardwareProfile scanProfile = baselineProfile;
  scanProfile.simd.prefixScanDependencyFactor = 2.5;
  auto scaled = evaluateOneStage(buildStage(), scanProfile);
  if (!scaled)
    FAIL() << llvm::toString(scaled.takeError());
  ASSERT_EQ(scaled->stages.front().implementations.size(), 2u);

  // Only the scan-class half is scaled: SIMD scan-shuffle cycles grow from
  // 320/32 = 10 to 10 * 2.5 = 25, so the stage total grows by 4 iterations *
  // (25 - 10) = 60 cycles.  The reduce-class half keeps the ideal rate and
  // SIMT keeps the identity factor, so both must not move.
  EXPECT_DOUBLE_EQ(scaled->stages.front().implementations[0].totalCycles,
                   baseline->stages.front().implementations[0].totalCycles +
                       60.0);
  EXPECT_DOUBLE_EQ(scaled->stages.front().implementations[1].totalCycles,
                   baseline->stages.front().implementations[1].totalCycles);
  EXPECT_GT(scaled->stages.front().implementations[0].totalCycles,
            scaled->stages.front().implementations[1].totalCycles);

  // A recurrence whose shuffle pool is purely reduce-class must not consume
  // the scan factor even though the scan feature flag is set.
  LogicalStage reduceOnlyStage = buildStage();
  reduceOnlyStage.workload.scanShuffleLaneSteps = 0.0;
  auto reduceOnlyScaled =
      evaluateOneStage(std::move(reduceOnlyStage), scanProfile);
  if (!reduceOnlyScaled)
    FAIL() << llvm::toString(reduceOnlyScaled.takeError());
  EXPECT_DOUBLE_EQ(
      reduceOnlyScaled->stages.front().implementations[0].totalCycles,
      baseline->stages.front().implementations[0].totalCycles);

  // A recurrence without a prefix scan must not consume the scan factor.
  LogicalStage plainStage = buildStage();
  plainStage.features.hasPrefixScan = false;
  auto plainScaled = evaluateOneStage(std::move(plainStage), scanProfile);
  if (!plainScaled)
    FAIL() << llvm::toString(plainScaled.takeError());
  EXPECT_DOUBLE_EQ(plainScaled->stages.front().implementations[0].totalCycles,
                   baseline->stages.front().implementations[0].totalCycles);
}

TEST(SimdSimtCostModelTest, IndependentLoopUsesSimdRooflineAndSerialSimtCost) {
  LogicalStage stage =
      logicalStage("independent", StageCostModelKind::IndependentPipelinedLoop,
                   StageScheduleKind::IndependentPipelined, 4);
  stage.features.hasLoop = true;
  stage.features.hasPointerInduction = true;

  stage.workload.loadBytes = 640.0;
  stage.workload.storeBytes = 160.0;
  stage.workload.loadWarpInstructions = 20.0;
  stage.workload.storeWarpInstructions = 10.0;
  stage.workload.dotOperations = 1.0;
  stage.workload.dotFlops = 512.0;
  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_TRUE(stage.features.permitsSimdRoofline());
  EXPECT_LT(table->stages[0].implementations[0].totalCycles,
            table->stages[0].implementations[1].totalCycles);
}

TEST(SimdSimtCostModelTest, SimdMemoryChargesStartupForEveryLogicalTransfer) {
  LogicalStage stage =
      logicalStage("memory", StageCostModelKind::ContinuousTileMemory);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loadBytes = 640.0;
  stage.workload.storeBytes = 160.0;
  stage.workload.loadOperations = 2.0;
  stage.workload.storeOperations = 3.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.loadOperationSetupCycles = 5.0;
  profile.simd.storeOperationSetupCycles = 7.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const StageImplementationCost &simd = table->stages[0].implementations[0];
  EXPECT_DOUBLE_EQ(simd.resources.load, 2.0 * 5.0 + 640.0 / 32.0);
  EXPECT_DOUBLE_EQ(simd.resources.store, 3.0 * 7.0 + 160.0 / 16.0);
}

TEST(SimdSimtCostModelTest,
     ContinuousMemoryCurveModelsPhysicalGroupThenRouteAppliesWavesOnce) {
  LogicalStage stage =
      logicalStage("continuous_load", StageCostModelKind::ContinuousTileMemory);
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loadOperations = 4.0;
  stage.workload.loadBytes = 4096.0;
  stage.workload.loadWarpInstructions = 32.0;
  stage.workload.continuousLoadCounts[{/*bytes=*/1024, /*elements=*/256}] = 4.0;
  stage.legalSimtFactors = {1, 2, 4};

  HardwareProfile profile = hardwareProfile();
  profile.simd.continuousLoadGroupCyclesByFactorAndBytes = {
      {1, {{1024, 20.0}}}};
  profile.simd.continuousLoadGroupStartupCyclesByFactorAndBytes = {
      {1, {{1024, 5.0}}}};
  profile.simt.continuousLoadGroupCyclesByFactorAndBytes = {
      {1, {{1024, 100.0}}}, {2, {{1024, 130.0}}}, {4, {{1024, 190.0}}}};
  profile.simt.continuousMemoryGroupBaseWarpCount = 1;
  profile.simt.continuousLoadGroupStartupCyclesByFactorAndBytes = {
      {1, {{1024, 10.0}}}, {2, {{1024, 20.0}}}, {4, {{1024, 30.0}}}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 4u);
  EXPECT_DOUBLE_EQ(costs[0].resources.continuousLoadGroup, 85.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.continuousLoadGroup, 410.0);
  EXPECT_DOUBLE_EQ(costs[2].resources.continuousLoadGroup, 540.0);
  EXPECT_DOUBLE_EQ(costs[3].resources.continuousLoadGroup, 790.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.load, 0.0);
  EXPECT_DOUBLE_EQ(costs[3].resources.load, 0.0);

  table->logicalProgramCountHint = 224;
  table->physicalCoreCountHint = 56;
  auto routes = solveStageRoutes(*table, profile.scopeHandoff);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_EQ(routes->allSimt.routeSuperblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 340.0);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 790.0);

  // The current curve is FP32-specific. The same byte count with twice as
  // many logical warp instructions is not silently treated as FP32.
  stage.legalSimtFactors = {1};
  stage.workload.continuousLoadCounts.clear();
  stage.workload.continuousLoadCounts[{1024, 512}] = 4.0;
  stage.workload.loadWarpInstructions = 64.0;
  auto fp16 = evaluateOneStage(stage, profile);
  if (!fp16)
    FAIL() << llvm::toString(fp16.takeError());
  EXPECT_DOUBLE_EQ(fp16->stages.front()
                       .implementations.front()
                       .resources.continuousLoadGroup,
                   0.0);
  EXPECT_GT(fp16->stages.front().implementations.front().resources.load, 0.0);

  // A one-point profile is exact-shape evidence, not permission to invent a
  // bandwidth by extrapolating to an unmeasured working size.
  stage.workload.continuousLoadCounts.clear();
  stage.workload.continuousLoadCounts[{2048, 512}] = 4.0;
  stage.workload.loadBytes = 8192.0;
  stage.workload.loadWarpInstructions = 64.0;
  auto outsideMeasuredRange = evaluateOneStage(stage, profile);
  if (!outsideMeasuredRange)
    FAIL() << llvm::toString(outsideMeasuredRange.takeError());
  EXPECT_DOUBLE_EQ(outsideMeasuredRange->stages.front()
                       .implementations.front()
                       .resources.continuousLoadGroup,
                   0.0);
  EXPECT_GT(outsideMeasuredRange->stages.front()
                .implementations.front()
                .resources.load,
            0.0);
}

TEST(SimdSimtCostModelTest, PartialMemoryCurveCannotHideUncoveredTransfers) {
  auto stage =
      logicalStage("partial_memory", StageCostModelKind::ContinuousTileMemory);
  stage.legalSimtFactors = {1, 2};
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.loadOperations = stage.workload.storeOperations = 2;
  stage.workload.loadBytes = stage.workload.storeBytes = 1536;
  stage.workload.loadWarpInstructions = stage.workload.storeWarpInstructions =
      12;
  stage.workload.continuousLoadCounts = {{{1024, 256}, 1}, {{512, 128}, 1}};
  stage.workload.continuousStoreCounts = stage.workload.continuousLoadCounts;

  auto profile = hardwareProfile();
  profile.simd.loadOperationSetupCycles = 5;
  profile.simd.storeOperationSetupCycles = 7;
  profile.simd.continuousLoadGroupCyclesByFactorAndBytes = {{1, {{1024, 20}}}};
  profile.simd.continuousStoreGroupCyclesByFactorAndBytes = {{1, {{1024, 30}}}};
  profile.simt.continuousMemoryGroupBaseWarpCount = 1;
  profile.simt.continuousLoadGroupCyclesByFactorAndBytes = {{1, {{1024, 100}}},
                                                            {2, {{1024, 140}}}};
  profile.simt.continuousStoreGroupCyclesByFactorAndBytes = {
      {1, {{1024, 120}}}, {2, {{1024, 160}}}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_DOUBLE_EQ(costs[0].resources.load, 5 + 512.0 / 32);
  EXPECT_DOUBLE_EQ(costs[0].resources.store, 7 + 512.0 / 16);
  EXPECT_DOUBLE_EQ(costs[0].totalCycles, 20 + 30 + 21 + 39);
  EXPECT_DOUBLE_EQ(costs[1].resources.load, 4);
  EXPECT_DOUBLE_EQ(costs[1].resources.store, 4);
  EXPECT_DOUBLE_EQ(costs[1].totalCycles, 100 + 120 + 4 + 4);
  // The F2 envelope is charged once; the uncovered 512-byte transfers
  // happen in both logical programs, not just one.
  EXPECT_DOUBLE_EQ(costs[2].totalCycles, 140 + 160 + 2 * (4 + 4));
}

TEST(SimdSimtCostModelTest, MemoryEnvelopeCannotEraseCompositeStageTraffic) {
  auto baselineProfile = hardwareProfile();
  baselineProfile.simd.loadOperationSetupCycles = 5;
  baselineProfile.simd.storeOperationSetupCycles = 7;
  auto withCurve = baselineProfile;
  withCurve.simd.continuousLoadGroupCyclesByFactorAndBytes = {
      {1, {{1024, 20}}}};
  withCurve.simd.continuousStoreGroupCyclesByFactorAndBytes = {
      {1, {{1024, 30}}}};
  withCurve.simt.continuousMemoryGroupBaseWarpCount = 1;
  withCurve.simt.continuousLoadGroupCyclesByFactorAndBytes = {
      {1, {{1024, 100}}}};
  withCurve.simt.continuousStoreGroupCyclesByFactorAndBytes = {
      {1, {{1024, 120}}}};
  for (auto kind :
       {StageCostModelKind::LoopCarriedRecurrence,
        StageCostModelKind::RowwiseReduction, StageCostModelKind::PrefixScan,
        StageCostModelKind::IndependentPipelinedLoop,
        StageCostModelKind::TinyCubeRoofline}) {
    SCOPED_TRACE(mlir::ascend::stringifyStageCostModel(kind).str());
    auto stage = logicalStage("composite", kind);
    stage.workload.paysKernelSetup = false;
    stage.workload.loadOperations = stage.workload.storeOperations = 1;
    stage.workload.loadBytes = stage.workload.storeBytes = 1024;
    stage.workload.loadWarpInstructions = stage.workload.storeWarpInstructions =
        8;
    stage.workload.continuousLoadCounts = {{{1024, 256}, 1}};
    stage.workload.continuousStoreCounts = stage.workload.continuousLoadCounts;
    auto baseline = evaluateOneStage(stage, baselineProfile);
    auto candidate = evaluateOneStage(stage, withCurve);
    if (!baseline)
      FAIL() << llvm::toString(baseline.takeError());
    if (!candidate)
      FAIL() << llvm::toString(candidate.takeError());
    const auto &costs = candidate->stages.front().implementations;
    ASSERT_EQ(costs.size(), 2u);
    for (unsigned i = 0; i < costs.size(); ++i) {
      EXPECT_DOUBLE_EQ(costs[i].resources.continuousLoadGroup, 0);
      EXPECT_DOUBLE_EQ(costs[i].resources.continuousStoreGroup, 0);
      EXPECT_GT(costs[i].resources.load, 0);
      EXPECT_GT(costs[i].resources.store, 0);
      EXPECT_DOUBLE_EQ(costs[i].totalCycles,
                       baseline->stages.front().implementations[i].totalCycles);
    }
  }
}

TEST(SimdSimtCostModelTest, SimdCubeStageUsesCvPipelineCriticalPath) {
  LogicalStage stage =
      logicalStage("cube", StageCostModelKind::TinyCubeRoofline,
                   StageScheduleKind::IndependentPipelined, 4);
  stage.features.hasDot = true;
  stage.features.hasContiguousMemory = true;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 16.0;
  stage.workload.loadBytes = 2048.0;
  stage.workload.storeBytes = 512.0;
  stage.workload.dotOperations = 1.0;
  stage.workload.dotFlops = 8192.0;
  stage.workload.issueElements = 128.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());

  const StageImplementationCost &simd = table->stages[0].implementations[0];
  ASSERT_EQ(simd.implementation.mode, StageMode::SIMD);
  EXPECT_DOUBLE_EQ(simd.resources.setup, 10.0);
  EXPECT_DOUBLE_EQ(simd.resources.load, 64.0);
  EXPECT_DOUBLE_EQ(simd.resources.store, 32.0);
  EXPECT_DOUBLE_EQ(simd.resources.dot, 136.0);
  // SIMD/Cube CV resources overlap inside one Stage. Unclassified scalar
  // operations contribute to issue, not a second invented execution latency.
  EXPECT_DOUBLE_EQ(simd.resources.scalar, 0.0);
  EXPECT_DOUBLE_EQ(simd.resources.issue, 16.0 / 4.0);
  EXPECT_DOUBLE_EQ(simd.totalCycles, 10.0 + 4.0 * 136.0);
}

TEST(SimdSimtCostModelTest, DotStartupIsChargedPerDynamicMatrixInstance) {
  LogicalStage stage =
      logicalStage("dot_chain", StageCostModelKind::TinyCubeRoofline);
  stage.features.hasDot = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.dotOperations = 2.0;
  stage.workload.dotFlops = 8192.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const StageImplementationCost &simd = table->stages[0].implementations[0];
  EXPECT_DOUBLE_EQ(simd.resources.dot, 2.0 * 8.0 + 8192.0 / 64.0);
}

TEST(SimdSimtCostModelTest, TrueLoopCarriedDependencyDisablesSimdRoofline) {
  LogicalStage stage =
      logicalStage("dependent", StageCostModelKind::IndependentPipelinedLoop,
                   StageScheduleKind::IndependentPipelined, 4);
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;

  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.workload.loadBytes = 640.0;
  stage.workload.storeBytes = 160.0;
  stage.workload.dotOperations = 1.0;
  stage.workload.dotFlops = 512.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_FALSE(stage.features.permitsSimdRoofline());
  EXPECT_GT(table->stages[0].implementations[0].totalCycles, 0.0);
}

TEST(SimdSimtCostModelTest, ControlFlowUsesIssueFloorWithoutAdditiveRates) {
  LogicalStage stage =
      logicalStage("control", StageCostModelKind::ScalarControl,
                   StageScheduleKind::StraightLine, 2);
  stage.simdLegal = false;
  stage.features.hasLoop = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.conditionalBranches = 3;
  stage.workload.divergentBranches = 2;
  stage.workload.loopBackedges = 1;
  stage.workload.synchronizations = 1;
  stage.features.activeLaneRatio = 0.5;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages[0].implementations[0];
  // Divergent branches are a subset of conditional branches, not additional
  // issue instructions. Without an identifiable composite latency curve the
  // five emitted control/sync instructions contribute only to the issue floor.
  EXPECT_DOUBLE_EQ(cost.resources.issue, 1.25);
  EXPECT_DOUBLE_EQ(cost.totalCycles, 2.5);
  EXPECT_EQ(cost.formulaEvidence,
            "analytical_resource_formula+control_issue_floor_only");
}

TEST(SimdSimtCostModelTest, RecurrenceAccumulatesCriticalPathAndTraffic) {
  LogicalStage stage =
      logicalStage("recurrence", StageCostModelKind::LoopCarriedRecurrence,
                   StageScheduleKind::LoopCarriedSerial, 4);
  stage.simdLegal = false;
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;

  stage.workload.loadWarpInstructions = 10.0;
  stage.workload.storeWarpInstructions = 5.0;

  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_GT(table->stages[0].implementations[0].totalCycles, 100.0);
  EXPECT_GT(table->stages[0].implementations[0].resources.criticalPath, 0.0);
}

TEST(SimdSimtCostModelTest, RecurrenceDoesNotInventCostFromStageLiveOutBytes) {
  LogicalStage baseline = logicalStage(
      "baseline_recurrence", StageCostModelKind::LoopCarriedRecurrence,
      StageScheduleKind::LoopCarriedSerial);
  baseline.features.hasLoop = true;
  baseline.features.hasLoopCarriedDataDependency = true;
  LogicalStage withState = baseline;
  withState.id = "stateful_recurrence";
  withState.liveOutBytes = 800;

  auto baselineTable = evaluateOneStage(baseline);
  auto stateTable = evaluateOneStage(withState);
  if (!baselineTable)
    FAIL() << llvm::toString(baselineTable.takeError());
  if (!stateTable)
    FAIL() << llvm::toString(stateTable.takeError());
  const double baselineSimd =
      baselineTable->stages[0].implementations[0].totalCycles;
  const double stateSimd = stateTable->stages[0].implementations[0].totalCycles;
  EXPECT_DOUBLE_EQ(stateSimd, baselineSimd);
}

TEST(SimdSimtCostModelTest, RecurrenceLoopCountDoesNotShortenSerialIterations) {
  LogicalStage serial = logicalStage("serial_recurrence",
                                     StageCostModelKind::LoopCarriedRecurrence,
                                     StageScheduleKind::LoopCarriedSerial, 14);
  serial.simdLegal = false;
  serial.features.hasLoop = true;
  serial.features.hasLoopCarriedDataDependency = true;
  serial.workload.shuffleLaneSteps = 128.0;
  serial.workload.issueElements = 64.0;

  LogicalStage grouped = serial;
  grouped.id = "grouped_recurrence";
  grouped.features.parallelRecurrenceGroupCount = 4;
  HardwareProfile profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;

  LogicalStage twiceAsManyIterations = grouped;
  twiceAsManyIterations.iterationCount = 28;
  auto doubledTable = evaluateOneStage(twiceAsManyIterations, profile);
  if (!doubledTable)
    FAIL() << llvm::toString(doubledTable.takeError());
  auto serialTable = evaluateOneStage(std::move(serial), profile);
  auto groupedTable = evaluateOneStage(std::move(grouped), profile);
  if (!serialTable)
    FAIL() << llvm::toString(serialTable.takeError());
  if (!groupedTable)
    FAIL() << llvm::toString(groupedTable.takeError());
  const double serialCycles =
      serialTable->stages[0].implementations[0].totalCycles;
  const double groupedCycles =
      groupedTable->stages[0].implementations[0].totalCycles;
  EXPECT_DOUBLE_EQ(groupedCycles, serialCycles);
  const auto &resources = groupedTable->stages[0].implementations[0].resources;
  EXPECT_GE(groupedCycles, resources.setup + 14.0 * resources.issue);
  EXPECT_DOUBLE_EQ(doubledTable->stages[0].implementations[0].totalCycles -
                       resources.setup,
                   2.0 * (groupedCycles - resources.setup));
}

TEST(SimdSimtCostModelTest,
     RecurrenceCurveModelsPhysicalGroupThenRouteAppliesWavesOnce) {
  LogicalStage stage = logicalStage("stateful_recurrence",
                                    StageCostModelKind::LoopCarriedRecurrence,
                                    StageScheduleKind::LoopCarriedSerial, 16);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.features.hasReduction = true;
  stage.features.parallelRecurrenceGroupCount = 4;
  stage.workload.operationCounts.clear();
  stage.workload.shuffleLaneSteps = 1024.0;
  stage.workload.issueElements = 0.0;

  HardwareProfile profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.recurrenceGroupBaseWarpCount = 4;
  profile.simt.recurrenceReductionGroupCyclesByFactorAndLaneSteps = {
      {1, {{1024, 100.0}, {4096, 400.0}}},
      {2, {{1024, 130.0}, {4096, 520.0}}},
      {4, {{1024, 190.0}, {4096, 760.0}}}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_DOUBLE_EQ(costs[0].resources.recurrenceGroupCriticalPath, 100.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.recurrenceGroupCriticalPath, 130.0);
  EXPECT_DOUBLE_EQ(costs[2].resources.recurrenceGroupCriticalPath, 190.0);
  EXPECT_DOUBLE_EQ(costs[0].totalCycles, 1610.0);
  EXPECT_DOUBLE_EQ(costs[1].totalCycles, 2090.0);
  EXPECT_DOUBLE_EQ(costs[2].totalCycles, 3050.0);

  table->logicalProgramCountHint = 224;
  table->physicalCoreCountHint = 56;
  auto routes = solveStageRoutes(*table, profile.scopeHandoff);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_EQ(routes->allSimt.routeSuperblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 3050.0);

  stage.legalSimtFactors = {1};
  stage.workload.shuffleLaneSteps = 2048.0;
  auto interpolated = evaluateOneStage(stage, profile);
  if (!interpolated)
    FAIL() << llvm::toString(interpolated.takeError());
  EXPECT_DOUBLE_EQ(interpolated->stages.front()
                       .implementations.front()
                       .resources.recurrenceGroupCriticalPath,
                   200.0);
}

TEST(SimdSimtCostModelTest,
     MixedLocalStageUsesCandidatePhysicalRecurrenceGroupCost) {
  LogicalStage stage = logicalStage(
      "mixed_recurrence", StageCostModelKind::LoopCarriedRecurrence,
      StageScheduleKind::LoopCarriedSerial, /*iterations=*/16);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1};
  stage.localSimtMaterializable = true;
  stage.localSimtFactors = {1, 2, 4};
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.features.hasReduction = true;
  stage.workload.operationCounts.clear();
  stage.workload.shuffleLaneSteps = 1024.0;
  stage.workload.issueElements = 0.0;

  HardwareProfile profile = hardwareProfile();
  profile.simt.recurrenceGroupBaseWarpCount = 1;
  profile.simt.recurrenceReductionGroupCyclesByFactorAndLaneSteps = {
      {1, {{1024, 100.0}}}, {2, {{1024, 130.0}}}, {4, {{1024, 190.0}}}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 4u);
  EXPECT_FALSE(costs[0].implementation.localScope);
  EXPECT_TRUE(costs[1].implementation.localScope);
  EXPECT_TRUE(costs[2].implementation.localScope);
  EXPECT_TRUE(costs[3].implementation.localScope);
  EXPECT_LT(costs[1].totalCycles, costs[2].totalCycles);
  EXPECT_LT(costs[2].totalCycles, costs[3].totalCycles);
  EXPECT_GE(costs[3].totalCycles,
            costs[3].resources.setup + 16.0 * costs[3].resources.issue);
}

TEST(SimdSimtCostModelTest, IndirectMemoryRejectsUnclassifiedWorkload) {
  LogicalStage stage =
      logicalStage("indirect", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::PartiallyDependent);
  stage.features.hasIndirectMemory = true;
  stage.workload.loadBytes = 1024.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.indirectLoadBytes = 1024.0;
  stage.workload.indirectLoadTransactions = 8.0;

  auto table = evaluateOneStage(stage, hardwareProfile());
  ASSERT_FALSE(table);
  EXPECT_NE(
      llvm::toString(table.takeError()).find("no measured implementation"),
      std::string::npos);
}

TEST(SimdSimtCostModelTest,
     IndirectMemoryCurveChargesEveryOperationAndInterpolatesShape) {
  LogicalStage stage =
      logicalStage("indirect", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::PartiallyDependent);
  stage.features.hasIndirectMemory = true;
  stage.workload.loadBytes = 1536.0;
  stage.workload.loadWarpInstructions = 24.0;
  stage.workload.indirectLoadCounts[{512, 256}] = 1.0;
  stage.workload.indirectLoadCounts[{1024, 512}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.indirectLoadSystemCyclesByWarpInstructions = {{8, 100.0},
                                                             {16, 220.0}};
  profile.simd.indirectLoadStartupSystemCyclesByWarpInstructions = {{8, 10.0},
                                                                    {16, 30.0}};
  profile.simt.indirectLoadSystemCyclesByWarpInstructions = {{8, 20.0},
                                                             {16, 60.0}};
  profile.simt.indirectLoadStartupSystemCyclesByWarpInstructions = {{8, 5.0},
                                                                    {16, 15.0}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 2u);
  EXPECT_DOUBLE_EQ(costs[0].resources.load, 350.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.load, 95.0);
  EXPECT_EQ(costs[0].formulaEvidence, "measured_indirect_load_curve");
  EXPECT_EQ(costs[1].formulaEvidence, "measured_indirect_load_curve");

  stage.workload.loadBytes = 768.0;
  stage.workload.loadWarpInstructions = 12.0;
  stage.workload.indirectLoadCounts = {{{768, 384}, 1.0}};
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_DOUBLE_EQ(table->stages.front().implementations[0].resources.load,
                   180.0);
  EXPECT_DOUBLE_EQ(table->stages.front().implementations[1].resources.load,
                   50.0);

  stage.workload.loadBytes = 2560.0;
  stage.workload.loadWarpInstructions = 40.0;
  stage.workload.indirectLoadCounts = {{{512, 256}, 5.0}};
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_EQ(table->stages.front().implementations[0].formulaEvidence,
            "measured_indirect_load_curve_extrapolated");

  // The measured curve is FP16-specific. The same warp count carrying twice
  // the bytes (for example FP32) must not silently reuse it.
  stage.workload.loadBytes = 1024.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.indirectLoadCounts = {{{1024, 256}, 1.0}};
  table = evaluateOneStage(stage, profile);
  ASSERT_FALSE(table);
  EXPECT_NE(
      llvm::toString(table.takeError()).find("no measured implementation"),
      std::string::npos);
}

TEST(SimdSimtCostModelTest, ShortIndirectLoadPreservesElementWidth) {
  LogicalStage stage =
      logicalStage("short_load", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::PartiallyDependent);
  stage.features.hasIndirectMemory = true;
  stage.workload.loadBytes = 4;
  stage.workload.loadWarpInstructions = 1;
  stage.workload.indirectLoadCounts = {{{4, 2}, 1}};
  auto profile = hardwareProfile();
  for (auto *mode : {&profile.simd, &profile.simt}) {
    mode->indirectLoadSystemCyclesByWarpInstructions = {{1, 20}};
    mode->indirectLoadStartupSystemCyclesByWarpInstructions = {{1, 10}};
  }
  auto costs = evaluateOneStage(stage, profile);
  if (!costs)
    FAIL() << llvm::toString(costs.takeError());
  for (const auto &cost : costs->stages.front().implementations)
    EXPECT_DOUBLE_EQ(cost.resources.load, 30);

  // Same four bytes and same single issued warp, but now one FP32 element.
  // It must not match the FP16 curve just because the old key was identical.
  stage.workload.indirectLoadCounts = {{{4, 1}, 1}};
  costs = evaluateOneStage(stage, profile);
  ASSERT_FALSE(costs);
  EXPECT_NE(
      llvm::toString(costs.takeError()).find("no measured implementation"),
      std::string::npos);
}

TEST(SimdSimtCostModelTest,
     AtomicCostDoesNotRepriceOrdinaryLoadsAsIndirectMemory) {
  LogicalStage stage = logicalStage("atomic", StageCostModelKind::AtomicMemory,
                                    StageScheduleKind::PartiallyDependent);
  stage.features.hasAtomicMemory = true;
  stage.features.hasIndirectMemory = true;
  stage.features.hasContiguousMemory = true;
  stage.workload.loadBytes = 1024.0;
  stage.workload.loadWarpInstructions = 8.0;
  mlir::ascend::AtomicWorkload atomic;
  atomic.kind = "fadd";
  atomic.dataType = "f32";
  atomic.memorySemantic = "acq_rel";
  atomic.memoryScope = "gpu";
  atomic.logicalElements = 64.0;
  atomic.logicalOperationInstances = 1.0;
  atomic.contentionUnknown = true;
  stage.workload.atomicWorkloads.push_back(atomic);

  HardwareProfile profile = hardwareProfile();
  profile.simd.atomicRates["fadd.f32"] = {8.0, 2.0, 5.0, 3.0};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &simd = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(simd.resources.load, 32.0);
  EXPECT_DOUBLE_EQ(simd.resources.store, 0.0);
  EXPECT_DOUBLE_EQ(simd.resources.atomic, 30.0);
}

TEST(SimdSimtCostModelTest,
     WorkloadPreservesAtomicSemanticsWithoutCountingAStore) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%load_pointer: tensor<64x!tt.ptr<f32>>,
                        %index_pointer: tensor<64x!tt.ptr<i64>>,
                        %base_pointer: tensor<64x!tt.ptr<f32>>,
                        %value: tensor<64xf32>) {
        %regular = "tt.load"(%load_pointer)
          : (tensor<64x!tt.ptr<f32>>) -> tensor<64xf32>
        %index = "tt.load"(%index_pointer)
          : (tensor<64x!tt.ptr<i64>>) -> tensor<64xi64>
        %address = "tt.addptr"(%base_pointer, %index)
          : (tensor<64x!tt.ptr<f32>>, tensor<64xi64>)
            -> tensor<64x!tt.ptr<f32>>
        %mask = arith.constant dense<true> : tensor<64xi1>
        %old = "tt.atomic_rmw"(%address, %value, %mask)
          {atomic_rmw_op = 5 : i32, sem = 4 : i32, scope = 1 : i32}
          : (tensor<64x!tt.ptr<f32>>, tensor<64xf32>, tensor<64xi1>)
            -> tensor<64xf32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("atomic_workload", StageCostModelKind::AtomicMemory);
  auto function = module->lookupSymbol<mlir::func::FuncOp>("kernel");
  ASSERT_TRUE(function);
  mlir::Block &body = function.getBody().front();
  for (mlir::Operation &operation : body.without_terminator())
    stage.operations.push_back(&operation);
  partition.stages.push_back(std::move(stage));

  if (llvm::Error error = StageFeatureAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error =
          mlir::ascend::StageKindClassifier().analyze(partition, 8192))
    FAIL() << llvm::toString(std::move(error));

  const LogicalStage &analyzed = partition.stages.front();
  EXPECT_TRUE(analyzed.features.hasAtomicMemory);
  EXPECT_TRUE(analyzed.features.hasIndirectMemory);
  EXPECT_TRUE(analyzed.features.hasContiguousMemory);
  EXPECT_DOUBLE_EQ(analyzed.workload.storeBytes, 0.0);
  EXPECT_DOUBLE_EQ(analyzed.workload.storeWarpInstructions, 0.0);
  ASSERT_EQ(analyzed.workload.atomicWorkloads.size(), 1u);
  const auto &atomic = analyzed.workload.atomicWorkloads.front();
  EXPECT_EQ(atomic.kind, "fadd");
  EXPECT_EQ(atomic.dataType, "f32");
  EXPECT_EQ(atomic.memorySemantic, "acq_rel");
  EXPECT_EQ(atomic.memoryScope, "gpu");
  EXPECT_DOUBLE_EQ(atomic.logicalElements, 64.0);
  EXPECT_FALSE(atomic.resultUsed);
  EXPECT_TRUE(atomic.addressDependsOnLoadedIndex);
  EXPECT_TRUE(atomic.contentionUnknown);
  EXPECT_EQ(analyzed.costModelKind, StageCostModelKind::AtomicMemory);
}

TEST(SimdSimtCostModelTest, MixedRouteRejectsUnmaterializableSimtStage) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles, bool localScope = false) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1, localScope};
    cost.totalCycles = cycles;
    return cost;
  };
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.localSimtMaterializable = false;
  head.implementations = {makeCost(StageMode::SIMD, 1.0),
                          makeCost(StageMode::SIMT, 100.0)};
  mlir::ascend::LogicalStageCost payload;
  payload.id = "unmaterializable_payload";
  payload.localSimtMaterializable = false;
  payload.implementations = {makeCost(StageMode::SIMD, 100.0),
                             makeCost(StageMode::SIMT, 1.0)};
  table.stages = {head, payload};
  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_TRUE(routes->allSimt.legal);
  EXPECT_FALSE(routes->mixed.legal);
}

TEST(SimdSimtCostModelTest,
     MixedRouteReportsCheapestConstrainedRouteWhenLocalScopeLoses) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles, bool localScope = false) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1, localScope};
    cost.totalCycles = cycles;
    return cost;
  };

  mlir::ascend::LogicalStageCost gather;
  gather.id = "indirect_tile_gather";
  gather.localSimtMaterializable = true;
  gather.implementations = {makeCost(StageMode::SIMD, 100.0),
                            makeCost(StageMode::SIMT, 130.0),
                            makeCost(StageMode::SIMT, 130.0, true)};
  mlir::ascend::LogicalStageCost dot;
  dot.id = "tiny_cube_dot";
  dot.implementations = {makeCost(StageMode::SIMD, 40.0),
                         makeCost(StageMode::SIMT, 90.0)};
  table.stages = {gather, dot};

  ScopeHandoffCost scopeHandoff;
  scopeHandoff.fixedScopeCycles = 20.0;
  auto routes = solveStageRoutes(table, scopeHandoff);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  ASSERT_EQ(routes->mixed.implementations.size(), 2u);
  EXPECT_EQ(routes->mixed.implementations[0].mode, StageMode::SIMT);
  EXPECT_TRUE(routes->mixed.implementations[0].localScope);
  EXPECT_EQ(routes->mixed.implementations[1].mode, StageMode::SIMD);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 190.0);
}

TEST(SimdSimtCostModelTest, AllSimdDoesNotPayRouteConditionalAutoBlockify) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1, false};
    cost.totalCycles = cycles;
    return cost;
  };

  mlir::ascend::LogicalStageCost dispatch;
  dispatch.id = "physical_program_dispatch";
  dispatch.model = "auto_blockify_dispatch";
  dispatch.implementations = {makeCost(StageMode::SIMD, 40.0),
                              makeCost(StageMode::SIMT, 30.0)};
  mlir::ascend::LogicalStageCost payload;
  payload.id = "payload";
  payload.model = "scalar_issue";
  payload.implementations = {makeCost(StageMode::SIMD, 100.0),
                             makeCost(StageMode::SIMT, 80.0)};
  table.stages = {dispatch, payload};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.legal);
  ASSERT_EQ(routes->allSimd.logicalStageCycles.size(), 2u);
  EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[0], 0.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[1], 100.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 100.0);
  ASSERT_TRUE(routes->allSimt.legal);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 110.0);
}

TEST(SimdSimtCostModelTest, MixedRouteChargesOneMergedCompoundScope) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles, bool localScope = false) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, 1, localScope};
    cost.totalCycles = cycles;
    return cost;
  };
  mlir::ascend::LogicalStageCost head;
  head.id = "head";
  head.implementations = {makeCost(StageMode::SIMD, 1.0),
                          makeCost(StageMode::SIMT, 100.0)};
  mlir::ascend::LogicalStageCost gather;
  gather.id = "compound_anchor_gather";
  gather.localSimtMaterializable = true;
  gather.localSimtFactors = {1};
  gather.implementations = {makeCost(StageMode::SIMD, 100.0),
                            makeCost(StageMode::SIMT, 1.0),
                            makeCost(StageMode::SIMT, 1.0, true)};
  mlir::ascend::LogicalStageCost tail = head;
  tail.id = "tail";
  table.stages = {head, gather, tail};

  ScopeHandoffCost scopeHandoff;
  scopeHandoff.fixedScopeCycles = 20.0;
  auto routes = solveStageRoutes(table, scopeHandoff);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  // All Stage anchors form one scope: 1 head + (20 scope + 1 payload) + 1 tail.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 23.0);
  ASSERT_EQ(routes->mixed.scopeHandoffCycles.size(), 3u);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[1], 20.0);
}

TEST(SimdSimtCostModelTest,
     MixedWithoutV1ReplaysSimdPeerPerLogicalProgramAndScopesTail) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 225;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;

  auto makeCost = [](StageMode mode, int64_t factor, double cycles,
                     bool localScope = false) {
    StageImplementationCost cost;
    cost.implementation = {mode, factor, localScope};
    cost.totalCycles = cycles;
    return cost;
  };

  LogicalStageCost local;
  local.id = "local_scope";
  local.localSimtMaterializable = true;
  local.localSimtFactors = {1, 2, 4};
  local.implementations = {makeCost(StageMode::SIMD, 1, 100.0),
                           makeCost(StageMode::SIMT, 1, 40.0, true),
                           makeCost(StageMode::SIMT, 2, 100.0, true),
                           makeCost(StageMode::SIMT, 4, 10.0, true)};

  LogicalStageCost peer;
  peer.id = "simd_peer";
  peer.implementations = {makeCost(StageMode::SIMD, 1, 3.0)};
  table.stages = {local, peer};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  ASSERT_EQ(routes->mixed.routeSuperblockFactor, 4);
  // Q_v=ceil(225/56)=5: one F4 scope group plus one F1 tail.  The SIMD
  // peer is outside the scope and must execute for all five logical programs,
  // not for the two whole-kernel VF iterations.
  EXPECT_DOUBLE_EQ(routes->mixed.logicalStageCycles[0], 50.0);
  EXPECT_DOUBLE_EQ(routes->mixed.logicalStageCycles[1], 15.0);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 65.0);
  EXPECT_EQ(routes->mixed.runtimeStageExecutionDomains[0], "aiv_local_scope");
  EXPECT_EQ(routes->mixed.runtimeStageExecutionDomains[1], "aiv_vector");
  EXPECT_EQ(routes->mixed.runtimeStageTailModes[0],
            "factorF_plus_factor1_tail");
}

TEST(SimdSimtCostModelTest, MixedRuntimeUsesPerCoreFullGroupsAndFactorOneTail) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  // The same logical grid has Q=10 on 56 AIV cores but Q=19 on 28 AIC cores.
  // This Mixed route selects a SIMD Cube peer, so the unsplit MIX parent owns
  // the V1 driver and uses the AIC schedule.  A local SIMT scope is a peer
  // inside that parent loop, not an independent 56-slot logical grid.
  table.logicalProgramCountHint = 512;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;

  auto makeCost = [](StageMode mode, int64_t factor, double cycles,
                     bool localScope = false, double setup = 0.0) {
    StageImplementationCost cost;
    cost.implementation = {mode, factor, localScope};
    cost.totalCycles = cycles;
    cost.resources.setup = setup;
    return cost;
  };

  LogicalStageCost dispatch;
  dispatch.id = "dispatch";
  dispatch.model = "auto_blockify_dispatch";
  dispatch.implementations = {makeCost(StageMode::SIMD, 1, 7.0, false, 5.0)};

  LogicalStageCost loop;
  loop.id = "loop";
  loop.model = "auto_blockify_loop";
  loop.implementations = {makeCost(StageMode::SIMD, 1, 1.0)};

  LogicalStageCost scope;
  scope.id = "local_scope";
  scope.model = "loop_carried_recurrence";
  scope.features.insideAutoBlockifyV1Loop = true;
  scope.localSimtMaterializable = true;
  scope.implementations = {makeCost(StageMode::SIMD, 1, 100.0),
                           makeCost(StageMode::SIMT, 1, 20.0, true),
                           makeCost(StageMode::SIMT, 2, 15.0, true),
                           makeCost(StageMode::SIMT, 4, 10.0, true)};

  LogicalStageCost simdBody;
  simdBody.id = "simd_body";
  simdBody.model = "scalar_issue";
  simdBody.features.insideAutoBlockifyV1Loop = true;
  simdBody.features.hasDot = true;
  simdBody.implementations = {makeCost(StageMode::SIMD, 1, 3.0)};
  table.stages = {dispatch, loop, scope, simdBody};

  ScopeHandoffCost scopeHandoff;
  auto routes = solveStageRoutes(table, scopeHandoff);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());

  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_EQ(routes->mixed.runtimePhysicalProgramCount, 28);
  EXPECT_EQ(routes->mixed.runtimeSchedulingSlotCount, 28);
  EXPECT_EQ(routes->mixed.runtimeLogicalProgramsPerSchedulingSlot, 19);
  EXPECT_EQ(routes->mixed.runtimeFullGroupCount, 4);
  EXPECT_EQ(routes->mixed.runtimeTailProgramCount, 3);
  EXPECT_EQ(routes->mixed.runtimeLoopIterationCount, 7);
  ASSERT_EQ(routes->mixed.logicalStageCycles.size(), 4u);
  EXPECT_DOUBLE_EQ(routes->mixed.logicalStageCycles[0], 7.0);
  EXPECT_DOUBLE_EQ(routes->mixed.logicalStageCycles[1], 7.0);
  // The local SIMT scope follows the AIC parent Q_a=ceil(512/28)=19:
  // four F4 groups plus a three-program F1 tail (40+60), while the SIMD peer
  // is replayed for all 19 logical programs (4*12 + 3*3).
  EXPECT_DOUBLE_EQ(routes->mixed.logicalStageCycles[2], 100.0);
  EXPECT_DOUBLE_EQ(routes->mixed.logicalStageCycles[3], 57.0);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 7.0 + 7.0 + 100.0 + 57.0);
  ASSERT_EQ(routes->mixed.runtimeStageLogicalProgramsPerSchedulingSlot.size(),
            4u);
  EXPECT_EQ(routes->mixed.runtimeStageLogicalProgramsPerSchedulingSlot[2], 19);
  EXPECT_EQ(routes->mixed.runtimeFullGroupCount, 4);
  EXPECT_EQ(routes->mixed.runtimeStageFullGroupCounts[2], 4);
  EXPECT_EQ(routes->mixed.runtimeStageTailProgramCounts[2], 3);
  ASSERT_EQ(routes->mixed.runtimeStageExecutionDomains.size(), 4u);
  ASSERT_EQ(routes->mixed.runtimeStageTailModes.size(), 4u);
  EXPECT_EQ(routes->mixed.runtimeStageExecutionDomains[0], "aic_parent");
  EXPECT_EQ(routes->mixed.runtimeStageExecutionDomains[1], "aic_parent");
  EXPECT_EQ(routes->mixed.runtimeStageExecutionDomains[2], "aiv_local_scope");
  EXPECT_EQ(routes->mixed.runtimeStageExecutionDomains[3], "aic_parent");
  EXPECT_EQ(routes->mixed.runtimeStageTailModes[2],
            "factorF_plus_factor1_tail");
  EXPECT_EQ(routes->mixed.runtimeStageTailModes[3],
            "factorF_body_plus_factor1_tail");

  ASSERT_TRUE(routes->allSimd.legal);
  EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[0], 5.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[1], 0.0);
  EXPECT_EQ(routes->allSimd.runtimePhysicalProgramCount, 28);
  EXPECT_EQ(routes->allSimd.runtimeSchedulingSlotCount, 28);
  EXPECT_EQ(routes->allSimd.runtimeLogicalProgramsPerSchedulingSlot, 19);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 1962.0);

  // Without Cube, all-SIMD and Mixed return to the vector-core launch limit.
  table.stages[3].features.hasDot = false;
  auto vectorRoutes = solveStageRoutes(table, scopeHandoff);
  if (!vectorRoutes)
    FAIL() << llvm::toString(vectorRoutes.takeError());
  ASSERT_TRUE(vectorRoutes->mixed.legal);
  EXPECT_EQ(vectorRoutes->mixed.runtimePhysicalProgramCount, 56);
  EXPECT_EQ(vectorRoutes->mixed.runtimeSchedulingSlotCount, 56);
  EXPECT_EQ(vectorRoutes->mixed.runtimeLogicalProgramsPerSchedulingSlot, 10);
  EXPECT_EQ(vectorRoutes->allSimd.runtimeLogicalProgramsPerSchedulingSlot, 10);
  EXPECT_DOUBLE_EQ(vectorRoutes->allSimd.totalCycles, 1035.0);
}

TEST(SimdSimtCostModelTest, MixedLaunchUsesAivWhenDotStageSelectsSIMT) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 225;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;

  auto makeCost = [](StageMode mode, double cycles, bool localScope = false) {
    StageImplementationCost cost;
    cost.implementation = {mode, 1, localScope};
    cost.totalCycles = cycles;
    return cost;
  };
  LogicalStageCost dot;
  dot.id = "simt_dot";
  dot.features.hasDot = true;
  dot.localSimtMaterializable = true;
  dot.implementations = {makeCost(StageMode::SIMD, 100.0),
                         makeCost(StageMode::SIMT, 1.0, true)};
  LogicalStageCost vector;
  vector.id = "simd_vector";
  vector.implementations = {makeCost(StageMode::SIMD, 1.0),
                            makeCost(StageMode::SIMT, 100.0, true)};
  table.stages = {dot, vector};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.implementations[0].mode, StageMode::SIMT);
  EXPECT_EQ(routes->mixed.implementations[1].mode, StageMode::SIMD);
  EXPECT_EQ(routes->mixed.runtimePhysicalProgramCount, 56);
  EXPECT_EQ(routes->mixed.runtimeSchedulingSlotCount, 56);
  ASSERT_TRUE(routes->allSimd.legal);
  EXPECT_EQ(routes->allSimd.runtimePhysicalProgramCount, 28);
  EXPECT_EQ(routes->allSimd.runtimeSchedulingSlotCount, 28);
}

TEST(SimdSimtCostModelTest, SuperBlockDoesNotDivideAggregateThroughputTwice) {
  LogicalStage stage =
      logicalStage("simt_payload", StageCostModelKind::ScalarIssue);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.workload.operationCounts.clear();
  stage.workload.loadWarpInstructions = 40.0;
  stage.workload.storeWarpInstructions = 20.0;
  stage.workload.shuffleLaneSteps = 64.0;
  HardwareProfile profile = hardwareProfile();
  profile.simt.loadWarpInstructionsPerCycleByWarpCount = {
      {1, 1}, {2, 2}, {4, 4}};
  profile.simt.storeWarpInstructionsPerCycleByWarpCount = {
      {1, 1}, {2, 2}, {4, 4}};
  profile.simt.shuffleLanesPerCycleByWarpCount = {{1, 32}, {2, 64}, {4, 128}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  // F * (40/F + 20/F + 64/(32*F)) = 62 at every F. The old
  // extra latency divisor incorrectly produced 62/F.
  for (const auto &cost : costs)
    EXPECT_DOUBLE_EQ(cost.totalCycles, 10 + 62);
  profile.simt.loadWarpInstructionsPerCycleByWarpCount.clear();
  profile.simt.storeWarpInstructionsPerCycleByWarpCount.clear();
  profile.simt.shuffleLanesPerCycleByWarpCount.clear();
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  for (const auto &cost : table->stages.front().implementations)
    EXPECT_DOUBLE_EQ(cost.totalCycles,
                     10 + cost.implementation.superblockFactor * 62);
}

TEST(SimdSimtCostModelTest, PureSimtRouteUsesOneUniformSuperBlockFactor) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](int64_t factor, double cycles) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, factor};
    cost.totalCycles = cycles;
    return cost;
  };
  mlir::ascend::LogicalStageCost first;
  first.id = "first";
  first.implementations = {makeCost(1, 5.0), makeCost(2, 1.0),
                           makeCost(4, 3.0)};
  mlir::ascend::LogicalStageCost second;
  second.id = "second";
  second.implementations = {makeCost(1, 5.0), makeCost(2, 4.0),
                            makeCost(4, 1.0)};
  table.stages = {first, second};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimt.legal);
  EXPECT_EQ(routes->allSimt.routeSuperblockFactor, 4);
  ASSERT_EQ(routes->allSimt.implementations.size(), 2u);
  EXPECT_EQ(routes->allSimt.implementations[0].superblockFactor, 4);
  EXPECT_EQ(routes->allSimt.implementations[1].superblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 4.0);
}

TEST(SimdSimtCostModelTest, MixedScopeSuperBlockUsesSelectedFactorCost) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, int64_t factor, double cycles,
                      bool localScope = false) {
    mlir::ascend::StageImplementationCost cost;
    cost.implementation = {mode, factor, localScope};
    cost.totalCycles = cycles;
    return cost;
  };
  mlir::ascend::LogicalStageCost prefix;
  prefix.id = "simd_prefix";
  prefix.implementations = {makeCost(StageMode::SIMD, 1, 5.0),
                            makeCost(StageMode::SIMT, 1, 50.0),
                            makeCost(StageMode::SIMT, 2, 25.0),
                            makeCost(StageMode::SIMT, 4, 12.5),
                            makeCost(StageMode::SIMT, 1, 50.0, true),
                            makeCost(StageMode::SIMT, 2, 25.0, true),
                            makeCost(StageMode::SIMT, 4, 12.5, true)};
  prefix.localSimtMaterializable = true;
  prefix.localSimtFactors = {1, 2, 4};

  mlir::ascend::LogicalStageCost payload;
  payload.id = "local_simt_payload";
  payload.implementations = {makeCost(StageMode::SIMD, 1, 100.0),
                             makeCost(StageMode::SIMT, 1, 10.0),
                             makeCost(StageMode::SIMT, 2, 1.0),
                             makeCost(StageMode::SIMT, 4, 0.5),
                             makeCost(StageMode::SIMT, 1, 10.0, true),
                             makeCost(StageMode::SIMT, 2, 1.0, true),
                             makeCost(StageMode::SIMT, 4, 0.5, true)};
  payload.localSimtMaterializable = true;
  payload.localSimtFactors = {1, 2, 4};
  table.stages = {prefix, payload};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 5.5);
}

TEST(SimdSimtCostModelTest,
     FactoredMixedRouteUsesOneBackendMaterializableLocalScope) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, int64_t factor, double cycles,
                      bool localScope = false) {
    StageImplementationCost cost;
    cost.implementation = {mode, factor, localScope};
    cost.totalCycles = cycles;
    return cost;
  };

  LogicalStageCost first;
  first.id = "first_local_candidate";
  first.features.insideAutoBlockifyV1Loop = true;
  first.localSimtMaterializable = true;
  first.implementations = {makeCost(StageMode::SIMD, 1, 100.0),
                           makeCost(StageMode::SIMT, 4, 1.0, true)};
  LogicalStageCost second = first;
  second.id = "second_local_candidate";
  LogicalStageCost tail;
  tail.id = "simd_tail";
  tail.implementations = {makeCost(StageMode::SIMD, 1, 1.0)};
  table.stages = {first, second, tail};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 4);
  EXPECT_EQ(llvm::count_if(
                routes->mixed.implementations,
                [](const mlir::ascend::StageImplementation &implementation) {
                  return implementation.localScope;
                }),
            1);
  // One scope is SIMT (1 cycle); the other Stage remains SIMD and is cloned
  // once per grouped logical program (100 * F4); the outside tail is not.
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 402.0);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[0], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[1], 0.0);
  EXPECT_DOUBLE_EQ(routes->mixed.scopeHandoffCycles[2], 0.0);
}

TEST(SimdSimtCostModelTest,
     OperationGraphBoundaryOwnsEveryRootAndDerivesLiveValues) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %loaded = "tt.load"(%pointer) : (i64) -> tensor<16x16xf32>
        %result = scf.for %i = %c2 to %c16 step %c1
            iter_args(%state = %loaded) -> tensor<16x16xf32> {
          %next = arith.addf %state, %loaded : tensor<16x16xf32>
          scf.yield %next : tensor<16x16xf32>
        }
        "tt.store"(%pointer, %result) : (i64, tensor<16x16xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::Operation *recurrence = nullptr;
  module->walk([&](mlir::scf::ForOp loop) { recurrence = loop; });
  ASSERT_NE(recurrence, nullptr);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = recurrence;
  anchor.scopeOperations.push_back(recurrence);
  anchor.scopeInsertionPoint = recurrence;
  anchor.kind = mlir::ascend::SimtAnchorKind::TriangularSolveLoop;
  anchor.triangularSolve =
      triangularBt16StageFeatures().simtAnchors.triangularSolves.front();
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  auto structure =
      mlir::ascend::ProgramStructureAnalysis().analyze(*module, anchorPlan);
  if (!structure)
    FAIL() << llvm::toString(structure.takeError());
  EXPECT_EQ(structure->rootOperations.size(), 6u);

  auto result = StagePartitioner().partition(*module, anchorPlan,
                                             StagePartitionerOptions{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  const StagePartition &partition = *result;
  EXPECT_TRUE(partition.operationOwnershipComplete);

  int64_t ownedRootCount = 0;
  const LogicalStage *recurrenceStage = nullptr;
  for (const LogicalStage &stage : partition.stages) {
    ownedRootCount += static_cast<int64_t>(stage.operations.size());
    if (llvm::is_contained(stage.operations, recurrence))
      recurrenceStage = &stage;
  }
  EXPECT_EQ(ownedRootCount, partition.modeledOperationCount);
  ASSERT_NE(recurrenceStage, nullptr);
  EXPECT_EQ(recurrenceStage->operations.size(), 1u);
  EXPECT_FALSE(recurrenceStage->liveIns.empty());
  EXPECT_EQ(recurrenceStage->liveOuts.size(), 1u);
  EXPECT_EQ(recurrenceStage->simtAnchorIndices, std::vector<unsigned>({0}));
  EXPECT_TRUE(recurrenceStage->localSimtMaterializable);
  // The arith.addf is the scf.for body and therefore represents 256 element
  // additions on every one of the 14 dynamic recurrence iterations.  The
  // per-iteration Stage workload must remain 256, not be divided by 14.
  auto add = recurrenceStage->workload.operationCounts.find("f32.add");
  ASSERT_NE(add, recurrenceStage->workload.operationCounts.end());
  EXPECT_DOUBLE_EQ(add->second.at(256), 1.0);
}

TEST(SimdSimtCostModelTest,
     SameStatementSupportOperationsJoinTheDominantResourceStage) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64) {
        %index = arith.constant 0 : i64
        %mask = arith.cmpi eq, %index, %index : i64
        %value = "tt.load"(%pointer, %mask) : (i64, i1) -> f32
        %tail = arith.constant 1 : i64
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  llvm::SmallVector<mlir::Operation *> roots;
  module->walk([&](mlir::func::FuncOp function) {
    for (mlir::Operation &operation : function.getBody().front())
      if (!operation.hasTrait<mlir::OpTrait::IsTerminator>())
        roots.push_back(&operation);
  });
  ASSERT_EQ(roots.size(), 4u);
  auto statement = mlir::FileLineColLoc::get(&context, "kernel.py", 10, 1);
  for (mlir::Operation *operation : llvm::ArrayRef(roots).take_front(3))
    operation->setLoc(statement);
  roots.back()->setLoc(mlir::FileLineColLoc::get(&context, "kernel.py", 11, 1));

  mlir::ascend::ProgramStructure structure;
  structure.rootOperations.assign(roots.begin(), roots.end());
  auto result = mlir::ascend::StageBoundaryAnalysis().analyze(
      structure, mlir::ascend::SimtAnchorPlan{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_EQ(result->stages.size(), 2u);
  EXPECT_EQ(result->stages.front().operations.size(), 3u);
  EXPECT_EQ(result->stages.front().costModelKind,
            StageCostModelKind::ScalarLoad);
  ASSERT_EQ(result->stages.back().operations.size(), 1u);
  EXPECT_EQ(result->stages.back().operations.front(), roots.back());
}

TEST(SimdSimtCostModelTest,
     CompoundScopeOrderIsNormalizedBeforeStagePartitioning) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64) {
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %setup = arith.constant dense<0> : tensor<16xi32>
        %loaded = "tt.load"(%pointer) : (i64) -> tensor<16x16xf32>
        %result = scf.for %i = %c2 to %c16 step %c1
            iter_args(%state = %loaded) -> tensor<16x16xf32> {
          %next = arith.addf %state, %loaded : tensor<16x16xf32>
          scf.yield %next : tensor<16x16xf32>
        }
        "tt.store"(%pointer, %result) : (i64, tensor<16x16xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::Operation *setup = nullptr;
  mlir::Operation *recurrence = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "arith.constant" &&
        operation->getNumResults() == 1 &&
        mlir::isa<mlir::RankedTensorType>(operation->getResult(0).getType()))
      setup = operation;
    if (mlir::isa<mlir::scf::ForOp>(operation))
      recurrence = operation;
  });
  ASSERT_NE(setup, nullptr);
  ASSERT_NE(recurrence, nullptr);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = recurrence;
  anchor.scopeOperations = {setup, recurrence};
  anchor.scopeInsertionPoint = recurrence;
  anchor.kind = mlir::ascend::SimtAnchorKind::TriangularSolveLoop;
  anchor.triangularSolve =
      triangularBt16StageFeatures().simtAnchors.triangularSolves.front();
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  auto structure =
      mlir::ascend::ProgramStructureAnalysis().analyze(*module, anchorPlan);
  if (!structure)
    FAIL() << llvm::toString(structure.takeError());
  auto setupPosition = llvm::find(structure->rootOperations, setup);
  auto recurrencePosition = llvm::find(structure->rootOperations, recurrence);
  ASSERT_NE(setupPosition, structure->rootOperations.end());
  ASSERT_NE(recurrencePosition, structure->rootOperations.end());
  EXPECT_EQ(recurrencePosition - setupPosition, 1);

  auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());
  const LogicalStage *loadStage = nullptr;
  const LogicalStage *recurrenceStage = nullptr;
  for (const LogicalStage &stage : partition->stages) {
    if (llvm::any_of(stage.operations, [&](mlir::Operation *operation) {
          return operation->getName().getStringRef() == "tt.load";
        }))
      loadStage = &stage;
    if (llvm::is_contained(stage.operations, recurrence))
      recurrenceStage = &stage;
  }
  ASSERT_NE(loadStage, nullptr);
  ASSERT_NE(recurrenceStage, nullptr);
  EXPECT_EQ(loadStage->operations.size(), 1u);
  EXPECT_EQ(recurrenceStage->operations.size(), 2u);
  EXPECT_EQ(recurrenceStage->simtAnchorIndices, std::vector<unsigned>({0}));
}

TEST(SimdSimtCostModelTest,
     NestedLocalScopeDoesNotAdvertiseUnsupportedSuperBlockFactors) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64, %condition: i1) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          scf.if %condition {
            %value = "tt.load"(%pointer) : (i64) -> tensor<16xf32>
          }
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::scf::ForOp v1Loop;
  mlir::Operation *nestedLoad = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(operation))
      v1Loop = loop;
    if (operation->getName().getStringRef() == "tt.load")
      nestedLoad = operation;
  });
  ASSERT_TRUE(v1Loop);
  ASSERT_NE(nestedLoad, nullptr);
  v1Loop->setAttr("ta.auto_blockify_v1.loop", mlir::UnitAttr::get(&context));

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = nestedLoad;
  anchor.scopeOperations.push_back(nestedLoad);
  anchor.scopeInsertionPoint = nestedLoad;
  anchor.kind = mlir::ascend::SimtAnchorKind::DirectGather;
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  StagePartitionerOptions options;
  options.maximumSuperblockFactor = 4;
  options.scopeSuperblockMaterializable = true;
  auto result = StagePartitioner().partition(*module, anchorPlan, options);
  if (!result)
    FAIL() << llvm::toString(result.takeError());

  const LogicalStage *nestedStage = nullptr;
  for (const LogicalStage &stage : result->stages)
    if (!stage.simtAnchorIndices.empty())
      nestedStage = &stage;
  ASSERT_NE(nestedStage, nullptr);
  EXPECT_TRUE(nestedStage->localSimtMaterializable);
  EXPECT_FALSE(nestedStage->localSuperblockMaterializable);
  EXPECT_FALSE(nestedStage->operations.empty());
  EXPECT_EQ(nestedStage->localSimtFactors, (std::vector<int64_t>{1}));

  auto costs = StageCostEvaluator().evaluate(*result, hardwareProfile());
  if (!costs)
    FAIL() << llvm::toString(costs.takeError());
  auto nestedCost = llvm::find_if(costs->stages, [&](const auto &stage) {
    return stage.id == nestedStage->id;
  });
  ASSERT_NE(nestedCost, costs->stages.end());
  EXPECT_FALSE(nestedCost->sourceLocations.empty());
}

TEST(SimdSimtCostModelTest, GenericSemanticStagesDoNotRequireAWorkloadDomain) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @unrelated_kernel(%pointer: i64) {
        %zero = arith.constant dense<0.0> : tensor<16xf32>
        %loaded = "tt.load"(%pointer) : (i64) -> tensor<16xf32>
        %mask = arith.cmpf ogt, %loaded, %zero : tensor<16xf32>
        "tt.store"(%pointer, %loaded) : (i64, tensor<16xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::ascend::SimtAnchorPlan anchorPlan;
  auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());

  ASSERT_TRUE(partition->operationOwnershipComplete);
  EXPECT_EQ(partition->modeledOperationCount, 4);
  ASSERT_EQ(partition->stages.size(), 4u);
  EXPECT_EQ(partition->stages[0].costModelKind,
            StageCostModelKind::ScalarIssue);
  EXPECT_EQ(partition->stages[1].costModelKind,
            StageCostModelKind::ContinuousTileMemory);
  EXPECT_EQ(partition->stages[2].costModelKind,
            StageCostModelKind::PredicateMask);
  EXPECT_EQ(partition->stages[3].costModelKind,
            StageCostModelKind::ContinuousTileStore);
  // This synthetic i64 pointer has no stride evidence. Stage ownership and
  // bytes remain known, but it must not consume a measured flat-layout curve.
  const auto &load = partition->stages[1].workload;
  const auto &store = partition->stages[3].workload;
  EXPECT_TRUE(load.continuousLoadCounts.empty());
  EXPECT_TRUE(store.continuousStoreCounts.empty());
  EXPECT_DOUBLE_EQ(load.loadBytes, 64.0);
  EXPECT_DOUBLE_EQ(store.storeBytes, 64.0);
  EXPECT_DOUBLE_EQ(load.loadOperations, 1.0);
  EXPECT_DOUBLE_EQ(store.storeOperations, 1.0);
  ASSERT_EQ(load.memoryAccesses.size(), 1u);
  ASSERT_EQ(store.memoryAccesses.size(), 1u);
  EXPECT_EQ(load.memoryAccesses.front().layoutClass(),
            StageMemoryAccess::LayoutClass::Unknown);
  EXPECT_EQ(store.memoryAccesses.front().layoutClass(),
            StageMemoryAccess::LayoutClass::Unknown);
}

TEST(SimdSimtCostModelTest, ElementwiseClassificationUsesShapedTypeNotOpName) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  const std::pair<const char *, const char *> cases[] = {
      {"arith.addf", "tensor<8x16xf32>"},
      {"arith.subf", "tensor<16x16xf32>"},
      {"arith.mulf", "tensor<1xf32>"},
      {"arith.divf", "vector<4xf32>"},
      {"arith.maximumf", "tensor<32xf32>"},
      {"arith.minimumf", "tensor<32xf32>"},
      {"arith.addi", "tensor<64xi32>"},
      {"arith.subi", "tensor<64xi32>"},
      {"arith.shli", "tensor<64xi32>"},
      {"arith.xori", "tensor<64xi32>"},
      {"arith.subf", "f32"},
      {"arith.addi", "i32"}};
  for (const auto &[operation, type] : cases) {
    SCOPED_TRACE(std::string(operation) + " " + type);
    const std::string text =
        "module { func.func @elementwise(%a: " + std::string(type) +
        ", %b: " + type + ") -> " + type + " { %r = " + operation +
        " %a, %b : " + type + " return %r : " + type + " } }";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    mlir::ascend::SimtAnchorPlan anchors;
    auto partition = StagePartitioner().partition(*module, anchors,
                                                  StagePartitionerOptions{});
    if (!partition)
      FAIL() << llvm::toString(partition.takeError());
    ASSERT_EQ(partition->stages.size(), 1u);
    const bool shaped = llvm::StringRef(type).contains('<');
    EXPECT_EQ(partition->stages.front().costModelKind,
              shaped ? StageCostModelKind::VectorIssue
                     : StageCostModelKind::ScalarIssue);
    EXPECT_DOUBLE_EQ(partition->stages.front().workload.scalarOperations,
                     shaped ? 0.0 : 1.0);
  }
}

TEST(SimdSimtCostModelTest, TensorSelectIsVectorWorkButConstantsAreNot) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @select(%mask: tensor<64xi1>, %a: tensor<64xf32>)
          -> tensor<64xf32> {
        %zero = arith.constant dense<0.0> : tensor<64xf32>
        %out = arith.select %mask, %a, %zero : tensor<64xi1>, tensor<64xf32>
        return %out : tensor<64xf32>
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::ascend::SimtAnchorPlan anchors;
  auto partition =
      StagePartitioner().partition(*module, anchors, StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());
  ASSERT_EQ(partition->stages.size(), 2u);
  EXPECT_EQ(partition->stages[0].costModelKind,
            StageCostModelKind::ScalarIssue);
  EXPECT_EQ(partition->stages[1].costModelKind,
            StageCostModelKind::VectorIssue);
}

TEST(SimdSimtCostModelTest, BlockPointerStridesGateFlatMemoryCurves) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  for (const auto &[strides, packed] :
       std::vector<std::pair<std::string, bool>>{{"%c16, %c1", true},
                                                 {"%c1, %c16", true},
                                                 {"%c2048, %c1", false},
                                                 {"%dynamic, %c1", false},
                                                 {"%c0, %c1", false}}) {
    SCOPED_TRACE(strides);
    const std::string source = R"mlir(
      module {
        func.func @block_transfer(%base: i64, %dynamic: i64) {
          %c0 = arith.constant 0 : i64
          %c1 = arith.constant 1 : i64
          %c16 = arith.constant 16 : i64
          %c2048 = arith.constant 2048 : i64
          %offset = arith.constant 0 : i32
          %ptr = "tt.make_tensor_ptr"(%base, %c2048, %c2048, )mlir" +
                               strides + R"mlir(, %offset, %offset)
              : (i64, i64, i64, i64, i64, i32, i32) -> i64
          %advanced = "tt.advance"(%ptr, %offset, %offset)
              : (i64, i32, i32) -> i64
          %loaded = "tt.load"(%advanced) : (i64) -> tensor<16x16xf32>
          "tt.store"(%advanced, %loaded) : (i64, tensor<16x16xf32>) -> ()
          return
        }
      }
    )mlir";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    mlir::ascend::SimtAnchorPlan anchorPlan;
    auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                  StagePartitionerOptions{});
    if (!partition)
      FAIL() << llvm::toString(partition.takeError());
    unsigned memoryStages = 0;
    for (const auto &stage : partition->stages) {
      const auto &work = stage.workload;
      if (work.loadOperations > 0) {
        ++memoryStages;
        EXPECT_DOUBLE_EQ(work.loadOperations, 1.0);
        EXPECT_DOUBLE_EQ(work.loadBytes, 1024.0);
        EXPECT_DOUBLE_EQ(work.loadWarpInstructions, 8.0);
        EXPECT_EQ(work.continuousLoadCounts.size(), packed ? 1u : 0u);
      }
      if (work.storeOperations > 0) {
        ++memoryStages;
        EXPECT_DOUBLE_EQ(work.storeOperations, 1.0);
        EXPECT_DOUBLE_EQ(work.storeBytes, 1024.0);
        EXPECT_DOUBLE_EQ(work.storeWarpInstructions, 8.0);
        EXPECT_EQ(work.continuousStoreCounts.size(), packed ? 1u : 0u);
      }
    }
    EXPECT_EQ(memoryStages, 2u);
  }
}

TEST(SimdSimtCostModelTest, ClippedDynamicLoopUsesStaticTripCountCap) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  for (bool capped : {false, true}) {
    std::string source = R"mlir(
      module {
        func.func @kernel(%limit: i32, %initial: i32) -> i32 {
          %c1 = arith.constant 1 : i32
          %c2 = arith.constant 2 : i32
          %c16 = arith.constant 16 : i32
          %cap = arith.minsi %limit, %c16 : i32
          %result = scf.for %i = %c2 to UPPER step %c1
              iter_args(%state = %initial) -> (i32) : i32 {
            %next = arith.addi %state, %i : i32
            scf.yield %next : i32
          }
          %second = scf.for %j = %c2 to %cap step %c1
              iter_args(%state = %result) -> (i32) : i32 {
            %next = arith.addi %state, %j : i32
            scf.yield %next : i32
          }
          return %second : i32
        }
      }
    )mlir";
    source.replace(source.find("UPPER"), 5, capped ? "%cap" : "%limit");
    if (!capped)
      source.replace(source.find("to %cap"), 7, "to %limit");
    auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(module);
    mlir::ascend::SimtAnchorPlan anchors;
    auto partition = StagePartitioner().partition(*module, anchors,
                                                  StagePartitionerOptions{});
    if (!partition)
      FAIL() << llvm::toString(partition.takeError());
    if (llvm::Error error = StageFeatureAnalysis().analyze(*partition))
      FAIL() << llvm::toString(std::move(error));
    int64_t maximumIterations = 1;
    for (const auto &stage : partition->stages) {
      maximumIterations = std::max(maximumIterations, stage.iterationCount);
      EXPECT_EQ(stage.features.parallelRecurrenceGroupCount, 1);
    }
    EXPECT_EQ(maximumIterations, capped ? 14 : 1);
  }
}

TEST(SimdSimtCostModelTest, AdjacentStructuredLoopsRemainSerialStages) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @two_serial_loops(%pointer: i64) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          %first = "tt.load"(%pointer) : (i64) -> f32
        }
        scf.for %i = %c0 to %c4 step %c1 {
          %second = "tt.load"(%pointer) : (i64) -> f32
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::ascend::SimtAnchorPlan anchorPlan;
  auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());

  llvm::SmallVector<const LogicalStage *> loopStages;
  for (const LogicalStage &stage : partition->stages)
    if (stage.costModelKind == StageCostModelKind::IndependentPipelinedLoop)
      loopStages.push_back(&stage);
  ASSERT_EQ(loopStages.size(), 2u);
  EXPECT_NE(loopStages[0]->operations.front(),
            loopStages[1]->operations.front());
}

TEST(SimdSimtCostModelTest,
     LocalScopeReturningPointerTensorIsRejectedBeforeScoring) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%base: !tt.ptr<f16>) {
        %indices = arith.constant dense<0> : tensor<16xi32>
        %ptrs = "tt.addptr"(%base, %indices)
            : (!tt.ptr<f16>, tensor<16xi32>) -> tensor<16x!tt.ptr<f16>>
        %values = "tt.load"(%ptrs)
            : (tensor<16x!tt.ptr<f16>>) -> tensor<16xf16>
        %reduced = "tt.reduce"(%values) : (tensor<16xf16>) -> f16
        "tt.store"(%base, %reduced) : (!tt.ptr<f16>, f16) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  llvm::SmallVector<mlir::Operation *> roots;
  module->walk([&](mlir::func::FuncOp function) {
    for (mlir::Operation &operation : function.getBody().front())
      if (!operation.hasTrait<mlir::OpTrait::IsTerminator>())
        roots.push_back(&operation);
  });
  ASSERT_EQ(roots.size(), 5u);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = roots[1];
  anchor.scopeOperations = {roots[1]};
  anchor.scopeInsertionPoint = roots[1];
  anchor.kind = mlir::ascend::SimtAnchorKind::DirectGather;
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  mlir::ascend::ProgramStructure structure;
  structure.rootOperations.assign(roots.begin(), roots.end());
  auto result =
      mlir::ascend::StageBoundaryAnalysis().analyze(structure, anchorPlan);
  if (!result)
    FAIL() << llvm::toString(result.takeError());

  const LogicalStage *gather = nullptr;
  for (const LogicalStage &stage : result->stages)
    if (llvm::is_contained(stage.operations, roots[1]))
      gather = &stage;
  ASSERT_NE(gather, nullptr);
  EXPECT_FALSE(gather->localSimtMaterializable);
  EXPECT_TRUE(gather->localSimtFactors.empty());
  EXPECT_TRUE(gather->simtAnchorIndices.empty());
}

TEST(SimdSimtCostModelTest, PointerInductionLoopIsNotADataRecurrence) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%start: i64) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        %step = arith.constant 16 : i64
        %address = scf.for %i = %c0 to %c8 step %c1
            iter_args(%current = %start) -> i64 {
          %value = "tt.load"(%current) : (i64) -> f32
          %next = arith.addi %current, %step : i64
          scf.yield %next : i64
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *loop = nullptr;
  module->walk([&](mlir::scf::ForOp operation) { loop = operation; });
  ASSERT_NE(loop, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("pointer_loop", StageCostModelKind::ConversionPack,
                   StageScheduleKind::IndependentPipelined, 8);
  stage.operations.push_back(loop);
  partition.stages.push_back(std::move(stage));

  if (llvm::Error error = StageFeatureAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  if (llvm::Error error =
          mlir::ascend::StageKindClassifier().analyze(partition, 8192))
    FAIL() << llvm::toString(std::move(error));
  const LogicalStage &classified = partition.stages.front();
  EXPECT_TRUE(classified.features.hasLoop);
  EXPECT_TRUE(classified.features.hasPointerInduction);
  EXPECT_FALSE(classified.features.hasLoopCarriedDataDependency);
  EXPECT_EQ(classified.costModelKind,
            StageCostModelKind::IndependentPipelinedLoop);
}

TEST(SimdSimtCostModelTest, IncompatibleDominantStructuresRequireStageSplit) {
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("gather_dot", StageCostModelKind::TinyCubeRoofline,
                   StageScheduleKind::PartiallyDependent, 1);
  stage.features.hasDot = true;
  stage.features.hasIndirectMemory = true;
  partition.stages.push_back(std::move(stage));

  llvm::Error error =
      mlir::ascend::StageKindClassifier().analyze(partition, 16384);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("requires_split"),
            std::string::npos);
}

TEST(SimdSimtCostModelTest, VectorRoundingPreservesOperationAndLoopBoundaries) {
  for (int size : {16, 64, 65}) {
    for (int copies : {1, 2}) {
      mlir::MLIRContext context;
      context.getOrLoadDialect<mlir::arith::ArithDialect>();
      context.getOrLoadDialect<mlir::func::FuncDialect>();
      context.getOrLoadDialect<mlir::scf::SCFDialect>();
      const std::string type = "tensor<" + std::to_string(size) + "xf32>";
      const std::string text =
          "module { func.func @kernel(%a: " + type + ", %b: " + type +
          ") { %c0 = arith.constant 0 : index\n"
          "%c1 = arith.constant 1 : index\n"
          "%c3 = arith.constant 3 : index\n"
          "scf.for %i = %c0 to %c3 step %c1 {\n"
          "%v = arith.addf %a, %b : " +
          type + "\n" +
          (copies == 2 ? "%w = arith.addf %v, %b : " + type + "\n" : "") +
          "%p = arith.cmpf olt, %a, %b : " + type + "\n" +
          (copies == 2 ? "%q = arith.cmpf oge, %v, %b : " + type + "\n" : "") +
          "}\nreturn } }";
      auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
      ASSERT_TRUE(module);
      mlir::Operation *loop = nullptr;
      module->walk([&](mlir::scf::ForOp operation) { loop = operation; });
      ASSERT_NE(loop, nullptr);
      for (int normalization : {1, 3}) {
        StagePartition partition;
        partition.operationOwnershipComplete = true;
        auto stage =
            logicalStage("arithmetic", StageCostModelKind::ScalarMath,
                         StageScheduleKind::StraightLine, normalization);
        stage.operations.push_back(loop);
        partition.stages.push_back(std::move(stage));
        for (int repeat = 0; repeat < 2; ++repeat) {
          if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
            FAIL() << llvm::toString(std::move(error));
          const auto &work = partition.stages.front().workload;
          EXPECT_DOUBLE_EQ(work.operationCounts.lookup("f32.add").at(size) *
                               normalization,
                           3.0 * copies);
          EXPECT_DOUBLE_EQ(work.predicateCounts.at(size) * normalization,
                           3.0 * copies);
          auto table =
              StageCostEvaluator().evaluate(partition, hardwareProfile());
          if (!table)
            FAIL() << llvm::toString(table.takeError());
          const auto &costs = table->stages.front().implementations;
          ASSERT_EQ(costs.size(), 2u);
          EXPECT_DOUBLE_EQ(costs[0].resources.compute * normalization,
                           3.0 * copies * ((size + 63) / 64));
          EXPECT_DOUBLE_EQ(costs[1].resources.compute * normalization,
                           3.0 * copies * size);
          EXPECT_DOUBLE_EQ(costs[0].resources.predicate * normalization,
                           3.0 * copies * ((size + 63) / 64));
          EXPECT_DOUBLE_EQ(costs[1].resources.predicate * normalization,
                           3.0 * copies * size);
        }
      }
    }
  }
}

TEST(SimdSimtCostModelTest, OperationInstanceCountsRequireValidSizesAndCounts) {
  StageWorkload work;
  work.operationCounts["f32.add"][16] = 1.0;
  work.predicateCounts[16] = 1.0;
  EXPECT_TRUE(work.isFiniteAndNonNegative());
  work.predicateCounts[16] = -1.0;
  EXPECT_FALSE(work.isFiniteAndNonNegative());
  work.predicateCounts[16] = 0.0;
  EXPECT_TRUE(work.isFiniteAndNonNegative());
  work.operationCounts["f32.add"][0] = 1.0;
  EXPECT_FALSE(work.isFiniteAndNonNegative());
  work.operationCounts["f32.add"].erase(0);
  work.indirectLoadCounts[{512, 256}] = -1.0;
  EXPECT_FALSE(work.isFiniteAndNonNegative());
}

TEST(SimdSimtCostModelTest, MemoryLayoutClassUsesStrideRowWidthAndRank) {
  StageMemoryAccess strided;
  strided.shape = {16, 16};
  strided.strides = {2048, 1};
  strided.elementBits = 32;
  EXPECT_EQ(strided.layoutClass(), StageMemoryAccess::LayoutClass::StridedWide);
  EXPECT_EQ(strided.effectiveRank(), 2);
  EXPECT_EQ(strided.rowBytes(), 64);
  EXPECT_EQ(strided.rowStrideBytes(), 8192);
  EXPECT_EQ(strided.rowCount(), 16);
  EXPECT_EQ(strided.descriptorCount(2), 1);
  EXPECT_FALSE(strided.isFlatContiguous());

  StageMemoryAccess packed;
  packed.shape = {16, 16};
  packed.strides = {16, 1};
  packed.elementBits = 32;
  EXPECT_EQ(packed.layoutClass(),
            StageMemoryAccess::LayoutClass::ContiguousWide);
  EXPECT_EQ(packed.rowBytes(), 64);
  EXPECT_EQ(packed.rowStrideBytes(), 64);
  EXPECT_TRUE(packed.isFlatContiguous());

  StageMemoryAccess shortRow;
  shortRow.shape = {8, 2};
  shortRow.strides = {2, 1};
  shortRow.elementBits = 16;
  EXPECT_EQ(shortRow.layoutClass(),
            StageMemoryAccess::LayoutClass::ContiguousShort);
  EXPECT_EQ(shortRow.rowBytes(), 4);

  StageMemoryAccess sixDimensional;
  sixDimensional.shape = {2, 2, 2, 2, 2, 2};
  sixDimensional.strides = {32, 16, 8, 4, 2, 1};
  sixDimensional.elementBits = 32;
  EXPECT_EQ(sixDimensional.effectiveRank(), 6);
  EXPECT_EQ(sixDimensional.descriptorCount(5), 2);
  EXPECT_EQ(sixDimensional.descriptorCount(2), 16);
}

TEST(SimdSimtCostModelTest, MemoryLayoutDoesNotPriceHolesAsContiguousBytes) {
  StageMemoryAccess access;
  access.shape = {8, 16, 1};
  access.strides = {2048, 2, 1};
  access.elementBits = 32;
  EXPECT_EQ(access.rowBytes(), 4);
  EXPECT_EQ(access.rowStrideBytes(), 8);
  EXPECT_EQ(access.rowCount(), 128);
  EXPECT_TRUE(access.hasNonUnitInnerStride());
  EXPECT_EQ(access.layoutClass(), StageMemoryAccess::LayoutClass::StridedShort);
  auto other = access;
  other.shape[0] = 16;
  EXPECT_NE(access.layoutKey(), other.layoutKey());
  other = access;
  other.masked = true;
  EXPECT_NE(access.layoutKey(), other.layoutKey());
  other = access;
  other.strides[0] = std::nullopt;
  EXPECT_EQ(other.layoutClass(), StageMemoryAccess::LayoutClass::Unknown);
  EXPECT_FALSE(other.hasNonUnitInnerStride());
  StageMemoryAccess permuted;
  permuted.shape = {16, 8};
  permuted.strides = {1, 2048};
  permuted.elementBits = 32;
  EXPECT_TRUE(permuted.hasUnitInnerStride());
  EXPECT_EQ(permuted.rowBytes(), 64);
  EXPECT_EQ(permuted.rowCount(), 8);
}

TEST(SimdSimtCostModelTest,
     LayoutCountCurveRejectsExtrapolationAndInvalidEvidence) {
  StageMemoryLayoutCost curve{200.0, 100.0, 1, 10, "unit-test synthetic sweep"};
  ASSERT_TRUE(curve.isValid());
  EXPECT_EQ(curve.evaluate(0), 0.0);
  EXPECT_EQ(curve.evaluate(1), 200.0);
  EXPECT_EQ(curve.evaluate(4), 500.0);
  EXPECT_FALSE(curve.evaluate(11));
  EXPECT_FALSE(curve.evaluate(0.5));
  curve.evidence.clear();
  EXPECT_FALSE(curve.isValid());
}

TEST(SimdSimtCostModelTest,
     LayoutGroupMemoryIsCoveredOnceNotMultipliedByFactor) {
  auto stage = logicalStage("memory", StageCostModelKind::ContinuousTileMemory);
  stage.workload = {};
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  StageMemoryAccess access;
  access.shape = {16, 16};
  access.strides = {2048, 1};
  access.elementBits = 32;
  // Separate SSA operations with the same geometry form one count sweep.
  stage.workload.memoryAccesses.assign(4, access);
  stage.workload.loadBytes = 4096;
  stage.workload.loadOperations = 4;
  stage.workload.loadWarpInstructions = 32;
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  const auto key = access.layoutKey() + ":warps=8:factor=2";
  profile.simt.loadLayoutCosts[key] = {200.0, 100.0, 1, 10,
                                       "synthetic unit test"};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_DOUBLE_EQ(costs[1].resources.continuousLoadGroup, 500.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.load, 0.0);
  EXPECT_DOUBLE_EQ(costs[1].totalCycles, 500.0);
  // An F2 measurement must not price either F1 or F4.
  EXPECT_DOUBLE_EQ(costs[0].resources.continuousLoadGroup, 0.0);
  EXPECT_DOUBLE_EQ(costs[2].resources.continuousLoadGroup, 0.0);
  EXPECT_EQ(costs[0].formulaEvidence, "uncalibrated_layout_memory_fallback");
}

TEST(SimdSimtCostModelTest, LayoutCountCurveDoesNotCrossOtherMemoryOrSync) {
  auto stage = logicalStage("memory", StageCostModelKind::ContinuousTileMemory);
  stage.workload = {};
  stage.simdLegal = false;
  stage.legalSimtFactors = {2};
  StageMemoryAccess load;
  load.shape = {16, 16};
  load.strides = {2048, 1};
  load.elementBits = 32;
  auto store = load;
  store.store = true;
  stage.workload.memoryAccesses = {load, store, load};
  stage.workload.loadBytes = 2048;
  stage.workload.loadOperations = 2;
  stage.workload.loadWarpInstructions = 16;
  stage.workload.storeBytes = 1024;
  stage.workload.storeOperations = 1;
  stage.workload.storeWarpInstructions = 8;
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  const auto key = load.layoutKey() + ":warps=8:factor=2";
  profile.simt.loadLayoutCosts[key] = {200.0, 100.0, 1, 10, "synthetic test"};
  profile.simt.storeLayoutCosts[key] = {150.0, 50.0, 1, 10, "synthetic test"};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  auto cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.continuousLoadGroup, 400.0);
  EXPECT_DOUBLE_EQ(cost.resources.continuousStoreGroup, 150.0);
  // Two load sequences pay two first rounds, not one first + one increment.
  EXPECT_DOUBLE_EQ(cost.totalCycles, 550.0);
  stage.features.synchronizationCount = 1;
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.continuousLoadGroup, 0.0);
  EXPECT_DOUBLE_EQ(cost.resources.continuousStoreGroup, 0.0);
  EXPECT_GT(cost.resources.load, 0.0);
  EXPECT_GT(cost.resources.store, 0.0);
}

TEST(SimdSimtCostModelTest, StoreFootprintServiceCountsLayoutAndFactorOnce) {
  auto stage = logicalStage("store", StageCostModelKind::ContinuousTileStore);
  stage.workload = {};
  stage.simdLegal = false;
  stage.legalSimtFactors = {2};
  StageMemoryAccess access;
  access.shape = {16, 16};
  access.strides = {2048, 1};
  access.elementBits = 32;
  access.store = true;
  access.count = 10;
  stage.workload.memoryAccesses = {access};
  stage.workload.storeBytes = 10240;
  stage.workload.storeOperations = 10;
  stage.workload.storeWarpInstructions = 80;
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.storeWarpInstructionsPerCycle = 1000;
  profile.simt.storeOperationSetupCycles = 0;
  profile.simt.storeFootprintUnitBytes = 128;
  profile.simt.storeCyclesPerFootprintUnit = 2;
  profile.simt.storeIssueFloorByWarpCount = {{8, 30}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  auto cost = table->stages.front().implementations.front();
  // 16 rows/LB * F2 * 2 cycles/unit * 10 operations = 640/group.
  EXPECT_DOUBLE_EQ(cost.resources.storeLayoutService, 320.0);
  EXPECT_DOUBLE_EQ(cost.totalCycles, 640.0);
  EXPECT_EQ(cost.formulaEvidence,
            "layout_store_service_floor_startup_uncalibrated");

  stage.workload.memoryAccesses[0].strides = {16, 1};
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  cost = table->stages.front().implementations.front();
  // Packed layouts failed independent issue-floor validation and must not
  // inherit the new strided service term merely because bytes are equal.
  EXPECT_DOUBLE_EQ(cost.resources.storeLayoutService, 0.0);
  EXPECT_GT(cost.resources.store, 0.0);

  stage.workload.memoryAccesses[0].strides[0] = std::nullopt;
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_DOUBLE_EQ(table->stages.front()
                       .implementations.front()
                       .resources.storeLayoutService,
                   0.0);
  EXPECT_GT(table->stages.front().implementations.front().resources.store, 0.0);

  // A different warp shape has no measured service floor; no interpolation
  // across occupancy regimes and no extrapolation from an eight-warp point.
  stage.workload.memoryAccesses[0] = access;
  profile.baseSimtWarpCount = 2;
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_DOUBLE_EQ(table->stages.front()
                       .implementations.front()
                       .resources.storeLayoutService,
                   0.0);
}

TEST(SimdSimtCostModelTest, DotWorkloadPreservesBatchAndIterationMultiplicity) {
  for (int batch : {1, 4}) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.allowUnregisteredDialects();
    const std::string prefix = batch == 1 ? "" : "4x";
    const std::string lhs = "tensor<" + prefix + "16x32xf32>";
    const std::string rhs = "tensor<" + prefix + "32x16xf32>";
    const std::string out = "tensor<" + prefix + "16x16xf32>";
    const std::string text =
        "module { func.func @kernel(%a: " + lhs + ", %b: " + rhs +
        ") {\n"
        "%c0 = arith.constant 0 : index\n"
        "%c1 = arith.constant 1 : index\n"
        "%c3 = arith.constant 3 : index\n"
        "scf.for %i = %c0 to %c3 step %c1 {\n"
        "%v = \"tt.dot\"(%a, %b) : (" +
        lhs + ", " + rhs + ") -> " + out + "\n}\nreturn } }";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    mlir::Operation *loop = nullptr;
    module->walk([&](mlir::scf::ForOp operation) { loop = operation; });
    ASSERT_NE(loop, nullptr);
    for (int normalization : {1, 3}) {
      StagePartition partition;
      partition.operationOwnershipComplete = true;
      auto stage =
          logicalStage("dot_loop", StageCostModelKind::CubeRoofline,
                       StageScheduleKind::LoopCarriedSerial, normalization);
      stage.operations.push_back(loop);
      partition.stages.push_back(std::move(stage));
      for (int repeat = 0; repeat < 2; ++repeat) {
        if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
          FAIL() << llvm::toString(std::move(error));
        EXPECT_DOUBLE_EQ(partition.stages.front().workload.dotFlops *
                             normalization,
                         3.0 * batch * 2.0 * 16 * 32 * 16);
        EXPECT_DOUBLE_EQ(partition.stages.front().workload.dotOperations *
                             normalization,
                         3.0 * batch);
        ASSERT_EQ(partition.stages.front().workload.dotFlopCounts.size(), 1u);
        EXPECT_DOUBLE_EQ(
            partition.stages.front().workload.dotFlopCounts.begin()->second *
                normalization,
            3.0 * batch);
      }
    }
  }
}

TEST(SimdSimtCostModelTest, TinyDotClassificationUsesPerMatrixShape) {
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("dot_chain", StageCostModelKind::CubeRoofline);
  stage.features.hasDot = true;
  // Nine 16x16x16 matrices: aggregate FLOPs are 9*8192, but each matrix is
  // below the tiny-dot threshold and must retain the tiny-dot model family.
  stage.workload.dotOperations = 9.0;
  stage.workload.dotFlopCounts[8192] = 9.0;
  stage.workload.dotFlops = 9.0 * 8192.0;
  partition.stages.push_back(std::move(stage));

  if (llvm::Error error =
          mlir::ascend::StageKindClassifier().analyze(partition, 16384))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(partition.stages.front().costModelKind,
            StageCostModelKind::TinyCubeRoofline);
}

TEST(SimdSimtCostModelTest,
     PrefixScanWorkloadPreservesAxisAndIndependentSequences) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  const char *text = R"mlir(
    module {
      func.func @kernel(%input: tensor<4x8xf32>) {
        %result = "tt.scan"(%input) {axis = 0 : i32}
          : (tensor<4x8xf32>) -> tensor<4x8xf32>
        return
      }
    })mlir";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  ASSERT_TRUE(module);
  mlir::Operation *scan = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "tt.scan")
      scan = operation;
  });
  ASSERT_NE(scan, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto stage = logicalStage("scan", StageCostModelKind::PrefixScan,
                            StageScheduleKind::StraightLine, 1);
  stage.operations.push_back(scan);
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));

  const auto &work = partition.stages.front().workload;
  ASSERT_EQ(work.prefixScanCounts.size(), 1u);
  EXPECT_DOUBLE_EQ(
      work.prefixScanCounts.at(std::make_pair<int64_t, int64_t>(4, 8)), 1.0);
  EXPECT_DOUBLE_EQ(work.shuffleLaneSteps, 0.0);
  auto json = work.toJSON();
  const auto *instances = json.getArray("prefix_scan_instances_per_iteration");
  ASSERT_NE(instances, nullptr);
  ASSERT_EQ(instances->size(), 1u);
  const auto *instance = (*instances)[0].getAsObject();
  ASSERT_NE(instance, nullptr);
  EXPECT_EQ(instance->getInteger("axis_extent"), 4);
  EXPECT_EQ(instance->getInteger("independent_sequence_count"), 8);
  EXPECT_EQ(instance->getNumber("count_per_iteration"), 1.0);
}

TEST(SimdSimtCostModelTest,
     LoopWorkloadUsesSignedCapsWithoutClaimingExactTrips) {
  struct Case {
    const char *upper;
    int64_t lower;
    int64_t step;
    int64_t trips;
    int64_t bounded;
    int64_t unknown;
  };
  // Dynamic, exact, empty, non-unit-step, reversed/nested min, and an
  // unsigned expression which must not be interpreted as a signed bound.
  for (const Case &test : {
           Case{"%u = arith.minsi %remaining, %c16 : index", 2, 1, 14, 1, 0},
           Case{"%u = arith.constant 16 : index", 2, 1, 14, 0, 0},
           Case{"%u = arith.constant 16 : index", 18, 1, 0, 0, 0},
           Case{"%u = arith.minsi %c16, %remaining : index", 2, 2, 7, 1, 0},
           Case{"%v = arith.minsi %remaining, %c16 : index\n"
                "%u = arith.minsi %v, %c8 : index",
                2, 1, 6, 1, 0},
           Case{"%u = arith.addi %remaining, %c16 : index", 2, 1, 1, 0, 1},
           Case{"%u = arith.minui %remaining, %c16 : index", 2, 1, 1, 0, 1},
       }) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.allowUnregisteredDialects();
    const std::string text =
        "module { func.func @kernel(%remaining: index, %address: i64) {\n"
        "%c16 = arith.constant 16 : index\n"
        "%c8 = arith.constant 8 : index\n"
        "%lo = arith.constant " +
        std::to_string(test.lower) +
        " : index\n"
        "%step = arith.constant " +
        std::to_string(test.step) + " : index\n" + test.upper +
        "\n"
        "scf.for %i = %lo to %u step %step {\n"
        "%value = \"tt.load\"(%address) : (i64) -> tensor<16xf32>\n"
        "}\n return } }";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(module) << text;
    mlir::Operation *loop = nullptr;
    module->walk([&](mlir::scf::ForOp operation) { loop = operation; });
    ASSERT_NE(loop, nullptr);
    StagePartition partition;
    partition.operationOwnershipComplete = true;
    auto stage = logicalStage("generic_loop",
                              StageCostModelKind::IndependentPipelinedLoop,
                              StageScheduleKind::IndependentPipelined, 1);
    stage.operations.push_back(loop);
    partition.stages.push_back(std::move(stage));
    // Re-analysis must not accumulate diagnostic counts.
    for (int repeat = 0; repeat < 2; ++repeat) {
      if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
        FAIL() << llvm::toString(std::move(error));
      const auto &result = partition.stages.front();
      EXPECT_DOUBLE_EQ(result.workload.loadBytes, 64.0 * test.trips) << text;
      EXPECT_DOUBLE_EQ(result.workload.loopBackedges, test.trips) << text;
      EXPECT_EQ(result.features.upperBoundLoopCount, test.bounded) << text;
      EXPECT_EQ(result.features.unknownTripCountLoopCount, test.unknown)
          << text;
    }
  }
}

TEST(SimdSimtCostModelTest,
     ControlWorkloadCountsSiblingAndNestedLoopExecutions) {
  struct Case {
    const char *loops;
    double backedges;
    double branches;
  };
  for (const Case &test : {
           Case{"scf.for %i = %c0 to %c14 step %c1 {\n"
                "scf.if %flag { scf.yield }\n"
                "\"test.barrier\"() : () -> ()\n}\n"
                "scf.for %j = %c0 to %c6 step %c1 {\n"
                "scf.if %flag { scf.yield }\n"
                "\"test.barrier\"() : () -> ()\n}\n",
                20, 20},
           Case{"scf.for %i = %c0 to %c3 step %c1 {\n"
                "scf.for %j = %c0 to %c6 step %c1 {\n"
                "scf.if %flag { scf.yield }\n"
                "\"test.barrier\"() : () -> ()\n}\n}\n",
                21, 18},
           Case{"scf.for %i = %c3 to %c0 step %c1 {\n"
                "scf.if %flag { scf.yield }\n"
                "\"test.barrier\"() : () -> ()\n}\n",
                0, 0},
       }) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();
    context.allowUnregisteredDialects();
    const std::string text = "module { func.func @kernel(%flag: i1) {\n"
                             "%c0 = arith.constant 0 : index\n"
                             "%c1 = arith.constant 1 : index\n"
                             "%c3 = arith.constant 3 : index\n"
                             "%c6 = arith.constant 6 : index\n"
                             "%c14 = arith.constant 14 : index\n" +
                             std::string(test.loops) + "return } }";
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    ASSERT_TRUE(module) << text;
    for (int64_t normalization : {1, 14}) {
      StagePartition partition;
      partition.operationOwnershipComplete = true;
      auto stage =
          logicalStage("control_loops", StageCostModelKind::ScalarControl,
                       StageScheduleKind::LoopCarriedSerial, normalization);
      auto function = *module->getOps<mlir::func::FuncOp>().begin();
      for (mlir::Operation &operation : function.getBody().front())
        if (mlir::isa<mlir::scf::ForOp>(operation))
          stage.operations.push_back(&operation);
      partition.stages.push_back(std::move(stage));
      if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
        FAIL() << llvm::toString(std::move(error));
      const auto &work = partition.stages.front().workload;
      EXPECT_NEAR(work.loopBackedges * normalization, test.backedges, 1e-12);
      EXPECT_NEAR(work.conditionalBranches * normalization, test.branches,
                  1e-12);
      EXPECT_NEAR(work.synchronizations * normalization, test.branches, 1e-12);
    }
  }
}
