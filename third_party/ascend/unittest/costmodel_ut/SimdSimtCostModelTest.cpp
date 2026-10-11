#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/Analysis/SimtAnchorAnalysis.h"
#include "AscendModel/Analysis/SimtRegisterPressureAnalysis.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/RouteModel/StageCostModels.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <gtest/gtest.h>

using mlir::ascend::HardwareProfile;
using mlir::ascend::LogicalStage;
using mlir::ascend::LogicalStageCost;
using mlir::ascend::ScopeHandoffCost;
using mlir::ascend::SimdSimtFeatureSummary;
using mlir::ascend::SimtPrivateStackChargeRegion;
using mlir::ascend::SimtRegisterPressure;
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
using mlir::ascend::StageOperationRate;
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
    mode.operationRates["fp32.saturated"] = {1.0, 1.0};
    mode.operationRates["f32.mul"] = {1.0, 1.0};
    mode.operationRates["f32.max"] = {1.0, 1.0};
    mode.operationRates["f32.relu"] = {1.0, 1.0};
    mode.operationRates["convert.cast"] = {1.0, 1.0};
    mode.loadBytesPerCycle = 32.0;
    mode.storeBytesPerCycle = 16.0;
    mode.loadWarpInstructionsPerCycle = 1.0;
    mode.storeWarpInstructionsPerCycle = 1.0;
    mode.predicateOperationsPerCycle = 1.0;
    mode.reductionLaneStepsPerCycle = 1.0;
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

TEST(SimdSimtCostModelTest, SelectedElapsedReportPreservesSlotDomainAndTerm) {
  mlir::ascend::SimdSimtCostReport::SelectedRouteElapsedPrediction prediction;
  prediction.applied = true;
  prediction.calibrationName = "low-wave";
  prediction.candidate = "mixed_simd_simt";
  prediction.superblockFactor = 1;
  prediction.baseWarpCount = 4;
  prediction.runtimeSchedulingSlotCount = 6;
  prediction.minimumRuntimeSchedulingSlotCount = 4;
  prediction.maximumRuntimeSchedulingSlotCount = 28;
  prediction.minimumRuntimeLoopIterations = 1;
  prediction.maximumRuntimeLoopIterations = 1;
  prediction.minimumSelectionScore = 13300.0;
  prediction.maximumSelectionScore = 13500.0;
  prediction.selectionScoreScale = 1.0;
  prediction.schedulingSlotScale = 96.0;
  prediction.fixedSystemCycles = 4700.0;
  prediction.runtimeKernelArgumentCount = 6;
  prediction.runtimeArgumentSetupSystemCycles = 480.0;
  prediction.systemCounterFrequencyMHz = 1000.0;
  prediction.baseSelectionScore = 13400.0;
  prediction.predictedSystemCounterCycles =
      prediction.fixedSystemCycles +
      prediction.selectionScoreScale * prediction.baseSelectionScore +
      prediction.schedulingSlotScale * prediction.runtimeSchedulingSlotCount;
  prediction.predictedMicroseconds = prediction.predictedSystemCounterCycles /
                                     prediction.systemCounterFrequencyMHz;
  prediction.dotUseSignature = "structural-signature";
  prediction.formula =
      "max(0, fixed + score_scale * score + slot_scale * slots)";
  prediction.evidence = "unit evidence";
  auto json = prediction.toJSON();
  EXPECT_EQ(json.getInteger("runtime_scheduling_slot_count"), 6);
  EXPECT_EQ(json.getInteger("minimum_runtime_scheduling_slot_count"), 4);
  EXPECT_EQ(json.getInteger("maximum_runtime_scheduling_slot_count"), 28);
  EXPECT_EQ(json.getNumber("scheduling_slot_scale"), 96.0);
  EXPECT_EQ(json.getInteger("runtime_kernel_argument_count"), 6);
  EXPECT_EQ(json.getNumber("runtime_argument_setup_system_cycles"), 480.0);
  EXPECT_EQ(json.getNumber("predicted_system_counter_cycles"), 18676.0);
  EXPECT_EQ(json.getNumber("predicted_microseconds"), 18.676);
}

TEST(SimdSimtCostModelTest, SourceFitUsesRegisterExcessAndNonNegativeFeatures) {
  mlir::ascend::SimtSourceFit fit;
  fit.enabled = true;
  fit.intercept = 1.0;
  fit.directInstructionWeight = 2.0;
  fit.segmentWeight = 3.0;
  fit.strideRatioWeight = 4.0;
  fit.maskedWeight = 5.0;
  fit.peakLiveRegisterWeight = 6.0;
  fit.unrollFactorWeight = 7.0;
  EXPECT_TRUE(fit.isValid());
  EXPECT_DOUBLE_EQ(fit.evaluate(2.0, 3.0, 4.0, 5.0, 6.0, 3.0), 105.0);
  EXPECT_DOUBLE_EQ(fit.evaluate(-2.0, -3.0, -4.0, -5.0, -6.0, 1.0), 1.0);
  fit.peakLiveRegisterWeight = -1.0;
  EXPECT_FALSE(fit.isValid());
}

TEST(SimdSimtCostModelTest, ExplicitPrivateTrafficIsPricedSeparately) {
  auto stage =
      logicalStage("private", StageCostModelKind::ContinuousTileMemory);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1};
  stage.workload.privateStackLoadInstructions = 3.0;
  stage.workload.privateStackStoreInstructions = 2.0;
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.privateStackLoadInstructionsPerCycle = 2.0;
  profile.simt.privateStackStoreInstructionsPerCycle = 1.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.privateStackLoad, 1.5);
  EXPECT_DOUBLE_EQ(cost.resources.privateStackStore, 2.0);
  EXPECT_DOUBLE_EQ(cost.resources.load, 1.5);
  EXPECT_DOUBLE_EQ(cost.resources.store, 2.0);
  EXPECT_GT(cost.resources.issue, 0.0);
  ASSERT_FALSE(table->stages.front().simtCalibrationEstimates.empty());
  for (const auto &local : table->stages.front().simtCalibrationEstimates) {
    EXPECT_DOUBLE_EQ(local.resources.privateStackLoad, 1.5);
    EXPECT_DOUBLE_EQ(local.resources.privateStackStore, 2.0);
  }
}

TEST(SimdSimtCostModelTest,
     WholeSimtRecurrencePricesUncoveredRegisterPressureAsPrivateTraffic) {
  auto stage = logicalStage("pressure_recurrence",
                            StageCostModelKind::LoopCarriedRecurrence,
                            StageScheduleKind::LoopCarriedSerial, 1);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2};
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.wholeSimtRegisterPressure = SimtRegisterPressure{};
  stage.wholeSimtRegisterPressure->peakLiveUnitsPerThread = 200;

  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.spillTransactionsPerCycle = 0.1;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 2u);
  // The derived per-thread budget is capped at 127 for both F1 and F2.
  // 73 excess live units at 0.1 transaction/cycle cost 730 cycles per
  // logical program; F2 executes two such programs in the physical group.
  EXPECT_DOUBLE_EQ(costs[0].resources.spill, 730.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.spill, 730.0);
  EXPECT_DOUBLE_EQ(costs[0].totalCycles, 730.0);
  EXPECT_DOUBLE_EQ(costs[1].totalCycles, 1460.0);
  for (const auto &local : table->stages.front().simtCalibrationEstimates)
    EXPECT_DOUBLE_EQ(local.resources.spill, 0.0);

  stage.wholeSimtRegisterPressure->peakLiveUnitsPerThread = 100;
  auto belowBudget = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(belowBudget));
  for (const auto &cost : belowBudget->stages.front().implementations)
    EXPECT_DOUBLE_EQ(cost.resources.spill, 0.0);
}

TEST(SimdSimtCostModelTest,
     DotComponentPrivateStackFloorsSelectExactWholeSimtWarpCountAndOwner) {
  auto stage = logicalStage("component-floor", StageCostModelKind::VectorIssue);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.wholeSimtRegisterPressure = SimtRegisterPressure{};
  auto &facts = *stage.wholeSimtRegisterPressure;
  facts.ttirMaximumWeakDotComponentNodes = 9;
  facts.ttirMaximumWeakDotComponentGeometryKnown = true;
  facts.ttirMaximumWeakDotComponentM = 16;
  facts.ttirMaximumWeakDotComponentN = 16;
  facts.ttirMaximumWeakDotComponentK = 16;
  facts.ttirMaximumWeakDotComponentElementBits = 32;
  facts.ttirFunctionDotUseSignature = "unit-dot-use";
  facts.dotComponentPrivateStackChargeRegion =
      SimtPrivateStackChargeRegion::WholeFunction;
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.dotComponentPrivateStackFloors.reserve(3);
  profile.simt.dotComponentPrivateStackFloors.emplace_back();
  auto &floor = profile.simt.dotComponentPrivateStackFloors.back();
  floor.enabled = true;
  floor.activeWarpCount = 16;
  floor.superblockFactor = 4;
  floor.minimumComponentNodes = 8;
  floor.maximumComponentNodes = 9;
  floor.m = floor.n = floor.k = 16;
  floor.elementBits = 32;
  floor.dotUseSignature = "unit-dot-use";
  floor.loadInstructionFloor = 405;
  floor.storeInstructionFloor = 135;
  floor.combinedGroupCycles = 9000.0;
  floor.preRecurrenceFraction = 0.014111943244627112;
  floor.dotRegionFraction = 0.7509779704846496;
  floor.postDotFraction = 0.23491008627072332;
  floor.evidence = "frozen independent holdouts";
  profile.simt.dotComponentPrivateStackFloors.push_back(floor);
  auto &floorF2 = profile.simt.dotComponentPrivateStackFloors.back();
  floorF2.activeWarpCount = 8;
  floorF2.superblockFactor = 2;
  floorF2.loadInstructionFloor = 204;
  floorF2.storeInstructionFloor = 84;
  floorF2.combinedGroupCycles = 6000.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &implementations = table->stages.front().implementations;
  auto f1 = llvm::find_if(implementations, [](const auto &cost) {
    return cost.implementation.mode == StageMode::SIMT &&
           cost.implementation.superblockFactor == 1;
  });
  auto f4 = llvm::find_if(implementations, [](const auto &cost) {
    return cost.implementation.mode == StageMode::SIMT &&
           cost.implementation.superblockFactor == 4;
  });
  auto f2 = llvm::find_if(implementations, [](const auto &cost) {
    return cost.implementation.mode == StageMode::SIMT &&
           cost.implementation.superblockFactor == 2;
  });
  ASSERT_NE(f1, implementations.end());
  ASSERT_NE(f2, implementations.end());
  ASSERT_NE(f4, implementations.end());
  EXPECT_FALSE(f1->resources.dotComponentPrivateStackFloorApplied);
  EXPECT_DOUBLE_EQ(f1->resources.privateStackLoad, 0.0);
  EXPECT_TRUE(f2->resources.dotComponentPrivateStackFloorApplied);
  EXPECT_DOUBLE_EQ(f2->resources.privateStackCombinedFloor, 3000.0);
  EXPECT_TRUE(f4->resources.dotComponentPrivateStackFloorApplied);
  EXPECT_DOUBLE_EQ(f4->resources.privateStackLoad, 0.0);
  EXPECT_DOUBLE_EQ(f4->resources.privateStackStore, 0.0);
  EXPECT_DOUBLE_EQ(f4->resources.privateStackCombinedFloor, 2250.0);
  EXPECT_NE(f4->formulaEvidence.find(
                "conditional_dot_component_private_stack_combined_group_floor"),
            std::string::npos);
  for (const auto &local : table->stages.front().simtCalibrationEstimates)
    EXPECT_FALSE(local.resources.dotComponentPrivateStackFloorApplied);

  auto alternateF2 = floorF2;
  alternateF2.dotUseSignature = "alternate-dot-use";
  alternateF2.combinedGroupCycles = 7000.0;
  profile.simt.dotComponentPrivateStackFloors.push_back(alternateF2);
  stage.wholeSimtRegisterPressure->ttirFunctionDotUseSignature =
      "alternate-dot-use";
  auto alternate = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(alternate));
  auto alternateCost = llvm::find_if(
      alternate->stages.front().implementations, [](const auto &cost) {
        return cost.implementation.mode == StageMode::SIMT &&
               cost.implementation.superblockFactor == 2;
      });
  ASSERT_NE(alternateCost, alternate->stages.front().implementations.end());
  EXPECT_TRUE(alternateCost->resources.dotComponentPrivateStackFloorApplied);
  EXPECT_DOUBLE_EQ(alternateCost->resources.privateStackCombinedFloor, 3500.0);
  stage.wholeSimtRegisterPressure->ttirFunctionDotUseSignature = "unit-dot-use";

  const auto expectF2Region = [&](SimtPrivateStackChargeRegion region,
                                  double fraction) {
    stage.wholeSimtRegisterPressure->dotComponentPrivateStackChargeRegion =
        region;
    auto regional = evaluateOneStage(stage, profile);
    ASSERT_TRUE(bool(regional));
    const auto &regionalImplementations =
        regional->stages.front().implementations;
    auto regionalF2 =
        llvm::find_if(regionalImplementations, [](const auto &cost) {
          return cost.implementation.mode == StageMode::SIMT &&
                 cost.implementation.superblockFactor == 2;
        });
    ASSERT_NE(regionalF2, regionalImplementations.end());
    EXPECT_TRUE(regionalF2->resources.dotComponentPrivateStackFloorApplied);
    EXPECT_DOUBLE_EQ(regionalF2->resources.privateStackCombinedFloor,
                     3000.0 * fraction);
  };
  expectF2Region(SimtPrivateStackChargeRegion::BeforeRecurrence,
                 floor.preRecurrenceFraction);
  expectF2Region(SimtPrivateStackChargeRegion::DotRegion,
                 floor.dotRegionFraction);
  expectF2Region(SimtPrivateStackChargeRegion::AfterDot, floor.postDotFraction);

  stage.wholeSimtRegisterPressure->ttirMaximumWeakDotComponentNodes = 10;
  auto outside = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(outside));
  for (const auto &cost : outside->stages.front().implementations)
    EXPECT_FALSE(cost.resources.dotComponentPrivateStackFloorApplied);

  stage.wholeSimtRegisterPressure->ttirMaximumWeakDotComponentNodes = 9;
  stage.wholeSimtRegisterPressure->ttirFunctionDotUseSignature =
      "different-dot-use";
  auto differentUse = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(differentUse));
  for (const auto &cost : differentUse->stages.front().implementations)
    EXPECT_FALSE(cost.resources.dotComponentPrivateStackFloorApplied);

  // The paired private-stack residual is incremental above the existing
  // Stage envelope. In particular, a measured continuous-memory curve must
  // not hide it inside that Stage kind's resource max.
  stage.wholeSimtRegisterPressure->ttirFunctionDotUseSignature = "unit-dot-use";
  stage.wholeSimtRegisterPressure->dotComponentPrivateStackChargeRegion =
      SimtPrivateStackChargeRegion::WholeFunction;
  stage.costModelKind = StageCostModelKind::ContinuousTileMemory;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loadBytes = 1024.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.continuousLoadCounts[{/*bytes=*/1024, /*elements=*/256}] = 1.0;
  profile.simt.continuousMemoryGroupBaseWarpCount = 4;
  profile.simt.continuousLoadGroupCyclesByFactorAndBytes = {
      {1, {{1024, 100.0}}}, {2, {{1024, 150.0}}}, {4, {{1024, 220.0}}}};
  profile.simt.continuousLoadGroupStartupCyclesByFactorAndBytes = {
      {1, {{1024, 10.0}}}, {2, {{1024, 15.0}}}, {4, {{1024, 20.0}}}};
  auto withResidual = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(withResidual));
  const auto findF2 = [](const StageCostTable &table) {
    return llvm::find_if(table.stages.front().implementations,
                         [](const auto &cost) {
                           return cost.implementation.mode == StageMode::SIMT &&
                                  cost.implementation.superblockFactor == 2;
                         });
  };
  auto withResidualF2 = findF2(*withResidual);
  ASSERT_NE(withResidualF2, withResidual->stages.front().implementations.end());
  auto unitF2 = llvm::find_if(
      profile.simt.dotComponentPrivateStackFloors, [](const auto &candidate) {
        return candidate.activeWarpCount == 8 &&
               candidate.dotUseSignature == "unit-dot-use";
      });
  ASSERT_NE(unitF2, profile.simt.dotComponentPrivateStackFloors.end());
  unitF2->enabled = false;
  auto withoutResidual = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(withoutResidual));
  auto withoutResidualF2 = findF2(*withoutResidual);
  ASSERT_NE(withoutResidualF2,
            withoutResidual->stages.front().implementations.end());
  EXPECT_DOUBLE_EQ(withResidualF2->totalCycles - withoutResidualF2->totalCycles,
                   6000.0);

  auto overlapping = alternateF2;
  overlapping.maximumComponentNodes += 1;
  profile.simt.dotComponentPrivateStackFloors.push_back(overlapping);
  EXPECT_FALSE(profile.simt.isValid(StageMode::SIMT));
}

TEST(SimdSimtCostModelTest, PrivateSourceFitUsesWholeVfPeakExcess) {
  auto stage =
      logicalStage("fitted-private", StageCostModelKind::ContinuousTileMemory);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1};
  StageMemoryAccess access;
  access.shape = {16, 16};
  access.strides = {2048, 1};
  access.elementBits = 32;
  access.masked = true;
  stage.workload.memoryAccesses = {access};
  stage.wholeSimtRegisterPressure = SimtRegisterPressure{};
  stage.wholeSimtRegisterPressure->peakLiveUnitsPerThread = 72;
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 16;
  profile.simt.privateStackLoadInstructionsPerCycle = 2.0;
  profile.simt.privateLoadSourceFit.enabled = true;
  profile.simt.privateLoadSourceFit.peakLiveRegisterWeight = 1.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  // Derived budget for 16 active warps is 64; 72 live units leave 8 excess
  // units, and no semantic GM load is charged by this synthetic Stage.
  EXPECT_DOUBLE_EQ(cost.resources.privateStackLoad, 4.0);
  EXPECT_DOUBLE_EQ(cost.resources.load, 4.0);
  ASSERT_FALSE(table->stages.front().simtCalibrationEstimates.empty());
  for (const auto &local : table->stages.front().simtCalibrationEstimates) {
    EXPECT_DOUBLE_EQ(local.resources.privateStackLoad, 0.0);
    EXPECT_DOUBLE_EQ(local.resources.load, 0.0);
  }
}

TEST(SimdSimtCostModelTest,
     LoweringEvidenceOverridesTtirBoundAndCarriesUnrollFeature) {
  auto stage = logicalStage("lowering-evidence",
                            StageCostModelKind::ContinuousTileMemory);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1};
  stage.wholeSimtRegisterPressure = SimtRegisterPressure{};
  stage.wholeSimtRegisterPressure->peakLiveUnitsPerThread = 40;
  stage.wholeSimtRegisterPressure->loweredPeakLiveUnitsPerThread = 80;
  stage.wholeSimtRegisterPressure->backendUnrollFactor = 4;
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 16;
  profile.simt.privateStackLoadInstructionsPerCycle = 2.0;
  profile.simt.privateLoadSourceFit.enabled = true;
  profile.simt.privateLoadSourceFit.peakLiveRegisterWeight = 1.0;
  profile.simt.privateLoadSourceFit.unrollFactorWeight = 2.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  // Budget=64 for 16 warps: max(40,80)-64 + 2*(4-1) = 22 generated
  // private loads, priced at two instructions per cycle.
  EXPECT_DOUBLE_EQ(cost.resources.privateStackLoad, 11.0);
  EXPECT_DOUBLE_EQ(cost.resources.load, 11.0);
}

TEST(SimdSimtCostModelTest, SimtPressureSeesValuesRetainedAcrossOtherStages) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel() {
      %held = "test.source"() : () -> tensor<256xf32>
      %lo = arith.constant 0 : index
      %hi = arith.constant 16 : index
      %step = arith.constant 1 : index
      scf.for %i = %lo to %hi step %step {
        %tmp = "test.source"() : () -> tensor<256xf32>
        "test.sink"(%tmp) : (tensor<256xf32>) -> ()
      }
      "test.sink"(%held) : (tensor<256xf32>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  StagePartition partition;
  for (mlir::Operation &op : function.getBody().front()) {
    auto stage = logicalStage(std::to_string(partition.stages.size()),
                              StageCostModelKind::VectorIssue);
    stage.operations.push_back(&op);
    partition.stages.push_back(std::move(stage));
  }
  for (int repetition = 0; repetition < 2; ++repetition) {
    if (llvm::Error error =
            mlir::ascend::SimtRegisterPressureAnalysis().analyze(*module,
                                                                 partition, 4))
      FAIL() << llvm::toString(std::move(error));
    const auto &facts = partition.stages[4].wholeSimtRegisterPressure;
    ASSERT_TRUE(facts);
    EXPECT_EQ(facts->peakCrossStageUnitsPerThread, 2);
    EXPECT_GE(facts->peakTensorUnitsPerThread, 4);
    EXPECT_EQ(facts->baseWarpCount, 4);
    EXPECT_EQ(facts->unknownValueCount, 0);
  }
}

TEST(SimdSimtCostModelTest, SimtPressureKeepsImplicitLoopControlLive) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel(%hi: i32, %step: i32) {
      %lo = arith.constant 0 : i32
      scf.for %iv = %lo to %hi step %step : i32 {
        %tile = "test.source"() : () -> tensor<256xf32>
        "test.sink"(%tile) : (tensor<256xf32>) -> ()
      }
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  auto stage = logicalStage("loop", StageCostModelKind::VectorIssue);
  module->walk([&](mlir::scf::ForOp loop) {
    stage.operations.push_back(loop.getOperation());
  });
  partition.stages.push_back(std::move(stage));
  if (auto error = mlir::ascend::SimtRegisterPressureAnalysis().analyze(
          *module, partition, 4))
    FAIL() << llvm::toString(std::move(error));
  const auto &facts = *partition.stages[0].wholeSimtRegisterPressure;
  EXPECT_EQ(facts.peakScalarUnitsPerThread, 3); // bound, step, induction
  EXPECT_EQ(facts.peakTensorUnitsPerThread, 2);
  EXPECT_EQ(facts.toJSON().getBoolean("spill_cost_included"), false);
}

TEST(SimdSimtCostModelTest, SimtPressureReportsTtirPredicateDagStructure) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel() {
      %lhs = "test.source"() : () -> i64
      %rhs = "test.source"() : () -> i64
      %p0 = arith.cmpi ult, %lhs, %rhs : i64
      %p1 = arith.cmpi eq, %lhs, %rhs : i64
      %both = arith.andi %p0, %p1 : i1
      %selected = arith.select %both, %lhs, %rhs : i64
      "test.sink"(%selected) : (i64) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  auto stage = logicalStage("whole", StageCostModelKind::VectorIssue);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  stage.operations.push_back(function.getOperation());
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = mlir::ascend::SimtRegisterPressureAnalysis().analyze(
          *module, partition, 4))
    FAIL() << llvm::toString(std::move(error));
  const auto &facts = *partition.stages[0].wholeSimtRegisterPressure;
  EXPECT_EQ(facts.predicateDefinitionUnitsPerThread, 3);
  EXPECT_EQ(facts.ttirIntegerEqualityPredicateDefinitionUnitsPerThread, 1);
  EXPECT_EQ(facts.ttirIntegerOrderedPredicateDefinitionUnitsPerThread, 1);
  EXPECT_EQ(facts.ttirPredicateShapeDefinitionUnitsPerThread, 0);
  EXPECT_EQ(facts.ttirOtherPredicateDefinitionUnitsPerThread, 1);
  EXPECT_EQ(facts.ttirPredicateDagEdgeUnitsPerThread, 2);
  EXPECT_EQ(facts.ttirMaxPredicateDagDepth, 2);
  EXPECT_EQ(facts.ttirBooleanPredicateUseUnitsPerThread, 2);
  EXPECT_EQ(facts.ttirSelectPredicateUseUnitsPerThread, 1);
  auto json = facts.toJSON();
  EXPECT_EQ(json.getInteger("ttir_max_predicate_dag_depth"), 2);
  EXPECT_EQ(json.getBoolean("spill_cost_included"), false);
}

TEST(SimdSimtCostModelTest, SimtPressureSeparatesPredicateShapeExpansion) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel() {
      %lhs = "test.source"() : () -> tensor<128xi32>
      %rhs = "test.source"() : () -> tensor<128xi32>
      %p = arith.cmpi eq, %lhs, %rhs : tensor<128xi32>
      %expanded = "tt.expand_dims"(%p) {axis = 1 : i32}
        : (tensor<128xi1>) -> tensor<128x1xi1>
      %broadcast = "tt.broadcast"(%expanded)
        : (tensor<128x1xi1>) -> tensor<128x2xi1>
      %yes = "test.source"() : () -> tensor<128x2xi32>
      %no = "test.source"() : () -> tensor<128x2xi32>
      %selected = arith.select %broadcast, %yes, %no
        : tensor<128x2xi1>, tensor<128x2xi32>
      "test.sink"(%selected) : (tensor<128x2xi32>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  auto stage = logicalStage("whole", StageCostModelKind::VectorIssue);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  stage.operations.push_back(function.getOperation());
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = mlir::ascend::SimtRegisterPressureAnalysis().analyze(
          *module, partition, 4))
    FAIL() << llvm::toString(std::move(error));
  const auto &facts = *partition.stages[0].wholeSimtRegisterPressure;
  EXPECT_EQ(facts.predicateDefinitionUnitsPerThread, 4);
  EXPECT_EQ(facts.ttirIntegerEqualityPredicateDefinitionUnitsPerThread, 1);
  EXPECT_EQ(facts.ttirPredicateShapeDefinitionUnitsPerThread, 3);
  EXPECT_EQ(facts.ttirOtherPredicateDefinitionUnitsPerThread, 0);
  EXPECT_EQ(facts.ttirPredicateDagEdgeUnitsPerThread, 3);
  EXPECT_EQ(facts.ttirMaxPredicateDagDepth, 3);
  EXPECT_EQ(facts.predicateUseUnitsPerThread, 4);
  EXPECT_EQ(facts.ttirSelectPredicateUseUnitsPerThread, 2);
  EXPECT_EQ(facts.toJSON().getInteger(
                "ttir_predicate_shape_definition_units_per_thread"),
            3);
}

TEST(SimdSimtCostModelTest,
     SimtPressureReportsPredicateFanoutAndLoopInvariantUses) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel() {
      %predicate = "test.predicate"() : () -> tensor<128xi1>
      %lo = arith.constant 0 : index
      %hi = arith.constant 4 : index
      %step = arith.constant 1 : index
      scf.for %i = %lo to %hi step %step {
        "test.first_sink"(%predicate) : (tensor<128xi1>) -> ()
        "test.second_sink"(%predicate) : (tensor<128xi1>) -> ()
      }
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  auto stage = logicalStage("whole", StageCostModelKind::VectorIssue);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  stage.operations.push_back(function.getOperation());
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = mlir::ascend::SimtRegisterPressureAnalysis().analyze(
          *module, partition, 4))
    FAIL() << llvm::toString(std::move(error));
  const auto &facts = *partition.stages[0].wholeSimtRegisterPressure;
  EXPECT_EQ(facts.predicateDefinitionUnitsPerThread, 1);
  EXPECT_EQ(facts.predicateUseUnitsPerThread, 2);
  EXPECT_EQ(facts.maxPredicateFanout, 2);
  EXPECT_EQ(facts.loopInvariantPredicateUnitsPerThread, 1);
  EXPECT_EQ(facts.loopInvariantPredicateUseUnitsPerThread, 2);
  EXPECT_EQ(facts.peakLivePredicateUnitsPerThread, 1);
  EXPECT_EQ(facts.toJSON().getBoolean("spill_cost_included"), false);
}

TEST(SimdSimtCostModelTest,
     SimtPressureDeduplicatesShapeAliasesAndDoesNotScaleByF) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel() {
      %v = "test.source"() : () -> tensor<256xf32>
      %alias = "tt.expand_dims"(%v) : (tensor<256xf32>) -> tensor<1x256xf32>
      "test.sink"(%v, %alias) : (tensor<256xf32>, tensor<1x256xf32>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  auto stage = logicalStage("one_vf", StageCostModelKind::VectorIssue);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  for (mlir::Operation &op : function.getBody().front())
    stage.operations.push_back(&op);
  stage.legalSimtFactors = {1, 2, 4};
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = mlir::ascend::SimtRegisterPressureAnalysis().analyze(
          *module, partition, 4))
    FAIL() << llvm::toString(std::move(error));
  ASSERT_TRUE(partition.stages[0].wholeSimtRegisterPressure);
  EXPECT_EQ(
      partition.stages[0].wholeSimtRegisterPressure->peakLiveUnitsPerThread, 2);
  auto result = StageCostEvaluator().evaluate(partition, hardwareProfile());
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->stages[0].wholeSimtRegisterPressure->peakLiveUnitsPerThread,
            2);
  auto json = result->stages[0].toJSON();
  ASSERT_NE(json.getObject("whole_simt_register_pressure"), nullptr);
  EXPECT_EQ(json.getObject("whole_simt_register_pressure")
                ->getBoolean("spill_cost_included"),
            false);
}

TEST(SimdSimtCostModelTest, SimtPressureKeepsSchedulingShellSeparateFromBody) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel() {
      %lo = arith.constant 0 : index
      %hi = arith.constant 16 : index
      %step = arith.constant 1 : index
      scf.for %i = %lo to %hi step %step {
        %v = "test.source"() : () -> tensor<256xf32>
        "test.sink"(%v) : (tensor<256xf32>) -> ()
      }
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::scf::ForOp loop;
  module->walk([&](mlir::scf::ForOp op) { loop = op; });
  ASSERT_TRUE(loop);
  auto i64 = mlir::IntegerType::get(&context, 64);
  loop->setAttr("ascend.simt.lowered_peak_live_register_units",
                mlir::IntegerAttr::get(i64, 80));
  loop->setAttr("ascend.simt.unroll_factor", mlir::IntegerAttr::get(i64, 4));
  StagePartition partition;
  auto shell = logicalStage("shell", StageCostModelKind::AutoBlockifyLoop);
  shell.operations.push_back(loop.getOperation());
  partition.stages.push_back(std::move(shell));
  for (mlir::Operation &op : *loop.getBody()) {
    auto body = logicalStage(std::to_string(partition.stages.size()),
                             StageCostModelKind::VectorIssue);
    body.operations.push_back(&op);
    partition.stages.push_back(std::move(body));
  }
  if (llvm::Error error = mlir::ascend::SimtRegisterPressureAnalysis().analyze(
          *module, partition, 4))
    FAIL() << llvm::toString(std::move(error));
  ASSERT_TRUE(partition.stages[0].wholeSimtRegisterPressure);
  EXPECT_EQ(partition.stages[0]
                .wholeSimtRegisterPressure->loweredPeakLiveUnitsPerThread,
            80);
  EXPECT_EQ(partition.stages[0].wholeSimtRegisterPressure->backendUnrollFactor,
            4);
  ASSERT_TRUE(partition.stages[2].wholeSimtRegisterPressure);
  EXPECT_EQ(partition.stages[2]
                .wholeSimtRegisterPressure->peakCrossStageUnitsPerThread,
            2);
  EXPECT_EQ(
      partition.stages[2].wholeSimtRegisterPressure->peakTensorUnitsPerThread,
      2);
}

TEST(SimdSimtCostModelTest, SimtPressureKeepsScalarReplicationAndUnknownTypes) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel(%scalar: i64, %unknown: tensor<?xf32>) {
      %v = "test.source"() : () -> tensor<256xi64>
      "test.sink"(%v, %scalar, %unknown) : (tensor<256xi64>, i64, tensor<?xf32>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  auto stage = logicalStage("whole", StageCostModelKind::VectorIssue);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  for (mlir::Operation &op : function.getBody().front())
    stage.operations.push_back(&op);
  partition.stages.push_back(std::move(stage));
  for (int64_t warps : {4, 8}) {
    if (llvm::Error error =
            mlir::ascend::SimtRegisterPressureAnalysis().analyze(
                *module, partition, warps))
      FAIL() << llvm::toString(std::move(error));
    const auto &facts = *partition.stages[0].wholeSimtRegisterPressure;
    EXPECT_EQ(facts.peakTensorUnitsPerThread, warps == 4 ? 4 : 2);
    EXPECT_EQ(facts.peakScalarUnitsPerThread, 2);
    EXPECT_EQ(facts.unknownValueCount, 1);
  }
}

TEST(SimdSimtCostModelTest, SimtPressureKeepsBlockPointerExpansionUnresolved) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::triton::TritonDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel(%block: !tt.ptr<tensor<16x16xf32>>,
                              %lanes: tensor<128x!tt.ptr<f32>>) {
      "test.sink"(%block, %lanes)
        : (!tt.ptr<tensor<16x16xf32>>, tensor<128x!tt.ptr<f32>>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  auto stage = logicalStage("whole", StageCostModelKind::VectorIssue);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  stage.operations.push_back(&function.getBody().front().front());
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = mlir::ascend::SimtRegisterPressureAnalysis().analyze(
          *module, partition, 4))
    FAIL() << llvm::toString(std::move(error));
  const auto &facts = *partition.stages[0].wholeSimtRegisterPressure;
  EXPECT_EQ(facts.peakBlockPointerCount, 1);
  EXPECT_EQ(facts.peakBlockPointerRankSum, 2);
  EXPECT_EQ(facts.peakPointerUnitsPerThread, 4);
  EXPECT_EQ(facts.peakLiveUnitsPerThread, 4);
  EXPECT_EQ(facts.toJSON().getBoolean("spill_cost_included"), false);
}

TEST(SimdSimtCostModelTest, SimtPressureSamplesRetainedStateAtTheDot) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel() {
      %old = "test.source"() : () -> tensor<4096xf32>
      "test.sink"(%old) : (tensor<4096xf32>) -> ()
      %a = "test.source"() : () -> tensor<16x16xf32>
      %b = "test.source"() : () -> tensor<16x16xf32>
      %held = "test.source"() : () -> tensor<256xf32>
      %alias = "tt.reshape"(%held) : (tensor<256xf32>) -> tensor<16x16xf32>
      %zero = arith.constant dense<0.0> : tensor<16x16xf32>
      %dot = "tt.dot"(%a, %b, %zero)
          : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      "test.sink"(%dot, %held, %alias)
          : (tensor<16x16xf32>, tensor<256xf32>, tensor<16x16xf32>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  auto stage = logicalStage("whole", StageCostModelKind::VectorIssue);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  stage.operations.push_back(function.getOperation());
  // Composite ownership may expose both a region root and its child.
  // Context samples must still describe each dot exactly once.
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "tt.dot")
      stage.operations.push_back(op);
  });
  partition.stages.push_back(std::move(stage));
  for (int repeat = 0; repeat < 2; ++repeat) {
    if (llvm::Error error =
            mlir::ascend::SimtRegisterPressureAnalysis().analyze(*module,
                                                                 partition, 4))
      FAIL() << llvm::toString(std::move(error));
    const auto &facts = *partition.stages[0].wholeSimtRegisterPressure;
    EXPECT_GE(facts.peakLiveUnitsPerThread, 32);
    ASSERT_EQ(facts.dotSites.size(), 1u);
    const auto &dot = facts.dotSites.front();
    EXPECT_EQ(dot.m, 16);
    EXPECT_EQ(dot.n, 16);
    EXPECT_EQ(dot.k, 16);
    EXPECT_EQ(dot.elementBits, 32);
    EXPECT_EQ(dot.operandUnitsPerThread, 4);
    EXPECT_EQ(dot.retainedTensorUnitsPerThread, 2);
    EXPECT_EQ(dot.retainedScalarUnitsPerThread, 0);
    EXPECT_EQ(facts.ttirDotDefinitionCount, 1);
    EXPECT_EQ(facts.ttirDotDependencyEdgeCount, 0);
    EXPECT_EQ(facts.ttirDotDependencySourceCount, 1);
    EXPECT_EQ(facts.ttirDotDependencySinkCount, 1);
    EXPECT_EQ(facts.ttirMaxDotDependencyDepth, 1);
    auto json = facts.toJSON();
    ASSERT_NE(json.getArray("dot_live_contexts"), nullptr);
    EXPECT_EQ(json.getArray("dot_live_contexts")->size(), 1u);
    EXPECT_EQ(json.getBoolean("spill_cost_included"), false);
  }
}

TEST(SimdSimtCostModelTest, SimtPressureReportsDotDependencyDag) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel() {
      %a = "test.source"() : () -> tensor<16x16xf32>
      %b = "test.source"() : () -> tensor<16x16xf32>
      %zero = arith.constant dense<0.0> : tensor<16x16xf32>
      %dot0 = "tt.dot"(%a, %b, %zero)
          : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      %neg = arith.subf %zero, %dot0 : tensor<16x16xf32>
      %dot1 = "tt.dot"(%neg, %b, %zero)
          : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      %dot2 = "tt.dot"(%a, %b, %zero)
          : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      "test.sink"(%dot1, %dot2)
          : (tensor<16x16xf32>, tensor<16x16xf32>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  auto stage = logicalStage("whole", StageCostModelKind::VectorIssue);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  stage.operations.push_back(function.getOperation());
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = mlir::ascend::SimtRegisterPressureAnalysis().analyze(
          *module, partition, 4))
    FAIL() << llvm::toString(std::move(error));
  const auto &facts = *partition.stages[0].wholeSimtRegisterPressure;
  EXPECT_EQ(facts.ttirDotDefinitionCount, 3);
  EXPECT_EQ(facts.ttirDotDependencyEdgeCount, 1);
  EXPECT_EQ(facts.ttirDotDependencySourceCount, 2);
  EXPECT_EQ(facts.ttirDotDependencySinkCount, 2);
  EXPECT_EQ(facts.ttirMaxDotDependencyDepth, 2);
  EXPECT_EQ(facts.ttirMaximumWeakDotComponentNodes, 2);
  EXPECT_TRUE(facts.ttirMaximumWeakDotComponentGeometryKnown);
  EXPECT_EQ(facts.ttirMaximumWeakDotComponentM, 16);
  EXPECT_EQ(facts.ttirMaximumWeakDotComponentN, 16);
  EXPECT_EQ(facts.ttirMaximumWeakDotComponentK, 16);
  EXPECT_EQ(facts.ttirMaximumWeakDotComponentElementBits, 32);
  EXPECT_EQ(facts.dotComponentPrivateStackChargeRegion,
            SimtPrivateStackChargeRegion::WholeFunction);
  ASSERT_EQ(facts.dotDependencySites.size(), 3u);
  EXPECT_EQ(facts.dotDependencySites[0].ordinal, 0);
  EXPECT_TRUE(facts.dotDependencySites[0].predecessorOrdinals.empty());
  EXPECT_EQ(facts.dotDependencySites[0].successorCount, 1);
  EXPECT_EQ(facts.dotDependencySites[0].nonDotPureConsumerCount, 1);
  EXPECT_EQ(facts.dotDependencySites[0].sideEffectFrontierCount, 0);
  EXPECT_GT(facts.dotDependencySites[0].maxNonDotPureUseSpan, 0);
  EXPECT_EQ(facts.dotDependencySites[0].maxSideEffectUseSpan, 0);
  EXPECT_EQ(facts.dotDependencySites[1].ordinal, 1);
  EXPECT_EQ(facts.dotDependencySites[1].predecessorOrdinals,
            std::vector<int64_t>({0}));
  EXPECT_EQ(facts.dotDependencySites[1].depth, 2);
  EXPECT_EQ(facts.dotDependencySites[1].nonDotPureConsumerCount, 0);
  EXPECT_EQ(facts.dotDependencySites[1].sideEffectFrontierCount, 1);
  EXPECT_GT(facts.dotDependencySites[1].maxSideEffectUseSpan, 0);
  EXPECT_EQ(facts.dotDependencySites[2].ordinal, 2);
  EXPECT_TRUE(facts.dotDependencySites[2].predecessorOrdinals.empty());
  EXPECT_EQ(facts.dotDependencySites[2].successorCount, 0);
  EXPECT_EQ(facts.dotDependencySites[2].sideEffectFrontierCount, 1);
  auto json = facts.toJSON();
  EXPECT_EQ(json.getInteger("ttir_dot_definition_count"), 3);
  EXPECT_EQ(json.getInteger("ttir_dot_dependency_edge_count"), 1);
  EXPECT_EQ(json.getInteger("ttir_dot_dependency_source_count"), 2);
  EXPECT_EQ(json.getInteger("ttir_dot_dependency_sink_count"), 2);
  EXPECT_EQ(json.getInteger("ttir_max_dot_dependency_depth"), 2);
  EXPECT_EQ(json.getInteger("ttir_maximum_weak_dot_component_nodes"), 2);
  EXPECT_EQ(json.getBoolean("ttir_maximum_weak_dot_component_geometry_known"),
            true);
  EXPECT_EQ(json.getBoolean("dot_component_private_stack_charge_owner"), true);
  ASSERT_NE(json.getArray("ttir_dot_dependency_sites"), nullptr);
  EXPECT_EQ(json.getArray("ttir_dot_dependency_sites")->size(), 3u);
  EXPECT_EQ(json.getBoolean("spill_cost_included"), false);
}

TEST(SimdSimtCostModelTest, SimtPressureStopsDotDagAtUnknownSideEffects) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel() {
      %a = "test.source"() : () -> tensor<16x16xf32>
      %b = "test.source"() : () -> tensor<16x16xf32>
      %zero = arith.constant dense<0.0> : tensor<16x16xf32>
      %dot0 = "tt.dot"(%a, %b, %zero)
          : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      %opaque = "test.unknown_effect"(%dot0)
          : (tensor<16x16xf32>) -> tensor<16x16xf32>
      %dot1 = "tt.dot"(%opaque, %b, %zero)
          : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      "test.sink"(%dot1) : (tensor<16x16xf32>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  StagePartition partition;
  auto stage = logicalStage("whole", StageCostModelKind::VectorIssue);
  auto function = *module->getOps<mlir::func::FuncOp>().begin();
  stage.operations.push_back(function.getOperation());
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = mlir::ascend::SimtRegisterPressureAnalysis().analyze(
          *module, partition, 4))
    FAIL() << llvm::toString(std::move(error));
  const auto &facts = *partition.stages[0].wholeSimtRegisterPressure;
  EXPECT_EQ(facts.ttirDotDefinitionCount, 2);
  EXPECT_EQ(facts.ttirDotDependencyEdgeCount, 0);
  EXPECT_EQ(facts.ttirDotDependencySourceCount, 2);
  EXPECT_EQ(facts.ttirDotDependencySinkCount, 2);
  EXPECT_EQ(facts.ttirMaxDotDependencyDepth, 1);
  ASSERT_EQ(facts.dotDependencySites.size(), 2u);
  EXPECT_EQ(facts.dotDependencySites[0].nonDotPureConsumerCount, 0);
  EXPECT_EQ(facts.dotDependencySites[0].sideEffectFrontierCount, 1);
  EXPECT_GT(facts.dotDependencySites[0].maxSideEffectUseSpan, 0);
  EXPECT_TRUE(facts.dotDependencySites[1].predecessorOrdinals.empty());
}

TEST(SimdSimtCostModelTest,
     DotUseSignatureNormalizesUniformSideEffectOrdinalTranslation) {
  const auto signature = [](llvm::StringRef text) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.allowUnregisteredDialects();
    auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
    if (!module)
      return std::string();
    StagePartition partition;
    auto stage = logicalStage("whole", StageCostModelKind::VectorIssue);
    auto function = *module->getOps<mlir::func::FuncOp>().begin();
    stage.operations.push_back(function.getOperation());
    partition.stages.push_back(std::move(stage));
    if (llvm::Error error =
            mlir::ascend::SimtRegisterPressureAnalysis().analyze(
                *module, partition, 4)) {
      llvm::consumeError(std::move(error));
      return std::string();
    }
    return partition.stages[0]
        .wholeSimtRegisterPressure->ttirFunctionDotUseSignature;
  };

  const std::string base = signature(R"mlir(
    module { func.func @kernel() {
      %a = "test.source"() : () -> tensor<16x16xf32>
      %b = "test.source"() : () -> tensor<16x16xf32>
      %zero = arith.constant dense<0.0> : tensor<16x16xf32>
      %dot0 = "tt.dot"(%a, %b, %zero)
          : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      %dot1 = "tt.dot"(%b, %a, %zero)
          : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      "test.sink"(%dot0) : (tensor<16x16xf32>) -> ()
      %u = "test.unrelated"() : () -> i32
      "test.sink"(%dot1) : (tensor<16x16xf32>) -> ()
      return
    } })mlir");
  const std::string translated = signature(R"mlir(
    module { func.func @kernel() {
      %a = "test.source"() : () -> tensor<16x16xf32>
      %b = "test.source"() : () -> tensor<16x16xf32>
      %zero = arith.constant dense<0.0> : tensor<16x16xf32>
      %dot0 = "tt.dot"(%a, %b, %zero)
          : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      %dot1 = "tt.dot"(%b, %a, %zero)
          : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>)
          -> tensor<16x16xf32>
      %prefix0 = "test.unrelated"() : () -> i32
      %prefix1 = "test.unrelated"() : () -> i32
      %prefix2 = "test.unrelated"() : () -> i32
      "test.sink"(%dot0) : (tensor<16x16xf32>) -> ()
      %u = "test.unrelated"() : () -> i32
      "test.sink"(%dot1) : (tensor<16x16xf32>) -> ()
      return
    } })mlir");
  ASSERT_FALSE(base.empty());
  ASSERT_FALSE(translated.empty());
  EXPECT_EQ(base, translated);
}

TEST(SimdSimtCostModelTest, CvSubblocksPartitionPayloadNotControlOrIterations) {
  auto stage = logicalStage("payload", StageCostModelKind::VectorIssue);
  stage.iterationCount = 14;
  stage.legalSimtFactors = {1, 4};
  stage.localSimtMaterializable = true;
  stage.localSimtFactors = {1, 4};
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["f32.add"][128] = 1;
  stage.workload.scalarOperations = 3;
  stage.workload.loopBackedges = 1;
  stage.workload.loadBytes = 512;
  stage.workload.loadOperations = 1;
  stage.workload.loadWarpInstructions = 4;
  StageMemoryAccess access;
  access.shape = {128};
  access.strides = {1};
  access.elementBits = 32;
  stage.workload.memoryAccesses = {access};
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  auto before = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(before));
  profile.cvAivSubblocks = 2;
  auto after = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(after));
  EXPECT_EQ(after->stages[0].iterationCount, 14);
  EXPECT_DOUBLE_EQ(after->stages[0].workload.loadBytes, 512);
  for (const auto &old : before->stages[0].implementations) {
    auto same =
        llvm::find_if(after->stages[0].implementations, [&](const auto &c) {
          return c.implementation.mode == old.implementation.mode &&
                 c.implementation.localScope == old.implementation.localScope &&
                 c.implementation.superblockFactor ==
                     old.implementation.superblockFactor &&
                 c.implementation.aivSubblocks == 1;
        });
    ASSERT_NE(same, after->stages[0].implementations.end());
    EXPECT_DOUBLE_EQ(same->totalCycles, old.totalCycles);
  }
  for (const auto &cost : after->stages[0].implementations) {
    if (cost.implementation.aivSubblocks != 2)
      continue;
    ASSERT_TRUE(cost.perAivWorkload);
    const auto &work = *cost.perAivWorkload;
    EXPECT_DOUBLE_EQ(work.loadBytes, 256);
    EXPECT_DOUBLE_EQ(work.loadOperations, 1);
    EXPECT_DOUBLE_EQ(work.loadWarpInstructions, 2);
    EXPECT_DOUBLE_EQ(work.scalarOperations, 3);
    EXPECT_DOUBLE_EQ(work.loopBackedges, 1);
    EXPECT_EQ(work.memoryAccesses[0].shape, (std::vector<int64_t>{64}));
    EXPECT_EQ(work.memoryAccesses[0].strides[0], 1);
    EXPECT_DOUBLE_EQ(cost.resources.setup, 10);
    EXPECT_DOUBLE_EQ(cost.resources.compute, 64);
    const double f = cost.implementation.superblockFactor;
    EXPECT_DOUBLE_EQ(cost.totalCycles, 10 + f * 14 * (64 + 2));
  }
  profile.scopeHandoff.cvPartitionPayloadEfficiency = 0.5;
  auto inefficient = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(inefficient));
  for (int64_t factor : {int64_t(1), int64_t(4)}) {
    auto findLocal = [&](int64_t subblocks) {
      return llvm::find_if(
          inefficient->stages[0].implementations, [&](const auto &cost) {
            return cost.implementation.mode == StageMode::SIMT &&
                   cost.implementation.localScope &&
                   cost.implementation.superblockFactor == factor &&
                   cost.implementation.aivSubblocks == subblocks;
          });
    };
    const auto ideal =
        llvm::find_if(after->stages[0].implementations, [&](const auto &cost) {
          return cost.implementation.mode == StageMode::SIMT &&
                 cost.implementation.localScope &&
                 cost.implementation.superblockFactor == factor &&
                 cost.implementation.aivSubblocks == 2;
        });
    const auto full = findLocal(1);
    const auto corrected = findLocal(2);
    ASSERT_NE(ideal, after->stages[0].implementations.end());
    ASSERT_NE(full, inefficient->stages[0].implementations.end());
    ASSERT_NE(corrected, inefficient->stages[0].implementations.end());
    EXPECT_DOUBLE_EQ(corrected->cvPartitionIdealTotalCycles,
                     ideal->totalCycles);
    EXPECT_DOUBLE_EQ(corrected->cvPartitionPayloadEfficiency, 0.5);
    EXPECT_DOUBLE_EQ(corrected->totalCycles,
                     ideal->totalCycles +
                         0.5 * (full->totalCycles - ideal->totalCycles));
    EXPECT_DOUBLE_EQ(corrected->cvPartitionInefficiencyCorrectionCycles,
                     corrected->totalCycles - ideal->totalCycles);
  }
  // Geometry is not evidence of the backend's row/column partition axis.
  stage.workload.memoryAccesses[0].shape = {8, 16};
  stage.workload.memoryAccesses[0].strides = {128, 1};
  auto rankTwo = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(rankTwo));
  for (const auto &cost : rankTwo->stages[0].implementations)
    if (cost.perAivWorkload) {
      EXPECT_DOUBLE_EQ(cost.perAivWorkload->loadBytes, 512);
      EXPECT_EQ(cost.perAivWorkload->memoryAccesses[0].shape,
                (std::vector<int64_t>{8, 16}));
    }
}

TEST(SimdSimtCostModelTest, CvSubblockF4KeepsNineteenProgramsAndThreeF1Tails) {
  StageCostTable table;
  table.logicalProgramCountHint = 512;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.cvAivSubblocks = 2;
  auto cost = [](StageMode mode, int64_t f, double ticks,
                 int64_t subblocks = 1) {
    StageImplementationCost value;
    value.implementation = {mode, f, mode == StageMode::SIMT, subblocks};
    value.totalCycles = ticks;
    return value;
  };
  LogicalStageCost scope;
  scope.id = "scope";
  scope.localSimtMaterializable = true;
  scope.scopeInputTensorBytes = 1024;
  scope.implementations = {
      cost(StageMode::SIMD, 1, 100), cost(StageMode::SIMT, 1, 1),
      cost(StageMode::SIMT, 4, 1), cost(StageMode::SIMT, 1, 20, 2),
      cost(StageMode::SIMT, 4, 10, 2)};
  LogicalStageCost cube;
  cube.id = "cube";
  cube.features.hasDot = true;
  cube.implementations = {cost(StageMode::SIMD, 1, 3)};
  table.stages = {scope, cube};
  ScopeHandoffCost handoff;
  handoff.fixedScopeCycles = 3;
  handoff.cvCrossCoreSynchronizationCycles = 5;
  handoff.inputHandoffBytesPerCycle = 128;
  auto routes = solveStageRoutes(table, handoff);
  ASSERT_TRUE(bool(routes));
  ASSERT_TRUE(routes->mixed.legal);
  const auto &mixed = routes->mixed;
  EXPECT_EQ(mixed.routeSuperblockFactor, 4);
  EXPECT_EQ(mixed.implementations[0].aivSubblocks, 2);
  EXPECT_EQ(mixed.runtimeSchedulingSlotCount, 28);
  EXPECT_EQ(mixed.runtimeLogicalProgramsPerSchedulingSlot, 19);
  EXPECT_EQ(mixed.runtimeFullGroupCount, 4);
  EXPECT_EQ(mixed.runtimeTailProgramCount, 3);
  EXPECT_EQ(mixed.runtimeLoopIterationCount, 7);
  // F1 is only the SuperBlock tail width.  The implementation remains a
  // physical CV1:2 split (`aivSubblocks == 2`), so every one of the seven
  // local-scope launches pays the AIC/AIV coordination envelope.
  EXPECT_DOUBLE_EQ(mixed.scopeHandoffCycles[0], 7 * (3 + 5));
  EXPECT_DOUBLE_EQ(mixed.logicalStageCycles[0],
                   4 * 10 + 3 * 20 + 19 * 4 + 7 * (3 + 5));
  EXPECT_DOUBLE_EQ(mixed.logicalStageCycles[1], 19 * 3);
  // The same tensor Stage without a Cube peer must not borrow CV1:2 work.
  table.stages[1].features.hasDot = false;
  routes = solveStageRoutes(table, handoff);
  ASSERT_TRUE(bool(routes));
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.implementations[0].aivSubblocks, 1);
  EXPECT_EQ(routes->mixed.runtimeSchedulingSlotCount, 56);
  EXPECT_EQ(routes->mixed.runtimeLogicalProgramsPerSchedulingSlot, 10);
}

TEST(SimdSimtCostModelTest, SimtBodyEstimatesDoNotRequireAnAnchor) {
  for (auto kind : {StageCostModelKind::VectorIssue,
                    StageCostModelKind::ContinuousTileMemory,
                    StageCostModelKind::ContinuousTileStore,
                    StageCostModelKind::TinyCubeRoofline}) {
    auto stage = logicalStage("unanchored", kind);
    stage.legalSimtFactors = {1, 2, 4};
    stage.features.insideAutoBlockifyV1Loop = true;
    stage.workload.loadBytes = 512;
    stage.workload.loadWarpInstructions = 4;
    stage.workload.storeBytes = 256;
    stage.workload.storeWarpInstructions = 2;
    stage.workload.dotFlops = 1024;
    stage.workload.dotOperations = 1;
    auto profile = hardwareProfile();
    profile.cvAivSubblocks = 2;
    profile.baseSimtWarpCount = 4;
    auto withoutAnchor = evaluateOneStage(stage, profile);
    ASSERT_TRUE(bool(withoutAnchor));
    const auto &unanchored = withoutAnchor->stages[0];
    ASSERT_EQ(unanchored.simtCalibrationEstimates.size(), 6u);
    EXPECT_EQ(unanchored.implementations.size(), 4u);
    EXPECT_FALSE(unanchored.localSimtMaterializable);
    EXPECT_TRUE(unanchored.simtAnchorIndices.empty());
    for (const auto &cost : unanchored.implementations)
      EXPECT_FALSE(cost.implementation.localScope);

    stage.localSimtMaterializable = true;
    stage.localSimtFactors = {1, 2, 4};
    auto withAnchor = evaluateOneStage(stage, profile);
    ASSERT_TRUE(bool(withAnchor));
    for (const auto &estimate : unanchored.simtCalibrationEstimates) {
      const auto &implementation = estimate.implementation;
      auto found = llvm::find_if(
          withAnchor->stages[0].implementations, [&](const auto &cost) {
            return cost.implementation.localScope &&
                   cost.implementation.superblockFactor ==
                       implementation.superblockFactor &&
                   cost.implementation.aivSubblocks ==
                       implementation.aivSubblocks;
          });
      ASSERT_NE(found, withAnchor->stages[0].implementations.end());
      EXPECT_DOUBLE_EQ(estimate.totalCycles, found->totalCycles);
      EXPECT_DOUBLE_EQ(estimate.resources.load, found->resources.load);
      EXPECT_DOUBLE_EQ(estimate.resources.store, found->resources.store);
      EXPECT_DOUBLE_EQ(estimate.resources.compute, found->resources.compute);
      EXPECT_DOUBLE_EQ(estimate.resources.dot, found->resources.dot);
    }
    auto json = unanchored.toJSON();
    auto *estimates = json.getArray("simt_calibration_estimates");
    ASSERT_NE(estimates, nullptr);
    ASSERT_EQ(estimates->size(), 6u);
    for (const auto &value : *estimates) {
      const auto *estimate = value.getAsObject();
      ASSERT_NE(estimate, nullptr);
      EXPECT_TRUE(estimate->getBoolean("estimate_only").value_or(false));
      EXPECT_FALSE(estimate->getBoolean("local_implementation_available")
                       .value_or(true));
      EXPECT_FALSE(
          estimate->getBoolean("includes_scope_boundary_costs").value_or(true));
    }
  }
}

TEST(SimdSimtCostModelTest, CalibrationEstimatesCannotBecomeRouteCandidates) {
  auto stage = logicalStage("unanchored", StageCostModelKind::VectorIssue);
  stage.legalSimtFactors = {1, 2, 4};
  auto profile = hardwareProfile();
  profile.cvAivSubblocks = 2;
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(table));
  ASSERT_FALSE(table->stages[0].simtCalibrationEstimates.empty());
  // Even unrealistically cheap diagnostic costs must not create a route.
  for (auto &estimate : table->stages[0].simtCalibrationEstimates)
    estimate.totalCycles = 0;
  table->logicalProgramCountHint = 512;
  table->physicalCoreCountHint = 56;
  table->physicalAiCoreCountHint = 28;
  auto withEstimates = solveStageRoutes(*table, {});
  ASSERT_TRUE(bool(withEstimates));
  EXPECT_FALSE(withEstimates->mixed.legal);
  table->stages[0].simtCalibrationEstimates.clear();
  auto withoutEstimates = solveStageRoutes(*table, {});
  ASSERT_TRUE(bool(withoutEstimates));
  EXPECT_EQ(withEstimates->mixed.legal, withoutEstimates->mixed.legal);
  EXPECT_DOUBLE_EQ(withEstimates->allSimd.totalCycles,
                   withoutEstimates->allSimd.totalCycles);
  EXPECT_DOUBLE_EQ(withEstimates->allSimt.totalCycles,
                   withoutEstimates->allSimt.totalCycles);
}

TEST(SimdSimtCostModelTest, CalibrationRespectsResourceBoundNotModeEnablement) {
  auto stage = logicalStage("diagnostic", StageCostModelKind::VectorIssue);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  auto profile = hardwareProfile();
  profile.cvAivSubblocks = 2;
  profile.maximumSimtEstimationFactor = 2;
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(table));
  EXPECT_EQ(table->stages[0].implementations.size(), 1u);
  EXPECT_EQ(table->stages[0].simtCalibrationEstimates.size(), 4u);
  for (const auto &estimate : table->stages[0].simtCalibrationEstimates)
    EXPECT_LE(estimate.implementation.superblockFactor, 2);
  profile.maximumSimtEstimationFactor = 3;
  EXPECT_FALSE(profile.isValid());
}

TEST(SimdSimtCostModelTest, UnanchoredCalibrationUsesFixedCvParentAndF1Tail) {
  auto payload = logicalStage("payload", StageCostModelKind::VectorIssue);
  payload.features.insideAutoBlockifyV1Loop = true;
  auto cube = logicalStage("cube", StageCostModelKind::TinyCubeRoofline);
  cube.features.hasDot = true;
  cube.features.insideAutoBlockifyV1Loop = true;
  StagePartition partition;
  partition.stages = {payload, cube};
  auto profile = hardwareProfile();
  profile.cvAivSubblocks = 2;
  auto table = StageCostEvaluator().evaluate(partition, profile);
  ASSERT_TRUE(bool(table));
  table->logicalProgramCountHint = 512;
  table->physicalCoreCountHint = 56;
  table->physicalAiCoreCountHint = 28;
  auto routes = solveStageRoutes(*table, {});
  ASSERT_TRUE(bool(routes));
  EXPECT_FALSE(routes->mixed.legal);
  auto json = routes->toJSON();
  auto *calibration = json.getObject("simt_cost_calibration");
  ASSERT_NE(calibration, nullptr);
  EXPECT_EQ(calibration->getInteger("scheduling_slots"), 28);
  EXPECT_EQ(calibration->getInteger("aiv_partitions"), 2);
  auto *factors = calibration->getArray("factors");
  ASSERT_NE(factors, nullptr);
  ASSERT_EQ(factors->size(), 3u);
  auto *f4 = (*factors)[2].getAsObject();
  ASSERT_NE(f4, nullptr);
  EXPECT_EQ(f4->getInteger("superblock_factor"), 4);
  EXPECT_EQ(f4->getInteger("logical_programs_per_parent"), 19);
  EXPECT_EQ(f4->getInteger("full_group_count"), 4);
  EXPECT_EQ(f4->getInteger("tail_program_count"), 3);
  auto *stages = f4->getArray("stages");
  ASSERT_NE(stages, nullptr);
  ASSERT_EQ(stages->size(), 2u);
  auto *estimate = (*stages)[0].getAsObject();
  ASSERT_NE(estimate, nullptr);
  // 64 elements / 2 AIVs; one setup + 19 half-LB bodies. Neither the
  // absence of anchors nor the all-SIMT executable's Q=10 changes this Q.
  EXPECT_EQ(estimate->getNumber("body_system_cycles"), 10 + 19 * 32);
  EXPECT_FALSE(
      estimate->getBoolean("local_implementation_available").value_or(true));
  EXPECT_NE(estimate->get("boundary_data_system_cycles"), nullptr);
  EXPECT_FALSE(estimate->getNumber("boundary_data_system_cycles"));
  EXPECT_NE(estimate->get("fixed_scope_system_cycles"), nullptr);
  EXPECT_FALSE(estimate->getNumber("fixed_scope_system_cycles"));
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
  profile.simt.dotFlopsPerCycleByWarpCount = {{4, 64.0}, {16, 256.0}};
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
  EXPECT_DOUBLE_EQ(costs[0].resources.dot, 0.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.dot, 0.0);
}

TEST(SimdSimtCostModelTest,
     RecurrenceSuperBlockChargesEveryLogicalProgramsResources) {
  LogicalStage stage = logicalStage(
      "generic_recurrence", StageCostModelKind::LoopCarriedRecurrence,
      StageScheduleKind::LoopCarriedSerial, /*iterations=*/3);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.workload.operationCounts.clear();
  stage.workload.dotOperations = 1.0;
  stage.workload.dotFlops = 640.0;
  stage.workload.issueElements = 320.0;

  HardwareProfile profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.dotSetupCycles = 1.0;
  profile.simt.dotFlopsPerCycleByWarpCount = {
      {4, 64.0}, {8, 96.0}, {16, 128.0}};
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(table));
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  // Per-program dot costs are 11, 23/3, and 6 cycles at 4/8/16 warps.
  // The physical group still executes F copies, so its aggregate dot floors
  // are 11, 46/3, and 24 cycles respectively. Setup is paid once.
  EXPECT_DOUBLE_EQ(costs[0].totalCycles, 10.0 + 3.0 * 11.0);
  EXPECT_DOUBLE_EQ(costs[1].totalCycles, 10.0 + 3.0 * (46.0 / 3.0));
  EXPECT_DOUBLE_EQ(costs[2].totalCycles, 10.0 + 3.0 * 24.0);
}

TEST(SimdSimtCostModelTest,
     DependentDotOccupancyCurveDoesNotPriceStraightLineDot) {
  LogicalStage stage =
      logicalStage("straight_dot", StageCostModelKind::TinyCubeRoofline);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 4};
  stage.workload.operationCounts.clear();
  stage.workload.dotOperations = 1.0;
  stage.workload.dotFlops = 640.0;

  HardwareProfile profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.dotSetupCycles = 1.0;
  profile.simt.dotFlopsPerCycle = 64.0;
  profile.simt.dotFlopsPerCycleByWarpCount = {{4, 64.0}, {16, 128.0}};
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(table));
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 2u);
  EXPECT_DOUBLE_EQ(costs[0].resources.dot, 11.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.dot, 11.0);
  EXPECT_DOUBLE_EQ(costs[1].totalCycles, 10.0 + 4.0 * 11.0);
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
    stage.workload.operationCounts.clear();
    stage.workload
        .operationCounts["f32.add"][contiguousElements * segmentCount] = 1.0;
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
  auto makePartition = [](int64_t independentGroups) {
    StagePartition partition;
    LogicalStage stage =
        logicalStage("payload", StageCostModelKind::LoopCarriedRecurrence,
                     StageScheduleKind::LoopCarriedSerial, /*iterations=*/16);
    stage.features.hasLoop = true;
    stage.features.hasLoopCarriedDataDependency = true;
    stage.features.parallelRecurrenceGroupCount = independentGroups;
    stage.localSimtMaterializable = true;
    stage.localSuperblockMaterializable = true;
    stage.localSimtFactors = {1};
    partition.stages.push_back(std::move(stage));
    return partition;
  };

  StagePartition f1Only = makePartition(/*independentGroups=*/4);
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(f1Only, 4, false))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(f1Only.stages[0].localSimtFactors, (std::vector<int64_t>{1}));

  StagePartition scopeSuperblock = makePartition(/*independentGroups=*/4);
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(scopeSuperblock, 4, true, 4))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(scopeSuperblock.stages[0].localSimtFactors,
            (std::vector<int64_t>{1, 2, 4}));

  StagePartition wholeKernelWide = makePartition(/*independentGroups=*/16);
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(
          wholeKernelWide, /*maximumSuperblockFactor=*/16,
          /*scopeSuperblockMaterializable=*/true,
          /*maximumScopeSuperblockFactor=*/4))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(wholeKernelWide.stages[0].legalSimtFactors,
            (std::vector<int64_t>{1, 2, 4, 8, 16}));
  EXPECT_EQ(wholeKernelWide.stages[0].localSimtFactors,
            (std::vector<int64_t>{1, 2, 4}));

  // ABI-v2 creates an F1 V1 scheduling loop and refines only the selected
  // scope after bufferization, but local and whole-kernel factors still share
  // the same target/runtime warp-resource maximum.
  StagePartition mixedOnly = makePartition(/*independentGroups=*/4);
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

  StagePartition oneWorkGroup = makePartition(/*independentGroups=*/1);
  if (llvm::Error error =
          StageModeLegalityAnalysis().analyze(oneWorkGroup, 4, true, 4))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(oneWorkGroup.stages[0].localSimtFactors,
            (std::vector<int64_t>{1, 2, 4}));
}

TEST(SimdSimtCostModelTest,
     AdditiveScanWithOverflowInstrumentationRemainsSimdLegal) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%input: tensor<256xi32>) -> tensor<256xi32> {
        %scan = "tt.scan"(%input) ({
        ^bb0(%lhs: i32, %rhs: i32):
          %lhs64 = arith.extsi %lhs : i32 to i64
          %rhs64 = arith.extsi %rhs : i32 to i64
          %wide = arith.addi %lhs64, %rhs64 : i64
          %limit = arith.constant 2147483647 : i64
          %ok = arith.cmpi sle, %wide, %limit : i64
          "tt.assert"(%ok) {message = "overflow"} : (i1) -> ()
          %sum = arith.addi %lhs, %rhs : i32
          "tt.scan.return"(%sum) : (i32) -> ()
        }) {axis = 0 : i32, reverse = false}
          : (tensor<256xi32>) -> tensor<256xi32>
        return %scan : tensor<256xi32>
      }
    })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *scan = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "tt.scan")
      scan = operation;
  });
  ASSERT_NE(scan, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.operations = {scan};
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(partition, 4))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_TRUE(partition.stages.front().simdLegal);
}

TEST(SimdSimtCostModelTest, MaximumScanRemainsSimdLegal) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%input: tensor<64xf32>) -> tensor<64xf32> {
        %scan = "tt.scan"(%input) ({
        ^bb0(%lhs: f32, %rhs: f32):
          %maximum = arith.maximumf %lhs, %rhs : f32
          "tt.scan.return"(%maximum) : (f32) -> ()
        }) {axis = 0 : i32, reverse = false}
          : (tensor<64xf32>) -> tensor<64xf32>
        return %scan : tensor<64xf32>
      }
    })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *scan = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "tt.scan")
      scan = operation;
  });
  ASSERT_NE(scan, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.operations = {scan};
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(partition, 4))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_TRUE(partition.stages.front().simdLegal);
}

TEST(SimdSimtCostModelTest, UnsupportedSubtractiveScanRemainsSimdIllegal) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%input: tensor<64xf32>) -> tensor<64xf32> {
        %scan = "tt.scan"(%input) ({
        ^bb0(%lhs: f32, %rhs: f32):
          %difference = arith.subf %lhs, %rhs : f32
          "tt.scan.return"(%difference) : (f32) -> ()
        }) {axis = 0 : i32, reverse = false}
          : (tensor<64xf32>) -> tensor<64xf32>
        return %scan : tensor<64xf32>
      }
    })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *scan = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "tt.scan")
      scan = operation;
  });
  ASSERT_NE(scan, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.operations = {scan};
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(partition, 4))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_FALSE(partition.stages.front().simdLegal);
}

TEST(SimdSimtCostModelTest, NativeSortIsSimdOnly) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%input: tensor<32x16xf32>)
          -> tensor<32x16xf32> {
        %sort = "ascend.sort"(%input) {dim = 1 : i64, descending = false}
          : (tensor<32x16xf32>) -> tensor<32x16xf32>
        return %sort : tensor<32x16xf32>
      }
    })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *sort = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "ascend.sort")
      sort = operation;
  });
  ASSERT_NE(sort, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto stage = logicalStage("sort", StageCostModelKind::VectorIssue);
  stage.operations = {sort};
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(partition, 4))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_TRUE(partition.stages.front().simdLegal);
  EXPECT_FALSE(partition.stages.front().simtLegal);
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

TEST(SimdSimtCostModelTest, SimtSuperBlockFactorsHonorGenericUbCapacity) {
  auto makeDotPartition = [](int64_t m, int64_t n, int64_t k,
                             int64_t elementBits) {
    StagePartition partition;
    LogicalStage stage =
        logicalStage("dot_payload", StageCostModelKind::CubeRoofline);
    stage.wholeSimtRegisterPressure = SimtRegisterPressure{};
    auto &facts = *stage.wholeSimtRegisterPressure;
    facts.ttirMaximumWeakDotComponentNodes = 1;
    facts.ttirMaximumWeakDotComponentGeometryKnown = true;
    facts.ttirMaximumWeakDotComponentM = m;
    facts.ttirMaximumWeakDotComponentN = n;
    facts.ttirMaximumWeakDotComponentK = k;
    facts.ttirMaximumWeakDotComponentElementBits = elementBits;
    partition.stages.push_back(std::move(stage));
    EXPECT_FALSE(StageModeLegalityAnalysis().analyze(
        partition, /*maximumSuperblockFactor=*/4));
    return partition;
  };

  // A 64x64x64 FP32 dot exposes lhs, rhs and accumulator simultaneously:
  // (M*K + K*N + M*N) * 4 = 49,152 bytes per logical program. F2 fits the
  // target's 122,880-byte UB capacity; F4 does not.
  StagePartition wide = makeDotPartition(64, 64, 64, 32);
  ASSERT_FALSE(
      mlir::ascend::StageResourceLegalityAnalysis().applySimtUbCapacity(
          wide, /*usableUbBytes=*/122880));
  const LogicalStage &wideStage = wide.stages.front();
  EXPECT_EQ(wideStage.simtUbWorkingSetBytesPerProgram, 49152);
  EXPECT_EQ(wideStage.simtUsableUbBytes, 122880);
  EXPECT_EQ(wideStage.legalSimtFactors, (std::vector<int64_t>{1, 2}));
  EXPECT_EQ(wideStage.simtFactorsRejectedByUb, (std::vector<int64_t>{4}));
  EXPECT_TRUE(wideStage.simtLegal);

  // The same formula leaves small tiles unconstrained; there is no kernel or
  // shape whitelist behind the decision.
  StagePartition small = makeDotPartition(16, 16, 16, 32);
  ASSERT_FALSE(
      mlir::ascend::StageResourceLegalityAnalysis().applySimtUbCapacity(
          small, /*usableUbBytes=*/122880));
  EXPECT_EQ(small.stages.front().simtUbWorkingSetBytesPerProgram, 3072);
  EXPECT_EQ(small.stages.front().legalSimtFactors,
            (std::vector<int64_t>{1, 2, 4}));
  EXPECT_TRUE(small.stages.front().simtFactorsRejectedByUb.empty());
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

  // A semantic completion curve may already contain the fixed enter/exit
  // envelope. Boundary tensor traffic remains explicit, but the fixed 12
  // cycles must not be charged a second time.
  table.stages[1].implementations[2].localScopeCompletionInclusive = true;
  auto completionInclusive = solveStageRoutes(table, scopeHandoff);
  if (!completionInclusive)
    FAIL() << llvm::toString(completionInclusive.takeError());
  EXPECT_DOUBLE_EQ(completionInclusive->mixed.totalCycles, 90.0);
  EXPECT_DOUBLE_EQ(completionInclusive->mixed.scopeHandoffCycles[1], 0.0);
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
  stage.workload.prefixScanCounts[{32, 8, 1, 1, false, "add"}] = 1.0;

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

TEST(SimdSimtCostModelTest,
     StraightLineSimdScanUsesCombineAndSemanticGeometry) {
  LogicalStage stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.features.hasReduction = true;
  stage.features.hasPrefixScan = true;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 0.0;
  stage.workload.prefixScanCounts[{64, 16, 1, 1, false, "minmax"}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.straightLinePrefixScanCostsByCombine["minmax"] = {
      100.0, 2.0, 3.0, {}, "structural scan test"};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.prefixScanCriticalPath,
                   100.0 + 64.0 * 16.0 / 2.0 + 63.0 * 3.0);
  EXPECT_DOUBLE_EQ(cost.resources.prefixScanIssue, 0.0);
  EXPECT_NE(
      cost.formulaEvidence.find("straight_line_scan_structural_completion"),
      std::string::npos);

  stage.workload.prefixScanCounts.clear();
  stage.workload.prefixScanCounts[{256, 1, 1, 1, false, "minmax"}] = 1.0;
  profile.simd.straightLinePrefixScanCostsByCombine["minmax"]
      .singleSequenceCompletionCyclesByExtent = {{128, 700.0}, {512, 1100.0}};
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_DOUBLE_EQ(table->stages.front()
                       .implementations.front()
                       .resources.prefixScanCriticalPath,
                   833.33333333333337);

  stage.workload.prefixScanCounts.clear();
  stage.workload.prefixScanCounts[{5, 21, 7, 1, false, "minmax"}] = 1.0;
  auto &strided = profile.simd.straightLinePrefixScanCostsByCombine["minmax"];
  strided.nonUnitStrideStartupCycles = 76.0;
  strided.nonUnitStrideElementsPerCycle = 0.84;
  strided.nonUnitStrideMinimumAxisExtent = 3;
  strided.nonUnitStrideMaximumAxisExtent = 8;
  strided.nonUnitStrideMinimumAxisStride = 4;
  strided.nonUnitStrideMaximumAxisStride = 8;
  strided.nonUnitStrideMinimumSemanticElements = 35;
  strided.nonUnitStrideMaximumSemanticElements = 168;
  strided.nonUnitStrideEvidence = "strided structural scan test";
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_DOUBLE_EQ(table->stages.front()
                       .implementations.front()
                       .resources.prefixScanCriticalPath,
                   76.0 + 105.0 / 0.84);
}

TEST(SimdSimtCostModelTest,
     OperationProfileSeparatesGridStrideKernelStartupFromWaveBody) {
  LogicalStage stage =
      logicalStage("histogram", StageCostModelKind::VectorIssue);
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["histogram.i32"][256] = 1.0;
  stage.workload.issueElements = 256.0;
  HardwareProfile profile = hardwareProfile();
  StageOperationRate rate;
  rate.throughput = 0.02;
  rate.freeWorkUnits = 0.5;
  rate.gridStrideKernelStartupCycles = 1200.0;
  profile.simd.operationRates["histogram.i32"] = rate;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.compute, (4.0 - 0.5) / 0.02);
  EXPECT_DOUBLE_EQ(cost.resources.gridStrideStartup, 1200.0);
  EXPECT_LT(cost.totalCycles, cost.resources.gridStrideStartup);
}

TEST(SimdSimtCostModelTest,
     GridStrideScanCurveRetainsOwnedLoadAndStoreService) {
  LogicalStage stage =
      logicalStage("dynamic_scan", StageCostModelKind::PrefixScan);
  stage.simdLegal = false;
  stage.features.hasReduction = true;
  stage.features.hasPrefixScan = true;
  stage.features.insideGridStrideLoopBody = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.prefixScanCounts[{256, 1, 1, 1, false, "add"}] = 1.0;
  stage.workload.loadWarpInstructions = 4.0;
  stage.workload.storeWarpInstructions = 2.0;

  HardwareProfile profile = hardwareProfile();
  profile.simt.gridStrideForwardScanCyclesByExtent = {{256, 250.0}};
  profile.simt.gridStrideForwardScanStartupCyclesByExtent = {{256, 50.0}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.gridStridePrefixScanBody, 250.0);
  EXPECT_GT(cost.resources.load, 0.0);
  EXPECT_GT(cost.resources.store, 0.0);
  EXPECT_DOUBLE_EQ(cost.totalCycles,
                   250.0 + cost.resources.load + cost.resources.store);
}

TEST(SimdSimtCostModelTest,
     GridStrideSimdScanCurveAlreadyIncludesOwnedMemoryService) {
  LogicalStage stage =
      logicalStage("dynamic_scan", StageCostModelKind::PrefixScan);
  stage.simtLegal = false;
  stage.features.hasReduction = true;
  stage.features.hasPrefixScan = true;
  stage.features.insideGridStrideLoopBody = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.prefixScanCounts[{256, 1, 1, 1, true, "add"}] = 1.0;
  stage.workload.loadBytes = 4096.0;
  stage.workload.storeBytes = 4096.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.storeOperations = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.gridStrideReverseScanCyclesByExtent = {{256, 250.0}};
  profile.simd.gridStrideReverseScanStartupCyclesByExtent = {{256, 50.0}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.gridStridePrefixScanBody, 250.0);
  EXPECT_GT(cost.resources.load, 0.0);
  EXPECT_GT(cost.resources.store, 0.0);
  EXPECT_DOUBLE_EQ(cost.totalCycles, 250.0);
}

TEST(SimdSimtCostModelTest,
     GridStrideSimtScanUsesFactorSpecificBodyWithoutDividingSourceQ) {
  LogicalStage stage =
      logicalStage("dynamic_scan", StageCostModelKind::PrefixScan);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasReduction = true;
  stage.features.hasPrefixScan = true;
  stage.features.insideGridStrideLoopBody = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.prefixScanCounts[{256, 1, 1, 1, false, "add"}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simt.gridStrideForwardScanCyclesByExtent = {{256, 250.0}};
  profile.simt.gridStrideForwardScanStartupCyclesByExtent = {{256, 50.0}};
  profile.simt.gridStrideForwardScanCyclesByFactorAndExtent = {
      {2, {{256, 225.0}}}, {4, {{256, 310.0}}}};
  profile.simt.gridStrideForwardScanStartupCyclesByFactorAndExtent = {
      {2, {{256, 25.0}}}, {4, {{256, 70.0}}}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_DOUBLE_EQ(costs[0].resources.gridStridePrefixScanBody, 250.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.gridStrideStartup, 50.0);
  EXPECT_FALSE(costs[0].resources.gridStrideFactorSpecificResourceBody);
  EXPECT_DOUBLE_EQ(costs[1].resources.gridStridePrefixScanBody, 225.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.gridStrideStartup, 25.0);
  EXPECT_TRUE(costs[1].resources.gridStrideFactorSpecificResourceBody);
  EXPECT_DOUBLE_EQ(costs[2].resources.gridStridePrefixScanBody, 310.0);
  EXPECT_DOUBLE_EQ(costs[2].resources.gridStrideStartup, 70.0);
  EXPECT_TRUE(costs[2].resources.gridStrideFactorSpecificResourceBody);
}

TEST(SimdSimtCostModelTest,
     GridStrideSimtGatherUsesFactorSpecificResourceCurves) {
  LogicalStage stage =
      logicalStage("dynamic_gather", StageCostModelKind::IndirectGatherMemory);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasIndirectMemory = true;
  stage.features.insideGridStrideLoopBody = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.loadBytes = 512.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.indirectLoadCounts[{512, 256}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simt.gridStrideIndirectLoadCyclesByWarpInstructions = {{8, 500.0}};
  profile.simt.gridStrideIndirectLoadStartupCyclesByWarpInstructions = {
      {8, 50.0}};
  profile.simt.gridStrideIndirectLoadCyclesByFactorAndWarpInstructions = {
      {2, {{8, 600.0}}}, {4, {{8, 800.0}}}};
  profile.simt.gridStrideIndirectLoadStartupCyclesByFactorAndWarpInstructions =
      {{2, {{8, 60.0}}}, {4, {{8, 80.0}}}};

  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_DOUBLE_EQ(costs[0].resources.indirectLoadGroup, 500.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.gridStrideStartup, 50.0);
  EXPECT_FALSE(costs[0].resources.gridStrideFactorSpecificResourceBody);
  EXPECT_DOUBLE_EQ(costs[1].resources.indirectLoadGroup, 600.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.gridStrideStartup, 60.0);
  EXPECT_TRUE(costs[1].resources.gridStrideFactorSpecificResourceBody);
  EXPECT_DOUBLE_EQ(costs[2].resources.indirectLoadGroup, 800.0);
  EXPECT_DOUBLE_EQ(costs[2].resources.gridStrideStartup, 80.0);
  EXPECT_TRUE(costs[2].resources.gridStrideFactorSpecificResourceBody);
}

TEST(SimdSimtCostModelTest,
     GridStrideSimdGatherPrefersDynamicWarpCurveOverStaticElementCurve) {
  LogicalStage stage =
      logicalStage("dynamic_gather", StageCostModelKind::IndirectGatherMemory);
  stage.features.hasIndirectMemory = true;
  stage.features.insideGridStrideLoopBody = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loadBytes = 512.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.indirectLoadCounts[{512, 256}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.indirectLoadSystemCyclesByElements = {{256, 80.0}};
  profile.simd.indirectLoadStartupSystemCyclesByWarpInstructions = {{8, 10.0}};
  profile.simd.gridStrideIndirectLoadCyclesByWarpInstructions = {{8, 500.0}};
  profile.simd.gridStrideIndirectLoadStartupCyclesByWarpInstructions = {
      {8, 50.0}};

  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.indirectLoadGroup, 500.0);
  EXPECT_DOUBLE_EQ(cost.resources.gridStrideStartup, 50.0);
  EXPECT_DOUBLE_EQ(cost.resources.indirectLoadGroupStartup, 0.0);
}

TEST(SimdSimtCostModelTest, GridStrideSimtDotUsesFactorSpecificResourceCurves) {
  LogicalStage stage =
      logicalStage("dynamic_dot", StageCostModelKind::TinyCubeRoofline);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasDot = true;
  stage.features.insideGridStrideLoopBody = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.dotOperations = 1.0;
  stage.workload.dotFlops = 8192.0;
  stage.workload.dotFlopCounts[8192] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simt.gridStrideDotCyclesByFlops = {{8192, 700.0}};
  profile.simt.gridStrideDotStartupCyclesByFlops = {{8192, 70.0}};
  profile.simt.gridStrideDotCyclesByFactorAndFlops = {{2, {{8192, 900.0}}},
                                                      {4, {{8192, 1200.0}}}};
  profile.simt.gridStrideDotStartupCyclesByFactorAndFlops = {
      {2, {{8192, 90.0}}}, {4, {{8192, 120.0}}}};

  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_DOUBLE_EQ(costs[0].resources.dot, 700.0);
  EXPECT_FALSE(costs[0].resources.gridStrideFactorSpecificResourceBody);
  EXPECT_DOUBLE_EQ(costs[1].resources.dot, 900.0);
  EXPECT_TRUE(costs[1].resources.gridStrideFactorSpecificResourceBody);
  EXPECT_DOUBLE_EQ(costs[2].resources.dot, 1200.0);
  EXPECT_TRUE(costs[2].resources.gridStrideFactorSpecificResourceBody);
}

TEST(SimdSimtCostModelTest, GridStrideDotCurveAppliesToSimdLoopShell) {
  LogicalStage stage =
      logicalStage("dynamic_dot_shell", StageCostModelKind::TinyCubeRoofline);
  stage.simtLegal = false;
  stage.features.hasDot = true;
  stage.features.isGridStrideLoopShell = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.dotOperations = 1.0;
  stage.workload.dotFlops = 8192.0;
  stage.workload.dotFlopCounts[8192] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.gridStrideDotCyclesByFlops = {{8192, 750.0}};
  profile.simd.gridStrideDotStartupCyclesByFlops = {{8192, 1000.0}};

  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.dot, 750.0);
  EXPECT_DOUBLE_EQ(cost.resources.gridStrideStartup, 1000.0);
  EXPECT_FALSE(cost.resources.gridStrideFactorSpecificResourceBody);
}

TEST(SimdSimtCostModelTest, PrefixScanIssuePreservesEveryWarpStep) {
  LogicalStage stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.features.hasReduction = true;
  stage.features.hasPrefixScan = true;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 0.0;
  stage.workload.shuffleLaneSteps = 0.0;
  stage.workload.prefixScanCounts[{4, 8, 1, 1, false, "add"}] = 1.0;

  auto table = evaluateOneStage(std::move(stage));
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &simt = table->stages.front().implementations[1];
  // ceil(log2(4))=2, so 8 sequences have 16 warp-steps.  The issue workload
  // is exactly 16 * (shuffle + predicate + add), independent of active-lane
  // density at the two levels.
  EXPECT_DOUBLE_EQ(simt.resources.prefixScanIssue, 16.0 * 3.0 / 4.0);
}

TEST(SimdSimtCostModelTest,
     StraightLineSimtScanUsesCompletionInclusiveElementRate) {
  LogicalStage stage = logicalStage("scan", StageCostModelKind::PrefixScan);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1};
  stage.features.hasReduction = true;
  stage.features.hasPrefixScan = true;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 0.0;
  stage.workload.shuffleLaneSteps = 0.0;
  stage.workload.prefixScanCounts[{32, 16, 1, 1, false, "add"}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simt.straightLinePrefixScanElementsPerCycle = 10.0;
  profile.simt.straightLinePrefixScanCompletionEvidence = "unit test";
  auto table = evaluateOneStage(std::move(stage), profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  ASSERT_EQ(table->stages.front().implementations.size(), 1u);
  EXPECT_DOUBLE_EQ(table->stages.front().implementations.front().totalCycles,
                   51.2);
}

TEST(SimdSimtCostModelTest, ReversePrefixScanHonorsTargetExtentLegality) {
  auto makeScan = [](bool reverse) {
    LogicalStage stage = logicalStage("scan", StageCostModelKind::PrefixScan);
    stage.features.hasReduction = true;
    stage.features.hasPrefixScan = true;
    stage.workload.operationCounts.clear();
    stage.workload.scalarOperations = 0.0;
    stage.workload.issueElements = 0.0;
    stage.workload.prefixScanCounts[{512, 1, 1, 1, reverse, "add"}] = 1.0;
    return stage;
  };
  HardwareProfile profile = hardwareProfile();
  profile.simd.maximumReversePrefixScanExtent = 256;

  auto reverse = evaluateOneStage(makeScan(true), profile);
  if (!reverse)
    FAIL() << llvm::toString(reverse.takeError());
  ASSERT_EQ(reverse->stages.front().implementations.size(), 1u);
  EXPECT_EQ(reverse->stages.front().implementations.front().implementation.mode,
            StageMode::SIMT);

  auto forward = evaluateOneStage(makeScan(false), profile);
  if (!forward)
    FAIL() << llvm::toString(forward.takeError());
  ASSERT_EQ(forward->stages.front().implementations.size(), 2u);
  EXPECT_EQ(forward->stages.front().implementations.front().implementation.mode,
            StageMode::SIMD);
}

TEST(SimdSimtCostModelTest, SimdPricingPreservesShortTensorSegments) {
  LogicalStage stage =
      logicalStage("segmented", StageCostModelKind::VectorIssue);
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["f32.add"][12000] = 1.0;
  stage.workload.tensorOperationWorkloads.push_back(
      {"f32.add", 32, 12000.0, 80.0, 150});
  stage.workload.issueElements = 12000.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.vectorWidthBits = 2048;
  profile.simd.vectorWidth = 64;
  profile.simd.issueWidth = 64;
  profile.simd.operationRates["f32.add"].throughput = 1.0;
  profile.simd.operationRates["f32.add"].factor = 1.0;
  profile.simd.issueOperationsPerCycle = 1.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &simd = table->stages.front().implementations.front();
  // 80 independent rows each need ceil(150 fp32 / 64 lanes)=3 vector adds.
  EXPECT_DOUBLE_EQ(simd.resources.compute, 240.0);
  EXPECT_DOUBLE_EQ(simd.resources.issue, 240.0);
}

TEST(SimdSimtCostModelTest, ScalarMemoryUsesWhiteBoxResources) {
  LogicalStage stage =
      logicalStage("scalar_memory", StageCostModelKind::ScalarLoad);
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 2.0;
  stage.workload.scalarLoadCount = 1.0;
  stage.workload.scalarStoreCount = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.mainScalarLoadPrepCycles = 3.0;
  profile.simd.mainScalarLoadFillCycles = 11.0;
  profile.simd.mainScalarLoadOutstandingLines = 2.0;
  profile.simd.mainScalarLoadExtraLineLowCycles = 17.0;
  profile.simd.mainScalarLoadExtraLineHighCycles = 19.0;
  profile.simd.mainScalarLoadExtraLineHighThreshold = 4.0;
  profile.simd.mainScalarLoadIssueCycles = 2.0;
  profile.simd.mte3StorePrepCycles = 5.0;
  profile.simd.mte3StoreFillCycles = 13.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &simd = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(simd.resources.load, 14.0);
  EXPECT_DOUBLE_EQ(simd.resources.store, 18.0);
  // The load and store use independent scalar-memory pipelines, so their
  // white-box services compete in the Stage resource envelope.
  EXPECT_DOUBLE_EQ(simd.totalCycles, 18.0);
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

TEST(SimdSimtCostModelTest,
     SaturatedFallbackChargesEveryUnclassifiedComputeOperation) {
  auto stage = logicalStage("vector", StageCostModelKind::VectorIssue);
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["fp32.saturated.math.floor"][64] = 2.0;
  stage.workload.issueElements = 128.0;

  auto profile = hardwareProfile();
  profile.simd.operationRates["fp32.saturated"] = {2.0, 1.0};
  profile.simt.operationRates["fp32.saturated"] = {32.0, 1.0};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 2u);
  EXPECT_DOUBLE_EQ(costs[0].resources.compute, 1.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.compute, 4.0);
  for (const auto &cost : costs) {
    EXPECT_EQ(cost.formulaEvidence.find("unprofiled_vector_execution"),
              std::string::npos);
    EXPECT_NE(cost.formulaEvidence.find("fp32_saturated_compute_fallback"),
              std::string::npos);
  }

  // Older hardware profiles predate the explicit fp32.saturated alias.  The
  // measured f32.add saturation rate is their backward-compatible non-zero
  // lower bound, so compute coverage does not depend on a profile migration.
  auto legacy = profile;
  legacy.simd.operationRates.erase("fp32.saturated");
  legacy.simt.operationRates.erase("fp32.saturated");
  legacy.simd.operationRates["f32.add"] = {4.0, 1.0};
  legacy.simt.operationRates["f32.add"] = {64.0, 1.0};
  auto legacyTable = evaluateOneStage(stage, legacy);
  if (!legacyTable)
    FAIL() << llvm::toString(legacyTable.takeError());
  EXPECT_DOUBLE_EQ(
      legacyTable->stages.front().implementations[0].resources.compute, 0.5);
  EXPECT_DOUBLE_EQ(
      legacyTable->stages.front().implementations[1].resources.compute, 2.0);
  for (const auto &cost : legacyTable->stages.front().implementations)
    EXPECT_NE(cost.formulaEvidence.find("fp32_saturated_compute_fallback"),
              std::string::npos);

  // A later per-operation microbenchmark can override the fallback by adding
  // the exact retained workload key to the profile; no extractor change is
  // required.
  profile.simd.operationRates["fp32.saturated.math.floor"] = {8.0, 1.0};
  profile.simt.operationRates["fp32.saturated.math.floor"] = {64.0, 1.0};
  auto exact = evaluateOneStage(stage, profile);
  if (!exact)
    FAIL() << llvm::toString(exact.takeError());
  EXPECT_DOUBLE_EQ(exact->stages.front().implementations[0].resources.compute,
                   0.25);
  EXPECT_DOUBLE_EQ(exact->stages.front().implementations[1].resources.compute,
                   2.0);
  for (const auto &cost : exact->stages.front().implementations)
    EXPECT_EQ(cost.formulaEvidence.find("fp32_saturated_compute_fallback"),
              std::string::npos);
}

TEST(SimdSimtCostModelTest,
     ContinuousMemoryStageCannotDropFusedSaturatedCompute) {
  auto stage =
      logicalStage("memory_compute", StageCostModelKind::ContinuousTileMemory,
                   StageScheduleKind::IndependentPipelined);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["fp32.saturated.math.floor"][640] = 1.0;
  stage.workload.issueElements = 640.0;
  stage.workload.loadBytes = 64.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.loadWarpInstructions = 1.0;

  auto profile = hardwareProfile();
  profile.simd.operationRates["fp32.saturated"] = {1.0, 1.0};
  profile.simd.loadBytesPerCycle = 1.0e9;
  profile.simd.loadWarpInstructionsPerCycle = 1.0e9;
  profile.simd.issueOperationsPerCycle = 1.0e9;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.compute, 10.0);
  EXPECT_DOUBLE_EQ(cost.totalCycles, 10.0);
}

TEST(SimdSimtCostModelTest,
     StructuralGenericIssueDoesNotClaimMissingExecutionRate) {
  auto stage = logicalStage("vector", StageCostModelKind::VectorIssue);
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["generic.issue"][64] = 1.0;
  auto table = evaluateOneStage(stage);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  for (const auto &cost : table->stages.front().implementations)
    EXPECT_EQ(cost.formulaEvidence.find("unprofiled_vector_execution"),
              std::string::npos);
}

TEST(SimdSimtCostModelTest, UnprofiledNamedExecutionIsExplicitInReport) {
  auto stage = logicalStage("vector", StageCostModelKind::VectorIssue);
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["test.unknown_numeric_execution"][64] = 1.0;
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
  stage.workload.prefixScanCounts[{2, 1, 1, 1, false, "add"}] = 1.0;

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
     SimdRecurrenceAmortizesTransferStartupAcrossKnownIterations) {
  LogicalStage stage = logicalStage(
      "memory_recurrence", StageCostModelKind::LoopCarriedRecurrence,
      StageScheduleKind::LoopCarriedSerial, /*iterations=*/4);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
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
  profile.simd.loadBytesPerCycle = 32.0;
  profile.simd.storeBytesPerCycle = 16.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const StageImplementationCost &simd = table->stages[0].implementations[0];
  EXPECT_DOUBLE_EQ(simd.resources.setup, 2.0 * 5.0 + 3.0 * 7.0);
  EXPECT_DOUBLE_EQ(simd.resources.load, 640.0 / 32.0);
  EXPECT_DOUBLE_EQ(simd.resources.store, 160.0 / 16.0);
  EXPECT_NE(simd.formulaEvidence.find("resident_loop_transfer_setup_once"),
            std::string::npos);
}

TEST(SimdSimtCostModelTest,
     SimdRowwiseSameLayoutAdditionalLoadSetupsAreOwnedOnce) {
  LogicalStage stage =
      logicalStage("multi_input_rowwise", StageCostModelKind::RowwiseReduction,
                   StageScheduleKind::IndependentPipelined, /*iterations=*/4);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.features.hasLoop = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loadBytes = 3.0 * 64.0 * 64.0 * 4.0;
  stage.workload.loadOperations = 3.0;
  StageMemoryAccess load;
  load.shape = {64, 64};
  load.strides = {8192, 1};
  load.elementBits = 32;
  stage.workload.memoryAccesses = {load, load, load};

  HardwareProfile profile = hardwareProfile();
  profile.simd.loadOperationSetupCycles = 5.0;
  profile.simd.loadBytesPerCycle = 32.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const StageImplementationCost &simd = table->stages[0].implementations[0];
  EXPECT_DOUBLE_EQ(simd.resources.setup, 2.0 * 5.0);
  EXPECT_DOUBLE_EQ(simd.resources.load, 5.0 + stage.workload.loadBytes / 32.0);
  EXPECT_TRUE(simd.resources.rowwiseAdditionalLoadSetupsOwnedOnce);
  EXPECT_NE(
      simd.formulaEvidence.find("resident_rowwise_additional_load_setups_once"),
      std::string::npos);
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

TEST(SimdSimtCostModelTest,
     SimdRowwiseCompositeDoesNotDoubleCountVectorPipeService) {
  auto stage = logicalStage("rowwise", StageCostModelKind::RowwiseReduction,
                            StageScheduleKind::IndependentPipelined, 4);
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts["f32.add"] = {{256, 2.0}};
  stage.workload.reductionOperations = 2.0;
  stage.workload.shuffleLaneSteps = 1024.0;
  stage.workload.predicateCounts = {{256, 1.0}};
  stage.workload.loadOperations = 1.0;
  stage.workload.loadBytes = 4096.0;
  stage.workload.loadWarpInstructions = 32.0;
  stage.workload.issueElements = 1.0;

  auto profile = hardwareProfile();
  // Isolate the rowwise resource envelope. Irregular-tile completion has its
  // own structural test below, and the frontend issue floor has separate
  // tests; neither should mask a vector-resource double count here.
  profile.simd.irregularTileCompletionEnabled = false;
  profile.simd.issueOperationsPerCycle = 1.0e9;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  const auto &r = cost.resources;
  ASSERT_GT(r.compute, 0.0);
  ASSERT_GT(r.reduction + r.shuffle, 0.0);
  const double vectorExecution =
      r.predicate + std::max(r.compute + r.dot, r.reduction + r.shuffle);
  const double expected =
      4.0 * std::max({r.load + r.continuousLoadGroup + vectorExecution,
                      r.store + r.continuousStoreGroup, r.scalar + r.predicate,
                      r.issue});
  EXPECT_DOUBLE_EQ(cost.totalCycles, expected);
  const double oldAdditiveVectorService =
      4.0 * (r.load + r.continuousLoadGroup + r.compute + r.dot + r.predicate +
             r.reduction + r.shuffle);
  EXPECT_LT(cost.totalCycles, oldAdditiveVectorService);
}

TEST(SimdSimtCostModelTest,
     IndependentLoopSerializesTripsWithoutInventingPipelineDepth) {
  auto stage = logicalStage("pointwise_loop",
                            StageCostModelKind::IndependentPipelinedLoop,
                            StageScheduleKind::IndependentPipelined, 4);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["f32.mul"][64] = 1.0;
  stage.workload.issueElements = 64.0;
  stage.workload.loadBytes = 2048.0;
  stage.workload.storeBytes = 1024.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.storeOperations = 1.0;
  stage.workload.loadWarpInstructions = 16.0;
  stage.workload.storeWarpInstructions = 8.0;

  auto profile = hardwareProfile();
  profile.simd.issueOperationsPerCycle = 1.0e9;
  profile.simd.operationRates["f32.mul"].startupCycles = 7.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  const auto &r = cost.resources;
  const double load =
      r.load + r.continuousLoadGroup + r.indirectLoadGroup + r.sharedLayout;
  const double store = r.store + r.continuousStoreGroup;
  ASSERT_GT(r.compute, 0.0);
  EXPECT_DOUBLE_EQ(r.operationCompletion, 7.0);
  EXPECT_DOUBLE_EQ(cost.totalCycles,
                   4.0 * (std::max({load, store, r.compute, r.issue}) +
                          r.operationCompletion));
  EXPECT_NE(cost.formulaEvidence.find("serialized_iteration_resource_roofline"),
            std::string::npos);
}

TEST(SimdSimtCostModelTest, OperationStartupHonorsMinimumLogicalElementDomain) {
  auto stage =
      logicalStage("mixed_scalar_tensor_max", StageCostModelKind::VectorIssue);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["f32.max"] = {{1, 1.0}, {64, 1.0}};

  auto profile = hardwareProfile();
  auto &rate = profile.simd.operationRates["f32.max"];
  rate.startupCycles = 7.0;
  rate.minimumStartupElements = 2;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &resources =
      table->stages.front().implementations.front().resources;
  EXPECT_DOUBLE_EQ(resources.operationCompletion, 7.0);
}

TEST(SimdSimtCostModelTest,
     SingleLoadGroupDoesNotReuseRankOneCurveForRankTwoMaskedTile) {
  auto stage = logicalStage("single_load_loop",
                            StageCostModelKind::IndependentPipelinedLoop,
                            StageScheduleKind::IndependentPipelined, 8);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.features.hasLoop = true;
  stage.features.hasContiguousMemory = true;
  stage.workload = {};
  stage.workload.loadBytes = 4096.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.loadWarpInstructions = 32.0;

  StageMemoryAccess load;
  load.shape = {1024};
  load.strides = {1};
  load.elementBits = 32;
  load.masked = true;
  stage.workload.memoryAccesses = {load};

  auto profile = hardwareProfile();
  profile.simd.loadOperationSetupCycles = 10.0;
  profile.simd.loadBytesPerCycle = 100.0;
  profile.simd.independentLoadGroupKernelStartupCycles = 50.0;
  profile.simd.independentLoadGroupIterationStartupCycles = 5.0;
  profile.simd.independentLoadGroupBytesPerCycle = 20.0;
  profile.simd.independentLoadGroupMinimumOperations = 1;
  profile.simd.independentLoadGroupMaximumOperations = 4;

  auto rankOne = evaluateOneStage(stage, profile);
  if (!rankOne)
    FAIL() << llvm::toString(rankOne.takeError());
  const auto &rankOneResources =
      rankOne->stages.front().implementations.front().resources;
  EXPECT_DOUBLE_EQ(rankOneResources.setup, 50.0);
  EXPECT_DOUBLE_EQ(rankOneResources.continuousLoadGroup, 5.0 + 4096.0 / 20.0);
  EXPECT_DOUBLE_EQ(rankOneResources.load, 0.0);

  load.shape = {16, 64};
  load.strides = {64, 1};
  stage.workload.memoryAccesses = {load};
  auto rankTwo = evaluateOneStage(stage, profile);
  if (!rankTwo)
    FAIL() << llvm::toString(rankTwo.takeError());
  const auto &rankTwoResources =
      rankTwo->stages.front().implementations.front().resources;
  EXPECT_DOUBLE_EQ(rankTwoResources.setup, 0.0);
  EXPECT_DOUBLE_EQ(rankTwoResources.continuousLoadGroup, 0.0);
  EXPECT_DOUBLE_EQ(rankTwoResources.load, 10.0 + 4096.0 / 100.0);
}

TEST(SimdSimtCostModelTest,
     IndependentLoadGroupPricesFlatPrefixAndLeavesStridedResidual) {
  auto stage = logicalStage("partial_load_group",
                            StageCostModelKind::IndependentPipelinedLoop,
                            StageScheduleKind::IndependentPipelined, 12);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.features.hasLoop = true;
  stage.features.hasContiguousMemory = true;
  stage.workload = {};
  stage.workload.loadBytes = 12288.0;
  stage.workload.loadOperations = 6.0;
  stage.workload.loadWarpInstructions = 384.0;

  StageMemoryAccess first;
  first.shape = {64, 8};
  first.strides = {8, 1};
  first.elementBits = 32;
  first.count = 2.0;
  StageMemoryAccess second = first;
  StageMemoryAccess strided = first;
  strided.strides = {96, 1};
  stage.workload.memoryAccesses = {first, second, strided};

  auto profile = hardwareProfile();
  profile.simd.loadOperationSetupCycles = 10.0;
  profile.simd.loadBytesPerCycle = 100.0;
  profile.simd.independentLoadGroupKernelStartupCycles = 50.0;
  profile.simd.independentLoadGroupIterationStartupCycles = 5.0;
  profile.simd.independentLoadGroupBytesPerCycle = 20.0;
  profile.simd.independentLoadGroupMinimumOperations = 1;
  profile.simd.independentLoadGroupMaximumOperations = 4;

  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.setup, 50.0);
  EXPECT_DOUBLE_EQ(cost.resources.continuousLoadGroup,
                   2.0 * (5.0 + 4096.0 / 20.0));
  EXPECT_DOUBLE_EQ(cost.resources.load, 2.0 * 10.0 + 4096.0 / 100.0);
  EXPECT_DOUBLE_EQ(cost.totalCycles,
                   50.0 + 12.0 * (2.0 * (5.0 + 4096.0 / 20.0) + 2.0 * 10.0 +
                                  4096.0 / 100.0));
}

TEST(SimdSimtCostModelTest,
     SimdReductionsShareOneFixedStartupWithinDynamicStageInvocation) {
  auto stage =
      logicalStage("fused_reductions", StageCostModelKind::RowwiseReduction,
                   StageScheduleKind::IndependentPipelined, 1);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.features.hasReduction = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 0.0;
  stage.workload.reductionOperations = 1.0;
  stage.workload.shuffleLaneSteps = 80.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.reductionOperationSetupCycles = 100.0;
  profile.simd.reductionLaneStepsPerCycle = 8.0;
  auto oneReduction = evaluateOneStage(stage, profile);
  if (!oneReduction)
    FAIL() << llvm::toString(oneReduction.takeError());

  stage.workload.reductionOperations = 2.0;
  auto twoReductions = evaluateOneStage(stage, profile);
  if (!twoReductions)
    FAIL() << llvm::toString(twoReductions.takeError());

  const auto &one = oneReduction->stages.front().implementations.front();
  const auto &two = twoReductions->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(one.resources.reduction, 110.0);
  EXPECT_DOUBLE_EQ(two.resources.reduction, 110.0);
}

TEST(SimdSimtCostModelTest,
     SimdMultiResultReductionUsesScalarizedLoweringSurface) {
  auto stage = logicalStage("arg_reduce", StageCostModelKind::RowwiseReduction);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.features.hasReduction = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 0.0;
  stage.workload.reductionOperations = 1.0;
  stage.workload.shuffleLaneSteps = 0.0;
  stage.workload.scalarizedMultiResultReductionCounts[{3, 64, 2, 32}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.reductionOperationSetupCycles = 5.0;
  profile.simd.scalarizedMultiResultReductionMinimumExtent = 32;
  profile.simd.scalarizedMultiResultReductionReferenceResults = 2;
  profile.simd.scalarizedMultiResultReductionSetupCycles = 11.0;
  profile.simd.scalarizedMultiResultReductionElementsPerCycle = 2.0;
  profile.simd.scalarizedMultiResultReductionEvidence = "unit surface";
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());

  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.scalarizedMultiResultReduction, 107.0);
  EXPECT_DOUBLE_EQ(cost.resources.reduction, 107.0);
  EXPECT_NE(
      cost.formulaEvidence.find("scalarized_multi_result_reduction_surface"),
      std::string::npos);

  stage.workload.scalarizedMultiResultReductionCounts.clear();
  stage.workload.scalarizedMultiResultReductionCounts[{3, 16, 2, 32}] = 1.0;
  auto belowDomain = evaluateOneStage(stage, profile);
  if (!belowDomain)
    FAIL() << llvm::toString(belowDomain.takeError());
  const auto &control = belowDomain->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(control.resources.scalarizedMultiResultReduction, 0.0);
  EXPECT_DOUBLE_EQ(control.resources.reduction, 5.0);
}

TEST(SimdSimtCostModelTest, SimdNativeSortUsesBoundedRowSurface) {
  auto stage = logicalStage("sort", StageCostModelKind::VectorIssue);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 0.0;
  stage.workload.sortCounts[{128, 16, 32}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.sortMinimumExtent = 8;
  profile.simd.sortMaximumExtent = 32;
  profile.simd.sortElementBits = 32;
  profile.simd.sortSetupCycles = 7.0;
  profile.simd.sortRowsPerCycle = 2.0;
  profile.simd.sortEvidence = "unit surface";
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());

  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.sort, 71.0);
  EXPECT_DOUBLE_EQ(cost.resources.compute, 71.0);
  EXPECT_NE(cost.formulaEvidence.find("native_sort_row_surface"),
            std::string::npos);

  stage.workload.sortCounts.clear();
  stage.workload.sortCounts[{128, 64, 32}] = 1.0;
  auto outsideDomain = evaluateOneStage(stage, profile);
  if (!outsideDomain)
    FAIL() << llvm::toString(outsideDomain.takeError());
  EXPECT_DOUBLE_EQ(
      outsideDomain->stages.front().implementations.front().resources.sort,
      0.0);
}

TEST(SimdSimtCostModelTest,
     SimtMultiResultReductionUsesTailClassAndCeilingRowGroups) {
  auto stage =
      logicalStage("multi_result", StageCostModelKind::RowwiseReduction);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1};
  stage.features.hasReduction = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 0.0;
  stage.workload.reductionOperations = 1.0;
  stage.workload.shuffleLaneSteps = 0.0;
  stage.workload.scalarizedMultiResultReductionCounts[{24, 125, 2, 32}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.multiResultReductionGroupBaseWarpCount = 4;
  profile.simt.multiResultReductionResultCount = 2;
  profile.simt.multiResultReductionElementBits = 32;
  profile.simt.multiResultReductionPaddedAxisExtent = 128;
  profile.simt.multiResultReductionGroupEvidence = "unit tail surface";
  profile.simt.multiResultReductionGroupCyclesByTailClassAndRows = {
      {"non_power_divisible_by_4", {{8, 100.0}, {32, 400.0}}},
      {"tail_mod_4_eq_2", {{8, 200.0}, {32, 800.0}}},
      {"tail_odd", {{8, 300.0}, {32, 1200.0}}}};

  auto odd = evaluateOneStage(stage, profile);
  if (!odd)
    FAIL() << llvm::toString(odd.takeError());
  const auto &oddCost = odd->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(oddCost.resources.multiResultReductionGroupBody, 1200.0);
  EXPECT_DOUBLE_EQ(oddCost.totalCycles, 1200.0);
  EXPECT_NE(oddCost.formulaEvidence.find(
                "simt_multi_result_reduction_tail_row_surface"),
            std::string::npos);

  stage.workload.scalarizedMultiResultReductionCounts.clear();
  stage.workload.scalarizedMultiResultReductionCounts[{24, 126, 2, 32}] = 1.0;
  auto even = evaluateOneStage(stage, profile);
  if (!even)
    FAIL() << llvm::toString(even.takeError());
  EXPECT_DOUBLE_EQ(even->stages.front()
                       .implementations.front()
                       .resources.multiResultReductionGroupBody,
                   800.0);

  stage.workload.scalarizedMultiResultReductionCounts.clear();
  stage.workload.scalarizedMultiResultReductionCounts[{24, 124, 2, 32}] = 1.0;
  auto aligned = evaluateOneStage(stage, profile);
  if (!aligned)
    FAIL() << llvm::toString(aligned.takeError());
  EXPECT_DOUBLE_EQ(aligned->stages.front()
                       .implementations.front()
                       .resources.multiResultReductionGroupBody,
                   400.0);

  // A power-of-two axis and rows outside the calibrated interval retain the
  // analytical reduction formula instead of extrapolating a tail template.
  stage.workload.scalarizedMultiResultReductionCounts.clear();
  stage.workload.scalarizedMultiResultReductionCounts[{24, 128, 2, 32}] = 1.0;
  auto powerOfTwo = evaluateOneStage(stage, profile);
  if (!powerOfTwo)
    FAIL() << llvm::toString(powerOfTwo.takeError());
  EXPECT_DOUBLE_EQ(powerOfTwo->stages.front()
                       .implementations.front()
                       .resources.multiResultReductionGroupBody,
                   0.0);

  stage.workload.scalarizedMultiResultReductionCounts.clear();
  stage.workload.scalarizedMultiResultReductionCounts[{64, 125, 2, 32}] = 1.0;
  auto outsideRows = evaluateOneStage(stage, profile);
  if (!outsideRows)
    FAIL() << llvm::toString(outsideRows.takeError());
  EXPECT_DOUBLE_EQ(outsideRows->stages.front()
                       .implementations.front()
                       .resources.multiResultReductionGroupBody,
                   0.0);
}

TEST(SimdSimtCostModelTest,
     SimtRowwiseGroupSurfaceUsesBoundedTtirGeometryWithoutFactorScaling) {
  auto stage =
      logicalStage("rowwise_surface", StageCostModelKind::RowwiseReduction,
                   StageScheduleKind::IndependentPipelined, 3);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasReduction = true;
  stage.features.hasContiguousMemory = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.reductionOperations = 2.0;
  stage.workload.shuffleLaneSteps = 1.0;
  StageMemoryAccess input;
  input.shape = {24, 96};
  input.strides = {96, 1};
  input.elementBits = 32;
  stage.workload.memoryAccesses = {input};

  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.rowwiseReductionGroupBaseWarpCount = 4;
  profile.simt.rowwiseReductionGroupReferenceOperations = 2;
  profile.simt.rowwiseReductionGroupEvidence = "synthetic TTIR geometry sweep";
  profile.simt.rowwiseReductionGroupCyclesByFactorRowsAndExtent = {
      {1,
       {{16, {{64, 100.0}, {128, 200.0}}}, {32, {{64, 300.0}, {128, 500.0}}}}},
      {2,
       {{16, {{64, 200.0}, {128, 300.0}}}, {32, {{64, 400.0}, {128, 600.0}}}}},
      {4,
       {{16, {{64, 300.0}, {128, 400.0}}}, {32, {{64, 500.0}, {128, 700.0}}}}}};

  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  // Bilinear interpolation at rows=24, extent=96.  Every value describes the
  // complete physical group, so F2/F4 are not multiplied by F again.
  EXPECT_DOUBLE_EQ(costs[0].resources.rowwiseReductionGroupBody, 275.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.rowwiseReductionGroupBody, 375.0);
  EXPECT_DOUBLE_EQ(costs[2].resources.rowwiseReductionGroupBody, 475.0);
  EXPECT_DOUBLE_EQ(costs[0].totalCycles, 3.0 * 275.0);
  EXPECT_DOUBLE_EQ(costs[1].totalCycles, 3.0 * 375.0);
  EXPECT_DOUBLE_EQ(costs[2].totalCycles, 3.0 * 475.0);

  // Reduction geometry belongs to the reduction op, so the same structural
  // surface remains available when Stage partitioning separates its producer
  // load into an earlier Stage.
  stage.workload.memoryAccesses.clear();
  stage.workload.reductionGeometryCounts[{24, 96, 32}] = 2.0;
  auto detached = evaluateOneStage(stage, profile);
  if (!detached)
    FAIL() << llvm::toString(detached.takeError());
  EXPECT_DOUBLE_EQ(detached->stages.front()
                       .implementations[2]
                       .resources.rowwiseReductionGroupBody,
                   475.0);

  profile.simt.rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent[1] = {
      {1, {{1, {{64, 50.0}, {128, 75.0}}}}},
      {2, {{1, {{64, 60.0}, {128, 85.0}}}}},
      {4, {{1, {{64, 70.0}, {128, 95.0}}}}}};
  stage.workload.reductionOperations = 1.0;
  stage.workload.reductionGeometryCounts.clear();
  stage.workload.reductionGeometryCounts[{1, 96, 32}] = 1.0;
  auto oneReduction = evaluateOneStage(stage, profile);
  if (!oneReduction)
    FAIL() << llvm::toString(oneReduction.takeError());
  EXPECT_DOUBLE_EQ(oneReduction->stages.front()
                       .implementations[2]
                       .resources.rowwiseReductionGroupBody,
                   82.5);

  stage.workload.reductionGeometryCounts.clear();
  stage.workload.memoryAccesses = {input};
  stage.workload.memoryAccesses.front().shape = {24, 256};
  stage.workload.memoryAccesses.front().strides = {256, 1};
  auto outside = evaluateOneStage(stage, profile);
  if (!outside)
    FAIL() << llvm::toString(outside.takeError());
  for (const auto &cost : outside->stages.front().implementations)
    EXPECT_DOUBLE_EQ(cost.resources.rowwiseReductionGroupBody, 0.0);
}

TEST(SimdSimtCostModelTest,
     SimtRowwiseAnalyticalFallbackUsesWarpReductionGeometry) {
  auto stage = logicalStage("large_rowwise_fallback",
                            StageCostModelKind::RowwiseReduction,
                            StageScheduleKind::IndependentPipelined, 1);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasReduction = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.reductionOperations = 1.0;
  // Two independent extent-1024 rows contain 20480 source-level lane steps,
  // but a SIMT warp executes 32 per-lane chunks followed by five cross-lane
  // shuffle levels.  The fallback must use that lowering geometry.
  stage.workload.shuffleLaneSteps = 20480.0;
  stage.workload.reductionGeometryCounts[{2, 1024, 32}] = 1.0;

  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 1;
  profile.simt.rowwiseReductionGroupCyclesByFactorRowsAndExtent.clear();
  profile.simt.rowwiseReductionGroupCyclesByOperationFactorRowsAndExtent
      .clear();
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  for (size_t i = 0; i < costs.size(); ++i) {
    const auto &cost = costs[i];
    const double factor = static_cast<double>(int64_t{1} << i);
    EXPECT_DOUBLE_EQ(cost.resources.rowwiseReductionGroupBody, 0.0);
    EXPECT_GT(cost.resources.reduction, 0.0);
    EXPECT_GT(cost.resources.shuffle, 0.0);
    EXPECT_DOUBLE_EQ(cost.totalCycles,
                     factor *
                         std::max(cost.resources.scalar + cost.resources.load +
                                      cost.resources.store +
                                      cost.resources.criticalPath,
                                  cost.resources.issue));
  }

  // The complete F4 resident group covers all 128 lanes, so the dependency
  // envelope remains owned once while only the shared issue floor scales with
  // F.
  stage.workload.shuffleLaneSteps = 896.0;
  stage.workload.reductionGeometryCounts.clear();
  stage.workload.reductionGeometryCounts[{1, 128, 32}] = 1.0;
  auto shortAxis = evaluateOneStage(stage, profile);
  ASSERT_TRUE(bool(shortAxis));
  const auto &f4 = shortAxis->stages.front().implementations.back();
  EXPECT_DOUBLE_EQ(f4.totalCycles,
                   std::max(f4.resources.scalar + f4.resources.load +
                                f4.resources.store + f4.resources.criticalPath,
                            4.0 * f4.resources.issue));
}

TEST(SimdSimtCostModelTest,
     SimtUniformScalarStoreLatencyIsOwnedOncePerPhysicalGroup) {
  auto stage =
      logicalStage("uniform_scalar_store", StageCostModelKind::ScalarStore,
                   StageScheduleKind::StraightLine, 1);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.scalarStoreCount = 1.0;
  stage.workload.issueElements = 1.0;

  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.simtUniformStoreBaseCycles = 64.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  for (size_t i = 0; i < costs.size(); ++i) {
    const auto &cost = costs[i];
    const double factor = static_cast<double>(int64_t{1} << i);
    EXPECT_DOUBLE_EQ(cost.totalCycles,
                     std::max(cost.resources.scalar + cost.resources.store,
                              factor * cost.resources.issue));
  }
  EXPECT_LT(costs[2].totalCycles, 4.0 * costs[0].totalCycles);
}

TEST(SimdSimtCostModelTest,
     SimdIndirectLoadCurvePreservesSubWarpActiveElements) {
  auto stage =
      logicalStage("indirect_load", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::StraightLine, 1);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.features.hasIndirectMemory = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loadBytes = 32.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.loadWarpInstructions = 1.0;
  stage.workload.indirectLoadCounts[{32, 8}] = 1.0;

  auto profile = hardwareProfile();
  profile.simd.indirectLoadSystemCyclesByElements = {{8, 80.0}, {32, 320.0}};
  profile.simd.indirectLoadStartupSystemCyclesByWarpInstructions = {{1, 10.0}};
  auto eightElements = evaluateOneStage(stage, profile);
  if (!eightElements)
    FAIL() << llvm::toString(eightElements.takeError());
  EXPECT_DOUBLE_EQ(eightElements->stages.front()
                       .implementations.front()
                       .resources.indirectLoadGroup,
                   80.0);

  stage.workload.loadBytes = 128.0;
  stage.workload.indirectLoadCounts.clear();
  stage.workload.indirectLoadCounts[{128, 32}] = 1.0;
  auto fullWarp = evaluateOneStage(stage, profile);
  if (!fullWarp)
    FAIL() << llvm::toString(fullWarp.takeError());
  EXPECT_DOUBLE_EQ(fullWarp->stages.front()
                       .implementations.front()
                       .resources.indirectLoadGroup,
                   320.0);
}

TEST(SimdSimtCostModelTest,
     SimdIndirectStoreCurveUsesDistinctDestinationSegments) {
  auto stage =
      logicalStage("indirect_store", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::StraightLine, 1);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.features.hasIndirectMemory = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.storeBytes = 32.0;
  stage.workload.storeOperations = 1.0;
  stage.workload.storeWarpInstructions = 1.0;
  stage.workload.indirectStoreCounts[{32, 8}] = 1.0;

  auto profile = hardwareProfile();
  profile.simd.indirectStoreSystemCyclesBySegments = {{8, 800.0}, {32, 1600.0}};
  auto scalarScatter = evaluateOneStage(stage, profile);
  if (!scalarScatter)
    FAIL() << llvm::toString(scalarScatter.takeError());
  const auto &scalar = scalarScatter->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(scalar.resources.store, 800.0);
  EXPECT_FALSE(scalar.resources.indirectMemoryAnalyticalFallback);
  EXPECT_NE(
      scalar.formulaEvidence.find("measured_indirect_store_segment_curve"),
      std::string::npos);

  stage.workload.indirectStoreCounts.clear();
  stage.workload.storeBytes = 1024.0;
  stage.workload.storeWarpInstructions = 8.0;
  stage.workload.segmentedIndirectStoreCounts[{1024, 256, 32}] = 1.0;
  auto segmentedScatter = evaluateOneStage(stage, profile);
  if (!segmentedScatter)
    FAIL() << llvm::toString(segmentedScatter.takeError());
  const auto &segmented =
      segmentedScatter->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(segmented.resources.store, 800.0);
  EXPECT_FALSE(segmented.resources.indirectMemoryAnalyticalFallback);
}

TEST(SimdSimtCostModelTest,
     SimtSegmentedIndirectLoadUsesFactorSegmentExtentSurface) {
  auto stage = logicalStage("segmented_indirect_load",
                            StageCostModelKind::IndirectGatherMemory,
                            StageScheduleKind::StraightLine, 1);
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  stage.features.hasIndirectMemory = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loadBytes = 4096.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.loadWarpInstructions = 64.0;
  stage.workload.segmentedIndirectLoadCounts[{4096, 2048, 1024}] = 1.0;

  auto profile = hardwareProfile();
  profile.simt
      .segmentedIndirectLoadGroupCyclesByFactorSegmentsAndContiguousElements = {
      {1, {{2, {{256, 100.0}, {2048, 300.0}}}}},
      {2, {{2, {{256, 200.0}, {2048, 400.0}}}}},
      {4, {{2, {{256, 300.0}, {2048, 500.0}}}}}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_DOUBLE_EQ(costs[0].resources.indirectLoadGroup,
                   100.0 + (300.0 - 100.0) * 768.0 / 1792.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.indirectLoadGroup,
                   200.0 + (400.0 - 200.0) * 768.0 / 1792.0);
  EXPECT_DOUBLE_EQ(costs[2].resources.indirectLoadGroup,
                   300.0 + (500.0 - 300.0) * 768.0 / 1792.0);
  for (const auto &cost : costs)
    EXPECT_FALSE(cost.resources.indirectMemoryAnalyticalFallback);
}

TEST(SimdSimtCostModelTest,
     SimdSegmentedIndirectLoadUsesElementWidthSpecificSurface) {
  auto stage = logicalStage("segmented_indirect_load",
                            StageCostModelKind::IndirectGatherMemory,
                            StageScheduleKind::StraightLine, 1);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.features.hasIndirectMemory = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loadBytes = 1024.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.segmentedIndirectLoadCounts[{1024, 256, 32}] = 1.0;

  auto profile = hardwareProfile();
  profile.simd
      .segmentedIndirectLoadGroupCyclesByFactorSegmentsAndContiguousElements = {
      {1, {{8, {{32, 200.0}}}}}};
  profile.simd
      .segmentedIndirectLoadFp32GroupCyclesByFactorSegmentsAndContiguousElements =
      {{1, {{8, {{32, 600.0}}}}}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.indirectLoadGroup, 600.0);
  EXPECT_DOUBLE_EQ(cost.resources.load, 0.0);
  EXPECT_DOUBLE_EQ(cost.totalCycles, 600.0);
  EXPECT_FALSE(cost.resources.indirectMemoryAnalyticalFallback);
  EXPECT_NE(
      cost.formulaEvidence.find("measured_segmented_indirect_load_surface"),
      std::string::npos);
}

TEST(SimdSimtCostModelTest, SimdIrregularTileUsesStructuralCompletionBounds) {
  auto profile = hardwareProfile();
  profile.simd.irregularTileCompletionEnabled = true;
  profile.simd.irregularTilePhysicalThresholdElements = 100;
  profile.simd.irregularTilePhysicalExcessCyclesPerElement = 1.0;
  profile.simd.irregularTileActiveCyclesPerElement = 2.0;
  profile.simd.irregularTileTailCyclesPerInactiveElement = 3.0;
  profile.simd.irregularTileRowGroupSize = 32;
  profile.simd.irregularTileSmallRowMaximumRows = 72;
  profile.simd.irregularTileLargeRowMaskedFloorScale = 0.5;
  profile.simd.irregularTileAlignedTwoRowGroupPhysicalMaterializationScale =
      0.25;
  profile.simd.irregularTileLargeWorkingSetFullPhysicalMinimumBytes = 4096;
  profile.simd.irregularTileAlignedFirstRowGroupCyclesPerElement = 0.5;
  profile.simd.irregularTileAlignedAdditionalRowGroupCyclesPerElement = 0.25;
  profile.simd.irregularTileCompletionEvidence = "synthetic structural fit";

  auto stage =
      logicalStage("masked", StageCostModelKind::IndependentPipelinedLoop,
                   StageScheduleKind::IndependentPipelined, 2);
  StageMemoryAccess masked;
  masked.shape = {32, 256};
  masked.strides = {256, 1};
  masked.elementBits = 32;
  masked.masked = true;
  masked.activeElementRatio = 150.0 / 256.0;
  stage.workload.memoryAccesses = {masked};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &maskedCost = table->stages.front().implementations.front();
  const double physical = 32.0 * 256.0;
  const double materialization = physical - 100.0;
  const double floor = materialization + 2.0 * 32.0 * 150.0;
  EXPECT_DOUBLE_EQ(maskedCost.resources.irregularPhysicalMaterialization,
                   materialization);
  EXPECT_DOUBLE_EQ(maskedCost.resources.irregularFullPhysicalMaterialization,
                   materialization);
  EXPECT_DOUBLE_EQ(maskedCost.resources.irregularMaskedTileFloor, floor);
  EXPECT_DOUBLE_EQ(maskedCost.totalCycles, 10.0 + 2.0 * floor);

  stage.id = "partial_second_group_masked";
  stage.iterationCount = 1;
  stage.workload.memoryAccesses[0].shape = {48, 256};
  stage.workload.memoryAccesses[0].strides = {256, 1};
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &partialSecondGroup =
      table->stages.front().implementations.front();
  const double partialPhysical = 48.0 * 256.0;
  const double partialMaterialization = partialPhysical - 100.0;
  EXPECT_DOUBLE_EQ(
      partialSecondGroup.resources.irregularPhysicalMaterialization,
      0.5 * partialMaterialization);
  EXPECT_DOUBLE_EQ(partialSecondGroup.resources.irregularMaskedTileFloor,
                   partialMaterialization + 2.0 * 48.0 * 150.0);

  stage.id = "two_groups_masked";
  stage.workload.memoryAccesses[0].shape = {64, 256};
  stage.workload.memoryAccesses[0].strides = {256, 1};
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &twoGroups = table->stages.front().implementations.front();
  const double twoGroupPhysical = 64.0 * 256.0;
  const double twoGroupMaterialization = twoGroupPhysical - 100.0;
  EXPECT_DOUBLE_EQ(twoGroups.resources.irregularPhysicalMaterialization,
                   0.25 * twoGroupMaterialization);
  EXPECT_DOUBLE_EQ(twoGroups.resources.irregularFullPhysicalMaterialization,
                   twoGroupMaterialization);
  EXPECT_DOUBLE_EQ(twoGroups.resources.irregularMaskedTileFloor,
                   twoGroupMaterialization + 2.0 * 64.0 * 150.0);

  stage.id = "large_masked";
  stage.iterationCount = 1;
  stage.workload.memoryAccesses[0].shape = {80, 256};
  stage.workload.memoryAccesses[0].strides = {256, 1};
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &largeMasked = table->stages.front().implementations.front();
  const double largePhysical = 80.0 * 256.0;
  const double largeMaterialization = 0.5 * (largePhysical - 100.0);
  const double largeFloor = largeMaterialization + 0.5 * 2.0 * 80.0 * 150.0;
  EXPECT_DOUBLE_EQ(largeMasked.resources.irregularPhysicalMaterialization, 0.0);
  EXPECT_DOUBLE_EQ(largeMasked.resources.irregularFullPhysicalMaterialization,
                   largeMaterialization);
  EXPECT_DOUBLE_EQ(largeMasked.resources.irregularMaskedTileFloor, largeFloor);

  stage.id = "tail";
  stage.iterationCount = 1;
  stage.workload.memoryAccesses[0].shape = {80, 150};
  stage.workload.memoryAccesses[0].strides = {150, 1};
  stage.workload.memoryAccesses[0].activeElementRatio = 1.0;
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &tailCost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(tailCost.resources.irregularMaskedTileFloor, 0.0);
  EXPECT_DOUBLE_EQ(tailCost.resources.irregularTailMaterialization,
                   80.0 * (192.0 - 150.0) * 3.0);

  stage.id = "full_group_unaligned_reduction";
  stage.workload.memoryAccesses[0].shape = {32, 150};
  stage.workload.memoryAccesses[0].strides = {150, 1};
  stage.workload.reductionOperations = 1.0;
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_DOUBLE_EQ(table->stages.front()
                       .implementations.front()
                       .resources.irregularMaskedTileFloor,
                   0.0);

  stage.id = "aligned_groups";
  stage.workload.reductionOperations = 0.0;
  stage.workload.memoryAccesses[0].shape = {64, 128};
  stage.workload.memoryAccesses[0].strides = {128, 1};
  stage.workload.memoryAccesses[0].masked = false;
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_DOUBLE_EQ(table->stages.front()
                       .implementations.front()
                       .resources.irregularMaskedTileFloor,
                   32.0 * 128.0 * (0.5 + 0.25));

  stage.id = "partial_aligned";
  stage.workload.memoryAccesses[0].shape = {48, 128};
  stage.workload.memoryAccesses[0].strides = {128, 1};
  stage.workload.memoryAccesses[0].masked = false;
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_DOUBLE_EQ(table->stages.front()
                       .implementations.front()
                       .resources.irregularMaskedTileFloor,
                   0.0);
}

TEST(SimdSimtCostModelTest,
     MaskedStridedCopyCompletionUsesDirectionAndMaskProvenance) {
  auto profile = hardwareProfile();
  mlir::ascend::MaskedStridedCopyCompletionCost scatter;
  scatter.baseSystemCycles = 300.0;
  scatter.additionalRowSystemCycles = 1.0;
  scatter.minimumRows = 128;
  scatter.maximumRows = 256;
  scatter.minimumIterations = 8;
  scatter.maximumIterations = 8;
  scatter.evidence = "synthetic masked scatter fit";
  profile.simd.maskedStridedStoreCopyCompletionByStrideBytes[64] = scatter;

  auto stage = logicalStage("masked_scatter",
                            StageCostModelKind::IndependentPipelinedLoop,
                            StageScheduleKind::IndependentPipelined, 8);
  stage.simtLegal = false;
  stage.legalSimtFactors.clear();
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loadBytes = 1024.0;
  stage.workload.storeBytes = 1024.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.storeOperations = 1.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.storeWarpInstructions = 8.0;

  StageMemoryAccess load;
  load.shape = {256};
  load.strides = {1};
  load.elementBits = 32;
  load.masked = true;
  StageMemoryAccess store = load;
  store.strides = {16};
  store.store = true;
  stage.workload.memoryAccesses = {load, store};

  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.store, 428.0);
  EXPECT_DOUBLE_EQ(cost.totalCycles, 8.0 * 428.0);

  // Reusing the exact load predicate is a different backend lowering and is
  // deliberately excluded from the separate-mask scatter curve.
  stage.workload.memoryAccesses[1].maskedStoreHasMatchingLoadDependency = true;
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &sharedMask = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(sharedMask.resources.store, 64.0);
  EXPECT_DOUBLE_EQ(sharedMask.totalCycles, 8.0 * 64.0);
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

TEST(SimdSimtCostModelTest,
     RecurrenceCriticalPathIncludesMeasuredIndirectLoadService) {
  LogicalStage stage = logicalStage("indirect_recurrence",
                                    StageCostModelKind::LoopCarriedRecurrence,
                                    StageScheduleKind::LoopCarriedSerial, 4);
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.features.hasIndirectMemory = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.operationCounts["f32.add"][32] = 1.0;
  stage.workload.loadBytes = 2048.0;
  stage.workload.loadWarpInstructions = 16.0;
  // One FP32 loaded-index operation is covered by the measured group curve;
  // the second operation remains on the generic analytical fallback.
  stage.workload.indirectLoadCounts = {{{1024, 256}, 1.0}};

  HardwareProfile profile = hardwareProfile();
  profile.simd.indirectLoadSystemCyclesByElements = {{256, 100.0}};
  profile.simd.indirectLoadStartupSystemCyclesByWarpInstructions = {{8, 10.0}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.indirectLoadGroup, 100.0);
  EXPECT_GT(cost.resources.load, 0.0);
  EXPECT_GT(cost.resources.criticalPath, 0.0);
  EXPECT_TRUE(cost.resources.indirectMemoryAnalyticalFallback);
  EXPECT_GE(cost.totalCycles,
            4.0 * (cost.resources.criticalPath + cost.resources.load +
                   cost.resources.indirectLoadGroup));
  EXPECT_NE(
      cost.formulaEvidence.find("analytical_indirect_memory_resource_fallback"),
      std::string::npos);
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
  profile.simt.recurrenceReductionGroupStartupCyclesByFactorAndLaneSteps = {
      {1, {{1024, 25.0}, {4096, 100.0}}},
      {2, {{1024, 35.0}, {4096, 140.0}}},
      {4, {{1024, 50.0}, {4096, 200.0}}}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 3u);
  EXPECT_DOUBLE_EQ(costs[0].resources.recurrenceGroupCriticalPath, 100.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.recurrenceGroupCriticalPath, 130.0);
  EXPECT_DOUBLE_EQ(costs[2].resources.recurrenceGroupCriticalPath, 190.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.recurrenceGroupStartup, 25.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.recurrenceGroupStartup, 35.0);
  EXPECT_DOUBLE_EQ(costs[2].resources.recurrenceGroupStartup, 50.0);
  EXPECT_DOUBLE_EQ(costs[0].totalCycles, 1635.0);
  EXPECT_DOUBLE_EQ(costs[1].totalCycles, 2125.0);
  EXPECT_DOUBLE_EQ(costs[2].totalCycles, 3100.0);

  table->logicalProgramCountHint = 224;
  table->physicalCoreCountHint = 56;
  auto routes = solveStageRoutes(*table, profile.scopeHandoff);
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_EQ(routes->allSimt.routeSuperblockFactor, 4);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 3100.0);

  stage.legalSimtFactors = {1};
  stage.workload.shuffleLaneSteps = 2048.0;
  auto interpolated = evaluateOneStage(stage, profile);
  if (!interpolated)
    FAIL() << llvm::toString(interpolated.takeError());
  EXPECT_DOUBLE_EQ(interpolated->stages.front()
                       .implementations.front()
                       .resources.recurrenceGroupCriticalPath,
                   200.0);
  EXPECT_DOUBLE_EQ(interpolated->stages.front()
                       .implementations.front()
                       .resources.recurrenceGroupStartup,
                   50.0);
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

TEST(SimdSimtCostModelTest, IndirectMemoryUsesGenericResourceFallback) {
  LogicalStage stage =
      logicalStage("indirect", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::PartiallyDependent);
  stage.features.hasIndirectMemory = true;
  stage.workload.loadBytes = 1024.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.indirectLoadCounts[std::make_pair(1024, 256)] = 1.0;

  auto table = evaluateOneStage(stage, hardwareProfile());
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  ASSERT_EQ(table->stages.front().implementations.size(), 2u);
  for (const auto &cost : table->stages.front().implementations) {
    EXPECT_TRUE(cost.resources.indirectMemoryAnalyticalFallback);
    EXPECT_GT(cost.resources.load, 0.0);
    EXPECT_EQ(cost.formulaEvidence,
              "analytical_indirect_memory_resource_fallback");
  }
}

TEST(SimdSimtCostModelTest,
     SegmentedIndirectLoadUsesProvenRunsInsteadOfScalarGatherCurve) {
  LogicalStage stage = logicalStage("segmented_indirect",
                                    StageCostModelKind::IndirectGatherMemory,
                                    StageScheduleKind::PartiallyDependent);
  stage.features.hasIndirectMemory = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loadBytes = 512.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.segmentedIndirectLoadCounts[{512, 256, 16}] = 1.0;

  HardwareProfile profile = hardwareProfile();
  profile.simd.loadOperationSetupCycles = 10.0;
  profile.simd.loadBytesPerCycle = 128.0;
  profile.simd.indirectLoadSystemCyclesByWarpInstructions = {{8, 1000.0}};
  profile.simd.indirectLoadStartupSystemCyclesByWarpInstructions = {{8, 100.0}};
  profile.simt.loadWarpInstructionsPerCycle = 2.0;
  profile.simt.indirectLoadSystemCyclesByWarpInstructions = {{8, 2000.0}};
  profile.simt.indirectLoadStartupSystemCyclesByWarpInstructions = {{8, 200.0}};

  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 2u);
  EXPECT_DOUBLE_EQ(costs[0].resources.indirectLoadGroup, 0.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.indirectLoadGroupStartup, 0.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.load, 16.0 * 10.0 + 512.0 / 128.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.indirectLoadGroup, 0.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.indirectLoadGroupStartup, 0.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.load, 8.0 / 2.0);
  for (const auto &cost : costs)
    EXPECT_NE(cost.formulaEvidence.find(
                  "segmented_indirect_load_analytical_fallback"),
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
  profile.simd.indirectLoadSystemCyclesByElements = {{256, 100.0},
                                                     {512, 220.0}};
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
  EXPECT_DOUBLE_EQ(costs[0].resources.indirectLoadGroup, 320.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.indirectLoadGroupStartup, 30.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.indirectLoadGroup, 80.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.indirectLoadGroupStartup, 15.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.load, 0.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.load, 0.0);
  EXPECT_EQ(costs[0].formulaEvidence, "measured_indirect_load_curve");
  EXPECT_EQ(costs[1].formulaEvidence, "measured_indirect_load_curve");

  stage.workload.loadBytes = 768.0;
  stage.workload.loadWarpInstructions = 12.0;
  stage.workload.indirectLoadCounts = {{{768, 384}, 1.0}};
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  EXPECT_DOUBLE_EQ(
      table->stages.front().implementations[0].resources.indirectLoadGroup,
      160.0);
  EXPECT_DOUBLE_EQ(table->stages.front()
                       .implementations[0]
                       .resources.indirectLoadGroupStartup,
                   20.0);
  EXPECT_DOUBLE_EQ(
      table->stages.front().implementations[1].resources.indirectLoadGroup,
      40.0);
  EXPECT_DOUBLE_EQ(table->stages.front()
                       .implementations[1]
                       .resources.indirectLoadGroupStartup,
                   10.0);

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
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  ASSERT_EQ(table->stages.front().implementations.size(), 2u);
  const auto &fp32Costs = table->stages.front().implementations;
  EXPECT_FALSE(fp32Costs[0].resources.indirectMemoryAnalyticalFallback);
  EXPECT_DOUBLE_EQ(fp32Costs[0].resources.indirectLoadGroup, 100.0);
  EXPECT_DOUBLE_EQ(fp32Costs[0].resources.indirectLoadGroupStartup, 10.0);
  EXPECT_DOUBLE_EQ(fp32Costs[0].resources.load, 0.0);
  EXPECT_EQ(fp32Costs[0].formulaEvidence, "measured_indirect_load_curve");
  EXPECT_TRUE(fp32Costs[1].resources.indirectMemoryAnalyticalFallback);
  EXPECT_DOUBLE_EQ(fp32Costs[1].resources.load, 8.0);
  EXPECT_EQ(fp32Costs[1].formulaEvidence,
            "analytical_indirect_memory_resource_fallback");
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
  profile.simd.indirectLoadSystemCyclesByElements = {{1, 10}, {2, 20}};
  auto costs = evaluateOneStage(stage, profile);
  if (!costs)
    FAIL() << llvm::toString(costs.takeError());
  for (const auto &cost : costs->stages.front().implementations) {
    EXPECT_DOUBLE_EQ(cost.resources.indirectLoadGroup, 20);
    EXPECT_DOUBLE_EQ(cost.resources.indirectLoadGroupStartup, 10);
    EXPECT_DOUBLE_EQ(cost.resources.load, 0);
  }

  // Same four bytes and same single issued warp, but now one FP32 element.
  // It must not match the FP16 curve just because the old key was identical.
  stage.workload.indirectLoadCounts = {{{4, 1}, 1}};
  costs = evaluateOneStage(stage, profile);
  if (!costs)
    FAIL() << llvm::toString(costs.takeError());
  const auto &fp32Costs = costs->stages.front().implementations;
  EXPECT_FALSE(fp32Costs[0].resources.indirectMemoryAnalyticalFallback);
  EXPECT_DOUBLE_EQ(fp32Costs[0].resources.indirectLoadGroup, 10.0);
  EXPECT_DOUBLE_EQ(fp32Costs[0].resources.indirectLoadGroupStartup, 10.0);
  EXPECT_DOUBLE_EQ(fp32Costs[0].resources.load, 0.0);
  EXPECT_EQ(fp32Costs[0].formulaEvidence, "measured_indirect_load_curve");
  EXPECT_TRUE(fp32Costs[1].resources.indirectMemoryAnalyticalFallback);
  EXPECT_DOUBLE_EQ(fp32Costs[1].resources.load, 1.0);
  EXPECT_EQ(fp32Costs[1].formulaEvidence,
            "analytical_indirect_memory_resource_fallback");
}

TEST(SimdSimtCostModelTest,
     MeasuredIndirectLoadGroupIsNotMultipliedBySuperBlockFactor) {
  LogicalStage stage =
      logicalStage("indirect_group", StageCostModelKind::IndirectGatherMemory,
                   StageScheduleKind::PartiallyDependent);
  stage.features.hasIndirectMemory = true;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;
  stage.workload.loadBytes = 512.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.indirectLoadCounts = {{{512, 256}, 1.0}};

  HardwareProfile profile = hardwareProfile();
  profile.simt.indirectLoadSystemCyclesByWarpInstructions = {{8, 20.0}};
  profile.simt.indirectLoadStartupSystemCyclesByWarpInstructions = {{8, 5.0}};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());

  const auto &estimates = table->stages.front().simtCalibrationEstimates;
  for (int64_t factor : {1, 2, 4}) {
    auto found = llvm::find_if(estimates, [factor](const auto &cost) {
      return cost.implementation.superblockFactor == factor &&
             cost.implementation.aivSubblocks == 1;
    });
    ASSERT_NE(found, estimates.end());
    EXPECT_DOUBLE_EQ(found->resources.indirectLoadGroup, 20.0);
    EXPECT_DOUBLE_EQ(found->resources.indirectLoadGroupStartup, 5.0);
    EXPECT_DOUBLE_EQ(found->resources.load, 0.0);
    EXPECT_DOUBLE_EQ(found->totalCycles, 20.0);
  }
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
  profile.simd.atomicRates["default"] = {8.0, 0.0, 0.0, 1.0};
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
     SimdRoutesRejectMaskedLoadedIndexVectorGatherLoweringGap) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles, bool localScope = false) {
    StageImplementationCost cost;
    cost.implementation = {mode, 1, localScope};
    cost.totalCycles = cycles;
    return cost;
  };

  LogicalStageCost gather;
  gather.id = "coalesced_scalar_indirect_load";
  gather.localSimtMaterializable = true;
  gather.localSimtFactors = {1};
  gather.legalSimtFactors = {1};
  StageMemoryAccess access;
  access.shape = {2};
  access.strides = {std::nullopt};
  access.elementBits = 16;
  access.masked = true;
  access.addressDependsOnLoadedIndex = true;
  gather.workload.memoryAccesses.push_back(access);
  gather.implementations = {makeCost(StageMode::SIMD, 1.0),
                            makeCost(StageMode::SIMT, 4.0),
                            makeCost(StageMode::SIMT, 4.0, true)};

  LogicalStageCost peer;
  peer.id = "simd_peer";
  peer.implementations = {makeCost(StageMode::SIMD, 1.0),
                          makeCost(StageMode::SIMT, 4.0)};
  table.stages = {gather, peer};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_FALSE(routes->allSimd.legal);
  EXPECT_TRUE(routes->allSimd.diagnosticCostAvailable);
  EXPECT_EQ(routes->allSimd.implementations.size(), 2u);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 2.0);
  EXPECT_EQ(
      routes->allSimd.ineligibilityReason,
      "masked_loaded_index_vector_gather_not_materializable_in_simd_backend");
  EXPECT_FALSE(routes->mixed.legal);
  EXPECT_FALSE(routes->mixed.diagnosticCostAvailable);
  EXPECT_EQ(
      routes->mixed.ineligibilityReason,
      "masked_loaded_index_vector_gather_not_materializable_in_simd_backend");
  EXPECT_TRUE(routes->allSimt.legal);
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

TEST(SimdSimtCostModelTest, MixedFactorOneSelectsExactlyOneLocalScope) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  auto makeCost = [&](StageMode mode, double cycles, bool localScope = false) {
    StageImplementationCost cost;
    cost.implementation = {mode, 1, localScope};
    cost.totalCycles = cycles;
    return cost;
  };

  LogicalStageCost first;
  first.id = "first_local_scope";
  first.localSimtMaterializable = true;
  first.localSimtFactors = {1};
  first.implementations = {makeCost(StageMode::SIMD, 100.0),
                           makeCost(StageMode::SIMT, 1.0, true)};

  LogicalStageCost second;
  second.id = "second_local_scope";
  second.localSimtMaterializable = true;
  second.localSimtFactors = {1};
  second.implementations = {makeCost(StageMode::SIMD, 80.0),
                            makeCost(StageMode::SIMT, 1.0, true)};

  LogicalStageCost peer;
  peer.id = "simd_peer";
  peer.implementations = {makeCost(StageMode::SIMD, 1.0)};
  table.stages = {first, second, peer};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(llvm::count_if(
                routes->mixed.implementations,
                [](const mlir::ascend::StageImplementation &implementation) {
                  return implementation.mode == StageMode::SIMT &&
                         implementation.localScope;
                }),
            1);
  EXPECT_EQ(routes->mixed.implementations[0].mode, StageMode::SIMT);
  EXPECT_EQ(routes->mixed.implementations[1].mode, StageMode::SIMD);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 82.0);
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

  auto makeCost = [](StageMode mode, double cycles, bool localScope = false,
                     int64_t factor = 1) {
    StageImplementationCost cost;
    cost.implementation = {mode, factor, localScope};
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

TEST(SimdSimtCostModelTest,
     SourceGridStridePureSimtFactorsReuseTheFactorOneBody) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 560;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;

  auto makeCost = [](int64_t factor, double cycles) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, factor};
    cost.totalCycles = cycles;
    return cost;
  };
  LogicalStageCost stage;
  stage.id = "source_grid_stride";
  stage.model = "independent_pipelined_loop";
  stage.features.hasLoop = true;
  stage.features.hasContiguousMemory = true;
  stage.features.isGridStrideLoopShell = true;
  stage.implementations = {makeCost(1, 100.0), makeCost(2, 250.0),
                           makeCost(4, 600.0)};
  table.stages = {stage};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  for (int64_t factor : {1, 2, 4}) {
    const auto &plan = routes->allSimtFactorPlans.at(factor);
    EXPECT_DOUBLE_EQ(plan.totalCycles, 1000.0);
    EXPECT_EQ(plan.runtimeStageLoopIterationCounts[0], 10);
  }
  EXPECT_FALSE(routes->allSimtFactorPlans.at(1).gridStrideFactorOneBodyApplied);
  EXPECT_TRUE(routes->allSimtFactorPlans.at(2).gridStrideFactorOneBodyApplied);
  EXPECT_TRUE(routes->allSimtFactorPlans.at(4).gridStrideFactorOneBodyApplied);
  EXPECT_EQ(routes->allSimtFactorPlans.at(4).runtimeStageTailModes[0],
            "source_grid_stride_factor1_body");
}

TEST(SimdSimtCostModelTest,
     PureSimtSuperBlockReportsTheMaterializedPhysicalLaunchGrid) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 128;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;

  auto makeCost = [](int64_t factor) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, factor};
    cost.totalCycles = 10.0;
    return cost;
  };
  LogicalStageCost stage;
  stage.id = "simt_group";
  stage.model = "scalar_issue";
  stage.implementations = {makeCost(1), makeCost(2), makeCost(4)};
  table.stages = {stage};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_EQ(routes->allSimtFactorPlans.at(1).runtimePhysicalProgramCount, 56);
  EXPECT_EQ(routes->allSimtFactorPlans.at(2).runtimePhysicalProgramCount, 56);
  EXPECT_EQ(routes->allSimtFactorPlans.at(4).runtimePhysicalProgramCount, 32);
  // Scheduling still uses all 56 resident core slots; F changes the launch
  // grid and per-slot grouping, not the hardware slot count.
  EXPECT_EQ(routes->allSimtFactorPlans.at(4).runtimeSchedulingSlotCount, 56);
}

TEST(SimdSimtCostModelTest,
     SourceGridStrideSimtScanKeepsFactorResourceRateAndSerialQ) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 224;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;

  auto makeCost = [](int64_t factor, double cycles, double scanBody = 0.0,
                     double startup = 0.0,
                     bool factorSpecificResource = false) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, factor};
    cost.totalCycles = cycles;
    cost.resources.gridStridePrefixScanBody = scanBody;
    cost.resources.gridStrideStartup = startup;
    cost.resources.gridStrideFactorSpecificResourceBody =
        factorSpecificResource;
    return cost;
  };
  LogicalStageCost shell;
  shell.id = "source_grid_stride";
  shell.model = "independent_pipelined_loop";
  shell.features.isGridStrideLoopShell = true;
  shell.implementations = {makeCost(1, 10.0), makeCost(2, 20.0),
                           makeCost(4, 40.0)};

  LogicalStageCost scan;
  scan.id = "factor_scan";
  scan.model = "prefix_scan";
  scan.features.insideGridStrideLoopBody = true;
  scan.features.hasPrefixScan = true;
  // Stage scoring stores F times the per-source-iteration scan body.
  scan.implementations = {makeCost(1, 250.0, 250.0, 50.0),
                          makeCost(2, 450.0, 225.0, 25.0, true),
                          makeCost(4, 1240.0, 310.0, 70.0, true)};
  table.stages = {shell, scan};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  // Every factor executes Q=4 source iterations. The shell keeps its F1
  // semantic body while the scan uses the selected active-warp resource rate.
  EXPECT_DOUBLE_EQ(routes->allSimtFactorPlans.at(1).totalCycles,
                   4.0 * (10.0 + 250.0) + 50.0);
  EXPECT_DOUBLE_EQ(routes->allSimtFactorPlans.at(2).totalCycles,
                   4.0 * (10.0 + 225.0) + 25.0);
  EXPECT_DOUBLE_EQ(routes->allSimtFactorPlans.at(4).totalCycles,
                   4.0 * (10.0 + 310.0) + 70.0);
  for (int64_t factor : {1, 2, 4})
    EXPECT_EQ(routes->allSimtFactorPlans.at(factor)
                  .runtimeStageLoopIterationCounts[1],
              4);
}

TEST(SimdSimtCostModelTest,
     FactorOneGridStrideGatherBoundaryOwnsFirstIndirectService) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 224;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;

  auto makeCost = [](double cycles, double indirect = 0.0,
                     double startup = 0.0) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, 1};
    cost.totalCycles = cycles;
    cost.resources.indirectLoadGroup = indirect;
    cost.resources.gridStrideStartup = startup;
    return cost;
  };
  LogicalStageCost shell;
  shell.id = "source_grid_stride";
  shell.model = "independent_pipelined_loop";
  shell.features.isGridStrideLoopShell = true;
  shell.implementations = {makeCost(0.0)};

  LogicalStageCost gather;
  gather.id = "dynamic_gather";
  gather.model = "indirect_gather_memory";
  gather.features.insideGridStrideLoopBody = true;
  gather.features.hasIndirectMemory = true;
  gather.implementations = {makeCost(100.0, 100.0, 50.0)};
  table.stages = {shell, gather};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimt.legal);
  // Q=4: the boundary owns the first 100-cycle indirect service, the other
  // three remain serialized, and the one-time grid startup remains charged.
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 3.0 * 100.0 + 50.0);
  EXPECT_TRUE(routes->allSimt.gridStrideFirstIndirectServiceOwnedByBoundary);
  EXPECT_DOUBLE_EQ(
      routes->allSimt.gridStrideFirstIndirectServiceAdjustmentCycles, -100.0);

  LogicalStageCost dot;
  dot.id = "dynamic_dot";
  dot.model = "tiny_cube_roofline";
  dot.features.insideGridStrideLoopBody = true;
  dot.features.hasDot = true;
  dot.implementations = {makeCost(10.0)};
  table.stages = {shell, gather, dot};
  routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimt.legal);
  // A gather-dot body keeps all four serialized gather services.  Its first
  // indirect interval is part of the composed body, not a pure-gather
  // boundary transfer.
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 50.0 + 4.0 * (100.0 + 10.0));
  EXPECT_FALSE(routes->allSimt.gridStrideFirstIndirectServiceOwnedByBoundary);
}

TEST(SimdSimtCostModelTest,
     SourceGridStrideFusedSimtScanKeepsFactorOneDirectMemory) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 224;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;

  auto makeCost = [](int64_t factor, double cycles, double scanBody,
                     double load, double store,
                     bool factorSpecificResource = false) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, factor};
    cost.totalCycles = cycles;
    cost.resources.gridStridePrefixScanBody = scanBody;
    cost.resources.load = load;
    cost.resources.store = store;
    cost.resources.gridStrideFactorSpecificResourceBody =
        factorSpecificResource;
    return cost;
  };
  LogicalStageCost shell;
  shell.id = "source_grid_stride";
  shell.model = "independent_pipelined_loop";
  shell.features.isGridStrideLoopShell = true;
  shell.implementations = {makeCost(1, 10.0, 0.0, 0.0, 0.0),
                           makeCost(2, 20.0, 0.0, 0.0, 0.0)};

  LogicalStageCost fusedScan;
  fusedScan.id = "fused_load_scan_store";
  fusedScan.model = "prefix_scan";
  fusedScan.features.insideGridStrideLoopBody = true;
  fusedScan.features.hasPrefixScan = true;
  fusedScan.features.hasContiguousMemory = true;
  // F1: scan 250 + direct memory 100 + 50 = 400.
  // F2: two programs at scan 225 + factor-rate memory 50 + 25 = 600.
  fusedScan.implementations = {makeCost(1, 400.0, 250.0, 100.0, 50.0),
                               makeCost(2, 600.0, 225.0, 50.0, 25.0, true)};
  table.stages = {shell, fusedScan};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  const auto &factor2 = routes->allSimtFactorPlans.at(2);
  // Q=4 source iterations. The shell uses its F1 body. The scan keeps its F2
  // resource rate, while fused direct memory uses the F1 semantic stream.
  EXPECT_DOUBLE_EQ(factor2.totalCycles, 4.0 * (10.0 + 225.0 + 100.0 + 50.0));
  EXPECT_TRUE(factor2.gridStrideFactorOneDirectMemoryApplied);
  EXPECT_DOUBLE_EQ(factor2.gridStrideFactorOneDirectMemoryAdjustmentCycles,
                   4.0 * (100.0 + 50.0 - 50.0 - 25.0));
  EXPECT_FALSE(
      routes->allSimtFactorPlans.at(1).gridStrideFactorOneDirectMemoryApplied);
}

TEST(SimdSimtCostModelTest,
     SourceGridStrideSimtAppliesStructuralGatherDotCompositionResidual) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 224;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdGridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry = {
      {1, {{8, {{8192, 5.0}}}}}};
  table.simdGridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry =
      {{1, {{8, {{8192, -7.0}}}}}};
  table.simdGridStrideIndirectDotCompositionEvidence =
      "synthetic SIMD structural resource-composition sweep";
  table.simtGridStrideIndirectDotCompositionBodyResidualByFactorAndGeometry = {
      {4, {{8, {{8192, 5.0}}}}}};
  table.simtGridStrideIndirectDotCompositionStartupResidualByFactorAndGeometry =
      {{4, {{8, {{8192, -7.0}}}}}};
  table.simtGridStrideIndirectDotCompositionEvidence =
      "synthetic structural resource-composition sweep";

  auto makeCost = [](int64_t factor, double cycles,
                     bool factorSpecificResource = false) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, factor};
    cost.totalCycles = cycles;
    cost.resources.gridStrideFactorSpecificResourceBody =
        factorSpecificResource;
    return cost;
  };
  auto makeSimdCost = [](double cycles) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMD, 1};
    cost.totalCycles = cycles;
    return cost;
  };

  LogicalStageCost shell;
  shell.id = "source_grid_stride";
  shell.model = "independent_pipelined_loop";
  shell.features.isGridStrideLoopShell = true;
  shell.implementations = {makeSimdCost(10.0), makeCost(1, 10.0),
                           makeCost(2, 20.0), makeCost(4, 40.0)};

  LogicalStageCost gather;
  gather.id = "dynamic_gather";
  gather.model = "indirect_gather_memory";
  gather.features.insideGridStrideLoopBody = true;
  gather.features.hasIndirectMemory = true;
  gather.workload.indirectLoadCounts[{1024, 256}] = 1.0;
  gather.implementations = {makeSimdCost(20.0), makeCost(1, 20.0),
                            makeCost(2, 40.0, true), makeCost(4, 80.0, true)};

  LogicalStageCost dot;
  dot.id = "dynamic_dot";
  dot.model = "tiny_cube_roofline";
  dot.features.insideGridStrideLoopBody = true;
  dot.features.hasDot = true;
  dot.workload.dotFlops = 8192.0;
  dot.workload.dotFlopCounts[8192] = 1.0;
  dot.implementations = {makeSimdCost(40.0), makeCost(1, 40.0),
                         makeCost(2, 80.0, true), makeCost(4, 160.0, true)};
  table.stages = {shell, gather, dot};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  const auto &factor4 = routes->allSimtFactorPlans.at(4);
  // Q=ceil(224/56)=4. The ordinary serialized body is
  // 4*(10 + 80/4 + 160/4)=280 cycles. The structural composition surface
  // contributes one signed startup adjustment plus Q body residuals:
  // -7 + 4*5 = 13.
  EXPECT_DOUBLE_EQ(factor4.totalCycles, 293.0);
  EXPECT_TRUE(factor4.gridStrideIndirectDotCompositionResidualApplied);
  EXPECT_EQ(factor4.gridStrideCompositionIndirectWarpInstructions, 8);
  EXPECT_EQ(factor4.gridStrideCompositionDotFlops, 8192);
  EXPECT_DOUBLE_EQ(factor4.gridStrideCompositionBodyResidualCycles, 5.0);
  EXPECT_DOUBLE_EQ(factor4.gridStrideCompositionStartupResidualCycles, -7.0);
  EXPECT_DOUBLE_EQ(factor4.gridStrideCompositionTotalResidualCycles, 13.0);
  EXPECT_EQ(factor4.gridStrideCompositionEvidence,
            "synthetic structural resource-composition sweep");
  // A dot-bearing all-SIMD route executes on 28 AIC parent slots, so its
  // source-loop body and its serialized structural residual both have Q=8.
  // The residual therefore contributes -7 + 8*5 = 33 cycles once.
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 8.0 * 70.0 + 33.0);
  EXPECT_TRUE(routes->allSimd.gridStrideIndirectDotCompositionResidualApplied);
  EXPECT_EQ(routes->allSimd.gridStrideCompositionIndirectWarpInstructions, 8);
  EXPECT_EQ(routes->allSimd.gridStrideCompositionDotFlops, 8192);
  EXPECT_DOUBLE_EQ(routes->allSimd.gridStrideCompositionBodyResidualCycles,
                   5.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.gridStrideCompositionStartupResidualCycles,
                   -7.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.gridStrideCompositionTotalResidualCycles,
                   33.0);
  EXPECT_EQ(routes->allSimd.gridStrideCompositionEvidence,
            "synthetic SIMD structural resource-composition sweep");
  EXPECT_FALSE(routes->allSimtFactorPlans.at(2)
                   .gridStrideIndirectDotCompositionResidualApplied);

  table.logicalProgramCountHint = 28;
  auto tailOnlyRoutes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!tailOnlyRoutes)
    FAIL() << llvm::toString(tailOnlyRoutes.takeError());
  const auto &tailOnlyFactor4 = tailOnlyRoutes->allSimtFactorPlans.at(4);
  EXPECT_EQ(tailOnlyFactor4.runtimeFullGroupCount, 0);
  EXPECT_EQ(tailOnlyFactor4.runtimeTailProgramCount, 1);
  EXPECT_FALSE(tailOnlyFactor4.gridStrideIndirectDotCompositionResidualApplied);
}

TEST(SimdSimtCostModelTest,
     TwoPassRowwiseUsesOnlyBoundedStructuralCompositionResidual) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 112;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdTwoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot = {
      {1, {{512, {{2, 30.0}}}}}};
  table.simdTwoPassRowwiseCompletionEvidence =
      "synthetic two-pass SIMD completion sweep";
  table.simtTwoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot = {
      {2, {{512, {{2, -20.0}}}}}};
  table.simtTwoPassRowwiseCompletionEvidence =
      "synthetic two-pass SIMT completion sweep";

  auto makeCost = [](StageMode mode, int64_t factor, double cycles) {
    StageImplementationCost cost;
    cost.implementation = {mode, factor};
    cost.totalCycles = cycles;
    return cost;
  };
  auto makeSegmentedLoad = [&](llvm::StringRef id) {
    LogicalStageCost stage;
    stage.id = id.str();
    stage.model = "indirect_gather_memory";
    stage.workload.segmentedIndirectLoadCounts[{2048, 1024, 512}] = 1.0;
    stage.implementations = {makeCost(StageMode::SIMD, 1, 10.0),
                             makeCost(StageMode::SIMT, 2, 20.0)};
    return stage;
  };
  LogicalStageCost reduction;
  reduction.id = "row_reduction";
  reduction.model = "rowwise_reduction";
  reduction.workload.reductionOperations = 1.0;
  reduction.implementations = {makeCost(StageMode::SIMD, 1, 15.0),
                               makeCost(StageMode::SIMT, 2, 25.0)};
  table.stages = {makeSegmentedLoad("first_pass_load"), reduction,
                  makeSegmentedLoad("second_pass_load")};

  StageCostTable baseline = table;
  baseline.simdTwoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot
      .clear();
  baseline.simtTwoPassRowwiseCompletionResidualByFactorExtentAndProgramsPerSlot
      .clear();
  auto baselineRoutes = solveStageRoutes(baseline, ScopeHandoffCost{});
  if (!baselineRoutes)
    FAIL() << llvm::toString(baselineRoutes.takeError());
  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());

  EXPECT_TRUE(routes->allSimd.twoPassRowwiseCompletionApplied);
  EXPECT_EQ(routes->allSimd.twoPassRowwiseContiguousExtent, 512);
  EXPECT_DOUBLE_EQ(routes->allSimd.twoPassRowwiseResidualCycles, 30.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles,
                   baselineRoutes->allSimd.totalCycles + 30.0);
  const auto &factor2 = routes->allSimtFactorPlans.at(2);
  EXPECT_TRUE(factor2.twoPassRowwiseCompletionApplied);
  EXPECT_EQ(factor2.twoPassRowwiseContiguousExtent, 512);
  EXPECT_DOUBLE_EQ(factor2.twoPassRowwiseResidualCycles, -20.0);
  EXPECT_DOUBLE_EQ(factor2.totalCycles,
                   baselineRoutes->allSimtFactorPlans.at(2).totalCycles - 20.0);

  table.stages.pop_back();
  auto incompleteRoutes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!incompleteRoutes)
    FAIL() << llvm::toString(incompleteRoutes.takeError());
  EXPECT_FALSE(incompleteRoutes->allSimd.twoPassRowwiseCompletionApplied);
  EXPECT_FALSE(incompleteRoutes->allSimtFactorPlans.at(2)
                   .twoPassRowwiseCompletionApplied);
}

TEST(SimdSimtCostModelTest,
     SingleResultRowReductionRecurrenceUsesSourceProgramCompletion) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 2;
  table.sourceLogicalProgramCountHint = 16;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdSingleResultRowReductionRecurrenceCompletionEnabled = true;
  table.simdSingleResultRowReductionRecurrenceMinimumExtent = 64;
  table.simdSingleResultRowReductionRecurrenceMaximumExtent = 256;
  table.simdSingleResultRowReductionRecurrenceMinimumIterations = 4;
  table.simdSingleResultRowReductionRecurrenceMaximumIterations = 16;
  table.simdSingleResultRowReductionRecurrenceMinimumPrograms = 8;
  table.simdSingleResultRowReductionRecurrenceMaximumPrograms = 32;
  table.simdSingleResultRowReductionRecurrenceIterationCycles = 20.0;
  table.simdSingleResultRowReductionRecurrenceStartupInterceptCycles = -30.0;
  table.simdSingleResultRowReductionRecurrenceStartupSqrtProgramCycles = 10.0;
  table.simdSingleResultRowReductionRecurrenceEvidence =
      "synthetic single-result recurrence sweep";

  StageImplementationCost simd;
  simd.implementation = {StageMode::SIMD, 1};
  simd.totalCycles = 100.0;
  LogicalStageCost recurrence;
  recurrence.id = "single_result_row_reduction_recurrence";
  recurrence.model = "rowwise_reduction";
  recurrence.iterationCount = 8;
  recurrence.features.hasLoopCarriedDataDependency = true;
  recurrence.features.hasReduction = true;
  recurrence.features.parallelRecurrenceGroupCount = 1;
  recurrence.workload.reductionGeometryCounts[{1, 128, 32}] = 1.0;
  recurrence.implementations = {simd};
  table.stages = {recurrence};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  // The transformed program count is two, but the measured contention domain
  // is the sixteen source programs: max(0, -30 + 10*sqrt(16)) + 8*20 = 170.
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 170.0);
  EXPECT_TRUE(
      routes->allSimd.simdSingleResultRowReductionRecurrenceCompletionApplied);
  EXPECT_EQ(routes->allSimd.simdSingleResultRowReductionRecurrenceExtent, 128);
  EXPECT_EQ(routes->allSimd.simdSingleResultRowReductionRecurrenceIterations,
            8);
  EXPECT_EQ(routes->allSimd.simdSingleResultRowReductionRecurrencePrograms, 16);
  EXPECT_DOUBLE_EQ(
      routes->allSimd.simdSingleResultRowReductionRecurrenceTargetCycles,
      170.0);
  EXPECT_DOUBLE_EQ(
      routes->allSimd.simdSingleResultRowReductionRecurrenceAdjustmentCycles,
      70.0);
  EXPECT_EQ(routes->allSimd.simdSingleResultRowReductionRecurrenceEvidence,
            "synthetic single-result recurrence sweep");
}

TEST(SimdSimtCostModelTest,
     SimtSingleResultRowReductionRecurrenceUsesIndependentCompletion) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 16;
  table.sourceLogicalProgramCountHint = 16;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simtSingleResultRowReductionRecurrenceCompletionEnabled = true;
  table.simtSingleResultRowReductionRecurrenceMinimumExtent = 128;
  table.simtSingleResultRowReductionRecurrenceMaximumExtent = 128;
  table.simtSingleResultRowReductionRecurrenceMinimumIterations = 4;
  table.simtSingleResultRowReductionRecurrenceMaximumIterations = 16;
  table.simtSingleResultRowReductionRecurrenceMinimumPrograms = 16;
  table.simtSingleResultRowReductionRecurrenceMaximumPrograms = 32;
  table.simtSingleResultRowReductionRecurrenceIterationCycles = 20.0;
  table.simtSingleResultRowReductionRecurrenceStartupInterceptCycles = -30.0;
  table.simtSingleResultRowReductionRecurrenceStartupSqrtProgramCycles = 10.0;
  table.simtSingleResultRowReductionRecurrenceEvidence =
      "synthetic whole-SIMT recurrence sweep";

  StageImplementationCost simt;
  simt.implementation = {StageMode::SIMT, 1};
  simt.totalCycles = 100.0;
  LogicalStageCost recurrence;
  recurrence.id = "single_result_row_reduction_recurrence";
  recurrence.model = "rowwise_reduction";
  recurrence.iterationCount = 8;
  recurrence.features.hasLoopCarriedDataDependency = true;
  recurrence.features.hasReduction = true;
  recurrence.features.parallelRecurrenceGroupCount = 1;
  recurrence.workload.reductionGeometryCounts[{1, 128, 32}] = 1.0;
  recurrence.implementations = {simt};
  table.stages = {recurrence};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 170.0);
  EXPECT_TRUE(
      routes->allSimt.simtSingleResultRowReductionRecurrenceCompletionApplied);
  EXPECT_EQ(routes->allSimt.simtSingleResultRowReductionRecurrenceExtent, 128);
  EXPECT_EQ(routes->allSimt.simtSingleResultRowReductionRecurrenceIterations,
            8);
  EXPECT_EQ(routes->allSimt.simtSingleResultRowReductionRecurrencePrograms, 16);
  EXPECT_DOUBLE_EQ(
      routes->allSimt.simtSingleResultRowReductionRecurrenceTargetCycles,
      170.0);
  EXPECT_DOUBLE_EQ(
      routes->allSimt.simtSingleResultRowReductionRecurrenceAdjustmentCycles,
      70.0);
  EXPECT_EQ(routes->allSimt.simtSingleResultRowReductionRecurrenceEvidence,
            "synthetic whole-SIMT recurrence sweep");
}

TEST(SimdSimtCostModelTest,
     SingleResultRowReductionRecurrenceRejectsDifferentTopology) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 16;
  table.sourceLogicalProgramCountHint = 16;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdSingleResultRowReductionRecurrenceCompletionEnabled = true;
  table.simdSingleResultRowReductionRecurrenceMinimumExtent = 64;
  table.simdSingleResultRowReductionRecurrenceMaximumExtent = 256;
  table.simdSingleResultRowReductionRecurrenceMinimumIterations = 4;
  table.simdSingleResultRowReductionRecurrenceMaximumIterations = 16;
  table.simdSingleResultRowReductionRecurrenceMinimumPrograms = 8;
  table.simdSingleResultRowReductionRecurrenceMaximumPrograms = 32;
  table.simdSingleResultRowReductionRecurrenceIterationCycles = 20.0;
  table.simdSingleResultRowReductionRecurrenceStartupInterceptCycles = -30.0;
  table.simdSingleResultRowReductionRecurrenceStartupSqrtProgramCycles = 10.0;

  StageImplementationCost simd;
  simd.implementation = {StageMode::SIMD, 1};
  simd.totalCycles = 100.0;
  LogicalStageCost recurrence;
  recurrence.id = "matrix_result_reduction_recurrence";
  recurrence.model = "rowwise_reduction";
  recurrence.iterationCount = 8;
  recurrence.features.hasLoopCarriedDataDependency = true;
  recurrence.features.hasReduction = true;
  recurrence.features.parallelRecurrenceGroupCount = 1;
  recurrence.workload.reductionGeometryCounts[{2, 128, 32}] = 1.0;
  recurrence.implementations = {simd};
  table.stages = {recurrence};

  auto matrixRoutes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!matrixRoutes)
    FAIL() << llvm::toString(matrixRoutes.takeError());
  EXPECT_DOUBLE_EQ(matrixRoutes->allSimd.totalCycles, 100.0);
  EXPECT_FALSE(matrixRoutes->allSimd
                   .simdSingleResultRowReductionRecurrenceCompletionApplied);

  table.stages.front().workload.reductionGeometryCounts.clear();
  table.stages.front().workload.reductionGeometryCounts[{1, 128, 32}] = 1.0;
  table.stages.front().features.hasPrefixScan = true;
  auto scanRoutes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!scanRoutes)
    FAIL() << llvm::toString(scanRoutes.takeError());
  EXPECT_DOUBLE_EQ(scanRoutes->allSimd.totalCycles, 100.0);
  EXPECT_FALSE(scanRoutes->allSimd
                   .simdSingleResultRowReductionRecurrenceCompletionApplied);
}

TEST(SimdSimtCostModelTest,
     MixedRouteUsesSimdSingleResultRowReductionCompletion) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  // Q=ceil(P/C)=2 lets the F2 candidate exercise a real full group. With
  // P<=C, F2 lowers entirely through its F1 tail and ties the F1 route.
  table.logicalProgramCountHint = 112;
  table.sourceLogicalProgramCountHint = 16;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdSingleResultRowReductionRecurrenceCompletionEnabled = true;
  table.simdSingleResultRowReductionRecurrenceMinimumExtent = 64;
  table.simdSingleResultRowReductionRecurrenceMaximumExtent = 256;
  table.simdSingleResultRowReductionRecurrenceMinimumIterations = 4;
  table.simdSingleResultRowReductionRecurrenceMaximumIterations = 16;
  table.simdSingleResultRowReductionRecurrenceMinimumPrograms = 8;
  table.simdSingleResultRowReductionRecurrenceMaximumPrograms = 32;
  table.simdSingleResultRowReductionRecurrenceIterationCycles = 20.0;
  table.simdSingleResultRowReductionRecurrenceStartupInterceptCycles = -30.0;
  table.simdSingleResultRowReductionRecurrenceStartupSqrtProgramCycles = 10.0;
  table.simdSingleResultRowReductionRecurrenceEvidence =
      "synthetic single-result recurrence sweep";

  auto makeCost = [](StageMode mode, double cycles, bool localScope = false,
                     int64_t factor = 1) {
    StageImplementationCost cost;
    cost.implementation = {mode, factor, localScope};
    cost.totalCycles = cycles;
    return cost;
  };

  LogicalStageCost recurrence;
  recurrence.id = "single_result_row_reduction_recurrence";
  recurrence.model = "rowwise_reduction";
  recurrence.iterationCount = 8;
  recurrence.features.hasLoopCarriedDataDependency = true;
  recurrence.features.hasReduction = true;
  recurrence.features.parallelRecurrenceGroupCount = 1;
  recurrence.workload.reductionGeometryCounts[{1, 128, 32}] = 1.0;
  recurrence.implementations = {makeCost(StageMode::SIMD, 100.0),
                                makeCost(StageMode::SIMT, 300.0)};

  LogicalStageCost gather;
  gather.id = "local_indirect_gather";
  gather.localSimtMaterializable = true;
  gather.localSimtFactors = {1, 2};
  gather.legalSimtFactors = {1, 2};
  gather.implementations = {makeCost(StageMode::SIMD, 100.0),
                            makeCost(StageMode::SIMT, 150.0),
                            makeCost(StageMode::SIMT, 20.0, true),
                            makeCost(StageMode::SIMT, 10.0, true, 2)};
  table.stages = {recurrence, gather};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->mixed.legal);
  EXPECT_EQ(routes->mixed.routeSuperblockFactor, 2);
  EXPECT_EQ(routes->mixed.implementations[0].mode, StageMode::SIMD);
  EXPECT_EQ(routes->mixed.implementations[1].mode, StageMode::SIMT);
  EXPECT_TRUE(routes->mixed.implementations[1].localScope);
  EXPECT_DOUBLE_EQ(routes->mixed.totalCycles, 180.0);
  EXPECT_TRUE(
      routes->mixed.simdSingleResultRowReductionRecurrenceCompletionApplied);
  EXPECT_DOUBLE_EQ(
      routes->mixed.simdSingleResultRowReductionRecurrenceTargetCycles, 170.0);
  EXPECT_DOUBLE_EQ(
      routes->mixed.simdSingleResultRowReductionRecurrenceAdjustmentCycles,
      -30.0);
}

TEST(SimdSimtCostModelTest,
     MultiResultPrefixScanUsesSourceProgramsAndMaskedActiveAxis) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 2;
  table.sourceLogicalProgramCountHint = 10;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdMultiResultPrefixScanCompletionEnabled = true;
  table.simdMultiResultPrefixScanMinimumPrograms = 2;
  table.simdMultiResultPrefixScanMaximumPrograms = 20;
  table.simdMultiResultPrefixScanMinimumPaddedAxis = 8;
  table.simdMultiResultPrefixScanMaximumPaddedAxis = 16;
  table.simdMultiResultPrefixScanRequiredResultCount = 2;
  table.simdMultiResultPrefixScanRequiredIndependentSequences = 8;
  table.simdMultiResultPrefixScanMinimumActiveRatio = 0.625;
  table.simdMultiResultPrefixScanMaximumActiveRatio = 1.0;
  table.simdMultiResultPrefixScanElapsedInterceptCycles = 1000.0;
  table.simdMultiResultPrefixScanElapsedSqrtProgramCycles = 0.0;
  table.simdMultiResultPrefixScanElapsedInactiveRatioCycles = 100.0;
  table.simdMultiResultPrefixScanElapsedPaddedAxisCycles = 10.0;
  table.simdMultiResultPrefixScanFixedBoundaryCycles = 1000.0;
  table.simdMultiResultPrefixScanEvidence = "synthetic multi-result scan sweep";

  auto simdCost = [](double cycles) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMD, 1};
    cost.totalCycles = cycles;
    return cost;
  };
  auto memoryStage = [&](llvm::StringRef id, bool store) {
    LogicalStageCost stage;
    stage.id = id.str();
    stage.model = store ? "continuous_tile_store" : "continuous_tile_memory";
    StageMemoryAccess access;
    access.shape = {8, 16};
    access.strides = {10, 1};
    access.elementBits = 32;
    access.masked = true;
    access.store = store;
    stage.workload.memoryAccesses = {access};
    stage.implementations = {simdCost(10.0)};
    return stage;
  };
  LogicalStageCost scan;
  scan.id = "two_result_prefix_scan";
  scan.model = "prefix_scan";
  scan.features.hasPrefixScan = true;
  scan.features.hasReduction = true;
  scan.workload.prefixScanCounts[{16, 8, 1, 2, false, "add"}] = 1.0;
  scan.implementations = {simdCost(20.0)};
  LogicalStageCost shell;
  shell.id = "auto_blockify_dispatch";
  shell.model = "auto_blockify_dispatch";
  shell.implementations = {simdCost(5.0)};
  table.stages = {shell,
                  memoryStage("load_value", false),
                  memoryStage("load_index", false),
                  scan,
                  memoryStage("store_value", true),
                  memoryStage("store_index", true)};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  // active_ratio=10/16. Absolute target is
  // 1000 + 100*(1-10/16) + 10*16 = 1197.5. The generic fixed boundary is
  // removed from the route score and will be added once by elapsed prediction.
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 197.5);
  EXPECT_TRUE(routes->allSimd.simdMultiResultPrefixScanCompletionApplied);
  EXPECT_EQ(routes->allSimd.simdMultiResultPrefixScanSourcePrograms, 10);
  EXPECT_EQ(routes->allSimd.simdMultiResultPrefixScanPaddedAxis, 16);
  EXPECT_EQ(routes->allSimd.simdMultiResultPrefixScanActiveAxis, 10);
  EXPECT_EQ(routes->allSimd.simdMultiResultPrefixScanResultCount, 2);
  EXPECT_EQ(routes->allSimd.simdMultiResultPrefixScanIndependentSequences, 8);
  EXPECT_DOUBLE_EQ(routes->allSimd.simdMultiResultPrefixScanActiveRatio, 0.625);
  EXPECT_DOUBLE_EQ(
      routes->allSimd.simdMultiResultPrefixScanAbsoluteTargetCycles, 1197.5);
  EXPECT_DOUBLE_EQ(routes->allSimd.simdMultiResultPrefixScanRouteTargetCycles,
                   197.5);
  EXPECT_DOUBLE_EQ(routes->allSimd.simdMultiResultPrefixScanAdjustmentCycles,
                   137.5);
}

TEST(SimdSimtCostModelTest,
     SimtPairwiseSelectionNetworkUsesStructuralRouteCompletion) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 1;
  table.sourceLogicalProgramCountHint = 1;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simtPairwiseSelectionNetworkCompletionEnabled = true;
  table.simtPairwiseSelectionNetworkMinimumInputElements = 256;
  table.simtPairwiseSelectionNetworkMaximumInputElements = 1024;
  table.simtPairwiseSelectionNetworkMinimumOutputColumns = 2;
  table.simtPairwiseSelectionNetworkMaximumOutputColumns = 8;
  table.simtPairwiseSelectionNetworkMinimumReductionStages = 2;
  table.simtPairwiseSelectionNetworkInterceptCycles = 10.0;
  table.simtPairwiseSelectionNetworkInputElementCycles = 0.5;
  table.simtPairwiseSelectionNetworkLog2OutputColumnCycles = 100.0;
  table.simtPairwiseSelectionNetworkEvidence =
      "synthetic pairwise selection network sweep";

  auto simtCost = [](double cycles) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, 1};
    cost.totalCycles = cycles;
    return cost;
  };
  auto memoryStage = [&](llvm::StringRef id, int64_t rows, int64_t columns,
                         bool store) {
    LogicalStageCost stage;
    stage.id = id.str();
    stage.model = store ? "continuous_tile_store" : "continuous_tile_memory";
    stage.features.hasContiguousMemory = true;
    StageMemoryAccess access;
    access.shape = {rows, columns};
    access.strides = {columns, 1};
    access.elementBits = 32;
    access.store = store;
    stage.workload.memoryAccesses = {access};
    stage.implementations = {simtCost(store ? 100.0 : 200.0)};
    return stage;
  };
  auto reductionStage = [&](llvm::StringRef id, int64_t rows) {
    LogicalStageCost stage;
    stage.id = id.str();
    stage.model = "rowwise_reduction";
    stage.features.hasReduction = true;
    stage.workload.reductionGeometryCounts[{rows, 2, 32}] = 1.0;
    stage.implementations = {simtCost(100.0)};
    return stage;
  };
  table.stages = {
      memoryStage("input", 8, 64, false), reductionStage("pairwise_0", 256),
      reductionStage("pairwise_1", 128), memoryStage("output", 8, 4, true)};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimt.legal);
  EXPECT_TRUE(routes->allSimt.simtPairwiseSelectionNetworkCompletionApplied);
  EXPECT_EQ(routes->allSimt.simtPairwiseSelectionNetworkInputRows, 8);
  EXPECT_EQ(routes->allSimt.simtPairwiseSelectionNetworkInputColumns, 64);
  EXPECT_EQ(routes->allSimt.simtPairwiseSelectionNetworkOutputColumns, 4);
  EXPECT_EQ(routes->allSimt.simtPairwiseSelectionNetworkReductionStages, 2);
  EXPECT_DOUBLE_EQ(routes->allSimt.simtPairwiseSelectionNetworkTargetCycles,
                   466.0);
  EXPECT_DOUBLE_EQ(routes->allSimt.simtPairwiseSelectionNetworkAdjustmentCycles,
                   -34.0);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 466.0);

  table.stages.back() = memoryStage("output", 8, 3, true);
  auto nonPowerOfTwo = solveStageRoutes(table, ScopeHandoffCost{});
  if (!nonPowerOfTwo)
    FAIL() << llvm::toString(nonPowerOfTwo.takeError());
  EXPECT_FALSE(
      nonPowerOfTwo->allSimt.simtPairwiseSelectionNetworkCompletionApplied);
  EXPECT_DOUBLE_EQ(nonPowerOfTwo->allSimt.totalCycles, 500.0);
}

TEST(SimdSimtCostModelTest,
     SimtMultiSequenceMinmaxScanUsesBoundedStructuralCompletion) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 1;
  table.sourceLogicalProgramCountHint = 1;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simtMultiSequenceMinmaxScanCompletionEnabled = true;
  table.simtMultiSequenceMinmaxScanMinimumIndependentSequences = 2;
  table.simtMultiSequenceMinmaxScanMaximumIndependentSequences = 32;
  table.simtMultiSequenceMinmaxScanMinimumExtent = 32;
  table.simtMultiSequenceMinmaxScanMaximumExtent = 64;
  table.simtMultiSequenceMinmaxScanBaseReductionCycles = 39.0;
  table.simtMultiSequenceMinmaxScanPerIndependentSequenceReductionCycles = 13.0;
  table.simtMultiSequenceMinmaxScanMaximumReductionCycles = 285.0;
  table.simtMultiSequenceMinmaxScanEvidence =
      "synthetic multi-sequence minmax scan sweep";

  StageImplementationCost simt;
  simt.implementation = {StageMode::SIMT, 1};
  simt.totalCycles = 500.0;
  LogicalStageCost scan;
  scan.id = "minmax_scan";
  scan.model = "prefix_scan";
  scan.features.hasPrefixScan = true;
  scan.features.hasReduction = true;
  scan.workload.prefixScanCounts[{32, 8, 1, 1, false, "minmax"}] = 1.0;
  scan.implementations = {simt};
  table.stages = {scan};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimt.legal);
  EXPECT_TRUE(routes->allSimt.simtMultiSequenceMinmaxScanCompletionApplied);
  EXPECT_EQ(routes->allSimt.simtMultiSequenceMinmaxScanIndependentSequences, 8);
  EXPECT_DOUBLE_EQ(routes->allSimt.simtMultiSequenceMinmaxScanReductionCycles,
                   143.0);
  EXPECT_DOUBLE_EQ(routes->allSimt.simtMultiSequenceMinmaxScanAdjustmentCycles,
                   -143.0);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 357.0);

  table.stages.front().workload.prefixScanCounts.clear();
  table.stages.front()
      .workload.prefixScanCounts[{32, 1, 1, 1, false, "minmax"}] = 1.0;
  auto singleSequence = solveStageRoutes(table, ScopeHandoffCost{});
  if (!singleSequence)
    FAIL() << llvm::toString(singleSequence.takeError());
  EXPECT_FALSE(
      singleSequence->allSimt.simtMultiSequenceMinmaxScanCompletionApplied);
  EXPECT_DOUBLE_EQ(singleSequence->allSimt.totalCycles, 500.0);

  table.stages.front().workload.prefixScanCounts.clear();
  table.stages.front().workload.prefixScanCounts[{32, 8, 1, 1, false, "mul"}] =
      1.0;
  auto multiplicative = solveStageRoutes(table, ScopeHandoffCost{});
  if (!multiplicative)
    FAIL() << llvm::toString(multiplicative.takeError());
  EXPECT_FALSE(
      multiplicative->allSimt.simtMultiSequenceMinmaxScanCompletionApplied);
  EXPECT_DOUBLE_EQ(multiplicative->allSimt.totalCycles, 500.0);

  table.stages.front().workload.prefixScanCounts.clear();
  table.stages.front()
      .workload.prefixScanCounts[{5, 21, 1, 1, false, "minmax"}] = 1.0;
  auto shortExtent = solveStageRoutes(table, ScopeHandoffCost{});
  if (!shortExtent)
    FAIL() << llvm::toString(shortExtent.takeError());
  EXPECT_FALSE(
      shortExtent->allSimt.simtMultiSequenceMinmaxScanCompletionApplied);
  EXPECT_DOUBLE_EQ(shortExtent->allSimt.totalCycles, 500.0);
}

TEST(SimdSimtCostModelTest,
     LargeSequenceMinmaxScanUsesCandidateSpecificCompletion) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 1;
  table.sourceLogicalProgramCountHint = 1;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdLargeSequenceMinmaxScanCompletionEnabled = true;
  table.simdLargeSequenceMinmaxScanMinimumIndependentSequences = 64;
  table.simdLargeSequenceMinmaxScanMaximumIndependentSequences = 512;
  table.simdLargeSequenceMinmaxScanMinimumExtent = 16;
  table.simdLargeSequenceMinmaxScanMaximumExtent = 16;
  table.simdLargeSequenceMinmaxScanPerIndependentSequenceCycles = 11.93;
  table.simdLargeSequenceMinmaxScanEvidence = "synthetic SIMD large scan";
  table.simtMultiSequenceMinmaxScanCompletionEnabled = true;
  table.simtMultiSequenceMinmaxScanMinimumIndependentSequences = 2;
  table.simtMultiSequenceMinmaxScanMaximumIndependentSequences = 32;
  table.simtMultiSequenceMinmaxScanMinimumExtent = 32;
  table.simtMultiSequenceMinmaxScanMaximumExtent = 64;
  table.simtMultiSequenceMinmaxScanBaseReductionCycles = 39.0;
  table.simtMultiSequenceMinmaxScanPerIndependentSequenceReductionCycles = 13.0;
  table.simtMultiSequenceMinmaxScanMaximumReductionCycles = 285.0;
  table.simtMultiSequenceMinmaxScanLargeMinimumIndependentSequences = 64;
  table.simtMultiSequenceMinmaxScanLargeMaximumIndependentSequences = 512;
  table.simtMultiSequenceMinmaxScanLargeMinimumExtent = 16;
  table.simtMultiSequenceMinmaxScanLargeMaximumExtent = 32;
  table.simtMultiSequenceMinmaxScanLargeBaseReductionCycles = 211.0;
  table.simtMultiSequenceMinmaxScanLargePerIndependentSequenceReductionCycles =
      2.75;
  table.simtMultiSequenceMinmaxScanLargeMaximumReductionCycles = 1611.0;
  table.simtMultiSequenceMinmaxScanEvidence = "synthetic SIMT large scan";

  StageImplementationCost simd;
  simd.implementation = {StageMode::SIMD, 1};
  simd.totalCycles = 1000.0;
  StageImplementationCost simt;
  simt.implementation = {StageMode::SIMT, 1};
  simt.totalCycles = 3000.0;
  LogicalStageCost scan;
  scan.id = "large_minmax_scan";
  scan.model = "prefix_scan";
  scan.features.hasPrefixScan = true;
  scan.features.hasReduction = true;
  scan.workload.prefixScanCounts[{16, 512, 1, 1, false, "minmax"}] = 1.0;
  scan.implementations = {simd, simt};
  table.stages = {scan};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_TRUE(routes->allSimd.simdLargeSequenceMinmaxScanCompletionApplied);
  EXPECT_EQ(routes->allSimd.simdLargeSequenceMinmaxScanIndependentSequences,
            512);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 1000.0 + 512.0 * 11.93);
  EXPECT_TRUE(routes->allSimt.simtMultiSequenceMinmaxScanCompletionApplied);
  EXPECT_EQ(routes->allSimt.simtMultiSequenceMinmaxScanExtent, 16);
  EXPECT_DOUBLE_EQ(routes->allSimt.simtMultiSequenceMinmaxScanReductionCycles,
                   1611.0);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 1389.0);
}

TEST(SimdSimtCostModelTest, MathCosKeepsItsSemanticOperationClass) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%source: tensor<1024xf32>) {
        %result = "math.cos"(%source) : (tensor<1024xf32>) -> tensor<1024xf32>
        "test.consume"(%result) : (tensor<1024xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *cosine = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "math.cos")
      cosine = operation;
  });
  ASSERT_NE(cosine, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto stage = logicalStage("cosine", StageCostModelKind::VectorIssue);
  stage.operations.push_back(cosine);
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));

  const StageWorkload &workload = partition.stages.front().workload;
  ASSERT_EQ(workload.operationCounts.size(), 1u);
  ASSERT_TRUE(workload.operationCounts.contains("f32.cos"));
  EXPECT_DOUBLE_EQ(workload.operationCounts.at("f32.cos").at(1024), 1.0);
  ASSERT_EQ(workload.tensorOperationWorkloads.size(), 1u);
  EXPECT_EQ(workload.tensorOperationWorkloads.front().operation, "f32.cos");
}

TEST(SimdSimtCostModelTest,
     WorkloadSeparatesComputeFallbackFromStructuralTensorOps) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%fp: tensor<128xf32>, %index: tensor<128xi32>,
                        %scalar: tensor<1xf32>, %scalar_fp: f32) {
        %floor = "math.floor"(%fp) : (tensor<128xf32>) -> tensor<128xf32>
        %scalar_floor = "math.floor"(%scalar_fp) : (f32) -> f32
        %exp2 = "math.exp2"(%fp) : (tensor<128xf32>) -> tensor<128xf32>
        %add = "arith.addi"(%index, %index) :
            (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
        %sub = "arith.subi"(%index, %index) :
            (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
        %mul = "arith.muli"(%index, %index) :
            (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
        %div = "arith.divsi"(%index, %index) :
            (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
        %rem = "arith.remsi"(%index, %index) :
            (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
        %min = "arith.minsi"(%index, %index) :
            (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
        %max = "arith.maxsi"(%index, %index) :
            (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
        %and = "arith.andi"(%index, %index) :
            (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
        %sin = "math.sin"(%fp) : (tensor<128xf32>) -> tensor<128xf32>
        %ceil = "math.ceil"(%fp) : (tensor<128xf32>) -> tensor<128xf32>
        %rsqrt = "math.rsqrt"(%fp) : (tensor<128xf32>) -> tensor<128xf32>
        %precise = "tt.precise_divf"(%fp, %fp) :
            (tensor<128xf32>, tensor<128xf32>) -> tensor<128xf32>
        %mulhi = "tt.mulhiui"(%index, %index) :
            (tensor<128xi32>, tensor<128xi32>) -> tensor<128xi32>
        %extern = "tt.extern_elementwise"(%fp) :
            (tensor<128xf32>) -> tensor<128xf32>
        %relu = "tt.extern_elementwise"(%fp) {symbol = "__hmf_reluf"} :
            (tensor<128xf32>) -> tensor<128xf32>
        %tanh = "tt.extern_elementwise"(%fp) {symbol = "__hmf_tanhf"} :
            (tensor<128xf32>) -> tensor<128xf32>
        %future = "tt.future_numeric"(%fp) :
            (tensor<128xf32>) -> tensor<128xf32>
        %bitcast = "tt.bitcast"(%fp) :
            (tensor<128xf32>) -> tensor<128xi32>
        %broadcast = "tt.broadcast"(%scalar) :
            (tensor<1xf32>) -> tensor<128xf32>
        "tt.elementwise_inline_asm"(%fp) : (tensor<128xf32>) -> ()
        "test.consume"(%floor, %scalar_floor, %exp2, %add, %sub, %mul, %div,
                       %rem, %min, %max, %and, %sin, %ceil, %rsqrt, %precise,
                       %mulhi, %extern, %relu, %tanh, %future, %bitcast,
                       %broadcast) :
            (tensor<128xf32>, f32, tensor<128xf32>, tensor<128xi32>,
             tensor<128xi32>, tensor<128xi32>, tensor<128xi32>,
             tensor<128xi32>, tensor<128xi32>, tensor<128xi32>,
             tensor<128xi32>, tensor<128xf32>, tensor<128xf32>,
             tensor<128xf32>, tensor<128xf32>, tensor<128xi32>,
             tensor<128xf32>, tensor<128xf32>, tensor<128xf32>,
             tensor<128xf32>, tensor<128xi32>, tensor<128xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto stage =
      logicalStage("compute_coverage", StageCostModelKind::VectorIssue);
  module->walk([&](mlir::Operation *operation) {
    const llvm::StringRef name = operation->getName().getStringRef();
    if (name == "math.floor" || name == "math.exp2" || name == "math.sin" ||
        name == "math.ceil" || name == "math.rsqrt" || name == "arith.addi" ||
        name == "arith.subi" || name == "arith.muli" || name == "arith.divsi" ||
        name == "arith.remsi" || name == "arith.minsi" ||
        name == "arith.maxsi" || name == "arith.andi" ||
        name == "tt.precise_divf" || name == "tt.mulhiui" ||
        name == "tt.extern_elementwise" || name == "tt.future_numeric" ||
        name == "tt.bitcast" || name == "tt.broadcast" ||
        name == "tt.elementwise_inline_asm")
      stage.operations.push_back(operation);
  });
  ASSERT_EQ(stage.operations.size(), 23u);
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));

  const StageWorkload &workload = partition.stages.front().workload;
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.math.floor"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.arith.addi"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.math.exp2"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.arith.divsi"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.arith.subi"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.arith.muli"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.arith.remsi"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.arith.minsi"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.arith.maxsi"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.arith.andi"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.math.sin"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.math.ceil"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.math.rsqrt"));
  EXPECT_TRUE(workload.operationCounts.contains("f32.div"));
  EXPECT_TRUE(workload.operationCounts.contains("fp32.saturated.tt.mulhiui"));
  EXPECT_TRUE(workload.operationCounts.contains(
      "fp32.saturated.tt.extern_elementwise"));
  EXPECT_TRUE(workload.operationCounts.contains("f32.relu"));
  EXPECT_TRUE(workload.operationCounts.contains(
      "fp32.saturated.tt.extern_elementwise.__hmf_tanhf"));
  EXPECT_TRUE(
      workload.operationCounts.contains("fp32.saturated.tt.future_numeric"));
  EXPECT_TRUE(workload.operationCounts.contains(
      "fp32.saturated.tt.elementwise_inline_asm"));
  EXPECT_TRUE(workload.operationCounts.contains("generic.issue"));
  EXPECT_DOUBLE_EQ(
      workload.operationCounts.at("fp32.saturated.math.floor").at(128), 1.0);
  EXPECT_DOUBLE_EQ(
      workload.operationCounts.at("fp32.saturated.math.floor").at(1), 1.0);
  EXPECT_DOUBLE_EQ(
      workload.operationCounts.at("fp32.saturated.arith.addi").at(128), 1.0);
  EXPECT_DOUBLE_EQ(workload.operationCounts.at("generic.issue").at(128), 2.0);
}

TEST(SimdSimtCostModelTest,
     MapElementwisePromotesScalarRegionComputeToFullTensorWidth) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%input: tensor<128xf32>) -> tensor<128xf32> {
        %mapped = "tt.map_elementwise"(%input) ({
        ^bb0(%arg0: f32):
          %floor = "math.floor"(%arg0) : (f32) -> f32
          "tt.map_elementwise.return"(%floor) : (f32) -> ()
        }) {pack = 1 : i32} : (tensor<128xf32>) -> tensor<128xf32>
        return %mapped : tensor<128xf32>
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto stage = logicalStage("mapped_compute", StageCostModelKind::VectorIssue);
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "tt.map_elementwise")
      stage.operations.push_back(operation);
  });
  ASSERT_EQ(stage.operations.size(), 1u);
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));

  const StageWorkload &workload = partition.stages.front().workload;
  const auto floor = workload.operationCounts.find("fp32.saturated.math.floor");
  ASSERT_NE(floor, workload.operationCounts.end());
  EXPECT_DOUBLE_EQ(floor->second.at(128), 1.0);
  EXPECT_EQ(floor->second.count(1), 0u);
  const auto tensor =
      llvm::find_if(workload.tensorOperationWorkloads,
                    [](const mlir::ascend::TensorOperationWorkload &item) {
                      return item.operation == "fp32.saturated.math.floor";
                    });
  ASSERT_NE(tensor, workload.tensorOperationWorkloads.end());
  EXPECT_DOUBLE_EQ(tensor->logicalElements, 128.0);
  EXPECT_EQ(tensor->elementBitWidth, 32);
  EXPECT_EQ(workload.operationCounts.count("generic.issue"), 0u);
}

TEST(SimdSimtCostModelTest,
     SimdStraightLineAddReductionUsesCombineClassCompletion) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 1;
  table.sourceLogicalProgramCountHint = 1;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdStraightLineAddReductionCompletionEnabled = true;
  table.simdStraightLineAddReductionMinimumRows = 16;
  table.simdStraightLineAddReductionMaximumRows = 64;
  table.simdStraightLineAddReductionMinimumExtent = 8;
  table.simdStraightLineAddReductionMaximumExtent = 32;
  table.simdStraightLineAddReductionSharedCompletionCycles = 100.0;
  table.simdStraightLineAddReductionEvidence =
      "synthetic additive reduction sweep";

  auto simdCost = [](double cycles) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMD, 1};
    cost.totalCycles = cycles;
    return cost;
  };
  LogicalStageCost load;
  load.id = "load";
  load.model = "continuous_tile_memory";
  load.features.hasContiguousMemory = true;
  StageMemoryAccess loadAccess;
  loadAccess.shape = {64, 16};
  loadAccess.strides = {3, 1};
  loadAccess.elementBits = 32;
  load.workload.memoryAccesses = {loadAccess};
  load.implementations = {simdCost(200.0)};

  LogicalStageCost reduction;
  reduction.id = "add_reduction";
  reduction.model = "rowwise_reduction";
  reduction.features.hasReduction = true;
  reduction.workload.reductionGeometryCounts[{64, 16, 32}] = 1.0;
  reduction.workload.reductionCombineClassCounts["add"] = 1.0;
  reduction.implementations = {simdCost(200.0)};

  LogicalStageCost store;
  store.id = "store";
  store.model = "continuous_tile_store";
  store.features.hasContiguousMemory = true;
  StageMemoryAccess storeAccess;
  storeAccess.shape = {64};
  storeAccess.strides = {1};
  storeAccess.elementBits = 32;
  storeAccess.store = true;
  store.workload.memoryAccesses = {storeAccess};
  store.implementations = {simdCost(100.0)};
  table.stages = {load, reduction, store};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.legal);
  EXPECT_TRUE(routes->allSimd.simdStraightLineAddReductionCompletionApplied);
  EXPECT_EQ(routes->allSimd.simdStraightLineAddReductionRows, 64);
  EXPECT_EQ(routes->allSimd.simdStraightLineAddReductionExtent, 16);
  EXPECT_DOUBLE_EQ(routes->allSimd.simdStraightLineAddReductionAdjustmentCycles,
                   -100.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 400.0);

  table.stages[1].workload.reductionCombineClassCounts.clear();
  table.stages[1].workload.reductionCombineClassCounts["minmax"] = 1.0;
  auto minmax = solveStageRoutes(table, ScopeHandoffCost{});
  if (!minmax)
    FAIL() << llvm::toString(minmax.takeError());
  EXPECT_FALSE(minmax->allSimd.simdStraightLineAddReductionCompletionApplied);
  EXPECT_DOUBLE_EQ(minmax->allSimd.totalCycles, 500.0);
}

TEST(SimdSimtCostModelTest,
     SimdTwoLoopSingleReductionChargesSerializedWaveCompletion) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 128;
  table.sourceLogicalProgramCountHint = 128;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdTwoLoopSingleReductionCompletionEnabled = true;
  table.simdTwoLoopSingleReductionMinimumSourcePrograms = 64;
  table.simdTwoLoopSingleReductionMaximumSourcePrograms = 256;
  table.simdTwoLoopSingleReductionMinimumRuntimeIterations = 2;
  table.simdTwoLoopSingleReductionMaximumRuntimeIterations = 5;
  table.simdTwoLoopSingleReductionRequiredExtent = 8192;
  table.simdTwoLoopSingleReductionFirstIterationCycles = 180.0;
  table.simdTwoLoopSingleReductionAdditionalIterationCycles = 430.0;
  table.simdTwoLoopSingleReductionEvidence =
      "synthetic two-loop reduction wave sweep";

  auto makeSimdStage = [](llvm::StringRef id, llvm::StringRef model,
                          double cycles) {
    LogicalStageCost stage;
    stage.id = id.str();
    stage.model = model.str();
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMD, 1};
    cost.totalCycles = cycles;
    stage.implementations = {cost};
    return stage;
  };
  auto first = makeSimdStage("first_loop", "independent_pipelined_loop", 100.0);
  first.features.hasLoop = true;
  auto reduction = makeSimdStage("reduction", "rowwise_reduction", 100.0);
  reduction.features.hasReduction = true;
  reduction.workload.reductionGeometryCounts[{1, 8192, 32}] = 1.0;
  reduction.workload.reductionCombineClassCounts["add"] = 1.0;
  auto second =
      makeSimdStage("second_loop", "independent_pipelined_loop", 100.0);
  second.features.hasLoop = true;
  table.stages = {first, reduction, second};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.legal);
  EXPECT_EQ(routes->allSimd.runtimeLoopIterationCount, 3);
  EXPECT_TRUE(routes->allSimd.simdTwoLoopSingleReductionCompletionApplied);
  EXPECT_EQ(routes->allSimd.simdTwoLoopSingleReductionRuntimeIterations, 3);
  EXPECT_DOUBLE_EQ(routes->allSimd.simdTwoLoopSingleReductionAdjustmentCycles,
                   1040.0);

  table.stages.pop_back();
  auto oneLoop = solveStageRoutes(table, ScopeHandoffCost{});
  if (!oneLoop)
    FAIL() << llvm::toString(oneLoop.takeError());
  EXPECT_FALSE(oneLoop->allSimd.simdTwoLoopSingleReductionCompletionApplied);
}

TEST(SimdSimtCostModelTest,
     MultiResultPrefixScanRejectsInconsistentMaskedAxis) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 2;
  table.sourceLogicalProgramCountHint = 10;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdMultiResultPrefixScanCompletionEnabled = true;
  table.simdMultiResultPrefixScanMinimumPrograms = 2;
  table.simdMultiResultPrefixScanMaximumPrograms = 20;
  table.simdMultiResultPrefixScanMinimumPaddedAxis = 8;
  table.simdMultiResultPrefixScanMaximumPaddedAxis = 16;
  table.simdMultiResultPrefixScanRequiredResultCount = 2;
  table.simdMultiResultPrefixScanRequiredIndependentSequences = 8;
  table.simdMultiResultPrefixScanMinimumActiveRatio = 0.625;
  table.simdMultiResultPrefixScanMaximumActiveRatio = 1.0;

  StageImplementationCost simd;
  simd.implementation = {StageMode::SIMD, 1};
  simd.totalCycles = 10.0;
  LogicalStageCost scan;
  scan.id = "two_result_prefix_scan";
  scan.model = "prefix_scan";
  scan.features.hasPrefixScan = true;
  scan.features.hasReduction = true;
  scan.workload.prefixScanCounts[{16, 8, 1, 2, false, "add"}] = 1.0;
  scan.implementations = {simd};
  table.stages = {scan};
  for (int index = 0; index < 4; ++index) {
    LogicalStageCost memory;
    memory.id = "memory_" + std::to_string(index);
    memory.model =
        index < 2 ? "continuous_tile_memory" : "continuous_tile_store";
    StageMemoryAccess access;
    access.shape = {8, 16};
    access.strides = {index == 3 ? 9 : 10, 1};
    access.elementBits = 32;
    access.masked = true;
    access.store = index >= 2;
    memory.workload.memoryAccesses = {access};
    memory.implementations = {simd};
    table.stages.push_back(memory);
  }

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 50.0);
  EXPECT_FALSE(routes->allSimd.simdMultiResultPrefixScanCompletionApplied);
}

TEST(SimdSimtCostModelTest,
     SourceGridStrideResidentMemoryReplacesOnlyDirectLoadStoreRates) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 112;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simtGridStrideResidentMemoryMaximumBytes = 64 * 1024 * 1024;
  table.simtGridStrideResidentLoadWarpInstructionsPerCycle = 4.0;
  table.simtGridStrideResidentStoreWarpInstructionsPerCycle = 2.0;
  table.simtGridStrideResidentMemoryEvidence = "synthetic resident sweep";

  auto makeCost = [](int64_t factor, double cycles) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, factor};
    cost.totalCycles = cycles;
    cost.resources.load = 40.0;
    cost.resources.store = 20.0;
    cost.resources.compute = 10.0;
    return cost;
  };
  LogicalStageCost stage;
  stage.id = "resident_direct_memory";
  stage.model = "independent_pipelined_loop";
  stage.features.hasLoop = true;
  stage.features.hasContiguousMemory = true;
  stage.features.isGridStrideLoopShell = true;
  stage.workload.loadBytes = 64.0;
  stage.workload.storeBytes = 32.0;
  stage.workload.loadWarpInstructions = 8.0;
  stage.workload.storeWarpInstructions = 4.0;
  stage.implementations = {makeCost(1, 70.0), makeCost(2, 140.0),
                           makeCost(4, 280.0)};
  table.stages = {stage};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  for (int64_t factor : {1, 2, 4}) {
    const auto &plan = routes->allSimtFactorPlans.at(factor);
    // Per source-loop body: compute 10 + load 8/4 + store 4/2 = 14.
    EXPECT_DOUBLE_EQ(plan.totalCycles, 28.0);
    EXPECT_TRUE(plan.gridStrideResidentMemoryApplied);
    EXPECT_DOUBLE_EQ(plan.gridStrideResidentMemoryWorkingSetBytes,
                     112.0 * 96.0);
    EXPECT_EQ(plan.gridStrideResidentMemoryEvidence,
              "synthetic resident sweep");
  }

  table.simtGridStrideResidentMemoryMaximumBytes = 1000;
  routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_DOUBLE_EQ(routes->allSimtFactorPlans.at(1).totalCycles, 140.0);
  EXPECT_FALSE(
      routes->allSimtFactorPlans.at(1).gridStrideResidentMemoryApplied);
}

TEST(SimdSimtCostModelTest,
     SourceGridStrideSimdScanCurveOwnsAdjacentDirectMemoryStages) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 112;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;

  auto makeSimdCost = [](double cycles, double scanBody = 0.0) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMD, 1};
    cost.totalCycles = cycles;
    cost.resources.gridStridePrefixScanBody = scanBody;
    return cost;
  };
  LogicalStageCost shell;
  shell.id = "source_grid_stride_shell";
  shell.model = "independent_pipelined_loop";
  shell.features.isGridStrideLoopShell = true;
  shell.implementations = {makeSimdCost(10.0)};

  LogicalStageCost load;
  load.id = "scan_load";
  load.model = "continuous_tile_memory";
  load.features.insideGridStrideLoopBody = true;
  load.features.hasContiguousMemory = true;
  load.implementations = {makeSimdCost(100.0)};

  LogicalStageCost scan;
  scan.id = "scan";
  scan.model = "prefix_scan";
  scan.features.insideGridStrideLoopBody = true;
  scan.features.hasPrefixScan = true;
  scan.features.hasReduction = true;
  scan.implementations = {makeSimdCost(250.0, 250.0)};

  LogicalStageCost store = load;
  store.id = "scan_store";
  store.model = "continuous_tile_store";
  store.implementations = {makeSimdCost(80.0)};
  table.stages = {shell, load, scan, store};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.legal);
  // Q=ceil(112/56)=2.  The shell and completion-inclusive scan are replayed;
  // the scan-owned 100+80 direct memory cycles are removed from both waves.
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 2.0 * (10.0 + 250.0));
  EXPECT_TRUE(routes->allSimd.gridStrideSimdScanMemoryOwnershipApplied);
  EXPECT_DOUBLE_EQ(routes->allSimd.gridStrideSimdScanMemoryOwnershipCycles,
                   2.0 * (100.0 + 80.0));
  EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[1], 0.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.logicalStageCycles[3], 0.0);
}

TEST(SimdSimtCostModelTest,
     StraightLineSimdLoadRunOwnsOneMteProgramStartupPerWave) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 112;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdLoadOperationStartupCycles = 10.0;

  auto makeLoad = [](llvm::StringRef id) {
    LogicalStageCost stage;
    stage.id = id.str();
    stage.model = "continuous_tile_memory";
    stage.features.hasContiguousMemory = true;
    stage.workload.loadOperations = 1.0;
    stage.workload.loadBytes = 4096.0;
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMD, 1};
    cost.resources.load = 120.0;
    cost.totalCycles = 120.0;
    stage.implementations = {cost};
    return stage;
  };
  table.stages = {makeLoad("load0"), makeLoad("load1"), makeLoad("load2")};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.legal);
  // Q=2. All three byte streams remain charged; only two duplicate MTE2
  // establishment costs are removed from each runtime wave.
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles,
                   2.0 * (3.0 * 120.0 - 2.0 * 10.0));
  EXPECT_TRUE(routes->allSimd.straightLineSimdLoadStartupOwnershipApplied);
  EXPECT_EQ(routes->allSimd.straightLineSimdLoadStartupOwnedOperationCount, 3);
  EXPECT_DOUBLE_EQ(routes->allSimd.straightLineSimdLoadStartupAdjustmentCycles,
                   -40.0);

  LogicalStageCost transparent;
  transparent.id = "zero_cost_index";
  transparent.model = "index_generation";
  StageImplementationCost transparentCost;
  transparentCost.implementation = {StageMode::SIMD, 1};
  transparentCost.totalCycles = 0.0;
  transparent.implementations = {transparentCost};
  table.stages = {makeLoad("load0"), transparent, makeLoad("load1"),
                  transparent, makeLoad("load2")};
  routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles,
                   2.0 * (3.0 * 120.0 - 2.0 * 10.0));
  EXPECT_TRUE(routes->allSimd.straightLineSimdLoadStartupOwnershipApplied);

  LogicalStageCost indirect;
  indirect.id = "loaded_index_gather";
  indirect.model = "indirect_gather_memory";
  indirect.features.hasIndirectMemory = true;
  StageImplementationCost indirectCost;
  indirectCost.implementation = {StageMode::SIMD, 1};
  indirectCost.resources.indirectLoadGroup = 50.0;
  indirectCost.totalCycles = 50.0;
  indirect.implementations = {indirectCost};
  table.stages.push_back(indirect);
  routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 2.0 * (3.0 * 120.0 + 50.0));
  EXPECT_FALSE(routes->allSimd.straightLineSimdLoadStartupOwnershipApplied);

  table.stages = {makeLoad("load0"), makeLoad("load1")};
  routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 2.0 * 2.0 * 120.0);
  EXPECT_FALSE(routes->allSimd.straightLineSimdLoadStartupOwnershipApplied);
}

TEST(SimdSimtCostModelTest, ShortSimdMultiMemoryKernelChargesOneGroupStartup) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 112;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdIndependentLoadGroupKernelStartupCycles = 50.0;

  auto makeMemory = [](llvm::StringRef id, bool store, double cycles) {
    LogicalStageCost stage;
    stage.id = id.str();
    stage.model = store ? "continuous_tile_store" : "continuous_tile_memory";
    stage.features.hasContiguousMemory = true;
    if (store)
      stage.workload.storeOperations = 1.0;
    else
      stage.workload.loadOperations = 1.0;
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMD, 1};
    if (store)
      cost.resources.store = cycles;
    else
      cost.resources.load = cycles;
    cost.totalCycles = cycles;
    stage.implementations = {cost};
    return stage;
  };
  LogicalStageCost compute;
  compute.id = "add";
  compute.model = "vector_issue";
  StageImplementationCost computeCost;
  computeCost.implementation = {StageMode::SIMD, 1};
  computeCost.resources.compute = 20.0;
  computeCost.totalCycles = 20.0;
  compute.implementations = {computeCost};
  table.stages = {makeMemory("load0", false, 100.0),
                  makeMemory("load1", false, 100.0),
                  makeMemory("load2", false, 100.0), compute,
                  makeMemory("store", true, 80.0)};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles,
                   2.0 * (100.0 + 100.0 + 100.0 + 20.0 + 80.0) + 50.0);
  EXPECT_TRUE(routes->allSimd.shortSimdMultiMemoryStartupApplied);
  EXPECT_EQ(routes->allSimd.shortSimdMultiMemoryLoadStageCount, 3);
  EXPECT_EQ(routes->allSimd.shortSimdMultiMemoryStoreStageCount, 1);

  table.stages.erase(table.stages.begin() + 2);
  routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles,
                   2.0 * (100.0 + 100.0 + 20.0 + 80.0));
  EXPECT_FALSE(routes->allSimd.shortSimdMultiMemoryStartupApplied);
}

TEST(SimdSimtCostModelTest,
     SubgridAutoBlockifyTransfersIndependentLoadStartupToBoundary) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 2;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simdIndependentLoadGroupKernelStartupCycles = 50.0;

  LogicalStageCost stage;
  stage.id = "subgrid_loop";
  stage.model = "independent_pipelined_loop";
  stage.features.hasLoop = true;
  stage.features.hasContiguousMemory = true;
  stage.features.insideAutoBlockifyV1Loop = true;
  stage.workload.loadOperations = 1.0;
  StageImplementationCost cost;
  cost.implementation = {StageMode::SIMD, 1};
  cost.resources.setup = 50.0;
  cost.resources.continuousLoadGroup = 100.0;
  cost.totalCycles = 150.0;
  stage.implementations = {cost};
  table.stages = {stage};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.legal);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 100.0);
  EXPECT_TRUE(routes->allSimd.lowOccupancySimdLoadGroupStartupOwnershipApplied);
  EXPECT_DOUBLE_EQ(
      routes->allSimd.lowOccupancySimdLoadGroupStartupAdjustmentCycles, -50.0);

  // A grouped flat prefix followed by residual strided/indirect loads owns a
  // distinct MTE program.  Its startup is not contained in the generic
  // sub-grid kernel boundary and must remain in the Stage.
  table.stages.front().implementations.front().resources.load = 25.0;
  table.stages.front().implementations.front().totalCycles = 175.0;
  routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 175.0);
  EXPECT_FALSE(
      routes->allSimd.lowOccupancySimdLoadGroupStartupOwnershipApplied);

  table.stages.front().implementations.front().resources.load = 0.0;
  table.stages.front().implementations.front().totalCycles = 150.0;
  table.logicalProgramCountHint = 56;
  routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 150.0);
  EXPECT_FALSE(
      routes->allSimd.lowOccupancySimdLoadGroupStartupOwnershipApplied);
}

TEST(SimdSimtCostModelTest,
     LargeSimdIrregularTileWorkingSetRestoresFullMaterialization) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 4;
  table.physicalCoreCountHint = 4;
  table.physicalAiCoreCountHint = 2;
  table.simdIrregularTileLargeWorkingSetFullPhysicalMinimumBytes = 4096;
  table.simdIrregularTileCompletionEvidence = "synthetic byte sweep";

  LogicalStageCost dispatch;
  dispatch.id = "dispatch";
  dispatch.model = "auto_blockify_dispatch";
  StageImplementationCost dispatchCost;
  dispatchCost.implementation = {StageMode::SIMD, 1};
  dispatch.implementations = {dispatchCost};

  LogicalStageCost rowwise;
  rowwise.id = "rowwise";
  rowwise.model = "rowwise_reduction";
  rowwise.iterationCount = 2;
  rowwise.features.insideAutoBlockifyV1Loop = true;
  rowwise.workload.loadBytes = 300.0;
  rowwise.workload.storeBytes = 100.0;
  StageImplementationCost rowwiseCost;
  rowwiseCost.implementation = {StageMode::SIMD, 1};
  rowwiseCost.resources.load = 10.0;
  rowwiseCost.resources.compute = 20.0;
  rowwiseCost.resources.irregularPhysicalMaterialization = 20.0;
  rowwiseCost.resources.irregularFullPhysicalMaterialization = 80.0;
  rowwiseCost.resources.irregularMaskedTileFloor = 50.0;
  rowwiseCost.totalCycles = 2.0 * 50.0;
  rowwise.implementations = {rowwiseCost};
  table.stages = {dispatch, rowwise};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.legal);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 100.0);
  EXPECT_FALSE(routes->allSimd.largeSimdIrregularTileWorkingSetApplied);

  table.stages[1].iterationCount = 3;
  table.stages[1].implementations[0].totalCycles = 3.0 * 50.0;
  routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimd.legal);
  EXPECT_DOUBLE_EQ(routes->allSimd.totalCycles, 3.0 * 110.0);
  EXPECT_TRUE(routes->allSimd.largeSimdIrregularTileWorkingSetApplied);
  EXPECT_DOUBLE_EQ(routes->allSimd.largeSimdIrregularTileWorkingSetBytes,
                   4800.0);
  EXPECT_DOUBLE_EQ(routes->allSimd.largeSimdIrregularTileAdjustmentCycles,
                   180.0);
  EXPECT_EQ(routes->allSimd.largeSimdIrregularTileEvidence,
            "synthetic byte sweep");
}

TEST(SimdSimtCostModelTest, StraightLineSimtScanOwnsPureGenericIndexPrologue) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 1;
  table.physicalCoreCountHint = 56;
  table.physicalAiCoreCountHint = 28;
  table.simtStraightLinePrefixScanElementsPerCycle = 10.0;
  table.simtStraightLinePrefixScanCompletionEvidence = "unit test";

  auto makeSimtCost = [](double cycles) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, 1};
    cost.totalCycles = cycles;
    return cost;
  };
  LogicalStageCost index;
  index.id = "pure_index";
  index.model = "index_generation";
  index.workload.operationCounts["generic.issue"][32] = 1.0;
  index.implementations = {makeSimtCost(250.0)};

  LogicalStageCost scan;
  scan.id = "scan";
  scan.model = "prefix_scan";
  scan.features.hasPrefixScan = true;
  scan.features.hasReduction = true;
  scan.implementations = {makeSimtCost(50.0)};

  LogicalStageCost memory;
  memory.id = "load";
  memory.model = "continuous_tile_memory";
  memory.features.hasContiguousMemory = true;
  memory.workload.loadOperations = 1.0;
  memory.implementations = {makeSimtCost(200.0)};
  table.stages = {index, scan, memory};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  ASSERT_TRUE(routes->allSimt.legal);
  EXPECT_DOUBLE_EQ(routes->allSimt.totalCycles, 250.0);
  EXPECT_DOUBLE_EQ(routes->allSimt.logicalStageCycles[0], 0.0);
  EXPECT_TRUE(routes->allSimt.straightLineSimtScanIndexOwnershipApplied);
  EXPECT_DOUBLE_EQ(routes->allSimt.straightLineSimtScanIndexOwnershipCycles,
                   -250.0);
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
  ASSERT_EQ(routes->allSimtFactorPlans.size(), 3u);
  EXPECT_DOUBLE_EQ(routes->allSimtFactorPlans.at(1).totalCycles, 10.0);
  EXPECT_DOUBLE_EQ(routes->allSimtFactorPlans.at(2).totalCycles, 5.0);
  EXPECT_DOUBLE_EQ(routes->allSimtFactorPlans.at(4).totalCycles, 4.0);
  auto factorJSON = routes->toJSON();
  const auto *factorRoutes = factorJSON.getObject("all_simt_factor_routes");
  ASSERT_NE(factorRoutes, nullptr);
  EXPECT_NE(factorRoutes->getObject("1"), nullptr);
  EXPECT_NE(factorRoutes->getObject("2"), nullptr);
  EXPECT_NE(factorRoutes->getObject("4"), nullptr);
}

TEST(SimdSimtCostModelTest,
     PureSimtGenericDotTailChargesOnlyActiveLogicalPrograms) {
  StageCostTable table;
  table.profileVersion = "unit-test-profile-v1";
  table.logicalProgramCountHint = 10;
  table.physicalCoreCountHint = 1;
  table.physicalAiCoreCountHint = 1;

  auto makeCost = [](int64_t factor, double cycles) {
    StageImplementationCost cost;
    cost.implementation = {StageMode::SIMT, factor};
    cost.totalCycles = cycles;
    cost.resources.dot = 24.0;
    return cost;
  };
  LogicalStageCost dot;
  dot.id = "generic_dot";
  dot.model = "tiny_cube_roofline";
  dot.features.hasDot = true;
  dot.features.insideAutoBlockifyV1Loop = true;
  dot.implementations = {makeCost(1, 24.0), makeCost(2, 48.0),
                         makeCost(4, 96.0)};
  table.stages = {dot};

  auto routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  const auto &f4 = routes->allSimtFactorPlans.at(4);
  ASSERT_EQ(f4.runtimeFullGroupCount, 2);
  ASSERT_EQ(f4.runtimeTailProgramCount, 2);
  ASSERT_EQ(f4.logicalStageCycles.size(), 1u);
  // Two full F4 groups execute 8 logical programs. The final F4 VF keeps its
  // group envelope, but only two of four guarded programs execute generic dot
  // arithmetic: 2*96 + 2*24 = 240, rather than 3*96 = 288.
  EXPECT_DOUBLE_EQ(f4.logicalStageCycles[0], 240.0);
  EXPECT_EQ(f4.runtimeStageTailModes[0], "factorF_predicated_tail");

  // A measured operand-path composite owns the complete physical-group
  // envelope and bypasses the analytical inactive-program credit.
  table.stages[0].implementations[2].resources.dotOperandGroup = 96.0;
  routes = solveStageRoutes(table, ScopeHandoffCost{});
  if (!routes)
    FAIL() << llvm::toString(routes.takeError());
  EXPECT_DOUBLE_EQ(routes->allSimtFactorPlans.at(4).logicalStageCycles[0],
                   288.0);
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
     UnknownLocationSupportPrefixJoinsTheDominantResourceStage) {
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
  for (mlir::Operation *operation : roots)
    operation->setLoc(mlir::UnknownLoc::get(&context));
  ASSERT_TRUE(llvm::all_of(roots, [](mlir::Operation *operation) {
    return mlir::isa<mlir::UnknownLoc>(operation->getLoc());
  }));

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

TEST(SimdSimtCostModelTest,
     GridStrideLoopSurvivesAutoBlockifyProgramIdRemapping) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%pointer: i64, %logicalPrograms: index) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %programs = "tt.get_num_programs"() : () -> index
        scf.for %resident = %c0 to %programs step %c1 {
          scf.for %program = %resident to %logicalPrograms step %programs {
            %value = "tt.gather"(%pointer) : (i64) -> tensor<16xf32>
          }
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::scf::ForOp schedulingLoop;
  mlir::scf::ForOp sourceLoop;
  mlir::Operation *nestedLoad = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (auto loop = llvm::dyn_cast<mlir::scf::ForOp>(operation)) {
      if (loop->getParentOfType<mlir::scf::ForOp>())
        sourceLoop = loop;
      else
        schedulingLoop = loop;
    }
    if (operation->getName().getStringRef() == "tt.gather")
      nestedLoad = operation;
  });
  ASSERT_TRUE(schedulingLoop);
  ASSERT_TRUE(sourceLoop);
  ASSERT_NE(nestedLoad, nullptr);
  schedulingLoop->setAttr("ta.auto_blockify_v1.loop",
                          mlir::UnitAttr::get(&context));

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = nestedLoad;
  anchor.scopeOperations.push_back(nestedLoad);
  anchor.scopeInsertionPoint = nestedLoad;
  anchor.kind = mlir::ascend::SimtAnchorKind::DirectGather;
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  auto result = StagePartitioner().partition(*module, anchorPlan,
                                             StagePartitionerOptions{});
  if (!result)
    FAIL() << llvm::toString(result.takeError());

  const LogicalStage *shellStage = nullptr;
  const LogicalStage *bodyStage = nullptr;
  for (const LogicalStage &stage : result->stages) {
    if (llvm::is_contained(stage.operations, sourceLoop.getOperation()))
      shellStage = &stage;
    if (llvm::is_contained(stage.operations, nestedLoad))
      bodyStage = &stage;
  }
  ASSERT_NE(shellStage, nullptr);
  ASSERT_NE(bodyStage, nullptr);
  EXPECT_TRUE(shellStage->features.isGridStrideLoopShell);
  EXPECT_TRUE(bodyStage->features.insideGridStrideLoopBody);
  EXPECT_TRUE(bodyStage->features.insideAutoBlockifyV1Loop);
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

TEST(SimdSimtCostModelTest, StaticRangeMaskReportsActiveElementRatio) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel() {
      %pointer = "test.source"() : () -> tensor<256xi64>
      %range = "tt.make_range"() {start = 0 : i32, end = 256 : i32}
        : () -> tensor<256xi32>
      %limit = arith.constant dense<150> : tensor<256xi32>
      %mask = arith.cmpi slt, %range, %limit : tensor<256xi32>
      %zero = arith.constant dense<0.0> : tensor<256xf32>
      %loaded = "tt.load"(%pointer, %mask, %zero)
        : (tensor<256xi64>, tensor<256xi1>, tensor<256xf32>)
        -> tensor<256xf32>
      "test.sink"(%loaded) : (tensor<256xf32>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::ascend::SimtAnchorPlan anchors;
  auto partition =
      StagePartitioner().partition(*module, anchors, StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());
  const LogicalStage *memoryStage = nullptr;
  for (const auto &stage : partition->stages)
    if (!stage.workload.memoryAccesses.empty())
      memoryStage = &stage;
  ASSERT_NE(memoryStage, nullptr);
  ASSERT_EQ(memoryStage->workload.memoryAccesses.size(), 1u);
  EXPECT_DOUBLE_EQ(
      memoryStage->workload.memoryAccesses.front().activeElementRatio,
      150.0 / 256.0);
  EXPECT_DOUBLE_EQ(memoryStage->features.activeLaneRatio, 150.0 / 256.0);
}

TEST(SimdSimtCostModelTest,
     NonAffineFloatingAddressDagIsRecordedOnTheMemoryAccess) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel(%scale: f32) {
      %range = "tt.make_range"() {start = 0 : i32, end = 32 : i32}
        : () -> tensor<32xi32>
      %range_f = arith.sitofp %range : tensor<32xi32> to tensor<32xf32>
      %scale_t = "tt.splat"(%scale) : (f32) -> tensor<32xf32>
      %scaled = arith.mulf %range_f, %scale_t : tensor<32xf32>
      %rounded = "math.floor"(%scaled) : (tensor<32xf32>) -> tensor<32xf32>
      %index = arith.fptosi %rounded : tensor<32xf32> to tensor<32xi32>
      %loaded = "tt.load"(%index) : (tensor<32xi32>) -> tensor<32xi32>
      %affine = "tt.load"(%range) : (tensor<32xi32>) -> tensor<32xi32>
      "test.sink"(%loaded, %affine)
        : (tensor<32xi32>, tensor<32xi32>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::ascend::SimtAnchorPlan anchors;
  auto partition =
      StagePartitioner().partition(*module, anchors, StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());

  std::vector<StageMemoryAccess> loads;
  for (const auto &stage : partition->stages)
    for (const auto &access : stage.workload.memoryAccesses)
      if (!access.store)
        loads.push_back(access);
  ASSERT_EQ(loads.size(), 2u);
  EXPECT_TRUE(loads[0].addressHasNonAffineTensorComputation);
  EXPECT_EQ(loads[0].layoutClass(), StageMemoryAccess::LayoutClass::Unknown);
  EXPECT_FALSE(loads[1].addressHasNonAffineTensorComputation);
  EXPECT_TRUE(loads[1].isFlatContiguous());
}

TEST(SimdSimtCostModelTest,
     TensorControlFlowSelectedAddressIsRecordedOnTheMemoryAccess) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel(%condition: i1) {
      %range = "tt.make_range"() {start = 0 : i32, end = 32 : i32}
        : () -> tensor<32xi32>
      %two = arith.constant dense<2> : tensor<32xi32>
      %selected = scf.if %condition -> (tensor<32xi32>) {
        scf.yield %range : tensor<32xi32>
      } else {
        %twice = arith.muli %range, %two : tensor<32xi32>
        scf.yield %twice : tensor<32xi32>
      }
      %selected_load = "tt.load"(%selected)
        : (tensor<32xi32>) -> tensor<32xf32>
      %affine_load = "tt.load"(%range)
        : (tensor<32xi32>) -> tensor<32xf32>
      "test.sink"(%selected_load, %affine_load)
        : (tensor<32xf32>, tensor<32xf32>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::ascend::SimtAnchorPlan anchors;
  auto partition =
      StagePartitioner().partition(*module, anchors, StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());

  std::vector<StageMemoryAccess> loads;
  for (const auto &stage : partition->stages)
    for (const auto &access : stage.workload.memoryAccesses)
      if (!access.store)
        loads.push_back(access);
  ASSERT_EQ(loads.size(), 2u);
  EXPECT_TRUE(loads[0].addressHasTensorControlFlowSelection);
  EXPECT_EQ(loads[0].layoutClass(), StageMemoryAccess::LayoutClass::Unknown);
  EXPECT_FALSE(loads[1].addressHasTensorControlFlowSelection);
  EXPECT_TRUE(loads[1].isFlatContiguous());
}

TEST(SimdSimtCostModelTest,
     MaskedStoreRecordsMatchingMaskedLoadValueDependency) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module { func.func @kernel(%pointer: tensor<8x8xi64>,
                               %mask: tensor<8x8xi1>) {
      %zero = arith.constant dense<0.0> : tensor<8x8xf16>
      %one = arith.constant dense<1.0> : tensor<8x8xf16>
      %other_mask = arith.constant dense<true> : tensor<8x8xi1>
      %loaded = "tt.load"(%pointer, %mask, %zero)
        : (tensor<8x8xi64>, tensor<8x8xi1>, tensor<8x8xf16>)
          -> tensor<8x8xf16>
      %sum = arith.addf %loaded, %one : tensor<8x8xf16>
      "tt.store"(%pointer, %sum, %mask)
        : (tensor<8x8xi64>, tensor<8x8xf16>, tensor<8x8xi1>) -> ()
      "tt.store"(%pointer, %sum, %other_mask)
        : (tensor<8x8xi64>, tensor<8x8xf16>, tensor<8x8xi1>) -> ()
      return
    } })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::ascend::SimtAnchorPlan anchors;
  auto partition =
      StagePartitioner().partition(*module, anchors, StagePartitionerOptions{});
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());

  std::vector<StageMemoryAccess> stores;
  for (const auto &stage : partition->stages)
    for (const auto &access : stage.workload.memoryAccesses)
      if (access.store)
        stores.push_back(access);
  ASSERT_EQ(stores.size(), 2u);
  EXPECT_TRUE(stores[0].maskedStoreHasMatchingLoadDependency);
  EXPECT_FALSE(stores[1].maskedStoreHasMatchingLoadDependency);
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
    // Scalar numeric operations retain ScalarIssue classification, but their
    // execution now uses the same operation-rate path at elements=1 instead
    // of disappearing into the issue-only scalar bucket.
    EXPECT_DOUBLE_EQ(partition->stages.front().workload.scalarOperations, 0.0);
    EXPECT_FALSE(partition->stages.front().workload.operationCounts.empty());
    if (!shaped)
      EXPECT_DOUBLE_EQ(
          partition->stages.front().workload.operationCounts.begin()->second.at(
              1),
          1.0);
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

TEST(SimdSimtCostModelTest,
     AtomicLocalScopeIsRejectedWhileWholeKernelSimtRemainsLegal) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%address: tensor<64x!tt.ptr<i32>>,
                        %value: tensor<64xi32>,
                        %mask: tensor<64xi1>) {
        %old = "tt.atomic_rmw"(%address, %value, %mask)
          {atomic_rmw_op = 0 : i32, sem = 4 : i32, scope = 1 : i32}
          : (tensor<64x!tt.ptr<i32>>, tensor<64xi32>, tensor<64xi1>)
            -> tensor<64xi32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *atomic = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "tt.atomic_rmw")
      atomic = operation;
  });
  ASSERT_NE(atomic, nullptr);

  mlir::ascend::SimtAnchorDescriptor anchor;
  anchor.operation = atomic;
  anchor.scopeOperations = {atomic};
  anchor.scopeInsertionPoint = atomic;
  anchor.kind = mlir::ascend::SimtAnchorKind::DirectGather;
  anchor.lowerability.mixed = true;
  anchor.materializable = true;
  mlir::ascend::SimtAnchorPlan anchorPlan;
  anchorPlan.anchors.push_back(std::move(anchor));

  mlir::ascend::ProgramStructure structure;
  structure.rootOperations = {atomic};
  auto result =
      mlir::ascend::StageBoundaryAnalysis().analyze(structure, anchorPlan);
  if (!result)
    FAIL() << llvm::toString(result.takeError());
  ASSERT_EQ(result->stages.size(), 1u);
  const LogicalStage &stage = result->stages.front();
  EXPECT_EQ(stage.costModelKind, StageCostModelKind::AtomicMemory);
  EXPECT_FALSE(stage.localSimtMaterializable);
  EXPECT_TRUE(stage.localSimtFactors.empty());
  EXPECT_TRUE(stage.simtAnchorIndices.empty());
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

TEST(SimdSimtCostModelTest,
     AutoBlockifyBodyAllowsDotWithLoopCarriedRecurrence) {
  StagePartition partition;
  partition.operationOwnershipComplete = true;
  LogicalStage stage =
      logicalStage("auto_blockify_body", StageCostModelKind::TinyCubeRoofline,
                   StageScheduleKind::PartiallyDependent, 1);
  stage.features.hasDot = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.features.insideAutoBlockifyV1Loop = true;
  stage.workload.dotOperations = 1.0;
  partition.stages.push_back(std::move(stage));

  if (llvm::Error error =
          mlir::ascend::StageKindClassifier().analyze(partition, 16384))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(partition.stages.front().costModelKind,
            StageCostModelKind::LoopCarriedRecurrence);
}

TEST(SimdSimtCostModelTest,
     LoadedScalarBaseIsUniformButLoadedTensorIndicesAreIndirect) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%base: !tt.ptr<f32>, %indices: !tt.ptr<i64>) {
        %scalar_index = "tt.load"(%indices) : (!tt.ptr<i64>) -> i64
        %base_vector = "tt.splat"(%base)
            : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
        %uniform_offsets = "tt.splat"(%scalar_index)
            : (i64) -> tensor<64xi64>
        %uniform_ptrs = "tt.addptr"(%base_vector, %uniform_offsets)
            : (tensor<64x!tt.ptr<f32>>, tensor<64xi64>)
              -> tensor<64x!tt.ptr<f32>>
        %uniform_values = "tt.load"(%uniform_ptrs)
            : (tensor<64x!tt.ptr<f32>>) -> tensor<64xf32>
        %zero = arith.constant dense<0> : tensor<64xi64>
        %index_base = "tt.splat"(%indices)
            : (!tt.ptr<i64>) -> tensor<64x!tt.ptr<i64>>
        %index_ptrs = "tt.addptr"(%index_base, %zero)
            : (tensor<64x!tt.ptr<i64>>, tensor<64xi64>)
              -> tensor<64x!tt.ptr<i64>>
        %loaded_indices = "tt.load"(%index_ptrs)
            : (tensor<64x!tt.ptr<i64>>) -> tensor<64xi64>
        %indirect_ptrs = "tt.addptr"(%base_vector, %loaded_indices)
            : (tensor<64x!tt.ptr<f32>>, tensor<64xi64>)
              -> tensor<64x!tt.ptr<f32>>
        %indirect_values = "tt.load"(%indirect_ptrs)
            : (tensor<64x!tt.ptr<f32>>) -> tensor<64xf32>
        %reduced_index = "tt.reduce"(%loaded_indices) ({
        ^bb0(%left: i64, %right: i64):
          %minimum = arith.minsi %left, %right : i64
          "tt.reduce.return"(%minimum) : (i64) -> ()
        }) {axis = 0 : i32} : (tensor<64xi64>) -> i64
        %reduced_offsets = "tt.splat"(%reduced_index)
            : (i64) -> tensor<64xi64>
        %reduced_base_ptrs = "tt.addptr"(%base_vector, %reduced_offsets)
            : (tensor<64x!tt.ptr<f32>>, tensor<64xi64>)
              -> tensor<64x!tt.ptr<f32>>
        %reduced_base_values = "tt.load"(%reduced_base_ptrs)
            : (tensor<64x!tt.ptr<f32>>) -> tensor<64xf32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  mlir::Operation *uniformLoad = nullptr;
  mlir::Operation *indirectLoad = nullptr;
  mlir::Operation *reducedBaseLoad = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() != "tt.load" ||
        operation->getNumResults() != 1)
      return;
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(
        operation->getResult(0).getType());
    if (!type || !type.getElementType().isF32())
      return;
    if (!uniformLoad)
      uniformLoad = operation;
    else if (!indirectLoad)
      indirectLoad = operation;
    else
      reducedBaseLoad = operation;
  });
  ASSERT_NE(uniformLoad, nullptr);
  ASSERT_NE(indirectLoad, nullptr);
  ASSERT_NE(reducedBaseLoad, nullptr);
  EXPECT_FALSE(mlir::ascend::isLoadedIndexDependentMemoryOp(uniformLoad));
  EXPECT_TRUE(mlir::ascend::isLoadedIndexDependentMemoryOp(indirectLoad));
  EXPECT_FALSE(mlir::ascend::isLoadedIndexDependentMemoryOp(reducedBaseLoad));
}

TEST(SimdSimtCostModelTest,
     LoadedIndexRowsPreserveContiguousSegmentsButColumnsStayElementwise) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%base: !tt.ptr<f16>, %indices: !tt.ptr<i64>,
                        %pitch: i64) {
        %range = "tt.make_range"() {start = 0 : i32, end = 16 : i32}
            : () -> tensor<16xi64>
        %index_base = "tt.splat"(%indices)
            : (!tt.ptr<i64>) -> tensor<16x!tt.ptr<i64>>
        %index_ptrs = "tt.addptr"(%index_base, %range)
            : (tensor<16x!tt.ptr<i64>>, tensor<16xi64>)
              -> tensor<16x!tt.ptr<i64>>
        %loaded_indices = "tt.load"(%index_ptrs)
            : (tensor<16x!tt.ptr<i64>>) -> tensor<16xi64>

        %loaded_rows_2d = "tt.expand_dims"(%loaded_indices) {axis = 1 : i32}
            : (tensor<16xi64>) -> tensor<16x1xi64>
        %pitch_rows = "tt.splat"(%pitch) : (i64) -> tensor<16x1xi64>
        %scaled_rows = arith.muli %loaded_rows_2d, %pitch_rows
            : tensor<16x1xi64>
        %row_offsets = "tt.broadcast"(%scaled_rows)
            : (tensor<16x1xi64>) -> tensor<16x16xi64>
        %columns_2d = "tt.expand_dims"(%range) {axis = 0 : i32}
            : (tensor<16xi64>) -> tensor<1x16xi64>
        %column_offsets = "tt.broadcast"(%columns_2d)
            : (tensor<1x16xi64>) -> tensor<16x16xi64>
        %row_gather_offsets = arith.addi %row_offsets, %column_offsets
            : tensor<16x16xi64>
        %row_base = "tt.splat"(%base)
            : (!tt.ptr<f16>) -> tensor<16x16x!tt.ptr<f16>>
        %row_ptrs = "tt.addptr"(%row_base, %row_gather_offsets)
            : (tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi64>)
              -> tensor<16x16x!tt.ptr<f16>>
        %row_values = "tt.load"(%row_ptrs)
            : (tensor<16x16x!tt.ptr<f16>>) -> tensor<16x16xf16>

        %loaded_columns_2d = "tt.expand_dims"(%loaded_indices)
            {axis = 0 : i32} : (tensor<16xi64>) -> tensor<1x16xi64>
        %loaded_columns = "tt.broadcast"(%loaded_columns_2d)
            : (tensor<1x16xi64>) -> tensor<16x16xi64>
        %rows_2d = "tt.expand_dims"(%range) {axis = 1 : i32}
            : (tensor<16xi64>) -> tensor<16x1xi64>
        %pitch_rows_direct = "tt.splat"(%pitch)
            : (i64) -> tensor<16x1xi64>
        %scaled_rows_direct = arith.muli %rows_2d, %pitch_rows_direct
            : tensor<16x1xi64>
        %direct_rows = "tt.broadcast"(%scaled_rows_direct)
            : (tensor<16x1xi64>) -> tensor<16x16xi64>
        %column_gather_offsets = arith.addi %direct_rows, %loaded_columns
            : tensor<16x16xi64>
        %column_base = "tt.splat"(%base)
            : (!tt.ptr<f16>) -> tensor<16x16x!tt.ptr<f16>>
        %column_ptrs = "tt.addptr"(%column_base, %column_gather_offsets)
            : (tensor<16x16x!tt.ptr<f16>>, tensor<16x16xi64>)
              -> tensor<16x16x!tt.ptr<f16>>
        %column_values = "tt.load"(%column_ptrs)
            : (tensor<16x16x!tt.ptr<f16>>) -> tensor<16x16xf16>
        "test.sink"(%row_values, %column_values)
            : (tensor<16x16xf16>, tensor<16x16xf16>) -> ()
        return
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
  const StageWorkload *rowGather = nullptr;
  const StageWorkload *columnGather = nullptr;
  for (const LogicalStage &stage : partition->stages) {
    if (!stage.workload.segmentedIndirectLoadCounts.empty())
      rowGather = &stage.workload;
    if (!stage.workload.indirectLoadCounts.empty())
      columnGather = &stage.workload;
  }
  ASSERT_NE(rowGather, nullptr);
  ASSERT_NE(columnGather, nullptr);
  EXPECT_EQ(rowGather->segmentedIndirectLoadCounts.count({512, 256, 16}), 1u);
  EXPECT_TRUE(rowGather->indirectLoadCounts.empty());
  ASSERT_EQ(rowGather->memoryAccesses.size(), 1u);
  EXPECT_TRUE(rowGather->memoryAccesses.front().addressDependsOnLoadedIndex);
  EXPECT_EQ(rowGather->memoryAccesses.front().provenContiguousElements(), 16);
  EXPECT_EQ(columnGather->indirectLoadCounts.count({512, 256}), 1u);
  EXPECT_TRUE(columnGather->segmentedIndirectLoadCounts.empty());
  ASSERT_EQ(columnGather->memoryAccesses.size(), 1u);
  EXPECT_EQ(columnGather->memoryAccesses.front().provenContiguousElements(), 1);
}

TEST(SimdSimtCostModelTest,
     AutoBlockifyBodyUsesAlgorithmicModelWhileControlShellStaysSynthetic) {
  StagePartition partition;
  partition.operationOwnershipComplete = true;

  LogicalStage shell =
      logicalStage("dispatch", StageCostModelKind::AutoBlockifyDispatch,
                   StageScheduleKind::StraightLine, 1);
  shell.workload.scalarOperations = 12.0;
  shell.workload.issueElements = 12.0;
  partition.stages.push_back(std::move(shell));

  LogicalStage body =
      logicalStage("body", StageCostModelKind::AutoBlockifyDispatch,
                   StageScheduleKind::LoopCarriedSerial, 16);
  body.features.hasLoop = true;
  body.features.hasLoopCarriedDataDependency = true;
  body.features.hasDot = true;
  body.features.hasReduction = true;
  body.features.hasIndirectMemory = true;
  body.workload.loadBytes = 32768.0;
  body.workload.loadOperations = 2.0;
  body.workload.dotOperations = 1.0;
  body.workload.dotFlops = 524288.0;
  body.workload.dotFlopCounts[524288] = 1.0;
  partition.stages.push_back(std::move(body));

  if (llvm::Error error =
          mlir::ascend::StageKindClassifier().analyze(partition, 16384))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(partition.stages[0].costModelKind,
            StageCostModelKind::AutoBlockifyDispatch);
  EXPECT_EQ(partition.stages[1].costModelKind,
            StageCostModelKind::LoopCarriedRecurrence);
  EXPECT_EQ(partition.stages[1].scheduleKind,
            StageScheduleKind::LoopCarriedSerial);
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
  work.indirectLoadCounts.clear();
  work.segmentedIndirectLoadCounts[{512, 256, 16}] = 1.0;
  EXPECT_TRUE(work.isFiniteAndNonNegative());
  work.segmentedIndirectLoadCounts.clear();
  work.segmentedIndirectLoadCounts[{512, 256, 1}] = 1.0;
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

  StageMemoryAccess partialIndirect;
  partialIndirect.shape = {16, 16};
  partialIndirect.strides = {std::nullopt, 1};
  partialIndirect.elementBits = 16;
  EXPECT_FALSE(partialIndirect.hasStaticLayout());
  EXPECT_EQ(partialIndirect.provenContiguousElements(), 16);

  StageMemoryAccess sixDimensional;
  sixDimensional.shape = {2, 2, 2, 2, 2, 2};
  sixDimensional.strides = {32, 16, 8, 4, 2, 1};
  sixDimensional.elementBits = 32;
  EXPECT_EQ(sixDimensional.effectiveRank(), 6);
  EXPECT_EQ(sixDimensional.descriptorCount(5), 2);
  EXPECT_EQ(sixDimensional.descriptorCount(2), 16);
}

TEST(SimdSimtCostModelTest,
     SimdNonAffineComputedAddressLoadPaysBoundedScalarLaneCompletion) {
  auto stage = logicalStage("computed-address",
                            StageCostModelKind::ContinuousTileMemory);
  stage.workload = {};
  StageMemoryAccess access;
  access.shape = {32};
  access.strides = {std::nullopt};
  access.elementBits = 32;
  access.addressHasNonAffineTensorComputation = true;
  stage.workload.memoryAccesses = {access};
  stage.workload.loadBytes = 128.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.loadWarpInstructions = 1.0;

  auto profile = hardwareProfile();
  profile.simd.computedAddressScalarizedLoadMinimumElements = 8;
  profile.simd.computedAddressScalarizedLoadMaximumElements = 64;
  profile.simd.computedAddressScalarizedLoadCyclesPerElement = 10.0;
  profile.simd.computedAddressScalarizedLoadEvidence = "synthetic unit rate";
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(table));
  const auto &costs = table->stages.front().implementations;
  ASSERT_GE(costs.size(), 2u);
  EXPECT_DOUBLE_EQ(costs[0].resources.computedAddressScalarizedLoad, 320.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.load, 324.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.computedAddressScalarizedLoad, 0.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.load, 1.0);

  stage.workload.memoryAccesses.front().addressDependsOnLoadedIndex = true;
  auto loadedIndex = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(loadedIndex));
  EXPECT_DOUBLE_EQ(loadedIndex->stages.front()
                       .implementations.front()
                       .resources.computedAddressScalarizedLoad,
                   0.0);
}

TEST(SimdSimtCostModelTest,
     SimdControlFlowSelectedAddressLoadPaysBoundedMeasuredCompletion) {
  auto stage = logicalStage("control-flow-address",
                            StageCostModelKind::ContinuousTileMemory);
  stage.workload = {};
  StageMemoryAccess access;
  access.shape = {32};
  access.strides = {std::nullopt};
  access.elementBits = 32;
  access.addressHasTensorControlFlowSelection = true;
  stage.workload.memoryAccesses = {access};
  stage.workload.loadBytes = 128.0;
  stage.workload.loadOperations = 1.0;
  stage.workload.loadWarpInstructions = 1.0;

  auto profile = hardwareProfile();
  profile.simd.controlFlowAddressScalarizedLoadResidualCyclesByElements = {
      {8, 80.0}, {64, 640.0}};
  profile.simd.controlFlowAddressScalarizedLoadEvidence =
      "synthetic bounded sweep";
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(table));
  const auto &costs = table->stages.front().implementations;
  ASSERT_GE(costs.size(), 2u);
  EXPECT_DOUBLE_EQ(costs[0].resources.controlFlowAddressScalarizedLoad, 320.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.load, 324.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.controlFlowAddressScalarizedLoad, 0.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.load, 1.0);

  stage.workload.memoryAccesses.front().addressDependsOnLoadedIndex = true;
  auto loadedIndex = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(loadedIndex));
  EXPECT_DOUBLE_EQ(loadedIndex->stages.front()
                       .implementations.front()
                       .resources.controlFlowAddressScalarizedLoad,
                   0.0);

  stage.workload.memoryAccesses.front().addressDependsOnLoadedIndex = false;
  stage.workload.memoryAccesses.front().shape = {128};
  auto outsideMeasuredRange = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(outsideMeasuredRange));
  EXPECT_DOUBLE_EQ(outsideMeasuredRange->stages.front()
                       .implementations.front()
                       .resources.controlFlowAddressScalarizedLoad,
                   0.0);
}

TEST(SimdSimtCostModelTest,
     SimdMaskedLoadStoreGroupPaysBoundedReadModifyWriteCompletion) {
  auto stage =
      logicalStage("masked-rmw", StageCostModelKind::ContinuousTileStore);
  stage.workload = {};
  StageMemoryAccess access;
  access.shape = {8, 8};
  access.strides = {8, 1};
  access.elementBits = 16;
  access.store = true;
  access.masked = true;
  access.activeElementRatio = 0.75;
  access.maskedStoreHasMatchingLoadDependency = true;
  stage.workload.memoryAccesses = {access};
  stage.workload.storeBytes = 128.0;
  stage.workload.storeOperations = 1.0;
  stage.workload.storeWarpInstructions = 2.0;

  auto profile = hardwareProfile();
  profile.simd.maskedLoadStoreRmwResidualCyclesByRowBytesAndActiveRows = {
      {16, {{4, 400.0}, {12, 920.0}}}};
  profile.simd.maskedLoadStoreRmwEvidence = "synthetic bounded surface";
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(table));
  const auto &costs = table->stages.front().implementations;
  ASSERT_GE(costs.size(), 2u);
  // Six active rows interpolate to 530 cycles between rows four and twelve.
  EXPECT_DOUBLE_EQ(costs[0].resources.maskedLoadStoreReadModifyWrite, 530.0);
  EXPECT_DOUBLE_EQ(costs[0].resources.store, 538.0);
  EXPECT_DOUBLE_EQ(costs[1].resources.maskedLoadStoreReadModifyWrite, 0.0);

  stage.workload.memoryAccesses.front().activeElementRatio = 0.25;
  auto outsideMeasuredRange = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(outsideMeasuredRange));
  EXPECT_DOUBLE_EQ(outsideMeasuredRange->stages.front()
                       .implementations.front()
                       .resources.maskedLoadStoreReadModifyWrite,
                   0.0);
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

TEST(SimdSimtCostModelTest,
     DotStoreSharedExchangeSeparatesGmBatchesOnlyForSimt) {
  auto stage = logicalStage("store", StageCostModelKind::ContinuousTileStore);
  stage.workload = {};
  stage.legalSimtFactors = {2};
  StageMemoryAccess access;
  access.shape = {16, 16};
  access.strides = {2048, 1};
  access.elementBits = 32;
  access.store = true;
  stage.workload.memoryAccesses.assign(4, access);
  access.fromDotAccumulator = true;
  stage.workload.memoryAccesses.insert(stage.workload.memoryAccesses.end(), 6,
                                       access);
  stage.workload.storeBytes = 10240;
  stage.workload.storeOperations = 10;
  stage.workload.storeWarpInstructions = 80;
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  const std::string gmKey = access.layoutKey() + ":warps=8:factor=2";
  profile.simt.storeLayoutCosts[gmKey] = {100, 20, 1, 10, "GM only"};
  auto before = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(before));
  profile.simt.dotStoreSharedLayoutCosts
      ["fma_to_row_major:16x16:bits=32:base_warps=4:factor=2"] = {
      80, 80, 2, 16, "isolated shared exchange, synthetic test"};
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(table));
  const auto &costs = table->stages.front().implementations;
  ASSERT_EQ(costs.size(), 2u);
  EXPECT_DOUBLE_EQ(costs[0].totalCycles,
                   before->stages.front().implementations[0].totalCycles);
  EXPECT_DOUBLE_EQ(costs[0].resources.sharedLayout, 0);
  // GM batches [4], [1], [1], [1], [1], [1], [1]: 160 + 600.
  EXPECT_DOUBLE_EQ(costs[1].resources.continuousStoreGroup, 760);
  // Six exchanges, already measured for the F2 group; no additional F2.
  EXPECT_DOUBLE_EQ(costs[1].resources.sharedLayout, 240);
  EXPECT_DOUBLE_EQ(costs[1].totalCycles, 1240);

  // A measured composite envelope explicitly covers the exchange already.
  profile.simt.storeLayoutCosts[gmKey].includesSharedLayout = true;
  auto composite = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(composite));
  EXPECT_DOUBLE_EQ(
      composite->stages.front().implementations[1].resources.sharedLayout, 0);
  EXPECT_DOUBLE_EQ(composite->stages.front().implementations[1].totalCycles,
                   280);
}

TEST(SimdSimtCostModelTest, SharedExchangeRequiresMeasuredShapeAndCount) {
  auto stage = logicalStage("store", StageCostModelKind::ContinuousTileStore);
  stage.workload = {};
  stage.simdLegal = false;
  stage.legalSimtFactors = {1, 2, 4};
  StageMemoryAccess access;
  access.shape = {16, 16};
  access.strides = {2048, 1};
  access.elementBits = 32;
  access.store = true;
  access.fromDotAccumulator = true;
  access.count = 6;
  stage.workload.memoryAccesses = {access};
  stage.workload.storeBytes = 6144;
  stage.workload.storeOperations = 6;
  stage.workload.storeWarpInstructions = 48;
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.dotStoreSharedLayoutCosts
      ["fma_to_row_major:16x16:bits=32:base_warps=4:factor=2"] = {80, 80, 2, 16,
                                                                  "test"};
  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(table));
  EXPECT_DOUBLE_EQ(
      table->stages.front().implementations[0].resources.sharedLayout, 0);
  EXPECT_DOUBLE_EQ(
      table->stages.front().implementations[1].resources.sharedLayout, 240);
  EXPECT_DOUBLE_EQ(
      table->stages.front().implementations[2].resources.sharedLayout, 0);
  for (int variant : {0, 1, 2, 3}) {
    auto unknown = stage;
    auto &a = unknown.workload.memoryAccesses.front();
    if (variant == 0)
      a.shape = {8, 16};
    if (variant == 1)
      a.fromDotAccumulator = false;
    if (variant == 2)
      a.count = 1;
    if (variant == 3)
      a.strides = {1, 16};
    auto cost = evaluateOneStage(unknown, profile);
    ASSERT_TRUE(static_cast<bool>(cost));
    EXPECT_DOUBLE_EQ(
        cost->stages.front().implementations[1].resources.sharedLayout, 0);
  }
}

TEST(SimdSimtCostModelTest,
     TtirStoreProvenanceFollowsOnlyShapePreservingDotValues) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @provenance(%ptr: i64, %a: tensor<16x16xf32>, %b: tensor<16x16xf32>) {
        %zero = arith.constant dense<0.0> : tensor<16x16xf32>
        %dot = "tt.dot"(%a, %b, %zero) : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
        %neg = arith.subf %zero, %dot : tensor<16x16xf32>
        %mixed = arith.addf %dot, %a : tensor<16x16xf32>
        "tt.store"(%ptr, %dot) : (i64, tensor<16x16xf32>) -> ()
        "tt.store"(%ptr, %neg) : (i64, tensor<16x16xf32>) -> ()
        "tt.store"(%ptr, %mixed) : (i64, tensor<16x16xf32>) -> ()
        "tt.store"(%ptr, %a) : (i64, tensor<16x16xf32>) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::ascend::SimtAnchorPlan anchorPlan;
  auto partition = StagePartitioner().partition(*module, anchorPlan,
                                                StagePartitionerOptions{});
  ASSERT_TRUE(static_cast<bool>(partition));
  std::vector<bool> sources;
  for (const auto &stage : partition->stages)
    for (const auto &access : stage.workload.memoryAccesses)
      if (access.store)
        sources.push_back(access.fromDotAccumulator);
  EXPECT_EQ(sources, (std::vector<bool>{true, true, false, false}));
}

TEST(SimdSimtCostModelTest, CompositeDotCostsReplaceOnlyWholeSimtF2Body) {
  auto stage = logicalStage("dot", StageCostModelKind::TinyCubeRoofline);
  stage.workload = {};
  stage.legalSimtFactors = {1, 2, 4};
  stage.workload.dotOperations = 3;
  stage.workload.dotFlops = 3 * 8192;
  stage.workload.dotFlopCounts[8192] = 3;
  const std::string path = "f32:16x16x16:lhs=computed:rhs=computed";
  stage.workload.dotOperandPaths[path] = {8192, 3};
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  auto before = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(before));
  profile.simt.dotOperandPathCosts[path + ":base_warps=4:factor=2"] = {
      700, 700, 1, 8, "synthetic independent probe slope"};
  auto after = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(after));
  const auto &costs = after->stages.front().implementations;
  ASSERT_EQ(costs.size(), 4u);
  EXPECT_DOUBLE_EQ(costs[2].resources.dotOperandGroup, 2100);
  EXPECT_DOUBLE_EQ(costs[2].resources.dot, 1050);
  EXPECT_DOUBLE_EQ(costs[2].totalCycles, 2100);
  for (unsigned i : {0u, 1u, 3u})
    EXPECT_DOUBLE_EQ(costs[i].totalCycles,
                     before->stages.front().implementations[i].totalCycles);
  const auto &local = after->stages.front().simtCalibrationEstimates;
  const auto &oldLocal = before->stages.front().simtCalibrationEstimates;
  ASSERT_EQ(local.size(), oldLocal.size());
  for (unsigned i = 0; i < local.size(); ++i)
    EXPECT_DOUBLE_EQ(local[i].totalCycles, oldLocal[i].totalCycles);
  profile.baseSimtWarpCount = 2;
  auto wrongWarp = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(wrongWarp));
  EXPECT_DOUBLE_EQ(
      wrongWarp->stages.front().implementations[2].resources.dotOperandGroup,
      0);
}

TEST(SimdSimtCostModelTest, CompositeDotPathIgnoresOnlyOuterProgramPitch) {
  auto stage = logicalStage("dot", StageCostModelKind::TinyCubeRoofline);
  stage.workload = {};
  stage.legalSimtFactors = {2};
  stage.workload.dotOperations = 2;
  stage.workload.dotFlops = 2 * 8192;
  stage.workload.dotFlopCounts[8192] = 2;
  const std::string measuredPath =
      "f32:16x16x16:lhs=computed:rhs=gm{strided_wide:r=2:row=64:"
      "stride=8192:rows=16:bits=32:masked=1:shape=16,16,:"
      "strides=2048,1,}";
  const std::string widerPitchPath =
      "f32:16x16x16:lhs=computed:rhs=gm{strided_wide:r=2:row=64:"
      "stride=16384:rows=16:bits=32:masked=1:shape=16,16,:"
      "strides=4096,1,}";
  stage.workload.dotOperandPaths[widerPitchPath] = {8192, 2};
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.dotOperandPathCosts[measuredPath + ":base_warps=4:factor=2"] = {
      700, 700, 1, 8, "synthetic pitch-independent tile probe"};

  auto table = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(table));
  ASSERT_EQ(table->stages.front().implementations.size(), 2u);
  const auto &f2 = table->stages.front().implementations[1];
  EXPECT_DOUBLE_EQ(f2.resources.dotOperandGroup, 1400);
  EXPECT_DOUBLE_EQ(f2.resources.dot, 700);
  EXPECT_DOUBLE_EQ(f2.totalCycles, 1400);
}

TEST(SimdSimtCostModelTest, PartialCompositeScalesOnlyTheUnmatchedDotResidual) {
  auto stage = logicalStage("dot", StageCostModelKind::TinyCubeRoofline);
  stage.workload = {};
  stage.legalSimtFactors = {2};
  stage.workload.dotOperations = 2;
  stage.workload.dotFlops = 2 * 8192;
  const std::string path = "f32:16x16x16:lhs=computed:rhs=computed";
  stage.workload.dotOperandPaths[path] = {8192, 1};
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  profile.simt.dotSetupCycles = 100;
  profile.simt.dotFlopsPerCycle = 8192;
  profile.simt.dotOperandPathCosts[path + ":base_warps=4:factor=2"] = {
      700, 700, 1, 8, "synthetic exact path"};

  auto fullResidual = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(fullResidual));
  ASSERT_EQ(fullResidual->stages.front().implementations.size(), 2u);
  EXPECT_DOUBLE_EQ(
      fullResidual->stages.front().implementations[1].resources.dot, 451);
  EXPECT_DOUBLE_EQ(fullResidual->stages.front().implementations[1].totalCycles,
                   902);

  profile.simt.dotPartialCompositeResidualRelativeScale = 0.5;
  auto scaledResidual = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(scaledResidual));
  const auto &cost = scaledResidual->stages.front().implementations[1];
  EXPECT_DOUBLE_EQ(cost.resources.dotOperandGroup, 700);
  EXPECT_DOUBLE_EQ(cost.resources.dot, 400.5);
  EXPECT_DOUBLE_EQ(cost.totalCycles, 801);
  EXPECT_NE(cost.formulaEvidence.find("+partial_composite_dot_residual"),
            std::string::npos);
}

TEST(SimdSimtCostModelTest, CompositeDotOwnsGmOnlyWhenAllConsumersAreCovered) {
  auto stage = logicalStage("loads", StageCostModelKind::ContinuousTileMemory);
  stage.workload = {};
  stage.legalSimtFactors = {2};
  StageMemoryAccess access;
  access.shape = {16, 16};
  access.strides = {2048, 1};
  access.elementBits = 32;
  access.count = 6;
  access.dotConsumerPaths = {"path-a", "path-b"};
  stage.workload.memoryAccesses = {access};
  stage.workload.loadBytes = 6144;
  stage.workload.loadOperations = 6;
  stage.workload.loadWarpInstructions = 48;
  stage.workload.issueElements = 1536;
  auto profile = hardwareProfile();
  profile.baseSimtWarpCount = 4;
  auto before = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(before));
  profile.simt.dotOperandPathCosts["path-a:base_warps=4:factor=2"] = {
      700, 700, 1, 8, "test"};
  auto partial = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(partial));
  EXPECT_DOUBLE_EQ(partial->stages.front().implementations[1].totalCycles,
                   before->stages.front().implementations[1].totalCycles);
  profile.simt.dotOperandPathCosts["path-b:base_warps=4:factor=2"] = {
      900, 900, 1, 8, "test"};
  auto covered = evaluateOneStage(stage, profile);
  ASSERT_TRUE(static_cast<bool>(covered));
  EXPECT_DOUBLE_EQ(
      covered->stages.front().implementations[1].resources.dotInputLoadsCovered,
      6);
  EXPECT_DOUBLE_EQ(covered->stages.front().implementations[1].totalCycles, 0);
  EXPECT_DOUBLE_EQ(covered->stages.front().implementations[0].totalCycles,
                   before->stages.front().implementations[0].totalCycles);
  // Raw workload remains visible; only the candidate's cost ownership moved.
  EXPECT_DOUBLE_EQ(covered->stages.front().workload.loadBytes, 6144);
}

TEST(SimdSimtCostModelTest, DotOperandPathsComeFromTtirUsesNotStageIds) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @arbitrary_name(%p: i64, %a: tensor<16x16xf32>) {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %c16 = arith.constant 16 : i32
        %c2048 = arith.constant 2048 : i32
        %z = arith.constant dense<0.0> : tensor<16x16xf32>
        %p2 = "tt.make_tensor_ptr"(%p, %c16, %c16, %c2048, %c1, %c0, %c0) : (i64, i32, i32, i32, i32, i32, i32) -> i64
        %g = "tt.load"(%p2) : (i64) -> tensor<16x16xf32>
        %c = arith.addf %a, %a : tensor<16x16xf32>
        %d1 = "tt.dot"(%c, %g, %z) : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
        %d2 = "tt.dot"(%g, %c, %d1) : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
        %d3 = "tt.dot"(%d1, %d2, %z) : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
        %escape = "tt.load"(%p2) : (i64) -> tensor<16x16xf32>
        %d4 = "tt.dot"(%escape, %c, %z) : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
        "tt.store"(%p, %escape) : (i64, tensor<16x16xf32>) -> ()
        %unknown = "tt.dot"(%a, %c, %z) : (tensor<16x16xf32>, tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::ascend::SimtAnchorPlan anchors;
  auto partition =
      StagePartitioner().partition(*module, anchors, StagePartitionerOptions{});
  ASSERT_TRUE(static_cast<bool>(partition));
  double dots = 0, known = 0;
  std::vector<unsigned> loadConsumers;
  for (const auto &stage : partition->stages) {
    dots += stage.workload.dotOperations;
    for (const auto &[key, entry] : stage.workload.dotOperandPaths)
      known += entry.second;
    for (const auto &a : stage.workload.memoryAccesses)
      if (!a.store)
        loadConsumers.push_back(a.dotConsumerPaths.size());
  }
  EXPECT_DOUBLE_EQ(dots, 5);
  EXPECT_DOUBLE_EQ(known, 4);
  EXPECT_EQ(loadConsumers, (std::vector<unsigned>{2, 0}));
}

TEST(SimdSimtCostModelTest, DotScaledUsesDotWorkloadAndCubeClassification) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::triton::TritonDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @scaled_dot(%a: tensor<16x16xi8>,
                            %b: tensor<32x16xi8>,
                            %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
        %result = tt.dot_scaled %a, %b, %c lhs = e2m1 rhs = e4m3
            {fastMath = false} : tensor<16x16xi8> * tensor<32x16xi8>
            -> tensor<16x16xf32>
        return %result : tensor<16x16xf32>
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::ascend::SimtAnchorPlan anchors;
  auto partition =
      StagePartitioner().partition(*module, anchors, StagePartitionerOptions{});
  ASSERT_TRUE(static_cast<bool>(partition));

  double dotOperations = 0.0;
  double dotFlops = 0.0;
  bool classifiedAsCube = false;
  bool featureHasDot = false;
  for (const auto &stage : partition->stages) {
    dotOperations += stage.workload.dotOperations;
    dotFlops += stage.workload.dotFlops;
    featureHasDot |= stage.features.hasDot;
    classifiedAsCube |=
        stage.costModelKind == StageCostModelKind::CubeRoofline ||
        stage.costModelKind == StageCostModelKind::TinyCubeRoofline;
  }
  EXPECT_DOUBLE_EQ(dotOperations, 1.0);
  EXPECT_DOUBLE_EQ(dotFlops, 2.0 * 16.0 * 32.0 * 16.0);
  EXPECT_TRUE(featureHasDot);
  EXPECT_TRUE(classifiedAsCube);
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
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  const char *text = R"mlir(
    module {
      func.func @kernel(%input: tensor<4x8xf32>) {
        %result = "tt.scan"(%input) ({
        ^bb0(%lhs: f32, %rhs: f32):
          %sum = arith.addf %lhs, %rhs : f32
          "tt.scan.return"(%sum) : (f32) -> ()
        }) {axis = 0 : i32}
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
  EXPECT_DOUBLE_EQ(work.prefixScanCounts.at(
                       std::make_tuple<int64_t, int64_t, int64_t, int64_t, bool,
                                       std::string>(4, 8, 8, 1, false, "add")),
                   1.0);
  EXPECT_DOUBLE_EQ(work.shuffleLaneSteps, 0.0);
  auto json = work.toJSON();
  const auto *instances = json.getArray("prefix_scan_instances_per_iteration");
  ASSERT_NE(instances, nullptr);
  ASSERT_EQ(instances->size(), 1u);
  const auto *instance = (*instances)[0].getAsObject();
  ASSERT_NE(instance, nullptr);
  EXPECT_EQ(instance->getInteger("axis_extent"), 4);
  EXPECT_EQ(instance->getInteger("independent_sequence_count"), 8);
  EXPECT_EQ(instance->getInteger("axis_stride_elements"), 8);
  EXPECT_EQ(instance->getInteger("result_count"), 1);
  EXPECT_EQ(instance->getBoolean("reverse"), false);
  EXPECT_EQ(instance->getString("combine_class"), "add");
  EXPECT_EQ(instance->getNumber("count_per_iteration"), 1.0);
}

TEST(SimdSimtCostModelTest, HistogramKeepsSupportedLocalSimtMaterialization) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%input: tensor<2048xi32>) {
        %counts = "tt.histogram"(%input) {num_bins = 256 : i32}
          : (tensor<2048xi32>) -> tensor<256xi32>
        return
      }
    })mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *histogram = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "tt.histogram")
      histogram = operation;
  });
  ASSERT_NE(histogram, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto stage = logicalStage("histogram", StageCostModelKind::VectorIssue);
  stage.operations.push_back(histogram);
  stage.localSimtMaterializable = true;
  stage.localSuperblockMaterializable = true;
  stage.simtAnchorIndices = {0};
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageModeLegalityAnalysis().analyze(
          partition, /*maximumSuperblockFactor=*/4,
          /*scopeSuperblockMaterializable=*/true,
          /*maximumScopeSuperblockFactor=*/4))
    FAIL() << llvm::toString(std::move(error));

  const auto &result = partition.stages.front();
  EXPECT_TRUE(result.simdLegal);
  EXPECT_TRUE(result.simtLegal);
  EXPECT_EQ(result.legalSimtFactors, (std::vector<int64_t>{1, 2, 4}));
  EXPECT_TRUE(result.localSimtMaterializable);
  EXPECT_FALSE(result.localSuperblockMaterializable);
  EXPECT_EQ(result.localSimtFactors, (std::vector<int64_t>{1}));

  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));
  const auto &work = partition.stages.front().workload;
  ASSERT_EQ(work.operationCounts.count("histogram.i32"), 1u);
  EXPECT_DOUBLE_EQ(work.operationCounts.at("histogram.i32").at(2048), 1.0);

  HardwareProfile profile = hardwareProfile();
  StageOperationRate histogramRate;
  histogramRate.throughput = 2.0;
  histogramRate.startupCycles = 10.0;
  histogramRate.freeWorkUnits = 128.0;
  histogramRate.includesLocalScopeHandoff = true;
  profile.simt.operationRates["histogram.i32"] = histogramRate;
  auto costs = StageCostEvaluator().evaluate(partition, profile);
  if (!costs)
    FAIL() << llvm::toString(costs.takeError());
  auto local = llvm::find_if(
      costs->stages.front().implementations, [](const auto &cost) {
        return cost.implementation.mode == StageMode::SIMT &&
               cost.implementation.localScope;
      });
  ASSERT_NE(local, costs->stages.front().implementations.end());
  EXPECT_DOUBLE_EQ(local->resources.compute, (2048.0 - 128.0) / 2.0);
  EXPECT_DOUBLE_EQ(local->resources.operationCompletion, 10.0);
  EXPECT_TRUE(local->localScopeCompletionInclusive);
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
     DiagnosticOnlyAssertionChainsDoNotConsumeStageResources) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(%source: tensor<64xi32>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        scf.for %i = %c0 to %c4 step %c1 {
          %real = arith.addi %source, %source : tensor<64xi32>
          %live_cmp = arith.cmpi slt, %real, %source : tensor<64xi32>
          "test.consume"(%real, %live_cmp)
            : (tensor<64xi32>, tensor<64xi1>) -> ()
        }
        scf.for %i = %c0 to %c4 step %c1 {
          %real = arith.addi %source, %source : tensor<64xi32>
          %live_cmp = arith.cmpi slt, %real, %source : tensor<64xi32>
          "test.consume"(%real, %live_cmp)
            : (tensor<64xi32>, tensor<64xi1>) -> ()
          %wide = arith.extsi %real : tensor<64xi32> to tensor<64xi64>
          %scaled = arith.muli %wide, %wide : tensor<64xi64>
          %limit = arith.extsi %source : tensor<64xi32> to tensor<64xi64>
          %overflow = arith.cmpi slt, %scaled, %limit : tensor<64xi64>
          %condition = arith.andi %overflow, %overflow : tensor<64xi1>
          "tt.assert"(%condition) : (tensor<64xi1>) -> ()
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  llvm::SmallVector<mlir::Operation *, 2> loops;
  module->walk([&](mlir::scf::ForOp operation) {
    loops.push_back(operation.getOperation());
  });
  ASSERT_EQ(loops.size(), 2u);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  for (auto indexed : llvm::enumerate(loops)) {
    auto stage = logicalStage("assert_chain_" + std::to_string(indexed.index()),
                              StageCostModelKind::IndependentPipelinedLoop,
                              StageScheduleKind::IndependentPipelined, 4);
    stage.operations.push_back(indexed.value());
    partition.stages.push_back(std::move(stage));
  }
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));

  const StageWorkload &baseline = partition.stages[0].workload;
  const StageWorkload &withDiagnostic = partition.stages[1].workload;
  ASSERT_FALSE(baseline.predicateCounts.empty());
  EXPECT_DOUBLE_EQ(withDiagnostic.issueElements, baseline.issueElements);
  EXPECT_DOUBLE_EQ(withDiagnostic.scalarOperations, baseline.scalarOperations);
  EXPECT_EQ(withDiagnostic.predicateCounts, baseline.predicateCounts);
  EXPECT_EQ(withDiagnostic.operationCounts, baseline.operationCounts);
  EXPECT_EQ(withDiagnostic.tensorOperationWorkloads.size(),
            baseline.tensorOperationWorkloads.size());
}

TEST(SimdSimtCostModelTest,
     FlatAddressScaffoldingIsOwnedByMemoryButIndirectArithmeticRemains) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @flat(%base: !tt.ptr<f32>) {
        %range = "tt.make_range"() {start = 0 : i32, end = 64 : i32}
          : () -> tensor<64xi32>
        %base_vector = "tt.splat"(%base)
          : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
        %address = "tt.addptr"(%base_vector, %range)
          : (tensor<64x!tt.ptr<f32>>, tensor<64xi32>)
            -> tensor<64x!tt.ptr<f32>>
        %value = "tt.load"(%address)
          : (tensor<64x!tt.ptr<f32>>) -> tensor<64xf32>
        %view = "tt.reshape"(%value)
          : (tensor<64xf32>) -> tensor<8x8xf32>
        %flat_view = "tt.reshape"(%view)
          : (tensor<8x8xf32>) -> tensor<64xf32>
        "tt.store"(%address, %flat_view)
          : (tensor<64x!tt.ptr<f32>>, tensor<64xf32>) -> ()
        return
      }
      func.func @indirect(%base: tensor<64x!tt.ptr<f32>>,
                          %index_ptr: tensor<64x!tt.ptr<i32>>) {
        %index = "tt.load"(%index_ptr)
          : (tensor<64x!tt.ptr<i32>>) -> tensor<64xi32>
        %two = arith.constant dense<2> : tensor<64xi32>
        %scaled = arith.muli %index, %two : tensor<64xi32>
        %address = "tt.addptr"(%base, %scaled)
          : (tensor<64x!tt.ptr<f32>>, tensor<64xi32>)
            -> tensor<64x!tt.ptr<f32>>
        %value = "tt.load"(%address)
          : (tensor<64x!tt.ptr<f32>>) -> tensor<64xf32>
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);

  auto analyze = [&](llvm::StringRef name) -> StageWorkload {
    StagePartition partition;
    partition.operationOwnershipComplete = true;
    auto stage = logicalStage("address", StageCostModelKind::VectorIssue);
    auto function = *llvm::find_if(module->getOps<mlir::func::FuncOp>(),
                                   [&](mlir::func::FuncOp candidate) {
                                     return candidate.getName() == name;
                                   });
    for (mlir::Operation &operation : function.getBody().front())
      stage.operations.push_back(&operation);
    partition.stages.push_back(std::move(stage));
    if (llvm::Error error = StageWorkloadAnalysis().analyze(partition)) {
      ADD_FAILURE() << llvm::toString(std::move(error));
      return {};
    }
    return partition.stages.front().workload;
  };

  const StageWorkload flat = analyze("flat");
  ASSERT_EQ(flat.operationCounts.size(), 1u);
  auto folded = flat.operationCounts.find("simd.folded_flat_address");
  ASSERT_NE(folded, flat.operationCounts.end());
  EXPECT_FALSE(folded->second.empty());
  EXPECT_TRUE(flat.tensorOperationWorkloads.empty());
  EXPECT_DOUBLE_EQ(flat.loadBytes, 256.0);
  EXPECT_DOUBLE_EQ(flat.storeBytes, 256.0);

  const StageWorkload indirect = analyze("indirect");
  auto addressCompute =
      indirect.operationCounts.find("fp32.saturated.arith.muli");
  ASSERT_NE(addressCompute, indirect.operationCounts.end());
  EXPECT_FALSE(addressCompute->second.empty());
  EXPECT_DOUBLE_EQ(indirect.loadBytes, 512.0);
}

TEST(SimdSimtCostModelTest,
     RuntimeScalarArgumentResolvesLoopTripCountAndRecurrenceKind) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(
          %upper: index {ascend.costmodel.runtime_value = 8192 : i64},
          %pointer: i64) {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %zero = arith.constant dense<0.0> : tensor<64x32xf32>
        %result = scf.for %i = %c0 to %upper step %c32
            iter_args(%state = %zero) -> tensor<64x32xf32> {
          %loaded = "tt.load"(%pointer) : (i64) -> tensor<64x32xf32>
          %next = arith.addf %state, %loaded : tensor<64x32xf32>
          scf.yield %next : tensor<64x32xf32>
        }
        return
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

  const LogicalStage *loopStage = nullptr;
  for (const LogicalStage &stage : partition->stages)
    if (stage.features.hasLoopCarriedDataDependency)
      loopStage = &stage;
  ASSERT_NE(loopStage, nullptr);
  EXPECT_EQ(loopStage->costModelKind,
            StageCostModelKind::LoopCarriedRecurrence);
  EXPECT_EQ(loopStage->iterationCount, 256);
  EXPECT_EQ(loopStage->features.unknownTripCountLoopCount, 0);
  EXPECT_DOUBLE_EQ(loopStage->workload.loadBytes, 64.0 * 32.0 * 4.0);
}

TEST(SimdSimtCostModelTest,
     RuntimeGridStrideLoopUsesLogicalProgramCountForDynamicWork) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(
          %upper: index {ascend.costmodel.runtime_value = 128 : i64},
          %pointer: i64) {
        %program = "tt.get_program_id"() {axis = 0 : i32} : () -> index
        %programs = "tt.get_num_programs"() {axis = 0 : i32} : () -> index
        %zero = arith.constant dense<0.0> : tensor<512xf32>
        %result = scf.for %row = %program to %upper step %programs
            iter_args(%state = %zero) -> tensor<512xf32> {
          %loaded = "tt.load"(%pointer) : (i64) -> tensor<512xf32>
          %next = arith.addf %state, %loaded : tensor<512xf32>
          scf.yield %next : tensor<512xf32>
        }
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::ascend::SimtAnchorPlan anchors;
  StagePartitionerOptions options;
  options.logicalProgramCountHint = 32;
  auto partition = StagePartitioner().partition(*module, anchors, options);
  if (!partition)
    FAIL() << llvm::toString(partition.takeError());

  const LogicalStage *loopStage = nullptr;
  for (const LogicalStage &stage : partition->stages)
    if (stage.features.hasLoopCarriedDataDependency)
      loopStage = &stage;
  ASSERT_NE(loopStage, nullptr);
  EXPECT_EQ(loopStage->iterationCount, 4);
  EXPECT_EQ(loopStage->features.unknownTripCountLoopCount, 0);
  EXPECT_DOUBLE_EQ(loopStage->workload.loadBytes, 512.0 * 4.0);
}

TEST(SimdSimtCostModelTest, RuntimeOneTripIterArgIsNotClassifiedAsRecurrence) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(
          %upper: index {ascend.costmodel.runtime_value = 64 : i64},
          %pointer: i64) {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %zero = arith.constant dense<0.0> : tensor<64x64xf32>
        %result = scf.for %i = %c0 to %upper step %c64
            iter_args(%state = %zero) -> tensor<64x64xf32> {
          %loaded = "tt.load"(%pointer) : (i64) -> tensor<64x64xf32>
          %next = arith.addf %state, %loaded : tensor<64x64xf32>
          scf.yield %next : tensor<64x64xf32>
        }
        return
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

  const LogicalStage *loopStage = nullptr;
  for (const LogicalStage &stage : partition->stages)
    if (stage.features.hasLoop)
      loopStage = &stage;
  ASSERT_NE(loopStage, nullptr);
  EXPECT_FALSE(loopStage->features.hasLoopCarriedDataDependency);
  EXPECT_EQ(loopStage->costModelKind,
            StageCostModelKind::IndependentPipelinedLoop);
  EXPECT_EQ(loopStage->iterationCount, 1);
}

TEST(SimdSimtCostModelTest,
     OneTripZeroFilledMaskedAccumulatorUsesExactActiveReductionExtent) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @kernel(
          %upper: index {ascend.costmodel.runtime_value = 128 : i64},
          %pointer: tensor<8192x!tt.ptr<f16>>) {
        %c0 = arith.constant 0 : index
        %c8192 = arith.constant 8192 : index
        %zero16 = arith.constant dense<0.0> : tensor<8192xf16>
        %zero32 = arith.constant dense<0.0> : tensor<8192xf32>
        %range = "tt.make_range"() {start = 0 : i32, end = 8192 : i32}
          : () -> tensor<8192xi32>
        %upper_i32 = arith.index_cast %upper : index to i32
        %bound = "tt.splat"(%upper_i32) : (i32) -> tensor<8192xi32>
        %result = scf.for %offset = %c0 to %upper step %c8192
            iter_args(%state = %zero32) -> tensor<8192xf32> {
          %offset_i32 = arith.index_cast %offset : index to i32
          %offsets = "tt.splat"(%offset_i32) : (i32) -> tensor<8192xi32>
          %indices = arith.addi %offsets, %range : tensor<8192xi32>
          %mask = arith.cmpi slt, %indices, %bound : tensor<8192xi32>
          %loaded = "tt.load"(%pointer, %mask, %zero16)
            : (tensor<8192x!tt.ptr<f16>>, tensor<8192xi1>, tensor<8192xf16>)
              -> tensor<8192xf16>
          %wide = arith.extf %loaded : tensor<8192xf16> to tensor<8192xf32>
          %square = arith.mulf %wide, %wide : tensor<8192xf32>
          %next = arith.addf %state, %square : tensor<8192xf32>
          scf.yield %next : tensor<8192xf32>
        }
        %sum = "tt.reduce"(%result) <{axis = 0 : i32}> ({
        ^bb0(%lhs: f32, %rhs: f32):
          %combined = arith.addf %lhs, %rhs : f32
          "tt.reduce.return"(%combined) : (f32) -> ()
        }) : (tensor<8192xf32>) -> f32
        "test.consume"(%sum) : (f32) -> ()
        return
      }
    }
  )mlir",
                                                        &context);
  ASSERT_TRUE(module);
  mlir::Operation *reduction = nullptr;
  module->walk([&](mlir::Operation *operation) {
    if (operation->getName().getStringRef() == "tt.reduce")
      reduction = operation;
  });
  ASSERT_NE(reduction, nullptr);

  StagePartition partition;
  partition.operationOwnershipComplete = true;
  auto stage =
      logicalStage("masked_reduction", StageCostModelKind::RowwiseReduction);
  stage.operations = {reduction};
  partition.stages.push_back(std::move(stage));
  if (llvm::Error error = StageWorkloadAnalysis().analyze(partition))
    FAIL() << llvm::toString(std::move(error));

  const StageWorkload &work = partition.stages.front().workload;
  ASSERT_EQ(work.oneTripMaskedAccumulatorReductionExtents.size(), 1u);
  EXPECT_DOUBLE_EQ(
      work.oneTripMaskedAccumulatorReductionExtents.at({8192, 128}), 1.0);
  EXPECT_DOUBLE_EQ(work.shuffleLaneSteps, 128.0 * 7.0);
  EXPECT_DOUBLE_EQ(work.reductionOperations, 1.0);
  ASSERT_EQ(work.reductionCombineClassCounts.size(), 1u);
  EXPECT_DOUBLE_EQ(work.reductionCombineClassCounts.at("add"), 1.0);
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

TEST(SimdSimtCostModelTest,
     LoopCarriedRecurrenceUsesNestedScanDependencyLowerBound) {
  LogicalStage stage =
      logicalStage("recurrence_scan", StageCostModelKind::LoopCarriedRecurrence,
                   StageScheduleKind::LoopCarriedSerial, /*iterations=*/4);
  stage.features.hasLoop = true;
  stage.features.hasLoopCarriedDataDependency = true;
  stage.features.hasPrefixScan = true;
  stage.simdLegal = false;
  stage.workload.paysKernelSetup = false;
  stage.workload.operationCounts.clear();
  stage.workload.scalarOperations = 0.0;
  stage.workload.issueElements = 0.0;
  stage.workload.prefixScanCounts[{32, 8, 1, 1, false, "add"}] = 1.0;

  HardwareProfile baselineProfile = hardwareProfile();
  baselineProfile.simt.operationRates["f32.add"].throughput = 100.0;
  baselineProfile.simt.predicateOperationsPerCycle = 100.0;
  baselineProfile.simt.prefixScanDependencyFactor = 1.0;
  auto baseline = evaluateOneStage(stage, baselineProfile);
  if (!baseline)
    FAIL() << llvm::toString(baseline.takeError());

  HardwareProfile dependentProfile = baselineProfile;
  dependentProfile.simt.prefixScanDependencyFactor = 4.0;
  auto dependent = evaluateOneStage(stage, dependentProfile);
  if (!dependent)
    FAIL() << llvm::toString(dependent.takeError());

  const auto &baseCost = baseline->stages.front().implementations.front();
  const auto &dependentCost = dependent->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(baseCost.resources.scanShuffle,
                   dependentCost.resources.scanShuffle);
  EXPECT_GT(dependentCost.totalCycles, baseCost.totalCycles);
}

TEST(SimdSimtCostModelTest,
     LoadedIndexAtomicUsesAddressClassRateWithoutChangingContiguousAtomic) {
  LogicalStage stage = logicalStage("atomic", StageCostModelKind::AtomicMemory,
                                    StageScheduleKind::PartiallyDependent);
  stage.features.hasAtomicMemory = true;
  mlir::ascend::AtomicWorkload atomic;
  atomic.kind = "add";
  atomic.dataType = "i32";
  atomic.memorySemantic = "acq_rel";
  atomic.memoryScope = "gpu";
  atomic.logicalElements = 256.0;
  atomic.logicalOperationInstances = 1.0;
  atomic.addressDependsOnLoadedIndex = true;
  atomic.contentionUnknown = true;
  stage.workload.atomicWorkloads.push_back(atomic);

  HardwareProfile profile = hardwareProfile();
  profile.simd.atomicRates["default"] = {2.0, 0.0, 0.0, 1.0};
  profile.simd.atomicRates["loaded_index"] = {0.5, 7.0, 0.0, 1.0};
  profile.simd.atomicRates["loaded_index"]
      .singlePhysicalProgramLogicalElementsPerCycle = 1.0;
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &loadedIndex = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(loadedIndex.resources.atomic, 519.0);
  EXPECT_DOUBLE_EQ(loadedIndex.resources.singlePhysicalProgramAtomic, 263.0);
  EXPECT_TRUE(loadedIndex.resources.hasSinglePhysicalProgramAtomicRate);

  table->logicalProgramCountHint = 1;
  table->physicalCoreCountHint = 56;
  table->physicalAiCoreCountHint = 56;
  auto oneProgram = solveStageRoutes(*table, profile.scopeHandoff);
  if (!oneProgram)
    FAIL() << llvm::toString(oneProgram.takeError());
  EXPECT_TRUE(oneProgram->allSimd.singlePhysicalProgramAtomicRateApplied);
  EXPECT_DOUBLE_EQ(
      oneProgram->allSimd.singlePhysicalProgramAtomicAdjustmentCycles, -256.0);
  EXPECT_DOUBLE_EQ(oneProgram->allSimd.totalCycles,
                   loadedIndex.totalCycles - 256.0);

  table->logicalProgramCountHint = 2;
  auto twoPrograms = solveStageRoutes(*table, profile.scopeHandoff);
  if (!twoPrograms)
    FAIL() << llvm::toString(twoPrograms.takeError());
  EXPECT_FALSE(twoPrograms->allSimd.singlePhysicalProgramAtomicRateApplied);
  EXPECT_DOUBLE_EQ(twoPrograms->allSimd.totalCycles, loadedIndex.totalCycles);

  stage.workload.atomicWorkloads.front().addressDependsOnLoadedIndex = false;
  stage.workload.atomicWorkloads.front().provenContiguousRunWidth = 256.0;
  table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &contiguous = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(contiguous.resources.atomic, 128.0);
}

TEST(SimdSimtCostModelTest, ControlFlowUsesProfileRatesAndLaneActivity) {
  LogicalStage stage =
      logicalStage("priced_control", StageCostModelKind::ScalarControl,
                   StageScheduleKind::StraightLine, 2);
  stage.simdLegal = false;
  stage.features.hasLoop = true;
  stage.features.conditionalBranchCount = 3;
  stage.features.divergentBranchCount = 2;
  stage.features.loopBackedgeCount = 1;
  stage.features.synchronizationCount = 1;
  stage.features.activeLaneRatio = 0.5;
  stage.workload.operationCounts.clear();
  stage.workload.issueElements = 0.0;

  HardwareProfile profile = hardwareProfile();
  profile.simt.controlFlow = {2.0, 3.0, 10.0, 7.0};
  auto table = evaluateOneStage(stage, profile);
  if (!table)
    FAIL() << llvm::toString(table.takeError());
  const auto &cost = table->stages.front().implementations.front();
  EXPECT_DOUBLE_EQ(cost.resources.loopControl, 2.0);
  EXPECT_DOUBLE_EQ(cost.resources.branchControl, 9.0);
  EXPECT_DOUBLE_EQ(cost.resources.divergence, 10.0);
  EXPECT_DOUBLE_EQ(cost.resources.synchronization, 7.0);
  EXPECT_GT(cost.totalCycles, 0.0);
}
