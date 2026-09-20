//===- StageCostModels.cpp - Per-stage analytical models -----------------===//

#include "AscendModel/RouteModel/StageCostModels.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <optional>
#include <system_error>

using namespace mlir;
using namespace mlir::ascend;

namespace {

static double iterations(const LogicalStage &stage) {
  return static_cast<double>(std::max<int64_t>(1, stage.iterationCount));
}

static double selectWarpCurveValue(const std::map<int64_t, double> &curve,
                                   double fallback, int64_t warpCount) {
  if (curve.empty())
    return fallback;
  auto selected = curve.lower_bound(warpCount);
  if (selected == curve.end())
    return std::prev(selected)->second;
  return selected->second;
}

/// Resolve occupancy-sensitive SIMT rates for one concrete implementation.
/// The backend materializer retains `baseWarpCount` warps for every logical
/// program and launches F such programs in one SuperBlock, so the physical VF
/// contains baseWarpCount * F active warps.  Selecting all candidates from the
/// base-warp point makes F2/F4 consume an F1 profile even though the generated
/// kernel has a different launch shape.
static StageModeProfile
resolveModeProfile(const HardwareProfile &hardware,
                   const StageImplementation &implementation) {
  StageModeProfile resolved =
      implementation.mode == StageMode::SIMD ? hardware.simd : hardware.simt;
  if (implementation.mode == StageMode::SIMD)
    return resolved;

  const int64_t activeWarps =
      std::max<int64_t>(1, hardware.baseSimtWarpCount) *
      std::max<int64_t>(1, implementation.superblockFactor);
  resolved.activeWarpCount = activeWarps;
  resolved.setupCycles = selectWarpCurveValue(
      resolved.setupCyclesByWarpCount, resolved.setupCycles, activeWarps);
  for (auto &entry : resolved.operationRates)
    entry.second.throughput =
        selectWarpCurveValue(entry.second.throughputByWarpCount,
                             entry.second.throughput, activeWarps);
  auto predicate = resolved.operationRates.find("predicate.cmp");
  if (predicate != resolved.operationRates.end() &&
      predicate->second.throughput > 0.0)
    resolved.predicateOperationsPerCycle =
        predicate->second.throughput / std::max(1.0, predicate->second.factor);
  resolved.shuffleLanesPerCycle =
      selectWarpCurveValue(resolved.shuffleLanesPerCycleByWarpCount,
                           resolved.shuffleLanesPerCycle, activeWarps);
  resolved.loadWarpInstructionsPerCycle =
      selectWarpCurveValue(resolved.loadWarpInstructionsPerCycleByWarpCount,
                           resolved.loadWarpInstructionsPerCycle, activeWarps);
  resolved.storeWarpInstructionsPerCycle =
      selectWarpCurveValue(resolved.storeWarpInstructionsPerCycleByWarpCount,
                           resolved.storeWarpInstructionsPerCycle, activeWarps);
  // Factor-specific group curves are measured at one explicit base warp
  // count.  They describe a different physical launch shape when num_warps
  // changes and must not be presented as measured evidence for that shape.
  if (resolved.recurrenceGroupBaseWarpCount != hardware.baseSimtWarpCount)
    resolved.recurrenceReductionGroupCyclesByFactorAndLaneSteps.clear();
  if (resolved.continuousMemoryGroupBaseWarpCount !=
      hardware.baseSimtWarpCount) {
    resolved.continuousLoadGroupCyclesByFactorAndBytes.clear();
    resolved.continuousStoreGroupCyclesByFactorAndBytes.clear();
    resolved.continuousLoadGroupStartupCyclesByFactorAndBytes.clear();
    resolved.continuousStoreGroupStartupCyclesByFactorAndBytes.clear();
  }
  return resolved;
}

static std::vector<std::string>
collectSourceLocations(const LogicalStage &stage) {
  std::vector<std::string> result;
  llvm::StringSet<> seen;
  for (Operation *operation : stage.operations) {
    std::string location;
    llvm::raw_string_ostream stream(location);
    operation->getLoc().print(stream);
    stream.flush();
    if (location.empty() || !seen.insert(location).second)
      continue;
    result.push_back(std::move(location));
  }
  return result;
}

static double serialBody(const StageResourceCycles &resources) {
  const double execution = resources.scalar + resources.load + resources.store +
                           resources.compute + resources.predicate +
                           resources.shuffle + resources.dot;
  // Issue is a shared front-end throughput bound, not an extra instruction
  // stream.  Adding it to execution double-counts every instruction.
  return std::max(execution, resources.issue);
}

static bool permitsSimdOverlap(const LogicalStage &stage) {
  return stage.scheduleKind == StageScheduleKind::IndependentPipelined &&
         stage.features.permitsSimdRoofline();
}

static bool isContinuousMemoryStage(StageCostModelKind kind) {
  return kind == StageCostModelKind::ContinuousTileMemory ||
         kind == StageCostModelKind::ContinuousTileStore ||
         kind == StageCostModelKind::ContinuousShortLoad ||
         kind == StageCostModelKind::CachePolicyStore;
}

struct LayoutCoverage {
  double loadBytes = 0.0;
  double storeBytes = 0.0;
  double loadOperations = 0.0;
  double storeOperations = 0.0;
  double loadWarpInstructions = 0.0;
  double storeWarpInstructions = 0.0;
  std::map<std::pair<int64_t, int64_t>, double> loadShapes;
  std::map<std::pair<int64_t, int64_t>, double> storeShapes;
};

static std::optional<std::pair<int64_t, int64_t>>
accessShape(const StageMemoryAccess &access) {
  if (!access.hasStaticLayout() || access.elementBits % 8 != 0)
    return std::nullopt;
  int64_t elements = 1;
  for (int64_t extent : access.shape) {
    if (extent <= 0 || elements > std::numeric_limits<int64_t>::max() / extent)
      return std::nullopt;
    elements *= extent;
  }
  const int64_t elementBytes = access.elementBits / 8;
  if (elementBytes <= 0 ||
      elements > std::numeric_limits<int64_t>::max() / elementBytes)
    return std::nullopt;
  return std::make_pair(elements * elementBytes, elements);
}

/// Only consume exact group evidence. Logical rank/row count does not prove
/// the selected DMA or SIMT lane mapping; do not invent a setup per row or
/// a physical LDG count by rounding TTIR elements. Unsupported layouts retain
/// an explicitly uncalibrated analytical estimate.
static std::optional<double> mapLayoutAccess(const StageMemoryAccess &access,
                                             const StageModeProfile &profile,
                                             bool simd, int64_t factor) {
  if (!accessShape(access) ||
      access.layoutClass() == StageMemoryAccess::LayoutClass::Unknown)
    return std::nullopt;
  const auto &profileCosts =
      access.store ? profile.storeLayoutCosts : profile.loadLayoutCosts;
  const std::string key = access.layoutKey() + ":warps=" +
                          std::to_string(simd ? 0 : profile.activeWarpCount) +
                          ":factor=" + std::to_string(simd ? 1 : factor);
  auto measured = profileCosts.find(key);
  return measured == profileCosts.end()
             ? std::nullopt
             : measured->second.evaluate(access.count);
}

/// Coverage of the logical tile is a lower bound on the sum of per-warp
/// coverage: warps may revisit a line, never merge two disjoint lines into one.
/// Only use the measured 8-byte/lane, full-warp domain; do not infer issued
/// instructions for short, broadcast, indirect or unknown-layout accesses.
/// Packed-layout issue floors did not transfer to the independent grouped
/// probe, so retain their previous estimate instead of claiming validation.
static double storeLayoutService(const StageWorkload &work,
                                 const StageModeProfile &profile,
                                 int64_t factor) {
  const int64_t unit = profile.storeFootprintUnitBytes;
  auto floor = profile.storeIssueFloorByWarpCount.find(profile.activeWarpCount);
  if (unit <= 0 || factor <= 0 ||
      floor == profile.storeIssueFloorByWarpCount.end())
    return 0.0;
  double groupService = 0.0;
  for (const auto &access : work.memoryAccesses) {
    auto shape = accessShape(access);
    if (!access.store || !shape || access.effectiveRank() > 2 ||
        access.isFlatContiguous() || !access.hasUnitInnerStride() ||
        static_cast<double>(shape->first) * factor !=
            static_cast<double>(profile.activeWarpCount) * 32 * 8)
      continue;
    int64_t units = 0;
    if (access.rowStrideBytes() >= access.rowBytes() &&
        access.rowStrideBytes() % unit == 0)
      units = access.rowCount() * llvm::divideCeil(access.rowBytes(), unit);
    if (units > 0)
      groupService +=
          access.count *
          std::max(floor->second, static_cast<double>(units) * factor *
                                      profile.storeCyclesPerFootprintUnit);
  }
  return groupService / factor;
}

/// Count issue instructions without collapsing distinct short operations into
/// one vector/warp. StageWorkload::issueElements remains a useful aggregate in
/// the report, but ceil(sum(elements) / width) is not a valid instruction
/// count when those elements belong to different operations.
static double countIssueInstructions(const StageWorkload &work,
                                     int64_t issueWidth, bool simd) {
  if (issueWidth <= 0)
    return 0.0;
  const double width = static_cast<double>(issueWidth);
  double instructions = work.scalarOperations;
  for (const auto &entry : work.operationCounts)
    for (const auto &[elements, count] : entry.second)
      instructions += count * std::ceil(static_cast<double>(elements) / width);
  for (const auto &[elements, count] : work.predicateCounts)
    instructions += count * std::ceil(static_cast<double>(elements) / width);

  // These are normalized element-work units, not a count of final LDG/STG.
  // Without a lowering contract, row geometry cannot change that unit.
  auto countMemory = [&](const auto &shapes, double &coveredWarpInstructions) {
    for (const auto &[shape, count] : shapes) {
      const int64_t warpInstructions =
          llvm::divideCeil(shape.second, int64_t(32));
      instructions += count * std::ceil(32.0 * warpInstructions / width);
      coveredWarpInstructions += count * warpInstructions;
    }
  };
  double coveredLoadWarpInstructions = 0.0;
  double coveredStoreWarpInstructions = 0.0;
  countMemory(work.continuousLoadCounts, coveredLoadWarpInstructions);
  countMemory(work.indirectLoadCounts, coveredLoadWarpInstructions);
  countMemory(work.continuousStoreCounts, coveredStoreWarpInstructions);
  instructions += std::ceil(
      32.0 *
      std::max(0.0, work.loadWarpInstructions - coveredLoadWarpInstructions) /
      width);
  instructions += std::ceil(
      32.0 *
      std::max(0.0, work.storeWarpInstructions - coveredStoreWarpInstructions) /
      width);
  instructions +=
      work.loopBackedges + work.conditionalBranches + work.synchronizations;
  return instructions;
}

/// Interpolate an isolated per-operation cost curve. Below the first measured
/// point, scale from the origin; above the last point, retain the last measured
/// local slope. This keeps workload size and dynamic occurrence count separate.
static double interpolateOperationCost(const std::map<int64_t, double> &curve,
                                       int64_t workSize) {
  if (curve.empty() || workSize <= 0)
    return 0.0;
  auto upper = curve.lower_bound(workSize);
  if (upper == curve.begin())
    return upper->second * static_cast<double>(workSize) / upper->first;
  if (upper == curve.end()) {
    auto last = std::prev(curve.end());
    if (last == curve.begin())
      return last->second * static_cast<double>(workSize) / last->first;
    auto previous = std::prev(last);
    const double slope = (last->second - previous->second) /
                         static_cast<double>(last->first - previous->first);
    return std::max(0.0, last->second + (workSize - last->first) * slope);
  }
  auto lower = std::prev(upper);
  const double ratio = static_cast<double>(workSize - lower->first) /
                       static_cast<double>(upper->first - lower->first);
  return lower->second + ratio * (upper->second - lower->second);
}

/// Continuous-memory curves are only evidence inside their measured byte
/// interval.  Extrapolating a cache/transaction curve beyond that interval
/// silently turns a local fit into an invented bandwidth model, so unmatched
/// shapes must retain the analytical fallback.
static std::optional<double>
interpolateBoundedOperationCost(const std::map<int64_t, double> &curve,
                                int64_t workSize) {
  if (curve.empty() || workSize < curve.begin()->first ||
      workSize > curve.rbegin()->first)
    return std::nullopt;
  return interpolateOperationCost(curve, workSize);
}

/// A one-time envelope is not an amount of work and must not scale from the
/// origin or continue with the last local slope.  Interpolate only between
/// measured shapes and retain the nearest endpoint outside that interval.
static double interpolateEnvelopeCost(const std::map<int64_t, double> &curve,
                                      int64_t workSize) {
  if (curve.empty() || workSize <= 0)
    return 0.0;
  auto upper = curve.lower_bound(workSize);
  if (upper == curve.begin())
    return upper->second;
  if (upper == curve.end())
    return curve.rbegin()->second;
  auto lower = std::prev(upper);
  const double ratio = static_cast<double>(workSize - lower->first) /
                       static_cast<double>(upper->first - lower->first);
  return lower->second + ratio * (upper->second - lower->second);
}

static bool extrapolatesIndirectEvidence(const LogicalStage &stage,
                                         const StageModeProfile &profile) {
  if (!stage.features.hasIndirectMemory ||
      profile.indirectLoadSystemCyclesByWarpInstructions.empty())
    return false;
  const int64_t first =
      profile.indirectLoadSystemCyclesByWarpInstructions.begin()->first;
  const int64_t last =
      profile.indirectLoadSystemCyclesByWarpInstructions.rbegin()->first;
  double operationCount = 0.0;
  for (const auto &[shape, count] : stage.workload.indirectLoadCounts) {
    const int64_t warpInstructions =
        llvm::divideCeil(shape.second, int64_t(32));
    if (warpInstructions < first || warpInstructions > last)
      return true;
    operationCount += count;
  }
  // The checked-in count sweep is fitted on 1, 2 and 4 operations.  Count=8
  // is a held-out capacity check, not a calibrated linear domain.
  return operationCount > 4.0;
}

static std::optional<StageResourceCycles>
mapWorkload(const LogicalStage &stage, const StageModeProfile &profile,
            StageMode mode, int64_t superblockFactor) {
  StageResourceCycles resources;
  const StageWorkload &work = stage.workload;
  const bool simd = mode == StageMode::SIMD;
  resources.setup = work.paysKernelSetup ? profile.setupCycles : 0.0;
  for (const auto &[name, counts] : work.operationCounts) {
    auto rate = profile.operationRates.find(name);
    if (rate == profile.operationRates.end() || rate->second.throughput <= 0.0)
      continue;
    for (const auto &[elements, count] : counts) {
      const double instructions =
          simd ? std::ceil(static_cast<double>(elements) / profile.vectorWidth)
               : static_cast<double>(elements);
      resources.compute +=
          count * instructions / rate->second.throughput * rate->second.factor;
    }
  }
  double prefixScanIssuedInstructions = 0.0;
  for (const auto &[shape, count] : work.prefixScanCounts) {
    const int64_t extent = shape.first;
    const int64_t independentSequences = shape.second;
    if (extent <= 1 || independentSequences <= 0 || count <= 0.0)
      continue;
    auto addRate = profile.operationRates.find("f32.add");
    if (addRate == profile.operationRates.end() ||
        addRate->second.throughput <= 0.0)
      continue;
    if (simd) {
      // One SIMD vector lane follows one independent sequence. Every vector
      // group executes an R-1 dependent add chain.
      const double groups = std::ceil(
          static_cast<double>(independentSequences) / profile.vectorWidth);
      const double vectorAdds =
          count * groups * static_cast<double>(extent - 1);
      resources.compute +=
          vectorAdds / addRate->second.throughput * addRate->second.factor;
      prefixScanIssuedInstructions += vectorAdds;
      resources.prefixScanCriticalPath += count *
                                          static_cast<double>(extent - 1) *
                                          profile.prefixScanStepLatencyCycles;
      continue;
    }

    // One SIMT warp follows one independent sequence. Each log2(R) level
    // issues one warp shuffle. Only lanes [offset, R) add the shuffled value,
    // while all R lanes evaluate the level predicate.
    int64_t depth = 0;
    int64_t activeAddsPerSequence = 0;
    for (int64_t offset = 1; offset < extent;) {
      ++depth;
      activeAddsPerSequence += extent - offset;
      if (offset > (extent - 1) / 2)
        break;
      offset <<= 1;
    }
    const double warpSteps = count * independentSequences * depth;
    const double activeAdds =
        count * independentSequences * activeAddsPerSequence;
    const double predicateOps = count * independentSequences * extent * depth;
    resources.compute +=
        activeAdds / addRate->second.throughput * addRate->second.factor;
    resources.predicate += predicateOps / profile.predicateOperationsPerCycle;
    resources.shuffle +=
        warpSteps * profile.issueWidth / profile.shuffleLanesPerCycle;
    // Preserve operation boundaries at every scan level.  A masked add and
    // its predicate are each one issued warp instruction even when fewer
    // than 32 lanes are active; lanes from different levels or independent
    // sequences cannot be packed into one instruction by taking
    // ceil(sum(active lanes) / 32).  Each level therefore issues one shuffle,
    // one predicate and one add per sequence.
    constexpr double instructionsPerWarpStep = 3.0;
    prefixScanIssuedInstructions += instructionsPerWarpStep * warpSteps;
    resources.prefixScanCriticalPath +=
        count * depth * profile.prefixScanStepLatencyCycles;
  }
  // Unclassified scalar operations have no execution-throughput evidence.
  // They still contribute distinct instructions to the measured issue floor
  // below, but assigning a second arbitrary scalar rate would fabricate a
  // mode-dependent cost and double-count the same instruction stream.
  if (stage.features.hasIndirectMemory) {
    const double loads = work.loadWarpInstructions;
    const double stores = work.storeWarpInstructions;
    double curvedLoadBytes = 0.0;
    double curvedLoadWarpInstructions = 0.0;
    double loadStartup = 0.0;
    for (const auto &[shape, count] : work.indirectLoadCounts) {
      const int64_t bytes = shape.first;
      const int64_t elements = shape.second;
      const int64_t warpInstructions = llvm::divideCeil(elements, int64_t(32));
      // The checked-in loaded-index curve was measured with one FP16 value per
      // active lane. Determine width before rounding to warps: a two-element
      // FP16 load is still FP16, not an unknown 4-byte-per-warp dtype.
      if (bytes != elements * 2)
        continue;
      resources.load +=
          count * interpolateOperationCost(
                      profile.indirectLoadSystemCyclesByWarpInstructions,
                      warpInstructions);
      loadStartup = std::max(
          loadStartup,
          interpolateEnvelopeCost(
              profile.indirectLoadStartupSystemCyclesByWarpInstructions,
              warpInstructions));
      curvedLoadBytes += count * bytes;
      curvedLoadWarpInstructions += count * warpInstructions;
    }
    resources.load += loadStartup;
    // The measured curve describes loaded-index loads only.  A generic
    // transaction-rate fallback hid missing StageWorkload classification and
    // assigned arbitrary latency to stores/atomics.  Reject that candidate
    // until a matching measured semantic curve exists.
    constexpr double coverageTolerance = 1.0e-9;
    if (profile.indirectLoadSystemCyclesByWarpInstructions.empty() ||
        profile.indirectLoadStartupSystemCyclesByWarpInstructions.empty() ||
        std::abs(work.loadBytes - curvedLoadBytes) > coverageTolerance ||
        std::abs(loads - curvedLoadWarpInstructions) > coverageTolerance ||
        stores > coverageTolerance)
      return std::nullopt;
  } else {
    LayoutCoverage layout;
    // A count curve describes one uninterrupted sequence. Do not merge two
    // matching layouts across a different access, or apply its overlap across
    // explicit synchronization/control flow. Lowering-only barriers require
    // separate evidence before a profile entry can be enabled.
    std::vector<StageMemoryAccess> accessGroups;
    for (const StageMemoryAccess &access : work.memoryAccesses) {
      if (!accessGroups.empty() && accessGroups.back().store == access.store &&
          accessGroups.back().layoutKey() == access.layoutKey())
        accessGroups.back().count += access.count;
      else
        accessGroups.push_back(access);
    }
    for (const auto &access : accessGroups) {
      if (!isContinuousMemoryStage(stage.costModelKind) ||
          stage.features.synchronizationCount != 0 || stage.features.hasLoop ||
          stage.features.conditionalBranchCount != 0)
        break;
      auto groupCost = mapLayoutAccess(access, profile, simd, superblockFactor);
      const auto shape = accessShape(access);
      if (!groupCost || !shape)
        continue;
      // Subtract the covered work in its original normalized unit. The new
      // profile's group envelope is not itself an instruction count.
      const double normalizedWarps =
          llvm::divideCeil(shape->second, int64_t(32));
      if (access.store) {
        resources.continuousStoreGroup += *groupCost;
        layout.storeBytes += access.count * shape->first;
        layout.storeOperations += access.count;
        layout.storeWarpInstructions += access.count * normalizedWarps;
        layout.storeShapes[*shape] += access.count;
      } else {
        resources.continuousLoadGroup += *groupCost;
        layout.loadBytes += access.count * shape->first;
        layout.loadOperations += access.count;
        layout.loadWarpInstructions += access.count * normalizedWarps;
        layout.loadShapes[*shape] += access.count;
      }
    }
    double curvedLoadOperations = 0.0;
    double curvedStoreOperations = 0.0;
    double curvedLoadBytes = 0.0;
    double curvedStoreBytes = 0.0;
    double curvedLoadWarpInstructions = 0.0;
    double curvedStoreWarpInstructions = 0.0;
    double curvedLoadStartup = 0.0;
    double curvedStoreStartup = 0.0;
    auto loadCurve = profile.continuousLoadGroupCyclesByFactorAndBytes.find(
        superblockFactor);
    auto storeCurve = profile.continuousStoreGroupCyclesByFactorAndBytes.find(
        superblockFactor);
    auto loadStartupCurve =
        profile.continuousLoadGroupStartupCyclesByFactorAndBytes.find(
            superblockFactor);
    auto storeStartupCurve =
        profile.continuousStoreGroupStartupCyclesByFactorAndBytes.find(
            superblockFactor);
    // These are complete contiguous-memory group envelopes. Other Stage
    // formulas do not consume continuousLoad/StoreGroup, so removing covered
    // bytes from their analytical resources would silently drop that work.
    // Keep their memory on the analytical path until a compatible composite
    // envelope is modeled explicitly.
    if (isContinuousMemoryStage(stage.costModelKind) &&
        loadCurve != profile.continuousLoadGroupCyclesByFactorAndBytes.end()) {
      for (const auto &[shape, count] : work.continuousLoadCounts) {
        const auto [bytes, elements] = shape;
        const auto covered = layout.loadShapes.find(shape);
        const double availableCount =
            std::max(0.0, count - (covered == layout.loadShapes.end()
                                       ? 0.0
                                       : covered->second));
        if (availableCount <= 0.0)
          continue;
        const int64_t warpInstructions =
            llvm::divideCeil(elements, int64_t(32));
        // The checked-in curve is an FP32 probe. Keep dtype/element-count
        // sensitivity explicit before rounding short operations to warps.
        if (bytes != elements * 4)
          continue;
        auto operationCost =
            interpolateBoundedOperationCost(loadCurve->second, bytes);
        if (!operationCost)
          continue;
        resources.continuousLoadGroup += availableCount * *operationCost;
        if (loadStartupCurve !=
            profile.continuousLoadGroupStartupCyclesByFactorAndBytes.end()) {
          auto startupCost =
              interpolateBoundedOperationCost(loadStartupCurve->second, bytes);
          if (startupCost)
            curvedLoadStartup = std::max(curvedLoadStartup, *startupCost);
        }
        curvedLoadOperations += availableCount;
        curvedLoadBytes += availableCount * bytes;
        curvedLoadWarpInstructions += availableCount * warpInstructions;
      }
    }
    if (isContinuousMemoryStage(stage.costModelKind) &&
        storeCurve !=
            profile.continuousStoreGroupCyclesByFactorAndBytes.end()) {
      for (const auto &[shape, count] : work.continuousStoreCounts) {
        const auto [bytes, elements] = shape;
        const auto covered = layout.storeShapes.find(shape);
        const double availableCount =
            std::max(0.0, count - (covered == layout.storeShapes.end()
                                       ? 0.0
                                       : covered->second));
        if (availableCount <= 0.0)
          continue;
        const int64_t warpInstructions =
            llvm::divideCeil(elements, int64_t(32));
        if (bytes != elements * 4)
          continue;
        auto operationCost =
            interpolateBoundedOperationCost(storeCurve->second, bytes);
        if (!operationCost)
          continue;
        resources.continuousStoreGroup += availableCount * *operationCost;
        if (storeStartupCurve !=
            profile.continuousStoreGroupStartupCyclesByFactorAndBytes.end()) {
          auto startupCost =
              interpolateBoundedOperationCost(storeStartupCurve->second, bytes);
          if (startupCost)
            curvedStoreStartup = std::max(curvedStoreStartup, *startupCost);
        }
        curvedStoreOperations += availableCount;
        curvedStoreBytes += availableCount * bytes;
        curvedStoreWarpInstructions += availableCount * warpInstructions;
      }
    }
    if (curvedLoadOperations > 0.0)
      resources.continuousLoadGroup += curvedLoadStartup;
    if (curvedStoreOperations > 0.0)
      resources.continuousStoreGroup += curvedStoreStartup;
    if (simd) {
      resources.load +=
          std::max(0.0, work.loadOperations - layout.loadOperations -
                            curvedLoadOperations) *
              profile.loadOperationSetupCycles +
          std::max(0.0, work.loadBytes - layout.loadBytes - curvedLoadBytes) /
              profile.loadBytesPerCycle;
      resources.store +=
          std::max(0.0, work.storeOperations - layout.storeOperations -
                            curvedStoreOperations) *
              profile.storeOperationSetupCycles +
          std::max(0.0,
                   work.storeBytes - layout.storeBytes - curvedStoreBytes) /
              profile.storeBytesPerCycle;
    } else {
      resources.load += std::max(0.0, work.loadWarpInstructions -
                                          layout.loadWarpInstructions -
                                          curvedLoadWarpInstructions) /
                        profile.loadWarpInstructionsPerCycle;
      resources.store += std::max(0.0, work.storeWarpInstructions -
                                           layout.storeWarpInstructions -
                                           curvedStoreWarpInstructions) /
                         profile.storeWarpInstructionsPerCycle;
    }
  }
  for (const auto &[elements, count] : work.predicateCounts) {
    const double instructions =
        simd ? std::ceil(static_cast<double>(elements) / profile.vectorWidth)
             : static_cast<double>(elements);
    resources.predicate +=
        count * instructions / profile.predicateOperationsPerCycle;
  }
  // Reduction shuffles are additional to the prefix-scan shuffles accumulated
  // above.  Assignment here used to erase the complete SIMT scan cost because
  // tt.scan intentionally does not populate shuffleLaneSteps.
  resources.shuffle += work.shuffleLaneSteps / profile.shuffleLanesPerCycle;
  if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence &&
      stage.features.hasReduction && work.shuffleLaneSteps > 0.0) {
    auto factorCurve =
        profile.recurrenceReductionGroupCyclesByFactorAndLaneSteps.find(
            superblockFactor);
    if (factorCurve !=
        profile.recurrenceReductionGroupCyclesByFactorAndLaneSteps.end())
      resources.recurrenceGroupCriticalPath = interpolateOperationCost(
          factorCurve->second,
          static_cast<int64_t>(std::ceil(work.shuffleLaneSteps)));
  }
  if (work.dotFlops > 0.0) {
    // A Stage can own multiple tiny dots. Their setup/dependency cost is not
    // amortized merely because Stage partitioning grouped them together.
    // This is the cost of one Stage body on one logical-program iteration;
    // it is deliberately not a CV-pipeline overlap model.  Cross-Stage
    // overlap must be proven by the lowered schedule before route composition
    // can replace a sum with a critical-path bound (see the calibration gate).
    resources.dot = work.dotOperations * profile.dotSetupCycles +
                    work.dotFlops / profile.dotFlopsPerCycle;
  }
  resources.issue = countIssueInstructions(work, profile.issueWidth, simd) /
                    profile.issueOperationsPerCycle;
  resources.prefixScanIssue =
      prefixScanIssuedInstructions / profile.issueOperationsPerCycle;
  resources.issue += resources.prefixScanIssue;
  if (stage.features.hasLoopCarriedDataDependency)
    resources.criticalPath = resources.scalar + resources.compute +
                             resources.predicate + resources.shuffle +
                             resources.dot;
  else if (stage.features.hasReduction)
    resources.criticalPath =
        resources.compute + resources.predicate + resources.shuffle;
  if (!simd && isContinuousMemoryStage(stage.costModelKind) &&
      resources.continuousStoreGroup == 0.0) {
    resources.storeLayoutService =
        storeLayoutService(work, profile, superblockFactor);
    // Both terms price the SAME transfers. Taking their max strengthens the
    // old flat-throughput bound without charging those bytes a second time.
    resources.store = std::max(resources.store, resources.storeLayoutService);
  }
  return resources;
}

static double estimateStage(const LogicalStage &stage,
                            const HardwareProfile &profile, StageMode mode,
                            const StageResourceCycles &resources);

static double applySuperBlock(const LogicalStage &stage,
                              const StageResourceCycles &resources,
                              const StageImplementation &implementation,
                              const HardwareProfile &profile,
                              double stageCycles) {
  if (implementation.mode != StageMode::SIMT ||
      implementation.superblockFactor == 1)
    return stageCycles;

  const double factor = static_cast<double>(implementation.superblockFactor);
  // SuperBlock creates `factor` independent logical-program groups on one
  // physical core.  It can hide latency across those groups, but it cannot
  // divide dependent arithmetic. Loop/branch/synchronization instructions are
  // already part of the shared issue lower bound; their independent latency is
  // deliberately not invented from an unidentifiable additive tuple.
  const double fixed = resources.setup;
  const double issueFloor =
      fixed + factor * iterations(stage) * resources.issue;
  // A recurrence is serial inside one logical program.  SuperBlock contributes
  // F independent logical programs to the same physical program, allowing the
  // scheduler to cover one program's dependency stalls with another program.
  // Normalize the critical-path portion per logical program, but retain the
  // aggregate issue floor: a larger factor cannot create additional issue
  // bandwidth.
  // This applies equally to whole-kernel and scope-local SuperBlock because
  // both materializers batch complete logical programs around the Stage.
  if (stage.costModelKind == StageCostModelKind::LoopCarriedRecurrence) {
    const double recurrenceBody = std::max(0.0, stageCycles - fixed);
    return std::max(issueFloor, fixed + recurrenceBody);
  }
  if ((stage.costModelKind == StageCostModelKind::ContinuousTileMemory ||
       stage.costModelKind == StageCostModelKind::ContinuousTileStore ||
       stage.costModelKind == StageCostModelKind::ContinuousShortLoad ||
       stage.costModelKind == StageCostModelKind::CachePolicyStore) &&
      (resources.continuousLoadGroup > 0.0 ||
       resources.continuousStoreGroup > 0.0)) {
    // Measured envelopes already cover F programs. Uncovered transfers and
    // other analytical resources still describe one logical program. Scale
    // only that work, then recompute the competing bounds; scaling the final
    // max would either multiply the envelope again or omit F-1 tails of work.
    auto grouped = resources;
    for (double *term :
         {&grouped.scalar, &grouped.load, &grouped.store, &grouped.compute,
          &grouped.predicate, &grouped.shuffle, &grouped.dot, &grouped.issue})
      *term *= factor;
    return estimateStage(stage, profile, implementation.mode, grouped);
  }
  const double body = std::max(0.0, stageCycles - fixed);
  // Resource rates describe aggregate throughput of the selected VF (already
  // resolved at baseWarps * F). The group issues F programs' work at that
  // rate. Dividing load/store/shuffle by F again double-counts the occupancy
  // benefit. These throughput-derived terms are not dependency latencies.
  return std::max(issueFloor, fixed + factor * body);
}

static double estimateStage(const LogicalStage &stage,
                            const HardwareProfile &profile, StageMode mode,
                            const StageResourceCycles &r) {
  const double count = iterations(stage);
  const double serial = r.setup + count * serialBody(r);
  switch (stage.costModelKind) {
  case StageCostModelKind::VectorIssue:
    // mapWorkload counts each tensor operation separately: SIMD uses
    // ceil(elements / vectorWidth) / operation throughput, SIMT uses lane
    // work / lane throughput. Issue is a competing bound, not scalar pricing
    // for the vector computation, and no cross-Stage overlap is assumed.
    return serial;
  case StageCostModelKind::AutoBlockifyDispatch:
  case StageCostModelKind::AutoBlockifyLoop: {
    const double dispatchCount =
        stage.costModelKind == StageCostModelKind::AutoBlockifyLoop ? count
                                                                    : 1.0;
    return r.setup + dispatchCount * std::max(r.scalar, r.issue);
  }
  case StageCostModelKind::ContinuousTileMemory:
  case StageCostModelKind::ContinuousTileStore:
  case StageCostModelKind::ContinuousShortLoad:
  case StageCostModelKind::CachePolicyStore:
    if (r.continuousLoadGroup > 0.0 || r.continuousStoreGroup > 0.0) {
      const double measuredMemory =
          r.continuousLoadGroup + r.continuousStoreGroup;
      const double otherExecution = r.scalar + r.load + r.store + r.compute +
                                    r.predicate + r.shuffle + r.dot;
      // Covered and uncovered transfers are disjoint workloads. Neither may
      // hide the other merely because only one shape has a measured curve.
      // The curve already contains its address/issue/sink envelope, so retain
      // the competing bound for non-memory operations instead of adding it.
      return r.setup + count * std::max({measuredMemory + r.load + r.store,
                                         otherExecution, r.issue});
    }
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup + count * std::max(r.scalar + r.predicate +
                                            std::max(r.load, r.store),
                                        r.issue);
    return serial;
  case StageCostModelKind::IndependentPipelinedLoop:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup +
             count * std::max({r.load, r.store, r.compute + r.dot + r.shuffle,
                               r.scalar + r.predicate, r.issue});
    return serial;
  case StageCostModelKind::LoopCarriedRecurrence: {
    const double analyticalCritical =
        r.criticalPath > 0.0
            ? std::max(r.criticalPath + r.load + r.store, r.issue)
            : serialBody(r);
    // The calibrated curve is a lower bound for the complete reduction
    // recurrence iteration of one physical group.  It already includes the
    // candidate SuperBlock factor, so neither this Stage formula nor route
    // composition may multiply it by F again.
    const double critical =
        std::max(analyticalCritical, r.recurrenceGroupCriticalPath);
    // Workload is already the aggregate work of all owned loops divided by
    // count. A loop count is not proof of concurrent execution and cannot
    // shorten a loop-carried chain. Any overlap needs a proven schedule.
    return r.setup + count * critical;
  }
  case StageCostModelKind::RowwiseReduction:
    return r.setup +
           count *
               std::max(r.scalar + r.load + r.store + r.criticalPath, r.issue);
  case StageCostModelKind::PrefixScan: {
    const double scanExecution = r.compute + r.predicate + r.shuffle;
    const double scanCritical =
        std::max(scanExecution, r.prefixScanCriticalPath);
    const double execution = r.scalar + r.load + r.store + scanCritical;
    return r.setup + count * std::max(execution, r.issue);
  }
  case StageCostModelKind::CubeRoofline:
  case StageCostModelKind::TinyCubeRoofline:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup + count * std::max(r.scalar + r.predicate + r.shuffle +
                                            std::max({r.load, r.compute + r.dot,
                                                      r.store}),
                                        r.issue);
    return serial;
  case StageCostModelKind::ConversionPack:
    if (mode == StageMode::SIMD && permitsSimdOverlap(stage))
      return r.setup +
             count * std::max(r.predicate + std::max({r.scalar + r.compute,
                                                      r.load, r.store}),
                              r.issue);
    return serial;
  default:
    if (mode == StageMode::SIMD)
      return r.setup +
             count *
                 (std::max({r.load, r.store, r.atomic,
                            r.compute + r.dot + r.shuffle,
                            r.scalar + r.predicate + controlBody(r), r.issue}) +
                  r.spill);
    return serial;
  }
}
static bool isDeclaredLegal(const LogicalStage &stage,
                            const StageImplementation &implementation) {
  if (!implementation.isValid())
    return false;
  if (implementation.mode == StageMode::SIMD)
    return stage.simdLegal && implementation.superblockFactor == 1 &&
           !implementation.localScope;
  if (!stage.simtLegal)
    return false;
  if (implementation.localScope)
    return stage.localSimtMaterializable &&
           llvm::is_contained(stage.localSimtFactors,
                              implementation.superblockFactor);
  return llvm::is_contained(stage.legalSimtFactors,
                            implementation.superblockFactor);
}

} // namespace

llvm::StringRef mlir::ascend::stringifyStageCostModel(StageCostModelKind kind) {
  switch (kind) {
  case StageCostModelKind::AutoBlockifyDispatch:
    return "auto_blockify_dispatch";
  case StageCostModelKind::AutoBlockifyLoop:
    return "auto_blockify_loop";
  case StageCostModelKind::ScalarIssue:
    return "scalar_issue";
  case StageCostModelKind::ScalarControl:
    return "scalar_control";
  case StageCostModelKind::ScalarMath:
    return "scalar_math";
  case StageCostModelKind::VectorIssue:
    return "vector_issue";
  case StageCostModelKind::IndexGeneration:
    return "index_generation";
  case StageCostModelKind::PredicateMask:
    return "predicate_mask";
  case StageCostModelKind::LoopPredicate:
    return "loop_predicate";
  case StageCostModelKind::ContinuousTileMemory:
    return "continuous_tile_memory";
  case StageCostModelKind::ContinuousTileStore:
    return "continuous_tile_store";
  case StageCostModelKind::ContinuousShortLoad:
    return "continuous_short_load";
  case StageCostModelKind::CachePolicyStore:
    return "cache_policy_store";
  case StageCostModelKind::IndirectScalarMemory:
    return "indirect_scalar_memory";
  case StageCostModelKind::IndirectGatherMemory:
    return "indirect_gather_memory";
  case StageCostModelKind::AtomicMemory:
    return "atomic_memory";
  case StageCostModelKind::IndependentPipelinedLoop:
    return "independent_pipelined_loop";
  case StageCostModelKind::LoopCarriedRecurrence:
    return "loop_carried_recurrence";
  case StageCostModelKind::RowwiseReduction:
    return "rowwise_reduction";
  case StageCostModelKind::PrefixScan:
    return "prefix_scan";
  case StageCostModelKind::CubeRoofline:
    return "cube_roofline";
  case StageCostModelKind::TinyCubeRoofline:
    return "tiny_cube_roofline";
  case StageCostModelKind::ConversionPack:
    return "conversion_pack";
  }
  llvm_unreachable("unknown StageCostModelKind");
}

bool StageMemoryLayoutCost::isValid() const {
  return std::isfinite(firstCycles) && firstCycles >= 0.0 &&
         std::isfinite(incrementalCycles) && incrementalCycles >= 0.0 &&
         minCount >= 1 && maxCount >= minCount && !evidence.empty();
}

std::optional<double> StageMemoryLayoutCost::evaluate(double count) const {
  if (!isValid() || !std::isfinite(count) || count < 0.0)
    return std::nullopt;
  if (count == 0.0)
    return 0.0;
  if (count < minCount || count > maxCount || std::floor(count) != count)
    return std::nullopt;
  const double cost = firstCycles + (count - 1.0) * incrementalCycles;
  return std::isfinite(cost) ? std::optional<double>(cost) : std::nullopt;
}

bool StageAtomicRate::isValid() const {
  return std::isfinite(logicalElementsPerCycle) &&
         logicalElementsPerCycle > 0.0 &&
         std::isfinite(operationStartupCycles) &&
         operationStartupCycles >= 0.0 &&
         std::isfinite(resultDependencyCycles) &&
         resultDependencyCycles >= 0.0 &&
         std::isfinite(unknownContentionMultiplier) &&
         unknownContentionMultiplier >= 1.0;
}

bool StageModeProfile::isValid(StageMode mode) const {
  const std::array<double, 9> common = {setupCycles,
                                        predicateOperationsPerCycle,
                                        shuffleLanesPerCycle,
                                        dotSetupCycles,
                                        dotFlopsPerCycle,
                                        issueOperationsPerCycle,
                                        prefixScanStepLatencyCycles,
                                        static_cast<double>(vectorWidth),
                                        static_cast<double>(issueWidth)};
  if (!std::all_of(
          common.begin(), common.end(),
          [](double value) { return std::isfinite(value) && value > 0.0; }) ||
      !std::isfinite(loadOperationSetupCycles) ||
      loadOperationSetupCycles < 0.0 ||
      !std::isfinite(storeOperationSetupCycles) ||
      storeOperationSetupCycles < 0.0)
    return false;
  if (mode == StageMode::SIMD) {
    if (!(loadBytesPerCycle > 0.0 && storeBytesPerCycle > 0.0))
      return false;
  } else if (!(loadWarpInstructionsPerCycle > 0.0 &&
               storeWarpInstructionsPerCycle > 0.0)) {
    return false;
  }
  const bool hasRecurrenceGroupCurve =
      !recurrenceReductionGroupCyclesByFactorAndLaneSteps.empty();
  const bool hasContinuousMemoryGroupCurve =
      !continuousLoadGroupCyclesByFactorAndBytes.empty() ||
      !continuousStoreGroupCyclesByFactorAndBytes.empty() ||
      !continuousLoadGroupStartupCyclesByFactorAndBytes.empty() ||
      !continuousStoreGroupStartupCyclesByFactorAndBytes.empty();
  if (mode == StageMode::SIMT &&
      ((hasRecurrenceGroupCurve && recurrenceGroupBaseWarpCount <= 0) ||
       (hasContinuousMemoryGroupCurve &&
        continuousMemoryGroupBaseWarpCount <= 0)))
    return false;
  const bool validIndirectCurve = llvm::all_of(
      indirectLoadSystemCyclesByWarpInstructions, [](const auto &entry) {
        return entry.first > 0 && std::isfinite(entry.second) &&
               entry.second > 0.0;
      });
  const bool validIndirectStartupCurve = llvm::all_of(
      indirectLoadStartupSystemCyclesByWarpInstructions, [](const auto &entry) {
        return entry.first > 0 && std::isfinite(entry.second) &&
               entry.second >= 0.0;
      });
  auto validFactorCurves = [](const auto &curves) {
    return llvm::all_of(curves, [](const auto &factorEntry) {
      return factorEntry.first > 0 && !factorEntry.second.empty() &&
             llvm::all_of(factorEntry.second, [](const auto &point) {
               return point.first > 0 && std::isfinite(point.second) &&
                      point.second > 0.0;
             });
    });
  };
  const auto validLayouts = [](const auto &curves) {
    return llvm::all_of(curves, [](const auto &entry) {
      return !entry.first.empty() && entry.second.isValid();
    });
  };
  const bool validStoreService =
      storeFootprintUnitBytes == 0
          ? storeCyclesPerFootprintUnit == 0.0 &&
                storeIssueFloorByWarpCount.empty()
          : storeFootprintUnitBytes > 0 &&
                std::isfinite(storeCyclesPerFootprintUnit) &&
                storeCyclesPerFootprintUnit > 0.0 &&
                !storeIssueFloorByWarpCount.empty() &&
                llvm::all_of(storeIssueFloorByWarpCount, [](const auto &entry) {
                  return entry.first > 0 && std::isfinite(entry.second) &&
                         entry.second > 0.0;
                });
  return validIndirectCurve && validIndirectStartupCurve && validStoreService &&
         validLayouts(loadLayoutCosts) && validLayouts(storeLayoutCosts) &&
         validFactorCurves(
             recurrenceReductionGroupCyclesByFactorAndLaneSteps) &&
         validFactorCurves(continuousLoadGroupCyclesByFactorAndBytes) &&
         validFactorCurves(continuousStoreGroupCyclesByFactorAndBytes) &&
         validFactorCurves(continuousLoadGroupStartupCyclesByFactorAndBytes) &&
         validFactorCurves(continuousStoreGroupStartupCyclesByFactorAndBytes) &&
         llvm::all_of(operationRates, [](const auto &entry) {
           return std::isfinite(entry.second.throughput) &&
                  entry.second.throughput > 0.0 &&
                  std::isfinite(entry.second.factor) &&
                  entry.second.factor > 0.0;
         });
}

bool HardwareProfile::isValid() const {
  return !profileVersion.empty() && !target.empty() && baseSimtWarpCount > 0 &&
         simd.isValid(StageMode::SIMD) && simt.isValid(StageMode::SIMT) &&
         scopeHandoff.isValid();
}

llvm::Expected<StageCostTable>
StageCostEvaluator::evaluate(const StagePartition &partition,
                             const HardwareProfile &profile) const {
  if (partition.stages.empty())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "StagePartition requires at least one Stage");
  if (!profile.isValid())
    return llvm::createStringError(std::errc::invalid_argument,
                                   "HardwareProfile is invalid");
  StageCostTable table;
  table.operationOwnershipComplete = partition.operationOwnershipComplete;
  table.modeledOperationCount = partition.modeledOperationCount;
  table.profileVersion = profile.profileVersion;
  llvm::StringSet<> stageIds;

  for (const LogicalStage &stage : partition.stages) {
    if (stage.id.empty() || !stageIds.insert(stage.id).second)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Stage ids must be non-empty and unique: '%s'", stage.id.c_str());
    if (stage.iterationCount <= 0 || !stage.features.isValid() ||
        !stage.workload.isFiniteAndNonNegative())
      return llvm::createStringError(
          std::errc::invalid_argument,
          "Stage '%s' has invalid iteration/features", stage.id.c_str());
    if (!stage.simdLegal && !stage.simtLegal)
      return llvm::createStringError(std::errc::invalid_argument,
                                     "Stage '%s' has no legal StageMode",
                                     stage.id.c_str());
    if (stage.simtLegal && stage.legalSimtFactors.empty() &&
        (!stage.localSimtMaterializable || stage.localSimtFactors.empty()))
      return llvm::createStringError(
          std::errc::invalid_argument,
          "SIMT Stage '%s' has no legal SuperBlock factor", stage.id.c_str());

    LogicalStageCost logicalCost;
    logicalCost.id = stage.id;
    logicalCost.model = stringifyStageCostModel(stage.costModelKind).str();
    logicalCost.schedule = stage.scheduleKind;
    logicalCost.iterationCount = stage.iterationCount;
    logicalCost.features = stage.features;
    logicalCost.workload = stage.workload;
    logicalCost.ownedOperationCount =
        static_cast<int64_t>(stage.operations.size());
    logicalCost.sourceLocations = collectSourceLocations(stage);
    logicalCost.liveInCount = static_cast<int64_t>(stage.liveIns.size());
    logicalCost.liveOutCount = static_cast<int64_t>(stage.liveOuts.size());
    logicalCost.liveInBytes = stage.liveInBytes;
    logicalCost.liveOutBytes = stage.liveOutBytes;
    logicalCost.scopeInputTensorBytes = stage.scopeInputTensorBytes;
    logicalCost.scopeOutputTensorBytes = stage.scopeOutputTensorBytes;
    logicalCost.simtAnchorIndices = stage.simtAnchorIndices;
    logicalCost.localSimtMaterializable = stage.localSimtMaterializable;
    logicalCost.localSuperblockMaterializable =
        stage.localSuperblockMaterializable;
    logicalCost.legalSimtFactors = stage.legalSimtFactors;
    logicalCost.localSimtFactors = stage.localSimtFactors;

    llvm::SmallVector<StageImplementation> implementations;
    if (stage.simdLegal)
      implementations.push_back({StageMode::SIMD, 1, false});
    if (stage.simtLegal)
      for (int64_t factor : stage.legalSimtFactors)
        implementations.push_back({StageMode::SIMT, factor, false});
    if (stage.simtLegal && stage.localSimtMaterializable)
      for (int64_t factor : stage.localSimtFactors)
        implementations.push_back({StageMode::SIMT, factor, true});

    for (const StageImplementation &implementation : implementations) {
      if (!isDeclaredLegal(stage, implementation))
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' has an illegal candidate",
                                       stage.id.c_str());
      const StageModeProfile modeProfile =
          resolveModeProfile(profile, implementation);
      std::optional<StageResourceCycles> resources =
          mapWorkload(stage, modeProfile, implementation.mode,
                      implementation.superblockFactor);
      if (!resources)
        continue;
      StageImplementationCost cost;
      cost.implementation = implementation;
      cost.resources = *resources;
      if (resources->recurrenceGroupCriticalPath > 0.0 ||
          resources->continuousLoadGroup > 0.0 ||
          resources->continuousStoreGroup > 0.0)
        cost.formulaEvidence = "factor_specific_measured_group_curve";
      else if (resources->storeLayoutService > 0.0)
        cost.formulaEvidence =
            "layout_store_service_floor_startup_uncalibrated";
      else if (llvm::any_of(stage.workload.memoryAccesses,
                            [](const StageMemoryAccess &access) {
                              return access.layoutClass() !=
                                     StageMemoryAccess::LayoutClass::Unknown;
                            }))
        cost.formulaEvidence = "uncalibrated_layout_memory_fallback";
      else if (stage.features.hasIndirectMemory)
        cost.formulaEvidence = extrapolatesIndirectEvidence(stage, modeProfile)
                                   ? "measured_indirect_load_curve_extrapolated"
                                   : "measured_indirect_load_curve";
      else if (implementation.mode == StageMode::SIMT &&
               implementation.superblockFactor > 1)
        cost.formulaEvidence = "uncalibrated_superblock_fallback";
      else
        cost.formulaEvidence = "analytical_resource_formula";
      if (stage.costModelKind == StageCostModelKind::VectorIssue) {
        cost.formulaEvidence += "+elementwise_workload";
        for (const auto &entry : stage.workload.operationCounts) {
          auto rate = modeProfile.operationRates.find(entry.first());
          if (rate == modeProfile.operationRates.end() ||
              rate->second.throughput <= 0.0) {
            cost.formulaEvidence += "+unprofiled_vector_execution";
            break;
          }
        }
      }
      if (stage.workload.scalarOperations > 0.0)
        cost.formulaEvidence += "+unclassified_scalar_issue_floor";
      if (stage.workload.loopBackedges > 0.0 ||
          stage.workload.conditionalBranches > 0.0 ||
          stage.workload.divergentBranches > 0.0 ||
          stage.workload.synchronizations > 0.0)
        cost.formulaEvidence += "+control_issue_floor_only";
      if (!stage.workload.prefixScanCounts.empty())
        cost.formulaEvidence += "+prefix_scan_per_level_issue";
      cost.totalCycles = applySuperBlock(
          stage, *resources, implementation, profile,
          estimateStage(stage, profile, implementation.mode, *resources));
      if (!cost.isValid())
        return llvm::createStringError(std::errc::invalid_argument,
                                       "Stage '%s' produced an invalid cost",
                                       stage.id.c_str());
      logicalCost.implementations.push_back(std::move(cost));
    }

    if (logicalCost.implementations.empty())
      return llvm::createStringError(
          std::errc::not_supported,
          "Stage '%s' has no measured implementation for its workload",
          stage.id.c_str());

    table.stages.push_back(std::move(logicalCost));
  }
  return table;
}
