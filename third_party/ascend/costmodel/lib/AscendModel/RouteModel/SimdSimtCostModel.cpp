//===- SimdSimtCostModel.cpp - Ascend SIMD/SIMT candidate model ----------===//
//
// The numerical model in this file is the versioned C++ candidate model.  It
// intentionally produces a relative per-program selection score, not an
// end-to-end kernel-time prediction.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "AscendModel/Analysis/SimtAnchorAnalysis.h"
#include "AscendModel/Analysis/StagePartitioner.h"
#include "AscendModel/Profile/MicrobenchmarkProfile.h"
#include "AscendModel/RouteModel/StageCostModels.h"
#include "ascend/include/Utils/SuperBlockFactor.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <set>
#include <system_error>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace mlir::ascend;

namespace {

constexpr llvm::StringLiteral kAllSimd = "all_simd";
constexpr llvm::StringLiteral kAllSimtOnly = "all_simt_only";
constexpr llvm::StringLiteral kMixedSimdSimt = "mixed_simd_simt";
constexpr int64_t kSupportedProfileSchemaVersion = 12;

struct StructuralProfile {
  int64_t tinyDotFlopsMax = 0;
};

struct CandidateProfile {
  HardwareProfile hardware;
  std::vector<std::string> compatibleTargets;
  std::string scoreUnit;
  std::string contentSha256;
  std::string selectionContentSha256;
  std::string microbenchmarkProfileVersion;
  std::string microbenchmarkProfileTarget;
  std::string microbenchmarkContentSha256;
  StructuralProfile structural;
};

static bool wildcardMatchInsensitive(llvm::StringRef pattern,
                                     llvm::StringRef value) {
  const std::string loweredPattern = pattern.lower();
  const std::string loweredValue = value.lower();
  size_t patternIndex = 0;
  size_t valueIndex = 0;
  size_t starIndex = std::string::npos;
  size_t starValueIndex = 0;
  while (valueIndex < loweredValue.size()) {
    if (patternIndex < loweredPattern.size() &&
        (loweredPattern[patternIndex] == '?' ||
         loweredPattern[patternIndex] == loweredValue[valueIndex])) {
      ++patternIndex;
      ++valueIndex;
      continue;
    }
    if (patternIndex < loweredPattern.size() &&
        loweredPattern[patternIndex] == '*') {
      starIndex = patternIndex++;
      starValueIndex = valueIndex;
      continue;
    }
    if (starIndex != std::string::npos) {
      patternIndex = starIndex + 1;
      valueIndex = ++starValueIndex;
      continue;
    }
    return false;
  }
  while (patternIndex < loweredPattern.size() &&
         loweredPattern[patternIndex] == '*')
    ++patternIndex;
  return patternIndex == loweredPattern.size();
}

/// Small fail-fast facade around llvm::json.  It permits a readable profile
/// parser while retaining a single actionable error message.
class ProfileJSONReader {
public:
  const llvm::json::Object *object(const llvm::json::Object &parent,
                                   llvm::StringRef key,
                                   llvm::StringRef context) {
    if (failed())
      return nullptr;
    if (const auto *value = parent.getObject(key))
      return value;
    setError(context + "." + key + " must be an object");
    return nullptr;
  }

  double number(const llvm::json::Object &parent, llvm::StringRef key,
                llvm::StringRef context) {
    if (failed())
      return 0.0;
    if (auto value = parent.getNumber(key))
      return *value;
    setError(context + "." + key + " must be a number");
    return 0.0;
  }

  int64_t integer(const llvm::json::Object &parent, llvm::StringRef key,
                  llvm::StringRef context) {
    if (failed())
      return 0;
    if (auto value = parent.getInteger(key))
      return *value;
    setError(context + "." + key + " must be an integer");
    return 0;
  }

  std::string string(const llvm::json::Object &parent, llvm::StringRef key,
                     llvm::StringRef context) {
    if (failed())
      return {};
    if (auto value = parent.getString(key))
      return value->str();
    setError(context + "." + key + " must be a string");
    return {};
  }

  double optionalNumber(const llvm::json::Object &parent, llvm::StringRef key,
                        double defaultValue) {
    if (auto value = parent.getNumber(key))
      return *value;
    return defaultValue;
  }

  bool failed() const { return !error.empty(); }
  llvm::StringRef getError() const { return error; }

  void setError(const llvm::Twine &message) {
    if (error.empty())
      error = message.str();
  }

private:
  std::string error;
};

static double resolveNumberOrMeasurement(
    const llvm::json::Object &object, llvm::StringRef numberKey,
    llvm::StringRef measurementKey, llvm::StringRef expectedUnit,
    const MicrobenchmarkProfile *microbench, ProfileJSONReader &reader,
    llvm::StringRef context) {
  if (auto reference = object.getString(measurementKey)) {
    if (!microbench) {
      reader.setError(context + "." + measurementKey +
                      " requires microbenchmark_profile");
      return 0.0;
    }
    llvm::StringRef expectedCycleDomain = "none";
    if (expectedUnit == "system_cycle" ||
        expectedUnit.ends_with("/system_cycle"))
      expectedCycleDomain = "SYS_CNT";
    auto value =
        microbench->requireValue(*reference, expectedUnit, expectedCycleDomain);
    if (!value) {
      reader.setError(llvm::toString(value.takeError()));
      return 0.0;
    }
    return *value;
  }
  return reader.number(object, numberKey, context);
}

static void readPositiveIntegerMeasurementCurve(
    const llvm::json::Object &owner, llvm::StringRef curveKey,
    llvm::StringRef expectedUnit, const MicrobenchmarkProfile *microbench,
    ProfileJSONReader &reader, llvm::StringRef context,
    std::map<int64_t, double> &curve) {
  const auto *object = owner.getObject(curveKey);
  if (!object)
    return;
  if (!microbench) {
    reader.setError(context + "." + curveKey +
                    " requires microbenchmark_profile");
    return;
  }
  llvm::StringRef expectedCycleDomain = "none";
  if (expectedUnit == "system_cycle" || expectedUnit.ends_with("/system_cycle"))
    expectedCycleDomain = "SYS_CNT";
  for (const auto &entry : *object) {
    int64_t point = 0;
    if (llvm::StringRef(entry.first).getAsInteger(10, point) || point <= 0) {
      reader.setError(context + "." + curveKey +
                      " keys must be positive integers");
      return;
    }
    const auto measurement = entry.second.getAsString();
    if (!measurement) {
      reader.setError(context + "." + curveKey +
                      " values must be measurement names");
      return;
    }
    auto value = microbench->requireValue(*measurement, expectedUnit,
                                          expectedCycleDomain);
    if (!value) {
      reader.setError(llvm::toString(value.takeError()));
      return;
    }
    curve[point] = *value;
  }
}

static void readFactorMeasurementCurves(
    const llvm::json::Object &owner, llvm::StringRef curveKey,
    llvm::StringRef expectedUnit, const MicrobenchmarkProfile *microbench,
    ProfileJSONReader &reader, llvm::StringRef context,
    std::map<int64_t, std::map<int64_t, double>> &curves) {
  const auto *factors = owner.getObject(curveKey);
  if (!factors)
    return;
  if (!microbench) {
    reader.setError(context + "." + curveKey +
                    " requires microbenchmark_profile");
    return;
  }
  llvm::StringRef expectedCycleDomain = "none";
  if (expectedUnit == "system_cycle" || expectedUnit.ends_with("/system_cycle"))
    expectedCycleDomain = "SYS_CNT";
  for (const auto &factorEntry : *factors) {
    const llvm::StringRef factorKey(factorEntry.first);
    int64_t factor = 0;
    if (factorKey.getAsInteger(10, factor) || factor <= 0) {
      reader.setError(context + "." + curveKey +
                      " factor keys must be positive integers");
      return;
    }
    const std::string factorContext =
        (context + "." + curveKey + "." + factorKey).str();
    const auto *points = factorEntry.second.getAsObject();
    if (!points) {
      reader.setError(factorContext + " must be an object");
      return;
    }
    auto &curve = curves[factor];
    for (const auto &pointEntry : *points) {
      int64_t laneSteps = 0;
      if (llvm::StringRef(pointEntry.first).getAsInteger(10, laneSteps) ||
          laneSteps <= 0) {
        reader.setError(factorContext + " keys must be positive integers");
        return;
      }
      const auto measurement = pointEntry.second.getAsString();
      if (!measurement) {
        reader.setError(factorContext + " values must be measurement names");
        return;
      }
      auto value = microbench->requireValue(*measurement, expectedUnit,
                                            expectedCycleDomain);
      if (!value) {
        reader.setError(llvm::toString(value.takeError()));
        return;
      }
      curve[laneSteps] = *value;
    }
  }
}

static StageOperationRate
resolveOpProfile(const llvm::json::Object &ops, llvm::StringRef opName,
                 llvm::StringRef throughputKey, llvm::StringRef expectedUnit,
                 const MicrobenchmarkProfile *microbench,
                 ProfileJSONReader &reader) {
  StageOperationRate result;
  const llvm::json::Value *raw = ops.get(opName);
  if (!raw) {
    reader.setError("missing operation profile " + opName);
    return result;
  }
  const auto *op = raw->getAsObject();
  if (!op) {
    reader.setError("operation profile " + opName + " must be an object");
    return result;
  }
  if (auto relative = op->getString("relative_to")) {
    StageOperationRate base = resolveOpProfile(
        ops, *relative, throughputKey, expectedUnit, microbench, reader);
    result.throughput = base.throughput;
    result.throughputByWarpCount = std::move(base.throughputByWarpCount);
    // A relative operation without a measured factor is not calibrated.  In
    // particular, an explicit JSON null must not silently become unit cost.
    result.factor = reader.number(*op, "factor", opName);
    return result;
  }
  result.throughput =
      resolveNumberOrMeasurement(*op, throughputKey, "throughput_measurement",
                                 expectedUnit, microbench, reader, opName);
  readPositiveIntegerMeasurementCurve(*op, "throughput_measurements",
                                      expectedUnit, microbench, reader, opName,
                                      result.throughputByWarpCount);
  result.factor = reader.optionalNumber(*op, "factor", 1.0);
  return result;
}

/// Match Python's json.dumps(value, sort_keys=True) representation.  Keeping
/// this stable makes profile_content_sha256 identical across the temporary
/// Python model and this C++ implementation.
static void emitPythonCanonicalJSON(const llvm::json::Value &value,
                                    llvm::raw_ostream &os) {
  if (const auto *object = value.getAsObject()) {
    std::vector<llvm::StringRef> keys;
    keys.reserve(object->size());
    for (const auto &entry : *object)
      keys.push_back(entry.first);
    llvm::sort(keys);
    os << '{';
    bool first = true;
    for (llvm::StringRef key : keys) {
      if (!first)
        os << ", ";
      first = false;
      os << llvm::json::Value(key.str()) << ": ";
      emitPythonCanonicalJSON(*object->get(key), os);
    }
    os << '}';
    return;
  }
  if (const auto *array = value.getAsArray()) {
    os << '[';
    bool first = true;
    for (const llvm::json::Value &element : *array) {
      if (!first)
        os << ", ";
      first = false;
      emitPythonCanonicalJSON(element, os);
    }
    os << ']';
    return;
  }
  os << value;
}

static std::string resolveProfileReference(llvm::StringRef ownerPath,
                                           llvm::StringRef reference) {
  if (llvm::sys::path::is_absolute(reference))
    return reference.str();
  llvm::SmallString<256> resolved(ownerPath);
  llvm::sys::path::remove_filename(resolved);
  llvm::sys::path::append(resolved, reference);
  llvm::sys::path::remove_dots(resolved, true);
  return resolved.str().str();
}

static void
readLayoutCostMap(ProfileJSONReader &reader,
                  const llvm::json::Object &resources, llvm::StringRef key,
                  llvm::StringRef context,
                  std::map<std::string, StageMemoryLayoutCost> &out) {
  const auto *object = resources.getObject(key);
  if (!object)
    return;
  for (const auto &entry : *object) {
    const auto *value = entry.second.getAsObject();
    const llvm::StringRef entryKey = entry.first;
    const std::string path = (context + "." + key + "." + entryKey).str();
    if (!value) {
      reader.setError(path + " must be an object");
      return;
    }
    StageMemoryLayoutCost cost;
    cost.firstCycles = reader.number(*value, "first_system_cycles", path);
    cost.incrementalCycles =
        reader.number(*value, "incremental_system_cycles", path);
    cost.minCount = reader.integer(*value, "min_count", path);
    cost.maxCount = reader.integer(*value, "max_count", path);
    cost.evidence = reader.string(*value, "evidence", path);
    if (!cost.isValid())
      reader.setError(path + " requires finite nonnegative costs, a valid "
                             "count domain and evidence");
    out[entryKey.str()] = cost;
  }
}

static void readStageResources(ProfileJSONReader &reader,
                               const llvm::json::Object &mode,
                               llvm::StringRef context,
                               const MicrobenchmarkProfile *microbench,
                               StageModeProfile &profile) {
  const auto *resources = reader.object(mode, "stage_resources", context);
  if (!resources)
    return;
  const std::string prefix = (context + ".stage_resources").str();
  profile.issueOperationsPerCycle = resolveNumberOrMeasurement(
      *resources, "issue_instructions_per_system_cycle",
      "issue_instructions_per_system_cycle_measurement",
      "instruction/system_cycle", microbench, reader, prefix);
  if (const auto *scan = resources->getObject("prefix_scan"))
    profile.prefixScanStepLatencyCycles = resolveNumberOrMeasurement(
        *scan, "step_latency_system_cycles", "step_latency_measurement",
        "system_cycle", microbench, reader, prefix + ".prefix_scan");
  if (const auto *recurrence =
          resources->getObject("loop_carried_recurrence")) {
    if (auto baseWarpCount = recurrence->getInteger("base_warp_count")) {
      if (*baseWarpCount <= 0)
        reader.setError(prefix +
                        ".loop_carried_recurrence.base_warp_count must be "
                        "positive");
      else
        profile.recurrenceGroupBaseWarpCount = *baseWarpCount;
    }
    readFactorMeasurementCurves(
        *recurrence, "reduction_group_iteration_system_cycle_measurements",
        "system_cycle", microbench, reader, prefix + ".loop_carried_recurrence",
        profile.recurrenceReductionGroupCyclesByFactorAndLaneSteps);
  }
  if (const auto *continuous = resources->getObject("continuous_memory")) {
    if (auto baseWarpCount = continuous->getInteger("base_warp_count")) {
      if (*baseWarpCount <= 0)
        reader.setError(prefix +
                        ".continuous_memory.base_warp_count must be positive");
      else
        profile.continuousMemoryGroupBaseWarpCount = *baseWarpCount;
    }
    readFactorMeasurementCurves(
        *continuous, "load_group_operation_system_cycle_measurements",
        "system_cycle", microbench, reader, prefix + ".continuous_memory",
        profile.continuousLoadGroupCyclesByFactorAndBytes);
    readFactorMeasurementCurves(
        *continuous, "store_group_operation_system_cycle_measurements",
        "system_cycle", microbench, reader, prefix + ".continuous_memory",
        profile.continuousStoreGroupCyclesByFactorAndBytes);
    readFactorMeasurementCurves(
        *continuous, "load_group_startup_system_cycle_measurements",
        "system_cycle", microbench, reader, prefix + ".continuous_memory",
        profile.continuousLoadGroupStartupCyclesByFactorAndBytes);
    readFactorMeasurementCurves(
        *continuous, "store_group_startup_system_cycle_measurements",
        "system_cycle", microbench, reader, prefix + ".continuous_memory",
        profile.continuousStoreGroupStartupCyclesByFactorAndBytes);
  }
  if (const auto *layout = resources->getObject("layout_memory")) {
    readLayoutCostMap(reader, *layout, "load", prefix + ".layout_memory",
                      profile.loadLayoutCosts);
    readLayoutCostMap(reader, *layout, "store", prefix + ".layout_memory",
                      profile.storeLayoutCosts);
  }
  if (const auto *service = resources->getObject("store_footprint_service")) {
    const std::string path = prefix + ".store_footprint_service";
    if (reader.string(*service, "evidence", path).empty())
      reader.setError(path + ".evidence must identify the calibration data");
    profile.storeFootprintUnitBytes =
        reader.integer(*service, "unit_bytes", path);
    profile.storeCyclesPerFootprintUnit =
        reader.number(*service, "system_cycles_per_unit", path);
    if (const auto *floors =
            reader.object(*service, "issue_floor_by_warps", path))
      for (const auto &entry : *floors) {
        const llvm::StringRef key = entry.first;
        int64_t warps = 0;
        auto value = entry.second.getAsNumber();
        if (key.getAsInteger(10, warps) || warps <= 0 || !value || *value <= 0)
          reader.setError(path + " has an invalid warp-count floor");
        else
          profile.storeIssueFloorByWarpCount[warps] = *value;
      }
  }
  if (const auto *indirect =
          reader.object(*resources, "indirect_memory", prefix)) {
    const std::string path = prefix + ".indirect_memory";
    readPositiveIntegerMeasurementCurve(
        *indirect, "load_system_cycle_measurements", "system_cycle", microbench,
        reader, path, profile.indirectLoadSystemCyclesByWarpInstructions);
    readPositiveIntegerMeasurementCurve(
        *indirect, "load_startup_system_cycle_measurements", "system_cycle",
        microbench, reader, path,
        profile.indirectLoadStartupSystemCyclesByWarpInstructions);
  }
}

static void readSetupCycleCurve(ProfileJSONReader &reader,
                                const llvm::json::Object &setup,
                                const MicrobenchmarkProfile *microbench,
                                StageModeProfile &profile) {
  const auto *curve = setup.getObject("empty_launch_measurements");
  if (!curve)
    return;
  if (!microbench) {
    reader.setError("SIMT setup curve requires microbenchmark_profile");
    return;
  }
  for (const auto &entry : *curve) {
    int64_t warpCount = 0;
    if (llvm::StringRef(entry.first).getAsInteger(10, warpCount) ||
        warpCount <= 0) {
      reader.setError("SIMT setup curve keys must be positive warp counts");
      return;
    }
    const auto measurement = entry.second.getAsString();
    if (!measurement) {
      reader.setError("SIMT setup curve values must be measurement names");
      return;
    }
    auto cycles = microbench->requireValue(*measurement, "system_cycle");
    if (!cycles) {
      reader.setError(llvm::toString(cycles.takeError()));
      return;
    }
    profile.setupCyclesByWarpCount[warpCount] = *cycles;
  }
}

static llvm::Expected<CandidateProfile>
loadCandidateProfile(llvm::StringRef requestedPath) {
  std::string path = requestedPath.empty() ? getDefaultSimdSimtProfilePath()
                                           : requestedPath.str();
  if (path.empty())
    return llvm::createStringError(std::errc::no_such_file_or_directory,
                                   "SIMD/SIMT profile path is empty; set "
                                   "TRITON_ASCEND_SIMD_SIMT_PROFILE");

  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "failed to read SIMD/SIMT profile '%s'",
                                   path.c_str());
  auto parsed = llvm::json::parse(buffer.get()->getBuffer());
  if (!parsed)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "failed to parse SIMD/SIMT profile '%s': %s",
                                   path.c_str(),
                                   llvm::toString(parsed.takeError()).c_str());
  const auto *root = parsed->getAsObject();
  if (!root)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT profile root must be an object");
  auto selectionSchemaVersion = root->getInteger("schema_version");
  if (!selectionSchemaVersion || *selectionSchemaVersion != 14)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT profile schema_version must be 14");

  CandidateProfile profile;
  ProfileJSONReader reader;
  std::optional<MicrobenchmarkProfile> microbenchmarkProfile;
  if (auto reference = root->getString("microbenchmark_profile")) {
    std::string resolved = resolveProfileReference(path, *reference);
    auto loaded = MicrobenchmarkProfile::loadFromFile(resolved);
    if (!loaded)
      return llvm::createStringError(
          std::errc::invalid_argument,
          "failed to load shared microbenchmark profile referenced by '%s': %s",
          path.c_str(), llvm::toString(loaded.takeError()).c_str());
    microbenchmarkProfile.emplace(std::move(*loaded));
    profile.microbenchmarkProfileVersion =
        microbenchmarkProfile->getProfileVersion().str();
    profile.microbenchmarkProfileTarget =
        microbenchmarkProfile->getTarget().str();
    profile.microbenchmarkContentSha256 =
        microbenchmarkProfile->getContentSha256().str();
  }
  const MicrobenchmarkProfile *microbench =
      microbenchmarkProfile ? &*microbenchmarkProfile : nullptr;

  HardwareProfile &hardware = profile.hardware;
  hardware.profileVersion = reader.string(*root, "profile_version", "profile");
  hardware.target = reader.string(*root, "target", "profile");
  if (const auto *targets = root->getArray("compatible_targets")) {
    for (const llvm::json::Value &target : *targets) {
      auto pattern = target.getAsString();
      if (!pattern || pattern->empty()) {
        reader.setError(
            "profile.compatible_targets entries must be non-empty strings");
        break;
      }
      profile.compatibleTargets.push_back(pattern->str());
    }
    if (profile.compatibleTargets.empty())
      reader.setError("profile.compatible_targets must not be empty");
  } else {
    reader.setError("profile.compatible_targets must be an array");
  }
  if (microbench && llvm::StringRef(hardware.target) != microbench->getTarget())
    reader.setError("selection profile target '" + hardware.target +
                    "' does not match shared microbenchmark target '" +
                    microbench->getTarget().str() + "'");
  profile.scoreUnit = reader.string(*root, "score_unit", "profile");
  const auto *calibration =
      reader.object(*root, "selection_calibration", "profile");
  if (calibration)
    profile.structural.tinyDotFlopsMax = reader.integer(
        *calibration, "tiny_dot_flops_max", "profile.selection_calibration");

  const auto *simd = reader.object(*root, "simd", "profile");
  if (simd) {
    int64_t vectorWidthBits = 0;
    if (simd->getString("vector_width_measurement")) {
      vectorWidthBits =
          static_cast<int64_t>(std::llround(resolveNumberOrMeasurement(
              *simd, "vector_width_bits", "vector_width_measurement", "bit",
              microbench, reader, "simd")));
    } else {
      vectorWidthBits = reader.integer(*simd, "vector_width_bits", "simd");
    }
    hardware.simd.vectorWidthBits = std::max<int64_t>(1, vectorWidthBits);
    hardware.simd.vectorWidth =
        std::max<int64_t>(1, hardware.simd.vectorWidthBits / 32);
    hardware.simd.issueWidth = hardware.simd.vectorWidth;
    if (const auto *startup =
            reader.object(*simd, "startup_system_cycles", "simd"))
      hardware.simd.setupCycles = resolveNumberOrMeasurement(
          *startup, "vector", "vector_measurement", "system_cycle", microbench,
          reader, "simd.startup_system_cycles");
    if (const auto *ops = reader.object(*simd, "ops", "simd")) {
      for (llvm::StringRef op :
           {"f32.add", "f32.sub", "f32.mul", "f32.div", "f32.max", "f32.abs",
            "f32.exp", "f32.log", "f32.sin", "f32.cos", "f32.trans",
            "predicate.cmp", "predicate.select", "convert.cast", "f32.clamp"})
        hardware.simd.operationRates[op] = resolveOpProfile(
            *ops, op, "throughput_vector_instructions_per_system_cycle",
            "vector_instruction/system_cycle", microbench, reader);
    }
    if (const auto *memory = reader.object(*simd, "memory", "simd")) {
      hardware.simd.loadBytesPerCycle = resolveNumberOrMeasurement(
          *memory, "vector_mte2_bytes_per_system_cycle",
          "load_throughput_measurement", "byte/system_cycle", microbench,
          reader, "simd.memory");
      hardware.simd.storeBytesPerCycle = resolveNumberOrMeasurement(
          *memory, "mte3_bytes_per_system_cycle",
          "store_throughput_measurement", "byte/system_cycle", microbench,
          reader, "simd.memory");
      if (memory->getString("load_startup_measurement"))
        hardware.simd.loadOperationSetupCycles = resolveNumberOrMeasurement(
            *memory, "load_operation_startup_system_cycles",
            "load_startup_measurement", "system_cycle", microbench, reader,
            "simd.memory");
      else
        hardware.simd.loadOperationSetupCycles = reader.optionalNumber(
            *memory, "load_operation_startup_system_cycles", 0.0);
      if (memory->getString("store_startup_measurement"))
        hardware.simd.storeOperationSetupCycles = resolveNumberOrMeasurement(
            *memory, "store_operation_startup_system_cycles",
            "store_startup_measurement", "system_cycle", microbench, reader,
            "simd.memory");
      else
        hardware.simd.storeOperationSetupCycles = reader.optionalNumber(
            *memory, "store_operation_startup_system_cycles", 0.0);
    }
    if (const auto *dot = reader.object(*simd, "dot", "simd")) {
      hardware.simd.dotSetupCycles = resolveNumberOrMeasurement(
          *dot, "startup_system_cycles", "startup_measurement", "system_cycle",
          microbench, reader, "simd.dot");
      hardware.simd.dotFlopsPerCycle = resolveNumberOrMeasurement(
          *dot, "flops_per_system_cycle", "throughput_measurement",
          "flop/system_cycle", microbench, reader, "simd.dot");
    }
    readStageResources(reader, *simd, "simd", microbench, hardware.simd);
    const auto predicate = hardware.simd.operationRates.lookup("predicate.cmp");
    hardware.simd.predicateOperationsPerCycle =
        predicate.throughput / std::max(1.0, predicate.factor);
    hardware.simd.shuffleLanesPerCycle = hardware.simd.vectorWidth;
  }

  const auto *simt = reader.object(*root, "simt", "profile");
  if (simt) {
    if (simt->getString("warp_size_measurement")) {
      hardware.simt.issueWidth =
          static_cast<int64_t>(std::llround(resolveNumberOrMeasurement(
              *simt, "warp_size", "warp_size_measurement", "lane", microbench,
              reader, "simt")));
    } else {
      hardware.simt.issueWidth = reader.integer(*simt, "warp_size", "simt");
    }
    hardware.simt.vectorWidth = 1;
    hardware.simt.vectorWidthBits = 1;
    if (const auto *setup =
            reader.object(*simt, "setup_system_cycles", "simt")) {
      hardware.simt.setupCycles = resolveNumberOrMeasurement(
          *setup, "empty_launch", "empty_launch_measurement", "system_cycle",
          microbench, reader, "simt.setup_system_cycles");
      readSetupCycleCurve(reader, *setup, microbench, hardware.simt);
    }
    if (const auto *ops = reader.object(*simt, "ops", "simt")) {
      for (llvm::StringRef op :
           {"f32.add", "f32.sub", "f32.mul", "f32.div", "f32.max", "f32.abs",
            "f32.exp", "f32.log", "f32.sin", "f32.cos", "f32.trans",
            "predicate.cmp", "predicate.select", "convert.cast", "f32.clamp"})
        hardware.simt.operationRates[op] =
            resolveOpProfile(*ops, op, "throughput_scalar_ops_per_system_cycle",
                             "scalar_op/system_cycle", microbench, reader);
    }
    if (const auto *dot = reader.object(*simt, "dot", "simt")) {
      hardware.simt.dotSetupCycles = resolveNumberOrMeasurement(
          *dot, "startup_system_cycles", "startup_measurement", "system_cycle",
          microbench, reader, "simt.dot");
      hardware.simt.dotFlopsPerCycle = resolveNumberOrMeasurement(
          *dot, "flops_per_system_cycle", "throughput_measurement",
          "flop/system_cycle", microbench, reader, "simt.dot");
    }
    const auto predicate = hardware.simt.operationRates.lookup("predicate.cmp");
    hardware.simt.predicateOperationsPerCycle =
        predicate.throughput / std::max(1.0, predicate.factor);
    if (const auto *shuffle = reader.object(*simt, "shuffle", "simt")) {
      hardware.simt.shuffleLanesPerCycle =
          hardware.simt.issueWidth *
          resolveNumberOrMeasurement(
              *shuffle, "warp_instructions_per_system_cycle",
              "throughput_measurement", "warp_instruction/system_cycle",
              microbench, reader, "simt.shuffle");
      std::map<int64_t, double> warpInstructionCurve;
      readPositiveIntegerMeasurementCurve(
          *shuffle, "throughput_measurements", "warp_instruction/system_cycle",
          microbench, reader, "simt.shuffle", warpInstructionCurve);
      for (const auto &[warpCount, rate] : warpInstructionCurve)
        hardware.simt.shuffleLanesPerCycleByWarpCount[warpCount] =
            hardware.simt.issueWidth * rate;
    }
    if (const auto *memory = reader.object(*simt, "memory", "simt")) {
      hardware.simt.loadWarpInstructionsPerCycle = resolveNumberOrMeasurement(
          *memory, "load_warp_instructions_per_system_cycle",
          "load_throughput_measurement", "warp_instruction/system_cycle",
          microbench, reader, "simt.memory");
      hardware.simt.storeWarpInstructionsPerCycle = resolveNumberOrMeasurement(
          *memory, "store_warp_instructions_per_system_cycle",
          "store_throughput_measurement", "warp_instruction/system_cycle",
          microbench, reader, "simt.memory");
      readPositiveIntegerMeasurementCurve(
          *memory, "load_throughput_measurements",
          "warp_instruction/system_cycle", microbench, reader, "simt.memory",
          hardware.simt.loadWarpInstructionsPerCycleByWarpCount);
      readPositiveIntegerMeasurementCurve(
          *memory, "store_throughput_measurements",
          "warp_instruction/system_cycle", microbench, reader, "simt.memory",
          hardware.simt.storeWarpInstructionsPerCycleByWarpCount);
    }
    readStageResources(reader, *simt, "simt", microbench, hardware.simt);
    if (const auto *resources = simt->getObject("stage_resources")) {
      if (const auto *handoff = resources->getObject("scope_handoff")) {
        hardware.scopeHandoff.fixedScopeCycles = reader.number(
            *handoff, "fixed_scope_system_cycles", "scope_handoff");
        hardware.scopeHandoff.inputHandoffBytesPerCycle = reader.number(
            *handoff, "input_handoff_bytes_per_system_cycle", "scope_handoff");
        hardware.scopeHandoff.outputHandoffBytesPerCycle = reader.number(
            *handoff, "output_handoff_bytes_per_system_cycle", "scope_handoff");
      }
    }
  }

  if (reader.failed())
    return llvm::createStringError(
        std::errc::invalid_argument, "invalid SIMD/SIMT profile '%s': %s",
        path.c_str(), reader.getError().str().c_str());
  if (!microbench)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT profile must reference "
                                   "microbenchmark_profile");
  if (!hardware.isValid())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT profile contains invalid hardware rates");
  std::string canonicalProfile;
  llvm::raw_string_ostream canonicalStream(canonicalProfile);
  emitPythonCanonicalJSON(*parsed, canonicalStream);
  canonicalStream.flush();
  llvm::ArrayRef<uint8_t> byteArray(
      reinterpret_cast<const uint8_t *>(canonicalProfile.data()),
      canonicalProfile.size());
  auto hash = llvm::SHA256::hash(byteArray);
  profile.selectionContentSha256 =
      llvm::toHex(llvm::ArrayRef<uint8_t>(hash), true);

  profile.contentSha256 = profile.selectionContentSha256;
  if (!profile.microbenchmarkContentSha256.empty()) {
    std::string combinedAssets =
        canonicalProfile +
        "\nshared_microbenchmark_sha256=" + profile.microbenchmarkContentSha256;
    llvm::ArrayRef<uint8_t> combinedBytes(
        reinterpret_cast<const uint8_t *>(combinedAssets.data()),
        combinedAssets.size());
    auto combinedHash = llvm::SHA256::hash(combinedBytes);
    profile.contentSha256 =
        llvm::toHex(llvm::ArrayRef<uint8_t>(combinedHash), true);
  }
  return profile;
}

static llvm::Expected<StageCostModelSummary> evaluateStageModel(
    const SimdSimtFeatureSummary &features, const CandidateProfile &profile,
    unsigned numWarps, bool wholeKernelSuperblockMaterializable,
    bool scopeSuperblockMaterializable, int64_t logicalProgramCountHint,
    int64_t physicalCoreCountHint, int64_t physicalAiCoreCountHint,
    ModuleOp module, const SimtAnchorPlan *anchorPlan,
    const SimdSimtCostModelOptions &options) {
  StagePartitionerOptions partitionerOptions;
  partitionerOptions.tinyDotFlopsMax = profile.structural.tinyDotFlopsMax;
  const int64_t warpLimitedFactorUpperBound =
      std::max<int64_t>(1, 64 / std::max<int64_t>(1, numWarps));
  int64_t commonFactorUpperBound =
      std::min<int64_t>(kMaximumSuperBlockFactor, warpLimitedFactorUpperBound);
  if (logicalProgramCountHint > 0)
    commonFactorUpperBound =
        std::min(commonFactorUpperBound, logicalProgramCountHint);
  partitionerOptions.maximumSuperblockFactor =
      (wholeKernelSuperblockMaterializable || features.autoBlockifyV1Applied)
          ? std::min(commonFactorUpperBound, maximumWholeKernelSuperblockFactor)
          : 1;
  partitionerOptions.maximumScopeSuperblockFactor =
      scopeSuperblockMaterializable
          ? std::min(commonFactorUpperBound, maximumScopeSuperblockFactor)
          : 1;
  partitionerOptions.scopeSuperblockMaterializable =
      scopeSuperblockMaterializable;
  StagePartitioner partitioner;
  if (!module || !anchorPlan)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Stage model requires PreparedTTIR and "
                                   "its anchor plan");
  auto partition =
      partitioner.partition(module, *anchorPlan, partitionerOptions);
  if (!partition)
    return partition.takeError();

  for (LogicalStage &stage : partition->stages) {
    llvm::erase_if(stage.legalSimtFactors, [&](int64_t factor) {
      return !llvm::is_contained(options.wholeKernelSuperblockFactors, factor);
    });
    llvm::erase_if(stage.localSimtFactors, [&](int64_t factor) {
      return !llvm::is_contained(options.scopeSuperblockFactors, factor);
    });
    stage.simtLegal &=
        !stage.legalSimtFactors.empty() ||
        (stage.localSimtMaterializable && !stage.localSimtFactors.empty());
  }

  HardwareProfile hardwareProfile = profile.hardware;
  hardwareProfile.baseSimtWarpCount = std::max<int64_t>(1, numWarps);
  StageCostEvaluator evaluator;
  auto costTable = evaluator.evaluate(*partition, hardwareProfile);
  if (!costTable)
    return costTable.takeError();
  costTable->logicalProgramCountHint = logicalProgramCountHint;
  costTable->physicalCoreCountHint = physicalCoreCountHint;
  costTable->physicalAiCoreCountHint = physicalAiCoreCountHint;
  auto routes = solveStageRoutes(*costTable, hardwareProfile.scopeHandoff);
  if (!routes)
    return routes.takeError();
  return std::move(*routes);
}

static llvm::SmallVector<std::pair<double, SimdSimtCandidateKind>>
legalCandidates(const SimdSimtCandidateScores &scores, bool allSimdLegal,
                bool allSimtLegal, bool mixedLegal) {
  llvm::SmallVector<std::pair<double, SimdSimtCandidateKind>> candidates;
  if (allSimdLegal)
    candidates.push_back({scores.allSimd, SimdSimtCandidateKind::AllSIMD});
  if (allSimtLegal)
    candidates.push_back(
        {scores.allSimtOnly, SimdSimtCandidateKind::AllSIMTOnly});
  if (mixedLegal)
    candidates.push_back(
        {scores.mixedSimdSimt, SimdSimtCandidateKind::MixedSIMDSIMT});
  llvm::stable_sort(candidates, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  return candidates;
}

static SimdSimtCandidateKind chooseBest(const SimdSimtCandidateScores &scores,
                                        bool allSimdLegal, bool allSimtLegal,
                                        bool mixedLegal) {
  return legalCandidates(scores, allSimdLegal, allSimtLegal, mixedLegal)
      .front()
      .second;
}

} // namespace

llvm::StringRef
mlir::ascend::stringifySimdSimtCandidate(SimdSimtCandidateKind candidate) {
  switch (candidate) {
  case SimdSimtCandidateKind::AllSIMD:
    return kAllSimd;
  case SimdSimtCandidateKind::AllSIMTOnly:
    return kAllSimtOnly;
  case SimdSimtCandidateKind::MixedSIMDSIMT:
    return kMixedSimdSimt;
  }
  llvm_unreachable("unknown SIMD/SIMT candidate");
}

llvm::json::Object SimdSimtCandidateScores::toJSON() const {
  return llvm::json::Object{{kAllSimd, allSimd},
                            {kAllSimtOnly, allSimtOnly},
                            {kMixedSimdSimt, mixedSimdSimt}};
}

static llvm::json::Object
toLowerabilityJSON(const CandidateLowerability &lowerability) {
  return llvm::json::Object{{kAllSimd, lowerability.allSimd},
                            {kAllSimtOnly, lowerability.allSimtOnly},
                            {kMixedSimdSimt, lowerability.mixed}};
}

static llvm::json::Object
toTriangularSolveFactsJSON(const TriangularSolveFacts &facts) {
  return llvm::json::Object{
      {"block_rows", facts.blockRows},
      {"block_columns", facts.blockColumns},
      {"accumulator_type", facts.accumulatorType},
      {"recurrence_start_row", facts.recurrenceStartRow},
      {"recurrence_loop_count", facts.recurrenceLoopCount},
      {"dense_dot_tail_ops", facts.denseDotTailOps},
      {"requires_cube_tail_partition", facts.requiresCubeTailPartition}};
}

llvm::json::Object SimtAnchorFeatureSummary::toJSON() const {
  llvm::json::Object result;
  result["count"] = count;
  llvm::json::Array triangularFacts;
  for (const TriangularSolveFacts &facts : triangularSolves)
    triangularFacts.push_back(toTriangularSolveFactsJSON(facts));
  result["triangular_solves"] = std::move(triangularFacts);
  result["kernel_lowerability"] = toLowerabilityJSON(kernelLowerability);
  return result;
}

llvm::json::Object SimdSimtFeatureSummary::toJSON() const {
  llvm::json::Object result;
  llvm::json::Object postTransform;
  postTransform["auto_blockify_v1_applied"] = autoBlockifyV1Applied;
  result["post_transform"] = std::move(postTransform);
  result["has_explicit_scope"] = hasExplicitScope;
  result["simt_anchors"] = simtAnchors.toJSON();
  return result;
}

llvm::json::Object SimdSimtCostReport::toJSON() const {
  llvm::json::Object result;
  result["schema_version"] = schemaVersion;
  result["model"] = model;
  result["profile_version"] = profileVersion;
  result["profile_target"] = profileTarget;
  result["actual_target"] = actualTarget;
  result["profile_content_sha256"] = profileContentSha256;
  result["selection_profile_content_sha256"] = selectionProfileContentSha256;
  llvm::json::Object sharedEvidence;
  sharedEvidence["profile_version"] = microbenchmarkProfileVersion;
  sharedEvidence["target"] = microbenchmarkProfileTarget;
  sharedEvidence["content_sha256"] = microbenchmarkProfileContentSha256;
  result["shared_microbenchmark_profile"] = std::move(sharedEvidence);
  result["unit"] = scoreUnit;
  result["candidate_costs"] = candidateCosts.toJSON();
  result["decision_kind"] = stringifySimdSimtCandidate(decision);
  llvm::json::Array selectableCandidates;
  if (allSimdCandidateLegal)
    selectableCandidates.push_back(kAllSimd);
  if (allSimtOnlyCandidateLegal)
    selectableCandidates.push_back(kAllSimtOnly);
  if (mixedCandidateLegal)
    selectableCandidates.push_back(kMixedSimdSimt);
  result["selectable_candidates"] = std::move(selectableCandidates);
  llvm::json::Array unsupportedValues;
  for (const std::string &value : unsupported)
    unsupportedValues.push_back(value);
  result["unmodeled_cost_terms"] = std::move(unsupportedValues);
  result["stage_model"] = stageModel.toJSON();

  if (includeFeaturesInJSON)
    result["features"] = features.toJSON();
  return result;
}

void SimdSimtCostReport::printJSON(llvm::raw_ostream &os, bool pretty) const {
  llvm::json::Object object = toJSON();
  if (pretty)
    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(object)));
  else
    os << llvm::json::Value(std::move(object));
}

std::string mlir::ascend::getDefaultSimdSimtProfilePath() {
  if (const char *environment = std::getenv("TRITON_ASCEND_SIMD_SIMT_PROFILE"))
    if (*environment)
      return environment;
#ifdef TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH
  return TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH;
#else
  return {};
#endif
}

llvm::Expected<SimdSimtFeatureSummary>
mlir::ascend::analyzeSimdSimtFeatures(ModuleOp module, bool compileOn91095) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");
  SimtAnchorPlan anchorPlan = buildMixedSimtAnchorPlan(module, compileOn91095);
  return analyzeSimdSimtFeatures(module, anchorPlan);
}

llvm::Expected<SimdSimtFeatureSummary>
mlir::ascend::analyzeSimdSimtFeatures(ModuleOp module,
                                      const SimtAnchorPlan &anchorPlan) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");

  SimdSimtFeatureSummary features;
  features.simtAnchors.count = llvm::count_if(
      anchorPlan.anchors,
      [](const SimtAnchorDescriptor &anchor) { return anchor.materializable; });
  features.simtAnchors.kernelLowerability = anchorPlan.kernelLowerability;

  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (anchor.triangularSolve)
      features.simtAnchors.triangularSolves.push_back(*anchor.triangularSolve);
  }

  module.walk([&](Operation *operation) {
    if (operation->hasAttr("ta.auto_blockify_v1") ||
        operation->hasAttr("ta.auto_blockify_v1.loop"))
      features.autoBlockifyV1Applied = true;
    features.hasExplicitScope |=
        operation->getName().getStringRef() == "scope.scope";
  });
  return features;
}

static llvm::Expected<SimdSimtCostReport>
estimateSimdSimtCandidatesImpl(const SimdSimtFeatureSummary &features,
                               const SimdSimtCostModelOptions &options,
                               ModuleOp module,
                               const SimtAnchorPlan *anchorPlan) {
  auto profileOrError = loadCandidateProfile(options.profilePath);
  if (!profileOrError)
    return profileOrError.takeError();
  CandidateProfile profile = std::move(*profileOrError);
  if (!options.actualTarget.empty() &&
      llvm::none_of(profile.compatibleTargets, [&](llvm::StringRef pattern) {
        return wildcardMatchInsensitive(pattern, options.actualTarget);
      }))
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT profile target '%s' is not "
                                   "compatible with actual target '%s'",
                                   profile.hardware.target.c_str(),
                                   options.actualTarget.c_str());

  SimdSimtCostReport report;
  report.profileVersion = profile.hardware.profileVersion;
  report.profileTarget = profile.hardware.target;
  report.actualTarget = options.actualTarget;
  report.profileContentSha256 = profile.contentSha256;
  report.selectionProfileContentSha256 = profile.selectionContentSha256;
  report.microbenchmarkProfileVersion = profile.microbenchmarkProfileVersion;
  report.microbenchmarkProfileTarget = profile.microbenchmarkProfileTarget;
  report.microbenchmarkProfileContentSha256 =
      profile.microbenchmarkContentSha256;
  report.scoreUnit = profile.scoreUnit;
  report.features = features;
  report.allSimdCandidateLegal =
      features.simtAnchors.kernelLowerability.allSimd;
  report.allSimtOnlyCandidateLegal =
      options.compileOn91095 && !features.hasExplicitScope &&
      features.simtAnchors.kernelLowerability.allSimtOnly;
  report.mixedCandidateLegal = !features.hasExplicitScope &&
                               options.compileOn91095 &&
                               features.simtAnchors.count > 0 &&
                               features.simtAnchors.kernelLowerability.mixed;
  report.includeFeaturesInJSON = options.includeFeaturesInJSON;

  const int64_t numWarps =
      std::max<int64_t>(1, static_cast<int64_t>(options.numWarps));
  auto stageModel = evaluateStageModel(
      features, profile, static_cast<unsigned>(numWarps),
      options.wholeKernelSuperblockMaterializable,
      options.scopeSuperblockMaterializable, options.logicalProgramCountHint,
      options.physicalVectorCoreCountHint, options.physicalAiCoreCountHint,
      module, anchorPlan, options);
  if (!stageModel)
    return stageModel.takeError();
  report.stageModel = std::move(*stageModel);
  report.candidateCosts.allSimd = report.stageModel.allSimd.totalCycles;
  report.candidateCosts.allSimtOnly = report.stageModel.allSimt.totalCycles;
  report.candidateCosts.mixedSimdSimt = report.stageModel.mixed.totalCycles;
  report.allSimdCandidateLegal &= report.stageModel.allSimd.legal;
  report.allSimtOnlyCandidateLegal &= report.stageModel.allSimt.legal;
  report.mixedCandidateLegal &= report.stageModel.mixed.legal;
  const unsigned legalCandidateCount =
      static_cast<unsigned>(report.allSimdCandidateLegal) +
      static_cast<unsigned>(report.allSimtOnlyCandidateLegal) +
      static_cast<unsigned>(report.mixedCandidateLegal);
  if (legalCandidateCount == 0)
    return llvm::createStringError(
        std::errc::not_supported,
        "Stage Route Model found no materializable candidate");
  report.decision =
      chooseBest(report.candidateCosts, report.allSimdCandidateLegal,
                 report.allSimtOnlyCandidateLegal, report.mixedCandidateLegal);
  return report;
}

llvm::Expected<SimdSimtCostReport> mlir::ascend::analyzeSimdSimtCandidates(
    ModuleOp module, const SimdSimtCostModelOptions &options) {
  if (!module)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "cannot analyze a null ModuleOp");
  SimtAnchorPlan anchorPlan =
      buildMixedSimtAnchorPlan(module, options.compileOn91095);
  return analyzeSimdSimtCandidates(module, anchorPlan, options);
}

llvm::Expected<SimdSimtCostReport> mlir::ascend::analyzeSimdSimtCandidates(
    ModuleOp module, const SimtAnchorPlan &anchorPlan,
    const SimdSimtCostModelOptions &options) {
  auto features = analyzeSimdSimtFeatures(module, anchorPlan);
  if (!features)
    return features.takeError();
  return estimateSimdSimtCandidatesImpl(*features, options, module,
                                        &anchorPlan);
}
