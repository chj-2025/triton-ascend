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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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
#include <limits>
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

struct StructuralProfile {
  int64_t tinyDotFlopsMax = 0;
};

struct CandidateProfile {
  HardwareProfile hardware;
  std::string scoreUnit;
  std::string contentSha256;
  std::string selectionContentSha256;
  std::string microbenchmarkProfileVersion;
  std::string microbenchmarkProfileTarget;
  std::string microbenchmarkContentSha256;
  StructuralProfile structural;
};

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

static std::optional<int64_t> getConstantInteger(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp || definingOp->getName().getStringRef() != "arith.constant")
    return std::nullopt;
  if (auto integer = definingOp->getAttrOfType<IntegerAttr>("value"))
    return integer.getInt();
  return std::nullopt;
}

static std::optional<int64_t> getKnownStaticLoopTripCount(Operation *op) {
  if (!op || op->getName().getStringRef() != "scf.for" ||
      op->getNumOperands() < 3)
    return std::nullopt;
  auto lower = getConstantInteger(op->getOperand(0));
  auto upper = getConstantInteger(op->getOperand(1));
  auto step = getConstantInteger(op->getOperand(2));
  if (!lower || !upper || !step || *step == 0)
    return std::nullopt;
  int64_t span = *upper - *lower;
  if (span > 0 && *step > 0)
    return std::max<int64_t>(1, (span + *step - 1) / *step);
  if (span < 0 && *step < 0) {
    int64_t positiveSpan = -span;
    int64_t positiveStep = -*step;
    return std::max<int64_t>(1,
                             (positiveSpan + positiveStep - 1) / positiveStep);
  }
  return std::nullopt;
}

static int64_t getModeledLoopTripCount(
    Operation *op,
    const llvm::DenseMap<Operation *, int64_t> &structuralTripEstimates) {
  if (auto knownTripCount = getKnownStaticLoopTripCount(op))
    return *knownTripCount;
  if (auto iterator = structuralTripEstimates.find(op);
      iterator != structuralTripEstimates.end())
    return iterator->second;
  return 1;
}

static int64_t getLoopMultiplier(
    Operation *op,
    const llvm::DenseMap<Operation *, int64_t> &structuralTripEstimates) {
  int64_t multiplier = 1;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getName().getStringRef() != "scf.for")
      continue;
    // AutoBlockify V1 wraps the original logical program in a physical-core
    // scheduling loop.  Its dispatch cost is modeled as a separate phase;
    // it must not multiply every algorithm operation as if it were an
    // algorithmic loop.
    if (parent->hasAttr("ta.auto_blockify_v1.schedule"))
      continue;
    int64_t tripCount =
        getModeledLoopTripCount(parent, structuralTripEstimates);
    if (tripCount > 0 &&
        multiplier <= std::numeric_limits<int64_t>::max() / tripCount)
      multiplier *= tripCount;
  }
  return multiplier;
}

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
    result.factor = reader.optionalNumber(*op, "factor", 1.0);
    return result;
  }
  result.throughput =
      resolveNumberOrMeasurement(*op, throughputKey, "throughput_measurement",
                                 expectedUnit, microbench, reader, opName);
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

static void readStageResources(ProfileJSONReader &reader,
                               const llvm::json::Object &mode,
                               llvm::StringRef context,
                               StageModeProfile &profile) {
  const auto *resources = reader.object(mode, "stage_resources", context);
  if (!resources)
    return;
  const std::string prefix = (context + ".stage_resources").str();
  profile.scalarOperationsPerCycle =
      reader.number(*resources, "scalar_operations_per_system_cycle", prefix);
  profile.issueOperationsPerCycle =
      reader.number(*resources, "issue_instructions_per_system_cycle", prefix);
  profile.spillTransactionsPerCycle =
      reader.number(*resources, "spill_transactions_per_system_cycle", prefix);
  if (const auto *indirect =
          reader.object(*resources, "indirect_memory", prefix)) {
    const std::string path = prefix + ".indirect_memory";
    profile.indirectLoadTransactionsPerCycle =
        reader.number(*indirect, "load_transactions_per_system_cycle", path);
    profile.indirectStoreTransactionsPerCycle =
        reader.number(*indirect, "store_transactions_per_system_cycle", path);
    profile.indirectDependencyLatencyCycles =
        reader.number(*indirect, "dependency_latency_system_cycles", path);
  }
  if (const auto *control = reader.object(*resources, "control_flow", prefix)) {
    const std::string path = prefix + ".control_flow";
    profile.controlFlow.loopBackedgeCycles =
        reader.number(*control, "loop_backedge_system_cycles", path);
    profile.controlFlow.conditionalBranchCycles =
        reader.number(*control, "conditional_branch_system_cycles", path);
    profile.controlFlow.divergentBranchPenaltyCycles =
        reader.number(*control, "divergent_branch_penalty_system_cycles", path);
    profile.controlFlow.synchronizationCycles =
        reader.number(*control, "synchronization_system_cycles", path);
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
  if (!selectionSchemaVersion || *selectionSchemaVersion != 10)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "SIMD/SIMT profile schema_version must be 10");

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
  if (microbench && llvm::StringRef(hardware.target) != microbench->getTarget())
    reader.setError("selection profile target '" + hardware.target +
                    "' does not match shared microbenchmark target '" +
                    microbench->getTarget().str() + "'");
  profile.scoreUnit = reader.string(*root, "score_unit", "profile");
  const auto *calibration =
      reader.object(*root, "selection_calibration", "profile");
  if (calibration) {
    if (const auto *structural =
            reader.object(*calibration, "simd_structural_penalty_ratio",
                          "profile.selection_calibration")) {
      profile.structural.tinyDotFlopsMax =
          reader.integer(*structural, "tiny_dot_flops_max", "structural");
    }
  }

  const auto *simd = reader.object(*root, "simd", "profile");
  if (simd) {
    if (simd->getString("vector_width_measurement")) {
      hardware.simd.vectorWidth = std::max<int64_t>(
          1, static_cast<int64_t>(std::llround(resolveNumberOrMeasurement(
                 *simd, "vector_width_bits", "vector_width_measurement", "bit",
                 microbench, reader, "simd"))) /
                 32);
    } else {
      hardware.simd.vectorWidth = std::max<int64_t>(
          1, reader.integer(*simd, "vector_width_bits", "simd") / 32);
    }
    hardware.simd.issueWidth = hardware.simd.vectorWidth;
    if (const auto *startup =
            reader.object(*simd, "startup_system_cycles", "simd"))
      hardware.simd.setupCycles =
          reader.number(*startup, "vector", "simd.startup_system_cycles");
    if (const auto *ops = reader.object(*simd, "ops", "simd")) {
      for (llvm::StringRef op :
           {"f32.add", "f32.sub", "f32.mul", "f32.div", "f32.max", "f32.abs",
            "f32.exp", "f32.log", "predicate.cmp", "predicate.select",
            "convert.cast", "f32.clamp"})
        hardware.simd.operationRates[op] = resolveOpProfile(
            *ops, op, "throughput_vector_instructions_per_system_cycle",
            "vector_instruction/system_cycle", microbench, reader);
    }
    if (const auto *memory = reader.object(*simd, "memory", "simd")) {
      hardware.simd.loadBytesPerCycle = reader.number(
          *memory, "vector_mte2_bytes_per_system_cycle", "simd.memory");
      hardware.simd.storeBytesPerCycle =
          reader.number(*memory, "mte3_bytes_per_system_cycle", "simd.memory");
    }
    if (const auto *dot = reader.object(*simd, "dot", "simd")) {
      hardware.simd.dotSetupCycles =
          reader.number(*dot, "startup_system_cycles", "simd.dot");
      hardware.simd.dotFlopsPerCycle =
          reader.number(*dot, "flops_per_system_cycle", "simd.dot");
    }
    readStageResources(reader, *simd, "simd", hardware.simd);
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
    if (const auto *setup =
            reader.object(*simt, "setup_system_cycles", "simt")) {
      hardware.simt.setupCycles = resolveNumberOrMeasurement(
          *setup, "empty_launch", "empty_launch_measurement", "system_cycle",
          microbench, reader, "simt.setup_system_cycles");
    }
    if (const auto *ops = reader.object(*simt, "ops", "simt")) {
      for (llvm::StringRef op :
           {"f32.add", "f32.sub", "f32.mul", "f32.div", "f32.max", "f32.abs",
            "f32.exp", "f32.log", "predicate.cmp", "predicate.select",
            "convert.cast", "f32.clamp"})
        hardware.simt.operationRates[op] =
            resolveOpProfile(*ops, op, "throughput_scalar_ops_per_system_cycle",
                             "scalar_op/system_cycle", microbench, reader);
    }
    if (const auto *dot = reader.object(*simt, "dot", "simt")) {
      hardware.simt.dotSetupCycles =
          reader.number(*dot, "startup_system_cycles", "simt.dot");
      hardware.simt.dotFlopsPerCycle =
          reader.number(*dot, "flops_per_system_cycle", "simt.dot");
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
    }
    readStageResources(reader, *simt, "simt", hardware.simt);
    if (const auto *resources = simt->getObject("stage_resources")) {
      if (const auto *superblock = resources->getObject("superblock")) {
        hardware.superblockUsefulFactorLimit =
            reader.integer(*superblock, "useful_factor_limit", "superblock");
        hardware.superblockPersistentStatePressureFreeFactor = reader.integer(
            *superblock, "persistent_state_pressure_free_factor", "superblock");
        hardware.superblockPersistentStateBytesPerCycle = reader.number(
            *superblock, "persistent_state_bytes_per_system_cycle",
            "superblock");
      }
      if (const auto *handoff = resources->getObject("scope_handoff")) {
        hardware.transition.simdToSimtCycles =
            hardware.transition.simtToSimdCycles = reader.number(
                *handoff, "fixed_directional_system_cycles", "scope_handoff");
        hardware.transition.simdUbLoadBytesPerCycle = reader.number(
            *handoff, "simd_ub_load_bytes_per_system_cycle", "scope_handoff");
        hardware.transition.simdUbStoreBytesPerCycle = reader.number(
            *handoff, "simd_ub_store_bytes_per_system_cycle", "scope_handoff");
        hardware.transition.simtUbLoadBytesPerThreadPerCycle = reader.number(
            *handoff, "simt_ub_load_bytes_per_thread_per_system_cycle",
            "scope_handoff");
        hardware.transition.simtUbStoreBytesPerThreadPerCycle = reader.number(
            *handoff, "simt_ub_store_bytes_per_thread_per_system_cycle",
            "scope_handoff");
      }
    }
    hardware.transition.simtWarpSize = hardware.simt.issueWidth;
  }

  if (reader.failed())
    return llvm::createStringError(
        std::errc::invalid_argument, "invalid SIMD/SIMT profile '%s': %s",
        path.c_str(), reader.getError().str().c_str());
  if (hardware.profileVersion != "david-v100-simd-simt-20260824-v19")
    return llvm::createStringError(
        std::errc::invalid_argument,
        "unsupported SIMD/SIMT profile version '%s' "
        "(expected david-v100-simd-simt-20260824-v19)",
        hardware.profileVersion.c_str());
  if (!microbench)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "SIMD/SIMT v19 profile must reference "
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

static llvm::Expected<std::optional<StageCostModelSummary>> evaluateStageModel(
    const SimdSimtFeatureSummary &features, const CandidateProfile &profile,
    unsigned numWarps, bool wholeKernelSuperblockMaterializable,
    bool scopeSuperblockMaterializable, int64_t logicalProgramCountHint,
    int64_t physicalCoreCountHint, ModuleOp module,
    const SimtAnchorPlan *anchorPlan) {
  StagePartitionerOptions partitionerOptions;
  partitionerOptions.tinyDotFlopsMax = profile.structural.tinyDotFlopsMax;
  partitionerOptions.maximumSuperblockFactor =
      (wholeKernelSuperblockMaterializable || scopeSuperblockMaterializable ||
       features.autoBlockifyV1Applied)
          ? 4
          : 1;
  const int64_t warpLimitedMaximum = numWarps <= 16   ? 4
                                     : numWarps <= 32 ? 2
                                                      : 1;
  partitionerOptions.maximumSuperblockFactor =
      std::min(partitionerOptions.maximumSuperblockFactor, warpLimitedMaximum);
  if (logicalProgramCountHint > 0) {
    const int64_t runtimeMaximum = logicalProgramCountHint >= 4   ? 4
                                   : logicalProgramCountHint >= 2 ? 2
                                                                  : 1;
    partitionerOptions.maximumSuperblockFactor =
        std::min(partitionerOptions.maximumSuperblockFactor, runtimeMaximum);
  }
  partitionerOptions.scopeSuperblockMaterializable =
      scopeSuperblockMaterializable;
  StagePartitioner partitioner;
  if (!module || !anchorPlan)
    return llvm::createStringError(std::errc::invalid_argument,
                                   "Stage model requires PreparedTTIR and "
                                   "its anchor plan");
  auto partition =
      partitioner.partition(module, *anchorPlan, features, partitionerOptions);
  if (!partition)
    return partition.takeError();
  if (!*partition)
    return std::optional<StageCostModelSummary>{};

  HardwareProfile hardwareProfile = profile.hardware;
  hardwareProfile.logicalWarpGroupCount = std::max<int64_t>(1, numWarps);
  StageCostEvaluator evaluator;
  auto costTable = evaluator.evaluate(**partition, hardwareProfile);
  if (!costTable)
    return costTable.takeError();
  costTable->logicalProgramCountHint = logicalProgramCountHint;
  costTable->physicalCoreCountHint = physicalCoreCountHint;
  auto routes = solveStageRoutes(*costTable, hardwareProfile.transition);
  if (!routes)
    return routes.takeError();
  return std::optional<StageCostModelSummary>{std::move(*routes)};
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
  result["conditional_branch_count"] = conditionalBranchCount;
  result["divergent_branch_count"] = divergentBranchCount;
  result["active_lane_ratio"] = activeLaneRatio;
  llvm::json::Array triangularFacts;
  for (const TriangularSolveFacts &facts : triangularSolves)
    triangularFacts.push_back(toTriangularSolveFactsJSON(facts));
  result["triangular_solves"] = std::move(triangularFacts);
  result["kernel_lowerability"] = toLowerabilityJSON(kernelLowerability);
  return result;
}

llvm::json::Object SimdSimtFeatureSummary::toJSON() const {
  llvm::json::Object result;
  result["load_ops"] = loadOps;
  result["store_ops"] = storeOps;
  result["reduce_ops"] = reduceOps;
  result["dot_ops"] = dotOps;
  result["loaded_index_dependent_memory_ops"] = loadedIndexDependentMemoryOps;
  result["dot_flops"] = dotFlops;
  result["static_loop_trip_count_max"] = staticLoopTripCountMax;
  result["conditional_branch_count"] = conditionalBranchCount;
  result["divergent_branch_count"] = divergentBranchCount;
  result["active_lane_ratio"] = activeLaneRatio;
  llvm::json::Object postTransform;
  postTransform["auto_blockify_v1_applied"] = autoBlockifyV1Applied;
  postTransform["auto_blockify_v1_loop_count"] = autoBlockifyV1LoopCount;
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
  llvm::DenseSet<Operation *> anchorRoots;
  llvm::DenseMap<Operation *, int64_t> structuralTripEstimates;
  features.simtAnchors.count = llvm::count_if(
      anchorPlan.anchors,
      [](const SimtAnchorDescriptor &anchor) { return anchor.materializable; });
  features.simtAnchors.kernelLowerability = anchorPlan.kernelLowerability;

  for (const SimtAnchorDescriptor &anchor : anchorPlan.anchors) {
    if (anchor.triangularSolve)
      features.simtAnchors.triangularSolves.push_back(*anchor.triangularSolve);
    if (!anchor.materializable)
      continue;
    for (Operation *operation : anchor.scopeOperations)
      if (operation)
        anchorRoots.insert(operation);
    if (anchor.scopeOperations.empty() && anchor.operation)
      anchorRoots.insert(anchor.operation);
    if (anchor.kind == SimtAnchorKind::TriangularSolveLoop)
      for (Operation *operation : anchor.scopeOperations)
        if (operation && operation->getName().getStringRef() == "scf.for")
          structuralTripEstimates[operation] = 14;
  }

  auto isInAnchor = [&](Operation *operation) {
    for (; operation; operation = operation->getParentOp())
      if (anchorRoots.contains(operation))
        return true;
    return false;
  };
  module.walk([&](Operation *operation) {
    if (operation->hasAttr("ta.auto_blockify_v1"))
      features.autoBlockifyV1Applied = true;
    if (operation->hasAttr("ta.auto_blockify_v1.loop")) {
      features.autoBlockifyV1Applied = true;
      ++features.autoBlockifyV1LoopCount;
    }
    if (operation->hasAttr("ta.auto_blockify_v1.schedule"))
      return;

    llvm::StringRef name = operation->getName().getStringRef();
    const bool inAnchor = isInAnchor(operation);
    const int64_t multiplier =
        getLoopMultiplier(operation, structuralTripEstimates);
    features.hasExplicitScope |= name == "scope.scope";
    features.loadOps += name == "tt.load";
    features.storeOps += name == "tt.store";
    features.reduceOps += name == "tt.reduce";
    features.dotOps += name == "tt.dot";

    if (name == "scf.if" || name == "cf.cond_br") {
      ++features.conditionalBranchCount;
      if (inAnchor)
        ++features.simtAnchors.conditionalBranchCount;
      const bool divergent =
          operation->getNumOperands() > 0 &&
          isa<RankedTensorType>(operation->getOperand(0).getType());
      features.divergentBranchCount += divergent;
      if (inAnchor)
        features.simtAnchors.divergentBranchCount += divergent;
    }
    if (name == "scf.for") {
      const int64_t trip =
          getModeledLoopTripCount(operation, structuralTripEstimates);
      features.staticLoopTripCountMax =
          std::max(features.staticLoopTripCountMax, trip);
    }
    if (isLoadedIndexDependentMemoryOp(operation))
      ++features.loadedIndexDependentMemoryOps;
    if (name == "tt.dot" && operation->getNumOperands() >= 2) {
      auto lhs = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
      auto rhs = dyn_cast<RankedTensorType>(operation->getOperand(1).getType());
      if (lhs && rhs && lhs.getRank() >= 2 && rhs.getRank() >= 2) {
        const int64_t m = lhs.getShape()[lhs.getRank() - 2];
        const int64_t k = lhs.getShape()[lhs.getRank() - 1];
        const int64_t n = rhs.getShape()[rhs.getRank() - 1];
        if (m > 0 && n > 0 && k > 0)
          features.dotFlops += 2 * m * n * k * multiplier;
      }
    }
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
      options.physicalVectorCoreCountHint, module, anchorPlan);
  if (!stageModel)
    return stageModel.takeError();
  if (*stageModel) {
    report.stageModel = std::move(**stageModel);
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
    report.decision = chooseBest(
        report.candidateCosts, report.allSimdCandidateLegal,
        report.allSimtOnlyCandidateLegal, report.mixedCandidateLegal);
    return report;
  }

  // Unknown Stage domains are deliberately not scored. The online Route
  // Model has no legacy whole-kernel fallback; the selector leaves the
  // existing backend-default lowering unchanged.
  report.stageModel.applied = false;
  report.unsupported.push_back("stage_model_not_applicable");
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
