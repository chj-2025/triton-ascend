//===- SuperBlockFactor.h - Supported SuperBlock factors -------*- C++ -*-===//

#ifndef TRITON_ASCEND_UTILS_SUPERBLOCKFACTOR_H
#define TRITON_ASCEND_UTILS_SUPERBLOCKFACTOR_H

#include <array>
#include <cstdint>

namespace mlir::ascend {

inline constexpr std::array<int64_t, 7> kSupportedSuperBlockFactors = {
    1, 2, 4, 8, 16, 32, 64};
inline constexpr int64_t kMaximumSuperBlockFactor =
    kSupportedSuperBlockFactors.back();
inline constexpr const char *kSupportedSuperBlockFactorsDescription =
    "1, 2, 4, 8, 16, 32 or 64";
inline constexpr std::array<int64_t, 3> kSupportedScopeSuperBlockFactors = {
    1, 2, 4};

constexpr bool isSupportedSuperBlockFactor(int64_t factor) {
  for (int64_t supported : kSupportedSuperBlockFactors)
    if (factor == supported)
      return true;
  return false;
}

constexpr bool isSupportedScopeSuperBlockFactor(int64_t factor) {
  for (int64_t supported : kSupportedScopeSuperBlockFactors)
    if (factor == supported)
      return true;
  return false;
}

} // namespace mlir::ascend

#endif // TRITON_ASCEND_UTILS_SUPERBLOCKFACTOR_H
