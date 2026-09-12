#pragma once

#include "threading.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tlmp {

inline constexpr std::uint32_t kRandomSeed = 0x4982026u;
inline volatile double g_observable_checksum = 0.0;

inline std::size_t checked_elements(std::size_t a, std::size_t b) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    throw std::overflow_error("tensor element count overflow");
  }
  return a * b;
}

inline void fill_random(std::vector<float>& values, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-0.5F, 0.5F);
  for (float& value : values) {
    value = distribution(generator);
  }
}

inline double checksum(const std::vector<float>& values) {
  // Sample every element. This happens outside timed regions and prevents the
  // optimizer from treating benchmark results as dead stores.
  double sum = 0.0;
  double compensation = 0.0;
  for (float value : values) {
    const double adjusted = static_cast<double>(value) - compensation;
    const double next = sum + adjusted;
    compensation = (next - sum) - adjusted;
    sum = next;
  }
  g_observable_checksum = sum;
  return sum;
}

inline bool close_enough(float actual, float expected, float absolute_tolerance,
                         float relative_tolerance) {
  const float difference = std::fabs(actual - expected);
  const float scale = std::max(std::fabs(actual), std::fabs(expected));
  return difference <= absolute_tolerance + relative_tolerance * scale;
}

class CacheFlusher {
 public:
  explicit CacheFlusher(std::size_t mebibytes, int threads = 1)
      : threads_(threads) {
    constexpr std::size_t bytes_per_mib = 1024U * 1024U;
    if (mebibytes != 0U &&
        mebibytes > std::numeric_limits<std::size_t>::max() / bytes_per_mib) {
      throw std::overflow_error("cache-flush allocation size overflow");
    }
    values_.assign(mebibytes * bytes_per_mib / sizeof(float), 0.125F);
  }

  void touch() const {
    // One float per conventional 64-byte cache line is enough to bring every
    // line into the hierarchy. This work is deliberately outside timed regions.
    constexpr std::size_t floats_per_cache_line = 64U / sizeof(float);
    if (values_.empty()) {
      return;
    }
    std::vector<double> partial(static_cast<std::size_t>(threads_), 0.0);
    parallel_chunks(values_.size() / floats_per_cache_line, threads_,
                    [&](std::size_t first, std::size_t last, int worker) {
      double local = 0.0;
      for (std::size_t line = first; line < last; ++line) {
        local += values_[line * floats_per_cache_line];
      }
      partial[worker] = local;
    });
    double sum = 0.0;
    for (double value : partial) {
      sum += value;
    }
    g_observable_checksum = sum;
  }

  bool enabled() const { return !values_.empty(); }

 private:
  int threads_;
  std::vector<float> values_;
};

inline std::string scientific(double value, int precision = 6) {
  std::ostringstream stream;
  stream << std::scientific << std::setprecision(precision) << value;
  return stream.str();
}

}  // namespace tlmp
