#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tlmp {

struct TimingStats {
  double median_ms = 0.0;
  double minimum_ms = 0.0;
  double maximum_ms = 0.0;
  double relative_range_percent = 0.0;
  std::vector<double> samples_ms;
};

inline double median(std::vector<double> samples) {
  if (samples.empty()) {
    throw std::invalid_argument("cannot compute median of no samples");
  }
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  if ((samples.size() % 2U) == 0U) {
    return 0.5 * (samples[middle - 1] + samples[middle]);
  }
  return samples[middle];
}

inline TimingStats summarize_timings(std::vector<double> samples_ms) {
  if (samples_ms.empty()) {
    throw std::invalid_argument("at least one timing iteration is required");
  }
  TimingStats stats;
  stats.median_ms = median(samples_ms);
  const auto limits = std::minmax_element(samples_ms.begin(), samples_ms.end());
  stats.minimum_ms = *limits.first;
  stats.maximum_ms = *limits.second;
  if (stats.median_ms > 0.0) {
    stats.relative_range_percent =
        100.0 * (stats.maximum_ms - stats.minimum_ms) / stats.median_ms;
  }
  stats.samples_ms = std::move(samples_ms);
  return stats;
}

template <typename Preparation, typename Function>
TimingStats measure_median_with_preparation(int warmup_iterations,
                                            int timed_iterations,
                                            Preparation&& prepare,
                                            Function&& function);

template <typename Function>
TimingStats measure_median(int warmup_iterations, int timed_iterations,
                           Function&& function) {
  const auto no_preparation = [] {};
  return measure_median_with_preparation(
      warmup_iterations, timed_iterations, no_preparation,
      std::forward<Function>(function));
}

template <typename Preparation, typename Function>
TimingStats measure_median_with_preparation(int warmup_iterations,
                                            int timed_iterations,
                                            Preparation&& prepare,
                                            Function&& function) {
  if (warmup_iterations < 0 || timed_iterations <= 0) {
    throw std::invalid_argument("warmups must be nonnegative and iterations positive");
  }
  for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
    prepare();
    function();
  }

  using Clock = std::chrono::steady_clock;
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(timed_iterations));
  for (int iteration = 0; iteration < timed_iterations; ++iteration) {
    prepare();
    const auto start = Clock::now();
    function();
    const auto stop = Clock::now();
    samples.push_back(
        std::chrono::duration<double, std::milli>(stop - start).count());
  }
  return summarize_timings(std::move(samples));
}

}  // namespace tlmp
