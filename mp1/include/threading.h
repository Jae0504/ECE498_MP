#pragma once

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace tlmp {

inline int parse_thread_count(const std::string& value,
                              const std::string& option = "--threads") {
  if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
    throw std::invalid_argument(option + " requires a positive integer");
  }
  const auto parsed = std::stoull(value);
  if (parsed == 0 || parsed > static_cast<unsigned long long>(
                                  std::numeric_limits<int>::max())) {
    throw std::invalid_argument(option + " requires a positive integer in int range");
  }
  return static_cast<int>(parsed);
}

// Check the actual OpenMP team before allocating/timing/saving measurements.
// A restricted runtime must not silently label fewer workers as N threads.
inline void configure_threads(int threads) {
  if (threads < 1 || threads > omp_get_num_procs()) {
    throw std::invalid_argument("--threads exceeds the available OpenMP processors (" +
                                std::to_string(omp_get_num_procs()) + ")");
  }
  if (threads > omp_get_thread_limit()) {
    throw std::invalid_argument("--threads exceeds OMP_THREAD_LIMIT");
  }
  omp_set_dynamic(0);
  omp_set_num_threads(threads);
  int actual = 1;
#pragma omp parallel num_threads(threads)
  {
#pragma omp single
    actual = omp_get_num_threads();
  }
  if (actual != threads) {
    throw std::runtime_error("requested " + std::to_string(threads) +
                             " threads, but OpenMP created " + std::to_string(actual));
  }
}

// Contiguous disjoint ranges, including tails and empty ranges. The serial
// path avoids entering an OpenMP region. All parallel work joins on return.
template <typename Function>
void parallel_chunks(std::size_t elements, int threads, Function function) {
  if (threads == 1) {
    function(0, elements, 0);
    return;
  }
#pragma omp parallel num_threads(threads)
  {
    const auto workers = static_cast<std::size_t>(omp_get_num_threads());
    const auto worker = static_cast<std::size_t>(omp_get_thread_num());
    const auto count = elements / workers;
    const auto tail = elements % workers;
    const auto first = worker * count + std::min(worker, tail);
    const auto last = first + count + (worker < tail ? 1 : 0);
    function(first, last, static_cast<int>(worker));
  }
}

}  // namespace tlmp
