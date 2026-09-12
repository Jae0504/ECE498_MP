#include "cpu_kernels.h"

#include "common.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace tlmp {
namespace {

bool compare_vectors(const std::string& name, const std::vector<float>& actual,
                     const std::vector<float>& expected,
                     float absolute_tolerance, float relative_tolerance,
                     bool verbose) {
  if (actual.size() != expected.size()) {
    std::cerr << "[FAIL] " << name << ": size mismatch\n";
    return false;
  }

  float maximum_absolute_error = 0.0F;
  float maximum_relative_error = 0.0F;
  std::size_t worst_index = 0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const float absolute_error = std::fabs(actual[index] - expected[index]);
    const float denominator = std::max(std::fabs(expected[index]), 1.0e-12F);
    const float relative_error = absolute_error / denominator;
    if (absolute_error > maximum_absolute_error) {
      maximum_absolute_error = absolute_error;
      maximum_relative_error = relative_error;
      worst_index = index;
    }
    if (!std::isfinite(actual[index]) ||
        !close_enough(actual[index], expected[index], absolute_tolerance,
                      relative_tolerance)) {
      std::cerr << "[FAIL] " << name << " at index " << index
                << ": actual=" << actual[index]
                << ", reference=" << expected[index]
                << ", absolute_error=" << absolute_error << '\n';
      return false;
    }
  }

  if (verbose) {
    std::cout << "[PASS] " << name << " (max_abs_error="
              << maximum_absolute_error << ", max_rel_error="
              << maximum_relative_error << ", worst_index=" << worst_index
              << ")\n";
  }
  return true;
}

bool test_gemm(bool verbose, int threads, std::size_t m) {
  const std::size_t n = m + 5, k = m + 3;
  std::vector<float> a(m * k), b(k * n), actual(m * n), expected(m * n);
  fill_random(a, kRandomSeed + 1U);
  fill_random(b, kRandomSeed + 2U);
  for (int repeat = 0; repeat < 2; ++repeat) {
    gemm_baseline(a.data(), b.data(), actual.data(), m, n, k, threads);
  }
  gemm_reference(a.data(), b.data(), expected.data(), m, n, k);
  return compare_vectors("GEMM M=" + std::to_string(m), actual, expected,
                         2.0e-5F, 2.0e-5F, verbose);
}

bool test_gemv(bool verbose, int threads, std::size_t n) {
  const std::size_t k = n + 3;
  std::vector<float> x(k), weights(k * n), actual(n), expected(n);
  fill_random(x, kRandomSeed + 3U);
  fill_random(weights, kRandomSeed + 4U);
  for (int repeat = 0; repeat < 2; ++repeat) {
    gemv_baseline(x.data(), weights.data(), actual.data(), n, k, threads);
  }
  gemv_reference(x.data(), weights.data(), expected.data(), n, k);
  return compare_vectors("GEMV N=" + std::to_string(n), actual, expected,
                         2.0e-5F, 2.0e-5F, verbose);
}

bool test_attention(bool verbose, int threads, std::size_t sequence) {
  const std::size_t dimension = sequence + 3;
  std::vector<float> query(sequence * dimension), key(sequence * dimension);
  std::vector<float> value(sequence * dimension), scores(sequence * sequence);
  std::vector<float> actual(sequence * dimension), expected(sequence * dimension);
  fill_random(query, kRandomSeed + 5U);
  fill_random(key, kRandomSeed + 6U);
  fill_random(value, kRandomSeed + 7U);
  for (int repeat = 0; repeat < 2; ++repeat) {
    attention_qk_baseline(query.data(), key.data(), scores.data(),
                          sequence, dimension, threads);
    softmax_rows_baseline(scores.data(), sequence, sequence, threads);
    attention_pv_baseline(scores.data(), value.data(), actual.data(),
                          sequence, dimension, threads);
  }
  attention_reference(query.data(), key.data(), value.data(), expected.data(),
                      sequence, dimension);
  for (std::size_t row = 0; row < sequence; ++row) {
    double row_sum = 0.0;
    for (std::size_t column = 0; column < sequence; ++column) {
      const float probability = scores[row * sequence + column];
      if (!std::isfinite(probability) || probability < 0.0F) {
        std::cerr << "[FAIL] Invalid softmax probability\n";
        return false;
      }
      row_sum += probability;
    }
    if (std::fabs(row_sum - 1.0) > 2.0e-5) {
      std::cerr << "[FAIL] Attention softmax row " << row << " sums to " << row_sum << '\n';
      return false;
    }
  }
  return compare_vectors("Attention S=" + std::to_string(sequence), actual,
                         expected, 3.0e-5F, 3.0e-5F, verbose);
}

}  // namespace

bool run_correctness_tests(bool verbose, int threads) {
  // Non-square shapes, vector tails, and fewer rows/columns than workers.
  for (std::size_t size : {3U, 8U, 19U}) {
    const bool gemm_ok = test_gemm(verbose, threads, size);
    const bool gemv_ok = test_gemv(verbose, threads, size);
    const bool attention_ok = test_attention(verbose, threads, size);
    if (!(gemm_ok && gemv_ok && attention_ok)) {
      std::cerr << "CPU correctness suite FAILED\n";
      return false;
    }
  }
  if (verbose) {
    std::cout << "All CPU correctness tests passed (threads=" << threads << ").\n";
  }
  return true;
}

}  // namespace tlmp
