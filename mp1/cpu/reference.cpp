#include "cpu_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tlmp {

void gemm_reference(const float* a, const float* b, float* c, std::size_t m,
                    std::size_t n, std::size_t k) {
  for (std::size_t row = 0; row < m; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      double sum = 0.0;
      for (std::size_t inner = 0; inner < k; ++inner) {
        sum += static_cast<double>(a[row * k + inner]) *
               static_cast<double>(b[inner * n + column]);
      }
      c[row * n + column] = static_cast<float>(sum);
    }
  }
}

void gemv_reference(const float* x, const float* weights, float* y,
                    std::size_t n, std::size_t k) {
  for (std::size_t column = 0; column < n; ++column) {
    double sum = 0.0;
    for (std::size_t inner = 0; inner < k; ++inner) {
      sum += static_cast<double>(x[inner]) *
             static_cast<double>(weights[inner * n + column]);
    }
    y[column] = static_cast<float>(sum);
  }
}

void attention_reference(const float* query, const float* key,
                         const float* value, float* output,
                         std::size_t sequence, std::size_t head_dimension) {
  std::vector<double> probabilities(sequence * sequence, 0.0);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dimension));

  for (std::size_t query_row = 0; query_row < sequence; ++query_row) {
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::size_t key_row = 0; key_row < sequence; ++key_row) {
      double dot = 0.0;
      for (std::size_t element = 0; element < head_dimension; ++element) {
        dot += static_cast<double>(
                   query[query_row * head_dimension + element]) *
               static_cast<double>(key[key_row * head_dimension + element]);
      }
      const double score = dot * scale;
      probabilities[query_row * sequence + key_row] = score;
      maximum = std::max(maximum, score);
    }

    double sum = 0.0;
    for (std::size_t key_row = 0; key_row < sequence; ++key_row) {
      double& probability =
          probabilities[query_row * sequence + key_row];
      probability = std::exp(probability - maximum);
      sum += probability;
    }
    for (std::size_t key_row = 0; key_row < sequence; ++key_row) {
      probabilities[query_row * sequence + key_row] /= sum;
    }
  }

  for (std::size_t row = 0; row < sequence; ++row) {
    for (std::size_t element = 0; element < head_dimension; ++element) {
      double sum = 0.0;
      for (std::size_t token = 0; token < sequence; ++token) {
        sum += probabilities[row * sequence + token] *
               static_cast<double>(value[token * head_dimension + element]);
      }
      output[row * head_dimension + element] = static_cast<float>(sum);
    }
  }
}

}  // namespace tlmp

