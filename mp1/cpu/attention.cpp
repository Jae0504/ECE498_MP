#include "cpu_kernels.h"
#include "threading.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace tlmp {

void attention_qk_baseline(const float* query, const float* key, float* scores,
                           std::size_t sequence, std::size_t head_dimension,
                           int threads) {
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_dimension));
  parallel_chunks(sequence, threads, [&](std::size_t first, std::size_t last, int) {
    for (std::size_t query_row = first; query_row < last; ++query_row) {
      const float* const q_row = query + query_row * head_dimension;
      for (std::size_t key_row = 0; key_row < sequence; ++key_row) {
        const float* const k_row = key + key_row * head_dimension;
        float sum = 0.0F;
        for (std::size_t element = 0; element < head_dimension; ++element) {
          sum += q_row[element] * k_row[element];
        }
        scores[query_row * sequence + key_row] = sum * scale;
      }
    }
  });
}

void softmax_rows_baseline(float* scores, std::size_t rows,
                           std::size_t columns, int threads) {
  parallel_chunks(rows, threads, [&](std::size_t first, std::size_t last, int) {
    for (std::size_t row = first; row < last; ++row) {
      float* const values = scores + row * columns;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::size_t column = 0; column < columns; ++column) {
        maximum = std::max(maximum, values[column]);
      }
      float sum = 0.0F;
      for (std::size_t column = 0; column < columns; ++column) {
        values[column] = std::exp(values[column] - maximum);
        sum += values[column];
      }
      const float inverse_sum = 1.0F / sum;
      for (std::size_t column = 0; column < columns; ++column) {
        values[column] *= inverse_sum;
      }
    }
  });
}

void attention_pv_baseline(const float* probabilities, const float* value,
                           float* output, std::size_t sequence,
                           std::size_t head_dimension, int threads) {
  parallel_chunks(sequence, threads, [&](std::size_t first, std::size_t last, int) {
    std::fill(output + first * head_dimension, output + last * head_dimension, 0.0F);
    for (std::size_t row = first; row < last; ++row) {
      float* const output_row = output + row * head_dimension;
      const float* const probability_row = probabilities + row * sequence;
      for (std::size_t token = 0; token < sequence; ++token) {
        const float probability = probability_row[token];
        const float* const value_row = value + token * head_dimension;
        for (std::size_t element = 0; element < head_dimension; ++element) {
          output_row[element] += probability * value_row[element];
        }
      }
    }
  });
}

}  // namespace tlmp
