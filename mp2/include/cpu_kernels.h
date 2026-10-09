#pragma once

#include <cstddef>

namespace tlmp {

// Educational baselines. All matrices are row-major float32 arrays.
void gemm_baseline(const float* a, const float* b, float* c, std::size_t m,
                   std::size_t n, std::size_t k);
void gemv_baseline(const float* x, const float* weights, float* y,
                   std::size_t n, std::size_t k);
void attention_qk_baseline(const float* query, const float* key, float* scores,
                           std::size_t sequence, std::size_t head_dimension);
void softmax_rows_baseline(float* scores, std::size_t rows,
                           std::size_t columns);
void attention_pv_baseline(const float* probabilities, const float* value,
                           float* output, std::size_t sequence,
                           std::size_t head_dimension);

// Independent, deliberately differently ordered correctness references. They
// accumulate in double precision.
void gemm_reference(const float* a, const float* b, float* c, std::size_t m,
                    std::size_t n, std::size_t k);
void gemv_reference(const float* x, const float* weights, float* y,
                    std::size_t n, std::size_t k);
void attention_reference(const float* query, const float* key,
                         const float* value, float* output,
                         std::size_t sequence, std::size_t head_dimension,
                         float* probabilities = nullptr,
                         float* scaled_scores = nullptr);

bool run_correctness_tests(bool verbose);

}  // namespace tlmp
