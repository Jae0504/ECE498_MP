#include "cpu_kernels.h"
#include "threading.h"

#include <algorithm>
#include <cstddef>

namespace tlmp {

void gemv_baseline(const float* x, const float* weights, float* y,
                   std::size_t n, std::size_t k, int threads) {
  // W is [K x N]. Partition output columns, retaining a contiguous inner loop.
  // Parallelizing K directly would race on y and require a reduction.
  parallel_chunks(n, threads, [&](std::size_t first, std::size_t last, int) {
    std::fill(y + first, y + last, 0.0F);
    for (std::size_t inner = 0; inner < k; ++inner) {
      const float x_value = x[inner];
      const float* const weight_row = weights + inner * n;
      for (std::size_t column = first; column < last; ++column) {
        y[column] += x_value * weight_row[column];
      }
    }
  });
}

}  // namespace tlmp
