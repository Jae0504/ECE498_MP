#include "cpu_kernels.h"
#include "threading.h"

#include <algorithm>
#include <cstddef>

namespace tlmp {

void gemm_baseline(const float* a, const float* b, float* c, std::size_t m,
                   std::size_t n, std::size_t k, int threads) {
  // Each worker owns complete output rows. The i-k-j arithmetic and the
  // contiguous inner loop are identical at every thread count.
  parallel_chunks(m, threads, [&](std::size_t first, std::size_t last, int) {
    std::fill(c + first * n, c + last * n, 0.0F);
    for (std::size_t row = first; row < last; ++row) {
      float* const c_row = c + row * n;
      const float* const a_row = a + row * k;
      for (std::size_t inner = 0; inner < k; ++inner) {
        const float a_value = a_row[inner];
        const float* const b_row = b + inner * n;
        for (std::size_t column = 0; column < n; ++column) {
          c_row[column] += a_value * b_row[column];
        }
      }
    }
  });
}

}  // namespace tlmp
