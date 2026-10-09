#include "gpu_kernels.cuh"

#include <cuda_runtime.h>

namespace tlmp::gpu {
namespace {

__global__ void gemm_kernel(const float* a, const float* b, float* c, int m,
                            int n, int k) {
  // MP2 TODO: Investigate data reuse and the current memory access pattern.
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || column >= n) {
    return;
  }

  float sum = 0.0F;
  for (int inner = 0; inner < k; ++inner) {
    sum += a[row * k + inner] * b[inner * n + column];
  }
  c[row * n + column] = sum;
}

}  // namespace

cudaError_t launch_gemm(const float* a, const float* b, float* c, int m, int n,
                        int k, cudaStream_t stream) {
  const dim3 threads(16U, 16U);
  const dim3 blocks((static_cast<unsigned int>(n) + threads.x - 1U) /
                        threads.x,
                    (static_cast<unsigned int>(m) + threads.y - 1U) /
                        threads.y);
  gemm_kernel<<<blocks, threads, 0, stream>>>(a, b, c, m, n, k);
  return cudaGetLastError();
}

}  // namespace tlmp::gpu
