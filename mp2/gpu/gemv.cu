#include "gpu_kernels.cuh"

#include <cuda_runtime.h>

namespace tlmp::gpu {
namespace {

__global__ void gemv_kernel(const float* x, const float* weights, float* y,
                            int n, int k) {
  // MP2 TODO: Explore how the block/grid configuration affects utilization.
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  if (column >= n) {
    return;
  }

  float sum = 0.0F;
  for (int inner = 0; inner < k; ++inner) {
    sum += x[inner] * weights[inner * n + column];
  }
  y[column] = sum;
}

}  // namespace

cudaError_t launch_gemv(const float* x, const float* weights, float* y, int n,
                        int k, cudaStream_t stream) {
  // Optimization example: compare the baseline with a smaller block.
  // To try it, comment out the 256-thread line and uncomment the 64-thread line.
  // Keep exactly one active definition, then rebuild and rerun correctness.
  constexpr int threads = 256;  // Naive baseline (default).
  // constexpr int threads = 64;  // Example optimization: more blocks.
  // For N=5632, this changes 22 blocks to 88; the grid adjusts automatically.
  // Measure the effect: a smaller block does not guarantee a speedup.
  const int blocks = (n + threads - 1) / threads;
  gemv_kernel<<<blocks, threads, 0, stream>>>(x, weights, y, n, k);
  return cudaGetLastError();
}

}  // namespace tlmp::gpu
