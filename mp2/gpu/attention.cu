#include "gpu_kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cfloat>

namespace tlmp::gpu {
namespace {

__global__ void attention_qk_kernel(const float* query, const float* key,
                                    float* scores, int sequence,
                                    int head_dimension, float scale) {
  // MP2 TODO: Investigate this memory access pattern and data reuse.
  const int key_row = blockIdx.x * blockDim.x + threadIdx.x;
  const int query_row = blockIdx.y * blockDim.y + threadIdx.y;
  if (query_row >= sequence || key_row >= sequence) {
    return;
  }

  float sum = 0.0F;
  for (int element = 0; element < head_dimension; ++element) {
    sum += query[query_row * head_dimension + element] *
           key[key_row * head_dimension + element];
  }
  scores[query_row * sequence + key_row] = sum * scale;
}

__global__ void softmax_rows_kernel(float* scores, int columns) {
  extern __shared__ float reduction[];
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  float* const values = scores + row * columns;

  float local_maximum = -FLT_MAX;
  for (int column = lane; column < columns; column += blockDim.x) {
    local_maximum = fmaxf(local_maximum, values[column]);
  }
  reduction[lane] = local_maximum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (lane < stride) {
      reduction[lane] = fmaxf(reduction[lane], reduction[lane + stride]);
    }
    __syncthreads();
  }
  const float maximum = reduction[0];
  // Every thread must read the maximum before reduction[] is reused for the
  // sum. This barrier is required for correctness; it is not an optimization.
  __syncthreads();

  float local_sum = 0.0F;
  for (int column = lane; column < columns; column += blockDim.x) {
    const float exponential = expf(values[column] - maximum);
    values[column] = exponential;
    local_sum += exponential;
  }
  reduction[lane] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (lane < stride) {
      reduction[lane] += reduction[lane + stride];
    }
    __syncthreads();
  }
  const float inverse_sum = 1.0F / reduction[0];
  for (int column = lane; column < columns; column += blockDim.x) {
    values[column] *= inverse_sum;
  }
}

__global__ void attention_pv_kernel(const float* probabilities,
                                    const float* value, float* output,
                                    int sequence, int head_dimension) {
  // MP2 TODO: Investigate this memory access pattern and data reuse.
  const int element = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= sequence || element >= head_dimension) {
    return;
  }

  float sum = 0.0F;
  for (int token = 0; token < sequence; ++token) {
    sum += probabilities[row * sequence + token] *
           value[token * head_dimension + element];
  }
  output[row * head_dimension + element] = sum;
}

}  // namespace

cudaError_t launch_attention_qk(const float* query, const float* key,
                                float* scores, int sequence,
                                int head_dimension, cudaStream_t stream) {
  const dim3 threads(16U, 16U);
  const dim3 blocks((static_cast<unsigned int>(sequence) + threads.x - 1U) /
                        threads.x,
                    (static_cast<unsigned int>(sequence) + threads.y - 1U) /
                        threads.y);
  const float scale = 1.0F / sqrtf(static_cast<float>(head_dimension));
  attention_qk_kernel<<<blocks, threads, 0, stream>>>(
      query, key, scores, sequence, head_dimension, scale);
  return cudaGetLastError();
}

cudaError_t launch_softmax(float* scores, int rows, int columns,
                           cudaStream_t stream) {
  constexpr int threads = 256;
  const std::size_t shared_bytes = threads * sizeof(float);
  softmax_rows_kernel<<<rows, threads, shared_bytes, stream>>>(scores, columns);
  return cudaGetLastError();
}

cudaError_t launch_attention_pv(const float* probabilities,
                                const float* value, float* output,
                                int sequence, int head_dimension,
                                cudaStream_t stream) {
  const dim3 threads(32U, 8U);
  const dim3 blocks(
      (static_cast<unsigned int>(head_dimension) + threads.x - 1U) / threads.x,
      (static_cast<unsigned int>(sequence) + threads.y - 1U) / threads.y);
  attention_pv_kernel<<<blocks, threads, 0, stream>>>(
      probabilities, value, output, sequence, head_dimension);
  return cudaGetLastError();
}

}  // namespace tlmp::gpu
