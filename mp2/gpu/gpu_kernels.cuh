#pragma once

#include <cuda_runtime.h>

namespace tlmp::gpu {

cudaError_t launch_gemm(const float* a, const float* b, float* c, int m, int n,
                        int k, cudaStream_t stream = nullptr);
cudaError_t launch_gemv(const float* x, const float* weights, float* y, int n,
                        int k, cudaStream_t stream = nullptr);
cudaError_t launch_attention_qk(const float* query, const float* key,
                                float* scores, int sequence,
                                int head_dimension,
                                cudaStream_t stream = nullptr);
cudaError_t launch_softmax(float* scores, int rows, int columns,
                           cudaStream_t stream = nullptr);
cudaError_t launch_attention_pv(const float* probabilities,
                                const float* value, float* output,
                                int sequence, int head_dimension,
                                cudaStream_t stream = nullptr);

}  // namespace tlmp::gpu

