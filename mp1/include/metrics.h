#pragma once

#include <cstddef>

namespace tlmp {

struct AnalyticMetrics {
  double flops = 0.0;
  double minimum_bytes = 0.0;

  double arithmetic_intensity() const {
    return flops / minimum_bytes;
  }
};

inline AnalyticMetrics gemm_metrics(std::size_t m, std::size_t n,
                                    std::size_t k) {
  const double md = static_cast<double>(m);
  const double nd = static_cast<double>(n);
  const double kd = static_cast<double>(k);
  return {2.0 * md * nd * kd,
          sizeof(float) * (md * kd + kd * nd + md * nd)};
}

inline AnalyticMetrics gemv_metrics(std::size_t n, std::size_t k) {
  const double nd = static_cast<double>(n);
  const double kd = static_cast<double>(k);
  return {2.0 * nd * kd, sizeof(float) * (kd + kd * nd + nd)};
}

struct AttentionAnalyticMetrics {
  AnalyticMetrics qk;
  AnalyticMetrics softmax;
  AnalyticMetrics pv;
  AnalyticMetrics total;
};

inline AttentionAnalyticMetrics attention_metrics(std::size_t sequence,
                                                   std::size_t head_dimension) {
  const double s = static_cast<double>(sequence);
  const double d = static_cast<double>(head_dimension);

  // QK includes one scale multiplication per score. For softmax, five scalar
  // operation-equivalents per element represent max/subtract/exp/sum/divide.
  // An exponential is not truly equivalent to one FLOP, so this is explicitly
  // an educational approximation rather than a hardware FLOP count.
  const AnalyticMetrics qk{
      2.0 * s * s * d + s * s,
      sizeof(float) * (2.0 * s * d + s * s)};
  const AnalyticMetrics softmax{
      5.0 * s * s,
      sizeof(float) * (2.0 * s * s)};
  const AnalyticMetrics pv{
      2.0 * s * s * d,
      sizeof(float) * (s * s + 2.0 * s * d)};
  return {qk,
          softmax,
          pv,
          {qk.flops + softmax.flops + pv.flops,
           qk.minimum_bytes + softmax.minimum_bytes + pv.minimum_bytes}};
}

inline double gflops(double flops, double milliseconds) {
  return flops / (milliseconds * 1.0e6);
}

}  // namespace tlmp

