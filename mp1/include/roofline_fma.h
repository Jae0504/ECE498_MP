#pragma once

#include <cstddef>
#include <stdexcept>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace tlmp {

#if defined(__AVX512F__) && defined(__FMA__)
inline constexpr int kFmaLanes = 16;
inline constexpr int kMaxFmaAccumulators = 24;
using FmaVector = __m512;
inline FmaVector fma_set(float x) { return _mm512_set1_ps(x); }
inline FmaVector fma_step(FmaVector a, FmaVector b, FmaVector c) {
  return _mm512_fmadd_ps(a, b, c);
}
inline float fma_sum(FmaVector a) { return _mm512_reduce_add_ps(a); }
#elif defined(__AVX2__) && defined(__FMA__)
inline constexpr int kFmaLanes = 8;
inline constexpr int kMaxFmaAccumulators = 12;
using FmaVector = __m256;
inline FmaVector fma_set(float x) { return _mm256_set1_ps(x); }
inline FmaVector fma_step(FmaVector a, FmaVector b, FmaVector c) {
  return _mm256_fmadd_ps(a, b, c);
}
inline float fma_sum(FmaVector a) {
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, a);
  float sum = 0;
  for (float lane : lanes) sum += lane;
  return sum;
}
#else
inline constexpr int kFmaLanes = 0;
inline constexpr int kMaxFmaAccumulators = 0;
#endif

inline bool valid_fma_accumulators(int count) {
  return count <= kMaxFmaAccumulators &&
      (count == 1 || count == 2 || count == 4 || count == 8 ||
       count == 12 || count == 16 || count == 24);
}

#if defined(__AVX2__) && defined(__FMA__)
// Named registers and compile-time counts keep the hot loop free of array
// accesses. Barriers prevent the compiler from merging identical FMA chains.
template <int Accumulators>
__attribute__((noinline)) float register_fma_kernel(std::size_t repeats) {
  static_assert(Accumulators <= kMaxFmaAccumulators);
  const FmaVector b = fma_set(0.999999F), c = fma_set(0.00001F);
#define TLMP_INIT(i) [[maybe_unused]] FmaVector a##i = fma_set(0.01F * (i + 1));
#define TLMP_EACH(M) \
  M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) \
  M(8) M(9) M(10) M(11) M(12) M(13) M(14) M(15) \
  M(16) M(17) M(18) M(19) M(20) M(21) M(22) M(23)
  TLMP_EACH(TLMP_INIT)
  for (std::size_t i = 0; i < repeats; ++i) {
#define TLMP_STEP(j) if constexpr (Accumulators > j) { \
    a##j = fma_step(a##j, b, c); asm volatile("" : "+v"(a##j)); }
    TLMP_EACH(TLMP_STEP)
  }
  float sum = 0;
#define TLMP_SUM(j) if constexpr (Accumulators > j) sum += fma_sum(a##j);
  TLMP_EACH(TLMP_SUM)
#undef TLMP_INIT
#undef TLMP_STEP
#undef TLMP_SUM
#undef TLMP_EACH
  return sum;
}
#endif

inline float register_fma(std::size_t repeats, int accumulators) {
  if (!valid_fma_accumulators(accumulators)) {
    throw std::invalid_argument("unsupported FMA accumulator count for this build; "
                                "register mode needs AVX2/FMA or AVX-512/FMA");
  }
#if defined(__AVX2__) && defined(__FMA__)
  switch (accumulators) {
    case 1: return register_fma_kernel<1>(repeats);
    case 2: return register_fma_kernel<2>(repeats);
    case 4: return register_fma_kernel<4>(repeats);
    case 8: return register_fma_kernel<8>(repeats);
    case 12: return register_fma_kernel<12>(repeats);
#if defined(__AVX512F__)
    case 16: return register_fma_kernel<16>(repeats);
    case 24: return register_fma_kernel<24>(repeats);
#endif
  }
#endif
  throw std::invalid_argument("unsupported FMA accumulator count");
}

}  // namespace tlmp
