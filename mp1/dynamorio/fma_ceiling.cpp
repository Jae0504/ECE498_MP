// Share the native MP1 register kernel; retain the original default work/checksum.
#include "roofline_fma.h"
#include "threading.h"
#include "timer.h"
#include <immintrin.h>
#include <iostream>
#include <iomanip>
#include <vector>
int main(int argc,char** argv) {
  try {
  if (argc > 4) throw std::invalid_argument("usage: fma_ceiling [threads [repeats [accumulators]]]");
  if(!__builtin_cpu_supports("avx512f") || !__builtin_cpu_supports("fma")) {std::cerr<<"AVX-512/FMA unavailable\n";return 1;}
  int threads=argc>1?tlmp::parse_thread_count(argv[1]):8;
  std::size_t reps=argc>2?tlmp::parse_thread_count(argv[2], "repeats"):20000000;
  int accumulators=argc>3?tlmp::parse_thread_count(argv[3], "accumulators"):16;
  if (!tlmp::valid_fma_accumulators(accumulators)) throw std::invalid_argument("unsupported accumulator count");
  tlmp::configure_threads(threads);
  std::vector<float> sums(threads);
  auto timing=tlmp::measure_median_with_preparation(2,9,[]{},[&]{
#pragma omp parallel num_threads(threads)
    {sums[omp_get_thread_num()]=tlmp::register_fma(reps, accumulators);}
  });
  double checksum=0;for(float v:sums) checksum+=v;
  double flops=2.0*tlmp::kFmaLanes*accumulators*reps*threads;
  std::cout<<"threads,repeats,warmups,iterations,median_ms,min_ms,max_ms,FLOPs,GFLOP_per_s,checksum,accumulators,lanes\n"
    <<std::setprecision(17)<<threads<<','<<reps<<",2,9,"<<timing.median_ms<<','<<timing.minimum_ms<<','<<timing.maximum_ms<<','<<flops<<','<<flops/(timing.median_ms*1e6)<<','<<checksum<<','<<accumulators<<','<<tlmp::kFmaLanes<<'\n';
  } catch (const std::exception& error) { std::cerr << "error: " << error.what() << '\n'; return 1; }
}
