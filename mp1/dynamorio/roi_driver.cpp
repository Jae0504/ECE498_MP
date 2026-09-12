// Trace one full invocation of the unchanged MP1 kernels through DR's start/stop API.
#include "common.h"
#include "cpu_kernels.h"
#include "tinyllama_config.h"
#include "dr_api.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("usage: roi_driver gemm|gemv|attention S threads");
    const std::string kernel=argv[1];
    const std::size_t s=std::stoul(argv[2]), n=tinyllama::kIntermediateSize,
      k=tinyllama::kHiddenSize, d=tinyllama::kHeadDim;
    const int threads=std::stoi(argv[3]);
    if (s<1 || s>2048) throw std::runtime_error("invalid sequence");
    tlmp::configure_threads(threads);
    if (!tlmp::run_correctness_tests(true, threads)) return 2;
    std::vector<float> a,b,c,extra,score;
    if (kernel=="gemm") {
      a.resize(s*k); b.resize(k*n); c.resize(s*n);
      tlmp::fill_random(a,tlmp::kRandomSeed+10); tlmp::fill_random(b,tlmp::kRandomSeed+11);
    } else if (kernel=="gemv") {
      a.resize(k); b.resize(k*n); c.resize(n);
      tlmp::fill_random(a,tlmp::kRandomSeed+20); tlmp::fill_random(b,tlmp::kRandomSeed+21);
    } else if (kernel=="attention") {
      a.resize(s*d); b.resize(s*d); extra.resize(s*d); c.resize(s*d); score.resize(s*s);
      tlmp::fill_random(a,tlmp::kRandomSeed+30); tlmp::fill_random(b,tlmp::kRandomSeed+31);
      tlmp::fill_random(extra,tlmp::kRandomSeed+32);
    } else throw std::runtime_error("unknown kernel");
    auto compute=[&] {
      if (kernel=="gemm") tlmp::gemm_baseline(a.data(),b.data(),c.data(),s,n,k,threads);
      else if (kernel=="gemv") tlmp::gemv_baseline(a.data(),b.data(),c.data(),n,k,threads);
      else {
        tlmp::attention_qk_baseline(a.data(),b.data(),score.data(),s,d,threads);
        tlmp::softmax_rows_baseline(score.data(),s,s,threads);
        tlmp::attention_pv_baseline(score.data(),extra.data(),c.data(),s,d,threads);
      }
    };
    compute(); compute(); // Same two native warm-ups as the benchmark, outside trace.
    const auto expected=c; // Full-size native reference for trace integrity (not independent correctness).
    if (!std::getenv("DYNAMORIO_OPTIONS")) throw std::runtime_error("DYNAMORIO_OPTIONS must specify offline output");
    if (dr_app_setup()!=0) throw std::runtime_error("dr_app_setup failed");
    using Clock=std::chrono::steady_clock;
    dr_app_start();
    const auto begin=Clock::now();
    compute();
    const auto end=Clock::now();
    dr_app_stop_and_cleanup();
    double max_error=0;
    for (std::size_t i=0;i<c.size();++i) {
      if (!tlmp::close_enough(c[i],expected[i],1e-5F,1e-5F)) {
        std::cerr<<"traced result differs at "<<i<<"\n"; return 3;
      }
      max_error=std::max(max_error,double(std::abs(c[i]-expected[i])));
    }
    std::cout<<std::setprecision(17)<<"trace_checksum="<<tlmp::checksum(c)
      <<"\ntrace_full_output_max_error="<<max_error
      <<"\ninstrumented_region_wall_ms="<<std::chrono::duration<double,std::milli>(end-begin).count()
      <<"\nNOTE: instrumented time is overhead diagnostic, never native kernel performance.\n";
  } catch (const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
