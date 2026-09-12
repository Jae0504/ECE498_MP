// Optional phase ROIs link the unchanged Attention functions from the native build.
#include "common.h"
#include "cpu_kernels.h"
#include "dr_api.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
int main(int argc,char** argv) {
  try {
    if(argc!=4) throw std::runtime_error("usage: attention_phase_driver qk|softmax|pv S threads");
    std::string phase=argv[1];size_t s=std::stoul(argv[2]),d=64;int threads=std::stoi(argv[3]);
    if(s<1||s>2048||(phase!="qk"&&phase!="softmax"&&phase!="pv"))throw std::runtime_error("invalid phase/shape");
    tlmp::configure_threads(threads);
    if(!tlmp::run_correctness_tests(true,threads))return 2;
    std::vector<float> q(s*d),k(s*d),v(s*d),scores(s*s),output(s*d);
    tlmp::fill_random(q,tlmp::kRandomSeed+30);tlmp::fill_random(k,tlmp::kRandomSeed+31);tlmp::fill_random(v,tlmp::kRandomSeed+32);
    auto qk=[&]{tlmp::attention_qk_baseline(q.data(),k.data(),scores.data(),s,d,threads);};
    auto softmax=[&]{tlmp::softmax_rows_baseline(scores.data(),s,s,threads);};
    auto pv=[&]{tlmp::attention_pv_baseline(scores.data(),v.data(),output.data(),s,d,threads);};
    qk();const auto raw_scores=scores;softmax();pv();
    auto compute=[&]{if(phase=="qk")qk();else if(phase=="softmax")softmax();else pv();};
    for(int i=0;i<2;++i){if(phase=="softmax")scores=raw_scores;compute();}
    const auto expected=phase=="pv"?output:scores;
    if(phase=="softmax")scores=raw_scores;
    if(!std::getenv("DYNAMORIO_OPTIONS"))throw std::runtime_error("missing offline options");
    if(dr_app_setup()!=0)throw std::runtime_error("DR setup failed");
    dr_app_start();const auto begin=std::chrono::steady_clock::now();
    compute();
    const auto end=std::chrono::steady_clock::now();dr_app_stop_and_cleanup();
    const auto& actual=phase=="pv"?output:scores;double error=0;
    for(size_t i=0;i<actual.size();++i){if(!tlmp::close_enough(actual[i],expected[i],1e-5F,1e-5F))return 3;error=std::max(error,double(std::abs(actual[i]-expected[i])));}
    std::cout<<std::setprecision(17)<<"trace_checksum="<<tlmp::checksum(actual)<<"\ntrace_full_output_max_error="<<error<<"\ninstrumented_region_wall_ms="<<std::chrono::duration<double,std::milli>(end-begin).count()<<"\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
