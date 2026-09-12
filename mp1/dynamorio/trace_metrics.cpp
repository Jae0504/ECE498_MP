// Actual data-reference byte widths and unique 64-byte data lines, using the DR reader.
// Each traced thread has independent counters; merge sets only after workers finish.
#include "drmemtrace/analyzer.h"
#include <cstdint>
#include <iostream>
#include <mutex>
#include <unordered_set>
#include <vector>
using namespace dynamorio::drmemtrace;
class metrics_tool : public analysis_tool_t {
  struct counts {uint64_t loads=0,stores=0,bytes=0;std::unordered_set<uint64_t> lines;};
  counts serial;
  std::vector<counts*> shards;
  std::mutex lock;
  static bool consume(counts& c,const memref_t& r) {
    if (r.data.type!=TRACE_TYPE_READ && r.data.type!=TRACE_TYPE_WRITE) return true;
    if (r.data.type==TRACE_TYPE_READ) ++c.loads; else ++c.stores;
    c.bytes+=r.data.size;
    if(r.data.size) {
      const uint64_t first=r.data.addr/64,last=(r.data.addr+r.data.size-1)/64;
      for(uint64_t line=first;line<=last;++line) c.lines.insert(line);
    }
    return true;
  }
 public:
  ~metrics_tool() override {for(auto* c:shards) delete c;}
  bool process_memref(const memref_t& r) override {return consume(serial,r);}
  bool parallel_shard_supported() override {return true;}
  void* parallel_shard_init_stream(int,void*,memtrace_stream_t*) override {
    auto* c=new counts;
    std::lock_guard<std::mutex> guard(lock); shards.push_back(c);return c;
  }
  bool parallel_shard_memref(void* data,const memref_t& r) override {return consume(*static_cast<counts*>(data),r);}
  bool parallel_shard_exit(void*) override {return true;}
  bool print_results() override {
    for(auto* c:shards) {
      serial.loads+=c->loads;serial.stores+=c->stores;serial.bytes+=c->bytes;
      serial.lines.insert(c->lines.begin(),c->lines.end());
    }
    std::cout<<"{\"Loads\":"<<serial.loads<<",\"Stores\":"<<serial.stores
      <<",\"Dynamic_Memory_Reference_Bytes\":"<<serial.bytes
      <<",\"Unique_Cache_Lines\":"<<serial.lines.size()<<"}\n";
    return true;
  }
};
int main(int argc,char** argv) {
  if(argc!=2) {std::cerr<<"usage: trace_metrics converted_trace_directory\n";return 1;}
  metrics_tool m; analysis_tool_t* ts[]={&m};
  analyzer_t a(argv[1],ts,1,8);
  if(!a || !a.run() || !a.print_stats()) {std::cerr<<a.get_error_string()<<"\n";return 2;}
}
