#include "common.h"
#include "csv_output.h"
#include "roofline_plot.h"
#include "roofline_fma.h"
#include "timer.h"

#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

struct Options {
  std::string ceiling = "all";
  std::size_t stream_elements = 67'108'864;
  std::size_t read_elements = 201'326'592;
  std::size_t fma_elements = 512;
  std::size_t fma_repeats = 20'000'000;
  std::string compute_mode = "register";
  std::string memory_mode = "streaming";
  int accumulators = tlmp::kMaxFmaAccumulators >= 16 ? 16 : 8;
  int threads = 1;
  int roof_threads = 0;
  int warmups = 1;
  int iterations = 5;
  bool csv = false;
  bool csv_header = true;
  std::string output = "results/ceilings.csv";
  bool no_save = false;
  bool no_plot = false;
  std::string kernels;
};

struct CeilingResult {
  std::string ceiling;
  std::string configuration;
  tlmp::TimingStats timing;
  double work = 0.0;
  double rate = 0.0;
  std::string unit;
  double result_checksum = 0.0;
  int warmups = 0;
  int iterations = 0;
};

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void stream_triad_kernel(float* a, const float* b, const float* c,
                         std::size_t elements) {
  constexpr float scalar = 1.000123F;
  for (std::size_t index = 0; index < elements; ++index) {
    a[index] = b[index] + scalar * c[index];
  }
}

// The caller partitions whole 64-byte lines so every worker's streaming stores
// are aligned, disjoint, and complete before its final fence and the team join.
__attribute__((noinline))
void streaming_triad(float* a, const float* b, const float* c, std::size_t n) {
  std::size_t i = 0;
#if defined(__AVX512F__)
  const __m512 scalar = _mm512_set1_ps(1.000123F);
  for (; i + 15 < n; i += 16) {
    _mm512_stream_ps(a+i, _mm512_fmadd_ps(scalar, _mm512_load_ps(c+i), _mm512_load_ps(b+i)));
  }
#elif defined(__AVX2__) && defined(__FMA__)
  const __m256 scalar = _mm256_set1_ps(1.000123F);
  for (; i + 7 < n; i += 8) {
    _mm256_stream_ps(a+i, _mm256_fmadd_ps(scalar, _mm256_load_ps(c+i), _mm256_load_ps(b+i)));
  }
#endif
  for (; i < n; ++i) a[i] = b[i] + 1.000123F * c[i];
#if defined(__AVX2__)
  _mm_sfence();
#endif
}

class AlignedFloats {
 public:
  explicit AlignedFloats(std::size_t n) : count_(n) {
    const std::size_t bytes = tlmp::checked_elements(n, sizeof(float));
    if (bytes > std::numeric_limits<std::size_t>::max()-63) throw std::bad_alloc();
    // No serial zero-fill: workers first-touch their own pages below.
    data_.reset(static_cast<float*>(std::aligned_alloc(64, (bytes+63)/64*64)));
    if (!data_) throw std::bad_alloc();
  }
  float* data() { return data_.get(); }
  std::size_t size() const { return count_; }
 private:
  std::size_t count_;
  std::unique_ptr<float, decltype(&std::free)> data_{nullptr, &std::free};
};

template <class Function>
void stream_chunks(std::size_t n, int threads, Function function) {
  tlmp::parallel_chunks(n/16 + (n%16 != 0), threads,
      [&](std::size_t first, std::size_t last, int worker) {
        function(std::min(first*16,n), std::min(last*16,n), worker);
      });
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
double stream_read_kernel(const float* values, std::size_t elements) {
#if defined(__AVX512F__)
  __m512 s0 = _mm512_setzero_ps(), s1 = s0, s2 = s0, s3 = s0;
  __m512 s4 = s0, s5 = s0, s6 = s0, s7 = s0;
  std::size_t index = 0;
  for (; index + 127 < elements; index += 128) {
    s0 = _mm512_add_ps(s0, _mm512_loadu_ps(values+index));
    s1 = _mm512_add_ps(s1, _mm512_loadu_ps(values+index+16));
    s2 = _mm512_add_ps(s2, _mm512_loadu_ps(values+index+32));
    s3 = _mm512_add_ps(s3, _mm512_loadu_ps(values+index+48));
    s4 = _mm512_add_ps(s4, _mm512_loadu_ps(values+index+64));
    s5 = _mm512_add_ps(s5, _mm512_loadu_ps(values+index+80));
    s6 = _mm512_add_ps(s6, _mm512_loadu_ps(values+index+96));
    s7 = _mm512_add_ps(s7, _mm512_loadu_ps(values+index+112));
  }
  double sum = static_cast<double>(_mm512_reduce_add_ps(s0)) + _mm512_reduce_add_ps(s1)
      + _mm512_reduce_add_ps(s2) + _mm512_reduce_add_ps(s3) + _mm512_reduce_add_ps(s4)
      + _mm512_reduce_add_ps(s5) + _mm512_reduce_add_ps(s6) + _mm512_reduce_add_ps(s7);
  for (; index < elements; ++index) sum += values[index];
#elif defined(__AVX2__)
  __m256 sum0 = _mm256_setzero_ps();
  __m256 sum1 = _mm256_setzero_ps();
  __m256 sum2 = _mm256_setzero_ps();
  __m256 sum3 = _mm256_setzero_ps();
  std::size_t index = 0;
  for (; index + 31U < elements; index += 32U) {
    sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(values + index));
    sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(values + index + 8U));
    sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(values + index + 16U));
    sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(values + index + 24U));
  }
  sum0 = _mm256_add_ps(sum0, sum1);
  sum2 = _mm256_add_ps(sum2, sum3);
  sum0 = _mm256_add_ps(sum0, sum2);
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, sum0);
  float sum = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] +
              lanes[5] + lanes[6] + lanes[7];
  for (; index < elements; ++index) {
    sum += values[index];
  }
#else
  float sums[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  std::size_t index = 0;
  for (; index + 7U < elements; index += 8U) {
    sums[0] += values[index];
    sums[1] += values[index + 1U];
    sums[2] += values[index + 2U];
    sums[3] += values[index + 3U];
    sums[4] += values[index + 4U];
    sums[5] += values[index + 5U];
    sums[6] += values[index + 6U];
    sums[7] += values[index + 7U];
  }
  float sum = sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] +
              sums[6] + sums[7];
  for (; index < elements; ++index) {
    sum += values[index];
  }
#endif
  return sum;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void fma_kernel(float* a, const float* b, const float* c,
                std::size_t elements, std::size_t repeats) {
  for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
    for (std::size_t index = 0; index < elements; ++index) {
      a[index] = a[index] * b[index] + c[index];
    }
  }
}

void print_usage(const char* program) {
  std::cout
      << "OpenMP educational Roofline ceiling microbenchmarks\n\n"
      << "Usage: " << program << " [options]\n\n"
      << "  --ceiling bandwidth|compute|all\n"
      << "  --stream-elements N  Float elements per array (default: 67108864)\n"
      << "  --read-elements N    Read-only stream length (default: 201326592)\n"
      << "  --fma-elements N     Legacy array elements per thread (default: 512)\n"
      << "  --fma-repeats N      FMA passes (default: 20000000)\n"
      << "  --compute-mode register|array (default: register; array is the legacy reference)\n"
      << "  --accumulators N     Independent FMA chains: 1,2,4,8,12,16,24 (ISA dependent)\n"
      << "  --memory-mode streaming|cached (default: streaming stores)\n"
      << "  --threads N          OpenMP workers (default: 1)\n"
      << "  --roof-threads N     Common roof scope (default: largest count in ceiling CSV)\n"
      << "  --warmup N           Warm-up iterations (default: 1)\n"
      << "  --iterations N       Timed iterations (default: 5)\n"
      << "  --output PATH        Append CSV results (default: results/ceilings.csv)\n"
      << "  --no-save            Disable CSV saving and automatic plots\n"
      << "  --no-plot            Save CSV without updating Roofline PNG/PDF\n"
      << "  --kernels PATH       Kernel CSV (default: kernel_results.csv beside --output)\n"
      << "  --format human|csv   Terminal output format\n"
      << "  --no-header          Omit the CSV header on stdout only\n";
}

std::size_t parse_size(const std::string& value, const std::string& option,
                       bool allow_zero = false) {
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed, 10);
  if (consumed != value.size() || (!allow_zero && parsed == 0ULL)) {
    throw std::invalid_argument(option + " requires a positive integer");
  }
  return static_cast<std::size_t>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    const auto next = [&]() -> std::string {
      if (++index >= argc) {
        throw std::invalid_argument(argument + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--ceiling") {
      options.ceiling = next();
    } else if (argument == "--stream-elements") {
      options.stream_elements = parse_size(next(), argument);
    } else if (argument == "--read-elements") {
      options.read_elements = parse_size(next(), argument);
    } else if (argument == "--fma-elements") {
      options.fma_elements = parse_size(next(), argument);
    } else if (argument == "--fma-repeats") {
      options.fma_repeats = parse_size(next(), argument);
    } else if (argument == "--compute-mode") {
      options.compute_mode = next();
    } else if (argument == "--accumulators") {
      options.accumulators = tlmp::parse_thread_count(next(), argument);
    } else if (argument == "--memory-mode") {
      options.memory_mode = next();
    } else if (argument == "--threads") {
      options.threads = tlmp::parse_thread_count(next());
    } else if (argument == "--roof-threads") {
      options.roof_threads = tlmp::parse_thread_count(next(), argument);
    } else if (argument == "--warmup") {
      options.warmups = static_cast<int>(parse_size(next(), argument, true));
    } else if (argument == "--iterations") {
      options.iterations = static_cast<int>(parse_size(next(), argument));
    } else if (argument == "--format") {
      const std::string format = next();
      if (format == "csv") {
        options.csv = true;
      } else if (format == "human") {
        options.csv = false;
      } else {
        throw std::invalid_argument("--format must be human or csv");
      }
    } else if (argument == "--output") {
      options.output = next();
    } else if (argument == "--no-save") {
      options.no_save = true;
    } else if (argument == "--no-header") {
      options.csv_header = false;
    } else if (argument == "--no-plot") {
      options.no_plot = true;
    } else if (argument == "--kernels") {
      options.kernels = next();
      if (options.kernels.empty()) {
        throw std::invalid_argument("--kernels requires a nonempty file path");
      }
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.ceiling != "bandwidth" && options.ceiling != "compute" &&
      options.ceiling != "all") {
    throw std::invalid_argument("--ceiling must be bandwidth, compute, or all");
  }
  if (options.compute_mode != "register" && options.compute_mode != "array")
    throw std::invalid_argument("--compute-mode must be register or array");
  if (options.memory_mode != "streaming" && options.memory_mode != "cached")
    throw std::invalid_argument("--memory-mode must be streaming or cached");
  if (options.compute_mode == "register" && options.ceiling != "bandwidth" &&
      !tlmp::valid_fma_accumulators(options.accumulators))
    throw std::invalid_argument("register FMA unavailable or invalid --accumulators; use a native AVX2/FMA build or --compute-mode array");
  return options;
}

CeilingResult benchmark_bandwidth(const Options& options) {
  AlignedFloats a(options.stream_elements), b(options.stream_elements), c(options.stream_elements);
  stream_chunks(a.size(), options.threads, [&](std::size_t first, std::size_t last, int) {
    for (std::size_t i = first; i < last; ++i) { a.data()[i]=0; b.data()[i]=0.5F; c.data()[i]=0.25F; }
  });

  const tlmp::TimingStats timing = tlmp::measure_median(
      options.warmups, options.iterations, [&] {
        stream_chunks(a.size(), options.threads,
            [&](std::size_t first, std::size_t last, int) {
          auto kernel = options.memory_mode == "streaming" ? streaming_triad : stream_triad_kernel;
          kernel(a.data() + first, b.data() + first, c.data() + first, last - first);
        });
      });
  // STREAM convention: two input reads plus one output write. Write-allocate
  // and cache effects can make physical traffic differ from this byte count.
  const double bytes = 3.0 * sizeof(float) *
                       static_cast<double>(options.stream_elements);
  const double gigabytes_per_second = bytes / (timing.median_ms * 1.0e6);
  const float expected = std::fma(1.000123F, 0.25F, 0.5F);
  // Check all stores, including unaligned-size tails, outside the timed region.
  std::vector<double> checksums(options.threads, 0);
  std::vector<int> errors(options.threads, 0);
  stream_chunks(a.size(), options.threads, [&](std::size_t first, std::size_t last, int worker) {
    for (std::size_t i = first; i < last; ++i) {
      if (a.data()[i] != expected) errors[worker] = 1;
      checksums[worker] += a.data()[i];
    }
  });
  double checksum = 0;
  for (int worker = 0; worker < options.threads; ++worker) {
    if (errors[worker]) throw std::runtime_error("triad output verification failed");
    checksum += checksums[worker];
  }
  std::ostringstream configuration;
  configuration << "elements=" << options.stream_elements
                << ";arrays=3;threads=" << options.threads
                << ";stores=" << options.memory_mode << ";first_touch=parallel;verified=1";
  return {"memory_bandwidth", configuration.str(), timing, bytes,
          gigabytes_per_second, "GB/s", checksum, options.warmups,
          options.iterations};
}

CeilingResult benchmark_read_bandwidth(const Options& options) {
  AlignedFloats values(options.read_elements);
  stream_chunks(values.size(), options.threads, [&](std::size_t first, std::size_t last, int) {
    std::fill(values.data()+first, values.data()+last, 0.5F);
  });
  std::vector<double> partial(static_cast<std::size_t>(options.threads), 0.0);
  const tlmp::TimingStats timing = tlmp::measure_median(
      options.warmups, options.iterations, [&] {
        stream_chunks(values.size(), options.threads,
            [&](std::size_t first, std::size_t last, int worker) {
          double sum = 0;
          // Bound each FP32 reduction so even multi-GiB streams at one thread
          // retain an exact checksum for the constant, exactly representable input.
          while (first < last) {
            const auto length = std::min(last-first, std::size_t{1} << 24);
            sum += stream_read_kernel(values.data()+first, length);
            first += length;
          }
          partial[worker] = sum;
        });
        double sum = 0.0;
        for (double value : partial) {
          sum += value;
        }
        tlmp::g_observable_checksum = sum;
      });
  const double bytes = sizeof(float) * static_cast<double>(values.size());
  const double gigabytes_per_second = bytes / (timing.median_ms * 1.0e6);
  if (tlmp::g_observable_checksum != 0.5 * static_cast<double>(values.size()))
    throw std::runtime_error("read-only output verification failed");
  std::ostringstream configuration;
  configuration << "elements=" << options.read_elements
                << ";streams=1;threads=" << options.threads
                << ";first_touch=parallel;verified=1";
  return {"memory_bandwidth_read", configuration.str(), timing, bytes,
          gigabytes_per_second, "GB/s", tlmp::g_observable_checksum,
          options.warmups, options.iterations};
}

CeilingResult benchmark_compute(const Options& options) {
  if (options.compute_mode == "register") {
    // Independent scalar reference checks every compiled accumulator variant.
    float expected = 0;
    for (int i = 0; i < options.accumulators; ++i) {
      float a = 0.01F * (i+1);
      for (int j = 0; j < 7; ++j) a = std::fma(a, 0.999999F, 0.00001F);
      expected += a * tlmp::kFmaLanes;
    }
    if (!tlmp::close_enough(tlmp::register_fma(7, options.accumulators), expected, 1e-4F, 1e-5F))
      throw std::runtime_error("register FMA verification failed");
    std::vector<float> partial(options.threads);
    const auto timing = tlmp::measure_median(options.warmups, options.iterations, [&] {
      tlmp::parallel_chunks(options.threads, options.threads,
          [&](std::size_t first, std::size_t last, int) {
        for (auto worker=first; worker<last; ++worker)
          partial[worker] = tlmp::register_fma(options.fma_repeats, options.accumulators);
      });
    });
    double checksum = 0;
    for (float value : partial) checksum += value;
    const double flops = 2.0 * tlmp::kFmaLanes * options.accumulators *
        static_cast<double>(options.fma_repeats) * options.threads;
    std::ostringstream configuration;
    configuration << "mode=register;lanes=" << tlmp::kFmaLanes
                  << ";accumulators=" << options.accumulators
                  << ";repeats=" << options.fma_repeats << ";threads=" << options.threads
                  << ";verified=1";
    return {"fma_compute", configuration.str(), timing, flops,
            flops / (timing.median_ms * 1e6), "GFLOP/s", checksum,
            options.warmups, options.iterations};
  }
  // Keep the small working set per worker fixed as the team grows.
  // Padding separates writable worker regions even without aligned malloc.
  const auto stride = tlmp::checked_elements(options.fma_elements / 16U + 2U, 16U);
  const auto total = tlmp::checked_elements(stride, static_cast<std::size_t>(options.threads));
  std::vector<float> a(total, 0.0F), b(total), c(total);
  for (int worker = 0; worker < options.threads; ++worker) {
    const auto offset = static_cast<std::size_t>(worker) * stride;
    for (std::size_t index = 0; index < options.fma_elements; ++index) {
      a[offset + index] = 0.125F;
      b[offset + index] = 0.99990F + static_cast<float>(index % 7U) * 1.0e-7F;
      c[offset + index] = 1.0e-5F + static_cast<float>(index % 5U) * 1.0e-7F;
    }
  }
  const tlmp::TimingStats timing = tlmp::measure_median(
      options.warmups, options.iterations, [&] {
        tlmp::parallel_chunks(static_cast<std::size_t>(options.threads), options.threads,
            [&](std::size_t first, std::size_t last, int) {
          for (auto worker = first; worker < last; ++worker) {
            const auto offset = worker * stride;
            fma_kernel(a.data() + offset, b.data() + offset, c.data() + offset,
                       options.fma_elements, options.fma_repeats);
          }
        });
      });
  const double flops = 2.0 * static_cast<double>(options.fma_elements) *
                       static_cast<double>(options.fma_repeats) * options.threads;
  const double gigaflops = flops / (timing.median_ms * 1.0e6);
  std::ostringstream configuration;
  configuration << "elements=" << options.fma_elements
                << ";repeats=" << options.fma_repeats
                << ";elements_scope=per_thread;threads=" << options.threads << ";mode=array";
  return {"fma_compute", configuration.str(), timing, flops, gigaflops,
          "GFLOP/s", tlmp::checksum(a), options.warmups, options.iterations};
}

std::string number(double value) {
  std::ostringstream stream;
  stream << std::setprecision(12) << value;
  return stream.str();
}

void print_csv_header(std::ostream& output = std::cout) {
  output << "ceiling,configuration,median_ms,min_ms,max_ms,"
               "relative_range_percent,work,rate,unit,checksum,warmups,iterations\n";
}

void print_result(const CeilingResult& result, bool csv,
                  std::ostream& output = std::cout) {
  if (csv) {
    const std::vector<std::string> fields = {
        result.ceiling,
        result.configuration,
        number(result.timing.median_ms),
        number(result.timing.minimum_ms),
        number(result.timing.maximum_ms),
        number(result.timing.relative_range_percent),
        number(result.work),
        number(result.rate),
        result.unit,
        number(result.result_checksum),
        std::to_string(result.warmups),
        std::to_string(result.iterations)};
    for (std::size_t index = 0; index < fields.size(); ++index) {
      if (index != 0U) {
        output << ',';
      }
      output << fields[index];
    }
    output << '\n';
    return;
  }

  output << "ceiling: " << result.ceiling << '\n'
            << "configuration: " << result.configuration << '\n'
            << std::fixed << std::setprecision(4)
            << "median runtime: " << result.timing.median_ms << " ms"
            << " (min=" << result.timing.minimum_ms
            << ", max=" << result.timing.maximum_ms << ")\n"
            << "relative timing range: "
            << result.timing.relative_range_percent << "%\n"
            << "measured ceiling: " << result.rate << ' ' << result.unit
            << '\n'
            << std::scientific << std::setprecision(9)
            << "checksum: " << result.result_checksum << "\n\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    tlmp::configure_threads(options.threads);
    std::ostringstream header;
    print_csv_header(header);
    tlmp::CsvOutput csv_file(options.output, header.str(), !options.no_save);
    const auto emit = [&](const CeilingResult& result) {
      std::ostringstream row;
      print_result(result, true, row);
      csv_file.append(row.str());
      print_result(result, options.csv);
    };
    if (options.csv && options.csv_header) {
      print_csv_header();
    }
    if (options.ceiling == "bandwidth" || options.ceiling == "all") {
      emit(benchmark_bandwidth(options));
      emit(benchmark_read_bandwidth(options));
    }
    if (options.ceiling == "compute" || options.ceiling == "all") {
      emit(benchmark_compute(options));
    }
    if (!options.no_save) {
      std::cerr << "CSV results appended to " << options.output << '\n';
      if (!options.no_plot) {
        const auto directory = std::filesystem::path(options.output).parent_path();
        const auto kernels = options.kernels.empty()
            ? (directory / "kernel_results.csv").string() : options.kernels;
        tlmp::update_roofline(kernels, options.output,
                             (directory / "roofline.csv").string(), options.roof_threads);
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
