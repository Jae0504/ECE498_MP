#include "common.h"
#include "cpu_kernels.h"
#include "csv_output.h"
#include "metrics.h"
#include "roofline_plot.h"
#include "timer.h"
#include "threading.h"
#include "tinyllama_config.h"

#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
  std::string kernel;
  std::size_t sequence = 512;
  int threads = 1;
  int roof_threads = 0;
  int warmups = 1;
  int iterations = 5;
  std::size_t flush_cache_mib = 0;
  bool test_only = false;
  bool skip_tests = false;
  bool csv = false;
  bool csv_header = true;
  std::string output = "results/kernel_results.csv";
  bool no_save = false;
  bool no_plot = false;
  std::string ceilings;
};

struct BenchmarkResult {
  std::string kernel;
  std::string configuration;
  tlmp::TimingStats timing;
  tlmp::AnalyticMetrics analytic;
  double result_checksum = 0.0;
  int warmups = 0;
  int iterations = 0;
  std::size_t sequence = 1;
  std::size_t flush_cache_mib = 0;
  bool has_attention_phases = false;
  tlmp::TimingStats qk_timing;
  tlmp::TimingStats softmax_timing;
  tlmp::TimingStats pv_timing;
  tlmp::AttentionAnalyticMetrics attention_analytic;
};

void print_usage(const char* program) {
  std::cout
      << "TinyLlama CPU architecture microbenchmarks\n\n"
      << "Usage:\n"
      << "  " << program << " --test\n"
      << "  " << program
      << " --kernel gemm|gemv|attention|all [options]\n\n"
      << "Options:\n"
      << "  --seq N            GEMM/attention sequence length, 1..2048 (default: 512)\n"
      << "  --threads N        OpenMP workers (default: 1)\n"
      << "  --roof-threads N   Common roof scope (default: largest count in ceiling CSV)\n"
      << "  --warmup N         Warm-up iterations (default: 1)\n"
      << "  --iterations N     Timed iterations used for the median (default: 5)\n"
      << "  --flush-cache-mib N  Touch N MiB outside timing before each run (default: 0)\n"
      << "  --output PATH      Append CSV results (default: results/kernel_results.csv)\n"
      << "  --no-save          Disable CSV saving and automatic plots\n"
      << "  --no-plot          Save CSV without updating Roofline PNG/PDF\n"
      << "  --ceilings PATH    Ceiling CSV (default: ceilings.csv beside --output)\n"
      << "  --format human|csv Terminal output format (default: human)\n"
      << "  --no-header        Omit the CSV header on stdout only\n"
      << "  --skip-tests       Skip preflight correctness (profiling script only)\n"
      << "  --help              Show this help\n\n"
      << "Explicit-loop float32 code with OpenMP; defaults to one thread; no BLAS.\n";
}

std::size_t parse_size(const std::string& text, const std::string& option) {
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(text, &consumed, 10);
  if (consumed != text.size() || parsed == 0ULL) {
    throw std::invalid_argument(option + " requires a positive integer");
  }
  return static_cast<std::size_t>(parsed);
}

std::size_t parse_nonnegative_size(const std::string& text,
                                   const std::string& option) {
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(text, &consumed, 10);
  if (consumed != text.size()) {
    throw std::invalid_argument(option + " requires a nonnegative integer");
  }
  return static_cast<std::size_t>(parsed);
}

int parse_nonnegative_int(const std::string& text, const std::string& option,
                          bool allow_zero) {
  std::size_t consumed = 0;
  const long parsed = std::stol(text, &consumed, 10);
  if (consumed != text.size() || parsed < 0 || (!allow_zero && parsed == 0)) {
    throw std::invalid_argument(option +
                                (allow_zero ? " requires a nonnegative integer"
                                            : " requires a positive integer"));
  }
  return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    const auto next_value = [&]() -> std::string {
      if (++index >= argc) {
        throw std::invalid_argument(argument + " requires a value");
      }
      return argv[index];
    };

    if (argument == "--kernel") {
      options.kernel = next_value();
    } else if (argument == "--seq") {
      options.sequence = parse_size(next_value(), argument);
    } else if (argument == "--threads") {
      options.threads = tlmp::parse_thread_count(next_value());
    } else if (argument == "--roof-threads") {
      options.roof_threads = tlmp::parse_thread_count(next_value(), argument);
    } else if (argument == "--warmup") {
      options.warmups = parse_nonnegative_int(next_value(), argument, true);
    } else if (argument == "--iterations" || argument == "--iters") {
      options.iterations =
          parse_nonnegative_int(next_value(), argument, false);
    } else if (argument == "--flush-cache-mib") {
      options.flush_cache_mib =
          parse_nonnegative_size(next_value(), argument);
    } else if (argument == "--format") {
      const std::string format = next_value();
      if (format == "csv") {
        options.csv = true;
      } else if (format == "human") {
        options.csv = false;
      } else {
        throw std::invalid_argument("--format must be human or csv");
      }
    } else if (argument == "--no-header") {
      options.csv_header = false;
    } else if (argument == "--output") {
      options.output = next_value();
    } else if (argument == "--no-save") {
      options.no_save = true;
    } else if (argument == "--no-plot") {
      options.no_plot = true;
    } else if (argument == "--ceilings") {
      options.ceilings = next_value();
      if (options.ceilings.empty()) {
        throw std::invalid_argument("--ceilings requires a nonempty file path");
      }
    } else if (argument == "--test") {
      options.test_only = true;
    } else if (argument == "--skip-tests") {
      options.skip_tests = true;
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  return options;
}

std::string gemm_configuration(std::size_t sequence) {
  std::ostringstream stream;
  stream << "S=" << sequence << ";M=" << sequence
         << ";N=" << tinyllama::kIntermediateSize
         << ";K=" << tinyllama::kHiddenSize;
  return stream.str();
}

std::string cache_configuration(std::size_t flush_cache_mib) {
  if (flush_cache_mib == 0U) {
    return ";cache=warm";
  }
  return ";cache=cold;flush_mib=" + std::to_string(flush_cache_mib);
}

BenchmarkResult benchmark_gemm(std::size_t sequence, int warmups,
                               int iterations, std::size_t flush_cache_mib, int threads) {
  const std::size_t m = sequence;
  const std::size_t n = tinyllama::kIntermediateSize;
  const std::size_t k = tinyllama::kHiddenSize;
  std::vector<float> a(tlmp::checked_elements(m, k));
  std::vector<float> b(tlmp::checked_elements(k, n));
  std::vector<float> c(tlmp::checked_elements(m, n));
  tlmp::fill_random(a, tlmp::kRandomSeed + 10U);
  tlmp::fill_random(b, tlmp::kRandomSeed + 11U);
  const tlmp::CacheFlusher cache_flusher(flush_cache_mib, threads);

  const tlmp::TimingStats timing = tlmp::measure_median_with_preparation(
      warmups, iterations, [&] { cache_flusher.touch(); }, [&] {
        tlmp::gemm_baseline(a.data(), b.data(), c.data(), m, n, k, threads);
      });
  BenchmarkResult result;
  result.kernel = "gemm";
  result.sequence = sequence;
  result.configuration =
      gemm_configuration(sequence) + cache_configuration(flush_cache_mib);
  result.timing = timing;
  result.analytic = tlmp::gemm_metrics(m, n, k);
  result.result_checksum = tlmp::checksum(c);
  result.warmups = warmups;
  result.iterations = iterations;
  result.flush_cache_mib = flush_cache_mib;
  return result;
}

BenchmarkResult benchmark_gemv(int warmups, int iterations,
                               std::size_t flush_cache_mib, int threads) {
  const std::size_t n = tinyllama::kIntermediateSize;
  const std::size_t k = tinyllama::kHiddenSize;
  std::vector<float> x(k);
  std::vector<float> weights(tlmp::checked_elements(k, n));
  std::vector<float> y(n);
  tlmp::fill_random(x, tlmp::kRandomSeed + 20U);
  tlmp::fill_random(weights, tlmp::kRandomSeed + 21U);
  const tlmp::CacheFlusher cache_flusher(flush_cache_mib, threads);

  const tlmp::TimingStats timing = tlmp::measure_median_with_preparation(
      warmups, iterations, [&] { cache_flusher.touch(); }, [&] {
        tlmp::gemv_baseline(x.data(), weights.data(), y.data(), n, k, threads);
      });
  std::ostringstream configuration;
  configuration << "tokens=1;N=" << n << ";K=" << k;
  BenchmarkResult result;
  result.kernel = "gemv";
  result.configuration =
      configuration.str() + cache_configuration(flush_cache_mib);
  result.timing = timing;
  result.analytic = tlmp::gemv_metrics(n, k);
  result.result_checksum = tlmp::checksum(y);
  result.warmups = warmups;
  result.iterations = iterations;
  result.flush_cache_mib = flush_cache_mib;
  return result;
}

BenchmarkResult benchmark_attention(std::size_t sequence, int warmups,
                                    int iterations,
                                    std::size_t flush_cache_mib, int threads) {
  const std::size_t dimension = tinyllama::kHeadDim;
  std::vector<float> query(tlmp::checked_elements(sequence, dimension));
  std::vector<float> key(tlmp::checked_elements(sequence, dimension));
  std::vector<float> value(tlmp::checked_elements(sequence, dimension));
  std::vector<float> scores(tlmp::checked_elements(sequence, sequence));
  std::vector<float> output(tlmp::checked_elements(sequence, dimension));
  tlmp::fill_random(query, tlmp::kRandomSeed + 30U);
  tlmp::fill_random(key, tlmp::kRandomSeed + 31U);
  tlmp::fill_random(value, tlmp::kRandomSeed + 32U);
  const tlmp::CacheFlusher cache_flusher(flush_cache_mib, threads);

  const auto run_all_phases = [&] {
    tlmp::attention_qk_baseline(query.data(), key.data(), scores.data(),
                                sequence, dimension, threads);
    tlmp::softmax_rows_baseline(scores.data(), sequence, sequence, threads);
    tlmp::attention_pv_baseline(scores.data(), value.data(), output.data(),
                                sequence, dimension, threads);
  };
  for (int iteration = 0; iteration < warmups; ++iteration) {
    cache_flusher.touch();
    run_all_phases();
  }

  using Clock = std::chrono::steady_clock;
  std::vector<double> total_samples;
  std::vector<double> qk_samples;
  std::vector<double> softmax_samples;
  std::vector<double> pv_samples;
  total_samples.reserve(static_cast<std::size_t>(iterations));
  qk_samples.reserve(static_cast<std::size_t>(iterations));
  softmax_samples.reserve(static_cast<std::size_t>(iterations));
  pv_samples.reserve(static_cast<std::size_t>(iterations));

  for (int iteration = 0; iteration < iterations; ++iteration) {
    cache_flusher.touch();
    const auto total_start = Clock::now();
    const auto qk_start = total_start;
    tlmp::attention_qk_baseline(query.data(), key.data(), scores.data(),
                                sequence, dimension, threads);
    const auto qk_stop = Clock::now();
    tlmp::softmax_rows_baseline(scores.data(), sequence, sequence, threads);
    const auto softmax_stop = Clock::now();
    tlmp::attention_pv_baseline(scores.data(), value.data(), output.data(),
                                sequence, dimension, threads);
    const auto pv_stop = Clock::now();

    qk_samples.push_back(
        std::chrono::duration<double, std::milli>(qk_stop - qk_start).count());
    softmax_samples.push_back(std::chrono::duration<double, std::milli>(
                                  softmax_stop - qk_stop)
                                  .count());
    pv_samples.push_back(std::chrono::duration<double, std::milli>(
                             pv_stop - softmax_stop)
                             .count());
    total_samples.push_back(std::chrono::duration<double, std::milli>(
                                pv_stop - total_start)
                                .count());
  }

  const tlmp::AttentionAnalyticMetrics analytic =
      tlmp::attention_metrics(sequence, dimension);
  std::ostringstream configuration;
  configuration << "S=" << sequence << ";D=" << dimension
                << ";heads=1;unmasked=1"
                << cache_configuration(flush_cache_mib);

  BenchmarkResult result;
  result.kernel = "attention";
  result.sequence = sequence;
  result.configuration = configuration.str();
  result.timing = tlmp::summarize_timings(std::move(total_samples));
  result.analytic = analytic.total;
  result.result_checksum = tlmp::checksum(output);
  result.warmups = warmups;
  result.iterations = iterations;
  result.flush_cache_mib = flush_cache_mib;
  result.has_attention_phases = true;
  result.qk_timing = tlmp::summarize_timings(std::move(qk_samples));
  result.softmax_timing =
      tlmp::summarize_timings(std::move(softmax_samples));
  result.pv_timing = tlmp::summarize_timings(std::move(pv_samples));
  result.attention_analytic = analytic;
  return result;
}

std::string number(double value) {
  std::ostringstream stream;
  stream << std::setprecision(12) << value;
  return stream.str();
}

void print_csv_header(std::ostream& output = std::cout) {
  output
      << "kernel,configuration,median_ms,min_ms,max_ms,relative_range_percent,"
         "approx_flops,measured_gflops,estimated_minimum_bytes,"
         "arithmetic_intensity,qk_ms,qk_gflops,softmax_ms,softmax_gops,"
         "pv_ms,pv_gflops,checksum,warmups,iterations,sequence,flush_cache_mib\n";
}

void print_csv(const BenchmarkResult& result, std::ostream& output = std::cout) {
  std::vector<std::string> fields = {
      result.kernel,
      result.configuration,
      number(result.timing.median_ms),
      number(result.timing.minimum_ms),
      number(result.timing.maximum_ms),
      number(result.timing.relative_range_percent),
      number(result.analytic.flops),
      number(tlmp::gflops(result.analytic.flops, result.timing.median_ms)),
      number(result.analytic.minimum_bytes),
      number(result.analytic.arithmetic_intensity())};

  if (result.has_attention_phases) {
    fields.push_back(number(result.qk_timing.median_ms));
    fields.push_back(number(tlmp::gflops(result.attention_analytic.qk.flops,
                                         result.qk_timing.median_ms)));
    fields.push_back(number(result.softmax_timing.median_ms));
    fields.push_back(number(tlmp::gflops(
        result.attention_analytic.softmax.flops,
        result.softmax_timing.median_ms)));
    fields.push_back(number(result.pv_timing.median_ms));
    fields.push_back(number(tlmp::gflops(result.attention_analytic.pv.flops,
                                         result.pv_timing.median_ms)));
  } else {
    fields.insert(fields.end(), 6, std::string());
  }

  fields.push_back(number(result.result_checksum));
  fields.push_back(std::to_string(result.warmups));
  fields.push_back(std::to_string(result.iterations));
  fields.push_back(std::to_string(result.sequence));
  fields.push_back(std::to_string(result.flush_cache_mib));
  for (std::size_t index = 0; index < fields.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << fields[index];
  }
  output << '\n';
}

void print_human(const BenchmarkResult& result) {
  std::cout << "kernel: " << result.kernel << '\n'
            << "configuration: " << result.configuration << '\n'
            << std::fixed << std::setprecision(4)
            << "median runtime: " << result.timing.median_ms << " ms"
            << " (min=" << result.timing.minimum_ms
            << ", max=" << result.timing.maximum_ms << ")\n"
            << "relative timing range: "
            << result.timing.relative_range_percent << "%\n"
            << std::scientific << std::setprecision(6)
            << "approximate FLOPs: " << result.analytic.flops << '\n'
            << std::fixed << std::setprecision(4)
            << "measured throughput: "
            << tlmp::gflops(result.analytic.flops, result.timing.median_ms)
            << " GFLOP/s\n"
            << std::scientific << std::setprecision(6)
            << "estimated minimum bytes: " << result.analytic.minimum_bytes
            << " bytes\n"
            << std::fixed << std::setprecision(4)
            << "analytical arithmetic intensity: "
            << result.analytic.arithmetic_intensity() << " FLOP/byte\n";

  if (result.has_attention_phases) {
    std::cout << "phase QK^T+scale: " << result.qk_timing.median_ms << " ms, "
              << tlmp::gflops(result.attention_analytic.qk.flops,
                               result.qk_timing.median_ms)
              << " approximate GFLOP/s\n"
              << "phase softmax: " << result.softmax_timing.median_ms
              << " ms, "
              << tlmp::gflops(result.attention_analytic.softmax.flops,
                               result.softmax_timing.median_ms)
              << " approximate GOP/s\n"
              << "phase P*V: " << result.pv_timing.median_ms << " ms, "
              << tlmp::gflops(result.attention_analytic.pv.flops,
                               result.pv_timing.median_ms)
              << " GFLOP/s\n";
  }
  std::cout << std::scientific << std::setprecision(9)
            << "checksum: " << result.result_checksum << '\n'
            << "timing policy: " << result.warmups << " warm-up, "
            << result.iterations << " measured iterations; median reported\n"
            << "note: minimum bytes are analytical, not measured DRAM traffic.\n";
  if (result.timing.relative_range_percent > 10.0) {
    std::cout << "warning: timing range exceeds 10%; inspect load, affinity, and frequency.\n";
  }
}

void emit_result(const BenchmarkResult& result, bool csv) {
  if (csv) {
    print_csv(result);
  } else {
    print_human(result);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    tlmp::configure_threads(options.threads);
    if (options.test_only) {
      return tlmp::run_correctness_tests(true, options.threads) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (options.kernel.empty()) {
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
    if (options.sequence >
        static_cast<std::size_t>(tinyllama::kMaxPositionEmbeddings)) {
      throw std::invalid_argument(
          "--seq exceeds TinyLlama max_position_embeddings=2048");
    }
    if (options.kernel != "gemm" && options.kernel != "gemv" &&
        options.kernel != "attention" && options.kernel != "all") {
      throw std::invalid_argument(
          "--kernel must be gemm, gemv, attention, or all");
    }
    if (!options.skip_tests && !tlmp::run_correctness_tests(false, options.threads)) {
      return EXIT_FAILURE;
    }

    std::ostringstream header;
    print_csv_header(header);
    tlmp::CsvOutput csv_file(options.output, header.str(), !options.no_save);
    const auto emit = [&](BenchmarkResult result) {
      result.configuration += ";threads=" + std::to_string(options.threads);
      std::ostringstream row;
      print_csv(result, row);
      csv_file.append(row.str());
      emit_result(result, options.csv);
    };

    if (options.csv && options.csv_header) {
      print_csv_header();
    }

    if (options.kernel == "gemm") {
      emit(benchmark_gemm(options.sequence, options.warmups,
                          options.iterations, options.flush_cache_mib, options.threads));
    } else if (options.kernel == "gemv") {
      emit(benchmark_gemv(options.warmups, options.iterations,
                          options.flush_cache_mib, options.threads));
    } else if (options.kernel == "attention") {
      emit(benchmark_attention(options.sequence, options.warmups,
                               options.iterations, options.flush_cache_mib, options.threads));
    } else if (options.kernel == "all") {
      for (std::size_t sequence : {128U, 512U, 2048U}) {
        emit(benchmark_gemm(sequence, options.warmups,
                            options.iterations, options.flush_cache_mib, options.threads));
      }
      emit(benchmark_gemv(options.warmups, options.iterations,
                          options.flush_cache_mib, options.threads));
      for (std::size_t sequence : {128U, 512U}) {
        emit(benchmark_attention(sequence, options.warmups,
                                 options.iterations, options.flush_cache_mib, options.threads));
      }
    }
    if (!options.no_save) {
      std::cerr << "CSV results appended to " << options.output << '\n';
      if (!options.no_plot) {
        const auto directory = std::filesystem::path(options.output).parent_path();
        const auto ceilings = options.ceilings.empty()
            ? (directory / "ceilings.csv").string() : options.ceilings;
        tlmp::update_roofline(options.output, ceilings,
                             (directory / "roofline.csv").string(), options.roof_threads);
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
