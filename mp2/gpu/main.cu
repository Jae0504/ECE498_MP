#include "gpu_kernels.cuh"

#include "common.h"
#include "cpu_kernels.h"
#include "metrics.h"
#include "timer.h"
#include "tinyllama_config.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kGpuUnavailableExitCode = 77;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
  ~CudaEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  cudaEvent_t get() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    check_cuda(cudaMalloc(reinterpret_cast<void**>(&pointer_),
                          count * sizeof(T)),
               "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() { return pointer_; }
  const T* get() const { return pointer_; }
  std::size_t count() const { return count_; }

 private:
  T* pointer_ = nullptr;
  std::size_t count_ = 0;
};

template <typename T>
void copy_to_device(DeviceBuffer<T>& destination,
                    const std::vector<T>& source) {
  if (destination.count() != source.size()) {
    throw std::invalid_argument("host/device buffer size mismatch");
  }
  check_cuda(cudaMemcpy(destination.get(), source.data(),
                        source.size() * sizeof(T), cudaMemcpyHostToDevice),
             "cudaMemcpy host-to-device");
}

template <typename T>
void copy_to_host(std::vector<T>& destination,
                  const DeviceBuffer<T>& source) {
  if (destination.size() != source.count()) {
    throw std::invalid_argument("device/host buffer size mismatch");
  }
  check_cuda(cudaMemcpy(destination.data(), source.get(),
                        destination.size() * sizeof(T), cudaMemcpyDeviceToHost),
             "cudaMemcpy device-to-host");
}

struct DeviceInfo {
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  std::size_t global_memory = 0;
};

bool query_device(DeviceInfo& info, std::string& failure) {
  int count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&count);
  if (count_status != cudaSuccess) {
    failure = std::string("cudaGetDeviceCount failed: ") +
              cudaGetErrorString(count_status);
    return false;
  }
  if (count == 0) {
    failure = "CUDA runtime reports zero devices";
    return false;
  }

  cudaDeviceProp properties{};
  const cudaError_t property_status = cudaGetDeviceProperties(&properties, 0);
  if (property_status != cudaSuccess) {
    failure = std::string("cudaGetDeviceProperties failed: ") +
              cudaGetErrorString(property_status);
    return false;
  }
  info.name = properties.name;
  info.compute_major = properties.major;
  info.compute_minor = properties.minor;
  info.global_memory = properties.totalGlobalMem;
  return true;
}

struct Options {
  std::string kernel;
  std::size_t sequence = 512;
  int warmups = 1;
  int iterations = 5;
  bool test_only = false;
  bool required_test_only = false;
  bool skip_tests = false;
  bool csv = false;
  bool csv_header = true;
};

struct BenchmarkResult {
  std::string kernel;
  std::string configuration;
  tlmp::TimingStats timing;
  tlmp::AnalyticMetrics analytic;
  double result_checksum = 0.0;
  int warmups = 0;
  int iterations = 0;
  bool has_attention_phases = false;
  tlmp::TimingStats qk_timing;
  tlmp::TimingStats softmax_timing;
  tlmp::TimingStats pv_timing;
  tlmp::AttentionAnalyticMetrics attention_analytic;
};

void print_usage(const char* program) {
  std::cout
      << "TinyLlama educational CUDA microbenchmarks\n\n"
      << "Usage:\n"
      << "  " << program << " --test\n"
      << "  " << program << " --test-required\n"
      << "  " << program
      << " --kernel gemm|gemv|attention|all [options]\n\n"
      << "  --seq N            Sequence length (default: 512)\n"
      << "  --warmup N         Warm-up iterations (default: 1)\n"
      << "  --iterations N     Timed iterations (default: 5)\n"
      << "  --format human|csv\n"
      << "  --no-header\n"
      << "  --skip-tests       Skip preflight test after it was run separately\n"
      << "  --test             Run small and non-multiple correctness cases\n"
      << "  --test-required    Run required TinyLlama-size correctness cases\n"
      << "  --help\n";
}

std::size_t parse_size(const std::string& text, const std::string& option) {
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(text, &consumed, 10);
  if (consumed != text.size() || parsed == 0ULL) {
    throw std::invalid_argument(option + " requires a positive integer");
  }
  return static_cast<std::size_t>(parsed);
}

int parse_int(const std::string& text, const std::string& option,
              bool allow_zero) {
  std::size_t consumed = 0;
  const long parsed = std::stol(text, &consumed, 10);
  if (consumed != text.size() || parsed < 0 || (!allow_zero && parsed == 0)) {
    throw std::invalid_argument(option + " has an invalid integer value");
  }
  return static_cast<int>(parsed);
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
    if (argument == "--kernel") {
      options.kernel = next();
    } else if (argument == "--seq") {
      options.sequence = parse_size(next(), argument);
    } else if (argument == "--warmup") {
      options.warmups = parse_int(next(), argument, true);
    } else if (argument == "--iterations" || argument == "--iters") {
      options.iterations = parse_int(next(), argument, false);
    } else if (argument == "--format") {
      const std::string format = next();
      if (format == "csv") {
        options.csv = true;
      } else if (format == "human") {
        options.csv = false;
      } else {
        throw std::invalid_argument("--format must be human or csv");
      }
    } else if (argument == "--no-header") {
      options.csv_header = false;
    } else if (argument == "--test") {
      options.test_only = true;
    } else if (argument == "--test-required") {
      options.required_test_only = true;
    } else if (argument == "--skip-tests") {
      options.skip_tests = true;
    } else if (argument == "--help" || argument == "-h") {
      print_usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.test_only && options.required_test_only) {
    throw std::invalid_argument(
        "--test and --test-required are separate test suites");
  }
  return options;
}

int checked_int(std::size_t value, const char* description) {
  if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error(std::string(description) + " exceeds CUDA int range");
  }
  return static_cast<int>(value);
}

template <typename Launch>
tlmp::TimingStats measure_cuda_kernel(int warmups, int iterations,
                                      Launch&& launch) {
  for (int iteration = 0; iteration < warmups; ++iteration) {
    check_cuda(launch(), "CUDA kernel launch (warm-up)");
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after warm-up");

  CudaEvent start;
  CudaEvent stop;
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(iterations));
  for (int iteration = 0; iteration < iterations; ++iteration) {
    check_cuda(cudaEventRecord(start.get()), "cudaEventRecord start");
    check_cuda(launch(), "CUDA kernel launch");
    check_cuda(cudaEventRecord(stop.get()), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop.get()), "cudaEventSynchronize");
    float milliseconds = 0.0F;
    check_cuda(cudaEventElapsedTime(&milliseconds, start.get(), stop.get()),
               "cudaEventElapsedTime");
    samples.push_back(milliseconds);
  }
  return tlmp::summarize_timings(std::move(samples));
}

bool compare_vectors(const std::string& name, const std::vector<float>& actual,
                     const std::vector<float>& expected, float absolute_tolerance,
                     float relative_tolerance, bool verbose) {
  if (actual.size() != expected.size()) {
    std::cerr << "[FAIL] " << name << ": size mismatch (actual="
              << actual.size() << ", expected=" << expected.size() << ")\n";
    return false;
  }
  std::size_t mismatch_count = 0;
  double maximum_absolute_error = 0.0;
  double maximum_relative_error = 0.0;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    const double absolute_error =
        std::fabs(static_cast<double>(actual[index]) -
                  static_cast<double>(expected[index]));
    const double scale =
        std::max(std::fabs(static_cast<double>(actual[index])),
                 std::fabs(static_cast<double>(expected[index])));
    const double relative_error = scale > 0.0 ? absolute_error / scale : 0.0;
    maximum_absolute_error = std::max(maximum_absolute_error, absolute_error);
    maximum_relative_error = std::max(maximum_relative_error, relative_error);
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]) ||
        !tlmp::close_enough(actual[index], expected[index], absolute_tolerance,
                            relative_tolerance)) {
      if (mismatch_count < 5U) {
        std::cerr << "[FAIL] " << name << " at index " << index
                  << ": GPU=" << actual[index]
                  << ", CPU reference=" << expected[index]
                  << ", absolute error=" << absolute_error
                  << ", relative error=" << relative_error << '\n';
      }
      ++mismatch_count;
    }
  }
  if (mismatch_count != 0U) {
    std::cerr << "[FAIL] " << name << ": " << mismatch_count
              << " mismatches out of " << actual.size()
              << " elements; tolerance=" << absolute_tolerance << "+"
              << relative_tolerance << "*scale\n";
    return false;
  }
  if (verbose) {
    std::cout << "[PASS] " << name << " (" << actual.size()
              << " elements, max_abs_error=" << maximum_absolute_error
              << ", max_rel_error=" << maximum_relative_error
              << ", tolerance=" << absolute_tolerance << "+"
              << relative_tolerance << "*scale)\n";
  }
  return true;
}

bool validate_softmax_rows(const std::string& name,
                           const std::vector<float>& probabilities, int rows,
                           int columns, double row_sum_tolerance,
                           bool verbose) {
  double maximum_row_sum_error = 0.0;
  for (int row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (int column = 0; column < columns; ++column) {
      const float probability = probabilities[static_cast<std::size_t>(row) *
                                                  columns +
                                              column];
      if (!std::isfinite(probability)) {
        std::cerr << "[FAIL] " << name << ": non-finite probability at row "
                  << row << ", column " << column << '\n';
        return false;
      }
      if (probability < 0.0F) {
        std::cerr << "[FAIL] " << name << ": negative probability at row "
                  << row << ", column " << column << ": " << probability
                  << '\n';
        return false;
      }
      sum += static_cast<double>(probability);
    }
    const double error = std::fabs(sum - 1.0);
    maximum_row_sum_error = std::max(maximum_row_sum_error, error);
    if (error > row_sum_tolerance) {
      std::cerr << "[FAIL] " << name << ": row " << row << " sums to "
                << std::setprecision(12) << sum << " (error=" << error
                << ", tolerance=" << row_sum_tolerance << ")\n";
      return false;
    }
  }
  if (verbose) {
    std::cout << "[PASS] " << name
              << " probabilities are finite, nonnegative, and rows sum to 1"
              << " (max_row_sum_error=" << maximum_row_sum_error
              << ", tolerance=" << row_sum_tolerance << ")\n";
  }
  return true;
}

bool run_gemm_correctness_case(int m, int n, int k,
                               float absolute_tolerance,
                               float relative_tolerance, std::uint32_t seed,
                               bool verbose) {
  const std::size_t a_elements =
      tlmp::checked_elements(static_cast<std::size_t>(m),
                             static_cast<std::size_t>(k));
  const std::size_t b_elements =
      tlmp::checked_elements(static_cast<std::size_t>(k),
                             static_cast<std::size_t>(n));
  const std::size_t c_elements =
      tlmp::checked_elements(static_cast<std::size_t>(m),
                             static_cast<std::size_t>(n));
  std::vector<float> a(a_elements);
  std::vector<float> b(b_elements);
  std::vector<float> actual(c_elements);
  std::vector<float> expected(c_elements);
  tlmp::fill_random(a, seed);
  tlmp::fill_random(b, seed + 1U);
  tlmp::gemm_reference(a.data(), b.data(), expected.data(), m, n, k);
  DeviceBuffer<float> device_a(a.size());
  DeviceBuffer<float> device_b(b.size());
  DeviceBuffer<float> device_c(actual.size());
  copy_to_device(device_a, a);
  copy_to_device(device_b, b);
  check_cuda(tlmp::gpu::launch_gemm(device_a.get(), device_b.get(),
                                    device_c.get(), m, n, k),
             "GEMM test launch");
  copy_to_host(actual, device_c);
  std::ostringstream name;
  name << "CUDA GEMM M=" << m << ",N=" << n << ",K=" << k;
  return compare_vectors(name.str(), actual, expected, absolute_tolerance,
                         relative_tolerance, verbose);
}

bool run_gemv_correctness_case(int n, int k, float absolute_tolerance,
                               float relative_tolerance, std::uint32_t seed,
                               bool verbose) {
  std::vector<float> x(static_cast<std::size_t>(k));
  std::vector<float> weights(
      tlmp::checked_elements(static_cast<std::size_t>(k),
                             static_cast<std::size_t>(n)));
  std::vector<float> actual(static_cast<std::size_t>(n));
  std::vector<float> expected(static_cast<std::size_t>(n));
  tlmp::fill_random(x, seed);
  tlmp::fill_random(weights, seed + 1U);
  tlmp::gemv_reference(x.data(), weights.data(), expected.data(), n, k);
  DeviceBuffer<float> device_x(x.size());
  DeviceBuffer<float> device_weights(weights.size());
  DeviceBuffer<float> device_y(actual.size());
  copy_to_device(device_x, x);
  copy_to_device(device_weights, weights);
  check_cuda(tlmp::gpu::launch_gemv(device_x.get(), device_weights.get(),
                                    device_y.get(), n, k),
             "GEMV test launch");
  copy_to_host(actual, device_y);
  std::ostringstream name;
  name << "CUDA GEMV N=" << n << ",K=" << k;
  return compare_vectors(name.str(), actual, expected, absolute_tolerance,
                         relative_tolerance, verbose);
}

bool run_attention_correctness_case(int sequence, int dimension,
                                    float output_absolute_tolerance,
                                    float output_relative_tolerance,
                                    float probability_absolute_tolerance,
                                    float probability_relative_tolerance,
                                    double row_sum_tolerance,
                                    std::uint32_t seed, bool verbose) {
  const std::size_t tensor_elements =
      tlmp::checked_elements(static_cast<std::size_t>(sequence),
                             static_cast<std::size_t>(dimension));
  const std::size_t score_elements =
      tlmp::checked_elements(static_cast<std::size_t>(sequence),
                             static_cast<std::size_t>(sequence));
  std::vector<float> query(tensor_elements);
  std::vector<float> key(tensor_elements);
  std::vector<float> value(tensor_elements);
  std::vector<float> actual_output(tensor_elements);
  std::vector<float> expected_output(tensor_elements);
  std::vector<float> actual_scores(score_elements);
  std::vector<float> expected_scores(score_elements);
  std::vector<float> actual_probabilities(score_elements);
  std::vector<float> expected_probabilities(score_elements);
  tlmp::fill_random(query, seed);
  tlmp::fill_random(key, seed + 1U);
  tlmp::fill_random(value, seed + 2U);
  tlmp::attention_reference(query.data(), key.data(), value.data(),
                            expected_output.data(), sequence, dimension,
                            expected_probabilities.data(),
                            expected_scores.data());

  DeviceBuffer<float> device_query(tensor_elements);
  DeviceBuffer<float> device_key(tensor_elements);
  DeviceBuffer<float> device_value(tensor_elements);
  DeviceBuffer<float> device_scores(score_elements);
  DeviceBuffer<float> device_output(tensor_elements);
  copy_to_device(device_query, query);
  copy_to_device(device_key, key);
  copy_to_device(device_value, value);
  check_cuda(tlmp::gpu::launch_attention_qk(
                 device_query.get(), device_key.get(), device_scores.get(),
                 sequence, dimension),
             "Attention QK test launch");
  copy_to_host(actual_scores, device_scores);
  check_cuda(tlmp::gpu::launch_softmax(device_scores.get(), sequence, sequence),
             "Attention softmax test launch");
  check_cuda(tlmp::gpu::launch_attention_pv(
                 device_scores.get(), device_value.get(), device_output.get(),
                 sequence, dimension),
             "Attention PV test launch");
  copy_to_host(actual_probabilities, device_scores);
  copy_to_host(actual_output, device_output);

  std::ostringstream prefix;
  prefix << "CUDA Attention S=" << sequence << ",D=" << dimension;
  const bool scores_ok = compare_vectors(
      prefix.str() + " QK^T+scale", actual_scores, expected_scores,
      probability_absolute_tolerance, probability_relative_tolerance,
      verbose);
  const bool probabilities_ok = compare_vectors(
      prefix.str() + " softmax", actual_probabilities,
      expected_probabilities, probability_absolute_tolerance,
      probability_relative_tolerance, verbose);
  const bool rows_ok = validate_softmax_rows(
      prefix.str() + " softmax", actual_probabilities, sequence, sequence,
      row_sum_tolerance, verbose);
  const bool output_ok = compare_vectors(
      prefix.str() + " output", actual_output, expected_output,
      output_absolute_tolerance, output_relative_tolerance, verbose);
  return scores_ok && probabilities_ok && rows_ok && output_ok;
}

bool run_gpu_fast_tests(bool verbose) {
  const bool gemm_small = run_gemm_correctness_case(
      8, 8, 8, 3.0e-5F, 3.0e-5F, tlmp::kRandomSeed + 201U, verbose);
  const bool gemv_small = run_gemv_correctness_case(
      8, 8, 3.0e-5F, 3.0e-5F, tlmp::kRandomSeed + 203U, verbose);
  const bool attention_small = run_attention_correctness_case(
      8, 8, 8.0e-5F, 8.0e-5F, 8.0e-5F, 8.0e-5F, 1.0e-5,
      tlmp::kRandomSeed + 205U, verbose);
  const bool gemm_boundary = run_gemm_correctness_case(
      127, 257, 63, 2.0e-4F, 2.0e-4F, tlmp::kRandomSeed + 211U, verbose);
  const bool gemv_boundary = run_gemv_correctness_case(
      257, 63, 2.0e-4F, 2.0e-4F, tlmp::kRandomSeed + 213U, verbose);
  const bool attention_boundary = run_attention_correctness_case(
      129, 64, 3.0e-4F, 3.0e-4F, 8.0e-5F, 8.0e-5F, 2.0e-5,
      tlmp::kRandomSeed + 215U, verbose);
  if (!(gemm_small && gemv_small && attention_small && gemm_boundary &&
        gemv_boundary && attention_boundary)) {
    std::cerr << "CUDA fast/boundary correctness suite FAILED\n";
    return false;
  }
  if (verbose) {
    std::cout << "All CUDA small and non-multiple correctness tests passed.\n";
  }
  return true;
}

bool run_gpu_required_tests(bool verbose) {
  const bool gemm_s128 = run_gemm_correctness_case(
      128, tinyllama::kIntermediateSize, tinyllama::kHiddenSize, 1.0e-3F,
      1.0e-3F, tlmp::kRandomSeed + 221U, verbose);
  const bool gemm_s512 = run_gemm_correctness_case(
      512, tinyllama::kIntermediateSize, tinyllama::kHiddenSize, 1.0e-3F,
      1.0e-3F, tlmp::kRandomSeed + 223U, verbose);
  const bool gemv = run_gemv_correctness_case(
      tinyllama::kIntermediateSize, tinyllama::kHiddenSize, 1.0e-3F,
      1.0e-3F, tlmp::kRandomSeed + 225U, verbose);
  const bool attention_s128 = run_attention_correctness_case(
      128, tinyllama::kHeadDim, 3.0e-4F, 3.0e-4F, 8.0e-5F, 8.0e-5F,
      2.0e-5, tlmp::kRandomSeed + 227U, verbose);
  const bool attention_s512 = run_attention_correctness_case(
      512, tinyllama::kHeadDim, 3.0e-4F, 3.0e-4F, 8.0e-5F, 8.0e-5F,
      2.0e-5, tlmp::kRandomSeed + 229U, verbose);
  if (!(gemm_s128 && gemm_s512 && gemv && attention_s128 &&
        attention_s512)) {
    std::cerr << "CUDA required-size correctness suite FAILED\n";
    return false;
  }
  if (verbose) {
    std::cout << "All required TinyLlama-size correctness tests passed.\n";
  }
  return true;
}

BenchmarkResult benchmark_gemm(std::size_t sequence, int warmups,
                               int iterations) {
  const std::size_t m = sequence;
  const std::size_t n = tinyllama::kIntermediateSize;
  const std::size_t k = tinyllama::kHiddenSize;
  std::vector<float> a(tlmp::checked_elements(m, k));
  std::vector<float> b(tlmp::checked_elements(k, n));
  std::vector<float> c(tlmp::checked_elements(m, n));
  tlmp::fill_random(a, tlmp::kRandomSeed + 10U);
  tlmp::fill_random(b, tlmp::kRandomSeed + 11U);
  DeviceBuffer<float> device_a(a.size());
  DeviceBuffer<float> device_b(b.size());
  DeviceBuffer<float> device_c(c.size());
  copy_to_device(device_a, a);
  copy_to_device(device_b, b);

  const tlmp::TimingStats timing = measure_cuda_kernel(
      warmups, iterations, [&] {
        return tlmp::gpu::launch_gemm(
            device_a.get(), device_b.get(), device_c.get(), checked_int(m, "M"),
            checked_int(n, "N"), checked_int(k, "K"));
      });
  copy_to_host(c, device_c);
  std::ostringstream configuration;
  configuration << "S=" << sequence << ";M=" << m << ";N=" << n
                << ";K=" << k;
  BenchmarkResult result;
  result.kernel = "gemm";
  result.configuration = configuration.str();
  result.timing = timing;
  result.analytic = tlmp::gemm_metrics(m, n, k);
  result.result_checksum = tlmp::checksum(c);
  result.warmups = warmups;
  result.iterations = iterations;
  return result;
}

BenchmarkResult benchmark_gemv(int warmups, int iterations) {
  const std::size_t n = tinyllama::kIntermediateSize;
  const std::size_t k = tinyllama::kHiddenSize;
  std::vector<float> x(k);
  std::vector<float> weights(tlmp::checked_elements(k, n));
  std::vector<float> y(n);
  tlmp::fill_random(x, tlmp::kRandomSeed + 20U);
  tlmp::fill_random(weights, tlmp::kRandomSeed + 21U);
  DeviceBuffer<float> device_x(x.size());
  DeviceBuffer<float> device_weights(weights.size());
  DeviceBuffer<float> device_y(y.size());
  copy_to_device(device_x, x);
  copy_to_device(device_weights, weights);

  const tlmp::TimingStats timing = measure_cuda_kernel(
      warmups, iterations, [&] {
        return tlmp::gpu::launch_gemv(
            device_x.get(), device_weights.get(), device_y.get(),
            checked_int(n, "N"), checked_int(k, "K"));
      });
  copy_to_host(y, device_y);
  std::ostringstream configuration;
  configuration << "tokens=1;N=" << n << ";K=" << k;
  BenchmarkResult result;
  result.kernel = "gemv";
  result.configuration = configuration.str();
  result.timing = timing;
  result.analytic = tlmp::gemv_metrics(n, k);
  result.result_checksum = tlmp::checksum(y);
  result.warmups = warmups;
  result.iterations = iterations;
  return result;
}

BenchmarkResult benchmark_attention(std::size_t sequence, int warmups,
                                    int iterations) {
  const std::size_t dimension = tinyllama::kHeadDim;
  const std::size_t tensor_elements =
      tlmp::checked_elements(sequence, dimension);
  const std::size_t score_elements =
      tlmp::checked_elements(sequence, sequence);
  std::vector<float> query(tensor_elements);
  std::vector<float> key(tensor_elements);
  std::vector<float> value(tensor_elements);
  std::vector<float> output(tensor_elements);
  tlmp::fill_random(query, tlmp::kRandomSeed + 30U);
  tlmp::fill_random(key, tlmp::kRandomSeed + 31U);
  tlmp::fill_random(value, tlmp::kRandomSeed + 32U);
  DeviceBuffer<float> device_query(tensor_elements);
  DeviceBuffer<float> device_key(tensor_elements);
  DeviceBuffer<float> device_value(tensor_elements);
  DeviceBuffer<float> device_scores(score_elements);
  DeviceBuffer<float> device_output(tensor_elements);
  copy_to_device(device_query, query);
  copy_to_device(device_key, key);
  copy_to_device(device_value, value);
  const int s = checked_int(sequence, "sequence");
  const int d = checked_int(dimension, "head dimension");

  const auto launch_all = [&] {
    check_cuda(tlmp::gpu::launch_attention_qk(
                   device_query.get(), device_key.get(), device_scores.get(), s,
                   d),
               "Attention QK warm-up launch");
    check_cuda(tlmp::gpu::launch_softmax(device_scores.get(), s, s),
               "Attention softmax warm-up launch");
    check_cuda(tlmp::gpu::launch_attention_pv(
                   device_scores.get(), device_value.get(), device_output.get(),
                   s, d),
               "Attention PV warm-up launch");
  };
  for (int iteration = 0; iteration < warmups; ++iteration) {
    launch_all();
  }
  check_cuda(cudaDeviceSynchronize(), "Attention warm-up synchronization");

  CudaEvent start;
  CudaEvent qk_stop;
  CudaEvent softmax_stop;
  CudaEvent pv_stop;
  std::vector<double> total_samples;
  std::vector<double> qk_samples;
  std::vector<double> softmax_samples;
  std::vector<double> pv_samples;
  total_samples.reserve(static_cast<std::size_t>(iterations));
  qk_samples.reserve(static_cast<std::size_t>(iterations));
  softmax_samples.reserve(static_cast<std::size_t>(iterations));
  pv_samples.reserve(static_cast<std::size_t>(iterations));

  for (int iteration = 0; iteration < iterations; ++iteration) {
    check_cuda(cudaEventRecord(start.get()), "Attention start event");
    check_cuda(tlmp::gpu::launch_attention_qk(
                   device_query.get(), device_key.get(), device_scores.get(), s,
                   d),
               "Attention QK launch");
    check_cuda(cudaEventRecord(qk_stop.get()), "Attention QK stop event");
    check_cuda(tlmp::gpu::launch_softmax(device_scores.get(), s, s),
               "Attention softmax launch");
    check_cuda(cudaEventRecord(softmax_stop.get()),
               "Attention softmax stop event");
    check_cuda(tlmp::gpu::launch_attention_pv(
                   device_scores.get(), device_value.get(), device_output.get(),
                   s, d),
               "Attention PV launch");
    check_cuda(cudaEventRecord(pv_stop.get()), "Attention PV stop event");
    check_cuda(cudaEventSynchronize(pv_stop.get()),
               "Attention event synchronization");

    float total_ms = 0.0F;
    float qk_ms = 0.0F;
    float softmax_ms = 0.0F;
    float pv_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&total_ms, start.get(), pv_stop.get()),
               "Attention total elapsed time");
    check_cuda(cudaEventElapsedTime(&qk_ms, start.get(), qk_stop.get()),
               "Attention QK elapsed time");
    check_cuda(cudaEventElapsedTime(&softmax_ms, qk_stop.get(),
                                    softmax_stop.get()),
               "Attention softmax elapsed time");
    check_cuda(cudaEventElapsedTime(&pv_ms, softmax_stop.get(), pv_stop.get()),
               "Attention PV elapsed time");
    total_samples.push_back(total_ms);
    qk_samples.push_back(qk_ms);
    softmax_samples.push_back(softmax_ms);
    pv_samples.push_back(pv_ms);
  }
  copy_to_host(output, device_output);

  const tlmp::AttentionAnalyticMetrics analytic =
      tlmp::attention_metrics(sequence, dimension);
  std::ostringstream configuration;
  configuration << "S=" << sequence << ";D=" << dimension
                << ";heads=1;unmasked=1";
  BenchmarkResult result;
  result.kernel = "attention";
  result.configuration = configuration.str();
  result.timing = tlmp::summarize_timings(std::move(total_samples));
  result.analytic = analytic.total;
  result.result_checksum = tlmp::checksum(output);
  result.warmups = warmups;
  result.iterations = iterations;
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

void print_csv_header() {
  std::cout
      << "kernel,configuration,median_ms,p10_ms,p90_ms,min_ms,max_ms,"
         "relative_range_percent,"
         "approx_flops,measured_gflops,estimated_minimum_bytes,"
         "arithmetic_intensity,qk_ms,qk_p10_ms,qk_p90_ms,qk_gflops,"
         "softmax_ms,softmax_p10_ms,softmax_p90_ms,softmax_gops,"
         "pv_ms,pv_p10_ms,pv_p90_ms,pv_gflops,device,checksum,warmups,"
         "iterations\n";
}

void print_csv(const BenchmarkResult& result, const DeviceInfo& device) {
  std::vector<std::string> fields = {
      result.kernel,
      result.configuration,
      number(result.timing.median_ms),
      number(result.timing.p10_ms),
      number(result.timing.p90_ms),
      number(result.timing.minimum_ms),
      number(result.timing.maximum_ms),
      number(result.timing.relative_range_percent),
      number(result.analytic.flops),
      number(tlmp::gflops(result.analytic.flops, result.timing.median_ms)),
      number(result.analytic.minimum_bytes),
      number(result.analytic.arithmetic_intensity())};
  if (result.has_attention_phases) {
    fields.push_back(number(result.qk_timing.median_ms));
    fields.push_back(number(result.qk_timing.p10_ms));
    fields.push_back(number(result.qk_timing.p90_ms));
    fields.push_back(number(tlmp::gflops(result.attention_analytic.qk.flops,
                                         result.qk_timing.median_ms)));
    fields.push_back(number(result.softmax_timing.median_ms));
    fields.push_back(number(result.softmax_timing.p10_ms));
    fields.push_back(number(result.softmax_timing.p90_ms));
    fields.push_back(number(tlmp::gflops(
        result.attention_analytic.softmax.flops,
        result.softmax_timing.median_ms)));
    fields.push_back(number(result.pv_timing.median_ms));
    fields.push_back(number(result.pv_timing.p10_ms));
    fields.push_back(number(result.pv_timing.p90_ms));
    fields.push_back(number(tlmp::gflops(result.attention_analytic.pv.flops,
                                         result.pv_timing.median_ms)));
  } else {
    fields.insert(fields.end(), 12, std::string());
  }
  fields.push_back(device.name);
  fields.push_back(number(result.result_checksum));
  fields.push_back(std::to_string(result.warmups));
  fields.push_back(std::to_string(result.iterations));
  for (std::size_t index = 0; index < fields.size(); ++index) {
    if (index != 0U) {
      std::cout << ',';
    }
    std::cout << fields[index];
  }
  std::cout << '\n';
}

void print_human(const BenchmarkResult& result, const DeviceInfo& device) {
  std::cout << "device: " << device.name << " (compute capability "
            << device.compute_major << '.' << device.compute_minor << ")\n"
            << "kernel: " << result.kernel << '\n'
            << "configuration: " << result.configuration << '\n'
            << std::fixed << std::setprecision(4)
            << "median kernel runtime: " << result.timing.median_ms << " ms"
            << " (p10=" << result.timing.p10_ms
            << ", p90=" << result.timing.p90_ms
            << ", min=" << result.timing.minimum_ms
            << ", max=" << result.timing.maximum_ms << ")\n"
            << "relative timing range: "
            << result.timing.relative_range_percent << "%\n"
            << "measured throughput: "
            << tlmp::gflops(result.analytic.flops, result.timing.median_ms)
            << " GFLOP/s\n"
            << std::scientific << std::setprecision(6)
            << "approximate FLOPs: " << result.analytic.flops << '\n'
            << "estimated minimum bytes: " << result.analytic.minimum_bytes
            << " bytes\n"
            << std::fixed << std::setprecision(4)
            << "analytical arithmetic intensity: "
            << result.analytic.arithmetic_intensity() << " FLOP/byte\n";
  if (result.has_attention_phases) {
    std::cout << "phase QK^T+scale: " << result.qk_timing.median_ms
              << " ms (p10=" << result.qk_timing.p10_ms
              << ", p90=" << result.qk_timing.p90_ms << ")\n"
              << "phase softmax: " << result.softmax_timing.median_ms
              << " ms (p10=" << result.softmax_timing.p10_ms
              << ", p90=" << result.softmax_timing.p90_ms << ")\n"
              << "phase P*V: " << result.pv_timing.median_ms
              << " ms (p10=" << result.pv_timing.p10_ms
              << ", p90=" << result.pv_timing.p90_ms << ")\n";
  }
  std::cout << std::scientific << std::setprecision(9)
            << "checksum: " << result.result_checksum << '\n'
            << "samples: " << result.iterations
            << " independent unbatched CUDA-event measurements after "
            << result.warmups << " warm-ups\n"
            << "note: GEMM/GEMV samples contain one kernel; Attention samples contain one three-kernel sequence.\n"
            << "note: CUDA events time kernels only; allocation and copies are excluded.\n"
            << "note: minimum bytes are analytical, not measured DRAM traffic.\n";
}

void emit(const BenchmarkResult& result, const DeviceInfo& device, bool csv) {
  if (csv) {
    print_csv(result, device);
  } else {
    print_human(result, device);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    DeviceInfo device;
    std::string failure;
    if (!query_device(device, failure)) {
      std::cerr << "GPU unavailable: " << failure << '\n'
                << "The CUDA sources can still be compiled, but runtime tests "
                   "and profiling are deferred.\n";
      return kGpuUnavailableExitCode;
    }

    if (options.test_only) {
      return run_gpu_fast_tests(true) ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    if (options.required_test_only) {
      return run_gpu_required_tests(true) ? EXIT_SUCCESS : EXIT_FAILURE;
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
    if (!options.skip_tests && !run_gpu_fast_tests(false)) {
      return EXIT_FAILURE;
    }
    if (options.csv && options.csv_header) {
      print_csv_header();
    }

    if (options.kernel == "gemm") {
      emit(benchmark_gemm(options.sequence, options.warmups,
                          options.iterations),
           device, options.csv);
    } else if (options.kernel == "gemv") {
      emit(benchmark_gemv(options.warmups, options.iterations), device,
           options.csv);
    } else if (options.kernel == "attention") {
      emit(benchmark_attention(options.sequence, options.warmups,
                               options.iterations),
           device, options.csv);
    } else if (options.kernel == "all") {
      for (std::size_t sequence : {128U, 512U, 2048U}) {
        emit(benchmark_gemm(sequence, options.warmups, options.iterations),
             device, options.csv);
      }
      emit(benchmark_gemv(options.warmups, options.iterations), device,
           options.csv);
      for (std::size_t sequence : {128U, 512U}) {
        emit(benchmark_attention(sequence, options.warmups,
                                 options.iterations),
             device, options.csv);
      }
    } else {
      throw std::invalid_argument(
          "--kernel must be gemm, gemv, attention, or all");
    }
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
