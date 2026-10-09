# MP2: GPU Experiments on NVIDIA DGX Spark

## Learning objective

Run the same TinyLlama-derived GEMM, GEMV, and simplified dense-attention
workloads as MP1 on the NVIDIA DGX Spark GPU. Start from the intentionally naive CUDA
kernels in this repository, profile them, and then implement and evaluate your
own optimizations.

The starter kernels are deliberately simple. They do not contain shared-memory
tiled GEMM, split-K GEMV, fused attention, Tensor Core intrinsics, or optimized
library replacements. Keep an unchanged baseline measurement so that every
later result has a fair comparison point. Your work is to implement and explain
successive improvements using coalescing, tiling, occupancy tuning, and Tensor
Core experiments, then compare your best correct kernels with MP1 CPU results
and optimized libraries. The starter supplies a working naive baseline and
correctness/timing infrastructure; optimized kernels and library adapters are
student work.

**All commands below run from the repository root, `ECE498_MPs/`, in the same
terminal.** Select a fresh result directory for the baseline:

```bash
export RESULT_DIR="$PWD/mp2/results/naive"
```

## Required workloads

All arrays contain row-major `float32` values.

| Kernel | Shape | Required cases |
|---|---|---|
| GEMM (prefill) | `[S x 2048] * [2048 x 5632]` | `S=128`, `S=512` |
| GEMV (one-token decode) | `[1 x 2048] * [2048 x 5632]` | one token |
| One dense attention head | `Q,K,V=[S x 64]` | `S=128`, `S=512` |

Attention launches QK-transpose-plus-scale, row softmax, and P-times-V as three
separate kernels, with scale `1/sqrt(64)` and no causal mask. This is not full
GQA or full TinyLlama inference.

### Continue from MP1

Reuse `mp1/results/kernel_results.csv` and the two-part
[MP1 report](../mp1/REPORT_GUIDE.md): Roofline analysis and DynamoRIO analysis.
Transfer those saved results from EWS if you use a separate Spark machine.
MP2 needs no MP1 executable, trace tool, submodule, model weights, or downloaded
dataset. `cpu/reference.cpp` is a correctness oracle, not an MP1 timing baseline.

Match CPU and GPU rows by kernel and tensor dimensions. MP1's `configuration`
also contains `threads` and `cache`, so the full strings are not equal. Use
the required **8-thread** MP1 measurements, and report GEMV's warm and
displaced-cache comparisons separately. Optional 1-thread comparisons must
be labeled separately. Preserve MP1's FLOP and analytical-byte formulas in
`include/metrics.h`, including the approximate softmax operation count.

```text
GPU speedup over CPU = MP1 native median_ms / GPU median_ms
optimization speedup = naive GPU median_ms / optimized GPU median_ms
library-relative speedup = library median_ms / optimized GPU median_ms
```

MP1 times a native CPU invocation including output clearing and OpenMP overhead;
MP2 times GPU work using CUDA events, excluding allocation and copies. Explain
these timing boundaries and the different host systems. Compare against MP1's
native timings; use its DynamoRIO measurements to explain behavior.

## Requirements

- NVIDIA DGX Spark (GB10, ARM64 Linux) with a working NVIDIA driver;
- CUDA Toolkit 13.0 or newer on Spark;
- `nvcc`, CMake 3.18 or newer, and a C++17 host compiler;
- Nsight Compute (`ncu`) for hardware-counter profiling;
- Nsight Systems (`nsys`) for the lightweight attention timeline;
- Compute Sanitizer for memory, initialization, synchronization, and race checks.

The default build targets GB10 compute capability 12.1 (`sm_121`), following
[NVIDIA's Spark build guidance](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/compilation.html)
and [supported software versions](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/porting/dependencies.html).
Build the host executable on Spark; an x86-64 executable built on EWS does not
run on its ARM64 CPU. For another GPU, set `CMAKE_CUDA_ARCHITECTURES` manually
or `CUDA_ARCHITECTURES` for scripts (for example, `90` for H100). Record that
hardware as a separate experiment.

Record the environment before interpreting results:

```bash
bash mp2/scripts/inspect_environment.sh
```

The scripts search `CUDA_HOME`, `CUDAToolkit_ROOT`, `PATH`,
`/usr/local/cuda/bin`, and versioned `/usr/local/cuda-*/bin` directories. An
instructor can select exact tools with:

```bash
NVCC_BIN=/path/to/nvcc \
NCU_BIN=/path/to/ncu \
NSYS_BIN=/path/to/nsys \
COMPUTE_SANITIZER_BIN=/path/to/compute-sanitizer \
bash mp2/scripts/inspect_environment.sh
```

An explicit override must name a usable executable; the scripts do not
silently replace an invalid override. Compiler discovery rejects an `nvcc`
that cannot target `CUDA_ARCHITECTURES` (default `121`), even if its version
command succeeds. If a tool is missing, use the course-provided CUDA
installation or ask course staff to provision it.

## Build and correctness

On DGX Spark, using its CUDA 13+ installation:

```bash
cmake -S mp2 -B mp2/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=121
cmake --build mp2/build -j 4
```

Run both correctness suites:

```bash
mp2/build/gpu_bench --test
mp2/build/gpu_bench --test-required
```

`--test` covers the original aligned `8x8` cases and non-multiple boundaries:

- GEMM `M=127,N=257,K=63`;
- GEMV `N=257,K=63`;
- Attention `S=129,D=64`.

`--test-required` checks every required educational configuration: GEMM
`S=128/512`, the one-token GEMV, and Attention `S=128/512,D=64`. Every output
element is compared with an independent CPU reference. Attention additionally
checks the QK-plus-scale and softmax intermediate arrays, rejects NaN/Inf and
negative probabilities, and checks that every softmax row sums approximately
to one. Checksums are printed by benchmarks but never determine test success.

CTest runs the same suites; the required-size test is labeled `long`:

```bash
ctest --test-dir mp2/build --output-on-failure
```

With no usable GPU, the executable and experiment scripts return `77`, and
CTest marks GPU tests **Skipped**. A successful compile or skipped CTest run
does not establish GPU correctness. `run_gpu.sh` writes `status=success` only
after both test suites and every requested timing case finish successfully.
If Python 3 is available, CTest also runs `script_workflow`, which checks
result preservation and failure handling with fake tools; it does not execute
GPU kernels.

The FP32 comparison rule is:

```text
abs(actual - expected)
<= absolute_tolerance + relative_tolerance * max(abs(actual), abs(expected))
```

The deterministic tests currently use:

| Output | Absolute tolerance | Relative tolerance |
|---|---:|---:|
| Small GEMM/GEMV | `3e-5` | `3e-5` |
| Boundary GEMM/GEMV | `2e-4` | `2e-4` |
| Required GEMM/GEMV | `1e-3` | `1e-3` |
| Attention QK/softmax | `8e-5` | `8e-5` |
| Small Attention output | `8e-5` | `8e-5` |
| Boundary/required Attention output | `3e-4` | `3e-4` |

Softmax row-sum tolerances are `1e-5` for the small case and `2e-5` for the
boundary/required cases. Test output records the observed maximum errors.

Run all four Compute Sanitizer checks with:

```bash
bash mp2/scripts/validate_gpu.sh
```

This runs memcheck, initcheck, synccheck, and racecheck on the small/boundary
suite and writes regenerated logs under `$RESULT_DIR/sanitizer/`.

## Normal performance experiment

Run the required experiment matrix with:

```bash
bash mp2/scripts/run_gpu.sh
```

The script first runs both correctness suites. It then records GEMM S=128/512,
one-token GEMV, and Attention S=128/512 in `$RESULT_DIR/kernel_results.csv`.
It also builds the benchmark, so the manual build above is optional. Wait for
`GPU results written to ...` and check `$RESULT_DIR/runtime_status.txt`.
Attention S=128 uses 101 measured samples; the CSV reports median, p10, and p90
for the total and each phase. Percentiles use linear interpolation at
`p * (sample_count - 1)` in sorted samples.

Each GEMM/GEMV timing sample contains one kernel launch. Each Attention sample
contains one QK/softmax/PV launch sequence, with separate CUDA events for its
three phases. Repeated-kernel batching is not used.

CUDA-event intervals exclude allocation, random initialization, host-to-device
copies, and device-to-host copies. The total Attention interval can include
GPU-side gaps between its three kernels. It is not end-to-end inference
latency.

Optional S=2048 diagnostic cases can be added with:

```bash
RESULT_DIR="$PWD/mp2/results/large" INCLUDE_LARGE=1 bash mp2/scripts/run_gpu.sh
```

An existing `kernel_results.csv` is never overwritten. For each optimization,
set a fresh `RESULT_DIR`, such as `mp2/results/tiled_v1`, and save the matching
source revision and compiler flags. `BUILD_DIR` optionally selects a separate
build tree; use a fresh tree when changing CUDA compilers. Without `RESULT_DIR`,
scripts use `mp2/results/`. A CSV is published only after the full experiment
matrix completes, so an interrupted run cannot masquerade as complete results.

## Nsight profiling

Generate the compact baseline profile with:

```bash
SKIP_BASELINE_RUN=1 bash mp2/scripts/profile_gpu.sh
```

Run this after `run_gpu.sh` for the same source revision and `RESULT_DIR`.
It checks correctness again without overwriting the normal timing CSV.
The required Nsight Compute baseline reports are limited to:

- GEMM S=512;
- one-token GEMV;
- Attention S=128.

The selected sections cover launch dimensions, achieved occupancy, SM and
memory throughput, L1/TEX behavior, and cache behavior without collecting the
entire metric catalog. `PROFILE_ATTENTION_S512=1` adds an instructor diagnostic
case. Nsight Systems records a CUDA-only Attention S=128 timeline with CPU
sampling disabled, so restricted `perf_event_paranoid` settings are nonfatal.

### Timing warning

Normal runtime and GFLOP/s must come from the CUDA-event experiment generated
by `run_gpu.sh`.

For Nsight Compute analysis, use Nsight's own `Duration` and hardware counters.
Do **not** use CUDA-event runtime or GFLOP/s printed by the application while
Nsight Compute replay is active as a normal performance result. Replay changes
execution conditions and may launch a kernel multiple times.

The `.ncu-rep`, `.nsys-rep`, SQLite, stdout, and status files under
`$RESULT_DIR/profile/` are generated artifacts and are not source files.
Inspect `ncu_status.txt`, `nsys_status.txt`, and the corresponding stderr files;
counter-permission errors may also appear in stdout. The script returns `1`
if a profiler fails, is unavailable, or does not produce a nonempty report;
it returns `77` if GPU execution is unavailable. A counter-permission failure
does not invalidate successful correctness tests or normal CUDA-event timings.

## Student implementation and optimization

1. Establish and retain the naive baseline for all five required cases. Explain
   how threads map to outputs and memory addresses in `gpu/gemm.cu`,
   `gpu/gemv.cu`, and `gpu/attention.cu`.
2. Implement incremental optimizations in those files while preserving the
   `launch_*` interfaces. Investigate coalescing and shared-memory tiling for
   matrix operations, parallel reductions and launch size for GEMV, and
   block size/register/shared-memory effects on occupancy. Record unsuccessful
   attempts too, with evidence explaining their behavior.
3. Evaluate a Tensor Core matrix-multiply variant and report its input,
   multiplication, accumulation, and output precision. Keep the FP32 comparison
   separate from reduced-precision experiments; validate error against the
   independent reference and justify any separate tolerance. Do not loosen the
   supplied FP32 tests to make a reduced-precision implementation pass.
4. Re-run both correctness suites and Compute Sanitizer after each kernel
   change, then collect normal event timings in a fresh result directory.
   Re-profile the optimized versions of the relevant baseline cases to explain
   the speedup and remaining bottlenecks. A library call alone is the library
   comparison, not a student-written kernel optimization.
5. Implement and time optimized-library comparisons under the same shapes,
   inputs, precision, warmups, sample counts, and timing boundaries. Compare
   all five cases with the naive GPU and the corresponding MP1 CPU results.

The phase interfaces and phase tests support optimizing the existing three
Attention kernels. Fused Attention is an optional extension: it also needs an
end-to-end reference check and a timing path that measures the fused kernel.
Retain the original analytical AI as the MP1 comparison convention and label
any revised traffic model separately.

### First optimization example: GEMV block size

[gpu/gemv.cu](gpu/gemv.cu) includes a commented optimization example in
`launch_gemv`. The default is the original 256-thread baseline:

```cpp
constexpr int threads = 256;  // Naive baseline (default).
// constexpr int threads = 64;  // Example optimization: more blocks.
```

First, leave the default active and save its measurements:

```bash
RESULT_DIR="$PWD/mp2/results/gemv_block256" bash mp2/scripts/run_gpu.sh
```

Next, **comment out the 256-thread line and uncomment the 64-thread line**.
Keep exactly one definition active:

```cpp
// constexpr int threads = 256;  // Naive baseline (default).
constexpr int threads = 64;  // Example optimization: more blocks.
```

The grid is calculated from `threads`, so no other code change is needed.
For the required `N=5632` outputs, this changes the launch from 22 blocks to
88 blocks. More blocks can distribute the work across more of GB10's 48 SMs.
Each thread still computes one output, and the arithmetic and FP32 precision
are unchanged.

Rebuild, validate, and save a separate measurement. Both scripts below rebuild
the executable, so the edited source takes effect:

```bash
RESULT_DIR="$PWD/mp2/results/gemv_block64" bash mp2/scripts/validate_gpu.sh
RESULT_DIR="$PWD/mp2/results/gemv_block64" bash mp2/scripts/run_gpu.sh
```

Use fresh directory names if these results already exist. Compare the `gemv`
row in the two `kernel_results.csv` files:

```text
block-size speedup = block256 median_ms / block64 median_ms
```

A ratio above 1 means the example improved performance. This is a small
optimization experiment, not a guaranteed speedup: our GB10 trials showed only
a small difference, with overlapping ranges across repeated runs. Memory
traffic is unchanged, and a smaller block does not automatically improve
achieved occupancy or performance. Compare `p10_ms`/`p90_ms`, repeat both
configurations under the same conditions, and report a tie or slowdown if that
is what you measure. This example introduces the edit/rebuild/validate/measure
workflow; continue with your own kernel optimizations.

### Optimized-library comparison

Use [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) or an equivalent
documented optimized library. The toolkit supplies cuBLAS; add
`CUDA::cublas` to your target's `target_link_libraries` when implementing
the adapter. The starter does not yet expose a library implementation selector.

| Workload | Library comparison |
|---|---|
| GEMM | `cublasSgemm` or `cublasGemmEx`; account for the row-major input layout |
| GEMV | `cublasSgemv` with the corresponding layout/transpose convention |
| Attention | cuBLAS QK and PV plus the same validated softmax, timing the whole sequence |

Label the last row **cuBLAS + custom softmax**, since cuBLAS does not supply
the softmax phase. Alternatively, use a full attention library with the same
single-head, unmasked semantics and scale. Verify every output element before
timing; handle creation, allocation, and one-time preparation stay outside the
timed region. Include any per-invocation conversion or packing cost, or report
it separately and explicitly identify the prepacked timing scope. Document
the library version and math mode; FP32 storage alone does not identify the
internal multiplication precision. Report Tensor Core / reduced-precision
results separately with their observed errors.

## Validation status and target hardware

The current starter was executed on 2026-10-01 on one NVIDIA H100 PCIe
(compute capability 9.0, 81,559 MiB, driver 580.167.08):

- Release `sm_90` build with `nvcc` 13.2.78;
- all six small/boundary and five required-size correctness cases passed;
- Compute Sanitizer 2026.1.1 memcheck/initcheck/synccheck: 0 errors on the
  small/boundary suite; racecheck: 0 hazards, 0 errors, 0 warnings;
- all five required normal CUDA-event measurements completed;
- Nsight Systems 2025.6.3 generated the Attention timeline, including all
  three GPU phases;
- Nsight Compute 2026.1.1 was blocked by `ERR_NVGPUCTRPERM` on this host.
  Its three reports were not generated. The profiling script records the
  counter restriction and returns failure; the kernel/sanitizer results pass.

These checks required access to the host GPU. An earlier run inside a restricted
sandbox could not see the NVIDIA device and skipped execution.

Compilation for the default `sm_121` target also passed. H100 execution does
not establish GB10/ARM64 execution or expected Spark performance. Before course
deployment, repeat correctness, sanitizer, timing, and profiling on the course
DGX Spark and retain the generated logs. Where counters are restricted, follow
the fallback evidence instructions in the report guide.

## Report

Use [REPORT_GUIDE.md](REPORT_GUIDE.md). Keep analytical FLOPs/bytes identical
to MP1 so CPU/GPU comparisons refer to the same workloads. Use the EWS CPU
timings and analysis from the two-part [MP1 report](../mp1/REPORT_GUIDE.md)
for this comparison, identifying the CPU worker count and GEMV cache condition
used. Keep speedups against 1-thread and 8-thread CPU results distinct. Generated
build and result files are intentionally excluded from Git.
