# ECE 498 MP1: CPU Architecture Profiling

This repository provides an architecture-focused machine problem built around
TinyLlama-1.1B tensor dimensions:

- [`mp1/`](mp1/): CPU timing on the provided EWS servers, analytical
  operation/data-movement models, source-level data-access and attention
  analysis, OpenMP scaling across 1/2/4/8 threads, DynamoRIO trace analysis,
  cache simulation, and a common Roofline.

Intentionally naive kernel implementations are provided as the starting point.
Students run a controlled baseline experiment, validate its results, and explain
its bottlenecks using timing, data-access, and Roofline evidence.

Follow the [MP1 instructions](mp1/README.md) for build, Roofline, and individual
measurement commands, and the [MP1 report guide](mp1/REPORT_GUIDE.md) for the report.

## Clone

```bash
git clone https://github.com/Jae0504/ECE498_MP.git
cd ECE498_MP
```

No `--recurse-submodules` option is needed. TinyLlama is not a build or runtime
dependency, so its source repository is intentionally not included as a Git
submodule. The benchmark uses random tensors and needs no model weights.

The verified model-configuration snapshot and its authoritative source links
are retained under [`reference/`](reference/).

## MP2

Use the course DGX Spark with CUDA 13+, CMake, and a C++17 compiler. MP2
targets the GB10 GPU (`sm_121`) and builds directly on Spark's ARM64 Linux
host. It uses generated tensors and requires no model weights or submodules.
Follow the [MP2 instructions](mp2/README.md) and
[report guide](mp2/REPORT_GUIDE.md).

From the repository root, save the initial measurements in a fresh directory:

```sh
export RESULT_DIR="$PWD/mp2/results/naive"
bash mp2/scripts/run_gpu.sh
bash mp2/scripts/validate_gpu.sh
SKIP_BASELINE_RUN=1 bash mp2/scripts/profile_gpu.sh
```

`run_gpu.sh` builds and checks the baseline before measuring all five required
cases. `validate_gpu.sh` runs all four Compute Sanitizer checks. GPU latency
uses CUDA events and excludes allocation and host/device copies. Exit code
`77` means GPU execution was unavailable, not a passing run. Nsight Compute
requires access to GPU performance counters; use the profiling instructions
and record any permission restriction. Use normal CUDA-event timings for
speedups, not application timings collected during Nsight Compute replay.

Keep the original results and source revision, then implement and evaluate
coalescing, shared-memory tiling, occupancy tuning, and a Tensor Core variant.
Validate each change and save its measurements in a fresh result directory.
Report reduced-precision Tensor Core experiments separately from the unchanged
FP32 correctness checks.

Begin with the [GEMV block-size example](mp2/README.md#first-optimization-example-gemv-block-size).
In `mp2/gpu/gemv.cu`, comment out the active 256-thread definition and uncomment
the 64-thread definition, keeping exactly one active. Rebuild, validate, and
compare with the saved baseline. This small experiment demonstrates the
optimization workflow; a performance improvement is not guaranteed.

Compare all five cases against the naive GPU baseline, MP1's native 8-thread
CPU results, and an optimized library such as cuBLAS. Students implement the
optimized kernels and library adapters. Reuse MP1's native CPU CSV and two-part
report, keeping GEMV warm-cache and displaced-cache comparisons separate.
Explain the CPU/GPU timing boundaries, improvements, unsuccessful attempts,
and remaining bottlenecks in the MP2 report.

## Workload rationale

TinyLlama is a decoder-only Transformer and contains no convolution layer.
Accordingly, these MPs use:

- GEMM for multi-token prefill;
- GEMV for one-token autoregressive decode;
- one-head materialized attention for sequence-length scaling and phase
  analysis.

GEMV is especially useful here because it provides a clean low-arithmetic
intensity contrast with the otherwise identical GEMM projection. A convolution
workload can be added separately if a later course version requires that exact
kernel family.
