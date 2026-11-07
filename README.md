# ECE 498 Machine Problems

This repository contains the course machine problems:

- [MP1: CPU Architecture Profiling](mp1/README.md): native CPU timing,
  data-access analysis, OpenMP scaling, DynamoRIO traces, cache simulation,
  and Roofline analysis using TinyLlama-1.1B tensor dimensions.
- [MP2: GPU Experiments on NVIDIA DGX Spark](mp2/README.md): run the same
  kernels on the GPU, optimize them, and compare with MP1 CPU results and
  optimized libraries.
- [MP3: AI Accelerator DSE — Parameter Optimization](mp3/README.md): execute
  tiny miniLLAMA using supplied INT16 PE-array RTL outputs and explore
  parameters of the fixed hardware implementation.
- [MP4: AI Accelerator DSE — Architecture and Implementation](mp4/README.md):
  implement architectural changes in SystemVerilog and compare the baseline,
  best MP3 configuration, and new design.

MP1 and MP2 provide intentionally naive kernel implementations as the starting point.
Students run a controlled baseline experiment, validate its results, and explain
its bottlenecks using timing, data-access, and Roofline evidence.

Follow the [MP1 instructions](mp1/README.md) for build, Roofline, and individual
measurement commands, and the [MP1 report guide](mp1/REPORT_GUIDE.md) for the report.

## Clone

```bash
git clone https://github.com/Jae0504/ECE498_MP.git
cd ECE498_MP
```

No `--recurse-submodules` option is needed. Each MP's setup initializes only
its required submodules. MP1 retains its pinned `dynamorio/` dependency.
MP3/MP4 require `miniLLAMA/`, `cocotb/`, and `verilator/`; setup initializes
their pinned revisions and patches a generated miniLLAMA copy. See
[MP3/MP4 dependencies](mp3/docs/DEPENDENCIES.md).

MP1/MP2 use random tensors with TinyLlama-1.1B dimensions and need no model
weights or TinyLlama source checkout. The verified model-configuration snapshot
and its authoritative source links are retained under [`reference/`](reference/).
MP3/MP4 execute a separate tiny miniLLAMA model with seeded random weights by default.

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

## MP3 and MP4

The required path is **miniLLAMA → Python/cocotb → Verilator RTL → miniLLAMA**.
MP3 tunes a supplied INT16/INT64 PE-array baseline; MP4 edits its SystemVerilog
copy. Actual RTL matrix results determine model logits and generated tokens.
The shared teaching baseline has 16×4 PEs, 8 KiB of input buffers, independent
8 B/cycle A/B/C ports, and an assumed 100 MHz clock.

| Assignment | Student edits | Evaluation |
| --- | --- | --- |
| MP3 | Configuration JSON and a list of parameter candidates; keep RTL fixed | Baseline and every candidate on prefill and decode; select using a declared objective |
| MP4 | `mp4/rtl_student/*.sv`; keep the selected MP3 resource budget | Reproduce MP3 first, then compare baseline, MP3 best, and modified RTL |

Run from the repository root and use fresh result directories:

```sh
sh mp3/setup_rtl.sh
MP_PY="$PWD/mp3/.venv/bin/python"

# MP3: edit the candidate list, then evaluate it on both workloads.
cp mp3/configs/rtl_candidates.json mp3/my_candidates.json
# Edit mp3/my_candidates.json before running the sweep.
"$MP_PY" mp3/rtl_run.py sweep \
  --candidates mp3/my_candidates.json --out mp3/results/dse_01
cat mp3/results/dse_01/sweep.md
cp mp3/results/dse_01/best_design.json mp3/best_design.json

# MP4: prepare the editable copy and reproduce MP3 before changing RTL.
"$MP_PY" mp3/tools/prepare_rtl_student.py
"$MP_PY" mp3/rtl_run.py check --rtl-root mp4/rtl_student \
  --design mp3/best_design.json --out mp4/results/initial_check_01
"$MP_PY" mp3/rtl_run.py run --rtl-root mp4/rtl_student \
  --design mp3/best_design.json --workload mp3/configs/workload_decode.json \
  --out mp4/results/unchanged_decode_01
```

The supplied candidates start the search; extend them with your own hypotheses.
For MP4, edit `controller.sv`, `tile_buffer.sv`, `pe_array.sv`, `pe.sv`, or
`accelerator_top.sv` under `mp4/rtl_student/`, then run `check`, `run`, and
`synth` with `--rtl-root mp4/rtl_student`. The complete commands for comparing
both workloads are in [MP3](mp3/README.md) and [MP4](mp4/README.md).
Use the [MP3 report template](mp3/report_template.md) and
[MP4 report guide](mp4/REPORT_GUIDE.md) for submission requirements.

See the [shared RTL workflow](mp3/RTL_WORKFLOW.md) for Linux/Python/compiler
prerequisites, local tool setup, student file locations and verification.
Setup builds the pinned Verilator source and installs cocotb and YoWASP Yosys
in `mp3/.venv`; no commercial license, GPU or sudo is required. Initial setup
needs internet; prepared experiments run offline. MP4 shares this environment.

TTFT/TPOT/latency are simulated **matrix-path contributions** from observed
RTL cycles. PyTorch executes non-matrix operators, outside that timing scope.
Yosys supplies generic logic/register/memory counts. Activity costs are
uncalibrated dimensionless proxies; whole-accelerator physical power and area
are unavailable in the supplied flow. See [PPA scope](mp3/docs/rtl_ppa.md).

## Workload rationale

These decoder-only Transformer workloads use:

- GEMM for multi-token prefill;
- GEMV for one-token autoregressive decode;
- attention matrix products for sequence-length scaling and phase analysis.

MP1/MP2 use standalone TinyLlama-sized kernels; MP3/MP4 execute a tiny miniLLAMA
model sized for CPU-based RTL simulation. MP1/MP2's standalone attention case
measures one head with materialized attention scores.

GEMV is especially useful here because it provides a clean low-arithmetic
intensity contrast with the otherwise identical GEMM projection. A convolution
workload can be added separately if a later course version requires that exact
kernel family.
