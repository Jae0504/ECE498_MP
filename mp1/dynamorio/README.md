# MP1 DynamoRIO: clone, build, measure

For the student **build → Roofline → native timing → selected metrics** workflow,
start with [mp1/README.md](../README.md). It includes kernel parameters, each metric,
Attention phase commands, and memory-budget settings for EWS. A shorter Korean
reference is available in [WORKFLOW.md](WORKFLOW.md). `--metrics` selects
`native`, `counts`, `opcodes`, `reuse`, `cache`, or `memory`; `--phase qk|softmax|pv`
selects an Attention trace ROI. This page retains advanced build troubleshooting
and the full-report workflow used when `--metrics` is omitted.

Use a repository commit containing this directory and the root `dynamorio` gitlink.
The official submodule URL is `https://github.com/DynamoRIO/dynamorio.git`, pinned to
`f50545b9fe31c536787de9f1e714fba99be7f810`. The build script initializes it and the
required nested submodules. It never uses `git submodule update --remote`.

```sh
git clone https://github.com/Jae0504/ECE498_MPs.git
cd ECE498_MPs
sh mp1/dynamorio/build.sh
```

This is a CPU-specific profile, tested on Linux x86_64, Intel Xeon 6761P,
GCC/G++ 13.3.0, CMake 3.28.3 and Python 3.12.3. The host needs a C/C++ toolchain,
CMake, Git, GNU make/time, taskset, Python with venv support and zlib development
files. Python 3.9 or newer is required. The lock file uses Python-version markers:
Python 3.9–3.11 selects a compatible pinned plotting stack (Matplotlib 3.9.4,
NumPy 2.0.2 and ContourPy 1.3.0); Python 3.12+ keeps the original gnr2 pins.
The interpreter inside `.venv` determines the selection. Build logs include
`python_version.log`, `python_packages.log` and `python_dependencies_check.log`.
The AVX-512F/FMA ceiling requires a supporting CPU. Missing server tools
are reported; no sudo or system-wide package installation is performed.

`build.sh [--jobs 8]` is incremental. It installs the pinned Python packages only
in `mp1/dynamorio/.venv`, builds LZ4 1.10.0 inside `mp1/dynamorio/build-deps` with
the custom allocator API needed by static tracing, builds DynamoRIO in Release,
builds the unchanged MP1 kernels and ROI helpers, runs all four original CTest
tests, and checks drrun. It does not rely on an earlier checkout's binaries, venv,
or `/usr/local/lib/liblz4.so`. Successful binary/source hashes and the correctness
gate are recorded in `mp1/dynamorio/build/build_state.json`. Build logs are under
`results/dynamorio_build/<timestamp>/`. Run `build.sh` again after moving the clone,
changing compiled source, or using a different CPU.
When CMake resets an existing cache after a compiler-path change, the build
reapplies its options to retain the local LZ4 path and Release configuration.

## Recover from a Python 3.9 dependency failure

An older copy of `requirements.lock` pinned ContourPy 1.3.3, which requires
Python >=3.11, and other packages requiring newer Python versions. If a Python
3.9 server reports `No matching distribution found`, update this file to the
version-marker lock supplied here, then rerun:

```sh
sh mp1/dynamorio/build.sh
```

The existing `.venv` is reused. The build invokes `.venv/bin/python -m pip`
directly, so activation is unnecessary. The fix selects compatible versions for
the entire plotting dependency set, including Python 3.9 backports. It does not
change the C++ kernels or replace a performance measurement. The original full
native/trace experiment was verified on gnr2; another server's toolchain and
native measurements must still pass its own build and correctness gates.

## Recover from an LZ4 feature-test failure

`configure_dynamorio: PASS` means only that CMake configuration returned zero.
If the next line reports an unusable LZ4 custom allocator, the build has stopped;
wait for `Build PASS` before measuring.

The pinned DynamoRIO source replaces `CMAKE_C_FLAGS`/`CMAKE_CXX_FLAGS` internally.
The earlier wrapper therefore could find the local LZ4 library while compiling
against an older system header. The updated wrapper passes the local include path
through the `CFLAGS`/`CXXFLAGS` environment that DynamoRIO preserves, supplies the
same environment to the build, and explicitly selects local headers for the ROI
helpers. It clears only `HAVE_LZ4_CUSTOM_MEM` with CMake `-U` so a previous failed
probe is rerun, while retaining the rest of the build tree. The test must pass;
the wrapper never forces it to true.

For a checkout still using the older wrapper, run this from the repository root
to supply the same include paths and retest the cached failure:

```sh
(
  export CFLAGS="-I$PWD/mp1/dynamorio/build-deps/lz4/lib ${CFLAGS:-}"
  export CXXFLAGS="-I$PWD/mp1/dynamorio/build-deps/lz4/lib ${CXXFLAGS:-}"
  cmake -S dynamorio -B dynamorio/build -U HAVE_LZ4_CUSTOM_MEM &&
  sh mp1/dynamorio/build.sh
)
```

After updating `build_local.py`, rerun `sh mp1/dynamorio/build.sh`. The existing
venv, dependency sources and build directory are reused. If the probe still
fails, inspect the `CMakeError.log` or `CMakeConfigureLog.yaml` copied into the
reported build-log directory for the actual compiler/linker error.

## Measure each workload

Run from any directory. An optional argument selects the shared output directory;
the default is `<repository>/results/dynamorio_student` (or `MP1_DR_RESULTS`).
Each script measures native performance and processes one complete trace through
counts, opcode mix, exact reuse distance, cache simulation and byte/unique-line
analysis. Every Attention script also measures its three separate phase ROIs.

```sh
sh mp1/dynamorio/measure_ceilings.sh       results/my_run
sh mp1/dynamorio/measure_gemv.sh           results/my_run
sh mp1/dynamorio/measure_gemm_s128.sh      results/my_run
sh mp1/dynamorio/measure_gemm_s512.sh      results/my_run
sh mp1/dynamorio/measure_attention_s128.sh results/my_run
sh mp1/dynamorio/measure_attention_s512.sh results/my_run
```

Or execute those exact shell entry points sequentially:

```sh
sh mp1/dynamorio/measure_all.sh results/my_run
```

Build and run everything in one command:

```sh
sh mp1/dynamorio/reproduce.sh results/my_run
```

Ceilings are measured once per result directory, automatically if necessary.
Requesting a non-GEMV workload first runs GEMV as a small full-trace validation
and includes that pilot as a normal GEMV row. The default is eight distinct
physical cores, two warmups, nine native samples, median kernel-region timing,
FP32 TinyLlama projection N=5632/K=2048 and single-head Attention D=64. The original
kernel code, warm-up policy, OMP settings, analytical formulas and trace analyses
are retained. No repeated full traces are used to choose more favorable counts.

The scripts refuse to overwrite the historical `results/dynamorio` experiment,
mix changed binaries/CPU sets in a result set, append a duplicate native row,
or run concurrently against one output directory. A fresh result directory means
a new measurement. Explicitly reuse completed stages after interruption with:

```sh
sh mp1/dynamorio/measure_all.sh results/my_run --resume
```

A partially written raw trace is not a complete trace and is never silently
reused. If tracing itself failed, preserve that output and start a fresh directory.
Completed expensive analyses can be reused with `--resume`; `run_trace.py` also
reuses an identical completed analysis stage unless `--force` is explicitly given; an incomplete phase
set needs its missing stages finished with `run_trace.py` or a fresh directory.
Raw `.raw.lz4` files are removed only after full successful analysis; converted ZIP
traces and metadata are retained. GEMM128 requires 6 GiB free and GEMM512 18 GiB;
small cases require 2.25 GiB. Commands stop on a time budget or less than 2 GiB
free disk and report failure, not a valid truncated measurement.

## CPU allocation on a shared server

Choose eight allowed logical CPUs that correspond to eight different physical
cores. For example, if assigned cores 8–15:

```sh
MP1_CPUS=8-15 taskset -c 8-15 sh mp1/dynamorio/measure_all.sh results/my_run
```

`MP1_CPUS` pins native and ROI execution. The enclosing `taskset` (or scheduler
CPU allocation) also confines offline analysis jobs, which otherwise inherit the
shell's allowed CPU set. The automatic selection takes the first eight physical
cores allowed to the process; it cannot allocate exclusive resources. Do not
assign every student CPUs 0–7. Use separate clones/output directories and scheduled
or disjoint CPU allocations. Native measurements never overlap offline analysis in this runner. Independent
count/reuse and cache/byte passes can overlap on the same fully converted GEMM
trace; stage locks prevent duplicate full analyses. Do not time native kernels alongside other heavy analysis on the same allocated cores.

## Results and comparison with the original run

After each workload completes, these files contain the completed workload rows:

- `kernel_characterization.csv` (the original exact eleven main columns)
- `trace_characterization.csv`, `ceilings.csv`, `roofline.csv`
- `roofline.png`, `cache_miss_rates.png`, `instruction_memory_mix.png`, `reuse_distance.png`
- `attention_phase_trace_characterization.csv` and `attention_phases.png` when relevant
- `DYNAMORIO_MP1_REPORT.md`, `environment.txt`, `experiment_metadata.json`
- `reference_comparison.csv`, `ceiling_reference_comparison.csv`, `phase_reference_comparison.csv` when relevant
- `verification.json`, `validation.txt`
- `commands.jsonl`, `trace_resources.csv`, raw tool logs and compressed traces

`reference/gnr2/` contains small CSVs copied from the actual original experiment,
plus provenance. **Analytical values and deterministic output checksums must
match. Runtime, GFLOP/s and all dynamic/cache/reuse numbers are compared, not
forced to equal the reference.** Scheduling, warm worker state, compiler output,
ASLR/cache mapping and shared-server load can change observations. PASS means
correctness, full-trace integrity and formula/cross-file checks succeeded.
`reference_comparison.csv` exposes the differences instead of hiding them.

The cache configuration deliberately preserves the original **reference model**:
8 cores, L1D 48 KiB/12-way, L1I 64 KiB/16-way, private L2 2 MiB/16-way,
shared LLC 256 MiB/16-way, LRU, virtual addresses, no prefetch, empty initial caches.
The real gnr2 LLC is 336 MiB; its set count is unsupported by this simulator at
16 ways. This model must not be called measured hardware cache misses or silently
presented as the hardware geometry of a different server. Main native timings
are warm; cache simulation starts cold. Dynamic operand bytes are not DRAM traffic.
Reuse statistics include instruction and data lines in per-thread histories.
Separate Attention phase cache results cannot be summed to obtain total misses.

To regenerate artifacts from an existing deployment run, without collecting a
second trace, run from this repository root:

```sh
export MP1_DR_RESULTS="$PWD/results/my_run"
mp1/dynamorio/.venv/bin/python mp1/dynamorio/summarize_phases.py --available # if Attention is present
mp1/dynamorio/.venv/bin/python mp1/dynamorio/summarize.py --available
mp1/dynamorio/.venv/bin/python mp1/dynamorio/write_run_report.py "$MP1_DR_RESULTS"
mp1/dynamorio/.venv/bin/python mp1/dynamorio/validate_deployment.py "$MP1_DR_RESULTS" --complete
```

Omit `--complete` when intentionally measuring fewer than all five workloads.
Report regeneration preserves the recorded experiment end time; measurement
wrappers update it when a new workload completes.
The older `write_report.py` and `validate_results.py` reproduce the historical
instructor report; deployment runs use `write_run_report.py` and
`validate_deployment.py` so another server is never mislabeled as the old run.

Packaging does not remove the full GEMM trace cost. The original five-workload
experiment retained about 6 GiB of compressed traces per user. Shared instructor
traces or scheduled batches remain preferable to unrestricted 30-user full runs.

## Roofline saturation calibration

The native ceiling code now uses parallel first touch, aligned buffers,
non-temporal triad stores with fences, and an AVX-512 read kernel on supported
hosts. `fma_ceiling` shares its register implementation with the default MP1
`roofline_bench`; its default 16 accumulators retain the reference work and
checksum. The standalone eight-core measurement remains a reference for that
allocation, not proof of whole-socket bandwidth saturation.

For a separate audit, run `mp1/scripts/calibrate_roofline.py` against
`mp1/build-dynamorio-native/roofline_bench`; see the MP1 README for the command.
It sweeps larger out-of-cache working sets, physical-core counts and FMA
accumulators, and can record hardware cycles. Increase `--max-threads` only
within your CPU allocation. `plot_roofline_calibration.py` renders its report.
Those calibration CSVs use the standard MP1 ceiling schema, so use them with
`mp1/scripts/generate_roofline.py`, not the DynamoRIO-specific summarizer. Keep
the thread scope explicit when choosing the requested or saturated roof.
