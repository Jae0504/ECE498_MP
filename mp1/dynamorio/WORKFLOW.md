# Generate the Roofline, measure native performance, and analyze selected metrics

The complete student guide is [mp1/README.md](../README.md). It covers building,
tuning native parameters, commands for each metric, and comparisons using
1, 2, 4, and 8 threads. This document is a quick reference for the 8-thread wrappers.

Run the commands below from the **repository root**. This MP1 baseline uses
8 physical cores. Roofline ceilings and execution times are measured on the
actual CPU. DynamoRIO traces the selected ROI in the same GEMM/GEMV/Attention
kernels under `mp1/cpu/` once, then reuses the saved trace for analysis.

## 1. Build and prepare paths

```sh
sh mp1/dynamorio/build.sh --jobs 8

# Select your 8 allocated CPUs. Replace 0-7 with your assigned CPU IDs.
export MP1_CPUS=0-7
MP1_PY="$PWD/mp1/dynamorio/.venv/bin/python"
MP1_CAL="$PWD/results/my_roofline"
MP1_RUN="$PWD/results/my_metrics"
```

Continue after `Build PASS`. If the build has already passed with the same
source and CPU, rebuilding is unnecessary. Use a fresh output directory for
each new experiment. Do not place files in `MP1_RUN` yet: the first measurement
command initializes it and records the environment and build information.

## 2. Validate CPU ceilings and generate the Roofline first

```sh
taskset -c "$MP1_CPUS" "$MP1_PY" mp1/scripts/calibrate_roofline.py \
  --bench mp1/build-dynamorio-native/roofline_bench \
  --threads 8 --max-threads 8 --sizes-mib 1024 2048 \
  --output "$MP1_CAL"

"$MP1_PY" mp1/scripts/plot_roofline_calibration.py "$MP1_CAL"

"$MP1_PY" mp1/scripts/generate_roofline.py \
  --ceilings "$MP1_CAL/ceilings.csv" \
  --kernels "$MP1_RUN/native_kernel_results.csv" \
  --roof-threads 8 --output "$MP1_CAL/roofline.csv"
```

This generates `my_roofline/roofline.png` and `.pdf`. Because no kernels have
been measured yet, the plot contains only the roof curves, without kernel
points. It is expected that the `--kernels` file does not exist yet.

The commands measure total working sets of 1 GiB and 2 GiB sequentially,
varying the core count across 1, 2, 4, and 8 and the number of independent FMA
accumulators. The defaults use 2 warmups and 9 measurements per condition,
with 2 rounds that reverse the condition order. Inspect saturation results in
`SATURATION_REPORT.md`, `verification.json`, `bandwidth_saturation.png`, and
`fma_saturation.png`.

**The 8-core ceiling is a reference for that allocation; it does not guarantee
memory saturation across the full socket.** Previous validation on a physical
Xeon 6761P server found a bandwidth plateau at 48-64 physical cores. That result
does not automatically apply to the current EWS allocation. Use the new report
to assess saturation for this run. Working sets that are too small relative
to the LLC may fail the saturation criteria.

The largest working set must be at most one quarter of `MemAvailable`. The
1 GiB and 2 GiB settings above are suitable for an EWS allocation with about
13 GiB available. If less memory is available, check `free -h` and reduce both
sizes. Omitting `--sizes-mib` selects large arrays automatically based on the
LLC size, which can cause memory-limit errors in a VM.

To reuse a completed calibration or resume an interrupted run, add `--resume`
to the first command. Use a new output path if the configuration or binary has
changed. On an Intel Xeon 6761P, add `--fma-per-cycle 2` to also check FMA
utilization using hardware cycles. This option requires `perf` permissions;
when using a different CPU, specify that CPU's throughput.

## 3. Measure native time and GFLOP/s for each kernel

To measure GEMV alone:

```sh
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemv.sh "$MP1_RUN" \
  --metrics native
```

To measure all five cases, run the following. Completed GEMV measurements are reused.

```sh
for mp1_case in gemv gemm128 gemm512 attention128 attention512; do
  taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_case.sh \
    "$mp1_case" "$MP1_RUN" --metrics native
done
```

`my_metrics/native_kernel_results.csv` records median time and GFLOP/s from
9 measurements after 2 warmups, along with algorithmic FLOPs, minimum bytes,
and arithmetic intensity (AI). Attention rows also include the individual
medians of `qk_ms`, `softmax_ms`, and `pv_ms` measured during full Attention runs.

Rerun the plotting command to add kernel points to the previously generated Roofline.

```sh
"$MP1_PY" mp1/scripts/generate_roofline.py \
  --ceilings "$MP1_CAL/ceilings.csv" \
  --kernels "$MP1_RUN/native_kernel_results.csv" \
  --roof-threads 8 --output "$MP1_CAL/roofline.csv"
```

## 4. Select trace metrics and cache simulation

List the desired names below after `--metrics`, separated by spaces.

| Selection | Results | Method |
|---|---|---|
| `native` | Execution time, GFLOP/s, Attention phase times; FLOPs/minimum bytes/AI | CPU timing and algorithmic formulas |
| `counts` | Instructions, Loads, Stores | Basic trace counts |
| `opcodes` | Execution counts by opcode, SIMD/FMA instruction distribution | Trace opcode mix |
| `reuse` | Mean/median reuse distance, histogram, data cold fraction | Exact reuse-distance analysis |
| `cache` | L1D/L2/LLC accesses, misses, miss rate | Cache-model simulation |
| `memory` | Dynamic Memory Reference Bytes, Unique Cache Lines | Data operand bytes and unique 64-byte cache lines |

To inspect GEMV instruction/load/store counts, then add cache and memory analysis
using the same trace:

```sh
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemv.sh "$MP1_RUN" \
  --metrics counts
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemv.sh "$MP1_RUN" \
  --metrics cache memory
```

To analyze only cache miss rates for GEMM S=128, then add opcode and reuse analysis:

```sh
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemm_s128.sh "$MP1_RUN" \
  --metrics cache
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemm_s128.sh "$MP1_RUN" \
  --metrics opcodes reuse
```

The default ROI is the entire kernel. Initialization, correctness tests, and
warmups are outside the ROI. OpenMP and call-management instructions inside
the ROI may appear in the trace. `counts` runs automatically once even when
only another metric such as `cache` is selected, because it is required for
ZIP conversion and validation of a complete 8-thread trace. Unrequested
opcode/reuse/cache/memory analyses and GEMV pilots are not run.

**`cache` uses simulation.** Do not use `instrumented_region_wall_ms` for
execution time or GFLOP/s. The cache model in `cache_model.cfg` specifies
L1D 48 KiB, L2 2 MiB, LLC 256 MiB, 8 cores, LRU, cold start, virtual addresses,
and no prefetching. This reference model approximates the previous gnr2
system's actual 336 MiB LLC. It does not represent hardware miss-counter
measurements or the actual cache specifications of the current EWS system.
The operand bytes reported by `memory` are not DRAM bytes. The mean and median
reported by `reuse` include each thread's instruction and data cache-line history.

## 5. Separate Attention into QK, softmax, and PV

To analyze full Attention and the softmax ROI separately:

```sh
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_attention_s128.sh "$MP1_RUN" \
  --metrics counts cache
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_attention_s128.sh "$MP1_RUN" \
  --phase softmax --metrics counts cache
```

To analyze each of the three phases:

```sh
for mp1_phase in qk softmax pv; do
  taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_attention_s128.sh "$MP1_RUN" \
    --phase "$mp1_phase" --metrics counts cache memory
done
```

For S=512, use `measure_attention_s512.sh` instead. Input preparation for each
phase occurs outside tracing. Each phase uses an independent trace and a
simulation with a cold cache, so do not sum phase misses and interpret them
as full Attention misses. Obtain native phase timings with `--metrics native`
in step 3. Combinations such as `--phase softmax --metrics native` are rejected
with an error.

## Results and reruns

| Path | Contents |
|---|---|
| `$MP1_CAL/roofline.png`, `.pdf`, `.csv` | Calibrated 8-core Roofline and native kernel points |
| `$MP1_CAL/SATURATION_REPORT.md` | Saturation validation for this ceiling measurement |
| `$MP1_RUN/native_kernel_results.csv` | Native performance and analytical metrics |
| `$MP1_RUN/metrics/gemv.json` | GEMV trace metrics completed so far |
| `$MP1_RUN/metrics/attention128_softmax.json` | S=128 softmax ROI metrics |
| `$MP1_RUN/logs/{counts,opcodes,reuse,cache,memory}_*.log` | Raw output from the selected analyses |
| `$MP1_RUN/traces/` | Traces available for further analysis |
| `$MP1_RUN/commands.jsonl`, `experiment_metadata.json` | Commands, CPU placement, and source/binary information |

The `--metrics` commands automatically reuse completed steps and native
measurement series. Select a new `MP1_RUN` when fresh timing samples or traces
are needed. Also use a new directory if tracing was interrupted and left an
incomplete trace. Measurements of selected metrics preserve the raw traces.
GEMM128 checks for at least 6 GiB of free disk space; GEMM512 checks for 18 GiB.

To generate the complete report, figures, and validation CSVs together, use
`sh mp1/dynamorio/measure_all.sh results/my_full_report` with a separate output
directory. This workflow measures its own 8-core reference ceiling and all
metrics. JSON files from selected measurements are not full-suite PASS reports.
Calibration and full-suite `ceilings.csv` files have different formats, so
always use **`$MP1_CAL/ceilings.csv`** for the plots in this guide.

## When full-socket saturation validation is needed

The following reference commands apply to an allocation of 64 physical cores
on a physical Xeon 6761P server. Use the earlier commands for the standard
8-thread EWS experiment. Save experiments on larger servers to a separate path.

```sh
taskset -c 0-63 "$MP1_PY" mp1/scripts/calibrate_roofline.py \
  --bench mp1/build-dynamorio-native/roofline_bench \
  --threads 8 --max-threads 64 --fma-per-cycle 2 \
  --output results/my_socket_calibration
"$MP1_PY" mp1/scripts/plot_roofline_calibration.py results/my_socket_calibration
```

When the saturation criteria are met, `saturated_ceilings.csv` contains the
ceiling for the maximum measured core count, which is 64 here. For comparisons
with an 8-core workload, use the original `ceilings.csv` with `--roof-threads 8`.
If you use the 64-core roof, state that resource scope in both the plot and
its interpretation.
