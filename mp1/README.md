# MP1: CPU Architecture Profiling and Roofline Analysis

Measure the provided FP32 GEMM, GEMV, and attention kernels on the EWS servers.
Use native timing, analytical operation/data-movement models, DynamoRIO traces,
and cache simulation to explain their architectural behavior. Keep the provided
implementations and required dimensions for the baseline; label any additional
optimization separately.

Follow this order, running one experiment at a time:

1. [Build](#build).
2. [Set the CPU allocation and output paths](#set-the-cpu-allocation-and-output-paths).
3. [Measure CPU ceilings and generate the Roofline](#measure-ceilings-and-automatically-update-your-roofline-plot).
4. [Measure each kernel's native CPU performance](#measure-native-cpu-metrics).
5. [Collect DynamoRIO metrics to explain the Roofline results](#dynamorio-instruction-cache-and-reuse-analysis).
6. [Write the two-part report](REPORT_GUIDE.md).

**All commands below run from the repository root, `ECE498_MPs/`, in the same
terminal.** The course baseline uses **8 workers and a common 8-thread Roofline**.
Native timing supports 1/2/4/8 workers; the packaged trace profile uses 8 workers.

## What is measured where?

| Quantity | How it is obtained | Command / output |
|---|---|---|
| Kernel runtime, timing variation, Attention phase times | Native CPU execution without DynamoRIO instrumentation | `cpu_bench`; native CSV |
| FLOPs, minimum bytes, arithmetic intensity (AI) | Analytical formulas for the selected tensor dimensions | `cpu_bench`; native CSV |
| GFLOP/s | Analytical FLOPs divided by native median time | `cpu_bench`; native CSV |
| Roofline bandwidth and compute ceilings | Native streaming-memory and register-FMA microbenchmarks | `calibrate_roofline.py`; ceiling CSV |
| Instructions, loads, stores | Count references in one complete ROI trace | `--metrics counts` |
| Opcode distribution | Count executed instruction mnemonics in that trace | `--metrics opcodes` |
| Reuse distance | Analyze cache-line reuse in the trace | `--metrics reuse` |
| L1D/L2/LLC accesses and misses | Simulate the configured cache hierarchy using the trace | `--metrics cache` |
| Operand bytes and unique data cache lines | Sum data-reference widths and count distinct 64-byte lines | `--metrics memory` |

DynamoRIO executes on the CPU too, but instrumented execution time is not the
kernel's native performance. The Roofline uses analytical AI and native GFLOP/s.
Trace operand bytes and simulated cache misses are different quantities from
measured DRAM traffic.

## Build

After the [clone instructions](../README.md#clone), run:

```sh
sh mp1/dynamorio/build.sh --jobs 8
```

This builds both the native CPU benchmarks and the DynamoRIO tools. It initializes
the pinned root `dynamorio/` submodule and required nested dependencies, prepares
Python packages, builds Release binaries, runs the four CTest tests,
and checks DynamoRIO. No virtual-environment activation or separate native build
is needed. Wait for **`Build PASS`** before measuring.

## Set the CPU allocation and output paths

Use eight allowed CPUs with distinct core IDs in the topology visible to the OS.
Replace `0-7` with the allocation assigned to you. Native measurements and offline
analysis run sequentially on this allocation.

```sh
export MP1_CPUS=0-7
export OMP_PLACES=threads
export OMP_PROC_BIND=close
export OMP_DYNAMIC=false
export OMP_WAIT_POLICY=PASSIVE
export GOMP_SPINCOUNT=0

MP1_PY="$PWD/mp1/dynamorio/.venv/bin/python"
MP1_NATIVE="$PWD/mp1/build-dynamorio-native"
MP1_CAL="$PWD/mp1/results/calibration"
MP1_CPU="$PWD/mp1/results"
MP1_RUN="$PWD/mp1/results/dynamorio"
```

Set these variables again in each new terminal. Native CSVs and Roofline plots
go directly in `mp1/results/`; calibration and trace files go in its
`calibration/` and `dynamorio/` subdirectories. The commands create these paths.

## Measure ceilings and automatically update your Roofline plot

Run the calibration script **before measuring the kernels**:

```sh
taskset -c "$MP1_CPUS" "$MP1_PY" mp1/scripts/calibrate_roofline.py \
  --bench "$MP1_NATIVE/roofline_bench" \
  --threads 8 --max-threads 8 \
  --sizes-mib 1024 2048 --rounds 2 --iterations 9 \
  --output "$MP1_CAL"

"$MP1_PY" mp1/scripts/plot_roofline_calibration.py "$MP1_CAL"

"$MP1_PY" mp1/scripts/generate_roofline.py \
  --ceilings "$MP1_CAL/ceilings.csv" \
  --kernels "$MP1_CPU/kernel_results.csv" \
  --roof-threads 8 --output "$MP1_CPU/roofline.csv"
```

This creates `mp1/results/roofline.png` and `.pdf`. The kernel CSV does not
need to exist yet: initially the plot contains the ceiling curves without kernel
points. The native commands below add points and update these same plots.

### Calibration parameters

| Option | Meaning | Values in the command above |
|---|---|---|
| `--bench PATH` | Native ceiling executable | The Release benchmark built by `build.sh` |
| `--threads N` | Requested roof scope and accumulator-sweep worker count | `8` |
| `--max-threads N` | Maximum allowed core count in the sweep | `8`, producing 1/2/4/8-core measurements |
| `--sizes-mib A B` | Total working-set sizes, tested sequentially; at least two distinct sizes | `1024 2048`: 1 GiB and 2 GiB, not per thread |
| `--rounds N` | Repeat the condition sweep, reversing order on alternate rounds | `2` |
| `--iterations N` | Timed samples per condition, after two fixed warmups | `9` |
| `--output DIR` | Fresh directory for calibration results | `$MP1_CAL` |

The largest working set must not exceed **one quarter of available memory**.
With about 13 GiB available, the 1/2 GiB sizes above fit that budget and run
sequentially.

If `Largest stream would exceed one quarter of available memory` appears, check
`free -h` and reduce `--sizes-mib` while retaining two distinct sizes. Use the
`available` column to check the budget.

### Read the ceiling results

| Ceiling CSV row | `work` | `rate` / `unit` |
|---|---|---|
| `memory_bandwidth` | Useful triad bytes: `12 * stream_elements` | GB/s |
| `memory_bandwidth_read` | Read bytes: `4 * read_elements` | GB/s |
| `fma_compute` | `2 * lanes * accumulators * repeats * threads` FLOPs | GFLOP/s |

Calibration writes rates for every tested thread count. `--roof-threads 8`
selects matching 8-thread memory and compute results for the common roof:

```text
bandwidth = max(triad_GB_per_s, read_only_GB_per_s)
compute = register_FMA_GFLOP_per_s
ridge_point = compute / bandwidth
roofline_bound(AI) = min(compute, AI * bandwidth)
attained_percent = 100 * measured_GFLOP_per_s / roofline_bound(AI)
```

## Measure native CPU metrics

Use `cpu_bench` to vary dimensions, workers, sample count, or cache displacement.
Every command prints a summary, appends one CSV row, and updates the Roofline
outside the timed region. Run the commands individually and inspect each result.
All native metrics are reported together in that row; separate runtime, FLOP,
and AI runs are unnecessary. `--metrics` belongs to the wrappers below, not to
`cpu_bench`.

### Native parameters

| Option | Meaning | CLI default / course use |
|---|---|---|
| `--kernel` | Workload | Select `gemm`, `gemv`, or `attention` explicitly |
| `--seq S` | GEMM rows/tokens or Attention sequence length | Default `512`; integer `1..2048`; GEMV always uses one token |
| `--threads N` | Workers sharing one problem | Default `1`; baseline `8`; optional scaling `1`, `2`, `4`, `8` |
| `--warmup N` | Untimed warmup invocations | Default `1`; examples use `2` |
| `--iterations N` | Timed samples summarized by their median | Default `5`; examples use `9`; keep the policy consistent |
| `--flush-cache-mib N` | Separate buffer touched outside timing before each invocation | Default `0`; compare warm GEMV (`0`) with displaced-cache GEMV (`528` MiB for the reported 264 MiB L3 total) |
| `--ceilings PATH` | Standard ceiling CSV used for automatic plotting | Explicitly use `$MP1_CAL/ceilings.csv` |
| `--roof-threads N` | Common roof scope; does not set kernel workers | Use `8`, including for 1/2/4-thread kernel runs |
| `--output PATH` | Append native measurements to this CSV | `$MP1_CPU/kernel_results.csv` |
| `--no-plot` | Save CSV without rendering plots after this command | Optional; regenerate plots later |

### GEMM: multi-token prefill

The fixed projection is `[S x 2048] * [2048 x 5632]`, so `M=S`, `K=2048`,
`N=5632`. Measure both required sequence lengths:

```sh
taskset -c "$MP1_CPUS" "$MP1_NATIVE/cpu_bench" --kernel gemm --seq 128 \
  --threads 8 --warmup 2 --iterations 9 --roof-threads 8 \
  --ceilings "$MP1_CAL/ceilings.csv" --output "$MP1_CPU/kernel_results.csv"

taskset -c "$MP1_CPUS" "$MP1_NATIVE/cpu_bench" --kernel gemm --seq 512 \
  --threads 8 --warmup 2 --iterations 9 --roof-threads 8 \
  --ceilings "$MP1_CAL/ceilings.csv" --output "$MP1_CPU/kernel_results.csv"
```

Read `median_ms`, `measured_gflops`, `approx_flops`, `estimated_minimum_bytes`,
and `arithmetic_intensity`. Increasing `S` increases work linearly and amortizes
the fixed weight matrix across more tokens. The command does not change `K` or `N`.

### GEMV: one-token decode, warm and displaced-cache

The shape is `[1 x 2048] * [2048 x 5632]`. GEMV ignores `--seq`; vary cache
conditions and thread count instead:

```sh
taskset -c "$MP1_CPUS" "$MP1_NATIVE/cpu_bench" --kernel gemv --flush-cache-mib 0 \
  --threads 8 --warmup 2 --iterations 9 --roof-threads 8 \
  --ceilings "$MP1_CAL/ceilings.csv" --output "$MP1_CPU/kernel_results.csv"

taskset -c "$MP1_CPUS" "$MP1_NATIVE/cpu_bench" --kernel gemv --flush-cache-mib 528 \
  --threads 8 --warmup 2 --iterations 9 --roof-threads 8 \
  --ceilings "$MP1_CAL/ceilings.csv" --output "$MP1_CPU/kernel_results.csv"
```

For the reported `L3: 264 MiB (12 instances)`, use a total flush buffer of
`528` MiB (`2 * 264`), taking the reported L3 sum as a conservative size basis.
Workers divide this buffer; it is not allocated per thread. The buffer touch is
outside timing. `0` means no deliberate cache displacement; `cache=cold` labels
the displacement experiment.

Compare native runtime/GFLOP/s while keeping the FLOPs and analytical AI fixed.
This experiment is distinct from the cold-start cache simulation below.

### Attention: total time and QK / softmax / PV times

The workload uses one head with `Q,K,V=[S x 64]`. It computes scaled `QK^T`, row
softmax, and `PV`, materializing the `S x S` scores without a causal mask.

```sh
taskset -c "$MP1_CPUS" "$MP1_NATIVE/cpu_bench" --kernel attention --seq 128 \
  --threads 8 --warmup 2 --iterations 9 --roof-threads 8 \
  --ceilings "$MP1_CAL/ceilings.csv" --output "$MP1_CPU/kernel_results.csv"

taskset -c "$MP1_CPUS" "$MP1_NATIVE/cpu_bench" --kernel attention --seq 512 \
  --threads 8 --warmup 2 --iterations 9 --roof-threads 8 \
  --ceilings "$MP1_CAL/ceilings.csv" --output "$MP1_CPU/kernel_results.csv"
```

| Native column | Meaning |
|---|---|
| `median_ms` | Median time across the complete three-phase Attention invocation |
| `qk_ms`, `qk_gflops` | Scaled QK phase median time and modeled rate |
| `softmax_ms`, `softmax_gops` | Softmax phase median time and approximate operation-equivalent rate |
| `pv_ms`, `pv_gflops` | PV phase median time and modeled rate |

Phase times are measured within the complete native Attention invocation.
The sum of the three phase medians need not equal the median total time.
GEMM/GEMV rows leave these phase fields blank.

### Additional native sequence lengths

For GEMM and Attention, replace `--seq 128` in the commands above with `256`
to add an intermediate point. Other native sizes in `1..2048` are optional;
the required measurements remain `128`/`512`. Trace commands support only the
fixed cases listed below.

### Timing region and plotting

Native timing includes kernel output clearing, OpenMP entry/exit, and synchronization.
Allocation, input preparation, warmups, cache displacement, and result writes
are outside timing. Attention phase timing includes timer overhead.

Figures are saved beside the native CSV as `roofline.png`/`.pdf` and
`roofline_gemm`, `roofline_gemv`, `roofline_attention` PNG/PDF files.
Error bars show the min/max timing range.

To regenerate plots from saved measurements:

```sh
"$MP1_PY" mp1/scripts/generate_roofline.py \
  --ceilings "$MP1_CAL/ceilings.csv" \
  --kernels "$MP1_CPU/kernel_results.csv" \
  --roof-threads 8 --output "$MP1_CPU/roofline.csv"
```

## DynamoRIO instruction, cache and reuse analysis

These commands use the same kernel library as the native benchmark. They run
two warmups outside tracing and collect **one complete selected ROI**. Allocation
and input preparation are outside the ROI; OpenMP/runtime and wrapper instructions
within it can appear in the counts. Use the native runs above for performance timing.

**Keep `--metrics` on the commands below.** Without it, a wrapper launches the
full profiling workflow, including additional native measurements and traces.

### Workloads and fixed trace parameters

| Case for `measure_case.sh` | Equivalent wrapper | Shape / parameters |
|---|---|---|
| `gemv` | `measure_gemv.sh` | One token, `K=2048`, `N=5632` |
| `gemm128` | `measure_gemm_s128.sh` | `S=128`, `K=2048`, `N=5632` |
| `gemm512` | `measure_gemm_s512.sh` | `S=512`, `K=2048`, `N=5632` |
| `attention128` | `measure_attention_s128.sh` | `S=128`, `D=64`, one head |
| `attention512` | `measure_attention_s512.sh` | `S=512`, `D=64`, one head |

All wrappers live under `mp1/dynamorio/`. They use **8 workers**, **2 warmups**,
and **one traced invocation**. Choose the case to select `S`; native `--seq`,
`--threads`, `--iterations`, and `--flush-cache-mib` options do not apply here.

Collect trace metrics for each case, then use the values relevant to your explanation:

```sh
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemv.sh "$MP1_RUN" \
  --metrics counts memory reuse cache opcodes

taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemm_s128.sh "$MP1_RUN" \
  --metrics counts memory reuse cache opcodes

taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_gemm_s512.sh "$MP1_RUN" \
  --metrics counts memory reuse cache opcodes

taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_attention_s128.sh "$MP1_RUN" \
  --metrics counts memory reuse cache opcodes

taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_attention_s512.sh "$MP1_RUN" \
  --metrics counts memory reuse cache opcodes
```

Results are saved in `$MP1_RUN/metrics/<case>.json`. Later requests reuse the
saved trace and completed analyses.

### Select each metric independently

Set the case once, then run the commands for the metrics you need. Replace `gemv`
with any case in the table above to apply the same commands to GEMM or Attention:

```sh
MP1_CASE=gemv

# Instructions, loads, and stores
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_case.sh \
  "$MP1_CASE" "$MP1_RUN" --metrics counts

# Executed opcode distribution, including SIMD/FMA instructions
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_case.sh \
  "$MP1_CASE" "$MP1_RUN" --metrics opcodes

# Mean/median reuse distance and histogram
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_case.sh \
  "$MP1_CASE" "$MP1_RUN" --metrics reuse

# Simulated L1D/L2/LLC accesses, misses, and miss rates
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_case.sh \
  "$MP1_CASE" "$MP1_RUN" --metrics cache

# Dynamic operand bytes and unique data cache lines
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_case.sh \
  "$MP1_CASE" "$MP1_RUN" --metrics memory

cat "$MP1_RUN/metrics/${MP1_CASE}.json"
```

`counts` also runs automatically when another trace metric needs it.

| Selection | Fields under `metrics` in the JSON | Interpretation |
|---|---|---|
| `counts` | `Instructions`, `Loads`, `Stores`, `Threads`, `Skipped_Memref_Markers` | Executed instructions (fetched + non-fetched) and data-reference counts; a vector load is one reference, not one per lane |
| `opcodes` | Mnemonic-to-count mapping | Dynamic opcode mix; convert SIMD/FMA instructions to work only with the correct lane count and operation semantics |
| `reuse` | `Mean_Reuse_Distance`, `Median_Reuse_Distance`, `Data_Cold_Reference_Fraction`, `histogram` | Reuse of 64-byte instruction **and** data lines in per-thread histories; mean/median are not data-only or global inter-thread reuse |
| `cache` | `L1D_Accesses`, `L1D_Misses`, `L1D_Miss_Rate`, and corresponding `L2_*`/`LLC_*` | Misses divided by accesses reaching each level; rates are fractions, so multiply by 100 for percent |
| `memory` | `Loads`, `Stores`, `Dynamic_Memory_Reference_Bytes`, `Unique_Cache_Lines` | Operand bytes include repeated accesses served from cache; unique 64-byte lines describe the data footprint. Neither measures DRAM traffic |

The `cache` selection uses [cache_model.cfg](dynamorio/cache_model.cfg): eight
simulated cores, 64-byte lines, per-core L1D 48 KiB and L2 2 MiB, and shared
LLC 256 MiB. Caches start empty and prefetching is disabled. These are simulated
misses, not native hardware counters; L2/LLC include instruction and data traffic.

Report cache counts alongside `100 * Miss_Rate`. For a data-reuse CDF, normalize
the histogram's `Data_References` counts by their sum and show the cold fraction
separately. Use these values in the
[DynamoRIO analysis](REPORT_GUIDE.md#2-dynamorio-analysis) of your report.

### Optional Attention phase traces

Without `--phase`, the ROI is the full Attention invocation. If phase details
help explain its behavior, use `--phase qk`, `softmax`, or `pv` with the desired
metrics:

```sh
taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_attention_s128.sh "$MP1_RUN" \
  --phase qk --metrics counts memory reuse cache opcodes

taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_attention_s128.sh "$MP1_RUN" \
  --phase softmax --metrics counts memory reuse cache opcodes

taskset -c "$MP1_CPUS" sh mp1/dynamorio/measure_attention_s128.sh "$MP1_RUN" \
  --phase pv --metrics counts memory reuse cache opcodes
```

Use `measure_attention_s512.sh` for `S=512`. You can also request any
of these analyses separately. Phase results are saved as
`metrics/attention128_qk.json`, `attention128_softmax.json`, and
`attention128_pv.json` (or the `attention512_*` equivalents).

Phase input preparation and warmups occur outside tracing. Each phase has its
own cold simulated cache, so phase miss counts cannot be summed to obtain
full-Attention misses. Read native phase times from `qk_ms`, `softmax_ms`, and `pv_ms`.

## Optional: compare 1, 2, 4, and 8 threads

To investigate scaling, repeat a native configuration at `--threads 1`, `2`,
and `4` after its 8-thread baseline. Keep shape, cache condition,
warmups, and timed sample count fixed, and keep `--roof-threads 8` throughout.
For example, add the other thread counts for GEMM S=128:

```sh
for mp1_threads in 1 2 4; do
  taskset -c "$MP1_CPUS" "$MP1_NATIVE/cpu_bench" --kernel gemm --seq 128 \
    --threads "$mp1_threads" --warmup 2 --iterations 9 --roof-threads 8 \
    --ceilings "$MP1_CAL/ceilings.csv" --output "$MP1_CPU/kernel_results.csv"
done
```

Apply the same changes to other configurations as needed.
Do this with the native executable; the packaged trace profile remains at eight
workers. Calibration already provides ceilings at 1/2/4/8 for comparison.

```text
speedup(t) = median_ms(1) / median_ms(t)
parallel_efficiency(t) = speedup(t) / t
```

Whole-problem FLOPs and analytical bytes/AI do not change with thread count.
To compare performance against each worker count's own ceiling, regenerate a
separate plot with the matching `--roof-threads N`.

## Read the CSV and derive your metrics

| Native CSV columns | Interpretation |
|---|---|
| `kernel`, `configuration`, `sequence`, `flush_cache_mib` | Workload and shape/cache parameters; `configuration` includes `threads=N` |
| `median_ms`, `min_ms`, `max_ms`, `relative_range_percent` | Native timing; range is `100 * (max-min) / median` |
| `approx_flops`, `estimated_minimum_bytes`, `arithmetic_intensity` | Analytical quantities, not hardware counters |
| `measured_gflops` | Whole-problem analytical work divided by median wall time |
| `qk_ms`, `softmax_ms`, `pv_ms` and phase rates | Native Attention phase measurements |
| `warmups`, `iterations` | Sampling policy |

Each CSV row summarizes one command's samples. The dimension models in
[include/metrics.h](include/metrics.h) are:

```text
GEMM FLOPs           = 2 S N K
GEMM minimum bytes   = 4 (S K + K N + S N)
GEMV FLOPs           = 2 N K
GEMV minimum bytes   = 4 (K + K N + N)
Arithmetic intensity = FLOPs / minimum bytes
Measured GFLOP/s     = FLOPs / (median_ms * 1e6)
```

Attention uses educational operation-equivalents: QK including scale is
`2 S^2 D + S^2`, softmax is `5 S^2`, and PV is `2 S^2 D`. Total work is
`4 S^2 D + 6 S^2`; the sum of the phase minimum-byte models is `16 (S D + S^2)`.
An exponential is not literally one hardware FLOP, and intermediate score
accesses can be served from cache. Neither the analytical phase sum nor trace
operand bytes measures actual DRAM traffic.

Inspect [cpu/gemm.cpp](cpu/gemm.cpp), [cpu/gemv.cpp](cpu/gemv.cpp), and
[cpu/attention.cpp](cpu/attention.cpp) to explain contiguous accesses, reuse,
working-set sizes, and reduction/parallel overhead. Combine source, native,
trace, and simulated-cache evidence while keeping their different scopes clear.

## Result files and report

The README commands save measurement results under `mp1/results/`:

| Path | Contents |
|---|---|
| `mp1/results/kernel_results.csv` | Native kernel measurements |
| `mp1/results/roofline.csv`, `.png`, `.pdf` | Derived Roofline coordinates, bounds, and plots |
| `mp1/results/calibration/ceilings.csv` | Measured bandwidth and compute ceilings |
| `mp1/results/calibration/bandwidth_saturation.*`, `fma_saturation.*` | Calibration plots |
| `mp1/results/dynamorio/metrics/<case>[_<phase>].json` | Selected trace metrics |
| `mp1/results/dynamorio/logs/<metric>_<case>[_<phase>].log` | Original analysis output |

**MP1 report will be submitted with MP2.**

Follow [REPORT_GUIDE.md](REPORT_GUIDE.md). The report has two parts:

1. **Roofline analysis:** plot GEMM, GEMV, and Attention on the common Roofline,
   and explain whether each configuration is closer to memory-bound or compute-bound.
2. **DynamoRIO analysis:** use numerical trace metrics to explain the reasons
   for those conclusions.

For build troubleshooting and advanced profiling, see the
[DynamoRIO guide](dynamorio/README.md).
