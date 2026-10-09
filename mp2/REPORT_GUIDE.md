# MP2 Report Guide

Submit a concise report plus the requested timing CSV and selected profiler
evidence. Every performance result must follow a passing correctness test.

## 1. Environment and correctness

Report:

1. GPU name, compute capability, memory, driver, toolkit, `nvcc`, and target
   architecture.
2. Whether the device is the course target or a separate validation system.
3. Results of the small, non-multiple, and required-size tests, including the
   documented FP32 tolerance.
4. Attention NaN/Inf status and the maximum softmax row-sum error.
5. Memcheck, initcheck, synccheck, and racecheck summaries.
6. Whether Nsight hardware counters were accessible without elevated
   privileges.

Do not use a checksum as the correctness criterion.

## 2. Normal CUDA-event timing

For GEMM S=128, GEMM S=512, GEMV, Attention S=128, and Attention S=512,
report:

| Kernel/configuration | Median | p10 | p90 | Approx. FLOPs | GFLOP/s | Analytical AI |
|---|---:|---:|---:|---:|---:|---:|

For Attention, include median and p10/p90 for QK-plus-scale, softmax, and P*V.
State the warm-up count, sample count, and that these are unbatched
measurements: one kernel per GEMM/GEMV sample and one three-kernel sequence per
Attention sample.

Use only `run_gpu.sh`'s normal CUDA-event CSV for this table. Allocation,
initialization, and host/device copies are outside the timed interval. The
three-kernel Attention total can include GPU-side gaps between phases, but it
is not end-to-end inference latency.

Answer:

1. Which workload has the highest and lowest throughput, and why?
2. Why does one-token GEMV use memory and execution resources differently from
   GEMM?
3. How does sequence length change parallelism, work, and reuse?
4. When Attention S grows by 4x, how do work and time scale?
5. Which result is most sensitive to launch or insufficient-work overhead?

## 3. Nsight baseline cases

Profile the naive GEMM S=512, GEMV, and Attention S=128 baseline, then the
corresponding optimized cases needed to explain your changes. For each case,
report the available forms of:

- Nsight kernel `Duration`;
- grid and block dimensions;
- achieved occupancy and waves per SM;
- SM/compute throughput;
- DRAM and L1/TEX throughput;
- L1/L2 hit rates or sectors/request when available.

Metric names vary by Nsight and GPU. Preserve the exact name and unit reported
on your machine and do not invent unsupported values.

Nsight Compute replay warning: use Nsight's own `Duration` and counters in this
section. Never substitute the CUDA-event runtime/GFLOP/s printed by the target
application under replay for the normal results in Section 2.

Answer:

1. Is each baseline limited by compute throughput, memory behavior,
   insufficient parallelism, launch overhead, or its naive mapping? Support the
   answer with at least two measurements.
2. Trace adjacent thread indices to global-memory addresses. Which accesses
   are coalesced?
3. Can high achieved occupancy coexist with poor useful efficiency here?
4. Is the ordinary FP32 baseline using tile-level Tensor Core matrix
   operations?
5. For Attention, what do the three phase durations and the Systems timeline
   reveal about short-kernel overhead and idle gaps?

If counters are permission-restricted, include the exact error and use source
and launch analysis for unavailable observations. Do not change system policy
or fabricate metrics.

## 4. CPU/GPU comparison and student optimization

Use the corresponding EWS CPU timings and analysis from the two-part
[MP1 report](../mp1/REPORT_GUIDE.md). State the CPU worker count for every
CPU/GPU speedup. The required MP1 baseline is 8 threads; 1-thread results are
optional and must be labeled separately. Show warm and displaced-cache GEMV
comparisons separately, and use MP1's Roofline and DynamoRIO data-access
analysis to explain the CPU bottlenecks. Match tensor dimensions, not the
complete `configuration` string, which includes extra CPU settings.

Include all five shared cases in the comparison:

| Kernel/shape | MP1 CPU threads/cache | CPU ms | Naive GPU ms | Best GPU ms | Library ms | Naive/best speedup | CPU/best speedup |
|---|---|---:|---:|---:|---:|---:|---:|

Identify the implementation revision and results directory for every GPU
column. The library measurements must cover the same operation; label
Attention implemented as cuBLAS matrix products plus custom softmax accurately.
State the library version, math mode, input/accumulator/output datatypes,
warmups, samples, and timing boundary. Compare native CPU times against normal
CUDA-event times and explain the CPU invocation overhead and excluded GPU
copies. MP2's CPU correctness reference is not an MP1 performance measurement.

For each implementation step, include:

| Change | Expected effect | Correctness/error | Runtime/speedup | Profiler evidence | Remaining bottleneck |
|---|---|---|---|---|---|

Discuss coalescing, tiling/data reuse, and occupancy tuning, and include your
Tensor Core experiment. Identify the multiply and accumulation precision and
report reduced-precision errors separately from the unchanged FP32 tolerance.
Explain improvements and regressions; do not infer that a larger block or
higher occupancy must be faster. Submit the changed sources/build instructions,
the preserved naive results, each optimization's CSV, and the library results.

Answer:

1. What is the GPU speedup for each shared required configuration?
2. Which CPU bottleneck was relieved, and which limitation remained or changed
   form?
3. Why can identical analytical arithmetic intensity produce very different
   measured GFLOP/s?
4. Which baseline has the most optimization headroom, according to evidence?
5. For each optimization you implement, which metric confirms that it helped
   for the intended architectural reason?
6. How does your best correct implementation compare with an optimized
   library under the same shape, datatype, timing boundary, and warm-up policy?

Keep baseline, optimized, and library results clearly labeled. Clearly
distinguish analytical minimum bytes, CUDA-event measurements, Nsight hardware
measurements, and any theoretical specification-sheet limits.
