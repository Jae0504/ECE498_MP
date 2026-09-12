# MP1 Report Guide

Submit a concise report with the two sections below. Follow the
[README commands](README.md); measurement results are saved under `mp1/results/`.
Include the native/ceiling CSVs and DynamoRIO JSON results and analysis logs
used as evidence.

## 1. Roofline analysis

Plot the native GEMM, GEMV, and Attention measurements together on the common
8-thread Roofline, using `--threads 8 --roof-threads 8`. Include:

- GEMM at S=128 and S=512.
- GEMV with `--flush-cache-mib 0` and `528`.
- Attention at S=128 and S=512.

Use analytical arithmetic intensity (FLOP/byte) on the x-axis and native
throughput (GFLOP/s) on the y-axis. Show the measured bandwidth and compute
ceilings, and identify each configuration in the figure. The README commands
generate this figure as `mp1/results/roofline.png` and `.pdf`.

For each configuration, explain whether it is closer to **memory-bound** or
**compute-bound** behavior. Support your explanation with its AI, measured
GFLOP/s, and position relative to the bandwidth and compute ceilings. Distinguish
the Roofline's prediction from an observed bottleneck; if the measurements
do not support a clear diagnosis, explain the ambiguity.

## 2. DynamoRIO analysis

Use DynamoRIO results for the same GEMM, GEMV, and Attention cases to explain
**why** they show the behavior discussed in Section 1. Select relevant evidence
from `counts`, `memory`, `reuse`, `cache`, and `opcodes`, and include actual
numerical values in your explanation.

| Evidence | What to examine |
|---|---|
| Memory accesses | Loads/stores, dynamic operand bytes, and unique data-line footprint |
| Reuse | Data reuse-distance distribution and cold-reference fraction |
| Cache simulation | L1D/L2/LLC access counts, misses, and miss ratios |
| Instructions | Scalar/vector arithmetic, FMA, shuffle, and other relevant opcode counts |

Connect these observations to the kernel's access and instruction patterns,
and explain how they support or challenge your Roofline diagnosis. For
Attention, use native phase times and QK/softmax/PV traces when they help
explain the result.

Operand bytes and simulated cache misses are not measured DRAM traffic.
Opcode counts do not measure execution-unit utilization. If the available
metrics cannot establish a specific cause, state what remains uncertain.
