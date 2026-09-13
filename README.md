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
cd ECE498_MPs
```

No `--recurse-submodules` option is needed. TinyLlama is not a build or runtime
dependency, so its source repository is intentionally not included as a Git
submodule. The benchmark uses random tensors and needs no model weights.

The verified model-configuration snapshot and its authoritative source links
are retained under [`reference/`](reference/).

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
