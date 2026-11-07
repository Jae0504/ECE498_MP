# Shared MP3/MP4 RTL workflow

## What executes

```text
miniLLAMA prefill / cached decode (PyTorch)
  Linear, attention QK^T and AV operands
       ↓ Python per-row A / per-column B INT16 quantization
  cocotb external memory ↔ Verilator ↔ synthesizable PE-array RTL
       ↓ actual signed INT64 output handshakes
  independent scoreboard → dequantize RTL outputs → next model operator
       ↓
  logits and tokens; observed RTL cycles and memory transactions
```

GEMM/GEMV describe shapes; attention QK^T/AV describe roles of those products.
Do not add role and shape-category totals and double-count work. Each batch/head
matrix executes using its own values. No shape cache skips numerical or timing
execution; only compiled executables are cached by source, parameters and tools.

## Dependencies and setup

Tested: Linux aarch64, Python 3.12.3, PyTorch 2.6.0 CPU, NumPy 1.26.4,
cocotb `2.0.0.dev643+rae55ced6`, Verilator `5.053 devel`, YoWASP Yosys 0.70.
The copied cocotb VERSION file can differ from its installed development package
version; identify the actual source revision and installed version.

| Tool | Job |
| --- | --- |
| Verilator | Compiles synthesizable SystemVerilog to a cycle simulation executable |
| cocotb | Python testbench: clock, command, external memory and output checks |
| Yosys / yowasp-yosys | Synthesizes actual RTL into generic logic and abstract memories |
| PyTorch | Model, non-matrix operators, validation; does not supply RTL matrix outputs |

From the repository root:

```sh
sh mp3/setup_rtl.sh
# Check an already prepared installation without reinstalling tools:
sh mp3/setup_rtl.sh --skip-install
```

Setup initializes the root `miniLLAMA/`, `cocotb/`, and `verilator/` submodules,
including empty directories from a normal clone, and verifies tool source pins.
Use a Git clone for setup so miniLLAMA can be prepared from its pinned commit.
The RTL tool pins are:

- Verilator: `770b7d009e59071311d4173f8c86b4ac6085f1e9`.
- cocotb: `ae55ced614b3f4ba89d77da5bd635024546d0737`.

Build prerequisites: Git, C++ compiler, make, autoconf, flex, bison, matching
Python development headers and `FlexLexer.h`. On Debian/Ubuntu, missing headers
are downloaded with `apt download` and extracted under `mp3/.deps/rtl_sysroot/`.
No system packages are installed and sudo is not used. Other systems must
provide those headers separately. Initial setup needs internet; prepared
experiments run offline. YoWASP's first startup compiles WASM and can be slow.
A native `yosys` on PATH is also accepted and recorded.

`setup_rtl.sh` calls `mp3/setup.sh` for the base RTL Python requirements, then
builds/installs the RTL tools. `--skip-install` verifies an existing environment.
It prepares the patched miniLLAMA source used by the model frontend.
See [dependencies](docs/DEPENDENCIES.md) for the source pins and Python packages.
The public CLI is `mp3/rtl_run.py`: `check`, `run`, `sweep`, `synth`, `compare`.

## Baseline and files

`configs/rtl_baseline.json` selects 16×4 PEs, 3072 A words and 1024 B words
(8192 input-storage bytes), four INT16 lanes per read port, and 100 MHz.
64-bit PE accumulators add 4096 register bits, outside the input SRAM budget.
C writes one 64-bit value per cycle. A and B each read up to 8 B/cycle through
independent ports; this is not one shared 8 B/cycle DRAM channel. The external
memory is a deterministic port-service model, not DDR timing.

This is a teaching baseline with an array shape poorly matched to short prompts
and single-token decode. The earlier 8×8, 2048/2048-word configuration is preserved
as `configs/rtl_reference_8x8.json` for an additional comparison. The JSON selected
by the runner overrides the RTL module's default parameters. Existing result
directories retain their original configuration and must not be relabeled after
this baseline change. Hold budgets fixed and report tradeoffs on both workloads.

The output-stationary array broadcasts A along rows and B along columns; it
is not a neighbor-to-neighbor systolic array. The controller loops N tile,
M tile, then K panel. A has one bank per row, B one bank per column. Panels fit
`min(a_buffer_words/array_rows, b_buffer_words/array_cols)` K elements. Remainder
tiles and K-panel accumulation are supported. After the last panel, the valid
output rectangle is drained. Baseline has no C SRAM, prefetch or load/compute overlap.

Paths in this table are relative to `mp3/`:

| File | Responsibility |
| --- | --- |
| `rtl/pe.sv` | INT16 multiply and INT64 accumulation |
| `rtl/pe_array.sv` | Broadcast and per-PE enables |
| `rtl/tile_buffer.sv` | Banked panel storage |
| `rtl/controller.sv` | Memory requests, tile control and output scheduling |
| `rtl/accelerator_top.sv` | Integration and external interface |
| `rtl_backend/driver.py` | Clock, external memory, observed counters and independent scoreboard |
| `rtl_backend/model.py` | Model hooks, quantization/dequantization and logit validation |
| `rtl_backend/frontend.py` | miniLLAMA loading, matrix interception, and independent operation recording |
| `rtl_backend/runner.py` | Build selection, simulator launch and provenance |
| `rtl_backend/sweep.py` | Declared candidate study, fixed-budget checks, comparisons, and selection |
| `rtl_backend/reporting.py` | Token timeline, timing and dimensionless activity proxy |
| `rtl_backend/synthesis.py` | Yosys synthesis and structural counts |

MP3 edits JSON only. MP4 edits the five RTL modules under `mp4/rtl_student/`.
The shared model frontend, simulation harness, and reports live under `rtl_backend/`.

## RTL interface

`start` is accepted while idle with positive 16-bit M, N, K. Dimensions are
latched, one matrix is in flight, and reset is synchronous/active-high.

- A addresses are row-major INT16 word offsets; B addresses are column-major
  INT16 word offsets. Python performs B layout conversion outside timing.
- Inputs present `*_req`, `*_addr`, `*_count`. Memory asserts `*_ready` with
  packed `*_data` (lane 0 in bits 15:0). A request remains stable until the
  accepting edge; count is 1..`memory_lanes`.
- Output presents `c_valid`, row-major **element** address `c_addr` and signed
  64-bit `c_data`. All remain stable until `c_ready` accepts them.
- `done` follows the last accepted output. All M×N elements must appear exactly
  once. The harness checks bounds, stability, completeness and actual values.
- `pe_enable`, `active_rows`, `active_cols` describe a rectangular group of
  one-MAC-per-cycle PEs. Counters use these signals for useful MACs and logical
  broadcast-buffer reads. SIMD PEs, nonrectangular masks or changed memory
  behavior require an explicit observability/accounting extension.

Timing is measured by Python counting clock edges, not a student-modifiable
RTL cycle counter. Memory only responds to requests; Python does not choose
tiles, reuse, MAC scheduling or numerical outputs. Input wait counts are
**port-cycles** and may overlap. Total cycles already include every wait.

The abstract SRAM has asynchronous reads and up to `memory_lanes` writes per
bank per cycle. This is synthesizable storage, not a selected SRAM macro.
Physical bank/port realization and critical-path timing need separate evaluation.
Reported buffer reads count useful operand broadcasts, not all electrical
activity of unused banks. The activity proxy is not a switching-power analysis.

## Correctness and numerical scope

```sh
MP3_PY="$PWD/mp3/.venv/bin/python"
"$MP3_PY" -m pytest -q mp3/tests
"$MP3_PY" mp3/rtl_run.py check --out mp3/results/rtl_check
"$MP3_PY" mp3/tools/verify_rtl.py --out mp3/results/rtl_integration
"$MP3_PY" mp3/rtl_run.py run --out mp3/results/rtl_smoke
```

`check` covers signed extrema, GEMV, partial tiles, K panels, backpressure,
repeated shapes with different values and zeros. `verify_rtl.py` also uses
non-power-of-two banks/lanes, tiny buffers and memory waits, and corrupts a
copied PE to verify that wrong hardware outputs fail the numerical scoreboard.
It does not modify selected source files.

Quantization uses symmetric max-absolute scaling (A per row, B per column),
round-to-nearest/ties-to-even and range -32767..32767. Zero vectors use scale 1.
The RTL supports the full signed INT16 range, including -32768 in direct tests.
INT64 accumulation cannot overflow for supported K≤65535. Each hardware output
must exactly match an independent NumPy INT64 matrix product.

Actual RTL integers are dequantized to FP32 and returned to the live model.
The complete cached model is repeated with independent quantized products;
logits must match exactly. A separate FP32 full-prefix reference checks the
executed token prefixes and reports quantization error. The course tiny-model
acceptance threshold is `atol=1e-3`, `rtol=1e-2`, not a general INT16 accuracy
guarantee. FP32 cached vs full-prefix inference is also checked at `atol=1e-5`,
`rtol=1e-4`.

Per-column quantization of attention V depends on sequence length; full-prefix
INT16 recomputation need not match cached INT16 hidden states. The exact INT16
reference therefore uses the same cached inference semantics. FP32 next-token
agreement is reported on executed prefixes, not claimed as language quality.
Default weights are seeded random; repeated generated characters are expected.
Use `--checkpoint` with matching `--model` for a course-trained checkpoint.

## How time is measured

The driver counts rising edges from accepted start through the accepted last
output/done, inclusive. Reset and inter-matrix launch gaps are excluded. The
1×1×1 no-wait case takes six cycles, including command, setup, load, control,
compute and drain. **`latency_us = cycles / clock_mhz`**.
Waveform clock edges are rounded to the simulator's 1 ps resolution; reported
latency uses integer cycle counts and the requested clock assumption.

- TTFT: all matrix cycles in the prefill pass divided by frequency.
- TPOT: one cached decode pass for each subsequent generated token.
- Mean TPOT: total decode matrix time / (`new_tokens - 1`); `null` for one token.
- Request latency: prefill plus all decode matrix time.

These are **matrix-path contributions**, not end-to-end system timing. They
include RTL control, panel loads, memory waits, MACs and result transfers.
They exclude quantization, layout conversion, launches, host software, embedding,
softmax, normalization, RoPE, activations and residuals. PyTorch executes these
non-matrix operators. Assumed 100 MHz is not timing-closed Fmax.
`host_timing.json` reports software wall time separately; it never determines
the accelerator speedup.

## Results and comparisons

For a complete fixed-budget parameter study, run:

```sh
"$MP3_PY" mp3/rtl_run.py sweep --candidates mp3/configs/rtl_candidates.json \
  --out mp3/results/dse_01
cat mp3/results/dse_01/sweep.md
```

This checks each configuration and executes both workloads, retaining all results
and an objective chosen before execution. `selected.json` records the chosen
candidate; `best_design.json` can be passed directly to MP4. The default study
holds PE count, total input SRAM, memory service, and clock fixed. Candidate
names are unique, and invalid configurations are rejected before simulation.
The earlier standalone commands remain useful for an individual experiment:

```sh
"$MP3_PY" mp3/rtl_run.py run --workload mp3/configs/workload_prefill.json \
  --out mp3/results/rtl_prefill
"$MP3_PY" mp3/rtl_run.py run --workload mp3/configs/workload_decode.json \
  --out mp3/results/rtl_decode
# Compare the SAME workload, using the original baseline first:
"$MP3_PY" mp3/rtl_run.py compare BASELINE_DIRECTORY CANDIDATE_DIRECTORY \
  --out mp3/results/rtl_comparison
```

| Artifact | Contents |
| --- | --- |
| `summary.json`, `analysis.md` | Counters, TTFT/TPOT/latency, scope, resources and cost proxy |
| `kernels.csv` | Every executed batch/head matrix, dimensions, cycles and transfers |
| `token_timeline.csv` | Per-token and cumulative matrix time |
| `trace.json` | Actual model events, generated IDs/text and workload identity |
| `numerical_validation.json`, `logits.pt` | Numerical checks and actual/reference logits |
| `design.json`, `request.json`, `provenance.json` | Parameters, model/workload, source hashes and tools |
| `build.log`, `simulation.log`, `tests.xml` | Compile and cocotb pass/fail evidence |
| `host_timing.json` | Software wall time, excluded from hardware metrics |
| `comparison.csv/json/md` | Same-workload baseline-relative differences |

Add `--waves` for a waveform; start with a small check run. Comparison rejects
mismatched workloads/weights/quantization and inconsistent tokens. Result
directories must be fresh; saved results are not silently overwritten.

## Power and area

```sh
"$MP3_PY" mp3/rtl_run.py synth --out mp3/results/rtl_synthesis
"$MP3_PY" mp3/rtl_run.py synth --rtl-root mp4/rtl_student \
  --design mp3/best_design.json --out mp4/results/rtl_synthesis
```

Yosys processes the actual RTL, preserves memories, maps arithmetic/control to
generic gates and runs structural checks. `synthesis.json` reports generic
logic cells (excluding metadata/memory cells), register cells, memory bits and
ports. `synthesize.ys`, the source snapshot, log and two netlists reproduce it.
Equal PE/SRAM budgets do not imply equal controller/interconnect logic or
identical physical area. Generic gate counts are not standard-cell area.

The simulator reports a **dimensionless activity cost**:
`MACs + 2×buffer_bytes + 50×external_memory_bytes`. Average activity cost divides
this score by cycles. Coefficients are illustrative and uncalibrated; they omit
clock, control, leakage and interconnect. Do not label scores watts, joules or
square micrometers. Lower traffic can reduce total energy score while shorter
runtime raises average activity; energy and power are different quantities.

Physical power and area remain `null`. A technology-based extension requires:

1. A public or school-provided Liberty cell library, operating corner and
   synthesis mapping (e.g. a properly licensed Nangate45 or SKY130 flow).
   This release does not silently assume a PDK.
2. SRAM macros/models or explicitly synthesized register memories. Memory
   bit count alone does not determine memory area or power.
3. Timing constraints and STA for Fmax; placement/routing and parasitics for
   stronger area/timing estimates.
4. Workload VCD/SAIF activity, voltage/frequency/corner and a power engine with
   compatible models for dynamic and leakage estimates.

The baseline passes numerical simulation and Yosys structural synthesis; it is
not claimed to be tapeout-ready or physically characterized. MP3 can compare
resources, activity and synthesis counts now. Physical PPA applies where a
supported technology flow is available.

See [RTL PPA](docs/rtl_ppa.md) for the required three-design synthesis comparison
and the evidence needed for a separate technology-based extension.
