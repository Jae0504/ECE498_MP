# MP3: AI Accelerator DSE — Parameter Optimization

Execute tiny miniLLAMA with a supplied **synthesizable INT16 PE-array RTL**
accelerator. Explore supported parameters while keeping its RTL unchanged.
MP4 starts with the same hardware and asks you to change its RTL.

The required path is **miniLLAMA → Python/cocotb → Verilator RTL → miniLLAMA**.
Actual integer accumulator outputs determine logits and generated tokens.

## Start

Run from the repository root. See the [shared RTL workflow](RTL_WORKFLOW.md)
for dependencies, interfaces, validation, timing and power/area limits.

```sh
sh mp3/setup_rtl.sh
MP3_PY="$PWD/mp3/.venv/bin/python"
"$MP3_PY" mp3/rtl_run.py check --out mp3/results/rtl_check
"$MP3_PY" mp3/rtl_run.py run --out mp3/results/rtl_baseline_smoke
cp mp3/configs/rtl_baseline.json mp3/my_design.json
```

Use fresh result directories for every run. Setup initializes the pinned
miniLLAMA/cocotb/Verilator submodules, including empty submodule directories
from a normal clone. It prepares only miniLLAMA for the RTL model path.
Linux, Python 3.12, a C++ compiler, make, autoconf, flex, bison and Git were
used for validation. No GPU, commercial license or sudo is needed.

## What you change

Edit `mp3/my_design.json`; preserve `mp3/rtl/*.sv` and the baseline JSON.

| Field | Hardware meaning |
| --- | --- |
| `array_rows`, `array_cols` | PE dimensions; one INT16 MAC per enabled PE per cycle |
| `a_buffer_words`, `b_buffer_words` | INT16 input-memory words; divisible by respective row/column count |
| `memory_lanes` | INT16 words per accepted beat on each independent read port |
| `memory_wait_cycles` | External memory wait cycles per read beat |
| `clock_mhz` | Assumed clock for converting cycles to time; hold fixed without STA evidence |

Teaching baseline: 16×4 PEs, 6 KiB A storage and 2 KiB B storage, INT64 accumulators,
8 B/cycle on each A/B read port, 8 B/cycle on the C write port, assumed 100 MHz.
The output-stationary array broadcasts operands; it is not a neighbor-to-neighbor
systolic array. Loading and compute do not overlap in the baseline.

This starting configuration deliberately leaves room for workload-driven tuning:
short prompts and single-token decode use few array rows. The earlier balanced
8×8 configuration remains available as `configs/rtl_reference_8x8.json`; include
it as a reference in your exploration. The teaching baseline is not a claim
about the worst possible design. All configurations use real RTL execution;
there are no artificial delays or changes to the timing calculation.

Pipeline, dataflow and precision are fixed in MP3: this starter has no selectable
implementations of those mechanisms. Implementing them is MP4 work. An unused
or unsupported JSON field is rejected.

## Evaluate baseline and parameter candidates

Copy `configs/rtl_candidates.json` to `my_candidates.json` and edit its named
parameter overrides. The baseline is always included automatically. Each
candidate must preserve PE count, total SRAM, port bandwidth, memory wait,
and clock. The default objective gives equal weight to baseline-normalized
prefill and decode request latency; declare different weights before running
if your study has another objective.

```sh
cp mp3/configs/rtl_candidates.json mp3/my_candidates.json
# Edit mp3/my_candidates.json before this command.
"$MP3_PY" mp3/rtl_run.py sweep --candidates mp3/my_candidates.json \
  --out mp3/results/dse_01
cat mp3/results/dse_01/sweep.md
cp mp3/results/dse_01/best_design.json mp3/best_design.json
```

`sweep` performs ten standalone RTL checks per configuration and actual model
runs on both `workload_prefill.json` and `workload_decode.json`. It records the
objective and configurations before execution, retains every candidate's
results, rejects invalid budgets before simulation, and writes `selected.json`
and `best_design.json`. A selected design may improve one metric and regress
another; report both. Extend the two supplied candidates with your own hypotheses.

The study directory contains `baseline/{prefill,decode}`, one directory per
candidate, and `comparison_{prefill,decode}`. To evaluate one configuration
independently and obtain generic synthesis counts:

```sh
"$MP3_PY" mp3/rtl_run.py run --design mp3/my_design.json \
  --workload mp3/configs/workload_decode.json --out mp3/results/rtl_trial_decode
"$MP3_PY" mp3/rtl_run.py compare \
  mp3/results/dse_01/baseline/decode mp3/results/rtl_trial_decode \
  --out mp3/results/rtl_decode_comparison
"$MP3_PY" mp3/rtl_run.py synth --design mp3/best_design.json \
  --out mp3/results/rtl_best_synthesis
```

`summary.json`, `kernels.csv`, `token_timeline.csv` and `comparison.md` report
**matrix-path** TTFT, TPOT and request latency from RTL cycles, not CPU wall
time. PyTorch still executes non-matrix operators. Every run checks exact
integer products and quantized-reference logits, and separately reports FP32 error.

## Area, power and submission

`synth` runs Yosys on the selected RTL/configuration and reports actual generic
logic cells, flip-flops and inferred memory bits. These are not physical area.
Energy/activity scores are dimensionless, uncalibrated teaching proxies.
Whole-accelerator physical power and area are `null`; see [PPA scope](docs/rtl_ppa.md)
for the required structural comparison and limits of the supplied flow.

Submit the [report](report_template.md), selected `best_design.json`, exact
commands, model/workload/weight identity, summaries, kernel/timeline CSVs,
numerical validation and synthesis/provenance files. Preserve the baseline
source hash. Do not submit environments, build caches or large netlists unless requested.

## Folder guide

| Path | Role in the required assignment |
| --- | --- |
| `configs/rtl_*.json`, `model_tiny.json`, `workload_*.json` | Baseline, candidate list, model, and workload inputs |
| `rtl/` | Fixed, synthesizable baseline; preserve it in MP3 |
| `rtl_run.py`, `rtl_backend/` | CLI, model hooks, RTL driver, sweep, reports, and synthesis |
| `tools/` | Dependency preparation, MP4 starter copy, and integration checks |
| `tests/test_rtl_*.py`, setup tests | Default RTL regression suite |
| `patches/miniLLAMA.patch`, `vendor_versions.json` | Reproducible model preparation |
| `RTL_WORKFLOW.md`, `docs/` | Interface, timing, dependencies, and PPA documentation |
| `results/`, `.deps/`, `.venv/` | Generated locally; not committed |
