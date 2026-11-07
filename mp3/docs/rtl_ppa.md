# RTL performance, power, and area evidence

| Measurement | Executable tool path | Scope |
| --- | --- | --- |
| Cycles, matrix-path TTFT/TPOT/request latency | `rtl_run.py run` through cocotb and Verilator | Actual RTL execution at an assumed clock |
| Generic logic cells, register bits, input-memory bits/ports | `rtl_run.py synth` through Yosys | Selected complete RTL; memories remain abstract |
| Dimensionless activity score | Counters from `rtl_run.py run` | Illustrative MAC, buffer, and external-memory costs |

## Required structural comparison

Run from the repository root with a fresh output directory each time:

```sh
MP3_PY="$PWD/mp3/.venv/bin/python"
"$MP3_PY" mp3/rtl_run.py synth --out mp3/results/baseline_synth_01
"$MP3_PY" mp3/rtl_run.py synth --design mp3/best_design.json \
  --out mp3/results/best_synth_01
"$MP3_PY" mp3/rtl_run.py synth --rtl-root mp4/rtl_student \
  --design mp3/best_design.json --out mp4/results/changed_synth_01
```

Inspect `synthesis.json`, the source snapshot, and the synthesis log. Generic
cells are not standard-cell area. Memory bit counts are not SRAM macro area.
The activity score in `summary.json` is dimensionless and must not be reported
as watts or joules. Whole-accelerator `physical_power_w` and `physical_area_um2`
remain `null`; `timing_closed` remains false.

## Technology-based extensions

The supplied flow provides structural synthesis and dimensionless activity
scores. Report whole-accelerator physical area and power as N/A in the report.

A separate technology-based analysis needs a cell library and operating corner,
technology mapping of the selected RTL, compatible SRAM models or explicitly
synthesized register memories, and timing constraints. Power estimates also need
that design's workload switching activity and an appropriate power engine.
Placement, routing, clock trees, and parasitics are needed for stronger physical
estimates. See the [shared workflow](../RTL_WORKFLOW.md#power-and-area).

Document the tool versions, libraries, memory organization, frequency, voltage,
activity source, and included components. Use the same analysis scope for the
baseline, MP3 best, and MP4 design. A single-PE estimate does not characterize
the full accelerator or its controller, interconnect, and memories.
