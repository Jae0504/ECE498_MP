# MP4: AI Accelerator DSE — Architecture and RTL Implementation

Continue from MP3 and implement an architectural change in **SystemVerilog**.
Python/cocotb supplies external memory and returns your RTL's actual outputs
to miniLLAMA. Compare the original baseline, MP3 best and your implemented RTL.

## Start from the same hardware

Use the MP3 environment and run from the repository root:

```sh
MP3_PY="$PWD/mp3/.venv/bin/python"
"$MP3_PY" mp3/tools/prepare_rtl_student.py
"$MP3_PY" mp3/rtl_run.py check --rtl-root mp4/rtl_student \
  --design mp3/best_design.json --out mp4/results/rtl_initial_check
"$MP3_PY" mp3/rtl_run.py run --rtl-root mp4/rtl_student \
  --design mp3/best_design.json --workload mp3/configs/workload_decode.json \
  --out mp4/results/rtl_unchanged_decode
```

Run the [MP3 candidate study](../mp3/README.md#evaluate-baseline-and-parameter-candidates)
and save its selected configuration as `mp3/best_design.json` first.
`mp4/rtl_student/` starts identical to `mp3/rtl/`. The preparation tool
creates a missing copy and refuses to overwrite student edits. Confirm the
unchanged copy reproduces the saved MP3 result before editing it.

Save a reference run for each workload before editing. These paths are used by
the three-way comparison below:

```sh
for PHASE in prefill decode; do
  "$MP3_PY" mp3/rtl_run.py run --design mp3/best_design.json \
    --workload "mp3/configs/workload_${PHASE}.json" \
    --out "mp3/results/rtl_best_${PHASE}"
done
"$MP3_PY" mp3/rtl_run.py compare \
  mp3/results/rtl_best_decode mp4/results/rtl_unchanged_decode \
  --out mp4/results/rtl_reproduction
```

The unchanged copy should have identical cycles and tokens to MP3 best. It
does not contain the instructor's optimization; your RTL edits create the new
architecture. Use fresh output directories after each revision.

## Where to implement your design

| Actual file | Responsibility / possible work |
| --- | --- |
| `mp4/rtl_student/pe.sv` | Signed multiply, accumulation, PE registers and gating |
| `mp4/rtl_student/pe_array.sv` | Operand broadcast, PE enables, spatial organization and results |
| `mp4/rtl_student/tile_buffer.sv` | Banking, storage, ports and buffering |
| `mp4/rtl_student/controller.sv` | Tile loops, memory requests and load/compute/output scheduling |
| `mp4/rtl_student/accelerator_top.sv` | Integration and external interface |

Work can extend beyond the PE: investigate reuse, banking, load/compute overlap,
a GEMV datapath, pipelining or a different dataflow. Explain added logic,
registers, ports and memories. Changing supported JSON fields alone is MP3 work.

Use `--rtl-root mp4/rtl_student` on **every** check, run and synth command for
your implementation. The runner fingerprints and recompiles changed RTL; no
package reinstall is needed. Omitting the option runs the original MP3 RTL.

Preserve the [interface/accounting contract](../mp3/RTL_WORKFLOW.md#rtl-interface)
for direct compatibility. A new protocol, precision, layout or MAC organization
may need an explicit adapter patch and new independent reference too.
Never substitute software-reference outputs or modify reported cycle totals
to claim an optimization. Multiple in-flight matrices are not supported by
the starter protocol.

## Verify and compare

```sh
"$MP3_PY" mp3/rtl_run.py check --rtl-root mp4/rtl_student \
  --design mp3/best_design.json --out mp4/results/rtl_changed_check
for PHASE in prefill decode; do
  "$MP3_PY" mp3/rtl_run.py run --rtl-root mp4/rtl_student \
    --design mp3/best_design.json --workload "mp3/configs/workload_${PHASE}.json" \
    --out "mp4/results/rtl_changed_${PHASE}"
  "$MP3_PY" mp3/rtl_run.py compare \
    "mp3/results/dse_01/baseline/${PHASE}" "mp3/results/rtl_best_${PHASE}" \
    "mp4/results/rtl_changed_${PHASE}" --out "mp4/results/rtl_comparison_${PHASE}"
done
"$MP3_PY" mp3/rtl_run.py synth --out mp3/results/rtl_baseline_synthesis
"$MP3_PY" mp3/rtl_run.py synth --design mp3/best_design.json \
  --out mp3/results/rtl_best_synthesis_mp4
"$MP3_PY" mp3/rtl_run.py synth --rtl-root mp4/rtl_student \
  --design mp3/best_design.json --out mp4/results/rtl_changed_synthesis
```

The commands use the baseline produced by `mp3/results/dse_01`; substitute
your own study directory when needed. Hold model, weights,
token count, memory service and main resource budgets fixed; explain changes.
Include boundary/backpressure tests, tests of the new mechanism, and a case
where it does not help. Add `--waves` for waveform debugging.

The [shared RTL workflow](../mp3/RTL_WORKFLOW.md) explains exact integer checks,
FP32 error, matrix-only timing, tools and PPA limits. Yosys counts are actual
implementation evidence, but do not prove Fmax or physical power/area.
Submit RTL, necessary adapter patches and reproducible evidence following the
[report guide](REPORT_GUIDE.md) and [report template](report_template.md).

## Folder guide

`rtl_student/` is the student-editable hardware. `results/` contains generated
measurements and is ignored by Git. The environment and evaluation code are
shared with `mp3/`; MP4 needs no second Python environment or extra submodule.
