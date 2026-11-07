# MP4 Report Guide

Submit a concise report that **compares the original baseline, the best MP3
configuration, and your new RTL design using PPA and workload-level metrics,
and explains why the changes help or hurt**.

**The submission deadline will be announced separately by the instructor.**

Follow the [MP4 README](README.md) for execution and comparison commands. The
[report template](report_template.md) is an optional starting structure.
Use the [shared RTL workflow](../mp3/RTL_WORKFLOW.md) and
[PPA guide](../mp3/docs/rtl_ppa.md) for measurement definitions and tool limitations.

## 1. Designs, objective, and comparison conditions

Identify all three designs:

| Design | RTL implementation | Configuration |
| --- | --- | --- |
| Original baseline | Supplied `mp3/rtl/` | Supplied `mp3/configs/rtl_baseline.json` |
| Best MP3 configuration | The same supplied RTL | Your selected MP3 parameter configuration |
| New MP4 design | Your modified `mp4/rtl_student/` | The selected MP3 configuration for the primary comparison |

State the MP3 selection objective and why that configuration was selected.
Record the configuration values, source revisions or fingerprints, tool versions,
and result directories. Identify the original baseline by its actual configuration
and revision rather than by a directory name alone.

Use the same model, weights, prompt, batch size, generated-token count, numerical
format, and memory-service conditions across the three designs for each workload.
Keep PE count, total input SRAM, external bandwidth, and assumed clock fixed for
the primary comparison. Account for additional control registers, logic, ports,
and interconnect. Report experiments that change these conditions separately.

Confirm that the unchanged MP4 starter reproduces the selected MP3 configuration
before introducing your RTL changes.

## 2. Architecture changes and correctness

Explain the bottleneck identified in MP3 and your proposed architectural or
microarchitectural change. Examples include datapath, buffering, pipeline,
dataflow, scheduling, or specialized-unit changes. Include:

- A diagram or small execution schedule showing the relevant data movement and control.
- The SystemVerilog files changed and the mechanism implemented in each.
- The expected effect on computation, memory traffic, stalls, or utilization.
- Additional resources and any necessary changes to the Python interface or accounting.

Changing configuration parameters alone does not constitute an MP4 architecture
implementation. Use `--rtl-root mp4/rtl_student` when checking, executing, and
synthesizing your implementation, and preserve the executed source fingerprints.

Every reported performance result must pass correctness validation. Summarize
standalone RTL checks and model-integrated validation, including exact integer
matrix results, quantized-reference logits, and the separately reported FP32
error. Include partial tiles, GEMV, K panels, backpressure, and tests specific
to your new mechanism. Explain any limitations of the checks; generated text
or a checksum alone is not sufficient correctness evidence.

## 3. Workload-level performance comparison

Evaluate all three designs on both supplied workloads:

- `mp3/configs/workload_prefill.json`: long-prompt prefill.
- `mp3/configs/workload_decode.json`: short-prompt prefill followed by cached decode.

Record the actual prompt length and output-token count. Use separate rows for
each workload and design:

| Workload | Design | Total RTL cycles | TTFT (us) | Mean TPOT (us) | Request latency (us) |
| --- | --- | ---: | ---: | ---: | ---: |
| Prefill | Original baseline | | | N/A | |
| Prefill | MP3 best | | | N/A | |
| Prefill | MP4 new | | | N/A | |
| Decode | Original baseline | | | | |
| Decode | MP3 best | | | | |
| Decode | MP4 new | | | | |

Report MP4 speedup or regression relative to **both** the original baseline and
MP3 best. Identify the metric used for every percentage or speedup. For example,
request speedup is `reference request latency / new request latency`; latency
reduction is `100 * (reference latency - new latency) / reference latency`.

These times are simulated **matrix-path contributions** derived from observed
RTL cycles and an assumed clock: `latency_us = cycles / clock_mhz`. TTFT covers
matrix execution through the first token; mean TPOT covers subsequent tokens.
For `N` generated tokens, request latency is `TTFT + (N - 1) * mean TPOT`.
Use N/A for mean TPOT when only one token is generated.

Use `summary.json`, `comparison.md`, and `token_timeline.csv` as evidence. Explain
differences between TTFT, TPOT, and total request latency; improving one does not
guarantee that the others improve. CPU wall time, Python validation time, and
Verilator compilation time are not accelerator latency.

## 4. Power, performance, and area evidence

Run `rtl_run.py synth` for all three designs and compare the available evidence:

| Design | Generic logic cells | Register bits | Input-memory bits / ports | Activity score per request | Evidence scope |
| --- | ---: | ---: | --- | ---: | --- |
| Original baseline | | | | | |
| MP3 best | | | | | |
| MP4 new | | | | | |

Use `synthesis.json` and the synthesis logs for structural counts, and the run's
`cost_proxy` and activity counters for the dimensionless activity score. Compare
activity separately for prefill and decode. Explain how added logic, registers,
buffer organization, and reduced or increased data movement affect the design
tradeoff. Generic cell counts are structural area proxies; the activity score
is an uncalibrated energy-related proxy, not measured power.

If technology-based area, power, or timing results are available, report them
in an additional table with units, tool/model, library or process, operating
corner, clock, switching activity, and included components. Use a consistent
scope and method for the three designs. Distinguish total energy per request
from average power.

With the supplied default flow, whole-accelerator physical area and power are
unavailable. Label these entries **N/A**, explain the missing characterization,
and provide the structural/activity comparison above. Do not invent physical
values or relabel proxy scores as watts, joules, or square micrometers.
Any additional technology-based analysis must cover the stated design and
memory organization, with its tools, libraries, and assumptions documented.

An assumed clock is not a measured Fmax. A pipelining change that adds cycles
does not establish a speedup unless a justified timing analysis supports a
higher operating frequency.

## 5. Explain why the changes help or hurt

Connect the observed results to your implementation using `kernels.csv`,
traffic and stall counters, utilization, and selected waveforms or schedules.
Discuss GEMM, GEMV, and attention QK/AV behavior where relevant. GEMM/GEMV are
shape categories and QK/AV are operation roles; do not double-count them as
independent contributions to total time.

Answer:

1. Which bottleneck did the change target, and which measurements show that it changed?
2. Which kernels or phases improved, regressed, or remained unchanged, and why?
3. Did lower memory traffic reduce cycles, or did another bottleneck remain?
4. How do the benefits compare with the added logic, registers, memory ports, or activity?
5. Why do prefill and decode respond differently to the same architecture?
6. Under the declared objective and constraints, is the MP4 design preferable to MP3 best?

Include an ablation that isolates the mechanism when multiple changes are
combined, and a workload or matrix shape where it gives little benefit or
causes a regression. Report actual observations even if the design does not
improve performance. A supported explanation of a regression is more useful
than assuming that a more complex design must be faster.

## 6. Reproduction and limitations

Make the report understandable without rerunning your experiments. Include the
configuration and RTL identifiers, exact commands, and paths to the evidence
used for every table or figure. Reference the numerical checks, comparisons,
kernel/timeline CSVs, and synthesis reports. Summarize the changed RTL and any
adapter changes sufficiently to connect the reported results to the implementation.

State which parts of inference remain outside the timed hardware path, including
quantization, layout conversion, softmax, normalization, RoPE, activations, and
host overhead. Explain remaining correctness, timing, and PPA limitations.
Keep measured simulation results, analytical estimates, structural proxies,
and technology-based estimates clearly labeled throughout the report.
