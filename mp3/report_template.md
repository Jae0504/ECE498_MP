# MP3 report

Name / team:

## Objective and constraints

State the optimization objective, PE/SRAM/bandwidth budgets, clock assumption,
and criteria for choosing the best configuration.
Include the candidate list and workload weights recorded by `rtl_run.py sweep`
before execution, or an equivalent declaration for a manual study.
Use the supplied OS INT16/INT64 RTL baseline. Identify supported JSON fields
you changed; keep RTL source, arithmetic and dataflow unchanged.

## Reproduction

List source fingerprints, environment, model/workload configurations, weight
source, seeds, exact commands, and artifact locations.
Include the common RTL source hash and numerical validation results for all
designs. Use `rtl_run.py run` and `rtl_run.py synth` throughout this comparison.

## Baseline characterization

Explain the request timeline and the dominant GEMM, GEMV, QKᵀ, and AV costs.
Discuss utilization, stalls, and traffic using measured simulator counters.

## Exploration

| Design | Array | Input SRAM / port bandwidth | Prefill cycles | Decode cycles | Matrix us | External bytes | Activity score | Generic cells / memory bits |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | | | | | | | | |
| Candidate | | | | | | | | |

Explain your hypotheses, experiment coverage, interactions, and regressions.
Provide a separate comparison for each workload.
Include all candidates in `exploration.json` and `sweep.md`, not just the winner.
Report simulated matrix-only TTFT, mean TPOT and request latency with speedups
and reductions from `comparison.md`. Relate these changes to observed cycles,
stalls and traffic, not CPU/simulator wall time.

## Selected configuration

Justify the winner under the stated objective and budgets. Show why a different
objective or workload might select a different configuration.

## Limitations and MP4 direction

Identify untimed work, uncalibrated activity-cost assumptions, and numerical limits.
Report exact integer/quantized-logit checks and FP32 errors separately. Generic
cells and memory bits are not physical area; power and physical area are unavailable
unless a technology flow is supplied. State the assumed clock and lack of STA.
Describe an architectural bottleneck worth investigating in MP4.
Explain why the proposed MP4 work requires a changed mechanism beyond the
parameter settings explored here.
