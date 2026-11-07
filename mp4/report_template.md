# MP4 report

Name / team:

## Bottleneck and proposed architecture

Use MP3 evidence to motivate the change. Include an architecture diagram and
describe datapath, buffering, scheduling, dependencies, and added resources.

## Implementation

List changed SystemVerilog modules and explain the changed hardware mechanism.
Include any adapter/protocol patch, added registers, memory ports and costs.
Distinguish supplied course infrastructure, unchanged MP3 parameters and your
new implementation. A parameter change by itself is MP3 work.

## Correctness

Provide reference comparisons, small schedule examples, resource/dependency
checks, partial-tile cases, and test commands/results. State what remains unverified.

## Three-way evaluation

| Workload | Design | Matrix cycles / ms | Prefill / decode | Traffic / stalls | Energy / power | Area | Evidence type |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | Original baseline | | | | | | |
| | MP3 best | | | | | | |
| | MP4 new | | | | | | |

Show baseline-relative and MP3-relative changes. Distinguish proxy PPA from
calibrated or implementation-based evidence. Include an ablation and a case
where the change does not help.
The original baseline uses course RTL with the supplied OS INT16/INT64 configuration;
MP3 best uses the same source with the selected configuration; MP4 new uses
your implementation. Keep the numerical execution mode consistent for all three.

For each workload, include simulated matrix-only TTFT, mean TPOT and complete
request latency, with baseline-relative speedups and reduction percentages
from `comparison.md`. Explain differences between prefill and decode behavior;
use each run's `token_timeline.csv` to inspect how decode latency changes with context.
Keep the clock assumption fixed when isolating cycle improvements.

## Full-stack interpretation

Explain how the changed kernels affect the request timeline and which work
remains outside the hardware timing model. Justify any claims of overlap,
persistent reuse, numerical accuracy, or full-system latency.

## Reproduction and limitations

List configurations, source hashes, commands, raw artifact locations, resource
budgets, frequency assumptions, cost coefficients, and limitations.
Include Yosys `synthesis.json` and logs for all three designs. Distinguish generic
logic/register/memory counts from technology area, and dimensionless activity
scores from watts or joules. Report exact integer and quantized-logit checks,
FP32 error, backpressure tests and architecture-specific correctness tests.
