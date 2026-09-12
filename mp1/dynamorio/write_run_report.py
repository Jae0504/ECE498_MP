"""Generate a fresh-run report from this run's data, never the instructor's historical prose."""
import argparse,csv,datetime,json,sys
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);parser.add_argument('--finalize-now',action='store_true');args=parser.parse_args()
out=args.output.resolve();data=json.loads((out/'analysis.json').read_text());meta=json.loads((out/'experiment_metadata.json').read_text());cfg=meta['configuration']
def table(rows):
    if not rows:return 'No completed rows.\n'
    keys=list(rows[0]);return '| '+' | '.join(keys)+' |\n|'+'|'.join(['---']*len(keys))+'|\n'+'\n'.join('| '+' | '.join(f'{r[k]:.6g}' if isinstance(r[k],float) else str(r[k]) for k in keys)+' |' for r in rows)+'\n'
generated=datetime.datetime.now(datetime.timezone.utc)
end_path=out/'experiment_end.json'
if end_path.exists() and not args.finalize_now:
    now=datetime.datetime.fromisoformat(json.loads(end_path.read_text())['end_utc'])
else:
    now=generated;end_path.write_text(json.dumps({'end_utc':now.isoformat()},indent=2)+'\n')
start=datetime.datetime.fromisoformat(meta['start_utc']);elapsed=(now-start).total_seconds()
trace_bytes=sum(p.stat().st_blocks*512 for p in (out/'traces').rglob('*') if p.is_file())
phase_path=out/'attention_phase_trace_characterization.csv';phases=list(csv.DictReader(phase_path.open())) if phase_path.exists() else []
# Interpret the current measurements; incomplete result sets stay explicit.
main_by_name={r['Kernel']:r for r in data['main']}
trace_by_name={r['Kernel']:r for r in data['trace']}
roof_by_name={r['Kernel']:r for r in data['roofline']}
phase_by_name={r['Kernel']:r for r in data['phases']}
diagnoses=[]
for label,m in main_by_name.items():
    t=trace_by_name[label];r=roof_by_name[label]
    evidence=(f"**{label}:** AI {m['Algorithmic_AI_FLOP_per_Byte']:.4g} FLOP/Byte gives a {r['Predicted_Bound'].lower()} prediction. "
              f"Native throughput is {m['GFLOP_per_s']:.4g} GFLOP/s, {r['Roofline_Efficiency_Percent']:.3g}% of the predicted roof. "
              f"The trace contains {t['Instructions']:,} instructions, {t['Loads']:,} loads and {t['Stores']:,} stores "
              f"({t['Load_Store_per_FLOP']:.4g} memory operands/useful operation). "
              f"Simulated L1D/LLC local miss rates are {100*t['L1D_Miss_Rate']:.4g}%/{100*t['LLC_Miss_Rate']:.4g}%. "
              f"Finite combined I+D reuse distance has mean {t['Mean_Reuse_Distance']:.4g} and median {t['Median_Reuse_Distance']:.4g} lines; "
              f"the per-thread data cold-reference fraction is {100*t['Data_Cold_Reference_Fraction']:.4g}%. ")
    if label.startswith('GEMM'):
        conclusion='The unblocked loop repeatedly transfers weights and updates outputs through the cache hierarchy. The evidence supports a mixed implementation-throughput and locality/data-movement limitation; the compute-side model label is not evidence of saturated FMA units.'
    elif label=='GEMV':
        conclusion='Low algorithmic AI makes this workload data-movement sensitive. Warm native LLC residency, vectorized instruction throughput and passive OpenMP wake-up costs all matter; the cold simulated LLC miss rate does not establish native DRAM bandwidth saturation.'
    else:
        ph=phase_by_name[label];phase=max(['QK','Softmax','PV'],key=lambda k:ph[k+'_ms'])
        conclusion=(f"{phase} has the largest phase median ({ph[phase+'_ms']:.4g} ms). "
                    'Reduction order, scalar/exp work, locality and thread/phase overhead support a mixed diagnosis; phase medians and a non-cycle-accurate trace do not determine stall percentages.')
    diagnoses.append(evidence+conclusion)
diagnosis_text='\n\n'.join(diagnoses)
comparisons=[]
if all(k in main_by_name for k in ['GEMM S=128','GEMV']):
    g=main_by_name['GEMM S=128'];v=main_by_name['GEMV'];gt=trace_by_name[g['Kernel']];vt=trace_by_name[v['Kernel']]
    comparisons.append(f"**GEMM versus GEMV:** GEMM128 AI is {g['Algorithmic_AI_FLOP_per_Byte']/v['Algorithmic_AI_FLOP_per_Byte']:.3f} times GEMV AI because the same weights can serve many token rows while useful work grows with M. In this implementation, load+store operands/useful operation are {gt['Load_Store_per_FLOP']:.6f} for GEMM128 versus {vt['Load_Store_per_FLOP']:.6f} for GEMV: high ideal AI does not guarantee fewer dynamic operands per FLOP. GEMM's cold-data fraction is {100*gt['Data_Cold_Reference_Fraction']:.4f}% versus {100*vt['Data_Cold_Reference_Fraction']:.4f}% for GEMV; its simulated LLC miss rate is {100*gt['LLC_Miss_Rate']:.4f}% versus {100*vt['LLC_Miss_Rate']:.4f}%. These reflect reuse across token rows. GEMV's smaller finite reuse-distance mean includes local output/spatial/instruction reuse and omits first weight observations, so it is not proof of better cross-token weight locality.")
if all(k in main_by_name for k in ['GEMM S=128','GEMM S=512']):
    a=main_by_name['GEMM S=128'];b=main_by_name['GEMM S=512'];ta=trace_by_name[a['Kernel']];tb=trace_by_name[b['Kernel']]
    comparisons.append(f"**GEMM128 to GEMM512:** AI grows {b['Algorithmic_AI_FLOP_per_Byte']/a['Algorithmic_AI_FLOP_per_Byte']:.3f}x, useful work {b['FLOPs']/a['FLOPs']:.3f}x and runtime {b['Runtime_ms']/a['Runtime_ms']:.3f}x; throughput changes {100*(b['GFLOP_per_s']/a['GFLOP_per_s']-1):+.3f}%. Mean reuse distance changes {ta['Mean_Reuse_Distance']:.2f} to {tb['Mean_Reuse_Distance']:.2f} lines (medians {ta['Median_Reuse_Distance']:g} and {tb['Median_Reuse_Distance']:g}). Simulated L1D misses change {100*ta['L1D_Miss_Rate']:.4f}% to {100*tb['L1D_Miss_Rate']:.4f}%; LLC misses {100*ta['LLC_Miss_Rate']:.4f}% to {100*tb['LLC_Miss_Rate']:.4f}%. More output rows amortize compulsory LLC misses, while the unblocked full-weight sweep and repeated output updates persist. Ideal reuse therefore need not improve L1 locality or realized GFLOP/s.")
if all(k in main_by_name for k in ['Attention S=128','Attention S=512']):
    a=main_by_name['Attention S=128'];b=main_by_name['Attention S=512'];ta=trace_by_name[a['Kernel']];tb=trace_by_name[b['Kernel']]
    comparisons.append(f"**Attention128 to Attention512:** runtime scales {b['Runtime_ms']/a['Runtime_ms']:.3f}x, analytical work {b['FLOPs']/a['FLOPs']:.3f}x, instructions {tb['Instructions']/ta['Instructions']:.3f}x, load+store references {(tb['Loads']+tb['Stores'])/(ta['Loads']+ta['Stores']):.3f}x and operand bytes {tb['Dynamic_Memory_Reference_Bytes']/ta['Dynamic_Memory_Reference_Bytes']:.3f}x. Simulated L1D miss rates are {100*ta['L1D_Miss_Rate']:.4f}% and {100*tb['L1D_Miss_Rate']:.4f}%; LLC rates {100*ta['LLC_Miss_Rate']:.4f}% and {100*tb['LLC_Miss_Rate']:.4f}%. The materialized score/probability working sets grow quadratically and phase overhead is amortized differently; native phase times above identify the largest phase in this actual run.")
comparison_text='\n\n'.join(comparisons) or 'The corresponding workload pairs have not all completed.'
peak_path=out/'peak_trace_snapshot.json'
peak_bytes=json.loads(peak_path.read_text())['all_trace_allocated_bytes'] if peak_path.exists() else None
peak_text=(f"The sampled global trace peak was {peak_bytes/2**30:.3f} GiB; 30 coincident peaks would require about {30*peak_bytes/2**30:.1f} GiB." if peak_bytes else 'A global trace peak was not recorded.')
class_cost=(f"The completed set retains {trace_bytes/2**30:.3f} GiB per user, or an estimated {30*trace_bytes/2**30:.1f} GiB for 30 copies. "
            f"{peak_text} Thirty serial repetitions at this observed elapsed time would total about {30*elapsed/3600:.2f} hours; this is an extrapolation, not a measured concurrent-class runtime. "
            'Thirty simultaneous eight-core runs request 240 distinct physical cores. Shared caches, memory bandwidth and storage further prevent linear scaling.')
verdict_reason=('All five workloads and separate phase traces are available, but unrestricted full-trace execution by 30 students is not recommended. The scripts are usable for instructor runs or scheduled, resource-allocated batches; provide sufficient per-user storage and shared instructor traces before assigning the complete workload set.'
                if len(data['main'])==5 and len(phases)==6 else 'This output directory is still a partial experiment. Complete and validate all required workloads before deployment; class-wide full-trace costs also require scheduling and storage planning.')
report=f'''# MP1 DynamoRIO measurement on {meta['host']}

This report contains **{len(data['main'])} completed full workload(s)**. Values below come from this result directory. No timings, cache values or phase metrics are copied from the reference. Check `validation.txt`, `verification.json`, `reference_comparison.csv` and raw `logs/` after the validator completes.

## Configuration and correctness

CPU: **{cfg['cpu_model']}**. Native and ROI execution use eight distinct physical cores, CPU set **{cfg['cpu_set']}**, eight threads. Compiler: `{cfg['compiler']}`. DynamoRIO submodule: `{cfg['dynamorio_commit']}`; project-local LZ4: `{cfg['lz4_commit']}`. Full topology, OS and tool versions: [environment.txt](environment.txt). The build gate ran all four original CTest tests; each native/ROI benchmark also runs correctness checks. Every traced full output is checked element-by-element against its native expected output, and checksums are compared with the original gnr2 workload.

Native measurements use the unchanged MP1 kernel-region timer, **2 warmups / 9 samples / median**, no cache flush in the main rows. OpenMP settings: `{cfg['omp']}`. Allocation, RNG, validation and printing are outside native timing and ROI tracing; output clears, OpenMP synchronization, math-library code and small ROI boundary/clock overhead are included. Instrumented wall time is only an overhead diagnostic. Attention's QK+scale, softmax, PV and total medians retain the original phase separation; separate phase medians need not sum exactly to the total median.

## Four types of numbers

- **MEASURED:** uninstrumented time, useful analytical work/time reported as GFLOP/s, and native FMA/bandwidth ceilings.
- **ANALYTICAL:** dimension-based FLOPs, specified minimum bytes, algorithmic AI and simple Roofline prediction.
- **TRACE-DERIVED:** dynamic instructions, memory operands and bytes, opcode mix, unique data lines and reuse statistics.
- **SIMULATED:** cold cache accesses/misses and local miss rates. These are not measured hardware cache misses.

## Formulas

FP32 projection: N=5632, K=2048. GEMM M=S: FLOPs=`2*M*N*K`, bytes=`4*(M*K+K*N+M*N)`. GEMV M=1 uses the same formula. One simplified Attention head has D=64: QK work=`2*S*S*D+S*S`, softmax=`5*S*S` operation-equivalents, PV=`2*S*S*D`, total=`4*S*S*D+6*S*S`. Attention minimum-byte model is the existing **phase sum** `16*(S*D+S*S)`, including materialized intermediate phase inputs/outputs. This is not the fused whole-operation compulsory external-I/O bound `16*S*D`, and not minimum DRAM traffic. Exp/max are not literally single hardware FLOPs. AI=`work/minimum_bytes`; native GFLOP/s=`work/(median_ms*1e6)`.

## Main characterization

Miss rates in CSVs and this table are fractions in [0,1].

{table(data['main'])}

## Roofline

Native register-FMA reference: **{data['compute']:.6f} GFLOP/s**. Sixteen independent AVX-512 accumulators, 16 lanes, 2 FLOPs/FMA, 20,000,000 iterations, 8 threads. Native bandwidth reference: **{data['bandwidth']:.6f} GB/s**, the greater of measured STREAM-like read and triad medians, each using a 1.5 GiB working set. Triad useful bytes exclude write-allocate traffic. These are measured references for the selected cores, not hardware DRAM counters or a guarantee of whole-machine saturation. Ridge=`compute/bandwidth`=**{data['ridge']:.6f} FLOP/Byte**.

Prediction=`min(compute,AI*bandwidth)`; efficiency=`100*native_GFLOPs/prediction`. Compute/Bandwidth labels describe that analytical model; low efficiency alone does not prove latency stalls.

{table(data['roofline'])}

![Roofline](roofline.png)

## Trace, cache and reuse

One complete ROI per workload, with LZ4 raw and ZIP converted traces; no sampling, shortened shapes, instruction truncation, L0 filtering or reuse-distance cap. The same trace feeds basic_counts, opcode_mix, reuse_distance, cache_simulator and the data-byte/unique-line helper. Instructions include fetched plus non-fetched REP iterations. Loads/stores are data operands, not disjoint instruction categories or FP32 elements. Data bytes sum recorded operand widths; unique data lines union all intersected 64-byte virtual lines across threads. These bytes are **not DRAM traffic**.

The fixed reference model is `{cfg['cache_model']}`. L1I is 64 KiB/16-way. It reproduces the instructor's model rather than silently claiming to match every server. On gnr2 the actual LLC is 336 MiB; this simulator cannot represent its non-power-of-two set count at 16 ways, so the modeled LLC is 256 MiB. Caches use LRU, coherence and virtual addresses. Native timing follows warmups, but untraced warmups leave the simulator cold. LLC is unified and its local denominator only includes requests reaching it. Cache-line splits can yield more accesses than memory operands. Scheduling, physical mapping, LLC slicing and prefetch/stall behavior are not faithful hardware models.

{table(data['trace'])}

`Load_Store_per_Instruction=(Loads+Stores)/Instructions`; `Load_Store_per_FLOP=(Loads+Stores)/analytical_work`. L1D misses/accesses sum all eight private caches before division; LLC uses its own local misses/accesses. Reuse mean/median are the tool's finite, unbinned, per-thread combined instruction+data distances, pooled across threads; cold accesses are excluded. The CDF selects data-reference histogram counts but distance still includes instruction lines. Unique-line helper splits operands; reuse uses their starting address. Geometric histogram bins use multiplier 1.5. Cold-data fraction=`(data_accesses-sum(data_histogram_counts))/data_accesses`, with first observations counted separately by thread. Small GEMV reuse distance can reflect output/spatial/instruction reuse without cross-token weight reuse.

![Cache rates](cache_miss_rates.png)
![Instruction and memory ratios](instruction_memory_mix.png)
![Reuse distances](reuse_distance.png)

## Attention phases

{table(data['phases'])}

{table(phases) if phases else 'No Attention phase measurements are part of the completed workload set yet.'}

{('![Native phases](attention_phases.png)' if phases else '')}

Separate phase trace caches start independently cold; their misses cannot be summed to reconstruct the total-Attention trace. Phase native times come from the original whole-Attention benchmark. No phase-specific trace metric is inferred from a total trace.

## Interpretation and comparison

{diagnosis_text}

{comparison_text}

The provided unblocked i-k-j GEMM sweeps the entire 44 MiB weight matrix for each output row. Algorithmic AI increases with token count, while repeated loads/stores and the long weight reuse path can persist. Compare per-FLOP operands, cache rates and native throughput together. GEMV has about 0.5 FLOP/Byte and little cross-token weight reuse, but warm LLC residency and passive thread wake-up costs complicate attributing time to DRAM. Attention combines QK reductions, scalar/exp work, contiguous PV updates and thread/phase overhead. Inspect its recorded phase times before diagnosing a dominant phase. This evidence supports conservative mixed implementation/data-movement diagnoses, not precise stall shares or proof of peak arithmetic saturation.

`reference_comparison.csv` compares all main/extended metrics against the actual archived gnr2 results. Analytical values and deterministic output checksums must match. Native time/GFLOP/s, runtime instruction counts, ASLR-dependent virtual addresses/cache mapping and reuse statistics are observations that can differ across runs. A PASS means the experiments and identities were verified; it does **not** mean all numeric values were identical. CPU/compiler/cache settings and source/binary hashes are recorded in `experiment_metadata.json`.

## Cost, limitations and student deployment

{table(data['resources'])}

{class_cost}

Recorded elapsed time for this measurement set: **{elapsed:.2f} s**. Retained trace allocation, including any phase traces: **{trace_bytes} bytes ({trace_bytes/2**30:.3f} GiB)**. Independent offline passes may overlap after conversion; summed stage costs are not elapsed time. Per-command generation/analysis times, peak disk samples and child RSS are in `trace_resources.csv`; `commands.jsonl` records commands and exit status. Raw records are removed after successful full analysis; converted ZIP records, maps and encodings remain. No hardware cycles, DRAM traffic, exact frequency or stall percentage was collected. Large native timing ranges and shared-server contention limit fine comparisons.

Build in each user's own clone with `sh mp1/dynamorio/build.sh`. Run each workload with its `measure_*.sh [RESULTS_DIR]` wrapper, or all of them with `measure_all.sh`. Set `MP1_CPUS` to an instructor/scheduler-assigned list of eight physical cores; giving every student CPUs 0–7 would cause contention. The default chooses the first eight allowed physical cores. Linux x86_64 with AVX-512F/FMA is required by this CPU-specific ceiling; other architectures need a revised ceiling. The fixed reference cache model is explicitly simulated, not auto-detected hardware. Full GEMM512 needs substantial analysis time and at least 18 GiB free before the script starts. This packaging does not remove the original full-trace cost: use shared instructor traces, smaller supported individual exercises or scheduled batches for a class of 30.

**NOT RECOMMENDED FOR STUDENT USE**

{verdict_reason}
'''
(out/'DYNAMORIO_MP1_REPORT.md').write_text(report)
(out/'experiment_summary.json').write_text(json.dumps({'start_utc':meta['start_utc'],'end_utc':now.isoformat(),'report_utc':generated.isoformat(),'elapsed_seconds':elapsed,'retained_trace_allocated_bytes':trace_bytes,'complete_workloads':[r['Kernel'] for r in data['main']]},indent=2)+'\n')
print('Fresh-run report generated solely from measured outputs: '+str(out))
