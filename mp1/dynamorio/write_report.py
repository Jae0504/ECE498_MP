"""Write the course report from validated CSV/JSON artifacts; no synthetic measurements."""
import argparse,csv,datetime,json,re,shutil
from pathlib import Path
from run_logged import OUT,ROOT,disk

def table(rows,columns):
    def fmt(v):
        if isinstance(v,float):return f'{v:,.5g}'
        if isinstance(v,int):return f'{v:,}'
        return str(v)
    return '| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(r[c]) for c in columns)+' |' for r in rows)+'\n'

d=json.loads((OUT/'analysis.json').read_text());main=d['main'];tr=d['trace'];roof=d['roofline'];res=d['resources'];ph=d['phases'];aux=d['aux']
start=datetime.datetime.fromisoformat(json.loads((OUT/'experiment_start.json').read_text())['start_utc'])
parser=argparse.ArgumentParser()
parser.add_argument('--finalize-now',action='store_true',help='Record this time as the experiment end; ordinary regeneration preserves the recorded end.')
args=parser.parse_args()
end_path=OUT/'experiment_end.json'
if end_path.exists() and not args.finalize_now:
    end=datetime.datetime.fromisoformat(json.loads(end_path.read_text())['end_utc'])
else:
    end=datetime.datetime.now(datetime.timezone.utc)
    end_path.write_text(json.dumps({'end_utc':end.isoformat()},indent=2)+'\n')
elapsed=(end-start).total_seconds()
free=shutil.disk_usage(OUT).free
snapshot_path=OUT/'peak_trace_snapshot.json'
snapshot=json.loads(snapshot_path.read_text()) if snapshot_path.exists() else None
peak_sentence=(f"A recorded global snapshot while GEMM512 raw and converted records coexisted occupied **{snapshot['all_trace_allocated_bytes']/2**30:.3f} GiB**. This is an observed footprint, not a continuously measured global maximum." if snapshot else 'Per-trace sampled peak footprints are recorded above; no continuous global maximum was collected.')
retained=disk(OUT/'traces')[1]
generation=sum(r['Trace_Generation_s'] for r in res);analysis=sum(r['Conversion_and_Analysis_s'] for r in res)
raw=sum(r['Raw_Trace_Logical_Bytes'] for r in res);converted=sum(r['Converted_Trace_Logical_Bytes'] for r in res)
resource_rows=[{'Kernel':r['Kernel'],'Trace s':r['Trace_Generation_s'],'Convert/analyze s':r['Conversion_and_Analysis_s'],'Raw GiB':r['Raw_Trace_Logical_Bytes']/2**30,'Converted GiB':r['Converted_Trace_Logical_Bytes']/2**30,'Peak GiB':r['Peak_Trace_Disk_Allocated_Bytes']/2**30,'Retained GiB':r['Retained_Trace_Disk_Allocated_Bytes']/2**30,'ROI overhead x':r['Instrumented_ROI_over_Native_Median'],'Max RSS MiB':r['Max_Child_RSS_KiB']/1024} for r in res]
cache_rows=[{'Kernel':r['Kernel'],'L1D accesses':r['L1D_Accesses'],'L1D misses':r['L1D_Misses'],'L1D miss %':100*r['L1D_Miss_Rate'],'LLC accesses':r['LLC_Accesses'],'LLC misses':r['LLC_Misses'],'LLC miss %':100*r['LLC_Miss_Rate']} for r in tr]
reuse_rows=[{'Kernel':r['Kernel'],'Mean distance':r['Mean_Reuse_Distance'],'Median distance':r['Median_Reuse_Distance'],'Data cold %':100*r['Data_Cold_Reference_Fraction'],'Unique data lines':r['Unique_Cache_Lines'],'Loads+stores/FLOP':r['Load_Store_per_FLOP']} for r in tr]
noise=[{'Kernel':r['Kernel'],'Minimum ms':float(aux[name]['native']['min_ms']),'Median ms':r['Runtime_ms'],'Maximum ms':float(aux[name]['native']['max_ms']),'Range/median %':float(aux[name]['native']['relative_range_percent'])} for r,(name,*_) in zip(main,[('gemm128',),('gemm512',),('gemv',),('attention128',),('attention512',)])]
opcode_rows=[{'Kernel':m['Kernel'],**{op:aux[name]['opcodes'].get(op,0) for op in ['vfmadd213ps','vmulps','vaddss','vshufps']}} for m,name in zip(main,['gemm128','gemm512','gemv','attention128','attention512'])]
phase_trace=list(csv.DictReader((OUT/'attention_phase_trace_characterization.csv').open()))
for row in phase_trace:
    for key in list(row):
        if key not in ['Kernel','Phase']: row[key]=float(row[key]) if ('ms' in key or 'Rate' in key) else int(row[key])
env=(OUT/'environment.txt').read_text()
os_name=re.search(r'^PRETTY_NAME="(.*)"',env,re.M).group(1)
kernel_line=re.search(r'\$ uname -a\n([^\n]+)',env).group(1)
git_version=re.search(r'^git version (.+)$',env,re.M).group(1)
failures=[json.loads(l) for l in (OUT/'commands.jsonl').read_text().splitlines() if json.loads(l)['returncode']!=0]
report=f'''# DynamoRIO MP1 characterization report

All values below come from execution on **gnr2** in this project. Native workload times never come from DynamoRIO. Complete raw benchmark output, trace-tool output, command arguments, environment variables and GNU time records are in `logs/` and `commands.jsonl`. The five complete problem sizes were retained; no instruction sampling, trace filtering, shortened GEMM, or substituted workload was used.

## 1. Environment and existing MP1

The existing prototype is `mp1/`: `CMakeLists.txt`, `cpu/main.cpp`, `cpu/gemm.cpp`, `cpu/gemv.cpp`, `cpu/attention.cpp`, `include/tinyllama_config.h`, `include/metrics.h`, `cpu/roofline.cpp`, and `scripts/generate_roofline.py`. Prior CSVs/plots are under `mp1/results/`; prior batch commands include `scripts/run_cpu.sh` and `scripts/run_thread_scaling.sh`. Their data were inspected and preserved. The TinyLlama snapshot is `reference/tinyllama_1.1b_chat_v1.0_config.json` (hidden size 2048, intermediate size 5632, head dimension 64).

This machine exposes an Intel Xeon 6761P, 64 physical cores / 128 logical CPUs and roughly 1.3 TiB RAM. It is **not the 12-vCPU VM described in the old README**. CPU frequency bounds exposed by sysfs/lscpu are 0.8–3.9 GHz; the idle snapshot is not a measured operating frequency during a benchmark. The cache hierarchy is 48 KiB L1D (12-way), 64 KiB L1I (16-way), 2 MiB L2 (16-way) per core and a shared 336 MiB LLC (16-way). Cache lines are 64 bytes. The native compiler is GCC/G++ 13.3.0 and CMake is 3.28.3. OS: **{os_name}**; Git **{git_version}**. Kernel inventory: `{kernel_line}`. Full hostname, topology, memory and frequency records: [environment.txt](environment.txt). Binary and original source hashes are in `binary_source_sha256.txt`.

The starting filesystem had about 20 GB free, despite abundant RAM. This is a material constraint for student traces.

**Correctness gate:** the existing build passed all four CTest tests before profiling. A separate CPU-native Release build (`mp1/build-dynamorio-native`) also passed all four tests, including kernel references, 1/2/4/8-thread execution checks, CSV output and automatic Roofline checks. Each native benchmark and each ROI driver executes the existing independent small-shape reference suite at eight threads. The ROI driver additionally compares every full-size output element to its uninstrumented warm-up output: all five had maximum absolute difference 0. Native and traced checksums agree. Full-size comparison validates tracing integrity; it does not replace the independent small-shape reference tests.

## 2. DynamoRIO version and build

Official repository: https://github.com/DynamoRIO/dynamorio.git, already cloned by the user into `dynamorio/`. No second clone was made. Exact commit: **f50545b9fe31c536787de9f1e714fba99be7f810**, branch `master`; clone reflog: **2026-09-09 17:00:22 -0500**. `drrun -version` reports **11.91.20705, build 0**. The repository's README, current local documentation/source and [official build instructions](https://dynamorio.org/page_building.html) were inspected.

The source remains unmodified. Required `elfutils` and `zlib` submodules were initialized at their pinned revisions. The optional `libipt` submodule was not needed; Intel PT conversion tools were not built. Release, `DEBUG=OFF`, out-of-tree `dynamorio/build`, 8 build jobs, no documentation/tests/sample builds. No sudo and no system package installation were used.

Two build integration problems were resolved locally: CMake found existing LZ4 in `/usr/local/lib`, but the linker needed `-L/usr/local/lib` and an rpath; upstream's exported Release-to-RelWithDebInfo mapping did not match this Release source build, so **our helper project's** CMake overrides that imported-target mapping. Missing-submodule, link, and mapping failures are retained in `logs/`. Neither change patches DynamoRIO source.

The built executables include `bin64/drrun` and `clients/bin64/drmemtrace_launcher`. `-t drmemtrace` and `-t drcachesim` identify tool configurations; the latter is a compatibility entry for the analysis frontend. `basic_counts`, `opcode_mix`, `reuse_distance`, and `cache_simulator` were all actually executed. This frontend prints usage with exit 1 for `-help`; the error/usage output was retained and supported options were then verified by execution. `view` requires `-exit_after_records`, not `-sim_refs`; a corrected 30-record view succeeded.

## 3. Four categories of numbers

| Category | Numbers and meaning |
|---|---|
| MEASURED | Uninstrumented kernel-region median runtime; analytical work divided by that time gives native GFLOP/s; native register-FMA and streaming bandwidth reference measurements |
| ANALYTICAL | Dimension-based FLOPs/operation-equivalents, specified minimum-byte model, algorithmic AI, Roofline bound and attained percentage |
| TRACE-DERIVED | Dynamic executed instructions, data operands, operand byte widths, unique virtual data lines, decoded opcode counts and per-thread reuse distances |
| SIMULATED | Accesses/misses and local miss rates from the configured cold-start cache simulator; **not measured hardware cache misses** |

Native GFLOP/s is a rate based on analytical work and measured time, not a hardware floating-point counter. Memory-reference bytes are not DRAM bytes. No stall cycles or latency-stall percentages are reported.

## 4. Native measurement methodology

All native kernels and ceilings ran sequentially, outside DynamoRIO, on logical CPUs **0–7**, which are eight distinct physical cores (their SMT siblings are 64–71). Every command uses `--threads 8`, `OMP_PLACES=threads`, `OMP_PROC_BIND=close`, `OMP_WAIT_POLICY=PASSIVE`, and `GOMP_SPINCOUNT=0`. Passive waiting limits trace pollution from spin loops; it also makes wake-up/synchronization overhead visible in short native kernels. Native and traced code uses the same policy.

The original kernel-region timers were retained: **2 warm-ups, 9 timed invocations, median reported**, no cache displacement for the five main rows. Allocation, RNG initialization, correctness, output checking and I/O remain outside the timers. Output zeroing and OpenMP team synchronization are inside. Attention retains QK+scale, softmax, PV and total timers; separate phase medians need not sum exactly to the total median. No production kernel or its normal CLI was rewritten.

{table(noise,['Kernel','Minimum ms','Median ms','Maximum ms','Range/median %'])}

Large timing ranges on short workloads limit fine-grained comparisons. These nine-sample ranges are observed ranges, not confidence intervals. The first valid run for each required workload supplies the main table; no fastest-run selection was used.

## 5. Analytical formulas and model scope

All tensors are row-major FP32. Projection dimensions: N=5632, K=2048; M=S for GEMM and M=1 for GEMV. One unmasked dense attention head has D=64.

- GEMM: `FLOPs = 2*M*N*K`; `Min_Bytes = 4*(M*K + K*N + M*N)`. Read A and B and write C once, excluding output write allocation and repeated loop traffic.
- GEMV: `FLOPs = 2*N*K`; `Min_Bytes = 4*(K + K*N + N)`.
- Attention preserves the existing **materialized, phase-based educational model**: QK FLOPs=`2*S*S*D+S*S`, softmax operation-equivalents=`5*S*S`, PV FLOPs=`2*S*S*D`; total=`4*S*S*D+6*S*S`. QK minimum bytes=`4*(2*S*D+S*S)`, softmax=`8*S*S`, PV=`4*(S*S+2*S*D)`; phase sum=`16*(S*D+S*S)`.
- `Algorithmic_AI = FLOPs/Min_Bytes`; `GFLOP/s = FLOPs/(Runtime_ms*1e6)`.

**Attention caveat:** the phase sum counts required phase inputs/outputs, including score/probability intermediates. It is not a universal fused-attention lower bound and not minimum DRAM traffic. If Q/K/V are read once and O written once, a whole-operation compulsory input/output bound alone is `16*S*D`; intermediates may stay in cache. The report deliberately preserves MP1's existing phase model and makes that distinction explicit. Exponentials and max operations are not literally single hardware FLOPs; Attention throughput/AI are therefore approximate operation-equivalent rates. Roofline conclusions for Attention depend on that convention.

## 6. Trace and cache methodology

A small standalone `mp1/dynamorio/roi_driver.cpp` links the **same unchanged native kernel static library** and DynamoRIO's static drmemtrace client. This follows the documented [start/stop ROI API](https://dynamorio.org/sec_drcachesim_partial.html). Two warm-ups, allocation, random generation, correctness checks and the native expected-output copy happen before `dr_app_setup()`/`dr_app_start()`. One full kernel invocation runs until `dr_app_stop_and_cleanup()`. The measured instrumented-region wall time is recorded only as an overhead diagnostic.

ROI includes the kernel calls, output clears, OpenMP runtime entry/joins/waits, necessary math-library code, the tiny dispatch wrapper and two clock reads. Setup/cleanup transition code near the boundaries can appear. The simulator does not include operating-system syscall bodies. This is close kernel isolation, not a claim that every instruction belongs to an arithmetic loop. The eight threads compute disjoint pieces of one problem. Instruction counts include synchronization and can vary slightly with scheduling; one complete trace per required case was used.

Trace collection uses **LZ4 raw compression**, with no L0 filtering, instruction sampling, size truncation or reduced shapes. Offline conversion uses **ZIP compression** to fit disk space. One converted trace feeds all analyses. `basic_counts:opcode_mix:reuse_distance` runs as a combined analysis with eight jobs; GEMM512 uses `-reuse_skip_dist 2000` and other cases use 500. A separate single-shard 24-million-instruction prefix test compared skip distances 500/2000/8192, yielding identical reported access counts, means, medians and standard deviations; 2000 took 21.08 s versus 39.77 s for 500. This changes the exact algorithm implementation cost, not trace coverage or the distance limit. The prefix is used only for tuning and never substituted into the main tables. Full tuning outputs are in `reuse_tuning.json`; cache simulation runs with `-core_serial` and eight modeled cores; the helper byte/unique-line analyzer uses eight thread shards. Its parallel results were checked exactly against a serial analysis of the GEMV trace.

Cache configuration is in [cache_model.cfg](cache_model.cfg): actual L1I/L1D/L2 capacities and associativities, 64-byte lines, LRU, coherent private caches, no modeled hardware prefetcher, a unified shared **256 MiB / 16-way LLC**, virtual addresses, and dynamic offline scheduling. Actual LLC is 336 MiB; with 16 ways it has a non-power-of-two set count that this simulator cannot represent. The 256 MiB model is a documented approximation, not the machine's measured LLC capacity. All five tensor working sets fit even this smaller LLC. Slice hashing, physical address mapping, real replacement/prefetch policies, OS scheduling and timing effects are not modeled accurately.

**Cold simulation versus warm native timing:** the traced invocation follows native warm-ups, but those warm-ups are outside the trace. Simulation therefore starts with empty caches and includes compulsory misses. These are not steady-state hardware miss rates and cannot be used to claim that the warm native execution incurred the same LLC misses. LLC is unified, so its request stream can include instruction misses and coherence-related effects as well as data demand.

## 7. Main characterization

The exact requested eleven-column CSV is [kernel_characterization.csv](kernel_characterization.csv). Miss-rate columns are fractions in [0,1], not percent.

{table(main,list(main[0]))}

## 8. Roofline ceilings and results

Old results were not reused: previous saved ceiling rows were single-thread values and did not establish a comparable native eight-core ceiling for this run. The measured register-FMA reference is **{d['compute']:.3f} GFLOP/s**. Its loop has sixteen independent AVX-512 accumulators, sixteen FP32 lanes and two operations per FMA: `512 * 20,000,000 * 8 = 81,920,000,000` FLOPs per sample, two warm-ups and nine samples. Disassembly confirms sixteen register-register `vfmadd132ps` instructions in the loop with no loads/stores there. This calibrates a favorable FP32 vector throughput; scalar reductions and exp cannot necessarily attain it.

Native STREAM-like triad and read benchmarks each touch **1.5 GiB**, larger than the shared 336 MiB LLC. Triad models twelve bytes per element (two reads + one write), read models four. The reference uses the larger measured rate: **{d['bandwidth']:.3f} GB/s**. Modeled triad bytes exclude extra write-allocate traffic. These are empirical references for this CPU set and workload, not certified machine-wide saturation or DRAM-counter measurements. Full bandwidth rows are in `native_bandwidth.csv`, and FMA raw output is in `logs/native_register_fma.log`.

Ridge = compute/bandwidth = **{d['ridge']:.4f} FLOP/Byte**. For each row: `Predicted_Roofline = min(compute, Algorithmic_AI*bandwidth)`, and `Roofline_Efficiency_Percent = 100*Measured/Predicted`. The bound label reflects only this simple model.

{table(roof,list(roof[0]))}

![Roofline](roofline.png)

Vertical error bars on workload points span the throughput corresponding to the slowest and fastest of the nine timed samples, not confidence intervals. The bandwidth line is a large-working-set reference. Native main workloads are warm and may be served by caches, so this classic DRAM-style reference does not identify their actual active memory level. Algorithmic minimum bytes do not account for the provided loop order's repeated traffic. A large gap below this optimistic roof is evidence to investigate, not proof of latency stalls.

## 9. Cache and data-access results

{table(cache_rows,list(cache_rows[0]))}

![Simulated cache rates](cache_miss_rates.png)

![Instruction and memory ratios](instruction_memory_mix.png)

Derivations in [trace_characterization.csv](trace_characterization.csv):

- Instructions = fetched instructions + non-fetched repeated instruction iterations, consistent with opcode_mix's total executed instructions. The REP convention matters for output clears.
- Loads/stores count `TRACE_TYPE_READ` / `TRACE_TYPE_WRITE` memory operands, not FP32 elements and not disjoint instruction classes. A read-modify-write operation can contribute both.
- `Load_Store_per_Instruction = (Loads+Stores)/Instructions`; `Load_Store_per_FLOP = (Loads+Stores)/analytical_work`.
- `Dynamic_Memory_Reference_Bytes = sum(memref.data.size)` over reads and writes only. SIMD accesses use the recorded vector width. This is address-trace traffic at the instruction interface, **not DRAM traffic**.
- Unique data lines = cardinality of `floor(address/64)` over every 64-byte line intersected by a data operand; per-thread sets are unioned across the trace. Instruction lines and prefetches are excluded.
- L1D accesses/misses sum the eight L1D caches; rate = total misses / total (hits+misses), not an unweighted average of core percentages.
- LLC rate = LLC misses / LLC (hits+misses), a local miss rate. Its denominator excludes child hits. Cache-line splits can cause more simulator accesses than load/store operands.

The kernels preserve their normal alias checks, loop-control instructions, vector widths and reductions. Full opcode counts are in `opcode_mix.csv`; SIMD FMA presence alone does not establish peak utilization. Selected mnemonic counts (zero means the completed opcode analysis reported no such instruction):

{table(opcode_rows,list(opcode_rows[0]))}

For each projection trace, the observed `vfmadd213ps` count multiplied by 8 FP32 lanes and 2 FLOPs equals its full analytical FLOP count exactly, an additional coverage check. Disassembly (`logs/kernel_disassembly.txt`) shows **256-bit YMM** FMAs in GEMM/GEMV, versus 512-bit ZMM FMAs in the optimized ceiling. The GEMM OpenMP inner loop also reloads an eight-byte loop bound from `0x48(%rsp)` on each vector iteration; the GEMV loop keeps that bound in a register. That directly explains an extra dynamic load and instruction per GEMM vector update, beyond tensor accesses, without attributing a precise stall cost. Attention QK emits vector multiplies followed by scalar `vaddss` and shuffle/reduction instructions; these are visible in total Attention's opcode mix.

## 10. Reuse-distance results

{table(reuse_rows,list(reuse_rows[0]))}

![Reuse-distance CDF](reuse_distance.png)

The built-in tool uses 64-byte tags, per-thread histories, and an unlimited distance list (`distance_limit=0`, no pruned addresses). Mean and median are the tool's exact unbinned statistics over **finite reuses in the combined instruction+data stream**, pooled across thread histograms. Cold references have no finite distance and are excluded. These columns are **not data-only stack distances** and not an average across thread means.

The plot selects the tool's data-reference histogram counts, renormalizes by their sum, and shows a conditional CDF of finite data reuses. Distances still count intervening distinct instruction and data lines. Geometric bins have multiplier 1.5; exported bin endpoints/counts are exact, but within-bin positions are not known. The tool tags the starting address of an operand for reuse analysis; the custom unique-line count also covers split lines, so their unique-line counts need not agree.

`Data_Cold_Reference_Fraction = (tool_data_accesses - sum(data_histogram_counts))/tool_data_accesses`. This counts cold references in per-thread histories, including first observations of a shared line by different threads. It is not a hardware compulsory-miss rate. A small overall reuse median can arise from instruction repetition and spatial reuse within one cache line, including in GEMV; it does not prove weight reuse across tokens.

## 11. GEMM versus GEMV and sequence scaling

GEMV uses a weight once for one input token. Its AI is **{main[2]['Algorithmic_AI_FLOP_per_Byte']:.4f}**, approximately 0.5 FLOP/Byte because the large `K*N` weight matrix dominates minimum bytes. GEMM shares that matrix across M tokens: AI increases to **{main[0]['Algorithmic_AI_FLOP_per_Byte']:.3f}** at S=128 and **{main[1]['Algorithmic_AI_FLOP_per_Byte']:.3f}** at S=512. Work grows exactly 4x while the analytical bytes grow only {main[1]['Min_Bytes']/main[0]['Min_Bytes']:.3f}x. The weight tensor alone is **44 MiB**.

The provided GEMM is unblocked i-k-j: each output row sweeps the entire weight matrix. C rows are 22 KiB and fit in L1D, but a full weight reuse across output rows spans hundreds of thousands of lines and exceeds private L2 capacity. GEMV partitions contiguous output columns; each worker streams its weight slice and repeatedly updates a much smaller output slice. Thus GEMV can show short output/spatial reuse even without cross-token weight reuse.

Native GEMM throughput changes from **{main[0]['GFLOP_per_s']:.3f} to {main[1]['GFLOP_per_s']:.3f} GFLOP/s**, a factor of **{main[1]['GFLOP_per_s']/main[0]['GFLOP_per_s']:.4f}**, despite the higher AI. Runtime grows **{main[1]['Runtime_ms']/main[0]['Runtime_ms']:.4f}x**. Load/store operands per useful FLOP are **{tr[0]['Load_Store_per_FLOP']:.5f}, {tr[1]['Load_Store_per_FLOP']:.5f}, {tr[2]['Load_Store_per_FLOP']:.5f}** for GEMM128, GEMM512 and GEMV. The unblocked implementation does not convert higher algorithmic AI into proportionally fewer dynamic memory operands per FLOP.

Trace-derived reuse means are **{tr[0]['Mean_Reuse_Distance']:.2f}, {tr[1]['Mean_Reuse_Distance']:.2f}, {tr[2]['Mean_Reuse_Distance']:.2f}** lines, respectively, under the combined-stream definition above. Larger GEMM offers more repeated sweeps, but does not shorten the row-to-row weight reuse path. The simulated L1D local miss rate changes from **{100*tr[0]['L1D_Miss_Rate']:.4f}% to {100*tr[1]['L1D_Miss_Rate']:.4f}%**, and LLC from **{100*tr[0]['LLC_Miss_Rate']:.4f}% to {100*tr[1]['LLC_Miss_Rate']:.4f}%**. First-touch amortization can improve LLC miss rates without shortening the reuse path or improving throughput. In comparison with GEMV, GEMM's low cold LLC local miss rate supports stronger inter-row weight reuse at the shared-cache level; GEMV's shorter combined-stream reuse mean mainly reflects instruction, output and spatial reuse. The cache/reuse tables should be read together; no single miss-rate percentage establishes a compute or bandwidth bottleneck.

A supplemental 1024 MiB cache-displacement GEMV native run is saved in `native_gemv_displaced.csv`. The flusher touches one float per line outside timing. It also wakes OpenMP workers before timing, whereas an empty flusher does not. Consequently warm/displaced differences here combine cache state, worker readiness and timing variability; they do not isolate DRAM bandwidth or certify a cold LLC.

## 12. Attention scaling and phases

{table(ph,list(ph[0]))}

Attention S=512 has exactly **16x** the educational work of S=128. Main native total runtime scales **{main[4]['Runtime_ms']/main[3]['Runtime_ms']:.4f}x**, dynamic instructions **{tr[4]['Instructions']/tr[3]['Instructions']:.4f}x**, and data load/store operands **{(tr[4]['Loads']+tr[4]['Stores'])/(tr[3]['Loads']+tr[3]['Stores']):.4f}x**. Fixed runtime/synchronization costs and SIMD/scalar loop structure are plausible contributors to the difference between runtime and work scaling, especially for the smaller case. The observed timing variation limits precise causal attribution.

QK is the largest native phase in both cases: the scalar accumulation order constrains vector reduction efficiency; strict floating-point reduction order is retained. Softmax adds max/reduction/exp work, and PV has a contiguous vectorizable inner loop. The score matrix grows from 64 KiB to 1 MiB; Q/K/V each grow from 32 KiB to 128 KiB. These shapes change the relationship to L1/L2 capacities while remaining well below the modeled LLC. Total-Attention simulated L1D local misses rise from **{100*tr[3]['L1D_Miss_Rate']:.4f}% to {100*tr[4]['L1D_Miss_Rate']:.4f}%**; LLC local rates rise from **{100*tr[3]['LLC_Miss_Rate']:.4f}% to {100*tr[4]['LLC_Miss_Rate']:.4f}%**. The larger private-cache footprint is consistent with the L1D change. The LLC percentages use only requests reaching LLC and should not be read as the fraction of all data references going to DRAM.

Separate phase tracing proved practical: **six additional complete phase ROIs** were collected through `attention_phase_driver.cpp`, linking the same unchanged kernel library. Each phase passed the existing reference tests and an element-by-element comparison to its native phase output (maximum error 0). Allocation, input preparation, two native phase warm-ups and expected-output copies are outside each phase ROI. The softmax input is restored to the original scaled scores before tracing, so the experiment does not apply softmax twice.

{table(phase_trace,list(phase_trace[0]))}

![Native Attention phases](attention_phases.png)

Native phase times in this table are from the original whole-Attention benchmark. Trace counts/cache rates are from separate phase ROIs, with independent empty simulated caches; **they are not a partition of the total trace and their cache misses must not be added to reconstruct the total**. In particular, separate softmax's high cold LLC local miss rate does not include the score residency established by QK in the total-Attention trace. The total trace remains the main characterization row.

The phase traces strengthen the QK diagnosis: QK has the largest instruction count at both sequence lengths and the largest native time. At S=512, PV issues more dynamic data-reference bytes than QK yet completes faster natively, consistent with its efficient contiguous SIMD loop versus QK's scalar reductions. This is evidence of instruction/implementation differences, not a derived stall-cycle decomposition. Phase cache and reuse stdout remains available in the individually named logs.

## 13. Per-workload diagnosis

'''
for i,(name,label,k,s) in enumerate([('gemm128','GEMM S=128','gemm',128),('gemm512','GEMM S=512','gemm',512),('gemv','GEMV','gemv',1),('attention128','Attention S=128','attention',128),('attention512','Attention S=512','attention',512)]):
    m,t,r=main[i],tr[i],roof[i]
    if k=='gemm':
        diagnosis='Likely mixed data-movement/locality and implementation-throughput limitation. Repeated weight sweeps and load/store/loop work prevent the unblocked code from exploiting its optimistic algorithmic AI. The observed low cold-start LLC miss rate supports LLC reuse, but does not make L1/L2-to-LLC movement free. The data do not establish saturation of FP arithmetic units.'
    elif k=='gemv':
        diagnosis='Algorithmically bandwidth/data-movement sensitive, with a large streamed weight matrix and little cross-token reuse. Warm weights may come from LLC; substantial native timing variability and passive OpenMP wake-up costs prevent attributing the entire gap to DRAM bandwidth. Short output/spatial reuse is compatible with weak weight reuse.'
    elif s==128:
        diagnosis='Insufficient useful work relative to thread/phase overhead, plus scalar/reduction work: a mixed limitation. The phase-sum model may label this Bandwidth, but the tensors fit cache and the roof label alone is not evidence of actual DRAM saturation. QK dominates native time; exp/reduction and synchronization matter.'
    else:
        diagnosis='Mixed QK reduction/instruction-throughput and cache/data-movement limitation. More work amortizes fixed phase/thread overhead, but strict QK reductions and scalar softmax still differ greatly from a register-FMA microbenchmark. Cache simulation indicates capacity/locality behavior, not stall time.'
    report+=f'''### {label}

AI **{m['Algorithmic_AI_FLOP_per_Byte']:.4f} FLOP/Byte**; simple Roofline predicts **{r['Predicted_Bound']}** at **{r['Predicted_Roofline_GFLOP_per_s']:.3f} GFLOP/s**. Native **{m['GFLOP_per_s']:.3f} GFLOP/s**, **{r['Roofline_Efficiency_Percent']:.3f}%** of that reference. Trace: **{t['Instructions']:,}** instructions, **{t['Loads']:,}** loads and **{t['Stores']:,}** stores; **{t['Load_Store_per_Instruction']:.4f}** data operands/instruction. Simulated local L1D/LLC miss rates: **{100*t['L1D_Miss_Rate']:.4f}% / {100*t['LLC_Miss_Rate']:.4f}%**. Finite combined-stream reuse mean/median: **{t['Mean_Reuse_Distance']:.2f} / {t['Median_Reuse_Distance']:.0f}** lines; per-thread data cold fraction **{100*t['Data_Cold_Reference_Fraction']:.3f}%**.

{diagnosis}

'''
report+=f'''## 14. Limits and failures

- No hardware performance counters, DRAM byte counters, cycle-accurate execution, IPC, latency-stall share or exact effective benchmark frequency were collected. These metrics are unsupported by this methodology, rather than zero.
- Full ROI traces include user-space runtime/clock/math overhead and can perturb scheduling. Native and instrumented wall times are kept separate.
- Native timings are warm; cache simulation is cold and idealized. Virtual addresses, 256 MiB surrogate LLC, dynamic trace scheduling, omitted prefetch and OS execution limit hardware fidelity.
- FLOPs/bytes are analytical conventions. Attention's exp/max accounting and phase-sum bytes make its roof especially approximate.
- Separate phase traces start with empty simulated caches and different ROI/runtime boundaries; their statistics cannot be summed to reconstruct total Attention. Reuse statistics are per-thread combined-stream finite distances, not data-only or cross-thread reuse distances.
- Build integration failures (including the optional second static client's CMake global initialization/target regeneration) and unsupported help/view flag attempts were repaired without modifying DynamoRIO source. Successful runs, stderr and original failures remain reviewable. No failed run supplies a numeric workload metric.
- The helper parallel analyzer was replay-validated against the serial pilot; counts also match `basic_counts` for every full trace. Simulator accesses can exceed memory operands due to line splitting.

## 15. Trace cost and feasibility for 30 students

{table(resource_rows,list(resource_rows[0]))}

Trace time includes process startup, native warm-ups, attach/detach and output validation. ROI overhead compares just the recorded instrumented interval to the native kernel median; it is not native performance. Analysis time includes raw-to-ZIP conversion plus the combined count/opcode/reuse pass, a cache pass and a byte/unique-line pass. Some independent offline analyses overlapped after native measurements had finished; these are sums of measured per-command costs, not elapsed wall time or isolated single-job throughput. Exact UTC starts are retained in the command log. Offline analysis threads were not CPU-pinned, so these costs remain sensitive to shared-machine scheduling. The per-stage breakdown is in [trace_resources.csv](trace_resources.csv). Cache passes use serial core scheduling; other passes use up to eight worker jobs. RSS is GNU time's maximum child-process RSS, not a sum of simultaneous processes. Disk peak is sampled every 0.5 seconds, so brief transient peaks can be missed.

Five trace generations total **{generation:.2f} s**; conversion/analysis total **{analysis:.2f} s ({analysis/60:.2f} min)**. Generated raw trace data/metadata total **{raw/2**30:.3f} GiB**. Converted records for those five traces total **{converted/2**30:.3f} GiB**; the six optional isolated Attention phase traces are additional. All retained trace directories, including phase traces, occupy **{retained/2**30:.3f} GiB allocated**. The validated raw `.raw.lz4` records were deleted only after conversion and a full successful count/opcode/reuse read; cache and byte-count validation also completed before the report, retaining ZIP records, module maps, encodings and all logs. `cleanup.jsonl` records exact removed paths/sizes. No shortened trace replaces a full workload.

A simple **estimate**, not a 30-user measurement: retaining the five required and six supplemental traces for each of 30 students needs about **{30*retained/2**30:.1f} GiB**, excluding sources/builds, versus roughly 20 GB initially free here. Concurrent in-flight raw+converted data requires still more. One-student timings do not predict queueing or throughput under 30-user contention; thirty eight-job analyses can request 240 CPU workers. A safe disk plan and scheduling are required. The fixed CPU set 0–7 is for this one-user experiment; assigning that same set to all students would cause direct contention. Use scheduled time slots or distinct instructor-assigned CPU sets and measure ceilings for each allowed resource scope. The large GEMM reuse/cache analyses and trace storage dominate the exercise's cost. ZIP metadata records the uncompressed converted trace size in `trace_resources.csv`; GEMM S=512 alone represents **{res[1]['Converted_Trace_Uncompressed_Bytes']/2**30:.3f} GiB of uncompressed trace records**. This is trace-file volume, separate from operand bytes or tensor sizes. Analysis must still decode that stream even when on-disk files compress well.

{peak_sentence}

ROI and lossless compression were actually used. L0 filtering, instruction windows, distance caps, or smaller trace-only matrices were investigated through current documentation but not used in reported full-workload data: they would change counts, cache/reuse state or representativeness. If adopted for students, keep native full-size rows, label trace-only dimensions/sampling, and leave unavailable full-workload metrics NA instead of copying sampled counts into them.

## 16. Student deployment recommendation and reproduction

Keep the native and analytical part of MP1 as the universal student activity. Provide a pinned, prebuilt local DynamoRIO package plus this ROI wrapper; the two build integration fixes should not be a required architecture exercise. Distribute one instructor-produced full trace set or precomputed trace-derived/cache tables with raw tool logs. Let students generate GEMV and small Attention traces, and run large GEMM analysis in scheduled batches or against shared read-only instructor traces. Require explicit separation of measured/analytical/trace-derived/simulated values and the cold-model disclaimer. If every student must generate all five full traces, reserve disk capacity and concurrency limits before rollout.

Reproduce on this same CPU from the repository root, using a fresh destination:

```sh
MP1_DR_RESULTS="$PWD/results/dynamorio_repro" sh mp1/dynamorio/reproduce.sh
```

The full-run script requires **at least 18 GiB free** before starting, including space for the 2 GiB runtime stop guard. At report generation this filesystem had **{free/2**30:.3f} GiB free** while preserving the completed experiment. A second full trace set therefore needs additional available storage; the script refuses to start if that condition is unmet. Regenerating artifacts from existing logs needs no second trace set. Future report regeneration preserves the recorded experiment end time; `--finalize-now` explicitly updates it when completing a new experiment.

Regenerate the submitted CSVs, plots and report from retained logs, without re-tracing:

```sh
mp1/venv/bin/python mp1/dynamorio/summarize_phases.py
mp1/venv/bin/python mp1/dynamorio/summarize.py
mp1/venv/bin/python mp1/dynamorio/write_report.py
mp1/venv/bin/python mp1/dynamorio/validate_results.py
```

`reproduce.sh` contains all configure/build/test/native/ROI/analysis commands; `commands.jsonl` records the exact commands actually executed, including absolute trace directories, environment and exit status. The individual commands and resulting artifacts were run and inspected in this experiment; the fresh-directory wrapper is assembled from those executed stages, not a claim of an additional second full experiment. MP1's existing local `venv` supplies Matplotlib; no packages were installed for this work.

Start UTC: **{start.isoformat()}**. Report finalized UTC: **{end.isoformat()}**. Total elapsed session/experiment time including inspection, builds, analysis and report preparation: **{elapsed:.2f} s ({elapsed/60:.2f} min)**. This differs from the sum of native or profiling times. Output file validation is recorded separately in `validation.txt`.

This verdict applies to the proposed **30-student, all-five-full-traces workflow on the currently available filesystem**. The measurements and prototype work, but the observed storage/analysis costs and required cache-model caveats make unrestricted deployment impractical. Shared traces, prebuilt tools and staged analysis can turn it into a usable assignment; they require an explicit deployment change and resource allocation before rollout.

**NOT RECOMMENDED FOR STUDENT USE**
'''
(OUT/'DYNAMORIO_MP1_REPORT.md').write_text(report)
(OUT/'experiment_summary.json').write_text(json.dumps({'start_utc':start.isoformat(),'end_utc':end.isoformat(),'total_elapsed_seconds':elapsed,'required_trace_generation_seconds':generation,'required_conversion_analysis_seconds':analysis,'raw_trace_logical_bytes_generated':raw,'converted_trace_logical_bytes':converted,'retained_trace_allocated_bytes':retained,'observed_global_trace_allocated_bytes':snapshot['all_trace_allocated_bytes'] if snapshot else None,'free_disk_bytes_at_report':free,'verdict':'NOT RECOMMENDED FOR STUDENT USE','failed_logged_commands':[{'name':r['name'],'returncode':r['returncode']} for r in failures]},indent=2)+'\n')
print(f'Report saved. Elapsed {elapsed:.2f}s; retained trace disk {retained} bytes.')
