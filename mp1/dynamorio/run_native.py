from run_logged import ROOT, OUT, run, CPUSET
OUT.mkdir(parents=True,exist_ok=True)
bench=ROOT/'mp1/build-dynamorio-native/cpu_bench'
ceil=ROOT/'mp1/build-dynamorio-native/roofline_bench'
if (OUT/'native_kernel_results.csv').exists(): raise SystemExit('Refusing to append a second measurement series; use a fresh MP1_DR_RESULTS directory.')
run('native_correctness_8', ['taskset','-c',CPUSET,bench,'--test','--threads','8'])
for name,k,s in [('gemm128','gemm',128),('gemm512','gemm',512),('gemv','gemv',1),('attention128','attention',128),('attention512','attention',512)]:
    run('native_'+name,['taskset','-c',CPUSET,bench,'--kernel',k,'--seq',str(s),'--threads','8','--warmup','2','--iterations','9','--flush-cache-mib','0','--no-plot','--output',OUT/'native_kernel_results.csv'])
run('native_bandwidth',['taskset','-c',CPUSET,ceil,'--ceiling','bandwidth','--stream-elements','134217728','--read-elements','402653184','--threads','8','--warmup','2','--iterations','9','--no-plot','--output',OUT/'native_bandwidth.csv'])
run('native_register_fma',['taskset','-c',CPUSET,ROOT/'mp1/dynamorio/build/fma_ceiling','8','20000000'])
# Supplemental native data-movement experiment; the requested main GEMV remains warm.
run('native_gemv_displaced',['taskset','-c',CPUSET,bench,'--kernel','gemv','--threads','8','--warmup','2','--iterations','9','--flush-cache-mib','1024','--no-plot','--output',OUT/'native_gemv_displaced.csv'])
