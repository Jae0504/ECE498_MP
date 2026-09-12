"""One independent workload, sharing measured ceilings and accumulated results."""
import argparse,csv,datetime,fcntl,json,math,os,shutil,subprocess,sys
from pathlib import Path
from deployment import ROOT,HERE,BUILD_STATE,WORK,selected_cpus,require_build
from selected_metrics import METRICS,measure_selected
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('case',choices=['ceilings',*[w[0] for w in WORK]])
p.add_argument('output',nargs='?',type=Path,default=Path(os.environ.get('MP1_DR_RESULTS',ROOT/'results/dynamorio_student')))
p.add_argument('--resume',action='store_true',help='Reuse explicitly completed stages; never append a second native series.')
p.add_argument('--metrics',nargs='+',choices=METRICS,help='Run only these analyses, reusing one trace. Default: the complete report workflow.')
p.add_argument('--phase',choices=['full','qk','softmax','pv'],default='full',help='Attention ROI for --metrics (default: the full kernel).')
args=p.parse_args();out=args.output.resolve()
if args.metrics and args.case=='ceilings':p.error('--metrics requires a workload, not ceilings.')
if args.phase!='full' and (not args.case.startswith('attention') or not args.metrics):p.error('--phase requires an attention workload with --metrics.')
if args.phase!='full' and 'native' in args.metrics:p.error('For native phase times, use --metrics native --phase full; its CSV includes qk_ms, softmax_ms and pv_ms.')
if not __debug__:raise SystemExit('Run without PYTHONOPTIMIZE: validation requires assertions.')
if out==ROOT/'results/dynamorio':raise SystemExit('Preserve the original experiment: choose a new output directory.')
if any(c.isspace() for c in str(ROOT)+str(out)):raise SystemExit('Use project/output paths without whitespace; upstream tracing/linker options require this profile restriction.')
os.environ['MP1_DR_RESULTS']=str(out);os.environ['MP1_CPUS']=selected_cpus()
state=require_build()
from run_logged import run,CPUSET,ENV
out.mkdir(parents=True,exist_ok=True)
# Validate tracing on GEMV before a larger independently requested workload.
if not args.metrics and args.case not in ['ceilings','gemv'] and not (out/'gemv.completed.json').exists():
    print('Running the complete GEMV validation first in the same result set.',flush=True)
    subprocess.run(['sh',str(HERE/'measure_gemv.sh'),str(out),'--resume'],check=True)
lock=(out/'.measurement.lock').open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Another measurement owns this result directory. Use separate directories and assigned CPU sets.')
metadata_path=out/'experiment_metadata.json'
configuration={'cpu_model':state['cpu_model'],'cpu_set':CPUSET,'threads':8,'warmups':2,'iterations':9,'cache_model':'gnr2-reference: L1D 48KiB/12-way, L2 2MiB/16-way, LLC 256MiB/16-way, cold, virtual addresses, no prefetch','cache_model_sha256':__import__('hashlib').sha256((HERE/'cache_model.cfg').read_bytes()).hexdigest(),'compiler':state['compiler'],'dynamorio_commit':state['dynamorio_commit'],'lz4_commit':state['lz4_commit'],'sources':state['sources'],'binaries':state['binaries'],'omp':{k:ENV[k] for k in ['OMP_PLACES','OMP_PROC_BIND','OMP_WAIT_POLICY','GOMP_SPINCOUNT']}}
if metadata_path.exists():
    metadata=json.loads(metadata_path.read_text())
    if metadata['configuration']!=configuration:raise SystemExit('CPU set, compiler, cache model or binaries changed within this result set. Use a fresh output directory.')
else:
    existing=[q for q in out.iterdir() if q.name!='.measurement.lock']
    if existing:raise SystemExit('This directory is not a managed measurement set. Choose a fresh output directory.')
    metadata={'host':__import__('socket').gethostname(),'root':str(ROOT),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'configuration':configuration}
    metadata_path.write_text(json.dumps(metadata,indent=2)+'\n')
    shutil.copyfile(HERE/'cache_model.cfg',out/'cache_model.cfg')
    subprocess.run([sys.executable,str(HERE/'record_environment.py')],check=True)
    run('drrun_version',[ROOT/'dynamorio/build/bin64/drrun','-version'])
    run('drrun_echo',[ROOT/'dynamorio/build/bin64/drrun','--','/bin/echo','DYNAMORIO_OK'])

def events():
    f=out/'commands.jsonl'
    return {r['name']:r for r in (json.loads(l) for l in f.read_text().splitlines())} if f.exists() else {}
def success(name):return events().get(name,{}).get('returncode')==0

if args.metrics:
    name,label,kernel,sequence=next(w for w in WORK if w[0]==args.case)
    measure_selected(out,name,kernel,sequence,args.metrics,args.phase)
    sys.exit(0)

def ceilings():
    done=out/'ceilings.completed.json'
    if done.exists():return
    bench=ROOT/'mp1/build-dynamorio-native/roofline_bench'
    if not success('native_bandwidth'):
        if (out/'native_bandwidth.csv').exists():raise SystemExit('Incomplete ceiling CSV exists; preserve it and use a fresh output directory.')
        run('native_bandwidth',['taskset','-c',CPUSET,bench,'--ceiling','bandwidth','--stream-elements','134217728','--read-elements','402653184','--threads','8','--warmup','2','--iterations','9','--no-plot','--output',out/'native_bandwidth.csv'])
    if not success('native_register_fma'):
        run('native_register_fma',['taskset','-c',CPUSET,HERE/'build/fma_ceiling','8','20000000'])
    bw=max(float(r['rate']) for r in csv.DictReader((out/'native_bandwidth.csv').open()))
    fma=next(csv.DictReader((out/'logs/native_register_fma.log').open()));compute=float(fma['GFLOP_per_s'])
    reference_fma=next(csv.DictReader((HERE/'reference/gnr2/register_fma.csv').open()))
    for field in ['threads','repeats','warmups','iterations','FLOPs','checksum']:
        assert float(fma[field])==float(reference_fma[field]),'FMA work/output mismatch: '+field
    assert math.isclose(compute,float(fma['FLOPs'])/(float(fma['median_ms'])*1e6),rel_tol=1e-10)

    assert bw>0 and compute>0
    row={'compute_ceiling_GFLOP_per_s':compute,'bandwidth_ceiling_GB_per_s':bw,'ridge_point_FLOP_per_Byte':compute/bw}
    with (out/'ceilings.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    done.write_text(json.dumps({'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'measured':row},indent=2)+'\n')

if (out/(args.case+'.completed.json')).exists() and not args.resume:
    raise SystemExit('This case already completed. Choose a new output directory for a new measurement, or pass --resume to reuse it explicitly.')
required_gib={'gemm128':6,'gemm512':18}.get(args.case,2.25)
if not (args.resume and (out/(args.case+'.completed.json')).exists()) and shutil.disk_usage(out).free<required_gib*2**30:
    raise SystemExit(f'{args.case} requires at least {required_gib} GiB free for full losslessly compressed traces and the 2 GiB stop guard.')
if args.case!='ceilings' and not args.resume and not (out/(args.case+'.completed.json')).exists():
    prior=events()
    if any(stage+'_'+args.case in prior for stage in ['native','trace','stats','cache','memory']):
        raise SystemExit('An incomplete prior run exists. Use --resume explicitly or choose a fresh output directory.')
ceilings()
if args.case=='ceilings':
    print('Native ceilings recorded: '+str(out/'ceilings.csv'));sys.exit(0)
if (out/(args.case+'.completed.json')).exists() and args.resume:
    subprocess.run([sys.executable,str(HERE/'validate_deployment.py'),str(out)],check=True)
    print('Explicit --resume: preserved completed '+args.case);sys.exit(0)
name,label,kernel,sequence=next(w for w in WORK if w[0]==args.case)
bench=ROOT/'mp1/build-dynamorio-native/cpu_bench'
if not success('native_'+name):
    if (out/'native_kernel_results.csv').exists():
        rows=list(csv.DictReader((out/'native_kernel_results.csv').open()))
        if any(r['kernel']==kernel and int(r['sequence'])==sequence for r in rows):raise SystemExit('An incomplete native row exists. Refusing to mix or silently replace measurement series.')
    run('native_'+name,['taskset','-c',CPUSET,bench,'--kernel',kernel,'--seq',sequence,'--threads','8','--warmup','2','--iterations','9','--flush-cache-mib','0','--no-plot','--output',out/'native_kernel_results.csv'])
if kernel=='gemv' and not success('native_gemv_displaced'):
    run('native_gemv_displaced',['taskset','-c',CPUSET,bench,'--kernel','gemv','--threads','8','--warmup','2','--iterations','9','--flush-cache-mib','1024','--no-plot','--output',out/'native_gemv_displaced.csv'])
if not success('trace_'+name):
    subprocess.run([sys.executable,str(HERE/'run_trace.py'),name,kernel,str(sequence),'collect'],check=True)
if kernel=='gemm' and not all(success(stage+'_'+name) for stage in ['stats','cache','memory']):
    # Conversion must close all ZIP shards/schedules before a second reader starts.
    stats=None
    if not success('stats_'+name):
        stats=subprocess.Popen([sys.executable,str(HERE/'run_trace.py'),name,kernel,str(sequence),'stats'])
    cache_run=subprocess.run([sys.executable,str(HERE/'overlap_cache.py'),name,kernel,str(sequence)])
    stats_code=stats.wait() if stats else 0
    if cache_run.returncode or stats_code:raise SystemExit('A full trace analysis failed; inspect logs before resuming. No shortened result was accepted.')
else:
    for stage in ['stats','cache','memory']:
        if not success(stage+'_'+name):
            subprocess.run([sys.executable,str(HERE/'run_trace.py'),name,kernel,str(sequence),stage],check=True)
# Delete raw records only once every full analysis has succeeded. ZIP traces stay.
subprocess.run([sys.executable,str(HERE/'cleanup_raw.py'),name],check=True)
if kernel=='attention':
    phase_names=[f'attention{sequence}_{phase}' for phase in ['qk','softmax','pv']]
    if not all(success('memory_'+phase) for phase in phase_names):
        if any((out/'traces'/phase).exists() for phase in phase_names):raise SystemExit('Partial phase run exists; use run_trace.py to finish it before retrying. No existing trace was overwritten.')
        subprocess.run([sys.executable,str(HERE/'run_attention_phases.py'),str(sequence)],check=True)
    subprocess.run([sys.executable,str(HERE/'summarize_phases.py'),'--available'],check=True)
subprocess.run([sys.executable,str(HERE/'summarize.py'),'--available'],check=True)
subprocess.run([sys.executable,str(HERE/'write_run_report.py'),str(out),'--finalize-now'],check=True)
subprocess.run([sys.executable,str(HERE/'validate_deployment.py'),str(out)],check=True)
(out/(name+'.completed.json')).write_text(json.dumps({'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'label':label,'full_workload':True},indent=2)+'\n')
print(f'{label}: full measurement PASS. Results: {out}',flush=True)
