"""Trace once per workload; reuse its compressed offline trace for each analysis."""
import argparse, fcntl, json
from run_logged import ROOT, OUT, run, CPUSET
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('name')
p.add_argument('kernel',choices=['gemm','gemv','attention'])
p.add_argument('sequence',type=int)
p.add_argument('stages',nargs='*',help='collect, counts, opcodes, reuse, stats, cache, memory')
p.add_argument('--phase',choices=['full','qk','softmax','pv'],default='full')
p.add_argument('--force',action='store_true')
args=p.parse_args()
name,kernel,seq=args.name,args.kernel,str(args.sequence)
force=args.force
stages=args.stages or ['collect','stats','cache','memory']
if any(s not in ['collect','counts','opcodes','reuse','stats','cache','memory'] for s in stages):
    p.error('Unknown trace stage.')
if args.phase!='full' and kernel!='attention':p.error('--phase requires attention')
base=OUT/'traces'/name
launcher=ROOT/'dynamorio/build/clients/bin64/drmemtrace_launcher'
def analyze(stage,command):
    # Independent readers may overlap, but a given stage is executed only once.
    locks=OUT/'.stage_locks';locks.mkdir(exist_ok=True)
    with (locks/(stage+'_'+name+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        log=OUT/'commands.jsonl'
        records=[json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        old=next((r for r in reversed(records) if r['name']==stage+'_'+name),None)
        if not force and old and old['returncode']==0 and old['command']==list(map(str,command)) and (OUT/'logs'/f'{stage}_{name}.log').is_file():
            print('Reused completed '+stage+'_'+name,flush=True);return
        run(stage+'_'+name,command,monitor=base,timeout=3600)

if 'collect' in stages:
    if base.exists() and any(base.iterdir()):raise SystemExit('Trace directory already exists; refusing duplicate trace.')
    base.mkdir(parents=True,exist_ok=True)
    options=f"-stderr_mask 0xc -client_lib ';;-offline -raw_compress lz4 -outdir {base}'"
    driver=([ROOT/'mp1/dynamorio/build/roi_driver',kernel,seq,'8'] if args.phase=='full'
            else [ROOT/'mp1/dynamorio/build/attention_phase_driver',args.phase,seq,'8'])
    run('trace_'+name,['taskset','-c',CPUSET,*driver],env={'DYNAMORIO_OPTIONS':options},monitor=base,timeout=1800)
paths=list(base.glob('drmemtrace.*.dir'))
if len(paths)!=1: raise SystemExit(f'Expected exactly one trace in {base}: {paths}')
trace=paths[0]
if 'counts' in stages:
    # Also completes raw-to-ZIP conversion before any other selected reader.
    analyze('counts',[launcher,'-indir',trace,'-jobs','8','-compress','zip','-tool','basic_counts'])
if 'opcodes' in stages:
    analyze('opcodes',[launcher,'-indir',trace,'-jobs','8','-compress','zip','-tool','opcode_mix'])
if 'reuse' in stages:
    analyze('reuse',[launcher,'-indir',trace,'-jobs','8','-compress','zip','-tool','reuse_distance','-reuse_distance_histogram','-reuse_histogram_bin_multiplier','1.5','-reuse_skip_dist',('2000' if name=='gemm512' else '500'),'-report_top','10'])
if 'stats' in stages:
    analyze('stats',[launcher,'-indir',trace,'-jobs','8','-compress','zip','-tool','basic_counts:opcode_mix:reuse_distance','-reuse_distance_histogram','-reuse_histogram_bin_multiplier','1.5','-reuse_skip_dist',('2000' if name=='gemm512' else '500'),'-report_top','10'])
if 'cache' in stages:
    analyze('cache',[launcher,'-indir',trace,'-jobs','8','-tool','cache_simulator','-config_file',OUT/'cache_model.cfg','-cores','8','-core_serial'])
if 'memory' in stages:
    analyze('memory',[ROOT/'mp1/dynamorio/build/trace_metrics',trace/'trace'])
