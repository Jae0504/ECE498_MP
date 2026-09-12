"""Supplemental full-size, individual phase ROIs; never substitute them for total traces."""
import subprocess,sys
from run_logged import ROOT,OUT,run,CPUSET
sizes=list(map(int,sys.argv[1:])) or [128,512]
if any(s not in [128,512] for s in sizes):raise SystemExit('Only S=128 and S=512 are supported.')
for s in sizes:
    for phase in ['qk','softmax','pv']:
        name=f'attention{s}_{phase}';base=OUT/'traces'/name
        if base.exists() and any(base.iterdir()):raise SystemExit('Refusing duplicate phase trace: '+name)
        base.mkdir(parents=True,exist_ok=True)
        options=f"-stderr_mask 0xc -client_lib ';;-offline -raw_compress lz4 -outdir {base}'"
        run('trace_'+name,['taskset','-c',CPUSET,ROOT/'mp1/dynamorio/build/attention_phase_driver',phase,str(s),'8'],env={'DYNAMORIO_OPTIONS':options},monitor=base)
        subprocess.run([sys.executable,str(ROOT/'mp1/dynamorio/run_trace.py'),name,'attention',str(s),'stats','cache','memory'],check=True)
        subprocess.run([sys.executable,str(ROOT/'mp1/dynamorio/cleanup_raw.py'),name],check=True)
