"""Run each public shell entry point once, with one shared native ceiling set."""
import argparse,os,subprocess,sys
from pathlib import Path
from deployment import ROOT,HERE
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',nargs='?',type=Path,default=Path(os.environ.get('MP1_DR_RESULTS',ROOT/'results/dynamorio_student')));p.add_argument('--resume',action='store_true');args=p.parse_args()
for script in ['measure_ceilings.sh','measure_gemv.sh','measure_gemm_s128.sh','measure_gemm_s512.sh','measure_attention_s128.sh','measure_attention_s512.sh']:
    cmd=['sh',str(HERE/script),str(args.output.resolve())]
    if args.resume:cmd.append('--resume')
    print('Running '+script,flush=True);subprocess.run(cmd,check=True)
subprocess.run([sys.executable,str(HERE/'validate_deployment.py'),str(args.output.resolve()),'--complete'],check=True)
print('All five shell measurement entry points and six Attention phase ROIs completed.')
