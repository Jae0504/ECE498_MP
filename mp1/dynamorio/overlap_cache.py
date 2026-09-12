"""Start cache/memory passes when another process has completed all trace conversion.
Native measurements must already be complete. Never read partially written ZIP files.
"""
import gzip, json, subprocess, sys, time, zipfile
from run_logged import ROOT,OUT
name,kernel,s=sys.argv[1:4]
start=time.monotonic()
events=OUT/'commands.jsonl'
initial_rows=[json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
initial_stats=next((r for r in reversed(initial_rows) if r['name']=='stats_'+name),None)
initial_start=initial_stats['start_utc'] if initial_stats else None
while True:
    events=OUT/'commands.jsonl'
    if events.exists():
        rows=[json.loads(line) for line in events.read_text().splitlines()]
        latest=next((r for r in reversed(rows) if r['name']=='stats_'+name),None)
        if latest and latest['returncode']!=0 and latest['start_utc']!=initial_start:raise SystemExit('Stats/conversion failed; not analyzing an incomplete trace.')
    traces=list((OUT/'traces'/name).glob('drmemtrace.*.dir'))
    try:
        assert len(traces)==1
        folder=traces[0]/'trace'
        shards=list(folder.glob('*.trace.zip'));assert len(shards)==8
        for path in [*shards,folder/'cpu_schedule.bin.zip']:
            with zipfile.ZipFile(path) as z:assert z.namelist()
        with gzip.open(folder/'serial_schedule.bin.gz','rb') as f:
            while f.read(1024*1024):pass
        break
    except (AssertionError,OSError,EOFError,zipfile.BadZipFile):
        if time.monotonic()-start>600:raise SystemExit('No complete converted trace within 600s')
        time.sleep(1)
print('Eight complete ZIP shards and closed schedule files verified; starting independent offline passes.',flush=True)
subprocess.run([sys.executable,str(ROOT/'mp1/dynamorio/run_trace.py'),name,kernel,s,'cache','memory'],check=True)
