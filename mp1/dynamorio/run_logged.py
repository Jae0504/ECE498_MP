"""Record real commands, elapsed time, resource usage and sampled disk footprint."""
import datetime, json, os, pathlib, re, shlex, shutil, signal, subprocess, time
ROOT=pathlib.Path(__file__).resolve().parents[2]
OUT=pathlib.Path(os.environ.get('MP1_DR_RESULTS', ROOT/'results/dynamorio')).resolve()
CPUSET=os.environ.get('MP1_CPUS','0-7')
ENV={**os.environ,'OMP_PLACES':'threads','OMP_PROC_BIND':'close','OMP_WAIT_POLICY':'PASSIVE','GOMP_SPINCOUNT':'0','LC_ALL':'C'}

def disk(path):
    logical=allocated=0
    if path and pathlib.Path(path).exists():
        for p in pathlib.Path(path).rglob('*'):
            if p.is_file():
                s=p.stat();logical+=s.st_size;allocated+=s.st_blocks*512
    return logical,allocated

def stop_owned_group(process):
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
    # GNU time may exit before its child; stop surviving members of our own group.
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    process.wait()

def run(name, command, env=None, monitor=None, timeout=3600):
    (OUT/'logs').mkdir(parents=True,exist_ok=True)
    log=OUT/'logs'/f'{name}.log';timelog=OUT/'logs'/f'{name}.time'
    row={'name':name,'command':list(map(str,command)),'cwd':str(ROOT),'environment':{k:ENV[k] for k in ['OMP_PLACES','OMP_PROC_BIND','OMP_WAIT_POLICY','GOMP_SPINCOUNT','LC_ALL']},'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if env: row['environment'].update(env)
    print('$ '+shlex.join(row['command']),flush=True)
    start=time.monotonic();peak=0;global_peak=0;last_progress=start
    with log.open('w') as f:
        p=subprocess.Popen(['/usr/bin/time','-v','-o',str(timelog),*row['command']],cwd=ROOT,env={**ENV,**(env or {})},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        interrupted=None
        try:
            while p.poll() is None:
                if monitor:
                    peak=max(peak,disk(monitor)[1])
                    total=disk(OUT/'traces')[1]
                    if total>global_peak:
                        global_peak=total
                        snapshot=OUT/'peak_trace_snapshot.json'
                        previous=json.loads(snapshot.read_text()) if snapshot.exists() else {}
                        if total>previous.get('all_trace_allocated_bytes',0):
                            snapshot.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'all_trace_allocated_bytes':total,'scope':'Global allocated trace footprint sampled every 0.5 seconds.'},indent=2)+'\n')
                if time.monotonic()-last_progress>=30:
                    print(f'{name}: still running, {time.monotonic()-start:.0f}s elapsed',flush=True);last_progress=time.monotonic()
                if time.monotonic()-start>timeout or (monitor and shutil.disk_usage(OUT).free<2*1024**3):
                    row['stopped_reason']='time budget or less than 2 GiB free disk'
                    break
                time.sleep(0.5)
        except BaseException as exc:
            interrupted=exc;row['stopped_reason']='Interrupted: '+type(exc).__name__
        finally:
            if interrupted is not None or 'stopped_reason' in row:stop_owned_group(p)
            else:p.wait()
    elapsed_observed=time.monotonic()-start
    recorded=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([0-9:.]+)',timelog.read_text())
    elapsed=0.0
    if recorded:
        for field in recorded.group(1).split(':'): elapsed=60*elapsed+float(field)
    else: elapsed=elapsed_observed
    row.update(elapsed_seconds=elapsed,orchestration_seconds=elapsed_observed,returncode=p.returncode,log=str(log),time_log=str(timelog),peak_sampled_disk_bytes=max(peak,disk(monitor)[1]) if monitor else None)
    if monitor:row['final_logical_bytes'],row['final_allocated_bytes']=disk(monitor)
    with (OUT/'commands.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(f'{name}: exit={p.returncode}, elapsed={row["elapsed_seconds"]:.3f}s',flush=True)
    if interrupted is not None:raise interrupted
    if p.returncode: raise RuntimeError(f'{name} failed; see {log}')
    return row

if __name__=='__main__':
    import sys
    run(sys.argv[1],sys.argv[2:])
