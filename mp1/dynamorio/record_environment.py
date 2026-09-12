import datetime,json,os,subprocess
from pathlib import Path
from run_logged import OUT,ROOT
OUT.mkdir(parents=True,exist_ok=True)
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
commands=[['hostname'],['cat','/etc/os-release'],['uname','-a'],['lscpu'],['lscpu','-C'],['lscpu','-e=CPU,CORE,SOCKET,NODE,ONLINE'],['cat','/proc/meminfo'],['gcc','--version'],['g++','--version'],['cmake','--version'],['git','--version'],['df','-B1',str(OUT)],['git','rev-parse','HEAD'],['git','-C','dynamorio','remote','-v'],['git','-C','dynamorio','log','-1','--format=fuller'],['git','-C','dynamorio','branch','--show-current'],['git','-C','dynamorio','reflog','--date=iso','-1'],['git','-C','dynamorio','submodule','status'],['cat','/proc/self/cgroup']]
with (OUT/'environment.txt').open('w') as f:
    f.write('Recorded UTC: '+start+'\nAllowed CPUs: '+str(sorted(os.sched_getaffinity(0)))+'\n')
    for cmd in commands:
        r=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        f.write('\n$ '+' '.join(cmd)+'\n'+r.stdout+f'\nexit={r.returncode}\n')
    for root in ['/sys/devices/system/cpu/cpu0/cpufreq','/sys/devices/system/cpu/cpu0/cache']:
        for p in sorted(Path(root).rglob('*')):
            if p.is_file():
                try:f.write(str(p)+': '+p.read_text().strip()+'\n')
                except (OSError,UnicodeError):pass
(OUT/'experiment_start.json').write_text(json.dumps({'start_utc':start},indent=2)+'\n')
