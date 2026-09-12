"""Shared, recorded settings for the deployable eight-core MP1 experiment."""
import hashlib,json,os,platform,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=ROOT/'mp1/dynamorio'
DR_COMMIT='f50545b9fe31c536787de9f1e714fba99be7f810'
LZ4_COMMIT='ebb370ca83af193212df4dcbadcc5d87bc0de2f0'
VENV=HERE/'.venv'
BUILD_STATE=HERE/'build/build_state.json'
WORK=[('gemm128','GEMM S=128','gemm',128),('gemm512','GEMM S=512','gemm',512),('gemv','GEMV','gemv',1),('attention128','Attention S=128','attention',128),('attention512','Attention S=512','attention',512)]

def cpu_model():
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):return line.split(':',1)[1].strip()
    return platform.processor()

def source_hashes():
    paths=[ROOT/'mp1/CMakeLists.txt',HERE/'CMakeLists.txt',*sorted((ROOT/'mp1/cpu').glob('*.cpp')),*sorted((ROOT/'mp1/include').glob('*')),*sorted((ROOT/'mp1/tests').glob('*.py')),*sorted(HERE.glob('*.cpp'))]
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}

def selected_cpus():
    allowed=os.sched_getaffinity(0)
    def key(cpu):
        p=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        return tuple((p/n).read_text().strip() for n in ['physical_package_id','core_id'])
    requested=os.environ.get('MP1_CPUS')
    if requested:
        cpus=[]
        for item in requested.split(','):
            ends=item.split('-');cpus.extend(range(int(ends[0]),int(ends[-1])+1))
    else:
        cpus=[];seen=set()
        for cpu in sorted(allowed):
            if key(cpu) not in seen:cpus.append(cpu);seen.add(key(cpu))
            if len(cpus)==8:break
    if len(cpus)!=8 or len(set(cpus))!=8 or not set(cpus)<=allowed or len({key(c) for c in cpus})!=8:
        raise SystemExit('MP1_CPUS must select exactly eight allowed CPUs on distinct physical cores. Ask the server scheduler/instructor for an eight-core allocation.')
    return ','.join(map(str,cpus))

def require_build():
    if not BUILD_STATE.exists():raise SystemExit('Build first: sh mp1/dynamorio/build.sh')
    state=json.loads(BUILD_STATE.read_text())
    if state['root']!=str(ROOT) or state['sources']!=source_hashes() or state['cpu_model']!=cpu_model():
        raise SystemExit('Checkout location, sources or CPU changed. Rebuild with sh mp1/dynamorio/build.sh')
    if subprocess.check_output(['git','-C',str(ROOT/'dynamorio'),'rev-parse','HEAD'],text=True).strip()!=DR_COMMIT:
        raise SystemExit('DynamoRIO differs from the pinned submodule. Run build.sh.')
    for rel,digest in state['binaries'].items():
        p=ROOT/rel
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise SystemExit(f'Build artifact changed or missing: {p}; run build.sh')
    return state
