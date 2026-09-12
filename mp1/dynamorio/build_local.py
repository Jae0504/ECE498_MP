"""Build a fresh clone without relying on another checkout, venv or /usr/local LZ4."""
import argparse,datetime,fcntl,hashlib,json,os,platform,shlex,shutil,subprocess,sys,time
from deployment import ROOT,HERE,DR_COMMIT,LZ4_COMMIT,VENV,BUILD_STATE,cpu_model,source_hashes
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--jobs',type=int,default=8);args=p.parse_args()
if args.jobs<1:raise SystemExit('--jobs must be positive')
if sys.version_info<(3,9):raise SystemExit('build.sh requires Python 3.9 or newer.')
if platform.system()!='Linux' or platform.machine()!='x86_64':raise SystemExit('This measured profile requires Linux x86_64.')
flags=set(next(l.split(':',1)[1].split() for l in (ROOT/'/proc/cpuinfo').read_text().splitlines() if l.startswith('flags')))
if not {'avx512f','fma'}<=flags:raise SystemExit('The original register-FMA ceiling requires AVX-512F/FMA; this CPU needs a different documented ceiling implementation.')
for tool in ['git','cmake','gcc','g++','make','taskset','objdump']:
    if not shutil.which(tool):raise SystemExit(f'Required server tool missing: {tool}. No system packages are installed by this script.')
logs=ROOT/'results/dynamorio_build'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S');logs.mkdir(parents=True)
lock_path=HERE/'.build.lock';lock=lock_path.open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit('Another local build is running.')
if BUILD_STATE.exists():BUILD_STATE.unlink() # A failed build must not leave a valid gate.
def run(name,cmd,env=None,allow_failure=False):
    cmd=list(map(str,cmd));print('$ '+shlex.join(cmd),flush=True);start=time.monotonic()
    with (logs/(name+'.log')).open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
    row={'name':name,'command':cmd,'elapsed_s':time.monotonic()-start,'exit_code':r.returncode}
    if env is not None:row['compiler_environment']={key:env.get(key,'') for key in ['CFLAGS','CXXFLAGS']}
    with (logs/'commands.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if r.returncode:
        if allow_failure:return r.returncode
        raise SystemExit(f'{name} failed (exit {r.returncode}); see {logs/(name+".log")}')
    print(f'{name}: PASS ({row["elapsed_s"]:.2f}s)',flush=True)
    return 0
def configure(name,cmd,env=None):
    code=run(name,cmd,env,allow_failure=True)
    # Switching an existing cache from cc/c++ to gcc/g++ makes CMake configure
    # twice internally and discard explicit -D options on the second pass.
    # Reapply those options after the compiler choice has settled.
    if 'You have changed variables that require your cache to be deleted.' in (logs/(name+'.log')).read_text():
        run(name+'_restore_options',cmd,env)
    elif code:
        raise SystemExit(f'{name} failed (exit {code}); see {logs/(name+".log")}')
entry=subprocess.check_output(['git','ls-files','--stage','dynamorio'],cwd=ROOT,text=True).split()
if entry[:2]!=['160000',DR_COMMIT]:raise SystemExit('Checkout must contain the pinned dynamorio gitlink and .gitmodules; use the complete repository commit.')
if (ROOT/'dynamorio/.git').exists():
    for diff in [['diff','--quiet'],['diff','--cached','--quiet']]:
        if subprocess.run(['git','-C',str(ROOT/'dynamorio'),*diff]).returncode:raise SystemExit('DynamoRIO has local source edits; preserve them before building the pinned submodule.')
run('submodule',['git','submodule','update','--init','dynamorio'])
assert subprocess.check_output(['git','-C','dynamorio','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==DR_COMMIT
run('nested_submodules',['git','-C','dynamorio','submodule','update','--init','third_party/elfutils','third_party/zlib'])
if not (VENV/'bin/python').exists():run('venv',[sys.executable,'-m','venv',VENV])
run('python_version',[VENV/'bin/python','-c',"import sys; print(sys.version, flush=True); sys.exit(0 if sys.version_info >= (3, 9) else 'The project venv requires Python 3.9 or newer.')"])
run('python_dependencies',[VENV/'bin/python','-m','pip','install','--disable-pip-version-check','-r',HERE/'requirements.lock'])
run('python_dependencies_check',[VENV/'bin/python','-m','pip','check'])
run('python_packages',[VENV/'bin/python','-m','pip','list','--format=json'])
# Build LZ4 locally with the custom-allocation API required by static ROI clients.
deps=HERE/'build-deps';deps.mkdir(exist_ok=True);lz4=deps/'lz4'
if not (lz4/'.git').exists():run('clone_lz4',['git','clone','--branch','v1.10.0','--depth','1','https://github.com/lz4/lz4.git',lz4])
if subprocess.check_output(['git','-C',str(lz4),'rev-parse','HEAD'],text=True).strip()!=LZ4_COMMIT:raise SystemExit('Local LZ4 revision differs from the pinned version; refusing to reset it.')
run('build_lz4',['make','-C',lz4/'lib','-j',args.jobs,'CPPFLAGS=-DXXH_NAMESPACE=LZ4_ -DLZ4F_PUBLISH_STATIC_FUNCTIONS','lib-release'])
libdir=lz4/'lib';link=f'-L{libdir} -Wl,-rpath,{libdir}'
# Upstream overwrites CMAKE_[C,CXX]_FLAGS but explicitly retains environment flags.
# Keep the local headers ahead of any system/CPATH headers for both languages.
dr_env=os.environ.copy()
for flag in ['CFLAGS','CXXFLAGS']:
    dr_env[flag]=(f'-I{shlex.quote(str(libdir))} '+dr_env.get(flag,'')).strip()

common=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_C_COMPILER='+shutil.which('gcc'),'-DCMAKE_CXX_COMPILER='+shutil.which('g++')]
configure('configure_dynamorio',['cmake','-S','dynamorio','-B','dynamorio/build',*common,'-DDEBUG=OFF','-DBUILD_DOCS=OFF','-DBUILD_TESTS=OFF','-DBUILD_SAMPLES=OFF',f'-Dliblz4={libdir}/liblz4.so','-U','HAVE_LZ4_CUSTOM_MEM',f'-DCMAKE_SHARED_LINKER_FLAGS={link}',f'-DCMAKE_EXE_LINKER_FLAGS={link}'],env=dr_env)
# Preserve the real feature-test diagnostics, including cached compiler failures.
for diagnostic in ['CMakeCache.txt','CMakeFiles/CMakeError.log','CMakeFiles/CMakeConfigureLog.yaml']:
    source=ROOT/'dynamorio/build'/diagnostic
    if source.exists():shutil.copyfile(source,logs/source.name)

cache=(ROOT/'dynamorio/build/CMakeCache.txt').read_text()
if f'liblz4:FILEPATH={libdir}/liblz4.so' not in cache and f'liblz4:UNINITIALIZED={libdir}/liblz4.so' not in cache:
    raise SystemExit('DynamoRIO configuration lost the pinned local LZ4 path; inspect configure_dynamorio logs.')
if 'HAVE_LZ4_CUSTOM_MEM:INTERNAL=1' not in cache:raise SystemExit(f'BUILD STOPPED: the LZ4 custom-allocator compile/link test failed. DynamoRIO and ROI helpers have not finished building. See {logs}/CMakeError.log or {logs}/CMakeConfigureLog.yaml and {logs}/configure_dynamorio.log.')
if 'ZLIB_LIBRARY_RELEASE:FILEPATH=' not in cache or 'ZLIB_LIBRARY_RELEASE:FILEPATH=ZLIB_LIBRARY_RELEASE-NOTFOUND' in cache:raise SystemExit('The server needs zlib development files for ZIP trace support. No package was installed system-wide.')
run('build_dynamorio',['cmake','--build','dynamorio/build','-j',args.jobs],env=dr_env)
configure('configure_mp1',['cmake','-S','mp1','-B','mp1/build-dynamorio-native',*common,'-DCPU_NATIVE=ON',f'-DPython3_EXECUTABLE={VENV}/bin/python'])
run('build_mp1',['cmake','--build','mp1/build-dynamorio-native','-j',args.jobs])
run('correctness',['ctest','--test-dir','mp1/build-dynamorio-native','--output-on-failure'])
if '100% tests passed, 0 tests failed out of 4' not in (logs/'correctness.log').read_text():raise SystemExit('Expected all four original MP1 CTest tests, including plotting.')
configure('configure_helpers',['cmake','-S','mp1/dynamorio','-B','mp1/dynamorio/build',*common,f'-DDynamoRIO_DIR={ROOT}/dynamorio/build/cmake',f'-DCMAKE_C_FLAGS=-I{libdir}',f'-DCMAKE_CXX_FLAGS=-I{libdir}',f'-DCMAKE_EXE_LINKER_FLAGS={link}'])
run('build_helpers',['cmake','--build','mp1/dynamorio/build','-j',args.jobs])
run('drrun_version',['dynamorio/build/bin64/drrun','-version'])
run('drrun_smoke',['dynamorio/build/bin64/drrun','--','/bin/echo','DYNAMORIO_OK'])
binaries=['mp1/build-dynamorio-native/cpu_bench','mp1/build-dynamorio-native/roofline_bench','mp1/build-dynamorio-native/libcpu_kernels.a',*[f'mp1/dynamorio/build/{n}' for n in ['roi_driver','attention_phase_driver','trace_metrics','fma_ceiling']],'dynamorio/build/bin64/drrun','dynamorio/build/clients/bin64/drmemtrace_launcher']
state={'root':str(ROOT),'built_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cpu_model':cpu_model(),'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],'dynamorio_commit':DR_COMMIT,'lz4_commit':LZ4_COMMIT,'sources':source_hashes(),'binaries':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in binaries},'correctness_tests_passed':4,'logs':str(logs)}
BUILD_STATE.write_text(json.dumps(state,indent=2)+'\n')
print(f'Build PASS. Logs: {logs}')
