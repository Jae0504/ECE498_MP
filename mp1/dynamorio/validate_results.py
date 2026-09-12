"""Validate recorded outputs, arithmetic identities and artifacts; never synthesize data."""
import csv, json, math, re
from pathlib import Path
from PIL import Image
from run_logged import OUT
required=['environment.txt','kernel_characterization.csv','trace_characterization.csv','ceilings.csv','roofline.csv','roofline.png','cache_miss_rates.png','instruction_memory_mix.png','reuse_distance.png','DYNAMORIO_MP1_REPORT.md']
messages=[]
for name in required:
    p=OUT/name;assert p.is_file() and p.stat().st_size>0,name
    messages.append(f'PASS nonempty {name} ({p.stat().st_size} bytes)')
    if p.suffix=='.png':
        with Image.open(p) as im:im.verify()
expected=['Kernel','Runtime_ms','GFLOP_per_s','FLOPs','Min_Bytes','Algorithmic_AI_FLOP_per_Byte','Instructions','Loads','Stores','L1D_Miss_Rate','LLC_Miss_Rate']
with (OUT/'kernel_characterization.csv').open() as f:
    reader=csv.DictReader(f);assert reader.fieldnames==expected;main=list(reader)
labels=['GEMM S=128','GEMM S=512','GEMV','Attention S=128','Attention S=512']
assert [r['Kernel'] for r in main]==labels
tr=list(csv.DictReader((OUT/'trace_characterization.csv').open()))
roof=list(csv.DictReader((OUT/'roofline.csv').open()))
ceil=next(csv.DictReader((OUT/'ceilings.csv').open()))
compute=float(ceil['compute_ceiling_GFLOP_per_s']);bandwidth=float(ceil['bandwidth_ceiling_GB_per_s'])
assert math.isclose(float(ceil['ridge_point_FLOP_per_Byte']),compute/bandwidth,rel_tol=1e-10)
for m,t,r in zip(main,tr,roof):
    for col in expected[1:]:assert math.isfinite(float(m[col])),(m['Kernel'],col)
    flops=float(m['FLOPs']);ms=float(m['Runtime_ms']);ai=flops/float(m['Min_Bytes']);g=flops/(ms*1e6)
    assert ms>0 and int(m['Instructions'])>0 and int(m['Loads'])>0 and int(m['Stores'])>0
    assert math.isclose(ai,float(m['Algorithmic_AI_FLOP_per_Byte']),rel_tol=1e-10)
    assert math.isclose(g,float(m['GFLOP_per_s']),rel_tol=1e-10)
    for level in ['L1D','LLC']:
        rate=float(m[level+'_Miss_Rate']);assert 0<=rate<=1
        assert math.isclose(rate,float(t[level+'_Misses'])/float(t[level+'_Accesses']),rel_tol=1e-10)
    pred=min(compute,ai*bandwidth)
    assert math.isclose(pred,float(r['Predicted_Roofline_GFLOP_per_s']),rel_tol=1e-10)
    assert math.isclose(100*g/pred,float(r['Roofline_Efficiency_Percent']),rel_tol=1e-10)
    assert r['Predicted_Bound']==('Compute' if ai*bandwidth>=compute else 'Bandwidth')
    assert int(t['Dynamic_Memory_Reference_Bytes'])>=int(t['Loads'])+int(t['Stores'])
    messages.append('PASS formulas and cross-file consistency: '+m['Kernel'])
commands=[json.loads(l) for l in (OUT/'commands.jsonl').read_text().splitlines()]
byname={r['name']:r for r in commands}
for name in ['gemm128','gemm512','gemv','attention128','attention512']:
    for stage in ['trace','stats','cache','memory']:assert byname[stage+'_'+name]['returncode']==0
    trace,=(OUT/'traces'/name).glob('drmemtrace.*.dir')
    assert len(list((trace/'trace').glob('*.trace.zip')))==8
    log=(OUT/'logs'/f'trace_{name}.log').read_text()
    assert 'trace_full_output_max_error=0' in log
    assert 'All CPU correctness tests passed (threads=8).' in log
    messages.append('PASS eight trace shards and full-output correctness: '+name)
phase_rows=list(csv.DictReader((OUT/'attention_phase_trace_characterization.csv').open()))
assert len(phase_rows)==6
for row in phase_rows:
    assert int(row['Instructions'])>0 and int(row['Dynamic_Memory_Reference_Bytes'])>0
    assert 0<=float(row['L1D_Miss_Rate'])<=1 and 0<=float(row['LLC_Miss_Rate'])<=1
for size in [128,512]:
    for phase in ['qk','softmax','pv']:
        name=f'attention{size}_{phase}'
        for stage in ['trace','stats','cache','memory']:assert byname[stage+'_'+name]['returncode']==0
        log=(OUT/'logs'/f'trace_{name}.log').read_text()
        assert 'trace_full_output_max_error=0' in log
with Image.open(OUT/'attention_phases.png') as im:im.verify()
messages.append('PASS six supplemental Attention phase traces, phase CSV and plot.')
report=(OUT/'DYNAMORIO_MP1_REPORT.md').read_text()
for word in ['MEASURED','ANALYTICAL','TRACE-DERIVED','SIMULATED','NOT RECOMMENDED FOR STUDENT USE']:
    assert word in report
messages.append('PASS all required artifacts, rows, identities and category labels.')
(OUT/'validation.txt').write_text('\n'.join(messages)+'\n')
print('\n'.join(messages))
