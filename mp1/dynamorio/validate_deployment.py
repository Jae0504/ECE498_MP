"""Validate real outputs and report differences from the recorded reference, without fitting data."""
import argparse,csv,json,math,re
from pathlib import Path
from PIL import Image
from deployment import HERE,WORK
p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);p.add_argument('--complete',action='store_true');args=p.parse_args();out=args.output.resolve()
if not __debug__:raise SystemExit('Validation requires Python assertions.')
def rows(path):return list(csv.DictReader(path.open()))
fields=['Kernel','Runtime_ms','GFLOP_per_s','FLOPs','Min_Bytes','Algorithmic_AI_FLOP_per_Byte','Instructions','Loads','Stores','L1D_Miss_Rate','LLC_Miss_Rate']
main=rows(out/'kernel_characterization.csv');trace=rows(out/'trace_characterization.csv');roofs=rows(out/'roofline.csv');ceil=rows(out/'ceilings.csv')[0]
assert list(main[0])==fields
labels=[w[1] for w in WORK];seen=[r['Kernel'] for r in main]
assert seen==[label for label in labels if label in seen] and len(set(seen))==len(seen)
if args.complete:assert seen==labels
assert [r['Kernel'] for r in trace]==seen==[r['Kernel'] for r in roofs]
events={r['name']:r for r in (json.loads(l) for l in (out/'commands.jsonl').read_text().splitlines())}
ref=HERE/'reference/gnr2';baseline={r['Kernel']:r for r in rows(ref/'kernel_characterization.csv')};old_trace={r['Kernel']:r for r in rows(ref/'trace_characterization.csv')};native=rows(out/'native_kernel_results.csv');old_native={r['kernel']+r['sequence']:r for r in rows(ref/'native_kernel_results.csv')}
checks=[];comparison=[]
compute=float(ceil['compute_ceiling_GFLOP_per_s']);bw=float(ceil['bandwidth_ceiling_GB_per_s'])
assert compute>0 and bw>0 and math.isclose(compute/bw,float(ceil['ridge_point_FLOP_per_Byte']),rel_tol=1e-10)
for m,t,r in zip(main,trace,roofs):
    label=m['Kernel'];name,_,kernel,seq=next(w for w in WORK if w[1]==label)
    for field in fields[1:]:assert math.isfinite(float(m[field])),(label,field)
    for field in ['FLOPs','Min_Bytes','Algorithmic_AI_FLOP_per_Byte']:
        assert float(m[field])==float(baseline[label][field]),(label,field)
    assert math.isclose(float(m['GFLOP_per_s']),float(m['FLOPs'])/(float(m['Runtime_ms'])*1e6),rel_tol=1e-10)
    for cache in ['L1D','LLC']:
        rate=float(m[cache+'_Miss_Rate']);assert 0<=rate<=1
        assert math.isclose(rate,float(t[cache+'_Misses'])/float(t[cache+'_Accesses']),rel_tol=1e-10)
    pred=min(compute,float(m['Algorithmic_AI_FLOP_per_Byte'])*bw)
    assert math.isclose(pred,float(r['Predicted_Roofline_GFLOP_per_s']),rel_tol=1e-10)
    assert math.isclose(100*float(m['GFLOP_per_s'])/pred,float(r['Roofline_Efficiency_Percent']),rel_tol=1e-10)
    assert r['Predicted_Bound']==('Compute' if float(m['Algorithmic_AI_FLOP_per_Byte'])*bw>=compute else 'Bandwidth')
    n=next(n for n in native if n['kernel']==kernel and int(n['sequence'])==seq)
    assert n['warmups']=='2' and n['iterations']=='9'
    assert math.isclose(float(n['checksum']),float(old_native[kernel+str(seq)]['checksum']),rel_tol=1e-9,abs_tol=1e-9)
    for stage in ['trace','stats','cache','memory']:assert events[stage+'_'+name]['returncode']==0
    log=(out/'logs'/f'trace_{name}.log').read_text();assert 'trace_full_output_max_error=0' in log and 'All CPU correctness tests passed (threads=8).' in log
    stats=(out/'logs'/f'stats_{name}.log').read_text();assert int(re.search(r'^\s*(\d+) total skipped memref markers',stats,re.M)[1])==0
    path,=(out/'traces'/name).glob('drmemtrace.*.dir');assert len(list((path/'trace').glob('*.trace.zip')))==8
    mem=json.loads((out/'logs'/f'memory_{name}.log').read_text())
    assert int(t['Loads'])==mem['Loads'] and int(t['Stores'])==mem['Stores']
    assert int(t['Dynamic_Memory_Reference_Bytes'])==mem['Dynamic_Memory_Reference_Bytes'] and int(t['Unique_Cache_Lines'])==mem['Unique_Cache_Lines']
    checks.append(f'PASS {label}: analytical equality, reference checksum, full trace, counts, cache ratios, roofline formulas')
    now={**m,**t};old={**baseline[label],**old_trace[label]}
    for field in now:
        if field=='Kernel':continue
        a,b=float(old[field]),float(now[field]);category='TRACE-DERIVED'
        if field in ['Runtime_ms','GFLOP_per_s']:category='MEASURED'
        elif field in ['FLOPs','Min_Bytes','Algorithmic_AI_FLOP_per_Byte']:category='ANALYTICAL'
        elif field.startswith(('L1D_','LLC_')):category='SIMULATED'
        comparison.append({'Kernel':label,'Metric':field,'Category':category,'Reference':a,'Current':b,'Delta':b-a,'Relative_Change_Percent':100*(b-a)/a if a else 'NA','Miss_Rate_Change_Percentage_Points':100*(b-a) if field.endswith('_Miss_Rate') else 'NA'})
for stage in ['native_bandwidth','native_register_fma']:assert events[stage]['returncode']==0
fma=next(csv.DictReader((out/'logs/native_register_fma.log').open()));reference_fma=rows(ref/'register_fma.csv')[0]
for field in ['threads','repeats','warmups','iterations','FLOPs','checksum']:
    assert float(fma[field])==float(reference_fma[field]),'FMA work/output mismatch: '+field
assert math.isclose(compute,float(fma['FLOPs'])/(float(fma['median_ms'])*1e6),rel_tol=1e-10)
for b in rows(out/'native_bandwidth.csv'):
    assert float(b['work'])==1610612736 and b['warmups']=='2' and b['iterations']=='9'
    assert math.isclose(float(b['rate']),float(b['work'])/(float(b['median_ms'])*1e6),rel_tol=1e-10)
checks.append('PASS native FMA work/checksum and both streaming-bandwidth formulas.')

for filename in ['environment.txt','DYNAMORIO_MP1_REPORT.md','roofline.png','cache_miss_rates.png','instruction_memory_mix.png','reuse_distance.png']:
    path=out/filename;assert path.is_file() and path.stat().st_size>0
    if path.suffix=='.png':
        with Image.open(path) as im:im.verify()
phase_count=0
for label in seen:
    if not label.startswith('Attention'):continue
    seq=int(label.split('=')[1])
    for phase in ['qk','softmax','pv']:
        name=f'attention{seq}_{phase}'
        for stage in ['trace','stats','cache','memory']:assert events[stage+'_'+name]['returncode']==0
        assert 'trace_full_output_max_error=0' in (out/'logs'/f'trace_{name}.log').read_text()
        phase_count+=1
if phase_count:
    assert len(rows(out/'attention_phase_trace_characterization.csv'))==phase_count
    with Image.open(out/'attention_phases.png') as im:im.verify()
def compare_table(filename,current_rows,reference_rows,identity):
    records=[];old_by_key={tuple(r[k] for k in identity):r for r in reference_rows}
    for r in current_rows:
        key=tuple(r[k] for k in identity);old=old_by_key[key]
        for field in r:
            if field in identity:continue
            a,b=float(old[field]),float(r[field])
            records.append({**{k:r[k] for k in identity},'Metric':field,'Reference':a,'Current':b,'Delta':b-a,'Relative_Change_Percent':100*(b-a)/a if a else 'NA'})
    with (out/filename).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
compare_table('ceiling_reference_comparison.csv',[ceil],rows(ref/'ceilings.csv'),[])
if phase_count:
    compare_table('phase_reference_comparison.csv',rows(out/'attention_phase_trace_characterization.csv'),rows(ref/'attention_phase_trace_characterization.csv'),['Kernel','Phase'])
checks.append(f'PASS {len(main)} complete workload(s), {phase_count} Attention phase(s), required CSVs and figures.')
checks.append('PASS verifies measurement integrity and analytical/output equality. Native timing and trace/cache values are compared numerically, not forced to equal the reference.')
with (out/'reference_comparison.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(comparison[0]));w.writeheader();w.writerows(comparison)
(out/'validation.txt').write_text('\n'.join(checks)+'\n')
(out/'verification.json').write_text(json.dumps({'integrity_passed':True,'analytical_and_output_equality':True,'complete_workloads':seen,'phase_count':phase_count,'reference_numeric_equality_required':False,'comparison':'reference_comparison.csv'},indent=2)+'\n')
print('\n'.join(checks))
