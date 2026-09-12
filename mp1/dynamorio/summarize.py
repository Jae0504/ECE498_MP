"""Derive characterization tables and figures exclusively from saved experiment outputs."""
import argparse, csv, datetime, json, math, re, statistics, sys, zipfile
from pathlib import Path
from run_logged import OUT, ROOT, disk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORK=[('gemm128','GEMM S=128','gemm',128),('gemm512','GEMM S=512','gemm',512),('gemv','GEMV','gemv',1),('attention128','Attention S=128','attention',128),('attention512','Attention S=512','attention',512)]
def read(name): return (OUT/'logs'/f'{name}.log').read_text()
def value(pattern,text,cast=int):
    m=re.search(pattern,text,re.M)
    if not m: raise ValueError('Missing measured field: '+pattern)
    return cast(m.group(1).replace(',',''))
def csvwrite(name,rows,fields=None):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def table(rows,columns):
    def fmt(v):
        if isinstance(v,float):return f'{v:,.4g}'
        if isinstance(v,int):return f'{v:,}'
        return str(v)
    return '| '+' | '.join(columns)+' |\n|'+'|'.join(['---']*len(columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(r[c]) for c in columns)+' |' for r in rows)+'\n'
def parse_cache(text):
    caches={}
    for m in re.finditer(r'^\s*(L1D\d+|LLC) \([^\n]*\) stats:\n((?:[ \t]+[^\n]*\n)+)',text,re.M):
        caches[m[1]]=(value(r'^\s*Hits:\s*([\d,]+)',m[2]),value(r'^\s*Misses:\s*([\d,]+)',m[2]))
    assert len([k for k in caches if k.startswith('L1D')])==8,caches
    hits=sum(v[0] for k,v in caches.items() if k.startswith('L1D'))
    misses=sum(v[1] for k,v in caches.items() if k.startswith('L1D'))
    lh,lm=caches['LLC']
    return {'L1D_Accesses':hits+misses,'L1D_Misses':misses,'L1D_Miss_Rate':misses/(hits+misses),'LLC_Accesses':lh+lm,'LLC_Misses':lm,'LLC_Miss_Rate':lm/(lh+lm)}

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--available',action='store_true',help='Summarize only completed workloads; never substitute partial traces.')
args=parser.parse_args()
native=list(csv.DictReader((OUT/'native_kernel_results.csv').open()))
commands=[json.loads(line) for line in (OUT/'commands.jsonl').read_text().splitlines()]
cmd_by_name={r['name']:r for r in commands}
if args.available:
    WORK=[w for w in WORK if all(cmd_by_name.get(stage+'_'+w[0],{}).get('returncode')==0 for stage in ['trace','stats','cache','memory'])]
else:assert len(native)==5
assert WORK,'No fully analyzed workloads are available.'
assert len({(r['kernel'],r['sequence']) for r in native})==len(native),'Duplicate native measurement rows.'
bw=list(csv.DictReader((OUT/'native_bandwidth.csv').open()))
bandwidth=max(float(r['rate']) for r in bw)
fma=list(csv.DictReader(read('native_register_fma').splitlines()))[0]
compute=float(fma['GFLOP_per_s']);ridge=compute/bandwidth
csvwrite('ceilings.csv',[{'compute_ceiling_GFLOP_per_s':compute,'bandwidth_ceiling_GB_per_s':bandwidth,'ridge_point_FLOP_per_Byte':ridge}])
main=[];extended=[];roofs=[];phases=[];histograms={};opcodes=[];resources=[];aux={}
commands=[json.loads(line) for line in (OUT/'commands.jsonl').read_text().splitlines()]
cmd_by_name={r['name']:r for r in commands}
for name,label,kernel,s in WORK:
    n=next(r for r in native if r['kernel']==kernel and int(r['sequence'])==s)
    runtime=float(n['median_ms']);flops=float(n['approx_flops']);minimum=float(n['estimated_minimum_bytes'])
    if kernel=='gemm': expected_flops=2*s*5632*2048; expected_bytes=4*(s*2048+2048*5632+s*5632)
    elif kernel=='gemv': expected_flops=2*5632*2048;expected_bytes=4*(2048+2048*5632+5632)
    else:expected_flops=4*s*s*64+6*s*s;expected_bytes=16*(s*64+s*s)
    assert flops==expected_flops and minimum==expected_bytes
    ai=flops/minimum;gflops=flops/(runtime*1e6)
    assert math.isclose(gflops,float(n['measured_gflops']),rel_tol=1e-10)
    raw=read('stats_'+name)
    fetched=value(r'^\s*([\d,]+) total \(fetched\) instructions',raw)
    unfetched=value(r'^\s*([\d,]+) total non-fetched instructions',raw)
    instructions=fetched+unfetched
    loads=value(r'^\s*([\d,]+) total data loads',raw)
    stores=value(r'^\s*([\d,]+) total data stores',raw)
    assert value(r'^\s*([\d,]+) total threads',raw)==8
    assert value(r'^\s*([\d,]+) total skipped memref markers',raw)==0
    assert value(r'^\s*([\d,]+) : total executed instructions',raw)==instructions
    mem=json.loads(read('memory_'+name))
    assert (mem['Loads'],mem['Stores'])==(loads,stores)
    trace_log=read('trace_'+name)
    assert value(r'^trace_full_output_max_error=(\S+)',trace_log,float)==0
    assert math.isclose(value(r'^trace_checksum=(\S+)',trace_log,float),float(n['checksum']),rel_tol=1e-9,abs_tol=1e-9)
    cache=parse_cache(read('cache_'+name))
    reuse=raw.split('Reuse distance tool aggregated results:',1)[1].split('Reuse distance threshold',1)[0]
    mean=value(r'^Reuse distance mean: (\S+)',reuse,float)
    median=value(r'^Reuse distance median: (\S+)',reuse,float)
    assert value(r'^Distance limit: (\d+)',reuse)==0
    assert value(r'^Pruned addresses: (\d+)',reuse)==0
    hist=[]
    for m in re.finditer(r'^\s*(\d+) -\s*(\d+)\s+(\d+)\s+[\d.]+%\s+[\d.]+%\s*:\s*(\d+)',reuse,re.M):
        hist.append(tuple(map(int,m.groups())))
    assert hist
    histograms[label]=hist
    data_refs=value(r'^Data accesses: (\d+)',reuse)
    assert data_refs==loads+stores
    reused_data=sum(h[3] for h in hist)
    cold_fraction=(data_refs-reused_data)/data_refs
    ext={'Kernel':label,'Instructions':instructions,'Loads':loads,'Stores':stores,'Load_Store_per_Instruction':(loads+stores)/instructions,'Dynamic_Memory_Reference_Bytes':mem['Dynamic_Memory_Reference_Bytes'],'Unique_Cache_Lines':mem['Unique_Cache_Lines'],**cache,'Mean_Reuse_Distance':mean,'Median_Reuse_Distance':median,'Data_Cold_Reference_Fraction':cold_fraction,'Load_Store_per_FLOP':(loads+stores)/flops}
    extended.append(ext)
    main.append({'Kernel':label,'Runtime_ms':runtime,'GFLOP_per_s':gflops,'FLOPs':int(flops),'Min_Bytes':int(minimum),'Algorithmic_AI_FLOP_per_Byte':ai,'Instructions':instructions,'Loads':loads,'Stores':stores,'L1D_Miss_Rate':cache['L1D_Miss_Rate'],'LLC_Miss_Rate':cache['LLC_Miss_Rate']})
    prediction=min(compute,ai*bandwidth)
    roofs.append({'Kernel':label,'Algorithmic_AI':ai,'Measured_GFLOP_per_s':gflops,'Predicted_Roofline_GFLOP_per_s':prediction,'Roofline_Efficiency_Percent':100*gflops/prediction,'Predicted_Bound':'Compute' if ai*bandwidth>=compute else 'Bandwidth'})
    if kernel=='attention':
        phases.append({'Kernel':label,'QK_ms':float(n['qk_ms']),'Softmax_ms':float(n['softmax_ms']),'PV_ms':float(n['pv_ms']),'Total_ms':runtime,'Min_Total_ms':float(n['min_ms']),'Max_Total_ms':float(n['max_ms'])})
    optext=raw.split('Opcode mix tool results:',1)[1].split('Reuse distance tool aggregated results:',1)[0]
    optext=re.split(r'\n\s*\d+ : sets of categories',optext)[0]
    # Printed mnemonics can be shared by distinct internal opcodes; aggregate them.
    ops={}
    for m in re.finditer(r'^\s*(\d+) :\s+(.+)$',optext,re.M):
        op=m[2].strip()
        if op=='total executed instructions':continue
        if op.startswith('Category') or op.startswith('category'):continue
        ops[op]=ops.get(op,0)+int(m[1])
    assert sum(ops.values())==instructions,(label,sum(ops.values()),instructions)
    if kernel in ['gemm','gemv']: assert ops['vfmadd213ps']*16==int(flops)
    aux[name]={'native':n,'fetched':fetched,'non_fetched':unfetched,'opcodes':ops,'reused_data':reused_data,'data_refs':data_refs}
    for op,count in ops.items():opcodes.append({'Kernel':label,'Opcode':op,'Count':count})
    base=OUT/'traces'/name
    path=next(base.glob('drmemtrace.*.dir'))
    gen=cmd_by_name['trace_'+name]
    analysis=[cmd_by_name[stage+'_'+name] for stage in ['stats','cache','memory']]
    inst_ms=value(r'^instrumented_region_wall_ms=(\S+)',trace_log,float)
    peak=max(r['peak_sampled_disk_bytes'] or 0 for r in [gen,*analysis])
    rss=[]
    for r in [gen,*analysis]:
        t=Path(r['time_log']).read_text()
        rss.append(value(r'^\s*Maximum resident set size \(kbytes\): (\d+)',t))
    uncompressed=sum(info.file_size for p in (path/'trace').glob('*.trace.zip') for info in zipfile.ZipFile(p).infolist())
    resources.append({'Converted_Trace_Uncompressed_Bytes':uncompressed,'Kernel':label,'Trace_Generation_s':gen['elapsed_seconds'],'Instrumented_ROI_ms':inst_ms,'Instrumented_ROI_over_Native_Median':inst_ms/runtime,'Raw_Trace_Logical_Bytes':gen['final_logical_bytes'],'Converted_Trace_Logical_Bytes':disk(path/'trace')[0],'Conversion_and_Analysis_s':sum(r['elapsed_seconds'] for r in analysis),'Stats_and_Conversion_s':analysis[0]['elapsed_seconds'],'Cache_Analysis_s':analysis[1]['elapsed_seconds'],'Memory_Analysis_s':analysis[2]['elapsed_seconds'],'Peak_Trace_Disk_Allocated_Bytes':peak,'Retained_Trace_Disk_Allocated_Bytes':disk(base)[1],'Max_Child_RSS_KiB':max(rss)})

csvwrite('kernel_characterization.csv',main)
csvwrite('trace_characterization.csv',extended)
csvwrite('roofline.csv',roofs)
csvwrite('attention_phases.csv',phases,['Kernel','QK_ms','Softmax_ms','PV_ms','Total_ms','Min_Total_ms','Max_Total_ms'])
csvwrite('trace_resources.csv',resources)
csvwrite('opcode_mix.csv',opcodes)
csvwrite('reuse_histograms.csv',[{'Kernel':k,'Distance_Min':h[0],'Distance_Max':h[1],'All_Reused_References':h[2],'Data_Reused_References':h[3]} for k,hs in histograms.items() for h in hs])

plt.rcParams.update({'font.size':10,'figure.dpi':120,'savefig.dpi':180})
styles={'gemm128':('#1768ac','o'),'gemm512':('#ff7f0e','s'),'gemv':('#249447','D'),'attention128':('#8c4ca6','^'),'attention512':('#cc4452','v')}
colors=[styles[w[0]][0] for w in WORK];marks=[styles[w[0]][1] for w in WORK]
host=re.search(r'\$ hostname\n([^\n]+)',(OUT/'environment.txt').read_text()).group(1)
x=np.logspace(-1,3,400)
fig,ax=plt.subplots(figsize=(10,6))
ax.loglog(x,bandwidth*x,'--',color='#909090',label=f'Measured bandwidth reference: {bandwidth:.1f} GB/s')
ax.axhline(compute,color='#333333',ls=':',label=f'Measured FMA reference: {compute:.1f} GFLOP/s')
ax.loglog(x,np.minimum(compute,bandwidth*x),color='black',lw=2,label='Analytical Roofline')
ax.axvline(ridge,color='#777777',ls='-.',label=f'Ridge: {ridge:.2f} FLOP/Byte')
ax.scatter([ridge],[compute],marker='x',color='black',s=70,zorder=4)
for r,c,m,(_,_,kernel,sequence) in zip(roofs,colors,marks,WORK):
    nr=next(n for n in native if n['kernel']==kernel and int(n['sequence'])==sequence)
    slow=float(nr['approx_flops'])/(float(nr['max_ms'])*1e6)
    fast=float(nr['approx_flops'])/(float(nr['min_ms'])*1e6)
    ax.errorbar(r['Algorithmic_AI'],r['Measured_GFLOP_per_s'],yerr=[[r['Measured_GFLOP_per_s']-slow],[fast-r['Measured_GFLOP_per_s']]],fmt='none',ecolor=c,capsize=3,alpha=.75)
    ax.scatter(r['Algorithmic_AI'],r['Measured_GFLOP_per_s'],s=80,c=c,marker=m,zorder=5,label=r['Kernel']+' (native)')
    dy=-15 if r['Kernel']=='Attention S=128' else 9
    ax.annotate(r['Kernel'],(r['Algorithmic_AI'],r['Measured_GFLOP_per_s']),xytext=(6,dy),textcoords='offset points',fontsize=9)
ax.set(xlabel='Arithmetic Intensity (FLOP/Byte)',ylabel='Performance (GFLOP/s)',xlim=(.1,1000),ylim=(1,compute*2),title=f'MP1 on {host}: 8 physical cores, native performance / algorithmic AI')
ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=8,loc='lower left',ncol=2)
fig.text(.5,.01,'Attention uses phase-sum bytes / operation-equivalents. Error bars: observed 9-sample throughput range, not confidence intervals.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.03,1,1));fig.savefig(OUT/'roofline.png');plt.close(fig)

labels=[r['Kernel'].replace(' S=','\nS=') for r in main];pos=np.arange(len(WORK))
fig,ax=plt.subplots(figsize=(9,5))
ax.bar(pos-.19,[100*r['L1D_Miss_Rate'] for r in extended],.38,label='Simulated L1D local miss rate')
ax.bar(pos+.19,[100*r['LLC_Miss_Rate'] for r in extended],.38,label='Simulated LLC local miss rate')
ax.set_xticks(pos,labels);ax.set_ylabel('Misses / cache accesses (%)');ax.set_title('Cold-start cache simulation: 8 cores, 256 MiB modeled LLC')
ax.legend();ax.grid(axis='y',alpha=.2)
fig.text(.5,.01,'L1D is access-weighted across cores. LLC denominator counts only requests reaching LLC.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.03,1,1));fig.savefig(OUT/'cache_miss_rates.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
for off,col,text in [(-.25,'Instructions','Instructions'),(0,'Loads','Data loads'),(.25,'Stores','Data stores')]:
    vals=[r[col]/r['Instructions'] for r in extended]
    ax.bar(pos+off,vals,.25,label=text)
ax.set_xticks(pos,labels);ax.set_ylabel('Count / dynamic instruction count');ax.set_title('Trace-derived instruction and memory-reference ratios');ax.legend()
fig.text(.5,.01,'Loads and stores count memory operands, not mutually exclusive instruction categories or FP32 elements.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.03,1,1));fig.savefig(OUT/'instruction_memory_mix.png');plt.close(fig)
fig,(ax,coldax)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[2.4,1]})
for label,c in zip([r['Kernel'] for r in main],colors):
    hs=histograms[label];tot=sum(h[3] for h in hs)
    ax.step([h[1]+1 for h in hs],100*np.cumsum([h[3] for h in hs])/tot,where='post',label=label,color=c)
ax.set_xscale('log');ax.set(xlabel='Reuse distance + 1 (distinct 64-byte lines; instruction + data stream)',ylabel='Cumulative reused data references (%)',ylim=(0,101),title='Data-reference reuse CDF, conditional on a finite reuse')
ax.legend();ax.grid(True,which='both',alpha=.2)
coldax.barh(np.arange(len(WORK)),[100*r['Data_Cold_Reference_Fraction'] for r in extended],color=colors)
coldax.set_yticks(np.arange(len(WORK)),[r['Kernel'].replace(' S=',' ') for r in main],fontsize=8)
coldax.invert_yaxis();coldax.set_xlabel('Cold data references (%)');coldax.set_title('Per-thread first observations',fontsize=10);coldax.grid(axis='x',alpha=.2)
fig.text(.5,.01,'Per-thread distances pooled; cold references excluded from CDF. Exact counts in geometric bins (multiplier 1.5).',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.03,1,1));fig.savefig(OUT/'reuse_distance.png');plt.close(fig)

payload={'main':main,'trace':extended,'roofline':roofs,'resources':resources,'phases':phases,'compute':compute,'bandwidth':bandwidth,'ridge':ridge,'aux':aux}
(OUT/'analysis.json').write_text(json.dumps(payload,indent=2)+'\n')
print(f'Validated {len(WORK)} complete traces against native checksums and independent memory counts; wrote CSVs and four plots.')
