"""Create supporting phase metrics from six actual isolated phase traces."""
import csv,json,re,sys
from run_logged import OUT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def count(pattern,text):return int(re.search(pattern,text,re.M).group(1))
def cache(text):
    pairs={}
    for m in re.finditer(r'^\s*(L1D\d+|LLC) \([^\n]*\) stats:\n((?:[ \t]+[^\n]*\n)+)',text,re.M):
        pairs[m[1]]=[count(r'^\s*'+k+r':\s*(\d+)',m[2]) for k in ['Hits','Misses']]
    assert len(pairs)==9
    h=sum(v[0] for k,v in pairs.items() if k!='LLC');m=sum(v[1] for k,v in pairs.items() if k!='LLC');lh,lm=pairs['LLC']
    return m/(h+m),lm/(lh+lm)
rows=[];native=list(csv.DictReader((OUT/'native_kernel_results.csv').open()))
sizes=[s for s in [128,512] if all((OUT/'logs'/f'memory_attention{s}_{phase}.log').exists() for phase in ['qk','softmax','pv'])] if '--available' in sys.argv else [128,512]
assert sizes,'No complete Attention phase set exists.'
for s in sizes:
    n=next(r for r in native if r['kernel']=='attention' and int(r['sequence'])==s)
    for phase,label,field in [('qk','QK+scale','qk_ms'),('softmax','Softmax','softmax_ms'),('pv','PV','pv_ms')]:
        name=f'attention{s}_{phase}';stats=(OUT/'logs'/f'stats_{name}.log').read_text()
        mem=json.loads((OUT/'logs'/f'memory_{name}.log').read_text())
        loads=count(r'^\s*(\d+) total data loads',stats);stores=count(r'^\s*(\d+) total data stores',stats)
        assert loads==mem['Loads'] and stores==mem['Stores']
        instructions=count(r'^\s*(\d+) total \(fetched\) instructions',stats)+count(r'^\s*(\d+) total non-fetched instructions',stats)
        assert instructions==count(r'^\s*(\d+) : total executed instructions',stats)
        assert count(r'^\s*(\d+) total threads',stats)==8
        assert 'trace_full_output_max_error=0' in (OUT/'logs'/f'trace_{name}.log').read_text()
        l1,ll=cache((OUT/'logs'/f'cache_{name}.log').read_text())
        rows.append({'Kernel':f'Attention S={s}','Phase':label,'Native_Phase_Median_ms':float(n[field]),'Instructions':instructions,'Loads':loads,'Stores':stores,'Dynamic_Memory_Reference_Bytes':mem['Dynamic_Memory_Reference_Bytes'],'Unique_Cache_Lines':mem['Unique_Cache_Lines'],'L1D_Miss_Rate':l1,'LLC_Miss_Rate':ll})
with (OUT/'attention_phase_trace_characterization.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,ax=plt.subplots(figsize=(8,5));bottom=np.zeros(len(sizes))
for phase,color in [('QK+scale','#1768ac'),('Softmax','#e99c32'),('PV','#249447')]:
    vals=np.array([r['Native_Phase_Median_ms'] for r in rows if r['Phase']==phase]);ax.bar(np.arange(len(sizes)),vals,bottom=bottom,label=phase,color=color);bottom+=vals
for i,s in enumerate(sizes):
    total=float(next(n for n in native if n['kernel']=='attention' and int(n['sequence'])==s)['median_ms'])
    ax.scatter(i,total,marker='D',facecolor='none',edgecolor='black',s=85,zorder=3,label='Measured total median' if i==0 else None)
ax.set_xticks(np.arange(len(sizes)),[f'Attention S={s}' for s in sizes]);ax.set_ylabel('Native median time (ms)');ax.set_title('Attention phase times, 8 threads');ax.legend();ax.grid(axis='y',alpha=.2)
fig.text(.5,.015,'Stacked phase medians need not sum exactly to the independently summarized total median.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.035,1,1));fig.savefig(OUT/'attention_phases.png',dpi=180);plt.close(fig)
print(f'{len(rows)} independent phase trace outputs validated; phase CSV and native timing plot saved.')
