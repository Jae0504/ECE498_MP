"""Time exact reuse algorithms on the same two-row trace prefix, without new tracing."""
import json,re
from run_logged import ROOT,OUT,run
trace=next((OUT/'traces/gemm128').glob('drmemtrace.*.dir'))
shard=sorted((trace/'trace').glob('*.trace.zip'))[0]
results=[]
for distance in [2000,8192,500]:
    name=f'reuse_tuning_{distance}'
    r=run(name,[ROOT/'dynamorio/build/clients/bin64/drmemtrace_launcher','-infile',shard,'-jobs','1','-tool','reuse_distance','-exit_after_instrs','24000000','-reuse_skip_dist',str(distance)],timeout=300)
    text=(OUT/'logs'/f'{name}.log').read_text()
    summary={k:re.search(r'^'+re.escape(k)+r': (.+)$',text,re.M).group(1) for k in ['Total accesses','Data accesses','Reuse distance mean','Reuse distance median','Reuse distance standard deviation','Distance limit','Pruned addresses']}
    results.append({'skip_distance':distance,'elapsed_seconds':r['elapsed_seconds'],'summary':summary})
assert all(r['summary']==results[0]['summary'] for r in results)
(OUT/'reuse_tuning.json').write_text(json.dumps(results,indent=2)+'\n')
print('Identical prefix statistics for all tuning choices; fastest:',min(results,key=lambda r:r['elapsed_seconds'])['skip_distance'])
