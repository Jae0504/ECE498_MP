"""Remove only generated raw trace records after full conversion and a complete successful analysis read.
Keep converted compressed traces, encodings, module maps and all analysis logs.
"""
import json,re,sys
from run_logged import OUT,disk
for name in sys.argv[1:]:
    events=[json.loads(s) for s in (OUT/'commands.jsonl').read_text().splitlines()]
    by_name={r['name']:r for r in events}
    for stage in ['trace','stats']:
        assert by_name[stage+'_'+name]['returncode']==0
    stats=(OUT/'logs'/f'stats_{name}.log').read_text()
    memory_path=OUT/'logs'/f'memory_{name}.log'
    if memory_path.exists() and memory_path.stat().st_size:
        memory=json.loads(memory_path.read_text())
        for field,term in [('Loads','loads'),('Stores','stores')]:
            assert memory[field]==int(re.search(r'^\s*(\d+) total data '+term,stats,re.M).group(1))
    assert int(re.search(r'^\s*(\d+) total skipped memref markers',stats,re.M).group(1))==0
    assert 'trace_full_output_max_error=0' in (OUT/'logs'/f'trace_{name}.log').read_text()
    base=OUT/'traces'/name
    trace,=base.glob('drmemtrace.*.dir')
    assert len(list((trace/'trace').glob('*.trace.zip')))==8
    removed=[]
    for p in (trace/'raw').glob('*.raw.lz4'):
        removed.append({'path':str(p),'bytes':p.stat().st_size});p.unlink()
    with (OUT/'cleanup.jsonl').open('a') as f:f.write(json.dumps({'name':name,'removed':removed,'retained_logical_bytes':disk(base)[0],'retained_allocated_bytes':disk(base)[1]})+'\n')
    print(f'{name}: removed {sum(p["bytes"] for p in removed)} raw bytes; retained {disk(base)[0]} bytes')
