"""Render saved calibration evidence without taking additional measurements."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mp1-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--comparison', type=Path, help='Optional original.csv/updated.csv from matched-size runs.')
    args = parser.parse_args()
    out = args.calibration
    meta = json.loads((out/'metadata.json').read_text())
    audit = json.loads((out/'verification.json').read_text())
    with (out/'samples.csv').open() as f: samples = list(csv.DictReader(f))
    with (out/'ceilings.csv').open() as f: ceilings = list(csv.DictReader(f))
    def cfg(row): return dict(p.split('=',1) for p in row['configuration'].split(';') if '=' in p)
    def save(fig, name):
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(out/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,ax = plt.subplots(figsize=(8,5))
    for kind,color,label in [('memory_bandwidth','#1768ac','Triad (streaming stores)'),
                              ('memory_bandwidth_read','#cf631b','Read-only')]:
        for index,size in enumerate(meta['sizes_mib']):
            rates=[]
            for threads in meta['counts']:
                selected=[float(r['rate']) for r in samples if r['ceiling']==kind
                          and int(cfg(r)['threads'])==threads
                          and int(cfg(r)['elements'])==size*2**20//(12 if kind=='memory_bandwidth' else 4)]
                rates.append(statistics.median(selected))
            ax.plot(meta['counts'],rates,marker='o',color=color,linestyle='-' if index else '--',
                    label=f'{label}, {size/1024:g} GiB')
    ax.axvline(meta['roof_threads'],color='gray',linestyle=':',label='Requested roof scope')
    ax.set(xlabel='Physical cores',ylabel='Useful bandwidth (GB/s)',
           title=f'Bandwidth saturation: {meta["rounds"]} rounds, {meta["iterations"]} samples/round')
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    save(fig,'bandwidth_saturation')
    fig,ax=plt.subplots(figsize=(8,5))
    accs=sorted(map(int,audit['fma_rates_by_accumulators']))
    ax.plot(accs,[audit['fma_rates_by_accumulators'][str(a)] for a in accs],marker='o')
    ax.set(xlabel='Independent register accumulators per thread',ylabel='FP32 GFLOP/s',
           title=f'Register FMA saturation: {meta["roof_threads"]} physical cores')
    ax.set_xticks(accs);ax.grid(alpha=.25)
    save(fig,'fma_saturation')
    scope=meta['roof_threads']; maximum=max(meta['counts'])
    def rate(threads,kind):
        return next(float(r['rate']) for r in ceilings if int(cfg(r)['threads'])==threads and r['ceiling']==kind)
    def bandwidth(threads): return max(rate(threads,k) for k in ('memory_bandwidth','memory_bandwidth_read'))
    fastest_kind=max(('memory_bandwidth','memory_bandwidth_read'),key=lambda k:rate(maximum,k))
    memory_saturated=any(p['ceiling']==fastest_kind and maximum in p['threads'] for p in audit['bandwidth_plateaus'])
    capacity=audit['perf'].get(str(maximum),{}).get('fma_capacity_fraction')
    compute_saturated=(0.95 <= capacity <= 1.05 if capacity is not None else
                       maximum==scope and audit['fma_accumulator_plateau'])
    saturated=memory_saturated and compute_saturated
    if saturated:
        chosen=[r for r in ceilings if int(cfg(r)['threads'])==maximum]
        with (out/'saturated_ceilings.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(chosen[0]));w.writeheader();w.writerows(chosen)
    elif (out/'saturated_ceilings.csv').exists():
        (out/'saturated_ceilings.csv').unlink()  # A derived file must not retain a stale certification.
    lines=['# Roofline saturation validation','',
           f'CPU allocation: `{meta["cpus"]}`. Requested scope: **{scope} physical cores**; '
           f'largest verified scope: **{maximum} physical cores**.',
           f'LLC: {meta["llc_bytes"]/2**20:g} MiB. Working sets: {meta["sizes_mib"]} MiB. '
           f'{meta["rounds"]} rounds in opposite orders, 2 warmups and {meta["iterations"]} samples per case. '
           'Reported rates combine series medians, not fastest samples.','',
           '| Scope | Bandwidth (GB/s) | FP32 FMA (GFLOP/s) |',
           '|---|---:|---:|']
    for n in sorted(set([scope,maximum])):
        lines.append(f'| {n} cores | {bandwidth(n):.3f} | {rate(n,"fma_compute"):.3f} |')
    lines += ['',f'Requested scope reaches a bandwidth plateau: **{audit["requested_scope_reaches_bandwidth_plateau"]}**.',
              f'Largest-scope selected bandwidth kernel reaches a plateau: **{memory_saturated}**.',
              f'Largest-scope compute saturation check: **{compute_saturated}**.',
              f'Matching-scope saturated ceiling export: **{saturated}**.',
              f'FMA accumulator plateau: **{audit["fma_accumulator_plateau"]}**.','',
              'A bandwidth pass requires the same access pattern at three core counts, across both sizes, '
              f'to stay within {100*meta["tolerance"]:g}% relative spread and within that margin of its observed best. '
              'Each triad array must be at least four times LLC. This verifies an empirical plateau, not a theoretical maximum.',
              '', f'The largest individual-series timing range was **{audit["max_timing_range_percent"]:.2f}%**. '
              'Individual slow samples are retained in the min/max ranges. The plateau test uses every '
              'round median, including both run orders; it does not claim that every invocation has '
              'the same latency or discard outliers to obtain its result.',
              '', 'Observed bandwidth plateaus:', '']
    for p in audit['bandwidth_plateaus']:
        lines.append(f'- {p["ceiling"]}, cores {p["threads"]}: {p["min_GBs"]:.3f}–{p["max_GBs"]:.3f} GB/s.')
    lines += ['', 'Hardware-cycle cross-check:', '']
    for n,info in audit['perf'].items():
        extra = (f', {100*info["fma_capacity_fraction"]:.2f}% of the specified FMA/cycle capacity'
                 if 'fma_capacity_fraction' in info else '')
        lines.append(f'- {n} cores: {info["fma_instructions_per_cycle"]:.4f} FMA instructions/core cycle{extra}.')
    lines += ['', 'Counters cover process setup plus all warmups and samples. The numerator includes all '
              'warmup/sample FMA invocations; setup adds cycles, so this is a conservative estimate. '
              'The throughput assumption is explicitly supplied, not inferred from a CPU name. '
              'For the tested Xeon 6761P, Intel documents two 512-bit FMA units per core: '
              '[Xeon 6 product brief](https://www.intel.com/content/www/us/en/products/docs/xeon-6-product-brief.html).',
              '', '![Bandwidth](bandwidth_saturation.png)', '', '![FMA](fma_saturation.png)', '']
    if args.baseline:
        with (args.baseline/'ceilings.csv').open() as f: baseline=list(csv.DictReader(f))
        old=next(float(r['rate']) for r in baseline if r['ceiling']=='fma_compute' and int(cfg(r)['threads'])==scope)
        lines += [f'Original array FMA at {scope} cores: **{old:.3f} GFLOP/s**. '
                  f'Updated register FMA: **{rate(scope,"fma_compute"):.3f} GFLOP/s** '
                  f'({rate(scope,"fma_compute")/old:.2f}x). '
                  'The original array case used 2,000,000 passes; the register case uses 20,000,000. '
                  'Both rates account for their actual work; both use 2 warmups and 9 samples. '
                  'Original files and commands remain in the sibling baseline directory.','']
    if args.comparison:
        with (args.comparison/'original.csv').open() as f: original=list(csv.DictReader(f))
        with (args.comparison/'updated.csv').open() as f: updated=list(csv.DictReader(f))
        lines += ['Matched-size original/updated comparison (separate validation series):','',
                  '| Benchmark | Working set | Original | Updated | Unit |',
                  '|---|---|---:|---:|---|']
        for old in original:
            oldcfg=cfg(old)
            new=next(r for r in updated if r['ceiling']==old['ceiling'] and
                     (old['ceiling']=='fma_compute' or cfg(r)['elements']==oldcfg['elements']))
            size=f'{float(old["work"])/2**30:g} GiB' if old['unit']=='GB/s' else 'register / legacy array'
            lines.append(f'| {old["ceiling"]} | {size} | {float(old["rate"]):.3f} | {float(new["rate"]):.3f} | {old["unit"]} |')
        lines += ['', 'Both bandwidth variants use the same sizes, core affinity and 2-warmup/9-sample '
                  'policy; order is reversed between the two sizes. FMA work is counted separately '
                  'for each implementation. This series is retained alongside the initial baseline, '
                  'including variation between series.','']
    lines += [f'The requested {scope}-core scope reaches the observed memory plateau: '
              f'{audit["requested_scope_reaches_bandwidth_plateau"]}. '
              'Where the requested allocation does not reach the plateau, more participating '
              'cores may be necessary; SIMD tuning alone need not remove that resource limit. '
              'Keep kernel and ceiling scopes explicit. '
              '`ceilings.csv` contains every measured scope; `saturated_ceilings.csv`, when present, '
              'contains both compute and bandwidth from the largest scope, never a mixture of core counts.',
              '', 'Useful-byte throughput is not a hardware DRAM-transaction counter. '
              'Parallel first touch and CPU pinning use the current NUMA policy; other hosts or allocations '
              'require their own calibration. Raw timing ranges, all commands, binary/source hashes, '
              'and hardware counters are retained in `samples.csv`, `commands.jsonl`, `metadata.json`, and `logs/`.', '']
    (out/'SATURATION_REPORT.md').write_text('\n'.join(lines))
    print(out/'SATURATION_REPORT.md')


if __name__=='__main__': main()
