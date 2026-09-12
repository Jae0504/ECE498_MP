"""Measure and audit empirical Roofline saturation; never infer it from a run's exit code.

Run native benchmarks sequentially, using distinct allowed physical cores. The
default maximum is the requested roof scope; explicitly increase --max-threads
to investigate socket memory saturation. Outputs retain both resource scopes.
"""
import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time


def configuration(row):
    return dict(item.split('=', 1) for item in row['configuration'].split(';') if '=' in item)


def physical_cpus():
    seen, result = set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        path = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = tuple((path / field).read_text().strip() for field in ('physical_package_id', 'core_id'))
        if key not in seen:
            seen.add(key)
            result.append(cpu)
    return result


def cache_bytes(cpus):
    caches = {}
    for cpu in cpus:
        for path in Path(f'/sys/devices/system/cpu/cpu{cpu}/cache').glob('index*'):
            if (path/'level').read_text().strip() != '3':
                continue
            size = (path/'size').read_text().strip()
            caches[(path/'shared_cpu_list').read_text().strip()] = int(size[:-1]) * {'K': 1024, 'M': 2**20}[size[-1]]
    if not caches:
        raise ValueError('Cannot identify the L3 cache topology required for this calibration.')
    return sum(caches.values())


def plateau(values, tolerance):
    return len(values) >= 3 and max(values) / min(values) <= 1 + tolerance


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', type=Path, default=project/'build/roofline_bench')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8, help='Resource scope of the requested roof.')
    parser.add_argument('--max-threads', type=int, help='Explicitly allowed maximum for the saturation sweep.')
    parser.add_argument('--sizes-mib', type=int, nargs='+', help='Total working sets; default gives each triad array at least 4x LLC.')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=9)
    parser.add_argument('--repeats', type=int, default=20000000)
    parser.add_argument('--tolerance', type=float, default=0.05)
    parser.add_argument('--fma-per-cycle', type=float, help='Documented FMA instruction throughput/core; enables cycle-utilization check.')
    parser.add_argument('--resume', action='store_true', help='Reuse completed commands with identical binaries, allocation and settings.')
    args = parser.parse_args()
    cpus = physical_cpus()
    maximum = args.max_threads or args.threads
    if not 1 <= args.threads <= maximum <= len(cpus):
        parser.error('Requested scope exceeds distinct allowed physical cores.')
    if args.rounds < 1 or args.iterations < 3 or args.repeats < 1 or not 0 < args.tolerance < 1:
        parser.error('Need positive rounds/repeats, at least three samples and tolerance in (0,1).')
    if args.fma_per_cycle is not None and args.fma_per_cycle <= 0:
        parser.error('--fma-per-cycle must be positive.')
    llc = cache_bytes(cpus[:maximum])
    per_array = 2 ** math.ceil(math.log2(4 * llc))
    sizes = sorted(set(args.sizes_mib or [3*per_array//2**20, 6*per_array//2**20]))
    if len(sizes) < 2 or min(sizes) < 1:
        parser.error('At least two positive working-set sizes are required.')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    if max(sizes)*2**20 > available / 4:
        parser.error('Largest stream would exceed one quarter of available memory; reduce --sizes-mib.')
    counts = sorted(set([n for n in (1,2,4,8,16,24,32,48,56,64) if n <= maximum] + [args.threads, maximum]))
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    (out/'logs').mkdir(exist_ok=args.resume)
    bench = args.bench.resolve()
    env = {**os.environ, 'OMP_PLACES': 'threads', 'OMP_PROC_BIND': 'close',
           'OMP_DYNAMIC': 'false', 'OMP_WAIT_POLICY': 'PASSIVE', 'GOMP_SPINCOUNT': '0'}
    metadata = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'bench': str(bench), 'binary_sha256': hashlib.sha256(bench.read_bytes()).hexdigest(),
                'cpus': cpus[:maximum], 'roof_threads': args.threads, 'counts': counts,
                'sizes_mib': sizes, 'llc_bytes': llc, 'rounds': args.rounds,
                'iterations': args.iterations, 'warmups': 2, 'tolerance': args.tolerance,
                'fma_per_cycle': args.fma_per_cycle,
                'environment': {k: v for k, v in env.items() if k.startswith(('OMP_', 'GOMP_'))},
                'lscpu': subprocess.check_output(['lscpu'], text=True),
                'source_hashes': {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [project/'cpu/roofline.cpp', project/'include/roofline_fma.h']}}
    previous = {}
    if args.resume:
        saved = json.loads((out/'metadata.json').read_text())
        for key in ('bench','binary_sha256','cpus','roof_threads','counts','sizes_mib','llc_bytes',
                    'rounds','iterations','warmups','tolerance','fma_per_cycle','environment','source_hashes'):
            if saved[key] != metadata[key]:
                raise ValueError('Cannot resume after configuration changed: '+key)
        metadata = saved
        previous = {r['name']:r for r in map(json.loads,(out/'commands.jsonl').read_text().splitlines())}
    else:
        (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    rows = []

    def run(name, threads, arguments, perf=False):
        cmd = ['taskset', '-c', ','.join(map(str,cpus[:threads])), str(bench),
               '--threads', str(threads), '--warmup', '2', '--iterations', str(args.iterations),
               '--no-save', '--format', 'csv', *map(str,arguments)]
        if perf:
            cmd = ['perf', 'stat', '-x', ';', '-e', 'cycles,instructions,ref-cycles', *cmd]
        if name in previous:
            old = previous[name]
            if old['command'] != cmd or old['returncode'] != 0:
                raise ValueError('Preserve failed or different prior command: '+name)
            p = subprocess.CompletedProcess(cmd, 0, (out/'logs'/f'{name}.stdout').read_text(),
                                            (out/'logs'/f'{name}.stderr').read_text())
        else:
            start = time.monotonic()
            p = subprocess.run(cmd, env=env, text=True, capture_output=True, timeout=600)
            (out/'logs'/f'{name}.stdout').write_text(p.stdout)
            (out/'logs'/f'{name}.stderr').write_text(p.stderr)
            with (out/'commands.jsonl').open('a') as f:
                f.write(json.dumps({'name': name, 'command': cmd, 'returncode': p.returncode,
                                   'seconds': time.monotonic()-start})+'\n')
        if p.returncode:
            if perf:
                return None, p.stderr
            raise RuntimeError(f'{name} failed: {p.stderr}')
        measured = list(csv.DictReader(io.StringIO(p.stdout)))
        for row in measured:
            if not math.isfinite(float(row['rate'])) or float(row['rate']) <= 0:
                raise RuntimeError(f'{name}: invalid measurement')
            row['experiment'] = name
            if not perf:
                rows.append(row)
            print(name, row['ceiling'], row['rate'], row['unit'], flush=True)
        if not perf:
            with (out/'samples.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        return measured, p.stderr

    # Reverse order on the second round to expose order/thermal bias.
    for round_ in range(args.rounds):
        cases = [(n, size) for n in counts for size in sizes]
        if round_ % 2:
            cases.reverse()
        for n, size in cases:
            run(f'bw_r{round_}_t{n}_{size}MiB', n, ['--ceiling', 'bandwidth',
                '--stream-elements', size*2**20//12, '--read-elements', size*2**20//4])
    flags = Path('/proc/cpuinfo').read_text().split('flags',1)[1].splitlines()[0].split()
    accumulators = [1,2,4,8,12,16,24] if 'avx512f' in flags else [1,2,4,8,12]
    chosen_acc = 16 if 16 in accumulators else 8
    for round_ in range(args.rounds):
        for acc in (accumulators if round_ % 2 == 0 else list(reversed(accumulators))):
            run(f'fma_r{round_}_t{args.threads}_a{acc}', args.threads,
                ['--ceiling','compute','--accumulators',acc,'--fma-repeats',args.repeats])
        for n in counts:
            if n != args.threads:
                run(f'fma_r{round_}_t{n}_a{chosen_acc}', n,
                    ['--ceiling','compute','--accumulators',chosen_acc,'--fma-repeats',args.repeats])

    def matches(row, threads, ceiling, size=None, acc=None):
        cfg = configuration(row)
        return (int(cfg['threads']) == threads and row['ceiling'] == ceiling
                and (size is None or int(cfg['elements']) == size*2**20//(12 if ceiling=='memory_bandwidth' else 4))
                and (acc is None or int(cfg.get('accumulators',0)) == acc))

    def median_rate(threads, ceiling, size=None, acc=None):
        return statistics.median(float(r['rate']) for r in rows if matches(r,threads,ceiling,size,acc))

    rates = {n: max(median_rate(n, ceiling, max(sizes))
                    for ceiling in ('memory_bandwidth','memory_bandwidth_read')) for n in counts}
    # Require the SAME access pattern to plateau across three distinct core
    # counts and both sizes. Do not switch between read/triad to manufacture it.
    saturated = []
    for ceiling in ('memory_bandwidth','memory_bandwidth_read'):
        for i in range(len(counts)-2):
            window = counts[i:i+3]
            # Test every round median so order effects cannot average away.
            values = [float(r['rate']) for r in rows for n in window for size in sizes
                      if matches(r,n,ceiling,size)]
            global_best = max(median_rate(n,ceiling,max(sizes)) for n in counts)
            if (plateau(values,args.tolerance) and min(values) >= global_best/(1+args.tolerance)
                    and min(sizes)*2**20/(3 if ceiling=='memory_bandwidth' else 1) >= 4*llc):
                saturated.append({'ceiling':ceiling,'threads':window,'min_GBs':min(values),'max_GBs':max(values)})
    fma_rates = {a: median_rate(args.threads,'fma_compute',acc=a) for a in accumulators}
    fma_plateau = plateau([float(r['rate']) for r in rows for a in accumulators[-3:]
                          if matches(r,args.threads,'fma_compute',acc=a)], args.tolerance)
    counters = {}
    if shutil.which('perf'):
        for n in sorted(set([args.threads,maximum])):
            measured, log = run(f'perf_fma_t{n}', n,
                ['--ceiling','compute','--accumulators',chosen_acc,'--fma-repeats',args.repeats],perf=True)
            if measured:
                events = {}
                for line in log.splitlines():
                    fields = line.split(';')
                    if len(fields) > 2:
                        try: events[fields[2].split(':')[0]] = float(fields[0])
                        except ValueError: pass
                if events.get('cycles',0) > 0:
                    r = measured[0]; lanes = int(configuration(r)['lanes'])
                    # Counters cover startup + all warmups/samples, not just one
                    # median sample. Extra validation/runtime cycles make this
                    # a conservative whole-process utilization estimate.
                    fmas = float(r['work']) * (2+args.iterations) / (2*lanes)
                    counters[n] = {'events':events,'fma_instructions_per_cycle':fmas/events['cycles']}
                    if args.fma_per_cycle:
                        counters[n]['fma_capacity_fraction'] = fmas/events['cycles']/args.fma_per_cycle
    ceilings = []
    for n in counts:
        for ceiling in ('memory_bandwidth','memory_bandwidth_read','fma_compute'):
            selected = [r for r in rows if matches(r,n,ceiling,
                         max(sizes) if ceiling != 'fma_compute' else None,
                         chosen_acc if ceiling == 'fma_compute' else None)]
            row = dict(selected[0]); row.pop('experiment')
            # Use all sample-series medians, never the fastest run.
            row['median_ms'] = statistics.median(float(r['median_ms']) for r in selected)
            row['min_ms'] = min(float(r['min_ms']) for r in selected)
            row['max_ms'] = max(float(r['max_ms']) for r in selected)
            row['rate'] = float(row['work'])/(row['median_ms']*1e6)
            row['relative_range_percent'] = 100*(row['max_ms']-row['min_ms'])/row['median_ms']
            row['configuration'] += f';calibration_rounds={args.rounds}'
            ceilings.append(row)
    with (out/'ceilings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(ceilings[0]));writer.writeheader();writer.writerows(ceilings)
    result = {'roof_threads':args.threads,'max_threads':maximum,'bandwidth_GBs':rates,
              'bandwidth_plateaus':saturated,'requested_scope_reaches_bandwidth_plateau':
              any(min(s['threads']) <= args.threads for s in saturated),
              'fma_rates_by_accumulators':fma_rates,'fma_accumulator_plateau':fma_plateau,
              'perf':counters,'max_timing_range_percent':max(float(r['relative_range_percent']) for r in rows),
              'notes':['Plateau tolerance applies to every round median across both sizes and three core counts; individual slow samples remain in timing ranges.',
                       'Memory rates count useful bytes, not DRAM controller transactions.',
                       'Core-limited and socket-saturated roofs retain separate thread counts.']}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    metadata.setdefault('finished_utc', datetime.datetime.now(datetime.timezone.utc).isoformat())
    metadata['analyzer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
