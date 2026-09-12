"""Run selected native/trace analyses without claiming a full-suite completion."""
import csv
import datetime
import json
import math
import re
import shutil
import subprocess
import sys

from deployment import HERE, ROOT

METRICS = ('native', 'counts', 'opcodes', 'reuse', 'cache', 'memory')


def value(pattern, text, cast=int):
    match = re.search(pattern, text, re.M)
    if not match:
        raise ValueError('Missing measured field: ' + pattern)
    return cast(match[1].replace(',', ''))


def parse_counts(text):
    result = {
        'Instructions': value(r'^\s*([\d,]+) total \(fetched\) instructions', text)
        + value(r'^\s*([\d,]+) total non-fetched instructions', text),
        'Loads': value(r'^\s*([\d,]+) total data loads', text),
        'Stores': value(r'^\s*([\d,]+) total data stores', text),
        'Threads': value(r'^\s*([\d,]+) total threads', text),
        'Skipped_Memref_Markers': value(r'^\s*([\d,]+) total skipped memref markers', text),
    }
    if result['Threads'] != 8 or result['Skipped_Memref_Markers'] != 0:
        raise ValueError('Expected a complete eight-thread trace without skipped references.')
    return result


def parse_opcodes(text, instructions):
    body = text.split('Opcode mix tool results:', 1)[1]
    body = re.split(r'\n\s*[\d,]+ : sets of categories', body)[0]
    result = {}
    for count, opcode in re.findall(r'^\s*([\d,]+) :\s+(.+)$', body, re.M):
        opcode = opcode.strip()
        if opcode == 'total executed instructions':
            if int(count.replace(',', '')) != instructions:
                raise ValueError('Opcode instruction count differs from basic_counts.')
            continue
        result[opcode] = result.get(opcode, 0) + int(count.replace(',', ''))
    if sum(result.values()) != instructions:
        raise ValueError('Opcode totals differ from basic_counts.')
    return result


def parse_cache(text):
    caches = {}
    for name, body in re.findall(r'^\s*(L1D\d+|L2_\d+|LLC) \([^\n]*\) stats:\n((?:[ \t]+[^\n]*\n)+)', text, re.M):
        caches[name] = (value(r'^\s*Hits:\s*([\d,]+)', body),
                        value(r'^\s*Misses:\s*([\d,]+)', body))
    result = {}
    for level, expected in [('L1D', 8), ('L2_', 8), ('LLC', 1)]:
        entries = [v for k, v in caches.items() if k.startswith(level)]
        if len(entries) != expected:
            raise ValueError(f'Expected {expected} {level} caches, found {len(entries)}.')
        hits, misses = (sum(v[i] for v in entries) for i in (0, 1))
        label = level.rstrip('_')
        result.update({label + '_Accesses': hits + misses, label + '_Misses': misses,
                       label + '_Miss_Rate': misses / (hits + misses) if hits + misses else None})
    return result


def parse_reuse(text, counts):
    body = text.split('Reuse distance tool aggregated results:', 1)[1].split('Reuse distance threshold', 1)[0]
    if value(r'^Distance limit: (\d+)', body) or value(r'^Pruned addresses: (\d+)', body):
        raise ValueError('Expected exact, unpruned reuse distances.')
    refs = value(r'^Data accesses: (\d+)', body)
    if refs != counts['Loads'] + counts['Stores']:
        raise ValueError('Reuse data accesses differ from basic_counts.')
    histogram = [dict(zip(('Min_Distance', 'Max_Distance', 'All_References', 'Data_References'), map(int, row)))
                 for row in re.findall(r'^\s*(\d+) -\s*(\d+)\s+(\d+)\s+[\d.]+%\s+[\d.]+%\s*:\s*(\d+)', body, re.M)]
    if not histogram:
        raise ValueError('Missing reuse histogram.')
    return {
        'Mean_Reuse_Distance': value(r'^Reuse distance mean: (\S+)', body, float),
        'Median_Reuse_Distance': value(r'^Reuse distance median: (\S+)', body, float),
        'Data_Cold_Reference_Fraction': (refs - sum(r['Data_References'] for r in histogram)) / refs if refs else None,
        'scope': '64-byte instruction and data lines, per-thread histories; mean/median include both',
        'histogram': histogram,
    }


def measure_selected(out, name, kernel, sequence, metrics, phase='full'):
    # Imported only after measure.py configures the output directory and affinity.
    from run_logged import run, CPUSET

    def events():
        path = out / 'commands.jsonl'
        return {r['name']: r for r in map(json.loads, path.read_text().splitlines())} if path.exists() else {}

    def success(stage):
        return events().get(stage + '_' + name, {}).get('returncode') == 0

    requested = list(dict.fromkeys(metrics))
    if 'native' in requested:
        native_path = out / 'native_kernel_results.csv'
        rows = list(csv.DictReader(native_path.open())) if native_path.exists() else []
        matching = [r for r in rows if r['kernel'] == kernel and int(r['sequence']) == sequence]
        if not success('native'):
            if matching:
                raise SystemExit('Incomplete native series exists; preserve it and use a fresh output directory.')
            run('native_' + name, ['taskset', '-c', CPUSET, ROOT/'mp1/build-dynamorio-native/cpu_bench',
                '--kernel', kernel, '--seq', sequence, '--threads', '8', '--warmup', '2', '--iterations', '9',
                '--flush-cache-mib', '0', '--no-plot', '--output', native_path])
        elif len(matching) != 1:
            raise SystemExit('Completed native CSV is missing or contains duplicate rows.')
        print('Native measurements: ' + str(native_path), flush=True)

    trace_metrics = [m for m in requested if m != 'native']
    if trace_metrics:
        trace_name = name if phase == 'full' else name + '_' + phase
        name = trace_name
        base = out/'traces'/name
        command = [sys.executable, str(HERE/'run_trace.py'), name, kernel, str(sequence)]
        if not success('trace'):
            required = {'gemm128': 6, 'gemm512': 18}.get(name, 2.25)
            if shutil.disk_usage(out).free < required * 2**30:
                raise SystemExit(f'{name} requires at least {required} GiB free for tracing.')
            subprocess.run([*command, 'collect', '--phase', phase], check=True)
        trace_log = (out/'logs'/f'trace_{name}.log').read_text()
        if value(r'^trace_full_output_max_error=(\S+)', trace_log, float) != 0:
            raise ValueError('Traced output failed correctness validation.')
        paths = list(base.glob('drmemtrace.*.dir'))
        if len(paths) != 1:
            raise ValueError('Expected exactly one complete trace.')
        # counts supplies conversion and integrity checks even for cache-only requests.
        subprocess.run([*command, 'counts'], check=True)
        counts = parse_counts((out/'logs'/f'counts_{name}.log').read_text())
        if len(list((paths[0]/'trace').glob('*.trace.zip'))) != 8:
            raise ValueError('Expected eight converted trace shards.')
        for stage in trace_metrics:
            if stage != 'counts':
                subprocess.run([*command, stage], check=True)

        report_path = out/'metrics'/f'{name}.json'
        report = json.loads(report_path.read_text()) if report_path.exists() else {
            'case': name, 'kernel': kernel, 'sequence': sequence, 'phase': phase,
            'threads': 8, 'trace': str(paths[0]), 'metrics': {},
        }
        result = report['metrics']
        # Keep mandatory integrity counts visible and label them as a dependency.
        result['counts'] = counts
        report['requested_metrics'] = sorted(set(report.get('requested_metrics', [])) | set(trace_metrics))
        report['validation_dependencies'] = ['counts']
        for stage in trace_metrics:
            raw = (out/'logs'/f'{stage}_{name}.log').read_text()
            if stage == 'opcodes':
                result[stage] = parse_opcodes(raw, counts['Instructions'])
            elif stage == 'cache':
                result[stage] = parse_cache(raw)
            elif stage == 'reuse':
                result[stage] = parse_reuse(raw, counts)
            elif stage == 'memory':
                memory = json.loads(raw)
                if any(memory[k] != counts[k] for k in ('Loads', 'Stores')):
                    raise ValueError('Memory reference totals differ from basic_counts.')
                result[stage] = memory
        native_path = out/'native_kernel_results.csv'
        if phase == 'full' and native_path.exists():
            native = [r for r in csv.DictReader(native_path.open()) if r['kernel'] == kernel and int(r['sequence']) == sequence]
            if native and not math.isclose(float(native[0]['checksum']), value(r'^trace_checksum=(\S+)', trace_log, float), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError('Native/trace checksums differ.')
        report['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report_path.parent.mkdir(exist_ok=True)
        temporary = report_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(report_path)
        print('Selected metrics: ' + str(report_path), flush=True)
    with (out/'selected_requests.jsonl').open('a') as f:
        f.write(json.dumps({'case': name, 'phase': phase, 'metrics': requested,
                            'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}) + '\n')
