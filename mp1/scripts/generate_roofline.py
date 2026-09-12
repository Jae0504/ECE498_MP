#!/usr/bin/env python3
"""Regenerate a measured Roofline and kernel points from append-only CSV files."""

import argparse
import csv
import math
import os
from pathlib import Path
import sys
import tempfile

KERNELS = ('gemm', 'gemv', 'attention')
FIELDS = ('kernel', 'configuration', 'arithmetic_intensity', 'measured_gflops',
          'roofline_bound', 'predicted_bound_type', 'threads', 'roof_threads',
          'bandwidth_gbs', 'compute_gflops', 'attained_percent')


def read_rows(path, required):
    # Either benchmark may run first; missing counterpart files are normal.
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f'{path}: expected CSV columns {", ".join(required)}')
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f'{path}: incomplete or malformed CSV row')
    return rows


def positive(value, name):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f'{name} must be finite and positive, got {value!r}')
    return result


def configuration_values(row):
    return dict(part.split('=', 1) for part in row.get('configuration', '').split(';')
                if '=' in part)


def thread_count(row):
    # Old CSVs predate --threads and represent a single worker.
    value = row.get('threads') or configuration_values(row).get('threads', '1')
    if not value.isascii() or not value.isdecimal() or int(value) <= 0:
        raise ValueError(f'invalid thread count: {value!r}')
    return int(value)


def select_ceilings(path, roof_threads=None):
    groups = {}
    for row in read_rows(path, ('ceiling', 'rate', 'unit')):
        name = row['ceiling']
        if name not in ('memory_bandwidth', 'memory_bandwidth_read', 'fma_compute'):
            continue
        threads = thread_count(row)
        group = groups.setdefault(threads, {'bandwidth': {}, 'compute': None})
        if name in ('memory_bandwidth', 'memory_bandwidth_read'):
            if row['unit'] != 'GB/s':
                raise ValueError(f'{path}: {name} requires GB/s')
            # New bandwidth pair for this thread count only. Never mix scopes
            # or use an older read result after a new triad is appended.
            if name == 'memory_bandwidth':
                group['bandwidth'] = {}
            group['bandwidth'][name] = positive(row['rate'], name)
        else:
            if row['unit'] != 'GFLOP/s':
                raise ValueError(f'{path}: fma_compute requires GFLOP/s')
            group['compute'] = positive(row['rate'], name)
    selected = roof_threads if roof_threads is not None else max(groups, default=None)
    group = groups.get(selected, {'bandwidth': {}, 'compute': None})
    return max(group['bandwidth'].values(), default=None), group['compute'], selected


def load_ceilings(path, roof_threads=None):
    return select_ceilings(path, roof_threads)[:2]


def load_kernels(path):
    rows = read_rows(path, FIELDS[:4])
    for row in rows:
        thread_count(row)
        if row['kernel'] not in KERNELS:
            raise ValueError(f'{path}: unknown kernel {row["kernel"]!r}')
        for key in ('arithmetic_intensity', 'measured_gflops'):
            positive(row[key], key)
        if all(row.get(key) for key in ('approx_flops', 'min_ms', 'max_ms')):
            work = positive(row['approx_flops'], 'approx_flops')
            low_time = positive(row['min_ms'], 'min_ms')
            high_time = positive(row['max_ms'], 'max_ms')
            rate = float(row['measured_gflops'])
            if low_time > high_time or not (work / high_time / 1e6 <= rate * 1.000001
                                          and rate <= work / low_time / 1e6 * 1.000001):
                raise ValueError(f'{path}: inconsistent timing range')
    return rows


def atomic_output(destination, writer):
    # A failed rendering leaves the previous complete image/PDF in place.
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + destination.stem + '-',
                                      suffix=destination.suffix, dir=destination.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        writer(temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_summary(path, kernels, bandwidth, compute, roof_threads=None):
    def write(destination):
        with destination.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for row in kernels:
                result = {key: row[key] for key in FIELDS[:4]}
                result.update(roofline_bound='', predicted_bound_type='',
                              threads=thread_count(row), roof_threads=roof_threads or '',
                              bandwidth_gbs=bandwidth if bandwidth is not None else '',
                              compute_gflops=compute if compute is not None else '',
                              attained_percent='')
                if bandwidth is not None and compute is not None:
                    memory = float(row['arithmetic_intensity']) * bandwidth
                    result['attained_percent'] = f'{100 * float(row["measured_gflops"]) / min(memory, compute):.9g}'
                    result.update(roofline_bound=f'{min(memory, compute):.9g}',
                                  predicted_bound_type='memory' if memory < compute else 'compute')
                writer.writerow(result)
    atomic_output(path, write)


def configuration_label(row):
    values = configuration_values(row)
    if row['kernel'] == 'gemv':
        flush = row.get('flush_cache_mib') or values.get('flush_mib', '0')
        return f'GEMV displaced ({flush} MiB)' if values.get('cache') == 'cold' else 'GEMV warm'
    sequence = row.get('sequence') or values.get('S', '?')
    title = 'GEMM' if row['kernel'] == 'gemm' else 'Attention'
    label = f'{title} S={sequence}'
    if values.get('cache') == 'cold':
        label += f' displaced ({values.get("flush_mib", "?")} MiB)'
    return label


def plot_figures(output, kernels, bandwidth, compute, kernel_path, ceiling_path, roof_threads=None):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        import numpy as np
    except ImportError as error:
        requirements = Path(__file__).resolve().parent.parent / 'requirements.txt'
        raise RuntimeError(f'PNG/PDF plotting requires Matplotlib in {sys.executable}. '
                           f'Install dependencies from {requirements}, then reconfigure '
                           'CMake if you change Python environments.') from error

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    thread_counts = sorted({thread_count(row) for row in kernels})
    colors = {1: '#1769aa', 2: '#d17810', 4: '#008577', 8: '#c0392b', 12: '#8055a0'}
    for index, threads in enumerate(thread_counts):
        colors.setdefault(threads, plt.get_cmap('tab20')(index % 20))
    markers = dict(zip(KERNELS, ('o', 's', '^')))
    ridge = compute / bandwidth if bandwidth is not None and compute is not None else None
    intensities = [float(r['arithmetic_intensity']) for r in kernels]
    if ridge is not None:
        intensities.append(ridge)
    xmin = min([0.05] + [value / 2 for value in intensities])
    xmax = max([1024.0] + [value * 2 for value in intensities])
    x = np.geomspace(xmin, xmax, 500)
    rates = [float(r['measured_gflops']) for r in kernels]
    for row in kernels:
        if all(row.get(key) for key in ('approx_flops', 'min_ms', 'max_ms')):
            rates.extend(float(row['approx_flops']) / float(row[key]) / 1e6
                         for key in ('min_ms', 'max_ms'))
    if compute is not None:
        rates.append(compute)
    elif bandwidth is not None:
        rates.append(bandwidth * xmax)
    ymin = min([0.1] + [rate / 2 for rate in rates])
    ymax = max([10.0] + [rate * 2 for rate in rates])
    missing = []
    if bandwidth is None:
        missing.append('bandwidth')
    if compute is None:
        missing.append('compute')
    scope = f'{roof_threads} threads' if roof_threads is not None else 'pending thread scope'
    status = ('Waiting for ' + ' and '.join(missing) if missing else 'Measured common reference roof') + f' | {scope}'

    for kernel in (None, *KERNELS):
        selected = [row for row in kernels if kernel is None or row['kernel'] == kernel]
        fig, ax = plt.subplots(figsize=(11.5, 6.6))
        fig.subplots_adjust(left=0.09, right=0.70, bottom=0.23, top=0.84)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, which='major', color='#e2e8f0', linewidth=0.7)
        ax.set_axisbelow(True)
        if bandwidth is not None:
            ax.plot(x, bandwidth * x, linestyle='--', color='#64748b', linewidth=1.4,
                    label=f'Bandwidth: {bandwidth:.2f} GB/s')
        if compute is not None:
            ax.axhline(compute, linestyle=':', color='#475569', linewidth=1.5,
                       label=f'FMA: {compute:.2f} GFLOP/s')
        if ridge is not None:
            ax.plot(x, np.minimum(compute, bandwidth * x), color='#172033',
                    linewidth=2.3, label='Roofline: min(FMA, AI x BW)')
            ax.axvline(ridge, color='#94a3b8', linestyle=':', linewidth=1,
                       label=f'Ridge: {ridge:.3g} FLOP/byte')
        annotations = {}
        for row in selected:
            ai = float(row['arithmetic_intensity'])
            rate = float(row['measured_gflops'])
            error = None
            if all(row.get(field) for field in ('approx_flops', 'min_ms', 'max_ms')):
                work = float(row['approx_flops'])
                low = work / float(row['max_ms']) / 1e6
                high = work / float(row['min_ms']) / 1e6
                error = [[max(0, rate - low)], [max(0, high - rate)]]
            ax.errorbar(ai, rate, yerr=error, marker=markers[row['kernel']],
                        color=colors[thread_count(row)], markersize=6, linestyle='none',
                        markerfacecolor='none' if configuration_values(row).get('cache') == 'cold'
                                        else colors[thread_count(row)],
                        alpha=0.8, capsize=3, elinewidth=1, zorder=4, label=None)
            # Label each shape/cache group once, at its highest point, rather
            # than stacking the same label at every thread count.
            key = row['kernel'], configuration_label(row)
            if key not in annotations or rate > float(annotations[key]['measured_gflops']):
                annotations[key] = row
        for row in annotations.values():
            ai, rate = float(row['arithmetic_intensity']), float(row['measured_gflops'])
            label = configuration_label(row).replace('GEMM ', '').replace('Attention ', '').replace('GEMV ', '')
            left = row['kernel'] == 'gemv' and configuration_values(row).get('cache') != 'cold'
            offset_y = -14 if row['kernel'] == 'gemm' else 10
            ax.annotate(label, (ai, rate), xytext=(-8 if left else 8, offset_y),
                        textcoords='offset points', ha='right' if left else 'left',
                        fontsize=8, color='#334155',
                        arrowprops={'arrowstyle': '-', 'color': '#94a3b8', 'linewidth': 0.6})
        if not selected:
            ax.text(0.5, 0.43, 'No kernel measurements yet', transform=ax.transAxes,
                    ha='center', color='#64748b')
        ax.set_xlabel('Analytical arithmetic intensity (FLOP/byte)')
        ax.set_ylabel('Measured aggregate throughput (GFLOP/s)')
        title = 'GEMM, GEMV, Attention' if kernel is None else ('Attention (total)' if kernel == 'attention' else kernel.upper())
        fig.suptitle(f'CPU Roofline - {title}', y=0.96, fontsize=17, fontweight='bold')
        fig.text(0.09, 0.885, f'{status} | {len(selected)} measured point(s)', color='#475569')
        handles, legend_labels = ax.get_legend_handles_labels()
        for name in KERNELS:
            if any(row['kernel'] == name for row in selected):
                handles.append(Line2D([], [], marker=markers[name], color='#475569', linestyle='none'))
                legend_labels.append(name.upper() if name != 'attention' else 'Attention')
        if any(row['kernel'] == 'gemv' and configuration_values(row).get('cache') == 'cold' for row in selected):
            handles.append(Line2D([], [], marker='s', color='#475569', markerfacecolor='none', linestyle='none'))
            legend_labels.append('GEMV displaced (open square)')
        for threads in thread_counts:
            if any(thread_count(row) == threads for row in selected):
                handles.append(Line2D([], [], marker='o', color=colors[threads], linestyle='none'))
                legend_labels.append(f'{threads} thread(s)')
        if handles:
            ax.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(1.02, 1),
                      fontsize=8, frameon=False)
        note = ('Each CSV row is one point; repeated runs are retained. Whiskers show min/max timing, not confidence intervals.\n'
                'AI uses analytical minimum bytes; attention uses operation-equivalents. Reference ceilings are not theoretical hardware peaks.\n'
                f'Common roof: {scope}; latest bandwidth pair and FMA at that count. Gap to this roof is not per-thread inefficiency.\n'
                f'Inputs: {kernel_path.name} | {ceiling_path.name}. Thread color; kernel marker; labels show shape/cache state.')
        fig.text(0.09, 0.05, note, fontsize=8, color='#475569', linespacing=1.7)
        stem = output.stem + (f'_{kernel}' if kernel else '')
        try:
            for extension in ('png', 'pdf'):
                destination = output.parent / f'{stem}.{extension}'
                atomic_output(destination, lambda path: fig.savefig(path, dpi=160, facecolor='white'))
        finally:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernels', type=Path, default=Path('results/kernel_results.csv'))
    parser.add_argument('--ceilings', type=Path, default=Path('results/ceilings.csv'))
    parser.add_argument('--output', type=Path, default=Path('results/roofline.csv'),
                        help='Summary CSV; PNG/PDF plots use the same directory and stem')
    parser.add_argument('--roof-threads', type=int,
                        help='Common ceiling thread count; default: largest count present in ceiling CSV')
    parser.add_argument('--csv-only', action='store_true', help='Regenerate the summary without rendering images')
    args = parser.parse_args()
    try:
        if args.roof_threads is not None and args.roof_threads <= 0:
            raise ValueError('--roof-threads must be positive')
        if args.output.suffix.lower() != '.csv':
            raise ValueError('--output must name a .csv summary file')
        if args.output.resolve() in (args.kernels.resolve(), args.ceilings.resolve()):
            raise ValueError('--output must not overwrite an input CSV')
        if args.kernels.resolve() == args.ceilings.resolve():
            raise ValueError('kernel and ceiling CSV paths must differ')
        bandwidth, compute, roof_threads = select_ceilings(args.ceilings, args.roof_threads)
        kernels = load_kernels(args.kernels)
        write_summary(args.output, kernels, bandwidth, compute, roof_threads)
        if not args.csv_only:
            plot_figures(args.output, kernels, bandwidth, compute, args.kernels, args.ceilings, roof_threads)
        print(f'Roofline updated: {args.output}' +
              ('' if args.csv_only else f', {args.output.with_suffix(".png")}, {args.output.with_suffix(".pdf")} (plus per-kernel plots)'),
              file=sys.stderr)
        print(f'  Common roof threads: {roof_threads}; points: {len(kernels)}; bandwidth: {bandwidth if bandwidth is not None else "pending"} GB/s; '
              f'FMA: {compute if compute is not None else "pending"} GFLOP/s', file=sys.stderr)
        if roof_threads is not None and any(thread_count(row) > roof_threads for row in kernels):
            print('  Note: some points use more threads than the selected common ceiling scope.', file=sys.stderr)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'Roofline plotting error: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
