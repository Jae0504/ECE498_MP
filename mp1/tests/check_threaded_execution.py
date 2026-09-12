"""Verify parallel correctness, fixed kernel work, and aggregate ceiling work."""
import csv
import io
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    cpu, ceiling = [str(Path(arg).resolve()) for arg in sys.argv[1:]]
    available = len(os.sched_getaffinity(0))
    counts = [n for n in (1, 2, 4, 8) if n <= available]
    with tempfile.TemporaryDirectory(prefix='mp1-threads-') as folder:
        def run(binary, args, *, error=None, env=None):
            result = subprocess.run([binary, *args], cwd=folder, text=True,
                                    capture_output=True, timeout=45, env=env)
            if error:
                assert result.returncode != 0, result.stdout
                assert error in result.stderr, result.stderr
            else:
                assert result.returncode == 0, result.stderr
            return result

        reference = {}
        for threads in counts:
            run(cpu, ['--test', '--threads', str(threads)])
            for kernel in ('gemm', 'gemv', 'attention'):
                output = run(cpu, ['--kernel', kernel, '--seq', '3', '--threads', str(threads),
                                   '--warmup', '1', '--iterations', '2', '--format', 'csv',
                                   '--no-save', '--flush-cache-mib', '1']).stdout
                row, = csv.DictReader(io.StringIO(output))
                assert f'threads={threads}' in row['configuration']
                assert math.isfinite(float(row['checksum']))
                if threads == 1:
                    reference[kernel] = row
                else:
                    for field in ('approx_flops', 'estimated_minimum_bytes', 'arithmetic_intensity'):
                        assert row[field] == reference[kernel][field]
                    assert math.isclose(float(row['checksum']), float(reference[kernel]['checksum']),
                                        rel_tol=1e-7, abs_tol=1e-7)

            output = run(ceiling, ['--ceiling', 'all', '--threads', str(threads),
                                   '--stream-elements', '257', '--read-elements', '259',
                                   '--compute-mode', 'array', '--fma-elements', '17', '--fma-repeats', '11',
                                   '--warmup', '1', '--iterations', '2', '--format', 'csv',
                                   '--no-save']).stdout
            rows = list(csv.DictReader(io.StringIO(output)))
            assert [float(row['work']) for row in rows] == [257 * 12, 259 * 4, 17 * 11 * 2 * threads]
            for row in rows:
                assert f'threads={threads}' in row['configuration']
                assert float(row['rate']) > 0
                key = row['ceiling']
                value = float(row['checksum']) / (threads if key == 'fma_compute' else 1)
                if threads == 1:
                    reference[key] = value
                else:
                    assert math.isclose(value, reference[key], rel_tol=1e-5, abs_tol=1e-5), row

        # Streaming stores need whole-line partitions. Exercise vector tails and
        # fewer cache lines than workers, including the legacy store comparison.
        for threads in counts:
            for size in (1, 15, 16, 17, 127, 128, 129):
                checksums = []
                for mode in ('cached', 'streaming'):
                    output = run(ceiling, ['--ceiling','bandwidth','--threads',str(threads),
                        '--stream-elements',str(size),'--read-elements',str(size),
                        '--memory-mode',mode,'--warmup','0','--iterations','1',
                        '--format','csv','--no-save']).stdout
                    triad, read = csv.DictReader(io.StringIO(output))
                    assert float(triad['work']) == size*12
                    assert float(read['work']) == size*4
                    assert float(read['checksum']) == size/2
                    assert 'verified=1' in triad['configuration']
                    checksums.append(float(triad['checksum']))
                assert checksums[0] == checksums[1]

        # Test the new default independently from the retained array contract.
        probe = subprocess.run([ceiling, '--ceiling','compute','--accumulators','1',
            '--fma-repeats','7','--warmup','0','--iterations','1','--format','csv','--no-save'],
            capture_output=True,text=True,timeout=20)
        if probe.returncode == 0:
            first, = csv.DictReader(io.StringIO(probe.stdout))
            cfg = dict(item.split('=',1) for item in first['configuration'].split(';'))
            lanes = int(cfg['lanes'])
            for acc in ([1,2,4,8,12,16,24] if lanes==16 else [1,2,4,8,12]):
                expected = None
                for threads in counts:
                    output = run(ceiling, ['--ceiling','compute','--threads',str(threads),
                        '--accumulators',str(acc),'--fma-repeats','7','--warmup','0',
                        '--iterations','1','--format','csv','--no-save']).stdout
                    row, = csv.DictReader(io.StringIO(output))
                    assert float(row['work']) == 2*lanes*acc*7*threads
                    checksum = float(row['checksum'])/threads
                    if expected is None: expected = checksum
                    assert math.isclose(checksum,expected,rel_tol=1e-10)
                    assert 'verified=1' in row['configuration']
            for acc in ('0','3','32'):
                run(ceiling,['--ceiling','compute','--accumulators',acc],error='error:')
        else:
            assert 'register FMA unavailable' in probe.stderr
        run(ceiling,['--compute-mode','invalid'],error='error:')
        run(ceiling,['--memory-mode','invalid'],error='error:')

        for binary in (cpu, ceiling):
            for value in ('0', '-1', '2.5', '2147483648'):
                run(binary, ['--threads', value], error='error:')
            if available >= 2:
                run(binary, ['--threads', '2'],
                    env={**os.environ, 'OMP_THREAD_LIMIT': '1'}, error='OMP_THREAD_LIMIT')
        assert not (Path(folder) / 'results').exists()
        print(f'Parallel reference checks, tails, empty partitions, metadata and work accounting passed: {counts}')


if __name__ == '__main__':
    main()
