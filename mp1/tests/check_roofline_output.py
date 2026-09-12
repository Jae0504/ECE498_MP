"""Check automatic Roofline creation, incremental updates, and CSV isolation."""

import csv
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def read(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    cpu, ceiling, plotter = [str(Path(arg).resolve()) for arg in sys.argv[1:]]
    spec = importlib.util.spec_from_file_location('plotter', plotter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='mp1-roofline-') as directory:
        root = Path(directory)

        def run(binary, *args):
            result = subprocess.run([binary, *args], cwd=root, text=True,
                                    capture_output=True, timeout=45)
            assert result.returncode == 0, result.stderr
            assert 'warning: automatic Roofline' not in result.stderr, result.stderr
            assert 'Roofline plotting error' not in result.stderr, result.stderr
            return result

        def check_images(folder):
            for stem in ('roofline', 'roofline_gemm', 'roofline_gemv', 'roofline_attention'):
                assert (folder / (stem + '.png')).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
                assert (folder / (stem + '.pdf')).read_bytes().startswith(b'%PDF-')
            assert not list(folder.glob('.roofline-*'))

        def check_bounds(folder):
            bandwidth, compute = module.load_ceilings(folder / 'ceilings.csv')
            for row in read(folder / 'roofline.csv'):
                ai = float(row['arithmetic_intensity'])
                expected = min(compute, ai * bandwidth)
                assert abs(float(row['roofline_bound']) / expected - 1) < 1e-8

        small = ('--kernel', 'attention', '--seq', '8', '--warmup', '1', '--iterations', '2')
        memory = ('--ceiling', 'bandwidth', '--stream-elements', '4096',
                  '--read-elements', '12288', '--warmup', '1', '--iterations', '2')
        compute = ('--ceiling', 'compute', '--fma-elements', '32',
                   '--fma-repeats', '1000', '--warmup', '1', '--iterations', '2')
        results = root / 'results'

        run(cpu, '--test')
        run(cpu, *small, '--no-save')
        assert not results.exists()
        response = run(ceiling, *memory, '--format', 'csv')
        assert response.stdout == (results / 'ceilings.csv').read_text()
        check_images(results)
        assert read(results / 'roofline.csv') == []
        assert module.load_ceilings(results / 'ceilings.csv')[1] is None
        skeleton = (results / 'roofline.png').read_bytes()

        run(ceiling, *compute)
        check_images(results)
        assert (results / 'roofline.png').read_bytes() != skeleton
        assert read(results / 'roofline.csv') == []
        skeleton = (results / 'roofline.png').read_bytes()

        response = run(cpu, '--kernel', 'gemm', '--seq', '1', '--warmup', '1',
                       '--iterations', '2', '--format', 'csv')
        assert response.stdout == (results / 'kernel_results.csv').read_text()
        assert (results / 'roofline.png').read_bytes() != skeleton
        assert len(read(results / 'roofline.csv')) == 1
        run(cpu, '--kernel', 'gemv', '--warmup', '1', '--iterations', '2')
        run(cpu, *small)
        assert [r['kernel'] for r in read(results / 'roofline.csv')] == ['gemm', 'gemv', 'attention']
        run(cpu, *small)
        assert len(read(results / 'roofline.csv')) == 4
        check_images(results)
        check_bounds(results)

        # Updating the compute ceiling must retain all existing kernel points.
        run(ceiling, *compute)
        assert len(read(results / 'roofline.csv')) == 4
        check_bounds(results)
        saved_plot = (results / 'roofline.png').read_bytes()
        saved_csv = (results / 'kernel_results.csv').read_bytes()
        run(cpu, *small, '--no-save')
        assert (results / 'kernel_results.csv').read_bytes() == saved_csv
        assert (results / 'roofline.png').read_bytes() == saved_plot
        run(cpu, *small, '--no-plot')
        assert len(read(results / 'kernel_results.csv')) == 5
        assert len(read(results / 'roofline.csv')) == 4
        assert (results / 'roofline.png').read_bytes() == saved_plot

        # Explicit counterpart paths and shell metacharacters are literal.
        custom = root / "custom 'series' $(touch SHOULD_NOT_EXIST)"
        custom.mkdir()
        custom_kernels = custom / 'points.csv'
        custom_ceilings = custom / 'limits.csv'
        run(ceiling, *compute, '--output', str(custom_ceilings), '--kernels', str(custom_kernels))
        check_images(custom)
        run(cpu, *small, '--output', str(custom_kernels), '--ceilings', str(custom_ceilings))
        assert len(read(custom / 'roofline.csv')) == 1
        assert read(custom / 'roofline.csv')[0]['roofline_bound'] == ''
        run(ceiling, *memory, '--output', str(custom_ceilings), '--kernels', str(custom_kernels))
        assert read(custom / 'roofline.csv')[0]['roofline_bound'] != ''
        assert not (root / 'SHOULD_NOT_EXIST').exists()
        assert (results / 'roofline.png').read_bytes() == saved_plot

        # Starting with kernels alone creates points, then ceilings fill in.
        fresh = root / 'kernel-first'
        run(cpu, *small, '--output', str(fresh / 'kernel_results.csv'))
        check_images(fresh)
        assert read(fresh / 'roofline.csv')[0]['roofline_bound'] == ''
        run(ceiling, *compute, '--output', str(fresh / 'ceilings.csv'))
        assert read(fresh / 'roofline.csv')[0]['roofline_bound'] == ''
        run(ceiling, *memory, '--output', str(fresh / 'ceilings.csv'))
        check_bounds(fresh)

        # Latest bandwidth pair / compute value, not fastest historical rows.
        selection = root / 'selection.csv'
        selection.write_text('ceiling,rate,unit\n'
                             'memory_bandwidth,100,GB/s\n'
                             'memory_bandwidth_read,120,GB/s\n'
                             'fma_compute,80,GFLOP/s\n'
                             'memory_bandwidth,20,GB/s\n'
                             'memory_bandwidth_read,15,GB/s\n'
                             'fma_compute,10,GFLOP/s\n')
        assert module.load_ceilings(selection) == (20, 10)
        with selection.open('a') as stream:
            stream.write('memory_bandwidth,5,GB/s\n')
        assert module.load_ceilings(selection) == (5, 10)

        # Keep ceiling scopes separate; latest 1-thread rows cannot replace 12.
        selection.write_text('ceiling,configuration,rate,unit\n'
                             'memory_bandwidth,threads=1,100,GB/s\n'
                             'memory_bandwidth_read,threads=1,120,GB/s\n'
                             'fma_compute,threads=1,80,GFLOP/s\n'
                             'memory_bandwidth,threads=12,40,GB/s\n'
                             'memory_bandwidth_read,threads=12,35,GB/s\n'
                             'fma_compute,threads=12,200,GFLOP/s\n'
                             'memory_bandwidth,threads=1,5,GB/s\n'
                             'fma_compute,threads=1,10,GFLOP/s\n')
        assert module.select_ceilings(selection) == (40, 200, 12)
        assert module.select_ceilings(selection, 1) == (5, 10, 1)
        assert module.select_ceilings(selection, 8) == (None, None, 8)
        with selection.open('a') as stream:
            stream.write('memory_bandwidth,threads=16,90,GB/s\n')
        assert module.select_ceilings(selection) == (90, None, 16)
        assert module.select_ceilings(selection, 12) == (40, 200, 12)

        # A rendering/input error must preserve raw results and old images.
        old_image = (custom / 'roofline.png').read_bytes()
        custom_ceilings.write_text('wrong,header\n')
        response = subprocess.run([cpu, *small, '--output', str(custom_kernels),
                                   '--ceilings', str(custom_ceilings)], cwd=root,
                                  capture_output=True, text=True, timeout=45)
        assert response.returncode == 0
        assert 'automatic Roofline update failed' in response.stderr
        assert len(read(custom_kernels)) == 2
        assert (custom / 'roofline.png').read_bytes() == old_image

        # An activated environment overrides the interpreter baked into CMake.
        active = root / 'active environment'
        (active / 'bin').mkdir(parents=True)
        marker = active / 'python-used.txt'
        interpreter = active / 'bin' / 'python3'
        interpreter.write_text(
            f'#!{sys.executable}\n'
            'import os, sys\nfrom pathlib import Path\n'
            f'Path({str(marker)!r}).write_text("active interpreter selected")\n'
            f'os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n'
        )
        interpreter.chmod(0o755)
        active_csv = active / 'results' / 'kernel_results.csv'
        response = subprocess.run([cpu, *small, '--output', str(active_csv)],
                                  cwd=root, capture_output=True, text=True, timeout=45,
                                  env={**os.environ, 'VIRTUAL_ENV': str(active)})
        assert response.returncode == 0, response.stderr
        assert marker.read_text() == 'active interpreter selected'
        assert 'automatic Roofline update failed' not in response.stderr
        check_images(active_csv.parent)

        # A broken active environment must not silently use a system Python.
        response = subprocess.run([cpu, *small, '--output', str(active_csv)],
                                  cwd=root, capture_output=True, text=True, timeout=45,
                                  env={**os.environ, 'VIRTUAL_ENV': str(root / 'missing-venv')})
        assert response.returncode == 0
        assert 'cannot start Roofline plotter with' in response.stderr
        assert len(read(active_csv)) == 2

        # Missing plotting dependencies also leave saved measurements intact.
        no_matplotlib = root / 'no-matplotlib'
        no_matplotlib.mkdir()
        (no_matplotlib / 'matplotlib.py').write_text("raise ImportError('missing Matplotlib test')\n")
        no_dependency_csv = no_matplotlib / 'kernel_results.csv'
        response = subprocess.run([cpu, *small, '--output', str(no_dependency_csv)],
                                  cwd=root, capture_output=True, text=True, timeout=45,
                                  env={**os.environ, 'PYTHONPATH': str(no_matplotlib)})
        assert response.returncode == 0
        assert len(read(no_dependency_csv)) == 1
        assert 'requires Matplotlib' in response.stderr
        assert 'automatic Roofline update failed' in response.stderr
        assert not (no_matplotlib / 'roofline.png').exists()

        # Reject invalid log coordinates and input/output collisions.
        invalid = root / 'invalid.csv'
        invalid.write_text('kernel,configuration,arithmetic_intensity,measured_gflops\n'
                           'gemm,S=1,nan,2\n')
        try:
            module.load_kernels(invalid)
            raise AssertionError('accepted non-finite intensity')
        except ValueError:
            pass
        before = invalid.read_bytes()
        rejected = subprocess.run([sys.executable, plotter, '--kernels', str(invalid),
                                   '--output', str(invalid)], cwd=root, capture_output=True, text=True)
        assert rejected.returncode != 0
        assert invalid.read_bytes() == before
        # Actual mixed-thread updates, plus explicit scope forwarded by both CLIs.
        if len(os.sched_getaffinity(0)) >= 2:
            mixed = root / 'mixed'
            for threads in ('1', '2'):
                for args in (memory, compute):
                    run(ceiling, *args, '--threads', threads, '--no-plot',
                        '--output', str(mixed / 'ceilings.csv'))
                run(cpu, *small, '--threads', threads, '--no-plot',
                    '--output', str(mixed / 'kernel_results.csv'))
            run(cpu, *small, '--threads', '1', '--output', str(mixed / 'kernel_results.csv'))
            summary = read(mixed / 'roofline.csv')
            assert [r['threads'] for r in summary] == ['1', '2', '1']
            assert all(r['roof_threads'] == '2' for r in summary)
            check_bounds(mixed)
            check_images(mixed)
            run(ceiling, *compute, '--threads', '1', '--roof-threads', '1',
                '--output', str(mixed / 'ceilings.csv'))
            assert all(r['roof_threads'] == '1' for r in read(mixed / 'roofline.csv'))
            run(cpu, *small, '--threads', '2', '--roof-threads', '2',
                '--output', str(mixed / 'kernel_results.csv'))
            assert all(r['roof_threads'] == '2' for r in read(mixed / 'roofline.csv'))
            assert module.thread_count({'configuration': 'S=128'}) == 1
            for value in ('0', '-1', '2.5'):
                try:
                    module.thread_count({'configuration': 'threads=' + value})
                    raise AssertionError('accepted invalid thread count')
                except ValueError:
                    pass
        print('Automatic skeletons, point updates, PNG/PDF, latest ceilings, custom paths, and failure recovery passed.')


if __name__ == '__main__':
    main()
