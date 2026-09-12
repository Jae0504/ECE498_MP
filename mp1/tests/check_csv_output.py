"""Exercise the benchmark CLI's persistent CSV contract in a fresh directory."""

import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    cpu, roofline = (str(Path(arg).resolve()) for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="mp1-csv-") as directory:
        root = Path(directory)

        def run(binary, *args, error=None):
            result = subprocess.run(
                [binary, *args, "--no-plot"], cwd=root, capture_output=True, text=True,
                timeout=30,
            )
            if error is not None:
                assert result.returncode != 0, result.stdout
                assert error in result.stderr, result.stderr
            else:
                assert result.returncode == 0, result.stderr
            return result

        def rows(path):
            with path.open(newline="") as stream:
                return list(csv.DictReader(stream))

        small = ("--kernel", "attention", "--seq", "8",
                 "--warmup", "1", "--iterations", "2")
        run(cpu, "--test")
        run(cpu, *small, "--no-save")
        assert not (root / "results").exists()

        result = run(cpu, *small, "--format", "csv")
        kernels = root / "results/kernel_results.csv"
        assert kernels.read_text() == result.stdout
        original = kernels.read_text()
        run(cpu, *small)
        records = rows(kernels)
        assert len(records) == 2
        assert kernels.read_text().startswith(original)
        assert kernels.read_text().count("kernel,configuration,") == 1
        for row in records:
            assert row["sequence"] == "8" and row["flush_cache_mib"] == "0"
            assert row["warmups"] == "1" and row["iterations"] == "2"
            assert float(row["approx_flops"]) == 4 * 8**2 * 64 + 6 * 8**2
            assert float(row["estimated_minimum_bytes"]) == 16 * (8 * 64 + 8**2)
            assert all(float(row[name]) > 0 for name in
                       ["median_ms", "qk_ms", "softmax_ms", "pv_ms"])

        saved = kernels.read_bytes()
        custom = root / "separate series/attention.csv"
        result = run(cpu, *small, "--flush-cache-mib", "1", "--output",
                     str(custom), "--format", "csv", "--no-header")
        assert kernels.read_bytes() == saved
        assert rows(custom)[0]["flush_cache_mib"] == "1"
        assert custom.read_text().splitlines()[1:] == result.stdout.splitlines()

        result = run(cpu, *small, "--no-save", "--format", "csv")
        assert len(list(csv.DictReader(io.StringIO(result.stdout)))) == 1
        assert kernels.read_bytes() == saved

        for content, message in [("wrong,header\n", "CSV header mismatch"),
                                 (original.rstrip("\n"), "incomplete last line")]:
            custom.write_text(content)
            run(cpu, *small, "--output", str(custom), error=message)
            assert custom.read_text() == content
        run(cpu, *small, "--output", "", error="nonempty file path")
        run(cpu, *small, "--output", str(kernels / "blocked.csv"), error="error:")

        for kernel in ["gemm", "gemv"]:
            run(cpu, "--kernel", kernel, "--seq", "1", "--warmup", "0",
                "--iterations", "1")
        for row in rows(kernels)[-2:]:
            assert row["sequence"] == "1"
            assert float(row["approx_flops"]) == 2 * 2048 * 5632
            assert row["qk_ms"] == row["softmax_ms"] == row["pv_ms"] == ""

        tiny_ceiling = ("--ceiling", "all", "--stream-elements", "64",
                        "--compute-mode", "array",
                        "--read-elements", "64", "--fma-elements", "32",
                        "--fma-repeats", "10", "--warmup", "0",
                        "--iterations", "2")
        result = run(roofline, *tiny_ceiling, "--format", "csv")
        ceilings = root / "results/ceilings.csv"
        assert ceilings.read_text() == result.stdout
        assert [float(row["work"]) for row in rows(ceilings)] == [768, 256, 640]
        run(roofline, *tiny_ceiling)
        assert len(rows(ceilings)) == 6
        assert ceilings.read_text().count("ceiling,configuration,") == 1
        saved = ceilings.read_bytes()
        run(roofline, *tiny_ceiling, "--no-save")
        assert ceilings.read_bytes() == saved
        custom.unlink()
        run(roofline, *tiny_ceiling, "--output", str(custom))
        assert len(rows(custom)) == 3
        assert ceilings.read_bytes() == saved
        print("CSV creation, append, metadata, redirection, and error checks passed.")


if __name__ == "__main__":
    main()
