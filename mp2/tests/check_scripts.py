#!/usr/bin/env python3
"""Test experiment orchestration with fake tools; this does not test CUDA kernels."""

import csv
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


class ScriptWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="mp2 workflow ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.scripts = self.root / "scripts"
        shutil.copytree(SCRIPTS, self.scripts)
        self.build = self.root / "build"
        self.results = self.root / "results"
        self.bin = self.root / "bin"
        self.build.mkdir()
        self.bin.mkdir()
        self.results.mkdir()
        self.env = os.environ.copy()
        for name in ("CUDA_HOME", "CUDAToolkit_ROOT", "CUDA_ARCHITECTURES",
                     "INCLUDE_LARGE", "SKIP_BASELINE_RUN"):
            self.env.pop(name, None)
        self.env.update(
            PATH=str(self.bin) + os.pathsep + os.defpath,
            BUILD_DIR=str(self.build), RESULT_DIR=str(self.results),
            NVCC_BIN=str(self.bin / "nvcc"),
        )
        self.executable(self.bin / "nvcc", """
if [ "$1" = --list-gpu-code ]; then printf 'sm_90\\nsm_121\\n';
else printf 'fake CUDA compiler\\n'; fi
""")
        self.executable(self.bin / "cmake", 'exit "${MP2_TEST_CMAKE_STATUS:-0}"\n')
        # Environment inventory is independent of the workflow under test.
        self.executable(self.scripts / "inspect_environment.sh",
                        'printf "fake environment\\n" > "$1"\n')
        self.executable(self.build / "gpu_bench", """
if [ "${MP2_TEST_MODE:-}" = unavailable ]; then exit 77; fi
case "$1" in
  --test) [ "${MP2_TEST_MODE:-}" != fast_failure ]; exit $? ;;
  --test-required) [ "${MP2_TEST_MODE:-}" != required_failure ]; exit $? ;;
esac
if [ "${MP2_TEST_MODE:-}" = timing_failure ] && [ "$2" = attention ]; then
  exit 1
fi
case " $* " in *' --no-header '*) ;; *) printf 'kernel,median_ms\\n' ;; esac
printf '%s,1.0\\n' "$2"
""")

    @staticmethod
    def executable(path, body):
        path.write_text("#!/bin/sh\n" + body)
        path.chmod(0o755)

    def run_script(self, name, **overrides):
        return subprocess.run(
            ["bash", str(self.scripts / name)],
            env=dict(self.env, **overrides), cwd=self.root,
            capture_output=True, text=True, timeout=30,
        )

    def assert_no_results(self):
        self.assertFalse((self.results / "kernel_results.csv").exists())
        self.assertEqual(list(self.results.glob(".kernel_results.*")), [])

    def test_success_and_baseline_preservation(self):
        result = self.run_script("run_gpu.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        csv_path = self.results / "kernel_results.csv"
        original = csv_path.read_bytes()
        with csv_path.open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["kernel"] for row in rows],
                         ["gemm", "gemm", "gemv", "attention", "attention"])
        status = (self.results / "runtime_status.txt").read_text()
        self.assertIn("status=success", status)
        result = self.run_script("run_gpu.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fresh RESULT_DIR", result.stderr)
        self.assertEqual(csv_path.read_bytes(), original)
        self.assertEqual((self.results / "runtime_status.txt").read_text(), status)

    def test_failed_correctness_or_timing_never_publishes_csv(self):
        for mode in ("fast_failure", "required_failure", "timing_failure"):
            with self.subTest(mode=mode):
                result = self.run_script("run_gpu.sh", MP2_TEST_MODE=mode)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assert_no_results()
                self.assertIn("status=failure",
                              (self.results / "runtime_status.txt").read_text())

    def test_configuration_failure_is_recorded(self):
        result = self.run_script("run_gpu.sh", MP2_TEST_CMAKE_STATUS="2")
        self.assertEqual(result.returncode, 2)
        self.assert_no_results()
        self.assertIn("stage=configure",
                      (self.results / "runtime_status.txt").read_text())

    def test_unavailable_gpu_is_never_success(self):
        for script, status_path in (
            ("run_gpu.sh", "runtime_status.txt"),
            ("validate_gpu.sh", "sanitizer/status.txt"),
            ("profile_gpu.sh", "profile/ncu_status.txt"),
        ):
            with self.subTest(script=script):
                result = self.run_script(script, MP2_TEST_MODE="unavailable")
                self.assertEqual(result.returncode, 77, result.stdout + result.stderr)
                self.assertIn("status=unavailable",
                              (self.results / status_path).read_text())
                self.assert_no_results()

    def test_profile_checks_current_runtime_despite_stale_status(self):
        (self.results / "runtime_status.txt").write_text("status=success\n")
        result = self.run_script("profile_gpu.sh", SKIP_BASELINE_RUN="1",
                                 MP2_TEST_MODE="unavailable")
        self.assertEqual(result.returncode, 77, result.stdout + result.stderr)
        self.assertTrue((self.results / "profile/PROFILE_DEFERRED.txt").exists())

    def test_incompatible_compiler_override_is_not_replaced(self):
        self.executable(self.bin / "nvcc", """
if [ "$1" = --list-gpu-code ]; then printf 'sm_90\\n';
else printf 'fake old CUDA compiler\\n'; fi
""")
        command = ["bash", "-c",
                   'source "$1"; discover_cuda_tool nvcc NVCC_BIN',
                   "_", str(self.scripts / "cuda_tools.sh")]
        result = subprocess.run(command, env=self.env, capture_output=True,
                                text=True, timeout=10)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Required CUDA architectures: 121", result.stderr)
        result = subprocess.run(command, env=dict(self.env, CUDA_ARCHITECTURES="90"),
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(self.bin / "nvcc"))

    def test_profiler_permission_error_on_stdout_is_recorded_as_failure(self):
        self.executable(self.bin / "ncu", """
case "$1" in
  --version) printf 'fake Nsight Compute\\n'; exit 0 ;;
  --list-sets|--list-sections) exit 0 ;;
esac
printf '==ERROR== ERR_NVGPUCTRPERM - GPU counter access denied\\n'
exit 1
""")
        self.executable(self.bin / "nsys", """
if [ "$1" = --version ]; then printf 'fake Nsight Systems\\n'; exit 0; fi
while [ "$#" -gt 0 ]; do
  if [ "$1" = -o ]; then
    printf 'fake report\\n' > "$2.nsys-rep"
    exit 0
  fi
  shift
done
exit 1
""")
        result = self.run_script("profile_gpu.sh", SKIP_BASELINE_RUN="1",
                                 NCU_BIN=str(self.bin / "ncu"),
                                 NSYS_BIN=str(self.bin / "nsys"))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        status = (self.results / "profile/ncu_status.txt").read_text()
        for case in ("ncu_gemm_s512", "ncu_gemv", "ncu_attention_s128"):
            self.assertIn(case + "_counter_permission=restricted", status)
            self.assertIn(case + "_report=missing", status)
        self.assertIn("status=failure", status)
        self.assertIn("status=success",
                      (self.results / "profile/nsys_status.txt").read_text())
        self.assertIn("GPU profiling incomplete", result.stderr)


if __name__ == "__main__":
    unittest.main()
