"""Integration regression and fault-injection check for the RTL course harness."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rtl_backend.config import read, write


def launch(*args):
    return subprocess.run([sys.executable, str(ROOT / "rtl_run.py"), *map(str, args)], capture_output=True, text=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rtl-root", type=Path, default=ROOT / "rtl")
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = read(ROOT / "configs/rtl_baseline.json")
    # Non-power-of-two banks/lanes, tiny panels, wait states, repeated transactions.
    cfg.update(array_rows=3, array_cols=5, a_buffer_words=21, b_buffer_words=45,
               memory_lanes=3, memory_wait_cycles=2)
    write(out / "design.json", cfg)
    result = launch("check", "--design", out / "design.json", "--rtl-root", args.rtl_root, "--out", out / "boundaries")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    broken = out / "broken_rtl"
    shutil.copytree(args.rtl_root, broken)
    pe = broken / "pe.sv"
    code = pe.read_text()
    if code.count("assign product = a * b;") != 1:
        raise RuntimeError("Fault injector expects baseline-compatible pe.sv; customize it for the changed datapath")
    pe.write_text(code.replace("assign product = a * b;", "assign product = a * b + 1;"))
    result = launch("check", "--design", out / "design.json", "--rtl-root", broken, "--out", out / "fault_rejected")
    log = out / "fault_rejected/simulation.log"
    if result.returncode == 0 or not log.is_file() or "Arrays are not equal" not in log.read_text():
        raise AssertionError("Corrupt RTL must fail the independent integer scoreboard, not just fail to compile")
    result = launch("run", "--design", out / "design.json", "--rtl-root", broken, "--out", out / "model_fault_rejected")
    log = out / "model_fault_rejected/simulation.log"
    if result.returncode == 0 or not log.is_file() or "Arrays are not equal" not in log.read_text():
        raise AssertionError("The live model path must also reject corrupt RTL outputs")
    write(out / "verification.json", dict(passed=True, boundary_and_wait_state_tests=True,
                                         corrupt_rtl_numerical_result_rejected=True,
                                         corrupt_rtl_in_live_model_rejected=True))
    print(out / "verification.json")


if __name__ == "__main__":
    main()
