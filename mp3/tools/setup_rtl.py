"""Install the public RTL tools locally; no system packages are installed."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PINS = {"verilator": ("https://github.com/verilator/verilator.git", "770b7d009e59071311d4173f8c86b4ac6085f1e9"),
        "cocotb": ("https://github.com/cocotb/cocotb.git", "ae55ced614b3f4ba89d77da5bd635024546d0737")}


def run(args, cwd=None, env=None):
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def source(name):
    path = REPO / name
    url, revision = PINS[name]
    if (REPO / ".git").exists():
        # A normal clone creates empty submodule directories. A directory alone
        # is not evidence that the pinned tool sources have been initialized.
        if not (path / ".git").exists():
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(f"Preserving unversioned tool directory: {path}")
            run(["git", "submodule", "update", "--init", "--", name], cwd=REPO)
    else:
        path = ROOT / ".deps/rtl_tools" / name
        if not (path / ".git").exists():
            if path.exists() and any(path.iterdir()):
                raise RuntimeError(f"Preserving unversioned tool directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
            run(["git", "init", path])
            run(["git", "fetch", "--depth", "1", url, revision], cwd=path)
            run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=path)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
    if actual != revision:
        raise RuntimeError(f"{name} revision {actual} differs from the course pin {revision}; preserve local work before updating")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=path).returncode:
        raise RuntimeError(f"{name} has source edits; preserve them before using the course tool baseline")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-install", action="store_true", help="Verify already installed tools only")
    args = p.parse_args()
    if not args.skip_install:
        required = ["git", "g++", "make", "autoconf", "flex", "bison"]
        missing = [name for name in required if not shutil.which(name)]
        if missing:
            raise SystemExit("Provision these build tools first: " + ", ".join(missing))
        verilator, cocotb_source = source("verilator"), source("cocotb")
        env = os.environ.copy()
        sysroot = ROOT / ".deps/rtl_sysroot"
        includes = sysroot / "usr/include"
        python_include = Path(sysconfig.get_path("include"))
        packages = []
        if not Path("/usr/include/FlexLexer.h").is_file() and not (includes / "FlexLexer.h").is_file():
            packages.append("libfl-dev")
        if not (python_include / "Python.h").is_file():
            python_include = includes / f"python{sys.version_info.major}.{sys.version_info.minor}"
            if not (python_include / "Python.h").is_file():
                packages.append(f"libpython{sys.version_info.major}.{sys.version_info.minor}-dev")
        if packages:
            if not shutil.which("apt") or not shutil.which("dpkg-deb"):
                raise SystemExit("Provide FlexLexer.h and matching Python development headers; automatic local extraction requires Debian/Ubuntu apt.")
            downloads = sysroot / "packages"
            downloads.mkdir(parents=True, exist_ok=True)
            run(["apt", "download", *packages], cwd=downloads)
            for package in downloads.glob("*.deb"):
                run(["dpkg-deb", "-x", package, sysroot])
        env["CPATH"] = str(includes) + (os.pathsep+env["CPATH"] if env.get("CPATH") else "")
        if not (verilator / "bin/verilator_bin").is_file():
            run(["autoconf"], cwd=verilator, env=env)
            run(["./configure", "--enable-ccwarn"], cwd=verilator, env=env)
            run(["make", "-C", "src", "-j4", "opt"], cwd=verilator, env=env)
        env["CMAKE_ARGS"] = f"-DPython_INCLUDE_DIR={python_include} -DPython_EXECUTABLE={sys.executable}"
        run([sys.executable, "-m", "pip", "install", cocotb_source, "yowasp-yosys==0.70.0.0.post1259"], env=env)
    from cocotb.task import bridge, resume
    from cocotb_tools.runner import get_runner
    for path in (REPO / "verilator/bin/verilator", ROOT / ".deps/rtl_tools/verilator/bin/verilator"):
        if path.is_file():
            run([path, "--version"])
            break
    else:
        run(["verilator", "--version"])
    print("cocotb", importlib.metadata.version("cocotb"))
    run([Path(sys.executable).parent / "yowasp-yosys", "-V"])
    print("RTL tools ready. Run: mp3/.venv/bin/python mp3/rtl_run.py check --out mp3/results/rtl_check")


if __name__ == "__main__":
    main()
