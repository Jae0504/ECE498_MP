import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

from .config import ROOT, REPO, digest, fingerprint, parameters, write


def optional_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(request, rtl_root, waves=False):
    from cocotb_tools.runner import get_runner
    rtl_root = Path(rtl_root).resolve()
    out = Path(request["out"]).resolve()
    if out.exists():
        raise ValueError(f"Use a fresh output directory: {out}")
    out.mkdir(parents=True)
    for local_verilator in (REPO / "verilator/bin", ROOT / ".deps/rtl_tools/verilator/bin"):
        if (local_verilator / "verilator_bin").is_file():
            os.environ["PATH"] = str(local_verilator) + os.pathsep + os.environ["PATH"]
            break
    if not shutil.which("verilator"):
        raise RuntimeError("Verilator missing; see RTL_WORKFLOW.md")
    tool_version = subprocess.check_output(["verilator", "--version"], text=True).strip()
    provenance = dict(rtl_root=str(rtl_root), rtl=fingerprint(rtl_root), verilator=tool_version,
                      cocotb=importlib.metadata.version("cocotb"), python=sys.version,
                      packages={name: optional_version(name) for name in ("torch", "numpy", "yowasp-yosys")},
                      harness={p.name: digest(p.read_text()) for p in Path(__file__).parent.glob("*.py")})
    if request["mode"] == "run":
        from .frontend import MINILLAMA_ROOT, source_hash
        provenance["model_source"] = source_hash(MINILLAMA_ROOT)
    key = digest(dict(rtl=provenance["rtl"], parameters=parameters(request["config"]),
                      verilator=tool_version, cocotb=provenance["cocotb"], waves=waves))[:20]
    cache_root = ROOT / ".deps/rtl_build"
    # Instructor RTL must not leave generated C++ outside its private directory.
    for ancestor in (*out.parents, *rtl_root.parents):
        if ancestor.name == "sample_answer":
            cache_root = ancestor / ".rtl_build"
            break
    build = cache_root / key
    request["out"] = str(out)
    request["provenance"] = provenance
    write(out / "request.json", request)
    write(out / "provenance.json", provenance)
    runner = get_runner("verilator")
    start = time.perf_counter()
    runner.build(sources=sorted(rtl_root.glob("*.sv")), hdl_toplevel="accelerator_top",
                 parameters=parameters(request["config"]), build_dir=build,
                 build_args=["-Wno-fatal", "-j", "4"], waves=waves,
                 timescale=("1ns", "1ps"), log_file=out / "build.log")
    build_seconds = time.perf_counter() - start
    # Fresh simulator process per request; only the compiled executable is cached.
    env = dict(ECE498_RTL_REQUEST=str(out / "request.json"),
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    start = time.perf_counter()
    runner.test(test_module="rtl_backend.cocotb_entry", hdl_toplevel="accelerator_top", build_dir=build,
                test_dir=out, extra_env=env, waves=waves, log_file=out / "simulation.log",
                results_xml=out / "tests.xml")
    xml = ET.parse(out / "tests.xml")
    if xml.findall(".//failure") or xml.findall(".//error") or not xml.findall(".//testcase"):
        raise RuntimeError(f"RTL validation failed; inspect {out / 'simulation.log'}")
    expected = "summary.json" if request["mode"] == "run" else "validation.json"
    if not (out / expected).is_file():
        raise RuntimeError(f"Simulator did not produce {expected}")
    write(out / "host_timing.json", dict(build_wall_seconds=build_seconds,
                                        simulation_and_validation_wall_seconds=time.perf_counter()-start,
                                        used_for_accelerator_latency=False))
    return out / expected
