"""Reproducible generic Yosys synthesis; no fabricated technology PPA."""
from collections import Counter
from pathlib import Path
import shutil
import subprocess
import sys

from .config import fingerprint, parameters, read, write


def synthesize(config, rtl_root, out):
    rtl_root, out = Path(rtl_root).resolve(), Path(out).resolve()
    if out.exists():
        raise ValueError("Use a fresh synthesis output directory")
    out.mkdir(parents=True)
    executable = shutil.which("yosys") or str(Path(sys.executable).parent / "yowasp-yosys")
    if not Path(executable).is_file():
        raise RuntimeError("Install yowasp-yosys in the course venv or provide native yosys")
    source = out / "source"
    source.mkdir()
    for path in sorted(rtl_root.glob("*.sv")):
        shutil.copy2(path, source / path.name)
    # Relative paths also work with the WASI filesystem sandbox used by YoWASP.
    script = "\n".join([
        "read_verilog -sv " + " ".join('source/' + p.name for p in sorted(source.glob("*.sv"))),
        "hierarchy -check -top accelerator_top " + " ".join(f"-chparam {k} {v}" for k, v in parameters(config).items()),
        "proc", "flatten", "opt", "memory_dff", "memory_collect", "opt", "check -assert",
        "stat", "write_json coarse_netlist.json",
        "techmap", "opt", "check -assert", "stat", "write_json generic_netlist.json", "",
    ])
    (out / "synthesize.ys").write_text(script)
    with (out / "synthesis.log").open("w") as log:
        subprocess.run([executable, "-s", "synthesize.ys"], cwd=out, stdout=log, stderr=subprocess.STDOUT, check=True)
    coarse = read(out / "coarse_netlist.json")["modules"]["accelerator_top"]["cells"]
    generic = read(out / "generic_netlist.json")["modules"]["accelerator_top"]["cells"]
    memories = []
    for name, cell in coarse.items():
        if cell["type"].startswith("$mem"):
            p = cell["parameters"]
            integer = lambda key: int(p[key], 2) if isinstance(p[key], str) else int(p[key])
            memories.append(dict(name=name, width=integer("WIDTH"), words=integer("SIZE"),
                                 bits=integer("WIDTH")*integer("SIZE"), read_ports=integer("RD_PORTS"),
                                 write_ports=integer("WR_PORTS")))
    report = dict(passed=True, rtl=fingerprint(rtl_root), design=config,
                  yosys=subprocess.check_output([executable, "-V"], text=True).strip(),
                  coarse_cells=dict(Counter(c["type"] for c in coarse.values())),
                  generic_cells=dict(Counter(c["type"] for c in generic.values())),
                  generic_cell_count=sum(c["type"].startswith("$_") for c in generic.values()), inferred_memories=memories,
                  inferred_memory_bits=sum(m["bits"] for m in memories),
                  physical_area_um2=None, physical_power_w=None, timing_closed=False,
                  scope="Actual synthesis of RTL with memories kept abstract. Generic gates are not standard cells; "
                        "memory bits are not SRAM macro area. No liberty, placement, routing, voltage or activity power analysis.")
    write(out / "synthesis.json", report)
    return out / "synthesis.json"
