"""Evaluate a declared, fixed-budget MP3 parameter study with the shared RTL."""
import math
from pathlib import Path
import re
import subprocess
import sys

from .config import ROOT, fingerprint, read, validate, write
from .reporting import compare

WORKLOADS = ("prefill", "decode")


def budget(config):
    return (config["array_rows"]*config["array_cols"],
            config["a_buffer_words"]+config["b_buffer_words"],
            config["memory_lanes"], config["memory_wait_cycles"], config["clock_mhz"])


def prepare_plan(baseline, plan):
    validate(baseline)
    if set(plan) != {"weights", "candidates"} or set(plan["weights"]) != set(WORKLOADS):
        raise ValueError("Candidate file needs weights for prefill/decode and a candidates list")
    weights = plan["weights"]
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w <= 0
           for w in weights.values()) or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("Workload weights must be positive, finite, and sum to one")
    if not isinstance(plan["candidates"], list) or not plan["candidates"]:
        raise ValueError("Provide at least one candidate")
    designs = [("baseline", dict(baseline))]
    names = {"baseline"}
    for candidate in plan["candidates"]:
        if set(candidate) != {"name", "parameters"} or not isinstance(candidate["parameters"], dict):
            raise ValueError("Each candidate needs name and parameters")
        name = candidate["name"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in names:
            raise ValueError("Use distinct lowercase candidate names; baseline is reserved")
        design = validate(dict(baseline, **candidate["parameters"]))
        if budget(design) != budget(baseline):
            raise ValueError(f"{name} changes PE count, total SRAM, bandwidth, memory wait, or clock")
        if any(design == prior for _, prior in designs):
            raise ValueError(f"{name} duplicates an existing configuration")
        names.add(name)
        designs.append((name, design))
    return dict(weights), designs


def rank(entries, weights):
    for entry in entries:
        entry["objective"] = sum(weights[w]*entry["metrics"][w]["request_latency_us"] /
                                 entries[0]["metrics"][w]["request_latency_us"] for w in WORKLOADS)
    return min(entries, key=lambda entry: entry["objective"])


def execute(args):
    weights, designs = prepare_plan(read(args.baseline), read(args.candidates))
    model = read(args.model)
    workloads = {w: read(ROOT / f"configs/workload_{w}.json") for w in WORKLOADS}
    checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if checkpoint and not checkpoint.is_file():
        raise ValueError(f"Missing checkpoint: {checkpoint}")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rtl = fingerprint(ROOT / "rtl")
    write(out / "objective.json", dict(weights=weights,
          objective="weighted sum of baseline-normalized request latency; lower is better",
          budget=dict(pes=budget(designs[0][1])[0], input_sram_bytes=2*budget(designs[0][1])[1],
                      memory_lanes=designs[0][1]["memory_lanes"],
                      memory_wait_cycles=designs[0][1]["memory_wait_cycles"],
                      clock_mhz=designs[0][1]["clock_mhz"]), rtl=rtl))
    write(out / "candidates.json", {"weights": weights, "designs": dict(designs)})
    write(out / "model.json", model)
    for w, workload in workloads.items():
        write(out / f"workload_{w}.json", workload)

    def launch(*command):
        subprocess.run([sys.executable, str(ROOT / "rtl_run.py"), *map(str, command)], check=True)

    entries = []
    for name, config in designs:
        directory = out / name
        directory.mkdir()
        design = directory / "design.json"
        write(design, config)
        launch("check", "--design", design, "--out", directory / "check")
        metrics = {}
        for w in WORKLOADS:
            command = ["run", "--design", design, "--model", out / "model.json",
                       "--workload", out / f"workload_{w}.json", "--out", directory / w]
            if checkpoint:
                command += ["--checkpoint", checkpoint]
            launch(*command)
            summary = read(directory / w / "summary.json")
            if not summary["validation"]["passed"] or summary["rtl_sha256"] != rtl["sha256"]:
                raise ValueError("Candidate validation failed or the shared RTL changed")
            metrics[w] = {key: summary[key] for key in ("ttft_us", "mean_tpot_us", "request_latency_us")}
        entries.append(dict(name=name, design=config, metrics=metrics))
    if fingerprint(ROOT / "rtl") != rtl:
        raise ValueError("Shared RTL changed during exploration")
    for w in WORKLOADS:
        compare([out / name / w for name, _ in designs], out / f"comparison_{w}")
    best = rank(entries, weights)
    write(out / "exploration.json", entries)
    write(out / "selected.json", best)
    write(out / "best_design.json", best["design"])
    lines = ["# MP3 RTL parameter study", "",
             "Objective: weighted baseline-normalized request latency. Lower is better.", "",
             f"Weights: prefill={weights['prefill']}, decode={weights['decode']}.", "",
             "| Design | Array | Prefill TTFT (us) | Decode TTFT (us) | Mean TPOT (us) | Decode request (us) | Objective |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for e in entries:
        p, d, c = e["metrics"]["prefill"], e["metrics"]["decode"], e["design"]
        lines.append(f"| {e['name']} | {c['array_rows']}x{c['array_cols']} | {p['ttft_us']:.3f} | "
                     f"{d['ttft_us']:.3f} | {d['mean_tpot_us']:.3f} | {d['request_latency_us']:.3f} | {e['objective']:.6f} |")
    lines += ["", f"Selected: `{best['name']}`. Configuration: `best_design.json`.", "",
              "Times cover the simulated RTL matrix path at an assumed clock. All candidates remain visible; "
              "the selected configuration need not minimize every metric."]
    (out / "sweep.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
    return out / "sweep.md"
