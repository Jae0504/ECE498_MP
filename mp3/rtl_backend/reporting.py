import csv
from pathlib import Path

from .config import read, write


def csv_write(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(request, trace, validation, calls, wall_seconds):
    config, out = request["config"], Path(request["out"])
    counters = ("cycles", "macs", "compute_cycles", "memory_read_bytes", "memory_write_bytes",
                "buffer_write_bytes", "buffer_read_bytes", "input_wait_port_cycles", "output_wait_cycles")
    totals = {key: sum(c[key] for c in calls) for key in counters}
    timeline, cumulative = [], 0
    for step in range(trace["workload"]["new_tokens"]):
        cycles = sum(c["cycles"] for c in calls if c["step"] == step)
        cumulative += cycles
        timeline.append(dict(token=step+1, phase="prefill" if step==0 else "decode", cycles=cycles,
                             latency_us=cycles/config["clock_mhz"], cumulative_us=cumulative/config["clock_mhz"]))
    prefill = timeline[0]["latency_us"]
    decode = sum(t["latency_us"] for t in timeline[1:])
    # Dimensionless teaching proxy: no process, voltage, capacitance or cell library.
    # Publish every coefficient; never present it as W, pJ or um^2.
    coefficients = dict(mac=1, buffer_byte=2, external_memory_byte=50)
    energy_score = (totals["macs"] + 2*(totals["buffer_read_bytes"]+totals["buffer_write_bytes"])
                    + 50*(totals["memory_read_bytes"]+totals["memory_write_bytes"]))
    summary = dict(execution_mode="rtl_int16_acc64", comparison_id=trace["comparison_id"],
                   design=config, rtl_sha256=request["provenance"]["rtl"]["sha256"],
                   totals=totals, ttft_us=prefill, mean_tpot_us=decode/(len(timeline)-1) if len(timeline)>1 else None,
                   request_latency_us=prefill+decode, generated_token_ids=trace["generated_token_ids"],
                   generated_text=trace["generated_text"], validation=validation,
                   useful_mac_utilization=totals["macs"]/(totals["cycles"]*config["array_rows"]*config["array_cols"]),
                   resources=dict(pes=config["array_rows"]*config["array_cols"],
                                  input_sram_bytes=2*(config["a_buffer_words"]+config["b_buffer_words"]),
                                  accumulator_bits=64*config["array_rows"]*config["array_cols"],
                                  a_port_bytes_per_cycle=2*config["memory_lanes"],
                                  b_port_bytes_per_cycle=2*config["memory_lanes"], c_port_bytes_per_cycle=8),
                   cost_proxy=dict(coefficients=coefficients, energy_score=energy_score,
                                   average_activity_cost_per_cycle=energy_score/totals["cycles"],
                                   physical_units=False, leakage_clock_control_interconnect_included=False),
                   physical_power_w=None, physical_area_um2=None, timing_closed=False,
                   model_and_validation_wall_seconds=wall_seconds,
                   scope="Serialized matrix operations only; includes RTL tiling, input memory wait and output transfer. "
                         "Excludes host/Python overhead, quantization, transpose/layout conversion, softmax, RoPE, normalization, "
                         "activations, residuals, embedding and inter-kernel launch gaps.")
    write(out / "summary.json", summary)
    write(out / "design.json", config)
    csv_write(out / "kernels.csv", calls)
    csv_write(out / "token_timeline.csv", timeline)
    (out / "analysis.md").write_text(
        f"# RTL matrix-path results\n\nTTFT: {prefill:.3f} us; mean TPOT: {summary['mean_tpot_us']} us; "
        f"request: {summary['request_latency_us']:.3f} us.\n\n"
        f"{totals['cycles']} measured RTL cycles; {len(calls)} independently checked integer matrices. "
        f"Generated token IDs: {trace['generated_token_ids']}. Seeded random weights are a functional demo, not a language-quality benchmark.\n\n"
        f"{summary['scope']} Clock is assumed, not proven by STA. TTFT/TPOT are the matrix contribution, not whole-system latency.\n\n"
        "Power and physical area are unavailable without technology characterization. `cost_proxy` is an uncalibrated "
        "dimensionless activity score; use `synth` for actual Yosys structural counts. These are different evidence.\n")


def compare(paths, out):
    paths, out = [Path(p) for p in paths], Path(out)
    runs = [read(p / "summary.json") for p in paths]
    if len(runs) < 2:
        raise ValueError("Compare at least two runs")
    if any(r["comparison_id"] != runs[0]["comparison_id"] or r["execution_mode"] != "rtl_int16_acc64" for r in runs):
        raise ValueError("Comparisons require the same workload, model, weights, quantization and RTL protocol")
    if any(not r["validation"]["passed"] or r["generated_token_ids"] != runs[0]["generated_token_ids"] for r in runs):
        raise ValueError("Numerical validation or generated-token consistency failed")
    if out.exists():
        raise ValueError("Use a fresh comparison output directory")
    out.mkdir(parents=True)
    rows = []
    for p, r in zip(paths, runs):
        rows.append(dict(run=str(p), cycles=r["totals"]["cycles"], ttft_us=r["ttft_us"],
                         mean_tpot_us=r["mean_tpot_us"], request_us=r["request_latency_us"],
                         request_speedup=runs[0]["request_latency_us"]/r["request_latency_us"],
                         ttft_speedup=runs[0]["ttft_us"]/r["ttft_us"],
                         memory_read_bytes=r["totals"]["memory_read_bytes"],
                         energy_score=r["cost_proxy"]["energy_score"]))
    csv_write(out / "comparison.csv", rows)
    write(out / "comparison.json", rows)
    lines = ["# RTL matrix-path comparison", "", "| Run | TTFT (us) | Mean TPOT (us) | Request (us) | Speedup |", "|---|---:|---:|---:|---:|"]
    for p, r in zip(paths, rows):
        tpot = f"{r['mean_tpot_us']:.3f}" if r["mean_tpot_us"] is not None else "N/A"
        lines.append(f"| {p.parent.name}/{p.name} | {r['ttft_us']:.3f} | {tpot} | {r['request_us']:.3f} | {r['request_speedup']:.3f}x |")
    lines.extend(["", "Matrix-only simulated time; assumed clock. No claim of measured CPU speedup, physical power or timing closure."])
    (out / "comparison.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))
