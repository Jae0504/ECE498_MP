#!/usr/bin/env python3
"""Extract function-scoped Cachegrind counts for the educational kernels."""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 5632
HEAD_DIM = 64

TARGET_FUNCTIONS = {
    "tlmp::gemm_baseline(": ("gemm", "total"),
    "tlmp::gemv_baseline(": ("gemv", "total"),
    "tlmp::attention_qk_baseline(": ("attention", "qk_scale"),
    "tlmp::softmax_rows_baseline(": ("attention", "softmax"),
    "tlmp::attention_pv_baseline(": ("attention", "pv"),
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pattern", default="teaching_*.out")
    parser.add_argument(
        "--include-profile",
        action="append",
        default=[],
        help="exact file name to include; may be repeated (overrides --pattern)",
    )
    return parser.parse_args()


def parse_cache_size(lines, level):
    expression = re.compile(
        rf"^desc: {re.escape(level)} cache:\s+(\d+) B, (\d+) B, (\d+)-way"
    )
    for line in lines:
        match = expression.match(line)
        if match:
            return tuple(int(value) for value in match.groups())
    return (0, 0, 0)


def parse_command(lines):
    command = next((line[5:] for line in lines if line.startswith("cmd: ")), "")
    kernel_match = re.search(r"(?:^|\s)--kernel\s+(\w+)", command)
    sequence_match = re.search(r"(?:^|\s)--seq\s+(\d+)", command)
    flush_match = re.search(r"(?:^|\s)--flush-cache-mib\s+(\d+)", command)
    if kernel_match is None:
        raise ValueError(f"cannot find --kernel in Cachegrind command: {command}")
    return (
        kernel_match.group(1),
        int(sequence_match.group(1)) if sequence_match else 512,
        int(flush_match.group(1)) if flush_match else 0,
    )


def phase_flops(kernel, phase, sequence):
    if kernel == "gemm":
        return 2 * sequence * HIDDEN_SIZE * INTERMEDIATE_SIZE
    if kernel == "gemv":
        return 2 * HIDDEN_SIZE * INTERMEDIATE_SIZE
    if phase == "qk_scale":
        return 2 * sequence * sequence * HEAD_DIM + sequence * sequence
    if phase == "softmax":
        return 5 * sequence * sequence
    if phase == "pv":
        return 2 * sequence * sequence * HEAD_DIM
    raise ValueError(f"unknown kernel/phase: {kernel}/{phase}")


def selected_function(name):
    for marker, label in TARGET_FUNCTIONS.items():
        if marker in name:
            return label
    return None


def parse_file(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    event_line = next((line for line in lines if line.startswith("events: ")), None)
    if event_line is None:
        raise ValueError(f"missing Cachegrind event header in {path}")
    events = event_line.split()[1:]
    required = {"Ir", "Dr", "Dw", "D1mr", "D1mw", "DLmr", "DLmw"}
    if not required.issubset(events):
        raise ValueError(f"missing required events in {path}: {sorted(required - set(events))}")

    counts = defaultdict(lambda: defaultdict(int))
    current = None
    for line in lines:
        if line.startswith("fn="):
            current = selected_function(line[3:])
            continue
        if current is None or not line or not (line[0].isdigit() or line[0] in "+-"):
            continue
        fields = line.split()
        if len(fields) != len(events) + 1:
            continue
        try:
            values = [int(value) for value in fields[1:]]
        except ValueError:
            continue
        for event, value in zip(events, values):
            counts[current][event] += value

    kernel, sequence, flush_mib = parse_command(lines)
    i1_size, i1_line, i1_assoc = parse_cache_size(lines, "I1")
    d1_size, d1_line, d1_assoc = parse_cache_size(lines, "D1")
    ll_size, ll_line, ll_assoc = parse_cache_size(lines, "LL")
    geometry = {
        "i1_bytes": i1_size,
        "i1_associativity": i1_assoc,
        "d1_bytes": d1_size,
        "d1_associativity": d1_assoc,
        "ll_bytes": ll_size,
        "ll_associativity": ll_assoc,
        "cache_line_bytes": ll_line or d1_line or i1_line,
    }

    rows = []
    for (function_kernel, phase), event_counts in counts.items():
        if function_kernel != kernel:
            continue
        rows.append(
            make_row(path, kernel, sequence, flush_mib, phase,
                     phase_flops(kernel, phase, sequence), event_counts, geometry)
        )

    if kernel == "attention" and rows:
        total_counts = defaultdict(int)
        for row_counts in counts.values():
            for event, value in row_counts.items():
                total_counts[event] += value
        total_flops = sum(int(row["approx_flops"]) for row in rows)
        rows.append(
            make_row(path, kernel, sequence, flush_mib, "total", total_flops,
                     total_counts, geometry)
        )
    return rows


def make_row(path, kernel, sequence, flush_mib, phase, flops, counts, geometry):
    line_bytes = geometry["cache_line_bytes"]
    ll_read_bytes = counts["DLmr"] * line_bytes

    def ratio(numerator, denominator):
        return "" if denominator == 0 else f"{numerator / denominator:.9g}"

    configuration = "tokens=1" if kernel == "gemv" else f"S={sequence}"
    return {
        "profile": path.name,
        "kernel": kernel,
        "configuration": configuration,
        "cache_state": "cold" if flush_mib else "warm",
        "flush_mib": flush_mib,
        "phase": phase,
        "approx_flops": flops,
        "Ir": counts["Ir"],
        "Dr": counts["Dr"],
        "Dw": counts["Dw"],
        "D1mr": counts["D1mr"],
        "D1mw": counts["D1mw"],
        "DLmr": counts["DLmr"],
        "DLmw": counts["DLmw"],
        "l1_read_miss_rate": ratio(counts["D1mr"], counts["Dr"]),
        "ll_read_miss_rate": ratio(counts["DLmr"], counts["Dr"]),
        "simulated_ll_read_bytes": ll_read_bytes,
        "flops_per_simulated_ll_read_byte": ratio(flops, ll_read_bytes),
        **geometry,
    }


def main():
    args = arguments()
    rows = []
    paths = (
        [args.input / name for name in args.include_profile]
        if args.include_profile
        else sorted(args.input.glob(args.pattern))
    )
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"missing Cachegrind profile: {path}")
        rows.extend(parse_file(path))
    if not rows:
        raise SystemExit(f"no kernel records found in {args.input}/{args.pattern}")

    fields = [
        "profile", "kernel", "configuration", "cache_state", "flush_mib",
        "phase", "approx_flops", "Ir", "Dr", "Dw", "D1mr", "D1mw",
        "DLmr", "DLmw", "l1_read_miss_rate", "ll_read_miss_rate",
        "simulated_ll_read_bytes", "flops_per_simulated_ll_read_byte",
        "i1_bytes", "i1_associativity", "d1_bytes", "d1_associativity",
        "ll_bytes", "ll_associativity", "cache_line_bytes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} function-scoped rows to {args.output}")


if __name__ == "__main__":
    main()
