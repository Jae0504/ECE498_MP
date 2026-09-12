#!/usr/bin/env python3
"""Parse perf stat -x ';' files and derive basic architecture ratios."""

import argparse
import csv
import re
from pathlib import Path


CONFIGURATIONS = {
    "gemm_s512": ("gemm", "S=512"),
    "gemv": ("gemv", "tokens=1"),
    "attention_s128": ("attention", "S=128;D=64"),
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--events-output", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    return parser.parse_args()


def parse_number(text):
    cleaned = text.strip().replace(",", "")
    if not cleaned or cleaned.startswith("<"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def label_for(path):
    stem = re.sub(r"_(core|cache|topdown)$", "", path.stem)
    return CONFIGURATIONS.get(stem)


def main():
    args = arguments()
    rows = []
    values = {}
    for path in sorted(args.input.glob("*.txt")):
        label = label_for(path)
        if label is None:
            continue
        kernel, configuration = label
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split(";")
            if len(fields) < 3:
                continue
            count = parse_number(fields[0])
            event = fields[2].strip()
            if count is None or not event:
                continue
            unit = fields[1].strip()
            run_percentage = parse_number(fields[4]) if len(fields) > 4 else None
            rows.append(
                {
                    "kernel": kernel,
                    "configuration": configuration,
                    "event": event,
                    "count": f"{count:.12g}",
                    "unit": unit,
                    "run_percentage": "" if run_percentage is None else f"{run_percentage:.6g}",
                    "source_file": path.name,
                }
            )
            values.setdefault((kernel, configuration), {})[event] = count

    args.events_output.parent.mkdir(parents=True, exist_ok=True)
    event_fields = [
        "kernel",
        "configuration",
        "event",
        "count",
        "unit",
        "run_percentage",
        "source_file",
    ]
    with args.events_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = [
        "kernel",
        "configuration",
        "cycles",
        "instructions",
        "ipc",
        "branches",
        "branch_misses",
        "branch_miss_rate",
        "cache_references",
        "cache_misses",
        "cache_miss_rate",
        "l1_loads",
        "l1_load_misses",
        "l1_miss_rate",
        "llc_loads",
        "llc_load_misses",
        "llc_miss_rate",
    ]
    with args.summary_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for (kernel, configuration), event_values in sorted(values.items()):
            def value(name):
                return event_values.get(name)

            def ratio(numerator, denominator):
                top, bottom = value(numerator), value(denominator)
                return "" if top is None or not bottom else f"{top / bottom:.9g}"

            writer.writerow(
                {
                    "kernel": kernel,
                    "configuration": configuration,
                    "cycles": value("cycles") or "",
                    "instructions": value("instructions") or "",
                    "ipc": ratio("instructions", "cycles"),
                    "branches": value("branches") or "",
                    "branch_misses": value("branch-misses") or "",
                    "branch_miss_rate": ratio("branch-misses", "branches"),
                    "cache_references": value("cache-references") or "",
                    "cache_misses": value("cache-misses") or "",
                    "cache_miss_rate": ratio("cache-misses", "cache-references"),
                    "l1_loads": value("L1-dcache-loads") or "",
                    "l1_load_misses": value("L1-dcache-load-misses") or "",
                    "l1_miss_rate": ratio("L1-dcache-load-misses", "L1-dcache-loads"),
                    "llc_loads": value("LLC-loads") or "",
                    "llc_load_misses": value("LLC-load-misses") or "",
                    "llc_miss_rate": ratio("LLC-load-misses", "LLC-loads"),
                }
            )

    print(f"parsed {len(rows)} event rows into {args.summary_output}")


if __name__ == "__main__":
    main()
