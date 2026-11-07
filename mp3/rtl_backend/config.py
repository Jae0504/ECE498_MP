import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
KEYS = {"array_rows", "array_cols", "a_buffer_words", "b_buffer_words",
        "memory_lanes", "memory_wait_cycles", "clock_mhz"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate(config):
    if set(config) != KEYS:
        raise ValueError(f"RTL configuration keys must be {sorted(KEYS)}")
    for key in KEYS - {"clock_mhz"}:
        minimum = 0 if key == "memory_wait_cycles" else 1
        if type(config[key]) is not int or not minimum <= config[key] <= 65535:
            raise ValueError(f"Invalid {key}")
    if (isinstance(config["clock_mhz"], bool) or not isinstance(config["clock_mhz"], (int, float))
            or not math.isfinite(config["clock_mhz"]) or not 1 <= config["clock_mhz"] <= 1000):
        raise ValueError("clock_mhz must be finite, 1..1000 (assumed, not timing-closed)")
    for operand, banks in (("a", "array_rows"), ("b", "array_cols")):
        if config[f"{operand}_buffer_words"] % config[banks]:
            raise ValueError(f"{operand} capacity must be divisible by its bank count")
        if config[f"{operand}_buffer_words"] < config[banks]:
            raise ValueError(f"{operand} needs at least one word per bank")
    return config


def parameters(config):
    return dict(ROWS=config["array_rows"], COLS=config["array_cols"],
                A_WORDS=config["a_buffer_words"], B_WORDS=config["b_buffer_words"],
                LANES=config["memory_lanes"])


def fingerprint(root):
    files = sorted(Path(root).glob("*.sv"))
    if not files:
        raise ValueError(f"No SystemVerilog files in {root}")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    return dict(sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), files=hashes)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
