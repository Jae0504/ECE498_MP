"""miniLLAMA loading and matrix hooks for RTL execution."""
from contextlib import contextmanager
import hashlib
import math
import sys

import torch
from torch import nn

from .config import ROOT, digest

SOURCES_ROOT = ROOT / ".deps"
MINILLAMA_ROOT = SOURCES_ROOT / "miniLLAMA"


def source_hash(directory):
    files = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(directory.rglob("*.py")) if "__pycache__" not in p.parts}
    return {"sha256": digest(files), "files": files}


class ByteTokenizer:
    """Fixed vocabulary: PAD=0, BOS=1, EOS=2, reserved=3, bytes=4..259."""

    @staticmethod
    def encode(text):
        return [1] + [byte + 4 for byte in text.encode("utf-8")]

    @staticmethod
    def decode(ids):
        return bytes(token - 4 for token in ids if 4 <= token < 260).decode("utf-8", errors="replace")


def model_config(raw):
    required = {"hidden_size", "num_layers", "num_heads", "num_kv_heads", "max_seq_len", "vocab_size", "seed"}
    if set(raw) != required:
        raise ValueError(f"Model keys must be {sorted(required)}")
    for name in required - {"seed"}:
        if type(raw[name]) is not int or raw[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(raw["seed"]) is not int or raw["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    h, heads, kv = raw["hidden_size"], raw["num_heads"], raw["num_kv_heads"]
    if h % heads or heads % kv or (h // heads) % 2:
        raise ValueError("hidden_size must divide into even head dimensions; heads must be divisible by KV heads")
    if raw["vocab_size"] != 260:
        raise ValueError("The course byte tokenizer requires vocab_size=260")
    return dict(vocab_size=260, n_head=heads, hidden_size=h, n_layer=raw["num_layers"],
                n_embd=h, n_local_heads=heads, n_local_kv_heads=kv, eps=1e-6,
                max_len=raw["max_seq_len"], rope_theta=10000.0, num_key_value_heads=kv,
                attention_dropout=0.0, rms_norm_eps=1e-6, block_size=raw["max_seq_len"])


def build_model(raw, checkpoint=None):
    if not (SOURCES_ROOT / "manifest.json").is_file() or not (MINILLAMA_ROOT / "model.py").is_file():
        raise RuntimeError("Prepare miniLLAMA first: sh mp3/setup_rtl.sh")
    if str(SOURCES_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCES_ROOT))
    from miniLLAMA.model import LLAMA

    config = model_config(raw)
    torch.set_num_threads(1)
    torch.manual_seed(raw["seed"])
    model = LLAMA(config)
    model.apply(model._init_weights)
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("model_config") != raw or saved.get("tokenizer") != "byte-v1":
            raise ValueError("Checkpoint model configuration/tokenizer does not match this run")
        model.load_state_dict(saved["state_dict"], strict=True)
    return model.eval()


def next_token(logits):
    scores = logits[:, -1, :].clone()
    scores[:, :4] = -torch.inf
    return scores.argmax(dim=-1, keepdim=True)


class KernelRecorder:
    def __init__(self, model):
        self.model = model
        self.events, self.handles, self.observers = [], [], []
        self.phase, self.step = "prefill", 0

    def record(self, name, role, m, n, k, repeat=1):
        self.events.append(dict(event_id=len(self.events), name=name, role=role,
                                kernel="gemv" if m == 1 or n == 1 else "gemm",
                                phase=self.phase, step=self.step,
                                m=int(m), n=int(n), k=int(k), repeat=int(repeat)))

    def linear_hook(self, name):
        def hook(module, inputs, output):
            x = inputs[0]
            self.record(name, "linear", x.numel() // x.shape[-1], module.out_features, module.in_features)
        return hook

    def attention_hook(self, name):
        def observe(role, lhs, rhs):
            self.record(name + "." + role, role, lhs.shape[-2], rhs.shape[-1],
                        lhs.shape[-1], math.prod(lhs.shape[:-2]))
        return observe

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self.linear_hook(name)))
            if hasattr(module, "matmul_observer"):
                self.observers.append((module, module.matmul_observer))
                module.matmul_observer = self.attention_hook(name)
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        for module, observer in self.observers:
            module.matmul_observer = observer


@contextmanager
def bind_matrix_executor(self, model):
    """Replace matrix operations and restore the model even if a check fails."""
    restored = []
    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                had_local = "forward" in module.__dict__
                restored.append((module, "forward", module.forward, had_local))

                def forward(x, module=module, name=name):
                    y = self.matmul(name, "linear", x.reshape(-1, module.in_features), module.weight.T)
                    y = y.reshape(*x.shape[:-1], module.out_features)
                    return y if module.bias is None else y + module.bias

                module.forward = forward
            if hasattr(module, "matmul_observer"):
                if not hasattr(module, "matmul_executor"):
                    raise RuntimeError("miniLLAMA source lacks the numerical execution hook; rerun source preparation")
                restored.append((module, "matmul_executor", module.matmul_executor, True))
                module.matmul_executor = lambda role, lhs, rhs, name=name: self.matmul(name + "." + role, role, lhs, rhs)
        yield self
    finally:
        for module, attribute, previous, had_local in reversed(restored):
            if had_local:
                setattr(module, attribute, previous)
            else:
                delattr(module, attribute)
