"""Use actual RTL outputs for all Linear and attention products in miniLLAMA."""
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from .frontend import ByteTokenizer, KernelRecorder, bind_matrix_executor, build_model, next_token
from .config import digest, write


def quantize(a, b):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite matrix operands")
    # Per-row A and per-column B, symmetric, round-to-nearest/ties-to-even.
    sa = np.max(np.abs(a.astype(np.float64)), axis=1, keepdims=True) / 32767
    sb = np.max(np.abs(b.astype(np.float64)), axis=0, keepdims=True) / 32767
    sa[sa == 0] = 1
    sb[sb == 0] = 1
    qa = np.clip(np.rint(a / sa), -32767, 32767).astype(np.int16)
    qb = np.clip(np.rint(b / sb), -32767, 32767).astype(np.int16)
    return qa, qb, sa, sb


class QuantizedExecutor:
    bind = bind_matrix_executor

    def __init__(self, rtl=None):
        self.rtl = rtl
        self.calls, self.events = [], []
        self.phase, self.step = "prefill", 0
        self.max_matrix_fp32_error = 0.0

    def matmul(self, name, role, lhs, rhs):
        if (lhs.device.type != "cpu" or rhs.device.type != "cpu" or
                lhs.dtype != torch.float32 or rhs.dtype != torch.float32 or
                lhs.ndim < 2 or lhs.ndim != rhs.ndim or lhs.shape[:-2] != rhs.shape[:-2] or
                lhs.shape[-1] != rhs.shape[-2]):
            raise ValueError("Expected matching CPU FP32 matrix batches")
        a, b = lhs.detach().numpy(), rhs.detach().numpy()
        result = np.empty((*a.shape[:-2], a.shape[-2], b.shape[-1]), np.float32)
        repeat = 0
        for index in np.ndindex(a.shape[:-2]):
            qa, qb, sa, sb = quantize(a[index], b[index])
            if self.rtl:
                integers, stats = self.rtl(qa, qb)
                self.calls.append(dict(case_id=len(self.calls), event_id=len(self.events), name=name,
                                       role=role, kernel="gemv" if qa.shape[0]==1 or qb.shape[1]==1 else "gemm",
                                       phase=self.phase, step=self.step, batch_head_index=str(index),
                                       m=qa.shape[0], n=qb.shape[1], k=qa.shape[1], **stats))
            else:
                integers = qa.astype(np.int64) @ qb.astype(np.int64)
            result[index] = (integers * sa * sb).astype(np.float32)
            repeat += 1
        output = torch.from_numpy(result)
        if self.rtl:
            self.max_matrix_fp32_error = max(self.max_matrix_fp32_error,
                                            (output - torch.matmul(lhs, rhs)).abs().max().item())
        self.events.append(dict(name=name, role=role, m=a.shape[-2], n=b.shape[-1], k=a.shape[-1], repeat=repeat))
        return output


@torch.inference_mode()
def execute_model(request, rtl):
    from .reporting import report
    model_cfg, workload, out = request["model"], request["workload"], Path(request["out"])
    if set(workload) != {"name", "prompt", "batch_size", "new_tokens"}:
        raise ValueError("Invalid workload keys")
    if not isinstance(workload["prompt"], str) or any(type(workload[k]) is not int or workload[k] < 1
                                                    for k in ("batch_size", "new_tokens")):
        raise ValueError("Invalid workload values")
    prompt = ByteTokenizer.encode(workload["prompt"])
    if len(prompt) + workload["new_tokens"] - 1 > model_cfg["max_seq_len"]:
        raise ValueError("Workload exceeds context length")
    model = build_model(model_cfg, request["checkpoint"])
    tokens = torch.tensor([prompt]*workload["batch_size"], dtype=torch.long)
    executor = QuantizedExecutor(rtl)
    generated, actual = [], []
    started = time.perf_counter()
    with executor.bind(model), KernelRecorder(model) as recorder:
        logits, _, cache = model(tokens, use_cache=True)
        actual.append(logits.clone())
        generated.append(next_token(logits))
        for step in range(1, workload["new_tokens"]):
            executor.phase = recorder.phase = "decode"
            executor.step = recorder.step = step
            logits, _, cache = model(generated[-1], past_key_values=cache, use_cache=True)
            actual.append(logits.clone())
            generated.append(next_token(logits))
    if len(executor.events) != len(recorder.events):
        raise AssertionError("Missed a model matrix event")
    for call, event in zip(executor.events, recorder.events):
        if any(call[k] != event[k] for k in call):
            raise AssertionError("Model observation and executed RTL events disagree")

    # Repeat cached inference with an independent INT64 matrix reference.
    # It validates the values, and never replaces an RTL output in the actual run.
    quantized_reference = []
    with QuantizedExecutor().bind(model):
        for step in range(workload["new_tokens"]):
            inp = tokens if step == 0 else generated[step-1]
            ref, _, ref_cache = model(inp, past_key_values=None if step == 0 else ref_cache, use_cache=True)
            torch.testing.assert_close(actual[step], ref, atol=0, rtol=0)
            quantized_reference.append(ref.clone())

    fp32_reference, fp32_tokens, fp32_error, fp32_cache_error = [], [], 0.0, 0.0
    prefix = tokens
    for step in range(workload["new_tokens"]):
        ref, _ = model(prefix)
        inp = tokens if step == 0 else generated[step-1]
        cached, _, fp_cache = model(inp, past_key_values=None if step == 0 else fp_cache, use_cache=True)
        torch.testing.assert_close(cached, ref, atol=1e-5, rtol=1e-4)
        fp32_cache_error = max(fp32_cache_error, (cached-ref).abs().max().item())
        fp32_error = max(fp32_error, (actual[step]-ref).abs().max().item())
        # Course tiny-model acceptance threshold; not a general INT16 accuracy guarantee.
        torch.testing.assert_close(actual[step], ref, atol=1e-3, rtol=1e-2)
        fp32_reference.append(ref.clone())
        fp32_tokens.append(next_token(ref))
        prefix = torch.cat((prefix, generated[step]), dim=1)
    ids = torch.cat(generated, dim=1).tolist()
    checkpoint_hash = hashlib.sha256(Path(request["checkpoint"]).read_bytes()).hexdigest() if request["checkpoint"] else None
    trace = dict(model=model_cfg, workload=workload, prompt_tokens=len(prompt),
                 generated_token_ids=ids, generated_text=[ByteTokenizer.decode(row) for row in ids],
                 weights="checkpoint" if checkpoint_hash else "seeded_random", checkpoint_sha256=checkpoint_hash,
                 events=recorder.events, result_source="RTL INT64 outputs dequantized to FP32",
                 comparison_id=digest(dict(model=model_cfg, workload=workload, checkpoint=checkpoint_hash,
                                           model_source=request["provenance"]["model_source"]["sha256"],
                                           quantization="symmetric-int16-rowA-columnB-v1", protocol="split-ports-v1")))
    validation = dict(passed=True, integer_matrices_bit_exact=True, quantized_reference_logits_bit_exact=True,
                      fp32_logits_max_abs_error=fp32_error, fp32_logit_atol=1e-3, fp32_logit_rtol=1e-2,
                      fp32_cached_vs_full_prefix_max_abs_error=fp32_cache_error,
                      fp32_matrix_max_abs_error=executor.max_matrix_fp32_error,
                      fp32_next_tokens_match_on_executed_prefix=torch.cat(fp32_tokens, dim=1).tolist()==ids,
                      reference_used_for_outputs=False, checked_matrix_count=len(executor.calls))
    write(out / "trace.json", trace)
    write(out / "numerical_validation.json", validation)
    torch.save(dict(rtl=actual, quantized_reference=quantized_reference, fp32_reference=fp32_reference), out / "logits.pt")
    report(request, trace, validation, executor.calls, time.perf_counter()-started)
