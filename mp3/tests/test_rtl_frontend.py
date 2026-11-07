import pytest
import torch

from rtl_backend.config import ROOT, read
from rtl_backend.frontend import ByteTokenizer, KernelRecorder, bind_matrix_executor, build_model, next_token


@pytest.mark.parametrize("batch,kv_heads", [(1, 2), (2, 1), (2, 4)])
def test_cached_chunk_matches_full_prefix(batch, kv_heads):
    config = read(ROOT / "configs/model_tiny.json")
    config.update(num_layers=2, num_kv_heads=kv_heads)
    model = build_model(config)
    ids = torch.tensor([ByteTokenizer.encode("abcdef")]*batch)
    with torch.inference_mode():
        _, _, cache = model(ids[:, :3], use_cache=True)
        cached, _, cache = model(ids[:, 3:], past_key_values=cache, use_cache=True)
        reference, _ = model(ids)
        torch.testing.assert_close(cached, reference, atol=1e-5, rtol=1e-4)
        assert cache[0][0].shape == (batch, kv_heads, ids.shape[1], 8)


def test_recorder_observes_real_prefill_and_cached_attention():
    model = build_model(read(ROOT / "configs/model_tiny.json"))
    ids = torch.tensor([ByteTokenizer.encode("AI chip")])
    with torch.inference_mode(), KernelRecorder(model) as recorder:
        logits, _, cache = model(ids, use_cache=True)
        for step in (1, 2):
            recorder.phase, recorder.step = "decode", step
            logits, _, cache = model(next_token(logits), past_key_values=cache, use_cache=True)
    qk = [e for e in recorder.events if e["role"] == "attention_qk"]
    assert [(e["m"], e["n"], e["k"], e["repeat"]) for e in qk] == [
        (8, 8, 8, 4), (1, 9, 8, 4), (1, 10, 8, 4)]
    assert all(e["kernel"] == "gemv" for e in recorder.events if e["phase"] == "decode")


def test_executor_controls_results_and_restores_model_after_failure():
    class ZeroExecutor:
        bind = bind_matrix_executor

        def matmul(self, name, role, lhs, rhs):
            return torch.zeros((lhs.shape[0], rhs.shape[1]))

    model = torch.nn.Linear(2, 3, bias=False)
    model.weight.data.fill_(1)
    x = torch.ones((1, 2))
    with pytest.raises(RuntimeError, match="test failure"):
        with ZeroExecutor().bind(model):
            assert not model(x).any()
            raise RuntimeError("test failure")
    assert "forward" not in model.__dict__
    torch.testing.assert_close(model(x), torch.full((1, 3), 2.0))


def test_byte_tokenizer_roundtrip():
    text = "AI \uac00\uc18d\uae30"
    assert ByteTokenizer.decode(ByteTokenizer.encode(text)) == text
