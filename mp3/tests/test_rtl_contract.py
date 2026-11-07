import numpy as np
import pytest

from rtl_backend.config import ROOT, read, validate
from rtl_backend.model import quantize


def test_zero_quantization_is_finite_and_exact():
    qa, qb, sa, sb = quantize(np.zeros((3, 7), np.float32), np.zeros((7, 2), np.float32))
    assert not qa.any() and not qb.any()
    assert np.isfinite(sa).all() and np.isfinite(sb).all()
    assert np.array_equal(qa.astype(np.int64) @ qb.astype(np.int64), np.zeros((3, 2)))


def test_signed_quantization_and_independent_row_scales():
    a = np.array([[-1, 0.5, 0], [-2**-12, 2**-13, 0]], np.float32)
    b = np.array([[2, 2**-10], [-1, -2**-11], [0, 0]], np.float32)
    qa, qb, sa, sb = quantize(a, b)
    assert qa.dtype == qb.dtype == np.int16
    assert np.array_equal(qa[0], qa[1])
    assert np.array_equal(qb[:, 0], qb[:, 1])
    result = (qa.astype(np.int64) @ qb.astype(np.int64))*sa*sb
    np.testing.assert_allclose(result, a.astype(np.float64) @ b.astype(np.float64), atol=1e-7, rtol=5e-5)


@pytest.mark.parametrize("key,value", [("a_buffer_words", 7), ("array_cols", 0), ("memory_lanes", True),
                                      ("memory_wait_cycles", -1), ("clock_mhz", float("nan")),
                                      ("clock_mhz", True), ("precision_bits", 8)])
def test_reject_unsupported_or_invalid_configuration(key, value):
    cfg = read(ROOT / "configs/rtl_baseline.json")
    cfg[key] = value
    with pytest.raises(ValueError):
        validate(cfg)


def test_reject_nonfinite_operands():
    with pytest.raises(ValueError):
        quantize(np.array([[np.nan]], np.float32), np.ones((1, 1), np.float32))
