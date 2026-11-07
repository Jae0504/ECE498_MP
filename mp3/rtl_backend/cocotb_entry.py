import os
from pathlib import Path

import cocotb
import numpy as np
from cocotb.task import bridge, resume

from .config import read, write
from .driver import ArrayDriver


@cocotb.test()
async def run_requested_job(dut):
    request = read(os.environ["ECE498_RTL_REQUEST"])
    driver = ArrayDriver(dut, request["config"])
    await driver.reset()
    out = Path(request["out"])
    if request["mode"] == "check":
        rng = np.random.default_rng(498)
        r, c = request["config"]["array_rows"], request["config"]["array_cols"]
        depth = min(request["config"]["a_buffer_words"]//r, request["config"]["b_buffer_words"]//c)
        shapes = [(1, 1, 1), (1, c+3, 7), (r+1, 1, 9), (r+1, c+1, 11),
                  (2, 3, depth+3), (r+1, c+1, 5), (r+1, c+1, 5)]
        cases = []
        for index, (m, n, k) in enumerate(shapes):
            a = rng.integers(-32768, 32768, (m, k), dtype=np.int16)
            b = rng.integers(-32768, 32768, (k, n), dtype=np.int16)
            if index == 0:
                a[:] = -32768
                b[:] = -32768
            result, stats = await driver.multiply(a, b, stall=index % 2 == 1)
            cases.append(dict(m=m, n=n, k=k, backpressure=index % 2 == 1, **stats))
        # A zero matrix must not retain accumulators or cached data from prior calls.
        _, stats = await driver.multiply(np.zeros((r+1, 3), np.int16), np.zeros((3, c+1), np.int16))
        cases.append(dict(m=r+1, n=c+1, k=3, zero=True, **stats))
        # Deliberately exceed INT32 accumulation; no wrap/saturation is allowed.
        _, stats = await driver.multiply(np.full((1, 9), -32768, np.int16),
                                         np.full((9, 1), -32768, np.int16))
        cases.append(dict(m=1, n=1, k=9, wider_than_int32=True, **stats))
        # Reset aborts an in-flight command and clears controller/PE/cache state.
        dut.dim_m.value = r+1
        dut.dim_n.value = c+1
        dut.dim_k.value = 17
        dut.start.value = 1
        await driver.tick()
        dut.start.value = 0
        await driver.tick()
        await driver.reset()
        _, stats = await driver.multiply(np.full((1, 2), -2, np.int16), np.full((2, 1), 3, np.int16))
        cases.append(dict(m=1, n=1, k=2, after_inflight_reset=True, **stats))
        write(out / "validation.json", dict(passed=True, integer_exact=True, cases=cases))
    else:
        from .model import execute_model

        @resume
        async def rtl_matmul(a, b):
            return await driver.multiply(a, b)

        await bridge(execute_model)(request, rtl_matmul)
