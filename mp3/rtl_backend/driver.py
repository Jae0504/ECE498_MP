"""External memory and scoreboard only. All tiling, MACs and outputs are RTL."""
import numpy as np
from cocotb.triggers import Timer


class ArrayDriver:
    def __init__(self, dut, config):
        self.dut, self.config = dut, config
        self.half_ns = 500 / config["clock_mhz"]

    async def tick(self):
        self.dut.clk.value = 0
        await Timer(self.half_ns, unit="ns", round_mode="round")
        self.dut.clk.value = 1
        await Timer(self.half_ns, unit="ns", round_mode="round")

    async def reset(self):
        for name in ("clk", "start", "dim_m", "dim_n", "dim_k", "a_ready", "b_ready", "a_data", "b_data", "c_ready"):
            getattr(self.dut, name).value = 0
        self.dut.rst.value = 1
        await self.tick()
        await self.tick()
        self.dut.rst.value = 0
        await self.tick()

    async def multiply(self, a, b, *, stall=False, timeout=10000000):
        if a.dtype != np.int16 or b.dtype != np.int16 or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("Expected INT16 [M,K] x [K,N]")
        m, k = a.shape
        n = b.shape[1]
        if any(not 1 <= size <= 65535 for size in (m, n, k)):
            raise ValueError("RTL dimensions must be 1..65535")
        d = self.dut
        if int(d.busy.value):
            raise RuntimeError("Accelerator is already busy")
        memories = {"a": np.ascontiguousarray(a).ravel(), "b": np.ascontiguousarray(b.T).ravel()}
        output = np.zeros(m*n, dtype=np.int64)
        seen = np.zeros(m*n, dtype=bool)
        pending = {"a": None, "b": None}
        wait = {"a": 0, "b": 0}
        last_c = None
        stats = dict(cycles=1, macs=0, compute_cycles=0, memory_read_bytes=0,
                     memory_write_bytes=0, buffer_write_bytes=0, buffer_read_bytes=0,
                     input_wait_port_cycles=0, output_wait_cycles=0)
        for key, value in (("dim_m", m), ("dim_n", n), ("dim_k", k), ("start", 1),
                           ("a_ready", 0), ("b_ready", 0), ("c_ready", 0)):
            getattr(d, key).value = value
        await self.tick()  # accepted start is cycle 1
        d.start.value = 0
        for cycle in range(1, timeout):
            for port in ("a", "b"):
                req = int(getattr(d, port + "_req").value)
                ready, data = 0, 0
                if req:
                    address = int(getattr(d, port + "_addr").value)
                    count = int(getattr(d, port + "_count").value)
                    packet = address, count
                    if pending[port] is None:
                        pending[port] = packet
                        wait[port] = self.config["memory_wait_cycles"]
                    elif pending[port] != packet:
                        raise AssertionError("Input request changed before memory accepted it")
                    if not 1 <= count <= self.config["memory_lanes"] or address < 0 or address+count > memories[port].size:
                        raise AssertionError(f"Invalid {port} request {packet}, memory size={memories[port].size}")
                    ready = int(wait[port] == 0 and (not stall or (cycle + (port == "b")) % 5 != 0))
                    if wait[port]:
                        wait[port] -= 1
                    if ready:
                        for lane, word in enumerate(memories[port][address:address+count]):
                            data |= (int(word) & 65535) << (16*lane)
                        stats["memory_read_bytes"] += 2*count
                        stats["buffer_write_bytes"] += 2*count
                        pending[port] = None
                    else:
                        stats["input_wait_port_cycles"] += 1
                elif pending[port] is not None:
                    raise AssertionError("RTL withdrew an unaccepted memory request")
                getattr(d, port + "_ready").value = ready
                getattr(d, port + "_data").value = data
            c_ready = int(not stall or cycle % 7 != 0)
            d.c_ready.value = c_ready
            if int(d.c_valid.value):
                address, value = int(d.c_addr.value), int(d.c_data.value)
                packet = address, value
                if last_c is not None and last_c != packet:
                    raise AssertionError("Output changed under backpressure")
                if not 0 <= address < output.size:
                    raise AssertionError("Output address out of bounds")
                if c_ready:
                    if seen[address]:
                        raise AssertionError("Duplicate output address")
                    output[address] = value - (1 << 64) if value & (1 << 63) else value
                    seen[address] = True
                    stats["memory_write_bytes"] += 8
                    last_c = None
                else:
                    last_c = packet
                    stats["output_wait_cycles"] += 1
            elif last_c is not None:
                raise AssertionError("RTL withdrew an unaccepted output")
            if int(d.pe_enable.value):
                rows, cols = int(d.active_rows.value), int(d.active_cols.value)
                if not 1 <= rows <= self.config["array_rows"] or not 1 <= cols <= self.config["array_cols"]:
                    raise AssertionError("Invalid active PE rectangle")
                stats["compute_cycles"] += 1
                stats["macs"] += rows*cols
                stats["buffer_read_bytes"] += 2*(rows+cols)
            await self.tick()
            stats["cycles"] += 1
            if int(d.done.value):
                break
        else:
            raise TimeoutError("RTL matrix execution exceeded cycle limit")
        if not seen.all() or any(value is not None for value in pending.values()) or last_c is not None:
            raise AssertionError("RTL completed with missing output or pending transfers")
        result = output.reshape(m, n)
        # Independent integer scoreboard. This reference NEVER supplies outputs.
        np.testing.assert_array_equal(result, a.astype(np.int64) @ b.astype(np.int64))
        if stats["macs"] != m*n*k:
            raise AssertionError("Observed PE enables disagree with required MAC count")
        return result, stats
