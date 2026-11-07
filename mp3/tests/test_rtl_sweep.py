from copy import deepcopy

import pytest

from rtl_backend.config import ROOT, read
from rtl_backend.sweep import prepare_plan, rank


def inputs():
    return read(ROOT / "configs/rtl_baseline.json"), read(ROOT / "configs/rtl_candidates.json")


@pytest.mark.parametrize("change", [
    {"clock_mhz": 200}, {"memory_lanes": 8}, {"memory_wait_cycles": 1},
    {"array_rows": 4}, {"a_buffer_words": 4096}, {"precision_bits": 8},
])
def test_budget_or_unsupported_changes_rejected_before_execution(change):
    baseline, plan = inputs()
    plan["candidates"][0]["parameters"].update(change)
    with pytest.raises(ValueError):
        prepare_plan(baseline, plan)


@pytest.mark.parametrize("name", ["baseline", "../escape", "two words", ""])
def test_candidate_names_cannot_overwrite_or_escape(name):
    baseline, plan = inputs()
    plan["candidates"][0]["name"] = name
    with pytest.raises(ValueError):
        prepare_plan(baseline, plan)


def test_declared_workload_weights_determine_selection():
    entries = [dict(name=name, metrics={"prefill": {"request_latency_us": p},
                                       "decode": {"request_latency_us": d}})
               for name, p, d in [("baseline", 100, 100), ("prefill", 40, 100), ("decode", 100, 40)]]
    assert rank(deepcopy(entries), {"prefill": 0.8, "decode": 0.2})["name"] == "prefill"
    assert rank(deepcopy(entries), {"prefill": 0.2, "decode": 0.8})["name"] == "decode"


@pytest.mark.parametrize("weights", [{"prefill": 0, "decode": 1}, {"prefill": float("nan"), "decode": 0.5},
                                     {"prefill": 1, "decode": 1}])
def test_invalid_objective_rejected(weights):
    baseline, plan = inputs()
    plan["weights"] = weights
    with pytest.raises(ValueError):
        prepare_plan(baseline, plan)
