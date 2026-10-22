"""Multi-period task family: MoM growth, n-month totals / averages, YTD."""
from collections import Counter
from pathlib import Path

import pytest
import yaml

from common import formulas as F
from generate_raft_data import build_example, validate_example

ROOT = Path(__file__).resolve().parents[1]


def test_growth_and_average_formulas():
    assert F.growth_pct(110, 100) == pytest.approx(0.10)
    assert F.growth_pct(-50, -100) == pytest.approx(0.5)  # less negative = improvement against |prior|
    with pytest.raises(F.FinanceMathError):
        F.growth_pct(5, 0)
    assert F.average([1, 2, 4]) == pytest.approx(7 / 3)
    with pytest.raises(F.FinanceMathError):
        F.average([])


def test_multi_period_examples():
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    variants: Counter = Counter()
    for i in range(300):
        ex = build_example(30_000 + i, cfg, "multi_period", missing=(i % 10 == 9), split="train")
        assert not validate_example(ex), validate_example(ex)
        meta = ex["meta"]
        if meta["missing"]:
            continue
        variant = meta["variant"].split(":")[0]
        variants[variant] += 1
        steps = ex["messages"][2]["tool_calls"][0]["function"]["arguments"]["steps"]
        formulas = {s["formula"] for s in steps}
        think = ex["messages"][2]["content"]
        if variant == "mom":
            assert "growth_pct" in formulas and "previous month" in think
        elif variant == "qoq":
            assert "growth_pct" in formulas and "Quarter-on-quarter" in think and "Quarterly Summary" in think
        elif variant == "average":
            assert "average" in formulas
        elif variant == "ytd":
            first = steps[0]["inputs"]["values"]
            assert "starts in Apr-" in think and len(first) >= 2
        else:
            assert formulas == {"total"}
    assert set(variants) == {"mom", "range_total", "average", "ytd", "qoq"}
