"""Lakh / crore units (formulas, generator) and the prompt contract (v1 adapter vs current)."""
import json
from pathlib import Path

import pytest
import yaml

from common import formulas as F
from common.prompts import CONTRACT, adapter_contract, build_tools, prompt_version
from common.tools import V1_FORMULAS, execute
from generate_raft_data import TASKS, build_example, validate_example

ROOT = Path(__file__).resolve().parents[1]


def test_unit_formulas():
    assert F.from_lakh(206.48) == pytest.approx(20_648_000)
    assert F.from_crore(2.5) == pytest.approx(25_000_000)
    assert F.from_thousands(20.27) == pytest.approx(20_270)
    assert F.from_millions(1.2) == pytest.approx(1_200_000)
    with pytest.raises(F.FinanceMathError):
        F.to_base_units(1, "dozen")


def test_calculator_converts_before_cac():
    resp = execute({"steps": [
        {"id": "sm_lakh", "formula": "total", "inputs": {"values": [12.5, 3.25]}},
        {"id": "sm", "formula": "from_lakh", "inputs": {"amount": "@sm_lakh"}},
        {"id": "cac", "formula": "cac", "inputs": {"sm_spend": "@sm", "new_customers": 630}}]})
    assert not resp.errors
    assert resp.results["sm"].display == "1,575,000"
    assert resp.results["cac"].value == pytest.approx(2500)


def test_v1_contract_is_frozen():
    # the v1 adapter (2026-10-21) was trained on prompt version 2013a079fb; it must keep seeing exactly that
    assert prompt_version("v1") == "2013a079fb"
    v1_desc = build_tools("v1")[0]["function"]["description"]
    assert "from_lakh" not in v1_desc and "from_lakh" not in V1_FORMULAS
    assert "from_lakh" in build_tools(CONTRACT)[0]["function"]["description"]


def test_adapter_contract(tmp_path):
    assert adapter_contract(None) == CONTRACT
    assert adapter_contract(tmp_path) == "v1"  # adapters without contract.json predate contracts
    (tmp_path / "contract.json").write_text(json.dumps({"contract": "v2"}), encoding="utf-8")
    assert adapter_contract(tmp_path) == "v2"


def _scaled_cfg() -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    cfg["scaled_units"] = 1.0
    return cfg


def test_scaled_unit_examples_validate_for_every_task():
    cfg = _scaled_cfg()
    conversions = 0
    for i, task in enumerate(sorted(TASKS)):
        for j in range(12):
            ex = build_example(7000 + i * 100 + j, cfg, task, missing=(j == 11), split="train")
            assert "scaled_units" in ex["meta"]["edge_cases"]
            assert not validate_example(ex), (task, validate_example(ex))
            ctx = ex["messages"][1]["content"]
            assert any(f"Units: {u}" in ctx for u in ("INR lakh", "INR crore", "USD thousands"))
            calls = ex["messages"][2].get("tool_calls")
            if calls:
                conversions += any(s["formula"].startswith("from_") for s in calls[0]["function"]["arguments"]["steps"])
                if task in ("net_revenue", "ebitda", "runway"):
                    assert any(f"Currency: {u}" in ex["messages"][4]["content"]
                               for u in ("INR lakh", "INR crore", "USD thousands"))
    assert conversions > 0


def test_validation_catches_misquoted_input():
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    ex = build_example(42, cfg, "net_revenue", False, "train")
    think = ex["messages"][2]["content"]
    line = next(ln for ln in think.splitlines() if ln.startswith("- ") and " = " in ln)
    value = line.split(" = ")[1].split(" ")[0]
    ex["messages"][2]["content"] = think.replace(line, line.replace(f" = {value} ", " = 999,999,999 "))
    assert any("quoted input" in e for e in validate_example(ex))


@pytest.mark.parametrize("text,expected", [
    ("Monthly P&L (₹ in lakhs)", "INR lakh"),
    ("Rs. Lakh", "INR lakh"),
    ("All figures in INR Lakhs unless stated", "INR lakh"),
    ("Amount (INR Cr)", "INR crore"),
    ("All figures in crores", "INR crore"),
    ("USD '000", "USD thousands"),
    ("USD ’000s", "USD thousands"),
    ("($ in millions)", "USD millions"),
    ("US$ mn", "USD millions"),
    ("$000s", "USD thousands"),
    ("Amounts in thousands", "thousands"),
    ("Units: INR", None),
    ("Mar-26", None),
    ("We grew to 2 mn users and raised $2m", None),
    ("Credit card fees (Cr. side)", None),
    ("USD Mar-26 actuals", None),
])
def test_detect_units(text, expected):
    from common.units import detect_units

    assert detect_units(text) == expected


def test_excel_units_from_title(tmp_path):
    from openpyxl import Workbook

    from ingest_financials import ingest_excel

    wb = Workbook()
    ws = wb.active
    ws.title = "P&L"
    ws.append(["Monthly P&L (₹ in lakhs)"])
    ws.append(["Line Item", "Feb-26", "Mar-26"])
    ws.append(["Gross Revenue", 210.5, 220.75])
    ws.append(["Discounts", 10.25, 11.5])
    path = tmp_path / "mis_lakh.xlsx"
    wb.save(path)
    chunks = ingest_excel(path)
    assert chunks and all(c.metadata.units == "INR lakh" for c in chunks)
    assert "Units: INR lakh" in chunks[0].text


def test_lakh_sample_workbook_units_and_cases(tmp_path):
    import math

    from eval_cases import build_lakh_cases
    from ingest_financials import ingest_excel
    from make_sample_mis import build_data, build_data_lakh, build_workbook

    chunks = ingest_excel(build_workbook(tmp_path / "mis_lakh.xlsx", lakh=True))
    units = {c.metadata.section: c.metadata.units for c in chunks}
    assert units["Revenue"] == "INR lakh"          # units row "INR (₹ in lakhs)"
    assert units["Bank"] == "INR lakh"             # only the title says "(Rs. in Lakhs)"
    assert units["Customer Metrics"] == "INR unless stated"
    assert not any("S&M" in c.text for c in chunks if c.metadata.section == "Customer Metrics")

    d, dl = build_data(), build_data_lakh()
    assert dl["gross_revenue"][0] == round(d["gross_revenue"][0] / 100_000, 2)
    assert dl["arpu"] == d["arpu"] and dl["new_customers"] == d["new_customers"]
    for t in range(1, len(dl["closing_cash"])):
        assert dl["opening_cash"][t] == dl["closing_cash"][t - 1]
    for case in build_lakh_cases():
        for vals in case.truth.values():
            assert vals and all(math.isfinite(x) for x in vals)
    cac = next(c for c in build_lakh_cases() if c.case_id == "rl-03").truth["cac"][0]
    assert 500 < cac < 10_000  # INR per customer, not lakh per customer
