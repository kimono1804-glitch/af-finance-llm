import json

import pytest

from common.prompts import parse_tool_calls, split_think
from common.tools import execute, format_value, run_tool_call


def test_chained_steps_with_refs_and_pct():
    resp = execute({"steps": [
        {"id": "net_revenue", "formula": "net_revenue",
         "inputs": {"gross_revenue": 1_250_000, "discounts": "50,000", "returns": 20_000}},
        {"id": "cm1", "formula": "cm1", "inputs": {"net_revenue": "@net_revenue", "cogs": 500_000}},
        {"id": "gm", "formula": "gross_margin", "inputs": {"net_revenue": "@net_revenue", "cogs": 500_000}},
        {"id": "ltv", "formula": "ltv", "inputs": {"arpu": 1000, "gross_margin": "@gm", "churn_rate_pct": 4}},
    ]})
    assert not resp.errors
    assert resp.results["net_revenue"].display == "1,180,000"
    assert resp.results["cm1"].value == 680_000
    assert resp.results["gm"].display == "57.63%"
    assert resp.results["ltv"].value == pytest.approx(1000 * (680_000 / 1_180_000) / 0.04)


def test_error_propagates_only_to_dependents():
    resp = execute({"steps": [
        {"id": "ltv", "formula": "ltv", "inputs": {"arpu": 100, "gross_margin_pct": 50, "churn_rate_pct": 0}},
        {"id": "ratio", "formula": "ltv_cac_ratio", "inputs": {"ltv": "@ltv", "cac": 100}},
        {"id": "cac", "formula": "cac", "inputs": {"sm_spend": 1000, "new_customers": 10}},
    ]})
    assert "churn_rate" in resp.errors["ltv"]
    assert "failed step" in resp.errors["ratio"]
    assert resp.results["cac"].value == 100


def test_bad_inputs_reported_not_raised():
    resp = execute({"steps": [
        {"id": "a", "formula": "eval", "inputs": {}},
        {"id": "b", "formula": "net_revenue", "inputs": {"gross_revenue": 1, "discounts": 0}},
        {"id": "c", "formula": "net_revenue", "inputs": {"gross_revenue": "__import__('os')", "discounts": 0, "returns": 0}},
        {"id": "d", "formula": "cac", "inputs": {"sm_spend": 1, "new_customers": 1, "extra": 2}},
        {"id": "e", "formula": "cac", "inputs": {"sm_spend": "@nope", "new_customers": 1}},
    ]})
    assert set(resp.errors) == {"a", "b", "c", "d", "e"}
    assert "missing input" in resp.errors["b"]


def test_list_inputs():
    resp = execute({"steps": [
        {"id": "var_ops", "formula": "total", "inputs": {"values": [100, "200", 300.5]}},
        {"id": "adj", "formula": "adjusted_ebitda", "inputs": {"ebitda": 1000, "adjustments": ["@var_ops", -100]}},
    ]})
    assert resp.results["var_ops"].value == 600.5
    assert resp.results["adj"].value == 1500.5


def test_display_kinds():
    assert format_value(0.12345, "percent") == "12.35%"
    assert format_value(3.456, "ratio") == "3.46x"
    assert format_value(7.26, "months") == "7.3 months"
    assert format_value(5000.0, "units") == "5,000 units (exact 5,000.00)"
    assert format_value(1234.4, "units") == "1,235 units (exact 1,234.40)"
    assert format_value(-105000, "currency") == "-105,000"


def test_run_tool_call_never_raises():
    out = json.loads(run_tool_call("finance_calculator", "{not json"))
    assert "_call" in out["errors"]
    out = json.loads(run_tool_call("rm_rf", {}))
    assert "unknown tool" in out["errors"]["_call"]
    out = json.loads(run_tool_call("finance_calculator", {"steps": [{"id": "Bad Id", "formula": "cac", "inputs": {}}]}))
    assert "_call" in out["errors"]


def test_parse_qwen3_output():
    text = ('<think>\nNeed CAC.\n</think>\n\n<tool_call>\n{"name": "finance_calculator", "arguments": '
            '{"steps": [{"id": "cac", "formula": "cac", "inputs": {"sm_spend": 10, "new_customers": 2}}]}}\n</tool_call>')
    think, rest = split_think(text)
    assert think == "Need CAC."
    calls = parse_tool_calls(rest)
    assert calls[0]["name"] == "finance_calculator"
    assert json.loads(run_tool_call(**calls[0]))["results"]["cac"]["value"] == 5
