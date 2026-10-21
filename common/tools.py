"""Safe finance calculator: the only place model tool calls turn into numbers.

The model sends {"steps": [{"id", "formula", "inputs"}, ...]}. Each formula name maps to a
function in common.formulas. There is no eval(); inputs must be numbers, numeric strings,
'@step_id' references to earlier steps, or lists of those.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from common import formulas as F
from common.schemas import CalcRequest, CalcResponse, StepResult

TOOL_NAME = "finance_calculator"


@dataclass(frozen=True)
class FormulaSpec:
    func: Callable[..., float]
    params: tuple[str, ...]
    kind: str  # currency | percent | ratio | months | units | number
    list_params: tuple[str, ...] = ()


REGISTRY: dict[str, FormulaSpec] = {
    "net_revenue": FormulaSpec(F.net_revenue, ("gross_revenue", "discounts", "returns"), "currency"),
    "gross_profit": FormulaSpec(F.gross_profit, ("net_revenue", "cogs"), "currency"),
    "gross_margin": FormulaSpec(F.gross_margin, ("net_revenue", "cogs"), "percent"),
    "contribution_margin": FormulaSpec(F.contribution_margin, ("net_revenue", "variable_costs"), "currency"),
    "cm1": FormulaSpec(F.cm1, ("net_revenue", "cogs"), "currency"),
    "cm2": FormulaSpec(F.cm2, ("cm1", "variable_ops_costs"), "currency"),
    "cm3": FormulaSpec(F.cm3, ("cm2", "variable_marketing"), "currency"),
    "margin_pct": FormulaSpec(F.margin_pct, ("value", "net_revenue"), "percent"),
    "ebit_from_net_income": FormulaSpec(F.ebit_from_net_income, ("net_income", "interest", "taxes"), "currency"),
    "ebitda": FormulaSpec(F.ebitda, ("ebit", "depreciation", "amortization"), "currency"),
    "adjusted_ebitda": FormulaSpec(F.adjusted_ebitda, ("ebitda", "adjustments"), "currency", ("adjustments",)),
    "net_burn": FormulaSpec(F.net_burn, ("cash_out", "cash_in"), "currency"),
    "runway_months": FormulaSpec(F.runway_months, ("cash_balance", "net_burn"), "months"),
    "burn_multiple": FormulaSpec(F.burn_multiple, ("net_burn", "net_new_arr"), "ratio"),
    "cac": FormulaSpec(F.cac, ("sm_spend", "new_customers"), "currency"),
    "ltv": FormulaSpec(F.ltv, ("arpu", "gross_margin", "churn_rate"), "currency"),
    "ltv_cac_ratio": FormulaSpec(F.ltv_cac_ratio, ("ltv", "cac"), "ratio"),
    "payback_period_months": FormulaSpec(F.payback_period_months, ("cac", "arpu", "gross_margin"), "months"),
    "break_even_units": FormulaSpec(F.break_even_units, ("fixed_costs", "price", "variable_cost_per_unit"), "units"),
    "break_even_revenue": FormulaSpec(F.break_even_revenue, ("fixed_costs", "price", "variable_cost_per_unit"), "currency"),
    "magic_number": FormulaSpec(F.magic_number, ("current_qtr_revenue", "prior_qtr_revenue", "prior_qtr_sm_spend"), "ratio"),
    "total": FormulaSpec(F.total, ("values",), "currency", ("values",)),
    "difference": FormulaSpec(F.difference, ("a", "b"), "currency"),
    "ratio": FormulaSpec(F.ratio, ("numerator", "denominator"), "ratio"),
    "percent_of": FormulaSpec(F.percent_of, ("part", "whole"), "percent"),
}


def formula_signatures() -> str:
    """One line per formula, used in the tool description shown to the model."""
    return "\n".join(f"- {name}({', '.join(spec.params)}) -> {spec.kind}" for name, spec in REGISTRY.items())


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_amount(v: float) -> str:
    """1180000 -> '1,180,000'; 1234.567 -> '1,234.57'."""
    if abs(v - round(v)) < 0.005:
        return f"{round(v):,}"
    return f"{v:,.2f}"


def format_value(v: float, kind: str) -> str:
    if kind == "percent":
        return f"{v * 100:.2f}%"
    if kind == "ratio":
        return f"{v:.2f}x"
    if kind == "months":
        return f"{v:.1f} months"
    if kind == "units":
        return f"{math.ceil(round(v, 9)):,} units (exact {v:,.2f})"
    return format_amount(v)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

class _InputError(Exception):
    pass


def _to_number(raw: Any, results: dict[str, StepResult], failed: set[str]) -> float:
    if isinstance(raw, bool):
        raise _InputError(f"boolean {raw!r} is not a number")
    if isinstance(raw, (int, float)):
        if not math.isfinite(raw):
            raise _InputError(f"non-finite number {raw!r}")
        return float(raw)
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("@"):
            ref = s[1:]
            if ref in failed:
                raise _InputError(f"depends on failed step '{ref}'")
            if ref not in results:
                raise _InputError(f"unknown reference '{s}' (refer only to earlier step ids)")
            return results[ref].value
        cleaned = s.replace(",", "").replace("_", "")
        try:
            return float(cleaned)
        except ValueError:
            raise _InputError(f"'{raw}' is not a number or @reference") from None
    raise _InputError(f"unsupported input type {type(raw).__name__}")


def _resolve_inputs(spec: FormulaSpec, inputs: dict[str, Any], results: dict[str, StepResult],
                    failed: set[str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for key, raw in inputs.items():
        is_pct = key.endswith("_pct")
        name = key[:-4] if is_pct else key
        if name not in spec.params:
            raise _InputError(f"unexpected input '{key}'; expected {list(spec.params)}")
        if name in kwargs:
            raise _InputError(f"input '{name}' given twice")
        if name in spec.list_params:
            if not isinstance(raw, list):
                raise _InputError(f"input '{name}' must be a list")
            vals = [_to_number(x, results, failed) for x in raw]
            kwargs[name] = [v / 100 for v in vals] if is_pct else vals
        else:
            if is_pct and isinstance(raw, str) and raw.strip().startswith("@"):
                raise _InputError(f"'{key}': pass step references without the _pct suffix")
            v = _to_number(raw, results, failed)
            kwargs[name] = v / 100 if is_pct else v
    missing = [p for p in spec.params if p not in kwargs]
    if missing:
        raise _InputError(f"missing input(s): {', '.join(missing)}")
    return kwargs


def execute(request: CalcRequest | dict[str, Any]) -> CalcResponse:
    """Run every step in order. A failing step is reported and does not stop independent steps."""
    req = request if isinstance(request, CalcRequest) else CalcRequest.model_validate(request)
    resp = CalcResponse()
    failed: set[str] = set()
    for step in req.steps:
        if step.id in resp.results or step.id in failed:
            resp.errors[step.id] = "duplicate step id"
            continue
        spec = REGISTRY.get(step.formula)
        if spec is None:
            resp.errors[step.id] = f"unknown formula '{step.formula}'"
            failed.add(step.id)
            continue
        try:
            kwargs = _resolve_inputs(spec, step.inputs, resp.results, failed)
            value = float(spec.func(**kwargs))
        except (_InputError, F.FinanceMathError) as exc:
            resp.errors[step.id] = str(exc)
            failed.add(step.id)
            continue
        resp.results[step.id] = StepResult(formula=step.formula, value=value, display=format_value(value, spec.kind))
    return resp


def run_tool_call(name: str, arguments: dict[str, Any] | str) -> str:
    """Entry point for model tool calls. Always returns a JSON string (never raises)."""
    if name != TOOL_NAME:
        return json.dumps({"results": {}, "errors": {"_call": f"unknown tool '{name}'"}})
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        resp = execute(args)
    except (ValidationError, json.JSONDecodeError, TypeError) as exc:
        return json.dumps({"results": {}, "errors": {"_call": f"invalid arguments: {exc}"[:500]}})
    return response_to_json(resp)


def response_to_json(resp: CalcResponse) -> str:
    """Compact JSON the model sees in <tool_response>. Same format in training and inference."""
    payload = {
        "results": {k: {"value": round(r.value, 6), "display": r.display} for k, r in resp.results.items()},
        "errors": resp.errors,
    }
    return json.dumps(payload, ensure_ascii=False)
