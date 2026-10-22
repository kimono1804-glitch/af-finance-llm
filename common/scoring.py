"""Scoring for model evaluation (eval_model.py).

A "trace" is what run_assistant.run_conversation returns:
  {"thinks": [str], "calls": [{"call": {"name", "arguments"}, "result": {...}, "unsupported_inputs": [...]}],
   "report": str}
A Gold describes what a correct answer contains. Ground truth always comes from Python (common.formulas),
never from a model.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from common.audit import context_numbers

# Formulas that are bookkeeping, not "the formula" a question is about.
HELPER_FORMULAS = {"total", "difference", "ratio", "percent_of", "margin_pct"}

# Equivalent ways to compute the same metric.
ALTERNATIVES: dict[str, set[str]] = {
    "cm1": {"cm1", "gross_profit", "contribution_margin"},
    "gross_profit": {"cm1", "gross_profit", "contribution_margin"},
    "cm2": {"cm2", "contribution_margin"},
    "cm3": {"cm3", "contribution_margin"},
    "ebit_from_net_income": {"ebit_from_net_income"},
    "break_even_units": {"break_even_units"},
    "break_even_revenue": {"break_even_revenue"},
}

# Numbers that legitimately appear in reports without coming from the context or the tool
# (benchmarks and thresholds in the CFO take, metric names like CM1/CM2/CM3).
# template thresholds: runway 6/12/18 mo, magic number 0.5/0.75, gross margin 40%, payback 12/24, LTV:CAC 1/3x, ...
ALLOWED_CONSTANTS = {0.0, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 14.0, 15.0, 18.0, 24.0, 40.0, 100.0}

MISSING_WORDS = ("missing", "not in the provided", "not provided", "not available", "not found",
                 "cannot compute", "can't compute", "no data", "does not appear", "not included")
ERROR_WORDS = ("not computable", "undefined", "impossible", "cannot", "can't", "never", "not possible")

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class Gold:
    case_id: str
    suite: str
    task: str
    question: str
    context: str                                   # the full user message the model saw
    truth: dict[str, list[float] | None]           # key -> acceptable values; None = must be reported as not computable
    required_formulas: list[set[str]] = field(default_factory=list)  # each set: any one formula satisfies it
    missing: str | None = None                     # name of the missing input (missing-data cases)
    missing_keywords: list[str] = field(default_factory=list)       # words the refusal must mention
    edge_cases: list[str] = field(default_factory=list)


def _close(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)


def _numbers(text: str) -> list[float]:
    return [abs(float(m.replace(",", ""))) for m in _NUM_RE.findall(text)]


def required_formulas_from_steps(steps: list[dict[str, Any]]) -> list[set[str]]:
    req = []
    for f in dict.fromkeys(s["formula"] for s in steps):
        if f not in HELPER_FORMULAS:
            req.append(ALTERNATIVES.get(f, {f}))
    return req


def trace_from_gold_messages(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn a dataset conversation into a trace (used to self-test the scorer: gold must score 100%)."""
    import json

    from common.prompts import split_think
    from common.tools import run_tool_call

    trace: dict[str, Any] = {"thinks": [], "calls": [], "report": ""}
    for m in messages[2:]:
        if m["role"] != "assistant":
            continue
        think, rest = split_think(m["content"])
        trace["thinks"].append(think)
        for tc in m.get("tool_calls", []):
            fn = tc["function"]
            trace["calls"].append({"call": {"name": fn["name"], "arguments": fn["arguments"]},
                                   "result": json.loads(run_tool_call(fn["name"], fn["arguments"])),
                                   "unsupported_inputs": []})
        if "tool_calls" not in m:
            trace["report"] = rest
    return trace


def score(trace: dict[str, Any], gold: Gold) -> dict[str, Any]:
    results = [r for c in trace.get("calls", []) for r in (c.get("result") or {}).get("results", {}).values()]
    values = [float(r["value"]) for r in results]
    report = trace.get("report") or ""
    low = report.lower()
    called = any(c.get("call", {}).get("name") == "finance_calculator" for c in trace.get("calls", []))
    used: set[str] = set()
    for c in trace.get("calls", []):
        args = (c.get("call") or {}).get("arguments")
        if isinstance(args, dict):
            for s in args.get("steps", []) or []:
                if isinstance(s, dict) and isinstance(s.get("formula"), str):
                    used.add(s["formula"])

    row: dict[str, Any] = {"case_id": gold.case_id, "suite": gold.suite, "task": gold.task,
                           "question": gold.question, "missing_case": gold.missing is not None,
                           "edge_cases": gold.edge_cases, "called_tool": called, "formulas_used": sorted(used)}

    if gold.missing is not None:
        mentions = any(w in low for w in MISSING_WORDS) and all(k.lower() in low for k in gold.missing_keywords)
        row["missing_handled"] = (not called) and mentions
        row["correct"] = row["end_to_end"] = row["missing_handled"]
        if not row["missing_handled"]:
            row["why"] = "called the calculator anyway" if called else "did not clearly name the missing input"
        return row

    hits: dict[str, bool] = {}
    hit_display: dict[str, str] = {}
    for key, acceptable in gold.truth.items():
        if acceptable is None:
            continue
        hits[key] = False
        for r in results:
            if any(_close(float(r["value"]), a) for a in acceptable):
                hits[key] = True
                hit_display[key] = r["display"]
                break
    expected_errors = [k for k, v in gold.truth.items() if v is None]
    errors_handled = (not expected_errors) or any(w in low for w in ERROR_WORDS)

    unsupported = [u for c in trace.get("calls", []) for u in c.get("unsupported_inputs", [])]
    allowed = context_numbers(gold.context) | {abs(v) for v in values} | ALLOWED_CONSTANTS
    for r in results:
        allowed |= set(_numbers(r["display"]))
    for c in trace.get("calls", []):  # numbers quoted in calculator error messages
        for e in (c.get("result") or {}).get("errors", {}).values():
            allowed |= set(_numbers(str(e)))
    hallucinated = sorted({n for n in _numbers(report) if not any(_close(n, a) for a in allowed)})

    row["value_recall"] = (sum(hits.values()) / len(hits)) if hits else 1.0
    row["numeric_correct"] = all(hits.values()) and errors_handled
    row["formula_correct"] = all(any(f in used for f in alts) for alts in gold.required_formulas)
    row["report_consistent"] = all(hit_display[k] in report for k in hit_display)
    row["faithful"] = not unsupported and not hallucinated
    row["unsupported_inputs"] = unsupported
    row["hallucinated_numbers"] = hallucinated[:10]
    row["expected_error_case"] = bool(expected_errors)
    row["errors_handled"] = errors_handled
    row["false_refusal"] = not called
    row["correct"] = row["numeric_correct"]
    row["end_to_end"] = row["numeric_correct"] and row["report_consistent"] and row["faithful"]
    if not row["end_to_end"]:
        reasons = []
        if not called:
            reasons.append("no calculator call")
        missed = [k for k, h in hits.items() if not h]
        if missed:
            reasons.append(f"wrong/missing values: {missed}")
        if not errors_handled:
            reasons.append("did not flag the not-computable metric")
        if not row["report_consistent"]:
            reasons.append("report does not copy tool values")
        if unsupported:
            reasons.append(f"inputs not in context: {unsupported[:3]}")
        if hallucinated:
            reasons.append(f"numbers in report not from context/tool: {hallucinated[:3]}")
        row["why"] = "; ".join(reasons)
    return row


METRICS = [
    ("numeric_accuracy", "Numeric accuracy", "non-missing cases where every expected value was computed (and not-computable metrics were flagged)"),
    ("value_recall", "Value recall", "share of expected values found in the calculator results"),
    ("formula_accuracy", "Formula accuracy", "used the right formula(s)"),
    ("faithfulness", "Context faithfulness", "no tool inputs or report numbers that are absent from the context/tool output"),
    ("report_consistency", "Report consistency", "report copies the calculator's display values"),
    ("end_to_end", "End-to-end correct", "numeric + report + faithfulness all pass"),
    ("missing_handling", "Missing-data handling", "missing-data cases: no tool call and names the missing input"),
    ("error_handling", "Edge-case handling", "zero churn / price below unit cost etc. reported as not computable"),
    ("false_refusal", "False refusals", "answerable cases where no calculator call was made (lower is better)"),
]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ans = [r for r in rows if not r["missing_case"]]
    miss = [r for r in rows if r["missing_case"]]
    err = [r for r in ans if r.get("expected_error_case")]

    def mean(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    return {
        "n": len(rows), "n_answerable": len(ans), "n_missing": len(miss), "n_error_cases": len(err),
        "numeric_accuracy": mean([float(r["numeric_correct"]) for r in ans]),
        "value_recall": mean([r["value_recall"] for r in ans]),
        "formula_accuracy": mean([float(r["formula_correct"]) for r in ans]),
        "faithfulness": mean([float(r["faithful"]) for r in ans]),
        "report_consistency": mean([float(r["report_consistent"]) for r in ans]),
        "end_to_end": mean([float(r["end_to_end"]) for r in rows]),
        "missing_handling": mean([float(r["missing_handled"]) for r in miss]),
        "error_handling": mean([float(r["errors_handled"]) for r in err]),
        "false_refusal": mean([float(r["false_refusal"]) for r in ans]),
        "latency_s": mean([r["latency_s"] for r in rows if "latency_s" in r]),
    }


def aggregate_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r[key])].append(r)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"
