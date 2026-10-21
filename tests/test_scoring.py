import copy
import math
from pathlib import Path

import yaml

from common.scoring import Gold, aggregate, required_formulas_from_steps, score, trace_from_gold_messages
from eval_cases import build_cases
from generate_raft_data import build_example

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))


def _gold(ex) -> Gold:
    meta, msgs = ex["meta"], ex["messages"]
    steps = msgs[2]["tool_calls"][0]["function"]["arguments"]["steps"] if "tool_calls" in msgs[2] else []
    return Gold(case_id=meta["id"], suite="synthetic", task=meta["task"], question="q", context=msgs[1]["content"],
                truth={k: (None if v is None else [v]) for k, v in (meta.get("truth") or {}).items()},
                required_formulas=required_formulas_from_steps(steps), missing=meta["missing"],
                missing_keywords=[meta["missing"].split("(")[0].strip().rstrip("%").strip()] if meta["missing"] else [])


def test_gold_answers_score_perfectly():
    rows = []
    for i, task in enumerate(["net_revenue", "contribution_margin", "ebitda", "break_even", "ltv_cac", "payback"]):
        for j in range(4):
            ex = build_example(500 + 10 * i + j, CFG, task, missing=(j == 3), split="eval")
            rows.append(score(trace_from_gold_messages(ex["messages"]), _gold(ex)))
    a = aggregate(rows)
    assert a["end_to_end"] == 1.0 and a["numeric_accuracy"] == 1.0 and a["false_refusal"] == 0.0


def test_wrong_value_and_hallucination_are_caught():
    ex = build_example(7, CFG, "contribution_margin", missing=False, split="eval")
    gold = _gold(ex)
    trace = trace_from_gold_messages(ex["messages"])
    bad = copy.deepcopy(trace)
    first = next(iter(bad["calls"][0]["result"]["results"].values()))
    first["value"] += 1000.0
    row = score(bad, gold)
    assert not row["numeric_correct"] and "wrong/missing values" in row["why"]

    bad = copy.deepcopy(trace)
    bad["report"] += "\nAlso, next quarter revenue will be 98,765,432."
    row = score(bad, gold)
    assert row["numeric_correct"] and not row["faithful"] and 98765432.0 in row["hallucinated_numbers"]


def test_missing_case_must_not_call_tool():
    ex = build_example(11, CFG, "net_revenue", missing=True, split="eval")
    gold = _gold(ex)
    assert score(trace_from_gold_messages(ex["messages"]), gold)["missing_handled"]
    fake = {"thinks": [], "report": "Net revenue is 100.",
            "calls": [{"call": {"name": "finance_calculator", "arguments": {"steps": []}},
                       "result": {"results": {}, "errors": {}}, "unsupported_inputs": []}]}
    row = score(fake, gold)
    assert not row["missing_handled"] and row["why"] == "called the calculator anyway"


def test_no_call_is_a_false_refusal():
    ex = build_example(13, CFG, "payback", missing=False, split="eval")
    row = score({"thinks": [], "calls": [], "report": "I cannot help."}, _gold(ex))
    assert row["false_refusal"] and not row["numeric_correct"]


def test_realworld_cases_have_finite_truth():
    cases = build_cases()
    assert len(cases) >= 15 and sum(1 for c in cases if c.missing) >= 3
    for c in cases:
        for vals in c.truth.values():
            assert vals and all(math.isfinite(v) for v in vals)
        assert c.needed_sections
    be = next(c for c in cases if c.case_id == "rw-14")
    assert be.truth["units_after"][0] > be.truth["units_now"][0]
