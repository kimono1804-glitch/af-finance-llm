import json
from collections import Counter
from pathlib import Path

import yaml

from generate_raft_data import build_example, generate, validate_example

ROOT = Path(__file__).resolve().parents[1]


def _cfg(n: int = 120) -> dict:
    cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    cfg["n_examples"] = n
    return cfg


def test_generated_examples_validate():
    examples = generate(_cfg())
    bad = {e["id"]: validate_example(e) for e in examples}
    assert not {k: v for k, v in bad.items() if v}


def test_deterministic():
    cfg = _cfg(20)
    a = [json.dumps(e) for e in generate(cfg)]
    b = [json.dumps(e) for e in generate(cfg)]
    assert a == b


def test_all_tasks_and_missing_examples_present():
    cfg = _cfg(300)
    examples = generate(cfg)
    tasks = Counter(e["meta"]["task"] for e in examples)
    assert set(tasks) == set(cfg["task_weights"])
    assert any(e["meta"]["missing"] for e in examples)
    assert any(e["meta"]["split"] == "test" for e in examples)


def test_conversation_shape():
    ex = build_example(3, _cfg(), "contribution_margin", missing=False, split="train")
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert ex["messages"][2]["content"].startswith("<think>")
    assert "<financial_context>" in ex["messages"][1]["content"]
    assert ex["messages"][4]["content"].count("<think>") == 1


def test_missing_example_has_no_tool_call():
    ex = build_example(5, _cfg(), "ltv_cac", missing=True, split="train")
    assert ex["meta"]["missing"]
    assert len(ex["messages"]) == 3 and "tool_calls" not in ex["messages"][2]
    assert "missing" in ex["messages"][2]["content"].lower()


def test_new_task_families_validate_with_edge_cases():
    cfg = _cfg()
    cfg["edge_cases"] = {**cfg["edge_cases"], "cash_positive": 0.5, "arr_decline": 0.5}
    for task in ("runway", "burn_multiple", "magic_number", "gross_margin"):
        for i in range(12):
            ex = build_example(2000 + i, cfg, task, missing=(task in ("magic_number", "gross_margin") and i % 4 == 0),
                               split="train")
            assert not validate_example(ex), (task, i, validate_example(ex))


def test_period_gap_examples():
    """Asked month outside every table: one assistant turn, no tool call, names the month, month not in context."""
    cfg = _cfg()
    cfg["missing_period_share"] = 1.0
    for task in ("ltv_cac", "ebitda", "runway", "burn_multiple"):
        ex = build_example(7, cfg, task, missing=True, split="train")
        assert ex["meta"]["missing_kind"] == "period", task
        asked = ex["meta"]["period"]
        context = ex["messages"][1]["content"].rsplit("Question:", 1)[0]
        assert asked not in context
        assert len(ex["messages"]) == 3 and "tool_calls" not in ex["messages"][2]
        assert asked in ex["messages"][2]["content"]
        assert not validate_example(ex), validate_example(ex)


def _question(ex: dict) -> str:
    return ex["messages"][1]["content"].rsplit("Question:", 1)[-1].strip()


def test_question_variation_keeps_period_and_differs_from_v1():
    from generate_raft_data import TASKS

    cfg = dict(_cfg(), scaled_units=0.0)  # wording only; lakh/crore scaling is tested in test_units_contract
    legacy = dict(cfg, compat_v1=True)
    changed = 0
    for i, task in enumerate(sorted(TASKS)):
        for j in range(15):
            idx = 500 + i * 100 + j
            ex, old = build_example(idx, cfg, task, False, "train"), build_example(idx, legacy, task, False, "train")
            q = _question(ex)
            assert not validate_example(ex)
            assert "lTV" not in q and not q[:1].isdigit()
            changed += q != _question(old)
            # the variation has its own rng: the tables and the answer stay the same
            assert ex["meta"]["truth"] == old["meta"]["truth"]
    assert changed > 30


def test_compat_v1_has_no_period_gaps():
    cfg = dict(_cfg(), compat_v1=True)
    exs = [build_example(900 + i, cfg, "net_revenue", True, "train") for i in range(30)]
    assert all(e["meta"].get("missing_kind") != "period" for e in exs)


def test_quarterly_distractor_note():
    cfg = _cfg()
    found = 0
    for i in range(60):
        ex = build_example(2000 + i, cfg, "ltv_cac", False, "train")
        ctx = ex["messages"][1]["content"]
        if "Quarterly Summary" in ctx.split("Question:")[0]:
            assert "has quarter totals" in ex["messages"][2]["content"]
            found += 1
    assert found > 0
