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
    examples = generate(_cfg(200))
    tasks = Counter(e["meta"]["task"] for e in examples)
    assert set(tasks) == {"net_revenue", "contribution_margin", "ebitda", "break_even", "ltv_cac", "payback"}
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
