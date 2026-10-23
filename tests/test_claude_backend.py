"""common.claude_llm: message conversion and output format, with a fake client (no network, no API key)."""

from types import SimpleNamespace as NS

import pytest

pytest.importorskip("anthropic")

from common.claude_llm import ClaudeLLM, anthropic_tools  # noqa: E402
from common.prompts import build_tools  # noqa: E402


class FakeMessages:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def _resp(content, stop="end_turn"):
    return NS(content=content, stop_reason=stop, model="claude-haiku-5-5", usage=NS(input_tokens=10, output_tokens=5))


def test_tools_match_the_fine_tuned_contract():
    tools = anthropic_tools("v2")
    assert [t["name"] for t in tools] == [t["function"]["name"] for t in build_tools("v2")]
    assert tools[0]["input_schema"] == build_tools("v2")[0]["function"]["parameters"]


def test_tool_round_trip_replays_raw_blocks(monkeypatch):
    llm = ClaudeLLM.__new__(ClaudeLLM)  # skip __init__ (no client / key needed)
    llm.model, llm.effort, llm.max_tokens, llm.fallbacks = "claude-haiku-5-5", "high", 16000, False
    llm.tools, llm._turns, llm.usage = anthropic_tools("v2"), [], {"input_tokens": 0, "output_tokens": 0, "requests": 0}
    turn1 = [NS(type="thinking", thinking="need net revenue"),
             NS(type="tool_use", id="tu_1", name="finance_calculator", input={"steps": []})]
    turn2 = [NS(type="text", text="## CFO Report")]
    fake = FakeMessages([_resp(turn1, "tool_use"), _resp(turn2)])
    llm.client = NS(messages=fake)

    msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q"}]
    out1 = llm.generate(msgs, 1536, 0.0)
    assert out1.startswith("<think>\nneed net revenue") and '"name": "finance_calculator"' in out1
    assert "temperature" not in fake.calls[0] and fake.calls[0]["system"] == "SYS"

    msgs += [{"role": "assistant", "content": "<think>\nx\n</think>\n\n", "tool_calls": []},
             {"role": "tool", "content": '{"results": {}}'}]
    out2 = llm.generate(msgs, 1536, 0.0)
    sent = fake.calls[1]["messages"]
    assert sent[1] == {"role": "assistant", "content": turn1}  # raw blocks replayed unchanged
    assert sent[2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu_1", "content": '{"results": {}}'}
    assert "## CFO Report" in out2
    assert len(llm._turns) == 2
