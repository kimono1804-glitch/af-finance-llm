"""Claude models as an evaluation backend (comparison baseline), through the official Anthropic SDK.

Same interface as run_assistant.FinanceLLM / ServerLLM: generate(messages, max_new_tokens, temperature) -> text in the
Qwen3 format the pipeline parses ("<think>...</think>" + "<tool_call>{json}</tool_call>" blocks). The model gets the
same system prompt and the same finance_calculator tool as the fine-tuned model.

Credentials: anthropic.Anthropic() resolves ANTHROPIC_API_KEY (or an `ant auth login` profile).

Notes
- Thinking is adaptive (always on for the current Opus / Sonnet). Each turn's raw response blocks are kept and replayed
  unchanged on the next request of the same conversation, which thinking blocks require.
- temperature / top_p are not sent (current models reject them); depth is set with output_config.effort.
- Server-side refusal fallback ("default" routing) is on for Opus / Sonnet. A case answered by a fallback model is
  marked in its think trace ("[served by ...]") so it can be excluded from a per-model comparison.
"""

from __future__ import annotations

import json
from typing import Any

from common.prompts import CONTRACT, build_tools

FALLBACK_BETA = "server-side-fallback-2026-07-01"
NO_FALLBACK = ("claude-haiku",)  # no server-side fallback for Haiku


def anthropic_tools(contract: str) -> list[dict[str, Any]]:
    """OpenAI-style tool list (what the fine-tuned model sees) -> Anthropic tool definitions."""
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]} for t in build_tools(contract)]


class ClaudeLLM:
    def __init__(self, model: str = "claude-opus-5-5", contract: str = CONTRACT, effort: str | None = "high",
                 max_tokens: int = 16000):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.contract = contract
        self.effort = effort
        self.max_tokens = max_tokens
        self.tools = anthropic_tools(contract)
        self.fallbacks = not model.startswith(NO_FALLBACK)
        self.label = f"claude ({model}, effort {effort or 'default'})"
        self._turns: list[Any] = []  # raw content blocks of each assistant turn in the current conversation
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

    def _to_api(self, messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        rest = messages[1:] if system else messages
        n_assistant = sum(1 for m in rest if m["role"] == "assistant")
        if n_assistant < len(self._turns):  # a new conversation started: forget the old turns
            self._turns = self._turns[:n_assistant]
        if n_assistant > len(self._turns):
            raise RuntimeError("assistant turn without stored response blocks (history was not produced by this backend)")

        out: list[dict[str, Any]] = []
        pending: list[str] = []  # tool_use ids of the last assistant turn, in order
        k = 0
        for m in rest:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = self._turns[k]
                k += 1
                out.append({"role": "assistant", "content": blocks})
                pending = [b.id for b in blocks if getattr(b, "type", "") == "tool_use"]
            elif m["role"] == "tool":
                result = {"type": "tool_result", "tool_use_id": pending.pop(0) if pending else "missing",
                          "content": m["content"]}
                # All results of one assistant turn go back in a single user message.
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(result)
                else:
                    out.append({"role": "user", "content": [result]})
        return system, out

    def generate(self, messages: list[dict[str, Any]], max_new_tokens: int, temperature: float) -> str:
        system, api_messages = self._to_api(messages)
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max(self.max_tokens, max_new_tokens),
                                  "system": system, "tools": self.tools, "messages": api_messages,
                                  "thinking": {"type": "adaptive", "display": "summarized"}}
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            response = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
        else:
            response = self.client.messages.create(**kwargs)
        self.usage["requests"] += 1
        self.usage["input_tokens"] += response.usage.input_tokens
        self.usage["output_tokens"] += response.usage.output_tokens

        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            self._turns.append(response.content)
            return f"(refused by {self.model}: {category})"

        thinking = [b.thinking for b in response.content if b.type == "thinking" and getattr(b, "thinking", "")]
        text = "".join(b.text for b in response.content if b.type == "text")
        calls = [b for b in response.content if b.type == "tool_use"]
        served = getattr(response, "model", self.model)
        note = f"\n[served by {served}]" if served and not str(served).startswith(self.model) else ""
        if response.stop_reason == "max_tokens":
            note += "\n[stopped at max_tokens]"
        self._turns.append(response.content)

        out = f"<think>\n{chr(10).join(thinking)}{note}\n</think>\n\n{text}"
        for b in calls:
            out += "\n<tool_call>\n" + json.dumps({"name": b.name, "arguments": b.input}) + "\n</tool_call>"
        return out
