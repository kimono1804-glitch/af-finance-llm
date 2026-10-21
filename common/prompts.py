"""System prompt, tool schema and message builders shared by data generation, training and inference.

Changing anything here changes what the model was trained on: regenerate the dataset and retrain.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from common.tools import TOOL_NAME, formula_signatures

SYSTEM_PROMPT = (
    "You are a startup CFO assistant. Answer only from the figures inside <financial_context>.\n"
    "Inside <think>: identify the period and metric asked for, extract every input with the table and row it "
    "came from, say which tables are irrelevant, and state the formula you will use.\n"
    f"Never do arithmetic yourself. Call {TOOL_NAME} with the extracted inputs and let it compute.\n"
    "If a required input is not in the context, do not call the tool and do not guess: state exactly which "
    "variable is missing.\n"
    "After the tool returns, write the final CFO report as Markdown, copying the tool's display values exactly."
)

TOOL_DESCRIPTION = (
    "Deterministic finance calculator. Send ordered steps; each step applies one formula. "
    "Inputs are numbers copied from the context (no thousands separators), '@step_id' to reuse an earlier "
    "step's result, or lists of those for list inputs. Append _pct to an input name to pass a percentage as "
    "written (churn_rate_pct: 3.5 means 3.5%). Formulas:\n" + formula_signatures()
)

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "snake_case result name"},
                                "formula": {"type": "string"},
                                "inputs": {"type": "object"},
                            },
                            "required": ["id", "formula", "inputs"],
                        },
                    }
                },
                "required": ["steps"],
            },
        },
    }
]


def build_context(chunks: Sequence[str]) -> str:
    body = "\n\n".join(f"[{i}] {c}" for i, c in enumerate(chunks, start=1))
    return f"<financial_context>\n{body}\n</financial_context>"


def build_user_message(chunks: Sequence[str], question: str) -> str:
    return f"{build_context(chunks)}\n\nQuestion: {question.strip()}"


# ---------------------------------------------------------------------------
# Parsing model output (Qwen3 format)
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def split_think(text: str) -> tuple[str, str]:
    """Return (think_text, remainder). Handles a missing opening tag (template may pre-fill it)."""
    m = _THINK_RE.search(text)
    if m:
        return m.group(1).strip(), (text[: m.start()] + text[m.end():]).strip()
    if "</think>" in text:
        head, tail = text.split("</think>", 1)
        return head.replace("<think>", "").strip(), tail.strip()
    return "", text.strip()


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract {"name", "arguments"} dicts from <tool_call> blocks. Malformed JSON is returned as an error entry."""
    calls: list[dict[str, Any]] = []
    for raw in _TOOL_RE.findall(text):
        try:
            obj = json.loads(raw)
            calls.append({"name": obj.get("name", ""), "arguments": obj.get("arguments", {})})
        except json.JSONDecodeError as exc:
            calls.append({"name": "_malformed", "arguments": {}, "error": f"bad JSON in tool call: {exc}"})
    return calls


def strip_tool_calls(text: str) -> str:
    return _TOOL_RE.sub("", text).strip()
