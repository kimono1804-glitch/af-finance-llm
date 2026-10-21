"""Faithfulness checks: every literal number the model passes to the calculator must exist in its context."""

from __future__ import annotations

import re
from typing import Any

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def context_numbers(text: str) -> set[float]:
    """All numbers in a context block, as absolute values ('1,250,000' -> 1250000.0, '3.25%' -> 3.25)."""
    return {abs(float(m.replace(",", ""))) for m in _NUM_RE.findall(text)}


def literal_numbers(obj: Any) -> list[float]:
    """Literal numbers inside tool-call inputs (skips '@step' references)."""
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        s = obj.strip().replace(",", "")
        try:
            return [float(s)]
        except ValueError:
            return []
    if isinstance(obj, list):
        return [x for o in obj for x in literal_numbers(o)]
    if isinstance(obj, dict):
        return [x for o in obj.values() for x in literal_numbers(o)]
    return []


def unsupported_inputs(arguments: dict[str, Any], context: str) -> list[str]:
    """Inputs of a finance_calculator call whose literal values do not appear in the context."""
    ctx = context_numbers(context)
    bad = []
    for step in arguments.get("steps", []) if isinstance(arguments, dict) else []:
        for key, val in (step.get("inputs") or {}).items():
            for v in literal_numbers(val):
                if abs(v) not in ctx:
                    bad.append(f"{step.get('id', '?')}.{key}={v:g}")
    return bad
