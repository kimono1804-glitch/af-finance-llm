"""Resolve relative period phrases ("last month", "latest quarter") to periods that exist in the ingested data.

The model is trained on explicit periods ("Mar-26", "Q4 FY26"), so questions are rewritten before retrieval.
"Last month" means the latest month present in the MIS (the last closed month), not the calendar month.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_RE = re.compile(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-(\d{2})$")
_QUARTER_RE = re.compile(r"^Q([1-4]) FY(\d{2})$")

_LATEST_MONTH = re.compile(r"\b(?:last|latest|this|current|most recent|the recent)\s+month\b", re.IGNORECASE)
_PREVIOUS_MONTH = re.compile(r"\b(?:previous|prior)\s+month\b|\bthe month before(?: last)?\b", re.IGNORECASE)
_LATEST_QUARTER = re.compile(r"\b(?:last|latest|this|current|most recent)\s+quarter\b", re.IGNORECASE)
_PREVIOUS_QUARTER = re.compile(r"\b(?:previous|prior)\s+quarter\b", re.IGNORECASE)


def month_key(label: str) -> tuple[int, int] | None:
    m = _MONTH_RE.match(label)
    return (int(m.group(2)), _MONTHS.index(m.group(1))) if m else None


def quarter_key(label: str) -> tuple[int, int] | None:
    m = _QUARTER_RE.match(label)
    return (int(m.group(2)), int(m.group(1))) if m else None


# Explicit periods written in a question. Strict on purpose: only real month names / abbreviations followed by a year,
# so "margin 40" or "decrease 10" never read as a month.
_ASKED_MONTH_RE = re.compile(
    r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?(?:\s*[-,/]\s*|\s+|\s*['’])(?:20)?(\d{2})(?![\d%])", re.IGNORECASE)
_ASKED_QUARTER_RE = re.compile(r"\bQ([1-4])\s*(?:of\s+)?FY\s?(?:20)?(\d{2})(?!\d)", re.IGNORECASE)


def asked_periods(question: str) -> list[str]:
    """Explicit months / quarters in a question, as table labels ('Mar-25', 'Q4 FY26'), in order, no duplicates."""
    found: list[tuple[int, str]] = []
    for m in _ASKED_MONTH_RE.finditer(question):
        word, tail = m.group(1), m.group(0)[len(m.group(1)):]
        if word == "may" and tail.strip()[:1].isdigit() and len(tail.strip()) == 2:
            continue  # lowercase "may 25" is usually the verb ("we may 25 ..."); "May-25", "may 2025" still count
        found.append((m.start(), f"{m.group(1)[:3].title()}-{m.group(2)}"))
    for m in _ASKED_QUARTER_RE.finditer(question):
        found.append((m.start(), f"Q{m.group(1)} FY{m.group(2)}"))
    return list(dict.fromkeys(label for _, label in sorted(found)))


def quarter_months(label: str) -> list[str]:
    """Indian FY quarter -> its months: Q1 FY26 = Apr-25, May-25, Jun-25; Q4 FY26 = Jan-26, Feb-26, Mar-26."""
    key = quarter_key(label)
    if key is None:
        return []
    fy, q = key
    out = []
    for k in range(3):
        idx = 3 + (q - 1) * 3 + k            # months since Jan of the FY's start year (Apr = 3)
        year = fy - 1 + idx // 12
        out.append(f"{_MONTHS[idx % 12]}-{year % 100:02d}")
    return out


def missing_periods(question: str, periods: Iterable[str]) -> list[str]:
    """Periods the question asks for that the data does not contain.

    A quarter counts as present if its label is in the data or all three of its months are.
    """
    labels = set(periods)
    out = []
    for p in asked_periods(question):
        if p in labels:
            continue
        if quarter_key(p) and all(m in labels for m in quarter_months(p)):
            continue
        out.append(p)
    return out


def coverage(periods: Iterable[str]) -> str:
    """'Apr-25 to Mar-26' (months), else quarters, else 'no dated periods'."""
    labels = set(periods)
    months = sorted((p for p in labels if month_key(p)), key=month_key)  # type: ignore[arg-type]
    if months:
        return months[0] if len(months) == 1 else f"{months[0]} to {months[-1]}"
    quarters = sorted((p for p in labels if quarter_key(p)), key=quarter_key)  # type: ignore[arg-type]
    if quarters:
        return quarters[0] if len(quarters) == 1 else f"{quarters[0]} to {quarters[-1]}"
    return "no dated periods"


def period_not_available_report(missing: list[str], periods: Iterable[str], sources: Iterable[str] = ()) -> str:
    """Deterministic answer when the asked period is not in the data. No numbers from other periods."""
    asked = ", ".join(missing)
    cov = coverage(periods)
    src = ", ".join(sorted(set(sources)))
    lines = [f"## Data not available: {asked}", "",
             f"The question asks for **{asked}**, but the uploaded data covers **{cov}**"
             + (f" (source: {src})." if src else "."),
             f"No calculation was done, so numbers from another period are not reported as {asked}.", "",
             f"To answer, upload the MIS that includes {asked}, or ask about a period within {cov}."]
    return "\n".join(lines)


def resolve_relative_periods(question: str, periods: Iterable[str]) -> tuple[str, list[str]]:
    """Return (rewritten question, notes). Periods come from chunk metadata."""
    labels = set(periods)
    months = sorted((p for p in labels if month_key(p)), key=month_key)  # type: ignore[arg-type]
    quarters = sorted((p for p in labels if quarter_key(p)), key=quarter_key)  # type: ignore[arg-type]
    notes: list[str] = []

    def sub(pattern: re.Pattern[str], options: list[str], back: int, what: str, q: str) -> str:
        if not pattern.search(q):
            return q
        if len(options) < back:
            notes.append(f"Could not resolve '{pattern.search(q).group(0)}': no {what} periods in the data.")
            return q
        target = options[-back]
        notes.append(f"Interpreted '{pattern.search(q).group(0)}' as {target} ({'latest' if back == 1 else 'second latest'} "
                     f"{what} in the data).")
        return pattern.sub(target, q)

    q = sub(_PREVIOUS_MONTH, months, 2, "month", question)
    q = sub(_LATEST_MONTH, months, 1, "month", q)
    q = sub(_PREVIOUS_QUARTER, quarters, 2, "quarter", q)
    q = sub(_LATEST_QUARTER, quarters, 1, "quarter", q)
    return q, notes
