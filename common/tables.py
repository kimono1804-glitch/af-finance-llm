"""Markdown rendering for financial tables.

Used by BOTH the synthetic data generator and the real-document ingester, so the model
sees identically-shaped chunks in training and at inference. Every chunk carries its
title, source, sheet, section, periods and units, plus the column headers and row labels.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from common.schemas import ChunkMetadata


def format_cell(value: object, is_percent: bool = False) -> str:
    """Render a cell the way a finance team would read it."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return ""
        if is_percent:
            return f"{value * 100:.2f}%"
        if abs(value - round(value)) < 1e-9:
            return f"{int(round(value)):,}"
        return f"{value:,.2f}"
    return str(value).strip().replace("|", "/").replace("\n", " ")


_INT_CELL = re.compile(r"^\(?-?\d{1,3}(?:,\d{3})*\)?$")
_DEC_CELL = re.compile(r"^\(?-?\d{1,3}(?:,\d{3})*\.\d+\)?$")


def align_row_decimals(row: Sequence[str]) -> list[str]:
    """If any number in a row has decimals, show whole numbers in that row with the same decimals.

    '1,240' between '1,240.04' and '1,239.99' reads like a different kind of value and led the model to take
    the neighbouring column. Label cells and percentages are left alone.
    """
    cells = list(row)
    decs = [len(c.split(".")[1].rstrip(")")) for c in cells if _DEC_CELL.match(c)]
    if not decs:
        return cells
    places = max(decs)
    out = []
    for c in cells:
        if _INT_CELL.match(c):
            c = c[:-1] + "." + "0" * places + ")" if c.endswith(")") else c + "." + "0" * places
        out.append(c)
    return out


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def period_span(periods: Sequence[str]) -> str:
    if not periods:
        return "n/a"
    if len(periods) == 1:
        return periods[0]
    return f"{periods[0]} to {periods[-1]}"


def render_chunk(title: str, meta: ChunkMetadata, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    """Header line + metadata line + Markdown table. Rows are already-formatted strings."""
    meta_parts = [f"Source: {meta.source}", f"Sheet: {meta.sheet}", f"Section: {meta.section}"]
    if meta.periods:
        meta_parts.append(f"Periods: {period_span(meta.periods)}")
    if meta.units:
        meta_parts.append(f"Units: {meta.units}")
    if meta.page is not None:
        meta_parts.append(f"Page: {meta.page}")
    return f"#### {title}\n" + " | ".join(meta_parts) + "\n" + markdown_table(headers, [align_row_decimals(r) for r in rows])


def render_text_chunk(title: str, meta: ChunkMetadata, text: str) -> str:
    """Narrative (non-table) chunk, e.g. notes from a PDF board pack."""
    meta_parts = [f"Source: {meta.source}", f"Section: {meta.section}"]
    if meta.page is not None:
        meta_parts.append(f"Page: {meta.page}")
    return f"#### {title}\n" + " | ".join(meta_parts) + "\n" + text.strip()
