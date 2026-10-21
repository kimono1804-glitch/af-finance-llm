"""Table-aware ingestion: Excel (openpyxl + pandas) and PDF (Docling) -> Markdown table chunks.

Rules (from CLAUDE.md):
- Every chunk carries its column headers, row labels, period(s), units and source metadata.
- A value is never separated from its header: long sections are split by rows and each piece repeats
  the header row.
- Chunks are rendered with common.tables.render_chunk, the same format the model was trained on.

Excel layout expected (typical MIS export):
  optional title rows ("Monthly P&L", "Units: INR")
  header row: first cell = row-label header ("Line Item"), other cells = periods or column names
  data rows; a row with a label but no values starts a new section ("Revenue", "Variable Costs")

Usage:
  python ingest_financials.py sample_startup_mis.xlsx [more.xlsx board_pack.pdf ...] [--out chunks.jsonl]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
import yaml

from common.schemas import ChunkMetadata, TableChunk
from common.tables import format_cell, render_chunk, render_text_chunk

ROOT = Path(__file__).resolve().parent

PERIOD_RE = re.compile(
    r"^(?:[A-Z][a-z]{2}-\d{2}|[A-Z][a-z]{2} \d{4}|FY\s?\d{2,4}(?:[- ]?Q[1-4])?|Q[1-4][- ]?(?:FY)?\s?\d{2,4}"
    r"|\d{4}-\d{2}|H[12][- ]?(?:FY)?\d{2,4}|FY\d{2} Plan|Base|Bull|Bear)$"
)
UNITS_RE = re.compile(r"(?:units?|currency)\s*[:=]\s*(.+)$", re.IGNORECASE)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40]


def _header_text(v: object) -> str:
    if isinstance(v, (dt.datetime, dt.date)):
        return v.strftime("%b-%y")
    return format_cell(v)


def _is_empty(v: object) -> bool:
    return v is None or (isinstance(v, float) and v != v) or (isinstance(v, str) and not v.strip())


def split_sections(rows: Sequence[Sequence[str]], default: str) -> list[tuple[str, list[list[str]]]]:
    """Group data rows into sections. A row with a label and no values is a section header."""
    sections: list[tuple[str, list[list[str]]]] = []
    name, cur = default, []
    for r in rows:
        label, values = r[0], r[1:]
        if label and all(not v for v in values):
            if cur:
                sections.append((name, cur))
            name, cur = label, []
            continue
        if not label and all(not v for v in values):
            continue
        cur.append(list(r))
    if cur:
        sections.append((name, cur))
    return sections


def make_chunks(*, title: str, source: str, sheet: str, headers: list[str], rows: list[list[str]],
                units: str, default_section: str, extra_periods: Iterable[str] = (), page: int | None = None,
                max_rows: int = 12) -> list[TableChunk]:
    periods = [h for h in headers[1:] if PERIOD_RE.match(h)] or list(extra_periods)
    chunks: list[TableChunk] = []
    for section, sec_rows in split_sections(rows, default_section):
        for start in range(0, len(sec_rows), max_rows):
            piece = sec_rows[start:start + max_rows]
            ctitle = f"{title} - {section}" if section != title else title
            meta = ChunkMetadata(source=source, sheet=sheet, section=section, periods=periods,
                                 row_labels=[r[0] for r in piece], units=units, page=page)
            cid = f"{_slug(Path(source).stem)}::{_slug(sheet)}::{_slug(section)}::{start // max_rows}"
            chunks.append(TableChunk(chunk_id=cid, title=ctitle, text=render_chunk(ctitle, meta, headers, piece),
                                     metadata=meta))
    return chunks


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def ingest_excel(path: Path, max_rows: int = 12) -> list[TableChunk]:
    import openpyxl

    wb = openpyxl.load_workbook(path, data_only=True)  # cached values, not formulas
    chunks: list[TableChunk] = []
    for ws in wb.worksheets:
        # pandas frame of raw values (structure), parallel frame of "is percent" flags (formatting)
        values = pd.DataFrame([[c.value for c in row] for row in ws.iter_rows()])
        if values.empty:
            continue
        pct = pd.DataFrame([["%" in (c.number_format or "") for c in row] for row in ws.iter_rows()])
        keep_cols = [c for c in values.columns if not values[c].map(_is_empty).all()]
        values, pct = values[keep_cols], pct[keep_cols]
        values = values[~values.map(_is_empty).all(axis=1)]
        pct = pct.loc[values.index]
        if values.empty:
            continue

        # Header row: first row whose first cell is text and that has >= 2 non-empty cells.
        header_idx = None
        for i, row in values.iterrows():
            non_empty = [v for v in row if not _is_empty(v)]
            if isinstance(row.iloc[0], str) and len(non_empty) >= 2:
                header_idx = i
                break
        if header_idx is None:
            continue

        title_rows = [str(r.iloc[0]).strip() for i, r in values.iterrows() if i < header_idx and not _is_empty(r.iloc[0])]
        title, units, extra_periods = ws.title, "", []
        for t in title_rows:
            m = UNITS_RE.search(t)
            if m:
                units = m.group(1).strip()
            elif PERIOD_RE.match(t):
                extra_periods.append(t)
            elif title == ws.title:
                title = t
        headers = [_header_text(v) or f"Col{j}" for j, v in enumerate(values.loc[header_idx])]
        rows = []
        for i in values.index[values.index > header_idx]:
            row = values.loc[i]
            rows.append([format_cell(v, bool(pct.loc[i, c])) if not _is_empty(v) else "" for c, v in row.items()])
        chunks += make_chunks(title=title, source=path.name, sheet=ws.title, headers=headers, rows=rows, units=units,
                              default_section=ws.title, extra_periods=extra_periods, max_rows=max_rows)
    return chunks


# ---------------------------------------------------------------------------
# PDF (Docling)
# ---------------------------------------------------------------------------

def ingest_pdf(path: Path, max_rows: int = 12, text_chunk_chars: int = 1200) -> list[TableChunk]:
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import SectionHeaderItem, TableItem, TextItem

    doc = DocumentConverter().convert(str(path)).document
    chunks: list[TableChunk] = []
    section, buf, buf_page = "Document", [], None

    def flush_text() -> None:
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            meta = ChunkMetadata(source=path.name, sheet="text", section=section, page=buf_page)
            cid = f"{_slug(path.stem)}::text::{_slug(section)}::{len(chunks)}"
            chunks.append(TableChunk(chunk_id=cid, title=section, text=render_text_chunk(section, meta, text), metadata=meta))
        buf = []

    for item, _level in doc.iterate_items():
        page = item.prov[0].page_no if getattr(item, "prov", None) else None
        if isinstance(item, TableItem):
            df = item.export_to_dataframe(doc=doc)
            if df.empty:
                continue
            headers = [_header_text(c) for c in df.columns]
            rows = [[format_cell(v) for v in r] for r in df.itertuples(index=False)]
            caption = item.caption_text(doc) or section
            chunks += make_chunks(title=caption, source=path.name, sheet=f"page {page}", headers=headers, rows=rows,
                                  units="", default_section=caption, page=page, max_rows=max_rows)
        elif isinstance(item, SectionHeaderItem):
            flush_text()
            section = item.text.strip() or section
        elif isinstance(item, TextItem):
            if not buf:
                buf_page = page
            buf.append(item.text)
            if sum(len(b) for b in buf) > text_chunk_chars:
                flush_text()
    flush_text()
    return chunks


# ---------------------------------------------------------------------------

def ingest_paths(paths: Iterable[str | Path], max_rows: int = 12) -> list[TableChunk]:
    out: list[TableChunk] = []
    for p in map(Path, paths):
        suffix = p.suffix.lower()
        if suffix in (".xlsx", ".xlsm"):
            out += ingest_excel(p, max_rows)
        elif suffix == ".pdf":
            out += ingest_pdf(p, max_rows)
        else:
            raise ValueError(f"unsupported file type: {p}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--config", default=str(ROOT / "configs" / "rag.yaml"))
    ap.add_argument("--out", help="write chunks as JSONL (keep real data inside git-ignored folders)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    chunks = ingest_paths(args.paths, cfg["max_rows_per_chunk"])
    for c in chunks:
        print(f"\n=== {c.chunk_id}\n{c.text}")
    print(f"\n{len(chunks)} chunks", file=sys.stderr)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(c.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
