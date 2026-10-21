"""PDF ingestion through Docling. Slow (loads Docling layout/table models): run with `pytest -m slow`."""

import pytest

from ingest_financials import ingest_excel, ingest_pdf
from make_sample_mis import build_board_pack_pdf, build_workbook


@pytest.mark.slow
def test_board_pack_pdf_matches_excel(tmp_path):
    pdf_chunks = ingest_pdf(build_board_pack_pdf(tmp_path / "board_pack.pdf"))
    xls_chunks = ingest_excel(build_workbook(tmp_path / "mis.xlsx"))

    def rows(chunks, section):
        c = next(c for c in chunks if c.metadata.section == section)
        return {line.split("|")[1].strip(): line for line in c.text.splitlines()[4:]}

    for section in ("Revenue", "Variable Costs", "Profit Summary"):
        assert rows(pdf_chunks, section) == rows(xls_chunks, section), section
    text = next(c for c in pdf_chunks if c.metadata.sheet == "text")
    assert "festive" in text.text and text.metadata.page == 1
    ue = next(c for c in pdf_chunks if "Unit Economics" in c.title)
    assert "| Monthly Churn % | 3.85% |" in ue.text and ue.metadata.periods[-1] == "Mar-26"
