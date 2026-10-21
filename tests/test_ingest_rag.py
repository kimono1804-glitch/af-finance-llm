from pathlib import Path

import pytest

from common.audit import unsupported_inputs
from ingest_financials import ingest_excel, split_sections
from make_sample_mis import build_workbook
from rag_pipeline import HybridRetriever, normalize_periods, tokenize


@pytest.fixture(scope="module")
def chunks(tmp_path_factory):
    path = build_workbook(tmp_path_factory.mktemp("mis") / "sample_startup_mis.xlsx")
    return ingest_excel(Path(path))


def test_every_chunk_carries_headers_and_metadata(chunks):
    assert len(chunks) >= 10
    for c in chunks:
        lines = c.text.splitlines()
        assert lines[0].startswith("#### ")
        assert "Source: sample_startup_mis.xlsx" in lines[1]
        assert lines[2].startswith("| ") and lines[3].startswith("|---")
        assert c.metadata.row_labels
        assert c.metadata.periods


def test_dates_and_percents_rendered(chunks):
    ue = next(c for c in chunks if c.metadata.sheet == "Unit Economics")
    assert "Mar-26" in ue.text and ue.metadata.periods[0] == "Apr-25"
    churn_line = next(line for line in ue.text.splitlines() if line.startswith("| Monthly Churn %"))
    assert churn_line.count("%") >= 12


def test_sections_split_from_section_rows(chunks):
    sections = {c.metadata.section for c in chunks if c.metadata.sheet == "P&L"}
    assert {"Revenue", "Cost of Revenue", "Variable Costs", "Profit Summary"} <= sections


def test_long_sections_split_keep_header():
    rows = [["Revenue", "", ""]] + [[f"Item {i}", "1", "2"] for i in range(30)]
    secs = split_sections(rows, "Sheet")
    assert secs[0][0] == "Revenue" and len(secs[0][1]) == 30


def test_period_normalisation_and_tokens():
    assert normalize_periods("CM2 for March 2026") == "CM2 for Mar-26"
    assert normalize_periods("in Jan 26") == "in Jan-26"
    toks = tokenize("S&M spend Mar-26 P&L")
    assert {"s&m", "mar-26", "mar", "26", "p&l"} <= set(toks)


def test_bm25_retrieves_the_right_sections(chunks):
    r = HybridRetriever(load_models=False)
    r.index(chunks)
    titles = [h.chunk.title for h in r.search("What were CM1 and CM2 for Mar-26?", top_k=5)]
    for needed in ("Revenue", "Cost of Revenue", "Variable Costs"):
        assert any(needed in t for t in titles), titles
    titles = [h.chunk.title for h in r.search("LTV:CAC for Jan-26", top_k=3)]
    assert any("Unit Economics" in t for t in titles)


def test_faithfulness_audit():
    ctx = "| Gross Revenue | 1,250,000 |\n| Churn | 3.25% |"
    args = {"steps": [{"id": "x", "formula": "ltv", "inputs": {"arpu": 1250000, "churn_rate_pct": 3.25,
                                                                 "gross_margin": "@gm", "other": 999}}]}
    assert unsupported_inputs(args, ctx) == ["x.other=999"]
