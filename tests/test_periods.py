from common.periods import resolve_relative_periods

PERIODS = ["Apr-25", "May-25", "Dec-25", "Jan-26", "Feb-26", "Mar-26", "Q3 FY26", "Q4 FY26", "FY27 Plan"]


def test_latest_and_previous_month():
    q, notes = resolve_relative_periods("What was CM2 last month?", PERIODS)
    assert q == "What was CM2 Mar-26?" and "Mar-26" in notes[0]
    q, _ = resolve_relative_periods("Compare with the previous month", PERIODS)
    assert q == "Compare with the Feb-26"


def test_year_boundary_ordering():
    q, _ = resolve_relative_periods("runway this month", ["Nov-25", "Dec-25", "Jan-26"])
    assert q == "runway Jan-26"


def test_quarters():
    q, _ = resolve_relative_periods("magic number for the latest quarter", PERIODS)
    assert q == "magic number for the Q4 FY26"
    q, _ = resolve_relative_periods("magic number for the prior quarter", PERIODS)
    assert q == "magic number for the Q3 FY26"


def test_unresolvable_and_untouched():
    q, notes = resolve_relative_periods("magic number last quarter", ["Mar-26"])
    assert q == "magic number last quarter" and "Could not resolve" in notes[0]
    q, notes = resolve_relative_periods("What was EBITDA for Mar-26?", PERIODS)
    assert q == "What was EBITDA for Mar-26?" and notes == []


from common.periods import asked_periods, missing_periods, period_not_available_report, quarter_months  # noqa: E402


def test_asked_periods_phrasings():
    assert asked_periods("LTV:CAC for Mar-25?") == ["Mar-25"]
    assert asked_periods("payback in Sep 2025 vs September 2025 vs Sep '25") == ["Sep-25"]
    assert asked_periods("magic number for Q4 of FY2026 and Q1 FY27") == ["Q4 FY26", "Q1 FY27"]


def test_asked_periods_ignores_lookalikes():
    for q in ("What is our gross margin 40% target?", "Why did churn decrease 10 points?",
              "What is the break-even under the FY27 plan?", "We may 25 hires later", "Market 25 share"):
        assert asked_periods(q) == [], q


def test_quarter_months_indian_fy():
    assert quarter_months("Q1 FY26") == ["Apr-25", "May-25", "Jun-25"]
    assert quarter_months("Q4 FY26") == ["Jan-26", "Feb-26", "Mar-26"]


def test_missing_periods():
    months = [f"{m}-25" for m in ("Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")] + ["Jan-26", "Feb-26", "Mar-26"]
    assert missing_periods("LTV:CAC for Mar-25?", months) == ["Mar-25"]
    assert missing_periods("net revenue in Apr 2026", months) == ["Apr-26"]
    assert missing_periods("CM2 for Mar-26", months) == []
    assert missing_periods("magic number Q4 FY26", months) == []          # all three months present
    assert missing_periods("magic number Q1 FY25", months) == ["Q1 FY25"]
    assert missing_periods("break-even under the FY27 plan", months) == []
    report = period_not_available_report(["Mar-25"], months, ["mis.xlsx"])
    assert "Mar-25" in report and "Apr-25 to Mar-26" in report and "not available" in report.lower()
