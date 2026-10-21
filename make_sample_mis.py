"""Build sample_startup_mis.xlsx: a synthetic (fake) D2C startup MIS for demos and tests.

Laid out like a typical founder MIS export: title rows, a "Units:" row, a header row of month columns
(real Excel dates formatted mmm-yy), section rows with no values, percent-formatted cells.
All numbers are deterministic (seeded). Derived rows (EBIT, Net Income, S&M total, Gross Margin %)
are computed with common.formulas and stored as values, so readers that ignore Excel formulas still work.

Usage:  python make_sample_mis.py [--out sample_startup_mis.xlsx]
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill

from common import formulas as F

ROOT = Path(__file__).resolve().parent
COMPANY = "Nimbus Botanics Pvt Ltd (synthetic sample)"
MONTHS = [dt.date(2025, 4, 1)] + [dt.date(2025 + (m // 12), m % 12 + 1, 1) for m in range(4, 15)]

# FY27 pricing plan (Pricing sheet). Shared with eval_cases.py for ground truth.
PRICING = {"price": 649, "variable_cost": 386.5, "fixed_costs": 9_850_000, "planned_hires": 1_420_000}

PL_SECTIONS = [
    ("Revenue", [("Gross Revenue", "gross_revenue"), ("Discounts & Coupons", "discounts"), ("Returns & Refunds", "returns")]),
    ("Cost of Revenue", [("COGS", "cogs")]),
    ("Variable Costs", [("Fulfillment & Logistics", "fulfillment"), ("Payment Gateway Fees", "payment_gateway"),
                        ("Packaging", "packaging")]),
    ("Marketing", [("Performance Marketing", "perf_marketing"), ("Brand Marketing", "brand_marketing")]),
    ("Operating Expenses", [("Salaries & Wages", "salaries"), ("Rent & Utilities", "rent"), ("Tech & Software", "tech"),
                            ("General & Admin", "g_and_a")]),
    ("D&A, Interest & Tax", [("Depreciation", "depreciation"), ("Amortization of Intangibles", "amortization"),
                             ("Amortization of Capitalized R&D", "amort_rnd"), ("Interest Expense", "interest"),
                             ("Income Tax", "taxes")]),
    ("Profit Summary", [("Operating Income (EBIT)", "ebit"), ("Net Profit / (Loss)", "net_income")]),
]

BOLD = Font(bold=True)
SECTION_FILL = PatternFill("solid", fgColor="DDEBF7")
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")


def r10(x: float) -> int:
    return int(round(x / 10.0)) * 10


def build_data(seed: int = 7) -> dict[str, list[float]]:
    rng = random.Random(seed)
    d: dict[str, list[float]] = {}
    hc = 34
    for t, _ in enumerate(MONTHS):
        if t in (3, 8):          # hiring jumps in Jul-25 and Dec-25 (step-fixed costs)
            hc += 6
        season = 1 + 0.12 * (t in (6, 7))   # Oct/Nov festive peak
        gr = r10(14_500_000 * (1.045 ** t) * season * rng.uniform(0.97, 1.03))
        disc = r10(gr * rng.uniform(0.07, 0.10) * (1.3 if season > 1 else 1))
        ret = r10(gr * rng.uniform(0.025, 0.04))
        nr = F.net_revenue(gr, disc, ret)
        cogs = r10(nr * rng.uniform(0.44, 0.47))
        ful, pg, pack = r10(nr * 0.072), r10(nr * 0.021), r10(nr * 0.018)
        perf = r10(nr * rng.uniform(0.17, 0.21))
        brand = r10(rng.uniform(450_000, 650_000))
        sal = r10(hc * 118_000)
        rent, tech, ga = 520_000, r10(310_000 + 6_000 * t), r10(rng.uniform(380_000, 460_000))
        dep, amort, rnd = 140_000, 60_000, 95_000
        ebit = F.difference(nr, F.total([cogs, ful, pg, pack, perf, brand, sal, rent, tech, ga, dep, amort, rnd]))
        interest = 85_000
        tax = r10(max(0.0, ebit - interest) * 0.25)
        ni = F.difference(F.difference(ebit, interest), tax)
        active = round(nr / 1_240)
        new = round(active * rng.uniform(0.11, 0.16))
        churn = round(rng.uniform(0.038, 0.052) * (1.35 if t == 9 else 1), 4)  # Jan-26 post-festive churn spike
        gm = round(F.gross_margin(nr, cogs), 4)
        row = dict(gross_revenue=gr, discounts=disc, returns=ret, cogs=cogs, fulfillment=ful, payment_gateway=pg,
                   packaging=pack, perf_marketing=perf, brand_marketing=brand, salaries=sal, rent=rent, tech=tech,
                   g_and_a=ga, depreciation=dep, amortization=amort, amort_rnd=rnd, interest=interest, taxes=tax,
                   ebit=ebit, net_income=ni, new_customers=new, active_customers=active,
                   arpu=round(nr / active, 2), gm=gm, churn=churn, sm_spend=F.total([perf, brand]), headcount=hc)
        for k, v in row.items():
            d.setdefault(k, []).append(v)
    # cash: opening balance after a seed round, burn from EBITDA-ish outflows
    cash = 182_000_000.0
    for t in range(len(MONTHS)):
        cash_in = r10(d["gross_revenue"][t] * 0.86)
        cash_out = r10(cash_in - d["ebit"][t] + 1_100_000)
        d.setdefault("opening_cash", []).append(cash)
        d.setdefault("cash_in", []).append(cash_in)
        d.setdefault("cash_out", []).append(cash_out)
        cash = cash + cash_in - cash_out
        d.setdefault("closing_cash", []).append(cash)
    return d


def _title(ws, title: str, units: str, extra: str | None = None) -> int:
    ws.append([title])
    ws["A1"].font = Font(bold=True, size=14)
    ws.append([f"Units: {units}"])
    row = 3
    if extra:
        ws.append([extra])
        row += 1
    ws.append([])
    return row + 1


def _header(ws, labels: list) -> None:
    ws.append(labels)
    for cell in ws[ws.max_row]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center")
        if isinstance(cell.value, dt.date):
            cell.number_format = "mmm-yy"


def _section(ws, name: str) -> None:
    ws.append([name])
    ws.cell(ws.max_row, 1).font = BOLD
    for c in range(1, len(MONTHS) + 2):
        ws.cell(ws.max_row, c).fill = SECTION_FILL


def _row(ws, label: str, values: list[float], fmt: str = "#,##0") -> None:
    ws.append([label, *values])
    for c in ws[ws.max_row][1:]:
        c.number_format = fmt


def build_workbook(out: Path) -> Path:
    d = build_data()
    wb = Workbook()

    ws = wb.active
    ws.title = "P&L"
    _title(ws, "Monthly P&L", "INR", COMPANY)
    _header(ws, ["Line Item", *MONTHS])
    for name, rows in PL_SECTIONS:
        _section(ws, name)
        for label, key in rows:
            _row(ws, label, d[key])
    ws.column_dimensions["A"].width = 34

    ws = wb.create_sheet("Unit Economics")
    _title(ws, "Unit Economics", "INR unless stated")
    _header(ws, ["Line Item", *MONTHS])
    _section(ws, "Customer Metrics")
    _row(ws, "New Customers", d["new_customers"])
    _row(ws, "Active Customers", d["active_customers"])
    _row(ws, "ARPU (monthly)", d["arpu"], "#,##0.00")
    _row(ws, "Gross Margin %", d["gm"], "0.00%")
    _row(ws, "Monthly Churn %", d["churn"], "0.00%")
    _row(ws, "Total S&M Spend", d["sm_spend"])
    ws.column_dimensions["A"].width = 26

    ws = wb.create_sheet("Pricing")
    _title(ws, "Pricing & Cost Structure", "INR per unit unless stated", "FY27 Plan")
    _header(ws, ["Item", "Value"])
    _section(ws, "Unit Cost & Fixed Cost Plan")
    for label, v, fmt in [("Average Selling Price (per unit)", PRICING["price"], "#,##0"),
                          ("Variable Cost per Unit", PRICING["variable_cost"], "#,##0.00"),
                          ("Monthly Fixed Costs", PRICING["fixed_costs"], "#,##0"),
                          ("Planned Hires - added monthly cost (from Apr-26)", PRICING["planned_hires"], "#,##0")]:
        ws.append([label, v])
        ws.cell(ws.max_row, 2).number_format = fmt
    ws.column_dimensions["A"].width = 48

    ws = wb.create_sheet("Cash")
    _title(ws, "Cash Position", "INR")
    _header(ws, ["Line Item", *MONTHS])
    _section(ws, "Bank")
    for label, key in [("Opening Cash Balance", "opening_cash"), ("Cash In", "cash_in"), ("Cash Out", "cash_out"),
                       ("Closing Cash Balance", "closing_cash")]:
        _row(ws, label, d[key])
    ws.column_dimensions["A"].width = 26

    ws = wb.create_sheet("Headcount")
    _title(ws, "Headcount by Department", "FTEs")
    _header(ws, ["Line Item", *MONTHS])
    _section(ws, "FTEs")
    hc = d["headcount"]
    _row(ws, "Engineering", [round(h * 0.38) for h in hc])
    _row(ws, "Sales & Growth", [round(h * 0.24) for h in hc])
    _row(ws, "Operations", [round(h * 0.26) for h in hc])
    _row(ws, "Total Headcount", hc)
    ws.column_dimensions["A"].width = 22

    wb.save(out)
    return out


def build_board_pack_pdf(out: Path) -> Path:
    """Same company as a board-pack PDF (narrative page + P&L + unit economics tables) for testing Docling ingestion."""
    from fpdf import FPDF

    d = build_data()
    labels = [m.strftime("%b-%y") for m in MONTHS]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(True, margin=10)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Nimbus Botanics - Board Pack FY26 (synthetic sample)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "CEO Commentary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, "Revenue grew through the festive season (Oct-Nov) and held up in Q4. Churn spiked in Jan-26 after "
                         "the festive cohort. Two hiring waves (Jul-25, Dec-25) raised payroll. The FY27 plan adds a further "
                         "hiring wave from Apr-26. All figures are in INR.")

    def table(title: str, rows: list[tuple[str, list[str]]]) -> None:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 6.5)
        with pdf.table(col_widths=(46, *([19] * len(labels))), text_align=("LEFT", *(["RIGHT"] * len(labels))),
                       line_height=4) as t:
            for label, cells in [("Line Item", labels), *rows]:
                r = t.row()
                r.cell(label)
                for c in cells:
                    r.cell(c)

    def fmt(v: float, pct: bool = False) -> str:
        return f"{v * 100:.2f}%" if pct else (f"{v:,.0f}" if abs(v - round(v)) < 1e-9 else f"{v:,.2f}")

    pl_rows: list[tuple[str, list[str]]] = []
    for name, rows in PL_SECTIONS:
        pl_rows.append((name, [""] * len(labels)))
        pl_rows += [(label, [fmt(x) for x in d[key]]) for label, key in rows]
    table("Monthly P&L (INR)", pl_rows)
    table("Unit Economics (INR unless stated)", [
        ("New Customers", [fmt(x) for x in d["new_customers"]]),
        ("ARPU (monthly)", [fmt(x) for x in d["arpu"]]),
        ("Gross Margin %", [fmt(x, True) for x in d["gm"]]),
        ("Monthly Churn %", [fmt(x, True) for x in d["churn"]]),
        ("Total S&M Spend", [fmt(x) for x in d["sm_spend"]]),
    ])
    pdf.output(str(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "sample_startup_mis.xlsx"))
    ap.add_argument("--pdf", help="also write a board-pack PDF of the same data to this path")
    args = ap.parse_args()
    out = build_workbook(Path(args.out))
    print(f"wrote {out}")
    if args.pdf:
        print(f"wrote {build_board_pack_pdf(Path(args.pdf))}")


if __name__ == "__main__":
    main()
