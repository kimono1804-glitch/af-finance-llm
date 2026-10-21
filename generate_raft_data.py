"""Generate a deterministic synthetic RAFT dataset for the startup-finance model.

Python creates every number (company P&Ls, unit economics, pricing) and computes every answer
through common.tools / common.formulas. Reasoning traces are templates filled with those values;
no LLM is involved, so the same seed always gives the same file.

Each example is a Qwen3 tool-calling conversation:
  system -> user (<financial_context> with relevant + distractor tables, then the question)
  -> assistant (<think> extract inputs, pick formulas </think> + finance_calculator call)
  -> tool (results JSON)
  -> assistant (<think> check results </think> + Markdown CFO report)
Missing-data examples stop after one assistant turn that names the missing variable.

Usage:
  python generate_raft_data.py                    # write dataset.jsonl, then validate it
  python generate_raft_data.py --validate-only    # re-check an existing dataset.jsonl
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from common import formulas as F
from common.audit import unsupported_inputs
from common.periods import month_key
from common.prompts import SYSTEM_PROMPT, TOOLS, build_user_message
from common.schemas import ChunkMetadata
from common.tables import format_cell, render_chunk
from common.tools import TOOL_NAME, execute, response_to_json

ROOT = Path(__file__).resolve().parent
_MONTH_LABEL_RE = re.compile(r"^[A-Z][a-z]{2}-\d{2}$")
PROMPT_VERSION = hashlib.sha1((SYSTEM_PROMPT + json.dumps(TOOLS, sort_keys=True)).encode()).hexdigest()[:10]

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

NAME_A = ["Nimbus", "Kestrel", "Saffron", "Indigo", "Lumen", "Orbitly", "Juniper", "Cobalt", "Tandem", "Monsoon",
          "Pebble", "Quartzly", "Harborline", "Fennel", "Banyan", "Vertexa", "Marigold", "Zephyrin", "Tamarind", "Corvid"]
NAME_B = ["Labs", "Foods", "Health", "Wear", "Cloud", "Pay", "Home", "Botanics", "Logistics", "Learning",
          "Beauty", "Analytics", "Pets", "Mobility", "Brews"]

LABELS: dict[str, list[str]] = {
    # P&L
    "gross_revenue": ["Gross Revenue", "Gross Sales", "Gross Revenue (Recognized)"],
    "discounts": ["Discounts", "Discounts & Coupons", "Trade Discounts"],
    "returns": ["Returns", "Returns & Refunds", "Sales Returns"],
    "billings": ["Billings", "Invoiced Billings"],
    "deferred_change": ["Change in Deferred Revenue", "Deferred Revenue Movement"],
    "cogs": ["COGS", "Cost of Goods Sold", "Cost of Revenue"],
    "fulfillment": ["Fulfillment & Logistics", "Shipping & Logistics", "Delivery Costs"],
    "payment_gateway": ["Payment Gateway Fees", "PG Charges", "Payment Processing"],
    "packaging": ["Packaging", "Packaging Materials"],
    "perf_marketing": ["Performance Marketing", "Paid Acquisition", "Digital Ads"],
    "brand_marketing": ["Brand Marketing", "Brand & Content"],
    "salaries": ["Salaries & Wages", "Employee Costs", "Payroll"],
    "rent": ["Rent", "Rent & Utilities"],
    "tech": ["Tech & Software", "SaaS Tools & Hosting"],
    "g_and_a": ["General & Admin", "G&A"],
    "depreciation": ["Depreciation", "Depreciation Expense"],
    "amortization": ["Amortization", "Amortization of Intangibles"],
    "amort_rnd": ["Amortization of Capitalized R&D", "Capitalized R&D Amortization"],
    "interest": ["Interest Expense", "Finance Costs"],
    "taxes": ["Income Tax", "Tax Expense"],
    "ebit": ["EBIT", "Operating Income (EBIT)", "Operating Profit (EBIT)"],
    "net_income": ["Net Income", "Net Profit / (Loss)", "Profit After Tax"],
    # Unit economics
    "new_customers": ["New Customers", "New Customers Acquired"],
    "active_customers": ["Active Customers", "Paying Customers (EoM)"],
    "arpu": ["ARPU (monthly)", "Avg Revenue per User / month"],
    "gm_pct": ["Gross Margin %", "Blended Gross Margin %"],
    "churn_pct": ["Monthly Churn %", "Customer Churn Rate (monthly)"],
    "sm_spend": ["Total S&M Spend", "Sales & Marketing Spend"],
    # Pricing
    "price": ["Average Selling Price (per unit)", "Price per Unit", "ASP (per unit)"],
    "vc": ["Variable Cost per Unit", "Unit Variable Cost"],
    "fixed": ["Monthly Fixed Costs", "Fixed Costs (monthly)"],
    "vc_materials": ["Materials per Unit", "Raw Materials / unit"],
    "vc_packaging": ["Packaging per Unit", "Packaging / unit"],
    "vc_shipping": ["Shipping per Unit", "Last-mile Delivery / unit"],
    "fx_salaries": ["Fixed Salaries (monthly)", "Team Salaries (monthly)"],
    "fx_rent": ["Rent (monthly)", "Office Rent (monthly)"],
    "fx_software": ["Software & Tools (monthly)", "Tools & Subscriptions (monthly)"],
    "hires": ["Planned Hires - added monthly cost", "New Hires Payroll (monthly, planned)"],
    # Cash
    "opening_cash": ["Opening Cash Balance", "Opening Bank Balance"],
    "cash_in": ["Cash In", "Total Cash Inflows", "Collections & Inflows"],
    "cash_out": ["Cash Out", "Total Cash Outflows", "Payments & Outflows"],
    "closing_cash": ["Closing Cash Balance", "Closing Bank Balance"],
    # ARR bridge
    "arr_open": ["Opening ARR", "ARR (start of month)"],
    "arr_new": ["New ARR", "New Business ARR"],
    "arr_expansion": ["Expansion ARR", "Upsell ARR"],
    "arr_churned": ["Churned ARR", "Lost ARR"],
    "arr_close": ["Closing ARR", "ARR (end of month)"],
}

PCT_KEYS = {"gm_pct", "churn_pct"}


def month_label(y: int, m: int) -> str:
    return f"{calendar.month_abbr[m]}-{y % 100:02d}"


def r10(x: float) -> int:
    """Round currency to the nearest 10 so MIS numbers look realistic."""
    return int(round(x / 10.0)) * 10


def loguniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def num(v: float) -> int | float:
    """Number as it should appear in a tool call: int when whole, else 2 decimals."""
    return int(round(v)) if abs(v - round(v)) < 1e-9 else round(v, 2)


def fmt(key: str, v: float) -> str:
    return f"{v:.2f}%" if key in PCT_KEYS else format_cell(v)


# ---------------------------------------------------------------------------
# Company simulation
# ---------------------------------------------------------------------------

@dataclass
class Company:
    name: str
    currency: str
    months: list[str]
    data: dict[str, list[float]]
    pricing: dict[str, float]
    labels: dict[str, str]
    edges: set[str]
    source: str
    window: list[int]
    target: int
    plan_label: str
    scale: float
    ym: list[tuple[int, int]] = field(default_factory=list)   # (year, month) for each entry in months

    @property
    def period(self) -> str:
        return self.months[self.target]


def simulate_company(rng: random.Random, edge_probs: dict[str, float]) -> Company:
    edges = {e for e, p in edge_probs.items() if rng.random() < p}
    currency = rng.choice(["INR", "USD"])
    n = rng.randint(12, 36)
    y, m = rng.randint(2022, 2024), rng.randint(1, 12)
    months, ym = [], []
    for _ in range(n):
        months.append(month_label(y, m))
        ym.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1

    w = rng.randint(3, 12)
    end = rng.randint(w - 1, n - 1)
    window = list(range(end - w + 1, end + 1))
    target = end if rng.random() < 0.6 else rng.choice(window)

    scale = loguniform(rng, 1.5e6, 9e7) if currency == "INR" else loguniform(rng, 4e4, 2.5e6)
    growth, season, phase = rng.uniform(-0.02, 0.09), rng.uniform(0, 0.15), rng.uniform(0, 2 * math.pi)
    disc_r = rng.uniform(0.02, 0.12)
    ret_r = 0.0 if rng.random() < 0.1 else rng.uniform(0.005, 0.07)
    cogs_r = rng.uniform(1.02, 1.3) if "negative_gross_margin" in edges else rng.uniform(0.25, 0.72)
    ful_r, pg_r, pack_r = rng.uniform(0.03, 0.10), rng.uniform(0.015, 0.03), rng.uniform(0.005, 0.03)
    perf_r = rng.uniform(0.05, 0.30)
    brand = scale * rng.uniform(0.01, 0.06)
    hc = rng.randint(6, 60)
    per_head = scale * rng.uniform(0.25, 0.8) / hc
    hires = set(rng.sample(range(1, n), k=rng.randint(1, 3))) if "step_fixed_costs" in edges else set()
    rent, tech0, ga = scale * rng.uniform(0.02, 0.08), scale * rng.uniform(0.01, 0.05), scale * rng.uniform(0.02, 0.06)
    dep, amort = scale * rng.uniform(0.005, 0.03), scale * rng.uniform(0.002, 0.02)
    amort_rnd = scale * rng.uniform(0.01, 0.05) if "capitalized_rnd" in edges else 0.0
    interest = 0.0 if rng.random() < 0.3 else scale * rng.uniform(0.002, 0.02)
    arpu = round(loguniform(rng, 250, 6000) if currency == "INR" else loguniform(rng, 8, 400), rng.choice([0, 2]))
    base_churn = rng.uniform(0.8, 8.0)
    cash = r10(scale * rng.uniform(15, 50))
    burn_extra = scale * rng.uniform(0.02, 0.15)
    arr = r10(scale * 12 * rng.uniform(0.3, 0.9))

    d: dict[str, list[float]] = defaultdict(list)
    for t in range(n):
        if t in hires:
            hc += rng.randint(3, 12)
        mult = (1 + growth) ** t * (1 + season * math.sin(2 * math.pi * t / 12 + phase)) * (1 + rng.gauss(0, 0.03))
        gr = r10(scale * max(mult, 0.2))
        disc = r10(gr * disc_r * rng.uniform(0.85, 1.15))
        ret = r10(gr * ret_r * rng.uniform(0.8, 1.2))
        nr = F.net_revenue(gr, disc, ret)
        cogs = r10(nr * cogs_r * rng.uniform(0.97, 1.03))
        ful, pg, pack = r10(nr * ful_r), r10(nr * pg_r), r10(nr * pack_r)
        perf, br = r10(nr * perf_r * rng.uniform(0.9, 1.1)), r10(brand * rng.uniform(0.8, 1.2))
        sal = r10(hc * per_head)
        tech = r10(tech0 * (1 + 0.01 * t))
        dep_t, am_t, rnd_t = r10(dep), r10(amort), r10(amort_rnd)
        ebit = F.difference(nr, F.total([cogs, ful, pg, pack, perf, br, sal, r10(rent), tech, r10(ga), dep_t, am_t, rnd_t]))
        intr = r10(interest)
        tax = r10(max(0.0, ebit - intr) * 0.25)
        ni = F.difference(F.difference(ebit, intr), tax)
        billings = r10(gr * rng.uniform(0.95, 1.25))
        churn = round(base_churn * rng.uniform(0.85, 1.15), 2)
        if "seasonal_churn_spike" in edges and t == target:
            churn = round(rng.uniform(10.0, 18.0), 2)
        if "zero_churn" in edges:
            churn = 0.0
        active = max(50, round(nr / arpu)) if nr > 0 else 50
        cash_in = r10(gr * rng.uniform(0.82, 0.95))
        if "cash_positive" in edges:
            cash_out = r10(cash_in * rng.uniform(0.8, 0.97))
        else:
            cash_out = max(10, r10(cash_in - ebit + burn_extra))
        if cash + cash_in - cash_out < scale * 2:  # an equity raise lands this month
            cash_in = r10(cash_in + scale * rng.uniform(15, 40))
        opening = cash
        cash = F.difference(F.total([opening, cash_in]), cash_out)
        new_arr, exp_arr = r10(arr * rng.uniform(0.02, 0.08)), r10(arr * rng.uniform(0.0, 0.02))
        lost_arr = r10(arr * (rng.uniform(0.08, 0.14) if "arr_decline" in edges else rng.uniform(0.005, 0.03)))
        arr_close = F.difference(F.total([arr, new_arr, exp_arr]), lost_arr)
        row = {
            "gross_revenue": gr, "discounts": disc, "returns": ret, "billings": billings,
            "deferred_change": F.difference(billings, gr), "cogs": cogs, "fulfillment": ful, "payment_gateway": pg,
            "packaging": pack, "perf_marketing": perf, "brand_marketing": br, "salaries": sal, "rent": r10(rent),
            "tech": tech, "g_and_a": r10(ga), "depreciation": dep_t, "amortization": am_t, "amort_rnd": rnd_t,
            "interest": intr, "taxes": tax, "ebit": ebit, "net_income": ni, "headcount": hc,
            "new_customers": max(5, round(active * rng.uniform(0.04, 0.25))), "active_customers": active,
            "arpu": arpu, "gm_pct": round(F.gross_margin(nr, cogs) * 100, 2), "churn_pct": churn,
            "sm_spend": F.total([perf, br]), "net_revenue": nr,
            "opening_cash": opening, "cash_in": cash_in, "cash_out": cash_out, "closing_cash": cash,
            "arr_open": arr, "arr_new": new_arr, "arr_expansion": exp_arr, "arr_churned": lost_arr, "arr_close": arr_close,
        }
        arr = arr_close
        for k, v in row.items():
            d[k].append(v)

    # Pricing / break-even inputs (a plan, not monthly actuals)
    price = round(loguniform(rng, 150, 9000) if currency == "INR" else loguniform(rng, 5, 400), rng.choice([0, 2]))
    if "price_below_variable_cost" in edges:
        vc = round(price * rng.uniform(1.02, 1.2), 2)
    else:
        vc = round(price * rng.uniform(0.3, 0.85), 2)
    mat = round(vc * rng.uniform(0.5, 0.7), 2)
    pkg = round(vc * rng.uniform(0.05, 0.15), 2)
    ship = round(vc - mat - pkg, 2)
    fx_sal, fx_rent, fx_sw = r10(scale * rng.uniform(0.1, 0.4)), r10(scale * rng.uniform(0.02, 0.06)), r10(scale * rng.uniform(0.01, 0.04))
    pricing = {
        "price": price, "vc": round(mat + pkg + ship, 2), "vc_materials": mat, "vc_packaging": pkg, "vc_shipping": ship,
        "fixed": F.total([fx_sal, fx_rent, fx_sw]), "fx_salaries": fx_sal, "fx_rent": fx_rent, "fx_software": fx_sw,
        "hires": r10(scale * rng.uniform(0.03, 0.12)),
        "price_bull": round(price * rng.uniform(1.03, 1.15), 2), "price_bear": round(price * rng.uniform(0.85, 0.97), 2),
        "vc_bull": round(vc * rng.uniform(0.92, 0.99), 2), "vc_bear": round(vc * rng.uniform(1.01, 1.1), 2),
    }

    name = f"{rng.choice(NAME_A)} {rng.choice(NAME_B)}"
    slug = name.lower().replace(" ", "_")
    end_label = months[end].replace("-", "")
    source = rng.choice([f"{slug}_mis_{end_label}.xlsx", f"{slug}_monthly_mis.xlsx", f"{slug}_board_pack_{end_label}.pdf"])
    labels = {k: rng.choice(v) for k, v in LABELS.items()}
    plan_label = f"FY{(int(months[end][-2:]) + 1) % 100:02d} Plan"
    return Company(name, currency, months, dict(d), pricing, labels, edges, source, window, target, plan_label, scale, ym)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

@dataclass
class Table:
    key: str
    title: str
    sheet: str
    section: str
    headers: list[str]
    rows: list[list[str]]
    periods: list[str]
    units: str
    source: str
    page: int | None = None

    def render(self) -> str:
        meta = ChunkMetadata(source=self.source, sheet=self.sheet, section=self.section, periods=self.periods,
                             row_labels=[r[0] for r in self.rows], units=self.units, page=self.page)
        return render_chunk(self.title, meta, self.headers, self.rows)


def _page(c: Company, rng: random.Random) -> int | None:
    return rng.randint(2, 18) if c.source.endswith(".pdf") else None


def monthly_table(c: Company, rng: random.Random, key: str, title: str, sheet: str, section: str,
                  row_keys: list[str], units: str | None = None, offset: int = 0) -> Table:
    idx = [i - offset for i in c.window]
    periods = [c.months[i] for i in idx]
    rows = [[c.labels.get(k, k), *[fmt(k, c.data[k][i]) for i in idx]] for k in row_keys]
    return Table(key, title, sheet, section, ["Line Item", *periods], rows, periods, units or c.currency, c.source, _page(c, rng))


PL_SECTIONS: dict[str, tuple[str, list[str]]] = {
    "pl_revenue": ("Revenue", ["gross_revenue", "discounts", "returns"]),
    "pl_cogs": ("Cost of Revenue", ["cogs"]),
    "pl_variable": ("Variable Costs", ["fulfillment", "payment_gateway", "packaging"]),
    "pl_marketing": ("Marketing", ["perf_marketing", "brand_marketing"]),
    "pl_opex": ("Operating Expenses", ["salaries", "rent", "tech", "g_and_a"]),
    "pl_below": ("D&A, Interest & Tax", ["depreciation", "amortization", "interest", "taxes"]),
    "pl_profit": ("Profit Summary", ["ebit", "net_income"]),
}


def pl_table(c: Company, rng: random.Random, key: str, rows: list[str] | None = None) -> Table:
    section, default_rows = PL_SECTIONS[key]
    rows = list(rows or default_rows)
    if key == "pl_revenue" and "deferred_revenue" in c.edges and rows == default_rows:
        rows = ["billings", "deferred_change", *rows]
    if key == "pl_below" and "capitalized_rnd" in c.edges and "amort_rnd" not in rows:
        rows.insert(2, "amort_rnd")
    return monthly_table(c, rng, key, f"Monthly P&L - {section}", "P&L", section, rows)


def unit_econ_table(c: Company, rng: random.Random, include_sm: bool, include_gm: bool) -> Table:
    rows = ["new_customers", "active_customers", "arpu"]
    if include_gm:
        rows.append("gm_pct")
    rows.append("churn_pct")
    if include_sm:
        rows.append("sm_spend")
    return monthly_table(c, rng, "unit_econ", "Unit Economics - Customer Metrics", "Unit Economics", "Customer Metrics", rows,
                         units=f"{c.currency} unless stated")


def pricing_table(c: Company, rng: random.Random, row_keys: list[str], scenarios: bool = False) -> Table:
    p = c.pricing
    if scenarios:
        headers = ["Item", "Base", "Bull", "Bear"]
        rows = [[c.labels["price"], format_cell(p["price"]), format_cell(p["price_bull"]), format_cell(p["price_bear"])],
                [c.labels["vc"], format_cell(p["vc"]), format_cell(p["vc_bull"]), format_cell(p["vc_bear"])],
                [c.labels["fixed"], format_cell(p["fixed"]), format_cell(p["fixed"]), format_cell(p["fixed"])]]
    else:
        headers = ["Item", "Value"]
        rows = []
        for k in row_keys:
            label = c.labels[k]
            if k == "hires":
                nxt = c.months[c.window[-1]]
                label = f"{label} (from {nxt})"
            rows.append([label, format_cell(p[k])])
    return Table("pricing", "Pricing & Cost Structure - Unit Cost & Fixed Cost Plan", "Pricing", "Unit Cost & Fixed Cost Plan", headers, rows,
                 [c.plan_label], f"{c.currency} per unit unless stated", c.source, _page(c, rng))


def cash_flow_table(c: Company, rng: random.Random) -> Table:
    return monthly_table(c, rng, "cash_flow", "Cash Flow - Bank", "Cash", "Cash Flow",
                         ["opening_cash", "cash_in", "cash_out", "closing_cash"])


def arr_table(c: Company, rng: random.Random) -> Table:
    return monthly_table(c, rng, "arr", "SaaS Metrics - ARR Bridge", "SaaS Metrics", "ARR Bridge",
                         ["arr_open", "arr_new", "arr_expansion", "arr_churned", "arr_close"])


def fy_quarters(c: Company) -> list[tuple[str, list[int]]]:
    """Complete Indian-FY quarters (Q1 = Apr-Jun) that end on or before the window's last month."""
    out, i = [], 0
    while i + 2 < len(c.ym) and i + 2 <= c.window[-1]:
        y, m = c.ym[i]
        if m in (1, 4, 7, 10):
            fy = y + 1 if m >= 4 else y
            out.append((f"Q{((m - 4) % 12) // 3 + 1} FY{fy % 100:02d}", [i, i + 1, i + 2]))
            i += 3
        else:
            i += 1
    return out


def quarter_values(c: Company, months: list[int]) -> dict[str, float]:
    return {"q_revenue": F.total([c.data["net_revenue"][i] for i in months]),
            "q_sm": F.total([c.data["sm_spend"][i] for i in months]),
            "q_customers": c.data["active_customers"][months[-1]]}


QUARTER_ROWS = {"q_revenue": "Net Revenue", "q_sm": "S&M Spend", "q_customers": "Ending Customers"}


def quarterly_table(c: Company, rng: random.Random, quarters: list[tuple[str, list[int]]]) -> Table:
    vals = [quarter_values(c, ms) for _, ms in quarters]
    rows = [[label, *[format_cell(v[k]) for v in vals]] for k, label in QUARTER_ROWS.items()]
    labels = [q for q, _ in quarters]
    return Table("quarterly", "Quarterly Summary - Revenue & S&M", "Quarterly", "Quarterly KPIs",
                 ["Line Item", *labels], rows, labels, c.currency, c.source, _page(c, rng))


# --- distractors -----------------------------------------------------------

def distractor_tables(c: Company, rng: random.Random) -> dict[str, Table]:
    periods = [c.months[i] for i in c.window]
    src = c.source

    def monthly(key, title, sheet, section, rows, units):
        return Table(key, title, sheet, section, ["Line Item", *periods], rows, periods, units, src, _page(c, rng))

    hc = [c.data["headcount"][i] for i in c.window]
    split = [rng.uniform(0.3, 0.45), rng.uniform(0.15, 0.25), rng.uniform(0.2, 0.3)]
    headcount = monthly("d_headcount", "Headcount by Department", "Headcount", "FTEs", [
        ["Engineering", *[format_cell(round(h * split[0])) for h in hc]],
        ["Sales", *[format_cell(round(h * split[1])) for h in hc]],
        ["Operations", *[format_cell(round(h * split[2])) for h in hc]],
        ["Total Headcount", *[format_cell(h) for h in hc]],
    ], "FTEs")
    sessions = [r10(c.data["active_customers"][i] * rng.uniform(8, 30)) for i in c.window]
    funnel = monthly("d_funnel", "Website & Funnel KPIs", "KPIs", "Acquisition Funnel", [
        ["Sessions", *[format_cell(s) for s in sessions]],
        ["Conversion Rate %", *[f"{rng.uniform(0.8, 4.5):.2f}%" for _ in periods]],
        ["Average Order Value", *[format_cell(r10(c.data["arpu"][i] * rng.uniform(0.8, 1.6))) for i in c.window]],
    ], f"{c.currency} / count")
    cash = cash_flow_table(c, rng)
    ret0 = rng.uniform(0.55, 0.85)
    cohort_rows = []
    for j, p in enumerate(periods[:3]):
        cohort_rows.append([f"Cohort {p}", "100.00%", *[f"{100 * ret0 ** (k + rng.uniform(0, 0.3)):.2f}%" for k in range(1, 4)]])
    cohort = Table("d_cohort", "Cohort Retention", "Cohorts", "Logo Retention by Signup Month",
                   ["Cohort", "M0", "M1", "M2", "M3"], cohort_rows, periods[:3], "% of cohort", src, _page(c, rng))
    inventory = monthly("d_inventory", "Inventory & Supply", "Ops", "Inventory", [
        ["Inventory Days", *[format_cell(rng.randint(25, 90)) for _ in periods]],
        ["Active SKUs", *[format_cell(rng.randint(20, 400)) for _ in periods]],
        ["Stock-out Rate %", *[f"{rng.uniform(0.5, 9):.2f}%" for _ in periods]],
    ], "days / count")
    wc = monthly("d_working_capital", "Working Capital", "Balance Sheet", "Receivables & Payables", [
        ["Accounts Receivable", *[format_cell(r10(c.scale * rng.uniform(0.2, 0.6))) for _ in periods]],
        ["Accounts Payable", *[format_cell(r10(c.scale * rng.uniform(0.2, 0.7))) for _ in periods]],
        ["DSO (days)", *[format_cell(rng.randint(15, 75)) for _ in periods]],
    ], c.currency)
    bench = Table("d_benchmarks", "Industry Benchmarks", "Benchmarks", "Peer Medians",
                  ["Metric", "Median", "Top Quartile"], [
                      ["LTV:CAC", "3.0x", "5.0x"], ["CAC Payback (months)", "14", "8"],
                      ["Gross Margin %", f"{rng.choice([45, 50, 55, 60])}.00%", f"{rng.choice([65, 70, 72])}.00%"],
                      ["Burn Multiple", "1.8x", "1.1x"]],
                  ["2025 survey"], "ratios / %", "peer_benchmarks_2025.pdf", rng.randint(3, 12))
    return {t.key: t for t in [headcount, funnel, cash, cohort, inventory, wc, bench]}


# ---------------------------------------------------------------------------
# Task plans
# ---------------------------------------------------------------------------

@dataclass
class Need:
    name: str        # human name used in reasoning ("Gross Revenue")
    table: str       # table key it lives in
    label: str       # row label in that table
    column: str      # column header (period / Value / Base ...)
    value: float
    shown: str       # exactly as rendered in the table
    key: str = ""    # data key (for missing-data mutation)


@dataclass
class Plan:
    task: str
    variant: str
    title: str
    period: str
    question: str
    tables: list[Table]
    needs: list[Need]
    steps: list[dict[str, Any]]
    formulas: list[str]
    notes: list[str]
    report_rows: list[tuple[str, str, str]]          # (metric, step id, formula text)
    truth: dict[str, float | None]                   # None = the calculator must report an error
    takeaway: Callable[[dict[str, str], dict[str, float], dict[str, str]], list[str]]
    missing_ok: list[int] = field(default_factory=list)
    uses_pl: bool = True


def mneed(c: Company, key: str, table: str, name: str | None = None, t: int | None = None) -> Need:
    t = c.target if t is None else t
    v = c.data[key][t]
    return Need(name or c.labels[key], table, c.labels[key], c.months[t], v, fmt(key, v), key)


def pneed(c: Company, key: str, name: str, column: str = "Value", value_key: str | None = None, label: str | None = None) -> Need:
    v = c.pricing[value_key or key]
    return Need(name, "pricing", label or c.labels[key], column, v, format_cell(v), key)


def step(id_: str, formula: str, **inputs: Any) -> dict[str, Any]:
    return {"id": id_, "formula": formula, "inputs": inputs}


def lit(n: Need) -> int | float:
    return num(n.value)


def pct(n: Need) -> float:
    return round(n.value, 2)


# --- 1. Net revenue -------------------------------------------------------

def plan_net_revenue(c: Company, rng: random.Random) -> Plan:
    P = c.period
    gr, di, re_ = (mneed(c, k, "pl_revenue") for k in ("gross_revenue", "discounts", "returns"))
    steps = [
        step("net_revenue", "net_revenue", gross_revenue=lit(gr), discounts=lit(di), returns=lit(re_)),
        step("revenue_leakage", "total", values=[lit(di), lit(re_)]),
        step("leakage_pct", "percent_of", part="@revenue_leakage", whole=lit(gr)),
    ]
    nr = F.net_revenue(gr.value, di.value, re_.value)
    leak = F.total([di.value, re_.value])
    truth = {"net_revenue": nr, "revenue_leakage": leak, "leakage_pct": F.percent_of(leak, gr.value)}
    notes = []
    if "deferred_revenue" in c.edges:
        notes.append(f'"{c.labels["billings"]}" and "{c.labels["deferred_change"]}" describe invoicing and deferral, '
                     "not recognized revenue, so they are excluded.")

    def take(disp, vals, errs):
        p = vals["leakage_pct"]
        if p > 0.12:
            return [f"Discounts and returns absorb {disp['leakage_pct']} of gross revenue, above a typical 5-12% band. "
                    "Review promo depth and return reasons before scaling volume."]
        if p < 0.05:
            return [f"Leakage is low at {disp['leakage_pct']} of gross revenue; pricing discipline looks good."]
        return [f"Leakage of {disp['leakage_pct']} of gross revenue is within a normal 5-12% band."]

    q = rng.choice([
        f"What was our net revenue for {P}?",
        f"Calculate net revenue for {P} from the MIS.",
        f"How much revenue did we actually keep in {P} after discounts and returns?",
        f"Give me {P} net revenue and how much leaked to discounts and returns.",
    ])
    return Plan("net_revenue", "deferred" if notes else "basic", "Net Revenue", P, q, [pl_table(c, rng, "pl_revenue")],
                [gr, di, re_], steps,
                ["Net Revenue = Gross Revenue - (Discounts + Returns)", "Leakage % = (Discounts + Returns) / Gross Revenue"],
                notes,
                [("Net Revenue", "net_revenue", "Gross Revenue - (Discounts + Returns)"),
                 ("Revenue Leakage (Discounts + Returns)", "revenue_leakage", "Discounts + Returns"),
                 ("Leakage % of Gross Revenue", "leakage_pct", "Leakage / Gross Revenue")],
                truth, take, missing_ok=[0, 1, 2])


# --- 2. Contribution margin ------------------------------------------------

def plan_contribution_margin(c: Company, rng: random.Random) -> Plan:
    P = c.period
    with_cm3 = rng.random() < 0.35
    gr, di, re_ = (mneed(c, k, "pl_revenue") for k in ("gross_revenue", "discounts", "returns"))
    cogs = mneed(c, "cogs", "pl_cogs")
    ful, pg, pack = (mneed(c, k, "pl_variable") for k in ("fulfillment", "payment_gateway", "packaging"))
    needs = [gr, di, re_, cogs, ful, pg, pack]
    steps = [
        step("net_revenue", "net_revenue", gross_revenue=lit(gr), discounts=lit(di), returns=lit(re_)),
        step("cm1", "cm1", net_revenue="@net_revenue", cogs=lit(cogs)),
        step("variable_ops_costs", "total", values=[lit(ful), lit(pg), lit(pack)]),
        step("cm2", "cm2", cm1="@cm1", variable_ops_costs="@variable_ops_costs"),
    ]
    nr = F.net_revenue(gr.value, di.value, re_.value)
    c1 = F.cm1(nr, cogs.value)
    vo = F.total([ful.value, pg.value, pack.value])
    c2 = F.cm2(c1, vo)
    truth = {"net_revenue": nr, "cm1": c1, "variable_ops_costs": vo, "cm2": c2}
    formulas = ["Net Revenue = Gross Revenue - (Discounts + Returns)", "CM1 = Net Revenue - COGS",
                "CM2 = CM1 - Variable Ops Costs (fulfillment + payment gateway + packaging)"]
    rows = [("Net Revenue", "net_revenue", "Gross Revenue - (Discounts + Returns)"),
            ("CM1", "cm1", "Net Revenue - COGS"),
            ("CM1 %", "cm1_pct", "CM1 / Net Revenue"),
            ("Variable Ops Costs", "variable_ops_costs", "Fulfillment + Payment Gateway + Packaging"),
            ("CM2", "cm2", "CM1 - Variable Ops Costs"),
            ("CM2 %", "cm2_pct", "CM2 / Net Revenue")]
    tables = [pl_table(c, rng, "pl_revenue"), pl_table(c, rng, "pl_cogs"), pl_table(c, rng, "pl_variable")]
    notes = []
    if with_cm3:
        perf = mneed(c, "perf_marketing", "pl_marketing")
        needs.append(perf)
        steps.append(step("cm3", "cm3", cm2="@cm2", variable_marketing=lit(perf)))
        truth["cm3"] = F.cm3(c2, perf.value)
        formulas.append("CM3 = CM2 - Variable (performance) Marketing")
        rows += [("CM3", "cm3", "CM2 - Performance Marketing"), ("CM3 %", "cm3_pct", "CM3 / Net Revenue")]
        tables.append(pl_table(c, rng, "pl_marketing"))
        notes.append(f'"{c.labels["brand_marketing"]}" is a fixed brand cost, not variable, so it stays out of CM3.')
    for cm in ["cm1", "cm2"] + (["cm3"] if with_cm3 else []):
        steps.append(step(f"{cm}_pct", "margin_pct", value=f"@{cm}", net_revenue="@net_revenue"))
        truth[f"{cm}_pct"] = F.margin_pct(truth[cm], nr)
    if "deferred_revenue" in c.edges:
        notes.append(f'"{c.labels["billings"]}" is invoicing, not recognized revenue; I start from {gr.label}.')

    def take(disp, vals, errs):
        out = []
        if vals["cm1"] < 0:
            out.append(f"CM1 is negative ({disp['cm1']}): COGS exceeds net revenue, so every sale loses money before "
                       "any other cost. Fix pricing or sourcing before spending on growth.")
        if vals["cm2"] < 0:
            out.append(f"CM2 is {disp['cm2']}; orders lose money after fulfillment and payment costs.")
        elif vals["cm2_pct"] < 0.15:
            out.append(f"CM2 margin of {disp['cm2_pct']} is thin; little room to fund acquisition.")
        else:
            out.append(f"CM2 margin of {disp['cm2_pct']} leaves healthy room to fund acquisition and fixed costs.")
        if with_cm3:
            out.append(f"After performance marketing, CM3 is {disp['cm3']} ({disp['cm3_pct']} of net revenue).")
        return out

    what = "CM1, CM2 and CM3" if with_cm3 else "CM1 and CM2"
    q = rng.choice([
        f"What are our {what} for {P}, in absolute terms and as % of net revenue?",
        f"Build the contribution margin waterfall ({what}) for {P}.",
        f"Compute {what} for {P}. Are we making money per order?",
        f"For {P}, walk me from gross revenue to {what.split(' and ')[-1]}.",
    ])
    return Plan("contribution_margin", "cm3" if with_cm3 else "cm2", "Contribution Margin", P, q, tables, needs, steps,
                formulas, notes, rows, truth, take, missing_ok=list(range(len(needs))))


# --- 3. EBITDA -------------------------------------------------------------

ADD_BACKS = ["ESOP one-time acceleration (non-cash)", "Fundraise legal & advisory fees", "Office relocation cost",
             "Severance from restructuring"]
DEDUCTIONS = ["Gain on sale of old equipment", "One-time insurance claim income", "Write-back of old payables"]
RECURRING = ["Monthly ESOP expense", "Annual audit fee accrual", "Recurring cloud credits"]


def plan_ebitda(c: Company, rng: random.Random) -> Plan:
    P, t = c.period, c.target
    variant = rng.choice(["from_ebit", "from_net_income"])
    adjusted = rng.random() < 0.3
    with_margin = rng.random() < 0.35
    rnd = "capitalized_rnd" in c.edges
    dep = mneed(c, "depreciation", "pl_below")
    am = mneed(c, "amortization", "pl_below")
    needs: list[Need] = []
    steps: list[dict[str, Any]] = []
    truth: dict[str, float | None] = {}
    notes: list[str] = []
    formulas: list[str] = []
    rows: list[tuple[str, str, str]] = []
    tables = [pl_table(c, rng, "pl_below")]

    if variant == "from_ebit":
        ebit_n = mneed(c, "ebit", "pl_profit")
        needs.append(ebit_n)
        ebit_ref: Any = lit(ebit_n)
        tables.append(pl_table(c, rng, "pl_profit"))
        notes.append(f'"{c.labels["interest"]}" and "{c.labels["taxes"]}" are not needed: EBIT is already before interest and taxes.')
    else:
        ni, it, tx = mneed(c, "net_income", "pl_profit"), mneed(c, "interest", "pl_below"), mneed(c, "taxes", "pl_below")
        needs += [ni, it, tx]
        steps.append(step("ebit", "ebit_from_net_income", net_income=lit(ni), interest=lit(it), taxes=lit(tx)))
        truth["ebit"] = F.ebit_from_net_income(ni.value, it.value, tx.value)
        ebit_ref = "@ebit"
        tables.append(pl_table(c, rng, "pl_profit", rows=["net_income"]))
        formulas.append("EBIT = Net Income + Interest + Taxes")
        notes.append("No EBIT row is reported, so I rebuild EBIT from net income by adding back interest and taxes.")
        rows.append(("EBIT", "ebit", "Net Income + Interest + Taxes"))

    needs += [dep, am]
    if rnd:
        rd = mneed(c, "amort_rnd", "pl_below")
        needs.append(rd)
        steps.append(step("amortization_total", "total", values=[lit(am), lit(rd)]))
        truth["amortization_total"] = F.total([am.value, rd.value])
        am_ref: Any = "@amortization_total"
        notes.append(f'"{rd.label}" is amortization of capitalized development cost, so it is added back with the other amortization.')
        rows.append(("Total Amortization", "amortization_total", "Amortization + Capitalized R&D Amortization"))
    else:
        am_ref = lit(am)
    steps.append(step("ebitda", "ebitda", ebit=ebit_ref, depreciation=lit(dep), amortization=am_ref))
    true_amort = am.value + (c.data["amort_rnd"][t] if rnd else 0)
    truth["ebitda"] = F.ebitda(c.data["ebit"][t], dep.value, true_amort)
    formulas.append("EBITDA = EBIT + Depreciation + Amortization")
    rows.append(("EBITDA", "ebitda", "EBIT + Depreciation + Amortization"))

    if adjusted:
        a_amt, b_amt = r10(c.scale * rng.uniform(0.01, 0.05)), r10(c.scale * rng.uniform(0.005, 0.03))
        other_p = c.months[c.window[0]] if c.window[0] != t else c.months[c.window[-1]]
        add_names = rng.sample(ADD_BACKS, 2)
        add_item = [add_names[0], P, format_cell(a_amt), "One-off expense (add back)"]
        gain_item = [rng.choice(DEDUCTIONS), P, format_cell(b_amt), "One-off gain (deduct)"]
        items = [add_item, gain_item,
                 [rng.choice(RECURRING), P, format_cell(r10(c.scale * rng.uniform(0.005, 0.02))), "Recurring - not an adjustment"]]
        if other_p != P:
            items.append([add_names[1], other_p, format_cell(r10(c.scale * rng.uniform(0.01, 0.04))), "One-off expense (add back)"])
        rng.shuffle(items)
        tables.append(Table("one_offs", "One-off Items (Management Notes)", "Notes", "EBITDA Adjustments",
                            ["Item", "Period", "Amount", "Nature"], items, sorted({r[1] for r in items}), c.currency, c.source, _page(c, rng)))
        needs += [Need(f"Add-back: {add_item[0]}", "one_offs", add_item[0], P, a_amt, add_item[2]),
                  Need(f"One-off gain: {gain_item[0]}", "one_offs", gain_item[0], P, b_amt, gain_item[2])]
        steps.append(step("adjusted_ebitda", "adjusted_ebitda", ebitda="@ebitda", adjustments=[num(a_amt), -num(b_amt)]))
        truth["adjusted_ebitda"] = F.adjusted_ebitda(truth["ebitda"], [a_amt, -b_amt])  # type: ignore[arg-type]
        formulas.append("Adjusted EBITDA = EBITDA + one-off expenses - one-off gains (documented items only)")
        notes.append(f"Only {P} one-off items are adjustments; recurring items and other periods are excluded.")
        rows.append(("Adjusted EBITDA", "adjusted_ebitda", "EBITDA + add-backs - one-off gains"))

    if with_margin:
        gr, di, re_ = (mneed(c, k, "pl_revenue") for k in ("gross_revenue", "discounts", "returns"))
        needs += [gr, di, re_]
        tables.append(pl_table(c, rng, "pl_revenue"))
        steps.append(step("net_revenue", "net_revenue", gross_revenue=lit(gr), discounts=lit(di), returns=lit(re_)))
        steps.append(step("ebitda_margin", "margin_pct", value="@ebitda", net_revenue="@net_revenue"))
        nr = F.net_revenue(gr.value, di.value, re_.value)
        truth["net_revenue"], truth["ebitda_margin"] = nr, F.margin_pct(truth["ebitda"], nr)  # type: ignore[arg-type]
        formulas.append("EBITDA Margin = EBITDA / Net Revenue")
        rows += [("Net Revenue", "net_revenue", "Gross Revenue - (Discounts + Returns)"),
                 ("EBITDA Margin", "ebitda_margin", "EBITDA / Net Revenue")]

    def take(disp, vals, errs):
        out = []
        if vals["ebitda"] < 0:
            out.append(f"EBITDA is negative at {disp['ebitda']}: the core business is burning cash before D&A. "
                       "Track runway closely.")
        else:
            out.append(f"EBITDA is positive at {disp['ebitda']}.")
        if "ebitda_margin" in vals:
            out.append(f"EBITDA margin is {disp['ebitda_margin']} of net revenue.")
        if "adjusted_ebitda" in vals:
            out.append(f"Adjusted EBITDA of {disp['adjusted_ebitda']} removes documented one-offs only; "
                       "present both figures to investors.")
        if rnd:
            out.append("Capitalized R&D amortization is added back, so cash spent on R&D does not show in EBITDA; "
                       "check cash burn separately.")
        return out

    want = "adjusted EBITDA" if adjusted else "EBITDA"
    q = rng.choice([
        f"What was {want} for {P}?",
        f"Reconcile {want} for {P} from the P&L.",
        f"Calculate {P} {want}{' and EBITDA margin' if with_margin else ''}.",
        f"I need the {P} {want} number for the board deck. Show the bridge.",
    ])
    return Plan("ebitda", variant + ("+adjusted" if adjusted else "") + ("+margin" if with_margin else ""),
                "Adjusted EBITDA" if adjusted else "EBITDA", P, q, tables, needs, steps, formulas, notes, rows, truth, take,
                missing_ok=[i for i, n in enumerate(needs) if n.table in ("pl_below", "pl_profit")])


# --- 4. Break-even ---------------------------------------------------------

def plan_break_even(c: Company, rng: random.Random) -> Plan:
    p = c.pricing
    P = c.plan_label
    impossible = "price_below_variable_cost" in c.edges
    if impossible:
        variant = "simple"
    elif "step_fixed_costs" in c.edges and rng.random() < 0.6:
        variant = "step_fixed"
    else:
        variant = rng.choice(["simple", "components", "scenarios"])
    steps: list[dict[str, Any]] = []
    truth: dict[str, float | None] = {}
    notes: list[str] = []
    rows: list[tuple[str, str, str]] = []
    formulas = ["Break-even units = Fixed Costs / (Price - Variable Cost per unit), rounded up",
                "Break-even revenue = Break-even units x Price"]

    def be(prefix: str, fixed: Any, price: Any, vc: Any, f_v: float, p_v: float, v_v: float, label: str):
        steps.append(step(f"{prefix}units", "break_even_units", fixed_costs=fixed, price=price, variable_cost_per_unit=vc))
        steps.append(step(f"{prefix}revenue", "break_even_revenue", fixed_costs=fixed, price=price, variable_cost_per_unit=vc))
        try:
            truth[f"{prefix}units"] = F.break_even_units(f_v, p_v, v_v)
            truth[f"{prefix}revenue"] = F.break_even_revenue(f_v, p_v, v_v)
        except F.FinanceMathError:
            truth[f"{prefix}units"] = truth[f"{prefix}revenue"] = None
        rows.append((f"Break-even Units{label}", f"{prefix}units", "Fixed Costs / (Price - Variable Cost)"))
        rows.append((f"Break-even Revenue{label}", f"{prefix}revenue", "Break-even Units x Price"))

    if variant == "scenarios":
        tables = [pricing_table(c, rng, [], scenarios=True)]
        needs = [pneed(c, "fixed", "Monthly Fixed Costs", "Base")]
        for sc in ["Base", "Bull", "Bear"]:
            sfx = "" if sc == "Base" else f"_{sc.lower()}"
            pn = pneed(c, "price", f"Price ({sc})", sc, f"price{sfx}")
            vn = pneed(c, "vc", f"Variable Cost per Unit ({sc})", sc, f"vc{sfx}")
            needs += [pn, vn]
            be(f"{sc.lower()}_", lit(needs[0]), lit(pn), lit(vn), needs[0].value, pn.value, vn.value, f" ({sc})")
        notes.append("Fixed costs are the same in all three scenarios; only price and unit cost move.")
        question = rng.choice([f"Run break-even for the base, bull and bear cases in the {P}.",
                               f"How many units do we need to sell monthly to break even under each scenario ({P})?"])
        missing_ok: list[int] = []
    else:
        price = pneed(c, "price", "Price per Unit")
        if variant == "components":
            parts = [pneed(c, k, c.labels[k]) for k in ("vc_materials", "vc_packaging", "vc_shipping")]
            fparts = [pneed(c, k, c.labels[k]) for k in ("fx_salaries", "fx_rent", "fx_software")]
            tables = [pricing_table(c, rng, ["price", "vc_materials", "vc_packaging", "vc_shipping",
                                             "fx_salaries", "fx_rent", "fx_software"])]
            needs = [price, *parts, *fparts]
            steps.append(step("variable_cost_per_unit", "total", values=[lit(n) for n in parts]))
            steps.append(step("fixed_costs", "total", values=[lit(n) for n in fparts]))
            truth["variable_cost_per_unit"] = F.total([n.value for n in parts])
            truth["fixed_costs"] = F.total([n.value for n in fparts])
            rows += [("Variable Cost per Unit", "variable_cost_per_unit", "Materials + Packaging + Shipping"),
                     ("Monthly Fixed Costs", "fixed_costs", "Salaries + Rent + Software")]
            notes.append("Unit cost and fixed costs are given as components, so I total them first.")
            be("break_even_", "@fixed_costs", lit(price), "@variable_cost_per_unit",
               truth["fixed_costs"], price.value, truth["variable_cost_per_unit"], "")  # type: ignore[arg-type]
            missing_ok = []
        elif variant == "step_fixed":
            vc, fixed = pneed(c, "vc", "Variable Cost per Unit"), pneed(c, "fixed", "Current Monthly Fixed Costs")
            hires = Need("Planned hires (added fixed cost)", "pricing", "", "Value", p["hires"], format_cell(p["hires"]), "hires")
            tables = [pricing_table(c, rng, ["price", "vc", "fixed", "hires"])]
            hires.label = tables[0].rows[3][0]
            needs = [price, vc, fixed, hires]
            steps.append(step("fixed_costs_after_hiring", "total", values=[lit(fixed), lit(hires)]))
            truth["fixed_costs_after_hiring"] = F.total([fixed.value, hires.value])
            rows.append(("Fixed Costs after Hiring", "fixed_costs_after_hiring", "Current Fixed + Planned Hires"))
            be("current_", lit(fixed), lit(price), lit(vc), fixed.value, price.value, vc.value, " (current)")
            be("after_hiring_", "@fixed_costs_after_hiring", lit(price), lit(vc), truth["fixed_costs_after_hiring"],  # type: ignore[arg-type]
               price.value, vc.value, " (after hiring)")
            notes.append("Fixed costs step up when the planned hires join, so break-even must be shown before and after.")
            missing_ok = [0, 1, 2]
        else:
            vc, fixed = pneed(c, "vc", "Variable Cost per Unit"), pneed(c, "fixed", "Monthly Fixed Costs")
            tables = [pricing_table(c, rng, ["price", "vc", "fixed"])]
            needs = [price, vc, fixed]
            be("break_even_", lit(fixed), lit(price), lit(vc), fixed.value, price.value, vc.value, "")
            missing_ok = [0, 1, 2]
        if impossible:
            notes.append("Price does not look higher than variable cost per unit; the calculator will confirm whether "
                         "break-even is possible at all.")
        question = rng.choice([
            f"What is our monthly break-even volume and revenue ({P})?",
            f"How many units must we sell each month to break even? Use the {P}.",
            f"Calculate break-even units and revenue from the pricing sheet.",
        ] + ([f"If we make the planned hires, how does our break-even change?"] if variant == "step_fixed" else []))

    def take(disp, vals, errs):
        if errs:
            return ["Break-even is impossible at current pricing: each unit sells below its variable cost, so more "
                    "volume only deepens losses. Raise price or cut unit cost before anything else."]
        if variant == "scenarios":
            return [f"Base case needs {disp['base_units']} a month; the bear case needs {disp['bear_units']} "
                    f"and the bull case {disp['bull_units']}. Plan capacity and cash for the bear case."]
        if variant == "step_fixed":
            return [f"Hiring lifts the monthly break-even from {disp['current_units']} to {disp['after_hiring_units']}. "
                    "Time the hires to when volume is on track to cover them."]
        return [f"The business needs {disp['break_even_units']} a month ({disp['break_even_revenue']} of revenue) "
                "to cover fixed costs."]

    return Plan("break_even", variant + ("+impossible" if impossible else ""), "Break-even Analysis", P, question,
                tables, needs, steps, formulas, notes, rows, truth, take, missing_ok=missing_ok, uses_pl=False)


# --- 5/6. LTV:CAC and payback -----------------------------------------------

def _acquisition_inputs(c: Company, rng: random.Random, needs: list[Need], steps: list[dict[str, Any]],
                        truth: dict[str, float | None], tables: list[Table], rows: list, formulas: list[str],
                        notes: list[str]) -> tuple[Any, bool, Need]:
    """Shared CAC + gross-margin inputs. Returns (gross-margin input value/ref, gm_is_pct, arpu need)."""
    sm_given = rng.random() < 0.6
    gm_given = rng.random() < 0.65
    nc = mneed(c, "new_customers", "unit_econ")
    arpu = mneed(c, "arpu", "unit_econ", "ARPU (monthly)")
    tables.append(unit_econ_table(c, rng, include_sm=sm_given, include_gm=gm_given))
    if sm_given:
        sm = mneed(c, "sm_spend", "unit_econ", "S&M Spend")
        needs += [sm, nc]
        sm_ref: Any = lit(sm)
        sm_val = sm.value
    else:
        perf, brand = mneed(c, "perf_marketing", "pl_marketing"), mneed(c, "brand_marketing", "pl_marketing")
        needs += [perf, brand, nc]
        tables.append(pl_table(c, rng, "pl_marketing"))
        steps.append(step("sm_spend", "total", values=[lit(perf), lit(brand)]))
        sm_val = truth["sm_spend"] = F.total([perf.value, brand.value])
        sm_ref = "@sm_spend"
        rows.append(("S&M Spend", "sm_spend", "Performance + Brand Marketing"))
        notes.append("There is no single S&M total, so S&M spend = performance marketing + brand marketing.")
    steps.append(step("cac", "cac", sm_spend=sm_ref, new_customers=lit(nc)))
    truth["cac"] = F.cac(sm_val, nc.value)
    formulas.append("CAC = Total S&M Spend / New Customers")
    rows.append(("CAC", "cac", "S&M Spend / New Customers"))
    needs.append(arpu)
    if gm_given:
        gm = mneed(c, "gm_pct", "unit_econ", "Gross Margin %")
        needs.append(gm)
        return pct(gm), True, arpu
    gr, di, re_ = (mneed(c, k, "pl_revenue") for k in ("gross_revenue", "discounts", "returns"))
    cogs = mneed(c, "cogs", "pl_cogs")
    needs += [gr, di, re_, cogs]
    tables += [pl_table(c, rng, "pl_revenue"), pl_table(c, rng, "pl_cogs")]
    steps.append(step("net_revenue", "net_revenue", gross_revenue=lit(gr), discounts=lit(di), returns=lit(re_)))
    steps.append(step("gross_margin", "gross_margin", net_revenue="@net_revenue", cogs=lit(cogs)))
    nr = F.net_revenue(gr.value, di.value, re_.value)
    truth["net_revenue"], truth["gross_margin"] = nr, F.gross_margin(nr, cogs.value)
    formulas += ["Net Revenue = Gross Revenue - (Discounts + Returns)", "Gross Margin = (Net Revenue - COGS) / Net Revenue"]
    rows += [("Net Revenue", "net_revenue", "Gross Revenue - (Discounts + Returns)"),
             ("Gross Margin", "gross_margin", "(Net Revenue - COGS) / Net Revenue")]
    notes.append("Unit economics has no gross margin row, so I derive it from the P&L for the same month.")
    return "@gross_margin", False, arpu


def plan_ltv_cac(c: Company, rng: random.Random) -> Plan:
    P = c.period
    needs: list[Need] = []
    steps: list[dict[str, Any]] = []
    truth: dict[str, float | None] = {}
    tables: list[Table] = []
    rows: list[tuple[str, str, str]] = []
    formulas: list[str] = []
    notes: list[str] = []
    gm_in, gm_pct, arpu = _acquisition_inputs(c, rng, needs, steps, truth, tables, rows, formulas, notes)
    churn = mneed(c, "churn_pct", "unit_econ", "Monthly Churn %")
    needs.append(churn)
    ltv_inputs: dict[str, Any] = {"arpu": lit(arpu), ("gross_margin_pct" if gm_pct else "gross_margin"): gm_in,
                                  "churn_rate_pct": pct(churn)}
    steps.append(step("ltv", "ltv", **ltv_inputs))
    steps.append(step("ltv_cac_ratio", "ltv_cac_ratio", ltv="@ltv", cac="@cac"))
    gm_val = c.data["gm_pct"][c.target] / 100 if gm_pct else truth["gross_margin"]
    try:
        truth["ltv"] = F.ltv(arpu.value, gm_val, churn.value / 100)  # type: ignore[arg-type]
        truth["ltv_cac_ratio"] = F.ltv_cac_ratio(truth["ltv"], truth["cac"])  # type: ignore[arg-type]
    except F.FinanceMathError:
        truth["ltv"] = truth["ltv_cac_ratio"] = None
    formulas += ["LTV = (ARPU x Gross Margin) / Monthly Churn", "LTV:CAC = LTV / CAC"]
    rows += [("LTV", "ltv", "(ARPU x Gross Margin) / Churn"), ("LTV:CAC", "ltv_cac_ratio", "LTV / CAC")]
    if "seasonal_churn_spike" in c.edges:
        notes.append(f"Churn in {P} ({churn.shown}) is far above the other months (a seasonal spike). The question "
                     f"asks for {P}, so I use it as reported and flag it.")
    if "zero_churn" in c.edges:
        notes.append(f"Churn for {P} is {churn.shown}. LTV divides by churn, so I let the calculator flag it rather than "
                     "assume a churn rate.")
    notes.append(f'"{c.labels["active_customers"]}" is not an input to LTV or CAC.')

    def take(disp, vals, errs):
        if "ltv" in errs:
            return ["LTV is undefined because observed churn is zero; with no churn the formula has no finite answer. "
                    "Use a cohort-based churn estimate (or an explicit, approved floor) before quoting LTV:CAC."]
        r = vals["ltv_cac_ratio"]
        out = []
        if r < 0:
            out.append(f"LTV:CAC is {disp['ltv_cac_ratio']}: negative gross margin makes every customer value-destroying.")
        elif r < 1:
            out.append(f"LTV:CAC of {disp['ltv_cac_ratio']} is below 1x; acquisition destroys value. Cut paid spend or fix margins.")
        elif r < 3:
            out.append(f"LTV:CAC of {disp['ltv_cac_ratio']} is below the 3x benchmark; improve retention or lower CAC before scaling.")
        else:
            out.append(f"LTV:CAC of {disp['ltv_cac_ratio']} clears the 3x benchmark; there is room to scale acquisition.")
        if "seasonal_churn_spike" in c.edges:
            out.append(f"{P} churn is a seasonal spike, so this LTV is a conservative low point.")
        return out

    q = rng.choice([f"What is our LTV:CAC for {P}?", f"Calculate CAC, LTV and LTV:CAC for {P}.",
                    f"Are our unit economics healthy? Give me LTV to CAC for {P}.",
                    f"Investor wants {P} LTV/CAC. Compute it from the data."])
    return Plan("ltv_cac", "ltv_cac", "LTV:CAC", P, q, tables, needs, steps, formulas, notes, rows, truth, take,
                missing_ok=[i for i, n in enumerate(needs) if n.table == "unit_econ"])


def plan_payback(c: Company, rng: random.Random) -> Plan:
    P = c.period
    needs: list[Need] = []
    steps: list[dict[str, Any]] = []
    truth: dict[str, float | None] = {}
    tables: list[Table] = []
    rows: list[tuple[str, str, str]] = []
    formulas: list[str] = []
    notes: list[str] = []
    gm_in, gm_pct, arpu = _acquisition_inputs(c, rng, needs, steps, truth, tables, rows, formulas, notes)
    pb_inputs: dict[str, Any] = {"cac": "@cac", "arpu": lit(arpu), ("gross_margin_pct" if gm_pct else "gross_margin"): gm_in}
    steps.append(step("payback_months", "payback_period_months", **pb_inputs))
    gm_val = c.data["gm_pct"][c.target] / 100 if gm_pct else truth["gross_margin"]
    try:
        truth["payback_months"] = F.payback_period_months(truth["cac"], arpu.value, gm_val)  # type: ignore[arg-type]
    except F.FinanceMathError:
        truth["payback_months"] = None
    formulas.append("Payback (months) = CAC / (ARPU x Gross Margin)")
    rows.append(("CAC Payback", "payback_months", "CAC / (ARPU x Gross Margin)"))
    notes.append(f'"{c.labels["churn_pct"]}" and "{c.labels["active_customers"]}" are not inputs to payback.')

    def take(disp, vals, errs):
        if "payback_months" in errs:
            return ["CAC is never paid back: gross margin per customer is zero or negative, so each customer adds cost. "
                    "Fix gross margin before any further acquisition spend."]
        m = vals["payback_months"]
        if m <= 12:
            return [f"Payback of {disp['payback_months']} is within 12 months; acquisition spend recovers quickly."]
        if m <= 24:
            return [f"Payback of {disp['payback_months']} is acceptable but ties up cash; watch runway as you scale spend."]
        return [f"Payback of {disp['payback_months']} is long; customers must stay over two years to repay acquisition. "
                "Reduce CAC or raise ARPU/margin."]

    q = rng.choice([f"What is our CAC payback period for {P}?", f"How many months does it take to recover CAC ({P})?",
                    f"Compute CAC and the payback period in months for {P}."])
    return Plan("payback", "payback", "CAC Payback", P, q, tables, needs, steps, formulas, notes, rows, truth, take,
                missing_ok=[i for i, n in enumerate(needs) if n.table == "unit_econ"])


# --- 7. Runway -------------------------------------------------------------

def plan_runway(c: Company, rng: random.Random) -> Plan:
    P = c.period
    cb = mneed(c, "closing_cash", "cash_flow", "Closing Cash Balance")
    ci = mneed(c, "cash_in", "cash_flow", "Cash In")
    co = mneed(c, "cash_out", "cash_flow", "Cash Out")
    steps = [step("net_burn", "net_burn", cash_out=lit(co), cash_in=lit(ci)),
             step("runway_months", "runway_months", cash_balance=lit(cb), net_burn="@net_burn")]
    nb = F.net_burn(co.value, ci.value)
    truth: dict[str, float | None] = {"net_burn": nb}
    try:
        truth["runway_months"] = F.runway_months(cb.value, nb)
    except F.FinanceMathError:
        truth["runway_months"] = None
    notes = [f'Runway uses the closing balance for {P}, not "{c.labels["opening_cash"]}".']

    def take(disp, vals, errs):
        if "runway_months" in errs:
            return [f"Net burn is {disp['net_burn']}: cash in exceeded cash out in {P}, so the company is not burning "
                    "cash and runway is not a constraint this month. Check whether that inflow was a one-off (e.g. a raise)."]
        m = vals["runway_months"]
        if m < 6:
            return [f"Runway of {disp['runway_months']} is critical. Start fundraising or cut burn immediately."]
        if m < 12:
            return [f"Runway of {disp['runway_months']}: begin a raise now; fundraises usually take 6+ months."]
        if m < 18:
            return [f"Runway of {disp['runway_months']} is workable; plan the next raise within two quarters."]
        return [f"Runway of {disp['runway_months']} is comfortable at the current burn."]

    q = rng.choice([f"What is our runway as of {P}?", f"How many months of cash do we have left based on {P} burn?",
                    f"Calculate net burn and runway for {P}.", f"At the {P} burn rate, when do we run out of money?"])
    return Plan("runway", "runway", "Runway", P, q, [cash_flow_table(c, rng)], [cb, ci, co], steps,
                ["Net Burn = Cash Out - Cash In", "Runway (months) = Cash Balance / Net Burn"], notes,
                [("Net Burn", "net_burn", "Cash Out - Cash In"), ("Runway", "runway_months", "Closing Cash / Net Burn")],
                truth, take, missing_ok=[], uses_pl=False)  # cash in/out derivable from opening/closing: never "missing"


# --- 8. Burn multiple -------------------------------------------------------

def plan_burn_multiple(c: Company, rng: random.Random) -> Plan:
    P = c.period
    variant = rng.choice(["bridge", "open_close"])
    ci, co = mneed(c, "cash_in", "cash_flow", "Cash In"), mneed(c, "cash_out", "cash_flow", "Cash Out")
    needs = [ci, co]
    steps = [step("net_burn", "net_burn", cash_out=lit(co), cash_in=lit(ci))]
    nb = F.net_burn(co.value, ci.value)
    truth: dict[str, float | None] = {"net_burn": nb}
    rows = [("Net Burn", "net_burn", "Cash Out - Cash In")]
    if variant == "open_close":
        ao, ac = mneed(c, "arr_open", "arr", "Opening ARR"), mneed(c, "arr_close", "arr", "Closing ARR")
        needs += [ao, ac]
        steps.append(step("net_new_arr", "difference", a=lit(ac), b=lit(ao)))
        nna = F.difference(ac.value, ao.value)
        arr_formula = "Closing ARR - Opening ARR"
        missing_ok: list[int] = []  # every input is derivable from the other rows
    else:
        an = mneed(c, "arr_new", "arr", "New ARR")
        ae = mneed(c, "arr_expansion", "arr", "Expansion ARR")
        al = mneed(c, "arr_churned", "arr", "Churned ARR")
        needs += [an, ae, al]
        steps += [step("gross_new_arr", "total", values=[lit(an), lit(ae)]),
                  step("net_new_arr", "difference", a="@gross_new_arr", b=lit(al))]
        gross = F.total([an.value, ae.value])
        truth["gross_new_arr"] = gross
        nna = F.difference(gross, al.value)
        arr_formula = "New ARR + Expansion ARR - Churned ARR"
        rows.append(("Gross New ARR", "gross_new_arr", "New + Expansion"))
        missing_ok = []
    truth["net_new_arr"] = nna
    steps.append(step("burn_multiple", "burn_multiple", net_burn="@net_burn", net_new_arr="@net_new_arr"))
    try:
        truth["burn_multiple"] = F.burn_multiple(nb, nna)
    except F.FinanceMathError:
        truth["burn_multiple"] = None
    formulas = ["Net Burn = Cash Out - Cash In", f"Net New ARR = {arr_formula}", "Burn Multiple = Net Burn / Net New ARR"]
    rows += [("Net New ARR", "net_new_arr", arr_formula), ("Burn Multiple", "burn_multiple", "Net Burn / Net New ARR")]
    notes = [f"The ARR bridge and the cash flow are both monthly, so I use the {P} column of each."]

    def take(disp, vals, errs):
        if "burn_multiple" in errs:
            return [f"Net new ARR is {disp['net_new_arr']}: ARR shrank in {P}, so burn multiple is undefined. The company "
                    "is spending cash without growing ARR; fix churn before adding acquisition spend."]
        b = vals["burn_multiple"]
        if b < 0:
            return [f"Burn multiple is {disp['burn_multiple']}: the company generated cash while growing ARR."]
        if b < 1:
            return [f"Burn multiple of {disp['burn_multiple']} is excellent: less cash burned than ARR added."]
        if b < 2:
            return [f"Burn multiple of {disp['burn_multiple']} is good for an early-stage company."]
        if b < 3:
            return [f"Burn multiple of {disp['burn_multiple']} needs attention; growth is getting expensive."]
        return [f"Burn multiple of {disp['burn_multiple']} is high: too much cash burned per unit of new ARR."]

    q = rng.choice([f"What was our burn multiple in {P}?", f"How efficient was our growth in {P}? Give me the burn multiple.",
                    f"Calculate net burn, net new ARR and burn multiple for {P}."])
    return Plan("burn_multiple", variant, "Burn Multiple", P, q, [cash_flow_table(c, rng), arr_table(c, rng)], needs, steps,
                formulas, notes, rows, truth, take, missing_ok=missing_ok, uses_pl=False)


# --- 9. Magic number ---------------------------------------------------------

def plan_magic_number(c: Company, rng: random.Random) -> Plan:
    qs = fy_quarters(c)
    if len(qs) < 2:
        return plan_runway(c, rng)
    shown = qs[-4:]
    qi = len(shown) - 1 if (rng.random() < 0.7 or len(shown) == 2) else rng.randint(1, len(shown) - 1)
    cur, prev = shown[qi], shown[qi - 1]
    vc, vp = quarter_values(c, cur[1]), quarter_values(c, prev[1])

    def qneed(name: str, key: str, q: str, value: float) -> Need:
        return Need(name, "quarterly", QUARTER_ROWS[key], q, value, format_cell(value), key)

    needs = [qneed(f"Net Revenue ({cur[0]})", "q_revenue", cur[0], vc["q_revenue"]),
             qneed(f"Net Revenue ({prev[0]})", "q_revenue", prev[0], vp["q_revenue"]),
             qneed(f"S&M Spend ({prev[0]})", "q_sm", prev[0], vp["q_sm"])]
    steps = [step("magic_number", "magic_number", current_qtr_revenue=lit(needs[0]), prior_qtr_revenue=lit(needs[1]),
                  prior_qtr_sm_spend=lit(needs[2]))]
    truth: dict[str, float | None] = {"magic_number": F.magic_number(vc["q_revenue"], vp["q_revenue"], vp["q_sm"])}
    notes = [f"Magic number uses the PRIOR quarter's S&M spend ({prev[0]}), not the S&M spend of {cur[0]}.",
             "Quarterly figures are given directly, so no monthly totals are needed."]

    def take(disp, vals, errs):
        m = vals["magic_number"]
        if m < 0:
            return [f"Magic number is {disp['magic_number']}: revenue fell quarter on quarter despite S&M spend."]
        if m < 0.5:
            return [f"Magic number of {disp['magic_number']} is below 0.5: S&M spend is inefficient; fix the funnel first."]
        if m < 0.75:
            return [f"Magic number of {disp['magic_number']} is moderate; optimise channels before adding budget."]
        return [f"Magic number of {disp['magic_number']} is above 0.75: S&M is efficient and can be scaled."]

    q = rng.choice([f"What is our magic number for {cur[0]}?",
                    f"Is our sales and marketing efficient? Compute the magic number for {cur[0]}.",
                    f"Calculate the SaaS magic number for {cur[0]} versus {prev[0]}."])
    return Plan("magic_number", "magic_number", "Magic Number", cur[0], q, [quarterly_table(c, rng, shown)], needs, steps,
                ["Magic Number = (Current Qtr Revenue - Prior Qtr Revenue) x 4 / Prior Qtr S&M Spend"], notes,
                [("Magic Number", "magic_number", "(Rev now - Rev prior) x 4 / Prior S&M")], truth, take,
                missing_ok=[0, 2], uses_pl=False)


# --- 10. Gross margin ---------------------------------------------------------

def plan_gross_margin(c: Company, rng: random.Random) -> Plan:
    P = c.period
    gr, di, re_ = (mneed(c, k, "pl_revenue") for k in ("gross_revenue", "discounts", "returns"))
    cogs = mneed(c, "cogs", "pl_cogs")
    steps = [step("net_revenue", "net_revenue", gross_revenue=lit(gr), discounts=lit(di), returns=lit(re_)),
             step("gross_profit", "gross_profit", net_revenue="@net_revenue", cogs=lit(cogs)),
             step("gross_margin", "gross_margin", net_revenue="@net_revenue", cogs=lit(cogs))]
    nr = F.net_revenue(gr.value, di.value, re_.value)
    truth: dict[str, float | None] = {"net_revenue": nr, "gross_profit": F.gross_profit(nr, cogs.value),
                                      "gross_margin": F.gross_margin(nr, cogs.value)}
    notes = ["Gross margin is measured on NET revenue (after discounts and returns), not gross revenue."]

    def take(disp, vals, errs):
        g = vals["gross_margin"]
        if g < 0:
            return [f"Gross margin is negative ({disp['gross_margin']}): COGS exceeds net revenue. Fix pricing or sourcing first."]
        if g < 0.4:
            return [f"Gross margin of {disp['gross_margin']} is thin for a consumer brand; watch COGS and discounting."]
        return [f"Gross margin of {disp['gross_margin']} is healthy."]

    q = rng.choice([f"What was our gross margin in {P}?", f"Calculate gross profit and gross margin for {P}.",
                    f"After COGS, how much of our {P} net revenue did we keep?"])
    return Plan("gross_margin", "gross_margin", "Gross Margin", P, q,
                [pl_table(c, rng, "pl_revenue"), pl_table(c, rng, "pl_cogs")], [gr, di, re_, cogs], steps,
                ["Net Revenue = Gross Revenue - (Discounts + Returns)", "Gross Profit = Net Revenue - COGS",
                 "Gross Margin = (Net Revenue - COGS) / Net Revenue"], notes,
                [("Net Revenue", "net_revenue", "Gross Revenue - (Discounts + Returns)"),
                 ("Gross Profit", "gross_profit", "Net Revenue - COGS"),
                 ("Gross Margin", "gross_margin", "Gross Profit / Net Revenue")], truth, take, missing_ok=[0, 1, 2, 3])


TASKS: dict[str, Callable[[Company, random.Random], Plan]] = {
    "net_revenue": plan_net_revenue,
    "contribution_margin": plan_contribution_margin,
    "ebitda": plan_ebitda,
    "break_even": plan_break_even,
    "ltv_cac": plan_ltv_cac,
    "payback": plan_payback,
    "runway": plan_runway,
    "burn_multiple": plan_burn_multiple,
    "magic_number": plan_magic_number,
    "gross_margin": plan_gross_margin,
}

# ---------------------------------------------------------------------------
# Example assembly
# ---------------------------------------------------------------------------

OPENERS = ["The question asks for {what} for {period}.", "Goal: {what} for {period}.", "I need {what} for {period}.",
           "Task: compute {what} ({period})."]
CLOSERS = ["All arithmetic goes to finance_calculator; I will not compute anything by hand.",
           "Sending these inputs to finance_calculator instead of doing the math myself.",
           "Next: pass the extracted inputs to finance_calculator."]


MONTH_NAMES = {abbr: full for abbr, full in zip(calendar.month_abbr[1:], calendar.month_name[1:])}


def say_period(rng: random.Random, label: str) -> str:
    """How a founder might write a period in a question. Tables always use 'Mar-26' / 'Q4 FY26'."""
    m = re.match(r"^([A-Z][a-z]{2})-(\d{2})$", label)
    if m:
        mon, yy = m.group(1), m.group(2)
        # never "April 25": it reads like a date (25 April)
        return rng.choice([label, label, f"{MONTH_NAMES[mon]} 20{yy}", f"{mon} 20{yy}", f"{mon} '{yy}"])
    q = re.match(r"^Q([1-4]) FY(\d{2})$", label)
    if q:
        return rng.choice([label, label, f"Q{q.group(1)} of FY{q.group(2)}", f"Q{q.group(1)} FY20{q.group(2)}"])
    return label


def absent_months(c: Company) -> list[str]:
    """Months just outside the table window (up to 12 before, up to 6 after): asked about, but not in any table."""
    def shift(y: int, m: int, k: int) -> str:
        t = y * 12 + (m - 1) + k
        return month_label(t // 12, t % 12 + 1)

    (y0, m0), (y1, m1) = c.ym[c.window[0]], c.ym[c.window[-1]]
    return [shift(y0, m0, -k) for k in range(1, 13)] + [shift(y1, m1, k) for k in range(1, 7)]


def apply_missing(plan: Plan, tables: list[Table], rng: random.Random) -> Need:
    need = plan.needs[rng.choice(plan.missing_ok)]
    table = next(t for t in tables if t.key == need.table)
    ri = next(i for i, r in enumerate(table.rows) if r[0] == need.label)
    if need.column in table.headers and need.column not in ("Value",) and rng.random() < 0.5:
        table.rows[ri][table.headers.index(need.column)] = "n/a"
        need.shown = "n/a"
    else:
        del table.rows[ri]
        need.shown = ""
    return need


def _what(plan: Plan) -> str:
    if plan.task != "contribution_margin":
        return plan.title
    return "CM1, CM2 and CM3" if plan.variant == "cm3" else "CM1 and CM2"


def period_gap_example(idx: int, plan: Plan, c: Company, tables: list[Table], ref: Callable[[str], str],
                       asked: str, rng: random.Random, task: str, split: str) -> dict[str, Any]:
    """Missing-data example where the asked month is outside every table. One assistant turn, no tool call.

    Teaches the behaviour the v1 adapter lacked: it used a neighbouring period's column and labelled it as the
    asked period.
    """
    first, last = c.months[c.window[0]], c.months[c.window[-1]]
    question = plan.question.replace(plan.period, say_period(rng, asked))
    user = build_user_message([t.render() for t in tables], question)
    dated = [t for t in tables if any(_MONTH_LABEL_RE.match(h) for h in t.headers)]
    think = [rng.choice(OPENERS).format(what=_what(plan), period=asked),
             f"Periods in the context: {first} to {last}"
             + (f" ({', '.join(ref(t.key) for t in dated)})." if dated else "."),
             f"{asked} is not in any table.",
             f"Missing: {asked} data. {plan.title} cannot be computed for {asked}, and I must not use another "
             f"period's column (such as {last if month_key(asked) > month_key(last) else first}) in its place. "
             "I will not call the calculator."]
    report = "\n".join([
        f"## CFO Report: {plan.title} ({asked}) - data missing",
        "",
        f"**Cannot compute {plan.title}:** {asked} is not in the provided data (the tables cover {first} to {last}).",
        "",
        f"**Next step:** share the MIS for {asked}, or ask about a period between {first} and {last}. "
        "No value has been assumed and no other period's figures were used.",
    ])
    meta: dict[str, Any] = {
        "id": f"raft-{idx:05d}", "task": task, "variant": plan.variant, "split": split, "period": asked,
        "company": c.name, "currency": c.currency, "edge_cases": sorted(c.edges), "missing": asked,
        "missing_kind": "period", "prompt_version": PROMPT_VERSION,
    }
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user},
            {"role": "assistant", "content": "<think>\n" + "\n".join(think) + "\n</think>\n\n" + report}]
    return {"id": meta["id"], "messages": msgs, "meta": meta}


def build_example(idx: int, cfg: dict[str, Any], task: str, missing: bool, split: str) -> dict[str, Any]:
    rng = random.Random(cfg["seed"] * 1_000_003 + idx)
    c = simulate_company(rng, cfg["edge_cases"])
    plan = TASKS[task](c, rng)
    # Two kinds of missing-data example: a needed input cell/row is absent ("cell"), or the question asks for a
    # month that no table covers ("period"). Period gaps also work for runway / burn multiple, whose inputs are
    # always derivable from other rows.
    gap_ok = bool(_MONTH_LABEL_RE.match(plan.period)) and plan.period in plan.question
    period_gap = False
    if missing:
        if gap_ok and (not plan.missing_ok or rng.random() < cfg.get("missing_period_share", 0.4)):
            period_gap = True
        elif not plan.missing_ok:
            missing = False
    relevant = plan.tables
    missing_need = apply_missing(plan, relevant, rng) if missing and not period_gap else None

    # Distractors: other P&L sections + unrelated tables. Missing-data examples avoid any table that
    # could be an alternative source for the missing value.
    pool = dict(distractor_tables(c, rng))
    rel_keys = {t.key for t in relevant}
    for k in rel_keys:
        pool.pop(k, None)
    if not missing:
        if "arr" not in rel_keys:
            pool["arr"] = arr_table(c, rng)
        qs = fy_quarters(c)
        if "quarterly" not in rel_keys and len(qs) >= 2:
            pool["quarterly"] = quarterly_table(c, rng, qs[-4:])
        for k in PL_SECTIONS:
            if k not in rel_keys:
                pool[k] = pl_table(c, rng, k)
        if "unit_econ" not in rel_keys:
            pool["unit_econ"] = unit_econ_table(c, rng, True, True)
        if "pricing" not in rel_keys:
            pool["pricing"] = pricing_table(c, rng, ["price", "vc", "fixed"])
        if plan.uses_pl and c.window[0] >= 12:
            prior = monthly_table(c, rng, "pl_revenue_prior", "Monthly P&L - Revenue (prior year)", "P&L",
                                  "Revenue", [k for k in PL_SECTIONS["pl_revenue"][1]], offset=12)
            pool[prior.key] = prior
    else:
        pool.pop("d_cohort", None)
    lo, hi = cfg["distractors"]["min"], cfg["distractors"]["max"]
    keys = sorted(pool)
    distract = [pool[k] for k in rng.sample(keys, k=min(len(keys), rng.randint(lo, hi)))]
    budget = cfg.get("max_context_chars")
    while budget and len(distract) > lo and sum(len(t.render()) for t in relevant + distract) > budget:
        distract.pop()
    tables = relevant + distract
    rng.shuffle(tables)
    pos = {t.key: i + 1 for i, t in enumerate(tables)}
    ref = lambda key: f"[{pos[key]}]"  # noqa: E731

    if period_gap:
        rendered = "\n".join(t.render() for t in tables)
        options = [p for p in absent_months(c) if p not in rendered]
        if options:
            return period_gap_example(idx, plan, c, tables, ref, rng.choice(options), rng, task, split)
        missing = False  # every nearby month appears somewhere (very rare): fall back to a normal example

    question = plan.question.replace(plan.period, say_period(rng, plan.period)) if plan.period in plan.question else plan.question
    user = build_user_message([t.render() for t in tables], question)

    # --- first assistant turn: extraction + formula choice
    think = [rng.choice(OPENERS).format(what=plan.title if plan.task != "contribution_margin" else
                                         ("CM1, CM2 and CM3" if plan.variant == "cm3" else "CM1 and CM2"),
                                         period=plan.period)]
    think.append("Relevant: " + ", ".join(f"{ref(t.key)} {t.title}" for t in relevant) + ".")
    think.append("Not needed: " + ", ".join(f"{ref(t.key)} {t.title}" for t in distract) + ".")
    if any(t.key == "pl_revenue_prior" for t in distract):
        think.append(f"{ref('pl_revenue_prior')} covers the prior year; I use only the {plan.period} column.")
    think.append("Inputs:")
    for n in plan.needs:
        if n is missing_need:
            think.append(f"- {n.name}: NOT FOUND for {n.column} in {ref(n.table)}")
        else:
            think.append(f'- {n.name} = {n.shown} ({ref(n.table)} row "{n.label}", {n.column})')
    for note in plan.notes:
        think.append(f"Note: {note}")
    think.append("Formulas:")
    think += [f"- {f}" for f in plan.formulas]

    meta: dict[str, Any] = {
        "id": f"raft-{idx:05d}", "task": task, "variant": plan.variant, "split": split, "period": plan.period,
        "company": c.name, "currency": c.currency, "edge_cases": sorted(c.edges), "missing": None,
        "prompt_version": PROMPT_VERSION,
    }
    msgs: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    if missing_need is not None:
        how = "is blank (n/a)" if missing_need.shown == "n/a" else "does not appear"
        think.append(f"Missing: {missing_need.name} for {missing_need.column} {how} in the context. "
                     f"{plan.title} cannot be computed without it, and I must not assume a value or borrow another "
                     "period's figure. I will not call the calculator.")
        table_rows = [f"| {n.name} | {'**missing**' if n is missing_need else n.shown} |" for n in plan.needs]
        report = "\n".join([
            f"## CFO Report: {plan.title} ({plan.period}) - data missing",
            "",
            f"**Cannot compute {plan.title}:** {missing_need.name} for {missing_need.column} is not in the provided data.",
            "",
            "| Required input | Value |",
            "|---|---|",
            *table_rows,
            "",
            f"**Next step:** share {missing_need.name} for {missing_need.column} (from the MIS or ledger) and I will "
            "rerun the calculation. No value has been assumed.",
        ])
        msgs.append({"role": "assistant", "content": f"<think>\n" + "\n".join(think) + "\n</think>\n\n" + report})
        meta["missing"] = missing_need.name
        meta["missing_kind"] = "cell"
        return {"id": meta["id"], "messages": msgs, "meta": meta}

    think.append(rng.choice(CLOSERS))
    args = {"steps": plan.steps}
    resp = execute(args)
    tool_json = response_to_json(resp)

    # ground-truth check (independent of the context text and the tool call)
    problems = compare_truth(plan.truth, resp)
    if problems:
        raise AssertionError(f"{meta['id']} ({task}/{plan.variant}): {problems}")

    disp = {k: r.display for k, r in resp.results.items()}
    vals = {k: r.value for k, r in resp.results.items()}
    errs = dict(resp.errors)

    t2 = ["Calculator results: " + "; ".join(f"{k} = {v}" for k, v in disp.items()) + "."
          if disp else "Calculator returned no results."]
    for k, e in errs.items():
        t2.append(f"{k} could not be computed: {e}.")
    if not errs:
        t2.append("No errors. I will copy these values exactly into the report.")
    else:
        t2.append("I will report the failed metrics as not computable with the reason, not estimate them.")

    report_lines = [f"## CFO Report: {plan.title} ({plan.period})",
                    # Only facts that are in the context: the company name never is (v1 taught the model to invent one).
                    f"Source: {c.source} | Currency: {c.currency}", "",
                    "| Metric | Value | Formula |", "|---|---|---|"]
    for metric, sid, formula in plan.report_rows:
        if sid in disp:
            report_lines.append(f"| {metric} | {disp[sid]} | {formula} |")
        else:
            report_lines.append(f"| {metric} | Not computable | {errs.get(sid, 'not computed')} |")
    report_lines += ["", "**Inputs used:** " + "; ".join(f"{n.name} {n.shown}" for n in plan.needs) + f" ({plan.period}).",
                     "", "**CFO take:**", *[f"- {line}" for line in plan.takeaway(disp, vals, errs)]]
    report = "\n".join(report_lines)

    msgs.append({"role": "assistant", "content": "<think>\n" + "\n".join(think) + "\n</think>\n\n",
                 "tool_calls": [{"type": "function", "function": {"name": TOOL_NAME, "arguments": args}}]})
    msgs.append({"role": "tool", "content": tool_json})
    msgs.append({"role": "assistant", "content": "<think>\n" + "\n".join(t2) + "\n</think>\n\n" + report})
    meta["truth"] = {k: (None if v is None else round(v, 6)) for k, v in plan.truth.items()}
    return {"id": meta["id"], "messages": msgs, "meta": meta}


def compare_truth(truth: dict[str, float | None], resp) -> list[str]:
    problems = []
    for k, v in truth.items():
        if v is None:
            if k not in resp.errors:
                problems.append(f"{k}: expected an error, got {resp.results.get(k)}")
            continue
        if k not in resp.results:
            problems.append(f"{k}: missing from results (errors={resp.errors.get(k)})")
            continue
        got = resp.results[k].value
        if abs(got - v) > 1e-6 * max(1.0, abs(v)):
            problems.append(f"{k}: tool={got} truth={v}")
    return problems


# ---------------------------------------------------------------------------
# Validation of a written dataset
# ---------------------------------------------------------------------------

def validate_example(ex: dict[str, Any]) -> list[str]:
    """Recompute every number in one example. Returns a list of problems (empty = valid)."""
    msgs, meta = ex["messages"], ex["meta"]
    errs: list[str] = []
    if meta.get("prompt_version") != PROMPT_VERSION:
        errs.append("prompt_version mismatch (regenerate the dataset)")
    if meta.get("missing"):
        if len(msgs) != 3 or "tool_calls" in msgs[2]:
            errs.append("missing-data example must have one assistant turn and no tool call")
        if "missing" not in msgs[2]["content"].lower():
            errs.append("missing-data report does not say what is missing")
        if meta.get("missing_kind") == "period":
            context = msgs[1]["content"].rsplit("Question:", 1)[0]
            if meta["period"] in context:
                errs.append(f"period-gap example: {meta['period']} appears in the context")
            if meta["period"] not in msgs[2]["content"]:
                errs.append("period-gap example: the answer does not name the asked period")
        return errs
    call = msgs[2]["tool_calls"][0]["function"]
    resp = execute(call["arguments"])
    if response_to_json(resp) != msgs[3]["content"]:
        errs.append("tool message differs from recomputed calculator output")
    errs += compare_truth(meta["truth"], resp)
    for bad in unsupported_inputs(call["arguments"], msgs[1]["content"]):
        errs.append(f"input {bad} does not appear in the context")
    report = msgs[4]["content"]
    for k, r in resp.results.items():
        if r.display not in report:
            errs.append(f"result {k} display '{r.display}' missing from final report")
    return errs


def validate_file(path: Path) -> int:
    bad = 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            n += 1
            problems = validate_example(ex)
            if problems:
                bad += 1
                if bad <= 10:
                    print(f"[MISMATCH] {ex['id']}: {problems}")
    print(f"validate: {n} examples, {bad} with mismatches")
    return bad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    master = random.Random(cfg["seed"])
    n = cfg["n_examples"]
    names = list(cfg["task_weights"])
    weights = [cfg["task_weights"][k] for k in names]
    tasks = master.choices(names, weights=weights, k=n)
    order = list(range(n))
    master.shuffle(order)
    test_ids = set(order[: round(n * cfg["test_fraction"])])
    master.shuffle(order)
    missing_ids = set(order[: round(n * cfg["missing_fraction"])])
    return [build_example(i, cfg, tasks[i], i in missing_ids, "test" if i in test_ids else "train") for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "data.yaml"))
    ap.add_argument("--n", type=int, help="override n_examples")
    ap.add_argument("--output", help="override output path")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.n:
        cfg["n_examples"] = args.n
    out = Path(args.output or ROOT / cfg["output"])

    if not args.validate_only:
        examples = generate(cfg)
        with out.open("w", encoding="utf-8", newline="\n") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        c = Counter((e["meta"]["task"], e["meta"]["split"]) for e in examples)
        print(f"wrote {len(examples)} examples -> {out}")
        for task in cfg["task_weights"]:
            print(f"  {task:<20} train={c[(task, 'train')]:>4} test={c[(task, 'test')]:>3}")
        print(f"  missing-data examples: {sum(1 for e in examples if e['meta']['missing'])}")
        edge = Counter(x for e in examples for x in e["meta"]["edge_cases"])
        print(f"  edge cases present: {dict(edge)}")
    return 1 if validate_file(out) else 0


if __name__ == "__main__":
    sys.exit(main())
