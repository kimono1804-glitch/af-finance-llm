"""Real-world evaluation cases: natural questions about sample_startup_mis.xlsx.

Expected values are computed with common.formulas from the same data make_sample_mis.py writes into the
workbook, so they are exact ground truth. Questions are phrased the way a founder would ask (month names,
"how much did we keep", etc.), unlike the templated training questions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common import formulas as F
from make_sample_mis import MONTHS, PRICING, build_data


@dataclass
class RealCase:
    case_id: str
    task: str
    question: str
    truth: dict[str, list[float] | None]
    required_formulas: list[set[str]] = field(default_factory=list)
    missing: str | None = None
    missing_keywords: list[str] = field(default_factory=list)
    needed_sections: set[str] = field(default_factory=set)   # for retrieval recall
    edge_cases: list[str] = field(default_factory=list)


def _label(i: int) -> str:
    return MONTHS[i].strftime("%b-%y")


def build_cases() -> list[RealCase]:
    d = build_data()
    idx = {_label(i): i for i in range(len(MONTHS))}

    def v(key: str, p: str) -> float:
        return d[key][idx[p]]

    def nr(p: str) -> float:
        return F.net_revenue(v("gross_revenue", p), v("discounts", p), v("returns", p))

    def cms(p: str) -> tuple[float, float, float]:
        c1 = F.cm1(nr(p), v("cogs", p))
        c2 = F.cm2(c1, F.total([v("fulfillment", p), v("payment_gateway", p), v("packaging", p)]))
        return c1, c2, F.cm3(c2, v("perf_marketing", p))

    def ebitda(p: str) -> float:
        return F.ebitda(v("ebit", p), v("depreciation", p), F.total([v("amortization", p), v("amort_rnd", p)]))

    def unit(p: str) -> dict[str, list[float]]:
        """CAC, LTV, LTV:CAC, payback. Gross margin may come from the sheet's GM% row or from the P&L."""
        cac = F.cac(v("sm_spend", p), v("new_customers", p))
        gms = [v("gm", p), F.gross_margin(nr(p), v("cogs", p))]
        ltvs = [F.ltv(v("arpu", p), g, v("churn", p)) for g in gms]
        return {"cac": [cac], "ltv": ltvs, "ltv_cac_ratio": [F.ltv_cac_ratio(x, cac) for x in ltvs],
                "payback_months": [F.payback_period_months(cac, v("arpu", p), g) for g in gms]}

    rev = {"Revenue"}
    cm = {"Revenue", "Cost of Revenue", "Variable Costs"}
    ebit_secs = {"D&A, Interest & Tax", "Profit Summary"}
    ue = {"Customer Metrics"}
    price = {"Unit Cost & Fixed Cost Plan"}
    pr, vc, fx, hires = PRICING["price"], PRICING["variable_cost"], PRICING["fixed_costs"], PRICING["planned_hires"]

    cases = [
        RealCase("rw-01", "net_revenue", "What was our net revenue for Mar-26?",
                 {"net_revenue": [nr("Mar-26")]}, [{"net_revenue"}], needed_sections=rev),
        RealCase("rw-02", "net_revenue", "How much revenue did we actually keep in November 2025 after discounts and refunds?",
                 {"net_revenue": [nr("Nov-25")]}, [{"net_revenue"}], needed_sections=rev),
        RealCase("rw-03", "contribution_margin", "What were CM1 and CM2 for Mar-26, in absolute terms and as % of net revenue?",
                 {"cm1": [cms("Mar-26")[0]], "cm2": [cms("Mar-26")[1]],
                  "cm2_pct": [F.margin_pct(cms("Mar-26")[1], nr("Mar-26"))]},
                 [{"net_revenue"}, {"cm1", "gross_profit", "contribution_margin"}, {"cm2", "contribution_margin"}],
                 needed_sections=cm),
        RealCase("rw-04", "contribution_margin", "Are we making money per order in Jan 2026? Give me CM2.",
                 {"cm2": [cms("Jan-26")[1]]}, [{"cm2", "contribution_margin"}], needed_sections=cm),
        RealCase("rw-05", "contribution_margin", "Walk me from gross revenue down to CM3 for Oct-25.",
                 {"cm3": [cms("Oct-25")[2]]}, [{"cm3", "contribution_margin"}],
                 needed_sections=cm | {"Marketing"}),
        RealCase("rw-06", "ebitda", "Reconcile EBITDA for Mar-26 from the P&L.",
                 {"ebitda": [ebitda("Mar-26")]}, [{"ebitda"}], needed_sections=ebit_secs,
                 edge_cases=["capitalized_rnd"]),
        RealCase("rw-07", "ebitda", "What was EBITDA in December 2025?",
                 {"ebitda": [ebitda("Dec-25")]}, [{"ebitda"}], needed_sections=ebit_secs,
                 edge_cases=["capitalized_rnd"]),
        RealCase("rw-08", "ebitda", "Calculate Feb-26 EBITDA and EBITDA margin.",
                 {"ebitda": [ebitda("Feb-26")], "ebitda_margin": [F.margin_pct(ebitda("Feb-26"), nr("Feb-26"))]},
                 [{"ebitda"}], needed_sections=ebit_secs | rev, edge_cases=["capitalized_rnd"]),
        RealCase("rw-09", "ltv_cac", "What is our LTV:CAC for Jan-26?",
                 {k: unit("Jan-26")[k] for k in ("cac", "ltv", "ltv_cac_ratio")},
                 [{"cac"}, {"ltv"}, {"ltv_cac_ratio"}], needed_sections=ue, edge_cases=["seasonal_churn_spike"]),
        RealCase("rw-10", "ltv_cac", "Investor wants our March 2026 LTV to CAC. Compute it from the MIS.",
                 {k: unit("Mar-26")[k] for k in ("cac", "ltv", "ltv_cac_ratio")},
                 [{"cac"}, {"ltv"}, {"ltv_cac_ratio"}], needed_sections=ue),
        RealCase("rw-11", "payback", "What is our CAC payback period for Feb-26?",
                 {k: unit("Feb-26")[k] for k in ("cac", "payback_months")},
                 [{"cac"}, {"payback_period_months"}], needed_sections=ue),
        RealCase("rw-12", "payback", "How many months does it take to recover CAC in Sep 2025?",
                 {k: unit("Sep-25")[k] for k in ("cac", "payback_months")},
                 [{"cac"}, {"payback_period_months"}], needed_sections=ue),
        RealCase("rw-13", "break_even", "What is our monthly break-even volume and revenue under the FY27 plan?",
                 {"units": [F.break_even_units(fx, pr, vc)], "revenue": [F.break_even_revenue(fx, pr, vc)]},
                 [{"break_even_units"}], needed_sections=price),
        RealCase("rw-14", "break_even", "If we make the planned hires, how does our monthly break-even change?",
                 {"units_now": [F.break_even_units(fx, pr, vc)],
                  "units_after": [F.break_even_units(F.total([fx, hires]), pr, vc)]},
                 [{"break_even_units"}], needed_sections=price, edge_cases=["step_fixed_costs"]),
        RealCase("rw-15", "net_revenue", "What was net revenue in Jul 2025 and how much leaked to discounts and returns?",
                 {"net_revenue": [nr("Jul-25")],
                  "leakage": [F.total([v("discounts", "Jul-25"), v("returns", "Jul-25")])]},
                 [{"net_revenue"}], needed_sections=rev),
        RealCase("rw-20", "net_revenue", "What was our net revenue last month?",
                 {"net_revenue": [nr("Mar-26")]}, [{"net_revenue"}], needed_sections=rev),
        RealCase("rw-21", "contribution_margin", "What was CM2 in the previous month?",
                 {"cm2": [cms("Feb-26")[1]]}, [{"cm2", "contribution_margin"}], needed_sections=cm),
        # Missing data: the period is not in the workbook. The model must say so and not compute.
        RealCase("rw-16", "ltv_cac", "What was our LTV:CAC for Mar-25?", {}, missing="Mar-25 data",
                 missing_keywords=["Mar-25"], needed_sections=ue),
        RealCase("rw-17", "net_revenue", "What was net revenue in Apr-26?", {}, missing="Apr-26 data",
                 missing_keywords=["Apr-26"], needed_sections=rev),
        RealCase("rw-18", "payback", "What was our CAC payback for Dec-24?", {}, missing="Dec-24 data",
                 missing_keywords=["Dec-24"], needed_sections=ue),
        RealCase("rw-19", "ebitda", "What was EBITDA for Jun-24?", {}, missing="Jun-24 data",
                 missing_keywords=["Jun-24"], needed_sections=ebit_secs),
    ]
    return cases
