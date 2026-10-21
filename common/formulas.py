"""Single source of truth for all startup finance math.

Every other module (data generation, the calculator tool, tests) imports from here.
Functions take and return plain floats. Percentages are fractions (0.42 = 42%).
Inputs that make a metric undefined raise FinanceMathError with a plain-English reason,
so the caller can report it instead of inventing a number.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


class FinanceMathError(ValueError):
    """A metric is undefined for the given inputs (e.g. zero churn, price <= variable cost)."""


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise FinanceMathError(f"{name} must be > 0 (got {value:g})")


# ---------------------------------------------------------------------------
# Revenue and margins
# ---------------------------------------------------------------------------

def net_revenue(gross_revenue: float, discounts: float, returns: float) -> float:
    """Net Revenue = Gross Revenue - (Discounts + Returns)."""
    return gross_revenue - (discounts + returns)


def gross_profit(net_revenue: float, cogs: float) -> float:
    """Gross Profit = Net Revenue - COGS (same value as CM1)."""
    return net_revenue - cogs


def gross_margin(net_revenue: float, cogs: float) -> float:
    """Gross Margin = (Net Revenue - COGS) / Net Revenue. Can be negative."""
    _require_positive("net_revenue", net_revenue)
    return (net_revenue - cogs) / net_revenue


def contribution_margin(net_revenue: float, variable_costs: float) -> float:
    """Contribution Margin = Net Revenue - Variable Costs."""
    return net_revenue - variable_costs


def cm1(net_revenue: float, cogs: float) -> float:
    """CM1 = Net Revenue - COGS."""
    return net_revenue - cogs


def cm2(cm1: float, variable_ops_costs: float) -> float:
    """CM2 = CM1 - variable ops costs (fulfillment, logistics, payment gateway, packaging)."""
    return cm1 - variable_ops_costs


def cm3(cm2: float, variable_marketing: float) -> float:
    """CM3 = CM2 - variable (performance) marketing."""
    return cm2 - variable_marketing


def margin_pct(value: float, net_revenue: float) -> float:
    """Any profit line as a fraction of net revenue (e.g. CM2 / Net Revenue)."""
    _require_positive("net_revenue", net_revenue)
    return value / net_revenue


# ---------------------------------------------------------------------------
# Profitability
# ---------------------------------------------------------------------------

def ebit_from_net_income(net_income: float, interest: float, taxes: float) -> float:
    """EBIT = Net Income + Interest + Taxes."""
    return net_income + interest + taxes


def ebitda(ebit: float, depreciation: float, amortization: float) -> float:
    """EBITDA = EBIT + Depreciation + Amortization (excludes interest and taxes by construction)."""
    return ebit + depreciation + amortization


def adjusted_ebitda(ebitda: float, adjustments: Iterable[float]) -> float:
    """Adjusted EBITDA = EBITDA +/- documented one-off items.

    Each adjustment is signed: add-backs positive, one-off gains negative.
    """
    return ebitda + sum(adjustments)


# ---------------------------------------------------------------------------
# Cash
# ---------------------------------------------------------------------------

def net_burn(cash_out: float, cash_in: float) -> float:
    """Net Burn = Cash Out - Cash In (monthly). Negative means cash-generative."""
    return cash_out - cash_in


def runway_months(cash_balance: float, net_burn: float) -> float:
    """Runway = Cash Balance / Net Burn (months)."""
    if net_burn <= 0:
        raise FinanceMathError(
            f"net_burn must be > 0 for a finite runway (got {net_burn:g}); the company is not burning cash"
        )
    if cash_balance < 0:
        raise FinanceMathError(f"cash_balance is negative ({cash_balance:g}); runway already exhausted")
    return cash_balance / net_burn


def burn_multiple(net_burn: float, net_new_arr: float) -> float:
    """Burn Multiple = Net Burn / Net New ARR."""
    _require_positive("net_new_arr", net_new_arr)
    return net_burn / net_new_arr


# ---------------------------------------------------------------------------
# Unit economics
# ---------------------------------------------------------------------------

def cac(sm_spend: float, new_customers: float) -> float:
    """CAC = Total Sales & Marketing Spend / New Customers."""
    _require_positive("new_customers", new_customers)
    return sm_spend / new_customers


def ltv(arpu: float, gross_margin: float, churn_rate: float) -> float:
    """LTV = (ARPU x Gross Margin) / Churn Rate. ARPU and churn on the same (monthly) basis.

    A negative gross margin gives a negative LTV (each customer destroys value).
    """
    _require_positive("churn_rate", churn_rate)
    return (arpu * gross_margin) / churn_rate


def ltv_cac_ratio(ltv: float, cac: float) -> float:
    """LTV:CAC = LTV / CAC."""
    _require_positive("cac", cac)
    return ltv / cac


def payback_period_months(cac: float, arpu: float, gross_margin: float) -> float:
    """Payback Period = CAC / (ARPU x Gross Margin), in months."""
    monthly_gp = arpu * gross_margin
    if monthly_gp <= 0:
        raise FinanceMathError(
            f"ARPU x gross_margin must be > 0 (got {monthly_gp:g}); CAC is never paid back"
        )
    return cac / monthly_gp


def break_even_units(fixed_costs: float, price: float, variable_cost_per_unit: float) -> float:
    """Break-even units = Fixed Costs / (Price - Variable Cost per unit). Exact (fractional) value."""
    unit_contribution = price - variable_cost_per_unit
    if unit_contribution <= 0:
        raise FinanceMathError(
            f"price ({price:g}) must exceed variable_cost_per_unit ({variable_cost_per_unit:g}); "
            "every unit sold loses money, so break-even is impossible"
        )
    if fixed_costs < 0:
        raise FinanceMathError(f"fixed_costs must be >= 0 (got {fixed_costs:g})")
    return fixed_costs / unit_contribution


def break_even_units_whole(fixed_costs: float, price: float, variable_cost_per_unit: float) -> int:
    """Break-even in whole units (rounded up: you cannot sell a fraction of a unit)."""
    return math.ceil(round(break_even_units(fixed_costs, price, variable_cost_per_unit), 9))


def break_even_revenue(fixed_costs: float, price: float, variable_cost_per_unit: float) -> float:
    """Break-even revenue = break-even units x price."""
    return break_even_units(fixed_costs, price, variable_cost_per_unit) * price


def magic_number(current_qtr_revenue: float, prior_qtr_revenue: float, prior_qtr_sm_spend: float) -> float:
    """Magic Number = (Current Qtr Revenue - Prior Qtr Revenue) x 4 / Prior Qtr S&M Spend."""
    _require_positive("prior_qtr_sm_spend", prior_qtr_sm_spend)
    return (current_qtr_revenue - prior_qtr_revenue) * 4 / prior_qtr_sm_spend


# ---------------------------------------------------------------------------
# Generic helpers (exposed to the calculator so the model never adds in its head)
# ---------------------------------------------------------------------------

def total(values: Iterable[float]) -> float:
    """Sum of a list of line items."""
    return float(sum(values))


def difference(a: float, b: float) -> float:
    """a - b."""
    return a - b


def ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator."""
    if denominator == 0:
        raise FinanceMathError("denominator must not be 0")
    return numerator / denominator


def percent_of(part: float, whole: float) -> float:
    """part / whole as a fraction (shown as a percentage), e.g. discounts as a share of gross revenue."""
    if whole == 0:
        raise FinanceMathError("whole must not be 0")
    return part / whole
