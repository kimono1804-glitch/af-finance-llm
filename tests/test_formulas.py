import math

import pytest

from common import formulas as F
from common.formulas import FinanceMathError


def test_net_revenue():
    assert F.net_revenue(1_250_000, 50_000, 20_000) == 1_180_000


def test_gross_margin_positive_and_negative():
    assert F.gross_margin(1000, 600) == pytest.approx(0.4)
    # COGS above revenue -> negative gross margin is a valid (bad) result, not an error
    assert F.gross_margin(1000, 1300) == pytest.approx(-0.3)


def test_gross_margin_zero_revenue_raises():
    with pytest.raises(FinanceMathError):
        F.gross_margin(0, 100)


def test_contribution_margin_layers():
    nr = F.net_revenue(1_000_000, 40_000, 10_000)
    c1 = F.cm1(nr, 400_000)
    c2 = F.cm2(c1, 120_000)
    c3 = F.cm3(c2, 150_000)
    assert (nr, c1, c2, c3) == (950_000, 550_000, 430_000, 280_000)
    assert F.contribution_margin(nr, 400_000 + 120_000 + 150_000) == c3
    assert F.margin_pct(c2, nr) == pytest.approx(430_000 / 950_000)


def test_ebitda_excludes_interest_and_taxes():
    ebit = F.ebit_from_net_income(net_income=-200_000, interest=30_000, taxes=0)
    assert ebit == -170_000
    assert F.ebitda(ebit, depreciation=25_000, amortization=40_000) == -105_000


def test_adjusted_ebitda_signed_adjustments():
    assert F.adjusted_ebitda(100_000, [50_000, -20_000]) == 130_000
    assert F.adjusted_ebitda(100_000, []) == 100_000


def test_burn_and_runway():
    burn = F.net_burn(cash_out=900_000, cash_in=600_000)
    assert burn == 300_000
    assert F.runway_months(3_000_000, burn) == pytest.approx(10.0)


@pytest.mark.parametrize("burn", [0, -50_000])
def test_runway_not_burning_raises(burn):
    with pytest.raises(FinanceMathError, match="not burning"):
        F.runway_months(1_000_000, burn)


def test_runway_negative_cash_raises():
    with pytest.raises(FinanceMathError):
        F.runway_months(-1, 100)


def test_cac_ltv_payback():
    c = F.cac(sm_spend=500_000, new_customers=250)
    assert c == 2000
    l = F.ltv(arpu=1000, gross_margin=0.6, churn_rate=0.04)
    assert l == pytest.approx(15_000)
    assert F.ltv_cac_ratio(l, c) == pytest.approx(7.5)
    assert F.payback_period_months(c, arpu=1000, gross_margin=0.6) == pytest.approx(2000 / 600)


def test_zero_churn_ltv_undefined():
    with pytest.raises(FinanceMathError, match="churn_rate"):
        F.ltv(1000, 0.6, 0)


def test_negative_margin_ltv_and_payback():
    assert F.ltv(1000, -0.1, 0.05) == pytest.approx(-2000)
    with pytest.raises(FinanceMathError, match="never paid back"):
        F.payback_period_months(2000, 1000, -0.1)
    with pytest.raises(FinanceMathError):
        F.payback_period_months(2000, 1000, 0.0)


def test_cac_zero_customers_raises():
    with pytest.raises(FinanceMathError):
        F.cac(100_000, 0)


def test_ltv_cac_zero_cac_raises():
    with pytest.raises(FinanceMathError):
        F.ltv_cac_ratio(10_000, 0)


def test_break_even():
    assert F.break_even_units(100_000, price=50, variable_cost_per_unit=30) == 5000
    assert F.break_even_units_whole(100_001, 50, 30) == 5001
    assert F.break_even_units_whole(100_000, 50, 30) == 5000
    assert F.break_even_revenue(100_000, 50, 30) == 250_000


@pytest.mark.parametrize("price,vc", [(30, 30), (25, 30)])
def test_break_even_price_not_above_variable_cost(price, vc):
    with pytest.raises(FinanceMathError, match="break-even is impossible"):
        F.break_even_units(100_000, price, vc)


def test_burn_multiple_and_magic_number():
    assert F.burn_multiple(300_000, 200_000) == pytest.approx(1.5)
    with pytest.raises(FinanceMathError):
        F.burn_multiple(300_000, 0)
    assert F.magic_number(1_200_000, 1_000_000, 800_000) == pytest.approx(1.0)
    with pytest.raises(FinanceMathError):
        F.magic_number(1, 1, 0)


def test_generic_helpers():
    assert F.total([1, 2, 3.5]) == 6.5
    assert F.difference(10, 4) == 6
    assert F.ratio(1, 4) == 0.25
    with pytest.raises(FinanceMathError):
        F.ratio(1, 0)
    assert math.isclose(F.total([]), 0.0)
    assert F.percent_of(25, 200) == 0.125
    with pytest.raises(FinanceMathError):
        F.percent_of(1, 0)
