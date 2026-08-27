"""Slice 1 validation: spread construction, credit, max loss, payoff, net Greeks."""

import pytest

from putspread.pricing import bs_put
from putspread.spread import (
    BullPutSpread,
    LegQuote,
    build_spread,
    payoff_at_expiry_per_contract,
    spread_mark_per_contract,
)


def make_amd_spread(credit=10.0) -> BullPutSpread:
    """The STRATEGY.md section 11 worked example, priced by hand."""
    short = LegQuote(strike=460.0, bid=24.5, ask=25.5, iv=0.89)
    long = LegQuote(strike=435.0, bid=14.7, ask=15.7, iv=0.91)
    return build_spread(
        underlying="AMD", spot=475.0, short=short, long=long,
        T=22 / 365, expiration_days=22, r=0.043,
        short_iv=0.89, long_iv=0.91, credit_per_share=credit,
    )


def test_worked_example_geometry():
    s = make_amd_spread()
    assert s.width == 25.0
    assert s.max_profit_per_contract == pytest.approx(1000.0)
    assert s.max_loss_per_contract == pytest.approx(1500.0)
    assert s.breakeven == pytest.approx(450.0)
    assert s.credit_to_width == pytest.approx(0.4)
    assert s.return_on_risk == pytest.approx(1000.0 / 1500.0)


def test_max_loss_respects_the_budget():
    """STRATEGY.md section 4: risk per contract is (width - credit) x 100."""
    s = make_amd_spread(credit=10.14)
    assert s.max_loss_per_contract == pytest.approx(1486.0)


def test_payoff_three_regimes():
    short_k, long_k, credit = 460.0, 435.0, 10.0
    # Above the short strike -> full credit.
    assert payoff_at_expiry_per_contract(500.0, short_k, long_k, credit) == pytest.approx(1000.0)
    assert payoff_at_expiry_per_contract(460.0, short_k, long_k, credit) == pytest.approx(1000.0)
    # Below the long strike -> max loss.
    assert payoff_at_expiry_per_contract(400.0, short_k, long_k, credit) == pytest.approx(-1500.0)
    assert payoff_at_expiry_per_contract(0.0, short_k, long_k, credit) == pytest.approx(-1500.0)
    # Between -> linear, and zero exactly at breakeven.
    assert payoff_at_expiry_per_contract(450.0, short_k, long_k, credit) == pytest.approx(0.0)
    assert payoff_at_expiry_per_contract(455.0, short_k, long_k, credit) == pytest.approx(500.0)


def test_payoff_is_monotone_and_bounded():
    prev = -1e18
    for spot in range(380, 520, 2):
        v = payoff_at_expiry_per_contract(float(spot), 460.0, 435.0, 10.0)
        assert -1500.0 - 1e-9 <= v <= 1000.0 + 1e-9
        assert v >= prev - 1e-9
        prev = v


def test_net_greeks_are_short_vega_positive_theta():
    """The entire thesis of section 2: short vega, positive theta, positive delta."""
    s = make_amd_spread()
    g = s.net_greeks
    assert g.delta > 0.0, "bull put spread is long the underlying"
    assert g.vega < 0.0, "credit spread must be SHORT vega -- IV crush helps"
    assert g.theta > 0.0, "credit spread must collect theta"


def test_credit_from_modeled_legs_is_positive():
    S, T, r = 475.0, 22 / 365, 0.043
    short_val = bs_put(S, 460.0, T, r, 0.89)
    long_val = bs_put(S, 435.0, T, r, 0.91)
    assert short_val - long_val > 0.0


def test_skew_direction_reduces_credit():
    """Section 7: a flat IV overstates credit. Higher long-leg IV must cost credit."""
    S, T, r = 475.0, 22 / 365, 0.043
    flat = bs_put(S, 460.0, T, r, 0.89) - bs_put(S, 435.0, T, r, 0.89)
    skewed = bs_put(S, 460.0, T, r, 0.89) - bs_put(S, 435.0, T, r, 0.91)
    assert skewed < flat


def test_inverted_strikes_rejected():
    with pytest.raises(ValueError, match="short strike above long strike"):
        build_spread(
            underlying="X", spot=100.0,
            short=LegQuote(strike=90.0, bid=1.0, ask=1.1),
            long=LegQuote(strike=95.0, bid=2.0, ask=2.1),
            T=0.1, expiration_days=36, r=0.04,
            short_iv=0.3, long_iv=0.32, credit_per_share=-1.0,
        )


def test_mark_decays_toward_intrinsic_as_time_passes():
    """Mark-to-market shrinks as T -> 0 when the spread is safely OTM."""
    marks = [
        spread_mark_per_contract(460.0, 435.0, 500.0, T, 0.043, 0.5, 0.52)
        for T in (30 / 365, 14 / 365, 3 / 365, 0.5 / 365)
    ]
    assert marks == sorted(marks, reverse=True)
    assert marks[-1] < 15.0


def test_leg_quote_liquidity_helpers():
    q = LegQuote(strike=100.0, bid=1.0, ask=1.2)
    assert q.mid == pytest.approx(1.1)
    assert q.spread_abs == pytest.approx(0.2)
    assert q.spread_pct_of_mid == pytest.approx(0.2 / 1.1)
    assert LegQuote(strike=100.0, bid=0.0, ask=0.0).spread_pct_of_mid == float("inf")
