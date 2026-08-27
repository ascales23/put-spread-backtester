"""Slice 2 -- single-trade evaluator, including the STRATEGY.md worked example."""

from datetime import date, timedelta

import pytest

from putspread.config import StrategyConfig
from putspread.evaluate import evaluate_candidate
from putspread.filters import FilterConfigurationError
from putspread.fills import FillConfig
from putspread.selection import SelectionError, select_long_strike, select_short_strike

R = 0.043
MID = FillConfig(model="mid")


def base_cfg(**kw) -> StrategyConfig:
    d = dict(symbols=("AMD",), max_loss_per_contract=1510.0, buffer_pct=0.03,
             require_earnings_data=False, acknowledge_missing_oi_volume=False,
             max_spread_pct_of_mid=0.12)
    d.update(kw)
    return StrategyConfig(**d)


# ------------------------------------------------------- the worked example


def test_worked_example_strikes_and_credit(amd_worked_example_chain):
    """STRATEGY.md section 11: buffer 3% off a 475 low target -> short 460; a
    ~$1,500 budget -> long 435, width 25, credit ~$10, max loss ~$1,500.

    Priced AT the low target, which is where section 3.1's trigger fires and what
    reproduces the reference tool's numbers. Our credit lands at $9.9 against the
    document's $10.14; the gap is the reference tool's own two-pass IV solve off
    live quotes, and is well inside a one-tick bid/ask.
    """
    chain = amd_worked_example_chain
    cfg = base_cfg()
    cand = evaluate_candidate(
        "AMD", chain.as_of, spot=475.0, low_target=475.0, chain=chain,
        cfg=cfg, fills=MID, r=R, calendar=None,
    )
    assert cand.accepted, cand.reject_reason
    s = cand.spread
    assert s.short.strike == 460.0
    assert s.long.strike == 435.0
    assert s.width == 25.0
    assert s.credit_per_share == pytest.approx(10.0, abs=0.5)
    assert s.max_loss_per_contract == pytest.approx(1500.0, abs=50.0)
    assert s.breakeven == pytest.approx(450.0, abs=0.5)


def test_worked_example_is_blocked_by_earnings(amd_worked_example_chain, amd_earnings_calendar):
    """The canonical case: AMD reported 2026-08-04, inside the expiration. Section 6.1
    makes this a hard block, and the 89% IV was event premium, not edge."""
    chain = amd_worked_example_chain
    assert chain.as_of < date(2026, 8, 4) < chain.expiration
    cand = evaluate_candidate(
        "AMD", chain.as_of, spot=475.0, low_target=475.0, chain=chain,
        cfg=base_cfg(require_earnings_data=True), fills=MID, r=R,
        calendar=amd_earnings_calendar,
    )
    assert not cand.accepted
    assert "earnings 2026-08-04" in cand.reject_reason


def test_earnings_filter_cannot_be_silently_skipped(amd_worked_example_chain):
    """No calendar + require_earnings_data must raise, never quietly trade."""
    with pytest.raises(FilterConfigurationError, match="mandatory"):
        evaluate_candidate(
            "AMD", amd_worked_example_chain.as_of, 475.0, 475.0, amd_worked_example_chain,
            base_cfg(require_earnings_data=True), MID, R, calendar=None,
        )


def test_earnings_after_close_on_expiry_day_does_not_block(amd_earnings_calendar):
    """Options settle at the close; a report after that close cannot touch them."""
    cal = amd_earnings_calendar
    assert cal.earnings_between("AMD", date(2026, 7, 23), date(2026, 8, 4)) == []
    assert cal.earnings_between("AMD", date(2026, 7, 23), date(2026, 8, 5)) == [date(2026, 8, 4)]


# ------------------------------------------------------- strike selection


def test_buffer_method_snaps_below_the_target(amd_worked_example_chain):
    k = select_short_strike(amd_worked_example_chain, 475.0, 475.0, R, base_cfg(buffer_pct=0.03))
    assert k == 460.0                       # 475 * 0.97 = 460.75, snapped down
    k2 = select_short_strike(amd_worked_example_chain, 475.0, 475.0, R, base_cfg(buffer_pct=0.10))
    assert k2 == 425.0                      # 475 * 0.90 = 427.5, snapped down


def test_delta_method_tracks_the_target_delta(amd_worked_example_chain):
    from putspread.chain import solve_leg_iv
    from putspread.pricing import bs_greeks

    chain = amd_worked_example_chain
    k = select_short_strike(chain, 475.0, 475.0, R, base_cfg(short_strike_method="delta",
                                                             target_short_delta=0.20))
    iv = solve_leg_iv(chain.puts[k], 475.0, chain.T, R)
    d = bs_greeks(475.0, k, chain.T, R, iv, "put").delta
    assert abs(abs(d) - 0.20) < 0.03


def test_delta_method_widens_the_strike_when_vol_is_higher():
    """The point of delta selection (section 3.3): it scales with vol, a % buffer does not."""
    from tests.conftest import make_chain
    as_of = date(2026, 7, 23)
    cfg = base_cfg(short_strike_method="delta", target_short_delta=0.20)
    low_vol = make_chain("X", as_of, as_of + timedelta(days=22), 475.0, atm_iv=0.30, skew_per_pct=0.0)
    high_vol = make_chain("X", as_of, as_of + timedelta(days=22), 475.0, atm_iv=0.90, skew_per_pct=0.0)
    k_low = select_short_strike(low_vol, 475.0, 475.0, R, cfg)
    k_high = select_short_strike(high_vol, 475.0, 475.0, R, cfg)
    assert k_high < k_low


def test_long_strike_is_the_widest_that_fits_the_budget(amd_worked_example_chain):
    chain = amd_worked_example_chain
    tight = select_long_strike(chain, 475.0, 460.0, R, base_cfg(max_loss_per_contract=800.0), MID)
    loose = select_long_strike(chain, 475.0, 460.0, R, base_cfg(max_loss_per_contract=1510.0), MID)
    assert tight.long.strike > loose.long.strike        # smaller budget -> narrower
    assert tight.max_loss_per_contract <= 800.0
    assert loose.max_loss_per_contract <= 1510.0


def test_budget_too_small_for_any_width_is_rejected(amd_worked_example_chain):
    with pytest.raises(SelectionError, match="max-loss budget"):
        select_long_strike(amd_worked_example_chain, 475.0, 460.0, R,
                           base_cfg(max_loss_per_contract=10.0), MID)


# ------------------------------------------------------- fills and filters


def test_realistic_fills_reduce_the_credit_versus_mid(amd_worked_example_chain):
    """Section 7: paying the spread must cost credit. If it did not, the fill model
    would be decorative."""
    chain = amd_worked_example_chain
    cfg = base_cfg()
    mid = evaluate_candidate("AMD", chain.as_of, 475.0, 475.0, chain, cfg, MID, R)
    real = evaluate_candidate("AMD", chain.as_of, 475.0, 475.0, chain, cfg,
                              FillConfig(model="realistic", fraction=0.5), R)
    nat = evaluate_candidate("AMD", chain.as_of, 475.0, 475.0, chain, cfg,
                             FillConfig(model="natural"), R)
    assert mid.spread.credit_per_share > real.spread.credit_per_share > nat.spread.credit_per_share


def test_missing_oi_and_volume_must_be_acknowledged(amd_worked_example_chain):
    """A filter you cannot evaluate must not silently pass."""
    from putspread.spread import LegQuote
    chain = amd_worked_example_chain
    stripped = type(chain)(
        symbol=chain.symbol, as_of=chain.as_of, expiration=chain.expiration,
        puts={k: LegQuote(strike=q.strike, bid=q.bid, ask=q.ask, iv=q.iv) for k, q in chain.puts.items()},
        calls=chain.calls,
    )
    with pytest.raises(FilterConfigurationError, match="open interest"):
        evaluate_candidate("AMD", chain.as_of, 475.0, 475.0, stripped, base_cfg(), MID, R)
    ok = evaluate_candidate("AMD", chain.as_of, 475.0, 475.0, stripped,
                            base_cfg(acknowledge_missing_oi_volume=True), MID, R)
    assert ok.accepted


def test_wide_markets_are_filtered_out(amd_worked_example_chain):
    from tests.conftest import make_chain
    wide = make_chain("AMD", amd_worked_example_chain.as_of,
                      amd_worked_example_chain.expiration, 475.0, spread_pct=0.60)
    cand = evaluate_candidate("AMD", wide.as_of, 475.0, 475.0, wide, base_cfg(), MID, R)
    assert not cand.accepted
    assert "bid/ask" in cand.reject_reason


def test_prob_otm_is_recorded_as_display_only(amd_worked_example_chain):
    cand = evaluate_candidate("AMD", amd_worked_example_chain.as_of, 475.0, 475.0,
                              amd_worked_example_chain, base_cfg(), MID, R)
    key = "prob_otm_risk_neutral_display"
    assert key in cand.diagnostics
    assert 0.0 < cand.diagnostics[key] < 1.0
