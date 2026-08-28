"""Slices 3 and 4 -- entry triggers, the historical P&L path, exits, and sizing."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from putspread.config import StrategyConfig, SupportConfig
from putspread.earnings import EarningsCalendar, EarningsDataMissing
from putspread.engine import Backtester
from putspread.exits import MarkState, evaluate_exits
from putspread.fills import FillConfig
from putspread.portfolio import OpenPosition, Portfolio
from putspread.rates import RateCurve
from putspread.support import confirmed_pivot_lows, entry_signals, support_series
from tests.conftest import FixtureChainProvider


def bars_from(prices: list[float], start: date = date(2024, 1, 1)) -> pd.DataFrame:
    """Daily bars on consecutive weekdays; high/low straddle the close by 1%."""
    idx, d = [], start
    while len(idx) < len(prices):
        if d.weekday() < 5:
            idx.append(d)
        d += timedelta(days=1)
    c = np.array(prices, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c * 1.01, "low": c * 0.99, "close": c}, index=pd.Index(idx, name="date")
    )


# ------------------------------------------------------- support, no lookahead


def test_pivot_low_confirmation_lag_is_respected():
    """A pivot is only knowable `right` bars after it prints. If this test ever fails
    the backtest is placing strikes under a low it could not have seen."""
    lows = np.array([10, 9, 8, 7, 5, 7, 8, 9, 10, 11], dtype=float)
    piv = confirmed_pivot_lows(lows, left=2, right=2)
    assert piv.tolist() == [[4, 6]]        # the low at index 4, confirmed at index 6


def test_support_series_never_uses_future_bars():
    """Truncating the history must not change any level already published."""
    prices = [100, 98, 95, 92, 90, 93, 96, 99, 102, 105, 103, 100, 97, 94, 91, 95, 98]
    bars = bars_from(prices)
    cfg = SupportConfig(pivot_left=2, pivot_right=2, min_distance_pct=0.0)
    full = support_series(bars, cfg)
    for cut in range(6, len(bars)):
        partial = support_series(bars.iloc[:cut], cfg)
        pd.testing.assert_series_equal(
            partial.dropna(), full.iloc[:cut].dropna(), check_freq=False
        )


def test_mechanical_entry_fires_on_a_fresh_touch_only():
    bars = bars_from([100, 99, 98, 97, 96, 100, 104, 102, 99, 96.5, 96.4, 99])
    support = pd.Series(96.0, index=bars.index)
    sig = entry_signals(bars, support, "mechanical")
    fired = [i for i, v in enumerate(sig) if v]
    # low = close*0.99, so a close near 97 dips under 96. Each fire needs the PRIOR
    # close above the level, which is what makes it a fresh pullback.
    assert fired
    for i in fired:
        assert bars["low"].iloc[i] <= 96.0 < bars["close"].iloc[i - 1]


def test_confirmation_entry_declines_a_failing_level():
    """Price touches support and keeps going: mechanical enters, confirmation does not."""
    bars = bars_from([100, 99, 97, 93, 89, 85, 81])
    support = pd.Series(97.0, index=bars.index)
    mech = entry_signals(bars, support, "mechanical")
    conf = entry_signals(bars, support, "confirmation", confirmation_bars=2)
    assert mech.sum() >= 1
    assert conf.sum() == 0


def test_confirmation_entry_takes_a_level_that_holds():
    bars = bars_from([100, 99, 97.5, 99.5, 101, 103])
    support = pd.Series(97.0, index=bars.index)
    conf = entry_signals(bars, support, "confirmation", confirmation_bars=2)
    assert conf.sum() >= 1


# ------------------------------------------------------- exits


def pos(**kw) -> OpenPosition:
    d = dict(symbol="X", entry_date=date(2024, 1, 2), expiration=date(2024, 1, 26),
             short_strike=95.0, long_strike=90.0, credit_per_share=1.50, contracts=1,
             max_loss_per_contract=350.0, entry_spot=100.0, support_level=96.0,
             short_iv=0.5, long_iv=0.52, entry_commission=1.30, entry_delta=10.0,
             entry_vega=-2.0, entry_theta=1.0, move_sigma=0.5, prob_otm_display=0.8)
    d.update(kw)
    return OpenPosition(**d)


def mark(**kw) -> MarkState:
    d = dict(spot=100.0, debit_to_close=1.0, short_mid=0.6, days_to_expiry=10, quotes_are_real=True)
    d.update(kw)
    return MarkState(**d)


def test_profit_target_triggers_at_the_configured_capture():
    cfg = StrategyConfig(profit_target_pct=0.50, require_earnings_data=False)
    assert evaluate_exits(pos(), mark(debit_to_close=0.70), cfg) == "profit_target"
    assert evaluate_exits(pos(), mark(debit_to_close=0.80), cfg) is None   # only 47% captured


def test_level_break_stop_fires_below_the_support_margin():
    cfg = StrategyConfig(stop_rule="level_break", stop_level_break_pct=0.01,
                         require_earnings_data=False)
    itm = mark(spot=94.0, short_mid=2.10, debit_to_close=2.9)   # intrinsic 1.00, real time value
    assert evaluate_exits(pos(), itm, cfg) == "level_break"                # 96 * 0.99 = 95.04
    assert evaluate_exits(pos(), mark(spot=95.5), cfg) is None


def test_credit_multiple_stop_fires_at_the_multiple():
    cfg = StrategyConfig(stop_rule="credit_multiple", stop_credit_multiple=2.0,
                         require_earnings_data=False)
    assert evaluate_exits(pos(), mark(debit_to_close=4.5), cfg) == "credit_multiple_stop"
    assert evaluate_exits(pos(), mark(debit_to_close=4.0), cfg) is None


def test_time_stop_fires_at_the_dte():
    cfg = StrategyConfig(time_stop_dte=14, require_earnings_data=False)
    assert evaluate_exits(pos(), mark(days_to_expiry=14), cfg) == "time_stop"
    assert evaluate_exits(pos(), mark(days_to_expiry=15), cfg) is None


def test_early_assignment_when_the_short_has_no_time_value_left():
    cfg = StrategyConfig(require_earnings_data=False, model_early_assignment=True)
    deep = mark(spot=88.0, short_mid=7.02, days_to_expiry=5)   # intrinsic 7.00, tv 0.02
    assert evaluate_exits(pos(), deep, cfg) == "early_assignment"
    fat = mark(spot=88.0, short_mid=8.20, days_to_expiry=5)    # tv 1.20, no reason to exercise
    assert evaluate_exits(pos(), fat, cfg) is None


def test_early_assignment_is_never_inferred_from_a_model_mark():
    """Assignment is a market fact; a Black-Scholes mark is not evidence of one."""
    cfg = StrategyConfig(require_earnings_data=False, model_early_assignment=True)
    modeled = mark(spot=88.0, short_mid=None, days_to_expiry=5, quotes_are_real=False)
    assert evaluate_exits(pos(), modeled, cfg) is None


def test_expiry_outranks_every_optional_rule():
    cfg = StrategyConfig(profit_target_pct=0.50, stop_rule="level_break",
                         time_stop_dte=21, require_earnings_data=False)
    assert evaluate_exits(pos(), mark(days_to_expiry=0, spot=50.0), cfg) == "expiry"


# ------------------------------------------------------- sizing


def test_position_size_respects_both_caps():
    cfg = StrategyConfig(starting_equity=100_000, max_risk_per_position_pct=0.02,
                         max_portfolio_risk_pct=0.02, require_earnings_data=False)
    p = Portfolio(cfg)
    assert p.size_position(500.0, 100_000) == 4          # $2,000 budget / $500
    p.positions.append(pos(max_loss_per_contract=1500.0, contracts=1))
    assert p.size_position(500.0, 100_000) == 1          # $2,000 - $1,500 open = $500
    p.positions.append(pos(max_loss_per_contract=500.0, contracts=1))
    assert p.size_position(500.0, 100_000) == 0          # budget exhausted -> skip


def test_portfolio_cap_binds_across_symbols():
    cfg = StrategyConfig(starting_equity=50_000, max_risk_per_position_pct=0.10,
                         max_portfolio_risk_pct=0.04, require_earnings_data=False)
    p = Portfolio(cfg)
    assert p.size_position(1000.0, 50_000) == 2          # portfolio cap $2,000 binds first


def test_cash_and_pnl_reconcile_on_a_round_trip():
    cfg = StrategyConfig(starting_equity=100_000, require_earnings_data=False)
    p = Portfolio(cfg)
    position = pos(contracts=3, credit_per_share=1.50, entry_commission=3.90)
    p.open_position(position)
    assert p.cash == pytest.approx(100_000 + 450.0 - 3.90)
    t = p.close_position(position, date(2024, 1, 20), 0.40, 97.0, "profit_target", 3.90)
    assert t.pnl == pytest.approx((1.50 - 0.40) * 100 * 3 - 3.90 - 3.90)
    assert p.cash == pytest.approx(100_000 + t.pnl)


# ------------------------------------------------------- engine integration


def calendar_for(symbol: str, lo: date, hi: date) -> EarningsCalendar:
    return EarningsCalendar(pd.DataFrame([
        {"symbol": symbol, "date": lo, "when": "After market close"},
        {"symbol": symbol, "date": hi, "when": "After market close"},
    ]))


def run_engine(prices: list[float], **cfg_kw):
    bars = bars_from(prices)
    path = bars["close"]
    provider = FixtureChainProvider("X", path, atm_iv=0.45, strike_step=1.0)
    cfg = StrategyConfig(
        symbols=("X",), start=str(bars.index[0]), end=str(bars.index[-1] + timedelta(days=60)),
        target_dte=22, dte_tolerance=10, max_loss_per_contract=1000.0,
        require_earnings_data=False, acknowledge_missing_oi_volume=True,
        max_spread_pct_of_mid=0.50, min_bid=0.01,
        support=SupportConfig(pivot_left=2, pivot_right=2, min_distance_pct=0.0),
        **cfg_kw,
    )
    bt = Backtester(provider, {"X": bars}, cfg, FillConfig(model="mid"),
                    RateCurve.constant(0.043), None)
    return bt.run()


def test_engine_opens_and_settles_a_winning_trade():
    """Price dips to support then recovers well above the short strike -> full credit."""
    prices = [100, 98, 95, 92, 90, 93, 96, 99, 102, 105, 103, 100, 96, 92, 89.5, 95, 99, 103]
    prices += [108] * 40
    res = run_engine(prices)
    assert res.trades, f"no trades; rejections: {res.rejections['reason'].tolist()[:5]}"
    t = res.trades[0]
    assert t.exit_reason == "expiry"
    assert t.pnl > 0
    assert t.pnl == pytest.approx(t.credit_per_share * 100 * t.contracts - t.commissions, abs=1e-6)


def test_engine_books_a_capped_loss_through_a_crash_gap():
    """A gap straight through both strikes must lose max loss, never more.

    This is the test that proves the path is real: a Black-Scholes-simulated path
    cannot produce a 35% overnight gap, and the whole reason section 7 forbids
    simulating the path is that it would understate exactly this outcome.
    """
    prices = [100, 98, 95, 92, 90, 93, 96, 99, 102, 105, 103, 100, 96, 92, 89.5]
    prices += [60] * 40                     # the gap
    res = run_engine(prices)
    assert res.trades
    t = res.trades[0]
    assert t.pnl == pytest.approx(-t.max_loss_per_contract * t.contracts, abs=1.0 + t.commissions)
    assert t.pnl >= -(t.max_loss_per_contract * t.contracts) - t.commissions - 1e-6


def test_engine_respects_one_position_per_symbol():
    prices = [100, 98, 95, 92, 90, 93, 96, 99, 102, 105, 103, 100, 96, 92, 89.5, 95, 99, 103] * 3
    res = run_engine(prices)
    entries = [(t.entry_date, t.exit_date) for t in res.trades]
    for i, (e1, x1) in enumerate(entries):
        for e2, _ in entries[i + 1:]:
            assert not (e1 < e2 < x1), "overlapping positions on one symbol"


def test_engine_refuses_to_run_without_earnings_coverage():
    bars = bars_from([100.0] * 30)
    cfg = StrategyConfig(symbols=("X",), start=str(bars.index[0]), end=str(bars.index[-1]),
                         require_earnings_data=True)
    bt = Backtester(FixtureChainProvider("X", bars["close"]), {"X": bars}, cfg,
                    FillConfig(), RateCurve.constant(0.04),
                    calendar_for("X", date(2030, 1, 1), date(2030, 4, 1)))
    with pytest.raises(EarningsDataMissing):
        bt.run()


def test_engine_is_deterministic():
    prices = [100, 98, 95, 92, 90, 93, 96, 99, 102, 105, 103, 100, 96, 92, 89.5, 95, 99] + [105] * 40
    a, b = run_engine(prices), run_engine(prices)
    assert [t.pnl for t in a.trades] == [t.pnl for t in b.trades]


def test_realistic_fills_never_beat_mid_fills():
    prices = [100, 98, 95, 92, 90, 93, 96, 99, 102, 105, 103, 100, 96, 92, 89.5, 95, 99] + [105] * 40
    bars = bars_from(prices)
    provider = FixtureChainProvider("X", bars["close"], atm_iv=0.45, strike_step=1.0)
    cfg = StrategyConfig(
        symbols=("X",), start=str(bars.index[0]), end=str(bars.index[-1] + timedelta(days=60)),
        target_dte=22, dte_tolerance=10, max_loss_per_contract=1000.0,
        require_earnings_data=False, acknowledge_missing_oi_volume=True,
        max_spread_pct_of_mid=0.50, min_bid=0.01,
        support=SupportConfig(pivot_left=2, pivot_right=2, min_distance_pct=0.0),
    )
    out = {}
    for model in ("mid", "realistic", "natural"):
        bt = Backtester(provider, {"X": bars}, cfg, FillConfig(model=model),
                        RateCurve.constant(0.043), None)
        r = bt.run()
        out[model] = sum(t.pnl for t in r.trades)
    assert out["mid"] >= out["realistic"] >= out["natural"]


def test_confirmation_and_mechanical_are_actually_different():
    """Guard against the disciplines collapsing into each other.

    If support were defined relative to TODAY's close it would always sit below it,
    no bar could ever close under its own support, and the section 3.2 A/B test would
    silently compare a rule against itself. This asserts they diverge on a path where
    the level breaks.
    """
    # A pivot low forms at 92, price rallies away, then returns and breaks through.
    bars = bars_from([100, 96, 92, 95, 99, 103, 100, 96, 91, 87, 84])
    cfg = SupportConfig(pivot_left=1, pivot_right=1, min_distance_pct=0.0)
    sup = support_series(bars, cfg)
    mech = entry_signals(bars, sup, "mechanical")
    conf = entry_signals(bars, sup, "confirmation", confirmation_bars=1)
    assert mech.sum() > 0
    assert not mech.equals(conf), "confirmation must not be identical to mechanical"
    assert conf.sum() <= mech.sum()


def test_support_level_is_known_before_the_bar_that_touches_it():
    """The level in force on day t must be derivable from bars up to t-1 only."""
    prices = [100, 98, 95, 92, 90, 93, 96, 99, 102, 105, 103, 100, 96, 92, 89.5]
    bars = bars_from(prices)
    cfg = SupportConfig(pivot_left=2, pivot_right=2, min_distance_pct=0.0)
    full = support_series(bars, cfg)
    for t in range(4, len(bars)):
        upto = support_series(bars.iloc[:t], cfg)
        if not np.isnan(full.iloc[t - 1]):
            assert upto.iloc[t - 1] == pytest.approx(full.iloc[t - 1])
