"""Deterministic in-memory fixtures.

These construct chains analytically so the ENGINE's logic can be tested without any
data file, per the build prompt ("the core must be testable without any data files").
They are test scaffolding only -- no backtest result is ever produced from them.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from putspread.chain import ChainProvider, ExpiryChain
from putspread.earnings import EarningsCalendar
from putspread.pricing import bs_call, bs_put
from putspread.spread import LegQuote

R = 0.043


def skewed_iv(strike: float, spot: float, atm_iv: float, skew_per_pct: float = 0.004) -> float:
    """IV rising as strikes fall -- the real shape of an equity put wing."""
    pct_otm = max(spot - strike, 0.0) / spot
    return atm_iv + skew_per_pct * pct_otm * 100.0


def linear_skew_iv(strike: float, anchors: tuple[tuple[float, float], tuple[float, float]]) -> float:
    """IV linear in strike through two quoted anchor points.

    This is exactly the section 7 approximation: "skew between the two quoted strikes
    is extrapolated linearly". Real skew curves; this does not, and the spec says so.
    """
    (k1, v1), (k2, v2) = anchors
    return v1 + (strike - k1) * (v2 - v1) / (k2 - k1)


def make_chain(
    symbol: str, as_of: date, expiration: date, spot: float, atm_iv: float = 0.89,
    strike_step: float = 5.0, n_strikes: int = 40, spread_pct: float = 0.04,
    skew_per_pct: float = 0.004,
    iv_anchors: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> ExpiryChain:
    """A synthetic but internally consistent board: BS prices plus a bid/ask band."""
    T = max((expiration - as_of).days, 0) / 365.0
    lo = spot - strike_step * n_strikes / 2
    puts, calls = {}, {}
    for i in range(n_strikes):
        k = round((lo + i * strike_step) / strike_step) * strike_step
        if k <= 0:
            continue
        iv = (linear_skew_iv(k, iv_anchors) if iv_anchors
              else skewed_iv(k, spot, atm_iv, skew_per_pct))
        pm = bs_put(spot, k, T, R, iv)
        cm = bs_call(spot, k, T, R, iv)
        half_p, half_c = max(pm * spread_pct / 2, 0.01), max(cm * spread_pct / 2, 0.01)
        puts[k] = LegQuote(strike=k, bid=round(max(pm - half_p, 0.0), 2),
                           ask=round(pm + half_p, 2), iv=iv, open_interest=500, volume=100)
        calls[k] = LegQuote(strike=k, bid=round(max(cm - half_c, 0.0), 2),
                            ask=round(cm + half_c, 2), iv=iv, open_interest=500, volume=100)
    return ExpiryChain(symbol=symbol, as_of=as_of, expiration=expiration, puts=puts, calls=calls)


class FixtureChainProvider(ChainProvider):
    """Chains generated on demand from a supplied daily price path."""

    def __init__(self, symbol: str, path: pd.Series, atm_iv: float = 0.60,
                 expiry_every: int = 7, strike_step: float = 5.0):
        self.symbol = symbol
        self.path = path
        self.atm_iv = atm_iv
        self.strike_step = strike_step
        # Weekly expirations on Fridays, out to ~90 days.
        first = min(path.index)
        last = max(path.index) + timedelta(days=120)
        d = first
        exps = []
        while d <= last:
            if d.weekday() == 4:
                exps.append(d)
            d += timedelta(days=1)
        self._expirations = exps

    def trading_dates(self, symbol, start, end):
        return [d for d in self.path.index if start <= d <= end]

    def expirations(self, symbol, as_of):
        return [e for e in self._expirations if e > as_of and (e - as_of).days <= 120]

    def chain(self, symbol, as_of, expiration):
        if as_of not in self.path.index or expiration <= as_of:
            return None
        return make_chain(symbol, as_of, expiration, float(self.path.loc[as_of]),
                          atm_iv=self.atm_iv, strike_step=self.strike_step)

    def spot(self, symbol, as_of):
        return float(self.path.loc[as_of]) if as_of in self.path.index else None


@pytest.fixture
def amd_worked_example_chain() -> ExpiryChain:
    """STRATEGY.md section 11, priced at the low target where the trigger fires."""
    as_of = date(2026, 7, 23)
    # The document's two quoted IVs: 89% at the 460 short, 91% at the 435 long.
    return make_chain(
        "AMD", as_of, as_of + timedelta(days=22), spot=475.0,
        strike_step=5.0, n_strikes=40, iv_anchors=((460.0, 0.89), (435.0, 0.91)),
    )


@pytest.fixture
def amd_earnings_calendar() -> EarningsCalendar:
    """The real AMD report date from the worked example: 2026-08-04, after the close."""
    return EarningsCalendar(pd.DataFrame([
        {"symbol": "AMD", "date": date(2026, 5, 5), "when": "After market close"},
        {"symbol": "AMD", "date": date(2026, 8, 4), "when": "After market close"},
        {"symbol": "AMD", "date": date(2026, 11, 3), "when": "After market close"},
    ]))
