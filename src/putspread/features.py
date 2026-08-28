"""Entry-time features for the learned trade filter.

Every feature here must be computable at the moment the trade is opened, from data
that existed at that moment. Anything else is lookahead, and in a study this small a
single leaked feature would manufacture a spectacular and entirely fake edge.

Feature families, and why each is here rather than being decoration:
  * volatility state -- STRATEGY.md section 2 argues the entry belongs in ELEVATED IV
    but nothing in the rule set ever tests that. IV rank and IV-minus-realized are the
    direct test of the spec's own central claim.
  * spread economics -- credit-to-width decides the breakeven win rate, and the
    backtest found the median trade collecting only 13.8% of width.
  * trend and location -- where price sits relative to its own recent range, which is
    what "support should hold" is implicitly a claim about.
  * market regime -- VIX level and change, so the model can separate a name-specific
    pullback from a market-wide one.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .chain import ChainProvider

#: Column order is fixed so a model trained in one process scores identically in
#: another. Adding a feature means retraining, never silently reordering.
FEATURE_COLUMNS = [
    "dte", "width_pct_spot", "credit_to_width", "return_on_risk",
    "moneyness_short", "moneyness_long", "short_iv", "iv_skew_slope",
    "iv_rank_252", "iv_pct_63", "iv_minus_rv20", "rv20", "rv60", "rv_ratio",
    "term_slope", "ret5", "ret20", "ret60", "dist_sma50", "drawdown_60",
    "support_dist", "support_break_depth", "move_sigma",
    "spread_pct_short", "spread_pct_long",
    "vix_level", "vix_chg5", "days_to_next_earnings",
    # The trade STRUCTURE is a feature, not a fixed constant. Harvesting every
    # candidate under many structures and letting the model see which one it is
    # turns "pick one parameter set for all time" into "pick the structure that
    # suits this particular setup" -- which is the thing the sweep failed to do.
    "cfg_buffer_pct", "cfg_target_dte", "cfg_target_delta", "cfg_is_delta_method",
]


def realized_vol(closes: pd.Series, window: int) -> float:
    """Annualized close-to-close realized volatility over `window` sessions."""
    if len(closes) < window + 1:
        return float("nan")
    r = np.diff(np.log(closes.to_numpy(dtype=float)[-(window + 1):]))
    return float(np.std(r, ddof=1) * np.sqrt(252.0))


def atm_iv_series(provider: ChainProvider, symbol: str, target_dte: int = 30) -> pd.Series:
    """Daily at-the-money implied vol for `symbol`, from the chain's own quotes.

    Uses the vendor's per-strike IV column rather than re-solving: this series is a
    regime feature, not a pricing input, and the data-quality run showed vendor IV
    agreeing with our solver to 2 basis points.
    """
    out: dict[date, float] = {}
    for d in provider.trading_dates(symbol, date.min, date.max):
        spot = provider.spot(symbol, d)
        if spot is None:
            continue
        exps = [e for e in provider.expirations(symbol, d) if (e - d).days > 5]
        if not exps:
            continue
        exp = min(exps, key=lambda e: abs((e - d).days - target_dte))
        ch = provider.chain(symbol, d, exp)
        if ch is None or not ch.puts:
            continue
        k = min(ch.puts, key=lambda k: abs(k - spot))
        iv = ch.puts[k].iv
        if iv and iv > 0:
            out[d] = float(iv)
    return pd.Series(out, name="atm_iv").sort_index()


def term_slope(provider: ChainProvider, symbol: str, d: date, spot: float) -> float:
    """Far-dated ATM IV minus near-dated ATM IV, in vol points.

    A steep positive slope means the front is cheap relative to the back; an inverted
    term structure usually means the market is pricing a near-term event. Selling the
    front into an inversion is a different trade from selling it into contango.
    """
    exps = provider.expirations(symbol, d)
    near = [e for e in exps if 5 < (e - d).days <= 21]
    far = [e for e in exps if 45 <= (e - d).days <= 120]
    if not near or not far:
        return float("nan")

    def atm(exp: date) -> float:
        ch = provider.chain(symbol, d, exp)
        if ch is None or not ch.puts:
            return float("nan")
        k = min(ch.puts, key=lambda k: abs(k - spot))
        iv = ch.puts[k].iv
        return float(iv) if iv else float("nan")

    return atm(min(far, key=lambda e: (e - d).days)) - atm(max(near, key=lambda e: (e - d).days))


def load_vix(vix_dir: str | Path) -> pd.Series:
    """Daily VIX close, or an empty series when the directory is absent."""
    files = sorted(Path(vix_dir).glob("*.parquet"))
    if not files:
        return pd.Series(dtype=float)
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.date
    return df.groupby("date")["close"].last().sort_index()


def _as_of(series: pd.Series, d: date) -> float:
    """Most recent value at or before `d`. NaN if the series starts later."""
    if series.empty:
        return float("nan")
    idx = series.index[series.index <= d]
    return float(series.loc[idx[-1]]) if len(idx) else float("nan")


def build_features(
    *,
    symbol: str,
    d: date,
    spot: float,
    support_level: float,
    candidate_diagnostics: dict,
    short_strike: float,
    long_strike: float,
    credit_per_share: float,
    max_loss_per_contract: float,
    short_iv: float,
    long_iv: float,
    dte: int,
    spread_pct_short: float,
    spread_pct_long: float,
    bars: pd.DataFrame,
    atm_iv: pd.Series,
    vix: pd.Series,
    term: float,
    days_to_next_earnings: float,
    cfg_buffer_pct: float = float("nan"),
    cfg_target_dte: float = float("nan"),
    cfg_target_delta: float = float("nan"),
    cfg_is_delta_method: float = 0.0,
) -> dict[str, float]:
    """One feature row for a candidate spread, using only information dated <= d."""
    hist = bars.loc[bars.index <= d]
    closes = hist["close"]
    highs = hist["high"]
    close = float(closes.iloc[-1]) if len(closes) else spot

    iv_hist = atm_iv.loc[atm_iv.index <= d]
    iv_now = float(iv_hist.iloc[-1]) if len(iv_hist) else float("nan")
    tail252, tail63 = iv_hist.tail(252), iv_hist.tail(63)
    lo, hi = (tail252.min(), tail252.max()) if len(tail252) > 20 else (np.nan, np.nan)

    rv20, rv60 = realized_vol(closes, 20), realized_vol(closes, 60)
    width = short_strike - long_strike

    def ret(n: int) -> float:
        return float(close / closes.iloc[-(n + 1)] - 1.0) if len(closes) > n else float("nan")

    sma50 = float(closes.tail(50).mean()) if len(closes) >= 50 else float("nan")
    hi60 = float(highs.tail(60).max()) if len(highs) >= 60 else float("nan")

    return {
        "dte": float(dte),
        "width_pct_spot": width / spot,
        "credit_to_width": credit_per_share / width if width > 0 else np.nan,
        "return_on_risk": (credit_per_share * 100.0) / max_loss_per_contract
        if max_loss_per_contract > 0 else np.nan,
        "moneyness_short": short_strike / spot - 1.0,
        "moneyness_long": long_strike / spot - 1.0,
        "short_iv": short_iv,
        # Skew per 1% of spot between the two strikes -- steeper skew means the wing
        # being bought is expensive relative to the wing being sold.
        "iv_skew_slope": (long_iv - short_iv) / (width / spot) if width > 0 else np.nan,
        "iv_rank_252": (iv_now - lo) / (hi - lo) if hi > lo else np.nan,
        "iv_pct_63": float((tail63 < iv_now).mean()) if len(tail63) > 10 else np.nan,
        "iv_minus_rv20": iv_now - rv20,
        "rv20": rv20,
        "rv60": rv60,
        "rv_ratio": rv20 / rv60 if rv60 and rv60 == rv60 and rv60 > 0 else np.nan,
        "term_slope": term,
        "ret5": ret(5),
        "ret20": ret(20),
        "ret60": ret(60),
        "dist_sma50": close / sma50 - 1.0 if sma50 == sma50 else np.nan,
        "drawdown_60": close / hi60 - 1.0 if hi60 == hi60 else np.nan,
        "support_dist": (spot - support_level) / spot,
        # How far the day's low pushed THROUGH the level -- a deep break is a
        # different event from a clean touch.
        "support_break_depth": (support_level - float(hist["low"].iloc[-1])) / spot
        if len(hist) else np.nan,
        "move_sigma": float(candidate_diagnostics.get("move_sigma", np.nan)),
        "spread_pct_short": spread_pct_short,
        "spread_pct_long": spread_pct_long,
        "vix_level": _as_of(vix, d),
        "vix_chg5": _as_of(vix, d) / _as_of(vix, d - pd.Timedelta(days=7).to_pytimedelta()) - 1.0
        if not vix.empty else np.nan,
        "days_to_next_earnings": days_to_next_earnings,
        "cfg_buffer_pct": cfg_buffer_pct,
        "cfg_target_dte": cfg_target_dte,
        "cfg_target_delta": cfg_target_delta,
        "cfg_is_delta_method": cfg_is_delta_method,
    }
