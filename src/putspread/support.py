"""Mechanical identification of the 'low target' (expected support level).

STRATEGY.md section 1 has a human supply the range thesis. A backtest cannot ask a
human 1,600 times, so support is derived from price history instead. Every rule here
is causal: a level available on day t uses only bars up to and including day t.

A pivot low is only *confirmed* pivot_right bars after it prints, so a pivot at index
i does not become usable until index i + pivot_right. Getting this wrong would let
the backtest place strikes under a low it could not yet have seen -- the most
seductive lookahead bug in this whole system, because it makes results better.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SupportConfig


@dataclass(frozen=True)
class SupportLevel:
    """A support level usable on `as_of`, with the bar it originated from."""

    level: float
    origin_index: int
    method: str


def confirmed_pivot_lows(lows: np.ndarray, left: int, right: int) -> np.ndarray:
    """Index of pivot lows, and the bar index at which each becomes confirmed.

    Returns an array of shape (n, 2): [pivot_index, confirmed_at_index].
    A pivot at i requires low[i] to be the strict minimum of the window
    [i-left, i+right]; it is knowable only at i+right.
    """
    n = len(lows)
    out = []
    for i in range(left, n - right):
        window = lows[i - left : i + right + 1]
        if lows[i] == window.min() and (window < lows[i]).sum() == 0:
            # Strictness: no other bar in the window may tie the low on the left side,
            # which keeps flat bases from producing a pivot at every bar.
            if (window[:left] <= lows[i]).sum() == 0:
                out.append((i, i + right))
    return np.array(out, dtype=int).reshape(-1, 2)


def support_series(bars: pd.DataFrame, cfg: SupportConfig) -> pd.Series:
    """Support level for every bar, causal. NaN where no valid level exists.

    `bars` must be indexed by date, ascending, with columns high/low/close, in RAW
    (split-unadjusted) prices so the level is comparable to option strikes.
    """
    close = bars["close"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    n = len(bars)
    levels = np.full(n, np.nan)

    # The level in force on day t is the one a trader identified at the close of day
    # t-1, so every comparison below uses the PRIOR close. Using today's close would
    # define support as "somewhere under wherever we ended up today", which forces the
    # level below today's close by construction -- and that quietly destroys the
    # section 3.2 A/B test, because a bar can then never close below its own support
    # and "confirmation" degenerates into "mechanical".
    prev_close = np.empty(n)
    prev_close[0] = np.nan
    prev_close[1:] = close[:-1]

    if cfg.method == "pivot_low":
        pivots = confirmed_pivot_lows(low, cfg.pivot_left, cfg.pivot_right)
        for t in range(n):
            usable = pivots[(pivots[:, 1] <= t) & (pivots[:, 0] >= t - cfg.lookback_days)]
            if len(usable) == 0:
                continue
            vals = low[usable[:, 0]]
            # Nearest confirmed pivot low that sat below yesterday's close.
            below = vals[vals < prev_close[t]]
            if len(below):
                levels[t] = below.max()

    elif cfg.method == "donchian":
        # Rolling low EXCLUDING today, so touching it today is a genuine event.
        roll = bars["low"].rolling(cfg.donchian_days).min().shift(1).to_numpy(dtype=float)
        levels = roll

    elif cfg.method == "sma":
        levels = bars["close"].rolling(cfg.sma_days).mean().shift(1).to_numpy(dtype=float)

    else:
        raise ValueError(f"unknown support method {cfg.method!r}")

    # A level is only a "pullback target" if it sat a sane distance below yesterday's
    # price -- close enough to be reachable, far enough to be an actual pullback.
    dist = (prev_close - levels) / prev_close
    levels = np.where(
        (dist >= cfg.min_distance_pct) & (dist <= cfg.max_distance_pct), levels, np.nan
    )
    return pd.Series(levels, index=bars.index, name="support")


def entry_signals(
    bars: pd.DataFrame, support: pd.Series, discipline: str, confirmation_bars: int = 1
) -> pd.Series:
    """Boolean series: True on bars where an entry is triggered.

    mechanical  -- the bar's LOW touches or breaches the support level. Only the FIRST
                   such bar counts; day three of sitting on the level is not a new
                   pullback.
    confirmation -- the same touch, but enter only once a bar CLOSES back above the
                   level, within `confirmation_bars` bars. Worse entry price, but it
                   declines the trades where the level is breaking, which is exactly
                   where this strategy's max loss comes from. A touch that never
                   closes back above is never taken.

    In both cases entry is priced at the CLOSE of the signal bar, using that day's
    end-of-day chain -- the only quotes the data actually contains.
    """
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    lvl = support.to_numpy(dtype=float)
    n = len(bars)
    touch = np.zeros(n, dtype=bool)

    for t in range(1, n):
        if np.isnan(lvl[t]):
            continue
        if low[t] <= lvl[t]:
            touch[t] = True

    # Only the first bar of a touch sequence is an entry opportunity.
    fresh = touch & ~np.concatenate(([False], touch[:-1]))

    if discipline == "mechanical":
        return pd.Series(fresh, index=bars.index, name="entry")

    if discipline != "confirmation":
        raise ValueError(f"unknown entry discipline {discipline!r}")

    signal = np.zeros(n, dtype=bool)
    for t in np.flatnonzero(fresh):
        level = lvl[t]
        # The touch bar itself counts: a bar that dips to support and closes back
        # above it is the classic intraday reversal the spec calls out.
        for k in range(t, min(t + confirmation_bars + 1, n)):
            if close[k] > level:
                signal[k] = True
                break
    return pd.Series(signal, index=bars.index, name="entry")
