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

    if cfg.method == "pivot_low":
        pivots = confirmed_pivot_lows(low, cfg.pivot_left, cfg.pivot_right)
        for t in range(n):
            usable = pivots[(pivots[:, 1] <= t) & (pivots[:, 0] >= t - cfg.lookback_days)]
            if len(usable) == 0:
                continue
            vals = low[usable[:, 0]]
            # Nearest confirmed pivot low that still sits below the current price.
            below = vals[vals < close[t]]
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

    # A level is only a "pullback target" if it sits a sane distance below price.
    dist = (close - levels) / close
    levels = np.where(
        (dist >= cfg.min_distance_pct) & (dist <= cfg.max_distance_pct), levels, np.nan
    )
    return pd.Series(levels, index=bars.index, name="support")


def entry_signals(
    bars: pd.DataFrame, support: pd.Series, discipline: str, confirmation_bars: int = 1
) -> pd.Series:
    """Boolean series: True on bars where an entry is triggered.

    mechanical  -- the bar's LOW touches or breaches the support level, and the prior
                   bar closed above it (so this is a fresh pullback into the level,
                   not day three of sitting on it).
    confirmation -- after such a touch, wait for a close back ABOVE the level within
                   `confirmation_bars` bars, and enter on that bar's close. Worse
                   entry price, but it declines the trades where the level is failing,
                   which is exactly where this strategy's max loss comes from.

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
        if low[t] <= lvl[t] and close[t - 1] > lvl[t]:
            touch[t] = True

    if discipline == "mechanical":
        return pd.Series(touch, index=bars.index, name="entry")

    if discipline != "confirmation":
        raise ValueError(f"unknown entry discipline {discipline!r}")

    signal = np.zeros(n, dtype=bool)
    for t in np.flatnonzero(touch):
        level = lvl[t]
        # The touch bar itself counts: a bar that dips to support and closes back
        # above it is the classic intraday reversal the spec calls out.
        for k in range(t, min(t + confirmation_bars + 1, n)):
            if close[k] > level:
                signal[k] = True
                break
    return pd.Series(signal, index=bars.index, name="entry")
