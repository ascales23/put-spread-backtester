"""Slice 5 -- performance metrics (STRATEGY.md section 9).

Win rates and expectancy here are REALIZED, computed from booked trades on the actual
price path. This module deliberately never imports prob_otm_rn: section 7 forbids
risk-neutral N(d2) from entering expectancy, and the import list is the enforcement.

Credit strategies are left-skewed -- many small wins, occasional large losses -- so
the mean is the least informative number on the page. The distribution stats and the
tail columns below are the ones that decide whether the edge is real.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Metrics:
    """Full metric set for one run."""

    n_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    expectancy_per_trade: float
    expectancy_per_day_in_trade: float
    profit_factor: float
    total_pnl: float
    total_return: float
    cagr: float
    max_drawdown: float
    max_drawdown_pct: float
    drawdown_duration_days: int
    sharpe: float
    sortino: float
    pnl_std: float
    pnl_skew: float
    worst_trade: float
    best_trade: float
    pnl_p05: float
    pnl_p95: float
    avg_days_in_trade: float
    largest_loss_as_pct_of_total_pnl: float
    start: date | None
    end: date | None

    def to_dict(self) -> dict:
        return asdict(self)


def _drawdown(equity: pd.Series) -> tuple[float, float, int]:
    """Max drawdown in dollars, in percent, and the longest underwater run in days."""
    if equity.empty:
        return 0.0, 0.0, 0
    peak = equity.cummax()
    dd = equity - peak
    dd_pct = dd / peak.replace(0, np.nan)
    underwater = dd < 0
    longest, run = 0, 0
    for flag in underwater:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return float(dd.min()), float(dd_pct.min() if dd_pct.notna().any() else 0.0), int(longest)


def compute_metrics(trades: pd.DataFrame, equity: pd.DataFrame, starting_equity: float) -> Metrics:
    """Metrics from a trade log and a daily equity curve.

    Sharpe and Sortino are computed on DAILY EQUITY RETURNS, not per-trade returns.
    Per-trade ratios flatter a strategy that trades rarely; the account only ever
    experiences the daily series, so that is what gets measured.
    """
    empty = trades is None or trades.empty
    pnl = trades["pnl"].to_numpy(dtype=float) if not empty else np.array([])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]

    if equity is not None and not equity.empty:
        eq = equity["equity"].astype(float)
        rets = eq.pct_change().dropna()
        ann = float(rets.mean() * TRADING_DAYS)
        vol = float(rets.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(rets) > 1 else 0.0
        downside = rets[rets < 0]
        dvol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(downside) > 1 else 0.0
        sharpe = ann / vol if vol > 0 else 0.0
        sortino = ann / dvol if dvol > 0 else 0.0
        dd, dd_pct, dd_days = _drawdown(eq)
        years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
        total_return = float(eq.iloc[-1] / starting_equity - 1.0)
        cagr = float((eq.iloc[-1] / starting_equity) ** (1 / years) - 1.0) if eq.iloc[-1] > 0 else -1.0
        start_d, end_d = eq.index[0], eq.index[-1]
    else:
        sharpe = sortino = dd = dd_pct = total_return = cagr = 0.0
        dd_days = 0
        start_d = end_d = None

    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    days_in = trades["days_in_trade"].to_numpy(dtype=float) if not empty else np.array([])
    total = float(pnl.sum()) if len(pnl) else 0.0

    return Metrics(
        n_trades=int(len(pnl)),
        win_rate=float((pnl > 0).mean()) if len(pnl) else 0.0,
        avg_win=float(wins.mean()) if len(wins) else 0.0,
        avg_loss=float(losses.mean()) if len(losses) else 0.0,
        expectancy_per_trade=float(pnl.mean()) if len(pnl) else 0.0,
        expectancy_per_day_in_trade=float((pnl / np.maximum(days_in, 1)).mean()) if len(pnl) else 0.0,
        profit_factor=(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0,
        total_pnl=total,
        total_return=total_return,
        cagr=cagr,
        max_drawdown=dd,
        max_drawdown_pct=dd_pct,
        drawdown_duration_days=dd_days,
        sharpe=sharpe,
        sortino=sortino,
        pnl_std=float(pnl.std(ddof=1)) if len(pnl) > 1 else 0.0,
        pnl_skew=float(pd.Series(pnl).skew()) if len(pnl) > 2 else 0.0,
        worst_trade=float(pnl.min()) if len(pnl) else 0.0,
        best_trade=float(pnl.max()) if len(pnl) else 0.0,
        pnl_p05=float(np.percentile(pnl, 5)) if len(pnl) else 0.0,
        pnl_p95=float(np.percentile(pnl, 95)) if len(pnl) else 0.0,
        avg_days_in_trade=float(days_in.mean()) if len(days_in) else 0.0,
        largest_loss_as_pct_of_total_pnl=(
            float(abs(pnl.min()) / abs(total)) if len(pnl) and total != 0 else float("nan")
        ),
        start=start_d,
        end=end_d,
    )


def pnl_histogram(trades: pd.DataFrame, bins: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Counts and bin edges of the per-trade P&L distribution (section 9)."""
    if trades is None or trades.empty:
        return np.array([]), np.array([])
    return np.histogram(trades["pnl"].to_numpy(dtype=float), bins=bins)


def exit_reason_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """How each exit rule actually performed -- which rule is earning its keep."""
    if trades is None or trades.empty:
        return pd.DataFrame()
    g = trades.groupby("exit_reason")["pnl"]
    out = pd.DataFrame({
        "trades": g.size(),
        "total_pnl": g.sum(),
        "avg_pnl": g.mean(),
        "win_rate": trades.assign(w=trades["pnl"] > 0).groupby("exit_reason")["w"].mean(),
    })
    return out.sort_values("total_pnl", ascending=False)
