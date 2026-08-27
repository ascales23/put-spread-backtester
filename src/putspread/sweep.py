"""Slice 6 -- parameter sweep with walk-forward out-of-sample validation.

The build prompt is blunt about this: "Do not report only in-sample sweep results."
So `walk_forward` is the primary entry point and `sweep` exists mainly to feed it.
A parameter set that wins in-sample and loses out-of-sample is flagged in the output,
not quietly averaged into a headline.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from typing import Callable, Iterable

import pandas as pd

from .config import StrategyConfig
from .fills import FillConfig
from .metrics import compute_metrics
from .runner import BacktestContext, run_backtest

Objective = Callable[[pd.Series], float]


def expand_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of the parameter grid, as a list of override dicts."""
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]


def _relevant(overrides: dict) -> dict:
    """Drop parameters that cannot matter for this combination.

    Sweeping target_short_delta while the method is 'buffer' produces five identical
    runs and five identical rows, which inflates the grid and, worse, makes an
    arbitrary duplicate look like a winner. Collapsing them keeps the comparison honest.
    """
    o = dict(overrides)
    if o.get("short_strike_method") != "delta":
        o.pop("target_short_delta", None)
    if o.get("short_strike_method") != "buffer":
        o.pop("buffer_pct", None)
    if o.get("stop_rule") != "credit_multiple":
        o.pop("stop_credit_multiple", None)
    if o.get("stop_rule") != "level_break":
        o.pop("stop_level_break_pct", None)
    return o


def sweep(
    base: StrategyConfig,
    grid: dict[str, list],
    ctx: BacktestContext,
    fills: FillConfig,
    start: str | None = None,
    end: str | None = None,
    min_trades: int = 10,
) -> pd.DataFrame:
    """Run every parameter combination over one window. Returns one row per set."""
    seen: set[tuple] = set()
    rows = []
    for overrides in expand_grid(grid):
        key = tuple(sorted(_relevant(overrides).items(), key=lambda kv: str(kv)))
        if key in seen:
            continue
        seen.add(key)
        cfg = replace(base, **overrides)
        if start:
            cfg = replace(cfg, start=start)
        if end:
            cfg = replace(cfg, end=end)
        res = run_backtest(cfg, ctx, fills)
        m = compute_metrics(res.trade_frame, res.equity_curve, cfg.starting_equity)
        row = {k: v for k, v in _relevant(overrides).items()}
        row.update(
            trades=m.n_trades, win_rate=m.win_rate, total_pnl=m.total_pnl,
            expectancy_per_trade=m.expectancy_per_trade, profit_factor=m.profit_factor,
            sharpe=m.sharpe, sortino=m.sortino, max_drawdown_pct=m.max_drawdown_pct,
            worst_trade=m.worst_trade, pnl_skew=m.pnl_skew,
            enough_trades=m.n_trades >= min_trades,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def default_objective(row: pd.Series) -> float:
    """Rank by realized expectancy per trade, penalized for a thin sample.

    Not Sharpe: a credit strategy that has not yet met its tail shows a magnificent
    Sharpe right up until the day it does not. Not total P&L either, which just picks
    whichever set traded most. Expectancy with a sample-size floor is the least
    gameable of the three, and sets below the floor are excluded outright.
    """
    if not row.get("enough_trades", False):
        return float("-inf")
    return float(row["expectancy_per_trade"])


def make_folds(start: str, end: str, n_folds: int = 4, train_years: float = 2.0) -> list[dict]:
    """Rolling walk-forward folds: train on `train_years`, test on the window after it."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    total = (e - s).days
    train_days = int(train_years * 365.25)
    test_days = max((total - train_days) // n_folds, 30)
    folds = []
    cursor = s
    while True:
        tr_start, tr_end = cursor, cursor + pd.Timedelta(days=train_days)
        te_start, te_end = tr_end + pd.Timedelta(days=1), tr_end + pd.Timedelta(days=test_days)
        if te_start >= e:
            break
        folds.append({
            "train_start": str(tr_start.date()), "train_end": str(tr_end.date()),
            "test_start": str(te_start.date()), "test_end": str(min(te_end, e).date()),
        })
        cursor = cursor + pd.Timedelta(days=test_days)
    return folds


def walk_forward(
    base: StrategyConfig,
    grid: dict[str, list],
    ctx: BacktestContext,
    fills: FillConfig,
    n_folds: int = 4,
    train_years: float = 2.0,
    objective: Objective = default_objective,
    min_trades: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Optimize in-sample per fold, then measure the SAME set out-of-sample.

    Returns (fold_table, chosen_params). The fold table puts in-sample and
    out-of-sample side by side so the overfitting gap is visible without arithmetic.
    """
    folds = make_folds(base.start, base.end, n_folds, train_years)
    fold_rows, chosen_rows = [], []

    for i, fold in enumerate(folds, 1):
        is_tbl = sweep(base, grid, ctx, fills, fold["train_start"], fold["train_end"], min_trades)
        if is_tbl.empty:
            continue
        scores = is_tbl.apply(objective, axis=1)
        if not (scores > float("-inf")).any():
            fold_rows.append({**fold, "fold": i, "status": "no in-sample set met the trade floor"})
            continue
        best = is_tbl.loc[scores.idxmax()]
        param_cols = [c for c in is_tbl.columns if c in grid]
        params = {c: best[c] for c in param_cols if pd.notna(best[c])}

        oos_cfg = replace(base, **params, start=fold["test_start"], end=fold["test_end"])
        oos = run_backtest(oos_cfg, ctx, fills)
        om = compute_metrics(oos.trade_frame, oos.equity_curve, base.starting_equity)

        fold_rows.append({
            "fold": i, **fold, **{f"param_{k}": v for k, v in params.items()},
            "is_trades": best["trades"], "is_expectancy": best["expectancy_per_trade"],
            "is_win_rate": best["win_rate"], "is_total_pnl": best["total_pnl"],
            "is_profit_factor": best["profit_factor"],
            "oos_trades": om.n_trades, "oos_expectancy": om.expectancy_per_trade,
            "oos_win_rate": om.win_rate, "oos_total_pnl": om.total_pnl,
            "oos_profit_factor": om.profit_factor,
            "overfit_gap": best["expectancy_per_trade"] - om.expectancy_per_trade,
            "oos_confirms": bool(om.n_trades > 0 and om.expectancy_per_trade > 0),
        })
        chosen_rows.append({"fold": i, **params})

    return pd.DataFrame(fold_rows), pd.DataFrame(chosen_rows)
