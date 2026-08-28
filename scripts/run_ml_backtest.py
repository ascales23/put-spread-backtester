#!/usr/bin/env python
"""Train the tail-risk model walk-forward and re-run the full backtest through it.

Three runs, same window, same fills, same everything else:
  baseline  -- the best fixed structure, no model
  filter    -- same structure, model declines elevated-tail-risk candidates
  selector  -- model also picks each trade's structure from the harvested grid

    python scripts/run_ml_backtest.py --out runs/ml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from putspread.config import StrategyConfig, SupportConfig       # noqa: E402
from putspread.engine import Backtester                          # noqa: E402
from putspread.fills import FillConfig                           # noqa: E402
from putspread.harvest import ENTRY_GRID, harvest_grid           # noqa: E402
from putspread.metrics import compute_metrics                    # noqa: E402
from putspread.ml import (                                       # noqa: E402
    TailRiskFilter, TailRiskSelector, apply_tail_rule, threshold_scan,
    walk_forward_predict,
)
from putspread.report import build_report                        # noqa: E402
from putspread.runner import DOLT_CAVEATS, BacktestContext       # noqa: E402

RATE_DIR = Path.home() / "marketdata" / "macro" / "treasury_3m" / "1day"
VIX_DIR = Path.home() / "marketdata" / "vix_index" / "VIX" / "1day"
UNIVERSE = ["AMD", "NVDA", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL",
            "NFLX", "MU", "AVGO", "SMCI", "COIN", "PLTR", "SPY"]

BEST_FIXED = dict(short_strike_method="buffer", buffer_pct=0.05, target_dte=14)


def summarize(name: str, result, equity_start: float) -> dict:
    m = compute_metrics(result.trade_frame, result.equity_curve, equity_start)
    return {
        "run": name, "trades": m.n_trades, "win_rate": m.win_rate,
        "total_pnl": m.total_pnl, "expectancy": m.expectancy_per_trade,
        "profit_factor": m.profit_factor, "sharpe": m.sharpe,
        "max_dd_pct": m.max_drawdown_pct, "worst_trade": m.worst_trade,
    }


def by_year(result) -> pd.Series:
    t = result.trade_frame
    if t.empty:
        return pd.Series(dtype=float)
    t = t.assign(year=pd.to_datetime(t["entry_date"]).dt.year)
    return t.groupby("year")["pnl"].sum()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=UNIVERSE)
    p.add_argument("--start", default="2020-01-22")
    p.add_argument("--end", default="2026-08-26")
    p.add_argument("--profit-target", type=float, default=0.5)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--min-train", type=int, default=200)
    p.add_argument("--data", default="data")
    p.add_argument("--harvest", default="runs/selector.harvest.parquet")
    p.add_argument("--out", default="runs/ml")
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    base = StrategyConfig(
        symbols=tuple(a.symbols), start=a.start, end=a.end,
        profit_target_pct=a.profit_target, acknowledge_missing_oi_volume=True,
        support=SupportConfig(),
    )
    fills = FillConfig(model="realistic", fraction=0.5)
    ctx = BacktestContext.build(list(base.symbols), a.data, RATE_DIR)

    hp = Path(a.harvest)
    if hp.exists():
        df = pd.read_parquet(hp)
        for c in ("entry_date", "exit_date", "expiration"):
            df[c] = pd.to_datetime(df[c]).dt.date
        print(f"harvest: {len(df):,} candidate-structure rows from {hp}")
    else:
        print(f"harvesting {len(ENTRY_GRID)} entry structures ...", flush=True)
        df = harvest_grid(ctx, base, fills, vix_dir=str(VIX_DIR))
        df.to_parquet(hp, index=False)

    print("\ntraining tail-risk model (target=big_loss) walk-forward ...")
    res = walk_forward_predict(df, target="big_loss", n_folds=a.folds, min_train=a.min_train)
    print(res.fold_table.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    accept = apply_tail_rule(res)
    scored = res.predictions.notna()
    print(f"\nscored out-of-fold {int(scored.sum()):,} of {len(df):,}; "
          f"a-priori rule accepts {int(accept.sum()):,} ({accept.sum() / max(scored.sum(), 1):.1%})")
    print("\n=== out-of-fold selectivity scan (DIAGNOSTIC -- threshold is NOT chosen here) ===")
    print(threshold_scan(df, -res.predictions, (0.0, 0.3, 0.5, 0.7, 0.9))[
        ["quantile", "trades", "coverage", "win_rate", "expectancy", "profit_factor", "lift"]
    ].to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    fixed_rows = df[(df["cfg_is_delta_method"] == 0.0)
                    & (df["cfg_buffer_pct"] == BEST_FIXED["buffer_pct"])
                    & (df["cfg_target_dte"] == BEST_FIXED["target_dte"])]
    trade_filter = TailRiskFilter.from_frame(fixed_rows, type(res)(
        res.predictions.loc[fixed_rows.index], res.folds, res.fold_table,
        res.importance, res.thresholds.loc[fixed_rows.index]))
    selector = TailRiskSelector.from_frame(df, res)

    first_scored = df.loc[scored, "entry_date"].min()
    caveats = list(DOLT_CAVEATS) + [
        "Trade selection by an XGBoost tail-risk model, scored WALK-FORWARD: every "
        "trade was ranked by a model trained only on trades that had already CLOSED "
        "before that trade was opened, and the accept threshold is the training "
        "window's own base rate of tail losses -- never tuned on the test set.",
        f"Opportunities before {first_scored} are the model's burn-in window and are "
        "never traded.",
    ]

    runs = {}
    print("\n=== full portfolio backtests ===")
    b = Backtester(ctx.provider, ctx.bars, base.with_(**BEST_FIXED), fills, ctx.rates,
                   ctx.calendar, list(DOLT_CAVEATS))
    runs["baseline (fixed 5%/14d)"] = b.run()

    b = Backtester(ctx.provider, ctx.bars, base.with_(**BEST_FIXED), fills, ctx.rates,
                   ctx.calendar, caveats, trade_filter=trade_filter)
    runs["filter (fixed structure)"] = b.run()

    b = Backtester(ctx.provider, ctx.bars, base, fills, ctx.rates, ctx.calendar,
                   caveats, entry_selector=selector)
    runs["selector (model picks structure)"] = b.run()

    table = pd.DataFrame([summarize(k, v, base.starting_equity) for k, v in runs.items()])
    print(table.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\n=== P&L by entry year ===")
    years = pd.DataFrame({k: by_year(v) for k, v in runs.items()})
    print(years.to_string(float_format=lambda x: f"{x:,.0f}"))

    best_name = table.loc[table["total_pnl"].idxmax(), "run"]
    print(f"\nbest by total P&L: {best_name}")

    for name, r in runs.items():
        slug = name.split(" ")[0]
        build_report(r, out.parent / f"{out.name}_{slug}.html",
                     f"Bull Put Spread -- {name}")
    table.to_csv(out.with_suffix(".summary.csv"), index=False)
    years.to_csv(out.with_suffix(".years.csv"))
    print(f"\nreports -> {out.parent}/{out.name}_*.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
