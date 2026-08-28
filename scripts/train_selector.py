#!/usr/bin/env python
"""Harvest every entry structure, train the XGBoost selector walk-forward, and
re-run the full backtest with the model choosing each trade's structure.

    python scripts/train_selector.py --out runs/selector
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from putspread.config import StrategyConfig, SupportConfig      # noqa: E402
from putspread.engine import Backtester                         # noqa: E402
from putspread.fills import FillConfig                          # noqa: E402
from putspread.harvest import ENTRY_GRID, harvest_grid          # noqa: E402
from putspread.metrics import compute_metrics                   # noqa: E402
from putspread.ml import (                                      # noqa: E402
    StructureSelector, best_per_opportunity, threshold_scan, walk_forward_predict,
)
from putspread.report import build_report                       # noqa: E402
from putspread.runner import DOLT_CAVEATS, BacktestContext      # noqa: E402

RATE_DIR = Path.home() / "marketdata" / "macro" / "treasury_3m" / "1day"
VIX_DIR = Path.home() / "marketdata" / "vix_index" / "VIX" / "1day"
UNIVERSE = ["AMD", "NVDA", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL",
            "NFLX", "MU", "AVGO", "SMCI", "COIN", "PLTR", "SPY"]


def period_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    d = df.copy()
    d["year"] = pd.to_datetime(d["entry_date"]).dt.year
    g = d.groupby("year").agg(
        trades=("pnl", "size"), pnl=("pnl", "sum"),
        win=("pnl", lambda x: (x > 0).mean()), exp=("pnl", "mean"),
    )
    g.columns = pd.MultiIndex.from_product([[label], g.columns])
    return g


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=UNIVERSE)
    p.add_argument("--start", default="2020-01-22")
    p.add_argument("--end", default="2026-08-26")
    p.add_argument("--profit-target", type=float, default=0.5)
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--min-train", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.0)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/selector")
    p.add_argument("--reuse", action="store_true")
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hp = out.with_suffix(".harvest.parquet")

    base = StrategyConfig(
        symbols=tuple(a.symbols), start=a.start, end=a.end,
        profit_target_pct=a.profit_target, acknowledge_missing_oi_volume=True,
        support=SupportConfig(),
    )
    fills = FillConfig(model="realistic", fraction=0.5)
    ctx = BacktestContext.build(list(base.symbols), a.data, RATE_DIR)

    if a.reuse and hp.exists():
        df = pd.read_parquet(hp)
        for c in ("entry_date", "exit_date", "expiration"):
            df[c] = pd.to_datetime(df[c]).dt.date
        print(f"reusing {hp} ({len(df):,} rows)")
    else:
        print(f"harvesting {len(ENTRY_GRID)} entry structures ...", flush=True)
        df = harvest_grid(ctx, base, fills, vix_dir=str(VIX_DIR))
        df.to_parquet(hp, index=False)
        print(f"\n{len(df):,} candidate-structure rows -> {hp}")

    if df.empty:
        print("nothing harvested")
        return 1

    opps = df.groupby(["symbol", "entry_date"]).ngroups
    print(f"\n{opps:,} distinct opportunities x {len(ENTRY_GRID)} structures")
    print(f"pooled: win {df['win'].mean():.1%}  expectancy ${df['pnl'].mean():,.2f}  "
          f"mean RoR {df['ror'].mean():.4f}")

    res = walk_forward_predict(df, target="ror", n_folds=a.folds, min_train=a.min_train)
    print("\n=== walk-forward folds ===")
    print(res.fold_table.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))

    print("\n=== out-of-fold selectivity scan (DIAGNOSTIC) ===")
    print(threshold_scan(df, res.predictions).to_string(
        index=False, float_format=lambda x: f"{x:,.3f}"))

    chosen = best_per_opportunity(df, res.predictions, a.threshold)
    print(f"\n=== model-chosen trade per opportunity (threshold {a.threshold}) ===")
    print(f"opportunities taken {len(chosen):,}  win {chosen['win'].mean():.1%}  "
          f"expectancy ${chosen['pnl'].mean():,.2f}  total ${chosen['pnl'].sum():,.0f}  "
          f"mean RoR {chosen['ror'].mean():.4f}")

    print("\n=== structures the model actually picked ===")
    picks = chosen.assign(
        method=chosen["cfg_is_delta_method"].map({1.0: "delta", 0.0: "buffer"}),
    ).groupby(["method", "cfg_target_dte"]).size().to_frame("trades")
    print(picks.to_string())

    print("\n=== by year: model-chosen vs each fixed structure's own best ===")
    scored_span = chosen["entry_date"].min()
    fixed = df[(df["cfg_is_delta_method"] == 0.0) & (df["cfg_buffer_pct"] == 0.05)
               & (df["cfg_target_dte"] == 14) & (df["entry_date"] >= scored_span)]
    both = period_table(chosen, "model").join(period_table(fixed, "fixed 5%/14d"), how="outer")
    print(both.to_string(float_format=lambda x: f"{x:,.2f}"))

    print("\n=== by symbol, model-chosen ===")
    bs = chosen.groupby("symbol").agg(
        trades=("pnl", "size"), pnl=("pnl", "sum"), win=("pnl", lambda x: (x > 0).mean()),
    ).sort_values("pnl", ascending=False)
    print(bs.to_string(float_format=lambda x: f"{x:,.2f}"))
    print(f"top-3 share of positive P&L: "
          f"{bs['pnl'].head(3).sum() / max(bs['pnl'].sum(), 1e-9):.1%}")

    if not res.importance.empty:
        print("\n=== mean feature importance (top 12) ===")
        print(res.importance.head(12).to_string(float_format=lambda x: f"{x:,.4f}"))

    # ---- the real test: a full portfolio backtest driven by the selector
    print("\n=== FULL BACKTEST with the selector wired into the engine ===")
    selector = StructureSelector.from_frame(df, res.predictions, a.threshold)
    caveats = list(DOLT_CAVEATS) + [
        "Entry structure chosen per trade by an XGBoost model scored WALK-FORWARD: "
        "each trade was ranked by a model trained only on trades that had already "
        "closed before that trade was opened.",
        f"Opportunities before {scored_span} fall in the model's burn-in window and "
        "are never traded, so this run cannot speak to the 2020-2022 period at all.",
    ]
    bt = Backtester(provider=ctx.provider, bars=ctx.bars, cfg=base, fills=fills,
                    rates=ctx.rates, calendar=ctx.calendar, data_caveats=caveats,
                    entry_selector=selector)
    result = bt.run()
    m = compute_metrics(result.trade_frame, result.equity_curve, base.starting_equity)
    print(f"trades {m.n_trades}  win {m.win_rate:.1%}  P&L ${m.total_pnl:,.0f}  "
          f"expectancy ${m.expectancy_per_trade:,.0f}/trade  PF {m.profit_factor:.2f}  "
          f"Sharpe {m.sharpe:.2f}  maxDD {m.max_drawdown_pct:.1%}")

    rep = build_report(result, out.with_suffix(".html"),
                       "Bull Put Spread -- XGBoost structure selector")
    df.assign(pred=res.predictions).to_parquet(out.with_suffix(".scored.parquet"), index=False)
    chosen.to_csv(out.with_suffix(".chosen.csv"), index=False)
    print(f"\nreport -> {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
