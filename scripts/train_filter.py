#!/usr/bin/env python
"""Harvest candidate trades, train the XGBoost filter walk-forward, and report.

    python scripts/train_filter.py --symbols AMD NVDA ... --buffer 0.05 --dte 14 \
        --profit-target 0.5 --out runs/filter
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from putspread.config import StrategyConfig, SupportConfig     # noqa: E402
from putspread.fills import FillConfig                         # noqa: E402
from putspread.harvest import harvest                          # noqa: E402
from putspread.ml import (                                     # noqa: E402
    evaluate_threshold, threshold_scan, walk_forward_predict,
)
from putspread.runner import BacktestContext                   # noqa: E402

RATE_DIR = Path.home() / "marketdata" / "macro" / "treasury_3m" / "1day"
VIX_DIR = Path.home() / "marketdata" / "vix_index" / "VIX" / "1day"

UNIVERSE = ["AMD", "NVDA", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL",
            "NFLX", "MU", "AVGO", "SMCI", "COIN", "PLTR", "SPY"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=UNIVERSE)
    p.add_argument("--start", default="2020-01-22")
    p.add_argument("--end", default="2026-08-26")
    p.add_argument("--buffer", type=float, default=0.05)
    p.add_argument("--dte", type=int, default=14)
    p.add_argument("--profit-target", type=float, default=0.5)
    p.add_argument("--method", default="buffer")
    p.add_argument("--target", default="ror", choices=["ror", "win", "big_loss"])
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--min-train", type=int, default=60)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/filter")
    p.add_argument("--reuse", action="store_true", help="reuse an existing harvest file")
    p.add_argument("--invert", action="store_true",
                   help="negate predictions, for targets where LOW is good (big_loss)")
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    harvest_path = out.with_suffix(".harvest.parquet")

    cfg = StrategyConfig(
        symbols=tuple(a.symbols), start=a.start, end=a.end,
        short_strike_method=a.method, buffer_pct=a.buffer, target_dte=a.dte,
        profit_target_pct=a.profit_target, acknowledge_missing_oi_volume=True,
        support=SupportConfig(),
    )
    fills = FillConfig(model="realistic", fraction=0.5)

    if a.reuse and harvest_path.exists():
        df = pd.read_parquet(harvest_path)
        df["entry_date"] = pd.to_datetime(df["entry_date"]).dt.date
        df["exit_date"] = pd.to_datetime(df["exit_date"]).dt.date
        print(f"reusing {harvest_path} ({len(df):,} candidates)")
    else:
        print("harvesting candidate trades ...", flush=True)
        ctx = BacktestContext.build(list(cfg.symbols), a.data, RATE_DIR)
        df = harvest(ctx, cfg, fills, vix_dir=str(VIX_DIR))
        df.to_parquet(harvest_path, index=False)
        print(f"{len(df):,} candidates -> {harvest_path}")

    if df.empty:
        print("no candidates harvested")
        return 1

    print(f"\ncandidates {len(df):,}  win rate {df['win'].mean():.1%}  "
          f"expectancy ${df['pnl'].mean():,.2f}/trade  "
          f"total ${df['pnl'].sum():,.0f}  big losses {df['big_loss'].sum()}")
    print(f"span {df['entry_date'].min()} .. {df['entry_date'].max()}")

    res = walk_forward_predict(df, target=a.target, n_folds=a.folds, min_train=a.min_train)
    if a.invert:
        res.predictions = -res.predictions
    print(f"\n=== walk-forward folds (target={a.target}) ===")
    print(res.fold_table.to_string(index=False, float_format=lambda x: f"{x:,.4f}")
          if not res.fold_table.empty else "no folds met the training-size floor")

    scored = res.predictions.notna().sum()
    print(f"\nscored out-of-fold: {scored:,} of {len(df):,} candidates "
          f"({scored / len(df):.0%}); the rest are burn-in")

    print("\n=== out-of-fold selectivity scan (DIAGNOSTIC -- not a selection) ===")
    scan = threshold_scan(df, res.predictions)
    print(scan.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # The deployed threshold is fixed a priori, not read off the scan above.
    fixed = 0.0 if a.target == "ror" else (-0.5 if a.invert else 0.5)
    deployed = evaluate_threshold(df, res.predictions, fixed)
    print(f"\n=== deployed rule: take when predicted {a.target} >= {fixed} ===")
    for k, v in deployed.items():
        print(f"  {k:24s} {v:,.4f}" if isinstance(v, float) else f"  {k:24s} {v}")

    if not res.importance.empty:
        print("\n=== mean feature importance (top 12) ===")
        print(res.importance.head(12).to_string(float_format=lambda x: f"{x:,.4f}"))

    df.assign(pred=res.predictions).to_parquet(out.with_suffix(".scored.parquet"), index=False)
    scan.to_csv(out.with_suffix(".scan.csv"), index=False)
    (out.with_suffix(".deployed.json")).write_text(json.dumps(deployed, indent=2, default=str))
    print(f"\nscored candidates -> {out.with_suffix('.scored.parquet')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
