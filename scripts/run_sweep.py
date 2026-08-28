#!/usr/bin/env python
"""Parameter sweep with walk-forward out-of-sample validation (Slice 6).

    python scripts/run_sweep.py --symbols AMD NVDA TSLA --folds 4 --out runs/sweep.html

Prints the in-sample vs out-of-sample table. A parameter set that only wins in-sample
is flagged in the `oos_confirms` column, not averaged away.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from putspread.config import SWEEP_GRID, StrategyConfig, SupportConfig   # noqa: E402
from putspread.fills import FillConfig                                   # noqa: E402
from putspread.runner import BacktestContext                             # noqa: E402
from putspread.sweep import sweep, walk_forward                          # noqa: E402

RATE_DIR = Path.home() / "marketdata" / "macro" / "treasury_3m" / "1day"

#: A tractable subset of the section 8 grid. The full cartesian product is thousands
#: of runs; these are the axes the spec says actually matter.
DEFAULT_GRID = {
    "short_strike_method": ["buffer", "delta"],
    "buffer_pct": [0.02, 0.03, 0.05],
    "target_short_delta": [0.10, 0.20, 0.30],
    "target_dte": [14, 22, 45],
    "entry_discipline": ["mechanical", "confirmation"],
    "profit_target_pct": [None, 0.50],
    "stop_rule": ["none", "level_break"],
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", default="2020-01-22")
    p.add_argument("--end", default="2026-08-26")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--train-years", type=float, default=2.0)
    p.add_argument("--min-trades", type=int, default=10)
    p.add_argument("--fill", default="realistic")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/sweep")
    p.add_argument("--full-grid", action="store_true")
    p.add_argument("--jobs", type=int, default=8)
    a = p.parse_args()

    base = StrategyConfig(
        symbols=tuple(a.symbols), start=a.start, end=a.end,
        acknowledge_missing_oi_volume=True, support=SupportConfig(),
    )
    ctx = BacktestContext.build(list(base.symbols), a.data, RATE_DIR)
    fills = FillConfig(model=a.fill, fraction=0.5)
    grid = SWEEP_GRID if a.full_grid else DEFAULT_GRID

    print(f"walk-forward: {a.folds} folds, {a.train_years}y train windows, "
          f"grid of {len(grid)} axes, fills={a.fill}\n", flush=True)
    folds, chosen = walk_forward(base, grid, ctx, fills, a.folds, a.train_years,
                                 min_trades=a.min_trades, n_jobs=a.jobs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out.with_suffix(".folds.csv"), index=False)

    print("=== walk-forward, in-sample vs out-of-sample ===")
    if folds.empty:
        print("no folds produced results")
    else:
        cols = [c for c in ("fold", "train_start", "test_start", "test_end",
                            "is_trades", "is_expectancy", "oos_trades", "oos_expectancy",
                            "overfit_gap", "oos_confirms") if c in folds.columns]
        print(folds[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
        if "oos_confirms" in folds:
            n = int(folds["oos_confirms"].sum())
            print(f"\nfolds where the in-sample winner also made money out-of-sample: "
                  f"{n} / {len(folds)}")

    print("\n=== full-sample sweep (IN-SAMPLE -- not a result, a diagnostic) ===")
    full = sweep(base, grid, ctx, fills, min_trades=a.min_trades, n_jobs=a.jobs)
    full.to_csv(out.with_suffix(".insample.csv"), index=False)
    show = full[full["enough_trades"]].sort_values("expectancy_per_trade", ascending=False)
    print(show.head(15).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print(f"\nCSVs -> {out.with_suffix('.folds.csv')}, {out.with_suffix('.insample.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
