#!/usr/bin/env python
"""Instantaneous-gap stress: what a market-wide crash costs the book on any given day.

The 2020 replay could not answer the leverage question, because the strategy barely
traded through it -- the max-loss budget and the liquidity filter refused nearly every
signal. That is reassuring but it is not a test: the book was never full when the gap
came.

This asks the question directly and without needing the crash to coincide with a full
book. For EVERY session in the backtest, take the positions actually open that day and
ask what an instantaneous market-wide decline would cost. Because these are defined-risk
verticals, any underlying that gaps below its long strike delivers exactly the max loss
and no more, so the answer is computable in closed form rather than simulated.

SPY fell 34.1% peak-to-trough in Feb-Mar 2020, so that is the headline shock.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from putspread.config import shipped_config                     # noqa: E402
from putspread.engine import Backtester                         # noqa: E402
from putspread.fills import FillConfig                          # noqa: E402
from putspread.ml import TailRiskFilter, WalkForwardResult, walk_forward_predict  # noqa: E402
from putspread.runner import BacktestContext                    # noqa: E402
from putspread.spread import payoff_at_expiry_per_contract      # noqa: E402

RATE_DIR = Path.home() / "marketdata" / "macro" / "treasury_3m" / "1day"
SHOCKS = (-0.10, -0.20, -0.34, -0.50)


class RecordingBacktester(Backtester):
    """Backtester that snapshots the open book every session."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.book_by_day: dict = {}

    def run(self):
        result = super().run()
        return result

    def record_equity_hook(self):
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", type=float, default=[1.0, 2.0, 4.0, 6.0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--harvest", default="runs/selector.harvest.parquet")
    ap.add_argument("--out", default="runs/gapstress")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)

    cfg0 = shipped_config()
    ctx = BacktestContext.build(list(cfg0.symbols), a.data, RATE_DIR)
    fills = FillConfig(model="realistic", fraction=0.5)

    df = pd.read_parquet(a.harvest)
    for c in ("entry_date", "exit_date", "expiration"):
        df[c] = pd.to_datetime(df[c]).dt.date
    res = walk_forward_predict(df, target="big_loss", n_folds=4, min_train=200)
    fx = df[(df["cfg_is_delta_method"] == 0.0) & (df["cfg_buffer_pct"] == 0.05)
            & (df["cfg_target_dte"] == 14)]
    flt = TailRiskFilter.from_frame(fx, WalkForwardResult(
        res.predictions.loc[fx.index], res.folds, res.fold_table,
        res.importance, res.thresholds.loc[fx.index]))

    rows = []
    for lev in a.levels:
        cfg = shipped_config(leverage=lev)
        bt = Backtester(ctx.provider, ctx.bars, cfg, fills, ctx.rates, ctx.calendar,
                        [], trade_filter=flt)
        result = bt.run()
        trades = result.trade_frame
        if trades.empty:
            continue

        eq = result.equity_curve
        # Rebuild the open book per session from the trade log: a position is open on
        # every session between its entry and its exit.
        open_on = defaultdict(list)
        for t in trades.itertuples(index=False):
            for d in eq.index:
                if t.entry_date <= d < t.exit_date:
                    open_on[d].append(t)

        per_shock = {s: [] for s in SHOCKS}
        for d, positions in open_on.items():
            equity = float(eq.loc[d, "equity"])
            if equity <= 0:
                continue
            for shock in SHOCKS:
                loss = 0.0
                for t in positions:
                    spot = bt._spot_for(t.symbol, d)
                    if spot is None:
                        continue
                    shocked = spot * (1.0 + shock)
                    # Defined risk: the payoff floor IS the max loss, by construction.
                    pnl = payoff_at_expiry_per_contract(
                        shocked, t.short_strike, t.long_strike, t.credit_per_share
                    ) * t.contracts
                    # Credit already banked at entry; the marginal hit is pnl minus
                    # the credit that is already reflected in equity.
                    loss += min(pnl, 0.0) if pnl < 0 else 0.0
                per_shock[shock].append((d, loss / equity))

        row = {"leverage": lev, "sessions_with_open_risk": len(open_on),
               "peak_open_risk_pct": float((eq["open_risk"] / eq["equity"]).max())}
        for shock in SHOCKS:
            vals = [v for _, v in per_shock[shock]]
            row[f"worst_{int(abs(shock)*100)}pct_gap"] = min(vals) if vals else 0.0
            row[f"median_{int(abs(shock)*100)}pct_gap"] = float(np.median(vals)) if vals else 0.0
        # When did the worst 34% case land?
        v34 = per_shock[-0.34]
        if v34:
            worst_day = min(v34, key=lambda x: x[1])
            row["worst_34_date"] = str(worst_day[0])
        rows.append(row)

    tbl = pd.DataFrame(rows)
    print("=== instantaneous market-wide gap, loss as a fraction of equity ===")
    print("(shipped config, ML filter on, realistic fills; every open book in the backtest)\n")
    print(tbl.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    tbl.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\n-> {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
