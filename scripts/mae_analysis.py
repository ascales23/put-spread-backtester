#!/usr/bin/env python
"""Where does a stop belong? Derive it from what trades actually did.

Maximum adverse excursion (MAE) is the standard tool: for every trade, how far
underwater did it go before it resolved? If eventual losers reliably dig deeper than
eventual winners ever do, a stop at that depth cuts losers while leaving winners
alone. If the two distributions overlap, every stop level is a tax on the winners --
and the honest answer is that no stop helps.

Excursions are measured in MULTIPLES OF CREDIT RECEIVED, because that is the unit a
credit-spread stop is written in ("close at 2x credit").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from putspread.config import StrategyConfig, SupportConfig     # noqa: E402
from putspread.fills import FillConfig                         # noqa: E402
from putspread.harvest import harvest                          # noqa: E402
from putspread.runner import BacktestContext                   # noqa: E402

RATE_DIR = Path.home() / "marketdata" / "macro" / "treasury_3m" / "1day"
VIX_DIR = Path.home() / "marketdata" / "vix_index" / "VIX" / "1day"
UNIVERSE = ["AMD", "NVDA", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL",
            "NFLX", "MU", "AVGO", "SMCI", "COIN", "PLTR", "SPY"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=UNIVERSE)
    ap.add_argument("--profit-target", type=float, default=None,
                    help="omit so trades run to expiry and excursions are unclipped")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/mae")
    ap.add_argument("--reuse", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hp = out.with_suffix(".harvest.parquet")

    # Deliberately NO profit target and NO stop: a trade already closed at +50% never
    # reveals how far it would have run, and one closed by a stop never reveals
    # whether it would have recovered. Excursions must be measured on unmanaged
    # trades or the analysis answers a question about the exits, not about the trades.
    cfg = StrategyConfig(
        symbols=tuple(a.symbols), profit_target_pct=a.profit_target, stop_rule="none",
        acknowledge_missing_oi_volume=True, support=SupportConfig(),
        short_strike_method="buffer", buffer_pct=0.05, target_dte=14,
    )
    if a.reuse and hp.exists():
        df = pd.read_parquet(hp)
    else:
        ctx = BacktestContext.build(list(cfg.symbols), a.data, RATE_DIR)
        print("harvesting unmanaged trades for excursion analysis ...", flush=True)
        df = harvest(ctx, cfg, FillConfig(model="realistic", fraction=0.5), str(VIX_DIR))
        df.to_parquet(hp, index=False)
    print(f"{len(df):,} unmanaged trades  win {df['win'].mean():.1%}  "
          f"expectancy ${df['pnl'].mean():,.2f}")

    win, loss = df[df["win"] == 1], df[df["win"] == 0]
    qs = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00]

    print("\n=== MAE: how far underwater, in multiples of credit received ===")
    tbl = pd.DataFrame({
        "winners": win["mae_credit_mult"].quantile(qs),
        "losers": loss["mae_credit_mult"].quantile(qs),
    })
    tbl.index = [f"p{int(q * 100)}" for q in qs]
    print(tbl.to_string(float_format=lambda x: f"{x:,.2f}"))

    print("\n=== what a stop at N x credit would have caught ===")
    rows = []
    for mult in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        w_hit = float((win["mae_credit_mult"] >= mult).mean())
        l_hit = float((loss["mae_credit_mult"] >= mult).mean())
        # Trades stopped at this level: losers cut (good), winners cut (bad).
        rows.append({
            "stop_x_credit": mult,
            "pct_losers_stopped": l_hit,
            "pct_winners_stopped": w_hit,
            "losers_caught_per_winner_lost": l_hit / w_hit if w_hit > 0 else np.inf,
            "avg_loser_pnl": float(loss["pnl"].mean()),
            "avg_winner_pnl_at_risk": float(win["pnl"].mean()),
        })
    scan = pd.DataFrame(rows)
    print(scan.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("\n=== MFE: how far in profit, in multiples of credit ===")
    mfe = pd.DataFrame({
        "winners": win["mfe_credit_mult"].quantile(qs),
        "losers": loss["mfe_credit_mult"].quantile(qs),
    })
    mfe.index = [f"p{int(q * 100)}" for q in qs]
    print(mfe.to_string(float_format=lambda x: f"{x:,.2f}"))
    print("\nA credit spread's MFE is capped at 1.0x credit by construction -- the most"
          "\nit can ever be worth is the whole credit. So the profit-target question is"
          "\nonly ever 'what fraction, how early', never 'how much more'.")

    print("\n=== separation ===")
    overlap = float((win["mae_credit_mult"] >= loss["mae_credit_mult"].median()).mean())
    print(f"median loser MAE          {loss['mae_credit_mult'].median():,.2f}x credit")
    print(f"median winner MAE         {win['mae_credit_mult'].median():,.2f}x credit")
    print(f"winners reaching the median loser's depth: {overlap:.1%}")
    print(f"exit prices from real quotes: {df['exit_quote_real'].mean():.1%}")

    scan.to_csv(out.with_suffix(".stopscan.csv"), index=False)
    tbl.to_csv(out.with_suffix(".mae.csv"))
    print(f"\n-> {out.with_suffix('.stopscan.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
