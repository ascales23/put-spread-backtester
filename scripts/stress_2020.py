#!/usr/bin/env python
"""Replay the February-March 2020 crash through a levered book.

Leverage produced one number nothing had tested: at 4x, up to 47% of equity sits
simultaneously at max loss. For a book of defined-risk verticals that is the
arithmetic worst case -- if every open spread finishes below its long strike on the
same day, that is the loss. It is not a tail estimate, and this is the historical
event most likely to realize it: correlated large-cap names, all gapping together,
entered precisely because they were falling toward support.

SPY fell 34.1% peak to trough over this window and the chain data samples 19 sessions
inside it, including 24 Feb, 9 Mar, 16 Mar and the 23 Mar bottom.

The ML tail-risk filter CANNOT participate: 2020 is inside its burn-in, so it has no
eligible training data. That is not protection, it is an accident of where the sample
starts, and this test therefore runs the strategy UNFILTERED -- which is also the
conservative reading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from putspread.config import shipped_config                     # noqa: E402
from putspread.fills import FillConfig                          # noqa: E402
from putspread.metrics import compute_metrics                   # noqa: E402
from putspread.report import build_report                       # noqa: E402
from putspread.runner import DOLT_CAVEATS, BacktestContext, run_backtest  # noqa: E402

RATE_DIR = Path.home() / "marketdata" / "macro" / "treasury_3m" / "1day"

#: The shipped universe restricted to names with chain data in early 2020.
#: TSLA, META, COIN, PLTR and SMCI have none, so they cannot take part.
STRESS_UNIVERSE = ("AAPL", "AMD", "AMZN", "AVGO", "GOOGL", "MSFT", "MU", "NFLX", "NVDA", "SPY")

WINDOWS = {
    "crash only (Feb-Apr)": ("2020-01-22", "2020-04-30"),
    "crash + recovery (to Jun)": ("2020-01-22", "2020-06-30"),
    "full 2020": ("2020-01-22", "2020-12-31"),
}


def summarise(name: str, lev: float, window: str, r, starting: float) -> dict:
    m = compute_metrics(r.trade_frame, r.equity_curve, starting)
    eq = r.equity_curve
    util = (eq["open_risk"] / eq["equity"]).replace([float("inf"), float("-inf")], pd.NA).dropna()
    daily = eq["equity"].pct_change().dropna()
    t = r.trade_frame
    return {
        "window": window, "exits": name, "leverage": lev,
        "trades": m.n_trades, "win_rate": m.win_rate,
        "total_pnl": m.total_pnl, "return": m.total_return,
        "max_dd_pct": m.max_drawdown_pct,
        "worst_day": float(daily.min()) if len(daily) else float("nan"),
        "peak_risk_pct": float(util.max()) if len(util) else float("nan"),
        "margin_breaches": int(len(r.margin_breaches)),
        "worst_trade": m.worst_trade,
        "max_losses_hit": int((t["pnl"] <= -t["max_loss_per_contract"] * t["contracts"] + 1.0).sum())
        if not t.empty else 0,
        "final_equity": float(eq["equity"].iloc[-1]) if not eq.empty else starting,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", type=float, default=[1.0, 2.0, 4.0, 6.0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="runs/stress2020")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)

    ctx = BacktestContext.build(list(STRESS_UNIVERSE), a.data, RATE_DIR)
    fills = FillConfig(model="realistic", fraction=0.5)
    rows, keep = [], {}

    for wname, (start, end) in WINDOWS.items():
        for lev in a.levels:
            for exits, req in (("modelled OK", False), ("real quotes only", True)):
                cfg = shipped_config(leverage=lev).with_(
                    symbols=STRESS_UNIVERSE, start=start, end=end,
                    require_real_quotes_for_exit=req,
                )
                r = run_backtest(cfg, ctx, fills)
                rows.append(summarise(exits, lev, wname, r, cfg.starting_equity))
                if wname == "crash + recovery (to Jun)" and not req:
                    keep[lev] = (r, cfg)

    tbl = pd.DataFrame(rows)
    cols = ["leverage", "exits", "trades", "win_rate", "total_pnl", "return",
            "max_dd_pct", "worst_day", "peak_risk_pct", "margin_breaches",
            "max_losses_hit", "worst_trade"]
    for wname in WINDOWS:
        print(f"\n=== {wname.upper()} ===")
        print(tbl[tbl["window"] == wname][cols]
              .to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # The equity path through the crash itself, at each leverage.
    print("\n=== equity path through the crash (4x, modelled exits) ===")
    if 4.0 in keep:
        r, cfg = keep[4.0]
        eq = r.equity_curve.copy()
        eq["drawdown"] = eq["equity"] / eq["equity"].cummax() - 1.0
        eq["risk_pct"] = eq["open_risk"] / eq["equity"]
        w = eq.loc[[d for d in eq.index if pd.Timestamp("2020-02-14") <= pd.Timestamp(d)
                    <= pd.Timestamp("2020-04-15")]]
        print(w.to_string(float_format=lambda x: f"{x:,.3f}"))

    for lev, (r, cfg) in keep.items():
        build_report(r, out.parent / f"stress2020_{lev:g}x.html",
                     f"Feb-Mar 2020 stress test -- {lev:g}x leverage",
                     extra_sections={"Test design": (
                         "<p>Shipped configuration, UNFILTERED: the XGBoost tail-risk "
                         "filter has no eligible training data this early and cannot "
                         "participate. Universe restricted to the ten shipped names "
                         "with chain data in early 2020.</p>")})
    tbl.to_csv(out.with_suffix(".csv"), index=False)
    print(f"\n-> {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
