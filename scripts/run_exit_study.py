#!/usr/bin/env python
"""Take-profit and stop-loss study, run at 4x leverage with the ML filter on.

Sweeps the section 5 exit rules, validates the winner walk-forward rather than
quoting the in-sample best, and bootstraps the result. Exits are the one place where
optimizing on the full sample is most tempting and most misleading: there are few
enough combinations that the best one always looks good in hindsight.

    python scripts/run_exit_study.py --leverage 4 --out runs/exits
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from putspread.config import StrategyConfig, SupportConfig       # noqa: E402
from putspread.fills import FillConfig                           # noqa: E402
from putspread.metrics import compute_metrics                    # noqa: E402
from putspread.ml import TailRiskFilter, WalkForwardResult, walk_forward_predict  # noqa: E402
from putspread.report import build_report                        # noqa: E402
from putspread.runner import DOLT_CAVEATS, BacktestContext, run_backtest  # noqa: E402
from scripts.run_leverage import block_bootstrap                 # noqa: E402

RATE_DIR = Path.home() / "marketdata" / "macro" / "treasury_3m" / "1day"
UNIVERSE = ["AMD", "NVDA", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL",
            "NFLX", "MU", "AVGO", "SMCI", "COIN", "PLTR", "SPY"]
BEST_FIXED = dict(short_strike_method="buffer", buffer_pct=0.05, target_dte=14)

#: The section 5 exit space. Stop multiples come from the MAE analysis: the median
#: loser digs to 3.3x credit while the median winner only reaches 0.14x, so the
#: interesting band is 1.5x-3x, where losers are caught and winners are not.
EXIT_GRID = {
    "profit_target_pct": [None, 0.25, 0.50, 0.75],
    "stop_rule": ["none", "credit_multiple", "level_break"],
    "stop_credit_multiple": [1.5, 2.0, 2.5, 3.0],
    "time_stop_dte": [None, 5, 3],
}


def exit_combos() -> list[dict]:
    keys = list(EXIT_GRID)
    seen, out = set(), []
    for combo in itertools.product(*(EXIT_GRID[k] for k in keys)):
        d = dict(zip(keys, combo))
        if d["stop_rule"] != "credit_multiple":
            d["stop_credit_multiple"] = 2.0        # inert; collapse the duplicates
        key = tuple(sorted(d.items(), key=lambda kv: str(kv)))
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def label(d: dict) -> str:
    pt = "hold" if d["profit_target_pct"] is None else f"PT{int(d['profit_target_pct'] * 100)}"
    if d["stop_rule"] == "credit_multiple":
        st = f"stop{d['stop_credit_multiple']:g}x"
    elif d["stop_rule"] == "level_break":
        st = "stopLVL"
    else:
        st = "nostop"
    ts = "" if d["time_stop_dte"] is None else f"+TS{d['time_stop_dte']}"
    return f"{pt}/{st}{ts}"


def evaluate(cfg, ctx, fills, flt, start=None, end=None) -> dict:
    c = cfg
    if start:
        c = c.with_(start=start)
    if end:
        c = c.with_(end=end)
    r = run_backtest(c, ctx, fills, trade_filter=flt)
    m = compute_metrics(r.trade_frame, r.equity_curve, c.starting_equity)
    t = r.trade_frame
    return {
        "trades": m.n_trades, "win_rate": m.win_rate, "total_pnl": m.total_pnl,
        "expectancy": m.expectancy_per_trade, "profit_factor": m.profit_factor,
        "sharpe": m.sharpe, "max_dd_pct": m.max_drawdown_pct,
        "avg_win": m.avg_win, "avg_loss": m.avg_loss, "worst_trade": m.worst_trade,
        "stopped": int((t["exit_reason"] == "credit_multiple_stop").sum()) if not t.empty else 0,
        "level_broke": int((t["exit_reason"] == "level_break").sum()) if not t.empty else 0,
        "modelled_exits": float(1.0 - t["exit_quote_real"].mean()) if not t.empty else 0.0,
        "_result": r,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=UNIVERSE)
    ap.add_argument("--start", default="2020-01-22")
    ap.add_argument("--end", default="2026-08-26")
    ap.add_argument("--leverage", type=float, default=4.0)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--paths", type=int, default=2000)
    ap.add_argument("--data", default="data")
    ap.add_argument("--harvest", default="runs/selector.harvest.parquet")
    ap.add_argument("--out", default="runs/exits")
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    base = StrategyConfig(
        symbols=tuple(a.symbols), start=a.start, end=a.end,
        acknowledge_missing_oi_volume=True, support=SupportConfig(),
        leverage=a.leverage, **BEST_FIXED,
    )
    fills = FillConfig(model="realistic", fraction=0.5)
    ctx = BacktestContext.build(list(base.symbols), a.data, RATE_DIR)

    df = pd.read_parquet(a.harvest)
    for c in ("entry_date", "exit_date", "expiration"):
        df[c] = pd.to_datetime(df[c]).dt.date
    res = walk_forward_predict(df, target="big_loss", n_folds=4, min_train=200)
    fx = df[(df["cfg_is_delta_method"] == 0.0) & (df["cfg_buffer_pct"] == 0.05)
            & (df["cfg_target_dte"] == 14)]
    flt = TailRiskFilter.from_frame(fx, WalkForwardResult(
        res.predictions.loc[fx.index], res.folds, res.fold_table,
        res.importance, res.thresholds.loc[fx.index]))

    combos = exit_combos()
    print(f"sweeping {len(combos)} exit configurations at {a.leverage:g}x leverage "
          f"with the ML filter on ...\n", flush=True)

    sweep_path = out.with_suffix(".sweep.csv")
    if sweep_path.exists():
        tbl = pd.read_csv(sweep_path)
        print(f"(reusing {sweep_path})")
    else:
        rows = []
        for d in combos:
            r = evaluate(base.with_(**d), ctx, fills, flt)
            r.pop("_result")
            rows.append({"config": label(d), **d, **r})
        tbl = pd.DataFrame(rows)
        tbl.to_csv(sweep_path, index=False)

    show = ["config", "trades", "win_rate", "total_pnl", "expectancy", "profit_factor",
            "sharpe", "max_dd_pct", "avg_win", "avg_loss", "stopped"]
    print("=== best 15 by total P&L (IN-SAMPLE -- a diagnostic, not a result) ===")
    print(tbl.sort_values("total_pnl", ascending=False).head(15)[show]
          .to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\n=== best 10 by Sharpe ===")
    print(tbl.sort_values("sharpe", ascending=False).head(10)[show]
          .to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
    print("\n=== shallowest 10 drawdowns ===")
    print(tbl.sort_values("max_dd_pct", ascending=False).head(10)[show]
          .to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    # ---- walk-forward: choose the exit config in-sample, measure it out-of-sample
    print(f"\n=== walk-forward over {a.folds} folds (choose on train, measure on test) ===")
    edges = pd.date_range(a.start, a.end, periods=a.folds + 2)
    wf = []
    for i in range(a.folds):
        tr_s, tr_e = str(edges[0].date()), str(edges[i + 1].date())
        te_s, te_e = str(edges[i + 1].date()), str(edges[i + 2].date())
        scored = []
        for d in combos:
            m = evaluate(base.with_(**d), ctx, fills, flt, tr_s, tr_e)
            m.pop("_result")
            if m["trades"] >= 15:
                scored.append((m["sharpe"], d, m))
        if not scored:
            continue
        best_sharpe, best_cfg, best_m = max(scored, key=lambda x: x[0])
        oos = evaluate(base.with_(**best_cfg), ctx, fills, flt, te_s, te_e)
        oos.pop("_result")
        wf.append({
            "fold": i + 1, "train_end": tr_e, "test": f"{te_s}..{te_e}",
            "chosen": label(best_cfg), "is_sharpe": best_sharpe,
            "is_pnl": best_m["total_pnl"], "oos_trades": oos["trades"],
            "oos_pnl": oos["total_pnl"], "oos_sharpe": oos["sharpe"],
            "oos_dd": oos["max_dd_pct"],
        })
    wf_tbl = pd.DataFrame(wf)
    print(wf_tbl.to_string(index=False, float_format=lambda x: f"{x:,.3f}")
          if not wf_tbl.empty else "no fold produced enough trades")

    # ---- head-to-head against the current exits, plus the MAE-implied stop
    print("\n=== head to head, full sample, 4x, ML filter ===")
    contenders = {
        "current: PT50 / no stop": dict(profit_target_pct=0.50, stop_rule="none",
                                        time_stop_dte=None),
        "PT50 + stop 2x credit": dict(profit_target_pct=0.50, stop_rule="credit_multiple",
                                      stop_credit_multiple=2.0, time_stop_dte=None),
        "PT50 + stop 2.5x credit": dict(profit_target_pct=0.50, stop_rule="credit_multiple",
                                        stop_credit_multiple=2.5, time_stop_dte=None),
        "PT50 + level-break stop": dict(profit_target_pct=0.50, stop_rule="level_break",
                                        time_stop_dte=None),
        "hold to expiry, no stop": dict(profit_target_pct=None, stop_rule="none",
                                        time_stop_dte=None),
    }
    if not wf_tbl.empty:
        chosen = wf_tbl["chosen"].mode()
        if len(chosen):
            print(f"(walk-forward most often chose: {chosen.iloc[0]})")

    hh, keep = [], {}
    for name, d in contenders.items():
        full = {"stop_credit_multiple": 2.0, **d}   # d wins where it sets the multiple
        m = evaluate(base.with_(**full), ctx, fills, flt)
        keep[name] = m.pop("_result")
        hh.append({"config": name, **m})
    hh_tbl = pd.DataFrame(hh)
    print(hh_tbl[["config", "trades", "win_rate", "total_pnl", "expectancy",
                  "profit_factor", "sharpe", "max_dd_pct", "avg_win", "avg_loss",
                  "worst_trade", "stopped", "modelled_exits"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print(f"\n=== block bootstrap, {a.paths:,} reorderings, at {a.leverage:g}x ===")
    bs_rows = []
    for name, r in keep.items():
        eq = r.equity_curve
        rets = eq["equity"].pct_change().dropna().to_numpy()
        bs = block_bootstrap(rets, a.paths, 10, 0)
        if bs.empty:
            continue
        bs_rows.append({
            "config": name,
            "median_return": bs["final_return"].median(),
            "p05_return": bs["final_return"].quantile(0.05),
            "median_dd": bs["max_dd"].median(),
            "worst5_dd": bs["max_dd"].quantile(0.05),
            "p_dd_over_25": float((bs["max_dd"] < -0.25).mean()),
            "ruin": float(bs["ruin_rate"].iloc[0]),
        })
    print(pd.DataFrame(bs_rows).to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    if not wf_tbl.empty:
        wf_tbl.to_csv(out.with_suffix(".walkforward.csv"), index=False)
    hh_tbl.to_csv(out.with_suffix(".headtohead.csv"), index=False)
    for name, r in keep.items():
        if "2x credit" in name or "no stop" in name:
            slug = name.replace(" ", "_").replace("/", "").replace(":", "")
            build_report(r, out.parent / f"exits_{slug}.html", f"4x leverage -- {name}")
    print(f"\n-> {out.with_suffix('.sweep.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
