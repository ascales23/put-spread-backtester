#!/usr/bin/env python
"""What leverage does to this strategy.

Runs the full backtest across a leverage ladder, with and without the XGBoost
tail-risk filter, and reports the things that decide whether leverage is survivable
rather than just the things that make it look good.

    python scripts/run_leverage.py --out runs/leverage

Two numbers matter more than P&L here:

  peak_risk_pct -- the largest fraction of equity that was simultaneously at max
      loss. For a book of defined-risk verticals this IS the worst case: if every
      open spread finished below its long strike on the same day, that is what the
      account loses. It is not a tail estimate; it is arithmetic.

  margin_breaches -- sessions where open max loss exceeded equity. Sizing cannot
      create these; only losses can, by shrinking equity under positions already on.
      A real account is liquidated here, at the worst possible moment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from putspread.config import StrategyConfig, SupportConfig       # noqa: E402
from putspread.engine import Backtester                          # noqa: E402
from putspread.fills import FillConfig                           # noqa: E402
from putspread.metrics import compute_metrics                    # noqa: E402
from putspread.ml import TailRiskFilter, WalkForwardResult, walk_forward_predict  # noqa: E402
from putspread.report import build_report                        # noqa: E402
from putspread.runner import DOLT_CAVEATS, BacktestContext       # noqa: E402

RATE_DIR = Path.home() / "marketdata" / "macro" / "treasury_3m" / "1day"
UNIVERSE = ["AMD", "NVDA", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL",
            "NFLX", "MU", "AVGO", "SMCI", "COIN", "PLTR", "SPY"]
BEST_FIXED = dict(short_strike_method="buffer", buffer_pct=0.05, target_dte=14)


def block_bootstrap(returns: np.ndarray, n_paths: int, block: int, seed: int) -> pd.DataFrame:
    """Resample daily returns in blocks and recompound.

    Blocks, not single days: losses in this strategy arrive in clusters (one bad
    tape takes out several open spreads at once), and an iid bootstrap would break
    exactly the dependence that creates the tail.

    The realized path is one draw from this distribution. Quoting it alone is how a
    lucky ordering gets mistaken for an edge.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    if n < block * 2:
        return pd.DataFrame()
    n_blocks = int(np.ceil(n / block))
    finals, dds, ruins = [], [], 0
    for _ in range(n_paths):
        starts = rng.integers(0, n - block, size=n_blocks)
        path = np.concatenate([returns[s:s + block] for s in starts])[:n]
        eq = np.cumprod(1.0 + path)
        if np.any(eq <= 0.0):
            ruins += 1
            finals.append(-1.0)
            dds.append(-1.0)
            continue
        peak = np.maximum.accumulate(eq)
        dds.append(float((eq / peak - 1.0).min()))
        finals.append(float(eq[-1] - 1.0))
    return pd.DataFrame({"final_return": finals, "max_dd": dds}).assign(ruin_rate=ruins / n_paths)


def analyse(name: str, lev: float, result, starting: float, n_paths: int, seed: int) -> dict:
    m = compute_metrics(result.trade_frame, result.equity_curve, starting)
    eq = result.equity_curve
    rets = eq["equity"].pct_change().dropna().to_numpy() if not eq.empty else np.array([])
    util = (eq["open_risk"] / eq["equity"]).replace([np.inf, -np.inf], np.nan).dropna() \
        if not eq.empty else pd.Series(dtype=float)
    bs = block_bootstrap(rets, n_paths, 10, seed) if len(rets) > 40 else pd.DataFrame()

    row = {
        "run": name, "leverage": lev, "trades": m.n_trades,
        "total_pnl": m.total_pnl, "total_return": m.total_return, "cagr": m.cagr,
        "max_dd_pct": m.max_drawdown_pct, "sharpe": m.sharpe,
        "worst_trade": m.worst_trade,
        "peak_risk_pct": float(util.max()) if len(util) else np.nan,
        "mean_risk_pct": float(util.mean()) if len(util) else np.nan,
        "margin_breaches": int(len(result.margin_breaches)),
    }
    if not bs.empty:
        row.update({
            "bs_median_return": float(bs["final_return"].median()),
            "bs_p05_return": float(bs["final_return"].quantile(0.05)),
            "bs_median_dd": float(bs["max_dd"].median()),
            "bs_p95_dd": float(bs["max_dd"].quantile(0.05)),   # worst 5% of drawdowns
            "bs_p_dd_over_25": float((bs["max_dd"] < -0.25).mean()),
            "bs_ruin_rate": float(bs["ruin_rate"].iloc[0]),
        })
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=UNIVERSE)
    p.add_argument("--start", default="2020-01-22")
    p.add_argument("--end", default="2026-08-26")
    p.add_argument("--levels", nargs="+", type=float,
                   default=[1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0])
    p.add_argument("--paths", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="data")
    p.add_argument("--harvest", default="runs/selector.harvest.parquet")
    p.add_argument("--out", default="runs/leverage")
    a = p.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    base = StrategyConfig(
        symbols=tuple(a.symbols), start=a.start, end=a.end, profit_target_pct=0.5,
        acknowledge_missing_oi_volume=True, support=SupportConfig(), **BEST_FIXED,
    )
    fills = FillConfig(model="realistic", fraction=0.5)
    ctx = BacktestContext.build(list(base.symbols), a.data, RATE_DIR)

    df = pd.read_parquet(a.harvest)
    for c in ("entry_date", "exit_date", "expiration"):
        df[c] = pd.to_datetime(df[c]).dt.date
    res = walk_forward_predict(df, target="big_loss", n_folds=4, min_train=200)
    fx = df[(df["cfg_is_delta_method"] == 0.0) & (df["cfg_buffer_pct"] == 0.05)
            & (df["cfg_target_dte"] == 14)]
    tail_filter = TailRiskFilter.from_frame(fx, WalkForwardResult(
        res.predictions.loc[fx.index], res.folds, res.fold_table,
        res.importance, res.thresholds.loc[fx.index]))

    rows, results = [], {}
    for lev in a.levels:
        for label, flt in (("unfiltered", None), ("ML filtered", tail_filter)):
            cfg = base.with_(leverage=lev)
            r = Backtester(ctx.provider, ctx.bars, cfg, fills, ctx.rates, ctx.calendar,
                           list(DOLT_CAVEATS), trade_filter=flt).run()
            rows.append(analyse(label, lev, r, cfg.starting_equity, a.paths, a.seed))
            results[(label, lev)] = r
        print(f"  {lev:>5.1f}x done", flush=True)

    tbl = pd.DataFrame(rows)
    pd.set_option("display.width", 260)

    for label in ("unfiltered", "ML filtered"):
        sub = tbl[tbl["run"] == label]
        print(f"\n=== {label.upper()} ===")
        print(sub[["leverage", "trades", "total_pnl", "total_return", "cagr",
                   "max_dd_pct", "sharpe", "peak_risk_pct", "margin_breaches"]]
              .to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
        print(f"\n  block-bootstrap over {a.paths:,} reorderings of the daily path:")
        print(sub[["leverage", "bs_median_return", "bs_p05_return", "bs_median_dd",
                   "bs_p95_dd", "bs_p_dd_over_25", "bs_ruin_rate"]]
              .to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    tbl.to_csv(out.with_suffix(".csv"), index=False)
    for lev in (1.0, 4.0):
        if ("ML filtered", lev) in results:
            build_report(results[("ML filtered", lev)],
                         out.parent / f"{out.name}_filtered_{lev:g}x.html",
                         f"Bull Put Spread -- ML filtered, {lev:g}x leverage")
    print(f"\ntable -> {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
