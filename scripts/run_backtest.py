#!/usr/bin/env python
"""Run one backtest and write an HTML report.

    python scripts/run_backtest.py --symbols AMD NVDA --dte 22 --buffer 0.03 \
        --profit-target 0.5 --out runs/amd_nvda.html
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from putspread.config import StrategyConfig, SupportConfig          # noqa: E402
from putspread.fills import FillConfig                              # noqa: E402
from putspread.metrics import compute_metrics                       # noqa: E402
from putspread.report import build_report                           # noqa: E402
from putspread.runner import BacktestContext, fill_sensitivity_table, run_backtest  # noqa: E402

RATE_DIR = Path.home() / "marketdata" / "macro" / "treasury_3m" / "1day"


def build_cfg(a) -> StrategyConfig:
    return StrategyConfig(
        symbols=tuple(a.symbols), start=a.start, end=a.end,
        entry_discipline=a.entry, short_strike_method=a.method,
        buffer_pct=a.buffer, target_short_delta=a.delta,
        max_loss_per_contract=a.max_loss, target_dte=a.dte, dte_tolerance=a.dte_tol,
        profit_target_pct=a.profit_target, stop_rule=a.stop, time_stop_dte=a.time_stop,
        starting_equity=a.equity, max_portfolio_risk_pct=a.portfolio_risk,
        max_risk_per_position_pct=a.position_risk,
        acknowledge_missing_oi_volume=True,     # this source carries no OI/volume
        support=SupportConfig(method=a.support, lookback_days=a.support_lookback),
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", default="2020-01-22")
    p.add_argument("--end", default="2026-08-26")
    p.add_argument("--entry", default="mechanical", choices=["mechanical", "confirmation"])
    p.add_argument("--method", default="buffer", choices=["buffer", "delta", "prob_otm"])
    p.add_argument("--buffer", type=float, default=0.03)
    p.add_argument("--delta", type=float, default=0.20)
    p.add_argument("--max-loss", type=float, default=1500.0)
    p.add_argument("--dte", type=int, default=22)
    p.add_argument("--dte-tol", type=int, default=7)
    p.add_argument("--profit-target", type=float, default=None)
    p.add_argument("--stop", default="none", choices=["none", "level_break", "credit_multiple"])
    p.add_argument("--time-stop", type=int, default=None)
    p.add_argument("--support", default="pivot_low", choices=["pivot_low", "donchian", "sma"])
    p.add_argument("--support-lookback", type=int, default=120)
    p.add_argument("--equity", type=float, default=100_000.0)
    p.add_argument("--portfolio-risk", type=float, default=0.10)
    p.add_argument("--position-risk", type=float, default=0.02)
    p.add_argument("--fill", default="realistic", choices=["mid", "realistic", "natural"])
    p.add_argument("--fill-fraction", type=float, default=0.5)
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/backtest.html")
    p.add_argument("--no-sensitivity", action="store_true")
    a = p.parse_args()

    cfg = build_cfg(a)
    ctx = BacktestContext.build(list(cfg.symbols), a.data, RATE_DIR)
    fills = FillConfig(model=a.fill, fraction=a.fill_fraction)

    res = run_backtest(cfg, ctx, fills)
    m = compute_metrics(res.trade_frame, res.equity_curve, cfg.starting_equity)
    sens = None if a.no_sensitivity else fill_sensitivity_table(cfg, ctx)

    out = build_report(res, a.out, f"Bull Put Spread -- {', '.join(cfg.symbols)}", sens)
    print(f"\nsignals {res.signals_seen}  trades {m.n_trades}  win {m.win_rate:.1%}  "
          f"P&L ${m.total_pnl:,.0f}  expectancy ${m.expectancy_per_trade:,.0f}/trade  "
          f"PF {m.profit_factor:.2f}  Sharpe {m.sharpe:.2f}  maxDD {m.max_drawdown_pct:.1%}")
    if sens is not None:
        print("\nfill sensitivity:\n" + sens.to_string(float_format=lambda x: f"{x:,.3f}"))
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
