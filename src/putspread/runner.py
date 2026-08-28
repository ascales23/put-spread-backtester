"""Wiring: build a ready-to-run backtest context from the parquet data directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .config import StrategyConfig
from .earnings import EarningsCalendar
from .engine import BacktestResult, Backtester
from .fills import FillConfig
from .providers.parquet_chain import ParquetChainProvider
from .rates import RateCurve

#: Caveats that apply to every run against this data source. Reported, never buried.
DOLT_CAVEATS = [
    "Chain data: post-no-preference/options on DoltHub -- REAL end-of-day bid/ask and "
    "per-strike IV for US equity options, 2019-02-09 onward, updated daily.",
    "This source carries NO open interest and NO volume, so the STRATEGY.md section 6.3 "
    "OI/volume filters cannot be evaluated. Liquidity is screened on bid/ask width and a "
    "minimum bid only.",
    "Quotes are end-of-day marks, not executable prints. Entries and exits are priced at "
    "the close of the signal day.",
    "Earnings dates: post-no-preference/earnings, real reported dates with "
    "before/after-market timing, coverage begins 2020-01-22.",
    "Underlying prices are recovered from the chains by put-call parity, so they are raw "
    "and split-consistent with the strikes.",
]


@dataclass
class BacktestContext:
    """Everything a run needs except the parameter set itself."""

    provider: ParquetChainProvider
    bars: dict[str, pd.DataFrame]
    rates: RateCurve
    calendar: EarningsCalendar
    caveats: list[str] = field(default_factory=lambda: list(DOLT_CAVEATS))

    @classmethod
    def build(
        cls,
        symbols: list[str],
        data_dir: str | Path = "data",
        rate_dir: str | Path | None = None,
    ) -> "BacktestContext":
        data_dir = Path(data_dir)
        rates = (
            RateCurve.from_parquet_dir(rate_dir) if rate_dir and Path(rate_dir).exists()
            else RateCurve.constant(0.04)
        )
        provider = ParquetChainProvider(
            data_dir / "chains", rates=rates, ohlcv_dir=data_dir / "ohlcv"
        )
        bars = {s: provider.daily_bars(s) for s in symbols}
        calendar = EarningsCalendar.from_parquet(data_dir / "earnings.parquet")
        return cls(provider=provider, bars=bars, rates=rates, calendar=calendar)


def run_backtest(
    cfg: StrategyConfig, ctx: BacktestContext, fills: FillConfig,
    trade_filter=None, entry_selector=None,
) -> BacktestResult:
    """One parameter set, one fill assumption, one pass over the path."""
    return Backtester(
        provider=ctx.provider, bars={s: ctx.bars[s] for s in cfg.symbols}, cfg=cfg,
        fills=fills, rates=ctx.rates, calendar=ctx.calendar, data_caveats=list(ctx.caveats),
        trade_filter=trade_filter, entry_selector=entry_selector,
    ).run()


def fill_sensitivity_table(cfg: StrategyConfig, ctx: BacktestContext) -> pd.DataFrame:
    """Section 9's decisive comparison: the same signals under three executions.

    If the edge only survives at mid, it is an artifact of the fill assumption.
    """
    from .metrics import compute_metrics

    rows = []
    for model, frac in (("mid", 0.0), ("realistic", 0.5), ("natural", 1.0)):
        f = FillConfig(model=model, fraction=frac,
                       commission_per_contract=cfg.commission_per_contract)
        res = run_backtest(cfg, ctx, f)
        m = compute_metrics(res.trade_frame, res.equity_curve, cfg.starting_equity)
        rows.append({
            "fill_model": model, "trades": m.n_trades, "win_rate": m.win_rate,
            "total_pnl": m.total_pnl, "expectancy_per_trade": m.expectancy_per_trade,
            "profit_factor": m.profit_factor, "sharpe": m.sharpe,
            "max_drawdown_pct": m.max_drawdown_pct,
        })
    return pd.DataFrame(rows).set_index("fill_model")
