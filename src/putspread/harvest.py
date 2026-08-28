"""Harvest every candidate trade and its realized outcome, for model training.

Deliberately runs WITHOUT portfolio constraints. The backtest can only take a trade
when capital and the one-per-symbol rule allow it, so its trade log is a biased
sample of the opportunity set -- biased by a queueing rule, not by anything the
market did. A filter trained on that log would learn the queue.

Each candidate is simulated in isolation at one contract, through the engine's own
exit logic, so a harvested outcome and a backtested one cannot drift apart.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from .config import StrategyConfig
from .engine import Backtester
from .evaluate import find_candidate
from .features import FEATURE_COLUMNS, atm_iv_series, build_features, load_vix, term_slope
from .fills import FillConfig
from .portfolio import OpenPosition
from .runner import BacktestContext
from .support import entry_signals, support_series

LABEL_COLUMNS = ["pnl", "ror", "win", "big_loss"]
META_COLUMNS = [
    "symbol", "entry_date", "exit_date", "expiration", "exit_reason",
    "short_strike", "long_strike", "credit_per_share", "max_loss_per_contract",
]


def _days_to_next_earnings(calendar, symbol: str, d: date) -> float:
    if calendar is None:
        return float("nan")
    ev = calendar.symbol_events(symbol)
    future = [x for x in ev["date"].tolist() if x > d]
    return float((min(future) - d).days) if future else float("nan")


def harvest(
    ctx: BacktestContext,
    cfg: StrategyConfig,
    fills: FillConfig,
    vix_dir: str | None = None,
) -> pd.DataFrame:
    """One row per candidate spread: entry-time features plus the realized outcome."""
    vix = load_vix(vix_dir) if vix_dir else pd.Series(dtype=float)
    start = pd.Timestamp(cfg.start).date()
    end = pd.Timestamp(cfg.end).date()
    rows = []

    for sym in cfg.symbols:
        bars = ctx.bars[sym]
        sup = support_series(bars, cfg.support)
        sig = entry_signals(bars, sup, cfg.entry_discipline, cfg.confirmation_bars)
        atm = atm_iv_series(ctx.provider, sym)

        engine = Backtester(
            provider=ctx.provider, bars={sym: bars}, cfg=cfg.with_(symbols=(sym,)),
            fills=fills, rates=ctx.rates, calendar=ctx.calendar,
        )

        for d in bars.index:
            if not (start <= d <= end) or d not in sig.index or not bool(sig.loc[d]):
                continue
            level = float(sup.loc[d])
            if level != level:
                continue
            dev = ctx.provider.parity_deviation(sym, d)
            if dev is not None and dev > cfg.max_parity_deviation:
                continue

            r = ctx.rates.get(d)
            cand = find_candidate(sym, d, level, ctx.provider, cfg, fills, r, ctx.calendar)
            if not cand.accepted or cand.spread is None:
                continue
            s = cand.spread
            spot = s.spot

            pos = OpenPosition(
                symbol=sym, entry_date=d, expiration=cand.diagnostics["expiration"],
                short_strike=s.short.strike, long_strike=s.long.strike,
                credit_per_share=s.credit_per_share, contracts=1,
                max_loss_per_contract=s.max_loss_per_contract, entry_spot=spot,
                support_level=level, short_iv=s.short_iv, long_iv=s.long_iv,
                entry_commission=fills.commissions_per_contract(2, 1),
                entry_delta=s.net_greeks.delta, entry_vega=s.net_greeks.vega,
                entry_theta=s.net_greeks.theta,
                move_sigma=float(cand.diagnostics.get("move_sigma", float("nan"))),
                prob_otm_display=float(
                    cand.diagnostics.get("prob_otm_risk_neutral_display", float("nan"))
                ),
            )
            trade = engine.simulate_isolated(pos)
            if trade is None:
                continue      # data ends before this position resolves

            feats = build_features(
                symbol=sym, d=d, spot=spot, support_level=level,
                candidate_diagnostics=cand.diagnostics,
                short_strike=s.short.strike, long_strike=s.long.strike,
                credit_per_share=s.credit_per_share,
                max_loss_per_contract=s.max_loss_per_contract,
                short_iv=s.short_iv, long_iv=s.long_iv, dte=s.expiration_days,
                spread_pct_short=s.short.spread_pct_of_mid,
                spread_pct_long=s.long.spread_pct_of_mid,
                bars=bars, atm_iv=atm, vix=vix,
                term=term_slope(ctx.provider, sym, d, spot),
                days_to_next_earnings=_days_to_next_earnings(ctx.calendar, sym, d),
                cfg_buffer_pct=cfg.buffer_pct,
                cfg_target_dte=float(cfg.target_dte),
                cfg_target_delta=cfg.target_short_delta,
                cfg_is_delta_method=float(cfg.short_strike_method == "delta"),
            )
            rows.append({
                "symbol": sym, "entry_date": d, "exit_date": trade.exit_date,
                "expiration": trade.expiration, "exit_reason": trade.exit_reason,
                "short_strike": trade.short_strike, "long_strike": trade.long_strike,
                "credit_per_share": trade.credit_per_share,
                "max_loss_per_contract": trade.max_loss_per_contract,
                "pnl": trade.pnl,
                "ror": trade.pnl / trade.max_loss_per_contract,
                "win": int(trade.pnl > 0),
                # The strategy dies from its left tail, so the tail gets its own label.
                "big_loss": int(trade.pnl < -0.5 * trade.max_loss_per_contract),
                **feats,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("entry_date").reset_index(drop=True)


#: Entry structures to harvest. Only ENTRY-side knobs vary: they change which spread
#: is constructed and nothing else, so a candidate under one structure is directly
#: comparable to the same day's candidate under another. Exit rules stay fixed at the
#: strategy level, because they act on a position over time rather than at entry.
ENTRY_GRID = [
    {"short_strike_method": "buffer", "buffer_pct": b, "target_dte": t}
    for b in (0.02, 0.03, 0.05, 0.08)
    for t in (7, 14, 22, 45)
] + [
    {"short_strike_method": "delta", "target_short_delta": dl, "target_dte": t}
    for dl in (0.10, 0.20, 0.30)
    for t in (7, 14, 22, 45)
]


def harvest_grid(
    ctx: BacktestContext,
    cfg: StrategyConfig,
    fills: FillConfig,
    grid: list[dict] | None = None,
    vix_dir: str | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Harvest every candidate under every entry structure in `grid`.

    Produces one row per (signal date, structure). The same underlying setup appears
    many times with different strikes and expiries, which is exactly the comparison
    the model needs in order to learn which structure suits which setup.
    """
    grid = grid if grid is not None else ENTRY_GRID
    frames = []
    for i, overrides in enumerate(grid, 1):
        variant = cfg.with_(**overrides)
        df = harvest(ctx, variant, fills, vix_dir)
        if progress:
            label = ",".join(f"{k}={v}" for k, v in overrides.items())
            print(f"  [{i}/{len(grid)}] {label:58s} {len(df):>5,} candidates", flush=True)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("entry_date").reset_index(drop=True)
