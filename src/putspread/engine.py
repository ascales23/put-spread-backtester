"""Slice 3 + 4 -- the backtest engine.

The central discipline of this module, from STRATEGY.md section 7 and the build
prompt: **P&L comes from the actual historical price path, gaps included.**
Black-Scholes appears here in exactly one place -- a fallback daily mark when a
strike is not quoted on some day and an early-exit rule needs a number. It never
generates the path, and realized P&L at expiry is pure intrinsic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .chain import ChainProvider, ExpiryChain, year_fraction
from .config import StrategyConfig
from .earnings import EarningsCalendar
from .evaluate import find_candidate
from .exits import MarkState, evaluate_exits
from .fills import FillConfig
from .portfolio import ClosedTrade, OpenPosition, Portfolio
from .rates import RateCurve
from .spread import CONTRACT_MULTIPLIER, LegQuote, payoff_at_expiry_per_contract, spread_mark_per_contract
from .support import entry_signals, support_series


@dataclass
class BacktestResult:
    """Everything one run produced."""

    cfg: StrategyConfig
    fills: FillConfig
    trades: list[ClosedTrade]
    equity_curve: pd.DataFrame
    rejections: pd.DataFrame
    data_caveats: list[str] = field(default_factory=list)
    signals_seen: int = 0

    @property
    def trade_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.trades])


class Backtester:
    """Walks the historical path for a set of symbols under one parameter set."""

    def __init__(
        self,
        provider: ChainProvider,
        bars: dict[str, pd.DataFrame],
        cfg: StrategyConfig,
        fills: FillConfig,
        rates: RateCurve,
        calendar: EarningsCalendar | None,
        data_caveats: list[str] | None = None,
    ) -> None:
        self.provider = provider
        self.bars = bars
        self.cfg = cfg
        self.fills = fills
        self.rates = rates
        self.calendar = calendar
        self.data_caveats = list(data_caveats or [])
        self.portfolio = Portfolio(cfg)

    # ------------------------------------------------------------------ marks

    def _leg_quotes(
        self, symbol: str, d: date, pos: OpenPosition
    ) -> tuple[LegQuote, LegQuote] | None:
        chain: ExpiryChain | None = self.provider.chain(symbol, d, pos.expiration)
        if chain is None:
            return None
        s, l = chain.puts.get(pos.short_strike), chain.puts.get(pos.long_strike)
        if s is None or l is None:
            return None
        return s, l

    def mark_position(self, pos: OpenPosition, d: date, spot: float) -> MarkState:
        """Debit to close today, from real quotes when they exist.

        Real EOD quotes are strictly better than a model mark: they carry the actual
        skew and the actual bid/ask the trade would have to cross. Black-Scholes is
        the fallback only, and the returned MarkState says which one was used so no
        downstream rule can mistake a model number for a market number.
        """
        dte = (pos.expiration - d).days
        legs = self._leg_quotes(pos.symbol, d, pos)
        if legs is not None:
            short_q, long_q = legs
            return MarkState(
                spot=spot,
                debit_to_close=self.fills.close_debit(short_q, long_q),
                short_mid=short_q.mid,
                days_to_expiry=dte,
                quotes_are_real=True,
            )
        r = self.rates.get(d)
        T = year_fraction(d, pos.expiration)
        model = spread_mark_per_contract(
            pos.short_strike, pos.long_strike, spot, T, r, pos.short_iv, pos.long_iv
        ) / CONTRACT_MULTIPLIER
        return MarkState(
            spot=spot, debit_to_close=max(model, 0.0), short_mid=None,
            days_to_expiry=dte, quotes_are_real=False,
        )

    # ------------------------------------------------------------------ exits

    def _settle_at_expiry(self, pos: OpenPosition, d: date, spot: float) -> ClosedTrade:
        """Realized P&L at expiry is INTRINSIC ONLY -- no vol assumption, no model.

        The short expires worthless above the strike and costs nothing to close, so
        no exit commission or slippage is charged there. In the money, the position
        settles at intrinsic and one round of closing costs applies, matching how a
        broker auto-exercises an ITM vertical.
        """
        pnl_per_contract = payoff_at_expiry_per_contract(
            spot, pos.short_strike, pos.long_strike, pos.credit_per_share
        )
        settled_itm = spot < pos.short_strike
        exit_commission = (
            self.fills.commissions_per_contract(2, 1) * pos.contracts if settled_itm else 0.0
        )
        debit = pos.credit_per_share - pnl_per_contract / CONTRACT_MULTIPLIER
        return self.portfolio.close_position(
            pos, d, debit, spot, "expiry", exit_commission
        )

    def _close_early(
        self, pos: OpenPosition, d: date, mark: MarkState, reason: str
    ) -> ClosedTrade:
        exit_commission = self.fills.commissions_per_contract(2, 1) * pos.contracts
        return self.portfolio.close_position(
            pos, d, mark.debit_to_close, mark.spot, reason, exit_commission
        )

    # ------------------------------------------------------------------ run

    def run(self) -> BacktestResult:
        start = pd.Timestamp(self.cfg.start).date()
        end = pd.Timestamp(self.cfg.end).date()

        # Mandatory earnings coverage check, once, before anything trades.
        if self.cfg.require_earnings_data:
            if self.calendar is None:
                raise RuntimeError(
                    "require_earnings_data=True but no earnings calendar was supplied"
                )
            for sym in self.cfg.symbols:
                self.calendar.assert_covers(sym, start, end)
        else:
            self.data_caveats.append(
                "EARNINGS FILTER DISABLED -- trades may span earnings reports."
            )

        signals: dict[str, pd.Series] = {}
        supports: dict[str, pd.Series] = {}
        for sym in self.cfg.symbols:
            b = self.bars[sym]
            sup = support_series(b, self.cfg.support)
            supports[sym] = sup
            signals[sym] = entry_signals(
                b, sup, self.cfg.entry_discipline, self.cfg.confirmation_bars
            )

        all_dates = sorted({d for sym in self.cfg.symbols for d in self.bars[sym].index})
        all_dates = [d for d in all_dates if start <= d <= end]
        signals_seen = 0

        for d in all_dates:
            # 1. Exits first: capital freed today is available for today's entries,
            #    which is how a real account behaves.
            for pos in list(self.portfolio.positions):
                spot = self._spot_for(pos.symbol, d)
                if spot is None:
                    continue
                mark = self.mark_position(pos, d, spot)
                open_pnl = (pos.credit_per_share - mark.debit_to_close) * CONTRACT_MULTIPLIER * pos.contracts
                pos.worst_mark = min(pos.worst_mark, open_pnl)
                pos.best_mark = max(pos.best_mark, open_pnl)

                if d >= pos.expiration:
                    # Settle on the EXPIRATION date's close, not today's. If the
                    # expiry fell on a day this symbol had no bar, today could be
                    # several sessions later and the underlying will have moved --
                    # booking that move into an already-settled position would be
                    # pure lookahead.
                    settle_spot = self._spot_for(pos.symbol, pos.expiration) or spot
                    self._settle_at_expiry(pos, pos.expiration, settle_spot)
                    continue
                reason = evaluate_exits(pos, mark, self.cfg)
                if reason:
                    self._close_early(pos, d, mark, reason)

            # 2. Entries.
            for sym in self.cfg.symbols:
                sig = signals[sym]
                if d not in sig.index or not bool(sig.loc[d]):
                    continue
                signals_seen += 1
                ok, why = self.portfolio.can_open(sym)
                if not ok:
                    self.portfolio.rejections.append((d, sym, why or "blocked"))
                    continue
                self._try_open(sym, d, float(supports[sym].loc[d]))

            # 3. Mark the book.
            open_pnl = 0.0
            for pos in self.portfolio.positions:
                spot = self._spot_for(pos.symbol, d)
                if spot is None:
                    continue
                m = self.mark_position(pos, d, spot)
                open_pnl += (pos.credit_per_share - m.debit_to_close) * CONTRACT_MULTIPLIER * pos.contracts
            self.portfolio.record_equity(d, open_pnl)

        eq = pd.DataFrame(
            self.portfolio.equity_curve, columns=["date", "equity", "open_risk"]
        ).set_index("date")
        rej = pd.DataFrame(self.portfolio.rejections, columns=["date", "symbol", "reason"])
        return BacktestResult(
            cfg=self.cfg, fills=self.fills, trades=self.portfolio.closed,
            equity_curve=eq, rejections=rej, data_caveats=self.data_caveats,
            signals_seen=signals_seen,
        )

    # ------------------------------------------------------------------ helpers

    def _spot_for(self, symbol: str, d: date) -> float | None:
        """RAW close for `symbol` on `d`, or the most recent prior close.

        Carrying the last close forward through a missing bar is right for a position
        that is still open; it is never used to create an entry, which requires a
        live chain on the day anyway.
        """
        b = self.bars.get(symbol)
        if b is None or b.empty:
            return None
        if d in b.index:
            return float(b.loc[d, "close"])
        prior = b.index[b.index <= d]
        return float(b.loc[prior[-1], "close"]) if len(prior) else None

    def _try_open(self, symbol: str, d: date, support_level: float) -> None:
        if support_level != support_level:  # NaN
            return
        r = self.rates.get(d)
        cand = find_candidate(
            symbol, d, support_level, self.provider, self.cfg, self.fills, r,
            self.calendar,
        )
        if not cand.accepted or cand.spread is None:
            self.portfolio.rejections.append((d, symbol, cand.reject_reason or "rejected"))
            return

        s = cand.spread
        equity = self.portfolio.equity()
        contracts = self.portfolio.size_position(s.max_loss_per_contract, equity)
        if contracts < 1:
            self.portfolio.rejections.append(
                (d, symbol, f"sizing allows 0 contracts at ${s.max_loss_per_contract:,.0f} risk")
            )
            return

        entry_commission = self.fills.commissions_per_contract(2, 1) * contracts
        self.portfolio.open_position(
            OpenPosition(
                symbol=symbol, entry_date=d, expiration=cand.diagnostics["expiration"],
                short_strike=s.short.strike, long_strike=s.long.strike,
                credit_per_share=s.credit_per_share, contracts=contracts,
                max_loss_per_contract=s.max_loss_per_contract, entry_spot=s.spot,
                support_level=support_level, short_iv=s.short_iv, long_iv=s.long_iv,
                entry_commission=entry_commission,
                entry_delta=s.net_greeks.delta, entry_vega=s.net_greeks.vega,
                entry_theta=s.net_greeks.theta,
                move_sigma=float(cand.diagnostics.get("move_sigma", float("nan"))),
                prob_otm_display=float(cand.diagnostics.get("prob_otm_risk_neutral_display", float("nan"))),
            )
        )
