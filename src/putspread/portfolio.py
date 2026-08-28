"""Slice 4 -- position sizing, capital, and the equity curve (STRATEGY.md section 4).

Sizing is driven ENTIRELY by realized max loss in dollars. No probability estimate of
any kind enters this module -- section 7 forbids using risk-neutral N(d2) as a win
rate in sizing, and the cleanest way to obey that is to never import it here.

Margin for a defined-risk vertical is the width less the credit received, i.e. exactly
the max loss, which is what brokers hold. So risk budget and margin are one number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from .config import StrategyConfig


@dataclass
class OpenPosition:
    """A live spread position."""

    symbol: str
    entry_date: date
    expiration: date
    short_strike: float
    long_strike: float
    credit_per_share: float
    contracts: int
    max_loss_per_contract: float
    entry_spot: float
    support_level: float
    short_iv: float
    long_iv: float
    entry_commission: float
    entry_delta: float
    entry_vega: float
    entry_theta: float
    move_sigma: float
    prob_otm_display: float
    worst_mark: float = 0.0          # most negative open P&L seen (MAE), dollars
    best_mark: float = 0.0           # most positive open P&L seen (MFE), dollars

    @property
    def width(self) -> float:
        return self.short_strike - self.long_strike

    @property
    def credit_dollars(self) -> float:
        """Gross credit received for the whole position, before commissions."""
        return self.credit_per_share * 100.0 * self.contracts

    @property
    def risk_dollars(self) -> float:
        """Total max loss at risk for this position, in dollars."""
        return self.max_loss_per_contract * self.contracts


@dataclass
class ClosedTrade:
    """A completed round trip."""

    symbol: str
    entry_date: date
    exit_date: date
    expiration: date
    short_strike: float
    long_strike: float
    width: float
    contracts: int
    credit_per_share: float
    exit_debit_per_share: float
    pnl: float                       # dollars, net of commissions both ways
    commissions: float
    exit_reason: str
    entry_spot: float
    exit_spot: float
    support_level: float
    short_iv: float
    long_iv: float
    max_loss_per_contract: float
    days_in_trade: int
    mae: float
    mfe: float
    move_sigma: float
    prob_otm_display: float

    @property
    def risk_dollars(self) -> float:
        return self.max_loss_per_contract * self.contracts

    @property
    def return_on_risk(self) -> float:
        r = self.risk_dollars
        return self.pnl / r if r > 0 else float("nan")

    @property
    def pnl_per_day(self) -> float:
        return self.pnl / max(self.days_in_trade, 1)


@dataclass
class Portfolio:
    """Account state: cash, open positions, and the realized/marked equity curve."""

    cfg: StrategyConfig
    cash: float = 0.0
    positions: list[OpenPosition] = field(default_factory=list)
    closed: list[ClosedTrade] = field(default_factory=list)
    equity_curve: list[tuple[date, float, float]] = field(default_factory=list)  # (date, equity, open_risk)
    rejections: list[tuple[date, str, str]] = field(default_factory=list)
    #: Dates where open max loss exceeded equity. Sizing cannot create this -- only
    #: losses can, by shrinking equity under positions already on. A real account
    #: gets a margin call here and is liquidated at the worst possible moment.
    margin_breaches: list[tuple[date, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.cash == 0.0:
            self.cash = self.cfg.starting_equity

    # ---------------------------------------------------------------- sizing

    @property
    def open_risk(self) -> float:
        """Total max loss across all open positions, dollars."""
        return sum(p.risk_dollars for p in self.positions)

    def equity(self, open_pnl: float = 0.0) -> float:
        """Cash plus mark-to-market on open positions."""
        return self.cash + open_pnl

    def size_position(self, max_loss_per_contract: float, equity: float) -> int:
        """Contracts to trade, enforcing the per-position cap, the portfolio cap, and
        the broker's margin limit.

        Section 4 calls the account-level cap "the dominant driver of long-run
        survival", so it is applied to the total open max loss, not per trade.
        `leverage` scales both caps; margin does NOT scale, because a defined-risk
        vertical is margined at its full max loss and an account cannot post more
        margin than it has. That third constraint is what makes leverage saturate.

        Returns 0 when even one contract would breach a limit -- the trade is
        skipped, never partially taken.
        """
        if max_loss_per_contract <= 0:
            return 0
        lev = self.cfg.leverage
        per_position_budget = self.cfg.max_risk_per_position_pct * lev * equity
        portfolio_budget = self.cfg.max_portfolio_risk_pct * lev * equity - self.open_risk
        margin_budget = self.cfg.max_margin_utilization * equity - self.open_risk
        budget = min(per_position_budget, portfolio_budget, margin_budget)
        if budget <= 0:
            return 0
        return max(int(math.floor(budget / max_loss_per_contract)), 0)

    def can_open(self, symbol: str) -> tuple[bool, str | None]:
        if len(self.positions) >= self.cfg.max_concurrent_positions:
            return False, f"at max concurrent positions ({self.cfg.max_concurrent_positions})"
        if self.cfg.one_position_per_symbol and any(p.symbol == symbol for p in self.positions):
            return False, f"already holding a {symbol} position"
        return True, None

    # ---------------------------------------------------------------- lifecycle

    def open_position(self, pos: OpenPosition) -> None:
        """Credit lands in cash immediately; margin is tracked as open_risk."""
        self.cash += pos.credit_dollars - pos.entry_commission
        self.positions.append(pos)

    def close_position(
        self, pos: OpenPosition, exit_date: date, exit_debit_per_share: float,
        exit_spot: float, reason: str, exit_commission: float,
    ) -> ClosedTrade:
        """Pay the debit to close and book the realized trade."""
        self.cash -= exit_debit_per_share * 100.0 * pos.contracts + exit_commission
        pnl = (
            (pos.credit_per_share - exit_debit_per_share) * 100.0 * pos.contracts
            - pos.entry_commission - exit_commission
        )
        trade = ClosedTrade(
            symbol=pos.symbol, entry_date=pos.entry_date, exit_date=exit_date,
            expiration=pos.expiration, short_strike=pos.short_strike,
            long_strike=pos.long_strike, width=pos.width, contracts=pos.contracts,
            credit_per_share=pos.credit_per_share,
            exit_debit_per_share=exit_debit_per_share, pnl=pnl,
            commissions=pos.entry_commission + exit_commission, exit_reason=reason,
            entry_spot=pos.entry_spot, exit_spot=exit_spot,
            support_level=pos.support_level, short_iv=pos.short_iv, long_iv=pos.long_iv,
            max_loss_per_contract=pos.max_loss_per_contract,
            days_in_trade=(exit_date - pos.entry_date).days,
            mae=pos.worst_mark, mfe=pos.best_mark,
            move_sigma=pos.move_sigma, prob_otm_display=pos.prob_otm_display,
        )
        self.closed.append(trade)
        self.positions.remove(pos)
        return trade

    def record_equity(self, d: date, open_pnl: float) -> None:
        eq = self.equity(open_pnl)
        self.equity_curve.append((d, eq, self.open_risk))
        if self.open_risk > eq:
            self.margin_breaches.append((d, eq, self.open_risk))
