"""Bull put spread construction and pricing (short higher put / long lower put).

Conventions:
  * All *_per_share fields are dollars per share; *_per_contract are dollars for one
    contract of 100 shares. Mixing these is the classic options-code bug, so the
    field names always say which one they are.
  * Sign convention is the SELLER's: credit received is positive, losses negative.
"""

from __future__ import annotations

from dataclasses import dataclass

from .pricing import Greeks, bs_greeks, prob_otm_rn

CONTRACT_MULTIPLIER = 100


@dataclass(frozen=True)
class LegQuote:
    """One quoted option leg. `bid`/`ask` are dollars per share."""

    strike: float
    bid: float
    ask: float
    iv: float | None = None          # annualized decimal; None until solved
    delta: float | None = None       # per $1 underlying, as quoted by the source
    open_interest: int | None = None
    volume: int | None = None

    @property
    def mid(self) -> float:
        """Mid price, dollars per share."""
        return 0.5 * (self.bid + self.ask)

    @property
    def spread_abs(self) -> float:
        """Bid/ask spread width, dollars per share."""
        return self.ask - self.bid

    @property
    def spread_pct_of_mid(self) -> float:
        """Bid/ask spread as a fraction of mid. inf when mid is zero."""
        m = self.mid
        return float("inf") if m <= 0 else self.spread_abs / m


@dataclass(frozen=True)
class BullPutSpread:
    """A priced bull put spread candidate at one point in time.

    `credit_per_share` is the NET credit received for selling the short put and
    buying the long put, at whatever fill assumption produced it.
    """

    underlying: str
    spot: float
    expiration_days: int              # calendar days to expiry at pricing time
    T: float                          # years to expiry, act/365
    short: LegQuote
    long: LegQuote
    credit_per_share: float
    short_iv: float
    long_iv: float
    net_greeks: Greeks                # for ONE spread (1 short + 1 long contract)
    risk_free_rate: float

    @property
    def width(self) -> float:
        """Strike width, dollars per share."""
        return self.short.strike - self.long.strike

    @property
    def max_profit_per_contract(self) -> float:
        """Max profit in dollars for one spread: the credit, kept in full."""
        return self.credit_per_share * CONTRACT_MULTIPLIER

    @property
    def max_loss_per_contract(self) -> float:
        """Max loss in dollars for one spread: (width - credit) x 100. Positive number."""
        return (self.width - self.credit_per_share) * CONTRACT_MULTIPLIER

    @property
    def breakeven(self) -> float:
        """Underlying price at expiry where the spread breaks even, dollars."""
        return self.short.strike - self.credit_per_share

    @property
    def credit_to_width(self) -> float:
        """Credit as a fraction of width -- the standard richness measure."""
        return self.credit_per_share / self.width if self.width > 0 else float("nan")

    @property
    def return_on_risk(self) -> float:
        """Max profit / max loss, the trade's headline reward-to-risk."""
        ml = self.max_loss_per_contract
        return self.max_profit_per_contract / ml if ml > 0 else float("inf")

    def prob_otm_display(self) -> float:
        """Risk-neutral N(d2) for the short strike. DISPLAY ONLY -- see pricing.prob_otm_rn."""
        return prob_otm_rn(self.spot, self.short.strike, self.T, self.risk_free_rate, self.short_iv)

    def payoff_at_expiry_per_contract(self, spot_at_expiry: float) -> float:
        """Realized P&L in dollars for one spread held to expiry, at `spot_at_expiry`.

        Intrinsic only -- no volatility assumption is needed at expiry, which is the
        whole point of holding to expiration (STRATEGY.md section 5).
        """
        return payoff_at_expiry_per_contract(
            spot_at_expiry, self.short.strike, self.long.strike, self.credit_per_share
        )


def payoff_at_expiry_per_contract(
    spot_at_expiry: float, short_strike: float, long_strike: float, credit_per_share: float
) -> float:
    """Bull put spread P&L in dollars for one contract at expiry (seller's sign).

    Above the short strike -> full credit. Below the long strike -> max loss.
    Between -> linear. Assignment/exercise is automatic and settles to intrinsic.
    """
    width = short_strike - long_strike
    short_intrinsic = max(short_strike - spot_at_expiry, 0.0)
    capped_loss = min(short_intrinsic, width)
    return (credit_per_share - capped_loss) * CONTRACT_MULTIPLIER


def spread_mark_per_contract(
    short_strike: float, long_strike: float, spot: float, T: float, r: float,
    short_iv: float, long_iv: float, q: float = 0.0,
) -> float:
    """Model value (debit to close) of the spread, dollars per contract, positive.

    Used ONLY for daily mark-to-market when an early-exit rule needs a mark
    (STRATEGY.md section 7). It never generates the realized P&L path.
    """
    short_val = bs_greeks(spot, short_strike, T, r, short_iv, "put", q).price
    long_val = bs_greeks(spot, long_strike, T, r, long_iv, "put", q).price
    return (short_val - long_val) * CONTRACT_MULTIPLIER


def build_spread(
    underlying: str,
    spot: float,
    short: LegQuote,
    long: LegQuote,
    T: float,
    expiration_days: int,
    r: float,
    short_iv: float,
    long_iv: float,
    credit_per_share: float,
    q: float = 0.0,
) -> BullPutSpread:
    """Assemble a priced spread with net Greeks for a 1-lot (short 1, long 1)."""
    if short.strike <= long.strike:
        raise ValueError(
            f"bull put spread needs short strike above long strike, got {short.strike} / {long.strike}"
        )
    g_short = bs_greeks(spot, short.strike, T, r, short_iv, "put", q).scaled(-1, CONTRACT_MULTIPLIER)
    g_long = bs_greeks(spot, long.strike, T, r, long_iv, "put", q).scaled(+1, CONTRACT_MULTIPLIER)
    return BullPutSpread(
        underlying=underlying,
        spot=spot,
        expiration_days=expiration_days,
        T=T,
        short=short,
        long=long,
        credit_per_share=credit_per_share,
        short_iv=short_iv,
        long_iv=long_iv,
        net_greeks=g_short + g_long,
        risk_free_rate=r,
    )
