"""Fill models. STRATEGY.md section 7: 'fills are modeled at mid in the tool; reality
is inside the bid/ask'.

Every price here is dollars per share. The seller's convention holds: opening a bull
put spread produces a positive credit, closing it costs a positive debit.

`mid` exists only as the optimistic reference point for the section 9 fill-sensitivity
comparison. It is not a defensible execution assumption and must never be the headline.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import FillModel
from .spread import CONTRACT_MULTIPLIER, LegQuote


def _half_spread(leg: LegQuote) -> float:
    return 0.5 * (leg.ask - leg.bid)


def sell_price(leg: LegQuote, model: FillModel, fraction: float) -> float:
    """Price received for SELLING one share-equivalent of this leg.

    mid       -- the midpoint (optimistic reference only)
    natural   -- the bid: you cross the spread and are filled now
    realistic -- `fraction` of the way from mid toward the bid
    """
    if model == "mid":
        return leg.mid
    if model == "natural":
        return leg.bid
    if model == "realistic":
        return leg.mid - fraction * _half_spread(leg)
    raise ValueError(f"unknown fill model {model!r}")


def buy_price(leg: LegQuote, model: FillModel, fraction: float) -> float:
    """Price paid for BUYING one share-equivalent of this leg."""
    if model == "mid":
        return leg.mid
    if model == "natural":
        return leg.ask
    if model == "realistic":
        return leg.mid + fraction * _half_spread(leg)
    raise ValueError(f"unknown fill model {model!r}")


@dataclass(frozen=True)
class FillConfig:
    model: FillModel = "realistic"
    fraction: float = 0.5
    commission_per_contract: float = 0.65   # per leg, per contract, each way

    def open_credit(self, short: LegQuote, long: LegQuote) -> float:
        """Net credit per share for opening the spread: sell short, buy long."""
        return sell_price(short, self.model, self.fraction) - buy_price(long, self.model, self.fraction)

    def close_debit(self, short: LegQuote, long: LegQuote) -> float:
        """Net debit per share for closing: buy back short, sell long.

        The spread is paid a SECOND time here. Round-tripping a credit spread costs
        the bid/ask twice, which is why holding to expiry (where the short expires
        worthless and costs nothing to close) is the spec's primary exit.
        """
        return buy_price(short, self.model, self.fraction) - sell_price(long, self.model, self.fraction)

    def commissions_per_contract(self, legs: int = 2, sides: int = 1) -> float:
        """Total commission in dollars for `legs` legs over `sides` transactions."""
        return self.commission_per_contract * legs * sides

    def open_credit_per_contract(self, short: LegQuote, long: LegQuote) -> float:
        """Dollars received for one spread, net of opening commissions."""
        return self.open_credit(short, long) * CONTRACT_MULTIPLIER - self.commissions_per_contract(2, 1)
