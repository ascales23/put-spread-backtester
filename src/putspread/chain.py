"""Option-chain domain model and the ChainProvider interface.

The provider interface is deliberately narrow so the data source is swappable
(STRATEGY.md / build prompt: "clean ChainProvider interface"). Nothing downstream
knows where quotes came from.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from .pricing import IVSolverError, implied_vol
from .spread import LegQuote

TRADING_DAYS_PER_YEAR = 252
DAYS_PER_YEAR = 365.0


def year_fraction(as_of: date, expiration: date) -> float:
    """Time to expiry in years, act/365. Zero on and after the expiration date."""
    return max((expiration - as_of).days, 0) / DAYS_PER_YEAR


@dataclass(frozen=True)
class ExpiryChain:
    """All put quotes for one underlying, one as-of date, one expiration.

    `puts` is keyed by strike (dollars). `calls` is optional and used only to recover
    a raw spot price by put-call parity; the strategy itself never trades calls.
    """

    symbol: str
    as_of: date
    expiration: date
    puts: dict[float, LegQuote]
    calls: dict[float, LegQuote]

    @property
    def days_to_expiry(self) -> int:
        return (self.expiration - self.as_of).days

    @property
    def T(self) -> float:
        """Years to expiry, act/365."""
        return year_fraction(self.as_of, self.expiration)

    def strikes(self) -> list[float]:
        """Put strikes, ascending."""
        return sorted(self.puts)

    def strike_increment(self) -> float:
        """Modal gap between adjacent put strikes, in dollars.

        The mode, not the mean: chains mix 1.0 spacing near the money with 5.0 or
        10.0 spacing in the wings, and the mean of that is a spacing that does not
        exist on the board.
        """
        ks = self.strikes()
        if len(ks) < 2:
            return 1.0
        gaps: dict[float, int] = {}
        for a, b in zip(ks, ks[1:]):
            g = round(b - a, 4)
            if g > 0:
                gaps[g] = gaps.get(g, 0) + 1
        return max(gaps.items(), key=lambda kv: (kv[1], -kv[0]))[0] if gaps else 1.0

    def nearest_strike_at_or_below(self, target: float) -> float | None:
        """Highest listed put strike <= target. None if the chain has none."""
        below = [k for k in self.puts if k <= target + 1e-9]
        return max(below) if below else None

    def nearest_strike(self, target: float) -> float | None:
        """Listed put strike closest to target."""
        return min(self.puts, key=lambda k: abs(k - target)) if self.puts else None


def implied_spot_from_parity(chain: ExpiryChain, r: float) -> float | None:
    """Recover the underlying's RAW price from the chain by put-call parity.

    S = C - P + K*exp(-r*T), evaluated at the strike closest to the money, where the
    call and put are both near ATM and therefore both liquid.

    This matters more than it looks: option strikes are historical and unadjusted,
    while most vendor price history is back-adjusted for splits. Comparing an
    adjusted close to a raw strike silently misprices every trade before a split.
    Parity gives a spot that is raw by construction and consistent with the very
    quotes being traded. Returns None when no strike has both sides quoted.
    """
    common = [k for k in chain.puts if k in chain.calls]
    if not common:
        return None
    disc = math.exp(-r * chain.T)
    # Seed with the strike whose call/put mids are closest -- that is the ATM strike,
    # and it needs no prior knowledge of spot.
    k = min(common, key=lambda k: abs(chain.calls[k].mid - chain.puts[k].mid))
    c, p = chain.calls[k], chain.puts[k]
    if c.mid <= 0 or p.mid <= 0:
        return None
    return c.mid - p.mid + k * disc


def solve_leg_iv(
    leg: LegQuote, spot: float, T: float, r: float, price: float | None = None
) -> float | None:
    """Solve this put leg's implied vol from its own quote (STRATEGY.md section 7).

    Per-leg, never one flat IV across both strikes: skew means the lower long strike
    carries higher IV, and a single IV overstates the credit -- always optimistically.

    `price` defaults to the leg's mid. Returns None when the quote carries no
    recoverable vol information rather than inventing one.
    """
    px = leg.mid if price is None else price
    try:
        return implied_vol(px, spot, leg.strike, T, r, "put")
    except IVSolverError:
        return None


class ChainProvider(ABC):
    """Read-only access to historical option chains for one or more symbols."""

    @abstractmethod
    def trading_dates(self, symbol: str, start: date, end: date) -> list[date]:
        """Dates in [start, end] on which a chain exists for `symbol`, ascending."""

    @abstractmethod
    def expirations(self, symbol: str, as_of: date) -> list[date]:
        """Expirations quoted for `symbol` on `as_of`, ascending."""

    @abstractmethod
    def chain(self, symbol: str, as_of: date, expiration: date) -> ExpiryChain | None:
        """One expiry's quotes, or None if not quoted that day."""

    @abstractmethod
    def spot(self, symbol: str, as_of: date) -> float | None:
        """RAW (split-unadjusted) underlying close on `as_of`, aligned with strikes."""

    def select_expiration(
        self, symbol: str, as_of: date, target_dte: int, tolerance: int
    ) -> date | None:
        """Expiration whose DTE is closest to `target_dte`, within `tolerance` days.

        Ties break toward the LONGER dated expiry: more time value means more credit,
        and it is the conservative choice when comparing two equally-distant boards.
        """
        candidates = [
            e for e in self.expirations(symbol, as_of)
            if abs((e - as_of).days - target_dte) <= tolerance and (e - as_of).days > 0
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: (abs((e - as_of).days - target_dte), -(e - as_of).days))
