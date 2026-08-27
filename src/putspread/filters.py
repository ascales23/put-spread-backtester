"""Hard do-not-trade filters (STRATEGY.md section 6).

Deliberately does NOT import prob_otm_rn: risk-neutral N(d2) must never gate a trade
or feed sizing (section 7). The move-plausibility check below uses a plain
lognormal-sigma distance, which is a stated approximation, not a probability claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from .config import StrategyConfig
from .earnings import EarningsCalendar
from .selection import StrikePair


class FilterConfigurationError(RuntimeError):
    """The configuration would silently disable a mandatory filter."""


@dataclass(frozen=True)
class FilterResult:
    """Outcome of running all filters on one candidate."""

    passed: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def reason(self) -> str:
        return "; ".join(self.reasons)


def check_earnings(
    symbol: str, entry: date, expiration: date,
    calendar: EarningsCalendar | None, cfg: StrategyConfig,
) -> tuple[bool, str | None]:
    """Section 6.1 -- mandatory. Blocks any position spanning an earnings report."""
    if calendar is None:
        if cfg.require_earnings_data:
            raise FilterConfigurationError(
                "no earnings calendar supplied and require_earnings_data is True. "
                "STRATEGY.md section 6.1 makes this filter mandatory; set "
                "require_earnings_data=False only to explicitly acknowledge that "
                "trades may span earnings, and every report will say so."
            )
        return True, None
    if not cfg.block_earnings_inside_expiry:
        return True, None
    hits = calendar.earnings_between(symbol, entry, expiration)
    if hits:
        return False, f"earnings {hits[0]} inside expiration {expiration}"
    return True, None


def check_liquidity(pair: StrikePair, cfg: StrategyConfig) -> tuple[bool, str | None]:
    """Section 6.3 -- open interest, volume, and bid/ask width.

    Raises if OI/volume are absent from the data and the caller has not explicitly
    acknowledged their absence: quietly passing a filter you cannot evaluate is the
    same failure mode as quietly disabling it.
    """
    for name, leg in (("short", pair.short), ("long", pair.long)):
        if leg.open_interest is None or leg.volume is None:
            if not cfg.acknowledge_missing_oi_volume:
                raise FilterConfigurationError(
                    "the chain source carries no open interest or volume, so the "
                    "section 6.3 OI/volume filters cannot be evaluated. Set "
                    "acknowledge_missing_oi_volume=True to proceed on the bid/ask "
                    "width and minimum-bid tests alone; every report will say so."
                )
        else:
            if leg.open_interest < cfg.min_open_interest:
                return False, f"{name} leg OI {leg.open_interest} < {cfg.min_open_interest}"
            if leg.volume < cfg.min_volume:
                return False, f"{name} leg volume {leg.volume} < {cfg.min_volume}"
        if leg.bid < cfg.min_bid:
            return False, f"{name} leg bid {leg.bid:.2f} < {cfg.min_bid:.2f}"
        if leg.spread_pct_of_mid > cfg.max_spread_pct_of_mid:
            return False, (
                f"{name} leg bid/ask {leg.spread_pct_of_mid:.1%} of mid "
                f"> {cfg.max_spread_pct_of_mid:.0%}"
            )
    return True, None


def move_sigma(spot: float, target: float, iv: float, days: int) -> float:
    """How many lognormal sigmas the move from `spot` to `target` is over `days`.

    Section 6.4's plausibility measure. A distance, not a probability -- reporting it
    as a probability would smuggle N(d2) back in through the side door.
    """
    if days <= 0 or iv <= 0 or spot <= 0 or target <= 0:
        return float("nan")
    sigma_window = iv * math.sqrt(days / 365.0)
    return abs(math.log(target / spot)) / sigma_window


def check_move_plausibility(
    spot: float, low_target: float, iv: float, days_to_low: int, cfg: StrategyConfig
) -> tuple[bool, str | None, float]:
    """Section 6.4 -- flag (default) or block an implausibly distant low target.

    In a backtest entries fire on an ACTUAL touch, so the required move has already
    happened and this can never block anything at entry. It is therefore computed and
    recorded as a diagnostic on the pre-trigger distance, preserving the spec's intent
    (how much of a stretch was this thesis?) without pretending to gate a trade the
    market already made.
    """
    ms = move_sigma(spot, low_target, iv, days_to_low)
    if ms != ms:  # NaN
        return True, None, ms
    if ms > cfg.max_move_sigma:
        msg = f"low target is {ms:.2f} sigma away over {days_to_low}d (> {cfg.max_move_sigma})"
        return (not cfg.block_implausible_move), msg, ms
    return True, None, ms
