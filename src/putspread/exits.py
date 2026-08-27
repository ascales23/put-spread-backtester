"""Exit rules (STRATEGY.md section 5), evaluated against REAL daily quotes.

Every rule returns a reason string or None. The engine applies them in a fixed
precedence so a day that satisfies two rules is not ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import StrategyConfig
from .portfolio import OpenPosition


@dataclass(frozen=True)
class MarkState:
    """What the engine knows about an open position on one day."""

    spot: float
    debit_to_close: float        # dollars per share, positive
    short_mid: float | None      # short leg mid, per share; None when unquoted
    days_to_expiry: int
    quotes_are_real: bool        # False when the mark came from Black-Scholes


def check_expiry(pos: OpenPosition, mark: MarkState) -> str | None:
    if mark.days_to_expiry <= 0:
        return "expiry"
    return None


def check_early_assignment(
    pos: OpenPosition, mark: MarkState, cfg: StrategyConfig, time_value_threshold: float = 0.10
) -> str | None:
    """Section 5 -- a deep-ITM short put with no time value left gets assigned.

    American puts are assigned when holding them costs the owner nothing: once the
    short's remaining time value is at or below a few cents, exercise is rational and
    the position is effectively closed whether or not the trader acts. Modeling it
    matters because it converts a defined-risk spread into a stock position at the
    worst possible moment.
    """
    if not cfg.model_early_assignment or mark.short_mid is None or not mark.quotes_are_real:
        return None
    if mark.spot >= pos.short_strike:
        return None
    intrinsic = pos.short_strike - mark.spot
    time_value = mark.short_mid - intrinsic
    if time_value <= time_value_threshold and mark.days_to_expiry <= 21:
        return "early_assignment"
    return None


def check_level_break(pos: OpenPosition, mark: MarkState, cfg: StrategyConfig) -> str | None:
    """Section 5 -- the thesis was 'support holds'. A decisive close below it is the
    thesis being wrong, and there is no reason to keep paying for that opinion."""
    if cfg.stop_rule != "level_break":
        return None
    if mark.spot < pos.support_level * (1.0 - cfg.stop_level_break_pct):
        return "level_break"
    return None


def check_credit_multiple_stop(pos: OpenPosition, mark: MarkState, cfg: StrategyConfig) -> str | None:
    """Section 5 -- close when the mark-to-market loss reaches N x the credit."""
    if cfg.stop_rule != "credit_multiple":
        return None
    open_loss_per_share = mark.debit_to_close - pos.credit_per_share
    if open_loss_per_share >= cfg.stop_credit_multiple * pos.credit_per_share:
        return "credit_multiple_stop"
    return None


def check_profit_target(pos: OpenPosition, mark: MarkState, cfg: StrategyConfig) -> str | None:
    """Section 5 -- close once `profit_target_pct` of the max credit is captured."""
    if cfg.profit_target_pct is None:
        return None
    if mark.debit_to_close <= (1.0 - cfg.profit_target_pct) * pos.credit_per_share:
        return "profit_target"
    return None


def check_time_stop(pos: OpenPosition, mark: MarkState, cfg: StrategyConfig) -> str | None:
    """Section 5 -- flat at N DTE regardless, to sidestep terminal gamma."""
    if cfg.time_stop_dte is None:
        return None
    if mark.days_to_expiry <= cfg.time_stop_dte:
        return "time_stop"
    return None


#: Precedence order. Expiry and assignment are facts, not choices, so they come first;
#: losses are cut before gains are taken so a whipsaw day cannot book a fake win.
EXIT_CHECKS = (
    check_expiry,
    check_early_assignment,
    check_level_break,
    check_credit_multiple_stop,
    check_profit_target,
    check_time_stop,
)


def evaluate_exits(pos: OpenPosition, mark: MarkState, cfg: StrategyConfig) -> str | None:
    """First triggered rule in precedence order, or None to keep holding."""
    for check in EXIT_CHECKS:
        reason = (
            check(pos, mark) if check is check_expiry
            else check(pos, mark, cfg)
        )
        if reason:
            return reason
    return None
