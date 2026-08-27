"""Slice 2 -- the single-trade evaluator.

Given a date, a spot, a chain and the strategy parameters, produce exactly one
candidate spread the way the reference tool does, or a documented reason it was
rejected. Pure with respect to I/O: the chain arrives through the provider interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from .chain import ChainProvider, ExpiryChain
from .config import StrategyConfig
from .earnings import EarningsCalendar
from .filters import check_earnings, check_liquidity, check_move_plausibility
from .fills import FillConfig
from .selection import SelectionError, StrikePair, select_long_strike, select_short_strike
from .spread import BullPutSpread, build_spread


@dataclass(frozen=True)
class Candidate:
    """The outcome of evaluating one entry opportunity."""

    symbol: str
    as_of: date
    accepted: bool
    spread: BullPutSpread | None = None
    reject_reason: str | None = None
    warnings: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    @property
    def ticket(self) -> str:
        """Human-readable trade ticket, or the reason there isn't one."""
        if not self.accepted or self.spread is None:
            return f"{self.symbol} {self.as_of}: REJECTED -- {self.reject_reason}"
        s = self.spread
        return (
            f"{self.symbol} {self.as_of}: SELL {s.short.strike:g}P / BUY {s.long.strike:g}P "
            f"exp {s.expiration_days}d, width ${s.width:g}, credit ${s.credit_per_share:.2f}, "
            f"max loss ${s.max_loss_per_contract:,.0f}, breakeven {s.breakeven:.2f}, "
            f"IV {s.short_iv:.1%}/{s.long_iv:.1%}, delta {s.net_greeks.delta:+.1f}, "
            f"vega {s.net_greeks.vega:+.1f}, theta {s.net_greeks.theta:+.1f}"
        )


def evaluate_candidate(
    symbol: str,
    as_of: date,
    spot: float,
    low_target: float,
    chain: ExpiryChain,
    cfg: StrategyConfig,
    fills: FillConfig,
    r: float,
    calendar: EarningsCalendar | None = None,
    days_to_low: int = 6,
    spot_before_move: float | None = None,
) -> Candidate:
    """Price one candidate spread and run every hard filter against it.

    Order matters: the earnings check runs FIRST and is cheap, so a name reporting
    inside the expiration never gets priced at all. STRATEGY.md section 11's worked
    example exists precisely to be rejected here.
    """
    diagnostics: dict = {
        "spot": spot, "low_target": low_target,
        "dte": chain.days_to_expiry, "expiration": chain.expiration,
    }

    ok, why = check_earnings(symbol, as_of, chain.expiration, calendar, cfg)
    if not ok:
        return Candidate(symbol, as_of, False, reject_reason=why, diagnostics=diagnostics)

    try:
        short_strike = select_short_strike(chain, spot, low_target, r, cfg)
        pair: StrikePair = select_long_strike(chain, spot, short_strike, r, cfg, fills)
    except SelectionError as exc:
        return Candidate(symbol, as_of, False, reject_reason=str(exc), diagnostics=diagnostics)

    ok, why = check_liquidity(pair, cfg)
    if not ok:
        return Candidate(symbol, as_of, False, reject_reason=why, diagnostics=diagnostics)

    warnings: list[str] = []
    ok, why, ms = check_move_plausibility(
        spot_before_move if spot_before_move is not None else spot,
        low_target, pair.short_iv, days_to_low, cfg,
    )
    diagnostics["move_sigma"] = ms
    if why:
        warnings.append(why)
    if not ok:
        return Candidate(symbol, as_of, False, reject_reason=why, warnings=tuple(warnings),
                         diagnostics=diagnostics)

    spread = build_spread(
        underlying=symbol, spot=spot, short=pair.short, long=pair.long,
        T=chain.T, expiration_days=chain.days_to_expiry, r=r,
        short_iv=pair.short_iv, long_iv=pair.long_iv,
        credit_per_share=pair.credit_per_share,
    )
    diagnostics.update(
        credit_to_width=spread.credit_to_width,
        return_on_risk=spread.return_on_risk,
        # Display only -- never used for sizing or expectancy (section 7).
        prob_otm_risk_neutral_display=spread.prob_otm_display(),
        strike_increment=chain.strike_increment(),
    )
    return Candidate(symbol, as_of, True, spread=spread, warnings=tuple(warnings),
                     diagnostics=diagnostics)


def find_candidate(
    symbol: str, as_of: date, low_target: float, provider: ChainProvider,
    cfg: StrategyConfig, fills: FillConfig, r: float,
    calendar: EarningsCalendar | None = None, days_to_low: int = 6,
    spot_before_move: float | None = None,
) -> Candidate:
    """Pick the expiration nearest the target DTE, then evaluate on that board."""
    spot = provider.spot(symbol, as_of)
    if spot is None:
        return Candidate(symbol, as_of, False, reject_reason="no underlying price for this date")
    exp = provider.select_expiration(symbol, as_of, cfg.target_dte, cfg.dte_tolerance)
    if exp is None:
        return Candidate(
            symbol, as_of, False,
            reject_reason=f"no expiration within {cfg.dte_tolerance}d of {cfg.target_dte} DTE",
        )
    chain = provider.chain(symbol, as_of, exp)
    if chain is None:
        return Candidate(symbol, as_of, False, reject_reason=f"no chain for {exp}")
    return evaluate_candidate(
        symbol, as_of, spot, low_target, chain, cfg, fills, r, calendar,
        days_to_low, spot_before_move,
    )
