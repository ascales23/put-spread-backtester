"""Strike selection (STRATEGY.md section 3.3).

Short strike: buffer below the low target, or a target delta, or a target
risk-neutral prob-OTM. Long strike: the WIDEST strike whose max loss still fits the
per-contract budget -- so width is derived, never chosen.
"""

from __future__ import annotations

from dataclasses import dataclass

from .chain import ExpiryChain, solve_leg_iv
from .config import ShortStrikeMethod, StrategyConfig
from .fills import FillConfig
from .pricing import bs_greeks, prob_otm_rn
from .spread import CONTRACT_MULTIPLIER, LegQuote


class SelectionError(Exception):
    """No strike pair satisfies the rules on this chain."""


@dataclass(frozen=True)
class StrikePair:
    short: LegQuote
    long: LegQuote
    short_iv: float
    long_iv: float
    credit_per_share: float
    max_loss_per_contract: float


def _tradeable_puts(chain: ExpiryChain, spot: float, cfg: StrategyConfig) -> dict[float, LegQuote]:
    """Puts that are OTM and carry a real two-sided quote."""
    return {
        k: q
        for k, q in chain.puts.items()
        if k < spot and q.bid >= cfg.min_bid and q.ask > q.bid >= 0.0
    }


def select_short_strike(
    chain: ExpiryChain, spot: float, low_target: float, r: float, cfg: StrategyConfig
) -> float:
    """Return the chosen short-put strike, snapped to the listed chain."""
    puts = _tradeable_puts(chain, spot, cfg)
    if not puts:
        raise SelectionError("no OTM puts with a usable two-sided quote")

    method: ShortStrikeMethod = cfg.short_strike_method
    if method == "buffer":
        # Section 3.3 flags this as volatility-blind on purpose; it is the baseline
        # the delta and prob-OTM methods are measured against.
        target = low_target * (1.0 - cfg.buffer_pct)
        below = [k for k in puts if k <= target + 1e-9]
        if not below:
            raise SelectionError(f"no listed strike at or below buffer target {target:.2f}")
        return max(below)

    if method == "delta":
        scored = []
        for k, q in puts.items():
            iv = solve_leg_iv(q, spot, chain.T, r)
            if iv is None:
                continue
            d = bs_greeks(spot, k, chain.T, r, iv, "put").delta
            scored.append((abs(abs(d) - cfg.target_short_delta), -k, k))
        if not scored:
            raise SelectionError("no put leg yielded a solvable IV for delta selection")
        return min(scored)[2]

    if method == "prob_otm":
        # Risk-neutral N(d2) used ONLY to place a strike, never as a win rate and
        # never in sizing or expectancy (section 7). Realized outcomes still come
        # entirely from the historical price path.
        best = None
        for k, q in puts.items():
            iv = solve_leg_iv(q, spot, chain.T, r)
            if iv is None:
                continue
            if prob_otm_rn(spot, k, chain.T, r, iv) >= cfg.target_prob_otm:
                if best is None or k > best:
                    best = k          # closest to the money that still clears the bar
        if best is None:
            raise SelectionError(
                f"no strike reaches risk-neutral prob-OTM {cfg.target_prob_otm:.2f}"
            )
        return best

    raise ValueError(f"unknown short-strike method {method!r}")


def select_long_strike(
    chain: ExpiryChain, spot: float, short_strike: float, r: float,
    cfg: StrategyConfig, fills: FillConfig,
) -> StrikePair:
    """Widest strike whose resulting max loss stays within the budget (section 3.3).

    Width is DERIVED: we walk candidate long strikes downward and keep the widest one
    that still fits (width - credit) x 100 <= budget. Credit is computed at the
    configured fill assumption, because a max loss computed off mid prices is a max
    loss the account will never actually have.
    """
    short_q = chain.puts.get(short_strike)
    if short_q is None:
        raise SelectionError(f"short strike {short_strike} not on the chain")
    short_iv = solve_leg_iv(short_q, spot, chain.T, r)
    if short_iv is None:
        raise SelectionError(f"short leg IV unsolvable at strike {short_strike}")

    candidates = sorted((k for k in chain.puts if k < short_strike), reverse=True)
    best: StrikePair | None = None
    for k in candidates:
        long_q = chain.puts[k]
        if long_q.ask <= 0.0 or long_q.ask < long_q.bid:
            continue
        width = short_strike - k
        if width < cfg.min_width:
            continue
        credit = fills.open_credit(short_q, long_q)
        if credit <= 0.0:
            continue
        max_loss = (width - credit) * CONTRACT_MULTIPLIER
        if max_loss > cfg.max_loss_per_contract:
            break                      # wider strikes only cost more; stop walking
        long_iv = solve_leg_iv(long_q, spot, chain.T, r)
        if long_iv is None:
            continue
        best = StrikePair(
            short=short_q, long=long_q, short_iv=short_iv, long_iv=long_iv,
            credit_per_share=credit, max_loss_per_contract=max_loss,
        )
    if best is None:
        raise SelectionError(
            f"no long strike below {short_strike} fits the "
            f"${cfg.max_loss_per_contract:,.0f} max-loss budget with a positive credit"
        )
    return best
