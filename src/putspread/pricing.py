"""Black-Scholes pricing, Greeks, and implied-volatility inversion.

Units, stated once and assumed everywhere in this module:
  S, K, prices : dollars per share (NOT per contract; multiply by 100 for contract $)
  T            : time to expiry in YEARS (act/365 unless the caller says otherwise)
  r, q         : continuously-compounded annual rates, decimal (0.05 == 5%)
  sigma        : annualized volatility, decimal (0.20 == 20% vol)

Greeks are returned in their conventional *reporting* scales:
  delta  : d(price)/d(S)                      -- per $1 of underlying
  gamma  : d2(price)/d(S)^2                   -- per $1^2
  vega   : d(price)/d(sigma) / 100            -- per 1 VOL POINT (1% move in sigma)
  theta  : d(price)/d(t) / 365                -- per CALENDAR DAY
  rho    : d(price)/d(r) / 100                -- per 1% move in r

This module is pure: no I/O, no globals, no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

OptionType = Literal["call", "put"]

_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf. Matches scipy.stats.norm.cdf to ~1e-15."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    """Black-Scholes d1 and d2. Caller must ensure T > 0 and sigma > 0."""
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    return d1, d1 - v


def bs_price(
    S: float, K: float, T: float, r: float, sigma: float, kind: OptionType, q: float = 0.0
) -> float:
    """Black-Scholes European option price, in dollars per share.

    Degenerate inputs (T <= 0 or sigma <= 0) collapse to discounted intrinsic value,
    which is the correct limit and keeps the expiry boundary well-defined.
    """
    if T <= 0.0 or sigma <= 0.0:
        fwd = S * math.exp(-q * T) - K * math.exp(-r * T)
        return max(fwd, 0.0) if kind == "call" else max(-fwd, 0.0)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    if kind == "call":
        return S * df_q * _norm_cdf(d1) - K * df_r * _norm_cdf(d2)
    return K * df_r * _norm_cdf(-d2) - S * df_q * _norm_cdf(-d1)


def bs_call(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes European call, dollars per share."""
    return bs_price(S, K, T, r, sigma, "call", q)


def bs_put(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes European put, dollars per share."""
    return bs_price(S, K, T, r, sigma, "put", q)


@dataclass(frozen=True)
class Greeks:
    """Per-share Greeks in reporting scale (see module docstring)."""

    price: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    def scaled(self, contracts: float, multiplier: int = 100) -> "Greeks":
        """Scale to a position of `contracts` contracts (negative == short)."""
        k = contracts * multiplier
        return Greeks(
            price=self.price * k,
            delta=self.delta * k,
            gamma=self.gamma * k,
            vega=self.vega * k,
            theta=self.theta * k,
            rho=self.rho * k,
        )

    def __add__(self, other: "Greeks") -> "Greeks":
        return Greeks(
            price=self.price + other.price,
            delta=self.delta + other.delta,
            gamma=self.gamma + other.gamma,
            vega=self.vega + other.vega,
            theta=self.theta + other.theta,
            rho=self.rho + other.rho,
        )


def bs_greeks(
    S: float, K: float, T: float, r: float, sigma: float, kind: OptionType, q: float = 0.0
) -> Greeks:
    """Price and Greeks for one European option, per share, in reporting scale."""
    if T <= 0.0 or sigma <= 0.0:
        price = bs_price(S, K, T, r, sigma, kind, q)
        itm = (S > K) if kind == "call" else (S < K)
        delta = (1.0 if kind == "call" else -1.0) if itm else 0.0
        return Greeks(price=price, delta=delta, gamma=0.0, vega=0.0, theta=0.0, rho=0.0)

    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    sqrtT = math.sqrt(T)
    df_r, df_q = math.exp(-r * T), math.exp(-q * T)
    pdf_d1 = _norm_pdf(d1)

    gamma = df_q * pdf_d1 / (S * sigma * sqrtT)
    vega = S * df_q * pdf_d1 * sqrtT / 100.0          # per vol point
    common_theta = -(S * df_q * pdf_d1 * sigma) / (2.0 * sqrtT)

    if kind == "call":
        delta = df_q * _norm_cdf(d1)
        theta = common_theta - r * K * df_r * _norm_cdf(d2) + q * S * df_q * _norm_cdf(d1)
        rho = K * T * df_r * _norm_cdf(d2) / 100.0
    else:
        delta = -df_q * _norm_cdf(-d1)
        theta = common_theta + r * K * df_r * _norm_cdf(-d2) - q * S * df_q * _norm_cdf(-d1)
        rho = -K * T * df_r * _norm_cdf(-d2) / 100.0

    return Greeks(
        price=bs_price(S, K, T, r, sigma, kind, q),
        delta=delta,
        gamma=gamma,
        vega=vega,
        theta=theta / 365.0,                           # per calendar day
        rho=rho,
    )


def prob_otm_rn(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Risk-neutral P(S_T > K) == N(d2), for a SHORT PUT finishing out of the money.

    WARNING -- STRATEGY.md section 7: this is NOT a real-world win probability. It is
    a risk-neutral quantity that systematically understates downside in high-IV names.
    It is exposed for DISPLAY ONLY. It must never feed sizing or expectancy; the
    backtest derives win rates from realized historical paths. See
    `putspread.filters` and `putspread.metrics`, which deliberately do not import it.
    """
    if T <= 0.0:
        return 1.0 if S > K else 0.0
    if sigma <= 0.0:
        return 1.0 if S * math.exp((r - q) * T) > K else 0.0
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    return _norm_cdf(d2)


class IVSolverError(ValueError):
    """Raised when implied volatility cannot be bracketed for the given price."""


def implied_vol(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    kind: OptionType,
    q: float = 0.0,
    *,
    lo: float = 1e-4,
    hi: float = 5.0,
    tol: float = 1e-10,
    price_tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Invert Black-Scholes for sigma by bisection. Returns annualized vol, decimal.

    Bisection (not Newton) is deliberate: it cannot diverge, and vega collapses for
    deep OTM strikes where this strategy lives, which is exactly where Newton breaks.

    `tol` is a tolerance on SIGMA, not on price. Converging on price instead would
    stop early wherever vega is near zero -- a deep-OTM strike in a low-vol name has
    a price that barely moves over a wide band of sigma, so a price-converged answer
    there can be off by tens of vol points. Sigma-bracket convergence is well defined
    everywhere BS is monotone in sigma, which it is.

    Raises IVSolverError if `price` is outside the no-arbitrage range reachable on
    [lo, hi] -- a quote that cannot be explained by any volatility is a data problem
    and must surface, not be silently clamped.
    """
    if T <= 0.0:
        raise IVSolverError("cannot solve IV at or after expiry (T <= 0)")
    if price <= 0.0:
        raise IVSolverError(f"non-positive option price {price!r}")

    # An in-the-money option's price is mostly intrinsic value, and intrinsic carries
    # no volatility information. Subtracting a large intrinsic from a large price to
    # recover a small time value destroys precision. Put-call parity converts the
    # problem to the OTM option at the same strike, whose price IS its time value;
    # both share the same implied volatility by construction. This is the standard
    # market practice of quoting vol off the OTM wing.
    fwd_less_strike = S * math.exp(-q * T) - K * math.exp(-r * T)
    if kind == "call" and fwd_less_strike > 0.0:
        kind, price = "put", price - fwd_less_strike
    elif kind == "put" and fwd_less_strike < 0.0:
        kind, price = "call", price + fwd_less_strike

    # Below this floor the remaining time value is at or under double-precision noise
    # for a price of this magnitude, so no sigma is recoverable. Say so rather than
    # returning a confident wrong number.
    time_value_floor = max(1e-10 * max(S, K), 1e-12)
    if price < -time_value_floor:
        raise IVSolverError(
            f"quote is below intrinsic value by {-price:.6f}; no volatility explains it"
        )
    if price <= time_value_floor:
        raise IVSolverError(
            f"time value {price:.3e} at or below the numerical floor {time_value_floor:.3e}; "
            "implied vol is not recoverable from this quote"
        )

    p_lo = bs_price(S, K, T, r, lo, kind, q)
    p_hi = bs_price(S, K, T, r, hi, kind, q)
    if price < p_lo - price_tol:
        raise IVSolverError(
            f"price {price:.6f} below intrinsic bound {p_lo:.6f} (sigma={lo}); check quote"
        )
    if price > p_hi + price_tol:
        raise IVSolverError(
            f"price {price:.6f} above price at sigma={hi} ({p_hi:.6f}); vol exceeds solver range"
        )

    a, b = lo, hi
    for _ in range(max_iter):
        if (b - a) < tol:
            break
        mid = 0.5 * (a + b)
        if bs_price(S, K, T, r, mid, kind, q) < price:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)
