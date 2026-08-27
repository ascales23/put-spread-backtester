"""Slice 1 validation: textbook values, put-call parity, Greeks, IV inversion."""

import math

import pytest
from scipy.stats import norm

from putspread.pricing import (
    Greeks,
    IVSolverError,
    bs_call,
    bs_greeks,
    bs_price,
    bs_put,
    implied_vol,
    prob_otm_rn,
    _norm_cdf,
)

# The canonical case from STRATEGY.md section 10 / CLAUDE_CODE_PROMPT slice 1.
S0, K0, T0, R0, SIG0 = 100.0, 100.0, 1.0, 0.05, 0.2


def test_textbook_call_value():
    assert bs_call(S0, K0, T0, R0, SIG0) == pytest.approx(10.4506, abs=1e-4)


def test_textbook_put_value():
    assert bs_put(S0, K0, T0, R0, SIG0) == pytest.approx(5.5735, abs=1e-4)


def test_norm_cdf_matches_scipy():
    for x in (-6.0, -2.5, -0.3, 0.0, 0.7, 3.1, 6.0):
        assert _norm_cdf(x) == pytest.approx(float(norm.cdf(x)), abs=1e-13)


@pytest.mark.parametrize("S", [60.0, 95.0, 100.0, 140.0])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("T", [0.02, 0.25, 1.0, 2.0])
@pytest.mark.parametrize("sigma", [0.1, 0.35, 0.9])
def test_put_call_parity(S, K, T, sigma):
    """C - P == S*e^-qT - K*e^-rT for every point on the grid."""
    r, q = 0.043, 0.0
    c = bs_call(S, K, T, r, sigma, q)
    p = bs_put(S, K, T, r, sigma, q)
    expected = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert (c - p) == pytest.approx(expected, abs=1e-10)


def test_put_call_parity_with_dividend_yield():
    S, K, T, r, sigma, q = 112.0, 105.0, 0.6, 0.04, 0.28, 0.021
    c, p = bs_call(S, K, T, r, sigma, q), bs_put(S, K, T, r, sigma, q)
    assert (c - p) == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


def test_price_collapses_to_intrinsic_at_expiry():
    assert bs_put(90.0, 100.0, 0.0, 0.05, 0.3) == pytest.approx(10.0)
    assert bs_put(110.0, 100.0, 0.0, 0.05, 0.3) == pytest.approx(0.0)
    assert bs_call(110.0, 100.0, 0.0, 0.05, 0.3) == pytest.approx(10.0)


def test_deep_otm_put_is_tiny_but_positive():
    p = bs_put(500.0, 200.0, 0.05, 0.04, 0.35)
    assert 0.0 <= p < 1e-6


# ---------------------------------------------------------------- Greeks


def test_greeks_match_finite_differences():
    S, K, T, r, sigma, q = 103.0, 100.0, 0.35, 0.045, 0.27, 0.0
    for kind in ("call", "put"):
        g = bs_greeks(S, K, T, r, sigma, kind, q)
        h = 1e-5
        # Gamma is a SECOND difference: dividing by h^2 amplifies float cancellation,
        # so it needs a much larger step than the first-order Greeks to be a valid check.
        hg = 1e-2
        fd_delta = (bs_price(S + h, K, T, r, sigma, kind, q) - bs_price(S - h, K, T, r, sigma, kind, q)) / (2 * h)
        fd_gamma = (
            bs_price(S + hg, K, T, r, sigma, kind, q)
            - 2 * bs_price(S, K, T, r, sigma, kind, q)
            + bs_price(S - hg, K, T, r, sigma, kind, q)
        ) / (hg * hg)
        fd_vega = (bs_price(S, K, T, r, sigma + h, kind, q) - bs_price(S, K, T, r, sigma - h, kind, q)) / (2 * h)
        fd_theta = -(bs_price(S, K, T + h, r, sigma, kind, q) - bs_price(S, K, T - h, r, sigma, kind, q)) / (2 * h)
        fd_rho = (bs_price(S, K, T, r + h, sigma, kind, q) - bs_price(S, K, T, r - h, sigma, kind, q)) / (2 * h)

        assert g.delta == pytest.approx(fd_delta, abs=1e-6)
        assert g.gamma == pytest.approx(fd_gamma, abs=1e-7)
        assert g.vega == pytest.approx(fd_vega / 100.0, abs=1e-6)     # per vol point
        assert g.theta == pytest.approx(fd_theta / 365.0, abs=1e-8)   # per calendar day
        assert g.rho == pytest.approx(fd_rho / 100.0, abs=1e-6)       # per 1% rate


def test_greek_signs_for_a_put():
    g = bs_greeks(100.0, 100.0, 0.5, 0.04, 0.25, "put")
    assert -1.0 < g.delta < 0.0        # long put is short the underlying
    assert g.gamma > 0.0
    assert g.vega > 0.0                # long option is long vol
    assert g.theta < 0.0               # long option bleeds
    assert g.rho < 0.0


def test_greeks_scaling_and_addition():
    g = bs_greeks(100.0, 95.0, 0.25, 0.04, 0.3, "put")
    short = g.scaled(-1)
    assert short.delta == pytest.approx(-g.delta * 100)
    assert short.theta == pytest.approx(-g.theta * 100)
    net = short + g.scaled(+1)
    assert net.delta == pytest.approx(0.0)
    assert net.vega == pytest.approx(0.0)


def test_greeks_at_expiry_are_degenerate():
    g = bs_greeks(90.0, 100.0, 0.0, 0.04, 0.3, "put")
    assert g.delta == -1.0 and g.gamma == 0.0 and g.vega == 0.0


# ---------------------------------------------------------------- IV inversion


@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.55, 0.89, 1.8, 3.5])
@pytest.mark.parametrize("K", [70.0, 100.0, 135.0])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_iv_round_trip(sigma, K, kind):
    """Price at a known sigma, invert, recover the same sigma.

    Includes deep-ITM strikes at low vol, where the option is pure intrinsic to
    within float noise. There the correct answer is not a number -- it is a refusal,
    because no sigma is recoverable. Both branches are asserted so the grid stays
    honest instead of quietly skipping the hard corners.
    """
    S, T, r = 100.0, 0.4, 0.045
    price = bs_price(S, K, T, r, sigma, kind)
    otm_equivalent = bs_price(S, K, T, r, sigma, "put" if K < S else "call")
    if otm_equivalent <= 1e-10 * max(S, K):
        with pytest.raises(IVSolverError, match="not recoverable"):
            implied_vol(price, S, K, T, r, kind)
    else:
        assert implied_vol(price, S, K, T, r, kind) == pytest.approx(sigma, abs=1e-6)


def test_iv_solves_itm_put_via_parity():
    """An ITM put with real time value must still invert cleanly through parity."""
    S, K, T, r, sigma = 100.0, 115.0, 0.5, 0.045, 0.42
    price = bs_put(S, K, T, r, sigma)
    assert implied_vol(price, S, K, T, r, "put") == pytest.approx(sigma, abs=1e-6)


def test_iv_refuses_pure_intrinsic_quote():
    """A quote with zero time value must raise, not return a fabricated vol."""
    S, K, T, r = 100.0, 150.0, 0.25, 0.0
    with pytest.raises(IVSolverError, match="not recoverable"):
        implied_vol(50.0, S, K, T, r, "put")


def test_iv_recovers_amd_style_high_vol():
    """The worked example lives near 89% IV on a far-OTM put; solver must hold there."""
    S, K, T, r, sigma = 535.0, 460.0, 22 / 365, 0.043, 0.89
    price = bs_put(S, K, T, r, sigma)
    assert implied_vol(price, S, K, T, r, "put") == pytest.approx(sigma, abs=1e-6)


def test_iv_raises_below_intrinsic():
    with pytest.raises(IVSolverError, match="below intrinsic"):
        implied_vol(1.0, 100.0, 120.0, 0.25, 0.04, "put")
    with pytest.raises(IVSolverError, match="below intrinsic"):
        implied_vol(0.5, 100.0, 130.0, 0.5, 0.04, "put")


def test_iv_raises_above_solver_range():
    with pytest.raises(IVSolverError, match="exceeds solver range"):
        implied_vol(99.0, 100.0, 100.0, 0.25, 0.04, "put")


def test_iv_raises_on_nonpositive_price_and_expired():
    with pytest.raises(IVSolverError):
        implied_vol(0.0, 100.0, 100.0, 0.25, 0.04, "put")
    with pytest.raises(IVSolverError):
        implied_vol(5.0, 100.0, 100.0, 0.0, 0.04, "put")


def test_iv_is_deterministic():
    args = (4.2, 100.0, 95.0, 0.3, 0.04, "put")
    assert implied_vol(*args) == implied_vol(*args)


# ---------------------------------------------------------------- N(d2) guard


def test_prob_otm_is_n_d2_and_display_only():
    S, K, T, r, sigma = 100.0, 90.0, 0.5, 0.04, 0.3
    p = prob_otm_rn(S, K, T, r, sigma)
    assert 0.0 < p < 1.0
    # Higher vol always lowers risk-neutral P(OTM) for an OTM short put.
    assert prob_otm_rn(S, K, T, r, 0.9) < p


def test_no_sizing_module_imports_prob_otm():
    """Guardrail: N(d2) must not reach sizing, expectancy, or the hard filters
    (STRATEGY.md section 7). Checked on the parsed import graph rather than on the
    source text, so the modules stay free to explain in prose why they refuse it."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "putspread"
    for name in ("portfolio.py", "metrics.py", "filters.py"):
        f = root / name
        if not f.exists():
            continue
        tree = ast.parse(f.read_text())
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "prob_otm_rn" not in imported, f"{name} must not import risk-neutral N(d2)"
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "prob_otm_rn" not in called, f"{name} must not call risk-neutral N(d2)"
