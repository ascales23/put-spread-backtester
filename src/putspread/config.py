"""Strategy parameters. Defaults and sweep ranges come from STRATEGY.md section 8."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

ShortStrikeMethod = Literal["buffer", "delta", "prob_otm"]
EntryDiscipline = Literal["mechanical", "confirmation"]
StopRule = Literal["none", "level_break", "credit_multiple"]
FillModel = Literal["mid", "realistic", "natural"]


@dataclass(frozen=True)
class SupportConfig:
    """How the 'low target' (expected support) is identified.

    STRATEGY.md assumes a human supplies the range thesis. A backtest cannot, so
    support must be derived mechanically from price history. This is the single
    largest departure from the written spec and is swept like any other parameter.
    """

    method: Literal["pivot_low", "donchian", "sma"] = "pivot_low"
    pivot_left: int = 5              # bars strictly higher on the left of the pivot
    pivot_right: int = 5             # bars strictly higher on the right (confirmation lag)
    lookback_days: int = 120         # how far back a pivot stays valid as support
    donchian_days: int = 60
    sma_days: int = 50
    min_distance_pct: float = 0.01   # support must sit at least this far below spot to be a "pullback"
    max_distance_pct: float = 0.25   # ...and not absurdly far


@dataclass(frozen=True)
class StrategyConfig:
    """One fully-specified parameter set for a backtest run."""

    # --- universe and window
    symbols: tuple[str, ...] = ("AMD",)
    start: str = "2020-01-22"        # first date with earnings-calendar coverage
    end: str = "2026-08-26"

    # --- entry (section 3)
    entry_discipline: EntryDiscipline = "mechanical"
    confirmation_bars: int = 1       # bars to wait for a close back above support
    support: SupportConfig = field(default_factory=SupportConfig)

    # --- strike selection (section 3.3)
    short_strike_method: ShortStrikeMethod = "buffer"
    buffer_pct: float = 0.03         # short strike this far BELOW the low target
    target_short_delta: float = 0.20 # |delta| of the short put, if method == "delta"
    target_prob_otm: float = 0.80    # risk-neutral N(d2), DISPLAY-DERIVED selection only
    max_loss_per_contract: float = 1500.0   # dollars; caps the long strike => caps width
    min_width: float = 1.0           # dollars; reject degenerate one-tick spreads
    target_dte: int = 22
    dte_tolerance: int = 7           # accept an expiration within +/- this many days

    # --- exits (section 5)
    profit_target_pct: float | None = None   # e.g. 0.50 == close at 50% of max credit
    stop_rule: StopRule = "none"
    stop_level_break_pct: float = 0.01       # close below support by this margin
    stop_credit_multiple: float = 2.0        # mark-to-market loss >= N x credit
    time_stop_dte: int | None = None         # close at N DTE regardless
    model_early_assignment: bool = True

    # --- filters (section 6)
    require_earnings_data: bool = True       # fail loudly rather than trade blind
    #: Instruments that genuinely never report. Listing one here is an assertion that
    #: its empty earnings history is a fact, not a data gap.
    non_reporting_symbols: tuple[str, ...] = (
        "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "SMH", "ARKK", "TLT", "GLD",
    )
    block_earnings_inside_expiry: bool = True
    min_open_interest: int = 100
    min_volume: int = 20
    max_spread_pct_of_mid: float = 0.12
    min_bid: float = 0.05                    # a leg with no real bid cannot be traded
    acknowledge_missing_oi_volume: bool = False
    max_parity_deviation: float = 0.02       # skip days whose chain disagrees with the close
    max_move_sigma: float = 1.5              # section 6.4 move-plausibility flag
    block_implausible_move: bool = False     # section 6.4 says flag, not necessarily block

    # --- fills (section 7)
    fill_model: FillModel = "realistic"
    realistic_fill_fraction: float = 0.5     # 0 == mid, 1 == fully at the natural price
    commission_per_contract: float = 0.65    # dollars per leg per contract, each way

    # --- portfolio (section 4)
    starting_equity: float = 100_000.0
    # Section 4 caps risk at two levels. The per-position cap is the spec's stated
    # "1-2% per position"; the portfolio cap is the account-level total it insists
    # must also be enforced. Setting them equal would silently allow only one open
    # position at a time, which is a very different strategy from the one specified.
    max_portfolio_risk_pct: float = 0.10     # total open max-loss / equity
    max_risk_per_position_pct: float = 0.02
    max_concurrent_positions: int = 10
    #: Multiplies BOTH risk caps below. 1.0 is the STRATEGY.md section 4 sizing;
    #: 4.0 puts four times the max loss to work per unit of equity.
    leverage: float = 1.0
    #: Hard broker constraint, not a preference. A defined-risk vertical is margined
    #: at its full max loss, so total open max loss can never exceed the account's
    #: equity -- there is nothing left to post. This is what makes leverage saturate
    #: rather than scale forever, and it binds before any of the caps above do.
    max_margin_utilization: float = 1.0
    one_position_per_symbol: bool = True

    # --- misc
    risk_free_rate_fallback: float = 0.04
    seed: int = 0

    def with_(self, **kw) -> "StrategyConfig":
        """Return a copy with fields replaced -- used by the sweep."""
        return replace(self, **kw)


#: STRATEGY.md section 8 sweep grid.
SWEEP_GRID: dict[str, list] = {
    "buffer_pct": [0.01, 0.02, 0.03, 0.05, 0.07, 0.10],
    "short_strike_method": ["buffer", "delta", "prob_otm"],
    "target_short_delta": [0.10, 0.15, 0.20, 0.25, 0.35],
    "target_dte": [7, 14, 22, 30, 45, 60],
    "entry_discipline": ["mechanical", "confirmation"],
    "profit_target_pct": [None, 0.25, 0.50, 0.75],
    "stop_rule": ["none", "level_break", "credit_multiple"],
    "time_stop_dte": [None, 21, 14, 7],
}
