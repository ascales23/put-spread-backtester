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
    #: Only act on an early-exit rule when the position's strikes are actually quoted
    #: that day. The chain sample is thin, so a mark often falls back to
    #: Black-Scholes -- and closing at a modelled price is an assumption, not a fill.
    #: Setting this True defers the exit to the next day with a real two-sided market,
    #: which is what an account would actually experience.
    require_real_quotes_for_exit: bool = False

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


#: The universe this strategy is actually run on: liquid, optionable, and volatile
#: enough that the credit is worth collecting.
SHIPPED_UNIVERSE = (
    "AMD", "NVDA", "TSLA", "META", "AAPL", "MSFT", "AMZN", "GOOGL",
    "NFLX", "MU", "AVGO", "SMCI", "COIN", "PLTR", "SPY",
)


def shipped_config(leverage: float = 1.0, **overrides) -> StrategyConfig:
    """The chosen live configuration, in one place so it cannot drift.

    Entry: 5% buffer below support, 14 DTE. Chosen by the sweep, and the only fixed
    structure whose edge survived pessimistic fills.

    Exits: take profit at 50% of max credit, stop at 2.5x credit received, and close
    at 3 DTE regardless. At 4x leverage this is the best configuration measured --
    $94,335 at profit factor 1.76, Sharpe 0.95 and a 13.0% drawdown, against $50,356
    / 1.40 / 0.49 / 23.7% for the same strategy with no stop.

    The stop LEVEL is not from the P&L sweep. It comes from the excursion study: the
    median losing trade digs to 3.26x credit underwater while the median winner
    reaches only 0.14x, and only 3.8% of winners ever reach the median loser's depth.
    A 2.5x stop therefore catches ~63% of losers while cutting ~5.2% of winners.

    THE ASSUMPTION THIS CONFIG RESTS ON, stated plainly because every number above
    depends on it: roughly 95% of the profit-target and stop exits fill at a
    Black-Scholes mark rather than at a quoted market, because this chain source does
    not quote an open position's contract every day. Set
    `require_real_quotes_for_exit=True` for the conservative bound, where the stop
    fires once in 182 trades and the whole thing collapses toward hold-to-expiry
    (~$58,952 at 4x, Sharpe 0.53, drawdown 28.5%). Both runs ship; the modelled-exit
    figures are the headline and the real-quote figures are the floor.

    The time stop is included on the same basis as the stop itself. Its measured
    benefit is large but, like the stop's, disappears under real-quote exits -- so it
    is kept or dropped with the stop, not judged separately.

    Deliberately NOT included: the level-break stop. It triggers on the underlying
    closing below support, which happens constantly in noise, and it turns a 92% win
    rate into 53% for -$23,855. That result needs no modelled price to be believed,
    which is exactly why it is the one exit conclusion that is safe to act on.
    """
    cfg = StrategyConfig(
        symbols=SHIPPED_UNIVERSE,
        short_strike_method="buffer",
        buffer_pct=0.05,
        target_dte=14,
        profit_target_pct=0.50,
        stop_rule="credit_multiple",
        stop_credit_multiple=2.5,
        time_stop_dte=3,
        leverage=leverage,
        acknowledge_missing_oi_volume=True,
        support=SupportConfig(),
    )
    return cfg.with_(**overrides) if overrides else cfg
