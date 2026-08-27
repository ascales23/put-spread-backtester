"""Earnings calendar. STRATEGY.md section 6.1 makes this filter MANDATORY.

The build prompt is explicit: "do not silently default the earnings filter off. If
earnings data isn't available, fail loudly." Hence `EarningsCalendar.assert_covers`,
which every backtest run calls before it is allowed to trade a symbol.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


class EarningsDataMissing(RuntimeError):
    """The earnings calendar does not cover a symbol/window the backtest wants to trade."""


@dataclass
class EarningsCalendar:
    """Historical earnings dates with before/after-market timing.

    `events` columns: symbol (str), date (datetime.date), when (str).
    """

    events: pd.DataFrame

    @classmethod
    def from_parquet(cls, path: str | Path) -> "EarningsCalendar":
        df = pd.read_parquet(path)
        df = df.rename(columns={"act_symbol": "symbol"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["when"] = df["when"].fillna("").astype(str)
        return cls(df[["symbol", "date", "when"]].sort_values(["symbol", "date"]).reset_index(drop=True))

    def coverage(self) -> tuple[date, date]:
        """Global (first, last) earnings date in the file."""
        return self.events["date"].min(), self.events["date"].max()

    def symbol_events(self, symbol: str) -> pd.DataFrame:
        return self.events[self.events["symbol"] == symbol]

    def assert_covers(self, symbol: str, start: date, end: date) -> None:
        """Raise unless this symbol has calendar coverage spanning [start, end].

        Coverage is judged on the FILE's global span, not the symbol's own event
        dates: a symbol with no rows inside the window may genuinely have not
        reported, but a window outside the file's span means we simply do not know,
        and trading blind through an unknown earnings date is precisely what the
        filter exists to prevent.
        """
        lo, hi = self.coverage()
        if start < lo or end > hi:
            raise EarningsDataMissing(
                f"earnings calendar covers {lo}..{hi} but the backtest window is "
                f"{start}..{end}; refusing to trade {symbol} through dates where "
                "earnings are unknown (STRATEGY.md section 6.1)"
            )
        if self.symbol_events(symbol).empty:
            raise EarningsDataMissing(
                f"no earnings events at all for {symbol} inside a covered window "
                f"({lo}..{hi}); this is more likely a symbol-mapping problem than a "
                "company that never reports"
            )

    def earnings_between(self, symbol: str, start: date, end: date) -> list[date]:
        """Earnings dates that a position opened at `start` and expiring `end` carries.

        The entry date itself is excluded (we trade at that day's close, after a
        before-open report and -- for an after-close report -- the position is opened
        into a known event, which the (start, end] window correctly still flags via
        the report date being > start only if it is a later day).

        Expiration-day timing matters: options settle at the close, so a report
        AFTER the close on expiration day cannot hurt the position, while a report
        BEFORE the open on expiration day can.
        """
        ev = self.symbol_events(symbol)
        hits = []
        for _, row in ev.iterrows():
            d, when = row["date"], row["when"].lower()
            if d <= start or d > end:
                continue
            if d == end and "after" in when:
                continue      # reports after the position has already settled
            hits.append(d)
        return hits

    def has_earnings_inside(self, symbol: str, start: date, end: date) -> bool:
        return bool(self.earnings_between(symbol, start, end))
