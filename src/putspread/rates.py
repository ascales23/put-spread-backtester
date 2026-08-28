"""Risk-free rate series. Minor input (STRATEGY.md section 7) but real, not invented."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass
class RateCurve:
    """Daily short rate as a decimal (0.043 == 4.3%), forward-filled to every date."""

    series: pd.Series
    fallback: float = 0.04

    @classmethod
    def from_parquet_dir(cls, path: str | Path, fallback: float = 0.04) -> "RateCurve":
        """Load daily 3-month treasury yields from a directory of monthly parquet files."""
        files = sorted(Path(path).glob("*.parquet"))
        if not files:
            return cls(pd.Series(dtype=float), fallback)
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], utc=True).dt.date
        # FRED-style files carry `value`; bar-style files carry `close`.
        col = "value" if "value" in df.columns else "close"
        s = df.groupby("date")[col].last().sort_index()
        # Source is quoted in percent; convert once, here, and never again.
        return cls(s / 100.0, fallback)

    @classmethod
    def constant(cls, rate: float) -> "RateCurve":
        return cls(pd.Series(dtype=float), rate)

    def get(self, d: date) -> float:
        """Rate on or most recently before `d`; the fallback if the series is empty."""
        if self.series.empty:
            return self.fallback
        idx = self.series.index
        prior = [x for x in idx if x <= d]
        return float(self.series.loc[prior[-1]]) if prior else float(self.series.iloc[0])
