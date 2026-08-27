"""ChainProvider backed by parquet extracted from the DoltHub options database.

Real end-of-day quotes: per-strike bid, ask, and the vendor's implied vol and delta.
The backtester does NOT trust the vendor's vol column for pricing -- STRATEGY.md
section 7 requires IV to be solved from the market bid/ask -- so `vol` is carried
only for data-quality cross-checks.

Known gap in this source: it carries no open interest and no volume, so the section
6.3 OI/volume filters cannot be evaluated. `filters.check_liquidity` refuses to run
unless the caller explicitly acknowledges that, and every report says so.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..chain import ChainProvider, ExpiryChain, implied_spot_from_parity
from ..rates import RateCurve
from ..spread import LegQuote


class ParquetChainProvider(ChainProvider):
    """Loads one symbol's full quote history into memory and indexes it by day."""

    def __init__(
        self,
        chain_dir: str | Path,
        rates: RateCurve | None = None,
        ohlcv_dir: str | Path | None = None,
        max_strikes_per_side: int = 200,
    ) -> None:
        self.chain_dir = Path(chain_dir)
        self.ohlcv_dir = Path(ohlcv_dir) if ohlcv_dir else None
        self.rates = rates or RateCurve.constant(0.04)
        self.max_strikes_per_side = max_strikes_per_side
        self._frames: dict[str, pd.DataFrame] = {}
        self._groups: dict[str, dict[tuple[date, date], pd.DataFrame]] = {}
        self._spots: dict[str, dict[date, float]] = {}
        self._chain_cache: dict[tuple[str, date, date], ExpiryChain] = {}

    # ---------------------------------------------------------------- loading

    def _frame(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._frames:
            path = self.chain_dir / f"{symbol}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"no chain data for {symbol} at {path}")
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            df["expiration"] = pd.to_datetime(df["expiration"]).dt.date
            for col in ("strike", "bid", "ask", "vol", "delta"):
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            # A crossed or inverted market is bad data, not a tradeable quote.
            df = df[(df["ask"] >= df["bid"]) & (df["bid"] >= 0)]
            df["call_put"] = df["call_put"].astype(str).str.lower().str[0]  # 'c' / 'p'
            self._frames[symbol] = df.reset_index(drop=True)
            self._groups[symbol] = {
                key: g for key, g in self._frames[symbol].groupby(["date", "expiration"], sort=False)
            }
        return self._frames[symbol]

    # ---------------------------------------------------------------- interface

    def trading_dates(self, symbol: str, start: date, end: date) -> list[date]:
        df = self._frame(symbol)
        ds = df.loc[(df["date"] >= start) & (df["date"] <= end), "date"]
        return sorted(ds.unique().tolist())

    def expirations(self, symbol: str, as_of: date) -> list[date]:
        df = self._frame(symbol)
        return sorted(df.loc[df["date"] == as_of, "expiration"].unique().tolist())

    def chain(self, symbol: str, as_of: date, expiration: date) -> ExpiryChain | None:
        key = (symbol, as_of, expiration)
        if key in self._chain_cache:
            return self._chain_cache[key]
        self._frame(symbol)
        g = self._groups[symbol].get((as_of, expiration))
        if g is None or g.empty:
            return None
        puts, calls = {}, {}
        for row in g.itertuples(index=False):
            leg = LegQuote(
                strike=float(row.strike), bid=float(row.bid), ask=float(row.ask),
                iv=float(row.vol) if not np.isnan(row.vol) else None,
                delta=float(row.delta) if not np.isnan(row.delta) else None,
                open_interest=None, volume=None,   # not carried by this source
            )
            (puts if row.call_put == "p" else calls)[leg.strike] = leg
        ch = ExpiryChain(symbol=symbol, as_of=as_of, expiration=expiration, puts=puts, calls=calls)
        self._chain_cache[key] = ch
        return ch

    def spot(self, symbol: str, as_of: date) -> float | None:
        """RAW underlying price, recovered from the chain by put-call parity.

        Deliberately NOT read from a price vendor: vendor history is split-adjusted
        while option strikes are not, so a back-adjusted close silently misprices
        every trade before a split. Parity uses the same quotes being traded, so the
        two can never disagree about what a share cost that day.
        """
        cache = self._spots.setdefault(symbol, {})
        if as_of in cache:
            return cache[as_of]
        r = self.rates.get(as_of)
        best: float | None = None
        for exp in self.expirations(symbol, as_of):
            dte = (exp - as_of).days
            if dte < 7:
                continue          # near-expiry parity is noisy; skip the front week
            ch = self.chain(symbol, as_of, exp)
            if ch is None:
                continue
            s = implied_spot_from_parity(ch, r)
            if s is not None and s > 0:
                best = s
                break             # first (nearest) usable expiry is the most liquid
        cache[as_of] = best
        return best

    # ---------------------------------------------------------------- extras

    def daily_bars(self, symbol: str) -> pd.DataFrame:
        """Raw daily OHLC for support detection, indexed by date.

        Prefers the vendor OHLCV file when present, but rescales it onto the RAW
        (parity) price level date by date, so highs and lows stay consistent with the
        strikes even across splits. Without a file it falls back to a parity-derived
        close-only frame, which still supports the donchian and sma rules.
        """
        parity = pd.Series(
            {d: self.spot(symbol, d) for d in self.trading_dates(symbol, date.min, date.max)}
        ).dropna()
        parity.index = pd.Index(parity.index, name="date")

        path = self.ohlcv_dir / f"{symbol}.parquet" if self.ohlcv_dir else None
        if path is None or not path.exists():
            return pd.DataFrame(
                {"open": parity, "high": parity, "low": parity, "close": parity}
            )

        px = pd.read_parquet(path)
        px["date"] = pd.to_datetime(px["date"]).dt.date
        px = px.set_index("date").sort_index()
        px = px.loc[px.index.isin(parity.index)]
        for c in ("open", "high", "low", "close"):
            px[c] = pd.to_numeric(px[c], errors="coerce").astype(float)

        # Scale each day's bar so its close matches the parity (raw) close. On days
        # with no split this factor is ~1.0; across a split it undoes the adjustment.
        factor = (parity.reindex(px.index) / px["close"]).replace([np.inf, -np.inf], np.nan)
        factor = factor.ffill().bfill()
        for c in ("open", "high", "low", "close"):
            px[c] = px[c] * factor
        return px[["open", "high", "low", "close"]].dropna()
