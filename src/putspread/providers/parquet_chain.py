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
        self._ohlcv_cache: dict[str, pd.DataFrame | None] = {}
        self._expirations: dict[str, dict[date, list[date]]] = {}
        self._dates: dict[str, list[date]] = {}

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
            # Precomputed date -> expirations. Scanning the frame on every call would
            # be O(rows) per lookup, and a sweep makes millions of these lookups
            # against a multi-million-row frame.
            exp_map: dict[date, list[date]] = {}
            for as_of, expiration in self._groups[symbol]:
                exp_map.setdefault(as_of, []).append(expiration)
            self._expirations[symbol] = {k: sorted(v) for k, v in exp_map.items()}
            self._dates[symbol] = sorted(exp_map)
        return self._frames[symbol]

    # ---------------------------------------------------------------- interface

    def trading_dates(self, symbol: str, start: date, end: date) -> list[date]:
        self._frame(symbol)
        return [d for d in self._dates[symbol] if start <= d <= end]

    def expirations(self, symbol: str, as_of: date) -> list[date]:
        self._frame(symbol)
        return self._expirations[symbol].get(as_of, [])

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

    def _ohlcv(self, symbol: str) -> pd.DataFrame | None:
        """Raw daily bars for `symbol`, or None when no OHLCV file was supplied."""
        if symbol in self._ohlcv_cache:
            return self._ohlcv_cache[symbol]
        path = self.ohlcv_dir / f"{symbol}.parquet" if self.ohlcv_dir else None
        if path is None or not path.exists():
            self._ohlcv_cache[symbol] = None
            return None
        px = pd.read_parquet(path)
        px["date"] = pd.to_datetime(px["date"]).dt.date
        px = px.set_index("date").sort_index()
        for c in ("open", "high", "low", "close"):
            px[c] = pd.to_numeric(px[c], errors="coerce").astype(float)
        self._ohlcv_cache[symbol] = px
        return px

    def parity_spot(self, symbol: str, as_of: date) -> float | None:
        """Underlying price recovered from the chain by put-call parity."""
        cache = self._spots.setdefault(symbol, {})
        if as_of in cache:
            return cache[as_of]
        r = self.rates.get(as_of)
        best: float | None = None
        for exp in self.expirations(symbol, as_of):
            if (exp - as_of).days < 7:
                continue          # near-expiry parity is noisy; skip the front week
            ch = self.chain(symbol, as_of, exp)
            if ch is None:
                continue
            s = implied_spot_from_parity(ch, r)
            if s is not None and s > 0:
                best = s
                break             # the nearest usable expiry is the most liquid
        cache[as_of] = best
        return best

    def spot(self, symbol: str, as_of: date) -> float | None:
        """RAW (split-unadjusted) underlying close, aligned with historical strikes.

        The vendor OHLCV is verified unadjusted -- NVDA prints 1224 the day before its
        2024 ten-for-one split and 121 the day after -- so it can be used directly
        against historical strikes. Parity is kept as an independent cross-check
        (`parity_deviation`) rather than as the price itself, because parity inherits
        the staleness of end-of-day option mids and would import that noise into every
        strike distance and every solved IV.
        """
        px = self._ohlcv(symbol)
        if px is not None and as_of in px.index:
            return float(px.loc[as_of, "close"])
        return self.parity_spot(symbol, as_of)

    def parity_deviation(self, symbol: str, as_of: date) -> float | None:
        """|parity spot / quoted close - 1| for this date, or None if not computable.

        A large deviation means the day's chain does not agree with where the stock
        actually closed -- stale or crossed end-of-day marks. Entries on such a day
        would be priced off quotes that never existed, so the engine skips them.
        """
        px = self._ohlcv(symbol)
        if px is None or as_of not in px.index:
            return None
        parity = self.parity_spot(symbol, as_of)
        close = float(px.loc[as_of, "close"])
        if parity is None or close <= 0:
            return None
        return abs(parity / close - 1.0)

    # ---------------------------------------------------------------- extras

    def daily_bars(self, symbol: str) -> pd.DataFrame:
        """Raw daily OHLC for support detection, restricted to days with a chain.

        Falls back to a parity-derived close-only frame when no OHLCV file exists,
        which still supports the donchian and sma support rules.
        """
        chain_days = set(self.trading_dates(symbol, date.min, date.max))
        px = self._ohlcv(symbol)
        if px is not None:
            out = px.loc[px.index.isin(chain_days), ["open", "high", "low", "close"]]
            return out.dropna()
        parity = pd.Series({d: self.parity_spot(symbol, d) for d in sorted(chain_days)}).dropna()
        parity.index = pd.Index(parity.index, name="date")
        return pd.DataFrame({"open": parity, "high": parity, "low": parity, "close": parity})
