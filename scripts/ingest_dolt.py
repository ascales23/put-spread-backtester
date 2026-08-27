#!/usr/bin/env python
"""Extract real historical option chains, earnings dates, and raw OHLCV from the
locally-cloned DoltHub databases into parquet the backtester can read.

Source: post-no-preference/{options,earnings,stocks} on DoltHub -- free, daily-updated,
real end-of-day quotes. This is the ONLY place the pipeline talks to the data source;
everything downstream reads parquet through the ChainProvider interface.

Usage:
    python scripts/ingest_dolt.py --symbols AMD NVDA TSLA --out data/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

DOLT = str(Path.home() / "bin" / "dolt")
DOLT_ROOT = Path.home() / "dolt_data"


def dolt_query_parquet(db: str, query: str, out: Path) -> None:
    """Run a query against a local dolt database and write parquet to `out`."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        proc = subprocess.run(
            [DOLT, "sql", "-r", "parquet", "-q", query],
            cwd=DOLT_ROOT / db, stdout=fh, stderr=subprocess.PIPE,
        )
    if proc.returncode != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"dolt query failed for {db}: {proc.stderr.decode()[:500]}")


def ingest_chains(symbols: list[str], out_dir: Path) -> None:
    for sym in symbols:
        dest = out_dir / "chains" / f"{sym}.parquet"
        print(f"[chains] {sym} -> {dest}", flush=True)
        dolt_query_parquet(
            "options",
            "select date, expiration, strike, call_put, bid, ask, vol, delta "
            f"from option_chain where act_symbol = '{sym}' order by date, expiration, strike",
            dest,
        )
        n = len(pd.read_parquet(dest, columns=["date"]))
        print(f"[chains] {sym}: {n:,} quote rows", flush=True)


def ingest_earnings(out_dir: Path) -> None:
    dest = out_dir / "earnings.parquet"
    print(f"[earnings] -> {dest}", flush=True)
    dolt_query_parquet(
        "earnings",
        "select act_symbol, date, `when` from earnings_calendar order by act_symbol, date",
        dest,
    )
    df = pd.read_parquet(dest)
    print(f"[earnings] {len(df):,} events, {df['act_symbol'].nunique():,} symbols, "
          f"{df['date'].min()} .. {df['date'].max()}", flush=True)


def ingest_ohlcv(symbols: list[str], out_dir: Path) -> None:
    for sym in symbols:
        dest = out_dir / "ohlcv" / f"{sym}.parquet"
        print(f"[ohlcv] {sym} -> {dest}", flush=True)
        dolt_query_parquet(
            "stocks",
            "select date, open, high, low, close, volume from ohlcv "
            f"where act_symbol = '{sym}' order by date",
            dest,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--skip", nargs="*", default=[], choices=["chains", "earnings", "ohlcv"])
    args = ap.parse_args()

    out = Path(args.out)
    if "earnings" not in args.skip:
        ingest_earnings(out)
    if "chains" not in args.skip:
        ingest_chains(args.symbols, out)
    if "ohlcv" not in args.skip:
        ingest_ohlcv(args.symbols, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
