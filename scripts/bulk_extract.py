#!/usr/bin/env python
"""One full pass over the 8 GB dolt options table, extracting the whole universe.

Filtering by act_symbol still costs a full table scan (the primary key leads with
date), so eighteen per-symbol queries would mean eighteen scans. This does one, then
splits the result per symbol.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

DOLT = str(Path.home() / "bin" / "dolt")
DOLT_ROOT = Path.home() / "dolt_data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "chains").mkdir(parents=True, exist_ok=True)
    raw = out / "_bulk_chains.parquet"
    in_list = ",".join(f"'{s}'" for s in args.symbols)

    print(f"scanning option_chain for {len(args.symbols)} symbols ...", flush=True)
    with raw.open("wb") as fh:
        proc = subprocess.run(
            [DOLT, "sql", "-r", "parquet", "-q",
             "select act_symbol, date, expiration, strike, call_put, bid, ask, vol, delta "
             f"from option_chain where act_symbol in ({in_list})"],
            cwd=DOLT_ROOT / "options", stdout=fh, stderr=subprocess.PIPE,
        )
    if proc.returncode != 0:
        print(proc.stderr.decode()[:2000], file=sys.stderr)
        return 1

    df = pd.read_parquet(raw)
    print(f"{len(df):,} quote rows total", flush=True)
    for sym, g in df.groupby("act_symbol"):
        dest = out / "chains" / f"{sym}.parquet"
        g.drop(columns=["act_symbol"]).sort_values(["date", "expiration", "strike"]).to_parquet(
            dest, index=False
        )
        print(f"  {sym:6s} {len(g):>10,} rows  {g['date'].min()} .. {g['date'].max()}", flush=True)
    raw.unlink()

    print("\nOHLCV ...", flush=True)
    (out / "ohlcv").mkdir(parents=True, exist_ok=True)
    with (out / "_bulk_ohlcv.parquet").open("wb") as fh:
        proc = subprocess.run(
            [DOLT, "sql", "-r", "parquet", "-q",
             "select act_symbol, date, open, high, low, close, volume from ohlcv "
             f"where act_symbol in ({in_list})"],
            cwd=DOLT_ROOT / "stocks", stdout=fh, stderr=subprocess.PIPE,
        )
    if proc.returncode != 0:
        print(proc.stderr.decode()[:2000], file=sys.stderr)
        return 1
    px = pd.read_parquet(out / "_bulk_ohlcv.parquet")
    for sym, g in px.groupby("act_symbol"):
        g.drop(columns=["act_symbol"]).sort_values("date").to_parquet(
            out / "ohlcv" / f"{sym}.parquet", index=False
        )
    print(f"{len(px):,} daily bars", flush=True)
    (out / "_bulk_ohlcv.parquet").unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
