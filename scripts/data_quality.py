#!/usr/bin/env python
"""Data-integrity report on the real chain data, run before trusting any backtest.

Checks that matter for THIS strategy:
  * do the chains agree with where the stock actually closed (put-call parity)?
  * can we solve an implied vol from the quotes we intend to trade?
  * does our solved IV agree with the vendor's own IV column?
  * how wide are the markets we would be selling into?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from putspread.chain import solve_leg_iv                       # noqa: E402
from putspread.providers.parquet_chain import ParquetChainProvider  # noqa: E402
from putspread.rates import RateCurve                          # noqa: E402

RATE_DIR = Path.home() / "marketdata" / "macro" / "treasury_3m" / "1day"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--sample-days", type=int, default=120)
    args = ap.parse_args()

    rates = RateCurve.from_parquet_dir(RATE_DIR) if RATE_DIR.exists() else RateCurve.constant(0.04)
    prov = ParquetChainProvider(Path(args.data) / "chains", rates=rates,
                                ohlcv_dir=Path(args.data) / "ohlcv")
    rows = []
    for sym in args.symbols:
        try:
            bars = prov.daily_bars(sym)
        except FileNotFoundError:
            print(f"{sym}: no chain file")
            continue
        days = prov.trading_dates(sym, bars.index.min(), bars.index.max())
        sample = days[:: max(len(days) // args.sample_days, 1)]

        devs, iv_ok, iv_fail, iv_err, spreads, n_strikes = [], 0, 0, [], [], []
        for d in sample:
            dev = prov.parity_deviation(sym, d)
            if dev is not None:
                devs.append(dev)
            spot = prov.spot(sym, d)
            if spot is None:
                continue
            exp = prov.select_expiration(sym, d, 22, 7)
            if exp is None:
                continue
            ch = prov.chain(sym, d, exp)
            if ch is None:
                continue
            otm = [q for k, q in ch.puts.items() if 0.80 * spot < k < spot and q.bid > 0]
            n_strikes.append(len(otm))
            for q in otm:
                iv = solve_leg_iv(q, spot, ch.T, rates.get(d))
                if iv is None:
                    iv_fail += 1
                    continue
                iv_ok += 1
                if q.iv:
                    iv_err.append(iv - q.iv)
                spreads.append(q.spread_pct_of_mid)

        rows.append({
            "symbol": sym,
            "chain_days": len(days),
            "first": days[0] if days else None,
            "last": days[-1] if days else None,
            "bars": len(bars),
            "parity_dev_median": np.median(devs) if devs else np.nan,
            "parity_dev_p95": np.percentile(devs, 95) if devs else np.nan,
            "parity_dev_over_2pct": float(np.mean(np.array(devs) > 0.02)) if devs else np.nan,
            "otm_puts_per_day": np.median(n_strikes) if n_strikes else np.nan,
            "iv_solve_rate": iv_ok / max(iv_ok + iv_fail, 1),
            "iv_vs_vendor_median": np.median(iv_err) if iv_err else np.nan,
            "iv_vs_vendor_p95_abs": np.percentile(np.abs(iv_err), 95) if iv_err else np.nan,
            "spread_pct_median": np.median(spreads) if spreads else np.nan,
            "spread_pct_under_12": float(np.mean(np.array(spreads) <= 0.12)) if spreads else np.nan,
        })

    out = pd.DataFrame(rows).set_index("symbol")
    pd.set_option("display.width", 220)
    print(out.to_string(float_format=lambda x: f"{x:,.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
