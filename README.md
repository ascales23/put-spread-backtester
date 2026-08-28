# Bull Put Spread Backtester

A backtester for the support-bounce credit-spread strategy specified in
[`docs/STRATEGY.md`](docs/STRATEGY.md). Sell a defined-risk put credit spread when a
liquid, high-IV underlying pulls back to a technical support level; profit if the
short strike finishes out of the money.

Everything runs against **real historical option chains** — real per-strike bid/ask
and implied vol, real earnings dates. Nothing in any reported result comes from
synthetic data.

## Data

| What | Source | Coverage |
|---|---|---|
| EOD option chains (bid, ask, IV, greeks, per strike) | [`post-no-preference/options`](https://www.dolthub.com/repositories/post-no-preference/options) on DoltHub | 2019-02-09 → present, daily |
| Earnings dates with before/after-market timing | [`post-no-preference/earnings`](https://www.dolthub.com/repositories/post-no-preference/earnings) | 2020-01-22 → present, 7,360 symbols |
| Daily OHLCV | [`post-no-preference/stocks`](https://www.dolthub.com/repositories/post-no-preference/stocks) | full history |
| Risk-free rate (3M treasury) | local `~/marketdata/macro/treasury_3m` | daily |

Free, no API keys, updated daily. Clone once, then extract to parquet:

```bash
cd ~/dolt_data
dolt clone post-no-preference/options      # ~9 GB
dolt clone post-no-preference/earnings
dolt clone post-no-preference/stocks

cd ~/put_spread
python scripts/ingest_dolt.py --symbols AMD NVDA TSLA --out data/
```

### Known limits of this source, stated up front

- **No open interest and no volume.** The STRATEGY.md §6.3 OI/volume filters cannot
  be evaluated. `filters.check_liquidity` refuses to run unless the caller passes
  `acknowledge_missing_oi_volume=True`, and every report carries the caveat.
  Liquidity is screened on bid/ask width and a minimum bid instead.
- **End-of-day marks, not prints.** Entries and exits are priced at the close of the
  signal day. A mark is not a fill; that is what the fill-sensitivity table is for.
- **Earnings coverage starts 2020-01-22**, so backtests start there. Running earlier
  raises `EarningsDataMissing` rather than trading through unknown earnings dates.

Underlying prices are recovered from the chains themselves by **put-call parity**
rather than read from a price vendor. Option strikes are historical and unadjusted
while vendor price history is split-adjusted; comparing the two silently misprices
every trade before a split. Parity gives a raw spot that cannot disagree with the
strikes being traded.

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
python -m pytest        # 257 tests
```

## Run one backtest

```bash
python scripts/run_backtest.py --symbols AMD NVDA TSLA \
    --dte 22 --buffer 0.03 --method buffer --entry mechanical \
    --max-loss 1500 --fill realistic --out runs/core.html
```

Writes a self-contained HTML report: equity and drawdown curves, the P&L histogram,
the fill-sensitivity table, the metric table, the trade log, and a breakdown of why
signals did not become trades.

## Sweep with walk-forward validation

```bash
python scripts/run_sweep.py --symbols AMD NVDA TSLA --folds 4 --train-years 2
```

Optimizes in-sample per fold, then measures **that same parameter set** on the
following out-of-sample window. The `overfit_gap` and `oos_confirms` columns are the
point of the exercise.

## Learned trade filter (XGBoost)

The parameter sweep found no fixed rule that generalizes, so the model works at the
level of the individual trade instead: score every candidate at entry and decline the
ones whose modelled tail risk is elevated.

```bash
python scripts/train_selector.py --out runs/selector   # harvest 28 entry structures
python scripts/run_ml_backtest.py --out runs/ml        # train + 3 comparable backtests
```

**Results, realistic fills, 15 symbols, 2020&ndash;2026:**

| Run | Trades | Win | Total P&L | Exp/trade | PF | Sharpe | Max DD |
|---|---|---|---|---|---|---|---|
| Baseline, fixed 5%/14d | 293 | 88.7% | $15,194 | $51.86 | 1.42 | 0.54 | −6.5% |
| **+ tail-risk filter** | 218 | 91.7% | $11,946 | **$54.80** | **1.61** | **0.59** | **−3.8%** |
| Model also picks structure | 316 | 90.8% | −$5,050 | −$15.98 | 0.84 | −0.22 | −9.9% |

The filter improves every risk-adjusted metric at every fill assumption and roughly
halves drawdown, at the cost of a quarter of the trades. It also repairs 2022, the
year the unfiltered strategy lost money (−$2,774 → +$115).

Two negative results worth keeping:

- **Regressing on return-on-risk fails.** Out-of-fold lift is negative at every
  selectivity level. The target is too skewed and too noisy. Classifying the *left
  tail* (`big_loss`) is what works &mdash; the strategy's problem is its tail, so that
  is what the model should predict.
- **Letting the model choose the trade structure destroys value.** As a veto it
  helps; as a designer it does not.

### How leakage is prevented

The sample is small enough that one leak would manufacture a convincing fake edge, so
this is enforced structurally and tested:

- A model scoring a trade entered on day *t* trains only on trades that had already
  **closed** before *t*. Training on trades merely *opened* before *t* leaks the
  future, since their labels were not yet known. With holds up to 60 days that
  distinction is not academic.
- The accept threshold is the **training window's own base rate** of tail losses,
  fixed a priori. The selectivity scan in the output is a diagnostic and is never used
  to pick it.
- Candidates the model cannot score (burn-in) are **declined**, never waved through.
- `tests/test_ml.py` asserts no training row overlaps its test window, that a planted
  signal is recovered, and that **pure noise yields no lift** &mdash; the test that
  would catch a leak.

The honest limitation: burn-in consumes 2020&ndash;21, so the filter cannot be
evaluated on the earliest period at all.

## Layout

```
src/putspread/
  pricing.py     Black-Scholes, Greeks, bisection IV inversion       (pure)
  spread.py      spread construction, credit, max loss, payoff       (pure)
  chain.py       ChainProvider interface, parity spot, per-leg IV
  support.py     mechanical support levels + entry triggers, causal
  selection.py   short strike (buffer/delta/prob-OTM), derived width
  filters.py     earnings, liquidity, move plausibility
  fills.py       mid / realistic / natural execution models
  evaluate.py    single-trade evaluator -> ticket or documented reject
  exits.py       hold-to-expiry, profit target, stops, time stop, assignment
  engine.py      the historical path walk
  portfolio.py   sizing, capital, equity curve
  metrics.py     §9 metrics
  report.py      single-file HTML report
  sweep.py       parameter sweep + walk-forward
  features.py    entry-time features (volatility state, economics, trend, regime)
  harvest.py     candidate harvest across entry structures, outcomes simulated
  ml.py          XGBoost filter, purged walk-forward, a-priori thresholds
  providers/     parquet-backed ChainProvider
```

Pricing and selection are pure functions; all I/O sits in `providers/` and `runner.py`.
The core is testable with no data files, which is what `tests/conftest.py` exercises.

## Guardrails enforced in code, not just in prose

- **Risk-neutral N(d2) is never a win rate.** `portfolio.py`, `metrics.py` and
  `filters.py` do not import `prob_otm_rn`, and a test parses their import graphs to
  keep it that way. It is recorded per trade as a display field only.
- **The P&L path is never simulated.** Realized P&L at expiry is intrinsic value on
  the actual historical close, gaps included. Black-Scholes marks an open position
  only when a strike is unquoted on a given day, and the resulting `MarkState` says
  the number came from a model — the early-assignment rule refuses to act on one.
- **The earnings filter cannot be silently skipped.** No calendar plus
  `require_earnings_data=True` raises; disabling it stamps a caveat on every report.
- **Mid fills are never a headline.** Any report generated at mid carries a banner
  saying the number is not achievable.
- **No-lookahead is tested.** A pivot low is usable only after its confirmation bar,
  and a test truncates history at every index to prove published levels never change.
