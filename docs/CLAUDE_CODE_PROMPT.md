# Prompt for Claude Code — Bull Put Spread Backtester

Paste everything below into Claude Code, in a fresh project directory, alongside
`STRATEGY.md`. Read `STRATEGY.md` first — it is the source of truth for intent and
rules; this prompt is the build order.

---

I'm building a backtester for the credit-spread strategy fully specified in
`STRATEGY.md` (in this directory). Read that file first and treat it as the spec.
Below is what to build and in what order. Ask me before making any assumption that
isn't resolved by STRATEGY.md — especially about data sources.

## Before writing code

1. Read `STRATEGY.md` end to end.
2. Tell me back, in a few sentences, what the strategy does and what the single
   biggest modeling risk is, so I can confirm you've understood it. Do not skip this.
3. Then propose a file/module layout and wait for my OK before implementing.

## Language and stack

- Python 3.11+.
- Use `uv` for env/deps if available, else `venv` + `pip`. Pin versions in
  `pyproject.toml` or `requirements.txt`.
- Core libs: `numpy`, `pandas`, `scipy` (for the normal CDF / root-finding).
  `matplotlib` for the required plots. Add nothing heavier without asking.
- No paid data APIs and no network calls baked into the core logic — the backtest
  must run against local data files (see Data below). Keep any data-fetching in a
  separate, optional module.

## Data — resolve this with me first

Historical options backtesting is only as good as its chain data. Do not assume a
source. Present me these options and let me pick before you build the data layer:

- (a) I provide historical daily option chains (CSV/parquet) with per-strike IV,
  bid/ask, OI, volume. Best case — use real fills and real skew.
- (b) I provide only underlying OHLCV, and we *model* the chain each day with
  Black-Scholes from an IV series (e.g., a historical IV/VIX-like input). Cheaper,
  but fills and skew are modeled, not real — mark every result as such.
- (c) A small bundled sample dataset you generate synthetically for development, so
  the engine can be built and tested before real data arrives.

Build the engine against a clean `ChainProvider` interface so the source is swappable.
Start with (c) for development unless I say otherwise.

## Build order (vertical slices, each runnable and tested before the next)

### Slice 1 — Pricing core
- Port the Black-Scholes put pricer, Greeks, and the bisection IV solver from the
  reference component described in STRATEGY.md §10. 
- Validate against the known textbook values (S=K=100, T=1, r=0.05, σ=0.2 →
  call 10.4506, put 5.5735) and put-call parity. Write these as unit tests; they must
  pass before moving on.
- Add the spread pricer (short put − long put), net credit, net Greeks.

### Slice 2 — Single-trade evaluator
- Given a date, an underlying price, a chain (from ChainProvider), and the strategy
  parameters, produce one candidate spread exactly as the reference tool does:
  short strike from the selection method, long strike from the max-loss budget,
  per-leg IV solved from quotes, liquidity/earnings filters applied.
- Output: the trade ticket (strikes, width, credit, max loss, breakeven, Greeks) or a
  documented reason it was filtered out.
- Test against the STRATEGY.md worked example (AMD): confirm it produces ~460/435,
  ~$10 credit, AND confirm the earnings filter blocks it.

### Slice 3 — Entry trigger + P&L path
- Walk the underlying's historical path. Detect entry triggers (mechanical and
  confirmation variants). On trigger, open the trade from Slice 2.
- Compute P&L along the **actual historical price path**, including gaps — do NOT
  simulate the path with Black-Scholes. Use BS only for daily mark-to-market of open
  positions if an early-exit rule needs it; realized P&L at expiry is intrinsic.
- Implement all exit rules from STRATEGY.md §5: hold-to-expiry, profit target, stop /
  level-break, time stop, and early-assignment modeling for deep-ITM shorts.
- Model realistic fills per STRATEGY.md §7: pay the bid/ask spread on entry and exit;
  make the fill assumption a parameter so I can compare mid vs. realistic.

### Slice 4 — Portfolio + sizing
- Enforce account-level position sizing (§4): cap total open max-loss at a fraction of
  equity. Handle multiple concurrent positions, capital allocation, and margin for
  defined-risk spreads.
- Track an equity curve.

### Slice 5 — Metrics + reporting
- All metrics in STRATEGY.md §9. Emphasize the P&L distribution/histogram and the
  fill-sensitivity comparison — those are the two that reveal whether the edge is real.
- Output a single HTML or Markdown report per run: parameters used, equity curve, DD
  curve, trade log, metric table, and the P&L histogram.

### Slice 6 — Parameter sweep + validation
- Sweep the §8 parameter grid. Parallelize if easy, but correctness first.
- CRITICAL: implement walk-forward / out-of-sample validation, not just in-sample
  optimization. Report in-sample vs. out-of-sample performance side by side so
  overfitting is visible. A parameter set that only wins in-sample must be flagged.

## Engineering standards

- Every slice: unit tests for the math, at least one integration test for the slice.
  Do not proceed to the next slice with failing tests.
- Pure functions for all pricing/selection logic; keep I/O and data-loading at the
  edges. The core must be testable without any data files.
- Type hints throughout. Docstrings that state units (dollars, years, annualized vol,
  etc.) — unit confusion is the classic options-code bug.
- Deterministic: seed any randomness; same inputs → same outputs.
- A `README.md` explaining how to run a single backtest and a sweep, and a documented
  example config.
- Commit after each passing slice with a clear message.

## Guardrails — things I want you to refuse to fudge

- Do not use risk-neutral N(d2) as a real win-rate anywhere in sizing or expectancy.
  It is an estimate for display only (STRATEGY.md §7).
- Do not simulate the P&L path with Black-Scholes. Use real historical prices with
  gaps. If chain data is modeled (data option b), say so loudly in every report.
- Do not report only in-sample sweep results.
- Do not silently default the earnings filter off. If earnings data isn't available,
  fail loudly or require me to explicitly acknowledge trades may span earnings.
- If realistic fills erase the edge, report that plainly — do not tune fills to make
  the strategy look good.

## First deliverable

Stop after: (1) your understanding summary, (2) proposed module layout, (3) the data-
source question answered by me. Then implement Slice 1 with passing validation tests
and show me before continuing.
