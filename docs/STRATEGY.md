# Bull Put Spread — Support-Bounce Credit Strategy

## 1. Intent

This strategy sells defined-risk put credit spreads on liquid large-cap equities
when price pulls back to a technical support level the trader expects to hold. The
thesis is directional-with-a-floor: not "the stock will rally hard," but "the stock
will not close below support by expiration." The spread is held to expiration and
profits from the short strike finishing out of the money.

It is deliberately **not** a long-premium directional bet. Earlier design work
established that buying calls or puts into elevated implied volatility is dominated
by IV crush — being right on direction while still losing money as vol collapses.
The credit spread inverts that exposure: it is short vega, so vol collapse works in
the trader's favor, and short theta-positive, so time decay works in the trader's
favor. The cost is a capped upside.

The trader supplies a *range thesis* — a low target (expected support), a high
target (expected recovery), and a timeframe for each. The system derives the strikes
and width from that thesis plus a risk budget. Everything that can be computed is
computed; the only irreducible human inputs are the thesis and the risk tolerance.

## 2. Why a bull put spread specifically

| Property | Long call | Bull put spread (this strategy) |
|---|---|---|
| Vega | Long — hurt by IV crush | Short — helped by IV crush |
| Theta | Negative — bleeds daily | Positive — collects daily |
| Max loss | Premium paid | Width − credit, defined |
| Win condition | Large fast up-move | Stays above short strike |
| Best entry vol regime | Low IV | High IV (sell rich premium) |

The strategy is designed to be entered when IV is elevated (which is typically
*when* a stock is falling toward support), because that is when credit received is
richest and the short-vega exposure is most valuable.

## 3. Entry logic

### 3.1 Trigger
Entry is triggered on the **underlying price**, not the option price. The thesis is
about where the stock goes; the option is the instrument. When the underlying
reaches the low target (the support level), the spread order is worked.

Rationale: a resting limit on the option itself may never fill because the option's
price at the trigger moment depends on IV, which cannot be pinned in advance. A
stock-triggered conditional order that then sends a spread order at a limit relative
to the *then-current* market is more reliable.

### 3.2 Confirmation vs. mechanical entry
Two entry disciplines, to be A/B tested:

- **Mechanical:** enter the moment the underlying touches the low target. Simple,
  always fills, no confirmation the level held.
- **Confirmation:** enter only after the underlying touches the low target *and*
  shows a stabilization signal (e.g., closes back above the level, or an intraday
  reversal bar). Worse average entry price, but avoids selling puts into a level
  that is breaking. For a premium-selling strategy whose max loss comes from the
  level failing, confirmation is expected to improve results.

### 3.3 Strike selection
- **Short strike:** placed a buffer percentage below the low target, snapped to the
  chain's strike increment. Default buffer 3%. NOTE: percentage buffers are
  volatility-blind; a fixed % is a weak rule in high-IV names. The backtest should
  also evaluate delta-based (e.g., short strike ≈ 0.20 delta) and
  probability-OTM-based selection, which scale with vol automatically.
- **Long strike:** the widest strike whose resulting max loss per contract stays
  within the risk budget. Wider = more credit and more absolute risk; the budget
  caps it.
- **Width** is therefore derived, not chosen directly.

## 4. Position sizing

- Risk is defined **per contract** as (width − credit) × 100.
- Max loss budget is stated per contract; number of contracts is a pure multiplier.
- Account-level sizing rule: total max loss across all open positions should not
  exceed a fixed fraction of account equity (default 1–2% per position). This is the
  dominant driver of long-run survival and must be enforced at the portfolio level,
  not per-trade.

## 5. Exit logic

**Primary: hold to expiration.** At expiry the spread is worth intrinsic value only,
so no exit-IV assumption is required.
- Above short strike → full credit (max profit).
- Below long strike → max loss.
- Between → linear.

**Optional early-management rules to backtest:**
- **Profit target:** close at 50% of max credit captured. Standard credit-spread
  discipline; reduces tail risk and frees capital, at the cost of some expected value.
- **Stop / thesis invalidation:** close if the underlying closes below the support
  level by a defined margin (the level has failed), or if the spread's mark-to-market
  loss reaches a multiple of credit received (e.g., 2×).
- **Time stop:** close at N days to expiration regardless, to avoid gamma risk in the
  final days.

Assignment note: legs are American-style. A deep-ITM short put near expiration can be
assigned early. The backtest should model early assignment when the short put is ITM
with minimal remaining time value, and any live implementation must close rather than
hold deep-ITM shorts into expiration.

## 6. Hard filters (do-not-trade conditions)

1. **Earnings inside the expiration.** If the underlying reports earnings before the
   option expires, elevated IV is an *event premium*, not a harvestable edge, and the
   position carries un-manageable overnight gap risk. Skip, or explicitly trade the
   post-earnings cycle once IV has normalized. This filter is mandatory.
2. **Other scheduled binary events** (FDA, major product events, macro prints for
   rate-sensitive names) inside the expiration — same treatment.
3. **Liquidity:** skip if either leg has open interest below a threshold (default
   100), daily volume below a threshold (default 20), or a bid/ask spread wider than
   a fraction of mid (default 12%). Illiquid spreads fill poorly and the modeled
   credit is not achievable.
4. **Move plausibility:** if the required drop from spot to the low target is more
   than ~1.5 standard deviations of the underlying's expected move over the days-to-
   low window (using current IV), the entry is unlikely to trigger — flag but do not
   necessarily block.

## 7. Pricing model

- Black-Scholes for European puts, used for both pricing and Greeks.
- Implied volatility is **solved from market bid/ask** (bisection inversion of BS),
  not assumed. Per-leg IV, because skew means the lower (long) strike carries higher
  IV than the short strike — using one flat IV overstates credit, always in the
  optimistic direction.
- Skew between the two quoted strikes is extrapolated linearly for any re-solved
  strikes. This is an approximation; real skew curves. The backtest should use actual
  per-strike IV from historical chains where available rather than extrapolating.
- Risk-free rate is a minor input; dividends are ignored for non-payers (fine for
  AMD; must be added for dividend payers, where they also affect early-assignment
  timing around ex-dates).

### Known model limitations (must be respected in the backtest)
- **Risk-neutral N(d2) is not a real-world probability.** It is used for a rough OTM
  estimate only and systematically understates downside in high-IV names. Do not use
  it as a win-rate input to sizing or expectancy.
- **Continuous-diffusion assumption breaks at gaps.** BS cannot price overnight jump
  risk. This is *why* the earnings filter exists; the backtest must use actual
  historical price paths (including gaps) for P&L, and use BS only for entry-day
  pricing and Greeks, never to simulate the P&L path.
- **Fills are modeled at mid in the tool; reality is inside the bid/ask.** The
  backtest must model realistic fills — at minimum, entering at a fraction of the way
  from mid toward the natural price, and paying the spread on both entry and exit.

## 8. Parameters (to expose and sweep in the backtest)

| Parameter | Default | Sweep range |
|---|---|---|
| Buffer % below low target | 3% | 1–10% |
| Short-strike selection method | buffer | {buffer, delta, prob-OTM} |
| Target short delta (if delta method) | 0.20 | 0.10–0.35 |
| Max loss per contract | $1,500 | budget-dependent |
| DTE at entry | 22 | 7–60 |
| Entry discipline | mechanical | {mechanical, confirmation} |
| Profit-target close | none (hold) | {none, 25%, 50%, 75%} |
| Stop rule | none | {none, level-break, 2× credit} |
| Time stop (DTE) | none | {none, 21, 14, 7} |
| Liquidity: min OI / min vol / max spread% | 100 / 20 / 12% | filter tuning |

## 9. Metrics the backtest must report

- Win rate, average win, average loss, expectancy per trade and per day-in-trade.
- Total return, CAGR, max drawdown, and drawdown duration.
- Sharpe and Sortino; profit factor.
- Distribution of P&L (histogram) — credit strategies have left-skewed returns
  (many small wins, occasional large losses); the tails matter more than the mean.
- Sensitivity of results to fill assumptions (mid vs. realistic) — if edge vanishes
  under realistic fills, the strategy is not real.
- Per-parameter-set comparison across the sweep, with attention to overfitting
  (out-of-sample / walk-forward validation, not just in-sample optimization).

## 10. Reference implementation

A single-position pricing/design calculator exists as a React component
(`spread-v2.jsx`) implementing the entry-side pricing: thesis inputs → proposed
strikes → two-pass IV solve from quotes → priced spread with Greeks, held-to-expiry
payoff, and the liquidity/earnings filters as warnings. It prices one candidate at
one point in time. The backtester generalizes this across historical time and price
paths. The Black-Scholes and IV-inversion functions in that component are correct
(validated against textbook values and put-call parity) and can be ported directly.

## 11. Worked example (the design case)

AMD, spot ~535, low target 475 (support) expected within ~6 days, high target 550,
22 DTE, IV ~89% (short) / ~91% (long). Buffer 3% → short strike 460, budget $1,500/ct
→ long strike ~435, width $25. Credit ≈ $10/contract. Full credit if AMD expires
above 460; max loss ≈ $1,486 below 435. **This exact trade is blocked by the earnings
filter** — AMD reported Aug 4, inside the Aug-14 expiration — and serves as the
canonical example of why the filter is mandatory: the 89% IV was an earnings premium,
not an edge.
