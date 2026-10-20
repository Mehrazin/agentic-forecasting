# exp71: Competitor Spread as Refi-Pressure Covariate — Report

**Result: ❌ HONEST NEGATIVE at current coverage.** The competitor spread
does not improve runoff forecasts through any channel tested.

## Data

- `competitor_rates` (Perplexity): 8 raw months with broker discount
  rates (2024-12 → 2026-04), forward-filled to 20 months.
- Spread = OSFI uninsured new-lending rate − broker discount 5yr rate.

## Three tests, three outcomes

1. **Router feature (exp71): zero effect.** v2 with spread feature gives
   *identical* forecasts to v2 without (Δ = 0.0%, n = 19 origins). The
   router's decisions never change.
2. **In-sample correlation: strong.** spread vs next-month total runoff
   rate: r = −0.85 (p < 0.001); controlling for rate level: r = −0.67
   (p = 0.001). Negative sign: wide spreads coincide with *low* runoff —
   wide spreads cluster in rising/volatile rate regimes where lock-in
   dominates the refi incentive.
3. **Out-of-sample predictive value: none.** Rolling spread-slope
   adjustment on naive: −295% MSE (n = 7 origins, p = 0.21). The
   in-sample correlation does not survive out-of-sample — with 8 raw
   broker months it cannot be distinguished from overfitting a handful
   of forward-filled points.

## Verdict

The DGP's simulated refi-spread mechanism does not transfer measurably to
aggregate real data *with this competitor-rate source*. Perplexity-
scraped rates are too sparse (8/21 months with broker data, one 429) for
a monthly covariate. A proper test needs either a real competitor-rate
time series (benchmark provider) or weekly pulls accumulated forward.

## Failure analysis (researcher pass, 2026-10-20)

**Why did it fail? Two stacked problems:**

1. **Same architectural limit as exp70:** the spread as a router feature
   produced *identical* forecasts (Δ = 0.0%). Regime labels are hand-rule
   defined on series-intrinsic features; the logistic router never weights
   the exogenous feature enough to flip a decision. Exogenous covariates
   have no channel through the current router — this is now a confirmed
   pattern (OSFI rate level in exp70, spread here). Fixing it would mean
   regime labels that condition on covariates — a v3 design change, not
   a data change.
2. **Data-source sparsity is the binding constraint.** The in-sample
   correlation (r = −0.85 raw, −0.67 controlling for rate level, both
   p ≤ 0.001) is tantalizing but rests on 8 raw broker months,
   forward-filled. Out-of-sample (n = 7 origins) the spread-slope
   adjustment *hurts* (−295% MSE). In-sample strength + out-of-sample
   collapse = overfitting a handful of forward-filled points. The
   negative sign itself is plausible lock-in economics (wide spreads
   cluster in rising-rate regimes where borrowers can't refinance).

**Variant tried:** direct correlation + rolling OOS slope adjustment
(collapse out-of-sample). **One more alternative worth one shot** when
coverage allows: the broker rate *change* (not spread level) as the
refi-incentive proxy, gated by the router's trust flag — but not before
coverage reaches 24+ real months.

**Recommendation:** keep the `competitor_rates` fetcher running weekly;
revisit the covariate design when coverage exceeds 24 real months. Until
then, exp71 is the documented floor, not a settled negative.

Files: `exp71_competitor_spread.py`, `exp71_results.json`.
