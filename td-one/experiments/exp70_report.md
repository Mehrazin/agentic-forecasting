# exp70: OSFI M4 Runoff Benchmark — Report

**Result: ❌ HONEST NEGATIVE (kill criterion triggered).** ProposedForecasterV2
does not beat naive on real Canadian mortgage runoff.

## Data

- OSFI M4 (StatCan table 10100006), 2016-07 → 2026-07, 121 months.
- Target: implied runoff rate = runoff / outstanding[t−1], per segment.
- Protocol: last 60 origins, 36m train, h=1, 5-fold blocked time CV.

## Results (CRPS, runoff-rate units)

| Segment | v2 | naive | Δ | p | win |
|---|---|---|---|---|---|
| **total** | 0.00207 | 0.00207 | −0.2% | 0.54 | 0.43 |
| insured | 0.00328 | 0.00308 | +6.5% | 0.199 | 0.37 |
| **uninsured** | 0.00262 | 0.00236 | **+10.7%** | **0.030** | 0.27 |
| variable_insured | 0.00828 | 0.00809 | +2.4% | 0.491 | 0.45 |
| variable_uninsured | 0.00281 | 0.00285 | −1.6% | 0.649 | 0.52 |

v2 ties best-single classical everywhere except uninsured, where it is
significantly *worse* — the router picks non-naive experts on a series
where naive is ex-ante best, and pays for it.

## Interpretation

This is the proposed method's first real-data runoff benchmark, and the
answer is a wash-to-negative: monthly aggregate runoff is dominated by
scheduled maturities and amortization — smooth, near-random-walk — and
the regime machinery adds variance without adding signal. The method's
wins (M4/Tourism/M3) are on *cross-sectional ensembles of many series*;
a single aggregate portfolio series is a different regime.

## Failure analysis (researcher pass, 2026-10-20)

**Why did it fail? Three diagnoses, all tested:**

1. **Target is near-unforecastable beyond naive.** AC1 of the runoff rate
   is 0.45–0.72 across segments; standalone expert CRPS: naive ≈ arima on
   every segment, ets worse, snaive far worse (AC12 only 0.11–0.36 — weak
   seasonality). There is almost no predictable innovation left after
   lag-1 persistence. The router's uninsured loss (+10.7%, p=0.030) is
   the cost of occasionally picking ets/snaive.
2. **The router has no channel for exogenous features.** Adding the OSFI
   rate level (the lock-in variable) as a router feature produced
   *identical* forecasts (Δ = 0.0%, p = 0.32): regime labels come from
   the spec's hand rules on series-intrinsic features, and the logistic
   router's weights on the extra feature never flip a decision at n=60.
   This is architectural — see exp71 for the same failure mode with the
   competitor spread.
3. **Horizon:** h=3 shows a faint v2 edge vs naive (p = 0.081, n=46 —
   flagged: crude tiled quantiles), h=6 degrades significantly
   (+24%, p = 0.023). No horizon rescues the design.

**Variants tried:** rate-level-augmented features (null), h=3/h=6
(marginal/worse). **Conclusion stands, now explained:** on a single
aggregate series the proposed method is the wrong tool; its edge is
cross-sectional. OSFI runoff remains valuable as the honest real-data
null benchmark for any future runoff claim.

## Caveats

- OSFI month M is observed as-of M+2 (publication lag not modeled in CV;
  affects all arms equally).
- Aggregate runoff mixes refi, moves, sales, defaults — not pure
  lender-switching attrition.

Files: `exp70_osfi_benchmark.py`, `exp70_results.json`.
