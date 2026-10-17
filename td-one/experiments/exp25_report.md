# exp25: News at Longer Horizons (h=3) — Report

**Date:** 2026-10-17
**Status:** Complete — hypothesis rejected; news significantly worse at h=3
**Script:** `experiments/exp25_h3_horizon.py` → `experiments/exp25_results.json`

---

## Question

exp21's diagnosis: at h=1, ETS adapts to level shifts within 1-2 months, so
news features can't beat adaptive smoothing. But at h=3, ETS *cannot* have
adapted to shifts it hasn't seen yet — does anticipation matter more at
longer horizons?

## Method

- Same Phase-3 news-leading-biased DGP as exp24 (≥60% news_lead > 0).
- h=3: forecast the switch rate 3 months ahead (48-month series, 30-month
  training window, 12 origins × 10 segments, n=120).
- Models: ETS(h=3), ARIMA(h=3), NEWS (OLS: y[t+3] ~ y[t],y[t-1],y[t-2] +
  news_rate_shift_3m, news_policy_shock_3m, news_total_3m — the features
  exp21 found sensible, PIT-safe with 1-month lag).

## Results

| Model | Mean CRPS (h=3) |
|---|---|
| ETS | **0.0769** |
| ARIMA | 0.0799 |
| NEWS | 0.0962 |
| oracle | 0.0670 |

**Paired t-tests (n=120):**
- NEWS vs ETS: Δ = +0.0193 (+25%), t = −3.29, **p = 0.0013** (significantly worse)
- NEWS vs ARIMA: Δ = +0.0163 (+20%), t = −2.64, **p = 0.0093** (significantly worse)
- Oracle gap closed by NEWS: **−195%**

## Diagnosis: hypothesis rejected

The news edge does NOT appear at h=3 — it gets worse. Three compounding
causes:

1. **OLS overfits.** 6 features (3 lags + 3 news) on ~27 effective training
   points (30-month window minus h=3 alignment). The news features are
   sparse, noisy counts; the regression chases noise.

2. **The news→outcome link attenuates with horizon.** A 1-2 month news lead
   helps at h=1 (barely) but at h=3 the relevant question is "what happens
   in months t+1..t+3" — news at t covers events whose effects mostly land
   *inside* the horizon, but the OLS can't distinguish "effect in month 1"
   from "effect in month 3."

3. **ETS(h=3) is already decent.** Simple exponential smoothing with a
   3-step forecast degrades gracefully; the 0.0769 CRPS leaves only a 13%
   gap to oracle (vs 21% at h=1 in exp21's setup). Less headroom for news
   to add value.

## What this means

Combined with exp21 (h=1: −21.6%) and exp24 (regime: −1.4% gap), exp25
completes the picture: **on this DGP, news features don't help at h=1 OR
h=3**, whether used directly (OLS), for selection (regime), or for
discovery (exp21's loop).

The remaining possibilities:
- (a) The news↔DGP coupling is too weak/noisy in the current design —
  even with the Phase-3 news-leading bias, article counts are a coarse
  proxy for event effects.
- (b) The task needs explicit onset classification ("will an event hit in
  the next 3 months?") rather than regression on counts.
- (c) Real news (not synthetic templates) may carry richer signal — but
  that's beyond the simulator.

This is the honest end of the "news as features" line. The infrastructure
(event simulator, sandbox, tools, decision policy) is validated and
reusable; the *news-feature* hypothesis is falsified on this DGP at both
horizons tested.

## Reproducibility

```
cd td-one && .venv/bin/python experiments/exp25_h3_horizon.py
```

Deterministic. ~70s. 120 origins × 3 model fits.
