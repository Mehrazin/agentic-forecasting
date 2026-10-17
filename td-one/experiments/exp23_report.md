# exp23: Diagnostic Model Selection — Report

**Date:** 2026-10-17
**Status:** Complete — clean negative for v1 rule-based policy
**Script:** `experiments/exp23_model_selection.py` → `experiments/exp23_results.json`

---

## Question

M5 (LLM router) failed because it was vibes-based. Can *diagnostic-driven*
selection — using statistical tests on the training window to pick from a
model zoo — beat single-best, random, and recency baselines?

## Method

**Data:** 10 event-augmented simulator segments (Phase 1), event density
0–8/year, 40 months each. Aggregated to monthly switch-rate series.

**Backtest:** 12 rolling origins per segment (120 total). 24-month training
window. At each origin:
1. Run diagnostics: `detect_changepoints`, `test_stationarity`,
   `decompose_series`, volatility ratio
2. Rule-based policy (defined a priori):
   - recent changepoint (last 6 obs) → ETS
   - non-stationary → ETS
   - trend_strength > 0.7 → ETS
   - seasonal_strength > 0.5 → ARIMA
   - vol_ratio > 1.2 → M0
   - default → ARIMA
3. Fit all 3 models (M0, ETS, ARIMA(1,1,1)), construct normal quantiles
   from in-sample residual std, score CRPS

**Baselines:** best single model (ARIMA, chosen in hindsight), oracle
(per-origin best), random (M5-vibes proxy), recency (last winner).

## Results

| Method        | Mean CRPS | vs best single |
|---------------|-----------|----------------|
| oracle        | 0.1042    | −12.2%         |
| single_arima  | 0.1187    | —              |
| **select**    | **0.1196**| **+0.8%**      |
| random        | 0.1203    | +1.4%          |
| single_ets    | 0.1210    | +1.9%          |
| recency       | 0.1230    | +3.6%          |
| single_m0     | 0.1300    | +9.5%          |

**Oracle gap closed by selection: −6.3%** (negative — selection moved
*away* from oracle relative to just using ARIMA).

**Paired t-tests (n=120):**
- select vs random: Δ = −0.0007, p = 0.80 (not significant)
- select vs single_arima: Δ = +0.0009, p = 0.51 (not significant)
- select vs oracle: Δ = +0.0154, p < 0.0001 (significantly worse)

**Selection accuracy:** 36.7% (random baseline: 33.3%).

## Diagnosis: why the policy failed

**1. The diagnostics don't discriminate.**
Mean diagnostic values by oracle winner:

| Oracle winner | n_recent_cps | nonstat | trend | seasonal | vol_ratio |
|---------------|--------------|---------|-------|----------|-----------|
| ets (n=43)    | 0.00         | 0.00    | 0.46  | 0.71     | 1.37      |
| arima (n=43)  | 0.00         | 0.05    | 0.45  | 0.73     | 1.25      |
| m0 (n=34)     | 0.00         | 0.12    | 0.48  | 0.83     | 1.39      |

The training-window diagnostics are nearly identical regardless of which
model wins next period. `n_recent_cps` is 0.00 for *all* groups — the
changepoint detector fires (98/120 origins have ≥1 changepoint), but they're
never in the last 6 observations, so the policy's first rule never triggers.

**2. The winner is determined by the future, not the past.**
On event-augmented data, which model wins 1-step-ahead depends on whether
an event hits *next* month — information that training-window diagnostics
cannot contain. The oracle wins by seeing the realized future; no
backward-looking statistic can replicate that.

**3. The policy asks the wrong question.**
It asks "what does the training series look like?" when it should ask
"what regime are we in?" The DGP has a latent regime (event density), and
the optimal model depends on the regime. Series-shape diagnostics don't
estimate the regime.

**4. Per-segment variance is extreme.**
Oracle gap closed ranges from +73% (seg2) to −326% (seg5) across segments
(n=12 origins each — noisy). Selection helps slightly in quiet segments
but hurts in eventful ones, suggesting the policy misfires exactly when
adaptation matters most.

## What this means

**v1 rule-based diagnostic selection is statistically indistinguishable
from random selection** (p=0.80) and does not beat "just use ARIMA"
(p=0.51).

This is a clean negative, and it's informative:

1. **Simple diagnostic rules don't capture the oracle's 12% edge.**
   The edge is real (oracle beats best single by 12.2%), but training-window
   series diagnostics can't access it.

2. **The path forward is regime-aware selection (v2), not series-shape
   selection.** The agent should estimate the latent event regime — e.g.,
   via `backtest_event_impact` on recent data, or by reading the news feed
   (Phase 1's information asymmetry). "Are we in a high-event regime?" is
   answerable; "which model wins next month?" is not, from series alone.

3. **This validates the roadmap's "data scientist" framing.** A human
   data scientist wouldn't just look at ACF plots — they'd ask *why* the
   series behaves this way, check the news, and reason about regimes.
   The v2 agent needs those capabilities.

## Reproducibility

```
cd td-one && .venv/bin/python experiments/exp23_model_selection.py
```

Deterministic (fixed seeds). Runtime ~90s. All 120 origins, 3 model fits
each, plus diagnostics.

## Files

- Script: `experiments/exp23_model_selection.py`
- Results: `experiments/exp23_results.json` (includes per-record diagnostics)
- This report: `experiments/exp23_report.md`
