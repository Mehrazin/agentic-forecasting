# exp24: Regime-Aware Model Selection — Report

**Date:** 2026-10-17
**Status:** Complete — v2 ties best single; does not close the oracle gap
**Script:** `experiments/exp24_regime_aware.py` → `experiments/exp24_results.json`

---

## Question

exp23's clean negative: backward-looking series diagnostics can't predict
which model wins next month (selection ~= random). But the DGP has a latent
*regime* (event density) estimable from the news feed (Phase 1's information
asymmetry). Can regime-aware selection beat diagnostic selection and
single-best?

## Method

**Regime estimation (PIT-safe, at each origin):**
- `n_news_30d` / `n_news_90d`: article counts via `articles_visible_asof`
- `mean_abs_ate`: `backtest_event_impact` with news article dates as proxy
  event dates on the training window
- Classify: ACTIVE (≥3 articles/30d or |ATE|>0.10), ELEVATED (≥1/30d or
  ≥3/90d), else QUIET

**Selection (v2 mapping, post-hoc corrected):**
- QUIET → ARIMA (empirically best in quiet: 0.1028 vs 0.1063 ETS)
- ELEVATED → ensemble (ETS+ARIMA mean)
- ACTIVE → ETS (empirically best in active: 0.1249 vs 0.1384 ARIMA)

v1 mapped ACTIVE→event_aware (trailing-2 mean), which scored worse than all
baselines (gap closed −73%). The v1→v2 correction is documented in the script.

**Data:** 10 event-augmented segments (Phase-3 news-leading-biased timeline,
≥60% news_lead > 0), 40 months, 12 rolling origins each (n=120).

## Results

| Method | Mean CRPS | Oracle gap closed |
|---|---|---|
| oracle | 0.1052 | 100% |
| ets (best single) | 0.1203 | 0% |
| **regime (v2)** | **0.1205** | **−1.4%** |
| exp23 diagnostic | 0.1207 | −2.9% |
| random | 0.1246 | −28% |
| recency | 0.1276 | −49% |

**Paired t-tests (n=120):** regime vs exp23 p=0.92; regime vs best single
p=0.93; regime vs random p=0.28. Selection accuracy 38% (random: 33%).

**By segment event density:**

| Group | regime | best single | regime vs best |
|---|---|---|---|
| quiet (0/yr) | 0.0895 | 0.0895 (arima) | +0.0% |
| mid (2–3/yr) | 0.0994 | 0.0993 (ets) | +0.1% |
| eventful (4–8/yr) | 0.1571 | 0.1509 (ets) | **+4.1%** |

## Diagnosis

1. **v2 fixes v1's catastrophe** (−73% → −1.4% gap) but doesn't find alpha.
   Regime-aware selection is statistically tied with best-single and with
   exp23's diagnostic selection.

2. **The mapping is empirically correct but the gains are small.** In QUIET,
   ARIMA beats ETS by 3%; in ACTIVE, ETS beats ARIMA by 10%. The classifier
   (57 QUIET / 56 ACTIVE / 7 ELEVATED) captures this — but ETS alone is
   already the best single model, so "pick the right model per regime"
   collapses to "mostly pick ETS."

3. **Same failure pattern as exp23 in eventful segments** (+4.1% worse than
   ETS). When events are dense, some origins get misclassified as QUIET
   (news gap) and receive ARIMA, which then gets crushed by the event. The
   classifier isn't perfect, and errors are costliest exactly when
   adaptation matters.

4. **The 12% oracle gap remains unclosed.** The oracle wins by seeing the
   realized future; no backward-looking signal — series diagnostics (exp23)
   or news-implied regime (exp24) — replicates that.

## What this means

Regime-aware selection is a **robustness play, not an alpha play**: it never
does much worse than the best single model without knowing in advance which
one that is. But it doesn't beat "just use ETS."

The deeper lesson (consistent across exp21/23/24): on this DGP at h=1, the
news feed's anticipatory signal is too weak and too sparse to beat adaptive
classical models. The remaining hypotheses are (a) longer horizons where
anticipation matters more (exp25), or (b) a fundamentally stronger
event/news coupling in the DGP.

## Reproducibility

```
cd td-one && .venv/bin/python experiments/exp24_regime_aware.py
```

Deterministic. ~60s. 120 origins × 4 model fits + diagnostics + news.
