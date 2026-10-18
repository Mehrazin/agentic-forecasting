# exp46: News-as-Selector with Dense News — Report

**Date:** 2026-10-18
**Verdict: DEFINITIVE NEGATIVE.** Dense news does not help regime/selection
classification on real data, for any target, any news variant, any dataset.
The news track for monthly forecasting is closed.

---

## 1. What changed since exp44

exp44 proved the news-as-selector *mechanism* on synthetic data (news improved
true-latent-state classification 36.7% → 52.2%, p=0.0068) but was blocked on
real data by sparsity: only 47/143 BoC meetings had news in their 30-day
windows. The Perplexity backfill raised coverage to **123/143 (86%)**, verified:

| Variant | Meetings covered |
|---|---|
| all sources | 123/143 (86%) |
| perplexity_boc only | 76/143 (53%) |
| rate-keyword-filtered | 87/143 (61%) |
| pre-backfill (any source) | 47/143 (33%) |

This experiment re-tests with dense news. If news can't help now, sparsity was
never the binding constraint.

## 2. Design improvements over exp44

1. **Primary BoC target is rule-surprise** (`rule != actual`, 22/143 = 15.4%),
   not EASY/HARD. EASY/HARD is a deterministic function of the same macro
   features used as inputs, making that classification circular; surprise is
   what a selector actually needs to predict (when will the rule fail?).
2. **Temporal-confound controls.** Raw article counts correlate with calendar
   year (r=0.42): 2025–2026 meetings average ~294 articles/window (general-news
   flood) vs 1–2 in 2009–2016. We test three news variants: all sources,
   `perplexity_boc`-only (targeted backfill), and rate-keyword-filtered titles.
3. **SMM forecast-difficulty target** (top-tercile |ETS q50 error|) alongside
   the calm/stressed regime replication, since backward features already hit
   90–96% on regime classification (ceiling effect leaves no room for news).
4. **Feature importance** (standardized logistic coefficients) and a
   news-only baseline.

Method: `LogisticRegression(class_weight='balanced')` on base vs base+news
features, blocked 5-fold CV, paired t-test on correctness indicators.
PIT-safe: articles with `pub_date` in `[D-30d, D)` only.

## 3. Results

### BoC rule-surprise (primary, n=143, 22 surprises)

| Features | Accuracy | AUC | vs base p |
|---|---|---|---|
| base (4 macro) | 0.695 | 0.666 | — |
| + news_all | 0.681 | 0.664 | 0.481 |
| + news_boc | 0.681 | 0.651 | 0.481 |
| + news_rate | 0.683 | 0.615 | 0.529 |
| news_only | 0.448 | 0.463 | 0.000 (worse) |

No variant helps. News-only is significantly *worse* than macro alone.

Stratified check (76 meetings with BoC-targeted news only): news_boc vs base
p=0.568, AUC drops 0.411 → 0.327. The negative is not driven by zero-news
meetings diluting the comparison.

### BoC EASY/HARD (exp44 replication)

| Features | Accuracy | vs base p |
|---|---|---|
| base | 0.580 | — |
| + news_all | 0.536 | 0.132 |
| + news_boc | 0.509 | 0.073 |

Replicates exp44's null (there: 0.264→0.295, p=0.20). News does not help;
point estimates are negative.

### SMM forecast difficulty (n=50, top-tercile |ETS error|)

| Features | Accuracy | AUC | vs base p |
|---|---|---|---|
| base (10 feats) | 0.620 | 0.641 | — |
| + news_all | 0.580 | 0.613 | 0.485 |
| + news_rate | 0.580 | 0.570 | 0.420 |

No improvement.

### SMM calm/stressed (exp44 replication)

base 0.900 vs +news 0.900, p=1.00. Ceiling effect confirmed.

### Feature importance (in-sample, descriptive)

BoC surprise (`news_boc` variant): `news_hawk` gets +0.595 (second-largest
|coef| after `yield_spread` +1.047) — the classifier *finds* an in-sample
pattern (hawkish chatter → surprise). It does not generalize (CV p=0.48).
SMM hard: `news_stress_share` −1.119 and `news_stress` +0.512 have opposite
signs — classic overfit signature on noise features.

## 4. Interpretation

The pattern is consistent and damning: **news features acquire non-zero
in-sample coefficients but fail to generalize out-of-sample, across every
target (surprise, regime, difficulty), every variant (all/boc-only/
rate-filtered), and both datasets.** Three reasons:

1. **Keyword counts are too crude.** Sampled windows show relevant macro
   context (CPI releases, BoC communications) but almost no directional
   keywords: mean `hawk_kw` is 0.88 before hikes, 0.00 before cuts, and 1.72
   before holds (contrarian noise — hike speculation during long holds that
   never materializes).
2. **Temporal confound.** `news_n` is a calendar proxy (r=0.42 with year),
   not a regime signal. The targeted-backfill and keyword-filtered variants
   were designed to fix this; they don't help either.
3. **Real regimes are macro-driven.** The macro features already capture the
   fundamentals; news adds no orthogonal information at monthly frequency.
   On synthetic data the mechanism worked because the DGP made news the
   *only* carrier of latent-state information — real markets don't work that way.

## 5. What this closes

- **News-as-feature** (exp19/21/25): dead.
- **News-as-selector, sparse** (exp44 real-data): dead.
- **News-as-selector, dense** (exp46): dead. Sparsity was never the binding
  constraint.

The synthetic mechanism result (p=0.0068) stands as a proof of concept for
*what news could do in principle*, but real monthly news — even dense,
targeted, PIT-safe news — carries no usable regime signal for these tasks.

## 6. Remaining (narrow) openings — not recommended

- Higher-frequency news (daily/intraday) around event dates specifically.
- LLM-extracted stance (not keyword counts) — expensive, and the base rate
  of decision-relevant content in windows looks low.
- A live news API for *monitoring* (detect regime breaks as they happen)
  rather than *forecasting*.

None of these are justified by the evidence above. **Recommendation: close
the news track for monthly forecasting and reallocate to the regime router
(exp43), which is the only combination method that beat equal weights.**

---

## Reproducibility

`cd td-one && .venv/bin/python experiments/exp46_news_dense.py` —
deterministic, ~12s. `--quick` for smoke test.
Results: `experiments/exp46_results.json`.
