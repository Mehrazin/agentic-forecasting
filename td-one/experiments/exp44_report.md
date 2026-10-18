# exp44: News-as-Selector — Report

**Date:** 2026-10-17
**Verdict: PARTIAL POSITIVE.** The mechanism is proven on synthetic data
(news significantly improves latent-regime prediction, p=0.0068), but the
effect is muted on real data by sparse news coverage and by backward
features already capturing the regime.

---

## Motivation

exp19/21/25 proved news fails as *features* — it does not improve the
forecast value. The literature says news works as *selection signals*: use
news to decide WHICH model to trust, not what the value should be.

exp44 tests this: take exp43's regime-probability router and ask whether
FORWARD-looking news signals improve regime classification over
backward-looking features alone. The target is regime accuracy, not CRPS —
if news cannot predict regimes, it is useless for monthly forecasting.

## Method

For origin date D, news features come from articles with
`pub_date in [D-30d, D)` — strictly before D (PIT-safe; same-day decision
reporting cannot leak). Keyword counts only, no LLM (deterministic).

- **BoC:** hawkish/dovish keyword counts, disagreement flag, net tone.
  Target: EASY/HARD regime. 143 meetings, 47 with news in window.
- **SMM:** article count, rate-keyword count, stress-keyword count/share.
  Target: calm/stressed. 50 origins, all with news.
- **Synthetic:** event simulator timeline (60 months, 13 events) + synthetic
  news feed (77 articles, PIT-safe by construction). Two targets:
  (a) backward-label regime (exp43-style, vol-based),
  (b) TRUE latent state = "event occurred in last 90d" (what news reports).

Classifier: LogisticRegression on standardized features, blocked 5-fold CV.
Comparison: backward-only vs backward+news. Paired t-test on correctness.

## Results

### Regime classification accuracy

| Dataset | Target | Base | +News | Δ | p |
|---|---|---|---|---|---|
| BoC | EASY/HARD | 0.264 | 0.295 | +0.031 | 0.20 |
| SMM | calm/stressed | 0.960 | 0.960 | +0.000 | 1.00 |
| Synth | backward labels | 0.872 | 0.850 | −0.021 | 0.32 |
| Synth | **TRUE latent state** | 0.367 | **0.522** | **+0.146** | **0.0068** |

### SMM full router CRPS (secondary)

News-augmented router CRPS = backward-only router CRPS to 4 decimals
(0.1843 vs 0.1843, p=1.0). Better regime prediction is moot when the
classifier is already at 96% — there is nothing left for news to add.

## Interpretation

**The synthetic true-label test is the clean experiment.** When ground truth
is "did an event recently occur" — the state news actually reports —
backward features alone achieve only 36.7% (they cannot see what has not
yet entered the series), while adding news raises this to 52.2%
(p=0.0068, consistent: news ≥ base in all 5 folds, strictly better in 3).
The mechanism works: news carries information about latent regime state
that backward features lack.

**Why it does not transfer to real data (yet):**

1. **BoC: news too sparse.** Only 47/143 meetings have any article in the
   30-day pre-meeting window (the 2021–2024 backfill is thin). The classifier
   cannot learn from mostly-zero features. Both base and news-augmented
   classifiers are bad (0.26–0.30) because EASY/HARD regimes shift over
   time — blocked folds put the COVID/inflation regime entirely in test.
2. **SMM: nothing left to predict.** Backward features already classify
   calm/stressed at 96%. News is redundant, not wrong.
3. **Synthetic backward labels: circular target.** Regimes defined from
   backward features are trivially predicted by backward features; news
   adds noise (p=0.32).

## What this closes

- News-as-**feature** (exp19/21/25): dead for monthly forecasting.
- News-as-**selector**: mechanism VALIDATED on synthetic (p=0.0068), but
  real-data deployment is blocked by (a) sparse historical news and
  (b) regimes already captured by backward features.

## Recommended next steps

1. **Live news API test.** The real use case is *today's* news predicting
   *this month's* regime — dense, current coverage, not the sparse backfill.
   If news-as-selector works anywhere, it is on live data.
2. **News for regime-shift detection specifically.** The value is at
   turning points (where backward features fail, as the 0.367 base shows),
   not in steady state. Gate the router: trust news only when backward
   features disagree.
3. **Do not close the news track**, but reframe it: news is a
   regime-shift detector, not a forecast input.

## Reproducibility

`cd td-one && .venv/bin/python experiments/exp44_news_selector.py`
Deterministic (fixed seeds), ~5s. News DB: `data/news/news_2026.db`
(gitignored, 9,581 articles).
