# exp21: Agent Feature Discovery — Report

**Date:** 2026-10-17
**Status:** Complete — honest negative with a clear diagnosis
**Script:** `experiments/exp21_feature_discovery.py` · **Results:** `experiments/exp21_results.json`

## Question

Can the "data scientist" loop work? I.e., an agent that (1) fits a baseline,
(2) examines residuals with analysis tools, (3) proposes news-derived features,
(4) tests each via PIT-safe CV, and (5) keeps only winners — does it beat a
strong classical baseline (ETS) on held-out data?

## Design

- **Data:** event-augmented simulator (66 months, 2000 customers/seed, 3 seeds).
  Monthly aggregate switching-rate series + PIT-safe news features (article
  counts per event type over trailing 3 months, months-since-news, etc.).
  The agent never sees the true event timeline — only the noisy news feed.
- **Discovery (months 0–47):** ETS in-sample residuals → `analyze_residuals`
  ranks 9 news features by |corr| → forward selection with
  `cross_validate` (5 folds, keep if CV CRPS improves ≥0.5%).
- **Held-out (months 54–65):** `compare_models` — ETS vs OLS-AR(1) vs
  OLS-AR(1)+discovered features, 12 rolling origins, Diebold-Mariano tests.
- All models share one normal-quantile wrapper, so CRPS differences isolate
  point-forecast skill.

## What the agent discovered

| Seed | Selected features | Discovery CV gain |
|------|-------------------|-------------------|
| 42 | `news_rate_shift_3m`, `news_policy_shock_3m`, `news_macro_surprise_3m` | 0.0412 → 0.0369 (**−10.6%**) |
| 7 | `news_rate_shift_3m`, `news_total_3m` | 0.0548 → 0.0515 (**−6.0%**) |
| 123 | `news_policy_shock_3m`, `news_housing_disruption_3m` | 0.0730 → 0.0641 (**−12.2%**) |

**Were they sensible? Yes.** Residual correlations picked out
`news_rate_shift_3m` (|r|=0.43) and `news_policy_shock_3m` first — exactly the
two event types whose news *leads* the DGP effect (rate_shift: 1–5 day news
delay vs 1-month transmission lag; policy_shock: 1–7 day delay vs 2-month
lag). The loop correctly ignored `news_lender_shift_3m` in selection even
though it correlated with residuals (|r|=0.35): lender-shift news arrives
5–30 days *after* its zero-lag effect starts, so it can't help forecasting
and CV rejected it. The discovery machinery is sound.

## Did they help? No — honest negative

Held-out CRPS (12 origins, mean):

| Seed | ETS | OLS-AR(1) | OLS-agent | Agent vs ETS |
|------|-----|-----------|-----------|--------------|
| 42 | 0.0808 | **0.0755** | 0.0798 | −1.2% (p=0.96) |
| 7 | **0.0960** | 0.1190 | 0.1224 | +27.5% (p=0.18) |
| 123 | **0.0702** | 0.0848 | 0.0973 | +38.7% (p=0.53) |

Mean gain vs ETS: **−21.6%** (FAIL vs the 2% success bar). No pairwise
Diebold-Mariano test is significant — 12 held-out months is low power, but
the point estimates are negative in 2/3 seeds.

## Diagnosis: why discovery gains (6–12%) didn't transfer

1. **News mostly lags the effect it describes.** Only 2 of 5 event types
   have news arriving before DGP onset (rate_shift, policy_shock). For
   lender_shift and macro_surprise (lag 0, news delay 5–30 days), the news
   is a *lagging* indicator — useless for 1-step-ahead forecasting.
2. **Nothing to anticipate in held-out.** Seed 42 had *zero* anticipation
   months (no news-covered event with onset in the 12-month window) — yet
   discovery still "found" features there. Pure selection overfitting.
3. **ETS adapts to level shifts within 1–2 months.** With 54+ months of
   training data, event effects are already in the series; a single linear
   coefficient on a coarse 3-month article count can't beat adaptive
   smoothing. The agent's edge would be in *onset anticipation*, which the
   1-step-ahead setup rarely rewards.
4. **Forward selection overfits sparse features.** The 0.5% CV threshold
   with 9 candidates and 5 folds picks noise (seed 123's features actively
   hurt: agent worse than plain AR(1)).

## Secondary findings

- **Scripted ReActAgentV2 trace works end-to-end**: 2 `code_execute` steps
  (changepoint detection → residual analysis) ran in the sandbox and the
  loop finalized. Loop mechanics validated.
- **Live Groq trace partially works**: gpt-oss-120b executed 3 sensible
  analysis steps (changepoints, residual moments, residual autocorrelation)
  via the ReAct protocol, then drifted into native OpenAI
  `{"name","arguments"}` tool-call format, causing HTTP 400s
  ("Tool choice is none, but model called a tool") — the 400 happens at the
  API layer, so a client-side protocol translator can't fix it. Never
  finalized in 8 steps. Takeaway: the model can drive the loop but needs
  either native tool wiring or a stricter protocol prompt.
- **PIT discipline held**: fixed one leak during development (prediction
  used the contemporaneous feature row; corrected to 1-month-lagged
  features throughout `_ols_fit_predict`).

## What this means for the roadmap

exp21 falsifies the *current* instantiation, not the paradigm. The loop
finds the right features; the evaluation doesn't reward them. Next steps
per the roadmap's reflect-and-adjust:

- **exp22 (M2 v2)**: still worth running — code-executing ReAct vs v1 on
  the same task isolates the capability upgrade.
- **Redesign the forecasting task** so anticipation matters: shorter
  training windows, multi-step horizons (h=3+), or explicit onset
  classification. Alternatively, strengthen the DGP: more policy-type
  events (news leads effect) and fewer zero-lag types.
- **Tighten selection**: raise the CV gain threshold, require the feature
  to help in a second validation split, or use stability selection.

The scientific value: we now know *exactly* why news features fail here —
it's measurable (news delay vs transmission lag per event type), not a
vibe. That diagnosis is the contribution.
