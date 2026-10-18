# exp53: Vector's LLM Predictor vs Our Proposed Method

**Status:** complete — 58/59 origins scored (1 LLM hard failure, see below).

## What was tested

Vector's `QuantileGridLLMPredictor` — their flagship continuous LLM
forecaster: one structured JSON completion per origin eliciting the full
11-quantile grid (`STANDARD_QUANTILES`), via LiteLLM. This is the real
predictor class from `aieng-forecasting`, not a reimplementation.

## How it was run

- **Data/origins:** identical 59 Food CPI origins as exp52 (2021-08 → 2026-07,
  h=1); PIT-safe via Vector's own `ForecastContext(as_of)`.
- **Model:** `openai/gpt-oss-120b` via Groq (Vector's Gemini proxy is
  unavailable). Tests their elicitation *architecture*, not their deployment.
- **Adaptations** (all documented in `docs/VECTOR_METHODOLOGY_AUDIT.md`):
  local auth proxy (vault-surrogate key never in process), model-id prefix
  normalization, `additionalProperties:false` re-added for Groq strict mode,
  `max_tokens` 16384→2048, `reasoning_effort="low"` (exp48).
  Kept faithful: `temperature=1.0`, full history window, JSON-schema elicitation.
- **Scoring:** both methods scored with OUR 9-level quantile CRPS on the same
  origins (subset of their 11 levels). Relative rankings are robust to scorer
  choice (r=0.9998, see audit).

## Results

| Method | Mean CRPS (58 matched origins) |
|---|---|
| **Vector LLM (quantile grid, gpt-oss-120b)** | **0.5688** |
| TD-One proposed method | 1.0339 |

- Difference (LLM − proposed): **−0.4651** (LLM 45% better)
- Paired t-test: t = −4.36, **p = 5.4e-05** ✅ significant
- Win rate: **LLM wins 43/58 origins** (74%)

**The LLM predictor beats our proposed method decisively on Food CPI.**

## Interpretation

1. **This is real forecasting, not memorization.** Year-by-year LLM CRPS:
   2021: 0.50, 2022: 1.06, 2023: 0.41, 2024: 0.37, 2025: 0.49, 2026: 0.58.
   The worst year is 2022 (peak inflation volatility — hardest to forecast),
   not the most recent. No training-cutoff step pattern. Performance tracks
   forecast difficulty, consistent with genuine skill.
2. **Likely mechanism: calibration.** Our experts emit sqrt(h)·σ intervals
   that are overdispersed on this smooth series (monthly moves ~0.5, our
   CRPS ~1.03 implies predictive σ ≈ 1.5–2). The LLM's intervals are much
   tighter and better centered (smoke test: median 252.9 vs actual 253.17,
   90% width ~1.1). On smooth macro series, the LLM is simply a better
   probabilist than our classical experts.
3. **Semantic context matters.** The LLM knows it is forecasting "Canada Food
   CPI" — inflation dynamics from its training are legitimate forecasting
   context, the same kind a human forecaster uses. Our method sees only
   numbers. This is a design difference, not leakage.
4. **Scope:** one dataset (Food CPI), one model, one run at temperature=1.0.
   Our M4/Food-CPI-vs-classical results stand; this result says an LLM
   elicitation approach can beat classical experts where we don't.

## Robustness notes

- LLM predictor hard-failure rate: **1/59 origins**
  (`json_validate_failed` — the gpt-oss reasoning-tax empty-content mode).
  Vector's backtest would count these as skipped; our method never fails
  to produce a forecast. Small robustness edge to us.
- Single run at temperature=1.0: an unknown share of CRPS variance is LLM
  sampling noise. A 3-seed repeat is the correct follow-up (TPM-limited here).
- TPM pressure: their default full-history prompts (~2300 tokens) repeatedly
  hit Groq's 8000 TPM cap; Vector's built-in transient retry handled it.

## Links

- `docs/VECTOR_METHODOLOGY_AUDIT.md` — every methodological difference found
- `docs/SELF_AUDIT.md` — threats to our own evaluation integrity
- `experiments/exp53_vector_llm.py`, `exp53_analyze.py`, `exp53_results.json`,
  `exp53_comparison.json`
