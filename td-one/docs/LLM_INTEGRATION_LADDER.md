# LLM Integration Ladder

**Goal:** Overtake Vector's `QuantileGridLLMPredictor` (0.5688 CRPS on Food CPI, exp53) while keeping our wins on M4/SMM/BoC/Synthetic.

**Status:** design approved 2026-10-18; rungs 1–5 evaluated 2026-10-19
(reports: `experiments/exp5{4,5,6,7,8}_report.md`). Rung 6 pending the two
new benchmark datasets.

**Ladder verdict (Food CPI, 59 origins):** rung 2 ✅ (proposed+llm 0.5057 beats
proposed 0.9661, p=9.8e-07; beats LLM alone 0.5158, p=0.037), rung 3 ❌ honest
negative (calibrated grids 1.1391; router can't use them), rung 4 ✅ V1 prompt
best (0.4262; V2's macro context adds nothing, p=0.20), rung 5 ⚠️ hybrid
0.4289 beats Vector 0.5688 by 24.6% (paired p=0.009) but loses to its own
best expert llm-promptV1 alone (0.4262, p=0.058) — the overtake is the LLM's,
not the router's. Guardrails reproduce exp49 bit-exact on M4/SMM/BoC/Synthetic.

## Why the LLM wins (exp53 diagnosis)

1. **Calibration, not memorization.** Our classical experts emit √h·σ intervals that are overdispersed on smooth macro series (monthly Food CPI moves ~0.5; our σ ≈ 1.5–2). The LLM's intervals are tighter and better centered (90% width ~1.1).
2. **Semantic context.** The LLM knows it is forecasting "Canada Food CPI" — inflation dynamics from pre-training are legitimate forecasting context. Our method sees only numbers.
3. **Direct quantile elicitation.** Structured JSON schema asks for q05–q95 directly; no distributional assumption, no sigma estimation step where error compounds.

## Pointers taken from Vector's design

- **Quantile-grid elicitation over distribution-fitting.** We fit σ then spread quantiles; they ask for the grid directly. Lesson: separate the *point* forecast from the *spread* — an LLM (or calibrator) can set spread better than √h·σ.
- **`series_description` / `user_prompt_suffix` config knobs.** Their predictor is *designed* for per-dataset customization. We adopt the same seam: prompts are data, not code.
- **`ForecastContext(as_of)` PIT discipline.** We mirror it: every prompt string is a pure function of (dataset, origin date, history ≤ origin).
- **Degenerate-case honesty.** Their `LastValuePredictor` is a documented floor. Our analogue: the router must be allowed to pick classical experts when the LLM is unavailable/uncertain — never force the LLM.

## Design rethink (what changes about our architecture)

- **Old belief:** "classical experts + statistical combination; LLM only as a gated sensor" (exp11/44 gating).
- **New belief:** the LLM is a *first-class expert* — the best probabilist on smooth, context-rich series. The router's job expands from "which classical model" to "which forecaster, including the LLM".
- **What stays:** regime router (exp43), gates (exp6), conformal (exp5), the kill list. The router is model-agnostic; adding an LLM expert is a data change, not an architecture change.
- **New module:** `LLMCalibrator` — LLM adjusts classical quantiles (spread correction) rather than replacing them. This is the hedge: if the LLM-as-expert overfits, the calibrator still harvests its calibration skill.

## Efficiency architecture (LLM calls are the bottleneck)

- Groq free tier: ~8000 TPM; exp53's 59 origins took 25–60 min.
- **Rule: run LLM inference once per (origin, prompt-variant); cache grids as JSON; all combination logic offline.**
- Rung 2 reuses `exp53_results.json` cache → **zero new LLM calls**.
- New LLM spend: rung 3 (59) + rung 4 (59 × 2 variants) ≈ 177 calls ≈ 1.5–3 h, sequential in one worker to respect TPM.

## Rung 1 — Semantic context for classical experts

- **Hypothesis:** adding dataset descriptions changes nothing for classical experts (they don't read text) — but formalizes the dataset→regime-spec mapping we currently do by hand.
- **Work:** `DATASET_SPECS` registry: dataset → (description, regime_spec, expert set). Verify exp49 numbers reproduce exactly with the registry wired in.
- **Expected:** null result, documented. The value is infrastructure for rungs 2–5.
- **LLM calls:** 0.
- **Result (exp54, ✅ expected null):** registry-wired forecaster reproduces exp49 exactly (max abs diff 1.8e-15 on M4, 0.0 on Food CPI). Report: `experiments/exp54_report.md`.

## Rung 2 — LLM as expert in the regime router

- **Hypothesis:** the exp43 router, given an "llm" expert alongside classical ones, learns *when* the LLM beats classical (e.g., smooth macro series) and routes accordingly.
- **Work:** build per-origin records where `quantiles["llm"]` and `crps["llm"]` come from the exp53 cache; run the standard blocked-5-fold CV fit/predict of `ProposedForecaster` with models=[..., "llm"]. The router's per-regime expert selection may pick "llm" for some regimes — that is the mechanism being tested.
- **Comparisons:** proposed+llm vs proposed (exp49) vs LLM alone (exp53).
- **LLM calls:** 0 (cache reuse).
- **PIT:** cache is per-origin from exp53, which used `ForecastContext(as_of)` — inherits its PIT safety.
- **Result (exp55, ✅ works):** proposed+llm 0.5057 beats proposed 0.9661 (p=9.8e-07) and beats LLM alone 0.5158 (p=0.037, win 0.63). Caveat: the router picks the LLM in *both* regimes — the gain is the LLM expert itself, not regime selection. Report: `experiments/exp55_report.md`.

## Rung 3 — LLM as calibrator of classical forecasts

- **Hypothesis:** the LLM's edge is calibration (spread), not location. Feeding it a classical forecast and asking for *adjusted* quantiles harvests the skill with less LLM autonomy (cheaper to trust, easier to gate).
- **Work:** new prompt: system = "You are a forecast calibrator…"; user = series meta + history + classical 9-quantile grid + instruction "return the calibrated 11-quantile grid". Uses `user_prompt_suffix` seam. Cache outputs.
- **Comparisons:** calibrated-classical vs raw classical vs LLM-alone.
- **LLM calls:** 59 (Food CPI).
- **PIT:** classical grid is computed on the train window only; prompt contains no post-origin data. Audited in `docs/PROMPT_LEAKAGE_AUDIT.md`.
- **Result (exp56, ❌ honest negative):** calibrated grids alone 1.1391 (worse than naive 0.9661); proposed+calibrated 1.1432 vs proposed 0.9661 (p=0.32, n.s., directionally worse). The router picks naive in every regime — it cannot use the calibrated grids profitably. The LLM's calibration skill needs direct quantile elicitation, not grid adjustment. Report: `experiments/exp56_report.md`.

## Rung 4 — Per-dataset prompts (PIT-safe)

- **Hypothesis:** richer pre-origin context (what the series *is*, recent regime description in words, all from history ≤ origin) improves the LLM expert without leaking labels.
- **Work:** prompt templates parameterized by (dataset, origin date):
  - Fixed block: series description, units, frequency, source (metadata, timeless).
  - Dynamic block: last observed value + date, 12m change, 3m change, "as of {date}" macro facts from a PIT-safe lookup table where every fact carries its own as-of date ≤ origin. **No numbers from after the origin. No actuals.**
  - Two variants: V1 (metadata + history stats in words), V2 (V1 + macro lookup). Implemented via `series_description` / `user_prompt_suffix`.
- **Audit:** `docs/PROMPT_LEAKAGE_AUDIT.md` — every template field traced to a pre-origin source; macro table versioned with as-of dates.
- **LLM calls:** 59 × 2 variants.
- **Comparisons:** V1 vs V2 vs generic (exp53) — is more context better, or does it distract?
- **Result (exp57, ✅ V1 wins, directional):** V1 0.4262 beats generic 0.5158 (p=0.057, suggestive but n.s.); V2 0.4599 adds nothing over V1 (p=0.20) — more context distracts rather than helps. Mini-router over prompt variants picks V1 in all 5 folds: selection adds nothing. Report: `experiments/exp57_report.md`.

## Rung 5 — Hybrid (the overtake attempt)

- **Architecture:** regime router over {classical experts, llm-generic, llm-V1, llm-V2, llm-calibrated} + gates + conformal. All components are per-origin cached grids; the router is fit offline per fold.
- **Work:** `experiments/exp58_hybrid.py` (numbering: exp54–57 are the rungs). Offline combination; no new LLM calls.
- **Comparisons:** hybrid vs each rung's best vs Vector LLM (0.5688) vs proposed (1.0339).
- **Guardrails:** rerun hybrid with LLM experts *ablated* on M4/SMM/BoC/Synthetic (no LLM outputs there) → must reproduce exp49 numbers exactly (no regression where the LLM is absent).
- **Success criterion:** hybrid < 0.5688 on Food CPI AND no guardrail regressions.
- **Result (exp58, ⚠️ overtake with a caveat):** hybrid 0.4289 beats Vector's 0.5688 by 24.6% (paired vs exp53's per-origin run: p=0.009, win 0.71) and guardrails reproduce exp49 bit-exact on M4 (1.8e-15), Synthetic (4.4e-16), SMM (0.0), BoC (0.0). BUT the hybrid loses to its own best expert llm-promptV1 alone (0.4262, p=0.058, win 0.39) — the router picks llm-promptV1 in both regimes and adds nothing beyond the dominant expert (same pattern as exp60's tourism TSFM). The overtake is the LLM's, not the router's. Report: `experiments/exp58_report.md`.

## Rung 6 — Generalizability (pending)

- The two new benchmark datasets (another agent). Run the rung-5 hybrid end-to-end: new LLM calls with the rung-4 prompt templates (they're dataset-parameterized by design) + classical experts + router.
- If the hybrid wins there too, the integration generalizes; if not, the Food CPI win was prompt/dataset-specific and we say so.

## Experiment numbering

- exp54 — rung 1 (context registry)
- exp55 — rung 2 (LLM as expert)
- exp56 — rung 3 (LLM calibrator)
- exp57 — rung 4 (per-dataset prompts)
- exp58 — rung 5 (hybrid)
- exp59 — rung 6 (new datasets, when they land)

## Files

- `docs/LLM_INTEGRATION_LADDER.md` (this file)
- `docs/PROMPT_LEAKAGE_AUDIT.md` — PIT audit for all prompt templates
- `td_one/llm_rungs/prompts.py` — prompt templates (pure functions of dataset/origin/history)
- `td_one/llm_rungs/datasets.py` — rung 1 dataset registry (description, regime spec, expert set, data path)
- `td_one/llm_rungs/cache.py` — JSON cache helpers for LLM grids
- `td_one/llm_rungs/combine.py` — offline combination (router over cached grids)
- `experiments/exp5{4,5,6,7,8}_*.py`, `exp5*_results.json`, `exp5*_report.md`
- `experiments/llm_food_cpi.py` — shared Food CPI classical-record builder for rungs 2–5 (verbatim exp52/53 pipeline)
- `experiments/exp5{5,6,7}_*_cache.json` — audited per-origin LLM grid caches (59/59 ok each)
