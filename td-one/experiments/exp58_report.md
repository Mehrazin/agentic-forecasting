# exp58 — Rung 5: hybrid router (classical + LLM experts)

**Date:** 2026-10-19T15:02:24.815391+00:00 | **Elapsed:** 246s
**Evaluated 59 Food CPI origins.**

## Headline

| comparison | mean CRPS | diff | t | p | win rate |
|---|---|---|---|---|---|
| hybrid vs classical proposed | 0.4289 vs 0.9661 | -0.5372 | -6.15 | 7.6e-08 | 0.86 |
| hybrid vs llm-generic alone | 0.4289 vs 0.5158 | -0.0869 | -1.87 | 0.06606 | 0.61 |
| hybrid vs llm-promptV1 alone | 0.4289 vs 0.4262 | +0.0027 | 1.94 | 0.05775 | 0.39 |
| hybrid vs llm-promptV2 alone | 0.4289 vs 0.4599 | -0.0310 | -1.18 | 0.2445 | 0.51 |
| hybrid vs llm-calibrated alone | 0.4289 vs 1.1391 | -0.7102 | -3.71 | 0.0004631 | 0.86 |
Mean CRPS: hybrid=0.4289, classical=0.9661, llm-generic=0.5158, llm-promptV1=0.4262, llm-promptV2=0.4599, llm-calibrated=1.1391. Vector LLM (exp53): 0.5688; proposed (exp53): 1.0339.

## Overtake verdict

✅ OVERTAKE (significant vs Vector, p=0.008964) — BUT the hybrid loses to its own best expert llm-promptV1 alone (0.4289 vs 0.4262, p=0.05775): the router adds no value when one expert dominates. The overtake is the LLM's, not the router's.

- Scalar: hybrid mean CRPS 0.4289 vs Vector LLM 0.5688 (exp53) — 24.6% better.
- Paired vs Vector's actual exp53 per-origin run (58 common origins): diff=-0.1403, t=-2.71, p=0.008964, win rate=0.71 — significant.

## Mechanism

With all four LLM experts available, the router's per-regime experts on the
full fit are: calm->llm-promptV1, stressed->llm-promptV1.

## Guardrail (classical-only vs exp49)

| dataset | n | max abs diff | verdict |
|---|---|---|---|
| m4 | 600 | 1.78e-15 | ✅ PASS |
| synthetic | 480 | 4.44e-16 | ✅ PASS |
| smm | 27 | 0 | ✅ PASS |
| boc | 143 | 0 | ✅ PASS |

All guardrails pass: ✅.

## Setup notes

- Hybrid = `ProposedForecaster(h=1, dataset="food_cpi")` with models
  naive, snaive, ets, arima, llm-generic, llm-promptV1, llm-promptV2, llm-calibrated — regime router + gates + conformal,
  all experts from per-origin cached grids; blocked 5-fold CV offline.
- Classical experts: naive/snaive/ets/arima (exp52/exp53 pipeline), raw Food
  CPI, 36-month train window, `series_features(period=12)`.
- `blocked_cv_predict` scores with a zeros(30) train proxy (exp47
  convention).
- Scalar references (Vector 0.5688, proposed 1.0339) are from exp53 on the
  same 59 origins; paired tests use aligned within-experiment CRPS vectors.

## PIT note

All LLM inputs are the audited per-origin caches (prompts pure functions of
(dataset, origin date, history <= origin); see
`docs/PROMPT_LEAKAGE_AUDIT.md`). Classical records use only train windows
<= origin. No post-origin data enters the evaluation.
