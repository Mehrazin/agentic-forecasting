# exp55 — Rung 2: LLM as an expert in the regime router

**Date:** 2026-10-19T14:45:06.761500+00:00 | **Elapsed:** 1s
**Cache:** variant=`llm-generic`, model=`openai/gpt-oss-120b (Groq, reasoning_effort=low)`,
59/59 origins ok; **evaluated 59 origins**.

## Headline

| comparison | mean CRPS | diff | t | p | win rate |
|---|---|---|---|---|---|
| proposed+llm vs proposed | 0.5057 vs 0.9661 | -0.4603 | -5.48 | 9.802e-07 | 0.81 |
| proposed+llm vs LLM alone | 0.5057 vs 0.5158 | -0.0101 | -2.13 | 0.03739 | 0.63 |
| proposed vs LLM alone | 0.9661 vs 0.5158 | +0.4503 | 5.34 | 1.631e-06 | 0.17 |
Mean CRPS: proposed+llm=0.5057, proposed=0.9661, LLM alone=0.5158.

## Mechanism

With the LLM available as an expert, the router's per-regime experts on the
full fit are: calm->llm, stressed->llm.
(The hypothesis under test: the router learns *when* the LLM beats classical
experts — e.g. on smooth macro stretches — and routes accordingly.)

## Setup notes

- Classical records: raw Food CPI values, 36-month train window per origin,
  `series_features(period=12)`; experts naive/snaive/ets/arima — the
  exp52/exp53 classical pipeline.
- Regime spec comes from the rung-1 registry (`dataset="food_cpi"` ->
  generic calm/stressed). exp53_analyze's proposed number (1.0339) used a
  hand-passed m4 spec with per-fold expanding-window train records, so it is
  context here, not the test statistic; the paired comparisons above are.
- `blocked_cv_predict` scores with a zeros(30) train proxy (exp47
  convention) rather than the real train window.
- Cache CRPS vs recomputed-from-grid max diff: 5.55e-17
  (validates the worker's stored CRPS against the evaluation scorer).

## PIT note

The cache is per-origin from the LLM worker; prompts must be pure functions
of (dataset, origin date, history <= origin) per `docs/LLM_INTEGRATION_LADDER.md`
rung 2. The classical records use only train windows <= origin. No
post-origin data enters the evaluation.
