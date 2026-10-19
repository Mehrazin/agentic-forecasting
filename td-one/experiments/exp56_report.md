# exp56 — Rung 3: LLM as calibrator of classical forecasts

**Date:** 2026-10-19T15:05:06.568481+00:00 | **Elapsed:** 1s
**Cache:** variant=`llm-calibrator`, model=`openai/gpt-oss-120b (Groq, reasoning_effort=low)`,
59/59 origins ok; **evaluated 59 origins**.

## Headline

| comparison | mean CRPS | diff | t | p | win rate |
|---|---|---|---|---|---|
| proposed+calibrated vs proposed | 1.1432 vs 0.9661 | +0.1771 | 1.01 | 0.3173 | 0.14 |
| proposed+calibrated vs calibrated alone | 1.1432 vs 1.1391 | +0.0041 | 0.12 | 0.901 | 0.56 |
| proposed vs calibrated alone | 0.9661 vs 1.1391 | -0.1730 | -0.97 | 0.3368 | 0.59 |
Mean CRPS: proposed+calibrated=1.1432, proposed=0.9661, calibrated alone=1.1391.

Classical experts alone: naive=0.9661, snaive=12.9264, ets=1.6708, arima=4.0975.

## Mechanism

With the calibrated grids available as an expert, the router's per-regime
experts on the full fit are: calm->naive, stressed->naive.

For reference: LLM-alone on the same origins is 0.5158 (exp55 cache,
regenerated generic prompt) and 0.5688 (exp53, Vector's run) — the
calibrated grids (1.1391) are far worse than the LLM forecasting directly.
The LLM's calibration skill needs direct quantile elicitation, not
post-hoc adjustment of a classical grid.

## Setup notes

- Classical records: the shared ladder pipeline (exp52/exp53): raw Food CPI,
  36-month train window per origin, `series_features(period=12)`; experts
  naive/snaive/ets/arima.
- The "llm" expert here holds the LLM's *calibrated* classical grids
  (rung-3 prompt asked the LLM to adjust the classical 9-quantile grid, not
  to forecast alone).
- `blocked_cv_predict` scores with a zeros(30) train proxy (exp47
  convention) rather than the real train window.
- Cache CRPS vs recomputed-from-grid max diff: 1.11e-16
  (validates the worker's stored CRPS against the evaluation scorer).

## PIT note

The cache is per-origin from the LLM worker; prompts must be pure functions
of (dataset, origin date, history <= origin) per `docs/LLM_INTEGRATION_LADDER.md`
rung 3. The classical records use only train windows <= origin. No
post-origin data enters the evaluation.
