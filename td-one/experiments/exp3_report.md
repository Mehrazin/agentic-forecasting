# Experiment 3 Report: M2 (ReAct LLM Agent) vs M0

**Date:** 2026-10-16 (run_date: 2025-06-30).
 **Mode:** real (Groq).
Forecast backend: timesfm_mock (interface-only mock on the real grid).

The ReAct agent runs the 3-tool trace
(data_segment → evidence → forecasting) per pilot segment; the
evaluated M2 forecast is the forecasting tool's rate-quantile output.
M0 is the trailing-6 Jeffreys Beta-Binomial on the same panels,
tested on the holdout batch the agent never saw (PIT-clean).

| Segment | M0 CRPS | M2 CRPS | M0 MAE | M2 MAE | M2 steps |
|---|---|---|---|---|---|
| seg-0 | 0.76 | 5.22 | 2.00 | 6.24 | 6 |
| seg-1 | 1.33 | 0.82 | 2.00 | 0.14 | 5 |
| seg-2 | 1.40 | nan | 2.00 | nan | 6 |

M2 beats M0 on CRPS in 1/2 finalized segments (real Groq).

## Caveats

- ⚠️ **Honesty note:** real-mode M2 still uses the `timesfm_mock`
  numerical backend (`exp3_results.json: "forecast_model":
  "timesfm_mock"`), so the seg-1 edge cannot be separated from the mock
  sharpness artifact. What this run demonstrates is that the agent can
  drive the tool loop to finalization and produce a winning trace — the
  orchestration claim, not the numerical claim.
- This was the **third real-mode attempt**: attempt 1 (naive retry) →
  1/3 finalized; attempt 2 (2.5s throttle) → 1/3; attempt 3 (patient
  retry: 6 attempts, 60s floor on 429) → 2/3 finalized + first win. The
  binding constraint on the free tier is rate limits, not model skill.
- The evidence-card diagnostic is **broken in real mode**: it returns
  the −1 sentinel for all three segments (no cards counted). Flagged as
  a known data-quality gap, not a measurement.
- M2 CRPS uses a normal approximation from the tool's 80% interval;
  rough calibration check only.
- Pilot panels are small synthetic batches; the scale-up pipeline
  (`scripts/scale_up.py`) evaluates on real monthly segment panels
  from the full simulator.
