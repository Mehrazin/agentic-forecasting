# Experiment 3 Report: M2 (ReAct LLM Agent) vs M0

**Date:** 2026-10-15 (run_date: 2025-06-30).
 **Mode:** mock (scripted LLM).
Forecast backend: timesfm_mock (interface-only mock on the real grid).

The scripted ReAct agent runs the 3-tool trace
(data_segment → evidence → forecasting) per pilot segment; the
evaluated M2 forecast is the forecasting tool's rate-quantile output.
M0 is the trailing-6 Jeffreys Beta-Binomial on the same panels,
tested on the holdout batch the agent never saw (PIT-clean).

| Segment | M0 CRPS | M2 CRPS | M0 MAE | M2 MAE | M2 steps | evid |
|---|---|---|---|---|---|---|
| seg-0 | 0.76 | 1.11 | 2.00 | 1.28 | 4 | 5 |
| seg-1 | 1.33 | 2.85 | 2.00 | 3.00 | 4 | 5 |
| seg-2 | 1.40 | 1.83 | 2.00 | 2.00 | 4 | 5 |

M2 beats M0 on CRPS in 0/3 segments (mock LLM; plumbing validation only).

## Caveats

- Mock mode: the LLM is scripted, so this validates the ReAct
  orchestration plumbing (tool dispatch, trace, finalize parsing),
  not LLM forecasting skill. A real-LLM run is the follow-up.
- M2 CRPS uses a normal approximation from the tool's 80% interval;
  rough calibration check only.
- Pilot panels are small synthetic batches; the scale-up pipeline
  (`scripts/scale_up.py`) evaluates on real monthly segment panels
  from the full simulator.
- Evidence retrieval returned 15 cards across segments (frozen illustrative corpus).
