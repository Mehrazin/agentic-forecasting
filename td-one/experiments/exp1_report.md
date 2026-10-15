# TD-One Experiment 1: M0 vs M1 (MVP-1)

**Date:** 2026-10-15
**Status:** ✅ Complete — pipeline validated end-to-end

## Setup

- **Data:** 500 synthetic customers, 24 monthly snapshots
- **Task:** Predict switch counts for 6 forward periods
- **Train:** 12 historical periods
- **Test:** 6 forward periods

## Models

| Model | Description | MAE |
|-------|-------------|-----|
| M0 | Historical baseline (Jeffreys Beta-Binomial) | 3.50 |
| M1a | Chronos-2 (mock, amazon/chronos-2) | 3.26 |
| M1b | TimesFM-2.5 (mock, google/timesfm-2.5-200m) | 3.32 |

## Observations

1. All three models achieve comparable MAE on this synthetic task.
   Mocks are calibrated to the historical rate, so this is expected.

2. The pipeline is validated end-to-end:
   - Synthetic data generation → PIT store → Model fitting → Forecasting → Evaluation

3. Next steps for real comparison:
   - Replace mocks with actual Chronos-2 / TimesFM-2.5 weights
   - Use 10,000 customers × 66 months (full spec)
   - Add M2 (LLM evidence) and M3 (hybrid) to the ladder

## Reproducibility

```bash
cd td-one
uv run python experiments/exp1_mvp1.py
```
