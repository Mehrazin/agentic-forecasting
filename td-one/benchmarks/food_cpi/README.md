# Food CPI

Food CPI forecasting benchmark — the dataset where LLMs proved their worth.

**Classical:** agent correctly discriminates seasonality across categories
(exp35/36). Proposed method: 1.0339 CRPS (59 origins).

**LLM track (the headline result):**
- exp53: Vector's `QuantileGridLLMPredictor` beats our proposed method
  decisively — 0.5688 vs 1.0339, p=5.4e-05. The LLM is a better probabilist
  on smooth macro series and knows the series identity ("Canada Food CPI").
- LLM ladder (exp55–59, Gemini Flash via Vector proxy): rung 5 hybrid 0.4526
  beats Vector by 20%, but best single LLM prompt (V1, 0.4262) beats the
  hybrid — the router adds nothing when one expert dominates. Rung 6 (exp59):
  clean negative on Tourism/M3 — the Food CPI win is dataset-specific.
- **Verdict:** LLM is a specialist for smooth macro series, not a general
  forecaster. Classical method stands elsewhere.

Data: `data.csv`, `data_multi.csv`. Scripts: `exp35_baseline.py`,
`../../experiments/exp53_vector_llm.py`, `../../experiments/exp55_run_llm.py`
(+ JSONs). Report: [`exp36_report.md`](exp36_report.md);
LLM results: [`../../experiments/exp53_report.md`](../../experiments/exp53_report.md),
[`../../experiments/exp58_report.md`](../../experiments/exp58_report.md),
[`../../experiments/exp59_report.md`](../../experiments/exp59_report.md).

Up: [`benchmarks/`](../README.md).
