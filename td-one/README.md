# TD-One Agentic Mortgage Renewal Attrition Forecasting System

Solo implementation for the Vector Agentic Forecasting Bootcamp by Vandad Davoodnia.

## Quickstart

```bash
cd td-one
uv sync
uv run pytest tests/ -q
```

## Structure

- `src/td_one/` — main package
  - `data_gen/` — synthetic mortgage data simulator
  - `data_service/` — point-in-time DuckDB store
  - `tools/` — agent tools (segment, evidence, forecasting)
  - `models/` — M0 baseline, TSFM wrappers, hybrid readout
  - `agent/` — ReAct controller (Groq-powered)
  - `evaluation/` — metrics, leakage auditor, backtest runner
- `configs/` — YAML configs for DGP, segments, models, experiments
- `experiments/` — exp1 (MVP1), exp2 (segments), exp3 (ablation)
- `tests/` — TDD test suite

## Key design decisions

- **LLM**: Groq `openai/gpt-oss-120b` (via vault surrogate, never in code)
- **FRED**: Mocked for development (file cache in `data/fred/`)
- **TSFMs**: Chronos-2 / TimesFM-2.5 (CPU-compatible, per survey)
- **Point-in-time**: All queries enforce `record_asof <= run_date`; leakage auditor verifies 0%
