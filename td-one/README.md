# TD-One Agentic Mortgage Renewal Attrition Forecasting System

Solo implementation for the Vector Agentic Forecasting Bootcamp by Vandad Davoodnia.

## Quickstart

```bash
cd td-one
uv sync
uv run --with pytest pytest tests/ -q
```

## Structure

- `src/td_one/` — main package
  - `data_gen/` — synthetic mortgage data simulator (history + structural outcome DGP)
  - `data_service/` — point-in-time DuckDB store (history + outcomes tables)
  - `tools/` — agent tool interfaces (not yet implemented)
  - `models/` — M0 baseline ✅, TSFM wrappers ✅, hybrid readout ✅ (math only)
  - `agent/` — forecast-agent helpers (evidence prompt, bounded adjustment;
    ReAct orchestration loop not yet implemented)
  - `evaluation/` — metrics, leakage auditor (backtest runner not yet implemented)
- `configs/` — YAML configs for DGP, segments, models, experiments
- `experiments/` — exp1 (M0 vs M1 mocks), exp2 (M0 vs M1 real weights),
  `REPRODUCIBILITY.md`, per-experiment reports + `*_results.json`
- `scripts/` — `bootstrap_models.py` (download + pin TSFM weights)
- `models/` — `MANIFEST.json` (pinned revisions); `weights/` (gitignored)
- `tests/` — TDD test suite

## Model weights

TSFM weights are bootstrapped reproducibly (never committed):

```bash
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
uv run --with huggingface_hub python scripts/bootstrap_models.py
```

See `experiments/REPRODUCIBILITY.md` for the full reproduction steps.

## Key design decisions

- **LLM**: Groq `openai/gpt-oss-120b` (via vault surrogate, never in code)
- **FRED**: Mocked for development (no live key needed yet; file-cache location TBD)
- **TSFMs**: Chronos-2 / TimesFM-2.5 (CPU-compatible, per survey)
- **Point-in-time**: All queries enforce `record_asof <= run_date`; leakage auditor verifies 0%
