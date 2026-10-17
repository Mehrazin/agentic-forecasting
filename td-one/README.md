# TD-One: Agentic Mortgage Attrition Forecasting

Solo implementation for the Vector Agentic Forecasting Bootcamp by Vandad Davoodnia.

We forecast borrower attrition at Canadian mortgage renewal: for each portfolio segment and month, how many renewing borrowers switch lenders vs stay? We compare a 4-rung modeling ladder — Bayesian baseline (M0), zero-shot TSFMs (M1), ReAct LLM agent (M2), hybrid (M3) — plus agentic extensions (M4 regime reader, M5 router) under strict point-in-time evaluation.

**Start here:** [docs/README.md](docs/README.md) (doc index) → [docs/METHODOLOGY.md](docs/METHODOLOGY.md) → [docs/RESULTS.md](docs/RESULTS.md)

## Quickstart

```bash
cd td-one
uv sync
uv run pytest tests/ -q
```

Run an experiment:
```bash
uv run python experiments/exp16_freddie_smm.py
```

## Setup & installation

**Prerequisites:** Python 3.12+, `uv`.

```bash
# 1. Clone and enter
git clone https://github.com/VDavoodnia/agentic-forecasting.git
cd agentic-forecasting/td-one

# 2. Install dependencies
uv sync

# 3. Download TSFM weights (never committed; ~1.3GB)
export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"
uv run --with huggingface_hub python scripts/bootstrap_models.py

# 4. (Optional) Fetch FRED data — needs API key via vault connector
uv run python scripts/fetch_fred.py

# 5. (Optional) Build news corpus — GDELT is free, Perplexity needs credit
uv run python scripts/fetch_news_gdelt.py --year 2025
uv run python scripts/fetch_news_perplexity.py --year 2025

# 6. Verify
uv run pytest tests/ -q
```

See [experiments/REPRODUCIBILITY.md](experiments/REPRODUCIBILITY.md) for full reproduction steps.

## Repository map

| Path | What it is |
|---|---|
| [docs/](docs/) | All documentation. Start with [docs/README.md](docs/README.md). |
| [docs/METHODOLOGY.md](docs/METHODOLOGY.md) | Methodology reference (problem, ladder, PIT, metrics). |
| [docs/RESULTS.md](docs/RESULTS.md) | Current experiment results (R2 series). |
| [docs/REDESIGN.md](docs/REDESIGN.md) | Design decisions: what we keep/kill/try next. |
| [docs/DIAGRAMS.md](docs/DIAGRAMS.md) | Architecture diagrams. |
| [docs/plots/](docs/plots/) | Result plots (SMM series, RAG eval, agent comparison). |
| [experiments/](experiments/) | All experiments. See [experiments/README.md](experiments/README.md). |
| [src/td_one/](src/td_one/) | Main package (see structure below). |
| [scripts/](scripts/) | Data fetchers, model bootstrap, pipelines. |
| [configs/](configs/) | YAML configs for DGP, segments, models. |
| [tests/](tests/) | TDD test suite. |

## Package structure

- `src/td_one/`
  - `data_gen/` — synthetic mortgage simulator (history + structural outcome DGP)
  - `data_service/` — point-in-time DuckDB store
  - `freddie/` — Freddie Mac loan-level loader (SMM computation)
  - `fred/` — FRED live client + vintage-safe loader
  - `news/` — news store (SQLite FTS5), RAG (BM25 → Groq rerank → abstain), Perplexity client
  - `tools/` — agent tools: data_segment, evidence, forecasting
  - `models/` — M0 baseline, TSFM wrappers (+ calibration), macro overlays, hybrid readout
  - `agent/` — ReAct loop, regime reader (M4), ensemble router (M5), Groq client, retry
  - `evaluation/` — metrics (CRPS/MAE/coverage), leakage auditor, backtest runner
  - `segments/` — segment discovery (PIT-safe panels)

## Key design decisions

- **LLM**: Groq `openai/gpt-oss-120b` (via vault surrogate, never in code). The LLM never outputs numbers — only structure (classes, weights, confidences).
- **FRED**: Live via vault connector (`custom.fred`); bulk cache in `data/fred/` via `scripts/fetch_fred.py`.
- **TSFMs**: Chronos-2 / TimesFM-2.5 (CPU-compatible). [LogScaleCalibrator](src/td_one/models/tsfm_calibration.py) fixes near-zero miscalibration.
- **News**: Offline-first SQLite FTS5 KB. GDELT (free breadth) + Perplexity (depth). RAG: BM25 → Groq rerank → abstain.
- **Point-in-time**: All queries enforce `record_asof <= run_date`; leakage auditor verifies 0%.

## Current status

| Track | Verdict |
|---|---|
| M4 regime reader + news RAG | ✅ Alive — 3.59 vs M0 4.82; 3/4 end-to-end |
| Conformal M0 | ✅ Strong baseline — nothing beats it cleanly on synthetic |
| TSFMs | ✅ Beat m0 30% on 63-mo real SMM (exp18, p=0.0012); beat classical ~20% on M4 |
| Macro (FRED overlay) | ⚠️ Method-specific — lagged macro helps classical (-0.357), residual conditioning hurts TSFM |
| M5 router | ❌ Dead — loses to inverse-CRPS even constrained |
| M2 ReAct / M3 hybrid | ❌ Dead — LLM numerical output is net-negative |

See [docs/REDESIGN.md](docs/REDESIGN.md) for the full kill/keep list with evidence.
