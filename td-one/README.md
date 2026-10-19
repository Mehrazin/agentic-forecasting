# TD-One: Agentic Mortgage Attrition Forecasting

Solo implementation for the Vector Agentic Forecasting Bootcamp by Vandad Davoodnia.

We forecast borrower attrition at Canadian mortgage renewal: for each portfolio segment and month, how many renewing borrowers switch lenders vs stay? We compare a 4-rung modeling ladder — Bayesian baseline (M0), zero-shot TSFMs (M1), ReAct LLM agent (M2), hybrid (M3) — plus agentic extensions (M4 regime reader, M5 router) under strict point-in-time evaluation.

**Start here:** [docs/README.md](docs/README.md) (doc index) → [docs/METHODOLOGY.md](docs/METHODOLOGY.md) → [docs/RESULTS.md](docs/RESULTS.md)

**Original plan:** [PLAN_SOLO.md](PLAN_SOLO.md) (the solo-build plan this implementation follows).

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
| [docs/plots/](docs/plots/) | Result plots — regenerate via `docs/plots/make_plots.py`. |
| [experiments/](experiments/) | All experiments (exp1–exp48). See [experiments/README.md](experiments/README.md). |
| [src/td_one/](src/td_one/) | Main package (see structure below; each subpackage has a README). |
| [scripts/](scripts/) | Data fetchers, model bootstrap, pipelines. |
| [configs/](configs/) | Run configs (reserved). |
| [tests/](tests/) | TDD test suite. |
| [benchmarks/](benchmarks/) | Real-data benchmarks: BoC, WTI, food CPI, M4, SMM. |
| [audits/](audits/) | News leakage + content audits. |
| [models/](models/) | Weight manifest + weights dir (gitignored). |
| [data/](data/) | Data dirs (gitignored) + per-dir READMEs. |

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
| Proposed method (exp47) | ✅ Alive — beats equal weights (0.1451 vs 0.1507, p=6.5e-09), best single, and exp43 router; confirmed on Tourism (exp60, p=1.9e-68) and M3 (exp61, p=2.7e-44) |
| LLM integration ladder (exp54–58) | ⚠️ Partial — rung 5 hybrid 0.4289 beats Vector's LLM 0.5688 by 24.6% (paired p=0.009) on Food CPI, but loses to its own best LLM expert (llm-promptV1 0.4262, p=0.058); rung 3 calibrator ❌ killed; guardrails reproduce exp49 bit-exact |
| exp43 regime router | ✅ Partial positive — first to beat equal weights (M4 0.1282 vs 0.1295, p=0.018) |
| Conformal M0 | ✅ Strong baseline — nothing beats it cleanly on synthetic |
| TSFMs | ✅ Beat m0 30% on 63-mo real SMM (exp18, p=0.0012); beat classical ~20% on M4 |
| Macro (FRED overlay) | ❌ Closed — 5-seed, no evidence macro helps (classical or TSFM) |
| M5 router | ❌ Dead — loses to inverse-CRPS even constrained |
| M2 ReAct / M3 hybrid | ❌ Dead — LLM numerical output is net-negative |
| News as feature (monthly) | ❌ Dead — exp46 definitive negative; exp19 frozen |
| Reasoning effort | ❌ "High" never helps — default to "low" for diagnostics (exp48) |

See [docs/REDESIGN.md](docs/REDESIGN.md) for the full kill/keep list with evidence.
