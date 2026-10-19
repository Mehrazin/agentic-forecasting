# `td_one` package

TD-One's Python package — everything the research runs on. The arc: synthetic
mortgage data (`data_gen/`) → PIT-safe storage (`data_service/`) → model ladder
(`models/`) → agent tools (`agent/tools/`, `tools/`) → evaluation
(`evaluation/`, `backtest/`) → the proposed method (`agent/proposed_method.py`)
→ LLM integration (`llm_rungs/`).

| Package | Owns |
|---|---|
| [`agent/`](agent/README.md) | LLM agents: ReAct loops, regime reader, decision policy, sandbox, Groq client, and the proposed method |
| [`agent/tools/`](agent/tools/README.md) | The 13 granular tools in the agent namespace (`@tracked()`) |
| [`tools/`](tools/README.md) | The 3 v1 action tools (`data_segment`, `evidence`, `forecasting`) the ReAct loop dispatches |
| [`models/`](models/README.md) | Model ladder: M0 baseline, conformal, TSFMs, hybrids, ensembles, shrinkage |
| [`data_gen/`](data_gen/README.md) | Mortgage simulator (v1/v2), event timeline, synthetic news |
| [`data_service/`](data_service/README.md) | PIT-safe DuckDB store (asof joins, no future leakage) |
| [`evaluation/`](evaluation/README.md) | Metrics (CRPS/Brier/coverage), leakage auditor, real benchmarks |
| [`backtest/`](backtest/README.md) | Backtest runner over forecast origins |
| [`segments/`](segments/README.md) | Segment discovery and cohort panels |
| [`schemas/`](schemas/README.md) | Pydantic schemas (from the Excel spec) |
| [`fred/`](fred/README.md) | FRED macro series: live client, cache, mock |
| [`freddie/`](freddie/README.md) | Freddie Mac loan-level loader (PIT-unsafe by itself — filter by origin) |
| [`news/`](news/README.md) | News RAG (BM25 + Groq rerank) and Perplexity fetcher |
| [`llm_rungs/`](llm_rungs/README.md) | LLM integration ladder: dataset registry, prompt templates, grid caches, offline combination (rungs 1–6) |

Entry points most people want: `agent/proposed_method.py` (the method),
`evaluation/metrics.py`, `data_service/pit_store.py`.
Up: [`src/`](../README.md) · [`td-one/`](../../README.md).
