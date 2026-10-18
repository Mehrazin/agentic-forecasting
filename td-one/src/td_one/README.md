# `td_one` package

TD-One's Python package. Subpackages:

| Package | Owns |
|---|---|
| [`agent/`](agent/README.md) | LLM agents: ReAct, regime reader, router, proposed method, sandbox, Groq client |
| [`tools/`](tools/README.md) | The 10 statistical tools agents can call (`@tracked()`) |
| [`models/`](models/README.md) | Model ladder: M0 baseline, TSFMs, hybrids, conformal, ensembles |
| [`data_gen/`](data_gen/README.md) | Mortgage simulator (v1/v2), event timeline, synthetic news |
| [`data_service/`](data_service/README.md) | PIT-safe DuckDB store (asof joins, no future leakage) |
| [`evaluation/`](evaluation/README.md) | Metrics (CRPS/Brier/coverage), leakage auditor, real benchmarks |
| [`backtest/`](backtest/README.md) | Backtest runner over origins |
| [`segments/`](segments/README.md) | Segment discovery and panels |
| [`schemas/`](schemas/README.md) | Pydantic schemas (from the Excel spec) |
| [`fred/`](fred/README.md) | FRED macro series: live client, cache, mock |
| [`freddie/`](freddie/README.md) | Freddie Mac loan-level loader (PIT-unsafe by itself — filter by origin) |
| [`news/`](news/README.md) | News RAG (BM25 + Groq rerank) and Perplexity fetcher |

Entry points most people want: `agent/proposed_method.py` (the method),
`evaluation/metrics.py`, `data_service/pit_store.py`.
Up: [`src/`](../README.md) · [`td-one/`](../../README.md).
