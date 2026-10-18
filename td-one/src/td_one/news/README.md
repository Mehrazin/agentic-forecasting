# `news/` — news RAG and fetchers

| Module | What it is |
|---|---|
| `rag.py` | News RAG: BM25 → Groq rerank → abstain (offline-first KB) |
| `perplexity.py` | Perplexity fetcher — **avoid spending credits** (GDELT first) |

Design: [`docs/RAG_DESIGN.md`](../../docs/RAG_DESIGN.md). Corpus:
[`data/news/`](../../data/news/). Up: [`td_one/`](../README.md).
