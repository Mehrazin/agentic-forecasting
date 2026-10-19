# `news/` — news RAG and fetchers

News retrieval for the agent's news tool. Two halves: the RAG that answers
queries from the cached corpus, and the fetchers that build the corpus.
The agent never hits a live news API — backfill scripts write into
`data/news/news_2026.db` (SQLite, gitignored) and the tool reads the cache.

| Module | What it is |
|---|---|
| `rag.py` | News RAG: BM25 → Groq rerank → abstain (offline-first KB) |
| `perplexity.py` | Perplexity fetcher — **avoid spending credits** (GDELT first) |

Design: [`docs/RAG_DESIGN.md`](../../docs/RAG_DESIGN.md). Corpus:
[`data/news/`](../../data/news/) (17,738 articles, 1979–2017, 4 macro topics).
Note: the news module is an honest negative on real data — the gate fires
90–100% but moves nothing. Kept for the pipeline, not the model.
Up: [`td_one/`](../README.md).
