# News RAG Design (TD-One)

Offline-first KB. The agent never calls a live API in the hot path.

## Why offline-first

We're backtesting on fixed 2026 data. The KB must be **reproducible**
(same query → same articles), **free** (no credit burn on repeated
experiments), and **PIT-safe by construction** (a frozen SQLite DB can't
leak the future). Live Perplexity is for KB *building/refreshing*
(offline, via `scripts/fetch_news_perplexity.py`), never for agents.

## KB schema

`td-one/data/news/news_2026.db` (SQLite + FTS5, gitignored, rebuilt by
scripts):

| table | contents |
|---|---|
| `articles` | `pub_date TEXT`, `title TEXT`, `url TEXT UNIQUE`, `source TEXT` |
| `articles_fts` | FTS5 index on `title` (BM25 ranking built in) |

Sources: GDELT DOC API (breadth, free) + Perplexity Agent API (depth,
`$20` credit). Dedupe by URL. ~1,400+ articles for Jan–Oct 2026 at last
count.

## Pipeline

```
RAGTool.search(query, before, limit)
  │
  ├─ Stage 1 (BM25): NewsStore.search(query, before, limit=20)
  │   FTS5 BM25 over titles, pub_date < before enforced in SQL.
  │   Returns up to 20 candidates, newest first.
  │
  ├─ Stage 2 (rerank): ONE Groq call, json_mode
  │   Prompt: query + all 20 candidates (date, title, snippet).
  │   Output: JSON array [{idx, relevance 0..1}].
  │   Sort by relevance desc, take top `limit`.
  │
  └─ Stage 3 (abstain): if max relevance < threshold (default 0.4)
      → return [], abstained=True. The caller falls back to the
      statistical baseline instead of acting on weak evidence.
```

### Rerank prompt (core)

System: "You are a relevance judge for mortgage-market news retrieval.
Score each article 0..1 for how directly it informs the analyst's
question. 1 = directly answers it; 0.5 = useful background;
0 = unrelated. Score junk 0 — never inflate. Reply ONLY with a JSON
array of {idx, relevance}."

User: the analyst question, then the candidates numbered 0..N-1 with
`[date] title — snippet`.

### Why LLM rerank on top of BM25

BM25 matches keywords, not intent. "Bank of Canada holds rate" and
"Bank of Canada hikes rate" both match "Bank of Canada rate" but imply
opposite regimes. The reranker reads meaning; BM25 just narrows the
field cheaply.

## PIT guarantees

1. `before` is mandatory (no default) — enforced in `RAGTool` and
   `NewsStore`.
2. Future cutoffs rejected (`before > today` → `ValueError`).
3. SQL filters `pub_date < before`; `RAGTool` re-checks every article
   client-side (belt and braces).
4. The KB is frozen for a backtest run — rebuilding mid-run is an
   operator action, never an agent action.

## Cost analysis (Groq free tier)

One rerank call per `RAGTool.search`: ~20 candidates × ~40 tokens ≈
800 input tokens + ~200 output tokens. At the 8,000 TPM free-tier
ceiling that's ~8 searches/minute sustained. The `NewsSearchTool`
already caps 10 searches/run; the RAG tool inherits that budget via
the shared `GroqChatClient.call_budget`. A 100-origin backtest with
2 searches/origin = 200 rerank calls ≈ well within the 1,000/day
free tier if spaced, or use a paid key.

## Failure modes

| Failure | Behavior |
|---|---|
| Groq 429 / empty / timeout | Fall back to BM25 order (log warning). Degraded but functional. |
| Rerank returns malformed JSON | Retry parse once (strip fences); else BM25 fallback. |
| All relevance < threshold | `abstained=True`, empty list. Caller uses baseline. |
| No BM25 candidates | Empty list, `abstained=True` (nothing to rerank). |
| Junk candidates (ads, dupes) | Reranker scores 0; BM25 fallback keeps newest-first order. |

## Tuning the abstention threshold

Default 0.4. Tune on a labeled set: sample ~50 (query, article) pairs,
have the reranker score them, pick the threshold maximizing F1 against
human relevant/irrelevant labels. Lower → more recall, more noise;
higher → cleaner context, more abstentions. Log the abstention rate per
experiment — a sudden jump means the KB is thin for that period.

## Evaluation: recall@k / precision@k

Evaluated on 100 labeled queries (`experiments/rag_eval_labels.json`) covering
Bank of Canada, mortgage rates, housing, and inflation topics with PIT-safe
cutoffs across 2026. BM25-only mode (deterministic, no Groq).

| k | Recall@k | Precision@k |
|---|----------|-------------|
| 1 | 0.156 | 0.750 |
| 3 | 0.373 | 0.628 |
| 5 | 0.486 | 0.515 |
| 10 | 0.528 | 0.327 |

![RAG recall/precision](plots/rag_recall_precision.png)

**Interpretation:** Precision@1 of 0.75 means when the system returns a single
article, it's usually relevant. Recall@10 of 0.53 means we find about half the
relevant articles in the top 10. The KB has heavy duplication (syndicated
articles), which hurts recall — deduping by story would help.

**Query preprocessing note:** Natural language queries must be converted to
FTS5 keywords (the eval does this via `to_fts_query`). Raw sentences with
FTS5's AND semantics return zero results. The `NEWS_SYSTEM_PROMPT` instructs
agents to write keyword queries.
