# `data/news/` — news corpus (gitignored)

`news_2026.db` — FTS5 SQLite, 7,100+ articles (2025–2026), enriched on every
fetcher hit. Backfill progress: `boc_backfill_progress.json`.

Fetchers: `scripts/fetch_news_gdelt.py` (free, preferred),
`scripts/fetch_news_perplexity.py` (costs credits — avoid).
Audits: [`audits/`](../audits/README.md). Up: [`data/`](../README.md).
