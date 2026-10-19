# `data/news/` — news corpus (gitignored)

`news_2026.db` — FTS5 SQLite, **17,738 articles (1979–2026)**, enriched on
every fetcher hit: ~8,890 GDELT + ~8,850 Perplexity (incl. the 1979–2017
macro backfill: 4 topics × 468 months, complete 2026-10-18).
Distribution: `../news_dist/` (multipart zip, rebuilt 2026-10-18).

Fetchers: `scripts/fetch_news_gdelt.py` (free, preferred; `--years` /
`--topics` / `--maxrecords` / `--sleep` for historical runs),
`scripts/fetch_news_perplexity.py` (costs credits — avoid).
News-as-selector: exp44 (mechanism), `experiments/monash_common.py`
(`news_features_for_origin`, used by exp60/exp61 `--news`).
Audits: [`audits/`](../audits/README.md). Up: [`data/`](../README.md).
