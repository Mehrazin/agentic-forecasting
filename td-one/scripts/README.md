# `scripts/` — data fetchers and utilities

| Script | What it does |
|---|---|
| `fetch_fred.py` | Download 19 FRED series → `data/fred/` |
| `fetch_news_gdelt.py` / `fetch_news_gdelt_gapfill.py` | GDELT news → `data/news/` (free, preferred) |
| `fetch_news_boc_backfill.py` | BoC 2009–2020 backfill → `data/news/` |
| `fetch_news_perplexity.py` | Perplexity news — **costs credits, avoid** |
| `bootstrap_models.py` | Download Chronos-2 + TimesFM-2.5 weights (see `models/MANIFEST.json`) |
| `scale_up.py` | Scale-up pipeline driver |

Runbook: [`docs/LIVE_NEWS_SETUP.md`](../docs/LIVE_NEWS_SETUP.md).
Up: [`td-one/`](../README.md).
