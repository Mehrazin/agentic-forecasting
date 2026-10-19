# `data/` — data (gitignored unless noted)

Historical data lives here. **Never commit data files** (exception: tiny
checked-in CSVs like the SMM monthly series). Rebuild via [`scripts/`](../scripts/).

| Dir | Contents | Rebuild |
|---|---|---|
| [`fred/`](fred/README.md) | 19 FRED series (parquet) | `scripts/fetch_fred.py` |
| [`freddie/`](freddie/README.md) | Freddie Mac raw + `smm_monthly_2021_2026.csv` (checked in) | raw files manual |
| [`m4/`](m4/README.md) | M4 Monthly train/test | manual download |
| [`news/`](news/README.md) | News corpus, FTS5 SQLite (`news_2026.db`) — 17,738 articles, 1979–2026 | `scripts/fetch_news_gdelt.py` |
| `news_dist/` | Split archives of the news DB (rebuilt 2026-10-18) | — |
| `scale_up/` | Scale-up histories/outcomes + PIT DuckDB | `scripts/scale_up.py` |
| `schemas/` | (reserved) | — |
| [`segments/`](segments/README.md) | Cached segment panels (parquet) | `src/td_one/segments/discovery.py` |
| `synthetic/` | (reserved) | — |

Up: [`td-one/`](../README.md).
