# News Corpus Distribution

Pre-built SQLite news database for TD-One's RAG system.

## Contents

- `news_2026_db.zip` (+ `.z01`, `.z02`, `.z03`): multipart zip of `news_2026.db`
- **9,581 articles** (2021–2026): 6,741 GDELT + 806 Perplexity (+ 2026 data)
- SQLite + FTS5, PIT-safe (all `pub_date` verified ≤ fetch date)

## Reassemble

```bash
cd td-one/data/news_dist
zip -s 0 news_2026_db.zip --out news_2026.db.zip
unzip news_2026.db.zip
mv news_2026.db ../news/news_2026.db
```

Or in Python:
```python
import zipfile
with zipfile.ZipFile('news_2026_db.zip') as z:  # reads all parts automatically
    z.extractall('.')
```

## Rebuild from scratch

```bash
uv run python scripts/fetch_news_gdelt.py --year 2021  # free, no key needed
uv run python scripts/fetch_news_perplexity.py --year 2021  # needs Perplexity credit
```

See `docs/RAG_DESIGN.md` for the schema and PIT guarantees.
