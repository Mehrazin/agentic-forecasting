# `fred/` — FRED macro series

Federal Reserve macro data: rates, inflation, labor, housing, FX, credit.
Live client, local parquet cache, and a mock for offline dev/tests.

| Module | What it is |
|---|---|
| `live.py` | Live FRED client via the vault key (`custom.fred`) |
| `cache.py` | Local parquet cache of curated macro series |
| `mock.py` | Mock client for offline dev/tests |

Populate: `.venv/bin/python scripts/fetch_fred.py` → [`data/fred/`](../../data/fred/).
Note: the macro track is closed (no evidence macro helps — classical or TSFM
side) — this stays for the data pipeline, not the model.
Up: [`td_one/`](../README.md).
