# `fred/` — FRED macro series

| Module | What it is |
|---|---|
| `live.py` | Live FRED client via the vault key (`custom.fred`) |
| `cache.py` | Local parquet cache (19 series) |
| `mock.py` | Mock client for offline dev/tests |

Populate: `.venv/bin/python scripts/fetch_fred.py` → [`data/fred/`](../../data/fred/).
Note: the macro track is closed (no evidence macro helps) — this stays for the
data pipeline, not the model. Up: [`td_one/`](../README.md).
