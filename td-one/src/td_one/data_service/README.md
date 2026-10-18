# `data_service/` — PIT-safe data store

`pit_store.py`: DuckDB store with asof joins. The hard rule — **no feature may
use information with `observed_asof > origin`**. Two mechanisms enforce it:
outcome filtering (`outcome_observed_at <= origin`) and the `events_visible_asof()`
/ `articles_visible_asof()` accessors. Anything bypassing these is a leak;
`evaluation/leakage.py` audits for it.

Up: [`td_one/`](../README.md).
