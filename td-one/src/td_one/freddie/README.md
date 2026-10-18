# `freddie/` — Freddie Mac loan-level loader

Reads pipe-delimited origination + monthly performance files, validates
schemas, builds cohort panels. **PIT SAFETY: the loader does NOT enforce
origin filtering** — callers must filter performance rows to
`reporting period < origin` (see `docs/DATA_AUDIT.md` P0 findings).

Data: [`data/freddie/`](../../data/freddie/) (raw files gitignored).
Up: [`td_one/`](../README.md).
