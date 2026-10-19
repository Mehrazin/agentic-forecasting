# `freddie/` — Freddie Mac loan-level loader

Reads pipe-delimited origination + monthly performance files, validates
schemas, builds cohort panels — the real mortgage data behind the SMM track.

**PIT SAFETY: the loader does NOT enforce origin filtering** — callers must
filter performance rows to `reporting period < origin` (see
`docs/DATA_AUDIT.md` P0 findings). The vintage files contain the full history;
using a row dated after your forecast origin is a leak.

Data: [`data/freddie/`](../../data/freddie/) (raw files gitignored).
Up: [`td_one/`](../README.md).
