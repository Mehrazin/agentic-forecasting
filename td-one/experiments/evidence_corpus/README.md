# Evidence corpus (frozen fixture)

`corpus.json` is a **frozen illustrative fixture** — 12 synthetic entries
(BoC/OSFI/5-year-bond items, 2024-01 to 2025-06) used by the M2 ReAct agent's
evidence tool. It is **not real news** and must not be cited as such.

The file is a top-level JSON list; each entry has `id`, `date` (ISO),
`title`, `text`, `source`. `td_one.tools.evidence.evidence_tool` enforces the
point-in-time cutoff (`date <= run_date`) on read.
