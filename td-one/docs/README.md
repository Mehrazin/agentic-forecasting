# TD-One Documentation Index

Start here. This page maps every document in `td-one/docs/` — what it is, who it's for, and where to go next.

## Start with these (5 min)

| Doc | What you get |
|---|---|
| [METHODOLOGY.md](METHODOLOGY.md) | The full methodology: problem, 4-rung ladder (M0–M3), PIT protocol, metrics. CVPR-style reference. |
| [RESULTS.md](RESULTS.md) | Current experiment numbers (R2 series). Every number links to a committed artifact. |
| [REDESIGN.md](REDESIGN.md) | What we keep, what we kill, what we try next — with evidence. Governs all future work. |

## Results & evidence

| Doc | What you get |
|---|---|
| [RESULTS.md](RESULTS.md) | R2 series: PIT-safe reruns. Primary metric CRPS. |
| [RESULTS_ARCHIVE.md](RESULTS_ARCHIVE.md) | exp1–exp15 (superseded by R2, preserved for provenance). |
| [DIAGRAMS.md](DIAGRAMS.md) | Architecture and experiment diagrams. |

## Audits (how we keep ourselves honest)

| Doc | What you get |
|---|---|
| [DATA_AUDIT.md](DATA_AUDIT.md) | 2026-10-16 data engineering audit: 2 P0, 5 P1, 5 P2 + fixes. |
| [CRITIC_SYNTHESIS.md](CRITIC_SYNTHESIS.md) | 6-angle critic panel findings, rounds 1–2. |
| [AGENT_AUDIT_M4.md](AGENT_AUDIT_M4.md) | Regime reader hardening: v2 prompt, ABSTAIN, counterargument. |
| [AGENT_AUDIT_M5.md](AGENT_AUDIT_M5.md) | Router post-mortem: why v1 failed, v2 design, kill decision. |
| [CITATION_AUDIT.md](CITATION_AUDIT.md) | 22 literature claims checked against primary sources. |

## Design docs

| Doc | What you get |
|---|---|
| [RAG_DESIGN.md](RAG_DESIGN.md) | News RAG: offline-first KB, BM25 → Groq rerank → abstain. Includes recall@k/precision@k eval. |
| [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md) | TSFM + agentic forecasting literature with verified citations. |
| [REAL_DATA_SURVEY.md](REAL_DATA_SURVEY.md) | Real dataset candidates for validation. |
| [SCALE_UP_RUNBOOK.md](SCALE_UP_RUNBOOK.md) | End-to-end pipeline: simulate → PIT → segments → backtest. |

## Data

| Asset | Location | Notes |
|---|---|---|
| News corpus | `data/news/news_2026.db` (gitignored) | 7,100+ articles (2025–2026). Rebuild via `scripts/fetch_news_gdelt.py` + `scripts/fetch_news_perplexity.py`. |
| SMM series | `data/freddie/smm_monthly_2021_2026.csv` | 63 months of Freddie Mac prepayment rates. |
| Plots | `docs/plots/` | `smm_2021_2026.png`, `rag_recall_precision.png`, `agent_m4_m5.png`. |
| FRED cache | `data/fred/` (gitignored) | 19 series. Rebuild via `scripts/fetch_fred.py`. |

## Experiments

See [experiments/README.md](../experiments/README.md) for the full experiment index (exp1–exp18, R2 series).
