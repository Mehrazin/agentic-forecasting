# TD-One Documentation Index

Start here. This page maps every document in `td-one/docs/` — what it is, who it's for, and where to go next. Everything under `td-one/` is self-contained: no document here depends on anything outside `td-one/`.

## Start with these (5 min)

| Doc | What you get |
|---|---|
| [FINAL_RESULTS.md](FINAL_RESULTS.md) | **Start here.** Final numbers only — no experiment codes, human-readable names, benchmarks in tables. |
| [METHODOLOGY.md](METHODOLOGY.md) | The full methodology: problem, glossary, all tracks (exp1–exp63 + LLM ladder). The reference. |
| [REDESIGN.md](REDESIGN.md) | What we keep, what we kill, what we try next — with evidence. Governs future work. |

## The story (for new readers)

| Doc | What you get |
|---|---|
| [reports/MANAGER_REPORT.md](reports/MANAGER_REPORT.md) | **Start here.** Plain-language project summary with glossary, diagrams, and tables (final, 2026-10-19). |
| [reports/FINAL_SYNTHESIS.md](reports/FINAL_SYNTHESIS.md) | Technical synthesis across all datasets and tracks (final, 2026-10-19). |

## Results & evidence

| Doc | What you get |
|---|---|
| [RESULTS.md](RESULTS.md) | Current results: all tracks (exp1–exp63 + LLM ladder). Primary metric CRPS. |
| [RESULTS_ARCHIVE.md](RESULTS_ARCHIVE.md) | exp1–exp15 (superseded by R2, preserved for provenance). |
| [POSITIVE_LIFT_AUDIT.md](POSITIVE_LIFT_AUDIT.md) | Inventory of every positive result (exp1–exp46 + benchmarks) used to design the proposed method. |
| [NONSIGNIFICANT_POSITIVES.md](NONSIGNIFICANT_POSITIVES.md) | Audit of every positive-but-nonsignificant effect (exp1–exp50) with combinability verdicts; evidence base for exp51. |
| [EVALUATION_INTEGRITY.md](EVALUATION_INTEGRITY.md) | Leakage audit, Monash protocol adherence, and Chronos-2/TimesFM-2.5 pretraining-contamination matrix (exp60/exp61). |
| [SELF_AUDIT.md](SELF_AUDIT.md) | Adversarial self-review: multiple comparisons, M4 development bias, BoC-discrete downgrade, PIT re-verification, CRPS verification. Headlines survive; margins disclosed. |
| [VECTOR_METHODOLOGY_AUDIT.md](VECTOR_METHODOLOGY_AUDIT.md) | Every methodological difference vs Vector's stack (CRPS scorer, quantile grids, LLM config, baselines); evidence base for exp52/exp53. |
| [DIAGRAMS.md](DIAGRAMS.md) | Architecture and experiment diagrams (mermaid — renders on GitHub). |
| [../experiments/README.md](../experiments/README.md) | Full experiment index, exp1–exp63 + LLM ladder, with verdicts and report links. |

## The agentic track

| Doc | What you get |
|---|---|
| [AGENTIC_ROADMAP.md](AGENTIC_ROADMAP.md) | The approved roadmap: from output-layer wrappers to true agents. Governs agentic work order. |
| [PROPOSED_METHOD.md](PROPOSED_METHOD.md) | The exp47 method: architecture, implementation (`src/td_one/agent/proposed_method.py`), verdict. |
| [AGENT_AUDIT_M4.md](AGENT_AUDIT_M4.md) | Regime reader hardening: v2 prompt, ABSTAIN, counterargument. |
| [AGENT_AUDIT_M5.md](AGENT_AUDIT_M5.md) | Router post-mortem: why v1 failed, v2 design, kill decision. |

## Tool infrastructure (audits & optimization)

| Doc | What you get |
|---|---|
| [TOOL_AUDIT.md](TOOL_AUDIT.md) | Tool call frequency, patterns, failures across exp21–exp47 and benchmarks. |
| [TOOL_OPTIMIZATION.md](TOOL_OPTIMIZATION.md) | Recommendations: which tools to add, drop, fix — implement before the final report. |
| [TOOL_USAGE_DATA.md](TOOL_USAGE_DATA.md) | Runtime tool-call data from `@tracked()` instrumentation. |

## Audits (how we keep ourselves honest)

| Doc | What you get |
|---|---|
| [DATA_AUDIT.md](DATA_AUDIT.md) | 2026-10-16 data engineering audit: 2 P0, 5 P1, 5 P2 + fixes. |
| [CRITIC_SYNTHESIS.md](CRITIC_SYNTHESIS.md) | 6-angle critic panel findings, rounds 1–2. |
| [CITATION_AUDIT.md](CITATION_AUDIT.md) | Literature claims checked against primary sources. |
| [reports/ARGON_REVIEW_2.md](reports/ARGON_REVIEW_2.md) | Second-pass review of the external auditor's 6 findings: what's addressed, where it's mistaken, what we missed. |

## Design docs

| Doc | What you get |
|---|---|
| [RAG_DESIGN.md](RAG_DESIGN.md) | News RAG: offline-first KB, BM25 → Groq rerank → abstain. Includes recall@k/precision@k eval. |
| [LITERATURE_SURVEY.md](LITERATURE_SURVEY.md) | TSFM + agentic forecasting literature with verified citations. |
| [LITERATURE_AGENTIC_SURVEY.md](LITERATURE_AGENTIC_SURVEY.md) | Agentic forecasting, model selection, news augmentation — informing the post-exp23/exp37 methodology. |
| [LITERATURE_SURVEY_AGENTIC.md](LITERATURE_SURVEY_AGENTIC.md) | Methodology-improvement survey: selection under instability, regime detection, leakage-safe feature discovery. |
| [REAL_DATA_SURVEY.md](REAL_DATA_SURVEY.md) | Real dataset candidates for validation. |
| [SCALE_UP_RUNBOOK.md](SCALE_UP_RUNBOOK.md) | End-to-end pipeline: simulate → PIT → segments → backtest. |
| [SYNTHETIC_CALIBRATION.md](SYNTHETIC_CALIBRATION.md) | Phase-4 event DGP calibration: 9.1pp → 2–3pp effects, decay, interactions. |
| [REDTEAM_TSF.md](REDTEAM_TSF.md) | TSFM red-team protocol (exp67): 7-shift loan taxonomy, breaking-point design, results summary. |
| [GROUPING_EXAMPLES.md](GROUPING_EXAMPLES.md) | Worked examples for the grouping track (exp66d): 4 cohorts end-to-end — rule → members → forecast → risk → citation-grounded explanation, incl. the failure case. |

## Plans

Active and historical plans live in [plans/](plans/). The agentic roadmap stays top-level because it governs current work.

| Doc | Status |
|---|---|
| [AGENTIC_ROADMAP.md](AGENTIC_ROADMAP.md) | Approved — trilogy complete (exp42–44), proposed method shipped (exp47) |
| [LOAN_ATTRITION_PLAN.md](LOAN_ATTRITION_PLAN.md) | Active — loan attrition program: Track 1 (grouping), Track 2 (TSFM red-team exp67 ✅ + guard exp68 ❌ + anticipatory exp69 ❌), integration |
| [plans/NEXT_PHASE_PLAN.md](plans/NEXT_PHASE_PLAN.md) | Complete — regime router shipped; live-news track closed |
| [plans/MULTI_DATASET_PLAN.md](plans/MULTI_DATASET_PLAN.md) | Complete — validated on M4, Tourism, M3 |
| [plans/OPTION1_JOINT_TRAINING_PLAN.md](plans/OPTION1_JOINT_TRAINING_PLAN.md) | Assessed — naive joint training likely fails; FFORMA-style meta-learning is the path |
| [plans/OPTION3_METALEARNING_PLAN.md](plans/OPTION3_METALEARNING_PLAN.md) | Designed — learn the selection policy from data (led to exp42) |

## Operations

| Doc | What you get |
|---|---|
| [LIVE_NEWS_SETUP.md](LIVE_NEWS_SETUP.md) | Live news: BoC backfill + cache enrichment runbook. |
| [../experiments/REPRODUCIBILITY.md](../experiments/REPRODUCIBILITY.md) | Reproduce any experiment: environment, weights, commands. |

## Data

| Asset | Location | Notes |
|---|---|---|
| News corpus | `../data/news/` (gitignored) | Articles 2025–2026, FTS5 SQLite. Enriched on every fetcher hit. |
| SMM series | `../data/freddie/smm_monthly_2021_2026.csv` | 63 months of Freddie Mac prepayment rates. |
| FRED cache | `../data/fred/` (gitignored) | 19 series. Rebuild via `../scripts/fetch_fred.py`. |
| M4 data | `../data/m4/` | M4 competition series used by the agentic track. |
| Plots | [plots/](plots/) | Regenerated, embedded figures — see its README for sources. |

## Agent instructions

| Doc | What you get |
|---|---|
| [AGENTS.md](AGENTS.md) | Working conventions for agents in `td-one/`: TDD, PIT-safety gate, metrics, honesty policy. |
| [GUIDELINES.md](GUIDELINES.md) | Contribution guidelines for this project. |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute: branches, commits, PRs. |
