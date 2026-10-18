# AGENTS.md — working in `td-one/`

Conventions for agents doing work here. General when possible, specific when
needed. The repo root `AGENTS.md` covers the bootcamp half; this file covers
TD-One, where the active research happens.

## Hard rules

### 1. PIT safety is a gate, not a guideline

No feature, label, or context may use information with `observed_asof >
origin`. Enforced by `data_service/pit_store.py` and the
`events_visible_asof()` / `articles_visible_asof()` accessors. Anything that
bypasses them is a **leak** — treat it like a failing test, not a follow-up.
`evaluation/leakage.py` audits for it; run it on any data-path change.

### 2. Docs are part of "done"

Any change to code, methods, datasets, or results must update the docs that
describe it **in the same change**. Before finishing:

1. `git grep` for what you touched across `docs/`, `experiments/`, READMEs.
2. Reconcile every hit: fix stale numbers, "planned/deferred" wording that's
   now real, file lists, commands, verdicts.
3. Update the layered docs together: the nearest doc, `docs/README.md`
   (index), `experiments/README.md` (verdicts), and `td-one/README.md`
   (status table) — not just the nearest one.

### 3. TDD + a regression test per fix

Every bug fix ships with a regression test. Test philosophy: non-obvious
logic, defensive contracts, error paths. Not Pydantic construction or trivial
behavior.

```bash
.venv/bin/pytest tests/ -q            # full suite
.venv/bin/ruff check src tests && .venv/bin/ruff format --check src tests
.venv/bin/mypy src
```

### 4. Honest negatives

A clean kill with evidence beats a vague positive. When an experiment fails,
record the verdict (`❌ honest negative`) in `experiments/README.md` with the
number, don't soften it, and don't rerun it without a new hypothesis. See
`docs/REDESIGN.md` for what each kill taught us.

## Research conventions

- **Metrics:** CRPS is primary (proper score on count distributions);
  secondary MAE and 80% interval coverage. Always report significance
  (paired t-test, win rate) — a bare mean difference is not a result.
- **Model ladder:** M0 (Beta-Binomial) → M1 (TSFMs) → M2/M3 (agents, dead) →
  M4/M5 (dead as wrappers) → proposed method (`agent/proposed_method.py`).
  New ideas get compared against the current best, not against M0 alone.
- **LLM:** Groq `openai/gpt-oss-120b` via `agent/groq_client.py` (vault
  surrogate — the key never enters the process). 2.5s throttle between calls;
  **wait out 429s** with the patient retry policy, don't work around them.
  Reasoning effort defaults to **low** for diagnostics (exp48: "high" never
  helps, sometimes hurts).
- **The LLM never outputs numbers** — only structure (regimes, confidences,
  selections). Numbers come from statistical code.
- **Tool tracking:** all agent tools carry `@tracked()`
  (`agent/tool_tracker.py`). Runtime data feeds `docs/TOOL_USAGE_DATA.md` →
  `docs/TOOL_AUDIT.md` → `docs/TOOL_OPTIMIZATION.md`. Check the audit before
  adding a tool; check the optimization list before the final report.

## Data conventions

- **Never commit data files.** `data/` is gitignored (READMEs inside are
  force-tracked docs, not data). Rebuild via `scripts/` (GDELT first —
  **avoid spending Perplexity credits**).
- **News cache:** enrich the local cache on every fetcher hit
  (`docs/LIVE_NEWS_SETUP.md`).
- **Weights:** gitignored; pinned in `models/MANIFEST.json`; loaders fail fast
  if missing. Fetch via `scripts/bootstrap_models.py`.

## Plots and diagrams

- Plots: generated only, via `docs/plots/make_plots.py` (shared style, source
  caption). Never hand-draw, never commit a plot you can't regenerate.
- Diagrams: mermaid, not ASCII. They render on GitHub.

## Commits

Work on `main`; commit early and often with messages that name what changed
and the evidence (`exp47: proposed method beats equal weights, p=6.5e-09`).
One logical change per commit; docs reconciliation rides along in the same
commit. Push when the work is green — approval cards may gate a push; wait,
don't work around them.
