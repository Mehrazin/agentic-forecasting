# Contributing to TD-One

## Branches and commits

- Work on `main`. Commit early and often — one logical change per commit.
- Commit messages name the change and the evidence:
  `exp47: proposed method beats equal weights (p=6.5e-09)`.
- Docs reconciliation rides in the same commit as the code change
  (see `docs/AGENTS.md` §2). A commit that changes behavior but leaves docs
  describing the old reality is a regression.

## Pull requests

PRs are welcome but not required for the core team — direct commits to `main`
are the norm here. If you open a PR:

1. Fill in the PR template (`.github/pull_request_template.md`).
2. CI must be green: `unit_tests.yml` + `code_checks.yml`.
3. Link the experiment report or audit that motivates the change.

## Adding an experiment

1. Script: `experiments/expNN_<name>.py` — one hypothesis, PIT-safe by
   construction, writes `expNN_results.json`.
2. Report: `experiments/expNN_report.md` — setup, numbers with significance,
   **verdict line**.
3. Register it: add a row to `experiments/README.md` (verdict key:
   ✅ ⚖️ ❌ 🔨 ▶️).
4. If it changes the story, update `docs/RESULTS.md` and the status table in
   `td-one/README.md`.

Numbering: continue from the highest `expNN` (exp26–exp39 were never
assigned — don't reuse them).

## Adding a tool

Check `docs/TOOL_AUDIT.md` first (does it already exist? was it killed?),
then `docs/TOOL_OPTIMIZATION.md` (is it on the add list?). New tools get
`@tracked()` from day one.

## Data and secrets

- Never commit data files or model weights.
- Never put API keys in code — use the vault surrogate pattern
  (`agent/groq_client.py`).

## Getting help

Start at [`docs/README.md`](README.md) (doc index) →
[`experiments/README.md`](../experiments/README.md) (what's been tried).
