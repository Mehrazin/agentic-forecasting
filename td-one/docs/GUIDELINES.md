# TD-One Guidelines

How we build here. These aren't suggestions — they're the working agreement
for anyone (human or agent) contributing to TD-One.

## What "done" means

A change is done when **all** of these hold:

1. **Tests green.** `.venv/bin/pytest tests/ -q`, `ruff check`, `mypy src`
   clean. A bug fix includes its regression test.
2. **PIT-safe.** No future leakage — verified by `evaluation/leakage.py` for
   data-path changes, by construction (asof accessors) otherwise.
3. **Docs reconciled.** The docs that describe what you changed say the new
   truth — in the same commit. See `AGENTS.md` §2.
4. **Honestly reported.** Numbers with significance, verdicts without spin.
   Negatives get recorded, not buried.

## Research standards

- **One hypothesis per experiment.** `experiments/expNN_name.py` +
  `expNN_results.json` + `expNN_report.md` with a verdict line.
- **Compare against the current best**, not a strawman. The ladder is in
  `experiments/README.md`.
- **No track closes without evidence.** A kill needs a number and a p-value
  (or a clean mechanism failure), recorded in `experiments/README.md` and
  `docs/REDESIGN.md`.
- **Kill your darlings in writing.** `docs/REDESIGN.md` §"what we kill" is a
  badge of honor, not a graveyard.

## Code standards

- Strict typing (`mypy src` clean), docstrings on public functions, no
  `Any`/dict shortcuts in new code.
- LLM retry: exponential backoff + jitter, honor `Retry-After`, patient on
  429s (Groq free tier). See `agent/retry.py`.
- Secrets via the vault surrogate — never in code, logs, or chat.

## Review

External audits (e.g. Argon) are treated as code review: verify each finding
against the repo, incorporate or document disagreement with file/line or test
evidence. Never restate an auditor's claims as your own findings, and never
report fixes complete before verification allows.
