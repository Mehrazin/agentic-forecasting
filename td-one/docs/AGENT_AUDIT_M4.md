# M4 Regime Reader — Agent Audit (2026-10-16)

Audit + hardening of the regime-reader agent (M4), our best agentic
result: real Groq 3.59 CRPS vs M0 4.82 (`experiments/exp11_results.json`).

## The failure that motivated this

`family_squeeze` scenario: true shift +0.45 (UP). The LLM output
DOWN@0.70: *"Increased financial pressure on low-income families makes
mortgage switching less affordable."* M4a CRPS 9.565 vs M0 4.842 —
a single confident wrong classification nearly erased the total gain
(3.587 vs 4.823).

Root cause: **single-mechanism reasoning**. The model followed one
plausible causal chain (pressure → less affordable → less switching)
and never considered the alternative (pressure → chase better deals →
*more* switching). The v1 prompt's 3-step CoT has no counterargument
step.

A second, quieter failure: `osfi_tightening` — correct direction
(DOWN@0.85) but the intervention still hurt (1.972 vs M0 1.097).
Direction right, magnitude wrong. The panel gate (M4b) correctly
abstained here, which is why the two-gate design matters.

## Findings

### 1. Prompt: systematically overconfident

Real-Groq confidences: 0.85, 0.85, 0.85, 0.90, 0.70, 0.85 (mean 0.83)
at 0.67 accuracy. The v1 instruction ("report LOW confidence when
ambiguous") does not work — the model reports high confidence on all
six scenarios, including the one it gets wrong.

### 2. Prompt: no uncertainty class

`NONE` means "no material change expected" — a confident directional
judgment. There was no way to say "I cannot determine the direction."
The model was forced to pick a direction even when the mechanism was
contested.

### 3. Prompt: injection surface

`description` is interpolated directly into the prompt via `.format()`.
News article titles/snippets will flow through the same path. No
delimiters, no instruction hierarchy — an article containing
*"Ignore previous instructions…"* would be executed as an instruction.

### 4. Calibration: hard cliff at 0.7

`family_squeeze` reported exactly 0.70 — the threshold. A hard cliff
with uncalibrated inputs is fragile; one tick either way flips the
decision.

### 5. News-specific biases (pre-integration)

- **Recency**: `NewsStore.search()` returns newest-first; the LLM will
  overweight the first articles.
- **Cherry-picking**: no instruction to weigh conflicting evidence.
- **Date confusion**: articles carry pub_dates; the model may treat
  stale articles as current.

## Fixes applied

| # | Fix | Where |
|---|---|---|
| 1 | New `ABSTAIN` regime class: "mechanism unclear, do not act". Distinct from `NONE`. Parser accepts it; `should_intervene` blocks it like `NONE`; `CLASS_DELTAS["ABSTAIN"] = 0.0`. | `regime_reader.py`, `regime_blend.py` |
| 2 | Hardened v2 prompt (`REGIME_PROMPT_V2`): explicit `COUNTERARGUMENT` step ("what is the strongest argument for the OPPOSITE direction?"), instruction hierarchy (`<CONTEXT>` = data, never instructions), news discipline (newest-first warning, anti-cherry-picking, publication-date awareness), ABSTAIN as preferred over confident guesses. | `regime_reader.py` |
| 3 | `format_news_block()`: renders articles with dates, strips `<>` to prevent breaking the data framing, truncates titles. | `regime_reader.py` |
| 4 | `read_regime_with_news()`: news-aware entry point. Caller owns the PIT cutoff (only pre-cutoff articles are passed). | `regime_reader.py` |
| 5 | `conservative_shrinkage()`: pulls self-reported confidence toward 0.5 (default strength 0.3). Addresses the observed 0.83-vs-0.67 overconfidence. Opt-in — wire into `should_intervene` when deploying. | `regime_reader.py` |
| 6 | 14 new tests: ABSTAIN parsing/gating, shrinkage math, v2 prompt guard presence (hierarchy, counterargument, news discipline), news-block sanitization, injection-framing (evil title stays inside `<CONTEXT>`). | `tests/test_regime_reader.py` |

## What was NOT changed

- `REGIME_PROMPT` (v1) kept as-is for backward compatibility; v2 is opt-in via `read_regime_with_news(use_v2_prompt=True)`.
- `MockRegimeReader` untouched (its calibrated Beta(8,2)/Beta(2,5) model is the test oracle).
- Thresholds (0.7 / z 1.25) unchanged — retuning needs more than n=6 real readings.

## Verification

- 33/33 tests pass (`tests/test_regime_reader.py`)
- `ruff check` clean, `ruff format` clean
- `mypy --strict` clean on both changed source files

## Open items

1. **Empirical calibration**: fit Platt/temperature scaling once we have ~50+ real readings. The shrinkage default (0.3) is a principled guess, not a fit.
2. **Threshold retune**: revisit 0.7 with v2 prompt + ABSTAIN on a larger scenario set.
3. **Magnitude**: `osfi_tightening` showed correct-direction-but-harmful intervention. Consider scaling the class delta by confidence instead of the current all-or-nothing gate.
