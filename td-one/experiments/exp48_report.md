# exp48: Reasoning-Effort Sweep — Final Report

## Setup
- 27 LLM calls: 3 tasks x 3 levels (low/medium/high) x 3 seeds
- Model: openai/gpt-oss-120b, max_tokens=1500
- Tasks: regime_diagnosis (accuracy), boc_discovery (binary hit), changepoint_diagnosis (rubric)

## Results

| Task | low | medium | high |
|---|---|---|---|
| regime_diagnosis (acc) | 1.000 (2.2s) | 1.000 (2.3s) | 1.000 (7.9s) |
| boc_discovery (hit) | 1.000 (2.9s) | 1.000 (4.1s) | **0.333 (17.6s, 1 error)** |
| changepoint (rubric) | 1.000 (2.5s) | 1.000 (1.7s) | 1.000 (4.7s) |

![Reasoning effort: accuracy and latency](../docs/plots/reasoning_sweep.png)

## Key findings

1. **"high" never helps, sometimes hurts.** On boc_discovery it dropped from
   1.000 to 0.333 — overthinking produced a plausible-but-wrong feature
   (rate_momentum x yield_spread instead of the inflation interaction).

2. **"high" causes empty-content failures.** Seed 0 on boc_discovery hit
   GroqEmptyContentError: the reasoning tax consumed the token budget,
   leaving nothing for the actual answer. This is the failure mode the
   client docstring already warned about.

3. **Latency penalty is severe.** "high" is 3.5-6x slower than "low"
   (7.9s vs 2.2s on regime_diagnosis; 17.6s vs 2.9s on boc_discovery).

4. **"low" is sufficient for diagnostic tasks.** All three tasks saturate
   at 1.0 with low effort. These are pattern-matching/diagnosis tasks,
   not deep multi-step reasoning — extra thinking adds noise, not signal.

## Recommendation

- **Default to "low"** for all current agent tasks (diagnosis, classification,
  feature proposal). It is fastest, cheapest, and equally accurate.
- **Reserve "medium"** for tasks where low shows instability across seeds
  (not observed here — all seeds agreed at low).
- **Avoid "high"** unless a future task demonstrably needs it AND the
  max_tokens budget is raised to accommodate the reasoning tax.
- The `reasoning_effort` parameter is wired into `GroqChatClient`
  (default "medium" preserved for backward compatibility); callers should
  explicitly pass `reasoning_effort="low"` for diagnostic workloads.

## Caveats

- n=3 seeds per cell is small; effect sizes are large but CIs are wide.
- Tasks were single-call prompts, not multi-step ReAct loops. "high" might
  help on genuinely hard multi-hop reasoning (untested).
- TPM impact not directly measured (client doesn't capture usage); latency
  is used as the cost proxy.
