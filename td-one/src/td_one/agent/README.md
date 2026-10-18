# `agent/` — LLM agents

| Module | What it is |
|---|---|
| `proposed_method.py` | **The method** (exp47): regime-probability router + gates + conformal |
| `regime_reader.py` | M4: LLM reads news → regime label + confidence (never numbers) |
| `ensemble_router.py` | M5: REATS-style ensemble router (killed by exp12 — loses to inv-CRPS) |
| `react.py` / `react_v2.py` | M2: ReAct tool loop (v2 = hardened) |
| `controller.py` | Groq agent controller |
| `decision_policy.py` | Shrinkage + corroboration gate (exp6/exp22) |
| `news_tool.py` | News evidence tool |
| `sandbox.py` | Sandboxed code execution (ported from AI2) |
| `groq_client.py` | Groq via vault-surrogate (key never in process; 2.5s throttle; patient 429 backoff) |
| `retry.py` | AI2-grade retry: exponential backoff + jitter, Retry-After honored |
| `tool_tracker.py` | `@tracked()` decorator — logs timing, hashed inputs, outputs, success/fail |

Design docs: [`docs/AGENTIC_ROADMAP.md`](../../docs/AGENTIC_ROADMAP.md),
[`docs/PROPOSED_METHOD.md`](../../docs/PROPOSED_METHOD.md).
Up: [`td_one/`](../README.md).
