# `agent/` — LLM agents and the proposed method

Everything with an LLM in the loop. The research arc runs through here: early
ReAct experiments (exp20–25) → the combination trilogy (exp42–44) → the
proposed method (exp47) → the LLM integration ladder (`llm_rungs/`).

| Module | What it is |
|---|---|
| `proposed_method.py` | **The method** (exp47/49): regime-probability router + trust flag + gates + conformal. Beats equal weights, best single, and exp43 alone on M4/Tourism/M3 |
| `regime_reader.py` | M4: LLM reads news → regime label + confidence (never numbers) |
| `ensemble_router.py` | M5: REATS-style ensemble router (killed by exp12 — loses to inv-CRPS) |
| `react.py` / `react_v2.py` | ReAct tool loops (v2 = hardened, code-executing, 13-tool namespace) |
| `controller.py` | Groq agent controller |
| `decision_policy.py` | Shrinkage + corroboration gate (exp6/exp22) |
| `news_tool.py` | News evidence tool (reads the SQLite cache, never hits a live API) |
| `sandbox.py` | Sandboxed code execution (ported from AI2) |
| `groq_client.py` | Groq via vault-surrogate (key never in process; 2.5s throttle; patient 429 backoff) |
| `retry.py` | AI2-grade retry: exponential backoff + jitter, Retry-After honored |
| `tool_tracker.py` | `@tracked()` decorator — logs timing, hashed inputs, outputs, success/fail |

Subpackage [`tools/`](tools/README.md) holds the 13 granular tool
implementations. Design docs: [`docs/AGENTIC_ROADMAP.md`](../../docs/AGENTIC_ROADMAP.md),
[`docs/PROPOSED_METHOD.md`](../../docs/PROPOSED_METHOD.md).
Up: [`td_one/`](../README.md).
