# `tools/` — the 3 v1 action tools

The coarse action tools the ReAct loop dispatches by name (`data_segment`,
`evidence`, `forecasting`). These wrap the data service, the news RAG, and the
model ladder respectively — one call each per ReAct step.

For the granular 13-tool namespace the agents actually reason with (residual
analysis, changepoints, regime probabilities, …), see
[`agent/tools/`](../agent/tools/README.md) — that's where the real work
happens. This layer is the older v1 interface, kept for the ReAct v1 loop.

| Module | Tools |
|---|---|
| `data_segment.py` | `data_segment_tool` — PIT-safe cohort panel for a segment × run date |
| `evidence.py` | `evidence_tool` — news/RAG evidence lookup |
| `forecasting.py` | `forecast` — run a named model over a panel |

Audits: [`docs/TOOL_AUDIT.md`](../../docs/TOOL_AUDIT.md).
Up: [`td_one/`](../README.md).
