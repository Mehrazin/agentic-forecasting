# `audits/` — news audits

Proves the news pipeline doesn't leak the future:

| File | What it is |
|---|---|
| [`NEWS_LEAKAGE_REPORT.md`](NEWS_LEAKAGE_REPORT.md) | Date-level audit: 33/34 pass, 0 leakage |
| [`NEWS_CONTENT_REPORT.md`](NEWS_CONTENT_REPORT.md) | Content-level audit: CLEAN — forward-looking language is about inputs, never targets |
| `news_leakage_audit.py` / `news_content_audit.py` | The auditors (+ result JSONs) |

Our guard is strictly stronger than date-only: triple-layer RAG defense.
Up: [`td-one/`](../README.md).
