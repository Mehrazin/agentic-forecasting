# `data_gen/` — simulator and synthetic data

| Module | What it is |
|---|---|
| `simulator.py` | Mortgage simulator v1: 10k loans × 66 months, planted shocks |
| `simulator_v2.py` | v2: renewal offers, lender competition, seasoning (subclasses v1) |
| `amortization.py` | Canadian amortization schedule (verified to the cent vs C001) |
| `events.py` | Event timeline: 5 exogenous event types, PIT-safe `events_visible_asof()` |
| `news_synthesis.py` | Template-based synthetic news (PIT-safe, 84% news-leading after Phase 3) |
| `scenarios.py` | Scenario definitions for experiments |

Calibration: [`docs/SYNTHETIC_CALIBRATION.md`](../../docs/SYNTHETIC_CALIBRATION.md).
Up: [`td_one/`](../README.md).
