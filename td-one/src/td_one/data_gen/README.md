# `data_gen/` — simulator and synthetic data

The synthetic world the early research ran in: a mortgage book with planted
shocks, exogenous events, and template-generated news. Built so agents could be
developed and debugged fast, then validated on real data. The simulator is
calibrated (9.1pp → 2.2pp effects, realistic decay, BoC-style interactions)
so lessons transfer — but real data is always the ground truth.

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
