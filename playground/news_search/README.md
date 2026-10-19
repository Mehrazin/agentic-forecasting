# News Search Grounding Playground

Small Gemini + Google Search grounding playground for collecting date-scoped news summaries. It was the original scaffolding for the energy/oil demo's news retrieval, but it is not wired into the formal `Predictor` interface or evaluation harness.

## Relationship to the project

This is exploration code, not a reference implementation. Its ideas have since been absorbed into the maintained stack: cutoff-aware `search_web` (with the leakage verifier) now lives in the shared agent factory (`aieng.forecasting.methods.agentic`), and the [energy/oil reference implementation](../implementations/energy_oil_forecasting/) is a fully scored backtest + protected eval that uses it. Start new work from there, not from this playground. For architecture principles and extension ideas, see `planning-docs/roadmap.md`.

## Setup

Dependencies are managed at the repository root:

```bash
uv sync
```

Credentials live in `.env` at the repository root. See `.env.example`.

```dotenv
GEMINI_API_KEY=...           # required for Gemini models with google_search
LANGFUSE_PUBLIC_KEY=...      # optional
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

## Running

```bash
cd playground/news_search

uv run python run.py
uv run python run.py --max-dates 3
uv run python run.py --stride 7
uv run python run.py --stride 7 --max-dates 3
```

Configuration lives in `configs/default.yaml`. Outputs are written to `outputs/<run_name>/<date_iso>.md` by default.

Free-tier Gemini keys can hit 429s quickly. Increase `delay_between_requests_sec` in the config or start with `--max-dates 1`.

> **Prefer the maintained path.** For cutoff-aware news retrieval in a scored forecaster, use the `search_web` capability in `aieng.forecasting.methods.agentic` (see the [energy/oil reference](../implementations/energy_oil_forecasting/)) instead of extending this playground.
