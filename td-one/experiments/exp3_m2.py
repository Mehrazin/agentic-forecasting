"""Experiment 3: M2 (ReAct LLM agent) vs M0 on synthetic segment panels.

The ReActAgent (td_one.agent.react) is driven by a scripted MockLLMClient
(mock mode) and wired to the real local tool implementations from
td_one.tools (data_segment_tool, evidence_tool, forecast). If td_one.tools
is not importable, the script prints a skip message and exits 0.

Pilot segment panels are built from the simulator DGP and written as parquet
to td-one/data/segments/ (the scale-up pipeline's location; it will overwrite
them at full scale). Simulator maturities cluster into few calendar months,
so panels are 12 *ordered renewal batches* labeled with sequential month-ends
— documented here, not calendar truth.

Evaluation is leakage-clean: the agent's run_date is the end of batch 11, so
data_segment_tool masks the holdout batch 12; M0 fits trailing-6 on batches
1..11 and is tested on batch 12, same as M2's one-step-ahead forecast.

The scripted agent performs the 3-tool trace
(data_segment → evidence → forecasting); the evaluated M2 forecast is the
forecasting tool's rate-quantile output. M2 CRPS uses a normal approximation
from the tool's 80% interval (documented in the report).
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from td_one.agent.react import MockLLMClient, ReActAgent  # noqa: E402
from td_one.evaluation.metrics import (  # noqa: E402
    crps_counts,
    interval_coverage,
    mae,
)
from td_one.models.baseline import HistoricalBaseline  # noqa: E402
from td_one.tools.data_segment import SEGMENTS_DIR  # noqa: E402

logger = logging.getLogger(__name__)

#: Latest panel month = experiment PIT cutoff.
RUN_DATE = date(2025, 6, 30)
#: Pilot segments.
N_SEGMENTS = 3
#: Mock-mode forecasting backend (interface-only mock on the real grid).
MOCK_MODEL = "timesfm_mock"

#: Injected tool callable: args dict -> observation string.
ToolFn = Callable[[dict[str, object]], str]


def _batch_months(n: int = 12) -> list[date]:
    """Sequential month-ends ending at RUN_DATE (pilot batch labels)."""
    months: list[date] = []
    y, m = RUN_DATE.year, RUN_DATE.month
    for _ in range(n):
        months.append(date(y, m, 1))
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    months.reverse()
    # Convert month-starts to month-ends.
    ends: list[date] = []
    for d in months:
        nm = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        ends.append(date.fromordinal(nm.toordinal() - 1))
    return ends


def build_panels(n_customers: int = 300, n_months: int = 66) -> list[dict[str, object]]:
    """Build 3 pilot panels: 12 ordered renewal batches from the simulator DGP."""
    from td_one.data_gen.simulator import MortgageSimulator

    months = _batch_months(12)
    panels: list[dict[str, object]] = []
    for i in range(N_SEGMENTS):
        sim = MortgageSimulator(n_customers=n_customers, n_months=n_months, seed=2000 + i)
        outcomes = [o for o in sim.generate_outcomes() if o.switched_lender_flag is not None]
        outcomes.sort(key=lambda o: o.scheduled_maturity_date)
        batches = np.array_split(np.arange(len(outcomes)), 12)
        n = np.zeros(12)
        a = np.zeros(12)
        for b, idx in enumerate(batches):
            n[b] = len(idx)
            a[b] = sum(int(outcomes[j].switched_lender_flag) for j in idx)
        panels.append(
            {
                "segment_id": f"seg-{i}",
                "months": months,
                "N": n,
                "A": a,
            }
        )
        logger.info("panel seg-%d: %d renewals in 12 batches", i, int(n.sum()))
    return panels


def write_pilot_segments(
    panels: list[dict[str, object]], segments_dir: Path = SEGMENTS_DIR
) -> None:
    """Write pilot panels as parquet in the scale-up pipeline's location."""
    segments_dir.mkdir(parents=True, exist_ok=True)
    for panel in panels:
        sid = str(panel["segment_id"])
        months = panel["months"]
        assert isinstance(months, list)
        df = pd.DataFrame(
            {
                "month": months,
                "n_renewals": np.asarray(panel["N"], dtype=int),
                "n_switches": np.asarray(panel["A"], dtype=int),
            }
        )
        df.to_parquet(segments_dir / f"{sid}.parquet", index=False)
        logger.info("wrote %s", segments_dir / f"{sid}.parquet")


def evaluate_m0(panels: list[dict[str, object]], seed: int = 42) -> list[dict[str, object]]:
    """M0: fit trailing-6 on batches 0..10, test on holdout batch 11."""
    rows: list[dict[str, object]] = []
    for panel in panels:
        n = np.asarray(panel["N"], dtype=float)
        a = np.asarray(panel["A"], dtype=float)
        m0 = HistoricalBaseline()
        m0.fit(n[:11], a[:11])
        n_test = int(n[11])
        actual = float(a[11])
        samples = m0.predict_counts(max(n_test, 1), seed=seed)
        q10, q50, q90 = m0.predict_quantiles(max(n_test, 1))
        rows.append(
            {
                "segment_id": str(panel["segment_id"]),
                "n_test": n_test,
                "actual": actual,
                "crps": float(crps_counts(samples[None, :], np.array([actual]))[0]),
                "mae": float(mae(np.array([q50]), np.array([actual]))),
                "coverage": float(
                    interval_coverage(np.array([q10]), np.array([q90]), np.array([actual]))
                ),
            }
        )
    return rows


def wire_tools() -> dict[str, ToolFn]:
    """Adapt the real td_one.tools implementations to the agent's callable contract.

    Raises:
        SystemExit(0): if td_one.tools is not importable.

    """
    try:
        from td_one.tools import data_segment_tool, evidence_tool, forecast
    except ImportError:
        print("SKIP: td_one.tools not importable; M2 agent run deferred (exit 0).")
        raise SystemExit(0) from None

    def _run_date(args: dict[str, object]) -> date:
        return date.fromisoformat(str(args["run_date"]))

    def data_segment(args: dict[str, object]) -> str:
        panel = data_segment_tool(str(args["segment_id"]), _run_date(args))
        return panel.to_json(orient="records", date_format="iso")

    def evidence(args: dict[str, object]) -> str:
        cards = evidence_tool(str(args.get("query", "")), _run_date(args))
        return json.dumps(cards)

    def forecasting(args: dict[str, object]) -> str:
        panel = data_segment_tool(str(args["segment_id"]), _run_date(args))
        quantiles = forecast(MOCK_MODEL, panel, _run_date(args))
        return json.dumps({str(k): v for k, v in quantiles.items()})

    return {
        "data_segment": data_segment,
        "evidence": evidence,
        "forecasting": forecasting,
    }


class _ScriptedFinalizer(MockLLMClient):
    """Scripted LLM: 3 tool calls, then a finalize built from the tool output.

    The forecasting tool's recorded observation ({level: rate}) becomes the
    finalize forecast after validation.
    """

    def __init__(self, holder: dict[str, str], segment_id: str, run_date_iso: str) -> None:
        super().__init__(
            [
                json.dumps(
                    {
                        "action": "data_segment",
                        "args": {"segment_id": segment_id, "run_date": run_date_iso},
                    }
                ),
                json.dumps(
                    {
                        "action": "evidence",
                        "args": {
                            "query": "Bank of Canada rate decision",
                            "run_date": run_date_iso,
                        },
                    }
                ),
                json.dumps(
                    {
                        "action": "forecasting",
                        "args": {"segment_id": segment_id, "run_date": run_date_iso},
                    }
                ),
                '{"action": "finalize", "forecast": {}}',
            ]
        )
        self._holder = holder

    def complete(self, prompt: str) -> str:
        if self.calls >= len(self._responses) - 1:
            # Finalize step: validate the recorded tool output into a forecast.
            self.calls += 1
            forecast: dict[str, float] = {}
            try:
                raw = json.loads(self._holder["last"])
                for key, value in raw.items():
                    q, rate = float(key), float(value)
                    if 0.0 <= q <= 1.0 and 0.0 <= rate <= 1.0:
                        forecast[str(key)] = rate
            except (KeyError, ValueError, TypeError, AttributeError):
                forecast = {}
            return json.dumps({"action": "finalize", "forecast": forecast})
        return super().complete(prompt)


def run_m2_agent(
    panels: list[dict[str, object]],
    tools: dict[str, ToolFn],
    seed: int = 42,
) -> list[dict[str, object]]:
    """Drive the scripted ReAct agent per panel; evaluate vs the holdout batch."""
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for panel in panels:
        sid = str(panel["segment_id"])
        months = panel["months"]
        assert isinstance(months, list)
        agent_run_date = months[10]  # masks holdout batch 11
        n = np.asarray(panel["N"], dtype=float)
        a = np.asarray(panel["A"], dtype=float)
        n_test = max(int(n[11]), 1)
        actual = float(a[11])
        holder: dict[str, str] = {}

        def recording_forecast(
            args: dict[str, object],
            _t: ToolFn = tools["forecasting"],
            _h: dict[str, str] = holder,
        ) -> str:
            out = _t(args)
            _h["last"] = out
            return out

        agent_tools = {**tools, "forecasting": recording_forecast}
        llm = _ScriptedFinalizer(holder, sid, agent_run_date.isoformat())
        agent = ReActAgent(llm, agent_tools, max_steps=6)
        result = agent.run(f"Forecast switching for segment {sid}", agent_run_date)

        forecast = result["forecast"] or {}
        q50 = forecast.get(0.5)
        if q50 is None:
            m2_crps, m2_mae, m2_cov = float("nan"), float("nan"), float("nan")
        else:
            q10 = float(forecast.get(0.1, q50))
            q90 = float(forecast.get(0.9, q50))
            # Normal approximation from the 80% interval, for CRPS.
            sd = max((q90 - q10) / 2.5631, 1e-6)
            samples = rng.normal(q50 * n_test, sd * n_test, size=1000)
            m2_crps = float(crps_counts(samples[None, :], np.array([actual]))[0])
            m2_mae = float(mae(np.array([q50 * n_test]), np.array([actual])))
            m2_cov = float(
                interval_coverage(
                    np.array([q10 * n_test]),
                    np.array([q90 * n_test]),
                    np.array([actual]),
                )
            )
        rows.append(
            {
                "segment_id": sid,
                "n_test": n_test,
                "actual": actual,
                "m2_crps": m2_crps,
                "m2_mae": m2_mae,
                "m2_coverage": m2_cov,
                "m2_steps": result["steps"],
                "evidence_cards": _evidence_card_count(result),
            }
        )
    return rows


def _evidence_card_count(result: dict[str, object]) -> int:
    """Count evidence cards retrieved in the trace (diagnostic)."""
    trace = result.get("trace")
    if not isinstance(trace, list):
        return -1
    for step in trace:
        if isinstance(step, dict) and step.get("action") == "evidence":
            try:
                return len(json.loads(str(step.get("observation", "[]"))))
            except (ValueError, TypeError):
                return -1
    return -1


def run_exp3(
    out_dir: str | Path,
    panels: list[dict[str, object]] | None = None,
    tools: dict[str, ToolFn] | None = None,
    seed: int = 42,
) -> Path:
    """Run the M2 experiment and write results JSON + report."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    if panels is None:
        panels = build_panels()
        write_pilot_segments(panels)
    if tools is None:
        tools = wire_tools()

    m0_rows = evaluate_m0(panels, seed=seed)
    m2_rows = run_m2_agent(panels, tools, seed=seed)

    combined: list[dict[str, object]] = []
    for m0r, m2r in zip(m0_rows, m2_rows, strict=True):
        combined.append({**m0r, **m2r})

    results = {
        "experiment": "exp3_m2",
        "date": RUN_DATE.isoformat(),
        "mode": "mock",
        "forecast_model": MOCK_MODEL,
        "panels": combined,
    }
    results_path = out_path / "exp3_results.json"
    results_path.write_text(json.dumps(results, indent=2, default=float))

    valid = [r for r in combined if r["m2_crps"] == r["m2_crps"]]
    n_better = sum(1 for r in valid if r["m2_crps"] < r["crps"])
    total_cards = sum(int(r["evidence_cards"]) for r in combined)
    lines = [
        "# Experiment 3 Report: M2 (ReAct LLM Agent) vs M0",
        "",
        f"**Date:** {datetime.now(UTC).date().isoformat()} (run_date: {RUN_DATE.isoformat()}).",
        " **Mode:** mock (scripted LLM).",
        f"Forecast backend: {MOCK_MODEL} (interface-only mock on the real grid).",
        "",
        "The scripted ReAct agent runs the 3-tool trace",
        "(data_segment → evidence → forecasting) per pilot segment; the",
        "evaluated M2 forecast is the forecasting tool's rate-quantile output.",
        "M0 is the trailing-6 Jeffreys Beta-Binomial on the same panels,",
        "tested on the holdout batch the agent never saw (PIT-clean).",
        "",
        "| Segment | M0 CRPS | M2 CRPS | M0 MAE | M2 MAE | M2 steps | evid |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in combined:
        lines.append(
            f"| {r['segment_id']} | {r['crps']:.2f} | {r['m2_crps']:.2f} |"
            f" {r['mae']:.2f} | {r['m2_mae']:.2f} | {r['m2_steps']} |"
            f" {r['evidence_cards']} |"
        )
    lines += [
        "",
        f"M2 beats M0 on CRPS in {n_better}/{len(valid)} segments"
        + (" (mock LLM; plumbing validation only)." if valid else "."),
        "",
        "## Caveats",
        "",
        "- Mock mode: the LLM is scripted, so this validates the ReAct",
        "  orchestration plumbing (tool dispatch, trace, finalize parsing),",
        "  not LLM forecasting skill. A real-LLM run is the follow-up.",
        "- M2 CRPS uses a normal approximation from the tool's 80% interval;",
        "  rough calibration check only.",
        "- Pilot panels are small synthetic batches; the scale-up pipeline",
        "  (`scripts/scale_up.py`) evaluates on real monthly segment panels",
        "  from the full simulator.",
        f"- Evidence retrieval returned {total_cards} cards across segments"
        " (frozen illustrative corpus).",
    ]
    (out_path / "exp3_report.md").write_text("\n".join(lines) + "\n")
    return results_path


def main() -> int:
    """CLI entrypoint."""
    print("exp3: mock mode (scripted LLM). Wiring td_one.tools ...")
    out = run_exp3(Path(__file__).resolve().parent, seed=42)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
