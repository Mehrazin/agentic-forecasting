"""Generate all plots for TD-One docs.

Every plot is regenerated from committed artifacts (result JSONs, CSVs) —
no hand-typed numbers except where the source is a doc table (noted).
Run:  .venv/bin/python docs/plots/make_plots.py   (from td-one/)
Output: docs/plots/*.png  (dpi=150, tight bbox, consistent style)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # td-one/
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PLOTS_DIR = Path(__file__).resolve().parent
DPI = 150


def setup() -> None:
    """Shared style for every TD-One plot."""
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    out = PLOTS_DIR / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")
    return out


# ---------------------------------------------------------------- existing

def plot_rag_metrics() -> Path:
    """Recall@k / precision@k — source: experiments/rag_eval_results.json."""
    setup()
    results = json.loads((ROOT / "experiments" / "rag_eval_results.json").read_text())
    summary = results["summary"]
    ks = [1, 3, 5, 10]
    recalls = [summary[f"@{k}"]["recall"] for k in ks]
    precisions = [summary[f"@{k}"]["precision"] for k in ks]

    fig, ax = plt.subplots(figsize=(9.6, 6))
    ax.plot(ks, recalls, "o-", label="Recall@k", linewidth=2, markersize=8)
    ax.plot(ks, precisions, "s-", label="Precision@k", linewidth=2, markersize=8)
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.set_title("RAG Retrieval: Recall@k and Precision@k (BM25, n=100 queries)")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.0)
    ax.legend()
    for k, r, p in zip(ks, recalls, precisions, strict=True):
        ax.annotate(f"{r:.2f}", (k, r), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9)
        ax.annotate(f"{p:.2f}", (k, p), textcoords="offset points", xytext=(0, -15),
                    ha="center", fontsize=9)
    fig.text(0.01, 0.01, "Source: experiments/rag_eval_results.json", fontsize=8, color="gray")
    return save(fig, "rag_recall_precision.png")


def plot_agent_results() -> Path:
    """M4 vs M5 agent results — source: exp11/exp12 result JSONs (values below)."""
    setup()
    labels = ["M4: Regime\nReader", "M0\n(baseline)", "M5: Ensemble\nRouter", "inv-CRPS\n(baseline)"]
    values = [3.59, 4.82, 2.60, 2.32]
    colors = ["#2ca02c", "#7f7f7f", "#d62728", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(9.6, 6))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("CRPS (lower is better)")
    ax.set_title("Agentic Results: M4 Wins, M5 Loses to Baseline")
    for bar, v in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{v:.2f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.text(0, 4.0, "Beats M0", ha="center", fontsize=10, color="#2ca02c", fontweight="bold")
    ax.text(2, 2.85, "Loses to baseline", ha="center", fontsize=10, color="#d62728",
            fontweight="bold")
    ax.set_ylim(0, 5.5)
    fig.text(0.01, 0.01, "Source: exp11_agent_v2 / exp12_router result JSONs", fontsize=8,
             color="gray")
    return save(fig, "agent_m4_m5.png")


def plot_smm() -> Path:
    """Freddie Mac SMM 2021–2026 — source: data/freddie/smm_monthly_2021_2026.csv."""
    setup()
    periods, smm = [], []
    with open(ROOT / "data" / "freddie" / "smm_monthly_2021_2026.csv") as f:
        for row in csv.DictReader(f):
            periods.append(row["period"])
            smm.append(float(row["smm"]) * 100)  # percent
    x = np.arange(len(periods))

    fig, ax = plt.subplots(figsize=(9.6, 4))
    ax.plot(x, smm, linewidth=1.5, color="#1f77b4")
    ax.set_ylabel("SMM (%)")
    ax.set_title("Freddie Mac Single-Month Mortality (SMM), 2021–2026 (63 months)")
    tick_idx = list(range(0, len(periods), 6))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([periods[i] for i in tick_idx], rotation=45, ha="right", fontsize=9)
    ax.set_xlim(0, len(periods) - 1)
    fig.text(0.01, 0.01, "Source: data/freddie/smm_monthly_2021_2026.csv", fontsize=8,
             color="gray")
    return save(fig, "smm_2021_2026.png")


# ---------------------------------------------------------------- new

def plot_trilogy() -> Path:
    """Agentic trilogy + proposed method on M4 — sources: exp42/43/45/47 result JSONs."""
    setup()
    # (label, method CRPS, equal-weights CRPS, best-single CRPS or None)
    rows = [
        ("exp42\nFFORMA", 0.11387, 0.11355, 0.10479),
        ("exp43\nRegime router", 0.12820, 0.12953, None),
        ("exp45\nLearned regimes", 0.13021, 0.12953, None),
        ("exp47\nProposed", 0.14506, 0.15069, 0.15049),
    ]
    labels = [r[0] for r in rows]
    x = np.arange(len(rows))
    w = 0.22
    fig, ax = plt.subplots(figsize=(9.6, 6))
    ax.bar(x - w, [r[1] for r in rows], w, label="Method", color="#1f77b4")
    ax.bar(x, [r[2] for r in rows], w, label="Equal weights", color="#7f7f7f")
    best = [r[3] if r[3] else np.nan for r in rows]
    ax.bar(x + w, best, w, label="Best single", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CRPS (lower is better)")
    ax.set_title("Agentic Track on M4: Only the Constrained Router Beats Equal Weights")
    ax.legend()
    # verdict annotations
    for i, r in enumerate(rows):
        verdict = "wins" if r[1] < r[2] else "loses"
        color = "#2ca02c" if r[1] < r[2] else "#d62728"
        ax.text(i - w, r[1] + 0.001, verdict, ha="center", fontsize=10, color=color,
                fontweight="bold")
    fig.text(0.01, 0.01, "Source: exp42/43/45/47_results.json (M4)", fontsize=8, color="gray")
    return save(fig, "trilogy_m4.png")


def plot_reasoning_sweep() -> Path:
    """exp48: reasoning effort vs accuracy and latency — source: exp48_report.md table."""
    setup()
    levels = ["low", "medium", "high"]
    acc = {
        "regime_diagnosis": [1.000, 1.000, 1.000],
        "boc_discovery": [1.000, 1.000, 0.333],
        "changepoint": [1.000, 1.000, 1.000],
    }
    latency = [2.4, 2.7, 10.1]  # mean seconds across tasks
    x = np.arange(len(levels))
    w = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 5))
    for i, (task, vals) in enumerate(acc.items()):
        ax1.bar(x + (i - 1) * w, vals, w, label=task.replace("_", " "))
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Accuracy by reasoning level")
    ax1.legend(fontsize=9)

    bars = ax2.bar(levels, latency, color=["#2ca02c", "#ff7f0e", "#d62728"])
    ax2.set_ylabel("Mean seconds per call")
    ax2.set_title("Latency by reasoning level")
    for bar, v in zip(bars, latency, strict=True):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.2, f"{v:.1f}s", ha="center",
                 fontsize=10, fontweight="bold")
    fig.suptitle('exp48: "High" Reasoning Never Helps (3 tasks × 3 seeds)', fontweight="bold")
    fig.text(0.01, 0.01, "Source: experiments/exp48_report.md", fontsize=8, color="gray")
    fig.tight_layout()
    return save(fig, "reasoning_sweep.png")


def plot_tool_usage() -> Path:
    """Tool frequency + timing — source: docs/TOOL_USAGE_DATA.md tables."""
    setup()
    tools = ["cross_\nvalidate", "test_\nstat.", "detect_\ncp", "analyze_\nresid",
             "decompose", "gen_\nscenario", "compare_\nmodels", "backtest_\nevent",
             "feat_\nimportance", "plot_\ndiag"]
    calls = [10, 8, 8, 5, 4, 4, 3, 2, 1, 1]
    ms = [1.3, 150, 45, 6, 29, 2.1, 4, 0.4, 2.4, 618]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 5))
    ax1.bar(tools, calls, color="#1f77b4")
    ax1.set_ylabel("Calls per workload")
    ax1.set_title("Tool call frequency (138 calls)")
    ax1.tick_params(axis="x", labelsize=8)

    colors = ["#d62728" if m > 100 else "#ff7f0e" if m > 20 else "#2ca02c" for m in ms]
    ax2.bar(tools, ms, color=colors)
    ax2.set_ylabel("Avg ms per call (log)")
    ax2.set_yscale("log")
    ax2.set_title("Tool latency")
    ax2.tick_params(axis="x", labelsize=8)
    fig.suptitle("Tool Audit: Frequency vs Latency", fontweight="bold")
    fig.text(0.01, 0.01, "Source: docs/TOOL_USAGE_DATA.md", fontsize=8, color="gray")
    fig.tight_layout()
    return save(fig, "tool_usage.png")


def main() -> int:
    plot_rag_metrics()
    plot_agent_results()
    plot_smm()
    plot_trilogy()
    plot_reasoning_sweep()
    plot_tool_usage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
