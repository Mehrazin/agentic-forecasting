"""Generate plots for TD-One docs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOTS_DIR = ROOT / "docs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_rag_metrics() -> Path:
    """Recall@k and precision@k curves."""
    results = json.loads((ROOT / "experiments" / "rag_eval_results.json").read_text())
    summary = results["summary"]
    ks = [1, 3, 5, 10]
    recalls = [summary[f"@{k}"]["recall"] for k in ks]
    precisions = [summary[f"@{k}"]["precision"] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, recalls, "o-", label="Recall@k", linewidth=2, markersize=8)
    ax.plot(ks, precisions, "s-", label="Precision@k", linewidth=2, markersize=8)
    ax.set_xlabel("k")
    ax.set_ylabel("Score")
    ax.set_title("RAG Retrieval: Recall@k and Precision@k (BM25, n=100 queries)")
    ax.set_xticks(ks)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Annotate values
    for k, r, p in zip(ks, recalls, precisions, strict=True):
        ax.annotate(
            f"{r:.2f}", (k, r), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9
        )
        ax.annotate(
            f"{p:.2f}", (k, p), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9
        )
    out = PLOTS_DIR / "rag_recall_precision.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def plot_agent_results() -> Path:
    """M4 vs M5 agent results bar chart."""
    # From exp11 (M4) and exp12 (M5) results
    labels = [
        "M4: Regime\nReader",
        "M0\n(baseline)",
        "M5: Ensemble\nRouter",
        "inv-CRPS\n(baseline)",
    ]
    values = [3.59, 4.82, 2.60, 2.32]
    colors = ["#2ca02c", "#7f7f7f", "#d62728", "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("CRPS (lower is better)")
    ax.set_title("Agentic Results: M4 Wins, M5 Loses to Baseline")
    # Add value labels
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{v:.2f}",
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    # Add verdict annotations
    ax.text(0, 4.0, "✓ Beats M0", ha="center", fontsize=10, color="#2ca02c", fontweight="bold")
    ax.text(
        2, 2.85, "✗ Loses to baseline", ha="center", fontsize=10, color="#d62728", fontweight="bold"
    )
    ax.set_ylim(0, 5.5)
    out = PLOTS_DIR / "agent_m4_m5.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def plot_r2_comparison() -> Path | None:
    """Old vs new CRPS if r2_comparison.md exists."""
    comp_path = ROOT / "experiments" / "r2_comparison.md"
    if not comp_path.exists():
        print("Skipping R2 plot (r2_comparison.md not found)")
        return None
    # TODO: Parse the markdown when R2 agent finishes
    print("R2 comparison found but parsing not implemented yet")
    return None


def main() -> int:
    plot_rag_metrics()
    plot_agent_results()
    plot_r2_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
