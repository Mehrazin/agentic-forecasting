"""Metrics for measuring recovery of planted DGP cohorts.

Answers: does a discovered grouping find the simulator's true
structure, or arbitrary clusters? Low scores here mean the
"explainability" is explaining noise — say so honestly.
"""

from __future__ import annotations

from sklearn.metrics import adjusted_rand_score


def adjusted_rand_index(true_labels: list[str], pred_labels: list[str]) -> float:
    """Compute the Adjusted Rand Index between cohorts and groups.

    1.0 = identical partitions (up to relabeling); ~0.0 = chance-level
    agreement; negative = worse than chance.
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError(f"label length mismatch: {len(true_labels)} vs {len(pred_labels)}")
    if not true_labels:
        raise ValueError("labels must be non-empty")
    return float(adjusted_rand_score(true_labels, pred_labels))


def best_match_per_cohort(true_labels: list[str], pred_labels: list[str]) -> dict[str, dict]:
    """For each true cohort, the predicted group with max F1.

    Returns ``{cohort: {group_id, precision, recall, f1, n_true,
    n_pred}}``. Precision = share of the matched group that is the
    cohort; recall = share of the cohort captured by the group.
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError(f"label length mismatch: {len(true_labels)} vs {len(pred_labels)}")
    cohorts = sorted(set(true_labels))
    groups = sorted(set(pred_labels))
    out: dict[str, dict] = {}
    for c in cohorts:
        true_idx = {i for i, t in enumerate(true_labels) if t == c}
        best: dict | None = None
        for g in groups:
            pred_idx = {i for i, p in enumerate(pred_labels) if p == g}
            tp = len(true_idx & pred_idx)
            precision = tp / len(pred_idx) if pred_idx else 0.0
            recall = tp / len(true_idx) if true_idx else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if best is None or f1 > best["f1"]:
                best = {
                    "group_id": g,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "n_true": len(true_idx),
                    "n_pred": len(pred_idx),
                }
        assert best is not None
        out[c] = best
    return out
