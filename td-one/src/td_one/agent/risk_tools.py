"""Risk-identification tools for the LLM risk classifier (exp66f).

Each tool produces a short text block appended to the group risk
classification prompt. They give the LLM *analyst-grade* context
beyond raw stats:

1. **SHAP risk drivers**: which features push *this group's* predicted
   attrition above/below the portfolio average (top positive/negative
   contributors with magnitudes).
2. **Counterfactual sensitivity**: which single-feature changes would
   flip the group's predicted risk class (e.g. "if mortgage_rate_pct
   fell 1pp, this group moves MEDIUM -> LOW").
3. **News-risk alignment**: news chunks most semantically similar to
   the group's risk profile, scored — grounded macro context per group
   (vs the title-search retrieval used in exp66d).
4. **Uplift amenability**: which groups would benefit most from a
   retention intervention, ranked by mean T-learner uplift.

All tools are computed from training-period data only (PIT). The
counterfactuals edit *feature values*, never outcomes; the class
cutoffs come from training-period predicted-probability tertiles.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def predicted_class(p: float, cutoffs: tuple[float, float]) -> str:
    """Risk class from predicted probability vs tertile cutoffs."""
    t1, t2 = cutoffs
    if p <= t1:
        return "LOW"
    if p <= t2:
        return "MEDIUM"
    return "HIGH"


def training_proba_cutoffs(proba: np.ndarray) -> tuple[float, float]:
    """LOW/MEDIUM and MEDIUM/HIGH cutoffs = training tertiles."""
    t1, t2 = np.quantile(np.asarray(proba, dtype=float), [1 / 3, 2 / 3])
    return float(t1), float(t2)


def format_shap_drivers(summary: dict) -> str:
    """Render a group SHAP driver summary as prompt text."""
    lines = [
        "SHAP risk drivers (log-odds contributions vs portfolio average, "
        f"group delta {summary['delta_logodds']:+.3f}):"
    ]
    if summary["top_up"]:
        ups = "; ".join(f"{d['feature']} (+{d['mean_shap']:.3f})" for d in summary["top_up"])
        lines.append(f"- Pushing risk UP: {ups}")
    if summary["top_down"]:
        downs = "; ".join(f"{d['feature']} ({d['mean_shap']:.3f})" for d in summary["top_down"])
        lines.append(f"- Pulling risk DOWN: {downs}")
    if not summary["top_up"] and not summary["top_down"]:
        lines.append("- No dominant drivers; group matches portfolio average.")
    return "\n".join(lines)


def _invert_base_map(base_map: dict[str, str]) -> dict[str, list[str]]:
    inv: dict[str, list[str]] = {}
    for feat, base in base_map.items():
        inv.setdefault(base, []).append(feat)
    return inv


def counterfactual_flips(
    model,
    x_group: pd.DataFrame,
    base_attrs: list[str],
    base_map: dict[str, str],
    feature_std: dict[str, float],
    cutoffs: tuple[float, float],
    top_k: int = 5,
) -> list[str]:
    """Single-feature edits that flip the group's predicted risk class.

    Numeric attributes are shifted by +/-1 training std; categorical
    attributes flip the group's dominant level to its runner-up. Only
    edits that change the predicted class are reported.
    """
    inv = _invert_base_map(base_map)
    p0 = float(model.predict_proba(x_group)[:, 1].mean())
    c0 = predicted_class(p0, cutoffs)
    findings: list[str] = []
    for attr in base_attrs[:top_k]:
        cols = inv.get(attr, [])
        if not cols:
            continue
        edited = x_group.copy()
        if len(cols) == 1 and attr in feature_std:
            # numeric attribute
            col = cols[0]
            std = feature_std[attr]
            if not np.isfinite(std) or std == 0:
                continue
            for sign, word in ((-1, "fell"), (1, "rose")):
                trial = edited.copy()
                trial[col] = trial[col] + sign * std
                p1 = float(model.predict_proba(trial)[:, 1].mean())
                c1 = predicted_class(p1, cutoffs)
                if c1 != c0:
                    findings.append(
                        f"If {attr} {word} by {std:.2g} (1 std), predicted risk "
                        f"would move {c0}->{c1} (p {p0:.3f}->{p1:.3f})."
                    )
                    break
        else:
            # categorical: flip dominant level to runner-up
            means = edited[cols].mean().sort_values(ascending=False)
            if len(means) < 2 or means.iloc[0] < 0.5:
                continue
            dom, runner = means.index[0], means.index[1]
            trial = edited.copy()
            trial[dom] = 0.0
            trial[runner] = 1.0
            p1 = float(model.predict_proba(trial)[:, 1].mean())
            c1 = predicted_class(p1, cutoffs)
            if c1 != c0:
                dom_lvl = dom[len(attr) + 1 :]
                run_lvl = runner[len(attr) + 1 :]
                findings.append(
                    f"If {attr} switched from '{dom_lvl}' to '{run_lvl}', predicted "
                    f"risk would move {c0}->{c1} (p {p0:.3f}->{p1:.3f})."
                )
    return findings


def format_counterfactuals(findings: list[str]) -> str:
    """Render counterfactual findings as prompt text."""
    if not findings:
        return (
            "Counterfactual sensitivity: no single-feature change among the "
            "top drivers flips this group's predicted risk class."
        )
    return "Counterfactual sensitivity (single-feature edits that flip the class):\n" + "\n".join(
        f"- {f}" for f in findings
    )


def news_risk_alignment(
    profile_text: str,
    search_fn,
    before,
    top_k: int = 3,
) -> str:
    """Retrieve news semantically aligned with a group risk profile.

    ``search_fn`` is ``SemanticNewsIndex.search`` (injected for
    testability); ``before`` is the PIT cutoff.
    """
    try:
        hits = search_fn(profile_text, before=before, top_k=top_k)
    except Exception:  # noqa: BLE001 — news is a bonus, not a gate
        return "News-risk alignment: retrieval failed."
    if not hits:
        return "News-risk alignment: no articles retrieved."
    lines = ["News-risk alignment (semantically closest to this group's risk profile):"]
    for i, h in enumerate(hits):
        lines.append(f"[{i}] {h.title} ({h.pub_date}, score={h.score:.3f}) [url:{i}]")
    return "\n".join(lines)


def uplift_amenability_text(group_mean_uplift: float, rank: int, n_groups: int) -> str:
    """Per-group amenability to a retention intervention."""
    if rank <= n_groups // 3:
        direction = "most"
    elif rank > 2 * n_groups // 3:
        direction = "least"
    else:
        direction = "moderately"
    return (
        f"Intervention amenability: this group's mean uplift is "
        f"{group_mean_uplift:+.4f} (rank {rank}/{n_groups}) — among the "
        f"{direction} responsive to retention offers."
    )
