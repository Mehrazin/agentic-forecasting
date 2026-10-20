"""SHAP-based attrition analysis: what drives switching heterogeneity.

Fits a gradient-boosting attrition model on **training-period**
borrower metadata, then computes exact TreeSHAP attributions
(``shap`` package) and SHAP interaction values. The motivation
(exp66e): the deterministic grouper clustered on *raw features* and
failed to recover the simulator's planted cohorts (ARI 0.010, chance
level — exp66d) because it clusters on feature similarity, not on
attrition relevance. The planted cohorts are sharp *interactions*
(rate_type x origination year x rate level; insured x credit band;
income x children) — interaction values should expose them, and
clustering on SHAP vectors should group borrowers by *why* they
switch rather than by *what they look like*.

PIT: the model is fit on training-period rows and the training-window
switching label only. Nothing post-origin enters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from td_one.agent.grouping import OUTCOME_COLUMNS

_ID_LIKE = frozenset({"mortgage_id", "customer_id", "snapshot_date"})
_DATE_SUFFIXES = ("_asof", "_date", "_at")


def _is_id_like(col: str) -> bool:
    return col in _ID_LIKE or col.endswith("_id")


# One-hot columns and engineered names -> base attribute, for matching
# discovered interactions against planted-cohort definitions.

# Planted cohorts as base-attribute sets (from simulator_v2's DGP).
PLANTED_COHORT_ATTRIBUTES: dict[str, frozenset[str]] = {
    "low_rate": frozenset({"rate_type", "origination_year", "mortgage_rate_pct"}),
    "osfi_exempt": frozenset({"insured_flag", "credit_score_band"}),
    "family_squeeze": frozenset({"household_income_cad", "children"}),
}


def planted_interaction_pairs() -> list[frozenset[str]]:
    """All 2-subsets of the planted cohorts' attribute sets."""
    from itertools import combinations

    pairs: list[frozenset[str]] = []
    for attrs in PLANTED_COHORT_ATTRIBUTES.values():
        for a, b in combinations(sorted(attrs), 2):
            pairs.append(frozenset({a, b}))
    return pairs


def base_attribute(feature_name: str, base_map: dict[str, str] | None = None) -> str:
    """Map an encoded feature column back to its base attribute.

    One-hot columns are ``<attr>_<value>``; the mapping is tracked at
    encoding time in ``base_map`` (see :func:`prepare_features`) so
    values containing underscores (e.g. ``credit_score_band_720_plus``)
    resolve correctly. Children-related attributes collapse to
    ``children``.
    """
    attr = base_map.get(feature_name, feature_name) if base_map else feature_name
    return _collapse_children(attr)


def _collapse_children(attr: str) -> str:
    """Collapse children-related attributes to ``children``."""
    if attr in ("children_under6", "children_6to17", "n_children"):
        return "children"
    return attr


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model-facing engineered columns (origination_year, n_children)."""
    out = df.copy()
    if "origination_date" in out.columns and "origination_year" not in out.columns:
        out["origination_year"] = pd.to_datetime(out["origination_date"]).dt.year.astype(float)
    if {"children_under6", "children_6to17"} <= set(
        out.columns
    ) and "n_children" not in out.columns:
        out["n_children"] = (
            pd.to_numeric(out["children_under6"], errors="coerce").fillna(0)
            + pd.to_numeric(out["children_6to17"], errors="coerce").fillna(0)
        ).astype(float)
    return out


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for c in df.columns:
        if _is_id_like(c):
            continue
        if any(c.endswith(s) for s in _DATE_SUFFIXES):
            continue
        if c in OUTCOME_COLUMNS:
            continue
        if df[c].nunique(dropna=False) < 2:
            continue  # zero-variance: no signal
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            categorical.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
    return numeric, categorical


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Engineer + encode borrower metadata into a numeric matrix.

    Returns ``(X, feature_names, base_map)`` where categorical
    attributes are one-hot encoded (column ``<attr>_<value>``),
    numerics are median-imputed, and ``base_map`` maps every encoded
    column back to its source attribute. Column order is
    deterministic.
    """
    df = engineer_features(df)
    numeric, categorical = _feature_columns(df)
    x_num = df[numeric].copy()
    for c in numeric:
        x_num[c] = pd.to_numeric(x_num[c], errors="coerce")
    x_num = x_num.fillna(x_num.median(numeric_only=True))
    pieces = [x_num.reset_index(drop=True)]
    base_map: dict[str, str] = {c: c for c in numeric}
    if categorical:
        x_cat = pd.get_dummies(df[categorical].astype(str), columns=categorical)
        for c in categorical:
            for out_col in x_cat.columns:
                if out_col.startswith(f"{c}_"):
                    base_map[out_col] = c
        pieces.append(x_cat.astype(float).reset_index(drop=True))
    x = pd.concat(pieces, axis=1)
    return x, list(x.columns), base_map


def fit_attrition_model(
    x: pd.DataFrame, y: np.ndarray, seed: int = 42
) -> GradientBoostingClassifier:
    """Fit a gradient-boosting attrition model (deterministic)."""
    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=seed,
    )
    model.fit(x, y)
    return model


def expected_log_odds(model: GradientBoostingClassifier) -> float:
    """TreeSHAP expected value E[f(x)] (baseline log-odds)."""
    import shap

    explainer = shap.TreeExplainer(model)
    ev = np.asarray(explainer.expected_value, dtype=float).ravel()
    return float(ev[-1]) if len(ev) > 1 else float(ev[0])


def shap_values(model: GradientBoostingClassifier, x: pd.DataFrame) -> np.ndarray:
    """Exact TreeSHAP values, shape (n_samples, n_features).

    Values are in log-odds units of the positive (switched) class.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(x)
    arr = np.asarray(sv, dtype=float)
    if arr.ndim == 3:  # list-of-classes form from some shap versions
        arr = arr[..., -1] if arr.shape[-1] == 2 else arr[0]
    return arr


def shap_interaction_values(model: GradientBoostingClassifier, x: pd.DataFrame) -> np.ndarray:
    """Exact SHAP interaction values, shape (n, F, F).

    ``phi_ij`` is the pairwise interaction effect for features i, j;
    the diagonal holds main effects.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    iv = explainer.shap_interaction_values(x)
    arr = np.asarray(iv, dtype=float)
    if arr.ndim == 4:  # class dimension present
        arr = arr[..., -1] if arr.shape[-1] == 2 else arr[0]
    return arr


def top_interactions(
    interaction_values: np.ndarray,
    feature_names: list[str],
    k: int = 20,
    base_map: dict[str, str] | None = None,
) -> list[dict]:
    """Top-k pairwise interactions by mean |phi_ij| over samples.

    Collapsed to base attributes (one-hot columns merged), self-pairs
    excluded. Returns ``[{pair: (attr_a, attr_b), attrs: frozenset,
    score: float}]`` sorted by score descending.
    """
    mean_abs = np.abs(interaction_values).mean(axis=0)  # (F, F)
    pair_scores: dict[frozenset[str], float] = {}
    n = len(feature_names)
    for i in range(n):
        for j in range(i + 1, n):
            a = base_attribute(feature_names[i], base_map)
            b = base_attribute(feature_names[j], base_map)
            if a == b:
                continue
            key = frozenset({a, b})
            score = float(mean_abs[i, j] + mean_abs[j, i])
            pair_scores[key] = max(pair_scores.get(key, 0.0), score)
    ranked = sorted(pair_scores.items(), key=lambda kv: -kv[1])
    return [{"pair": tuple(sorted(key)), "attrs": key, "score": score} for key, score in ranked[:k]]


def planted_pair_recall(top: list[dict], k: int) -> float:
    """Fraction of planted interaction pairs present in the top-k list."""
    planted = planted_interaction_pairs()
    found = {entry["attrs"] for entry in top[:k]}
    hits = sum(1 for p in planted if p in found)
    return hits / len(planted)


def cluster_on_shap(values: np.ndarray, n_clusters: int = 5, seed: int = 42) -> np.ndarray:
    """KMeans on per-borrower SHAP vectors (same protocol as exp66d)."""
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(np.asarray(values, dtype=float))


def group_shap_driver_summary(
    values: np.ndarray,
    feature_names: list[str],
    group_idx: np.ndarray,
    portfolio_mean: float,
    group_mean: float,
    top_k: int = 5,
    base_map: dict[str, str] | None = None,
) -> dict:
    """Per-group SHAP risk drivers vs the portfolio.

    ``values`` are SHAP log-odds contributions for all borrowers;
    ``group_idx`` selects the group's rows. Returns the top features
    pushing the group's predicted attrition (``up`` = positive mean
    SHAP) and pulling it down, with magnitudes in log-odds units.
    """
    group_vals = np.asarray(values, dtype=float)[group_idx]
    mean_shap = group_vals.mean(axis=0)
    order = np.argsort(-mean_shap)
    up = [
        {"feature": base_attribute(feature_names[i], base_map), "mean_shap": float(mean_shap[i])}
        for i in order[:top_k]
        if mean_shap[i] > 0
    ]
    down = [
        {"feature": base_attribute(feature_names[i], base_map), "mean_shap": float(mean_shap[i])}
        for i in order[::-1][:top_k]
        if mean_shap[i] < 0
    ]
    return {
        "portfolio_mean_logodds": float(portfolio_mean),
        "group_mean_logodds": float(group_mean),
        "delta_logodds": float(group_mean - portfolio_mean),
        "top_up": up,
        "top_down": down,
    }
