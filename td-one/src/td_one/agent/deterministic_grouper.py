"""Deterministic borrower grouping: clustering + tree-surrogate rules.

The exp66b pivot: LLM-proposed groupings are not reliably reproducible
run-to-run, which undermines their use for explainability (a risk
manager cannot trust groups that change between runs). This module
replaces LLM group *discovery* with a deterministic pipeline; the LLM
keeps its role in *explanation* generation via
``agent.risk_reporter.RiskReporter``.

Pipeline:
1. Standardize borrower metadata (numeric scaling + categorical
   one-hot), then KMeans (fixed seed).
2. Fit a shallow ``DecisionTreeClassifier`` to predict cluster labels
   from the same features. Each leaf becomes a group; its
   root-to-leaf path becomes an interpretable
   ``pandas.DataFrame.query()`` rule.
3. ``min_samples_leaf=min_group_n`` guarantees every group meets the
   size contract.

Deterministic: same input + same seed -> identical groups, always.
No LLM calls. Every rule passes ``grouping.validate_rule()`` (no
outcome columns, parseable query, minimum size).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from td_one.agent.grouping import OUTCOME_COLUMNS, GroupDefinition, validate_rule

DEFAULT_N_CLUSTERS = 5
DEFAULT_SEED = 42
DEFAULT_MIN_GROUP_N = 200
DEFAULT_MAX_DEPTH = 4

_ID_LIKE = frozenset({"mortgage_id", "snapshot_date"})
_DATE_LIKE_SUFFIXES = ("_asof", "_date", "_at")


def _default_numeric(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in _ID_LIKE or c in OUTCOME_COLUMNS:
            continue
        if any(c.endswith(s) for s in _DATE_LIKE_SUFFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


def _default_categorical(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if c in _ID_LIKE or c in OUTCOME_COLUMNS:
            continue
        if any(c.endswith(s) for s in _DATE_LIKE_SUFFIXES):
            continue
        if pd.api.types.is_bool_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            cols.append(c)
    return cols


def _build_features(
    df: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> tuple[np.ndarray, list[str], dict[str, tuple[float, float]]]:
    """Standardized feature matrix + feature names + un-scale map.

    One-hot names come out as ``<col>_<value>``; the rule extractor
    maps them back to ``col == 'value'`` / ``col != 'value'``.
    ``unscale`` maps each numeric feature to ``(mean, scale)`` so tree
    thresholds (in standardized units) convert back to raw units.
    """
    num_imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_num = scaler.fit_transform(num_imp.fit_transform(df[numeric]))
    cat_imp = SimpleImputer(strategy="most_frequent")
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    x_cat = ohe.fit_transform(cat_imp.fit_transform(df[categorical]))
    x_mat = np.concatenate([np.asarray(x_num, dtype=float), np.asarray(x_cat, dtype=float)], axis=1)
    unscale = {c: (float(scaler.mean_[i]), float(scaler.scale_[i])) for i, c in enumerate(numeric)}
    return x_mat, numeric + list(ohe.get_feature_names_out(categorical)), unscale


@dataclass(frozen=True)
class DeterministicGrouping:
    """Groups + row assignments from the deterministic pipeline."""

    groups: list[GroupDefinition]
    assignment: pd.Series  # index = df index, values = group_id
    n_clusters: int
    seed: int


def fit_deterministic_groups(
    df: pd.DataFrame,
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
    n_clusters: int = DEFAULT_N_CLUSTERS,
    seed: int = DEFAULT_SEED,
    min_group_n: int = DEFAULT_MIN_GROUP_N,
    max_tree_depth: int = DEFAULT_MAX_DEPTH,
) -> DeterministicGrouping:
    """Fit deterministic borrower groups.

    KMeans on standardized metadata, then a shallow decision tree
    predicts cluster labels; each leaf becomes a ``GroupDefinition``
    whose rule is the leaf's root-to-leaf path. Every rule is checked
    with ``validate_rule`` against ``df`` and ``min_group_n``.

    The tree partitions the feature space, so the leaf rules are
    mutually exclusive and exhaustive: every row matches exactly one.
    """
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
    num_cols = numeric if numeric is not None else _default_numeric(df)
    cat_cols = categorical if categorical is not None else _default_categorical(df)
    if not num_cols and not cat_cols:
        raise ValueError("no feature columns available for grouping")

    x_mat, feature_names, unscale = _build_features(df, num_cols, cat_cols)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(x_mat)
    tree = DecisionTreeClassifier(
        max_depth=max_tree_depth, min_samples_leaf=min_group_n, random_state=seed
    )
    tree.fit(x_mat, labels)

    groups, node_to_group = _extract_groups(
        tree, feature_names, cat_cols, unscale, df, min_group_n, seed
    )

    # Assign each row via the tree's own leaf lookup: the tree
    # partitions X, so every row lands in exactly one leaf.
    leaf_nodes = tree.apply(x_mat)
    group_ids = [g.group_id for g in groups]
    assignment = pd.Series(
        [group_ids[node_to_group[int(node)]] for node in leaf_nodes], index=df.index
    )
    return DeterministicGrouping(
        groups=groups, assignment=assignment, n_clusters=n_clusters, seed=seed
    )


def _condition(
    feature: str,
    threshold: float,
    goes_left: bool,
    cat_cols: list[str],
    unscale: dict[str, tuple[float, float]],
) -> str:
    """One tree split -> one pandas query condition.

    Tree thresholds are in standardized units for numeric features;
    convert back to raw units so the rule runs on the original
    dataframe.
    """
    for col in cat_cols:
        if feature.startswith(col + "_"):
            val = feature[len(col) + 1 :]
            # Bool columns one-hot into col_True / col_False: compare
            # against Python bools, not strings (a string comparison
            # on a bool column silently evaluates wrong).
            if val in ("True", "False"):
                lit = "True" if val == "True" else "False"
                return f"{col} != {lit}" if goes_left else f"{col} == {lit}"
            return f"{col} != '{val}'" if goes_left else f"{col} == '{val}'"
    mean, scale = unscale[feature]
    raw_thr = threshold * scale + mean
    op = "<=" if goes_left else ">"
    return f"{feature} {op} {raw_thr:.6g}"


def _extract_groups(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    cat_cols: list[str],
    unscale: dict[str, tuple[float, float]],
    df: pd.DataFrame,
    min_group_n: int,
    seed: int,
) -> tuple[list[GroupDefinition], dict[int, int]]:
    """Walk the tree; return (groups, {tree leaf node id -> group index}).

    Leaves are sorted by (majority cluster, rule text) for a
    deterministic group order independent of tree internals.
    """
    t = tree.tree_
    found: list[tuple[int, int, str]] = []  # (node_id, cluster, rule)

    def walk(node: int, path: list[str]) -> None:
        if t.children_left[node] == t.children_right[node]:  # leaf
            cluster = int(np.argmax(t.value[node][0]))
            rule = " and ".join(path) if path else "True"
            found.append((node, cluster, rule))
            return
        feat = feature_names[t.feature[node]]
        thr = float(t.threshold[node])
        walk(
            t.children_left[node],
            path + [_condition(feat, thr, True, cat_cols, unscale)],
        )
        walk(
            t.children_right[node],
            path + [_condition(feat, thr, False, cat_cols, unscale)],
        )

    walk(0, [])
    found.sort(key=lambda f: (f[1], f[2]))

    groups: list[GroupDefinition] = []
    node_to_group: dict[int, int] = {}
    for i, (node, cluster, rule) in enumerate(found):
        gd = validate_rule(rule, set(df.columns), min_group_n=min_group_n, df=df)
        if gd is None:
            raise RuntimeError(f"extracted rule failed validation: {rule!r}")
        groups.append(
            GroupDefinition(
                group_id=f"g{i}",
                rule=gd.rule,
                rationale=f"deterministic cluster {cluster} (kmeans seed={seed})",
            )
        )
        node_to_group[node] = i
    return groups, node_to_group
