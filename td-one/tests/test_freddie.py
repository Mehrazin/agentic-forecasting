"""Tests for the Freddie Mac loader."""

import pytest

from td_one.freddie import (
    DEFAULT_ZB_CODES,
    ORIG_COLUMNS,
    PERF_COLUMNS,
    PREPAY_ZB_CODES,
    load_orig,
    load_perf,
    monthly_prepay_rate,
)


def _orig_row(seq="F26Q10000001", **over):
    vals = [""] * len(ORIG_COLUMNS)
    vals[19] = seq
    vals[0] = "750"
    vals[12] = "5.5"
    for k, v in over.items():
        vals[ORIG_COLUMNS.index(k)] = v
    return "|".join(vals)


def _perf_row(seq="F26Q10000001", period="202602", zb="", **over):
    vals = [""] * len(PERF_COLUMNS)
    vals[0] = seq
    vals[1] = period
    vals[3] = "XX"
    vals[8] = zb
    vals[2] = "175000.00"
    for k, v in over.items():
        vals[PERF_COLUMNS.index(k)] = v
    return "|".join(vals)


def test_orig_schema_validation(tmp_path):
    p = tmp_path / "orig.txt"
    p.write_text(_orig_row() + "\n" + _orig_row(seq="F26Q10000002") + "\n")
    df = load_orig(p)
    assert len(df) == 2
    assert df["credit_score"].iloc[0] == 750


def test_orig_rejects_duplicate_seq(tmp_path):
    p = tmp_path / "orig.txt"
    p.write_text(_orig_row() + "\n" + _orig_row() + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_orig(p)


def test_prepay_codes():
    assert {"01"} == PREPAY_ZB_CODES
    assert {"03", "06", "09"} == DEFAULT_ZB_CODES


def test_smm_counts_only_voluntary(tmp_path):
    # 2 loans: one prepays (01), one defaults (03). SMM counts only the prepay.
    rows = [
        _perf_row(seq="A", period="202602"),
        _perf_row(seq="A", period="202603", zb="01"),
        _perf_row(seq="B", period="202602"),
        _perf_row(seq="B", period="202603", zb="03"),
    ]
    p = tmp_path / "perf.txt"
    p.write_text("\n".join(rows) + "\n")
    df = load_perf(p)
    smm = monthly_prepay_rate(df)
    mar = smm[smm["period"] == "2026-03-01"].iloc[0]
    assert mar["n_at_risk"] == 2
    assert mar["n_prepay"] == 1  # only the 01, not the 03
    assert mar["smm"] == pytest.approx(0.5)
