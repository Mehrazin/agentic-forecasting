"""Tests for the planted-cohort label function."""

import numpy as np

from td_one.data_gen.cohort_labels import COHORTS, cohort_label


def _base(**overrides):
    kw = {
        "rate_type": "variable",
        "origination_year": 2022,
        "mortgage_rate_pct": 5.0,
        "insured_flag": True,
        "credit_score_band": "660_719",
        "household_income_cad": 100000.0,
        "n_children": 1,
    }
    kw.update(overrides)
    return cohort_label(**kw)


def test_cohorts_tuple():
    assert set(COHORTS) == {"low_rate", "osfi_exempt", "family_squeeze", "control"}


def test_low_rate():
    assert _base(rate_type="fixed", origination_year=2021, mortgage_rate_pct=2.5) == "low_rate"


def test_low_rate_boundary_inclusive():
    assert _base(rate_type="fixed", origination_year=2021, mortgage_rate_pct=3.0) == "low_rate"


def test_low_rate_boundary_exclusive():
    assert _base(rate_type="fixed", origination_year=2021, mortgage_rate_pct=3.01) == "control"


def test_low_rate_wrong_year():
    assert _base(rate_type="fixed", origination_year=2020, mortgage_rate_pct=2.5) == "control"


def test_low_rate_takes_precedence_over_osfi():
    # Mirrors the simulator's check order: low_rate is tested first.
    assert (
        _base(
            rate_type="fixed",
            origination_year=2021,
            mortgage_rate_pct=2.5,
            insured_flag=False,
            credit_score_band="720_plus",
        )
        == "low_rate"
    )


def test_osfi_exempt():
    assert _base(insured_flag=False, credit_score_band="720_plus") == "osfi_exempt"


def test_osfi_exempt_needs_both():
    assert _base(insured_flag=False, credit_score_band="660_719") == "control"
    assert _base(insured_flag=True, credit_score_band="720_plus") == "control"


def test_family_squeeze():
    assert _base(household_income_cad=74000.0, n_children=3) == "family_squeeze"


def test_family_squeeze_boundaries():
    assert _base(household_income_cad=75000.0, n_children=3) == "control"
    assert _base(household_income_cad=74000.0, n_children=2) == "control"


def test_control_default():
    assert _base() == "control"


def test_numpy_bool_insured_flag():
    assert (
        cohort_label(
            rate_type="variable",
            origination_year=2022,
            mortgage_rate_pct=5.0,
            insured_flag=np.bool_(False),
            credit_score_band="720_plus",
            household_income_cad=100000.0,
            n_children=1,
        )
        == "osfi_exempt"
    )
