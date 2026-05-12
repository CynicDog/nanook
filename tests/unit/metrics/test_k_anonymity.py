from __future__ import annotations

import polars as pl
import pytest

from nanook.exceptions import ContextValidationError, MethodParameterError
from nanook.metrics.risk.k_anonymity import k_anonymity


def test_holds_when_every_class_meets_threshold(perfectly_k_anonymous):
    r = k_anonymity(perfectly_k_anonymous, qis=["zip", "age_bucket"], k=3)
    assert r.holds
    assert r.violations == 0
    assert r.violation_rate == 0.0
    assert r.sample_uniques == 0


def test_counts_records_below_threshold(adult_small):
    # Classes on (age, zip, sex): the 40-year-olds split 1/2/1 by sex, and the 50/M is alone.
    r = k_anonymity(adult_small, qis=["age", "zip", "sex"], k=3)
    assert not r.holds
    assert r.violations > 0


def test_k_must_be_at_least_two():
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(MethodParameterError):
        k_anonymity(df, qis=["x"], k=1)


def test_missing_column_raises():
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ContextValidationError):
        k_anonymity(df, qis=["nope"], k=2)


def test_empty_qis_raises():
    df = pl.DataFrame({"x": [1, 2]})
    with pytest.raises(ContextValidationError):
        k_anonymity(df, qis=[], k=2)


def test_empty_frame_holds_trivially():
    df = pl.DataFrame({"x": []}, schema={"x": pl.Int64})
    r = k_anonymity(df, qis=["x"], k=2)
    assert r.holds
    assert r.violations == 0


def test_sample_uniques_counts_singleton_classes():
    df = pl.DataFrame({"x": [1, 1, 2, 3]})
    r = k_anonymity(df, qis=["x"], k=2)
    assert r.sample_uniques == 2
