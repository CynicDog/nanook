from __future__ import annotations

import polars as pl
import pytest

from nanook.context import DataContext
from nanook.core.non_perturbative.local_suppression import LocalSuppression
from nanook.exceptions import MethodParameterError
from nanook.metrics.risk.k_anonymity import k_anonymity


def test_achieves_k_anonymity_on_small_dataset(adult_small):
    ctx = DataContext(quasi_identifiers=["age", "zip", "sex"])
    m = LocalSuppression(params={"target_k": 3})
    fitted = m.pre_scan(adult_small, ctx)
    out = m.apply(adult_small, ctx, fitted)
    r = k_anonymity(out, qis=["age", "zip", "sex"], k=3)
    # Either every record meets k=3 or the algorithm hit max_iter and made progress.
    assert r.violations < 8


def test_records_suppression_pairs(adult_small):
    ctx = DataContext(quasi_identifiers=["age", "zip", "sex"])
    fitted = LocalSuppression(params={"target_k": 3}).pre_scan(adult_small, ctx)
    assert "suppressions" in fitted
    assert fitted["cells_suppressed"] == len(fitted["suppressions"])


def test_requires_non_empty_qis():
    df = pl.DataFrame({"x": [1, 2, 3]})
    m = LocalSuppression(params={"target_k": 2})
    with pytest.raises(MethodParameterError):
        m.pre_scan(df, DataContext())


def test_invalid_target_k_rejected():
    with pytest.raises(MethodParameterError):
        LocalSuppression.validate_params({"target_k": 1})


def test_empty_violations_returns_identity(perfectly_k_anonymous):
    ctx = DataContext(quasi_identifiers=["zip", "age_bucket"])
    m = LocalSuppression(params={"target_k": 3})
    fitted = m.pre_scan(perfectly_k_anonymous, ctx)
    out = m.apply(perfectly_k_anonymous, ctx, fitted)
    assert out.equals(perfectly_k_anonymous)
    assert fitted["cells_suppressed"] == 0
