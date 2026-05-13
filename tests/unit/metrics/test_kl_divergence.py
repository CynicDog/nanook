from __future__ import annotations

import polars as pl
import pytest

from nanook.exceptions import MethodParameterError
from nanook.metrics.utility.kl_divergence import kl_divergence


def test_zero_when_distributions_match():
    df = pl.DataFrame({"c": ["a", "a", "b", "b"]})
    r = kl_divergence(df, df)
    assert r.scalar == pytest.approx(0.0, abs=1e-9)


def test_positive_when_protected_drifts():
    orig = pl.DataFrame({"c": ["a", "a", "a", "b"]})
    prot = pl.DataFrame({"c": ["a", "b", "b", "b"]})
    r = kl_divergence(orig, prot)
    assert r.scalar > 0.0


def test_epsilon_must_be_positive():
    df = pl.DataFrame({"c": ["a", "b"]})
    with pytest.raises(MethodParameterError):
        kl_divergence(df, df, epsilon=0.0)


def test_golden_value_kl_three_bin_categorical():
    # Original P: counts {a: 3, b: 2, c: 1} on n=6. P = (1/2, 1/3, 1/6).
    # Protected Q: counts {a: 2, b: 2, c: 2}. Q_raw = (1/3, 1/3, 1/3).
    # With epsilon ~ 0 (we use 1e-9), Laplace smoothing is negligible.
    # D_KL(P || Q) = 0.5 * ln(1.5) + (1/3) * ln(1) + (1/6) * ln(0.5)
    #             = 0.5 * 0.4054651 + 0 + (1/6) * (-0.6931472)
    #             ≈ 0.0872081.
    import math

    orig = pl.DataFrame({"c": ["a", "a", "a", "b", "b", "c"]})
    prot = pl.DataFrame({"c": ["a", "a", "b", "b", "c", "c"]})
    r = kl_divergence(orig, prot, epsilon=1e-9)
    expected = 0.5 * math.log(1.5) + (1 / 6) * math.log(0.5)
    assert r.scalar == pytest.approx(expected, abs=1e-6)
    assert r.per_column["c"] == pytest.approx(expected, abs=1e-6)
