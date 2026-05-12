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
