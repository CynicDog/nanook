from __future__ import annotations

import polars as pl
import pytest

from nanook.exceptions import MethodParameterError
from nanook.pipeline import Pipeline


def test_fluent_and_declarative_roundtrip():
    fluent = Pipeline().context(quasi_identifiers=["age", "zip"], sensitive=["diagnosis"])
    payload = fluent.to_dict()
    declarative = Pipeline.from_dict(payload)
    assert declarative.context_.quasi_identifiers == ("age", "zip")
    assert declarative.context_.sensitive == ("diagnosis",)
    assert declarative.to_dict() == payload


def test_unknown_top_level_key_rejected():
    with pytest.raises(MethodParameterError):
        Pipeline.from_dict({"version": 1, "ohno": 1})


def test_version_mismatch_rejected():
    with pytest.raises(MethodParameterError):
        Pipeline.from_dict({"version": 999})


def test_assess_returns_zero_loss_when_protected_is_original(adult_small):
    p = Pipeline().context(quasi_identifiers=["age", "zip"], sensitive=["diagnosis"])
    report = p.assess(adult_small, adult_small, k=3)
    assert report.utility.lambda_measure.scalar == 0.0
    assert report.utility.kl_divergence.scalar == pytest.approx(0.0, abs=1e-9)
    assert report.risk.k_anonymity is not None
    assert report.risk.l_diversity is not None


def test_apply_with_no_steps_is_identity():
    p = Pipeline().context(quasi_identifiers=["age"])
    df = pl.DataFrame({"age": [1, 2, 3]})
    out = p.apply(df)
    assert out.equals(df)


def test_to_dict_includes_seed():
    p = Pipeline(seed=42)
    assert p.to_dict()["seed"] == 42
