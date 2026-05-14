from __future__ import annotations

import polars as pl
import pytest

from nanook.context import DataContext
from nanook.core import METHOD_REGISTRY, list_method_schemas
from nanook.core.non_perturbative.suppression import Suppression
from nanook.exceptions import MethodParameterError
from nanook.pipeline import Pipeline


def test_drops_configured_column(adult_small):
    ctx = DataContext(quasi_identifiers=["age", "zip", "sex"])
    m = Suppression(column="age")
    out = m.apply(adult_small, ctx, m.pre_scan(adult_small, ctx))
    assert "age" not in out.columns
    assert set(out.columns) == set(adult_small.columns) - {"age"}
    assert out.height == adult_small.height


def test_missing_column_is_idempotent(adult_small):
    ctx = DataContext(quasi_identifiers=["age"])
    already_dropped = adult_small.drop("age")
    out = Suppression(column="age").apply(already_dropped, ctx, {})
    assert out.equals(already_dropped)


def test_requires_column():
    m = Suppression()
    with pytest.raises(MethodParameterError):
        m.apply(pl.DataFrame({"x": [1, 2]}), DataContext(), {})


def test_registered_with_drops_column_flag():
    assert "suppression" in METHOD_REGISTRY
    assert METHOD_REGISTRY["suppression"].drops_column is True
    assert METHOD_REGISTRY["suppression"].requires_pre_scan is False


def test_schema_entry_marks_drops_column():
    entry = next(e for e in list_method_schemas() if e["code"] == "suppression")
    assert entry["dropsColumn"] is True
    assert entry["category"] == "Non-perturbative"
    assert entry["params"] == []


def test_pipeline_round_trip_via_dict(adult_small):
    payload = {
        "version": 1,
        "context": {"quasi_identifiers": ["age"], "sensitive": ["diagnosis"]},
        "steps": [{"method": "suppression", "column": "age", "params": {}}],
    }
    p = Pipeline.from_dict(payload)
    out = p.apply(adult_small)
    assert "age" not in out.columns
    assert p.to_dict()["steps"][0]["method"] == "suppression"


def test_assess_tolerates_dropped_qi(adult_small):
    p = (
        Pipeline()
        .context(quasi_identifiers=["age", "zip", "sex"], sensitive=["diagnosis"])
        .suppression("age")
    )
    protected = p.apply(adult_small)
    report = p.assess(adult_small, protected, k=3)
    assert "age" not in protected.columns
    # k-anonymity still computed on the remaining QIs.
    assert report.risk.k_anonymity is not None
    assert report.risk.k_anonymity.qis == ("zip", "sex")


def test_assess_with_all_qis_dropped_yields_empty_risk(adult_small):
    p = Pipeline().context(quasi_identifiers=["age"]).suppression("age")
    protected = p.apply(adult_small)
    report = p.assess(adult_small, protected)
    assert report.risk.k_anonymity is None
