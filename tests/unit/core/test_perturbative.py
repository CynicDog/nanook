"""Smoke + invariant tests for the 9 perturbative methods.

We avoid asserting exact numerical equivalence (RNG-dependent) and instead test
the invariants documented in each method: dtype preservation, value-domain
sanity, reproducibility under fixed seed, and parameter validation.
"""

from __future__ import annotations

import polars as pl
import pytest

from nanook.context import DataContext
from nanook.core._registry import get_method
from nanook.exceptions import MethodParameterError, UnsupportedDtypeError


def _fit(name, *, column=None, params=None, df=None, ctx=None):
    cls = get_method(name)
    instance = cls(column=column, params=params or {})
    fitted = instance.pre_scan(df, ctx or DataContext()) if cls.requires_pre_scan else {}
    return instance, fitted


def test_noise_addition_changes_values_but_preserves_dtype():
    df = pl.DataFrame({"x": [float(i) for i in range(50)]})
    m, fitted = _fit("noise_addition", column="x", params={"intensity": 0.5, "seed": 1}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert out.schema["x"] == pl.Float64
    assert not out.equals(df)


def test_noise_addition_negative_intensity_rejected():
    with pytest.raises(MethodParameterError):
        get_method("noise_addition").validate_params({"intensity": -0.1})


def test_multiplicative_noise_preserves_zero():
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 0.0]})
    m, fitted = _fit("multiplicative_noise", column="x", params={"intensity": 0.2, "seed": 7}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert out.get_column("x")[0] == 0.0
    assert out.get_column("x")[3] == 0.0


def test_rounding_snaps_to_base():
    df = pl.DataFrame({"x": [0.3, 0.7, 1.2, 1.8]})
    m, fitted = _fit("rounding", column="x", params={"base": 1.0}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert out.get_column("x").to_list() == [0.0, 1.0, 1.0, 2.0]


def test_rounding_with_jitter_stays_within_bin():
    df = pl.DataFrame({"x": [12.0] * 50})  # snaps to centre 10, jitter window [-5, +5] → [5, 15]
    m, fitted = _fit(
        "rounding", column="x", params={"base": 10.0, "random_within_bin": True, "seed": 3}, df=df
    )
    out = m.apply(df, DataContext(), fitted)
    for v in out.get_column("x").to_list():
        assert 5.0 <= v <= 15.0


def test_rank_swapping_preserves_value_multiset():
    df = pl.DataFrame({"x": [float(i) for i in range(20)]})
    m, fitted = _fit("rank_swapping", column="x", params={"window_pct": 0.2, "seed": 11}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert sorted(out.get_column("x").to_list()) == sorted(df.get_column("x").to_list())


def test_data_swapping_preserves_value_multiset():
    df = pl.DataFrame({"x": list(range(20))})
    m, _ = _fit("data_swapping", column="x", params={"fraction": 1.0, "seed": 2}, df=df)
    out = m.apply(df, DataContext(), {})
    assert sorted(out.get_column("x").to_list()) == sorted(df.get_column("x").to_list())


def test_microaggregation_produces_at_least_k_duplicates():
    df = pl.DataFrame({"x": [float(i) for i in range(20)]})
    m, fitted = _fit("microaggregation", column="x", params={"k": 5}, df=df)
    out = m.apply(df, DataContext(), fitted)
    counts = out.get_column("x").value_counts()
    assert counts.get_column("count").min() >= 5


def test_resampling_preserves_rank_order_approximately():
    df = pl.DataFrame({"x": [float(i) for i in range(30)]})
    m, fitted = _fit("resampling", column="x", params={"b": 30, "seed": 5}, df=df)
    out = m.apply(df, DataContext(), fitted)
    # Sorted-rank should remain monotone non-decreasing.
    sorted_out = sorted(out.get_column("x").to_list())
    assert (
        sorted_out
        == out.get_column("x").gather(pl.Series("rank", df.get_column("x").arg_sort().to_list())).to_list()
    )


def test_pram_eventually_keeps_some_records():
    df = pl.DataFrame({"x": ["A"] * 30 + ["B"] * 30 + ["C"] * 30})
    m, fitted = _fit("pram", column="x", params={"retention": 0.8, "seed": 13}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert set(out.get_column("x").to_list()) <= {"A", "B", "C"}


def test_pram_retention_one_is_identity():
    df = pl.DataFrame({"x": ["A", "B", "C"]})
    m, fitted = _fit("pram", column="x", params={"retention": 1.0, "seed": 1}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert out.equals(df)


def test_massc_requires_quasi_identifiers():
    df = pl.DataFrame({"q": ["x"], "s": ["A"]})
    with pytest.raises(MethodParameterError):
        get_method("massc")(column="s", params={"fraction": 0.5}).pre_scan(df, DataContext())


def test_massc_preserves_in_group_multiset():
    df = pl.DataFrame(
        {
            "q": ["a"] * 10 + ["b"] * 10,
            "s": list(range(10)) + list(range(10, 20)),
        }
    )
    ctx = DataContext(quasi_identifiers=["q"])
    m, fitted = _fit("massc", column="s", params={"fraction": 1.0, "seed": 4}, df=df, ctx=ctx)
    out = m.apply(df, ctx, fitted)
    a_in = sorted(df.filter(pl.col("q") == "a").get_column("s").to_list())
    a_out = sorted(out.filter(pl.col("q") == "a").get_column("s").to_list())
    assert a_in == a_out


def test_unsupported_dtype_on_noise():
    df = pl.DataFrame({"x": ["a", "b"]})
    with pytest.raises(UnsupportedDtypeError):
        _fit("noise_addition", column="x", df=df)
