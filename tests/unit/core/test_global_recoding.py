from __future__ import annotations

import polars as pl
import pytest

from nanook.context import DataContext
from nanook.core.non_perturbative.global_recoding import GlobalRecoding
from nanook.exceptions import MethodParameterError


def _apply(df, **params):
    m = GlobalRecoding(column="x", params=params)
    return m.apply(df, DataContext(), {})


def test_continuous_index_labels():
    df = pl.DataFrame({"x": [5.0, 15.0, 25.0, 35.0]})
    out = _apply(df, bins=[0.0, 10.0, 20.0, 30.0, 40.0], label_mode="index")
    assert out.get_column("x").to_list() == [0, 1, 2, 3]


def test_continuous_midpoint_labels():
    df = pl.DataFrame({"x": [5.0, 15.0]})
    out = _apply(df, bins=[0.0, 10.0, 20.0], label_mode="midpoint")
    assert out.get_column("x").to_list() == [5.0, 15.0]


def test_categorical_mapping():
    df = pl.DataFrame({"x": ["a", "b", "c", "d"]})
    out = _apply(df, mapping={"a": "AB", "b": "AB", "c": "CD", "d": "CD"})
    assert out.get_column("x").to_list() == ["AB", "AB", "CD", "CD"]


def test_must_pass_exactly_one_form():
    with pytest.raises(MethodParameterError):
        GlobalRecoding.validate_params({})
    with pytest.raises(MethodParameterError):
        GlobalRecoding.validate_params({"bins": [0, 1], "mapping": {"a": "b"}})


def test_bins_must_be_strictly_increasing():
    with pytest.raises(MethodParameterError):
        GlobalRecoding.validate_params({"bins": [0, 1, 1, 2]})
