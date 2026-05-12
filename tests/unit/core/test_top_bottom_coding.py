from __future__ import annotations

import polars as pl
import pytest

from nanook.context import DataContext
from nanook.core.non_perturbative.top_bottom_coding import TopBottomCoding
from nanook.exceptions import MethodParameterError, UnsupportedDtypeError


def _run(df, **params):
    m = TopBottomCoding(column="x", params=params)
    fitted = m.pre_scan(df, DataContext())
    return m.apply(df, DataContext(), fitted), fitted


def test_two_sided_clips_to_quantile_bounds():
    df = pl.DataFrame({"x": [float(i) for i in range(100)]})
    out, params = _run(df, percentile=10, alternative="two_sided")
    col = out.get_column("x")
    # 5th and 95th percentiles roughly 4.95 / 94.05 on this uniform 0..99 column.
    assert col.min() >= params["lower_bound"] - 1e-9
    assert col.max() <= params["upper_bound"] + 1e-9


def test_less_only_clips_lower_tail():
    df = pl.DataFrame({"x": [float(i) for i in range(100)]})
    out, params = _run(df, percentile=10, alternative="less")
    assert params["lower_bound"] is not None
    assert params["upper_bound"] is None
    assert out.get_column("x").max() == pytest.approx(99.0)


def test_greater_only_clips_upper_tail():
    df = pl.DataFrame({"x": [float(i) for i in range(100)]})
    out, params = _run(df, percentile=10, alternative="greater")
    assert params["lower_bound"] is None
    assert params["upper_bound"] is not None
    assert out.get_column("x").min() == pytest.approx(0.0)


def test_invalid_percentile_rejected():
    with pytest.raises(MethodParameterError):
        TopBottomCoding.validate_params({"percentile": 0})
    with pytest.raises(MethodParameterError):
        TopBottomCoding.validate_params({"percentile": 100})


def test_non_numeric_column_rejected():
    df = pl.DataFrame({"x": ["a", "b", "c"]})
    m = TopBottomCoding(column="x", params={"percentile": 5})
    with pytest.raises(UnsupportedDtypeError):
        m.pre_scan(df, DataContext())
