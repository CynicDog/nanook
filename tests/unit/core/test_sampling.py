from __future__ import annotations

import polars as pl
import pytest

from nanook.context import DataContext
from nanook.core.non_perturbative.sampling import Sampling
from nanook.exceptions import MethodParameterError


def test_fraction_one_is_identity():
    df = pl.DataFrame({"x": range(100)})
    method = Sampling(params={"fraction": 1.0, "seed": 42})
    out = method.apply(df, DataContext(), {})
    assert out.equals(df)


def test_fraction_half_returns_about_half_with_fixed_seed():
    df = pl.DataFrame({"x": range(1000)})
    method = Sampling(params={"fraction": 0.5, "seed": 42})
    out = method.apply(df, DataContext(), {})
    assert 400 < out.height < 600  # well within sampling tolerance


def test_write_weights_adds_column():
    df = pl.DataFrame({"x": range(100)})
    method = Sampling(params={"fraction": 0.25, "seed": 7, "write_weights": True})
    out = method.apply(df, DataContext(), {})
    assert "_nk_weight" in out.columns
    assert out.get_column("_nk_weight")[0] == pytest.approx(4.0)


def test_invalid_fraction_rejected():
    with pytest.raises(MethodParameterError):
        Sampling.validate_params({"fraction": 1.5})
    with pytest.raises(MethodParameterError):
        Sampling.validate_params({"fraction": 0.0})


def test_seeded_runs_are_reproducible():
    df = pl.DataFrame({"x": range(200)})
    a = Sampling(params={"fraction": 0.3, "seed": 99}).apply(df, DataContext(), {})
    b = Sampling(params={"fraction": 0.3, "seed": 99}).apply(df, DataContext(), {})
    assert a.equals(b)
