from __future__ import annotations

import polars as pl
import pytest

from nanook.exceptions import MethodParameterError, UnsupportedDtypeError
from nanook.metrics.risk.t_closeness import t_closeness


def test_zero_emd_when_class_matches_population():
    df = pl.DataFrame({"zip": ["1", "1", "2", "2"], "x": [0.0, 1.0, 0.0, 1.0]})
    r = t_closeness(df, qis=["zip"], sensitive="x", t=0.1)
    assert r.holds
    assert r.max_emd == 0.0


def test_max_emd_when_class_is_homogeneous():
    df = pl.DataFrame({"zip": ["1", "1", "1", "2", "2", "2"], "x": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]})
    r = t_closeness(df, qis=["zip"], sensitive="x", t=0.1)
    assert not r.holds
    assert r.max_emd > 0.0


def test_t_must_be_in_unit_interval():
    df = pl.DataFrame({"zip": ["1"], "x": [0.0]})
    with pytest.raises(MethodParameterError):
        t_closeness(df, qis=["zip"], sensitive="x", t=1.5)


def test_ordinal_requires_numeric_sensitive():
    df = pl.DataFrame({"zip": ["1"], "x": ["A"]})
    with pytest.raises(UnsupportedDtypeError):
        t_closeness(df, qis=["zip"], sensitive="x", t=0.2, support="ordinal")
