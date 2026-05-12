from __future__ import annotations

import polars as pl
import pytest

from nanook.exceptions import MethodParameterError
from nanook.metrics.utility.il1s import il1s


def test_zero_when_unchanged():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    assert il1s(df, df).scalar == 0.0


def test_constant_column_is_skipped():
    orig = pl.DataFrame({"x": [5.0, 5.0, 5.0], "y": [1.0, 2.0, 3.0]})
    prot = pl.DataFrame({"x": [5.0, 5.0, 5.0], "y": [1.0, 2.0, 3.0]})
    r = il1s(orig, prot)
    assert "x" not in r.per_column
    assert r.scalar == 0.0


def test_mismatched_heights_raises():
    orig = pl.DataFrame({"x": [1.0]})
    prot = pl.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(MethodParameterError):
        il1s(orig, prot)
