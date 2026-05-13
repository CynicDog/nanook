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


def test_golden_value_il1s_handbook_pattern():
    # X = [0, 2, 4, 6]. mean = 3, Σ(x - mean)² = 20.
    # polars Series.std() returns the sample std (ddof=1), so
    # s_x = sqrt(20/3) ≈ 2.58199.
    # Z = [1, 1, 5, 5]. |X - Z| = [1, 1, 1, 1]. mean(|X - Z|) = 1.
    # IL1s_x = 1 / sqrt(20/3) = sqrt(3/20) ≈ 0.38730.
    import math

    orig = pl.DataFrame({"x": [0.0, 2.0, 4.0, 6.0]})
    prot = pl.DataFrame({"x": [1.0, 1.0, 5.0, 5.0]})
    r = il1s(orig, prot)
    expected = 1.0 / math.sqrt(20.0 / 3.0)
    assert r.scalar == pytest.approx(expected, abs=1e-9)
    assert r.per_column == {"x": pytest.approx(expected, abs=1e-9)}
