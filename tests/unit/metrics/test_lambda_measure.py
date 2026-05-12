from __future__ import annotations

import polars as pl
import pytest

from nanook.exceptions import MethodParameterError
from nanook.metrics.utility.lambda_measure import lambda_measure


def test_zero_loss_when_protected_equals_original():
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    r = lambda_measure(df, df)
    assert r.scalar == 0.0
    assert r.per_column == {"x": 0.0}


def test_handbook_example_one_tenth():
    orig = pl.DataFrame({"x": [0.0, 10.0]})
    prot = pl.DataFrame({"x": [1.0, 9.0]})
    # |0-1| + |10-9| = 2; mean = 1; range = 10; lambda_x = 0.1.
    assert lambda_measure(orig, prot).scalar == pytest.approx(0.1)


def test_constant_column_contributes_zero():
    orig = pl.DataFrame({"x": [5.0, 5.0, 5.0]})
    prot = pl.DataFrame({"x": [5.0, 5.0, 5.0]})
    assert lambda_measure(orig, prot).scalar == 0.0


def test_mismatched_heights_raises():
    orig = pl.DataFrame({"x": [1.0, 2.0]})
    prot = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(MethodParameterError):
        lambda_measure(orig, prot)
