from __future__ import annotations

import polars as pl
import pytest

from nanook.context import DataContext
from nanook.exceptions import ContextValidationError


def test_roundtrip_through_dict():
    ctx = DataContext(quasi_identifiers=["a", "b"], sensitive=["c"], weights="w")
    again = DataContext.from_dict(ctx.to_dict())
    assert again == ctx


def test_validate_rejects_missing_column():
    ctx = DataContext(quasi_identifiers=["nope"])
    df = pl.DataFrame({"x": [1]})
    with pytest.raises(ContextValidationError):
        ctx.validate(df)


def test_validate_rejects_role_overlap():
    ctx = DataContext(quasi_identifiers=["a"], sensitive=["a"])
    df = pl.DataFrame({"a": [1]})
    with pytest.raises(ContextValidationError):
        ctx.validate(df)


def test_unknown_dict_key_rejected():
    with pytest.raises(ContextValidationError):
        DataContext.from_dict({"foo": 1})


def test_validate_rejects_missing_weights():
    ctx = DataContext(weights="w")
    df = pl.DataFrame({"x": [1]})
    with pytest.raises(ContextValidationError):
        ctx.validate(df)
