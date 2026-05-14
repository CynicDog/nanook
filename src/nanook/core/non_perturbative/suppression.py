"""Suppression: drop the configured column entirely from the output frame.

The strongest non-perturbative protection short of refusing to release the data
at all. Used for direct identifiers (names, SSN, e-mail) where any preserved
signal is already a leak; SDC guidance treats column removal as the only
defensible treatment for those columns.

Streaming-safe: ``apply`` only needs the per-batch frame and rewrites it as
``df.drop(column)``. ``column`` is required — unlike ``local_suppression`` this
rule operates on a single configured column, not the whole quasi-identifier set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nanook.core._base import SDCMethod
from nanook.core._schema import schema
from nanook.exceptions import MethodParameterError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["Suppression"]


@schema(
    display_name="Suppression",
    category="Non-perturbative",
    applicable_dtypes=("ANY",),
    description=(
        "Drop the configured column from the output frame. The strongest "
        "non-perturbative treatment — used for direct identifiers where any "
        "preserved value would already be a re-identification leak."
    ),
    params=(),
)
class Suppression(SDCMethod):
    """Drop ``self.column`` from the frame.

    The rule takes no parameters: the action is fully determined by which
    column is configured. Applying it to a column already absent from the
    frame is a no-op, which keeps re-runs and partial replays idempotent.
    """

    name = "suppression"
    requires_pre_scan = False
    drops_column = True

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        return {}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = self.column
        if col is None:
            raise MethodParameterError("suppression: column is required")
        if col not in df.columns:
            return df
        return df.drop(col)
