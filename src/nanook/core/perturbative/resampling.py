"""Resampling: replace each value with the mean of multiple bootstrap samples taken at the same rank position.

Algorithm: draw ``b`` independent bootstrap samples of the column, sort each,
then replace the ``i``-th order statistic of the original column with the
average of the ``i``-th order statistics across the bootstraps. The method
preserves the rank order of values while smoothing tail uniqueness.

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/resampling.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nanook._internal.rng import generator
from nanook.core._base import SDCMethod
from nanook.core._registry import register_method
from nanook.exceptions import MethodParameterError, UnsupportedDtypeError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["Resampling"]


@register_method
class Resampling(SDCMethod):
    """Smooth ``self.column`` via averaging order statistics across bootstrap samples.

    Params:
        b: Number of bootstrap samples (``>= 1``). Higher ``b`` smooths more.
        seed: Optional integer seed.
    """

    name = "resampling"
    requires_pre_scan = True

    @classmethod
    def validate_params(cls, params: dict) -> None:
        if int(params.get("b", 10)) < 1:
            raise MethodParameterError("resampling: b must be >= 1")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        col = self.column
        if col is None:
            raise MethodParameterError("resampling: column is required")
        if not df.schema[col].is_numeric():
            raise UnsupportedDtypeError(f"resampling requires numeric column; got {df.schema[col]}")
        b = int(self.params.get("b", 10))
        rng = generator(self.params.get("seed"))
        x = df.get_column(col).cast(pl.Float64).to_numpy()
        n = x.size
        # Average order statistics across `b` bootstraps yields the smoothed sorted values.
        sorted_means = np.zeros(n)
        for _ in range(b):
            sample = rng.choice(x, size=n, replace=True)
            sorted_means += np.sort(sample)
        sorted_means /= b
        rank_order = np.argsort(x)
        smoothed = np.empty_like(sorted_means)
        smoothed[rank_order] = sorted_means
        return {"smoothed": smoothed.tolist()}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = self.column
        if col is None or col not in df.columns:
            return df
        return df.with_columns(pl.Series(col, params["smoothed"]))
