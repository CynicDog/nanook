"""Data Swapping: swap the value of ``column`` between random pairs of records.

Unlike rank-swapping, the pairing is uniformly random rather than constrained
to a rank-window — best used when the column's distribution should be preserved
exactly (every original value still appears, just permuted across rows).

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/data_swapping.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nanook._internal.rng import generator
from nanook.core._base import SDCMethod
from nanook.core._registry import register_method
from nanook.exceptions import MethodParameterError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["DataSwapping"]


@register_method
class DataSwapping(SDCMethod):
    """Swap ``self.column`` values between ``fraction`` of randomly paired rows.

    Params:
        fraction: Proportion of rows to involve in swaps, in ``(0, 1]``.
        seed: Optional integer seed.
    """

    name = "data_swapping"
    requires_pre_scan = False

    @classmethod
    def validate_params(cls, params: dict) -> None:
        f = float(params.get("fraction", 1.0))
        if not 0.0 < f <= 1.0:
            raise MethodParameterError("data_swapping: fraction must lie in (0, 1]")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        return {}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = self.column
        if col is None or col not in df.columns:
            return df
        n = df.height
        if n < 2:
            return df
        rng = generator(self.params.get("seed"))
        fraction = float(self.params.get("fraction", 1.0))
        k = max(2, int(n * fraction))
        if k % 2 == 1:
            k -= 1

        picked = rng.choice(n, size=k, replace=False)
        rng.shuffle(picked)
        values = df.get_column(col).to_numpy().copy()
        for i in range(0, k, 2):
            a, b = picked[i], picked[i + 1]
            values[a], values[b] = values[b], values[a]
        return df.with_columns(pl.Series(col, values))
