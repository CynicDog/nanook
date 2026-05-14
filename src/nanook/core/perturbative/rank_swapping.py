"""Rank Swapping: replace each value with the value of a neighbour close in rank.

Sort the column, then for each record randomly swap with one whose rank lies
within ``window_pct`` percent of its own rank. Preserves marginal distributions
and correlations close to ``window_pct``-bounded perturbations.

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/rank_swapping.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nanook._internal.rng import generator
from nanook.core._base import SDCMethod
from nanook.core._schema import ParamSchema, schema
from nanook.exceptions import MethodParameterError, UnsupportedDtypeError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["RankSwapping"]


@schema(
    display_name="Rank Swapping",
    category="Perturbative",
    applicable_dtypes=("NUMERIC",),
    description=(
        "Sort the column, then swap each value with another whose rank lies "
        "within ``window_pct`` of its own. Preserves the marginal distribution "
        "exactly while bounding the perturbation in rank space."
    ),
    params=(
        ParamSchema(
            name="window_pct",
            display_name="Rank Window",
            param_type="FLOAT",
            default=0.05,
            required=True,
            description=(
                "Half-width of the swap window as a fraction of n (e.g. 0.05 "
                "allows swaps within ±5% of the rank position)."
            ),
        ),
        ParamSchema(
            name="seed",
            display_name="Random Seed",
            param_type="INT",
            default=None,
            required=False,
            description="Optional integer seed for reproducibility.",
        ),
    ),
)
class RankSwapping(SDCMethod):
    """Swap each value with a neighbour whose rank lies within ``window_pct`` percent.

    Params:
        window_pct: Half-width of the swap window as a fraction of ``n`` (e.g.
            ``0.05`` allows swaps within ±5% of the rank position).
        seed: Optional integer seed.
    """

    name = "rank_swapping"
    requires_pre_scan = True

    @classmethod
    def validate_params(cls, params: dict) -> None:
        w = float(params.get("window_pct", 0.05))
        if not 0.0 < w <= 1.0:
            raise MethodParameterError("rank_swapping: window_pct must lie in (0, 1]")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        col = self.column
        if col is None:
            raise MethodParameterError("rank_swapping: column is required")
        if not df.schema[col].is_numeric():
            raise UnsupportedDtypeError(f"rank_swapping requires numeric column; got {df.schema[col]}")
        # Stash the ranks so apply() can do its work without re-sorting per batch.
        order = df.get_column(col).cast(pl.Float64).arg_sort()
        return {"order": order.to_list()}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = self.column
        if col is None or col not in df.columns:
            return df
        order: list[int] = params.get("order", [])
        n = len(order)
        if n <= 1:
            return df
        window = max(1, int(n * float(self.params.get("window_pct", 0.05))))
        rng = generator(self.params.get("seed"))

        sorted_values = df.get_column(col).cast(pl.Float64).gather(order).to_numpy()
        permutation = np.arange(n)
        for rank in range(n):
            lo = max(0, rank - window)
            hi = min(n - 1, rank + window)
            target = int(rng.integers(lo, hi + 1))
            permutation[rank], permutation[target] = permutation[target], permutation[rank]

        swapped_sorted = sorted_values[permutation]
        # Place swapped values back in original row order via the inverse of `order`.
        out_values = np.empty(n, dtype=sorted_values.dtype)
        out_values[np.asarray(order)] = swapped_sorted
        return df.with_columns(pl.Series(col, out_values))
