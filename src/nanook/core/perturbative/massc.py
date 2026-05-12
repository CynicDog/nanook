"""MASSC: Micro Agglomerative Statistical Sub-category Coding — categorical perturbation by group.

Records are first grouped on the quasi-identifiers, then within each group a
random subset of cells in the sensitive column is swapped with another cell
in the same group. This preserves both the group-level distribution and the
quasi-identifier integrity while breaking record-level linkability.

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/massc.md.
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

__all__ = ["MASSC"]


@register_method
class MASSC(SDCMethod):
    """Swap ``self.column`` values within quasi-identifier groups, perturbing ``fraction`` of records.

    Params:
        fraction: Fraction of in-group cells to swap, in ``(0, 1]``.
        seed: Optional integer seed.
    """

    name = "massc"
    requires_pre_scan = True

    @classmethod
    def validate_params(cls, params: dict) -> None:
        f = float(params.get("fraction", 0.5))
        if not 0.0 < f <= 1.0:
            raise MethodParameterError("massc: fraction must lie in (0, 1]")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:
        qis = list(ctx.quasi_identifiers)
        if not qis:
            raise MethodParameterError("massc: requires non-empty DataContext.quasi_identifiers")
        col = self.column
        if col is None:
            raise MethodParameterError("massc: column is required")
        if col not in df.columns:
            raise MethodParameterError(f"massc: column {col!r} not in frame")

        fraction = float(self.params.get("fraction", 0.5))
        rng = generator(self.params.get("seed"))

        # Build per-group row-index lists once during pre-scan; apply does only the swap.
        rows_by_group: dict[tuple, list[int]] = {}
        for i, row in enumerate(df.select(qis).iter_rows(named=True)):
            key = tuple(row[c] for c in qis)
            rows_by_group.setdefault(key, []).append(i)

        swaps: list[tuple[int, int]] = []
        for indices in rows_by_group.values():
            m = len(indices)
            if m < 2:
                continue
            n_swap = max(2, int(m * fraction))
            if n_swap % 2 == 1:
                n_swap -= 1
            picked = rng.choice(indices, size=n_swap, replace=False).tolist()
            rng.shuffle(picked)
            swaps.extend((picked[i], picked[i + 1]) for i in range(0, n_swap, 2))
        return {"swaps": swaps, "column": col}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = params["column"]
        if col not in df.columns:
            return df
        swaps: list[tuple[int, int]] = params.get("swaps", [])
        if not swaps:
            return df
        values = df.get_column(col).to_list()
        for a, b in swaps:
            values[a], values[b] = values[b], values[a]
        return df.with_columns(pl.Series(col, values, dtype=df.schema[col]))
