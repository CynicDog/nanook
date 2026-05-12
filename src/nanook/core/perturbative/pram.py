"""PRAM (Post-Randomisation Method): rewrite categorical values through a Markov transition matrix.

For each record, the protected category is drawn from a categorical
distribution conditioned on the original value. The transition matrix ``P`` is
either supplied explicitly or constructed during pre-scan to keep each
category's expected marginal close to the population marginal (the
invariant-marginal design).

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/pram.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nanook._internal.rng import generator
from nanook.core._base import SDCMethod
from nanook.core._registry import register_method
from nanook.exceptions import MethodParameterError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["PRAM"]


@register_method
class PRAM(SDCMethod):
    """Apply a per-cell Markov transition to ``self.column``.

    Params:
        retention: Probability of keeping the original value (diagonal entry).
            Must be in ``[0, 1]``. The remaining ``1 - retention`` mass is split
            proportionally across other categories by the population marginal.
        transition: Optional explicit ``categories × categories`` row-stochastic
            matrix; if supplied, ``retention`` is ignored.
        seed: Optional integer seed.
    """

    name = "pram"
    requires_pre_scan = True

    @classmethod
    def validate_params(cls, params: dict) -> None:
        r = float(params.get("retention", 0.8))
        if not 0.0 <= r <= 1.0:
            raise MethodParameterError("pram: retention must lie in [0, 1]")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        col = self.column
        if col is None:
            raise MethodParameterError("pram: column is required")
        retention = float(self.params.get("retention", 0.8))
        categories = df.get_column(col).drop_nulls().unique().sort().to_list()
        if not categories:
            return {"categories": [], "transition": []}
        custom = self.params.get("transition")
        if custom is not None:
            matrix = np.asarray(custom, dtype=float)
        else:
            counts = df.get_column(col).drop_nulls().value_counts()
            marginal_map = dict(counts.iter_rows())
            total = float(sum(marginal_map.values()))
            marginal = np.array([marginal_map.get(c, 0) / total for c in categories])
            matrix = _invariant_marginal_matrix(categories, retention, marginal)
        return {"categories": categories, "transition": matrix.tolist()}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = self.column
        if col is None or col not in df.columns:
            return df
        categories: list = params.get("categories", [])
        if not categories:
            return df
        matrix = np.asarray(params["transition"])
        cat_index = {c: i for i, c in enumerate(categories)}
        rng = generator(self.params.get("seed"))

        original = df.get_column(col).to_list()
        out = []
        for v in original:
            if v is None or v not in cat_index:
                out.append(v)
                continue
            probs = matrix[cat_index[v]]
            choice = rng.choice(len(categories), p=probs)
            out.append(categories[choice])
        return df.with_columns(pl.Series(col, out, dtype=df.schema[col]))


def _invariant_marginal_matrix(categories: list, retention: float, marginal: np.ndarray) -> np.ndarray:
    n = len(categories)
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i, i] = retention
        total_off = 1.0 - retention
        denom = 1.0 - marginal[i]
        if denom <= 0:
            matrix[i, i] = 1.0
            continue
        for j in range(n):
            if i != j:
                matrix[i, j] = total_off * marginal[j] / denom
    return matrix
