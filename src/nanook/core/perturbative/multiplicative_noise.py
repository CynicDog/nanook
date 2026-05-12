"""Multiplicative Noise: scale each value by ``1 + N(0, σ)``, preserving zeros.

Useful when an additive noise would generate impossible negatives (e.g. counts,
strictly positive incomes). Zero cells stay zero; all other values get scaled.
Reference: pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/multiplicative_noise.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nanook._internal.rng import generator
from nanook.core._base import SDCMethod
from nanook.core._registry import register_method
from nanook.exceptions import MethodParameterError, UnsupportedDtypeError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["MultiplicativeNoise"]


@register_method
class MultiplicativeNoise(SDCMethod):
    """Scale ``self.column`` by ``1 + N(0, intensity)``, leaving zeros untouched.

    Params:
        intensity: Standard deviation of the multiplicative factor.
        seed: Optional integer seed for reproducibility.
    """

    name = "multiplicative_noise"
    requires_pre_scan = False

    @classmethod
    def validate_params(cls, params: dict) -> None:
        if float(params.get("intensity", 0.05)) < 0.0:
            raise MethodParameterError("multiplicative_noise: intensity must be non-negative")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        return {}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = self.column
        if col is None or col not in df.columns:
            return df
        if not df.schema[col].is_numeric():
            raise UnsupportedDtypeError(f"multiplicative_noise requires numeric column; got {df.schema[col]}")
        intensity = float(self.params.get("intensity", 0.05))
        if intensity == 0.0:
            return df
        rng = generator(self.params.get("seed"))
        factors = 1.0 + rng.normal(0.0, intensity, df.height)
        return df.with_columns(
            pl.when(pl.col(col) == 0)
            .then(pl.col(col).cast(pl.Float64))
            .otherwise(pl.col(col).cast(pl.Float64) * pl.Series("_nk_factor", factors))
            .alias(col)
        )
