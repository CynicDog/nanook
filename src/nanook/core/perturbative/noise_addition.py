"""Noise Addition: add zero-mean Gaussian noise scaled to each column's standard deviation.

The scaling guarantees the noise distribution is invariant to unit choice; an
intensity of ``0.1`` always means "ten percent of one standard deviation",
regardless of whether the column is currency or counts.

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/noise_addition.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nanook._internal.rng import generator
from nanook.core._base import SDCMethod
from nanook.core._schema import ParamSchema, schema
from nanook.exceptions import MethodParameterError, UnsupportedDtypeError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["NoiseAddition"]


@schema(
    display_name="Noise Addition",
    category="Perturbative",
    applicable_dtypes=("NUMERIC",),
    description=(
        "Add zero-mean Gaussian noise scaled to the column's standard deviation. "
        "Intensity is unit-free: 0.10 always means ±10% of σ regardless of "
        "whether the column is currency or counts."
    ),
    params=(
        ParamSchema(
            name="intensity",
            display_name="Intensity",
            param_type="FLOAT",
            default=0.1,
            required=True,
            description=(
                "Noise scale as a fraction of column σ. Typical: 0.05 (light), 0.10 (moderate), 0.20 (heavy)."
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
class NoiseAddition(SDCMethod):
    """Add ``N(0, intensity · σ)`` noise to ``self.column``; numeric only.

    Params:
        intensity: Noise scale as a fraction of the column standard deviation.
            Typical: ``0.05`` (light), ``0.10`` (moderate), ``0.20`` (heavy).
        seed: Optional integer seed for reproducibility.
    """

    name = "noise_addition"
    requires_pre_scan = True

    @classmethod
    def validate_params(cls, params: dict) -> None:
        if float(params.get("intensity", 0.1)) < 0.0:
            raise MethodParameterError("noise_addition: intensity must be non-negative")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        col = self.column
        if col is None:
            raise MethodParameterError("noise_addition: column is required")
        if not df.schema[col].is_numeric():
            raise UnsupportedDtypeError(f"noise_addition requires numeric column; got {df.schema[col]}")
        sd = df.get_column(col).cast(pl.Float64).std() or 0.0
        return {"sigma": float(sd) * float(self.params.get("intensity", 0.1))}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = self.column
        if col is None or col not in df.columns:
            return df
        sigma = float(params.get("sigma", 0.0))
        if sigma == 0.0:
            return df
        rng = generator(self.params.get("seed"))
        noise = rng.normal(0.0, sigma, df.height)
        return df.with_columns((pl.col(col).cast(pl.Float64) + pl.Series("_nk_noise", noise)).alias(col))
