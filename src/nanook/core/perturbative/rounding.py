"""Rounding: snap each value to the nearest multiple of ``base``.

A coarser grid degrades re-identification by interval disclosure while keeping
each observation analytically usable. Optional ``random_within_bin`` jitters
the rounded value uniformly within the bin so distribution shapes are preserved.

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/rounding.md.
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

__all__ = ["Rounding"]


@schema(
    display_name="Rounding",
    category="Perturbative",
    applicable_dtypes=("NUMERIC",),
    description=(
        "Snap each value to a multiple of ``base``. Optionally jitter inside the "
        "bin to preserve shape rather than collapsing every cell to the centre."
    ),
    params=(
        ParamSchema(
            name="base",
            display_name="Bin Size",
            param_type="FLOAT",
            default=10.0,
            required=True,
            description="Positive bin width; values snap to round(x / base) · base.",
        ),
        ParamSchema(
            name="random_within_bin",
            display_name="Jitter Inside Bin",
            param_type="BOOL",
            default="false",
            required=False,
            description=(
                "When true, draw the rounded value uniformly inside the bin "
                "instead of snapping to its centre. Requires a seed for reproducibility."
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
class Rounding(SDCMethod):
    """Round ``self.column`` to a multiple of ``base``.

    Params:
        base: Positive bin width; values snap to ``round(x / base) · base``.
        random_within_bin: If True, sample uniformly within the bin instead of
            snapping to the centre. Requires ``seed`` for reproducibility.
        seed: Optional integer seed.
    """

    name = "rounding"
    requires_pre_scan = False

    @classmethod
    def validate_params(cls, params: dict) -> None:
        if float(params.get("base", 1.0)) <= 0.0:
            raise MethodParameterError("rounding: base must be positive")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        return {}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = self.column
        if col is None or col not in df.columns:
            return df
        if not df.schema[col].is_numeric():
            raise UnsupportedDtypeError(f"rounding requires numeric column; got {df.schema[col]}")
        base = float(self.params.get("base", 1.0))
        snapped = (pl.col(col).cast(pl.Float64) / base).round() * base
        if not self.params.get("random_within_bin", False):
            return df.with_columns(snapped.alias(col))
        rng = generator(self.params.get("seed"))
        jitter = rng.uniform(-base / 2.0, base / 2.0, df.height)
        return df.with_columns((snapped + pl.Series("_nk_jitter", jitter)).alias(col))
