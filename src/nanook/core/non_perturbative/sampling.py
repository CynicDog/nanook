"""Sampling: release a random subset of records so an intruder cannot confirm anyone's presence.

This implementation covers the SRS variant. ``fraction`` is the inclusion
probability ``π``; row inclusion is drawn independently per record (Bernoulli),
which is the engine-streaming-friendly approximation of SRS-without-replacement.
A ``_nk_weight`` column carrying the Horvitz-Thompson weight ``1/π`` is appended
when ``write_weights=True`` — analysts need it for unbiased estimation.

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/non_perturbative/sampling.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nanook._internal.rng import generator
from nanook.core._base import SDCMethod
from nanook.core._schema import ParamSchema, schema
from nanook.exceptions import MethodParameterError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["Sampling"]


@schema(
    display_name="Sampling",
    category="Non-perturbative",
    applicable_dtypes=("ANY",),
    description=(
        "Release a random subset of records via independent Bernoulli draws. "
        "Optionally append a Horvitz-Thompson weight column ``_nk_weight`` for "
        "unbiased downstream estimation."
    ),
    params=(
        ParamSchema(
            name="fraction",
            display_name="Inclusion Probability",
            param_type="FLOAT",
            default=0.5,
            required=True,
            description=("Per-record inclusion probability, in (0, 1]. 1.0 keeps every row."),
        ),
        ParamSchema(
            name="seed",
            display_name="Random Seed",
            param_type="INT",
            default=None,
            required=False,
            description="Optional integer seed for reproducibility.",
        ),
        ParamSchema(
            name="write_weights",
            display_name="Emit HT Weights",
            param_type="BOOL",
            default="false",
            required=False,
            description=("When true, append ``_nk_weight = 1 / fraction`` to the output frame."),
        ),
    ),
)
class Sampling(SDCMethod):
    """Independent Bernoulli sampling with optional Horvitz-Thompson weight emission.

    Params:
        fraction: Inclusion probability per record, in ``(0, 1]``. ``1.0`` keeps every row.
        seed: Optional integer seed for reproducibility.
        write_weights: If True, append ``_nk_weight = 1 / fraction`` to the output.
    """

    name = "sampling"
    requires_pre_scan = False

    @classmethod
    def validate_params(cls, params: dict) -> None:
        f = params.get("fraction", 1.0)
        if not 0.0 < float(f) <= 1.0:
            raise MethodParameterError("sampling: fraction must lie in (0, 1]")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        return {}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        fraction = float(self.params.get("fraction", 1.0))
        if fraction == 1.0:
            return df
        rng = generator(self.params.get("seed"))
        mask = rng.random(df.height) < fraction
        kept = df.filter(pl.Series(mask))
        if self.params.get("write_weights", False):
            kept = kept.with_columns(pl.lit(1.0 / fraction).alias("_nk_weight"))
        return kept
