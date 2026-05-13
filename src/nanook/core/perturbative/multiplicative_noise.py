"""Multiplicative Noise (Höhne 2004): log-normal multiplier with moment rescaling.

Each non-zero value is scaled by an independent log-normal factor centred on 1,
then the column is rescaled so the first two moments match the original
non-zero subset (`X^{aR} := (sigma_X / sigma_{X^a}) * (X^a - mu_{X^a}) + mu_X`).
Zeros stay zero. Suitable for strictly-positive variables (incomes, turnover)
where additive noise would generate impossible negatives or where a Gaussian
multiplier could flip signs.

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
    """Scale ``self.column`` by a log-normal multiplier, then rescale to restore the first two moments.

    Params:
        sigma_log: Standard deviation of the underlying normal in the log-normal
            multiplier ``Z ~ LogNormal(0, sigma_log**2)``. Typical range
            ``[0.05, 0.25]``; ``0.0`` short-circuits to identity.
        seed: Optional integer seed for reproducibility.
    """

    name = "multiplicative_noise"
    requires_pre_scan = False

    @classmethod
    def validate_params(cls, params: dict) -> None:
        if float(params.get("sigma_log", 0.1)) < 0.0:
            raise MethodParameterError("multiplicative_noise: sigma_log must be non-negative")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        return {}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = self.column
        if col is None or col not in df.columns:
            return df
        if not df.schema[col].is_numeric():
            raise UnsupportedDtypeError(f"multiplicative_noise requires numeric column; got {df.schema[col]}")
        sigma_log = float(self.params.get("sigma_log", 0.1))
        if sigma_log == 0.0:
            return df

        rng = generator(self.params.get("seed"))
        x = df.get_column(col).cast(pl.Float64).to_numpy()
        nz_mask = x != 0.0
        nz = x[nz_mask]
        if nz.size == 0:
            return df.with_columns(pl.col(col).cast(pl.Float64))

        # Step 1: positive log-normal multiplier (E[Z] = exp(sigma_log**2 / 2)).
        z = rng.lognormal(mean=0.0, sigma=sigma_log, size=nz.size)
        x_a = z * nz

        # Step 2: original and raw-perturbed moments on the non-zero subset.
        mu_x = float(nz.mean())
        sigma_x = float(nz.std(ddof=0))
        sigma_xa = float(x_a.std(ddof=0))

        # Step 3: Höhne moment-preserving rescaling. If the raw perturbed std
        # collapses (e.g. nz is a single point), keep the raw perturbed values.
        x_aR = (sigma_x / sigma_xa) * (x_a - float(x_a.mean())) + mu_x if sigma_xa > 0.0 else x_a

        out = x.copy()
        out[nz_mask] = x_aR
        return df.with_columns(pl.Series(col, out, dtype=pl.Float64))
