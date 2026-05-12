"""Global Recoding: replace fine-grained values with coarser groupings applied uniformly.

Two flavours:

- **Continuous**: cut the column at user-supplied ``bins`` and emit either the
  interval index, an interval-midpoint, or a printable label per ``label_mode``.
- **Categorical**: rewrite values through a user-supplied many-to-one ``mapping``.

The mapping is part of the (publishable) ``rule_stats`` so analysts can re-aggregate
companion variables to matching granularity.
Reference: pseudonymization-proposal/pseudo_code/sdc_methods/non_perturbative/global_recoding.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nanook.core._base import SDCMethod
from nanook.core._registry import register_method
from nanook.exceptions import MethodParameterError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["GlobalRecoding"]


@register_method
class GlobalRecoding(SDCMethod):
    """Map ``self.column`` through a publishable recoding rule.

    Params (continuous form):
        bins: Strictly increasing list of cut-points ``[c_0, c_1, …, c_M]`` defining
            ``M`` intervals; left-closed, right-open except the last.
        label_mode: ``"index"`` (default, returns ``0..M-1``), ``"midpoint"``
            (numeric midpoint per interval), or ``"label"`` (string ``"[c, c)"``).

    Params (categorical form):
        mapping: Dict of original-value -> coarser-value. Missing keys pass through.
    """

    name = "global_recoding"
    requires_pre_scan = False

    @classmethod
    def validate_params(cls, params: dict) -> None:
        bins = params.get("bins")
        mapping = params.get("mapping")
        if (bins is None) == (mapping is None):
            raise MethodParameterError("global_recoding: pass exactly one of `bins` or `mapping`")
        if bins is not None:
            if not isinstance(bins, list | tuple) or len(bins) < 2:
                raise MethodParameterError("global_recoding: bins must have at least two cut-points")
            for a, b in zip(bins, bins[1:], strict=False):
                if not a < b:
                    raise MethodParameterError("global_recoding: bins must be strictly increasing")
            label_mode = params.get("label_mode", "index")
            if label_mode not in ("index", "midpoint", "label"):
                raise MethodParameterError(f"global_recoding: unknown label_mode {label_mode!r}")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        return {}

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = self.column
        if col is None or col not in df.columns:
            return df

        bins = self.params.get("bins")
        if bins is not None:
            return self._apply_bins(df, col, bins, self.params.get("label_mode", "index"))

        mapping = self.params.get("mapping", {})
        return df.with_columns(pl.col(col).replace_strict(mapping, default=pl.col(col)).alias(col))

    @staticmethod
    def _apply_bins(df: pl.DataFrame, col: str, bins: list, label_mode: str) -> pl.DataFrame:
        cuts = [float(b) for b in bins]
        # polars `cut` returns the right-closed bin label; we re-map to the requested mode.
        binned = df.with_columns(
            pl.col(col)
            .cast(pl.Float64)
            .cut(
                breaks=cuts[1:-1],
                labels=[str(i) for i in range(len(cuts) - 1)],
                left_closed=True,
            )
            .alias("_nk_idx")
        )
        idx = binned.get_column("_nk_idx").cast(pl.Int64)
        if label_mode == "index":
            return df.with_columns(idx.alias(col))
        if label_mode == "midpoint":
            mids = [(cuts[j] + cuts[j + 1]) / 2.0 for j in range(len(cuts) - 1)]
            mapped = idx.replace_strict({i: mids[i] for i in range(len(mids))}, return_dtype=pl.Float64)
            return df.with_columns(mapped.alias(col))
        labels = [f"[{cuts[j]}, {cuts[j + 1]})" for j in range(len(cuts) - 1)]
        mapped = idx.replace_strict({i: labels[i] for i in range(len(labels))}, return_dtype=pl.String)
        return df.with_columns(mapped.alias(col))
