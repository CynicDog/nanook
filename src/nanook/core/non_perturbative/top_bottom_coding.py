"""Top and Bottom Coding: clip values above and below percentile thresholds computed from the full column.

The interior of the distribution is preserved exactly; only the tails — which
carry most of the re-identification weight on continuous attributes — are
collapsed to the threshold value.

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/non_perturbative/top_bottom_coding.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nanook.core._base import SDCMethod
from nanook.core._schema import ParamSchema, schema
from nanook.exceptions import MethodParameterError, UnsupportedDtypeError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["TopBottomCoding"]


@schema(
    display_name="Top/Bottom Coding",
    category="Non-perturbative",
    applicable_dtypes=("NUMERIC",),
    description=(
        "Clip values beyond percentile thresholds computed across the full column. "
        "The interior of the distribution stays exact; only the tails collapse to "
        "the threshold boundary."
    ),
    params=(
        ParamSchema(
            name="percentile",
            display_name="Percentile",
            param_type="FLOAT",
            default=5.0,
            required=True,
            description=(
                "Total percentage of mass to clip, in (0, 100). With both tails "
                "selected, it splits equally above and below."
            ),
        ),
        ParamSchema(
            name="alternative",
            display_name="Side",
            param_type="CODE",
            default="two_sided",
            required=False,
            code_options=(
                {"value": "two_sided", "label": "Both tails"},
                {"value": "less", "label": "Lower tail"},
                {"value": "greater", "label": "Upper tail"},
            ),
            description="Which tail(s) of the distribution to clip.",
        ),
    ),
)
class TopBottomCoding(SDCMethod):
    """Clip ``self.column`` to ``[lower, upper]`` percentile bounds derived during pre-scan.

    Params:
        percentile: Total percentage of mass to clip, in ``(0, 100)``. With
            ``alternative="two_sided"`` (default) it is split equally between tails.
        alternative: ``"two_sided"`` clips both tails; ``"less"`` clips only the
            lower; ``"greater"`` clips only the upper.
    """

    name = "top_bottom_coding"
    requires_pre_scan = True

    @classmethod
    def validate_params(cls, params: dict) -> None:
        p = float(params.get("percentile", 5))
        if not 0.0 < p < 100.0:
            raise MethodParameterError("top_bottom_coding: percentile must lie in (0, 100)")
        alt = params.get("alternative", "two_sided")
        if alt not in ("two_sided", "less", "greater"):
            raise MethodParameterError(f"top_bottom_coding: unknown alternative {alt!r}")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        col = self.column
        if col is None:
            raise MethodParameterError("top_bottom_coding: column is required")
        if not df.schema[col].is_numeric():
            raise UnsupportedDtypeError(f"top_bottom_coding requires a numeric column; got {df.schema[col]}")

        p = float(self.params.get("percentile", 5)) / 100.0
        alt = self.params.get("alternative", "two_sided")
        non_null = df.get_column(col).drop_nulls().cast(pl.Float64)

        def q(prob: float) -> float:
            # Linear interpolation matches pandas default — keeps re-runs stable across stacks.
            return float(non_null.quantile(prob, interpolation="linear") or non_null.min() or 0.0)

        lower: float | None
        upper: float | None
        if alt == "two_sided":
            half = p / 2.0
            lower, upper = q(half), q(1.0 - half)
        elif alt == "less":
            lower, upper = q(p), None
        else:
            lower, upper = None, q(1.0 - p)

        return {
            "lower_bound": lower,
            "upper_bound": upper,
            "percentile": self.params.get("percentile", 5),
            "alternative": alt,
        }

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        col = self.column
        if col is None or col not in df.columns:
            return df
        return df.with_columns(
            pl.col(col)
            .cast(pl.Float64)
            .clip(
                lower_bound=params["lower_bound"],
                upper_bound=params["upper_bound"],
            )
        )
