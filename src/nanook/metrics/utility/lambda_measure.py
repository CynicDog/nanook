"""λ measure: range-normalised mean absolute deviation between original and protected microdata.

The IHSN handbook's primary scalar summary of utility loss. Range-normalisation
makes columns of different units directly comparable in the file-level mean.
Reference: pseudonymization-proposal/pseudo_code/information_loss_metrics/lambda_measure.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nanook._internal.validate import require_columns
from nanook.exceptions import MethodParameterError
from nanook.report import LambdaReport

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["lambda_measure"]


def lambda_measure(
    original: pl.DataFrame,
    protected: pl.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> LambdaReport:
    """Compute λ_k per column and the file-level mean λ across columns.

    Args:
        original: Pre-SDC microdata.
        protected: Post-SDC microdata, row-aligned with ``original``.
        columns: Subset of continuous columns to compare. Defaults to the
            intersection of numeric columns present in both frames.

    Returns:
        A `LambdaReport` with the per-column dict and the file-level scalar.

    Raises:
        MethodParameterError: frames differ in row count.

    Examples:
        >>> import polars as pl
        >>> orig = pl.DataFrame({"x": [0.0, 10.0]})
        >>> prot = pl.DataFrame({"x": [1.0, 9.0]})
        >>> round(lambda_measure(orig, prot).scalar, 2)
        0.1
    """
    if original.height != protected.height:
        raise MethodParameterError(
            f"lambda_measure: frame heights differ ({original.height} vs {protected.height})"
        )
    cols = _select_numeric_columns(original, protected, columns)
    require_columns(original, cols, role="lambda_measure columns")
    require_columns(protected, cols, role="lambda_measure columns")

    per: dict[str, float] = {}
    for col in cols:
        x = original.get_column(col).cast(pl.Float64)
        z = protected.get_column(col).cast(pl.Float64)
        x_min = x.min()
        x_max = x.max()
        if x_min is None or x_max is None:
            per[col] = 0.0
            continue
        rng = float(x_max) - float(x_min)
        if rng == 0.0:
            per[col] = 0.0
            continue
        per[col] = float((x - z).abs().mean() or 0.0) / rng

    scalar = sum(per.values()) / len(per) if per else 0.0
    return LambdaReport(scalar=scalar, per_column=per)


def _select_numeric_columns(a: pl.DataFrame, b: pl.DataFrame, requested: Sequence[str] | None) -> list[str]:
    if requested is not None:
        return list(requested)
    numeric = {c for c, dt in a.schema.items() if dt.is_numeric()}
    return [c for c in b.columns if c in numeric]
