"""IL1s — Yancey-Winkler σ-scaled distance between original and protected microdata.

Complements λ: where λ is range-normalised and tail-insensitive, IL1s normalises
by the column standard deviation and reflects per-record deviation magnitude.
Reference: pseudonymization-proposal/pseudo_code/information_loss_metrics/il1s.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nanook._internal.validate import require_columns
from nanook.exceptions import MethodParameterError
from nanook.report import IL1sReport

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["il1s"]


def il1s(
    original: pl.DataFrame,
    protected: pl.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> IL1sReport:
    """Compute IL1s_k per non-constant column and the file-level mean.

    Args:
        original: Pre-SDC microdata.
        protected: Post-SDC microdata, row-aligned with ``original``.
        columns: Subset of continuous columns. Defaults to numeric columns
            common to both frames.

    Returns:
        An `IL1sReport`. Constant columns (``σ_k = 0``) are excluded — IL1s is
        undefined when the original column has no variation.

    Raises:
        MethodParameterError: frames differ in row count.

    Examples:
        >>> import polars as pl
        >>> orig = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        >>> prot = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
        >>> il1s(orig, prot).scalar
        0.0
    """
    if original.height != protected.height:
        raise MethodParameterError(f"il1s: frame heights differ ({original.height} vs {protected.height})")
    cols = _select_numeric_columns(original, protected, columns)
    require_columns(original, cols, role="il1s columns")
    require_columns(protected, cols, role="il1s columns")

    per: dict[str, float] = {}
    for col in cols:
        x = original.get_column(col).cast(pl.Float64)
        z = protected.get_column(col).cast(pl.Float64)
        sd = x.std()
        if sd is None or sd == 0.0:
            continue
        per[col] = float((x - z).abs().mean() or 0.0) / float(sd)

    scalar = sum(per.values()) / len(per) if per else 0.0
    return IL1sReport(scalar=scalar, per_column=per)


def _select_numeric_columns(a: pl.DataFrame, b: pl.DataFrame, requested: Sequence[str] | None) -> list[str]:
    if requested is not None:
        return list(requested)
    numeric = {c for c, dt in a.schema.items() if dt.is_numeric()}
    return [c for c in b.columns if c in numeric]
