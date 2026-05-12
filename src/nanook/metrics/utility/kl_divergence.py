"""KL divergence for categorical columns, with Laplace smoothing.

Asymmetric: ``D_KL(P || Q)`` measures information lost when treating the
protected marginal ``Q`` as an approximation of the original ``P``. Reference:
pseudonymization-proposal/pseudo_code/information_loss_metrics/kl_divergence.md.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from nanook._internal.validate import require_columns
from nanook.exceptions import MethodParameterError
from nanook.report import KLDivergenceReport

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["kl_divergence"]


def kl_divergence(
    original: pl.DataFrame,
    protected: pl.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    epsilon: float = 1e-6,
) -> KLDivergenceReport:
    """Compute per-column ``D_KL(P_k || Q_k)`` and the file-level mean across columns.

    Args:
        original: Pre-SDC microdata.
        protected: Post-SDC microdata, row-aligned with ``original``.
        columns: Subset of categorical columns. Defaults to columns whose dtype
            is String, Categorical, or Boolean in both frames.
        epsilon: Laplace smoothing applied to ``Q_k`` to avoid division by zero
            where ``P_k(v) > 0`` but ``Q_k(v) = 0``.

    Returns:
        A `KLDivergenceReport` with per-column divergences in nats.

    Raises:
        MethodParameterError: frames differ in row count or ``epsilon <= 0``.

    Examples:
        >>> import polars as pl
        >>> orig = pl.DataFrame({"c": ["a", "a", "b", "b"]})
        >>> prot = pl.DataFrame({"c": ["a", "a", "b", "b"]})
        >>> round(kl_divergence(orig, prot).scalar, 6)
        0.0
    """
    if original.height != protected.height:
        raise MethodParameterError(
            f"kl_divergence: frame heights differ ({original.height} vs {protected.height})"
        )
    if epsilon <= 0:
        raise MethodParameterError("kl_divergence: epsilon must be positive")
    cols = _select_categorical_columns(original, protected, columns)
    require_columns(original, cols, role="kl_divergence columns")
    require_columns(protected, cols, role="kl_divergence columns")

    n = original.height
    per: dict[str, float] = {}
    for col in cols:
        p_counts = _value_counts(original.get_column(col))
        q_counts = _value_counts(protected.get_column(col))
        support = set(p_counts) | set(q_counts)
        m = len(support) or 1
        divergence = 0.0
        for v in support:
            p = p_counts.get(v, 0) / n if n else 0.0
            q_raw = q_counts.get(v, 0) / n if n else 0.0
            q = (q_raw * n + epsilon) / (n + epsilon * m)
            if p > 0.0:
                divergence += p * math.log(p / q)
        per[col] = divergence

    scalar = sum(per.values()) / len(per) if per else 0.0
    return KLDivergenceReport(scalar=scalar, per_column=per, smoothing=epsilon)


def _value_counts(s: pl.Series) -> dict:
    return dict(s.value_counts().iter_rows())


def _select_categorical_columns(
    a: pl.DataFrame, b: pl.DataFrame, requested: Sequence[str] | None
) -> list[str]:
    if requested is not None:
        return list(requested)
    cat_dtypes = (pl.String, pl.Categorical, pl.Boolean, pl.Enum)
    cats = {c for c, dt in a.schema.items() if isinstance(dt, cat_dtypes)}
    return [c for c in b.columns if c in cats]
