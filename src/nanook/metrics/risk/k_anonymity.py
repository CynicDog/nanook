"""k-anonymity: every quasi-identifier combination must appear in at least ``k`` records.

Reference: Samarati and Sweeney (1998); pseudonymization-proposal/pseudo_code/risk_metrics/k_anonymity.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nanook._internal.grouping import equivalence_class_sizes
from nanook._internal.validate import require_columns, require_nonempty
from nanook.exceptions import MethodParameterError
from nanook.report import KAnonymityReport

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["k_anonymity"]


def k_anonymity(df: pl.DataFrame, *, qis: Sequence[str], k: int = 5) -> KAnonymityReport:
    """Count records whose quasi-identifier equivalence class is smaller than ``k``.

    Args:
        df: Microdata frame.
        qis: Quasi-identifier columns whose tuple must repeat at least ``k`` times.
        k: Anonymity threshold (typical: 3 or 5). Must be ``>= 2``.

    Returns:
        A `KAnonymityReport` with violation counts, sample uniques, and a boolean holding flag.

    Raises:
        MethodParameterError: ``k < 2`` or ``qis`` is empty.
        ContextValidationError: ``qis`` references a column not in ``df``.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"age": [30, 30, 31, 31, 31], "zip": ["1", "1", "2", "2", "2"]})
        >>> r = k_anonymity(df, qis=["age", "zip"], k=2)
        >>> (r.violations, r.violation_rate, r.holds)
        (0, 0.0, True)
        >>> k_anonymity(df, qis=["age", "zip"], k=3).holds
        False
    """
    if k < 2:
        raise MethodParameterError("k_anonymity: k must be >= 2")
    require_nonempty(qis, role="qis")
    require_columns(df, qis, role="qis")

    qis_t = tuple(qis)
    if df.height == 0:
        return KAnonymityReport(
            k=k, qis=qis_t, violations=0, violation_rate=0.0, sample_uniques=0, holds=True
        )

    sizes = equivalence_class_sizes(df, qis_t).get_column("_nk_class_size")
    # Each class of size s contributes s records below the threshold when s < k.
    violations = int(sizes.filter(sizes < k).sum() or 0)
    sample_uniques = int(sizes.filter(sizes == 1).sum() or 0)
    n = df.height
    return KAnonymityReport(
        k=k,
        qis=qis_t,
        violations=violations,
        violation_rate=violations / n,
        sample_uniques=sample_uniques,
        holds=violations == 0,
    )
