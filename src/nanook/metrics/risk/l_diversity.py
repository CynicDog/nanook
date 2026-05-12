"""l-diversity: each equivalence class must hold at least ``l`` well-represented sensitive values.

Three modes from Machanavajjhala et al. (2007): ``distinct``, ``entropy``, ``recursive``.
Reference: pseudonymization-proposal/pseudo_code/risk_metrics/l_diversity.md.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl

from nanook._internal.validate import require_columns, require_nonempty
from nanook.exceptions import MethodParameterError
from nanook.report import LDiversityReport

if TYPE_CHECKING:
    from collections.abc import Sequence

    from nanook._typing import LDiversityMode

__all__ = ["l_diversity"]


def l_diversity(
    df: pl.DataFrame,
    *,
    qis: Sequence[str],
    sensitive: str,
    l: int = 3,
    mode: LDiversityMode = "distinct",
    c: float = 1.0,
) -> LDiversityReport:
    """Evaluate l-diversity on ``sensitive`` over equivalence classes formed by ``qis``.

    Args:
        df: Microdata frame.
        qis: Quasi-identifier columns forming equivalence classes.
        sensitive: Single sensitive column whose in-class distribution is tested.
        l: Diversity threshold (``>= 2``).
        mode: One of ``"distinct"`` (count of distinct sensitive values per class),
            ``"entropy"`` (Shannon entropy ``>= log l``), or ``"recursive"``
            (``r_1 < c · (r_l + … + r_m)`` after sorting frequencies).
        c: Multiplier for the recursive (c, l) variant. Ignored otherwise.

    Returns:
        An `LDiversityReport` aggregating class-level pass/fail counts.

    Raises:
        MethodParameterError: ``l < 2`` or ``mode`` is unknown.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"zip": ["1","1","1","2","2","2"],
        ...                    "diag": ["A","A","B","A","B","C"]})
        >>> l_diversity(df, qis=["zip"], sensitive="diag", l=2).holds
        True
        >>> l_diversity(df, qis=["zip"], sensitive="diag", l=3).holds
        False
    """
    if l < 2:
        raise MethodParameterError("l_diversity: l must be >= 2")
    if mode not in ("distinct", "entropy", "recursive"):
        raise MethodParameterError(f"l_diversity: unknown mode {mode!r}")
    require_nonempty(qis, role="qis")
    require_columns(df, qis, role="qis")
    require_columns(df, [sensitive], role="sensitive")

    qis_t = tuple(qis)
    if df.height == 0:
        return LDiversityReport(
            l=l,
            qis=qis_t,
            sensitive=sensitive,
            mode=mode,
            violations=0,
            violation_rate=0.0,
            holds=True,
            c=c if mode == "recursive" else None,
        )

    grouped = df.group_by(qis_t).agg(pl.col(sensitive).alias("_nk_values"))
    classes = grouped.get_column("_nk_values").to_list()
    n_classes = len(classes)

    violations = 0
    for values in classes:
        if not _class_passes(values, l=l, mode=mode, c=c):
            violations += 1

    return LDiversityReport(
        l=l,
        qis=qis_t,
        sensitive=sensitive,
        mode=mode,
        violations=violations,
        violation_rate=violations / n_classes if n_classes else 0.0,
        holds=violations == 0,
        c=c if mode == "recursive" else None,
    )


def _class_passes(values: list, *, l: int, mode: LDiversityMode, c: float) -> bool:
    distinct = {}
    for v in values:
        if v is None:
            continue
        distinct[v] = distinct.get(v, 0) + 1

    if mode == "distinct":
        return len(distinct) >= l

    total = sum(distinct.values())
    if total == 0:
        return False

    if mode == "entropy":
        # Pass iff H_q >= log(l). 0 · log 0 := 0 by convention; loop skips zero counts naturally.
        entropy = -sum((cnt / total) * math.log(cnt / total) for cnt in distinct.values())
        return entropy >= math.log(l)

    # recursive (c, l)
    freqs = sorted(distinct.values(), reverse=True)
    if len(freqs) < l:
        return False
    r1 = freqs[0]
    tail = sum(freqs[l - 1 :])
    return r1 < c * tail
