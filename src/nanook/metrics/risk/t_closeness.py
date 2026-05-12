"""t-closeness: per-class sensitive distribution must be within EMD ``t`` of the population.

Reference: Li, Li, Venkatasubramanian (2007). See also
``pseudonymization-proposal/pseudo_code/risk_metrics/t_closeness.md``.

The ordinal/continuous path uses the closed-form CDF-difference Wasserstein-1
formula (pure numpy). The nominal path solves the transportation LP via
scipy.optimize.linprog — pulled in only on demand from the optional
``nominal-emd`` extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

from nanook._internal.emd import emd_nominal, emd_ordinal
from nanook._internal.validate import require_columns, require_nonempty
from nanook.exceptions import MethodParameterError, UnsupportedDtypeError
from nanook.report import TClosenessReport

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["t_closeness"]

_NUMERIC = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)


def t_closeness(
    df: pl.DataFrame,
    *,
    qis: Sequence[str],
    sensitive: str,
    t: float = 0.2,
    support: Literal["ordinal", "nominal"] = "ordinal",
    cost: np.ndarray | None = None,
) -> TClosenessReport:
    """Compute per-class EMD against the population marginal and flag classes above ``t``.

    Args:
        df: Microdata frame.
        qis: Quasi-identifier columns forming equivalence classes.
        sensitive: Sensitive column to test.
        t: Closeness threshold in ``(0, 1]``.
        support: ``"ordinal"`` uses the numeric value space and the closed-form CDF
            formula; ``"nominal"`` solves the transportation LP using ``cost``.
        cost: Pairwise non-negative cost matrix aligned to ``np.sort(unique(sensitive))``.
            Required when ``support == "nominal"``.

    Returns:
        A `TClosenessReport` with the maximum EMD and a violation count.

    Raises:
        MethodParameterError: ``t`` outside ``(0, 1]`` or nominal support without ``cost``.
        UnsupportedDtypeError: ordinal support requested on a non-numeric column.

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({"zip": ["1","1","1","2","2","2"], "x": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]})
        >>> r = t_closeness(df, qis=["zip"], sensitive="x", t=0.1)
        >>> r.holds, round(r.max_emd, 4)
        (False, 0.5)
    """
    if not 0.0 < t <= 1.0:
        raise MethodParameterError("t_closeness: t must lie in (0, 1]")
    if support not in ("ordinal", "nominal"):
        raise MethodParameterError(f"t_closeness: unknown support {support!r}")
    if support == "nominal" and cost is None:
        raise MethodParameterError("t_closeness: support='nominal' requires a cost matrix")
    require_nonempty(qis, role="qis")
    require_columns(df, qis, role="qis")
    require_columns(df, [sensitive], role="sensitive")

    qis_t = tuple(qis)
    if df.height == 0:
        return TClosenessReport(
            t=t,
            qis=qis_t,
            sensitive=sensitive,
            violations=0,
            max_emd=0.0,
            holds=True,
        )

    s_dtype = df.schema[sensitive]
    if support == "ordinal" and s_dtype not in _NUMERIC:
        raise UnsupportedDtypeError(
            f"t_closeness ordinal support requires a numeric sensitive column; got {s_dtype}"
        )

    # Population value space and marginal, used as the reference for every class.
    pop_values = df.get_column(sensitive).drop_nulls().unique().sort().to_numpy()
    if pop_values.size == 0:
        return TClosenessReport(t=t, qis=qis_t, sensitive=sensitive, violations=0, max_emd=0.0, holds=True)
    value_to_idx = {v: i for i, v in enumerate(pop_values.tolist())}
    pop = _empirical(df.get_column(sensitive), value_to_idx)

    grouped = df.group_by(qis_t).agg(pl.col(sensitive).alias("_nk_values"))

    max_emd = 0.0
    violations = 0
    for row in grouped.iter_rows(named=True):
        cls = _empirical_from_list(row["_nk_values"], value_to_idx)
        if support == "ordinal":
            emd = emd_ordinal(cls, pop, pop_values.astype(float))
        else:
            emd = emd_nominal(cls, pop, cost)  # type: ignore[arg-type]
        max_emd = max(max_emd, emd)
        if emd > t:
            violations += 1

    return TClosenessReport(
        t=t,
        qis=qis_t,
        sensitive=sensitive,
        violations=violations,
        max_emd=max_emd,
        holds=violations == 0,
    )


def _empirical(s: pl.Series, value_to_idx: dict) -> np.ndarray:
    counts = np.zeros(len(value_to_idx), dtype=float)
    for v in s.drop_nulls().to_list():
        counts[value_to_idx[v]] += 1.0
    total = counts.sum()
    return counts / total if total else counts


def _empirical_from_list(values: list, value_to_idx: dict) -> np.ndarray:
    counts = np.zeros(len(value_to_idx), dtype=float)
    for v in values:
        if v is None:
            continue
        idx = value_to_idx.get(v)
        if idx is not None:
            counts[idx] += 1.0
    total = counts.sum()
    return counts / total if total else counts
