from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence


def equivalence_class_sizes(df: pl.DataFrame, qis: Sequence[str]) -> pl.DataFrame:
    """Return a frame with ``qis + ['_nk_class_size']``, one row per distinct QI tuple.

    Centralises the group-by-and-count pattern so k-anonymity, l-diversity, and
    t-closeness share an identical grouping interpretation (including null handling).
    """
    return df.group_by(qis).agg(pl.len().alias("_nk_class_size"))


def per_record_class_size(df: pl.DataFrame, qis: Sequence[str]) -> pl.Series:
    """Return a Series aligned with ``df`` giving each record's equivalence-class size.

    Uses a window function so null-keyed records group together — a join on
    ``qis`` would drop them because SQL null-equality is unknown.
    """
    return df.select(pl.len().over(list(qis)).alias("_nk_class_size")).get_column("_nk_class_size")
