from __future__ import annotations

from typing import TYPE_CHECKING

from nanook.exceptions import ContextValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl


def require_columns(df: pl.DataFrame, cols: Sequence[str], *, role: str) -> None:
    """Raise ContextValidationError if any column in ``cols`` is missing from ``df``."""
    present = set(df.columns)
    missing = [c for c in cols if c not in present]
    if missing:
        raise ContextValidationError(f"{role} references absent columns: {missing}")


def require_nonempty(cols: Sequence[str], *, role: str) -> None:
    """Raise ContextValidationError if ``cols`` is empty."""
    if not cols:
        raise ContextValidationError(f"{role} must be a non-empty column list")
