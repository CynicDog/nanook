"""The `DataContext` value object: column-role declarations shared by every method and metric.

A context names which columns are quasi-identifiers, sensitive, sampling weights,
or hierarchy roots. SDC methods and risk metrics dispatch on these roles rather
than on positional column lists, which keeps pipelines readable and lets the
engine carry a single context across many steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nanook.exceptions import ContextValidationError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

__all__ = ["DataContext"]


@dataclass(frozen=True, slots=True)
class DataContext:
    """Declares the role each column plays in disclosure risk and utility analysis.

    Attributes:
        quasi_identifiers: Columns that, in combination, can re-identify a record.
        sensitive: Columns whose values must not leak even when quasi-identifiers do.
        weights: Single column holding survey sampling weights, if any. Used by
            individual-risk metrics and weighted utility measures.
        hierarchy: Optional ordered list of column-name groups, outer-to-inner.
            Currently advisory; reserved for hierarchical risk in a later release.

    Examples:
        >>> ctx = DataContext(quasi_identifiers=["age", "zip"], sensitive=["diagnosis"])
        >>> ctx.quasi_identifiers
        ('age', 'zip')
    """

    quasi_identifiers: tuple[str, ...] = ()
    sensitive: tuple[str, ...] = ()
    weights: str | None = None
    hierarchy: tuple[tuple[str, ...], ...] = field(default_factory=tuple)

    def __init__(
        self,
        quasi_identifiers: Sequence[str] = (),
        sensitive: Sequence[str] = (),
        weights: str | None = None,
        hierarchy: Sequence[Sequence[str]] = (),
    ) -> None:
        object.__setattr__(self, "quasi_identifiers", tuple(quasi_identifiers))
        object.__setattr__(self, "sensitive", tuple(sensitive))
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "hierarchy", tuple(tuple(level) for level in hierarchy))

    def validate(self, df: pl.DataFrame) -> None:
        """Check that every declared column exists in `df` and roles do not overlap.

        Raises:
            ContextValidationError: A declared column is missing, or a column is
                tagged as both quasi-identifier and sensitive.
        """
        present = set(df.columns)
        for role, cols in (
            ("quasi_identifiers", self.quasi_identifiers),
            ("sensitive", self.sensitive),
        ):
            missing = [c for c in cols if c not in present]
            if missing:
                raise ContextValidationError(
                    f"context.{role} references columns absent from frame: {missing}"
                )
        if self.weights is not None and self.weights not in present:
            raise ContextValidationError(f"context.weights references absent column: {self.weights!r}")
        overlap = set(self.quasi_identifiers) & set(self.sensitive)
        if overlap:
            raise ContextValidationError(
                f"columns declared as both quasi-identifier and sensitive: {sorted(overlap)}"
            )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict round-tripping through `from_dict`."""
        return {
            "quasi_identifiers": list(self.quasi_identifiers),
            "sensitive": list(self.sensitive),
            "weights": self.weights,
            "hierarchy": [list(level) for level in self.hierarchy],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> DataContext:
        """Inverse of `to_dict`. Unknown keys are rejected to catch typos early.

        Raises:
            ContextValidationError: `payload` contains a key not on `DataContext`.
        """
        known = {"quasi_identifiers", "sensitive", "weights", "hierarchy"}
        extra = set(payload) - known
        if extra:
            raise ContextValidationError(f"unknown context keys: {sorted(extra)}")
        return cls(
            quasi_identifiers=payload.get("quasi_identifiers", ()),
            sensitive=payload.get("sensitive", ()),
            weights=payload.get("weights"),
            hierarchy=payload.get("hierarchy", ()),
        )
