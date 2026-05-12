"""Shared type aliases and protocols.

Centralising these here keeps method and metric signatures uniform and avoids
duplicate `Sequence[str]` / `Literal[...]` declarations across the tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

type ColumnName = str
type ColumnList = Sequence[ColumnName]

type LDiversityMode = Literal["distinct", "entropy", "recursive"]
type Alternative = Literal["two_sided", "less", "greater"]
type ReplacementStat = Literal["mean", "median", "mode"]
