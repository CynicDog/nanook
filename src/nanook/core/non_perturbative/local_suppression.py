"""Local Suppression: null out cells in quasi-identifier columns until every class meets k.

Greedy heuristic. A record is "k-anonymous" once at least ``target_k`` records
agree with it on every still-revealed quasi-identifier column. At each step we
suppress the cell that reduces the violation count the most per unit of cost.

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/non_perturbative/local_suppression.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from nanook.core._base import SDCMethod
from nanook.core._schema import ParamSchema, schema
from nanook.exceptions import MethodParameterError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["LocalSuppression"]


@schema(
    display_name="Local Suppression",
    category="Non-perturbative",
    applicable_dtypes=("ANY",),
    description=(
        "Null cells in the configured identifier columns until every record's "
        "masked equivalence class has at least ``target_k`` members. The rule "
        "ignores any per-step column — it operates on every column flagged as "
        "Identifier."
    ),
    params=(
        ParamSchema(
            name="target_k",
            display_name="Target k",
            param_type="INT",
            default=5,
            required=True,
            description="Anonymity threshold to reach (>= 2).",
        ),
        ParamSchema(
            name="cost_priority",
            display_name="Column Cost Priority",
            param_type="MAP",
            default=None,
            required=False,
            description=(
                "Optional column → cost mapping. Higher values discourage "
                "suppression on that column. Defaults to 1.0 where absent."
            ),
        ),
        ParamSchema(
            name="max_iterations",
            display_name="Max Iterations",
            param_type="INT",
            default=None,
            required=False,
            description="Safety cap on the suppression loop. Defaults to (rows × identifier-columns).",
        ),
    ),
)
class LocalSuppression(SDCMethod):
    """Suppress quasi-identifier cells until every record's masked class has size ``>= target_k``.

    Suppression treats nulled cells as wildcards: a record's class is the set of
    records agreeing on every column where the record itself is not suppressed.
    The pre-scan stage decides the suppressions; the apply stage materialises
    them with ``polars`` ``when/then/otherwise``. ``column`` is unused — local
    suppression operates on the whole quasi-identifier set declared by the context.

    Params:
        target_k: Anonymity threshold to reach (``>= 2``).
        cost_priority: Optional mapping ``column -> cost``. Higher values discourage
            suppression on that column. Columns absent from the mapping have cost ``1.0``.
        max_iterations: Safety cap on the suppression loop (default: total cells).
    """

    name = "local_suppression"
    requires_pre_scan = True

    @classmethod
    def validate_params(cls, params: dict) -> None:
        k = int(params.get("target_k", 5))
        if k < 2:
            raise MethodParameterError("local_suppression: target_k must be >= 2")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:
        qis = list(ctx.quasi_identifiers)
        if not qis:
            raise MethodParameterError("local_suppression: requires non-empty DataContext.quasi_identifiers")
        target_k = int(self.params.get("target_k", 5))
        cost_priority: dict[str, float] = self.params.get("cost_priority") or {}
        max_iter = int(self.params.get("max_iterations", df.height * len(qis)))

        rows: list[list] = [[row[c] for c in qis] for row in df.select(qis).iter_rows(named=True)]
        suppressed: list[list[bool]] = [[False] * len(qis) for _ in rows]

        suppressions: list[tuple[int, str]] = []
        for _ in range(max_iter):
            sizes = _all_class_sizes(rows, suppressed)
            violating = [i for i, s in enumerate(sizes) if s < target_k]
            if not violating:
                break

            best = _best_suppression(rows, suppressed, violating, sizes, target_k, qis, cost_priority)
            if best is None:
                break
            i, j = best
            suppressed[i][j] = True
            suppressions.append((i, qis[j]))

        return {
            "suppressions": suppressions,
            "target_k": target_k,
            "qis": qis,
            "cells_suppressed": len(suppressions),
        }

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        suppressions: list[tuple[int, str]] = params.get("suppressions", [])
        if not suppressions:
            return df

        by_col: dict[str, set[int]] = {}
        for idx, col in suppressions:
            by_col.setdefault(col, set()).add(idx)

        out = df
        for col, idxs in by_col.items():
            mask = pl.Series("_nk_mask", [i in idxs for i in range(df.height)])
            out = out.with_columns(pl.when(mask).then(None).otherwise(pl.col(col)).alias(col))
        return out


def _all_class_sizes(rows: list[list], suppressed: list[list[bool]]) -> list[int]:
    """For each row, count records matching on all of its still-revealed columns."""
    sizes = []
    for i, row in enumerate(rows):
        revealed = [j for j, sup in enumerate(suppressed[i]) if not sup]
        size = sum(1 for r in rows if all(r[j] == row[j] for j in revealed))
        sizes.append(size)
    return sizes


def _best_suppression(
    rows: list[list],
    suppressed: list[list[bool]],
    violating: list[int],
    sizes: list[int],
    target_k: int,
    qis: list[str],
    cost_priority: dict[str, float],
) -> tuple[int, int] | None:
    """Pick the (row, column-index) suppression with the largest violation-reduction per unit cost."""
    best_score = -1.0
    best: tuple[int, int] | None = None
    for i in violating:
        for j in range(len(qis)):
            if suppressed[i][j]:
                continue
            new_sizes = _sizes_after_hypothetical(rows, suppressed, i, j)
            new_violations = sum(1 for s in new_sizes if s < target_k)
            old_violations = sum(1 for s in sizes if s < target_k)
            delta = old_violations - new_violations
            cost = cost_priority.get(qis[j], 1.0)
            score = delta / cost
            if score > best_score:
                best_score = score
                best = (i, j)
    if best_score <= 0.0:
        return None
    return best


def _sizes_after_hypothetical(rows: list[list], suppressed: list[list[bool]], i: int, j: int) -> list[int]:
    """Recompute every row's class size assuming we additionally suppress ``rows[i][j]``."""
    new_sizes = []
    for ii, row in enumerate(rows):
        revealed = [k for k in range(len(row)) if not suppressed[ii][k] and not (ii == i and k == j)]
        size = sum(1 for r in rows if all(r[k] == row[k] for k in revealed))
        new_sizes.append(size)
    return new_sizes
