"""Microaggregation: group records into clusters of size ``>= k`` and replace each cell with the cluster mean.

Implements the MDAV (Maximum Distance to Average Vector) heuristic in numpy.
For each step a centroid is computed, the farthest record from it forms a
cluster of its ``k`` nearest neighbours, and the procedure repeats on the
remaining records until fewer than ``2·k`` are left. Mean-replacement
guarantees k-anonymity on the microaggregated columns by construction.

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/microaggregation.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nanook.core._base import SDCMethod
from nanook.core._schema import ParamSchema, schema
from nanook.exceptions import MethodParameterError, UnsupportedDtypeError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["Microaggregation"]


@schema(
    display_name="Microaggregation",
    category="Perturbative",
    applicable_dtypes=("NUMERIC",),
    description=(
        "Cluster records into groups of at least k via MDAV (Maximum Distance "
        "to Average Vector) and replace each cell with its cluster mean. "
        "Mean-replacement guarantees k-anonymity on the microaggregated columns."
    ),
    params=(
        ParamSchema(
            name="k",
            display_name="Cluster Size (k)",
            param_type="INT",
            default=5,
            required=True,
            description="Minimum cluster size (>= 2).",
        ),
        ParamSchema(
            name="columns",
            display_name="Additional Columns",
            param_type="LIST",
            default=None,
            required=False,
            description=(
                "Optional explicit numeric column list overriding the rule's "
                "single column. Used for multivariate MDAV."
            ),
        ),
    ),
)
class Microaggregation(SDCMethod):
    """Replace each value with the mean of its k-nearest-neighbour cluster (MDAV).

    ``self.column`` may be a single column or a comma-separated list of numeric
    columns; multi-column microaggregation is the multivariate MDAV variant
    (clusters defined by Euclidean distance after per-column standardisation).

    Params:
        k: Minimum cluster size (``>= 2``).
        columns: Optional explicit list overriding ``self.column``.
    """

    name = "microaggregation"
    requires_pre_scan = True

    @classmethod
    def validate_params(cls, params: dict) -> None:
        if int(params.get("k", 5)) < 2:
            raise MethodParameterError("microaggregation: k must be >= 2")

    def _target_columns(self, df: pl.DataFrame) -> list[str]:  # noqa: ARG002
        cols = self.params.get("columns")
        if cols:
            return list(cols)
        if self.column is None:
            raise MethodParameterError("microaggregation: either `columns` param or self.column required")
        return [c.strip() for c in str(self.column).split(",") if c.strip()]

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:  # noqa: ARG002
        cols = self._target_columns(df)
        for c in cols:
            if not df.schema[c].is_numeric():
                raise UnsupportedDtypeError(
                    f"microaggregation requires numeric columns; got {df.schema[c]} for {c}"
                )
        k = int(self.params.get("k", 5))
        x = df.select(cols).cast(pl.Float64).to_numpy()
        # Standardise so each column contributes equally to the Euclidean metric.
        sds = x.std(axis=0)
        sds[sds == 0.0] = 1.0
        xs = (x - x.mean(axis=0)) / sds

        assignment = _mdav_cluster(xs, k)
        cluster_means = _means_per_cluster(x, assignment)
        return {
            "columns": cols,
            "k": k,
            "assignment": assignment.tolist(),
            "cluster_means": [m.tolist() for m in cluster_means],
        }

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        cols = params["columns"]
        assignment = np.asarray(params["assignment"])
        means = np.asarray(params["cluster_means"])

        out = df
        for j, c in enumerate(cols):
            values = means[assignment][:, j]
            out = out.with_columns(pl.Series(c, values))
        return out


def _mdav_cluster(x: np.ndarray, k: int) -> np.ndarray:
    """Assign every row in ``x`` to a cluster index using the MDAV heuristic."""
    n = x.shape[0]
    assignment = np.full(n, -1, dtype=np.int64)
    remaining = np.arange(n)
    cluster_id = 0

    # Step 2: while |U| >= 3k, form two clusters per pass — one seeded from
    # the centroid, and one seeded from the first seed (farthest-from-r).
    while remaining.size >= 3 * k:
        r_idx = _arg_max_distance(x, remaining, x[remaining].mean(axis=0))
        remaining, cluster_id = _form_cluster_at(x, remaining, k, r_idx, assignment, cluster_id)

        # Canonical MDAV (Domingo-Ferrer & Mateo-Sanz 2002, step 2): the second
        # cluster is seeded from the point farthest from `x_r`, not from the
        # pre-removal centroid.
        s_idx = _arg_max_distance(x, remaining, x[r_idx])
        remaining, cluster_id = _form_cluster_at(x, remaining, k, s_idx, assignment, cluster_id)

    # Step 3: while |U| >= 2k, form only the centroid-extreme cluster.
    while remaining.size >= 2 * k:
        r_idx = _arg_max_distance(x, remaining, x[remaining].mean(axis=0))
        remaining, cluster_id = _form_cluster_at(x, remaining, k, r_idx, assignment, cluster_id)

    # Step 4: residual (|U| < 2k but >= 0) becomes one final cluster.
    if remaining.size:
        assignment[remaining] = cluster_id
    return assignment


def _arg_max_distance(x: np.ndarray, remaining: np.ndarray, anchor: np.ndarray) -> int:
    """Return the index of the remaining row farthest (Euclidean) from ``anchor``."""
    return int(remaining[int(np.argmax(np.linalg.norm(x[remaining] - anchor, axis=1)))])


def _form_cluster_at(
    x: np.ndarray,
    remaining: np.ndarray,
    k: int,
    seed_idx: int,
    assignment: np.ndarray,
    cluster_id: int,
) -> tuple[np.ndarray, int]:
    """Form a cluster from the ``k`` remaining rows nearest to ``x[seed_idx]``."""
    d_seed = np.linalg.norm(x[remaining] - x[seed_idx], axis=1)
    nearest = remaining[np.argsort(d_seed)[:k]]
    assignment[nearest] = cluster_id
    removed_set = set(nearest.tolist())
    return np.array([r for r in remaining if r not in removed_set], dtype=np.int64), cluster_id + 1


def _means_per_cluster(x: np.ndarray, assignment: np.ndarray) -> list[np.ndarray]:
    cluster_ids = sorted(set(assignment.tolist()))
    return [x[assignment == c].mean(axis=0) for c in cluster_ids]
