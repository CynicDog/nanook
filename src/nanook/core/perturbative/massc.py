"""MASSC: Micro Agglomeration, Substitution, Subsampling, Calibration (Singh, Yu & Dunteman 2003).

Four-step categorical protection on the quasi-identifier tuple:

1. **Micro-agglomeration** clusters records into groups of size ``>= k`` via
   Hamming-distance MDAV on the QI tuple, yielding k-anonymous groups.
2. **Substitution** replaces each record's QI tuple with that of a randomly
   chosen cluster-mate, breaking the deterministic original→masked link
   while preserving the within-cluster distribution.
3. **Subsampling** retains ``floor(f_sub * n)`` records without replacement,
   adding an extra inclusion-uncertainty layer.
4. **Calibration** rakes design weights so the weighted marginals on the
   calibration variables match the original population totals.

The output frame has ``floor(f_sub * n)`` rows and one extra column carrying
the calibration weights. Utility metrics that assume equal-sized original
and protected frames (lambda, IL1s) are not meaningful against this output;
analysts should consume the calibration weights.

Reference: pseudonymization-proposal/pseudo_code/sdc_methods/perturbative/massc.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from nanook._internal import categorical_mdav, raking
from nanook._internal.rng import generator
from nanook.core._base import SDCMethod
from nanook.core._schema import ParamSchema, schema
from nanook.exceptions import MethodParameterError

if TYPE_CHECKING:
    from nanook.context import DataContext

__all__ = ["MASSC"]


@schema(
    display_name="MASSC",
    category="Perturbative",
    applicable_dtypes=("ANY",),
    requires_quasi_identifiers=True,
    is_pipeline_scope=True,
    description=(
        "Four-step (Micro-Agglomeration, Substitution, Subsampling, Calibration) "
        "categorical protection on the configured identifier columns. Drops rows "
        "and appends a calibration-weight column; utility metrics that assume "
        "equal-sized frames are not meaningful against this output."
    ),
    params=(
        ParamSchema(
            name="k",
            display_name="Cluster Size (k)",
            param_type="INT",
            default=5,
            required=True,
            description="Minimum cluster size for micro-agglomeration (>= 2).",
        ),
        ParamSchema(
            name="f_sub",
            display_name="Subsample Fraction",
            param_type="FLOAT",
            default=0.8,
            required=True,
            description="Fraction of rows to retain after subsampling, in (0, 1].",
        ),
        ParamSchema(
            name="calibration_vars",
            display_name="Calibration Variables",
            param_type="LIST",
            default=None,
            required=False,
            description=(
                "Columns to calibrate the weighted marginals on. Defaults to the identifier list when empty."
            ),
        ),
        ParamSchema(
            name="seed",
            display_name="Random Seed",
            param_type="INT",
            default=None,
            required=False,
            description="Optional integer seed for reproducibility.",
        ),
    ),
)
class MASSC(SDCMethod):
    """Run the canonical four-step MASSC on the context's quasi-identifiers.

    Params:
        k: Minimum cluster size for micro-agglomeration (``>= 2``).
        f_sub: Subsample fraction in ``(0, 1]``.
        calibration_vars: Columns to calibrate weighted marginals on. Defaults
            to the QI list when ``None``. Calibration columns are sourced
            from the substituted frame, so non-QI calibration variables must
            survive substitution unchanged.
        seed: Optional integer seed.
    """

    name = "massc"
    requires_pre_scan = True

    @classmethod
    def validate_params(cls, params: dict) -> None:
        if int(params.get("k", 5)) < 2:
            raise MethodParameterError("massc: k must be >= 2")
        f_sub = float(params.get("f_sub", 0.8))
        if not 0.0 < f_sub <= 1.0:
            raise MethodParameterError("massc: f_sub must lie in (0, 1]")

    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:
        qis = list(ctx.quasi_identifiers)
        if not qis:
            raise MethodParameterError("massc: requires non-empty DataContext.quasi_identifiers")
        k = int(self.params.get("k", 5))
        f_sub = float(self.params.get("f_sub", 0.8))
        cal_vars = list(self.params.get("calibration_vars") or qis)
        rng = generator(self.params.get("seed"))

        n = df.height
        if n < 2 * k:
            raise MethodParameterError(f"massc: frame must have >= 2*k = {2 * k} rows; got {n}")

        # Step 1: micro-agglomeration. Cluster by Hamming MDAV on QI codes,
        # then collapse each cluster's QI tuple to the per-column mode `q_g`
        # — this is what gives the output its k-anonymity on the QIs.
        codes = _encode_qi_codes(df, qis)
        assignment = categorical_mdav.assign(codes, k=k)

        qi_representative: dict[str, np.ndarray] = {}
        for c in qis:
            values = df.get_column(c).to_numpy()
            collapsed = values.copy()
            for cluster_id in np.unique(assignment):
                members = np.where(assignment == cluster_id)[0]
                vals, counts = np.unique(values[members], return_counts=True)
                collapsed[members] = vals[int(np.argmax(counts))]
            qi_representative[c] = collapsed

        # Step 2: within-cluster row permutation on the non-QI columns.
        # After step 1, the QI tuple is constant within each cluster, so the
        # pseudocode's "x_{i,K} := x_{j,K}" is a no-op on QIs. We permute the
        # non-QI columns so each surviving record is paired with a random
        # cluster-mate's other-column values — preserving the within-cluster
        # joint distribution while breaking the deterministic record link.
        non_qi_cols = [c for c in df.columns if c not in qis]
        row_permutation = np.arange(n, dtype=np.int64)
        if non_qi_cols:
            for cluster_id in np.unique(assignment):
                members = np.where(assignment == cluster_id)[0]
                row_permutation[members] = rng.permutation(members)

        # Step 3: subsample n_sub = floor(f_sub * n) without replacement.
        n_sub = max(1, int(np.floor(f_sub * n)))
        subsample_idx = np.sort(rng.choice(n, size=n_sub, replace=False))

        # Step 4: rake design weights against the population totals computed
        # from the original frame.
        targets = []
        for c in cal_vars:
            counts = df.get_column(c).value_counts(sort=False)
            value_col, count_col = counts.columns[0], counts.columns[1]
            targets.append({row[value_col]: row[count_col] for row in counts.iter_rows(named=True)})

        post_sub_cols = {}
        for c in cal_vars:
            if c in qis:
                post_sub_cols[c] = qi_representative[c][subsample_idx]
            else:
                source_values = df.get_column(c).to_numpy()
                post_sub_cols[c] = source_values[row_permutation][subsample_idx]
        post_sub_frame = pl.DataFrame(post_sub_cols)

        design_weight = np.full(n_sub, n / n_sub, dtype=np.float64)
        weights = raking.rake(post_sub_frame, design_weight, targets, cal_vars)

        return {
            "qis": qis,
            "non_qi_cols": non_qi_cols,
            "qi_representative": {c: arr.tolist() for c, arr in qi_representative.items()},
            "row_permutation": row_permutation.tolist(),
            "subsample_idx": subsample_idx.tolist(),
            "weights": weights.tolist(),
            "weights_col": ctx.weights or "weight",
        }

    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:  # noqa: ARG002
        qis: list[str] = params["qis"]
        non_qi_cols: list[str] = params["non_qi_cols"]
        qi_representative: dict[str, list] = params["qi_representative"]
        row_permutation = np.asarray(params["row_permutation"], dtype=np.int64)
        sub_idx = np.asarray(params["subsample_idx"], dtype=np.int64)
        weights = params["weights"]
        weights_col = params["weights_col"]

        # Round-tripping through numpy loses dtype fidelity: `to_numpy()` on
        # Categorical / String / null-bearing narrow-int columns returns object
        # (or NaN-padded float) arrays that Polars can't cast back to the
        # original schema. Use Polars-native gather for non-QI columns and a
        # String intermediate for Categorical QI reconstruction so all dtypes
        # round-trip cleanly.
        perm_list = row_permutation.tolist()
        substituted = df
        for c in qis:
            substituted = substituted.with_columns(
                _reconstruct_qi_series(c, qi_representative[c], df.schema[c])
            )
        for c in non_qi_cols:
            substituted = substituted.with_columns(df.get_column(c).gather(perm_list).alias(c))

        out = substituted[sub_idx.tolist()]
        return out.with_columns(pl.Series(weights_col, weights, dtype=pl.Float64))


def _reconstruct_qi_series(name: str, values: list, target_dtype: pl.DataType) -> pl.Series:
    """Build a Polars Series from a Python list, preserving ``target_dtype``.

    `pl.Series(name, values, dtype=Categorical)` routes through Polars' Object
    dtype, which cannot cast to Categorical. Reconstruct via a String
    intermediate so the cluster-mode round-trip survives Categorical QIs.
    """
    if isinstance(target_dtype, pl.Categorical):
        return pl.Series(name, values, dtype=pl.String).cast(target_dtype, strict=False)
    return pl.Series(name, values, dtype=target_dtype, strict=False)


def _encode_qi_codes(df: pl.DataFrame, qis: list[str]) -> np.ndarray:
    """Factorise each QI column to integer codes; stack into an (n, p) matrix."""
    n = df.height
    codes = np.empty((n, len(qis)), dtype=np.int64)
    for j, c in enumerate(qis):
        values = df.get_column(c).to_numpy()
        # `np.unique(..., return_inverse=True)` is stable across dtypes;
        # numeric, string, and boolean QIs all encode cleanly.
        _, inverse = np.unique(values, return_inverse=True)
        codes[:, j] = inverse
    return codes
