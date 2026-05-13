"""Smoke + invariant tests for the 9 perturbative methods.

We avoid asserting exact numerical equivalence (RNG-dependent) and instead test
the invariants documented in each method: dtype preservation, value-domain
sanity, reproducibility under fixed seed, and parameter validation.
"""

from __future__ import annotations

import polars as pl
import pytest

from nanook.context import DataContext
from nanook.core._registry import get_method
from nanook.exceptions import MethodParameterError, UnsupportedDtypeError


def _fit(name, *, column=None, params=None, df=None, ctx=None):
    cls = get_method(name)
    instance = cls(column=column, params=params or {})
    fitted = instance.pre_scan(df, ctx or DataContext()) if cls.requires_pre_scan else {}
    return instance, fitted


def test_noise_addition_changes_values_but_preserves_dtype():
    df = pl.DataFrame({"x": [float(i) for i in range(50)]})
    m, fitted = _fit("noise_addition", column="x", params={"intensity": 0.5, "seed": 1}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert out.schema["x"] == pl.Float64
    assert not out.equals(df)


def test_noise_addition_negative_intensity_rejected():
    with pytest.raises(MethodParameterError):
        get_method("noise_addition").validate_params({"intensity": -0.1})


def test_multiplicative_noise_preserves_zero():
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 0.0]})
    m, fitted = _fit("multiplicative_noise", column="x", params={"sigma_log": 0.2, "seed": 7}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert out.get_column("x")[0] == 0.0
    assert out.get_column("x")[3] == 0.0


def test_multiplicative_noise_preserves_first_two_moments():
    # Höhne 2004 step 3 guarantees mean and std of the non-zero subset match
    # the input exactly (up to float roundoff).
    df = pl.DataFrame({"x": [float(i + 1) for i in range(5000)]})  # strictly positive, no zeros
    m, fitted = _fit("multiplicative_noise", column="x", params={"sigma_log": 0.2, "seed": 0}, df=df)
    out = m.apply(df, DataContext(), fitted)
    x_in = df.get_column("x").to_numpy()
    x_out = out.get_column("x").to_numpy()
    assert abs(x_in.mean() - x_out.mean()) < 1e-9
    assert abs(x_in.std(ddof=0) - x_out.std(ddof=0)) < 1e-9
    # And the output is actually perturbed (not identity).
    assert not (x_in == x_out).all()


def test_multiplicative_noise_never_flips_sign_on_positive_input():
    # The pre-fix Gaussian factor `1 + N(0, sigma)` flipped signs at sigma=0.5
    # roughly 2% of the time. The canonical log-normal multiplier is positive
    # almost surely, so every cell in a strictly-positive column must stay
    # strictly positive.
    df = pl.DataFrame({"x": [1.0] * 10_000})
    m, fitted = _fit("multiplicative_noise", column="x", params={"sigma_log": 0.5, "seed": 0}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert (out.get_column("x") > 0.0).all()


def test_rounding_snaps_to_base():
    df = pl.DataFrame({"x": [0.3, 0.7, 1.2, 1.8]})
    m, fitted = _fit("rounding", column="x", params={"base": 1.0}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert out.get_column("x").to_list() == [0.0, 1.0, 1.0, 2.0]


def test_rounding_with_jitter_stays_within_bin():
    df = pl.DataFrame({"x": [12.0] * 50})  # snaps to centre 10, jitter window [-5, +5] → [5, 15]
    m, fitted = _fit(
        "rounding", column="x", params={"base": 10.0, "random_within_bin": True, "seed": 3}, df=df
    )
    out = m.apply(df, DataContext(), fitted)
    for v in out.get_column("x").to_list():
        assert 5.0 <= v <= 15.0


def test_rank_swapping_preserves_value_multiset():
    df = pl.DataFrame({"x": [float(i) for i in range(20)]})
    m, fitted = _fit("rank_swapping", column="x", params={"window_pct": 0.2, "seed": 11}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert sorted(out.get_column("x").to_list()) == sorted(df.get_column("x").to_list())


def test_data_swapping_preserves_value_multiset():
    df = pl.DataFrame({"x": list(range(20))})
    m, _ = _fit("data_swapping", column="x", params={"fraction": 1.0, "seed": 2}, df=df)
    out = m.apply(df, DataContext(), {})
    assert sorted(out.get_column("x").to_list()) == sorted(df.get_column("x").to_list())


def test_microaggregation_produces_at_least_k_duplicates():
    df = pl.DataFrame({"x": [float(i) for i in range(20)]})
    m, fitted = _fit("microaggregation", column="x", params={"k": 5}, df=df)
    out = m.apply(df, DataContext(), fitted)
    counts = out.get_column("x").value_counts()
    assert counts.get_column("count").min() >= 5


def test_microaggregation_second_seed_is_farthest_from_first():
    # Hand-crafted 2-D matrix where the canonical MDAV (second seed =
    # farthest from r) and the pre-fix variant (second seed = farthest from
    # the pre-removal centroid) disagree on which cluster a 'swing' point Q
    # joins. With raw 2-D coordinates:
    #   A = 3×(30, 0)         — pre-removal far_idx (extreme right)
    #   B = 2×(-10, 0)        — farthest from A in remaining
    #   C = 2×(0, 15)         — farthest from centroid in remaining
    #   Q = 1×(-7, 7)         — closer to B than to C
    #   D = 12×(5, -7.5)      — balances centroid; never extremal
    # Canonical second cluster = {B0, B1, Q}. Pre-fix second cluster =
    # {C0, C1, Q}. So Q's cluster id under canonical equals B's; under
    # pre-fix it equals C's. We test on `_mdav_cluster` directly so the
    # outer standardisation layer doesn't distort the geometry.
    import numpy as np

    from nanook.core.perturbative.microaggregation import _mdav_cluster

    x = np.array(
        [
            [30.0, 0.0],
            [30.0, 0.0],
            [30.0, 0.0],
            [-10.0, 0.0],
            [-10.0, 0.0],
            [0.0, 15.0],
            [0.0, 15.0],
            [-7.0, 7.0],
            *([[5.0, -7.5]] * 12),
        ]
    )
    assignment = _mdav_cluster(x, k=3)
    # Q (index 7) must share a cluster with B (indices 3, 4) under canonical
    # MDAV. Under the pre-fix variant Q would share with C (indices 5, 6).
    assert assignment[7] == assignment[3] == assignment[4]
    assert assignment[7] != assignment[5]
    assert assignment[7] != assignment[6]


def test_resampling_preserves_rank_order_approximately():
    df = pl.DataFrame({"x": [float(i) for i in range(30)]})
    m, fitted = _fit("resampling", column="x", params={"b": 30, "seed": 5}, df=df)
    out = m.apply(df, DataContext(), fitted)
    # Sorted-rank should remain monotone non-decreasing.
    sorted_out = sorted(out.get_column("x").to_list())
    assert (
        sorted_out
        == out.get_column("x").gather(pl.Series("rank", df.get_column("x").arg_sort().to_list())).to_list()
    )


def test_pram_eventually_keeps_some_records():
    df = pl.DataFrame({"x": ["A"] * 30 + ["B"] * 30 + ["C"] * 30})
    m, fitted = _fit("pram", column="x", params={"retention": 0.8, "seed": 13}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert set(out.get_column("x").to_list()) <= {"A", "B", "C"}


def test_pram_retention_one_is_identity():
    df = pl.DataFrame({"x": ["A", "B", "C"]})
    m, fitted = _fit("pram", column="x", params={"retention": 1.0, "seed": 1}, df=df)
    out = m.apply(df, DataContext(), fitted)
    assert out.equals(df)


def test_massc_requires_quasi_identifiers():
    df = pl.DataFrame({"q": ["x"] * 10, "s": ["A"] * 10})
    with pytest.raises(MethodParameterError):
        get_method("massc")(column=None, params={"k": 2, "f_sub": 0.8}).pre_scan(df, DataContext())


def _massc_frame(n: int = 60, *, seed: int = 0) -> pl.DataFrame:
    rng = __import__("numpy").random.default_rng(seed)
    return pl.DataFrame(
        {
            "age": rng.integers(20, 70, n).tolist(),
            "zip": rng.integers(1, 6, n).astype(str).tolist(),
            "sex": rng.choice(["M", "F"], n).tolist(),
            "diagnosis": rng.choice(["A", "B", "C"], n).tolist(),
        }
    )


def test_massc_output_is_k_anonymous_on_qis():
    df = _massc_frame(n=60, seed=0)
    ctx = DataContext(quasi_identifiers=["age", "zip", "sex"])
    m, fitted = _fit("massc", column=None, params={"k": 5, "f_sub": 1.0, "seed": 0}, df=df, ctx=ctx)
    out = m.apply(df, ctx, fitted)
    sizes = out.group_by(["age", "zip", "sex"]).len().get_column("len").to_list()
    assert min(sizes) >= 5


def test_massc_subsample_size():
    df = _massc_frame(n=60, seed=1)
    ctx = DataContext(quasi_identifiers=["age", "zip"])
    m, fitted = _fit("massc", column=None, params={"k": 3, "f_sub": 0.75, "seed": 1}, df=df, ctx=ctx)
    out = m.apply(df, ctx, fitted)
    assert out.height == 45  # floor(0.75 * 60)


def test_massc_weights_sum_to_population_total_per_calibration_cell():
    df = _massc_frame(n=60, seed=2)
    ctx = DataContext(quasi_identifiers=["age", "zip", "sex"])
    m, fitted = _fit(
        "massc",
        column=None,
        params={"k": 3, "f_sub": 0.8, "calibration_vars": ["sex"], "seed": 2},
        df=df,
        ctx=ctx,
    )
    out = m.apply(df, ctx, fitted)
    weights = out.get_column("weight").to_numpy()
    sex = out.get_column("sex").to_numpy()
    in_sex = df.get_column("sex").to_numpy()
    for v in set(in_sex):
        target = int((in_sex == v).sum())
        weighted_total = float(weights[sex == v].sum())
        assert abs(weighted_total - target) < 1e-6


def test_massc_substitution_breaks_qi_linkage():
    # On a seeded run, the substitution should overwrite the QI tuple for a
    # nontrivial share of records — they shouldn't all reproduce the original
    # tuple at the same row position.
    df = _massc_frame(n=60, seed=3)
    ctx = DataContext(quasi_identifiers=["age", "zip", "sex"])
    m, fitted = _fit("massc", column=None, params={"k": 3, "f_sub": 1.0, "seed": 3}, df=df, ctx=ctx)
    out = m.apply(df, ctx, fitted)
    changed = 0
    for c in ["age", "zip", "sex"]:
        changed += int((df.get_column(c).to_numpy() != out.get_column(c).to_numpy()).sum())
    # Across 60 rows × 3 QI columns = 180 cells, expect many to have changed.
    assert changed > 30


def test_unsupported_dtype_on_noise():
    df = pl.DataFrame({"x": ["a", "b"]})
    with pytest.raises(UnsupportedDtypeError):
        _fit("noise_addition", column="x", df=df)
