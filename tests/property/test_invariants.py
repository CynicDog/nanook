"""Property-based invariants for the perturbative methods (T4 in REVIEW.md).

Each property tests a documented invariant from the pseudocode against random
inputs — multiset preservation, identity at trivial parameter values,
monotonicity in noise intensity, and k-anonymity by construction for
microaggregation.
"""

from __future__ import annotations

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

import nanook as nk
from nanook.context import DataContext
from nanook.core._registry import get_method


def _fit_apply(name, *, column, params, df, ctx):
    cls = get_method(name)
    instance = cls(column=column, params=params)
    fitted = instance.pre_scan(df, ctx) if cls.requires_pre_scan else {}
    return instance.apply(df, ctx, fitted)


_float_values = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=4,
    max_size=60,
)


@given(values=_float_values, seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50, deadline=None)
def test_data_swapping_preserves_value_multiset(values, seed):
    df = pl.DataFrame({"x": values})
    out = _fit_apply(
        "data_swapping",
        column="x",
        params={"fraction": 1.0, "seed": seed},
        df=df,
        ctx=DataContext(),
    )
    assert sorted(out.get_column("x").to_list()) == sorted(values)


@given(values=_float_values, seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50, deadline=None)
def test_rank_swapping_preserves_value_multiset(values, seed):
    df = pl.DataFrame({"x": values})
    out = _fit_apply(
        "rank_swapping",
        column="x",
        params={"window_pct": 0.25, "seed": seed},
        df=df,
        ctx=DataContext(),
    )
    assert sorted(out.get_column("x").to_list()) == sorted(values)


@given(values=_float_values, seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=50, deadline=None)
def test_sampling_fraction_one_is_identity(values, seed):
    df = pl.DataFrame({"x": values})
    out = _fit_apply(
        "sampling",
        column=None,
        params={"fraction": 1.0, "seed": seed},
        df=df,
        ctx=DataContext(),
    )
    assert out.get_column("x").to_list() == values


@given(
    values=st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=30,
        max_size=60,
    ).filter(lambda xs: len(set(xs)) > 1),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=20, deadline=None)
def test_noise_addition_lambda_monotonic_in_intensity(values, seed):
    df = pl.DataFrame({"x": values})
    lambdas = []
    for intensity in (0.1, 1.0):
        out = _fit_apply(
            "noise_addition",
            column="x",
            params={"intensity": intensity, "seed": seed},
            df=df,
            ctx=DataContext(),
        )
        lambdas.append(nk.metrics.utility.lambda_measure(df, out).scalar)
    assert lambdas[0] <= lambdas[1] + 1e-9


@given(
    values=_float_values,
    k=st.integers(min_value=2, max_value=10),
)
@settings(max_examples=30, deadline=None)
def test_microaggregation_output_is_k_anonymous(values, k):
    # microaggregation guarantees by construction that every cluster mean is
    # shared by >= k records. The test parametrises k freely; the method
    # accepts any k >= 2.
    if len(values) < 2 * k:
        return  # skip combinations that have too few rows for the method
    df = pl.DataFrame({"x": values})
    out = _fit_apply(
        "microaggregation",
        column="x",
        params={"k": k},
        df=df,
        ctx=DataContext(),
    )
    counts = out.get_column("x").value_counts().get_column("count").to_list()
    assert min(counts) >= k
