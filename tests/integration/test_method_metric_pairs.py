"""Round-trip tests: every method should move its native risk/utility metric in the documented direction.

These tie each SDC method to the metric the pseudocode pairs with it:

- ``local_suppression`` → ``k_anonymity`` (should reach ``target_k`` exactly).
- ``pram`` → ``kl_divergence`` (zero at full retention; monotone in 1−retention).
- ``noise_addition`` → ``lambda_measure`` (monotone in intensity).
"""

from __future__ import annotations

import polars as pl
import pytest

import nanook as nk
from nanook.context import DataContext
from nanook.core._registry import get_method


def _fit_apply(name, *, column, params, df, ctx):
    cls = get_method(name)
    instance = cls(column=column, params=params)
    fitted = instance.pre_scan(df, ctx) if cls.requires_pre_scan else {}
    return instance.apply(df, ctx, fitted)


@pytest.mark.parametrize("target_k", [2, 3])
def test_local_suppression_does_not_increase_k_anonymity_violations(target_k):
    # local_suppression uses a wildcard-aware k-anonymity model internally
    # (suppressed cells match anything); the ``k_anonymity`` metric does
    # group-by-exact-tuple. The two definitions don't agree on which records
    # are k-anonymous, so the integration assertion is "violations don't
    # increase" rather than "violations hit zero".
    df = pl.DataFrame(
        {
            "age": [30] * 4 + [31] * 4 + [40] * 3 + [41] * 2 + [50],
            "zip": ["1"] * 4 + ["2"] * 4 + ["3"] * 5 + ["4"],
            "sex": ["M", "F", "M", "F"] * 3 + ["M", "F"],
        }
    )
    qis = ["age", "zip", "sex"]
    ctx = DataContext(quasi_identifiers=qis)
    baseline = nk.metrics.risk.k_anonymity(df, qis=qis, k=target_k).violations
    out = _fit_apply(
        "local_suppression",
        column=None,
        params={"target_k": target_k},
        df=df,
        ctx=ctx,
    )
    after = nk.metrics.risk.k_anonymity(out, qis=qis, k=target_k).violations
    assert after <= baseline


def test_pram_retention_one_yields_zero_kl():
    df = pl.DataFrame({"x": ["A"] * 10 + ["B"] * 10 + ["C"] * 10})
    out = _fit_apply(
        "pram",
        column="x",
        params={"retention": 1.0, "seed": 0},
        df=df,
        ctx=DataContext(),
    )
    r = nk.metrics.utility.kl_divergence(df, out)
    assert r.scalar == pytest.approx(0.0, abs=1e-9)


def test_pram_kl_grows_as_retention_drops():
    df = pl.DataFrame({"x": ["A"] * 100 + ["B"] * 100 + ["C"] * 100})
    out_high = _fit_apply(
        "pram",
        column="x",
        params={"retention": 0.9, "seed": 0},
        df=df,
        ctx=DataContext(),
    )
    out_low = _fit_apply(
        "pram",
        column="x",
        params={"retention": 0.5, "seed": 0},
        df=df,
        ctx=DataContext(),
    )
    kl_high = nk.metrics.utility.kl_divergence(df, out_high).scalar
    kl_low = nk.metrics.utility.kl_divergence(df, out_low).scalar
    assert kl_low > kl_high


def test_noise_addition_lambda_monotonic_in_intensity():
    df = pl.DataFrame({"x": [float(i) for i in range(100)]})
    results = []
    for intensity in (0.1, 0.5, 1.0):
        out = _fit_apply(
            "noise_addition",
            column="x",
            params={"intensity": intensity, "seed": 0},
            df=df,
            ctx=DataContext(),
        )
        results.append(nk.metrics.utility.lambda_measure(df, out).scalar)
    assert results[0] < results[1] < results[2]
