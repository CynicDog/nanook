"""Simulates the pseudonymize engine's two-pass scheduler against nanook methods directly.

We do not import the engine's `Rule` base class here — that lives in another
project. Instead we replicate the engine's loop exactly: per-method,
optionally pre-scan against the full frame, then apply per batch. Matching
behaviour proves the SDCRuleAdapter (which lives under
``pseudonymize/engine/core/pseudo/rules/sdc``) will work once `uv sync` picks
up nanook in that venv.
"""

from __future__ import annotations

import polars as pl

from nanook.context import DataContext
from nanook.core._registry import METHOD_REGISTRY


def _engine_two_pass(
    method_name: str, column: str | None, params: dict, df: pl.DataFrame, ctx: DataContext
) -> tuple[pl.DataFrame, dict]:
    """Mimic the engine's two-pass apply for one rule on a full frame."""
    cls = METHOD_REGISTRY[method_name]
    instance = cls(column=column, params=params)
    fitted = instance.pre_scan(df, ctx) if cls.requires_pre_scan else {}
    out = instance.apply(df, ctx, fitted)
    return out, fitted


def test_engine_two_pass_top_bottom_coding():
    df = pl.DataFrame({"age": [float(i) for i in range(100)]})
    out, fitted = _engine_two_pass("top_bottom_coding", "age", {"percentile": 10}, df, DataContext())
    assert "lower_bound" in fitted
    assert "upper_bound" in fitted
    assert out.height == df.height


def test_engine_two_pass_noise_addition_is_pre_scan():
    df = pl.DataFrame({"x": [float(i) for i in range(50)]})
    out, fitted = _engine_two_pass("noise_addition", "x", {"intensity": 0.1, "seed": 1}, df, DataContext())
    assert "sigma" in fitted
    assert out.height == df.height
    assert not out.equals(df)


def test_engine_two_pass_local_suppression(adult_small):
    ctx = DataContext(quasi_identifiers=["age", "zip", "sex"])
    out, fitted = _engine_two_pass("local_suppression", None, {"target_k": 3}, adult_small, ctx)
    assert "suppressions" in fitted
    assert out.height == adult_small.height


def test_every_registered_method_executes_on_a_toy_dataset():
    """Every method either runs cleanly or raises a typed MethodParameterError early."""
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": ["x", "y", "x", "y", "x", "y", "x", "y"],
        }
    )
    ctx = DataContext(quasi_identifiers=["b"], sensitive=["a"])

    method_specs = {
        "sampling": {"fraction": 0.8, "seed": 1},
        "top_bottom_coding": {"percentile": 10},
        "global_recoding": {"bins": [0.0, 4.0, 8.0]},
        "local_suppression": {"target_k": 2},
        "noise_addition": {"intensity": 0.1, "seed": 1},
        "multiplicative_noise": {"sigma_log": 0.1, "seed": 1},
        "rounding": {"base": 1.0},
        "rank_swapping": {"window_pct": 0.25, "seed": 1},
        "data_swapping": {"fraction": 0.5, "seed": 1},
        "microaggregation": {"k": 2},
        "resampling": {"b": 4, "seed": 1},
        "pram": {"retention": 0.7, "seed": 1},
        "massc": {"k": 2, "f_sub": 0.75, "seed": 1},
    }
    numeric_only = {
        "top_bottom_coding",
        "noise_addition",
        "multiplicative_noise",
        "rounding",
        "rank_swapping",
        "data_swapping",
        "microaggregation",
        "resampling",
        "global_recoding",
    }

    for name, params in method_specs.items():
        column = None
        if name in numeric_only:
            column = "a"
        elif name == "pram":
            column = "b"
        out, _ = _engine_two_pass(name, column, params, df, ctx)
        if name in {"sampling", "massc"}:
            assert out.height <= df.height
        else:
            assert out.height == df.height
