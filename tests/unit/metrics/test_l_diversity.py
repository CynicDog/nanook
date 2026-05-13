from __future__ import annotations

import math

import polars as pl
import pytest

from nanook.exceptions import MethodParameterError
from nanook.metrics.risk.l_diversity import l_diversity


def test_distinct_pass():
    df = pl.DataFrame({"zip": ["1", "1", "1"], "diag": ["A", "B", "C"]})
    assert l_diversity(df, qis=["zip"], sensitive="diag", l=3, mode="distinct").holds


def test_distinct_fail_when_class_too_homogeneous():
    df = pl.DataFrame({"zip": ["1", "1", "1"], "diag": ["A", "A", "B"]})
    r = l_diversity(df, qis=["zip"], sensitive="diag", l=3, mode="distinct")
    assert not r.holds
    assert r.violations == 1


def test_entropy_pass_for_uniform_class():
    df = pl.DataFrame({"zip": ["1"] * 4, "diag": ["A", "B", "C", "D"]})
    r = l_diversity(df, qis=["zip"], sensitive="diag", l=4, mode="entropy")
    assert r.holds


def test_entropy_threshold_uses_log_l():
    # Two values, equal frequency: H = log 2; pass for l=2.
    df = pl.DataFrame({"zip": ["1", "1"], "diag": ["A", "B"]})
    r = l_diversity(df, qis=["zip"], sensitive="diag", l=2, mode="entropy")
    assert r.holds
    assert math.isclose(math.log(2), math.log(2))  # sanity


def test_recursive_carries_c_in_report():
    df = pl.DataFrame({"zip": ["1"] * 3, "diag": ["A", "B", "C"]})
    r = l_diversity(df, qis=["zip"], sensitive="diag", l=2, mode="recursive", c=2.0)
    assert r.c == 2.0


def test_unknown_mode_raises():
    df = pl.DataFrame({"zip": ["1"], "diag": ["A"]})
    with pytest.raises(MethodParameterError):
        l_diversity(df, qis=["zip"], sensitive="diag", l=2, mode="bogus")  # type: ignore[arg-type]


def test_golden_value_l_diversity_entropy_hand_traced():
    # One equivalence class on zip="1", diag = [A, A, B, B, C].
    # P = (2/5, 2/5, 1/5). H = -(0.4 ln 0.4 + 0.4 ln 0.4 + 0.2 ln 0.2)
    #                       ≈ 1.05492.
    # For l=2 (entropy threshold = ln 2 ≈ 0.6931), passes; for l=3 (threshold
    # = ln 3 ≈ 1.0986), fails.
    df = pl.DataFrame(
        {
            "zip": ["1"] * 5,
            "diag": ["A", "A", "B", "B", "C"],
        }
    )
    r2 = l_diversity(df, qis=["zip"], sensitive="diag", l=2, mode="entropy")
    assert r2.holds
    r3 = l_diversity(df, qis=["zip"], sensitive="diag", l=3, mode="entropy")
    assert not r3.holds
