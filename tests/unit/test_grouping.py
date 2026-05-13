"""Null-handling regressions for QI grouping (T2 in REVIEW.md).

``_internal/grouping.py`` claims to handle nulls — polars' ``group_by`` treats
nulls as a single value, forming one null-keyed equivalence class. The metrics
that consume these helpers must round-trip that semantics. The l-diversity
``distinct`` mode additionally drops nulls from the sensitive column before
counting distinct values.
"""

from __future__ import annotations

import polars as pl

from nanook._internal.grouping import equivalence_class_sizes, per_record_class_size
from nanook.metrics.risk.k_anonymity import k_anonymity
from nanook.metrics.risk.l_diversity import l_diversity


def test_equivalence_class_sizes_treats_null_qi_as_one_class():
    df = pl.DataFrame({"q": [None, None, None, "a", "a"]})
    sizes = equivalence_class_sizes(df, ["q"])
    # Two classes: null (3) and "a" (2).
    rows = {tuple(row) for row in sizes.iter_rows()}
    assert rows == {(None, 3), ("a", 2)}


def test_per_record_class_size_reports_null_class_size_for_null_rows():
    df = pl.DataFrame({"q": [None, None, "a", "a"]})
    sizes = per_record_class_size(df, ["q"])
    assert sizes.to_list() == [2, 2, 2, 2]


def test_k_anonymity_null_qi_class_satisfies_when_at_threshold():
    df = pl.DataFrame({"q": [None, None, None, "a", "a", "a"]})
    r = k_anonymity(df, qis=["q"], k=3)
    assert r.holds
    assert r.violations == 0


def test_l_diversity_distinct_ignores_null_sensitive_values():
    # One equivalence class on zip="1". Without null exclusion, distinct
    # would be {A, None} = 2 and l=2 would pass. The pseudocode (and
    # implementation) drop nulls, leaving distinct = {A} = 1, which fails.
    df = pl.DataFrame({"zip": ["1", "1", "1"], "diag": ["A", "A", None]})
    r = l_diversity(df, qis=["zip"], sensitive="diag", l=2, mode="distinct")
    assert not r.holds
    assert r.violations == 1
