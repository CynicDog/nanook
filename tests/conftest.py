from __future__ import annotations

import polars as pl
import pytest


@pytest.fixture
def adult_small() -> pl.DataFrame:
    """A 12-row toy dataset with QIs (age, zip, sex), sensitive (diagnosis), and an income column."""
    return pl.DataFrame(
        {
            "age": [30, 30, 30, 30, 31, 31, 31, 31, 40, 40, 40, 50],
            "zip": ["1", "1", "1", "1", "2", "2", "2", "2", "3", "3", "3", "4"],
            "sex": ["M", "M", "F", "F", "M", "M", "F", "F", "M", "F", "F", "M"],
            "diagnosis": ["A", "B", "C", "A", "A", "B", "C", "A", "B", "C", "A", "B"],
            "income": [
                50_000.0,
                51_000,
                49_000,
                60_000,
                52_000,
                53_000,
                48_000,
                47_000,
                90_000,
                91_000,
                92_000,
                200_000,
            ],
        }
    )


@pytest.fixture
def perfectly_k_anonymous() -> pl.DataFrame:
    """Every record has at least 3 equivalents on (zip, age_bucket)."""
    return pl.DataFrame(
        {
            "zip": ["1"] * 3 + ["2"] * 3 + ["3"] * 4,
            "age_bucket": ["30s"] * 3 + ["40s"] * 3 + ["50s"] * 4,
            "diagnosis": ["A", "B", "C", "A", "B", "C", "A", "B", "C", "A"],
        }
    )
