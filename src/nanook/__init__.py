"""nanook: blazingly fast privacy engineering and statistical disclosure control.

Top-level surface kept intentionally small. The fluent `Pipeline` is the
recommended entry point; raw metric and method callables are reachable via
`nanook.metrics` and `nanook.core` for one-off use.

Examples:
    >>> import polars as pl
    >>> import nanook as nk
    >>> df = pl.DataFrame({"age": [30, 30, 31], "zip": ["10001", "10001", "10002"]})
    >>> nk.metrics.risk.k_anonymity(df, qis=["age", "zip"], k=2).holds
    False
"""

from __future__ import annotations

from nanook import core, metrics
from nanook.context import DataContext
from nanook.exceptions import (
    ContextValidationError,
    MethodParameterError,
    NanookError,
    UnsupportedDtypeError,
)
from nanook.pipeline import Pipeline
from nanook.report import (
    AssessmentReport,
    IL1sReport,
    KAnonymityReport,
    KLDivergenceReport,
    LambdaReport,
    LDiversityReport,
    RiskReport,
    TClosenessReport,
    UtilityReport,
)

__version__ = "0.1.0"

__all__ = [
    "AssessmentReport",
    "ContextValidationError",
    "DataContext",
    "IL1sReport",
    "KAnonymityReport",
    "KLDivergenceReport",
    "LDiversityReport",
    "LambdaReport",
    "MethodParameterError",
    "NanookError",
    "Pipeline",
    "RiskReport",
    "TClosenessReport",
    "UnsupportedDtypeError",
    "UtilityReport",
    "__version__",
    "core",
    "metrics",
]
