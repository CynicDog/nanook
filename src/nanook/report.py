"""Immutable report dataclasses returned by every metric and by `Pipeline.assess`.

Reports never carry the data they describe — they carry numeric summaries.
`to_dict` produces a JSON-serialisable payload that the pseudonymize engine
embeds in its execution stats, keeping nanook free of any I/O concern.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

__all__ = [
    "AssessmentReport",
    "IL1sReport",
    "KAnonymityReport",
    "KLDivergenceReport",
    "LDiversityReport",
    "LambdaReport",
    "RiskReport",
    "TClosenessReport",
    "UtilityReport",
]


@dataclass(frozen=True, slots=True)
class KAnonymityReport:
    """Result of a k-anonymity check on a chosen quasi-identifier set.

    Attributes:
        k: The threshold tested.
        qis: Quasi-identifier columns used.
        violations: Records whose equivalence class is smaller than `k`.
        violation_rate: `violations / n`, in `[0, 1]`.
        sample_uniques: Records that are unique on the quasi-identifier tuple.
        holds: True iff `violations == 0`.
    """

    k: int
    qis: tuple[str, ...]
    violations: int
    violation_rate: float
    sample_uniques: int
    holds: bool

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict; `qis` becomes a list."""
        return {**asdict(self), "qis": list(self.qis)}


@dataclass(frozen=True, slots=True)
class LDiversityReport:
    """Per-class l-diversity result for a single sensitive column.

    Attributes:
        l: The diversity threshold tested.
        qis: Quasi-identifier columns used to form classes.
        sensitive: The sensitive column tested.
        mode: One of "distinct", "entropy", "recursive".
        violations: Count of equivalence classes failing the test.
        violation_rate: `violations / G` where `G` is the class count.
        holds: True iff `violations == 0`.
        c: Multiplier for the recursive `(c, l)` variant; `None` otherwise.
    """

    l: int
    qis: tuple[str, ...]
    sensitive: str
    mode: Literal["distinct", "entropy", "recursive"]
    violations: int
    violation_rate: float
    holds: bool
    c: float | None = None

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict; `qis` becomes a list."""
        return {**asdict(self), "qis": list(self.qis)}


@dataclass(frozen=True, slots=True)
class TClosenessReport:
    """t-closeness result aggregating EMD across all equivalence classes.

    Attributes:
        t: The closeness threshold tested.
        qis: Quasi-identifier columns used.
        sensitive: Sensitive column compared against the population marginal.
        violations: Classes whose EMD exceeds `t`.
        max_emd: Largest EMD observed across classes.
        holds: True iff `violations == 0`.
    """

    t: float
    qis: tuple[str, ...]
    sensitive: str
    violations: int
    max_emd: float
    holds: bool

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict; `qis` becomes a list."""
        return {**asdict(self), "qis": list(self.qis)}


@dataclass(frozen=True, slots=True)
class LambdaReport:
    """Range-normalised mean absolute deviation between original and protected microdata.

    Attributes:
        scalar: File-level mean of `lambda_k` across columns, in `[0, 1]`.
        per_column: Map column-name -> `lambda_k`. Constant columns contribute 0.
    """

    scalar: float
    per_column: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict equivalent to this report."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IL1sReport:
    """Yancey-Winkler σ-scaled distance between original and protected microdata.

    Attributes:
        scalar: File-level mean of `IL1s_k` across retained columns.
        per_column: Map column-name -> `IL1s_k`. Constant columns are excluded.
    """

    scalar: float
    per_column: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict equivalent to this report."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KLDivergenceReport:
    """Categorical Kullback-Leibler divergence per column and file-level mean.

    Attributes:
        scalar: Mean of `D_KL,k` across categorical columns.
        per_column: Map categorical column-name -> `D_KL,k`, in nats.
        smoothing: Laplace smoothing constant used to avoid division by zero.
    """

    scalar: float
    per_column: dict[str, float] = field(default_factory=dict)
    smoothing: float = 1e-6

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict equivalent to this report."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RiskReport:
    """Bundle of risk metric reports; any individual slot may be `None` if not run."""

    k_anonymity: KAnonymityReport | None = None
    l_diversity: LDiversityReport | None = None
    t_closeness: TClosenessReport | None = None

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict; absent metrics are emitted as null."""
        return {
            "k_anonymity": self.k_anonymity.to_dict() if self.k_anonymity else None,
            "l_diversity": self.l_diversity.to_dict() if self.l_diversity else None,
            "t_closeness": self.t_closeness.to_dict() if self.t_closeness else None,
        }


@dataclass(frozen=True, slots=True)
class UtilityReport:
    """Bundle of utility metric reports; any individual slot may be `None` if not run."""

    lambda_measure: LambdaReport | None = None
    il1s: IL1sReport | None = None
    kl_divergence: KLDivergenceReport | None = None

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict; absent metrics are emitted as null."""
        return {
            "lambda_measure": self.lambda_measure.to_dict() if self.lambda_measure else None,
            "il1s": self.il1s.to_dict() if self.il1s else None,
            "kl_divergence": self.kl_divergence.to_dict() if self.kl_divergence else None,
        }


@dataclass(frozen=True, slots=True)
class AssessmentReport:
    """Pair of `RiskReport` + `UtilityReport`, the return type of `Pipeline.assess`."""

    risk: RiskReport
    utility: UtilityReport

    def to_dict(self) -> dict:
        """Return a JSON-friendly dict combining the risk and utility subreports."""
        return {"risk": self.risk.to_dict(), "utility": self.utility.to_dict()}
