from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import polars as pl

    from nanook.context import DataContext


class SDCMethod(ABC):
    """Base contract for every SDC method.

    Two execution modes mirror the pseudonymize engine's two-pass design:

    Streaming-safe (`requires_pre_scan = False`): `apply` is total — it needs
    only the row-group batch and the (possibly empty) params from `pre_scan`.

    Pre-scan required (`requires_pre_scan = True`): `pre_scan` runs once over
    the relevant columns to compute distribution-dependent parameters; `apply`
    uses them per batch.

    Subclasses set the class variable `name` to the canonical snake_case
    identifier used in the declarative pipeline JSON and in the engine
    registry under the `SDC_<UPPER>` adapter name.
    """

    name: ClassVar[str]
    requires_pre_scan: ClassVar[bool] = False
    drops_column: ClassVar[bool] = False

    def __init__(self, column: str | None = None, params: dict | None = None) -> None:
        self.column = column
        self.params: dict = dict(params or {})
        self._fitted: dict = {}

    @abstractmethod
    def pre_scan(self, df: pl.DataFrame, ctx: DataContext) -> dict:
        """Compute distribution-dependent parameters from a representative sample.

        Streaming-safe methods return an empty dict. Pre-scan methods return a
        JSON-serialisable dict consumed by `apply`.
        """

    @abstractmethod
    def apply(self, df: pl.DataFrame, ctx: DataContext, params: dict) -> pl.DataFrame:
        """Transform ``df`` using the pre-scan ``params``; return a new frame."""

    @classmethod  # noqa: B027 — opt-in validator hook; no-op by default
    def validate_params(cls, params: dict) -> None:  # noqa: ARG003
        """Optional override: raise MethodParameterError on invalid construction-time params.

        The default is a no-op; subclasses override to enforce per-method constraints.
        """
