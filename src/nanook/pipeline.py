"""Compose a `DataContext`, an ordered list of SDC steps, and an assessment.

Both forms documented in the user guide lower to the same internal step list:

- Declarative: `Pipeline.from_dict(payload)` for engine and JSON workflows.
- Fluent: chained builder methods like `.context(...)`, `.top_bottom(...)`,
  `.local_suppression(...)` for notebooks and scripts.

`to_dict` round-trips losslessly through `from_dict`, so the same definition
serialises into a PseudoConfig payload, a CLI input, or a notebook value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import numpy as np

from nanook.context import DataContext
from nanook.core._registry import get_method
from nanook.exceptions import MethodParameterError
from nanook.metrics.risk.k_anonymity import k_anonymity
from nanook.metrics.risk.l_diversity import l_diversity
from nanook.metrics.risk.t_closeness import t_closeness
from nanook.metrics.utility.il1s import il1s
from nanook.metrics.utility.kl_divergence import kl_divergence
from nanook.metrics.utility.lambda_measure import lambda_measure
from nanook.report import (
    AssessmentReport,
    RiskReport,
    UtilityReport,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    import polars as pl

__all__ = ["Pipeline"]


@dataclass(slots=True)
class _Step:
    method: str
    column: str | None
    params: dict

    def to_dict(self) -> dict:
        payload = {"method": self.method, "params": dict(self.params)}
        if self.column is not None:
            payload["column"] = self.column
        return payload


class Pipeline:
    """Compose a `DataContext`, an ordered list of SDC steps, and an assessment.

    A Pipeline is built fluently:

        >>> import nanook as nk
        >>> p = nk.Pipeline().context(quasi_identifiers=["age"], sensitive=["dx"])
        >>> p.context_.quasi_identifiers
        ('age',)

    or declaratively from a dict serialisable to JSON:

        >>> p2 = nk.Pipeline.from_dict({
        ...     "version": 1,
        ...     "context": {"quasi_identifiers": ["age"], "sensitive": ["dx"]},
        ...     "steps": [],
        ... })
        >>> p2.to_dict()["version"]
        1
    """

    VERSION = 1

    def __init__(self, seed: int | None = None) -> None:
        """Build an empty pipeline.

        ``seed``, if set, seeds every stochastic step that does not declare
        its own ``seed=`` parameter; per-step ``seed=`` values still win when
        present. Independent substreams per step are derived via
        :class:`numpy.random.SeedSequence` so step order deterministically
        maps to per-step seeds without any two steps sharing entropy.
        """
        self.context_: DataContext = DataContext()
        self.steps: list[_Step] = []
        self.seed: int | None = seed

    def context(
        self,
        quasi_identifiers: Sequence[str] = (),
        sensitive: Sequence[str] = (),
        weights: str | None = None,
        hierarchy: Sequence[Sequence[str]] = (),
    ) -> Self:
        """Set the column-role declarations and return ``self`` for chaining."""
        self.context_ = DataContext(
            quasi_identifiers=quasi_identifiers,
            sensitive=sensitive,
            weights=weights,
            hierarchy=hierarchy,
        )
        return self

    def step(self, method: str, *, column: str | None = None, **params: object) -> Self:
        """Append a step by canonical method name. Generic escape hatch for any registered method."""
        self.steps.append(_Step(method=method, column=column, params=dict(params)))
        return self

    def sampling(self, *, fraction: float, seed: int | None = None, write_weights: bool = False) -> Self:
        """Add a Bernoulli sampling step retaining each row independently with probability ``fraction``."""
        return self.step("sampling", fraction=fraction, seed=seed, write_weights=write_weights)

    def top_bottom(self, column: str, *, percentile: float = 5.0, alternative: str = "two_sided") -> Self:
        """Add a top/bottom-coding step on ``column``."""
        return self.step("top_bottom_coding", column=column, percentile=percentile, alternative=alternative)

    def global_recoding(
        self,
        column: str,
        *,
        bins: list | None = None,
        mapping: dict | None = None,
        label_mode: str = "index",
    ) -> Self:
        """Add a global-recoding step.

        Pass exactly one of ``bins`` (continuous) or ``mapping`` (categorical).
        """
        params: dict = {"label_mode": label_mode}
        if bins is not None:
            params["bins"] = list(bins)
        if mapping is not None:
            params["mapping"] = dict(mapping)
        return self.step("global_recoding", column=column, **params)

    def suppression(self, column: str) -> Self:
        """Drop ``column`` from the output. The strongest non-perturbative treatment."""
        return self.step("suppression", column=column)

    def local_suppression(self, *, target_k: int = 5, cost_priority: dict | None = None) -> Self:
        """Add a local-suppression step targeting ``target_k`` over the context's quasi-identifiers."""
        return self.step(
            "local_suppression",
            target_k=target_k,
            cost_priority=dict(cost_priority) if cost_priority else None,
        )

    def noise(self, column: str, *, intensity: float = 0.1, seed: int | None = None) -> Self:
        """Add additive Gaussian noise to ``column``, scaled to its standard deviation."""
        return self.step("noise_addition", column=column, intensity=intensity, seed=seed)

    def multiplicative_noise(self, column: str, *, sigma_log: float = 0.1, seed: int | None = None) -> Self:
        """Scale ``column`` by a log-normal multiplier with Höhne moment rescaling; zeros stay zero."""
        return self.step("multiplicative_noise", column=column, sigma_log=sigma_log, seed=seed)

    def rounding(
        self, column: str, *, base: float = 1.0, random_within_bin: bool = False, seed: int | None = None
    ) -> Self:
        """Snap ``column`` to a multiple of ``base``."""
        return self.step(
            "rounding",
            column=column,
            base=base,
            random_within_bin=random_within_bin,
            seed=seed,
        )

    def rank_swap(self, column: str, *, window_pct: float = 0.05, seed: int | None = None) -> Self:
        """Swap each value with a neighbour within ``window_pct`` of its rank."""
        return self.step("rank_swapping", column=column, window_pct=window_pct, seed=seed)

    def swap(self, column: str, *, fraction: float = 1.0, seed: int | None = None) -> Self:
        """Randomly swap ``column`` values between ``fraction`` of pairs."""
        return self.step("data_swapping", column=column, fraction=fraction, seed=seed)

    def microaggregate(self, column: str, *, k: int = 5) -> Self:
        """Replace each value in ``column`` with its k-nearest-neighbour cluster mean."""
        return self.step("microaggregation", column=column, k=k)

    def resample(self, column: str, *, b: int = 10, seed: int | None = None) -> Self:
        """Smooth ``column`` by averaging order statistics across ``b`` bootstrap samples."""
        return self.step("resampling", column=column, b=b, seed=seed)

    def pram(self, column: str, *, retention: float = 0.8, seed: int | None = None) -> Self:
        """Apply a categorical PRAM transition to ``column`` with the given retention probability."""
        return self.step("pram", column=column, retention=retention, seed=seed)

    def massc(
        self,
        *,
        k: int = 5,
        f_sub: float = 0.8,
        calibration_vars: list[str] | None = None,
        seed: int | None = None,
    ) -> Self:
        """Run the four-step MASSC (micro-agglomeration, QI substitution, subsampling, calibration)."""
        return self.step(
            "massc",
            k=k,
            f_sub=f_sub,
            calibration_vars=calibration_vars,
            seed=seed,
        )

    def to_dict(self) -> dict:
        """Serialise the pipeline to a JSON-friendly dict. Inverse of `from_dict`."""
        return {
            "version": self.VERSION,
            "context": self.context_.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Pipeline:
        """Build a `Pipeline` from a JSON payload.

        Raises:
            MethodParameterError: `version` is not the expected value or `payload`
                has unknown top-level keys.
        """
        known = {"version", "context", "steps", "seed"}
        extra = set(payload) - known
        if extra:
            raise MethodParameterError(f"Pipeline.from_dict: unknown keys {sorted(extra)}")
        version = payload.get("version", cls.VERSION)
        if version != cls.VERSION:
            raise MethodParameterError(
                f"Pipeline.from_dict: unsupported version {version}, expected {cls.VERSION}"
            )
        p = cls(seed=payload.get("seed"))
        p.context_ = DataContext.from_dict(payload.get("context", {}))
        for raw in payload.get("steps", []):
            p.steps.append(
                _Step(
                    method=raw["method"],
                    column=raw.get("column"),
                    params=dict(raw.get("params", {})),
                )
            )
        return p

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply each step in order, threading the resulting frame through.

        Each step's ``method`` is looked up in the global method registry,
        instantiated with the step's ``column`` and ``params``, pre-scanned
        when ``requires_pre_scan`` is set, and applied. Unknown method names
        raise `MethodParameterError` from the registry.

        When ``self.seed`` is set, independent substreams are spawned for
        each step via :class:`numpy.random.SeedSequence` and injected into
        any step whose own ``seed=`` is ``None``. Explicit per-step seeds
        always win.
        """
        self.context_.validate(df)
        if self.seed is not None and self.steps:
            substreams = np.random.SeedSequence(self.seed).spawn(len(self.steps))
        else:
            substreams = [None] * len(self.steps)
        out = df
        for step, ss in zip(self.steps, substreams, strict=True):
            cls = get_method(step.method)
            effective_params = dict(step.params)
            if effective_params.get("seed") is None and ss is not None:
                effective_params["seed"] = int(ss.generate_state(1, dtype=np.uint32)[0])
            instance = cls(column=step.column, params=effective_params)
            # Narrow the context to columns still present in ``out``: prior
            # column-drop steps (e.g. ``suppression``) may have removed QIs the
            # pipeline-level context still names. Context-driven methods such as
            # ``local_suppression``/``massc`` do ``df.select(context.qis)``, so
            # they would otherwise raise ColumnNotFoundError on the dropped column.
            step_ctx = _restrict_context_to_frame(self.context_, out)
            params = instance.pre_scan(out, step_ctx) if cls.requires_pre_scan else {}
            out = instance.apply(out, step_ctx, params)
        return out

    def assess(
        self,
        original: pl.DataFrame,
        protected: pl.DataFrame | None = None,
        *,
        k: int = 5,
        l: int = 3,
        t: float = 0.2,
        l_mode: str = "distinct",
    ) -> AssessmentReport:
        """Compute risk and utility metrics implied by the pipeline's context.

        ``protected`` defaults to ``original`` when the caller wants a baseline
        risk assessment without having applied any transformation yet — λ, IL1s,
        and KL are still emitted but evaluate to zero loss.

        Risk metrics fire conditionally on what the context declares:
        k-anonymity needs ``quasi_identifiers``; l-diversity and t-closeness
        additionally need ``sensitive``. t-closeness is only run when the
        sensitive column is numeric (ordinal support) — nominal support must
        be invoked directly via `nanook.metrics.risk.t_closeness`.
        """
        if protected is None:
            protected = original
        # Validate against ``original``: the context describes the input's roles.
        # ``protected`` may be missing columns that a step intentionally dropped
        # (e.g. ``suppression``), and that's expected — _assess_risk filters those
        # out so the risk view reflects what's actually still releasable.
        self.context_.validate(original)

        risk = _assess_risk(protected, self.context_, k=k, l=l, t=t, l_mode=l_mode)
        utility = _assess_utility(original, protected)
        return AssessmentReport(risk=risk, utility=utility)


def _restrict_context_to_frame(ctx: DataContext, df: pl.DataFrame) -> DataContext:
    """Return a context whose role-bearing columns all exist in ``df``.

    Identity when nothing was dropped; otherwise QIs, sensitive columns, and
    ``weights`` missing from ``df`` are filtered out. Used by `Pipeline.apply`
    to thread a coherent context through column-drop steps.
    """
    present = set(df.columns)
    qis = tuple(q for q in ctx.quasi_identifiers if q in present)
    sensitive = tuple(s for s in ctx.sensitive if s in present)
    weights = ctx.weights if ctx.weights in present else None
    if qis == ctx.quasi_identifiers and sensitive == ctx.sensitive and weights == ctx.weights:
        return ctx
    return DataContext(
        quasi_identifiers=qis,
        sensitive=sensitive,
        weights=weights,
        hierarchy=ctx.hierarchy,
    )


def _assess_risk(
    df: pl.DataFrame,
    ctx: DataContext,
    *,
    k: int,
    l: int,
    t: float,
    l_mode: str,
) -> RiskReport:
    # Skip QIs/sensitive columns the protected frame no longer carries (e.g. dropped
    # by ``suppression``) — they cannot contribute re-identification signal once gone.
    qis = [q for q in ctx.quasi_identifiers if q in df.columns]
    if not qis:
        return RiskReport()

    k_rep = k_anonymity(df, qis=qis, k=k)
    l_rep = None
    t_rep = None
    sensitive_present = [s for s in ctx.sensitive if s in df.columns]
    if sensitive_present:
        sensitive_col = sensitive_present[0]
        l_rep = l_diversity(df, qis=qis, sensitive=sensitive_col, l=l, mode=l_mode)  # type: ignore[arg-type]
        if df.schema[sensitive_col].is_numeric():
            t_rep = t_closeness(df, qis=qis, sensitive=sensitive_col, t=t)
    return RiskReport(k_anonymity=k_rep, l_diversity=l_rep, t_closeness=t_rep)


def _assess_utility(original: pl.DataFrame, protected: pl.DataFrame) -> UtilityReport:
    numeric = [c for c, dt in original.schema.items() if dt.is_numeric() and c in protected.columns]
    cat_dtypes = ("String", "Categorical", "Boolean", "Enum")
    categorical = [
        c for c, dt in original.schema.items() if type(dt).__name__ in cat_dtypes and c in protected.columns
    ]

    lambda_rep = lambda_measure(original, protected, columns=numeric) if numeric else None
    il1s_rep = il1s(original, protected, columns=numeric) if numeric else None
    kl_rep = kl_divergence(original, protected, columns=categorical) if categorical else None
    return UtilityReport(lambda_measure=lambda_rep, il1s=il1s_rep, kl_divergence=kl_rep)
