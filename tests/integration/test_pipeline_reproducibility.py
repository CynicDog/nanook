"""Pipeline-level reproducibility regressions for M4 seed propagation.

Pre-M4 ``Pipeline(seed=...)`` was stored but never propagated to per-step
RNGs, so stochastic steps without explicit ``seed=`` parameters fell back to
OS entropy. After M4, ``Pipeline.apply`` spawns one ``SeedSequence`` substream
per step from ``self.seed`` and injects it where the step has no explicit
seed. Explicit per-step seeds always win.
"""

from __future__ import annotations

import nanook as nk


def test_pipeline_with_seed_is_deterministic_without_per_step_seeds(adult_small):
    def build():
        return (
            nk.Pipeline(seed=42)
            .context(quasi_identifiers=["age", "zip"], sensitive=["diagnosis"])
            .sampling(fraction=0.8)
            .noise("income", intensity=0.1)
            .pram("diagnosis", retention=0.7)
        )

    out_a = build().apply(adult_small)
    out_b = build().apply(adult_small)
    assert out_a.equals(out_b)


def test_pipeline_seed_changes_propagate_to_stochastic_steps(adult_small):
    def with_seed(s: int):
        return (
            nk.Pipeline(seed=s)
            .context(quasi_identifiers=["age"], sensitive=["diagnosis"])
            .noise("income", intensity=0.1)
            .apply(adult_small)
        )

    assert not with_seed(1).equals(with_seed(2))


def test_explicit_step_seed_wins_over_pipeline_seed(adult_small):
    # Same explicit per-step seed under two different Pipeline seeds → equal
    # output. If the pipeline seed were leaking through, the outputs would
    # differ.
    def with_pipeline_seed(s: int):
        return (
            nk.Pipeline(seed=s)
            .context(quasi_identifiers=["age"])
            .noise("income", intensity=0.1, seed=999)
            .apply(adult_small)
        )

    assert with_pipeline_seed(1).equals(with_pipeline_seed(2))
