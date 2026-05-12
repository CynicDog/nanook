"""Disclosure-risk and information-loss metrics.

Two submodules:

- `nanook.metrics.risk`   — k-anonymity, l-diversity, t-closeness.
- `nanook.metrics.utility` — λ measure, IL1s, KL divergence.

Both are pure compute over `polars.DataFrame`s and return immutable report
dataclasses from `nanook.report`.
"""

from __future__ import annotations

from nanook.metrics import risk, utility

__all__ = ["risk", "utility"]
