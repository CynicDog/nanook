"""Statistical-disclosure-control methods.

Each method is a subclass of `_base.SDCMethod` exposing two operations:
``pre_scan(df, ctx)`` returning a parameter dict, and ``apply(df, ctx, params)``
returning a transformed frame. Streaming-safe methods set
``requires_pre_scan = False`` and ``pre_scan`` returns an empty dict.

The two-pass split mirrors the pseudonymize engine's `Rule` execution model;
that alignment is what lets the adapter layer in the engine stay a thin shim.
"""

from __future__ import annotations

# Import side-effect: each module registers its method class.
from nanook.core import non_perturbative as _non_perturbative  # noqa: F401
from nanook.core import perturbative as _perturbative  # noqa: F401
from nanook.core._base import SDCMethod
from nanook.core._registry import METHOD_REGISTRY, get_method, register_method

__all__ = ["METHOD_REGISTRY", "SDCMethod", "get_method", "register_method"]
