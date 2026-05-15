"""UI-friendly schema metadata for every registered SDC method.

The :class:`SDCMethod` base intentionally keeps no presentation metadata —
it carries only what the engine needs (``name``, ``requires_pre_scan``,
``drops_column``). Downstream applications (the pseudonymize studio in
particular) need richer information: a category, a display name, the parameter
types and defaults that drive their forms.

This module adds that surface as opt-in metadata attached at class definition
via :func:`schema`. The decorator both registers the class with
:data:`METHOD_REGISTRY` and pins a :class:`MethodSchema` on the class so the
catalogue can be re-serialised at any time via :func:`list_method_schemas`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nanook.core._registry import METHOD_REGISTRY, register_method
from nanook.exceptions import MethodParameterError

if TYPE_CHECKING:
    from nanook.core._base import SDCMethod

__all__ = [
    "MethodSchema",
    "ParamSchema",
    "list_method_schemas",
    "schema",
]


@dataclass(frozen=True)
class ParamSchema:
    """A single method parameter as the UI sees it."""

    name: str
    display_name: str
    param_type: str
    default: Any = None
    code_options: tuple[dict[str, str], ...] | None = None
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class MethodSchema:
    """A registered SDC method projected for the studio's rule selector."""

    name: str
    display_name: str
    category: str
    applicable_dtypes: tuple[str, ...]
    requires_pre_scan: bool
    drops_column: bool
    description: str
    params: tuple[ParamSchema, ...] = field(default_factory=tuple)
    requires_quasi_identifiers: bool = False
    requires_sensitive: bool = False
    is_pipeline_scope: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.name,
            "displayName": self.display_name,
            "category": self.category,
            "applicableTypes": list(self.applicable_dtypes),
            "requiresPreScan": self.requires_pre_scan,
            "dropsColumn": self.drops_column,
            "requiresQuasiIdentifiers": self.requires_quasi_identifiers,
            "requiresSensitive": self.requires_sensitive,
            "isPipelineScope": self.is_pipeline_scope,
            "description": self.description,
            "params": [_param_to_dict(p) for p in self.params],
        }


def _param_to_dict(p: ParamSchema) -> dict[str, Any]:
    return {
        "code": p.name,
        "displayName": p.display_name,
        "paramType": p.param_type,
        "defaultValue": None if p.default is None else str(p.default),
        "codeOptions": None if p.code_options is None else json.dumps(list(p.code_options)),
        "required": p.required,
        "description": p.description,
    }


def schema(
    *,
    display_name: str,
    category: str,
    applicable_dtypes: Sequence[str],
    description: str,
    params: Sequence[ParamSchema] = (),
    requires_quasi_identifiers: bool = False,
    requires_sensitive: bool = False,
    is_pipeline_scope: bool = False,
) -> Callable[[type[SDCMethod]], type[SDCMethod]]:
    """Attach a :class:`MethodSchema` to ``cls`` and register it.

    Replaces :func:`register_method` at the call site. The decorator reads
    ``cls.name``, ``cls.requires_pre_scan``, and ``cls.drops_column`` directly,
    so those class attributes must already be set when the decorator runs.

    ``requires_quasi_identifiers`` / ``requires_sensitive`` mirror the
    invariants the method enforces in ``pre_scan`` / ``apply``: methods that
    operate on the QI tuple (e.g. ``massc``, ``local_suppression``) must set
    the QI flag so callers can gate the UI before the engine rejects the run.

    ``is_pipeline_scope`` is a UI-facing hint: when ``True``, the step has no
    meaningful per-column placement — the engine ignores the step's ``column``
    field and operates on context columns instead. Studios should surface such
    methods at the pipeline level, not in a per-column rule selector.
    """

    def decorate(cls: type[SDCMethod]) -> type[SDCMethod]:
        if not getattr(cls, "name", None):
            raise MethodParameterError(f"@schema: {cls.__name__} has no `name`")
        cls._schema = MethodSchema(  # type: ignore[attr-defined]
            name=cls.name,
            display_name=display_name,
            category=category,
            applicable_dtypes=tuple(applicable_dtypes),
            requires_pre_scan=cls.requires_pre_scan,
            drops_column=cls.drops_column,
            description=description,
            params=tuple(params),
            requires_quasi_identifiers=requires_quasi_identifiers,
            requires_sensitive=requires_sensitive,
            is_pipeline_scope=is_pipeline_scope,
        )
        return register_method(cls)

    return decorate


def list_method_schemas() -> list[dict[str, Any]]:
    """Project every registered method as a JSON-ready dict, sorted by ``name``."""
    out: list[dict[str, Any]] = []
    for name in sorted(METHOD_REGISTRY):
        cls = METHOD_REGISTRY[name]
        s: MethodSchema | None = getattr(cls, "_schema", None)
        if s is None:
            # Fallback for methods that didn't go through @schema yet.
            out.append(
                {
                    "code": cls.name,
                    "displayName": cls.name,
                    "category": "Uncategorised",
                    "applicableTypes": [],
                    "requiresPreScan": cls.requires_pre_scan,
                    "dropsColumn": cls.drops_column,
                    "requiresQuasiIdentifiers": False,
                    "requiresSensitive": False,
                    "isPipelineScope": False,
                    "description": (cls.__doc__ or "").strip().split("\n", 1)[0],
                    "params": [],
                }
            )
        else:
            out.append(s.to_dict())
    return out
