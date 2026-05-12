from __future__ import annotations

from typing import TYPE_CHECKING

from nanook.exceptions import MethodParameterError

if TYPE_CHECKING:
    from nanook.core._base import SDCMethod

METHOD_REGISTRY: dict[str, type[SDCMethod]] = {}


def register_method(cls: type[SDCMethod]) -> type[SDCMethod]:
    """Decorator-or-call: insert ``cls`` into METHOD_REGISTRY keyed by ``cls.name``."""
    name = getattr(cls, "name", None)
    if not name:
        raise MethodParameterError(f"register_method: {cls.__name__} has no `name`")
    METHOD_REGISTRY[name] = cls
    return cls


def get_method(name: str) -> type[SDCMethod]:
    """Lookup helper that raises a typed error when ``name`` is not registered."""
    try:
        return METHOD_REGISTRY[name]
    except KeyError as exc:
        raise MethodParameterError(f"unknown SDC method: {name!r}") from exc
