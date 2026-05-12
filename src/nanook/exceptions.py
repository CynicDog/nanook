"""Exception hierarchy for nanook.

Every error raised by nanook code subclasses `NanookError`, so callers can
catch the whole library with a single `except`. Specific subclasses are
defined for the three boundaries where things actually go wrong: user-supplied
`DataContext`, user-supplied method parameters, and unsupported column dtypes.
"""

from __future__ import annotations

__all__ = [
    "ContextValidationError",
    "MethodParameterError",
    "NanookError",
    "UnsupportedDtypeError",
]


class NanookError(Exception):
    """Base class for all nanook errors.

    Carries an optional short `code` so downstream tooling (engine adapters,
    UI surfaces) can branch on the failure mode without parsing the message.
    """

    code: str | None = None

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.code


class ContextValidationError(NanookError):
    """Raised when a `DataContext` references columns absent from the frame or is internally inconsistent."""

    code = "context_invalid"


class MethodParameterError(NanookError):
    """Raised when an SDC method receives a parameter outside its supported domain."""

    code = "method_param_invalid"


class UnsupportedDtypeError(NanookError):
    """Raised when a metric or method is applied to a column whose dtype it cannot handle."""

    code = "unsupported_dtype"
