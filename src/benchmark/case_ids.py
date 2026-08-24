"""Shared validation for case IDs used as filesystem path components."""

from __future__ import annotations

import re
from typing import Any


_SAFE_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class CaseIdValidationError(ValueError):
    """Raised when a case ID is not a conservative path-safe slug."""


def is_safe_case_id(value: Any) -> bool:
    """Return whether *value* is a path-safe, non-normalized case ID.

    Case IDs are deliberately limited to ASCII letters, digits, dots,
    underscores, and hyphens.  They are suitable for use as one path
    component, but callers must still resolve paths beneath their trusted
    root.
    """

    return (
        isinstance(value, str)
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and _SAFE_CASE_ID.fullmatch(value) is not None
    )


def validate_case_id(value: Any, *, field: str = "case_id") -> str:
    """Return *value* unchanged or raise for an unsafe case ID.

    No trimming or normalization is performed: distinct trusted identities
    must not collapse to the same filesystem name.
    """

    if not is_safe_case_id(value):
        raise CaseIdValidationError(
            f"{field} must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}} "
            "and must not be '.', '..', absolute, or contain a path separator"
        )
    return value
