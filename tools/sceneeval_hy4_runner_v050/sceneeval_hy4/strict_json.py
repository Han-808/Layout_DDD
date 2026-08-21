"""Strict JSON parsing used for inputs, API envelopes, and model content."""

from __future__ import annotations

import json
from typing import Any


class StrictJSONError(ValueError):
    """Raised for syntax, non-standard constants, or duplicate object keys."""


def _reject_constant(value: str) -> None:
    raise StrictJSONError(f"non-standard JSON numeric constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJSONError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def loads_strict(text: str) -> Any:
    """Parse one complete RFC-style JSON value without repairs or extraction."""
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except StrictJSONError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise StrictJSONError(str(exc)) from exc


def loads_strict_bytes(data: bytes) -> Any:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise StrictJSONError(f"response is not valid UTF-8: {exc}") from exc
    return loads_strict(text)
