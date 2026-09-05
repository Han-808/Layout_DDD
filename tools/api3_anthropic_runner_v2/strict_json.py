"""Strict whole-response JSON parsing without extraction or repair."""

from __future__ import annotations

import json
from typing import Any


class StrictJSONError(ValueError):
    pass


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

