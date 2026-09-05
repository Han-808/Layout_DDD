from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")


def required_text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{label} must be non-empty")
    return result


def optional_text(value: Any) -> str | None:
    result = str(value or "").strip()
    return result or None


def identifiers(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
) -> tuple[str, ...]:
    if value is None and minimum == 0:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a JSON list")
    result = tuple(str(item).strip() for item in value)
    if (
        len(result) < minimum
        or any(not item for item in result)
        or len(result) != len(set(result))
    ):
        raise ValueError(
            f"{label} requires at least {minimum} unique non-empty values"
        )
    return result


def token(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not TOKEN_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lower_snake_case token")
    return result


def tokens(value: Any, label: str) -> tuple[str, ...]:
    result = identifiers(value, label)
    for item in result:
        token(item, label)
    return result


def confidence(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be finite and within [0, 1]")
    return result


def json_object(
    value: Any,
    label: str,
    *,
    forbidden_fields: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    result = deepcopy(dict(value))
    if forbidden_fields:
        reject_fields(
            result,
            path=label,
            forbidden_fields=forbidden_fields,
        )
    try:
        return json.loads(canonical_json(result))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON data") from exc


def reject_fields(
    value: Any,
    *,
    path: str,
    forbidden_fields: frozenset[str],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).strip().lower() in forbidden_fields:
                raise ValueError(
                    f"{path} may not contain decision field {key!r}"
                )
            reject_fields(
                item,
                path=f"{path}.{key}",
                forbidden_fields=forbidden_fields,
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_fields(
                item,
                path=f"{path}[{index}]",
                forbidden_fields=forbidden_fields,
            )


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    )


def stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()[:20]
    return f"{prefix}:{digest}"


def unique_texts(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if str(value).strip()
        )
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"unsupported JSON value {type(value).__name__}")
