"""Strict, path-free helpers for generation retrieval contracts v2."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


SCHEMA_SUFFIX = "_v2"
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]*$")


class RetrievalContractError(ValueError):
    """Raised when a retrieval descriptor or resource violates its contract."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_object(path: Path, *, maximum_bytes: int = 2_000_000) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) > maximum_bytes:
        raise RetrievalContractError(f"JSON exceeds {maximum_bytes} bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise RetrievalContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise RetrievalContractError(f"non-finite JSON number is forbidden: {value}")

    try:
        value = json.loads(
            data.decode("utf-8", errors="strict"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RetrievalContractError(f"invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RetrievalContractError("JSON root must be an object")
    return value


def exact_keys(
    value: Mapping[str, Any],
    *,
    label: str,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - set(value))
    extra = sorted(set(value) - allowed)
    if missing or extra:
        raise RetrievalContractError(
            f"{label} keys differ: missing={missing}, extra={extra}"
        )


def object_value(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RetrievalContractError(f"{label} must be an object")
    return value


def array_value(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RetrievalContractError(f"{label} must be an array")
    return value


def string_value(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RetrievalContractError(f"{label} must be a non-empty string")
    return value


def identifier(value: Any, *, label: str) -> str:
    text = string_value(value, label=label)
    if not _ID_RE.fullmatch(text):
        raise RetrievalContractError(f"{label} is not a portable identifier: {text!r}")
    return text

def positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RetrievalContractError(f"{label} must be a positive integer")
    return value


def nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RetrievalContractError(f"{label} must be a non-negative integer")
    return value


def finite_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RetrievalContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RetrievalContractError(f"{label} must be a finite number")
    return result


def sha256_value(value: Any, *, label: str) -> str:
    text = string_value(value, label=label)
    if len(text) != 64:
        raise RetrievalContractError(f"{label} must be a SHA-256 hex digest")
    try:
        int(text, 16)
    except ValueError as exc:
        raise RetrievalContractError(f"{label} must be a SHA-256 hex digest") from exc
    return text.lower()


def safe_relative_path(value: Any, *, label: str) -> str:
    text = string_value(value, label=label)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise RetrievalContractError(f"{label} must be a safe relative POSIX path")
    return text
