"""Relocatable access to package-owned runtime resources.

Explicit caller paths remain outside this module.  Runtime defaults resolve to
the packaged copy first and retain the historical checkout-root path only as a
compatibility fallback for source layouts that predate packaged resources.
"""

from __future__ import annotations

import atexit
from contextlib import ExitStack
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path, PurePosixPath
from threading import RLock


_PACKAGE = "benchmark._resources"
_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
_RESOURCE_STACK = ExitStack()
_RESOURCE_LOCK = RLock()
atexit.register(_RESOURCE_STACK.close)


def _normalized_relative_path(relative: str | Path) -> PurePosixPath:
    text = str(relative).replace("\\", "/").strip()
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"runtime resource path must be package-relative: {relative!r}")
    return path


@lru_cache(maxsize=None)
def packaged_resource_path(relative: str | Path) -> Path:
    """Materialize and return one immutable package resource for this process."""

    normalized = _normalized_relative_path(relative)
    traversable = files(_PACKAGE).joinpath(*normalized.parts)
    if not traversable.is_file():
        raise FileNotFoundError(f"packaged runtime resource is missing: {normalized}")
    with _RESOURCE_LOCK:
        return _RESOURCE_STACK.enter_context(as_file(traversable))


@lru_cache(maxsize=None)
def runtime_resource_path(relative: str | Path) -> Path:
    """Resolve a default runtime resource with package-first precedence."""

    normalized = _normalized_relative_path(relative)
    try:
        return packaged_resource_path(normalized)
    except FileNotFoundError as packaged_error:
        checkout_path = _CHECKOUT_ROOT.joinpath(*normalized.parts)
        if checkout_path.is_file():
            return checkout_path
        raise FileNotFoundError(
            f"runtime resource is unavailable in package and checkout: {normalized}"
        ) from packaged_error


__all__ = ["packaged_resource_path", "runtime_resource_path"]
