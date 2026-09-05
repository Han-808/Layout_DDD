"""Frozen two-message prompt construction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


_PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "prompt_protocol.txt"


@lru_cache(maxsize=1)
def protocol_text() -> str:
    text = _PROTOCOL_PATH.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        raise RuntimeError(f"prompt protocol must end with a newline: {_PROTOCOL_PATH}")
    return text[:-1]


def build_system_prompt() -> str:
    """Return the complete frozen layout protocol as the system message."""
    return protocol_text()


def build_user_prompt(description: str) -> str:
    """Return only the original SceneEval Description, byte-for-byte as text."""
    return description
