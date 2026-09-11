"""Benchmark-owned visual style specification for the Visual Quality judge.

Visual Quality is a VLM-primary category evaluated over standardized renders.
Without a frozen statement of what the appearance is supposed to be, the judge
can only apply generic taste. This module carries that statement.

The spec is benchmark-owned in exactly the same sense as a reference annotation:
it is authored ahead of generation, hash-verified through the case bundle, and
never derived from generator output or from runtime prompt parsing. Compilation
into judge text is deterministic so the same spec always produces the same
request payload.
"""

from __future__ import annotations

from typing import Any


VISUAL_STYLE_SPEC_VERSION = "visual_style_spec_v1"

TRUSTED_STYLE_SPEC_SOURCES = ("benchmark_annotation", "benchmark_owned", "trusted_case_bundle")
STYLE_SPEC_SOURCES = TRUSTED_STYLE_SPEC_SOURCES + ("manual", "programmatic", "diagnostic")


class VisualStyleSpecError(ValueError):
    """Raised when a visual style specification is structurally invalid."""


def validate_visual_style_spec(spec: Any, *, require_trusted_source: bool = False) -> dict[str, Any]:
    """Validate a visual style spec and return it unchanged."""

    if not isinstance(spec, dict):
        raise VisualStyleSpecError("visual style spec must be a JSON object")
    if spec.get("spec_version") != VISUAL_STYLE_SPEC_VERSION:
        raise VisualStyleSpecError(
            f"visual style spec_version must be {VISUAL_STYLE_SPEC_VERSION!r}"
        )
    source = spec.get("source")
    if source not in STYLE_SPEC_SOURCES:
        raise VisualStyleSpecError(
            f"visual style spec source must be one of {sorted(STYLE_SPEC_SOURCES)}"
        )
    if require_trusted_source and source not in TRUSTED_STYLE_SPEC_SOURCES:
        raise VisualStyleSpecError(
            f"official visual style spec source must be one of {sorted(TRUSTED_STYLE_SPEC_SOURCES)}"
        )
    if not isinstance(spec.get("frozen"), bool):
        raise VisualStyleSpecError("visual style spec frozen must be boolean")
    if require_trusted_source and spec.get("frozen") is not True:
        raise VisualStyleSpecError("official visual style spec must be frozen")
    directives = spec.get("directives")
    if not isinstance(directives, list) or not directives:
        raise VisualStyleSpecError("visual style spec directives must be a non-empty list")
    seen: set[str] = set()
    for index, directive in enumerate(directives):
        path = f"visual style spec directives[{index}]"
        if not isinstance(directive, dict):
            raise VisualStyleSpecError(f"{path} must be a JSON object")
        directive_id = str(directive.get("directive_id") or "").strip()
        if not directive_id:
            raise VisualStyleSpecError(f"{path}.directive_id must be a non-empty string")
        if directive_id in seen:
            raise VisualStyleSpecError(f"{path}.directive_id duplicates {directive_id!r}")
        seen.add(directive_id)
        if not str(directive.get("statement") or "").strip():
            raise VisualStyleSpecError(f"{path}.statement must be a non-empty string")
        if "required" in directive and not isinstance(directive.get("required"), bool):
            raise VisualStyleSpecError(f"{path}.required must be boolean")
    return spec


def compile_visual_style_prompt(spec: dict[str, Any]) -> str:
    """Render a validated spec into deterministic judge text."""

    validate_visual_style_spec(spec)
    scene_type = str(spec.get("scene_type") or "").strip()
    header = (
        f"Visual style specification for scene_type={scene_type}."
        if scene_type
        else "Visual style specification."
    )
    lines = [
        header,
        "Judge the rendered evidence against every directive below.",
    ]
    for index, directive in enumerate(spec["directives"], start=1):
        marker = "required" if directive.get("required", True) else "optional"
        statement = str(directive["statement"]).strip()
        lines.append(f"{index}. [{marker}] {statement}")
    return "\n".join(lines)


def visual_style_spec_summary(spec: dict[str, Any] | None) -> dict[str, Any]:
    """Return non-scoring provenance metadata for the evaluation report."""

    if spec is None:
        return {"available": False, "spec_version": None, "source": None, "directive_count": 0}
    return {
        "available": True,
        "spec_version": spec.get("spec_version"),
        "source": spec.get("source"),
        "frozen": bool(spec.get("frozen")),
        "scene_type": spec.get("scene_type"),
        "directive_count": len(spec.get("directives") or []),
        "directive_ids": [str(item.get("directive_id")) for item in spec.get("directives") or []],
    }
