"""Structured severity levels for semantic-placement defects.

The metric verdict remains binary for compatibility.  Severity is an
additive diagnostic dimension on each invalid semantic-placement defect; it
does not alter benchmark weights or aggregation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


CLEAR_SEMANTIC_MISPLACEMENT = "clear_semantic_misplacement"
MATERIAL_CONTEXTUAL_MISMATCH = "material_contextual_mismatch"
PLACEMENT_SEVERITY_LEVELS = (
    CLEAR_SEMANTIC_MISPLACEMENT,
    MATERIAL_CONTEXTUAL_MISMATCH,
)

_PLACEMENT_SEVERITY_RANK = {
    MATERIAL_CONTEXTUAL_MISMATCH: 1,
    CLEAR_SEMANTIC_MISPLACEMENT: 2,
}


def placement_severity_rank(value: Any) -> int:
    """Return the stable ordinal rank for a placement severity token."""

    return _PLACEMENT_SEVERITY_RANK.get(str(value or "").strip(), 0)


def validate_placement_defect_severity(defect: dict[str, Any]) -> str:
    """Validate and return one placement defect's required severity."""

    severity = str(defect.get("severity") or "").strip()
    if severity not in PLACEMENT_SEVERITY_LEVELS:
        raise ValueError(
            "semantic-placement defects require severity equal to one of "
            f"{list(PLACEMENT_SEVERITY_LEVELS)}, got {severity!r}"
        )
    return severity


def placement_severity_summary(
    defects: Iterable[Any],
) -> dict[str, Any]:
    """Summarize valid structured placement severities without inference."""

    counts = {level: 0 for level in PLACEMENT_SEVERITY_LEVELS}
    for defect in defects:
        if not isinstance(defect, dict):
            continue
        severity = str(defect.get("severity") or "").strip()
        if severity in counts:
            counts[severity] += 1
    highest = (
        CLEAR_SEMANTIC_MISPLACEMENT
        if counts[CLEAR_SEMANTIC_MISPLACEMENT]
        else MATERIAL_CONTEXTUAL_MISMATCH
        if counts[MATERIAL_CONTEXTUAL_MISMATCH]
        else "none"
    )
    return {
        "schema_version": "semantic_placement_severity_v1",
        "highest_severity": highest,
        "counts": counts,
        "strict_failure_present": bool(
            counts[CLEAR_SEMANTIC_MISPLACEMENT]
        ),
        "extended_issue_present": bool(sum(counts.values())),
        "affects_existing_metric_score": False,
    }


__all__ = [
    "CLEAR_SEMANTIC_MISPLACEMENT",
    "MATERIAL_CONTEXTUAL_MISMATCH",
    "PLACEMENT_SEVERITY_LEVELS",
    "placement_severity_rank",
    "placement_severity_summary",
    "validate_placement_defect_severity",
]
