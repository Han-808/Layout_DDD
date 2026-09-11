"""Structured severity levels for semantic-placement defects.

The Judge verdict remains binary for workflow compatibility.  Severity is the
controlled semantic input to the post-hoc object-equivalent burden scorer; it
does not change the Judge verdict or trigger additional evidence acquisition.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


CLEAR_SEMANTIC_MISPLACEMENT = "clear_semantic_misplacement"
MATERIAL_CONTEXTUAL_MISMATCH = "material_contextual_mismatch"
ATYPICAL = "atypical"
IMPLAUSIBLE = "implausible"
PLACEMENT_SEVERITY_LEVELS = (
    ATYPICAL,
    IMPLAUSIBLE,
)
LEGACY_PLACEMENT_SEVERITY_LEVELS = (
    CLEAR_SEMANTIC_MISPLACEMENT,
    MATERIAL_CONTEXTUAL_MISMATCH,
)

_LEGACY_TO_CANONICAL = {
    MATERIAL_CONTEXTUAL_MISMATCH: ATYPICAL,
    CLEAR_SEMANTIC_MISPLACEMENT: IMPLAUSIBLE,
}

_PLACEMENT_SEVERITY_RANK = {
    MATERIAL_CONTEXTUAL_MISMATCH: 1,
    CLEAR_SEMANTIC_MISPLACEMENT: 2,
    ATYPICAL: 1,
    IMPLAUSIBLE: 2,
}


def placement_severity_rank(value: Any) -> int:
    """Return the stable ordinal rank for a placement severity token."""

    return _PLACEMENT_SEVERITY_RANK.get(str(value or "").strip(), 0)


def canonical_placement_severity(value: Any) -> str:
    """Normalize a legacy placement token without weakening validation."""

    severity = str(value or "").strip()
    return _LEGACY_TO_CANONICAL.get(severity, severity)


def validate_placement_defect_severity(defect: dict[str, Any]) -> str:
    """Validate and return one placement defect's required severity."""

    severity = str(defect.get("severity") or "").strip()
    if severity not in {
        *PLACEMENT_SEVERITY_LEVELS,
        *LEGACY_PLACEMENT_SEVERITY_LEVELS,
    }:
        raise ValueError(
            "semantic-placement defects require severity equal to one of "
            f"{list(PLACEMENT_SEVERITY_LEVELS)}, got {severity!r}"
        )
    return canonical_placement_severity(severity)


def placement_severity_summary(
    defects: Iterable[Any],
) -> dict[str, Any]:
    """Summarize valid structured placement severities without inference."""

    counts = {level: 0 for level in PLACEMENT_SEVERITY_LEVELS}
    for defect in defects:
        if not isinstance(defect, dict):
            continue
        severity = canonical_placement_severity(defect.get("severity"))
        if severity in counts:
            counts[severity] += 1
    highest = (
        IMPLAUSIBLE
        if counts[IMPLAUSIBLE]
        else ATYPICAL
        if counts[ATYPICAL]
        else "none"
    )
    return {
        "schema_version": "object_equivalent_burden_v1",
        "highest_severity": highest,
        "counts": counts,
        "strict_failure_present": bool(
            counts[IMPLAUSIBLE]
        ),
        "extended_issue_present": bool(sum(counts.values())),
        "affects_existing_metric_score": True,
        "legacy_aliases_accepted": dict(_LEGACY_TO_CANONICAL),
    }


__all__ = [
    "ATYPICAL",
    "CLEAR_SEMANTIC_MISPLACEMENT",
    "IMPLAUSIBLE",
    "LEGACY_PLACEMENT_SEVERITY_LEVELS",
    "MATERIAL_CONTEXTUAL_MISMATCH",
    "PLACEMENT_SEVERITY_LEVELS",
    "canonical_placement_severity",
    "placement_severity_rank",
    "placement_severity_summary",
    "validate_placement_defect_severity",
]
