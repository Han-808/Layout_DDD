from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def validate_generic_visual_response(result: dict[str, Any]) -> dict[str, Any]:
    """Validate the legacy generic visual scoring response."""

    applicable = result.get("applicable", True)
    if not isinstance(applicable, bool):
        raise ValueError("VLM judge applicable must be boolean")
    if applicable:
        _score(result.get("score"), "score")
    elif result.get("score") is not None:
        raise ValueError("VLM judge score must be null when applicable is false")
    if result.get("confidence") is not None:
        _score(result.get("confidence"), "confidence")
    result["applicable"] = applicable
    return result


def validate_canonical_metric_response(
    result: dict[str, Any],
    *,
    allowed_scopes: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate the strict canonical metric verdict contract."""

    evidence_status = result.get("evidence_status")
    verdict = result.get("verdict")
    if evidence_status not in {"sufficient", "insufficient"}:
        raise ValueError(
            "canonical metric evidence_status must be sufficient or insufficient"
        )
    if verdict not in {"valid", "invalid", "ambiguous"}:
        raise ValueError(
            "canonical metric verdict must be valid, invalid, or ambiguous"
        )
    if evidence_status == "insufficient" and verdict != "ambiguous":
        raise ValueError(
            "insufficient canonical metric evidence requires verdict=ambiguous"
        )
    _score(result.get("confidence"), "confidence")
    defects = result.get("defects")
    if not isinstance(defects, list):
        raise ValueError("canonical metric defects must be a JSON list")
    if verdict == "invalid" and not defects:
        raise ValueError(
            "canonical metric invalid verdict requires an explicit significant defect"
        )
    if verdict == "valid" and defects:
        raise ValueError(
            "canonical metric valid verdict cannot retain defect records"
        )
    allowed = set(allowed_scopes)
    for defect in defects:
        if not isinstance(defect, dict):
            raise ValueError(
                "canonical metric defects must contain JSON objects"
            )
        if not str(defect.get("scope") or "").strip():
            raise ValueError(
                "canonical metric defects must identify their metric scope"
            )
        if allowed and defect.get("scope") not in allowed:
            raise ValueError(
                "canonical metric defect scope is outside the requested metric"
            )
        if not str(defect.get("reason") or "").strip():
            raise ValueError(
                "canonical metric defects must explain the significant defect"
            )
        target_ids = defect.get("target_ids")
        if (
            not isinstance(target_ids, list)
            or not target_ids
            or any(
                not isinstance(item, str) or not item.strip()
                for item in target_ids
            )
        ):
            raise ValueError(
                "canonical metric defects must identify non-empty target_ids"
            )
        if not str(defect.get("relation") or "").strip():
            raise ValueError(
                "canonical metric defects must identify the defective relation"
            )
    missing_evidence = result.get("missing_evidence")
    if not isinstance(missing_evidence, list):
        raise ValueError("canonical metric missing_evidence must be a JSON list")
    if evidence_status == "insufficient" and (
        not missing_evidence
        or any(
            not isinstance(item, str) or not item.strip()
            for item in missing_evidence
        )
    ):
        raise ValueError(
            "insufficient canonical metric evidence must name missing evidence"
        )
    if evidence_status == "insufficient" and defects:
        raise ValueError(
            "insufficient canonical metric evidence cannot assert defects"
        )
    if evidence_status == "sufficient" and missing_evidence:
        raise ValueError(
            "sufficient canonical metric evidence cannot retain missing_evidence"
        )
    return result


def validate_binary_judge_response(
    result: dict[str, Any],
    *,
    judge_label: str,
    confidence_label: str | None = None,
    confidence_required: bool = True,
) -> dict[str, Any]:
    """Validate a fail-closed valid/invalid judge response."""

    if result.get("verdict") not in {"valid", "invalid"}:
        raise ValueError(
            f"{judge_label} verdict must be exactly 'valid' or 'invalid'"
        )
    confidence = result.get("confidence")
    if confidence_required or confidence is not None:
        _score(
            confidence,
            "confidence",
            label=confidence_label or judge_label,
        )
    return result


def validate_camera_selection_response(
    result: dict[str, Any],
    *,
    available_view_ids: Iterable[str],
    max_views: int,
) -> list[str]:
    """Validate the role boundary and selected view IDs for a camera response."""

    if not isinstance(result, dict):
        raise ValueError("camera selector response must be a JSON object")
    forbidden = [key for key in ("verdict", "score") if key in result]
    if forbidden:
        raise ValueError(
            "camera selector response must not contain verdict or score"
        )
    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("camera selector response must include a non-empty reason")
    ids = result.get("selected_view_ids")
    if not isinstance(ids, list):
        raise ValueError("camera selector selected_view_ids must be a list")
    if any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError(
            "camera selector selected_view_ids must contain non-empty strings"
        )
    selected = [value.strip() for value in ids]
    if len(set(selected)) != len(selected):
        raise ValueError(
            "camera selector selected_view_ids must not contain duplicates"
        )
    available = set(available_view_ids)
    if (
        not selected
        or len(selected) > int(max_views)
        or any(value not in available for value in selected)
    ):
        raise ValueError("camera selector returned invalid selected_view_ids")
    return selected


def _score(value: Any, name: str, *, label: str = "VLM judge") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} {name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} {name} must be between 0 and 1")
    return result
