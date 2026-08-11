from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any


class ResponseSchemaRepairError(ValueError):
    """Raised after the one allowed response-schema repair also fails."""

    def __init__(
        self,
        message: str,
        *,
        schema_audit: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.schema_audit = deepcopy(schema_audit)


def response_schema_audit_from_exception(
    error: BaseException,
) -> dict[str, Any] | None:
    """Recover response-schema audit data through wrapped exception chains."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        audit = getattr(current, "schema_audit", None)
        if isinstance(audit, dict):
            return deepcopy(audit)
        current = current.__cause__ or current.__context__
    return None


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
    allowed_missing_observations: Iterable[str] = (),
    allowed_defect_target_ids: Iterable[str] = (),
    allowed_evidence_request_target_ids: Iterable[str] = (),
    required_defect_fields: Iterable[str] = (),
    allowed_defect_field_values: (
        Mapping[str, Iterable[str]] | None
    ) = None,
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
    allowed_defect_targets = set(allowed_defect_target_ids)
    allowed_evidence_targets = set(
        allowed_evidence_request_target_ids
    )
    required_fields = {
        str(field).strip()
        for field in required_defect_fields
        if str(field).strip()
    }
    field_values = {
        str(field): {
            str(item)
            for item in values
        }
        for field, values in (
            allowed_defect_field_values or {}
        ).items()
    }
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
        if allowed_defect_targets:
            outside_targets = sorted(
                set(target_ids) - allowed_defect_targets
            )
            if outside_targets:
                raise ValueError(
                    "canonical metric defect target_ids are outside the "
                    f"requested evidence scope: {outside_targets}"
                )
        if not str(defect.get("relation") or "").strip():
            raise ValueError(
                "canonical metric defects must identify the defective relation"
            )
        missing_fields = sorted(
            field
            for field in required_fields
            if field not in defect
            or (
                isinstance(defect.get(field), str)
                and not str(defect.get(field) or "").strip()
            )
            or defect.get(field) is None
        )
        if missing_fields:
            raise ValueError(
                "canonical metric defects are missing required fields: "
                f"{missing_fields}"
            )
        for field, allowed_values in field_values.items():
            if field not in defect:
                continue
            if str(defect[field]) not in allowed_values:
                raise ValueError(
                    f"canonical metric defect {field} must be one of "
                    f"{sorted(allowed_values)}, got {defect[field]!r}"
                )
    missing_evidence = result.get("missing_evidence")
    if not isinstance(missing_evidence, list):
        raise ValueError("canonical metric missing_evidence must be a JSON list")
    if any(
        not isinstance(item, str) or not item.strip()
        for item in missing_evidence
    ):
        raise ValueError(
            "canonical metric missing_evidence must contain non-empty strings"
        )
    allowed_observations = set(allowed_missing_observations)
    if allowed_observations:
        unknown_missing = sorted(
            set(missing_evidence) - allowed_observations
        )
        if unknown_missing:
            raise ValueError(
                "canonical metric missing_evidence must use exact allowed "
                f"Camera DSL tokens; unknown {unknown_missing}"
            )
    evidence_request = result.get("evidence_request")
    if evidence_status == "insufficient":
        if evidence_request is None:
            raise ValueError(
                "insufficient canonical metric evidence requires a "
                "structured evidence_request"
            )
        _validate_canonical_evidence_request(
            evidence_request,
            allowed_missing_observations=allowed_observations,
            allowed_target_ids=allowed_evidence_targets,
        )
        requested_missing = evidence_request["missing_observations"]
        if missing_evidence and list(dict.fromkeys(missing_evidence)) != list(
            dict.fromkeys(requested_missing)
        ):
            raise ValueError(
                "canonical metric missing_evidence conflicts with "
                "evidence_request.missing_observations"
            )
    elif evidence_request is not None:
        raise ValueError(
            "sufficient canonical metric evidence cannot retain an "
            "evidence_request"
        )
    if evidence_status == "insufficient" and defects:
        raise ValueError(
            "insufficient canonical metric evidence cannot assert defects"
        )
    return result


def _validate_canonical_evidence_request(
    value: Any,
    *,
    allowed_missing_observations: set[str],
    allowed_target_ids: set[str],
) -> None:
    if not isinstance(value, dict):
        raise ValueError(
            "canonical metric evidence_request must be a JSON object"
        )
    unknown_keys = set(value) - {
        "target_ids",
        "missing_observations",
        "view_goal",
        "metadata",
    }
    if unknown_keys:
        raise ValueError(
            "canonical metric evidence_request contains unsupported fields: "
            f"{sorted(unknown_keys)}"
        )
    target_ids = value.get("target_ids")
    if (
        not isinstance(target_ids, list)
        or not target_ids
        or any(
            not isinstance(item, str) or not item.strip()
            for item in target_ids
        )
        or len(target_ids) != len(set(target_ids))
    ):
        raise ValueError(
            "canonical metric evidence_request.target_ids must contain "
            "unique non-empty strings"
        )
    if allowed_target_ids:
        outside = sorted(set(target_ids) - allowed_target_ids)
        if outside:
            raise ValueError(
                "canonical metric evidence_request target IDs are outside "
                f"the authoritative scope: {outside}"
            )
    observations = value.get("missing_observations")
    if (
        not isinstance(observations, list)
        or not observations
        or any(
            not isinstance(item, str) or not item.strip()
            for item in observations
        )
        or len(observations) != len(set(observations))
    ):
        raise ValueError(
            "canonical metric evidence_request.missing_observations must "
            "contain unique non-empty strings"
        )
    if allowed_missing_observations:
        unknown = sorted(
            set(observations) - allowed_missing_observations
        )
        if unknown:
            raise ValueError(
                "canonical metric evidence_request must use exact allowed "
                f"Camera DSL tokens; unknown {unknown}"
            )
    if not str(value.get("view_goal") or "").strip():
        raise ValueError(
            "canonical metric evidence_request.view_goal must be non-empty"
        )
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(
            "canonical metric evidence_request.metadata must be a JSON object"
        )
    forbidden_authority = {
        "metric",
        "task",
        "rubric",
        "metric_scope",
        "judgment_scope",
        "camera_constraints",
        "constraints",
        "camera_repair_plans",
        "repair_plan",
        "selected_plan_id",
        "relaxable_constraints",
        "preferred_view_families",
        "forbidden_view_families",
        "camera_proposal",
        "pose",
    } & set(metadata)
    if forbidden_authority:
        raise ValueError(
            "canonical metric evidence_request metadata cannot redefine "
            "metric scope or camera policy: "
            f"{sorted(forbidden_authority)}"
        )


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
