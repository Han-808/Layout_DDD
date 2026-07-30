from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.visual_judge.evidence_sufficiency import (
    EVIDENCE_SUFFICIENCY_VERSION,
    REPAIR_CAMERA,
    SUFFICIENT,
    assess_preview_selection_sufficiency,
    assess_visual_evidence_sufficiency,
)
from benchmark.visual_judge.interfaces import (
    EvidenceGateRequest,
    EvidenceGateResult,
)
from benchmark.visual_judge.visual_config import DEFAULT_P0B_VISUAL_CONFIGS


EVIDENCE_GATE_VERSION = "deterministic_evidence_gate_v1"


class DeterministicEvidenceGate:
    """Fast technical readiness checks; this class never invokes a model."""

    backend = "deterministic"

    def check(self, request: EvidenceGateRequest) -> EvidenceGateResult:
        if not isinstance(request, EvidenceGateRequest):
            raise TypeError("EvidenceGate requires an EvidenceGateRequest")

        items = list(request.visual_evidence)
        deficiencies: list[dict[str, Any]] = []
        checks: list[str] = []
        checks_not_applicable: list[str] = []

        if request.manifest_path:
            checks.append("referenced_manifest_exists")
            if not Path(request.manifest_path).expanduser().is_file():
                deficiencies.append(
                    _deficiency("evidence_manifest_missing", "rerender")
                )
        else:
            checks_not_applicable.append("referenced_manifest_exists")

        if not items:
            deficiencies.append(
                _deficiency(
                    "visual_evidence_missing",
                    (
                        REPAIR_CAMERA
                        if request.evidence_goal.get(
                            "missing_evidence_camera_repairable"
                        )
                        is True
                        else "rerender"
                    ),
                )
            )
        else:
            checks.append("evidence_files_exist")
            deficiencies.extend(_file_deficiencies(items))
            checks.append("render_not_blank")
            deficiencies.extend(_explicit_readiness_deficiencies(items))
            checks.append("required_evidence_roles_present")
            deficiencies.extend(
                _role_count_deficiencies(items, request.evidence_goal)
            )

        metric = str(request.metric or "").strip().lower()
        metadata_items = [
            item for item in items if isinstance(item, dict)
        ]
        assessment: dict[str, Any] | None = None
        assessment_scope: str | None = None
        if metric in DEFAULT_P0B_VISUAL_CONFIGS and metadata_items:
            checks.append("metric_specific_visibility_and_packet_sufficiency")
            if len(metadata_items) == len(items):
                assessment_scope = "full_packet"
                assessment = assess_visual_evidence_sufficiency(
                    metric,
                    metadata_items,
                    request=_assessment_request(request),
                )
            else:
                assessment_scope = "metadata_subset"
                assessment = _assess_metadata_subset(
                    metric,
                    metadata_items,
                    request=request,
                )
            if assessment.get("status") != SUFFICIENT:
                deficiencies.extend(
                    deepcopy(assessment.get("deficiencies") or [])
                )
        else:
            checks_not_applicable.append(
                "metric_specific_visibility_and_packet_sufficiency"
            )

        deficiencies = _dedupe(deficiencies)
        if deficiencies:
            return _result(
                ready=False,
                deficiencies=deficiencies,
                checks=checks,
                checks_not_applicable=checks_not_applicable,
                request=request,
                assessment=assessment,
                assessment_scope=assessment_scope,
                metadata_evidence_count=len(metadata_items),
            )

        return EvidenceGateResult(
            ready=True,
            camera_repairable=False,
            reason_codes=("evidence_ready",),
            deficiencies=(),
            backend=self.backend,
            provenance={
                "schema_version": EVIDENCE_GATE_VERSION,
                "implementation_version": EVIDENCE_SUFFICIENCY_VERSION,
                "threshold_source": (
                    "existing_evidence_sufficiency_and_visual_config"
                ),
                "visual_config_id": (
                    DEFAULT_P0B_VISUAL_CONFIGS.get(metric, {}).get("config_id")
                ),
                "checks_applied": checks,
                "checks_not_applicable": checks_not_applicable,
                "assessment_scope": assessment_scope,
                "metadata_evidence_count": len(metadata_items),
                **(
                    {"assessment": deepcopy(assessment)}
                    if assessment is not None
                    else {}
                ),
            },
        )


def _assessment_request(request: EvidenceGateRequest) -> dict[str, Any]:
    result = deepcopy(request.context)
    result.setdefault("metric", request.metric)
    result.setdefault("scene", deepcopy(request.scene))
    result.setdefault("object_ids", list(request.target_ids))
    return result


def _assess_metadata_subset(
    metric: str,
    items: list[dict[str, Any]],
    *,
    request: EvidenceGateRequest,
) -> dict[str, Any]:
    """Apply existing metric-local thresholds to rich items in a mixed packet."""

    selected_view_ids: list[str] = []
    visibility_by_id: dict[str, dict[str, Any]] = {}
    poses_by_id: dict[str, dict[str, Any]] = {}
    for item in items:
        role = str(item.get("role") or "").strip().lower()
        if "global" in role:
            continue
        view_id = _metadata_view_id(item)
        if not view_id:
            continue
        if view_id not in selected_view_ids:
            selected_view_ids.append(view_id)
        visibility = item.get("visibility")
        if isinstance(visibility, dict):
            visibility_by_id[view_id] = deepcopy(visibility)
        pose = item.get("pose")
        if isinstance(pose, dict):
            poses_by_id[view_id] = deepcopy(pose)
    return assess_preview_selection_sufficiency(
        metric,
        selected_view_ids,
        visibility_by_id,
        request=_assessment_request(request),
        poses_by_id=poses_by_id,
    )


def _metadata_view_id(item: dict[str, Any]) -> str:
    value = item.get("view_id") or item.get("id")
    pose = item.get("pose")
    if value is None and isinstance(pose, dict):
        value = pose.get("id")
    return str(value or "")


def _file_deficiencies(items: list[Any]) -> list[dict[str, Any]]:
    deficiencies: list[dict[str, Any]] = []
    for item in items:
        path = _evidence_path(item)
        if path is None:
            deficiencies.append(
                _deficiency("evidence_path_missing", "rerender")
            )
            continue
        if not path.expanduser().is_file():
            deficiencies.append(
                _deficiency("evidence_file_missing", "rerender")
            )
        elif path.expanduser().stat().st_size <= 0:
            deficiencies.append(
                _deficiency("empty_render_file", "rerender")
            )
    return deficiencies


def _explicit_readiness_deficiencies(
    items: list[Any],
) -> list[dict[str, Any]]:
    deficiencies: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        visibility = (
            item.get("visibility")
            if isinstance(item.get("visibility"), dict)
            else {}
        )
        render_status = str(
            item.get("render_status") or visibility.get("status") or ""
        ).strip().lower()
        if render_status == "blank":
            deficiencies.append(_deficiency("blank_render", "rerender"))
        checks = {
            "target_visible": "target_not_visible",
            "projected_coverage_sufficient": "projected_coverage_insufficient",
            "jointly_visible": "required_targets_not_jointly_visible",
            "focus_in_frame": "focus_region_out_of_frame",
            "contact_in_frame": "contact_region_out_of_frame",
            "support_region_in_frame": "support_region_out_of_frame",
            "boundary_region_in_frame": "boundary_region_out_of_frame",
        }
        for key, reason in checks.items():
            value = (
                item.get(key)
                if item.get(key) is not None
                else visibility.get(key)
            )
            if value is False:
                deficiencies.append(_deficiency(reason, REPAIR_CAMERA))
        redundant = (
            item.get("redundant_view")
            if item.get("redundant_view") is not None
            else visibility.get("redundant_view")
        )
        if redundant is True:
            deficiencies.append(_deficiency("redundant_view", REPAIR_CAMERA))
    return deficiencies


def _role_count_deficiencies(
    items: list[Any],
    goal: dict[str, Any],
) -> list[dict[str, Any]]:
    if not isinstance(goal, dict):
        return []
    roles = [
        str(item.get("role") or "")
        for item in items
        if isinstance(item, dict)
    ]
    deficiencies: list[dict[str, Any]] = []
    required_roles = goal.get("required_roles")
    if isinstance(required_roles, list):
        for role in required_roles:
            role_name = str(role)
            if role_name and role_name not in roles:
                deficiencies.append(
                    _deficiency(
                        f"required_role_missing:{role_name}",
                        str(goal.get("missing_role_repairability") or "rerender"),
                    )
                )
    required_global = _optional_nonnegative_int(
        goal.get("required_global_view_count")
    )
    required_local = _optional_nonnegative_int(
        goal.get("required_local_view_count")
    )
    if required_global is not None:
        global_count = sum("global" in role for role in roles)
        if global_count < required_global:
            deficiencies.append(
                _deficiency("required_global_evidence_missing", "rerender")
            )
    if required_local is not None:
        local_count = sum("local" in role for role in roles)
        if local_count < required_local:
            deficiencies.append(
                _deficiency("required_local_evidence_missing", REPAIR_CAMERA)
            )
    return deficiencies


def _optional_nonnegative_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("EvidenceGate required view counts must be non-negative")
    return value


def _evidence_path(item: Any) -> Path | None:
    if isinstance(item, dict):
        value = item.get("path") or item.get("image_path")
    else:
        value = item
    if value is None or not str(value).strip():
        return None
    return Path(str(value))


def _deficiency(code: str, repairability: str) -> dict[str, str]:
    return {"code": str(code), "repairability": str(repairability)}


def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (
            str(item.get("code") or ""),
            str(item.get("repairability") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _result(
    *,
    ready: bool,
    deficiencies: list[dict[str, Any]],
    checks: list[str],
    checks_not_applicable: list[str],
    request: EvidenceGateRequest,
    assessment: dict[str, Any] | None = None,
    assessment_scope: str | None = None,
    metadata_evidence_count: int = 0,
) -> EvidenceGateResult:
    repairabilities = {
        str(item.get("repairability") or "") for item in deficiencies
    }
    camera_repairable = bool(deficiencies) and repairabilities == {
        REPAIR_CAMERA
    }
    return EvidenceGateResult(
        ready=ready,
        camera_repairable=camera_repairable,
        reason_codes=tuple(
            str(item.get("code") or "")
            for item in deficiencies
            if str(item.get("code") or "")
        ),
        deficiencies=tuple(deepcopy(deficiencies)),
        backend="deterministic",
        provenance={
            "schema_version": EVIDENCE_GATE_VERSION,
            "implementation_version": EVIDENCE_SUFFICIENCY_VERSION,
            "threshold_source": (
                "existing_evidence_sufficiency_and_visual_config"
            ),
            "visual_config_id": DEFAULT_P0B_VISUAL_CONFIGS.get(
                str(request.metric or "").lower(), {}
            ).get("config_id"),
            "checks_applied": checks,
            "checks_not_applicable": checks_not_applicable,
            "assessment_scope": assessment_scope,
            "metadata_evidence_count": metadata_evidence_count,
            **(
                {"assessment": deepcopy(assessment)}
                if assessment is not None
                else {}
            ),
        },
    )
