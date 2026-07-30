from __future__ import annotations

from copy import deepcopy
import json
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


EVIDENCE_GATE_VERSION = "deterministic_evidence_gate_v2"

CANONICAL_TECHNICAL_REQUIREMENTS: dict[str, dict[str, Any]] = {
    "functional_semantic_fidelity": {
        "require_context": True,
        "require_target_visibility": True,
        "require_projected_coverage": True,
        "require_joint_visibility": True,
        "required_global_view_count": 1,
        "required_local_view_count": 0,
        "require_view_redundancy_check": True,
    },
    "scale_consistency": {
        "require_context": True,
        "require_target_visibility": True,
        "require_projected_coverage": True,
        "require_joint_visibility": False,
        "required_global_view_count": 0,
        "required_local_view_count": 1,
        "require_view_redundancy_check": True,
    },
    "object_pairing_consistency": {
        "require_context": True,
        "require_target_visibility": True,
        "require_projected_coverage": True,
        "require_joint_visibility": True,
        "required_global_view_count": 1,
        "required_local_view_count": 1,
        "require_view_redundancy_check": True,
    },
    "style_consistency": {
        "require_context": True,
        "require_target_visibility": False,
        "require_projected_coverage": False,
        "require_joint_visibility": False,
        "required_global_view_count": 1,
        "required_local_view_count": 0,
        "require_view_redundancy_check": True,
    },
}
TECHNICAL_REQUIREMENT_KEYS = frozenset(
    next(iter(CANONICAL_TECHNICAL_REQUIREMENTS.values()))
)


class DeterministicEvidenceGate:
    """Fast technical readiness checks; this class never invokes a model."""

    backend = "deterministic"

    def __init__(
        self,
        *,
        allow_path_only_compatibility: bool = False,
        metric_requirements: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not isinstance(allow_path_only_compatibility, bool):
            raise TypeError(
                "EvidenceGate allow_path_only_compatibility must be boolean"
            )
        if metric_requirements is not None and not isinstance(
            metric_requirements, dict
        ):
            raise TypeError("EvidenceGate metric_requirements must be a mapping")
        unknown_metrics = set(metric_requirements or {}) - set(
            CANONICAL_TECHNICAL_REQUIREMENTS
        )
        if unknown_metrics:
            raise ValueError(
                "EvidenceGate has requirements for unknown metrics: "
                f"{sorted(unknown_metrics)}"
            )
        for metric, requirement_patch in (metric_requirements or {}).items():
            _validate_technical_requirement_patch(
                requirement_patch,
                label=f"EvidenceGate requirements for {metric}",
            )
        self.allow_path_only_compatibility = allow_path_only_compatibility
        self.metric_requirements = deepcopy(metric_requirements or {})

    def check(self, request: EvidenceGateRequest) -> EvidenceGateResult:
        if not isinstance(request, EvidenceGateRequest):
            raise TypeError("EvidenceGate requires an EvidenceGateRequest")

        raw_items = list(request.visual_evidence)
        (
            items,
            manifest_metadata_count,
            manifest_failure,
        ) = _with_manifest_metadata(
            raw_items,
            request.manifest_path,
        )
        deficiencies: list[dict[str, Any]] = []
        checks: list[str] = []
        checks_not_applicable: list[str] = []

        if request.manifest_path:
            checks.append("referenced_manifest_exists")
            if manifest_failure is not None:
                deficiencies.append(
                    _deficiency(manifest_failure, "manifest")
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
        technical_assessment: dict[str, Any] | None = None
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
        elif metric not in DEFAULT_P0B_VISUAL_CONFIGS:
            checks_not_applicable.append(
                "metric_specific_visibility_and_packet_sufficiency"
            )
        if metric in CANONICAL_TECHNICAL_REQUIREMENTS:
            path_only_compatibility = (
                self.allow_path_only_compatibility
                or request.evidence_goal.get(
                    "allow_path_only_compatibility"
                )
                is True
            )
            if path_only_compatibility:
                checks_not_applicable.append(
                    "canonical_technical_evidence_requirements"
                )
                technical_assessment = {
                    "status": "compatibility_bypass",
                    "allow_path_only_compatibility": True,
                }
            else:
                checks.append("canonical_technical_evidence_requirements")
                requirements, requirement_sources = (
                    _resolved_technical_requirements(
                        metric,
                        request=request,
                        injected=self.metric_requirements.get(metric),
                    )
                )
                technical_deficiencies = _technical_evidence_deficiencies(
                    items,
                    request=request,
                    requirements=requirements,
                )
                deficiencies.extend(technical_deficiencies)
                technical_assessment = {
                    "status": (
                        "ready"
                        if not technical_deficiencies
                        else "not_ready"
                    ),
                    "requirements": deepcopy(requirements),
                    "requirement_sources": deepcopy(
                        requirement_sources
                    ),
                    "metadata_evidence_count": len(metadata_items),
                    "manifest_metadata_count": manifest_metadata_count,
                    "allow_path_only_compatibility": False,
                }
        else:
            checks_not_applicable.append(
                "canonical_technical_evidence_requirements"
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
                technical_assessment=technical_assessment,
                manifest_metadata_count=manifest_metadata_count,
                manifest_failure=manifest_failure,
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
                "manifest_metadata_count": manifest_metadata_count,
                "manifest_status": (
                    "not_provided"
                    if request.manifest_path is None
                    else "valid"
                ),
                "allow_path_only_compatibility": (
                    self.allow_path_only_compatibility
                ),
                **(
                    {"assessment": deepcopy(assessment)}
                    if assessment is not None
                    else {}
                ),
                **(
                    {
                        "technical_assessment": deepcopy(
                            technical_assessment
                        )
                    }
                    if technical_assessment is not None
                    else {}
                ),
            },
        )


def _resolved_technical_requirements(
    metric: str,
    *,
    request: EvidenceGateRequest,
    injected: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    requirements = deepcopy(CANONICAL_TECHNICAL_REQUIREMENTS[metric])
    sources = {key: "metric_default" for key in requirements}
    if injected is not None:
        _validate_technical_requirement_patch(
            injected,
            label=f"EvidenceGate requirements for {metric}",
        )
        requirements.update(deepcopy(injected))
        sources.update({key: "gate_injection" for key in injected})
    explicit = request.evidence_goal.get("technical_requirements")
    if explicit is not None:
        _validate_technical_requirement_patch(
            explicit,
            label="EvidenceGate technical_requirements",
        )
        requirements.update(deepcopy(explicit))
        sources.update({key: "request_override" for key in explicit})

    phase = str(request.context.get("evidence_phase") or "").strip().lower()
    if metric == "functional_semantic_fidelity" and (
        phase in {"local_confirmation", "final"}
        or request.context.get("relevant_local_visual_evidence")
    ):
        requirements["required_local_view_count"] = max(
            1,
            _required_count(
                requirements.get("required_local_view_count"),
                "required_local_view_count",
            ),
        )
        sources["required_local_view_count"] = "evidence_phase_policy"

    for key in (
        "require_context",
        "require_target_visibility",
        "require_projected_coverage",
        "require_joint_visibility",
        "require_view_redundancy_check",
    ):
        if not isinstance(requirements.get(key), bool):
            raise ValueError(
                f"EvidenceGate technical requirement {key} must be boolean"
            )
    for key in (
        "required_global_view_count",
        "required_local_view_count",
    ):
        requirements[key] = _required_count(requirements.get(key), key)
    return requirements, sources


def _validate_technical_requirement_patch(
    value: Any,
    *,
    label: str,
) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a mapping")
    unknown = set(value) - TECHNICAL_REQUIREMENT_KEYS
    if unknown:
        raise ValueError(
            f"{label} contains unknown fields: {sorted(unknown)}"
        )


def _technical_evidence_deficiencies(
    items: list[Any],
    *,
    request: EvidenceGateRequest,
    requirements: dict[str, Any],
) -> list[dict[str, Any]]:
    repairability = _technical_repairability(request)
    deficiencies: list[dict[str, Any]] = []
    metadata_items = [item for item in items if isinstance(item, dict)]
    targets = tuple(
        target
        for target in request.target_ids
        if target and target != "scene"
    )

    if requirements["require_context"] and not request.scene:
        deficiencies.append(
            _deficiency("required_scene_context_missing", "context")
        )

    roles = [
        {
            str(item.get("role") or "").strip().lower()
            for item in group
            if str(item.get("role") or "").strip()
        }
        for group in _technical_camera_view_groups(metadata_items)
    ]
    global_count = sum(
        any("global" in role for role in group_roles)
        for group_roles in roles
    )
    local_count = sum(
        any(
            any(
                token in role
                for token in ("local", "focus", "group", "pair")
            )
            and "global" not in role
            for role in group_roles
        )
        and not any("global" in role for role in group_roles)
        for group_roles in roles
    )
    inferred_global, inferred_local = _conservative_role_counts(
        items,
        request=request,
        targets=targets,
    )
    global_count = max(global_count, inferred_global)
    local_count = max(local_count, inferred_local)
    if global_count < requirements["required_global_view_count"]:
        deficiencies.append(
            _deficiency("required_global_evidence_metadata_missing", repairability)
        )
    if local_count < requirements["required_local_view_count"]:
        deficiencies.append(
            _deficiency("required_local_evidence_metadata_missing", repairability)
        )

    if requirements["require_target_visibility"] and targets:
        visible_targets = _visible_targets(metadata_items, targets)
        if not metadata_items:
            deficiencies.append(
                _deficiency("target_visibility_metadata_missing", repairability)
            )
        elif set(targets) - visible_targets:
            deficiencies.append(
                _deficiency("target_visibility_not_established", repairability)
            )

    if requirements["require_projected_coverage"] and targets:
        coverage = [
            _technical_value(item, "projected_coverage_sufficient")
            for item in metadata_items
        ]
        if not any(value is True for value in coverage):
            code = (
                "projected_coverage_insufficient"
                if any(value is False for value in coverage)
                else "projected_coverage_metadata_missing"
            )
            deficiencies.append(_deficiency(code, repairability))

    if (
        requirements["require_joint_visibility"]
        and len(targets) > 1
    ):
        joint_values = [
            _technical_value(item, "jointly_visible")
            for item in metadata_items
        ]
        inferred_joint = any(
            not (set(targets) - _visible_targets([item], targets))
            for item in metadata_items
        )
        if not any(value is True for value in joint_values) and not inferred_joint:
            code = (
                "required_targets_not_jointly_visible"
                if any(value is False for value in joint_values)
                else "joint_visibility_metadata_missing"
            )
            deficiencies.append(_deficiency(code, repairability))

    if (
        requirements["require_view_redundancy_check"]
        and len(_technical_camera_view_groups(metadata_items)) > 1
    ):
        groups = _technical_camera_view_groups(metadata_items)
        redundancy = [
            _group_boolean_measurement(group, "redundant_view")
            for group in groups
        ]
        if any(value is True for value in redundancy):
            deficiencies.append(_deficiency("redundant_view", repairability))
        elif (
            len(metadata_items) != len(items)
            or len(redundancy) != len(groups)
            or any(value is None for value in redundancy)
        ):
            deficiencies.append(
                _deficiency("view_redundancy_metadata_missing", repairability)
            )
    return deficiencies


def _technical_camera_view_groups(
    items: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, item in enumerate(items):
        pose = item.get("pose")
        view_id = str(item.get("view_id") or "").strip()
        pair_id = str(item.get("pair_id") or "").strip()
        if isinstance(pose, dict) and pose:
            identity = "pose:" + json.dumps(
                pose,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        elif view_id:
            identity = f"view:{view_id}"
        elif pair_id:
            identity = f"pair:{pair_id}"
        else:
            path = _evidence_path(item)
            identity = (
                f"path:{path}"
                if path is not None
                else f"item:{index}"
            )
        groups.setdefault(identity, []).append(item)
    return list(groups.values())


def _group_boolean_measurement(
    items: list[dict[str, Any]],
    key: str,
) -> bool | None:
    values = [_technical_value(item, key) for item in items]
    if any(value is True for value in values):
        return True
    if any(value is False for value in values):
        return False
    return None


def _conservative_role_counts(
    items: list[Any],
    *,
    request: EvidenceGateRequest,
    targets: tuple[str, ...],
) -> tuple[int, int]:
    """Reuse existing metric packet policy without inventing image semantics."""

    if not items:
        return 0, 0
    metric = str(request.metric or "").strip().lower()
    context = request.context
    paths = {
        str(_evidence_path(item))
        for item in items
        if _evidence_path(item) is not None
    }
    global_refs = context.get("relevant_global_visual_evidence")
    local_refs = context.get("relevant_local_visual_evidence")
    global_count = (
        sum(str(Path(str(value))) in paths for value in global_refs)
        if isinstance(global_refs, list)
        else 0
    )
    local_count = (
        sum(str(Path(str(value))) in paths for value in local_refs)
        if isinstance(local_refs, list)
        else 0
    )
    if metric == "style_consistency" and not targets and len(items) == 1:
        global_count = max(global_count, 1)
    if (
        metric == "functional_semantic_fidelity"
        and not targets
        and len(items) == 1
    ):
        global_count = max(global_count, 1)
    return global_count, local_count


def _visible_targets(
    items: list[dict[str, Any]],
    targets: tuple[str, ...],
) -> set[str]:
    visible: set[str] = set()
    for item in items:
        fractions = _technical_value(item, "target_pixel_fractions")
        if isinstance(fractions, dict):
            for target in targets:
                value = fractions.get(target)
                if (
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and float(value) > 0.0
                ):
                    visible.add(target)
        target_visible = _technical_value(item, "target_visible")
        if target_visible is True:
            item_targets = item.get("target_ids")
            if isinstance(item_targets, list):
                visible.update(
                    target
                    for target in targets
                    if target in {str(value) for value in item_targets}
                )
            elif len(targets) == 1:
                visible.add(targets[0])
    return visible


def _technical_value(item: dict[str, Any], key: str) -> Any:
    if item.get(key) is not None:
        return item.get(key)
    visibility = item.get("visibility")
    if isinstance(visibility, dict):
        return visibility.get(key)
    return None


def _technical_repairability(request: EvidenceGateRequest) -> str:
    return (
        REPAIR_CAMERA
        if request.evidence_goal.get("missing_evidence_camera_repairable")
        is True
        else "rerender"
    )


def _required_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"EvidenceGate technical requirement {label} must be non-negative"
        )
    return value


def _with_manifest_metadata(
    items: list[Any],
    manifest_path: str | None,
) -> tuple[list[Any], int, str | None]:
    if not manifest_path:
        return list(deepcopy(items)), 0, None
    path = Path(manifest_path).expanduser()
    if not path.is_file():
        return (
            list(deepcopy(items)),
            0,
            "evidence_manifest_missing",
        )
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return (
            list(deepcopy(items)),
            0,
            "evidence_manifest_unreadable",
        )
    except json.JSONDecodeError:
        return (
            list(deepcopy(items)),
            0,
            "evidence_manifest_invalid",
        )
    if not isinstance(manifest, dict):
        return (
            list(deepcopy(items)),
            0,
            "evidence_manifest_invalid",
        )
    manifest_items: list[dict[str, Any]] = []
    found_evidence_list = False
    for key in ("render_evidence_items", "visual_evidence", "views"):
        if key not in manifest:
            continue
        found_evidence_list = True
        values = manifest[key]
        if not isinstance(values, list) or any(
            not isinstance(item, dict) for item in values
        ):
            return (
                list(deepcopy(items)),
                0,
                "evidence_manifest_invalid",
            )
        manifest_items.extend(values)
    if not found_evidence_list or not manifest_items:
        return (
            list(deepcopy(items)),
            0,
            "evidence_manifest_evidence_items_missing",
        )
    by_path = {
        str(_evidence_path(item)): item
        for item in manifest_items
        if _evidence_path(item) is not None
    }
    enriched: list[Any] = []
    matches = 0
    for item in items:
        item_path = _evidence_path(item)
        metadata = by_path.get(str(item_path)) if item_path is not None else None
        if metadata is None:
            enriched.append(deepcopy(item))
            continue
        matches += 1
        if isinstance(item, dict):
            merged = deepcopy(metadata)
            merged.update(deepcopy(item))
            enriched.append(merged)
        else:
            enriched.append(deepcopy(metadata))
    if items and matches == 0:
        return (
            list(deepcopy(items)),
            0,
            "evidence_manifest_evidence_mismatch",
        )
    return enriched, matches, None


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
        elif render_status in {
            "corrupt",
            "corrupted",
            "failed",
            "failure",
            "error",
            "invalid",
        }:
            deficiencies.append(
                _deficiency("corrupt_render_evidence", "render")
            )
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
    technical_assessment: dict[str, Any] | None = None,
    manifest_metadata_count: int = 0,
    manifest_failure: str | None = None,
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
            "manifest_metadata_count": manifest_metadata_count,
            "manifest_status": (
                manifest_failure
                or (
                    "valid"
                    if request.manifest_path is not None
                    else "not_provided"
                )
            ),
            **(
                {"assessment": deepcopy(assessment)}
                if assessment is not None
                else {}
            ),
            **(
                {
                    "technical_assessment": deepcopy(
                        technical_assessment
                    )
                }
                if technical_assessment is not None
                else {}
            ),
        },
    )
