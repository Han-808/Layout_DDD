from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.visual_judge.interfaces.evidence import (
    EvidenceRenderRequest,
    EvidenceRenderResult,
)
from benchmark.visual_judge.interfaces.camera import (
    CameraSelectionRequest,
    CameraSelectionResult,
)
from benchmark.visual_judge.camera_dsl import CameraConstraintSet
from benchmark.visual_judge.camera_repair import CameraRepairPlan
from benchmark.visual_judge.acquisition_state import (
    CameraAcquisitionState,
)
from benchmark.visual_judge.control_config import (
    VLMEvaluationControl,
)
from benchmark.visual_judge.experiment_telemetry import (
    CameraExperimentTelemetry,
)
from benchmark.visual_judge.interfaces.judge import (
    EvidenceRequest,
    JudgeRequest,
    JudgeResult,
)
from benchmark.visual_judge.orchestration.budget import (
    policy_for_stop_reason,
)
from benchmark.visual_judge.orchestration.camera_acquisition import (
    provenance_count,
    remaining_budget,
    selector_identity,
)


def _render_result_with_audit(
    result: EvidenceRenderResult,
    *,
    request: EvidenceRenderRequest,
) -> EvidenceRenderResult:
    provenance = deepcopy(result.provenance)
    _validate_render_cost_provenance(provenance)
    selected_ids = list(request.selection.selected_view_ids)
    reported_ids = provenance.get("selected_view_ids")
    if reported_ids is not None and reported_ids != selected_ids:
        raise ValueError(
            "evidence renderer provenance selected_view_ids must match "
            "the validated CameraSelector result"
        )
    reported_round = provenance.get("evidence_round")
    if reported_round is not None and reported_round != request.evidence_round:
        raise ValueError(
            "evidence renderer provenance evidence_round must match "
            "the current control round"
        )
    provenance.setdefault("scene_access", "read_only")
    provenance.setdefault("selected_view_ids", selected_ids)
    provenance.setdefault(
        "camera_actions",
        list(deepcopy(request.selection.camera_actions)),
    )
    provenance.setdefault("evidence_round", request.evidence_round)
    provenance.setdefault("selector_backend", request.selection.backend)
    provenance.setdefault(
        "selection_provenance",
        deepcopy(request.selection.provenance),
    )
    provenance.setdefault("budget", deepcopy(request.budget))
    return EvidenceRenderResult(
        visual_evidence=result.visual_evidence,
        merge_policy=result.merge_policy,
        camera_actions_executed=result.camera_actions_executed,
        manifest_path=result.manifest_path,
        next_candidate_views=result.next_candidate_views,
        next_allowed_actions=result.next_allowed_actions,
        replaces_candidate_views=result.replaces_candidate_views,
        replaces_allowed_actions=result.replaces_allowed_actions,
        backend=result.backend,
        provenance=provenance,
    )


def _validate_render_cost_provenance(
    provenance: dict[str, Any],
) -> None:
    for key in ("preview_render_count", "full_render_count"):
        if key not in provenance:
            continue
        value = provenance[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"evidence renderer provenance {key} must be a "
                "non-negative integer"
            )
    for key in ("render_gpu_time_seconds", "gpu_time_seconds"):
        if key not in provenance:
            continue
        value = provenance[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(
                f"evidence renderer provenance {key} must be finite and "
                "non-negative"
            )


def _evidence_fingerprint(items: list[Any]) -> str:
    payload = json.dumps(
        [_evidence_content_identity(item) for item in items],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evidence_content_identity(item: Any) -> Any:
    if isinstance(item, dict):
        raw_path = item.get("path") or item.get("image_path")
        if raw_path is not None and str(raw_path).strip():
            return _path_content_identity(raw_path)
        for key in ("image_sha256", "content_hash", "sha256"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return {"content_digest": value.strip().lower()}
        return _jsonable(item)
    if isinstance(item, (str, Path)):
        return _path_content_identity(item)
    return _jsonable(item)


def _path_content_identity(value: Any) -> dict[str, str]:
    path = Path(str(value)).expanduser()
    if not path.is_file():
        return {"path": str(path)}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"content_sha256": digest.hexdigest()}


def _evidence_refs(items: list[Any]) -> list[str]:
    refs: list[str] = []
    for index, item in enumerate(items):
        if isinstance(item, dict):
            value = (
                item.get("view_id")
                or item.get("id")
                or item.get("path")
                or item.get("image_path")
            )
        else:
            value = item
        refs.append(str(value) if value is not None else f"evidence_{index:02d}")
    return refs


def _rendered_view_count(
    items: tuple[Any, ...],
    *,
    trusted_view_ids: tuple[str, ...] = (),
) -> int:
    """Count independent camera views while keeping same-pose bundles intact."""

    identities: set[str] = set()
    trusted_ids = set(trusted_view_ids)
    for index, item in enumerate(items):
        if isinstance(item, dict):
            pose = item.get("pose")
            role = str(item.get("role") or "")
            pair_id = item.get("pair_id")
            view_id = str(item.get("view_id") or "").strip()
            if isinstance(pose, dict) and pose:
                value = "pose:" + json.dumps(
                    _jsonable(pose),
                    sort_keys=True,
                    separators=(",", ":"),
                )
            elif view_id and view_id in trusted_ids:
                value = f"trusted_view:{view_id}"
            elif (
                pair_id is not None
                and role
                in {
                    "collision_rgb",
                    "collision_pair_overlay",
                    "metric_local_contour",
                }
            ):
                value = f"verified_pair:{pair_id}"
            else:
                # A renderer-provided view_id alone is not proof that two
                # independent files share a camera pose.
                value = (
                    item.get("path")
                    or item.get("image_path")
                    or f"rendered_view_{index:02d}"
                )
        else:
            value = item
        identities.add(
            str(value)
            if value is not None
            else f"rendered_view_{index:02d}"
        )
    return len(identities)


def _rendered_evidence_refs(
    trace: list[dict[str, Any]],
) -> set[str]:
    refs: set[str] = set()
    for event in trace:
        if (
            event.get("stage") != "render"
            or event.get("status") != "completed"
        ):
            continue
        result = event.get("result")
        if not isinstance(result, dict):
            continue
        evidence = result.get("visual_evidence")
        if isinstance(evidence, list):
            refs.update(_evidence_refs(evidence))
    return refs


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("VLM evaluation audit values must be finite")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _latest_focus_target_ids(
    trace: list[dict[str, Any]],
    *,
    fallback: Any,
) -> list[str]:
    for event in reversed(trace):
        if (
            not isinstance(event, dict)
            or event.get("stage") != "acquisition_planner"
        ):
            continue
        evidence_request = event.get("evidence_request")
        if not isinstance(evidence_request, dict):
            continue
        target_ids = evidence_request.get("target_ids")
        if isinstance(target_ids, list) and target_ids:
            return [
                str(value)
                for value in target_ids
                if str(value).strip()
            ]
    if not isinstance(fallback, (list, tuple)):
        return []
    return [
        str(value)
        for value in fallback
        if str(value).strip()
    ]


def build_evaluation_audit(
    *,
    schema_version: str,
    control: VLMEvaluationControl,
    compatibility_selector: Any,
    deterministic_selector: Any | None,
    vlm_selector: Any | None,
    effective_policy: str,
    policy_source: str,
    renderer: Any,
    evidence_gate_backend: str,
    judge: Any,
    judge_result: JudgeResult | None,
    judge_request: JudgeRequest,
    final_status: str,
    final_confidence: float,
    final_evidence_request: EvidenceRequest | None,
    acquisition_state: CameraAcquisitionState,
    telemetry: CameraExperimentTelemetry,
    stop_reason: str,
    trace: list[dict[str, Any]],
    evidence: list[Any],
    selector_calls: int,
    actions_used: int,
    rounds_used: int,
    total_images_acquired: int,
) -> dict[str, Any]:
    judge_provenance = (
        deepcopy(judge_result.provenance)
        if judge_result is not None
        else {}
    )
    recovery_outcome = _evidence_recovery_outcome(
        trace,
        final_status=final_status,
    )
    telemetry_value = telemetry.to_dict(stop_reason=stop_reason)
    selection_events = [
        item
        for item in telemetry_value.get("events", [])
        if isinstance(item, dict)
        and item.get("kind") == "camera_selection"
    ]
    vlm_selection_events = [
        item
        for item in selection_events
        if item.get("stage") == "vlm"
    ]
    bank_events = [
        item
        for item in trace
        if isinstance(item, dict)
        and item.get("stage") == "trusted_candidate_bank"
        and item.get("status") == "completed"
    ]
    group_scope = judge_request.context.get("group_scope")
    group_scope = (
        group_scope if isinstance(group_scope, dict) else {}
    )
    focus_target_ids = _latest_focus_target_ids(
        trace,
        fallback=(
            judge_request.context.get("target_object_ids")
            or group_scope.get("member_ids")
            or []
        ),
    )
    return _jsonable(
        {
            "schema_version": schema_version,
            "evaluation": {
                "case_id": _case_id(judge_request),
                "task": judge_request.task,
                "metric": judge_request.metric,
                "final_status": final_status,
                "final_confidence": final_confidence,
                "deterministic_outcome": _deterministic_outcome(
                    judge_request
                ),
                "evidence_recovery_outcome": recovery_outcome,
                "final_evidence_request": (
                    final_evidence_request.to_dict()
                    if final_evidence_request is not None
                    else None
                ),
            },
            "control": control.manifest(),
            "camera_acquisition": {
                "requested_policy": control.camera_acquisition_policy,
                "effective_policy": effective_policy,
                "policy_source": policy_source,
                "state": acquisition_state.to_dict(),
                "selectors": {
                    "deterministic": selector_identity(
                        deterministic_selector
                    ),
                    "vlm": selector_identity(vlm_selector),
                },
                "remaining_budget": remaining_budget(
                    control=control,
                    selector_calls=selector_calls,
                    actions_used=actions_used,
                    rounds_used=rounds_used,
                    total_images_acquired=total_images_acquired,
                ),
            },
            "experiment_telemetry": telemetry_value,
            "preview_renderer_invoked": (
                int(telemetry_value.get("preview_render_count") or 0)
                > 0
            ),
            "preview_render_count": int(
                telemetry_value.get("preview_render_count") or 0
            ),
            "final_render_count": int(
                telemetry_value.get("full_render_count") or 0
            ),
            "production_camera_selector_backend": (
                vlm_selection_events[-1].get("selector_backend")
                if vlm_selection_events
                else None
            ),
            "effective_vlm_selection_mode": (
                vlm_selection_events[-1].get("selection_mode")
                if vlm_selection_events
                else None
            ),
            "semantic_selection_triggered": any(
                item.get("reason")
                == "semantic_selection_required"
                for item in telemetry_value.get("events", [])
                if isinstance(item, dict)
                and item.get("kind") == "camera_escalation"
            ),
            "trusted_candidate_count": (
                int(bank_events[-1].get("candidate_count") or 0)
                if bank_events
                else 0
            ),
            "group_id": group_scope.get("group_id"),
            "focus_target_ids": focus_target_ids,
            "authoritative_group_member_ids": list(
                group_scope.get("member_ids") or []
            ),
            "selector_backend": str(
                getattr(
                    compatibility_selector,
                    "backend",
                    control.camera_selector_backend,
                )
            ),
            "requested_selector_backend": control.camera_selector_backend,
            "selector_adapter": (
                f"{type(compatibility_selector).__module__}."
                f"{type(compatibility_selector).__qualname__}"
            ),
            "renderer_backend": str(
                getattr(
                    renderer,
                    "backend",
                    (
                        f"{type(renderer).__module__}."
                        f"{type(renderer).__qualname__}"
                    ),
                )
            ),
            "renderer_adapter": (
                f"{type(renderer).__module__}."
                f"{type(renderer).__qualname__}"
            ),
            "scene_access": "read_only",
            "evidence_gate_backend": evidence_gate_backend,
            "judge_backend": (
                judge_result.backend
                if judge_result is not None
                else f"{type(judge).__module__}.{type(judge).__qualname__}"
            ),
            "model": judge_provenance.get("model"),
            "endpoint": judge_provenance.get("endpoint"),
            "judge_provenance": judge_provenance,
            "rounds_used": rounds_used,
            "selector_calls_used": selector_calls,
            "camera_actions_used": actions_used,
            "initial_camera_usage": deepcopy(
                (
                    trace[0].get("initial_camera_usage")
                    if trace
                    and isinstance(trace[0], dict)
                    and isinstance(
                        trace[0].get("initial_camera_usage"),
                        dict,
                    )
                    else None
                )
            ),
            "current_packet_image_count": len(evidence),
            "total_images_acquired": total_images_acquired,
            "unique_rendered_evidence_count": len(
                _rendered_evidence_refs(trace)
            ),
            "images_used": _evidence_refs(evidence),
            "applied_failure_policy": policy_for_stop_reason(
                control,
                stop_reason,
            ),
            "trace": deepcopy(trace),
        }
    )


def record_selector_failure(
    *,
    telemetry: CameraExperimentTelemetry,
    trace: list[dict[str, Any]],
    selector: Any,
    stage: str,
    selection_request: CameraSelectionRequest,
    control: VLMEvaluationControl,
    evidence_round: int,
    episode_index: int,
    failure_kind: str,
    failure_error: str,
) -> None:
    usage = getattr(selector, "last_call_usage", None)
    has_runtime_usage = (
        stage == "vlm"
        and isinstance(usage, dict)
        and "vlm_call_count" in usage
    )
    telemetry.record_selector(
        stage=stage,
        outcome=failure_kind,
        candidate_count=len(selection_request.candidate_views),
        filtered_candidate_count=0,
        attempted_candidate_ids=(),
        selected_view_ids=(),
        attempted_plan_ids=(),
        selected_plan_id=None,
        selector_backend=str(getattr(selector, "backend", stage)),
        selection_mode=(
            str(
                selection_request.context.get(
                    "vlm_selection_mode",
                    control.vlm_selection_mode,
                )
            )
            if stage == "vlm"
            else None
        ),
        vlm_call_count=(
            provenance_count(
                usage if isinstance(usage, dict) else {},
                "vlm_call_count",
                default=1,
            )
            if stage == "vlm"
            else 0
        ),
        vlm_call_count_source=(
            "adapter_runtime_usage"
            if has_runtime_usage
            else (
                "conservative_interface_dispatch"
                if stage == "vlm"
                else "not_vlm"
            )
        ),
        evidence_round=evidence_round,
        episode_index=episode_index,
        provenance={
            "failure_kind": failure_kind,
            "error": failure_error,
        },
    )
    trace.append(
        {
            "stage": "camera_selector",
            "selection_stage": stage,
            "evidence_round": evidence_round,
            "status": "failed",
            "failure_kind": failure_kind,
            "error": failure_error,
        }
    )


def record_camera_selection(
    *,
    telemetry: CameraExperimentTelemetry,
    trace: list[dict[str, Any]],
    selection: CameraSelectionResult,
    selection_request: CameraSelectionRequest,
    repair_plans: tuple[CameraRepairPlan, ...],
    constraints: CameraConstraintSet,
    control: VLMEvaluationControl,
    stage: str,
    evidence_round: int,
    episode_index: int,
) -> None:
    selected_plan = next(
        (
            plan
            for plan in repair_plans
            if plan.plan_id == selection.selected_plan_id
        ),
        None,
    )
    telemetry.record_selector(
        stage=stage,
        outcome=selection.outcome,
        candidate_count=provenance_count(
            selection.provenance,
            "candidate_count",
            default=len(selection_request.candidate_views),
        ),
        filtered_candidate_count=provenance_count(
            selection.provenance,
            "filtered_candidate_count",
            default=len(selection.selected_view_ids),
        ),
        attempted_candidate_ids=selection.attempted_candidate_ids,
        selected_view_ids=selection.selected_view_ids,
        attempted_plan_ids=selection.attempted_plan_ids,
        selected_plan_id=selection.selected_plan_id,
        selector_backend=str(
            selection.provenance.get("selector_backend")
            or selection.backend
        ),
        selection_mode=(
            str(
                selection_request.context.get(
                    "vlm_selection_mode",
                    control.vlm_selection_mode,
                )
            )
            if stage == "vlm"
            else None
        ),
        vlm_call_count=(
            provenance_count(
                selection.provenance,
                "vlm_call_count",
                default=1,
            )
            if stage == "vlm"
            else 0
        ),
        vlm_call_count_source=(
            "selector_provenance"
            if stage == "vlm"
            and "vlm_call_count" in selection.provenance
            else (
                "conservative_interface_dispatch"
                if stage == "vlm"
                else "not_vlm"
            )
        ),
        evidence_round=evidence_round,
        episode_index=episode_index,
        relaxed_constraints=(
            selected_plan.relaxed_constraints
            if selected_plan is not None
            else ()
        ),
        has_camera_proposal=selection.camera_proposal is not None,
        provenance=selection.provenance,
    )
    trace.append(
        {
            "stage": "camera_selector",
            "selection_stage": stage,
            "evidence_round": evidence_round,
            "status": selection.outcome,
            "camera_constraints": constraints.to_dict(),
            "result": selection.to_dict(),
        }
    )


def _case_id(request: JudgeRequest) -> str | None:
    for source in (
        request.context,
        request.scene_context,
        request.claim_or_event,
    ):
        for key in ("case_id", "scene_id", "example_id", "sample_id"):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value)
    return None


def _deterministic_outcome(request: JudgeRequest) -> str | None:
    evidence = request.deterministic_evidence
    for key in ("status", "verdict", "outcome"):
        value = evidence.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _evidence_recovery_outcome(
    trace: list[dict[str, Any]],
    *,
    final_status: str,
) -> str:
    repair_attempted = any(
        event.get("stage") in {"camera_selector", "render"}
        for event in trace
    )
    if final_status in {"valid", "invalid"}:
        return "recovered" if repair_attempted else "not_required"
    return "unsuccessful" if repair_attempted else "not_attempted"


render_result_with_audit = _render_result_with_audit
evidence_content_identity = _evidence_content_identity
evidence_fingerprint = _evidence_fingerprint
evidence_refs = _evidence_refs
jsonable = _jsonable
rendered_evidence_refs = _rendered_evidence_refs
rendered_view_count = _rendered_view_count
record_camera_selection_audit = record_camera_selection
record_selector_failure_audit = record_selector_failure
