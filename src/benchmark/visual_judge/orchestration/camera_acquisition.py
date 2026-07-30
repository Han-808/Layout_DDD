from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from typing import Any

from benchmark.visual_judge.acquisition_state import (
    CameraAcquisitionState,
)
from benchmark.visual_judge.adapters.legacy_camera import (
    HybridCameraSelector,
    build_camera_selector,
    camera_selection_result_from_value,
)
from benchmark.visual_judge.adapters.active_camera import (
    ActiveVLMCameraSelector,
)
from benchmark.visual_judge.adapters.deterministic_camera import (
    DeterministicCameraRepairSolver,
    DeterministicLocalCameraSelector,
)
from benchmark.visual_judge.camera_ranking import (
    DeterministicCameraRankingConfig,
)
from benchmark.visual_judge.camera_dsl import (
    METRIC_CAMERA_REQUIREMENTS,
    CameraConstraintSet,
    active_constraint_references,
    canonical_camera_metric,
    camera_constraints_from_judge_request,
)
from benchmark.visual_judge.camera_repair import (
    CameraRepairPlan,
    diagnose_camera_constraint_conflicts,
    generate_camera_repair_plans,
)
from benchmark.visual_judge.camera_targets import (
    merge_authoritative_target_ids,
)
from benchmark.visual_judge.cascade_policy import (
    should_escalate_to_vlm,
)
from benchmark.visual_judge.control_config import (
    VLMEvaluationControl,
)
from benchmark.visual_judge.interfaces.camera import (
    CameraSelectionRequest,
    CameraSelectionResult,
    CameraSelector,
)
from benchmark.visual_judge.interfaces.evidence import (
    EvidenceGateResult,
)
from benchmark.visual_judge.interfaces.judge import (
    EvidenceRequest,
    JudgeRequest,
)


class CameraSelectorInvocationError(RuntimeError):
    """Engineering failure from one selector call or response validation."""

    def __init__(self, failure_kind: str, error: Exception) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.failure_kind = str(failure_kind)
        self.original_error = error


def selector_dispatch_charge(selector: CameraSelector) -> int:
    return (
        0
        if getattr(
            selector,
            "trusted_composite_provider_adapter",
            False,
        )
        is True
        else 1
    )


def invoke_selector_once(
    selector: CameraSelector,
    request: CameraSelectionRequest,
    *,
    default_backend: str,
) -> CameraSelectionResult:
    try:
        raw = selector.select(request)
    except Exception as exc:
        raise CameraSelectorInvocationError(
            "selector_exception",
            exc,
        ) from exc
    try:
        validation_request = request
        if (
            isinstance(raw, CameraSelectionResult)
            and getattr(
                selector,
                "validated_internal_candidate_bank",
                False,
            )
            is True
        ):
            validation_request = _request_with_internal_candidates(
                request,
                raw,
            )
        return camera_selection_result_from_value(
            raw,
            request=validation_request,
            backend=str(
                getattr(selector, "backend", default_backend)
            ),
        )
    except Exception as exc:
        raise CameraSelectorInvocationError(
            "invalid_selector_response",
            exc,
        ) from exc


def _request_with_internal_candidates(
    request: CameraSelectionRequest,
    result: CameraSelectionResult,
) -> CameraSelectionRequest:
    candidates = list(deepcopy(result.selected_views))
    known = {
        str(candidate.get("id") or "")
        for candidate in candidates
    }
    for candidate_id in result.attempted_candidate_ids:
        if candidate_id not in known:
            candidates.append({"id": candidate_id})
            known.add(candidate_id)
    return replace(
        request,
        candidate_views=tuple(candidates),
    )


def resolve_camera_selector(
    value: CameraSelector | Any | None,
    *,
    backend: str,
) -> CameraSelector:
    if getattr(value, "preserve_configured_adapter", False) is True:
        return value
    if backend == "vlm":
        return build_camera_selector(backend="vlm", vlm=value)
    if backend == "hybrid":
        if isinstance(value, tuple) and len(value) == 2:
            return build_camera_selector(
                backend="hybrid",
                vlm=value[0],
                deterministic=value[1],
            )
        if isinstance(value, HybridCameraSelector):
            return value
        raise ValueError(
            "hybrid CameraSelector requires (vlm, deterministic) backends "
            "or a HybridCameraSelector"
        )
    return build_camera_selector(
        backend=backend,
        existing=value if backend == "existing" else None,
        deterministic=value if backend == "deterministic" else None,
    )


def _as_explicit_selector(
    value: CameraSelector | Any | None,
    *,
    backend: str,
    vlm_selection_mode: str = "repair_plan",
    allow_freeform_pose: bool = False,
    deterministic_ranking_config: (
        DeterministicCameraRankingConfig | dict[str, Any] | None
    ) = None,
    deterministic_ranking_sources: dict[str, str] | None = None,
    max_repair_plans: int = 3,
    max_repair_plans_source: str = "default",
) -> CameraSelector | None:
    if value is None:
        return None
    if isinstance(value, ActiveVLMCameraSelector):
        if value.selection_mode != vlm_selection_mode:
            raise ValueError(
                "injected ActiveVLMCameraSelector selection_mode conflicts "
                "with camera_acquisition.vlm.selection_mode"
            )
        if value.max_repair_plans != max_repair_plans:
            raise ValueError(
                "injected ActiveVLMCameraSelector max_repair_plans conflicts "
                "with camera_acquisition.vlm.max_repair_plans"
            )
        return value
    if callable(getattr(value, "select", None)):
        return value
    if backend == "vlm":
        return ActiveVLMCameraSelector(
            value,
            selection_mode=vlm_selection_mode,
            repair_solver=(
                DeterministicCameraRepairSolver(
                    DeterministicLocalCameraSelector(
                        ranking_config=deterministic_ranking_config,
                        ranking_config_sources=(
                            deterministic_ranking_sources
                        ),
                    )
                )
                if vlm_selection_mode == "repair_plan"
                else None
            ),
            pose_validator=getattr(value, "pose_validator", None),
            allow_freeform_pose=allow_freeform_pose,
            max_repair_plans=max_repair_plans,
            max_repair_plans_source=max_repair_plans_source,
        )
    return resolve_camera_selector(value, backend=backend)


def resolve_acquisition_selectors(
    *,
    requested_policy: str,
    configured_backend: str,
    compatibility_selector: CameraSelector,
    compatibility_was_provided: bool,
    deterministic: CameraSelector | Any | None,
    vlm: CameraSelector | Any | None,
    vlm_selection_mode: str,
    allow_freeform_pose: bool,
    deterministic_ranking_config: (
        DeterministicCameraRankingConfig | dict[str, Any] | None
    ) = None,
    deterministic_ranking_sources: dict[str, str] | None = None,
    max_repair_plans: int = 3,
    max_repair_plans_source: str = "default",
) -> tuple[CameraSelector | None, CameraSelector | None, str, str]:
    """Keep one-selector legacy paths stable while enabling two-stage DI."""

    explicit = deterministic is not None or vlm is not None
    deterministic_selector = _as_explicit_selector(
        deterministic,
        backend="deterministic",
    )
    vlm_selector = _as_explicit_selector(
        vlm,
        backend="vlm",
        vlm_selection_mode=vlm_selection_mode,
        allow_freeform_pose=allow_freeform_pose,
        deterministic_ranking_config=deterministic_ranking_config,
        deterministic_ranking_sources=deterministic_ranking_sources,
        max_repair_plans=max_repair_plans,
        max_repair_plans_source=max_repair_plans_source,
    )
    policy = str(requested_policy)

    if explicit:
        if deterministic_selector is None and compatibility_was_provided:
            deterministic_selector = compatibility_selector
        if (
            deterministic_selector is None
            and (
                configured_backend == "deterministic"
                or not compatibility_was_provided
            )
        ):
            deterministic_selector = compatibility_selector
        if (
            vlm_selector is None
            and compatibility_was_provided
            and configured_backend == "vlm"
        ):
            vlm_selector = compatibility_selector
        effective = policy
        if policy == "deterministic_then_vlm":
            if vlm_selector is None:
                effective = "deterministic_only"
            elif deterministic_selector is None:
                effective = "vlm_only"
        return (
            deterministic_selector,
            vlm_selector,
            effective,
            "explicit_camera_acquisition",
        )

    # A historical CameraSelector may itself be a composite provider or
    # HybridCameraSelector. It remains one compatibility implementation and is
    # never reinterpreted as the research cascade.
    if policy == "fixed":
        return compatibility_selector, None, "fixed", "config"
    if policy == "vlm_only":
        if (
            not compatibility_was_provided
            or configured_backend != "vlm"
        ):
            return (
                None,
                None,
                "vlm_only",
                "vlm_selector_not_configured",
            )
        return (
            None,
            compatibility_selector,
            "vlm_only",
            "legacy_single_selector",
        )
    if configured_backend == "vlm":
        return (
            None,
            compatibility_selector,
            "vlm_only",
            "legacy_single_selector",
        )
    return (
        compatibility_selector,
        None,
        "deterministic_only",
        "legacy_single_selector",
    )


def remaining_budget(
    *,
    control: VLMEvaluationControl,
    selector_calls: int,
    actions_used: int,
    rounds_used: int,
    total_images_acquired: int,
) -> dict[str, int]:
    return {
        "remaining_images": max(
            0,
            control.max_total_images - total_images_acquired,
        ),
        "remaining_camera_actions": max(
            0,
            control.max_camera_actions - actions_used,
        ),
        "remaining_selector_calls": max(
            0,
            control.max_selector_calls - selector_calls,
        ),
        "remaining_evidence_rounds": max(
            0,
            control.max_evidence_rounds - rounds_used,
        ),
    }


def stage_budget_stop(
    *,
    control: VLMEvaluationControl,
    policy_source: str,
    state: CameraAcquisitionState,
) -> str | None:
    if policy_source == "legacy_single_selector":
        limit = control.max_evidence_rounds
    elif state.stage == "vlm":
        limit = control.vlm_max_rounds
    else:
        limit = control.deterministic_max_rounds
    used = (
        state.vlm_rounds_used
        if state.stage == "vlm"
        else state.deterministic_rounds_used
    )
    if used >= limit:
        return f"{state.stage}_stage_rounds_exhausted"
    return None


def build_selection_request(
    *,
    control: VLMEvaluationControl,
    request: JudgeRequest,
    evidence: list[Any],
    targets: tuple[str, ...],
    known_target_ids: tuple[str, ...],
    goal: dict[str, Any],
    constraints: CameraConstraintSet,
    candidates: tuple[dict[str, Any], ...],
    actions: tuple[str, ...],
    selector_context: dict[str, Any],
    state: CameraAcquisitionState,
    repair_plans: tuple[CameraRepairPlan, ...],
    deterministic_selection: CameraSelectionResult | None,
    selector_calls: int,
    actions_used: int,
    rounds_used: int,
    total_images_acquired: int,
) -> CameraSelectionRequest:
    stage_max_views = (
        control.vlm_max_selected_views
        if state.stage == "vlm"
        else control.deterministic_max_selected_views
    )
    context = deepcopy(selector_context)
    context.update(
        {
            "camera_acquisition_policy": state.policy,
            "camera_acquisition_stage": state.stage,
            "camera_repair_episode": state.episode_index,
            "camera_constraints": constraints.to_dict(),
            "known_target_ids": list(known_target_ids),
            "attempted_view_ids": list(state.attempted_view_ids),
            "attempted_plan_ids": list(state.attempted_plan_ids),
            "vlm_selection_mode": control.vlm_selection_mode,
            "camera_repair_plans": [
                plan.to_dict() for plan in repair_plans
            ],
            "camera_repair_plan_budget": {
                "effective": control.vlm_max_repair_plans,
                "source": control.sources.get(
                    "camera_acquisition.vlm.max_repair_plans",
                    "default",
                ),
            },
            "deterministic_rejected_candidates": (
                list(
                    deepcopy(
                        deterministic_selection.rejected_candidates
                    )
                )
                if deterministic_selection is not None
                else []
            ),
            "deterministic_candidate_evaluations": (
                candidate_conflict_evaluations(
                    deterministic_selection
                )
            ),
            "evidence_round": rounds_used + 1,
            "scene_access": "read_only",
        }
    )
    remaining = remaining_budget(
        control=control,
        selector_calls=selector_calls,
        actions_used=actions_used,
        rounds_used=rounds_used,
        total_images_acquired=total_images_acquired,
    )
    return CameraSelectionRequest(
        task=request.task,
        metric=request.metric,
        target_ids=targets,
        scene=deepcopy(request.scene_context),
        evidence_goal=deepcopy(goal),
        existing_visual_evidence=tuple(deepcopy(evidence)),
        budget={
            "max_views_per_round": min(
                control.max_views_per_round,
                stage_max_views,
                remaining["remaining_images"],
            ),
            "candidate_budget": control.deterministic_candidate_budget,
            **remaining,
        },
        constraints=constraints.to_dict(),
        candidate_views=candidates,
        allowed_actions=actions,
        evidence_round=rounds_used + 1,
        allow_freeform_pose=(
            state.stage == "vlm"
            and control.vlm_selection_mode == "freeform_pose"
            and control.allow_freeform_pose
        ),
        allow_scene_mutation=False,
        context=context,
    )


def escalation_allowed(
    *,
    control: VLMEvaluationControl,
    state: CameraAcquisitionState,
    vlm_selector_available: bool,
    remaining: dict[str, int],
    deterministic_outcome: str | None = None,
    gate_result: EvidenceGateResult | None = None,
    reason: str,
) -> bool:
    if not vlm_selector_available:
        return False
    return should_escalate_to_vlm(
        policy=state.policy,
        deterministic_outcome=deterministic_outcome,
        post_render_gate_result=gate_result,
        acquisition_state=state,
        budget=remaining,
        escalation={
            "on_no_feasible_candidate": (
                control.escalate_on_no_feasible_candidate
            ),
            "on_post_render_gate_insufficient": (
                control.escalate_on_post_render_gate_insufficient
            ),
            "on_selector_exception": (
                control.escalate_on_selector_exception
            ),
            "on_render_failure": control.escalate_on_render_failure,
        },
        reason=reason,
    )


def deterministic_escalation_reason(
    selection: CameraSelectionResult,
    *,
    constraints: CameraConstraintSet | None = None,
) -> str:
    codes = set(selection.reason_codes)
    if constraints is not None and diagnose_camera_constraint_conflicts(
        constraints,
        candidate_evaluations=candidate_conflict_evaluations(selection),
    ):
        return "camera_constraint_conflict"
    if (
        "candidate_already_attempted" in codes
        or "candidate_ranking_exhausted" in codes
    ):
        return "candidate_ranking_exhausted"
    return "no_feasible_candidate"


def repair_plans_for_vlm(
    *,
    constraints: CameraConstraintSet,
    deterministic_selection: CameraSelectionResult | None,
    max_plans: int = 3,
) -> tuple[CameraRepairPlan, ...]:
    conflicts = diagnose_camera_constraint_conflicts(
        constraints,
        candidate_evaluations=candidate_conflict_evaluations(
            deterministic_selection
        ),
    )
    return generate_camera_repair_plans(
        constraints,
        conflicts=conflicts,
        max_plans=max_plans,
    )


def apply_trusted_repair_policy(
    constraints: CameraConstraintSet,
    *,
    evidence: list[Any],
    selector_context: dict[str, Any],
    request_context: dict[str, Any] | None = None,
) -> CameraConstraintSet:
    """Attach controller-owned view-local relaxations, never Judge authority."""

    active = active_constraint_references(constraints)
    relaxable: list[str] = []
    sources: list[str] = []
    if (
        "global_context_preserved" in active
        and _has_global_anchor(
            evidence,
            request_context=request_context or {},
        )
    ):
        # A corrective view may trade away global framing only because the
        # packet already retains a global anchor; packet composition still
        # preserves that anchor after render.
        relaxable.append("global_context_preserved")
        sources.append("existing_global_anchor_safe_default")

    policy = selector_context.get("camera_constraint_policy")
    if policy is not None:
        if not isinstance(policy, dict):
            raise ValueError(
                "camera_constraint_policy dependency injection must be "
                "a mapping"
            )
        unknown = set(policy) - {"relaxable_constraints"}
        if unknown:
            raise ValueError(
                "camera_constraint_policy has unknown fields: "
                f"{sorted(unknown)}"
            )
        injected = policy.get("relaxable_constraints", ())
        if not isinstance(injected, (list, tuple)):
            raise ValueError(
                "camera_constraint_policy.relaxable_constraints must be "
                "a JSON list"
            )
        normalized = [
            str(value).strip()
            for value in injected
            if str(value).strip()
        ]
        if (
            len(normalized) != len(injected)
            or len(normalized) != len(set(normalized))
        ):
            raise ValueError(
                "trusted relaxable constraints must be unique non-empty "
                "references"
            )
        inactive = set(normalized) - active
        if inactive:
            raise ValueError(
                "trusted relaxable constraints must reference active "
                f"Camera DSL constraints: {sorted(inactive)}"
            )
        relaxable.extend(normalized)
        sources.append("dependency_injection")

    resolved = tuple(dict.fromkeys(relaxable))
    metadata = deepcopy(constraints.metadata)
    metadata["controller_repair_policy"] = {
        "requested": deepcopy(policy),
        "effective_relaxable_constraints": list(resolved),
        "sources": sources or ["conservative_default_empty"],
    }
    return replace(
        constraints,
        relaxable_constraints=resolved,
        metadata=metadata,
    )


def _has_global_anchor(
    evidence: list[Any],
    *,
    request_context: dict[str, Any],
) -> bool:
    if any(
        isinstance(item, dict)
        and "global" in str(item.get("role") or "").lower()
        for item in evidence
    ):
        return True
    global_refs = request_context.get(
        "relevant_global_visual_evidence"
    )
    if not isinstance(global_refs, (list, tuple)):
        return False
    evidence_refs = {
        str(
            item.get("path") or item.get("image_path")
            if isinstance(item, dict)
            else item
        )
        for item in evidence
    }
    return any(str(value) in evidence_refs for value in global_refs)


def candidate_conflict_evaluations(
    selection: CameraSelectionResult | None,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    if selection is None:
        return evaluations
    for rejected in selection.rejected_candidates:
        failed = rejected.get("failed_constraints")
        if not isinstance(failed, (list, tuple)) or not failed:
            continue
        evaluations.append(
            {
                "candidate_id": rejected.get("candidate_id"),
                "feasible": False,
                "failed_constraints": list(failed),
                "reason_codes": list(
                    rejected.get("reason_codes") or []
                ),
            }
        )
    return evaluations


def escalation_event(
    *,
    reason: str,
    request: EvidenceRequest | None,
    constraints: CameraConstraintSet,
    state: CameraAcquisitionState,
    remaining: dict[str, int],
    deterministic_selection: CameraSelectionResult | dict[str, Any] | None,
    gate_result: EvidenceGateResult | None,
) -> dict[str, Any]:
    if isinstance(deterministic_selection, CameraSelectionResult):
        selection_value = deterministic_selection.to_dict()
    elif isinstance(deterministic_selection, dict):
        selection_value = deepcopy(deterministic_selection)
    else:
        selection_value = None
    return {
        "stage": "camera_escalation",
        "from_stage": "deterministic",
        "to_stage": "vlm",
        "reason": reason,
        "episode_index": state.episode_index,
        "evidence_request": (
            request.to_dict() if request is not None else None
        ),
        "camera_constraints": constraints.to_dict(),
        "attempted_candidate_ids": (
            list(
                deterministic_selection.attempted_candidate_ids
            )
            if isinstance(
                deterministic_selection,
                CameraSelectionResult,
            )
            else list(
                (selection_value or {}).get(
                    "attempted_candidate_ids",
                    (),
                )
            )
        ),
        "attempted_view_ids": list(state.attempted_view_ids),
        "attempted_plan_ids": list(state.attempted_plan_ids),
        "deterministic_selection": selection_value,
        "evidence_gate_deficiencies": (
            list(deepcopy(gate_result.deficiencies))
            if gate_result is not None
            else []
        ),
        "remaining_budget": deepcopy(remaining),
        "evidence_round": state.total_rounds_used,
    }


def latest_selection(
    trace: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for event in reversed(trace):
        if event.get("stage") != "camera_selector":
            continue
        result = event.get("result")
        if isinstance(result, dict):
            return deepcopy(result)
    return None


def selector_identity(value: Any | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "backend": str(getattr(value, "backend", "unknown")),
        "adapter": f"{type(value).__module__}.{type(value).__qualname__}",
        "scene_access": "read_only",
    }


def camera_contract_failure_trace(
    error: Exception,
    *,
    evidence_round: int,
    source: str,
) -> dict[str, Any]:
    return {
        "stage": "camera_dsl",
        "evidence_round": evidence_round,
        "status": "failed",
        "failure_kind": "scene_contract_failure",
        "source": source,
        "error": f"{type(error).__name__}: {error}",
    }


def legacy_baseline_constraints(
    request: EvidenceRequest,
    *,
    metric: str,
    known_target_ids: tuple[str, ...],
    relation_type: str | None,
    error: Exception,
) -> CameraConstraintSet:
    metric_name = canonical_camera_metric(
        metric,
        relation_type=relation_type,
    )
    baseline = METRIC_CAMERA_REQUIREMENTS[
        metric_name
    ].baseline_observations
    normalized = EvidenceRequest(
        target_ids=request.target_ids,
        missing_observations=tuple(baseline),
        view_goal=request.view_goal,
        metadata={
            **deepcopy(request.metadata),
            "legacy_camera_dsl_normalization": {
                "original_missing_observations": list(
                    request.missing_observations
                ),
                "reason": f"{type(error).__name__}: {error}",
                "normalization": "metric_baseline_only",
            },
        },
    )
    return camera_constraints_from_judge_request(
        normalized,
        metric=metric,
        known_target_ids=known_target_ids,
        relation_type=relation_type,
    )


def known_target_ids(
    request: JudgeRequest,
    targets: tuple[str, ...],
) -> tuple[str, ...]:
    del targets
    values: list[str] = []

    for source in (request.claim_or_event, request.context):
        for key in (
            "target_ids",
            "target_object_ids",
            "object_ids",
            "subject_ids",
            "member_ids",
        ):
            raw = source.get(key)
            if isinstance(raw, (list, tuple)):
                values.extend(
                    str(item)
                    for item in raw
                    if str(item).strip()
                )
        for key in (
            "object_id",
            "subject_id",
            "anchor_id",
            "target_id",
        ):
            raw = source.get(key)
            if raw is not None and str(raw).strip():
                values.append(str(raw))

    return merge_authoritative_target_ids(
        values,
        request.scene_context,
    )


def request_relation_type(request: JudgeRequest) -> str | None:
    for source in (
        request.claim_or_event,
        request.context,
        request.deterministic_evidence,
    ):
        for key in (
            "relation_type",
            "event_type",
            "predicate",
            "type",
        ):
            value = source.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
        relation = source.get("relation")
        if isinstance(relation, str) and relation.strip():
            return relation.strip()
        if isinstance(relation, dict):
            for key in ("type", "predicate", "relation"):
                value = relation.get(key)
                if value is not None and str(value).strip():
                    return str(value).strip()
    return None


def render_count(
    provenance: dict[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    value = provenance.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"renderer provenance {key} must be a non-negative integer"
        )
    return value


def provenance_count(
    provenance: dict[str, Any],
    key: str,
    *,
    default: int,
) -> int:
    return render_count(provenance, key, default=default)


def render_duration(provenance: dict[str, Any]) -> float:
    value = provenance.get(
        "render_gpu_time_seconds",
        provenance.get("gpu_time_seconds", 0.0),
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            "renderer provenance render GPU time must be numeric"
        )
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(
            "renderer provenance render GPU time must be finite and "
            "non-negative"
        )
    return result
