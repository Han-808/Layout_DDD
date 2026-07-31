from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from benchmark.utils.io import write_json
from benchmark.visual_judge.control_config import (
    VLMEvaluationControl,
    resolve_vlm_evaluation_control,
)
from benchmark.visual_judge.acquisition_planner import (
    EvidenceAcquisitionPlanner,
    MetricAcquisitionPlanningRequest,
    MetricSpecificAcquisitionPlanner,
)
from benchmark.visual_judge.acquisition_state import (
    CameraAcquisitionState,
)
from benchmark.visual_judge.camera_dsl import (
    CameraConstraintSet,
)
from benchmark.visual_judge.evidence_gate import DeterministicEvidenceGate
from benchmark.visual_judge.experiment_telemetry import (
    CameraExperimentTelemetry,
)
from benchmark.visual_judge.interfaces.camera import (
    CameraSelectionRequest,
    CameraSelectionResult,
    CameraSelector,
)
from benchmark.visual_judge.interfaces.evidence import (
    EvidenceGate,
    EvidenceGateRequest,
    EvidenceGateResult,
    EvidenceRenderer,
)
from benchmark.visual_judge.interfaces.judge import (
    EvidenceRequest,
    Judge,
    JudgeRequest,
    JudgeResult,
)
from benchmark.visual_judge.adapters.deterministic_camera import (
    DeterministicLocalCameraSelector,
)
from benchmark.visual_judge.orchestration.audit import (
    build_evaluation_audit as _build_evaluation_audit,
    evidence_fingerprint as _evidence_fingerprint,
    evidence_refs as _evidence_refs,
    jsonable as _jsonable,
    record_camera_selection_audit as _record_camera_selection_audit,
    record_selector_failure_audit as _record_selector_failure_audit,
)
from benchmark.visual_judge.orchestration.budget import (
    budget_stop_reason as _budget_stop_reason,
    normalize_camera_usage as _normalize_camera_usage,
    usage_overrun_stop_reason as _usage_overrun_stop_reason,
)
from benchmark.visual_judge.orchestration.camera_acquisition import (
    apply_trusted_repair_policy as _apply_trusted_repair_policy,
    build_selection_request as _build_selection_request,
    camera_contract_failure_trace as _camera_contract_failure_trace,
    deterministic_escalation_reason as _deterministic_escalation_reason,
    escalation_allowed as _escalation_allowed,
    escalation_event as _escalation_event,
    known_target_ids as _known_target_ids,
    remaining_budget as _remaining_budget,
    render_count as _render_count,
    render_duration as _render_duration,
    repair_plans_for_vlm as _repair_plans_for_vlm,
    request_relation_type as _request_relation_type,
    resolve_acquisition_selectors as _resolve_acquisition_selectors,
    resolve_camera_selector as _resolve_camera_selector,
    stage_budget_stop as _stage_budget_stop,
)
from benchmark.visual_judge.orchestration.evidence_packet import (
    gate_reason as _gate_reason,
    gate_stop_reason as _gate_stop_reason,
    goal_from_judge_request as _goal_from_judge_request,
    merge_evidence as _merge_evidence,
    request_target_ids as _request_target_ids,
    validate_candidates as _validate_candidates,
)
from benchmark.visual_judge.orchestration.repair_executor import (
    CameraRepairExecutor,
)


VLM_CONTROL_LOOP_VERSION = "vlm_evaluation_control_loop_v1"
EVALUATION_STATUSES = {"valid", "invalid", "unresolved"}


@dataclass(frozen=True)
class VLMEvaluationResult:
    status: str
    confidence: float
    reason: str
    defects: tuple[dict[str, Any], ...]
    stop_reason: str
    visual_evidence: tuple[Any, ...]
    judge_result: JudgeResult | None
    evidence_request: EvidenceRequest | None
    audit: dict[str, Any]
    manifest_path: str | None = None

    def __post_init__(self) -> None:
        if self.status not in EVALUATION_STATUSES:
            raise ValueError(
                "VLM evaluation status must be valid, invalid, or unresolved"
            )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "status": self.status,
            "confidence": self.confidence,
            "reason": self.reason,
            "defects": list(deepcopy(self.defects)),
            "stop_reason": self.stop_reason,
            "visual_evidence": list(deepcopy(self.visual_evidence)),
            "judge_result": (
                self.judge_result.to_dict()
                if self.judge_result is not None
                else None
            ),
            "evidence_request": (
                self.evidence_request.to_dict()
                if self.evidence_request is not None
                else None
            ),
            "audit": deepcopy(self.audit),
            "manifest_path": self.manifest_path,
        })


class VLMEvaluationController:
    """Budgeted evidence-readiness and metric-decision orchestration."""

    def __init__(
        self,
        *,
        judge: Judge,
        renderer: EvidenceRenderer,
        camera_selector: CameraSelector | Any | None = None,
        deterministic_camera_selector: CameraSelector | Any | None = None,
        vlm_camera_selector: CameraSelector | Any | None = None,
        acquisition_planner: EvidenceAcquisitionPlanner | Any | None = None,
        evidence_gate: EvidenceGate | None = None,
        control: VLMEvaluationControl | None = None,
    ) -> None:
        judge_call = getattr(judge, "judge", None)
        if not callable(judge_call):
            raise TypeError("VLMEvaluationController requires Judge.judge(request)")
        renderer_call = getattr(renderer, "render", None)
        if not callable(renderer_call):
            raise TypeError(
                "VLMEvaluationController requires EvidenceRenderer.render(request)"
            )
        self.control = control or resolve_vlm_evaluation_control(
            existing_selector_available=camera_selector is not None
        )
        self.judge = judge
        self.renderer = renderer
        self.acquisition_planner = (
            acquisition_planner or MetricSpecificAcquisitionPlanner()
        )
        if not callable(getattr(self.acquisition_planner, "plan", None)):
            raise TypeError(
                "VLMEvaluationController requires "
                "EvidenceAcquisitionPlanner.plan(request)"
            )
        self.evidence_gate = evidence_gate or DeterministicEvidenceGate()
        ranking_sources = {
            key: self.control.sources.get(
                f"camera_acquisition.deterministic.ranking.{key}",
                "default",
            )
            for key in self.control.deterministic_ranking
        }
        self.camera_selector = (
            DeterministicLocalCameraSelector(
                ranking_config=self.control.deterministic_ranking,
                ranking_config_sources=ranking_sources,
            )
            if (
                camera_selector is None
                and self.control.camera_selector_backend
                in {"existing", "deterministic"}
            )
            else _resolve_camera_selector(
                camera_selector,
                backend=self.control.camera_selector_backend,
            )
        )
        (
            self.deterministic_camera_selector,
            self.vlm_camera_selector,
            self.effective_camera_acquisition_policy,
            self.camera_acquisition_policy_source,
        ) = _resolve_acquisition_selectors(
            requested_policy=self.control.camera_acquisition_policy,
            configured_backend=self.control.camera_selector_backend,
            compatibility_selector=self.camera_selector,
            compatibility_was_provided=camera_selector is not None,
            deterministic=deterministic_camera_selector,
            vlm=vlm_camera_selector,
            vlm_selection_mode=self.control.vlm_selection_mode,
            allow_freeform_pose=self.control.allow_freeform_pose,
            deterministic_ranking_config=(
                self.control.deterministic_ranking
            ),
            deterministic_ranking_sources=ranking_sources,
            max_repair_plans=self.control.vlm_max_repair_plans,
            max_repair_plans_source=self.control.sources.get(
                "camera_acquisition.vlm.max_repair_plans",
                "default",
            ),
        )

    def run(
        self,
        request: JudgeRequest | dict[str, Any],
        *,
        evidence_goal: dict[str, Any] | None = None,
        candidate_views: list[dict[str, Any]]
        | tuple[dict[str, Any], ...] = (),
        allowed_actions: list[str] | tuple[str, ...] = (),
        selector_context: dict[str, Any] | None = None,
        gate_manifest_path: str | None = None,
        control_manifest_path: str | Path | None = None,
        initial_camera_usage: dict[str, Any] | None = None,
    ) -> VLMEvaluationResult:
        judge_request = JudgeRequest.from_value(request)
        goal = deepcopy(evidence_goal or {})
        targets = _request_target_ids(judge_request)
        candidates = tuple(deepcopy(candidate_views))
        actions = tuple(str(item) for item in allowed_actions)
        selector_context = deepcopy(selector_context or {})
        _validate_candidates(candidates)
        if any(not action for action in actions):
            raise ValueError("allowed camera actions must be non-empty strings")

        trace: list[dict[str, Any]] = []
        evidence = list(deepcopy(judge_request.visual_evidence))
        total_images_acquired = len(evidence)
        manifest_path = gate_manifest_path
        initial_usage = _normalize_camera_usage(initial_camera_usage)
        selector_calls = initial_usage["selector_calls"]
        actions_used = initial_usage["camera_actions"]
        rounds_used = 0
        judged_fingerprints: set[str] = set()
        last_judge: JudgeResult | None = None
        pending_request: EvidenceRequest | None = None
        post_render_validation_error: str | None = None
        last_deterministic_selection: CameraSelectionResult | None = None
        known_target_ids = _known_target_ids(judge_request, targets)
        relation_type = _request_relation_type(judge_request)
        camera_constraints: CameraConstraintSet | None = None
        acquisition_state = CameraAcquisitionState.create(
            self.effective_camera_acquisition_policy
        )
        telemetry = CameraExperimentTelemetry(
            policy=self.effective_camera_acquisition_policy
        )
        repair_executor = CameraRepairExecutor(
            renderer=self.renderer,
            control=self.control,
        )
        finish = partial(
            self._finish,
            judge_request=judge_request,
            acquisition_state=acquisition_state,
            telemetry=telemetry,
        )

        def unresolved(
            *,
            reason: str,
            stop_reason: str,
        ) -> VLMEvaluationResult:
            return finish(
                status="unresolved",
                reason=reason,
                stop_reason=stop_reason,
                evidence=evidence,
                judge_result=last_judge,
                evidence_request=pending_request,
                trace=trace,
                selector_calls=selector_calls,
                actions_used=actions_used,
                rounds_used=rounds_used,
                total_images_acquired=total_images_acquired,
                control_manifest_path=control_manifest_path,
            )

        while True:
            gate_result = self._check_gate(
                request=judge_request,
                visual_evidence=evidence,
                target_ids=targets,
                evidence_goal=goal,
                manifest_path=manifest_path,
            )
            gate_trace = {
                "stage": "evidence_gate",
                "evidence_round": rounds_used,
                "result": gate_result.to_dict(),
                "images_used": _evidence_refs(evidence),
            }
            if rounds_used == 0 and not trace and initial_usage["observed"]:
                gate_trace["initial_camera_usage"] = deepcopy(initial_usage)
            trace.append(gate_trace)
            telemetry.record_gate(
                phase=(
                    "post_render"
                    if acquisition_state.last_render_stage is not None
                    else "pre_judge"
                ),
                evidence_round=rounds_used,
                episode_index=acquisition_state.episode_index,
                result=gate_result.to_dict(),
            )

            if total_images_acquired > self.control.max_total_images:
                return unresolved(
                    reason="visual evidence exceeds the resolved image budget",
                    stop_reason="max_total_images_exhausted",
                )

            usage_overrun = _usage_overrun_stop_reason(
                control=self.control,
                selector_calls=selector_calls,
                camera_actions=actions_used,
            )
            if usage_overrun is not None:
                return unresolved(
                    reason=(
                        "observed camera usage exceeds the resolved control "
                        "budget"
                    ),
                    stop_reason=usage_overrun,
                )

            if post_render_validation_error is not None:
                return unresolved(
                    reason=post_render_validation_error,
                    stop_reason="renderer_followup_contract_invalid",
                )

            if gate_result.ready:
                judge_fingerprint = _evidence_fingerprint(evidence)
                if judge_fingerprint in judged_fingerprints:
                    return unresolved(
                        reason=(
                            "the unchanged evidence packet was already judged"
                        ),
                        stop_reason="evidence_packet_already_judged",
                    )
                judged_fingerprints.add(judge_fingerprint)
                current_request = judge_request.with_visual_evidence(evidence)
                raw_judge = self.judge.judge(current_request)
                last_judge = JudgeResult.from_value(raw_judge)
                telemetry.record_judge(
                    evidence_round=rounds_used,
                    episode_index=acquisition_state.episode_index,
                    status=last_judge.status,
                )
                trace.append(
                    {
                        "stage": "judge",
                        "evidence_round": rounds_used,
                        "result": last_judge.to_dict(),
                        "images_used": _evidence_refs(evidence),
                    }
                )
                if last_judge.status in {"valid", "invalid"}:
                    return finish(
                        status=last_judge.status,
                        confidence=last_judge.confidence,
                        reason=last_judge.reason,
                        defects=last_judge.defects,
                        stop_reason="judge_conclusion",
                        evidence=evidence,
                        judge_result=last_judge,
                        evidence_request=None,
                        trace=trace,
                        selector_calls=selector_calls,
                        actions_used=actions_used,
                        rounds_used=rounds_used,
                        total_images_acquired=total_images_acquired,
                        control_manifest_path=control_manifest_path,
                    )
                pending_request = last_judge.evidence_request
                if not self.control.judge_allow_need_more_evidence:
                    return unresolved(
                        reason=last_judge.reason,
                        stop_reason="judge_evidence_request_disabled",
                    )
                if pending_request is None:  # Defensive; validator also checks.
                    raise ValueError(
                        "Judge need_more_evidence requires evidence_request"
                    )
                protected_group_targets = _protected_group_targets(
                    judge_request
                )
                if protected_group_targets:
                    outside_scope = sorted(
                        set(pending_request.target_ids)
                        - set(protected_group_targets)
                    )
                    if outside_scope:
                        trace.append(
                            {
                                "stage": "judge_evidence_request",
                                "evidence_round": rounds_used,
                                "status": "invalid",
                                "reason": (
                                    "Judge requested targets outside the "
                                    "immutable group scope"
                                ),
                                "outside_group_scope_target_ids": (
                                    outside_scope
                                ),
                                "group_scope_target_ids": list(
                                    protected_group_targets
                                ),
                            }
                        )
                        return unresolved(
                            reason=(
                                "Judge evidence request attempted to change "
                                "the grouping scope"
                            ),
                            stop_reason=(
                                "judge_evidence_request_outside_group_scope"
                            ),
                        )
                    pending_request = EvidenceRequest(
                        target_ids=protected_group_targets,
                        missing_observations=(
                            pending_request.missing_observations
                        ),
                        view_goal=pending_request.view_goal,
                        metadata={
                            **deepcopy(pending_request.metadata),
                            "group_scope_preserved": True,
                        },
                    )
                goal = _goal_from_judge_request(goal, pending_request)
                if pending_request.target_ids:
                    targets = pending_request.target_ids
                    known_target_ids = _known_target_ids(
                        judge_request,
                        targets,
                    )
                acquisition_state.start_episode()
                try:
                    camera_constraints = self.acquisition_planner.plan(
                        MetricAcquisitionPlanningRequest(
                            metric=judge_request.metric,
                            evidence_request=pending_request,
                            known_target_ids=tuple(known_target_ids),
                            relation_type=relation_type,
                        )
                    )
                except (TypeError, ValueError) as exc:
                    trace.append(
                        _camera_contract_failure_trace(
                            exc,
                            evidence_round=rounds_used,
                            source="judge_evidence_request",
                        )
                    )
                    return unresolved(
                        reason=(
                            "Judge evidence request could not be represented "
                            "by the metric-scoped Camera DSL"
                        ),
                        stop_reason="camera_constraint_contract_invalid",
                    )
                trace.append(
                    {
                        "stage": "acquisition_planner",
                        "evidence_round": rounds_used,
                        "episode_index": acquisition_state.episode_index,
                        "backend": str(
                            getattr(
                                self.acquisition_planner,
                                "backend",
                                type(self.acquisition_planner).__name__,
                            )
                        ),
                        "evidence_request": pending_request.to_dict(),
                        "camera_constraints": camera_constraints.to_dict(),
                    }
                )
            else:
                # EvidenceGate owns input integrity only. A missing, corrupt,
                # blank, or undecodable packet is an engineering/render
                # failure and cannot be translated into a camera request.
                # Metric sufficiency and all normal camera repair start from
                # Judge.need_more_evidence above.
                return unresolved(
                    reason=_gate_reason(gate_result),
                    stop_reason=_gate_stop_reason(gate_result),
                )

            if self.effective_camera_acquisition_policy == "fixed":
                return unresolved(
                    reason=(
                        "fixed camera acquisition policy cannot repair "
                        "insufficient evidence"
                    ),
                    stop_reason="fixed_views_insufficient",
                )

            budget_stop = _budget_stop_reason(
                control=self.control,
                rounds_used=rounds_used,
                selector_calls=selector_calls,
                total_images_acquired=total_images_acquired,
            )
            if budget_stop is not None:
                return unresolved(
                    reason="visual evidence budget exhausted before another repair",
                    stop_reason=budget_stop,
                )

            if camera_constraints is None:
                return unresolved(
                    reason="camera repair requires validated Camera DSL constraints",
                    stop_reason="camera_constraint_contract_invalid",
                )
            try:
                camera_constraints = _apply_trusted_repair_policy(
                    camera_constraints,
                    evidence=evidence,
                    selector_context=selector_context,
                    request_context=judge_request.context,
                )
            except (TypeError, ValueError) as exc:
                trace.append(
                    _camera_contract_failure_trace(
                        exc,
                        evidence_round=rounds_used,
                        source="controller_repair_policy",
                    )
                )
                return unresolved(
                    reason=(
                        "trusted camera repair policy could not be "
                        "validated"
                    ),
                    stop_reason="camera_constraint_contract_invalid",
                )

            while True:
                selector = self._selector_for_stage(acquisition_state.stage)
                if selector is None:
                    return unresolved(
                        reason=(
                            f"{acquisition_state.stage} camera selector is "
                            "not configured"
                        ),
                        stop_reason="camera_selector_unavailable",
                    )
                stage_stop = _stage_budget_stop(
                    control=self.control,
                    policy_source=(
                        self.camera_acquisition_policy_source
                    ),
                    state=acquisition_state,
                )
                if stage_stop is not None:
                    return unresolved(
                        reason="camera acquisition stage budget exhausted",
                        stop_reason=stage_stop,
                    )
                repair_plans = _repair_plans_for_vlm(
                    constraints=camera_constraints,
                    deterministic_selection=last_deterministic_selection,
                    max_plans=self.control.vlm_max_repair_plans,
                )
                selection_request = _build_selection_request(
                    control=self.control,
                    request=judge_request,
                    evidence=evidence,
                    targets=targets,
                    known_target_ids=known_target_ids,
                    goal=goal,
                    constraints=camera_constraints,
                    candidates=candidates,
                    actions=actions,
                    selector_context=selector_context,
                    state=acquisition_state,
                    repair_plans=repair_plans,
                    deterministic_selection=last_deterministic_selection,
                    selector_calls=selector_calls,
                    actions_used=actions_used,
                    rounds_used=rounds_used,
                    total_images_acquired=total_images_acquired,
                )
                selection_execution = repair_executor.select_once(
                    selector=selector,
                    request=selection_request,
                    default_backend=acquisition_state.stage,
                )
                selector_calls += selection_execution.selector_calls
                if selection_execution.selection is None:
                    failure_kind = str(
                        selection_execution.failure_kind
                        or "selector_exception"
                    )
                    failure_error = str(
                        selection_execution.error
                        or "camera selector failed"
                    )
                    _record_selector_failure_audit(
                        telemetry=telemetry,
                        trace=trace,
                        selector=selector,
                        stage=acquisition_state.stage,
                        selection_request=selection_request,
                        control=self.control,
                        evidence_round=rounds_used + 1,
                        episode_index=(
                            acquisition_state.episode_index
                        ),
                        failure_kind=failure_kind,
                        failure_error=failure_error,
                    )
                    return unresolved(
                        reason=(
                            "camera selector failed; previous evidence was retained"
                            if self.control.on_selector_failure
                            == "keep_previous_evidence"
                            else "camera selector failed"
                        ),
                        stop_reason="camera_selector_failed",
                    )
                selection = selection_execution.selection

                acquisition_state.record_selection(
                    outcome=selection.outcome,
                    # State tracks camera views actually selected/rendered.
                    # Evaluated/rejected bank IDs remain in selector telemetry
                    # so a trusted repair plan may re-evaluate them under a
                    # different constraint set.
                    attempted_view_ids=selection.selected_view_ids,
                    attempted_plan_ids=(
                        selection.attempted_plan_ids
                        or (
                            (selection.selected_plan_id,)
                            if selection.selected_plan_id is not None
                            else ()
                        )
                    ),
                )
                if acquisition_state.stage == "deterministic":
                    last_deterministic_selection = selection
                _record_camera_selection_audit(
                    telemetry=telemetry,
                    trace=trace,
                    selection=selection,
                    selection_request=selection_request,
                    repair_plans=repair_plans,
                    constraints=camera_constraints,
                    control=self.control,
                    stage=acquisition_state.stage,
                    evidence_round=rounds_used + 1,
                    episode_index=acquisition_state.episode_index,
                )
                if selection.outcome == "selected":
                    break

                escalation_reason = _deterministic_escalation_reason(
                    selection,
                    constraints=camera_constraints,
                )
                if (
                    acquisition_state.stage == "deterministic"
                    and self._escalate_after_selection(
                        state=acquisition_state,
                        reason=escalation_reason,
                        selection=selection,
                        pending_request=pending_request,
                        constraints=camera_constraints,
                        trace=trace,
                        telemetry=telemetry,
                        selector_calls=selector_calls,
                        actions_used=actions_used,
                        rounds_used=rounds_used,
                        total_images_acquired=total_images_acquired,
                    )
                ):
                    continue
                if acquisition_state.stage == "vlm":
                    acquisition_state.mark_vlm_failed(selection.outcome)
                return unresolved(
                    reason=selection.reason,
                    stop_reason=(
                        "vlm_no_feasible_candidate"
                        if acquisition_state.stage == "vlm"
                        else "no_feasible_candidate"
                    ),
                )

            # Capture content identity before the renderer can overwrite a
            # same-path corrective image in place.
            previous_fingerprint = _evidence_fingerprint(evidence)
            render_execution = repair_executor.render_once(
                selector=selector,
                selection=selection,
                selection_request=selection_request,
                judge_request=judge_request,
                evidence_goal=goal,
                previous_visual_evidence=evidence,
                selector_calls_used=selector_calls,
                camera_actions_used=actions_used,
            )
            selector_calls += render_execution.selector_calls
            actions_used += render_execution.camera_actions
            if render_execution.failure_kind == "budget_exhausted":
                return unresolved(
                    reason=str(render_execution.reason),
                    stop_reason=str(render_execution.stop_reason),
                )
            if render_execution.failure_kind == "render_failure":
                rejected_evidence = list(
                    render_execution.rejected_visual_evidence
                )
                trace.append(
                    {
                        "stage": "render",
                        "selection_stage": acquisition_state.stage,
                        "evidence_round": rounds_used + 1,
                        "status": "failed",
                        "failure_kind": "render_failure",
                        "error": render_execution.error,
                        "observed_internal_selector_calls": (
                            render_execution.selector_calls
                        ),
                        "observed_camera_actions": (
                            render_execution.camera_actions
                        ),
                        "provenance": deepcopy(
                            render_execution.failure_provenance or {}
                        ),
                        "rejected_visual_evidence": _evidence_refs(
                            rejected_evidence
                        ),
                    }
                )
                failure_provenance = (
                    render_execution.failure_provenance or {}
                )
                telemetry.record_render_failure(
                    stage=acquisition_state.stage,
                    preview_count=_render_count(
                        failure_provenance,
                        "preview_render_count",
                        default=0,
                    ),
                    full_count=_render_count(
                        failure_provenance,
                        "full_render_count",
                        default=len(rejected_evidence),
                    ),
                    gpu_time_seconds=_render_duration(
                        failure_provenance
                    ),
                    evidence_round=rounds_used + 1,
                    episode_index=acquisition_state.episode_index,
                    error=str(render_execution.error),
                )
                if rejected_evidence:
                    total_images_acquired += len(rejected_evidence)
                    rejected_gate = self._check_gate(
                        request=judge_request,
                        visual_evidence=rejected_evidence,
                        target_ids=targets,
                        evidence_goal=goal,
                        manifest_path=None,
                    )
                    trace.append(
                        {
                            "stage": "evidence_gate",
                            "evidence_round": rounds_used + 1,
                            "result": rejected_gate.to_dict(),
                            "images_used": _evidence_refs(
                                rejected_evidence
                            ),
                            "accepted_for_judging": False,
                        }
                    )
                    telemetry.record_gate(
                        phase="post_render_rejected",
                        evidence_round=rounds_used + 1,
                        episode_index=(
                            acquisition_state.episode_index
                        ),
                        result=rejected_gate.to_dict(),
                    )
                return unresolved(
                    reason=str(render_execution.reason),
                    stop_reason=str(render_execution.stop_reason),
                )
            rendered = render_execution.rendered
            if rendered is None:
                raise RuntimeError(
                    "CameraRepairExecutor returned no render or failure"
                )
            post_render_validation_error = (
                render_execution.post_render_validation_error
            )
            rendered_view_count = (
                render_execution.rendered_view_count
            )
            evidence = _merge_evidence(
                evidence,
                rendered,
                preserve_global_anchor=(
                    camera_constraints.require_global_anchor
                ),
            )
            current_fingerprint = _evidence_fingerprint(evidence)
            total_images_acquired += len(rendered.visual_evidence)
            rounds_used += 1
            acquisition_state.record_render_round()
            manifest_path = rendered.manifest_path
            if (
                rendered.replaces_candidate_views
                or rendered.next_candidate_views
            ):
                try:
                    _validate_candidates(rendered.next_candidate_views)
                except Exception as exc:
                    post_render_validation_error = (
                        "renderer returned invalid next candidate views: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    candidates = rendered.next_candidate_views
            if (
                rendered.replaces_allowed_actions
                or rendered.next_allowed_actions
            ):
                if any(
                    not isinstance(action, str) or not action
                    for action in rendered.next_allowed_actions
                ):
                    post_render_validation_error = (
                        "renderer returned invalid next allowed camera actions"
                    )
                else:
                    actions = rendered.next_allowed_actions
            trace.append(
                {
                    "stage": "render",
                    "selection_stage": (
                        acquisition_state.last_render_stage
                    ),
                    "evidence_round": rounds_used,
                    "status": "completed",
                    "result": rendered.to_dict(),
                    "rendered_view_count": rendered_view_count,
                    "packet_changed": (
                        previous_fingerprint != current_fingerprint
                    ),
                    "images_used": _evidence_refs(evidence),
                }
            )
            telemetry.record_render(
                stage=str(acquisition_state.last_render_stage),
                preview_count=_render_count(
                    rendered.provenance,
                    "preview_render_count",
                    default=0,
                ),
                full_count=_render_count(
                    rendered.provenance,
                    "full_render_count",
                    default=rendered_view_count,
                ),
                gpu_time_seconds=_render_duration(
                    rendered.provenance
                ),
                evidence_round=rounds_used,
                episode_index=acquisition_state.episode_index,
            )
            # Looping is intentional: the next operation is always
            # EvidenceGate.
            if previous_fingerprint == current_fingerprint:
                gate_after_no_change = self._check_gate(
                    request=judge_request,
                    visual_evidence=evidence,
                    target_ids=targets,
                    evidence_goal=goal,
                    manifest_path=manifest_path,
                )
                trace.append(
                    {
                        "stage": "evidence_gate",
                        "evidence_round": rounds_used,
                        "result": gate_after_no_change.to_dict(),
                        "images_used": _evidence_refs(evidence),
                    }
                )
                telemetry.record_gate(
                    phase="post_render",
                    evidence_round=rounds_used,
                    episode_index=acquisition_state.episode_index,
                    result=gate_after_no_change.to_dict(),
                )
                if (
                    gate_after_no_change.ready
                    and current_fingerprint not in judged_fingerprints
                ):
                    continue
                return unresolved(
                    reason="camera repair did not change the evidence packet",
                    stop_reason="evidence_packet_unchanged",
                )

    def _selector_for_stage(
        self,
        stage: str,
    ) -> CameraSelector | None:
        if stage == "vlm":
            return self.vlm_camera_selector
        return self.deterministic_camera_selector

    def _escalate_after_selection(
        self,
        *,
        state: CameraAcquisitionState,
        reason: str,
        selection: CameraSelectionResult,
        pending_request: EvidenceRequest | None,
        constraints: CameraConstraintSet,
        trace: list[dict[str, Any]],
        telemetry: CameraExperimentTelemetry,
        selector_calls: int,
        actions_used: int,
        rounds_used: int,
        total_images_acquired: int,
    ) -> bool:
        remaining = _remaining_budget(
            control=self.control,
            selector_calls=selector_calls,
            actions_used=actions_used,
            rounds_used=rounds_used,
            total_images_acquired=total_images_acquired,
        )
        allowed = _escalation_allowed(
            control=self.control,
            state=state,
            vlm_selector_available=(
                self.vlm_camera_selector is not None
            ),
            remaining=remaining,
            deterministic_outcome=selection.outcome,
            reason=reason,
        )
        if not allowed:
            return False
        event = _escalation_event(
            reason=reason,
            request=pending_request,
            constraints=constraints,
            state=state,
            remaining=remaining,
            deterministic_selection=selection,
        )
        state.escalate(reason)
        trace.append(event)
        telemetry.record_escalation(event)
        return True

    def _check_gate(
        self,
        *,
        request: JudgeRequest,
        visual_evidence: list[Any],
        target_ids: tuple[str, ...],
        evidence_goal: dict[str, Any],
        manifest_path: str | None,
    ) -> EvidenceGateResult:
        raw = self.evidence_gate.check(
            EvidenceGateRequest(
                task=request.task,
                metric=request.metric,
                target_ids=target_ids,
                scene=deepcopy(request.scene_context),
                visual_evidence=tuple(deepcopy(visual_evidence)),
                evidence_goal=deepcopy(evidence_goal),
                manifest_path=manifest_path,
                context=deepcopy(request.context),
            )
        )
        return EvidenceGateResult.from_value(raw)

    def _finish(
        self,
        *,
        status: str,
        reason: str,
        stop_reason: str,
        evidence: list[Any],
        judge_result: JudgeResult | None,
        evidence_request: EvidenceRequest | None,
        trace: list[dict[str, Any]],
        selector_calls: int,
        actions_used: int,
        rounds_used: int,
        total_images_acquired: int,
        control_manifest_path: str | Path | None,
        judge_request: JudgeRequest,
        acquisition_state: CameraAcquisitionState,
        telemetry: CameraExperimentTelemetry,
        confidence: float = 0.0,
        defects: tuple[dict[str, Any], ...] = (),
    ) -> VLMEvaluationResult:
        audit = _build_evaluation_audit(
            schema_version=VLM_CONTROL_LOOP_VERSION,
            control=self.control,
            compatibility_selector=self.camera_selector,
            deterministic_selector=self.deterministic_camera_selector,
            vlm_selector=self.vlm_camera_selector,
            effective_policy=self.effective_camera_acquisition_policy,
            policy_source=self.camera_acquisition_policy_source,
            renderer=self.renderer,
            evidence_gate_backend=self.control.evidence_gate_backend,
            judge=self.judge,
            judge_result=judge_result,
            judge_request=judge_request,
            final_status=status,
            final_confidence=confidence,
            final_evidence_request=evidence_request,
            acquisition_state=acquisition_state,
            telemetry=telemetry,
            stop_reason=stop_reason,
            trace=trace,
            evidence=evidence,
            selector_calls=selector_calls,
            actions_used=actions_used,
            rounds_used=rounds_used,
            total_images_acquired=total_images_acquired,
        )
        written_path: str | None = None
        if control_manifest_path is not None:
            target = Path(control_manifest_path).expanduser()
            write_json(target, audit)
            written_path = str(target)
        return VLMEvaluationResult(
            status=status,
            confidence=confidence,
            reason=reason,
            defects=tuple(deepcopy(defects)),
            stop_reason=stop_reason,
            visual_evidence=tuple(deepcopy(evidence)),
            judge_result=judge_result,
            evidence_request=evidence_request,
            audit=audit,
            manifest_path=written_path,
        )


def _protected_group_targets(
    request: JudgeRequest,
) -> tuple[str, ...]:
    scope = request.context.get("group_scope")
    if not isinstance(scope, dict):
        return ()
    values = scope.get("member_ids")
    if not isinstance(values, list):
        values = request.context.get("member_ids")
    if not isinstance(values, list):
        return ()
    return tuple(
        dict.fromkeys(
            str(value)
            for value in values
            if isinstance(value, (str, int)) and str(value).strip()
        )
    )
