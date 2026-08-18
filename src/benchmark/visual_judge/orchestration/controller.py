from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
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
    EvidenceReadinessRequest,
    EvidenceReadinessResult,
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
    SEMANTIC_SELECTION_OBSERVATIONS,
)
from benchmark.visual_judge.orchestration.audit import (
    build_evaluation_audit as _build_evaluation_audit,
    evidence_artifact_refs as _evidence_artifact_refs,
    evidence_fingerprint as _evidence_fingerprint,
    evidence_refs as _evidence_refs,
    jsonable as _jsonable,
    record_camera_selection_audit as _record_camera_selection_audit,
    record_selector_failure_audit as _record_selector_failure_audit,
)
from benchmark.visual_judge.orchestration.budget import (
    budget_stop_reason as _budget_stop_reason,
    normalize_acquisition_ledger as _normalize_acquisition_ledger,
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
from benchmark.visual_judge.orchestration.evidence_window import (
    compose_bounded_evidence_window as _compose_bounded_evidence_window,
    evidence_artifact_id as _evidence_artifact_id,
    resolve_bounded_evidence_window as _resolve_bounded_evidence_window,
    select_reusable_evidence as _select_reusable_evidence,
)
from benchmark.visual_judge.orchestration.repair_executor import (
    CameraRepairExecutor,
)


VLM_CONTROL_LOOP_VERSION = "vlm_evaluation_control_loop_v1"
EVALUATION_STATUSES = {"valid", "invalid", "unresolved"}
_CONTROLLER_PRIVATE_CONTEXT_KEYS = frozenset(
    {"functional_group_evidence_window"}
)


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
        forced_choice = deepcopy(
            self.audit.get(
                "budget_exhaustion_forced_choice",
                {"applied": False},
            )
        )
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
            "budget_exhaustion_forced_choice": forced_choice,
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
        candidate_bank_builder: Any | None = None,
        candidate_preview_renderer: Any | None = None,
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
        self.production_candidate_previews_required = bool(
            getattr(
                self.vlm_camera_selector,
                "requires_candidate_previews",
                False,
            )
        )
        resolved_builder = candidate_bank_builder
        if resolved_builder is None:
            eligible_selectors = tuple(
                selector
                for selector in (
                    self.deterministic_camera_selector,
                    (
                        self.camera_selector
                        if self.production_candidate_previews_required
                        else None
                    ),
                )
                if isinstance(
                    selector,
                    DeterministicLocalCameraSelector,
                )
                and (
                    self.effective_camera_acquisition_policy
                    != "vlm_only"
                    or self.production_candidate_previews_required
                )
            )
            for selector in eligible_selectors:
                candidate = getattr(
                    selector, "candidate_bank_builder", None
                )
                if callable(getattr(candidate, "build", None)):
                    resolved_builder = candidate
                    break
        if resolved_builder is not None and not callable(
            getattr(resolved_builder, "build", None)
        ):
            raise TypeError(
                "candidate_bank_builder must expose build(request)"
            )
        if candidate_preview_renderer is not None and not callable(
            getattr(candidate_preview_renderer, "render", None)
        ):
            raise TypeError(
                "candidate_preview_renderer must expose render(request)"
            )
        self.candidate_bank_builder = resolved_builder
        self.candidate_preview_renderer = candidate_preview_renderer

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
        initial_acquisition_ledger: dict[str, Any] | None = None,
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
        evidence_window = _resolve_bounded_evidence_window(
            judge_request.context,
            initial_evidence=evidence,
        )
        initial_window_artifact_ids = [
            _evidence_artifact_id(item) for item in evidence
        ]
        presented_window_artifact_ids = set(initial_window_artifact_ids)
        evidence_window_events: list[dict[str, Any]] = []
        if evidence_window is not None:
            initialization = {
                "schema_version": "bounded_evidence_window_v1",
                "policy": evidence_window.policy,
                "group_id": evidence_window.group_id,
                "check_id": evidence_window.check_id,
                "trigger": "initial_packet",
                "max_active_images": evidence_window.max_active_images,
                "fixed_artifact_ids": list(
                    evidence_window.fixed_artifact_ids
                ),
                "before_artifact_ids": [],
                "added_artifact_ids": list(
                    initial_window_artifact_ids
                ),
                "evicted_artifact_ids": [],
                "after_artifact_ids": list(
                    initial_window_artifact_ids
                ),
                "overflow_flush_applied": False,
                "physical_artifacts_deleted": False,
            }
            evidence_window_events.append(initialization)
            trace.append(
                {
                    "stage": "evidence_window",
                    "status": "initialized",
                    "evidence_round": 0,
                    "result": deepcopy(initialization),
                }
            )
        manifest_path = gate_manifest_path
        initial_usage = _normalize_camera_usage(initial_camera_usage)
        initial_ledger = _normalize_acquisition_ledger(
            initial_acquisition_ledger
        )
        acquired_artifact_ids = set(initial_ledger["artifact_ids"])
        total_images_acquired = int(
            initial_ledger["total_images_acquired"]
        )

        def register_artifacts(items: list[Any] | tuple[Any, ...]) -> int:
            nonlocal total_images_acquired
            added = 0
            for artifact_id in _evidence_artifact_refs(list(items)):
                # Ledgers written before artifact-scoped accounting stored
                # bare evidence refs.  Accept that additive legacy spelling
                # without charging the same path again.
                legacy_alias = (
                    artifact_id.removeprefix("path:")
                    if artifact_id.startswith("path:")
                    else None
                )
                if (
                    artifact_id in acquired_artifact_ids
                    or (
                        legacy_alias is not None
                        and legacy_alias in acquired_artifact_ids
                    )
                ):
                    continue
                acquired_artifact_ids.add(artifact_id)
                total_images_acquired += 1
                added += 1
            return added

        register_artifacts(evidence)
        selector_calls = (
            int(initial_ledger["selector_calls"])
            + initial_usage["selector_calls"]
        )
        actions_used = (
            int(initial_ledger["camera_actions"])
            + initial_usage["camera_actions"]
        )
        rounds_used = int(initial_ledger["evidence_rounds"])
        judged_fingerprints: set[str] = set()
        last_judge: JudgeResult | None = None
        pending_request: EvidenceRequest | None = None
        post_render_validation_error: str | None = None
        last_deterministic_selection: CameraSelectionResult | None = None
        candidate_bank_episode = -1
        preview_episode = -1
        vlm_selection_mode_override: str | None = None
        known_target_ids = _known_target_ids(judge_request, targets)
        relation_type = _request_relation_type(judge_request)
        functional_atomic_check = _functional_atomic_required_check(
            judge_request
        )
        functional_soft_contract_active = (
            functional_atomic_check is not None
        )
        functional_terminal_recovery_active = bool(
            judge_request.metric == "functional_consistency"
            and not _is_routing_screen_request(judge_request)
        )
        evidence_readiness_reviewer = (
            _resolve_evidence_readiness_reviewer(
                self.vlm_camera_selector,
                self.camera_selector,
            )
        )
        readiness_reviewed_fingerprints: set[str] = set()
        readiness_passed_fingerprints: set[str] = set()
        readiness_review_count = 0
        readiness_acquire_count = 0
        targeted_acquisition_attempts = 0
        usable_side_fallback_attempted = False
        usable_side_fallback_applied = False
        terminal_limited_evidence = False
        camera_constraints: CameraConstraintSet | None = None
        acquisition_state = CameraAcquisitionState.create(
            self.effective_camera_acquisition_policy,
            total_rounds_used=rounds_used,
            total_deterministic_rounds_used=int(
                initial_ledger["deterministic_rounds"]
            ),
            total_vlm_rounds_used=int(
                initial_ledger["vlm_rounds"]
            ),
        )
        telemetry = CameraExperimentTelemetry(
            policy=self.effective_camera_acquisition_policy
        )
        repair_executor = CameraRepairExecutor(
            renderer=self.renderer,
            control=self.control,
        )
        def finish(**kwargs: Any) -> VLMEvaluationResult:
            return self._finish(
                **kwargs,
                acquisition_ledger=_current_acquisition_ledger(
                    artifact_ids=acquired_artifact_ids,
                    total_images_acquired=total_images_acquired,
                    rounds_used=rounds_used,
                    selector_calls=selector_calls,
                    actions_used=actions_used,
                    state=acquisition_state,
                ),
                judge_request=judge_request,
                acquisition_state=acquisition_state,
                telemetry=telemetry,
                evidence_window=(
                    evidence_window.to_dict()
                    if evidence_window is not None
                    else None
                ),
                initial_window_artifact_ids=(
                    initial_window_artifact_ids
                ),
                evidence_window_events=evidence_window_events,
                functional_soft_evidence_contract={
                    "schema_version": (
                        "functional_soft_evidence_contract_v1"
                    ),
                    "active": functional_soft_contract_active,
                    "check_id": (
                        str(
                            functional_atomic_check.get("check_id")
                            or ""
                        )
                        if functional_atomic_check is not None
                        else None
                    ),
                    "camera_selector_review_count": (
                        readiness_review_count
                    ),
                    "camera_selector_acquire_count": (
                        readiness_acquire_count
                    ),
                    "camera_selector_passed": bool(
                        readiness_passed_fingerprints
                    ),
                    "targeted_acquisition_attempts": (
                        targeted_acquisition_attempts
                    ),
                    "usable_side_fallback_attempted": (
                        usable_side_fallback_attempted
                    ),
                    "usable_side_fallback_applied": (
                        usable_side_fallback_applied
                    ),
                    "terminal_limited_evidence": (
                        terminal_limited_evidence
                    ),
                    "decision_authority": "none",
                },
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

        def force_terminal_choice(
            *,
            trigger_stop_reason: str,
        ) -> VLMEvaluationResult:
            nonlocal last_judge, terminal_limited_evidence
            if pending_request is None:
                raise ValueError(
                    "terminal forced choice requires a pending evidence "
                    "request"
                )
            if (
                last_judge is not None
                and last_judge.status != "need_more_evidence"
            ):
                raise ValueError(
                    "terminal forced choice cannot replace a completed Judge "
                    "decision"
                )
            if (
                last_judge is not None
                and _is_routing_screen_request(judge_request)
            ):
                # A screen is a router, not a terminal decision authority.
                # Preserve need_more_evidence so the enclosing metric workflow
                # can supply its planned local/group evidence.  Forcing a
                # binary answer here would silently bypass that downstream
                # stage while reusing the same insufficient global packet.
                trace.append(
                    {
                        "stage": "terminal_choice_policy",
                        "outcome": "deferred_to_downstream_review",
                        "trigger_stop_reason": trigger_stop_reason,
                        "evidence_request": pending_request.to_dict(),
                    }
                )
                return unresolved(
                    reason=last_judge.reason,
                    stop_reason=(
                        f"{trigger_stop_reason}_screen_deferred"
                    ),
                )
            ambiguity_before_forcing = bool(
                last_judge is not None
                and last_judge.status == "need_more_evidence"
            )
            pre_force_judge_status = (
                last_judge.status
                if last_judge is not None
                else str(
                    pending_request.metadata.get("source")
                    or "pre_judge_evidence_acquisition"
                )
            )
            pre_force_reason = (
                last_judge.reason
                if last_judge is not None
                else str(
                    pending_request.metadata.get("reason")
                    or pending_request.view_goal
                )
            )
            original_evidence_request = (
                pending_request.to_dict()
                if pending_request is not None
                else None
            )
            evidence_artifacts_at_forcing = _evidence_refs(evidence)
            final_request = _terminal_forced_choice_judge_request(
                judge_request,
                evidence=evidence,
                previous_request=pending_request,
                trigger_stop_reason=trigger_stop_reason,
            )
            raw_judge = self.judge.judge(
                _judge_visible_request(final_request)
            )
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
                    "terminal_forced_choice": True,
                    "ambiguity_before_forcing": (
                        ambiguity_before_forcing
                    ),
                    "pre_force_judge_status": (
                        pre_force_judge_status
                    ),
                    "pre_force_reason": pre_force_reason,
                    "pre_force_evidence_request": (
                        original_evidence_request
                    ),
                    "original_evidence_request": (
                        original_evidence_request
                    ),
                    "available_image_count": len(evidence),
                    "evidence_artifacts_at_forcing": (
                        evidence_artifacts_at_forcing
                    ),
                    "budget_trigger_stop_reason": trigger_stop_reason,
                    "final_forced_verdict": last_judge.status,
                    "final_forced_confidence": last_judge.confidence,
                    "visual_evidence_policy": (
                        "all_available_then_judge_context_bounded"
                    ),
                }
            )
            if last_judge.status not in {"valid", "invalid"}:
                raise ValueError(
                    "terminal budget-exhaustion Judge must return valid or "
                    "invalid; need_more_evidence is forbidden"
                )
            terminal_limited_evidence = True
            return finish(
                status=last_judge.status,
                confidence=last_judge.confidence,
                reason=last_judge.reason,
                defects=last_judge.defects,
                stop_reason=(
                    f"{trigger_stop_reason}_forced_choice"
                ),
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
                if (
                    gate_result.ready
                    and (
                        pending_request is not None
                        or functional_atomic_check is not None
                    )
                ):
                    if pending_request is None:
                        pending_request = (
                            _functional_check_evidence_request(
                                functional_atomic_check,
                                source="initial_image_budget_exhausted",
                            )
                        )
                    return force_terminal_choice(
                        trigger_stop_reason="max_total_images_exhausted",
                    )
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
                if (
                    gate_result.ready
                    and (
                        pending_request is not None
                        or functional_atomic_check is not None
                    )
                ):
                    if pending_request is None:
                        pending_request = (
                            _functional_check_evidence_request(
                                functional_atomic_check,
                                source="initial_camera_budget_exhausted",
                            )
                        )
                    return force_terminal_choice(
                        trigger_stop_reason=usage_overrun,
                    )
                return unresolved(
                    reason=(
                        "observed camera usage exceeds the resolved control "
                        "budget"
                    ),
                    stop_reason=usage_overrun,
                )

            if post_render_validation_error is not None:
                if gate_result.ready and pending_request is not None:
                    return force_terminal_choice(
                        trigger_stop_reason=(
                            "renderer_followup_contract_invalid"
                        ),
                    )
                return unresolved(
                    reason=post_render_validation_error,
                    stop_reason="renderer_followup_contract_invalid",
                )

            if gate_result.ready:
                judge_fingerprint = _evidence_fingerprint(evidence)
                acquisition_source: str | None = None
                if functional_atomic_check is not None:
                    active_preflight = _active_functional_preflight(
                        judge_request,
                        evidence=evidence,
                    )
                    if active_preflight is not None:
                        pending_request = (
                            _functional_preflight_evidence_request(
                                functional_atomic_check,
                                active_preflight,
                            )
                        )
                        acquisition_source = "usable_side_soft_fallback"
                        usable_side_fallback_attempted = True
                        usable_side_fallback_applied = True
                        trace.append(
                            {
                                "stage": "functional_evidence_readiness",
                                "status": "acquire",
                                "source": acquisition_source,
                                "evidence_round": rounds_used,
                                "check_id": str(
                                    functional_atomic_check.get(
                                        "check_id"
                                    )
                                    or ""
                                ),
                                "result": {
                                    "outcome": "acquire",
                                    "missing_observations": list(
                                        pending_request.missing_observations
                                    ),
                                    "view_goal": (
                                        pending_request.view_goal
                                    ),
                                    "reason_codes": deepcopy(
                                        active_preflight.get(
                                            "reason_codes"
                                        )
                                        or []
                                    ),
                                    "decision_authority": "none",
                                },
                                "evidence_request": (
                                    pending_request.to_dict()
                                ),
                                "images_used": _evidence_refs(evidence),
                            }
                        )
                    elif (
                        evidence_readiness_reviewer is not None
                        # The Judge sees the initial target-aware packet
                        # first.  CameraSelector readiness is a supplement
                        # verifier after the Judge has requested more visual
                        # evidence, not a mandatory second VLM opinion before
                        # every atomic check.  Machine-detected usable-side
                        # failure still takes the preflight branch above.
                        and pending_request is not None
                        and judge_fingerprint
                        not in readiness_reviewed_fingerprints
                    ):
                        readiness_budget_stop = _budget_stop_reason(
                            control=self.control,
                            rounds_used=rounds_used,
                            selector_calls=selector_calls,
                            total_images_acquired=total_images_acquired,
                        )
                        if readiness_budget_stop is not None:
                            pending_request = (
                                _functional_check_evidence_request(
                                    functional_atomic_check,
                                    source=(
                                        "camera_selector_readiness_budget_"
                                        "exhausted"
                                    ),
                                )
                            )
                            return force_terminal_choice(
                                trigger_stop_reason=(
                                    readiness_budget_stop
                                )
                            )
                        readiness_request = (
                            _build_functional_readiness_request(
                                judge_request,
                                check=functional_atomic_check,
                                evidence=evidence,
                                budget=_remaining_budget(
                                    control=self.control,
                                    selector_calls=selector_calls,
                                    actions_used=actions_used,
                                    rounds_used=rounds_used,
                                    total_images_acquired=(
                                        total_images_acquired
                                    ),
                                ),
                                evidence_window=(
                                    evidence_window.to_dict()
                                    if evidence_window is not None
                                    else None
                                ),
                            )
                        )
                        readiness_reviewed_fingerprints.add(
                            judge_fingerprint
                        )
                        readiness_review_count += 1
                        try:
                            raw_readiness = (
                                evidence_readiness_reviewer.
                                review_evidence_readiness(
                                    readiness_request
                                )
                            )
                            readiness_result = (
                                EvidenceReadinessResult.from_value(
                                    raw_readiness,
                                    request=readiness_request,
                                    backend=str(
                                        getattr(
                                            evidence_readiness_reviewer,
                                            "backend",
                                            "unknown",
                                        )
                                    ),
                                )
                            )
                        except Exception as exc:
                            selector_calls += (
                                _evidence_readiness_call_count(
                                    evidence_readiness_reviewer
                                )
                            )
                            pending_request = (
                                _functional_check_evidence_request(
                                    functional_atomic_check,
                                    source=(
                                        "camera_selector_readiness_failed"
                                    ),
                                    reason=(
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                )
                            )
                            trace.append(
                                {
                                    "stage": (
                                        "functional_evidence_readiness"
                                    ),
                                    "status": "failed",
                                    "evidence_round": rounds_used,
                                    "check_id": str(
                                        functional_atomic_check.get(
                                            "check_id"
                                        )
                                        or ""
                                    ),
                                    "error": (
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                    "fallback": (
                                        "forced_binary_with_retained_"
                                        "evidence"
                                    ),
                                    "evidence_request": (
                                        pending_request.to_dict()
                                    ),
                                    "images_used": _evidence_refs(
                                        evidence
                                    ),
                                    "decision_authority": "none",
                                }
                            )
                            return force_terminal_choice(
                                trigger_stop_reason=(
                                    "camera_evidence_readiness_failed"
                                )
                            )
                        selector_calls += _evidence_readiness_call_count(
                            evidence_readiness_reviewer
                        )
                        trace.append(
                            {
                                "stage": "functional_evidence_readiness",
                                "status": readiness_result.outcome,
                                "source": "vlm_camera_selector",
                                "evidence_round": rounds_used,
                                "check_id": readiness_request.check_id,
                                "result": readiness_result.to_dict(),
                                **(
                                    {
                                        "evidence_request": {
                                            "target_ids": list(
                                                readiness_request.target_ids
                                            ),
                                            "missing_observations": list(
                                                readiness_result.
                                                missing_observations
                                            ),
                                            "view_goal": (
                                                readiness_result.view_goal
                                            ),
                                            "metadata": {
                                                "source": (
                                                    "camera_selector_"
                                                    "evidence_readiness_v1"
                                                ),
                                                "check_ids": list(
                                                    _functional_check_ids(
                                                        functional_atomic_check
                                                    )
                                                ),
                                                "decision_authority": (
                                                    "none"
                                                ),
                                            },
                                        }
                                    }
                                    if readiness_result.outcome == "acquire"
                                    else {}
                                ),
                                "images_used": _evidence_refs(evidence),
                                "decision_authority": "none",
                            }
                        )
                        if readiness_result.outcome == "pass":
                            readiness_passed_fingerprints.add(
                                judge_fingerprint
                            )
                        else:
                            readiness_acquire_count += 1
                            pending_request = EvidenceRequest(
                                target_ids=(
                                    readiness_request.target_ids
                                ),
                                missing_observations=(
                                    readiness_result.
                                    missing_observations
                                ),
                                view_goal=readiness_result.view_goal,
                                metadata={
                                    "source": (
                                        "camera_selector_evidence_"
                                        "readiness_v1"
                                    ),
                                    "check_ids": [
                                        *_functional_check_ids(
                                            functional_atomic_check
                                        )
                                    ],
                                    "check_type": (
                                        readiness_request.check_type
                                    ),
                                    "reason": readiness_result.reason,
                                    "supporting_evidence_refs": list(
                                        readiness_result.
                                        supporting_evidence_refs
                                    ),
                                    "decision_authority": "none",
                                },
                            )
                            acquisition_source = (
                                "camera_selector_evidence_readiness"
                            )

                if acquisition_source is None:
                    if judge_fingerprint in judged_fingerprints:
                        if (
                            functional_soft_contract_active
                            and pending_request is not None
                        ):
                            return force_terminal_choice(
                                trigger_stop_reason=(
                                    "evidence_packet_already_judged"
                                )
                            )
                        return unresolved(
                            reason=(
                                "the unchanged evidence packet was already "
                                "judged"
                            ),
                            stop_reason="evidence_packet_already_judged",
                        )
                    judged_fingerprints.add(judge_fingerprint)
                    pending_request = None
                    current_request = _judge_visible_request(
                        judge_request.with_visual_evidence(evidence)
                    )
                    raw_judge = self.judge.judge(current_request)
                    last_judge = JudgeResult.from_value(raw_judge)
                    raw_judge_response = deepcopy(
                        getattr(self.judge, "last_raw_response", None)
                    )
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
                            total_images_acquired=(
                                total_images_acquired
                            ),
                            control_manifest_path=(
                                control_manifest_path
                            ),
                        )
                    pending_request = last_judge.evidence_request
                    if not self.control.judge_allow_need_more_evidence:
                        if (
                            functional_soft_contract_active
                            and pending_request is not None
                        ):
                            return force_terminal_choice(
                                trigger_stop_reason=(
                                    "judge_evidence_request_disabled"
                                )
                            )
                        return unresolved(
                            reason=last_judge.reason,
                            stop_reason=(
                                "judge_evidence_request_disabled"
                            ),
                        )
                    if pending_request is None:
                        raise ValueError(
                            "Judge need_more_evidence requires "
                            "evidence_request"
                        )
                    (
                        judge_request,
                        registered_placement_check,
                    ) = _register_pending_placement_check(
                        judge_request,
                        raw_response=raw_judge_response,
                        evidence_request=pending_request,
                    )
                    if registered_placement_check is not None:
                        trace.append(
                            {
                                "stage": "placement_check_lifecycle",
                                "evidence_round": rounds_used,
                                "status": str(
                                    registered_placement_check.get(
                                        "handoff_status"
                                    )
                                    or "evidence_requested"
                                ),
                                "check": deepcopy(
                                    registered_placement_check
                                ),
                                "decision_authority": "none",
                            }
                        )
                    acquisition_source = "judge_need_more_evidence"

                if (
                    functional_soft_contract_active
                    and pending_request is not None
                ):
                    targeted_acquisition_attempts += 1
                protected_group_targets = _protected_group_targets(
                    judge_request
                )
                if protected_group_targets:
                    allowed_group_targets = (
                        _allowed_group_evidence_targets(
                            judge_request
                        )
                    )
                    outside_scope = sorted(
                        set(pending_request.target_ids)
                        - set(allowed_group_targets)
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
                                "allowed_external_target_ids": sorted(
                                    set(allowed_group_targets)
                                    - set(protected_group_targets)
                                ),
                            }
                        )
                        if functional_soft_contract_active:
                            return force_terminal_choice(
                                trigger_stop_reason=(
                                    "judge_evidence_request_outside_group_"
                                    "scope"
                                )
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
                        # Group membership remains immutable context, while a
                        # Judge may focus one repair view on a strict non-empty
                        # subset of that group.
                        target_ids=pending_request.target_ids,
                        missing_observations=(
                            pending_request.missing_observations
                        ),
                        view_goal=pending_request.view_goal,
                        metadata={
                            **deepcopy(pending_request.metadata),
                            "group_scope_preserved": True,
                            "authoritative_group_member_ids": list(
                                protected_group_targets
                            ),
                            "camera_focus_target_ids": list(
                                pending_request.target_ids
                            ),
                        },
                    )
                goal = _goal_from_judge_request(goal, pending_request)
                if pending_request.target_ids:
                    targets = pending_request.target_ids
                    known_target_ids = _known_target_ids(
                        judge_request,
                        targets,
                    )
                if evidence_window is not None:
                    reusable_records = _select_reusable_evidence(
                        evidence_window,
                        active_evidence=evidence,
                        target_ids=pending_request.target_ids,
                        missing_observations=(
                            pending_request.missing_observations
                        ),
                        excluded_artifact_ids=(
                            presented_window_artifact_ids
                        ),
                    )
                    remaining_image_budget = max(
                        0,
                        int(self.control.max_total_images)
                        - total_images_acquired,
                    )
                    reusable_records = reusable_records[
                        :remaining_image_budget
                    ]
                    if reusable_records:
                        reusable_evidence = [
                            deepcopy(item["visual_evidence"])
                            for item in reusable_records
                        ]
                        evidence, window_event = (
                            _compose_bounded_evidence_window(
                                evidence_window,
                                previous=evidence,
                                additions=reusable_evidence,
                                trigger="shared_bank_reuse",
                            )
                        )
                        reused_ids = [
                            str(item["artifact_id"])
                            for item in reusable_records
                        ]
                        window_event["reused_artifact_ids"] = reused_ids
                        window_event["camera_selector_invoked"] = False
                        register_artifacts(reusable_evidence)
                        presented_window_artifact_ids.update(reused_ids)
                        evidence_window_events.append(
                            deepcopy(window_event)
                        )
                        trace.append(
                            {
                                "stage": "evidence_bank_reuse",
                                "status": "completed",
                                "evidence_round": rounds_used,
                                "group_id": evidence_window.group_id,
                                "check_id": evidence_window.check_id,
                                "reused_artifact_ids": reused_ids,
                                "camera_selector_invoked": False,
                                "result": deepcopy(window_event),
                                "images_used": _evidence_refs(evidence),
                            }
                        )
                        # The changed packet must return to EvidenceGate and
                        # Judge before any camera-selection episode begins.
                        continue
                acquisition_state.start_episode()
                vlm_selection_mode_override = None
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
                    if functional_soft_contract_active:
                        return force_terminal_choice(
                            trigger_stop_reason=(
                                "camera_constraint_contract_invalid"
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
                if (
                    functional_soft_contract_active
                    and gate_result.ready
                    and pending_request is not None
                ):
                    return force_terminal_choice(
                        trigger_stop_reason=(
                            "fixed_views_insufficient"
                        )
                    )
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
                return force_terminal_choice(
                    trigger_stop_reason=budget_stop,
                )

            if camera_constraints is None:
                if (
                    functional_soft_contract_active
                    and gate_result.ready
                    and pending_request is not None
                ):
                    return force_terminal_choice(
                        trigger_stop_reason=(
                            "camera_constraint_contract_invalid"
                        )
                    )
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
                if functional_soft_contract_active:
                    return force_terminal_choice(
                        trigger_stop_reason=(
                            "camera_repair_policy_invalid"
                        )
                    )
                return unresolved(
                    reason=(
                        "trusted camera repair policy could not be "
                        "validated"
                    ),
                    stop_reason="camera_constraint_contract_invalid",
                )

            if (
                self.candidate_bank_builder is not None
                and candidate_bank_episode
                != acquisition_state.episode_index
                and (
                    self.production_candidate_previews_required
                    or not candidates
                    or all(
                        isinstance(item, dict)
                        and item.get("technical_feasibility") is True
                        for item in candidates
                    )
                )
            ):
                bank_request = _build_selection_request(
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
                    repair_plans=(),
                    deterministic_selection=None,
                    selector_calls=selector_calls,
                    actions_used=actions_used,
                    rounds_used=rounds_used,
                    total_images_acquired=total_images_acquired,
                )
                try:
                    bank = self.candidate_bank_builder.build(
                        bank_request,
                        constraints=camera_constraints,
                    )
                except (TypeError, ValueError) as exc:
                    trace.append(
                        {
                            "stage": "trusted_candidate_bank",
                            "episode_index": (
                                acquisition_state.episode_index
                            ),
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if functional_soft_contract_active:
                        return force_terminal_choice(
                            trigger_stop_reason=(
                                "camera_candidate_bank_failed"
                            )
                        )
                    return unresolved(
                        reason=(
                            "trusted technical camera candidate bank "
                            "could not be generated"
                        ),
                        stop_reason="camera_candidate_bank_failed",
                    )
                candidates = bank.candidates
                candidate_bank_episode = acquisition_state.episode_index
                trace.append(
                    {
                        "stage": "trusted_candidate_bank",
                        "episode_index": acquisition_state.episode_index,
                        "status": "completed",
                        "candidate_count": len(candidates),
                        "candidate_ids": [
                            str(item["id"]) for item in candidates
                        ],
                        "rejected_candidates": list(
                            deepcopy(bank.rejected_candidates)
                        ),
                        "backend": bank.backend,
                        "provenance": deepcopy(bank.provenance),
                    }
                )
            active_observations = (
                set(camera_constraints.required_observations)
                | set(camera_constraints.preserved_observations)
            )
            if (
                acquisition_state.stage == "vlm"
                and candidates
                and active_observations
                & SEMANTIC_SELECTION_OBSERVATIONS
            ):
                vlm_selection_mode_override = "candidate_only"
            elif acquisition_state.stage == "vlm" and not candidates:
                vlm_selection_mode_override = (
                    "freeform_pose"
                    if self.control.allow_freeform_pose
                    else None
                )

            while True:
                selector = self._selector_for_stage(acquisition_state.stage)
                if selector is None:
                    if (
                        functional_soft_contract_active
                        and gate_result.ready
                        and pending_request is not None
                    ):
                        return force_terminal_choice(
                            trigger_stop_reason=(
                                "camera_selector_unavailable"
                            )
                        )
                    return unresolved(
                        reason=(
                            f"{acquisition_state.stage} camera selector is "
                            "not configured"
                        ),
                        stop_reason="camera_selector_unavailable",
                    )
                if (
                    acquisition_state.stage == "vlm"
                    and not candidates
                    and not self.control.allow_freeform_pose
                ):
                    trace.append(
                        {
                            "stage": "trusted_candidate_bank",
                            "episode_index": acquisition_state.episode_index,
                            "status": "empty_terminalized",
                            "candidate_count": 0,
                        }
                    )
                    return force_terminal_choice(
                        trigger_stop_reason="trusted_candidate_bank_empty"
                    )
                stage_stop = _stage_budget_stop(
                    control=self.control,
                    policy_source=(
                        self.camera_acquisition_policy_source
                    ),
                    state=acquisition_state,
                )
                if stage_stop is not None:
                    return force_terminal_choice(
                        trigger_stop_reason=stage_stop,
                    )
                repair_plans = _repair_plans_for_vlm(
                    constraints=camera_constraints,
                    deterministic_selection=last_deterministic_selection,
                    max_plans=self.control.vlm_max_repair_plans,
                )
                if (
                    acquisition_state.stage == "vlm"
                    and candidates
                    and not repair_plans
                ):
                    # VLM-only selection and non-conflict deterministic
                    # exhaustion choose among trusted previews. Repair-plan
                    # mode is reserved for an actual diagnosed conflict.
                    vlm_selection_mode_override = "candidate_only"
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
                    vlm_selection_mode_override=(
                        vlm_selection_mode_override
                    ),
                )
                if (
                    acquisition_state.stage == "vlm"
                    and candidates
                    and vlm_selection_mode_override
                    == "candidate_only"
                    and preview_episode
                    != acquisition_state.episode_index
                ):
                    requires_previews = bool(
                        getattr(
                            selector,
                            "requires_candidate_previews",
                            False,
                        )
                    )
                    if self.candidate_preview_renderer is None:
                        if requires_previews:
                            if (
                                functional_terminal_recovery_active
                                and gate_result.ready
                                and pending_request is not None
                            ):
                                trace.append(
                                    {
                                        "stage": (
                                            "candidate_preview_render"
                                        ),
                                        "episode_index": (
                                            acquisition_state.episode_index
                                        ),
                                        "status": "unavailable",
                                        "fallback": (
                                            "forced_binary_with_retained_"
                                            "evidence"
                                        ),
                                    }
                                )
                                return force_terminal_choice(
                                    trigger_stop_reason=(
                                        "camera_preview_renderer_"
                                        "unavailable"
                                    )
                                )
                            return unresolved(
                                reason=(
                                    "production semantic camera selection "
                                    "requires trusted candidate previews"
                                ),
                                stop_reason=(
                                    "camera_preview_renderer_unavailable"
                                ),
                            )
                    else:
                        try:
                            preview_result = (
                                self.candidate_preview_renderer.render(
                                    selection_request
                                )
                            )
                        except Exception as exc:
                            trace.append(
                                {
                                    "stage": "candidate_preview_render",
                                    "episode_index": (
                                        acquisition_state.episode_index
                                    ),
                                    "status": "failed",
                                    "error": (
                                        f"{type(exc).__name__}: {exc}"
                                    ),
                                    "fallback": (
                                        "forced_binary_with_retained_"
                                        "evidence"
                                        if (
                                            functional_terminal_recovery_active
                                            and gate_result.ready
                                            and pending_request is not None
                                        )
                                        else None
                                    ),
                                }
                            )
                            if (
                                functional_terminal_recovery_active
                                and gate_result.ready
                                and pending_request is not None
                            ):
                                return force_terminal_choice(
                                    trigger_stop_reason=(
                                        "camera_preview_render_failed"
                                    )
                                )
                            return unresolved(
                                reason=(
                                    "trusted camera candidate preview "
                                    "rendering failed"
                                ),
                                stop_reason="camera_preview_render_failed",
                            )
                        candidates = tuple(
                            deepcopy(preview_result.candidates)
                        )
                        preview_episode = (
                            acquisition_state.episode_index
                        )
                        selection_request = replace(
                            selection_request,
                            candidate_views=candidates,
                        )
                        preview_count = int(
                            preview_result.provenance.get(
                                "preview_render_count",
                                len(candidates),
                            )
                        )
                        trace.append(
                            {
                                "stage": "candidate_preview_render",
                                "episode_index": (
                                    acquisition_state.episode_index
                                ),
                                "status": "completed",
                                "candidate_ids": [
                                    str(item["id"])
                                    for item in candidates
                                ],
                                "manifest_path": (
                                    preview_result.manifest_path
                                ),
                                "provenance": deepcopy(
                                    preview_result.provenance
                                ),
                            }
                        )
                        telemetry.record_preview_render(
                            stage="vlm",
                            preview_count=preview_count,
                            gpu_time_seconds=_render_duration(
                                preview_result.provenance
                            ),
                            evidence_round=rounds_used + 1,
                            episode_index=(
                                acquisition_state.episode_index
                            ),
                            provenance=preview_result.provenance,
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
                    # Selection is reached only after this packet passed the
                    # deterministic integrity gate and the Judge requested a
                    # repair.  A selector engineering failure must not erase
                    # that still-valid packet or leave the scientific decision
                    # open: ask the Judge for one final bounded choice using
                    # the retained evidence.  The selector failure remains a
                    # separately auditable degraded condition and is never a
                    # normal deterministic-to-VLM escalation.
                    if (
                        gate_result.ready
                        and pending_request is not None
                        and (
                            functional_soft_contract_active
                            or (
                                last_judge is not None
                                and last_judge.status
                                == "need_more_evidence"
                            )
                        )
                    ):
                        trace.append(
                            {
                                "stage": "terminal_choice_policy",
                                "outcome": (
                                    "forced_with_retained_evidence"
                                ),
                                "trigger_stop_reason": (
                                    "camera_selector_failed"
                                ),
                                "degraded": True,
                                "failure_kind": failure_kind,
                                "failure_error": failure_error,
                                "selection_stage": (
                                    acquisition_state.stage
                                ),
                                "retained_evidence": _evidence_refs(
                                    evidence
                                ),
                            }
                        )
                        return force_terminal_choice(
                            trigger_stop_reason=(
                                "camera_selector_failed"
                            ),
                        )
                    return unresolved(
                        reason=(
                            "camera selector failed without a previously "
                            "validated evidence packet"
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
                    if (
                        escalation_reason
                        == "semantic_selection_required"
                    ):
                        vlm_selection_mode_override = (
                            "candidate_only"
                            if candidates
                            else "freeform_pose"
                            if self.control.allow_freeform_pose
                            else None
                        )
                    continue
                if acquisition_state.stage == "vlm":
                    acquisition_state.mark_vlm_failed(selection.outcome)
                return force_terminal_choice(
                    trigger_stop_reason=(
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
                return force_terminal_choice(
                    trigger_stop_reason=str(
                        render_execution.stop_reason
                    ),
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
                failure_artifacts = (
                    _acquired_artifacts_from_provenance(
                        failure_provenance,
                        fallback=rejected_evidence,
                    )
                )
                if failure_artifacts:
                    register_artifacts(failure_artifacts)
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
            rendered_artifacts = _acquired_artifacts_from_provenance(
                rendered.provenance,
                fallback=list(rendered.visual_evidence),
            )
            register_artifacts(rendered_artifacts)
            if total_images_acquired > self.control.max_total_images:
                rounds_used += 1
                acquisition_state.record_render_round()
                manifest_path = rendered.manifest_path
                trace.append(
                    {
                        "stage": "render",
                        "selection_stage": (
                            acquisition_state.last_render_stage
                        ),
                        "evidence_round": rounds_used,
                        "status": "rejected_budget_overrun",
                        "result": rendered.to_dict(),
                        "rendered_view_count": rendered_view_count,
                        "acquired_artifact_count": len(
                            rendered_artifacts
                        ),
                        "total_images_acquired": (
                            total_images_acquired
                        ),
                        "max_total_images": (
                            self.control.max_total_images
                        ),
                        "accepted_for_judging": False,
                    }
                )
                telemetry.record_render(
                    stage=str(
                        acquisition_state.last_render_stage
                    ),
                    preview_count=_render_count(
                        rendered.provenance,
                        "preview_render_count",
                        default=0,
                    ),
                    full_count=_render_count(
                        rendered.provenance,
                        "full_render_count",
                        default=len(rendered_artifacts),
                    ),
                    gpu_time_seconds=_render_duration(
                        rendered.provenance
                    ),
                    evidence_round=rounds_used,
                    episode_index=acquisition_state.episode_index,
                )
                return force_terminal_choice(
                    trigger_stop_reason=(
                        "max_total_images_exhausted"
                    ),
                )
            window_event: dict[str, Any] | None = None
            if evidence_window is not None:
                evidence, window_event = (
                    _compose_bounded_evidence_window(
                        evidence_window,
                        previous=evidence,
                        additions=list(rendered.visual_evidence),
                        trigger="camera_render",
                    )
                )
                window_event["camera_selector_invoked"] = True
                evidence_window_events.append(deepcopy(window_event))
            else:
                evidence = _merge_evidence(
                    evidence,
                    rendered,
                    preserve_global_anchor=(
                        camera_constraints.require_global_anchor
                    ),
                )
            current_fingerprint = _evidence_fingerprint(evidence)
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
                    **(
                        {"evidence_window": deepcopy(window_event)}
                        if window_event is not None
                        else {}
                    ),
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
                    if (
                        functional_soft_contract_active
                        and pending_request is not None
                    ):
                        return force_terminal_choice(
                            trigger_stop_reason=(
                                "evidence_packet_unchanged"
                            )
                        )
                    continue
                if (
                    gate_after_no_change.ready
                    and functional_soft_contract_active
                    and pending_request is not None
                ):
                    return force_terminal_choice(
                        trigger_stop_reason=(
                            "evidence_packet_unchanged"
                        )
                    )
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
        acquisition_ledger: dict[str, Any],
        control_manifest_path: str | Path | None,
        judge_request: JudgeRequest,
        acquisition_state: CameraAcquisitionState,
        telemetry: CameraExperimentTelemetry,
        evidence_window: dict[str, Any] | None,
        initial_window_artifact_ids: list[str],
        evidence_window_events: list[dict[str, Any]],
        functional_soft_evidence_contract: dict[str, Any],
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
            acquisition_ledger=acquisition_ledger,
        )
        if evidence_window is not None:
            audit["evidence_window"] = {
                "schema_version": "bounded_evidence_window_v1",
                "policy": evidence_window["policy"],
                "group_id": evidence_window.get("group_id"),
                "check_id": evidence_window.get("check_id"),
                "max_active_images": evidence_window[
                    "max_active_images"
                ],
                "fixed_artifact_ids": list(
                    evidence_window["fixed_artifact_ids"]
                ),
                "initial_artifact_ids": list(
                    initial_window_artifact_ids
                ),
                "final_artifact_ids": [
                    _evidence_artifact_id(item) for item in evidence
                ],
                "events": deepcopy(evidence_window_events),
                "physical_artifacts_deleted": False,
            }
        if functional_soft_evidence_contract.get("active") is True:
            audit["functional_soft_evidence_contract"] = deepcopy(
                functional_soft_evidence_contract
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


def _functional_atomic_required_check(
    request: JudgeRequest,
) -> dict[str, Any] | None:
    if request.metric != "functional_consistency":
        return None
    packet = request.context.get("functional_probe_evidence")
    packet = packet if isinstance(packet, dict) else {}
    values = packet.get("required_checks")
    if not isinstance(values, list):
        values = request.context.get("required_functional_checks")
    checks = [
        deepcopy(item)
        for item in values or []
        if isinstance(item, dict)
    ]
    if not checks:
        return None
    if len(checks) == 1:
        check = checks[0]
    else:
        target_sets = {
            tuple(sorted(_functional_check_target_ids(item)))
            for item in checks
        }
        if (
            len(target_sets) != 1
            or any(
                str(item.get("owner_stage") or "")
                != "cross_group_relation"
                for item in checks
            )
        ):
            return None
        source_check_ids = [
            str(item.get("check_id") or "") for item in checks
        ]
        check = {
            "check_id": "cross_relation_episode::"
            + "::".join(source_check_ids),
            "source_check_ids": source_check_ids,
            "check_type": "cross_group_relation_bundle",
            "owner_stage": "cross_group_relation",
            "target_ids": list(next(iter(target_sets))),
            "group_ids": list(
                dict.fromkeys(
                    str(group_id)
                    for item in checks
                    for group_id in item.get("group_ids") or []
                    if str(group_id).strip()
                )
            ),
            "relation": next(
                (
                    deepcopy(item.get("relation"))
                    for item in checks
                    if item.get("relation") is not None
                ),
                None,
            ),
            "joint_task": next(
                (
                    str(item.get("joint_task"))
                    for item in checks
                    if str(item.get("joint_task") or "").strip()
                ),
                "",
            ),
            "constraint_kind": list(
                dict.fromkeys(
                    str(item.get("constraint_kind") or "")
                    for item in checks
                    if str(item.get("constraint_kind") or "").strip()
                )
            ),
            "failure_condition": list(
                dict.fromkeys(
                    str(item.get("failure_condition") or "")
                    for item in checks
                    if str(item.get("failure_condition") or "").strip()
                )
            ),
            "required_observations": list(
                dict.fromkeys(
                    str(observation)
                    for item in checks
                    for observation in (
                        item.get("required_observations")
                        or item.get("observation_kinds")
                        or []
                    )
                    if str(observation).strip()
                )
            ),
            "target_affordances": [
                deepcopy(affordance)
                for item in checks
                for affordance in item.get("target_affordances") or []
                if isinstance(affordance, dict)
            ],
        }
    if (
        not str(check.get("check_id") or "").strip()
        or not _functional_check_target_ids(check)
    ):
        return None
    return check


def _functional_check_ids(check: dict[str, Any]) -> tuple[str, ...]:
    values = check.get("source_check_ids") or [check.get("check_id")]
    return tuple(
        dict.fromkeys(
            str(item) for item in values if str(item).strip()
        )
    )


def _functional_check_target_ids(
    check: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(item)
            for item in check.get("target_ids") or []
            if str(item).strip() and str(item) != "scene"
        )
    )


def _functional_check_observations(
    check: dict[str, Any],
) -> tuple[str, ...]:
    values = (
        check.get("required_observations")
        or check.get("observation_kinds")
        or ["target_visible"]
    )
    return tuple(
        dict.fromkeys(
            str(item) for item in values if str(item).strip()
        )
    )


def _functional_check_evidence_request(
    check: dict[str, Any],
    *,
    source: str,
    reason: str | None = None,
) -> EvidenceRequest:
    observations = _functional_check_observations(check)
    return EvidenceRequest(
        target_ids=_functional_check_target_ids(check),
        missing_observations=observations,
        view_goal=(
            "show the typed Functional targets and the geometry needed for "
            "the declared observations"
        ),
        metadata={
            "source": str(source),
            "check_ids": list(_functional_check_ids(check)),
            "check_type": str(
                check.get("check_type")
                or check.get("predicate")
                or "functional_check"
            ),
            **({"reason": str(reason)} if reason else {}),
            "decision_authority": "none",
        },
    )


def _active_functional_preflight(
    request: JudgeRequest,
    *,
    evidence: list[Any],
) -> dict[str, Any] | None:
    value = request.context.get("functional_evidence_preflight")
    if not isinstance(value, dict) or value.get("active") is not True:
        return None
    initial = {
        _soft_evidence_artifact_id(item)
        for item in value.get("initial_evidence_refs") or []
        if str(item).strip()
    }
    current = {
        _soft_evidence_artifact_id(item) for item in evidence
    }
    if current - initial:
        return None
    return deepcopy(value)


def _soft_evidence_artifact_id(value: Any) -> str:
    try:
        return _evidence_artifact_id(value)
    except (TypeError, ValueError):
        return str(value)


def _functional_preflight_evidence_request(
    check: dict[str, Any],
    preflight: dict[str, Any],
) -> EvidenceRequest:
    allowed_observations = set(
        _functional_check_observations(check)
    )
    missing = tuple(
        dict.fromkeys(
            str(item)
            for item in preflight.get("missing_observations") or []
            if str(item) in allowed_observations
        )
    )
    if not missing:
        missing = tuple(
            item
            for item in (
                "interaction_side_visible",
                "front_back_disambiguated",
                "target_visible",
            )
            if item in allowed_observations
        ) or _functional_check_observations(check)
    allowed_targets = set(_functional_check_target_ids(check))
    targets = tuple(
        dict.fromkeys(
            str(item)
            for item in preflight.get("target_ids") or []
            if str(item) in allowed_targets
        )
    ) or _functional_check_target_ids(check)
    return EvidenceRequest(
        target_ids=targets,
        missing_observations=missing,
        view_goal=(
            "compare deterministic object-centric opposing views so the "
            "usable or interaction side and front/back direction become "
            "visually interpretable"
        ),
        metadata={
            "source": "functional_evidence_preflight_v1",
            "check_ids": list(_functional_check_ids(check)),
            "check_type": str(
                check.get("check_type")
                or check.get("predicate")
                or "functional_check"
            ),
            "reason_codes": deepcopy(
                preflight.get("reason_codes") or []
            ),
            "usable_side_fallback": True,
            "unresolved_usable_side_target_ids": list(targets),
            "fallback_policy": (
                "deterministic_opposing_target_views_then_selector_review"
            ),
            "decision_authority": "none",
        },
    )


def _build_functional_readiness_request(
    request: JudgeRequest,
    *,
    check: dict[str, Any],
    evidence: list[Any],
    budget: dict[str, int],
    evidence_window: dict[str, Any] | None,
) -> EvidenceReadinessRequest:
    packet = request.context.get("functional_probe_evidence")
    packet = packet if isinstance(packet, dict) else {}
    boundary = packet.get("boundary_clearance_evidence")
    boundary = boundary if isinstance(boundary, dict) else {}
    preflight = request.context.get("functional_evidence_preflight")
    group_scope = request.context.get("group_scope")
    owner_stage = str(check.get("owner_stage") or "")
    return EvidenceReadinessRequest(
        task=request.task,
        metric=request.metric,
        check_id=str(check.get("check_id") or ""),
        check_type=str(
            check.get("check_type")
            or check.get("predicate")
            or "functional_check"
        ),
        target_ids=_functional_check_target_ids(check),
        scene=deepcopy(request.scene_context),
        required_observations=_functional_check_observations(check),
        current_visual_evidence=tuple(deepcopy(evidence)),
        budget={
            str(key): int(value)
            for key, value in budget.items()
            if isinstance(value, int) and not isinstance(value, bool)
        },
        context={
            "evidence_phase": request.context.get("evidence_phase"),
            "route_scope": (
                "cross_group"
                if owner_stage == "cross_group_relation"
                else "group_local"
            ),
            "group_scope": (
                deepcopy(group_scope)
                if isinstance(group_scope, dict)
                else {}
            ),
            "functional_check": {
                key: deepcopy(check[key])
                for key in (
                    "check_id",
                    "source_check_ids",
                    "check_type",
                    "predicate",
                    "target_ids",
                    "group_ids",
                    "owning_group_id",
                    "relation",
                    "joint_task",
                    "constraint_kind",
                    "failure_condition",
                    "required_observations",
                    "observation_goals",
                    "target_affordances",
                )
                if key in check
            },
            "functional_evidence_preflight": (
                deepcopy(preflight)
                if isinstance(preflight, dict)
                else {}
            ),
            "usable_surface_hypotheses": [
                deepcopy(item)
                for item in boundary.get(
                    "usable_surface_hypotheses"
                )
                or []
                if isinstance(item, dict)
                and str(item.get("target_id") or "")
                in set(_functional_check_target_ids(check))
            ],
            "evidence_window_summary": (
                {
                    "policy": evidence_window.get("policy"),
                    "group_id": evidence_window.get("group_id"),
                    "check_id": evidence_window.get("check_id"),
                    "max_active_images": evidence_window.get(
                        "max_active_images"
                    ),
                    "fixed_artifact_ids": list(
                        evidence_window.get("fixed_artifact_ids") or []
                    ),
                }
                if isinstance(evidence_window, dict)
                else {}
            ),
        },
        allow_scene_mutation=False,
    )


def _resolve_evidence_readiness_reviewer(
    *selectors: Any,
) -> Any | None:
    for selector in selectors:
        if selector is None:
            continue
        if getattr(selector, "supports_evidence_readiness", True) is False:
            continue
        if callable(
            getattr(selector, "review_evidence_readiness", None)
        ):
            return selector
    return None


def _evidence_readiness_call_count(selector: Any) -> int:
    usage = getattr(selector, "last_call_usage", None)
    value = (
        usage.get("vlm_call_count")
        if isinstance(usage, dict)
        else None
    )
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 1


def _terminal_forced_choice_judge_request(
    request: JudgeRequest,
    *,
    evidence: list[Any],
    previous_request: EvidenceRequest | None,
    trigger_stop_reason: str,
) -> JudgeRequest:
    stop_reason = str(trigger_stop_reason or "").strip()
    if not stop_reason:
        raise ValueError(
            "budget-exhaustion finalization requires a stop reason"
        )
    context = deepcopy(request.context)
    # Keep the established additive request key for compatibility. The
    # trigger reason distinguishes budget exhaustion from an exhausted camera
    # acquisition path such as no_feasible_candidate.
    context["budget_exhaustion_finalization"] = {
        "required": True,
        "trigger_stop_reason": stop_reason,
        "ambiguity_before_forcing": previous_request is not None,
        "visual_evidence_policy": (
            "all_available_then_judge_context_bounded"
        ),
        "available_visual_count": len(evidence),
        "previous_missing_observations": (
            list(previous_request.missing_observations)
            if previous_request is not None
            else []
        ),
        "previous_evidence_request": (
            previous_request.to_dict()
            if previous_request is not None
            else None
        ),
    }
    return JudgeRequest(
        task=request.task,
        metric=request.metric,
        claim_or_event=deepcopy(request.claim_or_event),
        scene_context=deepcopy(request.scene_context),
        deterministic_evidence=deepcopy(
            request.deterministic_evidence
        ),
        visual_evidence=tuple(deepcopy(evidence)),
        rubric=deepcopy(request.rubric),
        context=context,
    )


def _is_routing_screen_request(request: JudgeRequest) -> bool:
    """Return whether a Judge call only routes a later review stage."""

    context = request.context
    if not isinstance(context, dict):
        return False
    return bool(
        str(context.get("decision_mode") or "").strip().lower()
        == "screen"
        or str(context.get("evidence_phase") or "").strip().lower()
        in {"global_screen", "required_area_global_screen"}
    )


def _judge_visible_request(request: JudgeRequest) -> JudgeRequest:
    """Remove Controller-only orchestration state from the model prompt.

    The shared evidence bank may contain artifacts that are not active for the
    current atomic check.  Passing that routing structure to the Judge would
    both waste context and leak inactive evidence provenance into a semantic
    decision.  The Controller and persisted audit retain the full structure;
    the Judge receives only the selected visual packet and ordinary metric
    context.
    """

    if not any(key in request.context for key in _CONTROLLER_PRIVATE_CONTEXT_KEYS):
        return request
    visible_context = {
        key: deepcopy(value)
        for key, value in request.context.items()
        if key not in _CONTROLLER_PRIVATE_CONTEXT_KEYS
    }
    return replace(request, context=visible_context)


def _acquired_artifacts_from_provenance(
    provenance: dict[str, Any] | None,
    *,
    fallback: list[Any],
) -> list[Any]:
    if isinstance(provenance, dict):
        values = provenance.get("acquired_artifact_paths")
        if not isinstance(values, list):
            provider_usage = provenance.get("provider_usage")
            values = (
                provider_usage.get("acquired_artifact_paths")
                if isinstance(provider_usage, dict)
                else None
            )
        if isinstance(values, list) and values:
            return list(deepcopy(values))
    return list(deepcopy(fallback))


def _current_acquisition_ledger(
    *,
    artifact_ids: set[str],
    total_images_acquired: int,
    rounds_used: int,
    selector_calls: int,
    actions_used: int,
    state: CameraAcquisitionState,
) -> dict[str, Any]:
    """Serialize this Controller episode's budget state for its caller."""

    return {
        "schema_version": "metric_camera_acquisition_ledger_v1",
        "artifact_ids": sorted(artifact_ids),
        "total_images_acquired": int(total_images_acquired),
        "evidence_rounds": int(rounds_used),
        "selector_calls": int(selector_calls),
        "camera_actions": int(actions_used),
        "deterministic_rounds": int(
            state.total_deterministic_rounds_used
        ),
        "vlm_rounds": int(state.total_vlm_rounds_used),
    }


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


def _allowed_group_evidence_targets(
    request: JudgeRequest,
) -> tuple[str, ...]:
    group_members = _protected_group_targets(request)
    external = request.context.get(
        "allowed_external_evidence_target_ids"
    )
    external_ids = (
        [
            str(value)
            for value in external
            if isinstance(value, (str, int)) and str(value).strip()
        ]
        if isinstance(external, list)
        else []
    )
    return tuple(dict.fromkeys((*group_members, *external_ids)))


def _register_pending_placement_check(
    request: JudgeRequest,
    *,
    raw_response: Any,
    evidence_request: EvidenceRequest,
) -> tuple[JudgeRequest, dict[str, Any] | None]:
    """Promote a typed placement proposal into the next immutable request."""

    if request.metric != "semantic_placement_consistency":
        return request, None
    metadata = evidence_request.metadata
    proposal = (
        metadata.get("placement_check_proposal")
        if isinstance(metadata, dict)
        else None
    )
    if proposal is None and isinstance(raw_response, dict):
        response_request = raw_response.get("evidence_request")
        response_metadata = (
            response_request.get("metadata")
            if isinstance(response_request, dict)
            and isinstance(response_request.get("metadata"), dict)
            else {}
        )
        proposal = response_metadata.get(
            "placement_check_proposal"
        )
    if proposal is None:
        return request, None
    from benchmark.evaluator.scene_quality.placement_checks import (
        build_pending_placement_check,
    )

    context = deepcopy(request.context)
    group_scope = context.get("group_scope")
    group_members = (
        {
            str(item)
            for item in group_scope.get("member_ids") or []
            if str(item).strip()
        }
        if isinstance(group_scope, dict)
        else set()
    )
    if group_members:
        known_ids = group_members
    else:
        known_ids = {
            str(item.get("id") or item.get("object_id"))
            for item in request.scene_context.get("objects") or []
            if isinstance(item, dict)
            and (item.get("id") or item.get("object_id"))
        }
    if not known_ids:
        raise ValueError(
            "placement proposal cannot be registered without trusted "
            "object identities"
        )
    proposal_subject = (
        str(proposal.get("subject_id") or "")
        if isinstance(proposal, dict)
        else ""
    )
    if proposal_subject not in set(evidence_request.target_ids):
        raise ValueError(
            "placement proposal subject must be included in the evidence "
            "request target IDs"
        )
    groups = context.get("object_groups")
    groups = groups if isinstance(groups, list) else []
    check = build_pending_placement_check(
        proposal,
        known_ids=known_ids,
        groups=groups,
        source_ref=(
            f"controller_evidence_request_round:"
            f"{context.get('evidence_phase') or 'unknown'}"
        ),
    )
    phase = str(context.get("evidence_phase") or "")
    expected_owner = (
        "scene_global"
        if phase in {
            "global_discovery",
            "residual_global_placement_review",
        }
        else "group_local"
        if phase in {"group_local_review", "initial_visual"}
        else None
    )
    owner_stage = str(check.get("owner_stage") or "")
    deferred_to_group = bool(
        expected_owner == "scene_global"
        and owner_stage == "group_local"
    )
    if (
        expected_owner is not None
        and owner_stage != expected_owner
        and not deferred_to_group
    ):
        raise ValueError(
            "placement evidence-request proposal is routed to the wrong "
            f"Judge stage: expected {expected_owner!r}"
        )
    collection_key = (
        "deferred_placement_checks"
        if deferred_to_group
        else "required_placement_checks"
    )
    required = [
        deepcopy(item)
        for item in context.get(collection_key) or []
        if isinstance(item, dict)
    ]
    by_id = {
        str(item.get("check_id") or ""): item
        for item in required
    }
    prior = by_id.get(str(check["check_id"]))
    if prior is not None:
        identity_fields = (
            "check_type",
            "subject_id",
            "owner_stage",
            "owning_group_id",
        )
        if any(
            prior.get(key) != check.get(key) for key in identity_fields
        ) or sorted(prior.get("context_ids") or []) != sorted(
            check.get("context_ids") or []
        ):
            raise ValueError(
                "placement proposal collides with an existing check identity"
            )
        registered = prior
    else:
        if deferred_to_group:
            check["handoff_status"] = "deferred_to_group_local"
            check["handoff_from_stage"] = "scene_global"
        required.append(deepcopy(check))
        registered = check
    context[collection_key] = required
    if deferred_to_group:
        return (
            JudgeRequest(
                task=request.task,
                metric=request.metric,
                claim_or_event=deepcopy(request.claim_or_event),
                scene_context=deepcopy(request.scene_context),
                deterministic_evidence=deepcopy(
                    request.deterministic_evidence
                ),
                visual_evidence=tuple(
                    deepcopy(request.visual_evidence)
                ),
                rubric=deepcopy(request.rubric),
                context=context,
            ),
            deepcopy(registered),
        )
    response_contract = (
        deepcopy(context.get("response_contract"))
        if isinstance(context.get("response_contract"), dict)
        else {}
    )
    placement_contract = (
        deepcopy(response_contract.get("placement_check_results"))
        if isinstance(
            response_contract.get("placement_check_results"),
            dict,
        )
        else {}
    )
    placement_contract.update(
        required=True,
        exact_check_ids=[
            str(item.get("check_id") or "")
            for item in required
        ],
    )
    response_contract["placement_check_results"] = (
        placement_contract
    )
    context["response_contract"] = response_contract
    return (
        JudgeRequest(
            task=request.task,
            metric=request.metric,
            claim_or_event=deepcopy(request.claim_or_event),
            scene_context=deepcopy(request.scene_context),
            deterministic_evidence=deepcopy(
                request.deterministic_evidence
            ),
            visual_evidence=tuple(
                deepcopy(request.visual_evidence)
            ),
            rubric=deepcopy(request.rubric),
            context=context,
        ),
        deepcopy(registered),
    )
