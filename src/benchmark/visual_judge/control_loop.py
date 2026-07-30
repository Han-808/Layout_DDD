from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from benchmark.utils.io import write_json
from benchmark.visual_judge.control_config import (
    VLMEvaluationControl,
    resolve_vlm_evaluation_control,
)
from benchmark.visual_judge.evidence_gate import DeterministicEvidenceGate
from benchmark.visual_judge.interfaces import (
    CameraSelectionRequest,
    CameraSelectionResult,
    CameraSelector,
    EvidenceGate,
    EvidenceGateRequest,
    EvidenceGateResult,
    EvidenceRequest,
    HybridCameraSelector,
    Judge,
    JudgeRequest,
    JudgeResult,
    build_camera_selector,
    camera_selection_result_from_value,
)


VLM_CONTROL_LOOP_VERSION = "vlm_evaluation_control_loop_v1"
EVALUATION_STATUSES = {"valid", "invalid", "unresolved"}
EVIDENCE_MERGE_POLICIES = {"append", "replace"}


class EvidenceRenderFailure(RuntimeError):
    """Render failure carrying verifiable usage already consumed by a backend."""

    def __init__(
        self,
        message: str,
        *,
        internal_selector_calls: int = 0,
        camera_actions_executed: int = 0,
        visual_evidence: tuple[Any, ...] = (),
        provenance: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.internal_selector_calls = _nonnegative_int(
            internal_selector_calls,
            "render failure internal_selector_calls",
        )
        self.camera_actions_executed = _nonnegative_int(
            camera_actions_executed,
            "render failure camera_actions_executed",
        )
        self.visual_evidence = tuple(deepcopy(visual_evidence))
        self.provenance = deepcopy(provenance or {})


@dataclass(frozen=True)
class EvidenceRenderRequest:
    """Input to an injected renderer or existing evidence-provider adapter."""

    judge_request: JudgeRequest
    selection: CameraSelectionResult
    evidence_goal: dict[str, Any]
    previous_visual_evidence: tuple[Any, ...]
    evidence_round: int
    budget: dict[str, int]
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = self.judge_request.to_dict()
        result.update(
            {
                "selection": self.selection.to_dict(),
                "selected_view_ids": list(self.selection.selected_view_ids),
                "camera_proposal": deepcopy(self.selection.camera_proposal),
                "camera_actions": list(deepcopy(self.selection.camera_actions)),
                "evidence_goal": deepcopy(self.evidence_goal),
                "previous_visual_evidence": list(
                    deepcopy(self.previous_visual_evidence)
                ),
                "evidence_round": self.evidence_round,
                "budget": deepcopy(self.budget),
            }
        )
        for key, value in self.context.items():
            result.setdefault(key, deepcopy(value))
        return result


@dataclass(frozen=True)
class EvidenceRenderResult:
    """Rendered evidence packet without any metric decision."""

    visual_evidence: tuple[Any, ...]
    merge_policy: str = "replace"
    camera_actions_executed: int = 0
    manifest_path: str | None = None
    next_candidate_views: tuple[dict[str, Any], ...] = ()
    next_allowed_actions: tuple[str, ...] = ()
    replaces_candidate_views: bool = False
    replaces_allowed_actions: bool = False
    backend: str = "unknown"
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        default_merge_policy: str = "replace",
        default_backend: str = "unknown",
        proposed_action_count: int = 0,
        authorized_action_count: int | None = None,
    ) -> EvidenceRenderResult:
        if isinstance(value, cls):
            result = value
        elif isinstance(value, (list, tuple)):
            result = cls(
                visual_evidence=tuple(deepcopy(value)),
                merge_policy=default_merge_policy,
                camera_actions_executed=proposed_action_count,
                backend=default_backend,
            )
        elif isinstance(value, dict):
            forbidden = [
                key
                for key in (
                    "verdict",
                    "score",
                    "scene_mutation",
                    "scene_patch",
                )
                if key in value
            ]
            if forbidden:
                raise ValueError(
                    "evidence renderer must not return metric verdict, score, "
                    "or scene mutation"
                )
            visual_evidence = value.get("visual_evidence")
            if visual_evidence is None:
                visual_evidence = value.get("render_evidence")
            if not isinstance(visual_evidence, (list, tuple)):
                raise ValueError(
                    "evidence renderer visual_evidence must be a JSON list"
                )
            replaces_candidates = "next_candidate_views" in value
            replaces_actions = "next_allowed_actions" in value
            next_candidates = value.get("next_candidate_views") or []
            next_actions = value.get("next_allowed_actions") or []
            if not isinstance(next_candidates, (list, tuple)) or not all(
                isinstance(item, dict) for item in next_candidates
            ):
                raise ValueError(
                    "evidence renderer next_candidate_views must be a JSON list"
                )
            if not isinstance(next_actions, (list, tuple)) or not all(
                isinstance(item, str) and item for item in next_actions
            ):
                raise ValueError(
                    "evidence renderer next_allowed_actions must be a JSON list"
                )
            provenance = value.get("provenance") or {}
            if not isinstance(provenance, dict):
                raise ValueError(
                    "evidence renderer provenance must be a JSON object"
                )
            actions_executed = value.get(
                "camera_actions_executed",
                proposed_action_count,
            )
            result = cls(
                visual_evidence=tuple(deepcopy(visual_evidence)),
                merge_policy=str(
                    value.get("merge_policy") or default_merge_policy
                ),
                camera_actions_executed=_nonnegative_int(
                    actions_executed,
                    "camera_actions_executed",
                ),
                manifest_path=(
                    str(value["manifest_path"])
                    if value.get("manifest_path")
                    else None
                ),
                next_candidate_views=tuple(deepcopy(next_candidates)),
                next_allowed_actions=tuple(next_actions),
                replaces_candidate_views=replaces_candidates,
                replaces_allowed_actions=replaces_actions,
                backend=str(value.get("backend") or default_backend),
                provenance=deepcopy(provenance),
            )
        else:
            raise ValueError(
                "evidence renderer response must be a JSON object or list"
            )
        if result.merge_policy not in EVIDENCE_MERGE_POLICIES:
            raise ValueError(
                "evidence renderer merge_policy must be append or replace"
            )
        _nonnegative_int(
            result.camera_actions_executed,
            "camera_actions_executed",
        )
        if authorized_action_count is None:
            if result.camera_actions_executed != proposed_action_count:
                raise ValueError(
                    "evidence renderer camera_actions_executed must match the "
                    "validated CameraSelector action count"
                )
        else:
            authorized = _nonnegative_int(
                authorized_action_count,
                "authorized_action_count",
            )
            if not (
                proposed_action_count
                <= result.camera_actions_executed
                <= authorized
            ):
                raise ValueError(
                    "evidence renderer camera_actions_executed must stay "
                    "within the trusted composite-provider reservation"
                )
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "visual_evidence": list(deepcopy(self.visual_evidence)),
            "merge_policy": self.merge_policy,
            "camera_actions_executed": self.camera_actions_executed,
            "manifest_path": self.manifest_path,
            "next_candidate_views": list(
                deepcopy(self.next_candidate_views)
            ),
            "next_allowed_actions": list(self.next_allowed_actions),
            "replaces_candidate_views": self.replaces_candidate_views,
            "replaces_allowed_actions": self.replaces_allowed_actions,
            "backend": self.backend,
            "provenance": deepcopy(self.provenance),
        }


@runtime_checkable
class EvidenceRenderer(Protocol):
    def render(self, request: EvidenceRenderRequest) -> EvidenceRenderResult: ...


class ExistingEvidenceRendererAdapter:
    """Wrap an existing callable evidence provider without changing its packet."""

    def __init__(
        self,
        provider: Callable[[dict[str, Any]], Any],
        *,
        merge_policy: str = "replace",
        backend: str = "existing",
    ) -> None:
        if not callable(provider):
            raise TypeError("existing evidence provider must be callable")
        if merge_policy not in EVIDENCE_MERGE_POLICIES:
            raise ValueError(
                "existing evidence provider merge_policy must be append or replace"
            )
        self.provider = provider
        self.merge_policy = merge_policy
        self.backend = str(backend)

    def render(self, request: EvidenceRenderRequest) -> EvidenceRenderResult:
        raw = self.provider(request.to_dict())
        result = EvidenceRenderResult.from_value(
            raw,
            default_merge_policy=self.merge_policy,
            default_backend=self.backend,
            proposed_action_count=_selection_action_count(request.selection),
        )
        provenance = deepcopy(result.provenance)
        provenance.setdefault(
            "adapter",
            f"{type(self).__module__}.{type(self).__qualname__}",
        )
        provenance.setdefault(
            "provider",
            f"{type(self.provider).__module__}.{type(self.provider).__qualname__}",
        )
        policy = getattr(self.provider, "policy_config", None)
        if isinstance(policy, dict):
            provenance.setdefault("existing_policy", deepcopy(policy))
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
        self.evidence_gate = evidence_gate or DeterministicEvidenceGate()
        self.camera_selector = _resolve_camera_selector(
            camera_selector,
            backend=self.control.camera_selector_backend,
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

        while True:
            gate_result = self._check_gate(
                request=judge_request,
                visual_evidence=evidence,
                target_ids=targets,
                evidence_goal=goal,
                manifest_path=manifest_path,
                after_render=rounds_used > 0,
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

            if total_images_acquired > self.control.max_total_images:
                return self._finish(
                    status="unresolved",
                    reason="visual evidence exceeds the resolved image budget",
                    stop_reason="max_total_images_exhausted",
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

            usage_overrun = _usage_overrun_stop_reason(
                control=self.control,
                selector_calls=selector_calls,
                camera_actions=actions_used,
            )
            if usage_overrun is not None:
                return self._finish(
                    status="unresolved",
                    reason=(
                        "observed camera usage exceeds the resolved control "
                        "budget"
                    ),
                    stop_reason=usage_overrun,
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

            if post_render_validation_error is not None:
                return self._finish(
                    status="unresolved",
                    reason=post_render_validation_error,
                    stop_reason="renderer_followup_contract_invalid",
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

            if gate_result.ready:
                judge_fingerprint = _evidence_fingerprint(evidence)
                if judge_fingerprint in judged_fingerprints:
                    return self._finish(
                        status="unresolved",
                        reason=(
                            "the unchanged evidence packet was already judged"
                        ),
                        stop_reason="evidence_packet_already_judged",
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
                judged_fingerprints.add(judge_fingerprint)
                current_request = judge_request.with_visual_evidence(evidence)
                raw_judge = self.judge.judge(current_request)
                last_judge = JudgeResult.from_value(raw_judge)
                trace.append(
                    {
                        "stage": "judge",
                        "evidence_round": rounds_used,
                        "result": last_judge.to_dict(),
                        "images_used": _evidence_refs(evidence),
                    }
                )
                if last_judge.status in {"valid", "invalid"}:
                    return self._finish(
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
                    return self._finish(
                        status="unresolved",
                        reason=last_judge.reason,
                        stop_reason="judge_evidence_request_disabled",
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
                if pending_request is None:  # Defensive; validator also checks.
                    raise ValueError(
                        "Judge need_more_evidence requires evidence_request"
                    )
                goal = _goal_from_judge_request(goal, pending_request)
                if pending_request.target_ids:
                    targets = pending_request.target_ids
            else:
                pending_request = _request_from_gate(
                    targets=targets,
                    gate_result=gate_result,
                    evidence_goal=goal,
                )
                if not gate_result.camera_repairable:
                    return self._finish(
                        status="unresolved",
                        reason=_gate_reason(gate_result),
                        stop_reason="evidence_not_camera_repairable",
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
                goal = _goal_from_gate(goal, gate_result)

            budget_stop = _budget_stop_reason(
                control=self.control,
                rounds_used=rounds_used,
                selector_calls=selector_calls,
                total_images_acquired=total_images_acquired,
            )
            if budget_stop is not None:
                return self._finish(
                    status="unresolved",
                    reason="visual evidence budget exhausted before another repair",
                    stop_reason=budget_stop,
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

            selection_request = CameraSelectionRequest(
                task=judge_request.task,
                metric=judge_request.metric,
                target_ids=targets,
                scene=deepcopy(judge_request.scene_context),
                evidence_goal=deepcopy(goal),
                existing_visual_evidence=tuple(deepcopy(evidence)),
                budget={
                    "max_views_per_round": min(
                        self.control.max_views_per_round,
                        self.control.max_total_images
                        - total_images_acquired,
                    ),
                    "remaining_images": (
                        self.control.max_total_images
                        - total_images_acquired
                    ),
                    "remaining_camera_actions": (
                        self.control.max_camera_actions - actions_used
                    ),
                    "remaining_selector_calls": (
                        self.control.max_selector_calls - selector_calls
                    ),
                    "remaining_evidence_rounds": (
                        self.control.max_evidence_rounds - rounds_used
                    ),
                },
                candidate_views=candidates,
                allowed_actions=actions,
                evidence_round=rounds_used + 1,
                allow_freeform_pose=self.control.allow_freeform_pose,
                allow_scene_mutation=self.control.allow_scene_mutation,
                context=deepcopy(selector_context),
            )
            selector_dispatch_charge = (
                0
                if getattr(
                    self.camera_selector,
                    "trusted_composite_provider_adapter",
                    False,
                )
                is True
                else 1
            )
            selector_calls += selector_dispatch_charge
            try:
                raw_selection = self.camera_selector.select(selection_request)
                selection = camera_selection_result_from_value(
                    raw_selection,
                    request=selection_request,
                    backend=str(
                        getattr(
                            self.camera_selector,
                            "backend",
                            self.control.camera_selector_backend,
                        )
                    ),
                )
            except Exception as exc:
                trace.append(
                    {
                        "stage": "camera_selector",
                        "evidence_round": rounds_used + 1,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                return self._finish(
                    status="unresolved",
                    reason=(
                        "camera selector failed; previous evidence was retained"
                        if self.control.on_selector_failure
                        == "keep_previous_evidence"
                        else "camera selector failed"
                    ),
                    stop_reason="camera_selector_failed",
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
            trace.append(
                {
                    "stage": "camera_selector",
                    "evidence_round": rounds_used + 1,
                    "status": "selected",
                    "result": selection.to_dict(),
                }
            )

            reservation = _trusted_composite_reservation(
                self.camera_selector,
                selection,
            )
            if (
                selector_calls + reservation["selector_calls"]
                > self.control.max_selector_calls
            ):
                return self._finish(
                    status="unresolved",
                    reason=(
                        "selected camera backend exceeds the remaining "
                        "selector-call budget"
                    ),
                    stop_reason="max_selector_calls_exhausted",
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
            proposed_actions = _selection_action_count(selection)
            if (
                actions_used
                + proposed_actions
                + reservation["camera_actions"]
                > self.control.max_camera_actions
            ):
                return self._finish(
                    status="unresolved",
                    reason="camera action budget exhausted before rendering",
                    stop_reason="max_camera_actions_exhausted",
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

            render_request = EvidenceRenderRequest(
                judge_request=judge_request.with_visual_evidence(evidence),
                selection=selection,
                evidence_goal=deepcopy(goal),
                previous_visual_evidence=tuple(deepcopy(evidence)),
                evidence_round=rounds_used + 1,
                budget=deepcopy(selection_request.budget),
                context=deepcopy(selector_context),
            )
            raw_render: Any = None
            try:
                raw_render = self.renderer.render(render_request)
                rendered = EvidenceRenderResult.from_value(
                    raw_render,
                    proposed_action_count=proposed_actions,
                    authorized_action_count=(
                        proposed_actions
                        + reservation["camera_actions"]
                        if reservation["trusted"]
                        else None
                    ),
                )
            except Exception as exc:
                rejected_evidence = _raw_render_evidence(raw_render)
                failure_selector_calls = 0
                failure_camera_actions = 0
                failure_provenance: dict[str, Any] = {}
                if isinstance(exc, EvidenceRenderFailure):
                    rejected_evidence = list(
                        deepcopy(exc.visual_evidence)
                    )
                    failure_selector_calls = (
                        exc.internal_selector_calls
                    )
                    failure_camera_actions = (
                        exc.camera_actions_executed
                    )
                    failure_provenance = deepcopy(exc.provenance)
                if (
                    reservation["trusted"]
                    and isinstance(raw_render, EvidenceRenderResult)
                ):
                    actions_used += raw_render.camera_actions_executed
                    selector_calls += _rendered_internal_selector_calls(
                        raw_render
                    )
                selector_calls += failure_selector_calls
                actions_used += failure_camera_actions
                failure_contract_error: str | None = None
                if reservation["trusted"]:
                    if (
                        failure_selector_calls
                        > reservation["selector_calls"]
                    ):
                        failure_contract_error = (
                            "failed renderer exceeded the trusted "
                            "composite-provider selector-call reservation"
                        )
                    if (
                        failure_camera_actions
                        > proposed_actions
                        + reservation["camera_actions"]
                    ):
                        failure_contract_error = (
                            "failed renderer exceeded the trusted "
                            "composite-provider camera-action reservation"
                        )
                elif (
                    failure_selector_calls > 0
                    or failure_camera_actions > proposed_actions
                ):
                    failure_contract_error = (
                        "failed renderer reported camera usage outside the "
                        "validated CameraSelector authorization"
                    )
                trace.append(
                    {
                        "stage": "render",
                        "evidence_round": rounds_used + 1,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "observed_internal_selector_calls": (
                            failure_selector_calls
                        ),
                        "observed_camera_actions": (
                            failure_camera_actions
                        ),
                        "provenance": failure_provenance,
                        "rejected_visual_evidence": _evidence_refs(
                            rejected_evidence
                        ),
                    }
                )
                if rejected_evidence:
                    total_images_acquired += len(rejected_evidence)
                    rejected_gate = self._check_gate(
                        request=judge_request,
                        visual_evidence=rejected_evidence,
                        target_ids=targets,
                        evidence_goal=goal,
                        manifest_path=None,
                        after_render=True,
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
                usage_overrun = _usage_overrun_stop_reason(
                    control=self.control,
                    selector_calls=selector_calls,
                    camera_actions=actions_used,
                )
                if failure_contract_error is not None:
                    failure_reason = failure_contract_error
                    failure_stop_reason = (
                        "renderer_followup_contract_invalid"
                    )
                elif usage_overrun is not None:
                    failure_reason = (
                        "failed camera evidence rendering consumed more than "
                        "the resolved control budget"
                    )
                    failure_stop_reason = usage_overrun
                else:
                    failure_reason = "camera evidence rendering failed"
                    failure_stop_reason = "render_failed"
                return self._finish(
                    status="unresolved",
                    reason=failure_reason,
                    stop_reason=failure_stop_reason,
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

            internal_selector_calls = _rendered_internal_selector_calls(
                rendered,
            )
            if (
                internal_selector_calls
                > reservation["selector_calls"]
            ):
                post_render_validation_error = (
                    "renderer exceeded the trusted composite-provider "
                    "selector-call reservation"
                )
            rendered_view_count = _rendered_view_count(
                rendered.visual_evidence
            )
            if (
                rendered_view_count
                > selection_request.budget["max_views_per_round"]
            ):
                post_render_validation_error = (
                    "renderer exceeded max_views_per_round with "
                    f"{rendered_view_count} independently rendered views"
                )

            previous_fingerprint = _evidence_fingerprint(evidence)
            evidence = _merge_evidence(evidence, rendered)
            current_fingerprint = _evidence_fingerprint(evidence)
            total_images_acquired += len(rendered.visual_evidence)
            actions_used += rendered.camera_actions_executed
            selector_calls += internal_selector_calls
            rounds_used += 1
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
            # Looping is intentional: the next operation is always EvidenceGate
            # (unless an explicit legacy-compatible override disables it).
            if previous_fingerprint == current_fingerprint:
                gate_after_no_change = self._check_gate(
                    request=judge_request,
                    visual_evidence=evidence,
                    target_ids=targets,
                    evidence_goal=goal,
                    manifest_path=manifest_path,
                    after_render=True,
                )
                trace.append(
                    {
                        "stage": "evidence_gate",
                        "evidence_round": rounds_used,
                        "result": gate_after_no_change.to_dict(),
                        "images_used": _evidence_refs(evidence),
                    }
                )
                return self._finish(
                    status="unresolved",
                    reason="camera repair did not change the evidence packet",
                    stop_reason="evidence_packet_unchanged",
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

    def _check_gate(
        self,
        *,
        request: JudgeRequest,
        visual_evidence: list[Any],
        target_ids: tuple[str, ...],
        evidence_goal: dict[str, Any],
        manifest_path: str | None,
        after_render: bool,
    ) -> EvidenceGateResult:
        should_check = self.control.evidence_gate_enabled and (
            not after_render
            or self.control.require_evidence_gate_after_render
        )
        if not should_check:
            return EvidenceGateResult(
                ready=True,
                camera_repairable=False,
                reason_codes=("evidence_gate_explicitly_disabled",),
                deficiencies=(),
                backend="disabled",
                provenance={
                    "schema_version": VLM_CONTROL_LOOP_VERSION,
                    "after_render": after_render,
                },
            )
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
        confidence: float = 0.0,
        defects: tuple[dict[str, Any], ...] = (),
    ) -> VLMEvaluationResult:
        judge_provenance = (
            deepcopy(judge_result.provenance)
            if judge_result is not None
            else {}
        )
        audit = _jsonable({
            "schema_version": VLM_CONTROL_LOOP_VERSION,
            "control": self.control.manifest(),
            "selector_backend": str(
                getattr(
                    self.camera_selector,
                    "backend",
                    self.control.camera_selector_backend,
                )
            ),
            "requested_selector_backend": (
                self.control.camera_selector_backend
            ),
            "selector_adapter": (
                f"{type(self.camera_selector).__module__}."
                f"{type(self.camera_selector).__qualname__}"
            ),
            "evidence_gate_backend": self.control.evidence_gate_backend,
            "judge_backend": (
                judge_result.backend
                if judge_result is not None
                else f"{type(self.judge).__module__}.{type(self.judge).__qualname__}"
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
            "applied_failure_policy": _policy_for_stop_reason(
                self.control,
                stop_reason,
            ),
            "trace": deepcopy(trace),
        })
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


def _resolve_camera_selector(
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


def _policy_for_stop_reason(
    control: VLMEvaluationControl,
    stop_reason: str,
) -> dict[str, str] | None:
    if stop_reason == "evidence_not_camera_repairable":
        return {
            "field": "on_non_camera_repairable_evidence",
            "value": control.on_non_camera_repairable_evidence,
        }
    if stop_reason.startswith("max_"):
        return {
            "field": "on_budget_exhausted",
            "value": control.on_budget_exhausted,
        }
    if stop_reason == "camera_selector_failed":
        return {
            "field": "on_selector_failure",
            "value": control.on_selector_failure,
        }
    if stop_reason == "render_failed":
        return {
            "field": "on_render_failure",
            "value": control.on_render_failure,
        }
    return None


def _merge_evidence(
    previous: list[Any],
    rendered: EvidenceRenderResult,
) -> list[Any]:
    additions = list(deepcopy(rendered.visual_evidence))
    if rendered.merge_policy == "replace":
        return additions
    combined = list(deepcopy(previous)) + additions
    result: list[Any] = []
    slots: dict[str, int] = {}
    for item in combined:
        slot = _evidence_slot_identity(item)
        if slot is not None and slot in slots:
            # A corrective render of the same view/representation supersedes
            # its technically deficient predecessor while retaining packet
            # ordering (for example global-first OOB evidence).
            result[slots[slot]] = item
            continue
        key = json.dumps(
            _evidence_content_identity(item),
            sort_keys=True,
            separators=(",", ":"),
        )
        seen = {
            json.dumps(
                _evidence_content_identity(existing),
                sort_keys=True,
                separators=(",", ":"),
            )
            for existing in result
        }
        if key in seen:
            continue
        if slot is not None:
            slots[slot] = len(result)
        result.append(item)
    return result


def _evidence_slot_identity(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    pose = item.get("pose")
    view_id = (
        item.get("view_id")
        or item.get("id")
        or (
            pose.get("id")
            if isinstance(pose, dict)
            else None
        )
    )
    if view_id is None or not str(view_id).strip():
        return None
    representation = (
        item.get("representation")
        or item.get("representation_type")
        or item.get("evidence_style")
        or item.get("role")
        or "default"
    )
    return f"{view_id}:{representation}"


def _raw_render_evidence(value: Any) -> list[Any]:
    if isinstance(value, EvidenceRenderResult):
        return list(deepcopy(value.visual_evidence))
    if isinstance(value, (list, tuple)):
        return list(deepcopy(value))
    if not isinstance(value, dict):
        return []
    items = value.get("visual_evidence")
    if items is None:
        items = value.get("render_evidence")
    if not isinstance(items, (list, tuple)):
        return []
    return list(deepcopy(items))


def _selection_action_count(result: CameraSelectionResult) -> int:
    return max(
        len(result.camera_actions),
        1 if result.camera_proposal is not None else 0,
    )


def _trusted_composite_reservation(
    selector: CameraSelector,
    result: CameraSelectionResult,
) -> dict[str, int | bool]:
    trusted = bool(
        getattr(
            selector,
            "trusted_composite_provider_adapter",
            False,
        )
    )
    if not trusted:
        return {
            "trusted": False,
            "selector_calls": 0,
            "camera_actions": 0,
        }
    provenance = result.provenance
    selector_calls = _nonnegative_int(
        provenance.get("max_internal_selector_calls", 0),
        "max_internal_selector_calls",
    )
    camera_actions = _nonnegative_int(
        provenance.get("max_internal_camera_actions", 0),
        "max_internal_camera_actions",
    )
    return {
        "trusted": True,
        "selector_calls": selector_calls,
        "camera_actions": camera_actions,
    }


def _rendered_internal_selector_calls(
    result: EvidenceRenderResult,
) -> int:
    return _nonnegative_int(
        result.provenance.get("internal_selector_calls", 0),
        "internal_selector_calls",
    )


def _normalize_camera_usage(
    value: dict[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {
            "observed": False,
            "selector_calls": 0,
            "camera_actions": 0,
        }
    if not isinstance(value, dict):
        raise TypeError("initial_camera_usage must be a JSON object")
    result = deepcopy(value)
    result["selector_calls"] = _nonnegative_int(
        value.get("selector_calls", 0),
        "initial_camera_usage.selector_calls",
    )
    result["camera_actions"] = _nonnegative_int(
        value.get("camera_actions", 0),
        "initial_camera_usage.camera_actions",
    )
    result["observed"] = True
    return result


def _usage_overrun_stop_reason(
    *,
    control: VLMEvaluationControl,
    selector_calls: int,
    camera_actions: int,
) -> str | None:
    if selector_calls > control.max_selector_calls:
        return "max_selector_calls_exhausted"
    if camera_actions > control.max_camera_actions:
        return "max_camera_actions_exhausted"
    return None


def _budget_stop_reason(
    *,
    control: VLMEvaluationControl,
    rounds_used: int,
    selector_calls: int,
    total_images_acquired: int,
) -> str | None:
    if rounds_used >= control.max_evidence_rounds:
        return "max_evidence_rounds_exhausted"
    if selector_calls >= control.max_selector_calls:
        return "max_selector_calls_exhausted"
    if total_images_acquired >= control.max_total_images:
        return "max_total_images_exhausted"
    return None


def _goal_from_judge_request(
    existing: dict[str, Any],
    request: EvidenceRequest,
) -> dict[str, Any]:
    result = deepcopy(existing)
    result.update(
        {
            "target_ids": list(request.target_ids),
            "missing_observations": list(request.missing_observations),
            "view_goal": request.view_goal,
            "judge_evidence_request": request.to_dict(),
        }
    )
    return result


def _goal_from_gate(
    existing: dict[str, Any],
    result: EvidenceGateResult,
) -> dict[str, Any]:
    goal = deepcopy(existing)
    goal.update(
        {
            "missing_observations": list(result.reason_codes),
            "view_goal": str(
                goal.get("view_goal") or "repair_technical_visual_evidence"
            ),
            "gate_deficiencies": list(deepcopy(result.deficiencies)),
        }
    )
    return goal


def _request_from_gate(
    *,
    targets: tuple[str, ...],
    gate_result: EvidenceGateResult,
    evidence_goal: dict[str, Any],
) -> EvidenceRequest:
    resolved_targets = targets or ("scene",)
    missing = tuple(gate_result.reason_codes) or (
        "technical_evidence_not_ready",
    )
    return EvidenceRequest(
        target_ids=resolved_targets,
        missing_observations=missing,
        view_goal=str(
            evidence_goal.get("view_goal")
            or "repair_technical_visual_evidence"
        ),
        metadata={
            "source": "evidence_gate",
            "camera_repairable": gate_result.camera_repairable,
            "deficiencies": list(deepcopy(gate_result.deficiencies)),
        },
    )


def _request_target_ids(request: JudgeRequest) -> tuple[str, ...]:
    values: list[Any] = []
    for source in (request.claim_or_event, request.context):
        for key in ("target_ids", "object_ids"):
            if isinstance(source.get(key), list):
                values.extend(source[key])
        for key in ("object_id", "subject_id", "target_id"):
            if source.get(key) is not None:
                values.append(source[key])
    resolved = tuple(
        dict.fromkeys(
            str(value) for value in values if str(value).strip()
        )
    )
    return resolved or ("scene",)


def _gate_reason(result: EvidenceGateResult) -> str:
    if result.reason_codes:
        return "visual evidence is not technically ready: " + ", ".join(
            result.reason_codes
        )
    return "visual evidence is not technically ready"


def _validate_candidates(
    candidates: tuple[dict[str, Any], ...],
) -> None:
    ids = [str(item.get("id") or "") for item in candidates]
    if any(not value for value in ids):
        raise ValueError("candidate views require non-empty IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("candidate view IDs must be unique")


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


def _rendered_view_count(items: tuple[Any, ...]) -> int:
    """Count independent camera views while keeping same-pose bundles intact."""

    identities: set[str] = set()
    for index, item in enumerate(items):
        if isinstance(item, dict):
            pose = item.get("pose")
            role = str(item.get("role") or "")
            pair_id = item.get("pair_id")
            if isinstance(pose, dict) and pose:
                value = "pose:" + json.dumps(
                    _jsonable(pose),
                    sort_keys=True,
                    separators=(",", ":"),
                )
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


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value
