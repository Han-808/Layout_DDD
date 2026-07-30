from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

from benchmark.visual_judge.control_config import VLMEvaluationControl
from benchmark.visual_judge.interfaces.camera import (
    CameraSelectionRequest,
    CameraSelectionResult,
    CameraSelector,
)
from benchmark.visual_judge.interfaces.evidence import (
    EvidenceRenderFailure,
    EvidenceRenderRequest,
    EvidenceRenderResult,
    EvidenceRenderer,
)
from benchmark.visual_judge.interfaces.judge import JudgeRequest
from benchmark.visual_judge.orchestration.audit import (
    jsonable,
    render_result_with_audit,
    rendered_view_count,
)
from benchmark.visual_judge.orchestration.budget import (
    rendered_internal_selector_calls,
    selection_action_count,
    trusted_composite_reservation,
    usage_overrun_stop_reason,
)
from benchmark.visual_judge.orchestration.camera_acquisition import (
    CameraSelectorInvocationError,
    invoke_selector_once,
    selector_dispatch_charge,
)
from benchmark.visual_judge.orchestration.evidence_packet import (
    raw_render_evidence,
)


@dataclass(frozen=True)
class SelectionExecution:
    selection: CameraSelectionResult | None
    selector_calls: int
    failure_kind: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class RenderExecution:
    rendered: EvidenceRenderResult | None
    render_request: EvidenceRenderRequest | None
    selector_calls: int
    camera_actions: int
    rendered_view_count: int = 0
    rejected_visual_evidence: tuple[Any, ...] = ()
    failure_kind: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    reason: str | None = None
    failure_provenance: dict[str, Any] | None = None
    post_render_validation_error: str | None = None


class CameraRepairExecutor:
    """Execute one selector policy and, when selected, one render.

    Cascade policy, stage transitions, EvidenceGate calls, packet composition,
    and Judge calls remain Controller responsibilities.
    """

    def __init__(
        self,
        *,
        renderer: EvidenceRenderer,
        control: VLMEvaluationControl,
    ) -> None:
        self.renderer = renderer
        self.control = control

    def select_once(
        self,
        *,
        selector: CameraSelector,
        request: CameraSelectionRequest,
        default_backend: str,
    ) -> SelectionExecution:
        dispatch_charge = selector_dispatch_charge(selector)
        try:
            selection = invoke_selector_once(
                selector,
                request,
                default_backend=default_backend,
            )
        except CameraSelectorInvocationError as exc:
            return SelectionExecution(
                selection=None,
                selector_calls=dispatch_charge,
                failure_kind=exc.failure_kind,
                error=str(exc),
            )
        return SelectionExecution(
            selection=selection,
            selector_calls=dispatch_charge,
        )

    def render_once(
        self,
        *,
        selector: CameraSelector,
        selection: CameraSelectionResult,
        selection_request: CameraSelectionRequest,
        judge_request: JudgeRequest,
        evidence_goal: dict[str, Any],
        previous_visual_evidence: list[Any],
        selector_calls_used: int,
        camera_actions_used: int,
    ) -> RenderExecution:
        reservation = trusted_composite_reservation(
            selector,
            selection,
        )
        if (
            selector_calls_used + int(reservation["selector_calls"])
            > self.control.max_selector_calls
        ):
            return RenderExecution(
                rendered=None,
                render_request=None,
                selector_calls=0,
                camera_actions=0,
                failure_kind="budget_exhausted",
                stop_reason="max_selector_calls_exhausted",
                reason=(
                    "selected camera backend exceeds the remaining "
                    "selector-call budget"
                ),
            )
        proposed_actions = selection_action_count(selection)
        if (
            camera_actions_used
            + proposed_actions
            + int(reservation["camera_actions"])
            > self.control.max_camera_actions
        ):
            return RenderExecution(
                rendered=None,
                render_request=None,
                selector_calls=0,
                camera_actions=0,
                failure_kind="budget_exhausted",
                stop_reason="max_camera_actions_exhausted",
                reason="camera action budget exhausted before rendering",
            )

        render_request = EvidenceRenderRequest(
            judge_request=judge_request.with_visual_evidence(
                previous_visual_evidence
            ),
            selection=selection,
            evidence_goal=deepcopy(evidence_goal),
            previous_visual_evidence=tuple(
                deepcopy(previous_visual_evidence)
            ),
            evidence_round=selection_request.evidence_round,
            budget=deepcopy(selection_request.budget),
            context=deepcopy(selection_request.context),
        )
        raw_render: Any = None
        try:
            raw_render = self.renderer.render(render_request)
            rendered = EvidenceRenderResult.from_value(
                raw_render,
                proposed_action_count=proposed_actions,
                authorized_action_count=(
                    proposed_actions + int(reservation["camera_actions"])
                    if reservation["trusted"]
                    else None
                ),
            )
            rendered = render_result_with_audit(
                rendered,
                request=render_request,
            )
        except Exception as exc:
            rejected = raw_render_evidence(raw_render)
            failure_selector_calls = 0
            failure_camera_actions = 0
            failure_provenance: dict[str, Any] = {}
            if isinstance(exc, EvidenceRenderFailure):
                rejected = list(deepcopy(exc.visual_evidence))
                failure_selector_calls = exc.internal_selector_calls
                failure_camera_actions = exc.camera_actions_executed
                try:
                    failure_provenance = jsonable(
                        deepcopy(exc.provenance)
                    )
                    _validate_failure_render_costs(failure_provenance)
                except (TypeError, ValueError) as provenance_error:
                    failure_provenance = {
                        "provenance_validation_error": (
                            f"{type(provenance_error).__name__}: "
                            f"{provenance_error}"
                        )
                    }
            if (
                reservation["trusted"]
                and isinstance(raw_render, EvidenceRenderResult)
            ):
                failure_selector_calls += (
                    rendered_internal_selector_calls(raw_render)
                )
                failure_camera_actions += (
                    raw_render.camera_actions_executed
                )
            failure_contract_error: str | None = None
            if reservation["trusted"]:
                if (
                    failure_selector_calls
                    > int(reservation["selector_calls"])
                ):
                    failure_contract_error = (
                        "failed renderer exceeded the trusted "
                        "composite-provider selector-call reservation"
                    )
                if (
                    failure_camera_actions
                    > proposed_actions + int(reservation["camera_actions"])
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
            usage_overrun = usage_overrun_stop_reason(
                control=self.control,
                selector_calls=(
                    selector_calls_used + failure_selector_calls
                ),
                camera_actions=(
                    camera_actions_used + failure_camera_actions
                ),
            )
            if failure_contract_error is not None:
                reason = failure_contract_error
                stop_reason = "renderer_followup_contract_invalid"
            elif usage_overrun is not None:
                reason = (
                    "failed camera evidence rendering consumed more than "
                    "the resolved control budget"
                )
                stop_reason = usage_overrun
            else:
                reason = "camera evidence rendering failed"
                stop_reason = "render_failed"
            return RenderExecution(
                rendered=None,
                render_request=render_request,
                selector_calls=failure_selector_calls,
                camera_actions=failure_camera_actions,
                rejected_visual_evidence=tuple(deepcopy(rejected)),
                failure_kind="render_failure",
                error=f"{type(exc).__name__}: {exc}",
                stop_reason=stop_reason,
                reason=reason,
                failure_provenance=failure_provenance,
            )

        internal_selector_calls = rendered_internal_selector_calls(
            rendered
        )
        validation_error: str | None = None
        if internal_selector_calls > int(reservation["selector_calls"]):
            validation_error = (
                "renderer exceeded the trusted composite-provider "
                "selector-call reservation"
            )
        view_count = rendered_view_count(
            rendered.visual_evidence,
            trusted_view_ids=selection.selected_view_ids,
        )
        if view_count > selection_request.budget["max_views_per_round"]:
            validation_error = (
                "renderer exceeded max_views_per_round with "
                f"{view_count} independently rendered views"
            )
        return RenderExecution(
            rendered=rendered,
            render_request=render_request,
            selector_calls=internal_selector_calls,
            camera_actions=rendered.camera_actions_executed,
            rendered_view_count=view_count,
            post_render_validation_error=validation_error,
        )


def _validate_failure_render_costs(
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
                f"failed renderer provenance {key} must be a "
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
                f"failed renderer provenance {key} must be finite and "
                "non-negative"
            )
