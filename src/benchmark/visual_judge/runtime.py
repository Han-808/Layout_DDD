from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from benchmark.visual_judge.contracts import (
    validate_generic_visual_response,
)
from benchmark.visual_judge.control_config import VLMEvaluationControl
from benchmark.visual_judge.control_loop import (
    EvidenceRenderFailure,
    EvidenceRenderRequest,
    EvidenceRenderResult,
    VLMEvaluationController,
    VLMEvaluationResult,
)
from benchmark.visual_judge.interfaces import (
    CameraSelectionRequest,
    CameraSelectionResult,
    ExistingJudgeAdapter,
    JudgeRequest,
    JudgeResult,
)
from benchmark.visual_judge.roles import DecisionContract


_METHOD_CONTRACTS = {
    "adjudicate_scene_quality": DecisionContract.CANONICAL_METRIC,
    "adjudicate_functional_semantic": DecisionContract.CANONICAL_METRIC,
    "adjudicate_p0b": DecisionContract.P0B_BINARY,
    "adjudicate_relation": DecisionContract.RELATION_BINARY,
    "adjudicate_spatial_fidelity": (
        DecisionContract.SPATIAL_FIDELITY_BINARY
    ),
}
_BINARY_CONTRACTS = {
    DecisionContract.P0B_BINARY.value,
    DecisionContract.RELATION_BINARY.value,
    DecisionContract.SPATIAL_FIDELITY_BINARY.value,
}
_EXISTING_PROVIDER_CANDIDATE_ID = "existing_provider_acquisition"


class EvidenceControlUnresolvedError(RuntimeError):
    """A binary compatibility method could not safely call its VLM Judge."""

    def __init__(self, result: VLMEvaluationResult) -> None:
        super().__init__(
            "visual evidence remained unresolved before a binary Judge "
            f"conclusion ({result.stop_reason})"
        )
        self.result = result


@dataclass
class _CapturedLegacyCall:
    call: Callable[[dict[str, Any]], Any]
    method_name: str
    responses: list[dict[str, Any]]

    def __getattr__(self, name: str) -> Any:
        if name != self.method_name:
            raise AttributeError(name)

        def invoke(request: dict[str, Any]) -> Any:
            raw = self.call(request)
            if isinstance(raw, dict):
                self.responses.append(deepcopy(raw))
            return raw

        return invoke


@dataclass
class _GenericCompatibilityJudgeAdapter:
    call: Callable[[dict[str, Any]], Any]
    responses: list[dict[str, Any]]
    backend: str = "existing"

    def judge(self, request: JudgeRequest) -> JudgeResult:
        payload = deepcopy(request.context)
        payload.update(
            {
                "task": request.task,
                "metric": request.metric,
                "category": request.task,
                "scene_summary": deepcopy(request.scene_context),
                "deterministic_evidence": deepcopy(
                    request.deterministic_evidence
                ),
                "render_evidence": [
                    _legacy_evidence_ref(item)
                    for item in request.visual_evidence
                ],
                "vlm_role": "judge",
                "decision_contract": (
                    DecisionContract.GENERIC_VISUAL_SCORE.value
                ),
                "judge_method": "evaluate",
            }
        )
        raw = self.call(payload)
        if not isinstance(raw, dict):
            raise ValueError(
                "generic visual evaluator response must be a JSON object"
            )
        validate_generic_visual_response(raw)
        self.responses.append(deepcopy(raw))
        return JudgeResult.from_value(
            {
                "status": "valid",
                # The legacy generic contract intentionally permits omitted
                # confidence.  The internal control result still needs a
                # numeric confidence, but the original response is returned
                # unchanged to the compatibility caller below.
                "confidence": (
                    0.0
                    if raw.get("confidence") is None
                    else raw.get("confidence")
                ),
                "reason": raw.get("summary")
                or "generic visual evaluation completed",
                "defects": [],
                "backend": self.backend,
                "provenance": {
                    "adapter": type(self).__name__,
                    "vlm_role": "judge",
                    "decision_contract": (
                        DecisionContract.GENERIC_VISUAL_SCORE.value
                    ),
                    "judge_method": "evaluate",
                    **{
                        key: deepcopy(raw[key])
                        for key in (
                            "model",
                            "endpoint",
                            "images_used",
                            "request_metadata",
                        )
                        if raw.get(key) is not None
                    },
                },
            }
        )


class _UnavailableEvidenceRenderer:
    def render(self, request: EvidenceRenderRequest) -> EvidenceRenderResult:
        del request
        raise RuntimeError(
            "no camera evidence provider is configured for another round"
        )


class _ExistingProviderCameraSelector:
    """Select one trusted acquisition token for a legacy composite provider."""

    def __init__(
        self,
        provider: Any,
        *,
        requested_backend: str,
        effective_backend: str,
        selector_binding: dict[str, Any],
    ) -> None:
        self.provider = provider
        self.requested_backend = str(requested_backend)
        self.backend = str(effective_backend)
        self.selector_binding = deepcopy(selector_binding)
        self.preserve_configured_adapter = True
        self.trusted_composite_provider_adapter = True

    def select(
        self,
        request: CameraSelectionRequest,
    ) -> CameraSelectionResult:
        candidate = next(
            (
                item
                for item in request.candidate_views
                if str(item.get("id") or "")
                == _EXISTING_PROVIDER_CANDIDATE_ID
            ),
            None,
        )
        if candidate is None:
            raise ValueError(
                "existing camera provider acquisition candidate is missing"
            )
        max_actions = _provider_max_actions(self.provider)
        max_selector_calls = _provider_max_selector_calls(
            self.provider
        )
        return CameraSelectionResult(
            selected_view_ids=(_EXISTING_PROVIDER_CANDIDATE_ID,),
            selected_views=(deepcopy(candidate),),
            reason=(
                "delegate one bounded evidence acquisition to the configured "
                "existing camera provider"
            ),
            backend=self.backend,
            evidence_round=request.evidence_round,
            provenance={
                "adapter": type(self).__name__,
                "requested_backend": self.requested_backend,
                "effective_backend": self.backend,
                "selector_binding": deepcopy(self.selector_binding),
                "provider": _qualified_name(self.provider),
                "provider_policy": _provider_policy(self.provider),
                "selection_kind": "legacy_composite_backend_acquisition",
                "max_internal_camera_actions": max_actions,
                "max_internal_selector_calls": max_selector_calls,
            },
        )


class _ExistingProviderEvidenceRenderer:
    """Execute the exact provider acquisition selected by its trusted token."""

    def __init__(
        self,
        provider: Any,
        *,
        usage_consumer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.provider = provider
        self.usage_consumer = usage_consumer

    def render(
        self,
        request: EvidenceRenderRequest,
    ) -> EvidenceRenderResult:
        if request.selection.selected_view_ids != (
            _EXISTING_PROVIDER_CANDIDATE_ID,
        ):
            raise ValueError(
                "existing provider renderer received an unrelated selection"
            )
        provider_request = _provider_request(request)
        try:
            raw = _call_provider(
                self.provider,
                provider_request,
                metric=request.judge_request.metric,
            )
            evidence = _provider_visual_evidence(raw)
            if not evidence:
                raise RuntimeError(
                    "existing camera evidence provider returned no visual "
                    "evidence"
                )
        except Exception as exc:
            try:
                usage = _provider_observed_usage(
                    self.provider,
                    (),
                    metric=request.judge_request.metric,
                )
            except Exception as usage_exc:
                raise EvidenceRenderFailure(
                    (
                        f"{type(exc).__name__}: {exc}; provider usage "
                        f"telemetry unavailable: {type(usage_exc).__name__}: "
                        f"{usage_exc}"
                    ),
                    provenance={
                        "adapter": type(self).__name__,
                        "provider": _qualified_name(self.provider),
                        "provider_policy": _provider_policy(self.provider),
                        "usage_observation_error": (
                            f"{type(usage_exc).__name__}: {usage_exc}"
                        ),
                    },
                ) from exc
            if self.usage_consumer is not None:
                self.usage_consumer(usage)
            raise EvidenceRenderFailure(
                f"{type(exc).__name__}: {exc}",
                internal_selector_calls=int(
                    usage.get("selector_calls", 0)
                ),
                camera_actions_executed=int(
                    usage.get("camera_actions", 0)
                ),
                visual_evidence=tuple(
                    usage.get("evidence_refs") or ()
                ),
                provenance={
                    "adapter": type(self).__name__,
                    "provider": _qualified_name(self.provider),
                    "provider_policy": _provider_policy(self.provider),
                    "provider_usage": deepcopy(usage),
                    "usage_source": (
                        "existing_provider_last_call_usage"
                    ),
                },
            ) from exc
        usage = _provider_observed_usage(
            self.provider,
            raw,
            metric=request.judge_request.metric,
        )
        if self.usage_consumer is not None:
            self.usage_consumer(usage)
        actual_actions = int(usage.get("camera_actions", 0))
        actual_selector_calls = int(usage.get("selector_calls", 0))
        return EvidenceRenderResult(
            visual_evidence=tuple(deepcopy(evidence)),
            merge_policy="append",
            camera_actions_executed=actual_actions,
            backend="existing",
            provenance={
                "adapter": type(self).__name__,
                "provider": _qualified_name(self.provider),
                "provider_policy": _provider_policy(self.provider),
                "selected_acquisition": _EXISTING_PROVIDER_CANDIDATE_ID,
                "scene_access": "read_only",
                "internal_selector_calls": actual_selector_calls,
                "actual_camera_actions": actual_actions,
                "provider_usage": deepcopy(usage),
                "usage_source": "existing_provider_last_call_usage",
            },
        )


class ControlledVLMJudge:
    """Backward-compatible public methods backed by the strict controller."""

    def __init__(
        self,
        judge: Any,
        *,
        control: VLMEvaluationControl,
        camera_provider: Any | None = None,
        camera_selector: Any | None = None,
        strict: bool | None = None,
    ) -> None:
        if judge is None:
            raise TypeError("ControlledVLMJudge requires an existing judge")
        self._judge = judge
        self.control = control
        self.camera_provider = camera_provider
        self.camera_selector = camera_selector
        # A Judge is controlled by default regardless of implementation
        # details.  Compatibility callers that are provably non-VLM may opt
        # out explicitly with strict=False; absence of `.model` must never
        # become an implicit evidence-gate bypass.
        self.strict = True if strict is None else bool(strict)
        self.audit_records: list[dict[str, Any]] = []
        self._consumed_provider_usage_ids: set[str] = set()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._judge, name)

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.evaluate(request)

    def evaluate(self, request: dict[str, Any]) -> dict[str, Any]:
        """Run generic-score compatibility through the same evidence boundary."""

        if not isinstance(request, dict):
            raise TypeError("evaluate request must be a JSON object")
        call = getattr(self._judge, "_evaluate_raw", None)
        if not callable(call):
            call = getattr(self._judge, "evaluate", None)
        if not callable(call) and callable(self._judge):
            call = self._judge
        if not callable(call):
            raise TypeError(
                "generic visual evaluator must be callable or expose evaluate"
            )
        if not self.strict:
            return call(request)

        responses: list[dict[str, Any]] = []
        judge_adapter = _GenericCompatibilityJudgeAdapter(
            call=call,
            responses=responses,
        )
        provider_available = (
            self.camera_provider is not None
            and not _compatibility_screen_request(request)
        )
        initial_camera_usage = (
            self._consume_provider_usage(
                request,
                packet=_visual_evidence_packet(request),
            )
            if provider_available
            else None
        )
        if provider_available:
            selector_binding = _bind_provider_selector_backend(
                self.camera_provider,
                requested_backend=self.control.camera_selector_backend,
                injected_selector=self.camera_selector,
                metric=str(
                    request.get("metric")
                    or request.get("category")
                    or "visual_quality"
                ),
            )
            selector: Any = _ExistingProviderCameraSelector(
                self.camera_provider,
                requested_backend=self.control.camera_selector_backend,
                effective_backend=str(
                    selector_binding["effective_backend"]
                ),
                selector_binding=selector_binding,
            )
            renderer: Any = _ExistingProviderEvidenceRenderer(
                self.camera_provider,
                usage_consumer=self._mark_provider_usage_consumed,
            )
            candidates = (
                {
                    "id": _EXISTING_PROVIDER_CANDIDATE_ID,
                    "kind": "legacy_composite_backend_acquisition",
                    "backend": "existing",
                },
            )
        else:
            selector = self.camera_selector
            renderer = _UnavailableEvidenceRenderer()
            candidates = tuple(_request_candidate_views(request))
        controller = VLMEvaluationController(
            judge=judge_adapter,
            renderer=renderer,
            camera_selector=selector,
            control=self.control,
        )
        core_request = _judge_request(request)
        result = controller.run(
            core_request,
            evidence_goal=_evidence_goal(
                request,
                camera_repairable=provider_available,
            ),
            candidate_views=candidates,
            allowed_actions=tuple(_request_allowed_actions(request)),
            selector_context=_selector_context(request),
            gate_manifest_path=_request_manifest_path(request),
            initial_camera_usage=initial_camera_usage,
        )
        self.audit_records.append(
            {
                "judge_method": "evaluate",
                "metric": core_request.metric,
                "status": result.status,
                "stop_reason": result.stop_reason,
                "audit": deepcopy(result.audit),
            }
        )
        if responses:
            return deepcopy(responses[-1])
        return {
            "applicable": False,
            "score": None,
            "confidence": 0.0,
            "summary": result.reason,
            "issues": [],
            "evidence": [],
            "vlm_role": "judge",
            "decision_contract": (
                DecisionContract.GENERIC_VISUAL_SCORE.value
            ),
            "judge_method": "evaluate",
            "images_used": list(result.audit.get("images_used") or []),
            "evidence_control_stop_reason": result.stop_reason,
        }

    def adjudicate_scene_quality(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self._adjudicate("adjudicate_scene_quality", request)

    def adjudicate_functional_semantic(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self._adjudicate(
            "adjudicate_functional_semantic",
            request,
        )

    def adjudicate_functional_semantics(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self.adjudicate_functional_semantic(request)

    def adjudicate_specification_fidelity(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self.adjudicate_functional_semantic(request)

    def adjudicate_p0b(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._adjudicate("adjudicate_p0b", request)

    def adjudicate_relation(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self._adjudicate("adjudicate_relation", request)

    def adjudicate_spatial_fidelity(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        return self._adjudicate(
            "adjudicate_spatial_fidelity",
            request,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "wrapper": _qualified_name(self),
            "underlying_judge": _qualified_name(self._judge),
            "strict_controller_enabled": self.strict,
            "camera_provider": (
                _qualified_name(self.camera_provider)
                if self.camera_provider is not None
                else None
            ),
            "camera_selector": (
                _qualified_name(self.camera_selector)
                if self.camera_selector is not None
                else None
            ),
            "controlled_call_count": len(self.audit_records),
            "controlled_calls": deepcopy(self.audit_records),
        }

    def _adjudicate(
        self,
        method_name: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise TypeError(f"{method_name} request must be a JSON object")
        legacy_call = _resolve_legacy_method(
            self._judge,
            method_name,
        )
        if not self.strict:
            return legacy_call(request)

        decision_contract = _METHOD_CONTRACTS[method_name]
        responses: list[dict[str, Any]] = []
        captured = _CapturedLegacyCall(
            call=legacy_call,
            method_name=method_name,
            responses=responses,
        )
        judge_adapter = ExistingJudgeAdapter(
            captured,
            method_name=method_name,
            decision_contract=decision_contract,
        )
        provider_available = (
            self.camera_provider is not None
            and not _compatibility_screen_request(request)
        )
        initial_camera_usage = (
            self._consume_provider_usage(
                request,
                packet=_visual_evidence_packet(request),
            )
            if provider_available
            else None
        )
        if provider_available:
            selector_binding = _bind_provider_selector_backend(
                self.camera_provider,
                requested_backend=self.control.camera_selector_backend,
                injected_selector=self.camera_selector,
                metric=str(
                    request.get("metric")
                    or request.get("family")
                    or request.get("category")
                    or ""
                ),
            )
            selector: Any = _ExistingProviderCameraSelector(
                self.camera_provider,
                requested_backend=self.control.camera_selector_backend,
                effective_backend=str(
                    selector_binding["effective_backend"]
                ),
                selector_binding=selector_binding,
            )
            renderer: Any = _ExistingProviderEvidenceRenderer(
                self.camera_provider,
                usage_consumer=self._mark_provider_usage_consumed,
            )
            candidates = (
                {
                    "id": _EXISTING_PROVIDER_CANDIDATE_ID,
                    "kind": "legacy_composite_backend_acquisition",
                    "backend": "existing",
                },
            )
        else:
            selector = self.camera_selector
            renderer = _UnavailableEvidenceRenderer()
            candidates = tuple(_request_candidate_views(request))

        controller = VLMEvaluationController(
            judge=judge_adapter,
            renderer=renderer,
            camera_selector=selector,
            control=self.control,
        )
        core_request = _judge_request(request)
        result = controller.run(
            core_request,
            evidence_goal=_evidence_goal(
                request,
                camera_repairable=provider_available,
            ),
            candidate_views=candidates,
            allowed_actions=tuple(_request_allowed_actions(request)),
            selector_context=_selector_context(request),
            gate_manifest_path=_request_manifest_path(request),
            initial_camera_usage=initial_camera_usage,
        )
        self.audit_records.append(
            {
                "judge_method": method_name,
                "metric": core_request.metric,
                "status": result.status,
                "stop_reason": result.stop_reason,
                "audit": deepcopy(result.audit),
            }
        )
        if result.status in {"valid", "invalid"} and responses:
            return deepcopy(responses[-1])
        if responses and decision_contract.value not in _BINARY_CONTRACTS:
            return deepcopy(responses[-1])
        if decision_contract.value in _BINARY_CONTRACTS:
            raise EvidenceControlUnresolvedError(result)
        return _canonical_unresolved_response(
            request,
            method_name=method_name,
            result=result,
        )

    def _consume_provider_usage(
        self,
        request: dict[str, Any],
        *,
        packet: list[Any],
    ) -> dict[str, Any] | None:
        if self.camera_provider is None:
            return None
        usage = _provider_observed_usage(
            self.camera_provider,
            packet,
            metric=str(
                request.get("metric")
                or request.get("family")
                or request.get("category")
                or ""
            ),
            allow_absent=True,
        )
        if not usage:
            return None
        usage_id = str(usage.get("call_id") or "")
        if usage_id and usage_id in self._consumed_provider_usage_ids:
            return None
        self._mark_provider_usage_consumed(usage)
        return usage

    def _mark_provider_usage_consumed(
        self,
        usage: dict[str, Any],
    ) -> None:
        usage_id = str(usage.get("call_id") or "")
        if usage_id:
            self._consumed_provider_usage_ids.add(usage_id)


def build_controlled_vlm_judge(
    judge: Any | None,
    *,
    control: VLMEvaluationControl,
    camera_provider: Any | None = None,
    camera_selector: Any | None = None,
    strict: bool | None = None,
) -> Any | None:
    if judge is None or isinstance(judge, ControlledVLMJudge):
        return judge
    return ControlledVLMJudge(
        judge,
        control=control,
        camera_provider=camera_provider,
        camera_selector=camera_selector,
        strict=strict,
    )


def _resolve_legacy_method(
    judge: Any,
    method_name: str,
) -> Callable[[dict[str, Any]], Any]:
    aliases = {
        "adjudicate_scene_quality": (
            "_adjudicate_scene_quality_raw",
            "adjudicate_scene_quality",
            "evaluate",
        ),
        "adjudicate_functional_semantic": (
            "_adjudicate_functional_semantic_raw",
            "adjudicate_functional_semantic",
            "adjudicate_functional_semantics",
            "adjudicate_specification_fidelity",
        ),
        "adjudicate_p0b": (
            "_adjudicate_p0b_raw",
            "adjudicate_p0b",
        ),
        "adjudicate_relation": (
            "_adjudicate_relation_raw",
            "adjudicate_relation",
        ),
        "adjudicate_spatial_fidelity": (
            "_adjudicate_spatial_fidelity_raw",
            "adjudicate_spatial_fidelity",
        ),
    }[method_name]
    for name in aliases:
        call = getattr(judge, name, None)
        if callable(call):
            return call
    if callable(judge):
        return judge
    raise TypeError(
        f"existing judge must be callable or expose {method_name}(request)"
    )


def _judge_request(request: dict[str, Any]) -> JudgeRequest:
    metric = str(
        request.get("metric")
        or request.get("family")
        or request.get("category")
        or "visual_metric"
    )
    task = str(request.get("category") or metric)
    claim_or_event = (
        request.get("claim")
        if isinstance(request.get("claim"), dict)
        else request.get("event")
        if isinstance(request.get("event"), dict)
        else request.get("relation")
        if isinstance(request.get("relation"), dict)
        else {}
    )
    scene_context = request.get("scene_summary")
    if not isinstance(scene_context, dict):
        scene_context = request.get("scene")
    if not isinstance(scene_context, dict):
        architecture = (
            request.get("architecture")
            if isinstance(request.get("architecture"), dict)
            else {}
        )
        scene_context = {
            "boundary": deepcopy(architecture.get("boundary")),
            "scene_height": architecture.get("height"),
            "objects": deepcopy(request.get("objects") or []),
        }
    deterministic = request.get("deterministic_evidence")
    if not isinstance(deterministic, dict):
        deterministic = request.get("detector_evidence")
    if not isinstance(deterministic, dict):
        deterministic = {}
    return JudgeRequest(
        task=task,
        metric=metric,
        claim_or_event=deepcopy(claim_or_event),
        scene_context=deepcopy(scene_context),
        deterministic_evidence=deepcopy(deterministic),
        visual_evidence=tuple(_visual_evidence_packet(request)),
        rubric=deepcopy(
            request.get("metric_rubric") or request.get("rubric")
        ),
        context=deepcopy(request),
    )


def _visual_evidence_packet(request: dict[str, Any]) -> list[Any]:
    paths = list(request.get("render_evidence") or [])
    metadata = request.get("local_render_evidence_metadata")
    if not isinstance(metadata, list):
        return deepcopy(paths)
    by_path = {
        str(item.get("path") or item.get("image_path")): item
        for item in metadata
        if isinstance(item, dict)
        and (item.get("path") or item.get("image_path"))
    }
    if not paths:
        return deepcopy(paths)
    return [
        deepcopy(by_path.get(str(path), path))
        for path in paths
    ]


def _evidence_goal(
    request: dict[str, Any],
    *,
    camera_repairable: bool,
) -> dict[str, Any]:
    policy = request.get("visual_evidence_policy")
    result = deepcopy(policy) if isinstance(policy, dict) else {}
    result.setdefault(
        "view_goal",
        f"show_metric_scoped_{request.get('metric') or 'visual'}_evidence",
    )
    result["missing_evidence_camera_repairable"] = bool(
        camera_repairable
    )
    return result


def _provider_request(request: EvidenceRenderRequest) -> dict[str, Any]:
    context = request.judge_request.context
    explicit = context.get("camera_evidence_request")
    if isinstance(explicit, dict):
        result = deepcopy(explicit)
    else:
        result = {
            "category": "visual_evidence_request",
            "metric": request.judge_request.metric,
            "event": deepcopy(request.judge_request.claim_or_event),
            "object_ids": list(
                request.evidence_goal.get("target_ids")
                or request.judge_request.claim_or_event.get("object_ids")
                or context.get("target_ids")
                or context.get("target_object_ids")
                or []
            ),
            "scene": deepcopy(request.judge_request.scene_context),
            "detector_evidence": deepcopy(
                request.judge_request.deterministic_evidence
            ),
            "natural_language_prompt": (
                context.get("natural_language_prompt")
                or context.get("prompt")
            ),
            "evidence_scope": "metric_scoped",
        }
    result["_camera_selection_phase"] = "active_fallback"
    result["_camera_evidence_deficiency"] = deepcopy(
        request.evidence_goal
    )
    result["_vlm_evidence_round"] = request.evidence_round
    result["access"] = "read_only_evidence_request"
    return result


def _call_provider(
    provider: Any,
    request: dict[str, Any],
    *,
    metric: str,
) -> Any:
    if metric in {
        "scale_consistency",
        "object_pairing_consistency",
        "style_consistency",
    }:
        call = getattr(provider, "provide_scene_quality_evidence", None)
        if callable(call):
            return call(request)
    if callable(provider):
        return provider(request)
    call = getattr(provider, "provide_scene_quality_evidence", None)
    if callable(call):
        return call(request)
    raise TypeError("camera evidence provider is not callable")


def _provider_visual_evidence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(deepcopy(value))
    if not isinstance(value, dict):
        raise ValueError(
            "camera evidence provider response must be a JSON object or list"
        )
    status = str(value.get("status") or "available").lower()
    if status in {"failed", "error", "unavailable", "insufficient"}:
        raise RuntimeError(
            "camera evidence provider reported "
            f"{status}: {value.get('reason') or value.get('error') or ''}"
        )
    for key in (
        "visual_evidence",
        "render_evidence_items",
        "render_evidence",
        "paths",
    ):
        items = value.get(key)
        if isinstance(items, (list, tuple)):
            return list(deepcopy(items))
    raise ValueError(
        "camera evidence provider response does not contain visual evidence"
    )


def _canonical_unresolved_response(
    request: dict[str, Any],
    *,
    method_name: str,
    result: VLMEvaluationResult,
) -> dict[str, Any]:
    missing = (
        list(result.evidence_request.missing_observations)
        if result.evidence_request is not None
        else [result.stop_reason]
    )
    value: dict[str, Any] = {
        "evidence_status": "insufficient",
        "verdict": "ambiguous",
        "confidence": 0.0,
        "reason": result.reason,
        "missing_evidence": missing,
        "defects": [],
        "vlm_role": "judge",
        "decision_contract": DecisionContract.CANONICAL_METRIC.value,
        "judge_method": method_name,
        "images_used": list(result.audit.get("images_used") or []),
        "evidence_control_stop_reason": result.stop_reason,
    }
    if method_name == "adjudicate_functional_semantic":
        value["canonical_verdict"] = "ambiguous"
        value["verdict"] = "insufficient_evidence"
        value["router_state"] = "insufficient_evidence"
        value["response_adapter"] = (
            "functional_semantic_insufficient_evidence_compat_v1"
        )
    return value


def _request_candidate_views(
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    for key in ("candidate_views", "candidates"):
        value = request.get(key)
        if isinstance(value, list) and all(
            isinstance(item, dict) for item in value
        ):
            return deepcopy(value)
    return []


def _request_allowed_actions(request: dict[str, Any]) -> list[str]:
    value = request.get("allowed_actions")
    if not isinstance(value, list):
        return []
    return [
        str(item)
        for item in value
        if isinstance(item, str) and item
    ]


def _selector_context(request: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "allow_adjustment",
        "color_legend",
        "corrective_proposals",
        "evidence_deficiency",
        "preview_degradation",
        "preview_role",
        "preview_visibility_warning",
        "selection_phase",
    ):
        if key in request:
            result[key] = deepcopy(request[key])
    return result


def _request_manifest_path(request: dict[str, Any]) -> str | None:
    for key in ("evidence_manifest_path", "render_manifest_path"):
        value = request.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _compatibility_screen_request(request: dict[str, Any]) -> bool:
    return (
        str(request.get("decision_mode") or "").strip().lower()
        == "screen"
        or str(request.get("evidence_phase") or "").strip().lower()
        in {"global_screen", "required_area_global_screen"}
    )


def _provider_policy(provider: Any) -> dict[str, Any] | None:
    value = getattr(provider, "policy_config", None)
    return deepcopy(value) if isinstance(value, dict) else None


def _provider_max_actions(provider: Any) -> int:
    policy = _provider_policy(provider) or {}
    for key in ("max_camera_actions", "max_steps"):
        value = policy.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    value = getattr(provider, "max_steps", 0)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _provider_max_selector_calls(provider: Any) -> int:
    policy = _provider_policy(provider) or {}
    value = policy.get("max_selector_calls")
    if (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0
    ):
        return value
    selection_source = str(
        policy.get("selection_source") or ""
    ).strip().lower()
    resolved_modes = policy.get("resolved_metric_modes")
    query_cov = bool(
        selection_source in {
            "runtime_selector",
            "frozen_vlm_selected_view_ids",
        }
        or (
            isinstance(resolved_modes, dict)
            and "query_cov"
            in {
                str(item).strip().lower()
                for item in resolved_modes.values()
            }
        )
        or str(policy.get("mode") or "").strip().lower()
        == "query_cov"
    )
    if not query_cov:
        return 0
    return _provider_max_actions(provider) + 1


def _bind_provider_selector_backend(
    provider: Any,
    *,
    requested_backend: str,
    injected_selector: Any | None,
    metric: str,
) -> dict[str, Any]:
    requested = str(requested_backend).strip().lower()
    capability = _provider_selector_capability(provider, metric=metric)
    result: dict[str, Any] = {
        "requested_backend": requested,
        "provider_capability": capability,
        "provider": _qualified_name(provider),
        "metric": str(metric),
        "injected_selector": (
            _qualified_name(injected_selector)
            if injected_selector is not None
            else None
        ),
    }
    if requested == "existing":
        result.update(
            {
                "effective_backend": "existing",
                "binding_verified": True,
                "binding_reason": "legacy_existing_provider_adapter",
            }
        )
        return result

    compatible = capability == requested
    if requested == "vlm" and capability == "hybrid":
        # A conditional deterministic→VLM provider is a hybrid backend, not a
        # pure VLM selector.  Treating it as VLM would silently retain the old
        # deterministic trigger and misstate the effective configuration.
        compatible = False
    if not compatible:
        raise ValueError(
            "camera_selector.backend="
            f"{requested!r} is incompatible with the configured legacy "
            f"camera provider capability {capability!r}; use backend='existing' "
            "to preserve that composite provider or inject a matching provider"
        )

    if requested in {"vlm", "hybrid"}:
        if injected_selector is None:
            raise ValueError(
                f"camera_selector.backend={requested!r} requires an injected "
                "selector bound into the configured camera provider"
            )
        if not _provider_contains_selector(provider, injected_selector):
            raise ValueError(
                f"camera_selector.backend={requested!r} does not match the "
                "selector bound into the configured legacy camera provider"
            )
        binding_reason = "injected_selector_identity_verified"
    else:
        binding_reason = "deterministic_provider_policy_verified"
    result.update(
        {
            "effective_backend": requested,
            "binding_verified": True,
            "binding_reason": binding_reason,
        }
    )
    return result


def _provider_selector_capability(
    provider: Any,
    *,
    metric: str,
) -> str:
    explicit = str(
        getattr(provider, "camera_selector_backend", "")
        or getattr(provider, "selector_backend", "")
    ).strip().lower()
    if explicit in {"existing", "deterministic", "vlm", "hybrid"}:
        return explicit
    if (
        getattr(provider, "deterministic_provider", None) is not None
        and getattr(provider, "active_provider", None) is not None
    ):
        return "hybrid"
    policy = _provider_policy(provider) or {}
    if str(policy.get("schema_version") or "").startswith(
        "conditional_vlm_active_camera_fallback"
    ):
        return "hybrid"
    resolved_modes = policy.get("resolved_metric_modes")
    metric_name = str(metric).strip().lower()
    if metric_name == "object_architecture_penetration":
        metric_name = "oob"
    resolved_mode = (
        resolved_modes.get(metric_name)
        if isinstance(resolved_modes, dict)
        else None
    )
    mode = str(resolved_mode or policy.get("mode") or "").strip().lower()
    if mode == "query_cov":
        return "vlm"
    if mode:
        return "deterministic"
    if getattr(provider, "selector", None) is not None:
        return "vlm"
    return "existing"


def _provider_contains_selector(
    provider: Any,
    selector: Any,
) -> bool:
    provider_ids = _selector_object_ids(provider)
    selector_ids = _selector_object_ids(selector)
    return bool(provider_ids & selector_ids)


def _selector_object_ids(value: Any) -> set[int]:
    pending = [value]
    seen: set[int] = set()
    result: set[int] = set()
    while pending:
        current = pending.pop()
        if current is None:
            continue
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)
        result.add(identity)
        if isinstance(current, (list, tuple)):
            pending.extend(current)
            continue
        for name in (
            "_selector",
            "selector",
            "primary",
            "fallback",
            "active_provider",
            "deterministic_provider",
        ):
            child = getattr(current, name, None)
            if child is not None:
                pending.append(child)
    return result


def _provider_observed_usage(
    provider: Any,
    packet: Any,
    *,
    metric: str,
    allow_absent: bool = False,
) -> dict[str, Any]:
    value = getattr(provider, "last_call_usage", None)
    if value is None:
        if allow_absent:
            return {}
        if (
            _provider_max_actions(provider) > 0
            or _provider_max_selector_calls(provider) > 0
        ):
            raise RuntimeError(
                "configured composite camera provider declares camera/selector "
                "capacity but does not expose actual last_call_usage telemetry"
            )
        return {
            "call_id": None,
            "metric": str(metric),
            "cache_hit": False,
            "evidence_refs": _packet_evidence_refs(packet),
            "manifest_path": None,
            "selector_calls": 0,
            "camera_actions": 0,
            "observability": "provider_declares_no_internal_selector_or_actions",
        }
    if not isinstance(value, dict):
        raise RuntimeError(
            "camera provider last_call_usage must be a JSON object"
        )
    usage = deepcopy(value)
    for key in ("selector_calls", "camera_actions"):
        raw = usage.get(key)
        if (
            not isinstance(raw, int)
            or isinstance(raw, bool)
            or raw < 0
        ):
            raise RuntimeError(
                f"camera provider last_call_usage.{key} must be a "
                "non-negative integer"
            )
    usage_metric = str(usage.get("metric") or "").strip().lower()
    expected_metric = str(metric or "").strip().lower()
    if usage_metric and expected_metric and usage_metric != expected_metric:
        if allow_absent:
            return {}
        raise RuntimeError(
            "camera provider last_call_usage metric does not match the "
            "current Judge request"
        )
    refs = usage.get("evidence_refs")
    if not isinstance(refs, list) or not all(
        isinstance(item, str) and item for item in refs
    ):
        raise RuntimeError(
            "camera provider last_call_usage.evidence_refs must be a list "
            "of non-empty strings"
        )
    packet_refs = set(_packet_evidence_refs(packet))
    if refs and packet_refs and not (set(refs) & packet_refs):
        if allow_absent:
            return {}
        raise RuntimeError(
            "camera provider last_call_usage does not describe the current "
            "visual evidence packet"
        )
    usage["metric"] = usage_metric or expected_metric
    usage["observability"] = "provider_actual_usage"
    return usage


def _packet_evidence_refs(packet: Any) -> list[str]:
    if isinstance(packet, dict):
        for key in (
            "visual_evidence",
            "render_evidence_items",
            "render_evidence",
            "paths",
        ):
            value = packet.get(key)
            if isinstance(value, (list, tuple)):
                packet = value
                break
        else:
            packet = []
    if not isinstance(packet, (list, tuple)):
        return []
    refs: list[str] = []
    for index, item in enumerate(packet):
        if isinstance(item, dict):
            value = (
                item.get("view_id")
                or item.get("id")
                or item.get("path")
                or item.get("image_path")
            )
        else:
            value = item
        refs.append(
            str(value)
            if value is not None
            else f"evidence_{index:02d}"
        )
    return refs


def _legacy_evidence_ref(value: Any) -> Any:
    if isinstance(value, dict):
        path = value.get("path") or value.get("image_path")
        if path is not None and str(path).strip():
            return str(path)
    return deepcopy(value)


def _qualified_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"
