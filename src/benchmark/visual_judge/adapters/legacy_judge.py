from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from benchmark.visual_judge.contracts import (
    validate_binary_judge_response,
    validate_canonical_metric_response,
    validate_generic_visual_response,
)
from benchmark.visual_judge.interfaces.judge import (
    JUDGE_STATUSES,
    EvidenceRequest,
    JudgeRequest,
    JudgeResult,
)
from benchmark.visual_judge.roles import DecisionContract
from benchmark.visual_judge.control_config import VLMEvaluationControl
from benchmark.visual_judge.orchestration.controller import (
    VLMEvaluationController,
    VLMEvaluationResult,
)
from benchmark.visual_judge.adapters.legacy_camera import (
    EXISTING_PROVIDER_CANDIDATE_ID as _EXISTING_PROVIDER_CANDIDATE_ID,
    ExistingProviderCameraSelector as _ExistingProviderCameraSelector,
    bind_provider_selector_backend as _bind_provider_selector_backend,
    provider_observed_usage as _provider_observed_usage,
)
from benchmark.visual_judge.adapters.legacy_renderer import (
    ExistingProviderEvidenceRenderer as _ExistingProviderEvidenceRenderer,
    UnavailableEvidenceRenderer as _UnavailableEvidenceRenderer,
    coerce_evidence_renderer as _coerce_evidence_renderer,
)


class ExistingJudgeAdapter:
    """Normalize one existing dedicated judge method to the Judge interface."""

    def __init__(
        self,
        judge: Any,
        *,
        method_name: str,
        decision_contract: str,
        backend: str = "existing",
    ) -> None:
        method = getattr(judge, method_name, None)
        if not callable(method):
            raise TypeError(f"existing Judge adapter requires {method_name}(request)")
        self._method = method
        self.method_name = str(method_name)
        self.decision_contract = str(
            getattr(decision_contract, "value", decision_contract)
        ).strip()
        if not self.decision_contract:
            raise ValueError(
                "existing Judge adapter requires an explicit decision_contract"
            )
        self.backend = str(backend)
        self.last_raw_response: Any | None = None

    def judge(self, request: JudgeRequest) -> JudgeResult:
        legacy_request = _legacy_judge_request(request)
        legacy_request["vlm_role"] = "judge"
        legacy_request["decision_contract"] = self.decision_contract
        legacy_request["judge_method"] = self.method_name
        raw = self._method(legacy_request)
        if isinstance(raw, JudgeResult):
            raw = raw.to_dict()
        self.last_raw_response = deepcopy(raw)
        if not isinstance(raw, dict):
            raise ValueError("existing Judge response must be a JSON object")
        if raw.get("status") in JUDGE_STATUSES:
            value = _validate_native_judge_contract(
                raw,
                request=request,
                decision_contract=self.decision_contract,
                method_name=self.method_name,
            )
            reported_backend = value.get("backend")
            value["backend"] = self.backend
            provenance = value.setdefault("provenance", {})
            if not isinstance(provenance, dict):
                raise ValueError(
                    "existing Judge provenance must be a JSON object"
                )
            if (
                reported_backend is not None
                and str(reported_backend) != self.backend
            ):
                provenance.setdefault(
                    "reported_backend",
                    str(reported_backend),
                )
            _set_authoritative_provenance(
                provenance,
                "adapter",
                type(self).__name__,
                boundary="Judge",
            )
            _set_authoritative_provenance(
                provenance,
                "method",
                self.method_name,
                boundary="Judge",
            )
            _set_authoritative_provenance(
                provenance,
                "judge_method",
                self.method_name,
                boundary="Judge",
            )
            _set_authoritative_provenance(
                provenance,
                "vlm_role",
                "judge",
                boundary="Judge",
            )
            _set_authoritative_provenance(
                provenance,
                "decision_contract",
                self.decision_contract,
                boundary="Judge",
            )
            for key in (
                "model",
                "endpoint",
                "images_used",
                "request_metadata",
            ):
                if raw.get(key) is not None:
                    provenance.setdefault(key, deepcopy(raw[key]))
            return JudgeResult.from_value(value)

        normalized = _validate_legacy_judge_contract(
            raw,
            request=request,
            decision_contract=self.decision_contract,
            method_name=self.method_name,
        )
        verdict = str(normalized.get("verdict") or "").strip()
        evidence_status = str(
            normalized.get("evidence_status") or ""
        ).strip()
        if evidence_status == "insufficient" or verdict in {
            "ambiguous",
            "need_more_evidence",
        }:
            missing = [
                str(item)
                for item in normalized.get("missing_evidence") or []
                if str(item).strip()
            ]
            evidence_request = raw.get("evidence_request")
            if not isinstance(evidence_request, dict):
                evidence_request = {
                    "target_ids": _request_target_ids(request) or ["scene"],
                    "missing_observations": (
                        missing
                        or ["target_visible"]
                    ),
                    "view_goal": "metric_scoped_visual_confirmation",
                    "metadata": {
                        "source_method": self.method_name,
                        "source_evidence_status": evidence_status or None,
                    },
                }
            return JudgeResult.from_value(
                {
                    "status": "need_more_evidence",
                    "confidence": raw.get("confidence", 0.0),
                    "reason": raw.get("reason") or "additional visual evidence required",
                    "defects": [],
                    "evidence_request": evidence_request,
                    "backend": self.backend,
                    "provenance": {
                        "adapter": type(self).__name__,
                        "method": self.method_name,
                        "vlm_role": "judge",
                        "decision_contract": self.decision_contract,
                        **_legacy_audit_provenance(raw),
                    },
                }
            )
        if verdict not in {"valid", "invalid"}:
            raise ValueError(
                "existing Judge verdict must be valid, invalid, or request more evidence"
            )
        return JudgeResult.from_value(
            {
                "status": verdict,
                "confidence": raw.get("confidence", 0.0),
                "reason": raw.get("reason") or f"{self.method_name} returned {verdict}",
                "defects": raw.get("defects") or [],
                "backend": self.backend,
                "provenance": {
                    "adapter": type(self).__name__,
                    "method": self.method_name,
                    "vlm_role": "judge",
                    "decision_contract": self.decision_contract,
                    **_legacy_audit_provenance(raw),
                },
            }
        )


def _legacy_judge_request(request: JudgeRequest) -> dict[str, Any]:
    result = deepcopy(request.context)
    result.update(
        {
            "task": request.task,
            "metric": request.metric,
            "claim_or_event": deepcopy(request.claim_or_event),
            "scene_context": deepcopy(request.scene_context),
            "deterministic_evidence": deepcopy(request.deterministic_evidence),
            "visual_evidence": list(deepcopy(request.visual_evidence)),
            "rubric": deepcopy(request.rubric),
        }
    )
    result["render_evidence"] = [
        _legacy_visual_evidence_ref(item)
        for item in request.visual_evidence
    ]
    result.setdefault("scene_summary", deepcopy(request.scene_context))
    result.setdefault("metric_rubric", deepcopy(request.rubric))
    if "event" not in result and request.claim_or_event:
        result["event"] = deepcopy(request.claim_or_event)
    return result


def _legacy_visual_evidence_ref(value: Any) -> Any:
    if isinstance(value, dict):
        path = value.get("path") or value.get("image_path")
        if path is not None and str(path).strip():
            return str(path)
    return deepcopy(value)


def _validate_legacy_judge_contract(
    value: dict[str, Any],
    *,
    request: JudgeRequest,
    decision_contract: str,
    method_name: str,
) -> dict[str, Any]:
    normalized = deepcopy(value)
    if decision_contract == DecisionContract.CANONICAL_METRIC.value:
        compatibility_verdict = str(
            normalized.get("canonical_verdict") or ""
        ).strip()
        if compatibility_verdict:
            normalized["verdict"] = compatibility_verdict
        validate_canonical_metric_response(
            normalized,
            allowed_scopes=_canonical_allowed_scopes(request),
        )
        return normalized
    binary_labels = {
        DecisionContract.P0B_BINARY.value: "P0b judge",
        DecisionContract.RELATION_BINARY.value: "relation judge",
        DecisionContract.SPATIAL_FIDELITY_BINARY.value: (
            "spatial-fidelity judge"
        ),
    }
    if decision_contract in binary_labels:
        validate_binary_judge_response(
            normalized,
            judge_label=binary_labels[decision_contract],
        )
        return normalized
    if decision_contract == DecisionContract.GENERIC_VISUAL_SCORE.value:
        validate_generic_visual_response(normalized)
        raise ValueError(
            "generic visual scoring is a compatibility response and cannot "
            "be converted to a metric Judge verdict"
        )
    raise ValueError(
        f"{method_name} uses an unsupported Judge decision_contract "
        f"{decision_contract!r}"
    )


def _validate_native_judge_contract(
    value: dict[str, Any],
    *,
    request: JudgeRequest,
    decision_contract: str,
    method_name: str,
) -> dict[str, Any]:
    """Apply metric-contract invariants to the unified Judge status schema."""

    normalized = deepcopy(value)
    status = str(normalized.get("status") or "").strip()
    legacy_verdict = str(
        normalized.get("canonical_verdict")
        or normalized.get("verdict")
        or ""
    ).strip()
    if legacy_verdict:
        compatible = {
            "valid": {"valid"},
            "invalid": {"invalid"},
            "need_more_evidence": {
                "ambiguous",
                "need_more_evidence",
                "insufficient_evidence",
            },
        }[status]
        if legacy_verdict not in compatible:
            raise ValueError(
                "Judge status conflicts with legacy verdict fields"
            )
    evidence_status = str(
        normalized.get("evidence_status") or ""
    ).strip()
    if evidence_status:
        expected = (
            "insufficient"
            if status == "need_more_evidence"
            else "sufficient"
        )
        if evidence_status != expected:
            raise ValueError(
                "Judge status conflicts with legacy evidence_status"
            )

    if decision_contract == DecisionContract.CANONICAL_METRIC.value:
        evidence_request = (
            EvidenceRequest.from_value(
                normalized.get("evidence_request")
            )
            if status == "need_more_evidence"
            else None
        )
        canonical = {
            "evidence_status": (
                "insufficient"
                if status == "need_more_evidence"
                else "sufficient"
            ),
            "verdict": (
                "ambiguous"
                if status == "need_more_evidence"
                else status
            ),
            "confidence": normalized.get("confidence"),
            "reason": normalized.get("reason"),
            "missing_evidence": (
                list(evidence_request.missing_observations)
                if evidence_request is not None
                else []
            ),
            "defects": deepcopy(normalized.get("defects") or []),
        }
        validate_canonical_metric_response(
            canonical,
            allowed_scopes=_canonical_allowed_scopes(request),
        )
        return normalized
    if decision_contract in {
        DecisionContract.P0B_BINARY.value,
        DecisionContract.RELATION_BINARY.value,
        DecisionContract.SPATIAL_FIDELITY_BINARY.value,
    }:
        # Binary compatibility wrappers may hide need_more_evidence externally,
        # but the internal Judge must still carry a structured request.
        JudgeResult.from_value(normalized)
        return normalized
    raise ValueError(
        f"{method_name} uses an unsupported Judge decision_contract "
        f"{decision_contract!r}"
    )


def _canonical_allowed_scopes(request: JudgeRequest) -> tuple[str, ...]:
    for source in (
        request.context.get("judgment_scope"),
        request.rubric,
    ):
        if not isinstance(source, dict):
            continue
        values = source.get("included") or source.get("allowed_scopes")
        if isinstance(values, (list, tuple)):
            return tuple(
                str(value)
                for value in values
                if isinstance(value, str) and value.strip()
            )
    return ()


def _set_authoritative_provenance(
    provenance: dict[str, Any],
    key: str,
    value: Any,
    *,
    boundary: str,
) -> None:
    if key in provenance and provenance[key] != value:
        raise ValueError(
            f"{boundary} response provenance.{key} conflicts with "
            "the authoritative adapter metadata"
        )
    provenance[key] = deepcopy(value)


def _request_target_ids(request: JudgeRequest) -> list[str]:
    values: list[Any] = []
    for source in (request.claim_or_event, request.context):
        for key in (
            "target_ids",
            "target_object_ids",
            "object_ids",
            "subject_ids",
            "member_ids",
        ):
            if isinstance(source.get(key), list):
                values.extend(source[key])
        for key in (
            "object_id",
            "subject_id",
            "anchor_id",
            "target_id",
        ):
            if source.get(key) is not None:
                values.append(source[key])
    return list(dict.fromkeys(str(value) for value in values if str(value).strip()))


def _legacy_audit_provenance(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value[key])
        for key in (
            "model",
            "endpoint",
            "images_used",
            "request_metadata",
        )
        if value.get(key) is not None
    }

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


class ControlledVLMJudge:
    """Backward-compatible public methods backed by the strict controller."""

    def __init__(
        self,
        judge: Any,
        *,
        control: VLMEvaluationControl,
        camera_provider: Any | None = None,
        camera_selector: Any | None = None,
        deterministic_camera_selector: Any | None = None,
        vlm_camera_selector: Any | None = None,
        evidence_renderer: Any | None = None,
        strict: bool | None = None,
    ) -> None:
        if judge is None:
            raise TypeError("ControlledVLMJudge requires an existing judge")
        self._judge = judge
        self.control = control
        self.camera_provider = camera_provider
        self.camera_selector = camera_selector
        self.deterministic_camera_selector = (
            deterministic_camera_selector
        )
        self.vlm_camera_selector = vlm_camera_selector
        self.evidence_renderer = evidence_renderer
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
        compatibility_screen = _compatibility_screen_request(request)
        provider_available = (
            self.camera_provider is not None
            and self.evidence_renderer is None
            and not compatibility_screen
        )
        independent_renderer_available = (
            self.evidence_renderer is not None
            and not compatibility_screen
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
        elif independent_renderer_available:
            selector = self.camera_selector
            renderer = _coerce_evidence_renderer(self.evidence_renderer)
            candidates = tuple(_request_candidate_views(request))
        else:
            selector = self.camera_selector
            renderer = _UnavailableEvidenceRenderer()
            candidates = tuple(_request_candidate_views(request))
        controller = VLMEvaluationController(
            judge=judge_adapter,
            renderer=renderer,
            camera_selector=selector,
            deterministic_camera_selector=(
                None
                if provider_available
                else self.deterministic_camera_selector
            ),
            vlm_camera_selector=(
                None
                if provider_available
                else self.vlm_camera_selector
            ),
            control=self.control,
        )
        core_request = _judge_request(request)
        result = controller.run(
            core_request,
            evidence_goal=_evidence_goal(
                request,
                camera_repairable=(
                    provider_available or independent_renderer_available
                ),
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
            "deterministic_camera_selector": (
                _qualified_name(self.deterministic_camera_selector)
                if self.deterministic_camera_selector is not None
                else None
            ),
            "vlm_camera_selector": (
                _qualified_name(self.vlm_camera_selector)
                if self.vlm_camera_selector is not None
                else None
            ),
            "evidence_renderer": (
                _qualified_name(self.evidence_renderer)
                if self.evidence_renderer is not None
                else None
            ),
            "scene_access": "read_only",
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
            allow_control_status=self.strict,
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
        compatibility_screen = _compatibility_screen_request(request)
        provider_available = (
            self.camera_provider is not None
            and self.evidence_renderer is None
            and not compatibility_screen
        )
        independent_renderer_available = (
            self.evidence_renderer is not None
            and not compatibility_screen
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
        elif independent_renderer_available:
            selector = self.camera_selector
            renderer = _coerce_evidence_renderer(self.evidence_renderer)
            candidates = tuple(_request_candidate_views(request))
        else:
            selector = self.camera_selector
            renderer = _UnavailableEvidenceRenderer()
            candidates = tuple(_request_candidate_views(request))

        controller = VLMEvaluationController(
            judge=judge_adapter,
            renderer=renderer,
            camera_selector=selector,
            deterministic_camera_selector=(
                None
                if provider_available
                else self.deterministic_camera_selector
            ),
            vlm_camera_selector=(
                None
                if provider_available
                else self.vlm_camera_selector
            ),
            control=self.control,
        )
        core_request = _judge_request(request)
        result = controller.run(
            core_request,
            evidence_goal=_evidence_goal(
                request,
                camera_repairable=(
                    provider_available or independent_renderer_available
                ),
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
            if decision_contract.value in _BINARY_CONTRACTS:
                return _binary_compatibility_response(
                    responses[-1],
                    method_name=method_name,
                    decision_contract=decision_contract,
                )
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
    deterministic_camera_selector: Any | None = None,
    vlm_camera_selector: Any | None = None,
    evidence_renderer: Any | None = None,
    strict: bool | None = None,
) -> Any | None:
    if judge is None or isinstance(judge, ControlledVLMJudge):
        return judge
    return ControlledVLMJudge(
        judge,
        control=control,
        camera_provider=camera_provider,
        camera_selector=camera_selector,
        deterministic_camera_selector=deterministic_camera_selector,
        vlm_camera_selector=vlm_camera_selector,
        evidence_renderer=evidence_renderer,
        strict=strict,
    )


def _resolve_legacy_method(
    judge: Any,
    method_name: str,
    *,
    allow_control_status: bool = True,
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
            *(
                ("_adjudicate_p0b_control",)
                if allow_control_status
                else ()
            ),
            "_adjudicate_p0b_raw",
            "adjudicate_p0b",
        ),
        "adjudicate_relation": (
            *(
                ("_adjudicate_relation_control",)
                if allow_control_status
                else ()
            ),
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


def _binary_compatibility_response(
    raw: dict[str, Any],
    *,
    method_name: str,
    decision_contract: DecisionContract,
) -> dict[str, Any]:
    if raw.get("verdict") in {"valid", "invalid"} and raw.get("status") is None:
        validate_binary_judge_response(
            raw,
            judge_label=method_name,
        )
        return deepcopy(raw)
    native = JudgeResult.from_value(raw)
    if native.status not in {"valid", "invalid"}:
        raise ValueError(
            "binary compatibility response requires a concluded internal Judge"
        )
    result: dict[str, Any] = {
        "verdict": native.status,
        "confidence": native.confidence,
        "reason": native.reason,
    }
    for key in (
        "vlm_role",
        "decision_contract",
        "judge_method",
        "images_used",
        "model",
        "endpoint",
        "request_metadata",
    ):
        if raw.get(key) is not None:
            result[key] = deepcopy(raw[key])
    result.setdefault("vlm_role", "judge")
    result.setdefault("decision_contract", decision_contract.value)
    result.setdefault("judge_method", method_name)
    validate_binary_judge_response(
        result,
        judge_label=method_name,
    )
    return result


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


def _legacy_evidence_ref(value: Any) -> Any:
    if isinstance(value, dict):
        path = value.get("path") or value.get("image_path")
        if path is not None and str(path).strip():
            return str(path)
    return deepcopy(value)


def _qualified_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"
