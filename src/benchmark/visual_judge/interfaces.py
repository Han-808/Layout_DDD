from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol, runtime_checkable

from benchmark.visual_judge.contracts import (
    validate_binary_judge_response,
    validate_camera_selection_response,
    validate_canonical_metric_response,
    validate_generic_visual_response,
)
from benchmark.visual_judge.roles import DecisionContract


JUDGE_STATUSES = {"valid", "invalid", "need_more_evidence"}


@dataclass(frozen=True)
class EvidenceRequest:
    """Structured observation request emitted by a Judge."""

    target_ids: tuple[str, ...]
    missing_observations: tuple[str, ...]
    view_goal: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> EvidenceRequest:
        if isinstance(value, cls):
            value = value.to_dict()
        if not isinstance(value, dict):
            raise ValueError(
                "Judge need_more_evidence requires a structured evidence_request"
            )
        target_ids = _string_tuple(value.get("target_ids"))
        if not target_ids:
            raise ValueError(
                "Judge evidence_request must identify target_ids; "
                "use 'scene' for a global request"
            )
        missing = _string_tuple(
            value.get("missing_observations")
            if value.get("missing_observations") is not None
            else value.get("missing_evidence")
        )
        view_goal = str(value.get("view_goal") or "").strip()
        if not missing:
            raise ValueError(
                "Judge evidence_request must name missing observations"
            )
        if not view_goal:
            raise ValueError("Judge evidence_request must provide a view_goal")
        metadata = value.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("Judge evidence_request metadata must be a JSON object")
        return cls(
            target_ids=target_ids,
            missing_observations=missing,
            view_goal=view_goal,
            metadata=deepcopy(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_ids": list(self.target_ids),
            "missing_observations": list(self.missing_observations),
            "view_goal": self.view_goal,
            "metadata": deepcopy(self.metadata),
        }


@dataclass(frozen=True)
class JudgeRequest:
    """Implementation-independent input to a metric-scoped Judge."""

    task: str
    metric: str
    claim_or_event: dict[str, Any]
    scene_context: dict[str, Any]
    deterministic_evidence: dict[str, Any]
    visual_evidence: tuple[Any, ...]
    rubric: Any
    context: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> JudgeRequest:
        if isinstance(value, cls):
            value = value.to_dict()
        if not isinstance(value, dict):
            raise TypeError("Judge request must be a JSON object")
        task = str(value.get("task") or value.get("category") or "").strip()
        metric = str(value.get("metric") or task).strip()
        if not task:
            raise ValueError("Judge request requires a task")
        if not metric:
            raise ValueError("Judge request requires a metric")
        claim_or_event = value.get("claim_or_event")
        if claim_or_event is None:
            claim_or_event = (
                value.get("claim")
                if isinstance(value.get("claim"), dict)
                else value.get("event")
            )
        scene_context = value.get("scene_context")
        if scene_context is None:
            scene_context = value.get("scene_summary") or value.get("scene")
        deterministic = value.get("deterministic_evidence")
        if deterministic is None:
            deterministic = value.get("detector_evidence")
        visual = value.get("visual_evidence")
        if visual is None:
            visual = value.get("render_evidence")
        context = value.get("context")
        return cls(
            task=task,
            metric=metric,
            claim_or_event=_dict_or_empty(claim_or_event, "claim_or_event"),
            scene_context=_dict_or_empty(scene_context, "scene_context"),
            deterministic_evidence=_dict_or_empty(
                deterministic, "deterministic_evidence"
            ),
            visual_evidence=tuple(_list_or_empty(visual, "visual_evidence")),
            rubric=deepcopy(value.get("rubric") or value.get("metric_rubric")),
            context=_dict_or_empty(context, "context"),
        )

    def with_visual_evidence(self, items: list[Any]) -> JudgeRequest:
        return JudgeRequest(
            task=self.task,
            metric=self.metric,
            claim_or_event=deepcopy(self.claim_or_event),
            scene_context=deepcopy(self.scene_context),
            deterministic_evidence=deepcopy(self.deterministic_evidence),
            visual_evidence=tuple(deepcopy(items)),
            rubric=deepcopy(self.rubric),
            context=deepcopy(self.context),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vlm_role": "judge",
            "task": self.task,
            "metric": self.metric,
            "claim_or_event": deepcopy(self.claim_or_event),
            "scene_context": deepcopy(self.scene_context),
            "deterministic_evidence": deepcopy(self.deterministic_evidence),
            "visual_evidence": list(deepcopy(self.visual_evidence)),
            "rubric": deepcopy(self.rubric),
            "context": deepcopy(self.context),
        }


@dataclass(frozen=True)
class JudgeResult:
    status: str
    confidence: float
    reason: str
    defects: tuple[dict[str, Any], ...] = ()
    evidence_request: EvidenceRequest | None = None
    backend: str = "unknown"
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> JudgeResult:
        if isinstance(value, cls):
            value = value.to_dict()
        if not isinstance(value, dict):
            raise ValueError("Judge response must be a JSON object")
        status = str(value.get("status") or "").strip()
        if status not in JUDGE_STATUSES:
            raise ValueError(
                "Judge status must be valid, invalid, or need_more_evidence"
            )
        confidence = _confidence(value.get("confidence"))
        reason = str(value.get("reason") or "").strip()
        if not reason:
            raise ValueError("Judge response must include a reason")
        defects = value.get("defects")
        if defects is None:
            defects = []
        if not isinstance(defects, list) or not all(
            isinstance(item, dict) for item in defects
        ):
            raise ValueError("Judge defects must be a JSON list of objects")
        evidence_request = (
            EvidenceRequest.from_value(value.get("evidence_request"))
            if status == "need_more_evidence"
            else None
        )
        if status != "need_more_evidence" and value.get("evidence_request") is not None:
            raise ValueError(
                "Judge evidence_request is only valid with need_more_evidence"
            )
        if status == "need_more_evidence" and defects:
            raise ValueError(
                "Judge need_more_evidence response cannot assert metric defects"
            )
        if status == "valid" and defects:
            raise ValueError("Judge valid response cannot retain metric defects")
        backend = str(value.get("backend") or "unknown")
        provenance = value.get("provenance")
        if provenance is None:
            provenance = {}
        if not isinstance(provenance, dict):
            raise ValueError("Judge provenance must be a JSON object")
        return cls(
            status=status,
            confidence=confidence,
            reason=reason,
            defects=tuple(deepcopy(defects)),
            evidence_request=evidence_request,
            backend=backend,
            provenance=deepcopy(provenance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "reason": self.reason,
            "defects": list(deepcopy(self.defects)),
            "evidence_request": (
                self.evidence_request.to_dict()
                if self.evidence_request is not None
                else None
            ),
            "backend": self.backend,
            "provenance": deepcopy(self.provenance),
        }


@runtime_checkable
class Judge(Protocol):
    def judge(self, request: JudgeRequest) -> JudgeResult: ...


@dataclass(frozen=True)
class EvidenceGateRequest:
    task: str
    metric: str
    target_ids: tuple[str, ...]
    scene: dict[str, Any]
    visual_evidence: tuple[Any, ...]
    evidence_goal: dict[str, Any] = field(default_factory=dict)
    manifest_path: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceGateResult:
    ready: bool
    camera_repairable: bool
    reason_codes: tuple[str, ...]
    deficiencies: tuple[dict[str, Any], ...]
    backend: str = "deterministic"
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: Any) -> EvidenceGateResult:
        if isinstance(value, cls):
            value = value.to_dict()
        if not isinstance(value, dict):
            raise ValueError("EvidenceGate response must be a JSON object")
        if "verdict" in value or "score" in value:
            raise ValueError("EvidenceGate must not return metric verdict or score")
        ready = value.get("ready")
        camera_repairable = value.get("camera_repairable")
        if not isinstance(ready, bool):
            raise ValueError("EvidenceGate ready must be boolean")
        if not isinstance(camera_repairable, bool):
            raise ValueError("EvidenceGate camera_repairable must be boolean")
        reason_codes = _string_tuple(value.get("reason_codes"))
        deficiencies = value.get("deficiencies")
        if deficiencies is None:
            deficiencies = []
        if not isinstance(deficiencies, list) or not all(
            isinstance(item, dict) for item in deficiencies
        ):
            raise ValueError("EvidenceGate deficiencies must be a JSON list")
        if ready and deficiencies:
            raise ValueError("ready EvidenceGate result cannot retain deficiencies")
        if ready and camera_repairable:
            raise ValueError(
                "ready EvidenceGate result cannot request camera repair"
            )
        if ready and any(
            code
            not in {
                "evidence_ready",
                "evidence_gate_explicitly_disabled",
            }
            for code in reason_codes
        ):
            raise ValueError(
                "ready EvidenceGate result cannot retain failure reason codes"
            )
        if not ready and not reason_codes and not deficiencies:
            raise ValueError(
                "not-ready EvidenceGate result must explain its deficiencies"
            )
        if camera_repairable and (
            not deficiencies
            or any(
                str(item.get("repairability") or "") != "camera"
                for item in deficiencies
            )
        ):
            raise ValueError(
                "camera-repairable EvidenceGate result requires only "
                "camera-repairable deficiencies"
            )
        provenance = value.get("provenance")
        if provenance is None:
            provenance = {}
        if not isinstance(provenance, dict):
            raise ValueError("EvidenceGate provenance must be a JSON object")
        return cls(
            ready=ready,
            camera_repairable=camera_repairable,
            reason_codes=reason_codes,
            deficiencies=tuple(deepcopy(deficiencies)),
            backend=str(value.get("backend") or "deterministic"),
            provenance=deepcopy(provenance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "camera_repairable": self.camera_repairable,
            "reason_codes": list(self.reason_codes),
            "deficiencies": list(deepcopy(self.deficiencies)),
            "backend": self.backend,
            "provenance": deepcopy(self.provenance),
        }


@runtime_checkable
class EvidenceGate(Protocol):
    def check(self, request: EvidenceGateRequest) -> EvidenceGateResult: ...


@dataclass(frozen=True)
class CameraSelectionRequest:
    task: str
    metric: str
    target_ids: tuple[str, ...]
    scene: dict[str, Any]
    evidence_goal: dict[str, Any]
    existing_visual_evidence: tuple[Any, ...]
    budget: dict[str, int]
    candidate_views: tuple[dict[str, Any], ...] = ()
    allowed_actions: tuple[str, ...] = ()
    evidence_round: int = 0
    allow_freeform_pose: bool = False
    allow_scene_mutation: bool = False
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "task": self.task,
            "metric": self.metric,
            "target_ids": list(self.target_ids),
            "scene": deepcopy(self.scene),
            "evidence_goal": deepcopy(self.evidence_goal),
            "existing_visual_evidence": list(
                deepcopy(self.existing_visual_evidence)
            ),
            "budget": deepcopy(self.budget),
            "candidate_views": list(deepcopy(self.candidate_views)),
            "allowed_actions": list(self.allowed_actions),
            "evidence_round": self.evidence_round,
            "allow_freeform_pose": self.allow_freeform_pose,
            "allow_scene_mutation": self.allow_scene_mutation,
            "context": deepcopy(self.context),
        }
        # Existing selectors consume these historical names.
        result["candidates"] = list(deepcopy(self.candidate_views))
        result["max_views"] = _positive_int(
            self.budget.get("max_views_per_round"),
            "CameraSelector max_views_per_round",
        )
        result["object_ids"] = list(self.target_ids)
        # Keep implementation-specific hints out of the stable dataclass while
        # still allowing the existing selector wrapper to consume its historical
        # request keys unchanged.
        for key in (
            "allow_adjustment",
            "color_legend",
            "corrective_proposals",
            "decision_contract",
            "evidence_deficiency",
            "judge_method",
            "preview_degradation",
            "preview_role",
            "preview_visibility_warning",
            "selection_phase",
            "vlm_role",
        ):
            if key in self.context and key not in result:
                result[key] = deepcopy(self.context[key])
        return result


@dataclass(frozen=True)
class CameraSelectionResult:
    selected_view_ids: tuple[str, ...]
    selected_views: tuple[dict[str, Any], ...] = ()
    camera_proposal: dict[str, Any] | None = None
    camera_actions: tuple[dict[str, Any], ...] = ()
    reason: str = ""
    backend: str = "unknown"
    evidence_round: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_view_ids": list(self.selected_view_ids),
            "selected_views": list(deepcopy(self.selected_views)),
            "camera_proposal": deepcopy(self.camera_proposal),
            "camera_actions": list(deepcopy(self.camera_actions)),
            "reason": self.reason,
            "backend": self.backend,
            "evidence_round": self.evidence_round,
            "provenance": deepcopy(self.provenance),
        }


@runtime_checkable
class CameraSelector(Protocol):
    def select(self, request: CameraSelectionRequest) -> CameraSelectionResult: ...


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
                        or ["metric_scoped_visual_confirmation"]
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


class ExistingCameraSelectorAdapter:
    """Expose an existing select/select_camera_views object through one interface."""

    def __init__(self, selector: Any, *, backend: str = "existing") -> None:
        call = getattr(selector, "select", None)
        self._stable_call = callable(call)
        if self._stable_call:
            self.method_name = "select"
        else:
            call = getattr(selector, "select_camera_views", None)
            self.method_name = "select_camera_views"
        if not callable(call):
            raise TypeError(
                "existing CameraSelector adapter requires select(request) or "
                "select_camera_views(request)"
            )
        self._selector = selector
        self._call = call
        self.backend = str(backend)

    def _invoke(
        self,
        request: CameraSelectionRequest,
        *,
        legacy_payload: dict[str, Any] | None = None,
    ) -> Any:
        if self._stable_call:
            return self._call(request)
        return self._call(
            request.to_dict() if legacy_payload is None else legacy_payload
        )

    def select(self, request: CameraSelectionRequest) -> CameraSelectionResult:
        raw = self._invoke(request)
        result = camera_selection_result_from_value(
            raw,
            request=request,
            backend=self.backend,
        )
        return _with_selector_provenance(
            result,
            adapter=type(self).__name__,
        )


class VLMCameraSelector(ExistingCameraSelectorAdapter):
    def __init__(self, selector: Any) -> None:
        super().__init__(selector, backend="vlm")

    def select(self, request: CameraSelectionRequest) -> CameraSelectionResult:
        audit = {
            "vlm_role": "vlm_camera_selector",
            "decision_contract": "camera_selection_v1",
            "judge_method": self.method_name,
        }
        audited_request = replace(
            request,
            context={
                **deepcopy(request.context),
                **audit,
            },
        )
        payload = audited_request.to_dict()
        raw = self._invoke(
            audited_request,
            legacy_payload=payload,
        )
        result = camera_selection_result_from_value(
            raw,
            request=audited_request,
            backend=self.backend,
        )
        selector_audit = _selector_object_audit(self._selector)
        selector_audit["images_used"] = [
            str(item.get("id"))
            for item in audited_request.candidate_views
        ]
        return _with_selector_provenance(
            result,
            adapter=type(self).__name__,
            vlm_role="vlm_camera_selector",
            decision_contract="camera_selection_v1",
            judge_method=self.method_name,
            audit=selector_audit,
        )


class DeterministicCameraSelector:
    """Deterministic adapter; strategy remains injectable and implementation-free."""

    def __init__(
        self,
        selection_fn: Callable[[CameraSelectionRequest], Any] | None = None,
    ) -> None:
        self._selection_fn = selection_fn

    def select(self, request: CameraSelectionRequest) -> CameraSelectionResult:
        if self._selection_fn is not None:
            return camera_selection_result_from_value(
                self._selection_fn(request),
                request=request,
                backend="deterministic",
            )
        limit = _positive_int(
            request.budget.get("max_views_per_round"),
            "CameraSelector max_views_per_round",
        )
        selected = list(request.candidate_views[:limit])
        if not selected:
            raise ValueError(
                "deterministic CameraSelector requires candidate views or an "
                "injected selection strategy"
            )
        return camera_selection_result_from_value(
            {
                "selected_view_ids": [str(item.get("id") or "") for item in selected],
                "selected_views": selected,
                "reason": "stable candidate order",
                "provenance": {"strategy": "stable_input_order"},
            },
            request=request,
            backend="deterministic",
        )


class HybridCameraSelector:
    """Try one selector and fall back to another without changing the interface."""

    def __init__(self, primary: CameraSelector, fallback: CameraSelector) -> None:
        self.primary = primary
        self.fallback = fallback

    def select(self, request: CameraSelectionRequest) -> CameraSelectionResult:
        try:
            result = camera_selection_result_from_value(
                self.primary.select(request),
                request=request,
                backend="hybrid_primary",
            )
            route = "primary"
        except Exception as exc:
            result = camera_selection_result_from_value(
                self.fallback.select(request),
                request=request,
                backend="hybrid_fallback",
            )
            route = "fallback"
            fallback_error = f"{type(exc).__name__}: {exc}"
        provenance = deepcopy(result.provenance)
        provenance["hybrid_route"] = route
        if route == "fallback":
            provenance["primary_error"] = fallback_error
        return CameraSelectionResult(
            selected_view_ids=result.selected_view_ids,
            selected_views=result.selected_views,
            camera_proposal=deepcopy(result.camera_proposal),
            camera_actions=result.camera_actions,
            reason=result.reason,
            backend="hybrid",
            evidence_round=result.evidence_round,
            provenance=provenance,
        )


def build_camera_selector(
    *,
    backend: str,
    existing: Any | None = None,
    deterministic: CameraSelector
    | Callable[[CameraSelectionRequest], Any]
    | None = None,
    vlm: Any | None = None,
) -> CameraSelector:
    """Resolve a configured backend without exposing its algorithm in the interface."""

    backend_name = str(backend).strip().lower()
    if backend_name not in {"existing", "deterministic", "vlm", "hybrid"}:
        raise ValueError(
            "CameraSelector backend must be existing, deterministic, vlm, or hybrid"
        )
    deterministic_selector = _coerce_deterministic_selector(deterministic)
    if backend_name == "deterministic":
        return deterministic_selector
    if backend_name == "existing":
        if existing is None:
            return deterministic_selector
        return _coerce_existing_selector(existing, backend="existing")
    if vlm is None:
        raise ValueError(
            f"CameraSelector backend {backend_name!r} requires a VLM selector"
        )
    vlm_selector = _coerce_existing_selector(vlm, backend="vlm")
    if backend_name == "vlm":
        return vlm_selector
    return HybridCameraSelector(vlm_selector, deterministic_selector)


def camera_selection_result_from_value(
    value: Any,
    *,
    request: CameraSelectionRequest,
    backend: str,
) -> CameraSelectionResult:
    typed_result = isinstance(value, CameraSelectionResult)
    if typed_result:
        raw = value.to_dict()
    elif isinstance(value, dict):
        raw = deepcopy(value)
    else:
        raise ValueError("CameraSelector response must be a JSON object")
    forbidden = [key for key in ("verdict", "score") if key in raw]
    if forbidden:
        raise ValueError(
            "CameraSelector response must not contain metric verdict or score"
        )
    if _contains_scene_mutation(raw):
        raise ValueError("CameraSelector response must not contain scene mutation")

    by_id: dict[str, dict[str, Any]] = {}
    for item in request.candidate_views:
        if not isinstance(item, dict):
            raise ValueError(
                "CameraSelector candidate_views must contain JSON objects"
            )
        candidate_id = str(item.get("id") or "").strip()
        if not candidate_id:
            raise ValueError(
                "CameraSelector candidate views require non-empty IDs"
            )
        if candidate_id in by_id:
            raise ValueError(
                "CameraSelector candidate view IDs must be unique"
            )
        by_id[candidate_id] = item
    available_ids = list(by_id)

    trusted_proposals = _trusted_corrective_proposals(request)
    proposal = raw.get("camera_proposal")
    raw_ids = raw.get("selected_view_ids")
    if raw_ids is None and proposal is not None:
        raw_ids = []
    if not isinstance(raw_ids, list):
        raise ValueError("CameraSelector selected_view_ids must be a list")
    if raw_ids:
        selected_ids = validate_camera_selection_response(
            {**raw, "selected_view_ids": raw_ids},
            available_view_ids=available_ids,
            max_views=_positive_int(
                request.budget.get("max_views_per_round"),
                "CameraSelector max_views_per_round",
            ),
        )
    else:
        selected_ids = []

    selected_views = raw.get("selected_views")
    if selected_views is None or (
        not selected_views and selected_ids
    ):
        selected_views = [deepcopy(by_id[item_id]) for item_id in selected_ids]
    if not isinstance(selected_views, list) or not all(
        isinstance(item, dict) for item in selected_views
    ):
        raise ValueError("CameraSelector selected_views must be a JSON list")
    if len(selected_views) != len(selected_ids):
        raise ValueError(
            "CameraSelector selected_views must match selected_view_ids"
        )
    for item, selected_id in zip(selected_views, selected_ids):
        item_id = str(item.get("id") or "")
        if item_id != selected_id or item != by_id[selected_id]:
            raise ValueError(
                "CameraSelector selected_views must exactly match trusted candidates"
            )

    action = raw.get("action")
    actions = raw.get("camera_actions")
    if action is not None and actions is not None:
        raise ValueError(
            "CameraSelector must not return both action and camera_actions"
        )
    if actions is None:
        actions = [action] if isinstance(action, dict) else []
    if action is not None and not isinstance(action, dict):
        raise ValueError("CameraSelector action must be null or a JSON object")
    if not isinstance(actions, list) or not all(
        isinstance(item, dict) for item in actions
    ):
        raise ValueError("CameraSelector camera_actions must be a JSON list")
    allowed = set(request.allowed_actions)
    resolved_actions: list[dict[str, Any]] = []
    for item in actions:
        resolved_actions.append(
            _validated_camera_action(
                item,
                available_ids=set(available_ids),
                allowed_actions=allowed,
                trusted_proposals=trusted_proposals,
            )
        )

    resolved_proposal: dict[str, Any] | None = None
    if proposal is not None:
        if not isinstance(proposal, dict):
            raise ValueError("CameraSelector camera_proposal must be a JSON object")
        if selected_ids or selected_views:
            raise ValueError(
                "CameraSelector selected views and camera_proposal are mutually exclusive"
            )
        if resolved_actions:
            raise ValueError(
                "CameraSelector camera_proposal and camera_actions are mutually exclusive"
            )
        if not request.allow_freeform_pose:
            proposal_id = str(proposal.get("proposal_id") or "")
            trusted = trusted_proposals.get(proposal_id)
            if trusted is None:
                raise ValueError(
                    "CameraSelector returned an unverifiable or free-form proposal"
                )
            if set(proposal) != {"proposal_id"} and (
                not typed_result or proposal != trusted
            ):
                raise ValueError(
                    "bounded CameraSelector proposal must reference only proposal_id"
                )
            _require_trusted_proposal(trusted)
            resolved_proposal = deepcopy(trusted)
        else:
            if proposal.get("validated") is False:
                raise ValueError(
                    "CameraSelector proposal explicitly failed validation"
                )
            pose_validator = request.context.get("pose_validator")
            if not callable(pose_validator):
                raise ValueError(
                    "free-form CameraSelector proposal requires an injected "
                    "pose validator"
                )
            if pose_validator(
                deepcopy(proposal),
                deepcopy(request.scene),
            ) is not True:
                raise ValueError(
                    "free-form CameraSelector proposal failed pose validation"
                )
            resolved_proposal = deepcopy(proposal)
    if not selected_ids and proposal is None:
        raise ValueError(
            "CameraSelector must select a known view or return a validated proposal"
        )
    reason = str(raw.get("reason") or "").strip()
    if not reason:
        raise ValueError("CameraSelector response must include a reason")
    provenance = raw.get("provenance")
    if provenance is None:
        provenance = {}
    if not isinstance(provenance, dict):
        raise ValueError("CameraSelector provenance must be a JSON object")
    for key in (
        "model",
        "endpoint",
        "images_used",
        "judge_method",
        "request_metadata",
    ):
        if raw.get(key) is not None:
            provenance.setdefault(key, deepcopy(raw[key]))
    if raw.get("backend") is not None:
        provenance.setdefault("reported_backend", str(raw["backend"]))
    if raw.get("evidence_round") is not None:
        provenance.setdefault(
            "reported_evidence_round",
            deepcopy(raw["evidence_round"]),
        )
    return CameraSelectionResult(
        selected_view_ids=tuple(selected_ids),
        selected_views=tuple(deepcopy(selected_views)),
        camera_proposal=resolved_proposal,
        camera_actions=tuple(resolved_actions),
        reason=reason,
        backend=str(backend),
        evidence_round=_nonnegative_int(
            request.evidence_round,
            "CameraSelector evidence_round",
        ),
        provenance=deepcopy(provenance),
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


def _trusted_corrective_proposals(
    request: CameraSelectionRequest,
) -> dict[str, dict[str, Any]]:
    values = request.context.get("corrective_proposals", [])
    if values is None:
        values = []
    if not isinstance(values, (list, tuple)):
        raise ValueError(
            "CameraSelector corrective_proposals must be a JSON list"
        )
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(
                "CameraSelector corrective_proposals must contain JSON objects"
            )
        proposal_id = str(value.get("proposal_id") or "").strip()
        if not proposal_id:
            raise ValueError(
                "CameraSelector corrective proposals require proposal_id"
            )
        if proposal_id in result:
            raise ValueError(
                "CameraSelector corrective proposal IDs must be unique"
            )
        result[proposal_id] = value
    return result


def _require_trusted_proposal(value: dict[str, Any]) -> None:
    if value.get("validated") is False:
        raise ValueError(
            "CameraSelector corrective proposal explicitly failed validation"
        )
    if _contains_scene_mutation(value):
        raise ValueError(
            "CameraSelector corrective proposal must not mutate the scene"
        )


def _validated_camera_action(
    value: dict[str, Any],
    *,
    available_ids: set[str],
    allowed_actions: set[str],
    trusted_proposals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    allowed_fields = {
        "proposal_id",
        "view_id",
        "type",
        "action",
        "family",
    }
    if not value or any(key not in allowed_fields for key in value):
        raise ValueError(
            "CameraSelector camera action contains empty or untrusted fields"
        )
    action_value = value.get("action")
    type_value = value.get("type")
    if (
        action_value is not None
        and type_value is not None
        and str(action_value).strip() != str(type_value).strip()
    ):
        raise ValueError("CameraSelector camera action type is inconsistent")
    proposal_id = _optional_text(value.get("proposal_id"), "proposal_id")
    requested_type = _optional_text(
        type_value if type_value is not None else action_value,
        "action type",
    )
    requested_view = _optional_text(value.get("view_id"), "view_id")
    requested_family = _optional_text(value.get("family"), "family")

    if proposal_id:
        trusted = trusted_proposals.get(proposal_id)
        if trusted is None:
            raise ValueError(
                "CameraSelector action references an unknown corrective proposal"
            )
        _require_trusted_proposal(trusted)
        trusted_type = _optional_text(
            trusted.get("action_primitive")
            if trusted.get("action_primitive") is not None
            else trusted.get("type")
            if trusted.get("type") is not None
            else trusted.get("action"),
            "trusted action type",
        )
        trusted_view = _optional_text(
            trusted.get("parent_view_id")
            if trusted.get("parent_view_id") is not None
            else trusted.get("view_id"),
            "trusted view_id",
        )
        trusted_family = _optional_text(
            trusted.get("family"),
            "trusted family",
        )
        if not trusted_type or not trusted_view:
            raise ValueError(
                "CameraSelector corrective proposal lacks a verifiable action"
            )
        if requested_type and requested_type != trusted_type:
            raise ValueError(
                "CameraSelector action type does not match trusted proposal"
            )
        if requested_view and requested_view != trusted_view:
            raise ValueError(
                "CameraSelector action view does not match trusted proposal"
            )
        if requested_family and requested_family != trusted_family:
            raise ValueError(
                "CameraSelector action family does not match trusted proposal"
            )
        action_type = trusted_type
        view_id = trusted_view
        family = trusted_family
    else:
        if "family" in value:
            raise ValueError(
                "CameraSelector action family requires a corrective proposal"
            )
        action_type = requested_type
        view_id = requested_view
        family = ""

    if not action_type or action_type not in allowed_actions:
        raise ValueError(
            "CameraSelector returned an action outside the allowlist"
        )
    if not view_id or view_id not in available_ids:
        raise ValueError(
            "CameraSelector action references an unknown candidate"
        )
    result = {"view_id": view_id, "type": action_type}
    if proposal_id:
        result["proposal_id"] = proposal_id
    if family:
        result["family"] = family
    return result


def _optional_text(value: Any, label: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"CameraSelector {label} must be a non-empty string")
    return value.strip()


def _contains_scene_mutation(value: Any) -> bool:
    if isinstance(value, dict):
        if any(
            key in {"scene_mutation", "scene_patch"}
            for key in value
        ):
            return True
        return any(_contains_scene_mutation(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_scene_mutation(item) for item in value)
    return False


def _selector_object_audit(selector: Any) -> dict[str, Any]:
    model = getattr(selector, "model", None)
    return {
        "model": (
            getattr(model, "model_id", None)
            or getattr(selector, "model_id", None)
        ),
        "endpoint": (
            getattr(model, "endpoint", None)
            or getattr(selector, "endpoint", None)
        ),
    }


def _coerce_existing_selector(value: Any, *, backend: str) -> CameraSelector:
    if backend == "vlm":
        if isinstance(value, VLMCameraSelector):
            return value
        return VLMCameraSelector(value)
    if callable(getattr(value, "select", None)):
        return value
    return ExistingCameraSelectorAdapter(value, backend=backend)


def _coerce_deterministic_selector(
    value: CameraSelector
    | Callable[[CameraSelectionRequest], Any]
    | None,
) -> CameraSelector:
    if value is None:
        return DeterministicCameraSelector()
    if callable(getattr(value, "select", None)):
        return value
    if callable(value):
        return DeterministicCameraSelector(value)
    raise TypeError(
        "deterministic CameraSelector backend requires select(request) "
        "or a selection callable"
    )


def _with_selector_provenance(
    result: CameraSelectionResult,
    *,
    adapter: str,
    vlm_role: str | None = None,
    decision_contract: str | None = None,
    judge_method: str | None = None,
    audit: dict[str, Any] | None = None,
) -> CameraSelectionResult:
    provenance = deepcopy(result.provenance)
    _set_authoritative_provenance(
        provenance,
        "adapter",
        str(adapter),
        boundary="CameraSelector",
    )
    if vlm_role is not None:
        _set_authoritative_provenance(
            provenance,
            "vlm_role",
            vlm_role,
            boundary="CameraSelector",
        )
    if decision_contract is not None:
        _set_authoritative_provenance(
            provenance,
            "decision_contract",
            decision_contract,
            boundary="CameraSelector",
        )
    if judge_method is not None:
        _set_authoritative_provenance(
            provenance,
            "judge_method",
            judge_method,
            boundary="CameraSelector",
        )
    for key, value in (audit or {}).items():
        if value is not None:
            if key in {"model", "endpoint"}:
                _set_authoritative_provenance(
                    provenance,
                    key,
                    deepcopy(value),
                    boundary="CameraSelector",
                )
            else:
                provenance.setdefault(key, deepcopy(value))
    return CameraSelectionResult(
        selected_view_ids=result.selected_view_ids,
        selected_views=result.selected_views,
        camera_proposal=deepcopy(result.camera_proposal),
        camera_actions=result.camera_actions,
        reason=result.reason,
        backend=result.backend,
        evidence_round=result.evidence_round,
        provenance=provenance,
    )


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
        for key in ("target_ids", "object_ids"):
            if isinstance(source.get(key), list):
                values.extend(source[key])
        for key in ("object_id", "subject_id", "target_id"):
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


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Judge confidence must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError("Judge confidence must be between 0 and 1")
    return result


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("structured identifier fields must be JSON lists")
    return tuple(dict.fromkeys(str(item) for item in value if str(item).strip()))


def _dict_or_empty(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"Judge {label} must be a JSON object")
    return deepcopy(value)


def _list_or_empty(value: Any, label: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"Judge {label} must be a JSON list")
    return list(deepcopy(value))


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value
