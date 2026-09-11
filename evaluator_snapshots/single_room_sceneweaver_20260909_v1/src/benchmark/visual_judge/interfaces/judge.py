from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


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
