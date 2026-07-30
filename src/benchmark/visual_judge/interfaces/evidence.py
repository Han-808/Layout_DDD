from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from benchmark.visual_judge.interfaces.camera import CameraSelectionResult
from benchmark.visual_judge.interfaces.judge import JudgeRequest


EVIDENCE_MERGE_POLICIES = {"append", "replace"}


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


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("structured identifier fields must be JSON lists")
    return tuple(dict.fromkeys(str(item) for item in value if str(item).strip()))

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
                "scene_access": "read_only",
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
            if _contains_scene_mutation_marker(value):
                raise ValueError(
                    "evidence renderer must not return scene mutation"
                )
            forbidden = [
                key
                for key in (
                    "verdict",
                    "score",
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
            reported_scene_access = value.get("scene_access")
            if (
                reported_scene_access is not None
                and str(reported_scene_access) != "read_only"
            ):
                raise ValueError(
                    "evidence renderer scene access must be read_only"
                )
            if reported_scene_access is not None:
                provenance = deepcopy(provenance)
                provenance.setdefault(
                    "scene_access",
                    str(reported_scene_access),
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
        scene_access = str(result.provenance.get("scene_access") or "read_only")
        if scene_access != "read_only":
            raise ValueError(
                "evidence renderer scene access must be read_only"
            )
        if _contains_scene_mutation_marker(result.provenance):
            raise ValueError(
                "evidence renderer provenance must not contain scene mutation"
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

def _contains_scene_mutation_marker(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in {
                "allow_scene_mutation",
                "scene_mutated",
                "scene_mutation",
                "scene_patch",
            }:
                return True
            if _contains_scene_mutation_marker(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(
            _contains_scene_mutation_marker(item)
            for item in value
        )
    return False

def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value
