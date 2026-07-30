from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


VLM_EVALUATION_CONTROL_VERSION = "vlm_evaluation_control_v1"

DEFAULT_VLM_EVALUATION_CONTROL: dict[str, Any] = {
    "schema_version": VLM_EVALUATION_CONTROL_VERSION,
    "camera_selector": {
        "backend": "existing",
        "allow_freeform_pose": False,
        "allow_scene_mutation": False,
    },
    "evidence_gate": {
        "enabled": True,
        "backend": "deterministic",
    },
    "judge": {
        "allow_need_more_evidence": True,
    },
    "budgets": {
        "max_evidence_rounds": 2,
        "max_views_per_round": 2,
        "max_total_images": 6,
        "max_camera_actions": 2,
        "max_selector_calls": 3,
    },
    "require_evidence_gate_after_render": True,
    "on_non_camera_repairable_evidence": "unresolved",
    "on_budget_exhausted": "unresolved",
    "on_selector_failure": "keep_previous_evidence",
    "on_render_failure": "unresolved",
}

_POLICY_VALUES = {
    "on_non_camera_repairable_evidence": {"unresolved"},
    "on_budget_exhausted": {"unresolved"},
    "on_selector_failure": {"keep_previous_evidence", "unresolved"},
    "on_render_failure": {"unresolved"},
}


@dataclass(frozen=True)
class VLMEvaluationControl:
    schema_version: str
    camera_selector_backend: str
    allow_freeform_pose: bool
    allow_scene_mutation: bool
    evidence_gate_enabled: bool
    evidence_gate_backend: str
    judge_allow_need_more_evidence: bool
    max_evidence_rounds: int
    max_views_per_round: int
    max_total_images: int
    max_camera_actions: int
    max_selector_calls: int
    require_evidence_gate_after_render: bool
    on_non_camera_repairable_evidence: str
    on_budget_exhausted: str
    on_selector_failure: str
    on_render_failure: str
    requested: dict[str, Any]
    sources: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "camera_selector": {
                "backend": self.camera_selector_backend,
                "allow_freeform_pose": self.allow_freeform_pose,
                "allow_scene_mutation": self.allow_scene_mutation,
            },
            "evidence_gate": {
                "enabled": self.evidence_gate_enabled,
                "backend": self.evidence_gate_backend,
            },
            "judge": {
                "allow_need_more_evidence": (
                    self.judge_allow_need_more_evidence
                ),
            },
            "budgets": {
                "max_evidence_rounds": self.max_evidence_rounds,
                "max_views_per_round": self.max_views_per_round,
                "max_total_images": self.max_total_images,
                "max_camera_actions": self.max_camera_actions,
                "max_selector_calls": self.max_selector_calls,
            },
            "require_evidence_gate_after_render": (
                self.require_evidence_gate_after_render
            ),
            "on_non_camera_repairable_evidence": (
                self.on_non_camera_repairable_evidence
            ),
            "on_budget_exhausted": self.on_budget_exhausted,
            "on_selector_failure": self.on_selector_failure,
            "on_render_failure": self.on_render_failure,
        }

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requested": deepcopy(self.requested),
            "effective": self.to_dict(),
            "sources": deepcopy(self.sources),
        }


def resolve_vlm_evaluation_control(
    value: dict[str, Any] | None = None,
    *,
    existing_max_views: int | None = None,
    existing_max_steps: int | None = None,
    existing_selector_available: bool | None = None,
    judge_max_images: int | None = None,
    overrides: dict[str, Any] | None = None,
) -> VLMEvaluationControl:
    """Resolve defaults, existing provider limits, and explicit overrides."""

    requested = deepcopy(DEFAULT_VLM_EVALUATION_CONTROL)
    sources = {
        path: "default"
        for path in (
            "camera_selector.backend",
            "camera_selector.allow_freeform_pose",
            "camera_selector.allow_scene_mutation",
            "evidence_gate.enabled",
            "evidence_gate.backend",
            "judge.allow_need_more_evidence",
            "budgets.max_evidence_rounds",
            "budgets.max_views_per_round",
            "budgets.max_total_images",
            "budgets.max_camera_actions",
            "budgets.max_selector_calls",
            "require_evidence_gate_after_render",
            "on_non_camera_repairable_evidence",
            "on_budget_exhausted",
            "on_selector_failure",
            "on_render_failure",
        )
    }
    if value is not None:
        _validate_patch(value)
        _deep_update(requested, value, sources=sources, source="config")
    if overrides is not None:
        _validate_patch(overrides)
        _deep_update(
            requested,
            overrides,
            sources=sources,
            source="dependency_injection",
        )

    effective = deepcopy(requested)
    backend = str(effective["camera_selector"]["backend"]).strip().lower()
    if existing_selector_available is not None:
        _boolean(
            existing_selector_available,
            "existing_selector_available",
        )
    if (
        backend == "existing"
        and existing_selector_available is False
        and existing_max_views is None
        and existing_max_steps is None
    ):
        effective["camera_selector"]["backend"] = "deterministic"
        sources["camera_selector.backend"] = (
            "fallback_no_existing_selector"
        )
        backend = "deterministic"
    if backend == "existing":
        if (
            existing_max_views is not None
            and sources["budgets.max_views_per_round"] == "default"
        ):
            effective["budgets"]["max_views_per_round"] = _positive_int(
                existing_max_views,
                "existing max_views",
            )
            sources["budgets.max_views_per_round"] = "existing_camera_provider"
        if existing_max_steps is not None:
            steps = _nonnegative_int(
                existing_max_steps,
                "existing max_steps",
            )
            if sources["budgets.max_camera_actions"] == "default":
                effective["budgets"]["max_camera_actions"] = steps
                sources["budgets.max_camera_actions"] = (
                    "existing_camera_provider"
                )
            if sources["budgets.max_selector_calls"] == "default":
                effective["budgets"]["max_selector_calls"] = steps + 1
                sources["budgets.max_selector_calls"] = (
                    "existing_camera_provider"
                )
    if judge_max_images is not None:
        capacity = _positive_int(judge_max_images, "judge max_images")
        requested_total = _positive_int(
            effective["budgets"]["max_total_images"],
            "max_total_images",
        )
        effective["budgets"]["max_total_images"] = min(
            requested_total,
            capacity,
        )
        if capacity < requested_total:
            sources["budgets.max_total_images"] = "judge_capacity"

    _validate_resolved(effective)
    return _from_mapping(
        effective,
        requested=requested,
        sources=sources,
    )


def _from_mapping(
    value: dict[str, Any],
    *,
    requested: dict[str, Any],
    sources: dict[str, str],
) -> VLMEvaluationControl:
    selector = value["camera_selector"]
    gate = value["evidence_gate"]
    judge = value["judge"]
    budgets = value["budgets"]
    return VLMEvaluationControl(
        schema_version=VLM_EVALUATION_CONTROL_VERSION,
        camera_selector_backend=str(selector["backend"]),
        allow_freeform_pose=bool(selector["allow_freeform_pose"]),
        allow_scene_mutation=bool(selector["allow_scene_mutation"]),
        evidence_gate_enabled=bool(gate["enabled"]),
        evidence_gate_backend=str(gate["backend"]),
        judge_allow_need_more_evidence=bool(
            judge["allow_need_more_evidence"]
        ),
        max_evidence_rounds=int(budgets["max_evidence_rounds"]),
        max_views_per_round=int(budgets["max_views_per_round"]),
        max_total_images=int(budgets["max_total_images"]),
        max_camera_actions=int(budgets["max_camera_actions"]),
        max_selector_calls=int(budgets["max_selector_calls"]),
        require_evidence_gate_after_render=bool(
            value["require_evidence_gate_after_render"]
        ),
        on_non_camera_repairable_evidence=str(
            value["on_non_camera_repairable_evidence"]
        ),
        on_budget_exhausted=str(value["on_budget_exhausted"]),
        on_selector_failure=str(value["on_selector_failure"]),
        on_render_failure=str(value["on_render_failure"]),
        requested=deepcopy(requested),
        sources=deepcopy(sources),
    )


def _validate_patch(value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise TypeError("VLM evaluation control patch must be a JSON object")
    allowed = set(DEFAULT_VLM_EVALUATION_CONTROL)
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"unknown VLM evaluation control fields: {sorted(unknown)}"
        )
    nested_allowed = {
        "camera_selector": set(
            DEFAULT_VLM_EVALUATION_CONTROL["camera_selector"]
        ),
        "evidence_gate": set(DEFAULT_VLM_EVALUATION_CONTROL["evidence_gate"]),
        "judge": set(DEFAULT_VLM_EVALUATION_CONTROL["judge"]),
        "budgets": set(DEFAULT_VLM_EVALUATION_CONTROL["budgets"]),
    }
    for key, allowed_keys in nested_allowed.items():
        if key not in value:
            continue
        nested = value[key]
        if not isinstance(nested, dict):
            raise TypeError(f"{key} control must be a JSON object")
        nested_unknown = set(nested) - allowed_keys
        if nested_unknown:
            raise ValueError(
                f"unknown {key} control fields: {sorted(nested_unknown)}"
            )


def _validate_resolved(value: dict[str, Any]) -> None:
    if value.get("schema_version") != VLM_EVALUATION_CONTROL_VERSION:
        raise ValueError(
            "VLM evaluation control schema_version must be "
            f"{VLM_EVALUATION_CONTROL_VERSION!r}"
        )
    selector = value["camera_selector"]
    if str(selector["backend"]) not in {
        "existing",
        "deterministic",
        "vlm",
        "hybrid",
    }:
        raise ValueError(
            "camera_selector.backend must be existing, deterministic, vlm, or hybrid"
        )
    _boolean(selector["allow_freeform_pose"], "allow_freeform_pose")
    _boolean(selector["allow_scene_mutation"], "allow_scene_mutation")
    gate = value["evidence_gate"]
    _boolean(gate["enabled"], "evidence_gate.enabled")
    if str(gate["backend"]) != "deterministic":
        raise ValueError("evidence_gate.backend must be deterministic")
    _boolean(
        value["judge"]["allow_need_more_evidence"],
        "judge.allow_need_more_evidence",
    )
    budgets = value["budgets"]
    _nonnegative_int(
        budgets["max_evidence_rounds"],
        "max_evidence_rounds",
    )
    _positive_int(budgets["max_views_per_round"], "max_views_per_round")
    _positive_int(budgets["max_total_images"], "max_total_images")
    _nonnegative_int(budgets["max_camera_actions"], "max_camera_actions")
    _positive_int(budgets["max_selector_calls"], "max_selector_calls")
    _boolean(
        value["require_evidence_gate_after_render"],
        "require_evidence_gate_after_render",
    )
    for key, allowed in _POLICY_VALUES.items():
        if value[key] not in allowed:
            raise ValueError(f"{key} must be one of {sorted(allowed)}")


def _deep_update(
    target: dict[str, Any],
    patch: dict[str, Any],
    *,
    sources: dict[str, str],
    source: str,
    prefix: str = "",
) -> None:
    for key, value in patch.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(
                target[key],
                value,
                sources=sources,
                source=source,
                prefix=path,
            )
        else:
            target[key] = deepcopy(value)
            if path != "schema_version":
                sources[path] = source


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value
