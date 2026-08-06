from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from benchmark.visual_judge.camera_ranking import (
    DEFAULT_DETERMINISTIC_CAMERA_RANKING,
    DeterministicCameraRankingConfig,
)


VLM_EVALUATION_CONTROL_VERSION = "vlm_evaluation_control_v2"

DEFAULT_VLM_EVALUATION_CONTROL: dict[str, Any] = {
    "schema_version": VLM_EVALUATION_CONTROL_VERSION,
    "camera_selector": {
        "backend": "existing",
        "allow_freeform_pose": False,
        "allow_scene_mutation": False,
    },
    "initial_group_camera": {
        "mode": "visibility_ranked",
        "selector": "deterministic",
    },
    "camera_acquisition": {
        "policy": "deterministic_then_vlm",
        "deterministic": {
            "max_rounds": 1,
            "candidate_budget": 8,
            "max_selected_views": 2,
            "ranking": deepcopy(DEFAULT_DETERMINISTIC_CAMERA_RANKING),
        },
        "vlm": {
            "max_rounds": 1,
            "selection_mode": "repair_plan",
            "max_selected_views": 2,
            "max_repair_plans": 3,
        },
        # These mirror the existing budgets. The resolver keeps both paths
        # synchronized so old configuration patches remain authoritative.
        "total": {
            "max_evidence_rounds": 2,
            "max_total_images": 6,
            "max_selector_calls": 3,
            "max_camera_actions": 2,
        },
        "escalation": {
            "on_no_feasible_candidate": True,
            # Retained in the additive config shape, but frozen off: evidence
            # sufficiency is now a Judge decision rather than a Gate signal.
            "on_post_render_gate_insufficient": False,
            "on_selector_exception": False,
            "on_render_failure": False,
        },
    },
    "evidence_gate": {
        "enabled": True,
        "backend": "deterministic",
        "allow_path_only_compatibility": False,
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
    "on_budget_exhausted": "force_choice",
    "on_selector_failure": "keep_previous_evidence",
    "on_render_failure": "unresolved",
}

_POLICY_VALUES = {
    "on_non_camera_repairable_evidence": {"unresolved"},
    "on_budget_exhausted": {"force_choice"},
    "on_selector_failure": {"keep_previous_evidence", "unresolved"},
    "on_render_failure": {"unresolved"},
}


@dataclass(frozen=True)
class VLMEvaluationControl:
    schema_version: str
    camera_selector_backend: str
    allow_freeform_pose: bool
    allow_scene_mutation: bool
    initial_group_camera_mode: str
    initial_group_camera_selector: str
    camera_acquisition_policy: str
    deterministic_max_rounds: int
    deterministic_candidate_budget: int
    deterministic_max_selected_views: int
    deterministic_ranking: dict[str, float]
    vlm_max_rounds: int
    vlm_selection_mode: str
    vlm_max_selected_views: int
    vlm_max_repair_plans: int
    escalate_on_no_feasible_candidate: bool
    escalate_on_post_render_gate_insufficient: bool
    escalate_on_selector_exception: bool
    escalate_on_render_failure: bool
    evidence_gate_enabled: bool
    evidence_gate_backend: str
    evidence_gate_allow_path_only_compatibility: bool
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
            "initial_group_camera": {
                "mode": self.initial_group_camera_mode,
                "selector": self.initial_group_camera_selector,
            },
            "camera_acquisition": {
                "policy": self.camera_acquisition_policy,
                "deterministic": {
                    "max_rounds": self.deterministic_max_rounds,
                    "candidate_budget": (
                        self.deterministic_candidate_budget
                    ),
                    "max_selected_views": (
                        self.deterministic_max_selected_views
                    ),
                    "ranking": deepcopy(self.deterministic_ranking),
                },
                "vlm": {
                    "max_rounds": self.vlm_max_rounds,
                    "selection_mode": self.vlm_selection_mode,
                    "max_selected_views": self.vlm_max_selected_views,
                    "max_repair_plans": self.vlm_max_repair_plans,
                },
                "total": {
                    "max_evidence_rounds": self.max_evidence_rounds,
                    "max_total_images": self.max_total_images,
                    "max_selector_calls": self.max_selector_calls,
                    "max_camera_actions": self.max_camera_actions,
                },
                "escalation": {
                    "on_no_feasible_candidate": (
                        self.escalate_on_no_feasible_candidate
                    ),
                    "on_post_render_gate_insufficient": (
                        self.escalate_on_post_render_gate_insufficient
                    ),
                    "on_selector_exception": (
                        self.escalate_on_selector_exception
                    ),
                    "on_render_failure": self.escalate_on_render_failure,
                },
            },
            "evidence_gate": {
                "enabled": self.evidence_gate_enabled,
                "backend": self.evidence_gate_backend,
                "allow_path_only_compatibility": (
                    self.evidence_gate_allow_path_only_compatibility
                ),
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
            "initial_group_camera.mode",
            "initial_group_camera.selector",
            "camera_acquisition.policy",
            "camera_acquisition.deterministic.max_rounds",
            "camera_acquisition.deterministic.candidate_budget",
            "camera_acquisition.deterministic.max_selected_views",
            *(
                "camera_acquisition.deterministic.ranking." + key
                for key in DEFAULT_DETERMINISTIC_CAMERA_RANKING
            ),
            "camera_acquisition.vlm.max_rounds",
            "camera_acquisition.vlm.selection_mode",
            "camera_acquisition.vlm.max_selected_views",
            "camera_acquisition.vlm.max_repair_plans",
            "camera_acquisition.total.max_evidence_rounds",
            "camera_acquisition.total.max_total_images",
            "camera_acquisition.total.max_selector_calls",
            "camera_acquisition.total.max_camera_actions",
            "camera_acquisition.escalation.on_no_feasible_candidate",
            (
                "camera_acquisition.escalation."
                "on_post_render_gate_insufficient"
            ),
            "camera_acquisition.escalation.on_selector_exception",
            "camera_acquisition.escalation.on_render_failure",
            "evidence_gate.enabled",
            "evidence_gate.backend",
            "evidence_gate.allow_path_only_compatibility",
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
    _synchronize_total_budgets(effective, sources)
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
    # Judge packet capacity and acquisition capacity are independent. The
    # former constrains one request; it must not silently shrink the shared
    # Controller image ledger across groups or repair episodes.
    if judge_max_images is not None:
        _positive_int(judge_max_images, "judge max_images")
    _mirror_effective_budgets_to_acquisition(effective, sources)

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
    initial_group_camera = value["initial_group_camera"]
    acquisition = value["camera_acquisition"]
    deterministic = acquisition["deterministic"]
    vlm = acquisition["vlm"]
    escalation = acquisition["escalation"]
    gate = value["evidence_gate"]
    judge = value["judge"]
    budgets = value["budgets"]
    return VLMEvaluationControl(
        schema_version=VLM_EVALUATION_CONTROL_VERSION,
        camera_selector_backend=str(selector["backend"]),
        allow_freeform_pose=bool(selector["allow_freeform_pose"]),
        allow_scene_mutation=bool(selector["allow_scene_mutation"]),
        initial_group_camera_mode=str(
            initial_group_camera["mode"]
        ),
        initial_group_camera_selector=str(
            initial_group_camera["selector"]
        ),
        camera_acquisition_policy=str(acquisition["policy"]),
        deterministic_max_rounds=int(deterministic["max_rounds"]),
        deterministic_candidate_budget=int(
            deterministic["candidate_budget"]
        ),
        deterministic_max_selected_views=int(
            deterministic["max_selected_views"]
        ),
        deterministic_ranking=(
            DeterministicCameraRankingConfig.from_value(
                deterministic["ranking"]
            ).to_dict()
        ),
        vlm_max_rounds=int(vlm["max_rounds"]),
        vlm_selection_mode=str(vlm["selection_mode"]),
        vlm_max_selected_views=int(vlm["max_selected_views"]),
        vlm_max_repair_plans=int(vlm["max_repair_plans"]),
        escalate_on_no_feasible_candidate=bool(
            escalation["on_no_feasible_candidate"]
        ),
        escalate_on_post_render_gate_insufficient=bool(
            escalation["on_post_render_gate_insufficient"]
        ),
        escalate_on_selector_exception=bool(
            escalation["on_selector_exception"]
        ),
        escalate_on_render_failure=bool(
            escalation["on_render_failure"]
        ),
        evidence_gate_enabled=bool(gate["enabled"]),
        evidence_gate_backend=str(gate["backend"]),
        evidence_gate_allow_path_only_compatibility=bool(
            gate["allow_path_only_compatibility"]
        ),
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
        "initial_group_camera": set(
            DEFAULT_VLM_EVALUATION_CONTROL["initial_group_camera"]
        ),
        "camera_acquisition": set(
            DEFAULT_VLM_EVALUATION_CONTROL["camera_acquisition"]
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
    acquisition = value.get("camera_acquisition")
    if isinstance(acquisition, dict):
        _validate_nested_patch(
            acquisition,
            "deterministic",
            DEFAULT_VLM_EVALUATION_CONTROL["camera_acquisition"][
                "deterministic"
            ],
        )
        deterministic = acquisition.get("deterministic")
        if isinstance(deterministic, dict):
            _validate_nested_patch(
                deterministic,
                "ranking",
                DEFAULT_VLM_EVALUATION_CONTROL[
                    "camera_acquisition"
                ]["deterministic"]["ranking"],
            )
        _validate_nested_patch(
            acquisition,
            "vlm",
            DEFAULT_VLM_EVALUATION_CONTROL["camera_acquisition"]["vlm"],
        )
        _validate_nested_patch(
            acquisition,
            "total",
            DEFAULT_VLM_EVALUATION_CONTROL["camera_acquisition"]["total"],
        )
        _validate_nested_patch(
            acquisition,
            "escalation",
            DEFAULT_VLM_EVALUATION_CONTROL["camera_acquisition"][
                "escalation"
            ],
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
    if selector["allow_scene_mutation"] is not False:
        raise ValueError(
            "camera_selector.allow_scene_mutation cannot be enabled; "
            "CameraSelector and evidence renderer scene access is read-only"
        )
    initial = value["initial_group_camera"]
    if str(initial["mode"]) != "visibility_ranked":
        raise ValueError(
            "official camera-policy evaluation requires "
            "initial_group_camera.mode=visibility_ranked"
        )
    if str(initial["selector"]) != "deterministic":
        raise ValueError(
            "official camera-policy evaluation requires "
            "initial_group_camera.selector=deterministic"
        )
    acquisition = value["camera_acquisition"]
    if str(acquisition["policy"]) not in {
        "fixed",
        "deterministic_only",
        "vlm_only",
        "deterministic_then_vlm",
    }:
        raise ValueError(
            "camera_acquisition.policy must be fixed, deterministic_only, "
            "vlm_only, or deterministic_then_vlm"
        )
    deterministic = acquisition["deterministic"]
    _nonnegative_int(
        deterministic["max_rounds"],
        "camera_acquisition.deterministic.max_rounds",
    )
    _positive_int(
        deterministic["candidate_budget"],
        "camera_acquisition.deterministic.candidate_budget",
    )
    _positive_int(
        deterministic["max_selected_views"],
        "camera_acquisition.deterministic.max_selected_views",
    )
    DeterministicCameraRankingConfig.from_value(
        deterministic["ranking"]
    )
    vlm = acquisition["vlm"]
    _nonnegative_int(
        vlm["max_rounds"],
        "camera_acquisition.vlm.max_rounds",
    )
    if str(vlm["selection_mode"]) not in {
        "candidate_only",
        "repair_plan",
        "freeform_pose",
    }:
        raise ValueError(
            "camera_acquisition.vlm.selection_mode must be candidate_only, "
            "repair_plan, or freeform_pose"
        )
    _positive_int(
        vlm["max_selected_views"],
        "camera_acquisition.vlm.max_selected_views",
    )
    _positive_int(
        vlm["max_repair_plans"],
        "camera_acquisition.vlm.max_repair_plans",
    )
    for key, enabled in acquisition["escalation"].items():
        _boolean(enabled, f"camera_acquisition.escalation.{key}")
    for key in ("on_selector_exception", "on_render_failure"):
        if acquisition["escalation"][key] is not False:
            raise ValueError(
                f"camera_acquisition.escalation.{key} cannot be enabled; "
                "engineering failures are not normal VLM escalation signals"
            )
    if (
        acquisition["escalation"]["on_post_render_gate_insufficient"]
        is not False
    ):
        raise ValueError(
            "camera_acquisition.escalation."
            "on_post_render_gate_insufficient cannot be enabled; "
            "metric evidence sufficiency belongs to Judge"
        )
    gate = value["evidence_gate"]
    _boolean(gate["enabled"], "evidence_gate.enabled")
    if gate["enabled"] is not True:
        raise ValueError(
            "evidence_gate.enabled cannot be disabled; every evidence packet "
            "must pass input-integrity validation before Judge"
        )
    if str(gate["backend"]) != "deterministic":
        raise ValueError("evidence_gate.backend must be deterministic")
    _boolean(
        gate["allow_path_only_compatibility"],
        "evidence_gate.allow_path_only_compatibility",
    )
    if gate["allow_path_only_compatibility"] is not False:
        raise ValueError(
            "evidence_gate.allow_path_only_compatibility cannot be enabled; "
            "EvidenceGate input-integrity checks cannot be bypassed"
        )
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
    if value["require_evidence_gate_after_render"] is not True:
        raise ValueError(
            "require_evidence_gate_after_render cannot be disabled"
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


def _validate_nested_patch(
    parent: dict[str, Any],
    key: str,
    defaults: dict[str, Any],
) -> None:
    if key not in parent:
        return
    value = parent[key]
    if not isinstance(value, dict):
        raise TypeError(f"camera_acquisition.{key} must be a JSON object")
    unknown = set(value) - set(defaults)
    if unknown:
        raise ValueError(
            "unknown camera_acquisition."
            f"{key} fields: {sorted(unknown)}"
        )


def _synchronize_total_budgets(
    value: dict[str, Any],
    sources: dict[str, str],
) -> None:
    total = value["camera_acquisition"]["total"]
    budgets = value["budgets"]
    mapping = {
        "max_evidence_rounds": "max_evidence_rounds",
        "max_total_images": "max_total_images",
        "max_selector_calls": "max_selector_calls",
        "max_camera_actions": "max_camera_actions",
    }
    for total_key, budget_key in mapping.items():
        total_path = f"camera_acquisition.total.{total_key}"
        budget_path = f"budgets.{budget_key}"
        total_source = sources[total_path]
        budget_source = sources[budget_path]
        if total_source != "default":
            budgets[budget_key] = deepcopy(total[total_key])
            sources[budget_path] = total_source
        elif budget_source != "default":
            total[total_key] = deepcopy(budgets[budget_key])
            sources[total_path] = f"legacy_{budget_source}"


def _mirror_effective_budgets_to_acquisition(
    value: dict[str, Any],
    sources: dict[str, str],
) -> None:
    total = value["camera_acquisition"]["total"]
    budgets = value["budgets"]
    for key in (
        "max_evidence_rounds",
        "max_total_images",
        "max_selector_calls",
        "max_camera_actions",
    ):
        total[key] = deepcopy(budgets[key])
        if sources[f"budgets.{key}"] != "default":
            sources[f"camera_acquisition.total.{key}"] = (
                sources[f"budgets.{key}"]
            )


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
