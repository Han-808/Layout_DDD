from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from typing import Any, Callable

from benchmark.visual_judge.contracts import (
    validate_camera_selection_response,
)
from benchmark.visual_judge.camera_dsl import (
    CAMERA_CONSTRAINT_REFERENCES,
)
from benchmark.visual_judge.interfaces.camera import (
    CameraSelectionRequest,
    CameraSelectionResult,
    CameraSelector,
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
        previously_attempted = {
            str(value)
            for value in request.context.get("attempted_view_ids", [])
            if str(value).strip()
        }
        unattempted = [
            item
            for item in request.candidate_views
            if str(item.get("id") or "") not in previously_attempted
        ]
        selected = list(unattempted[:limit])
        if not selected:
            available_ids = [
                str(item.get("id") or "")
                for item in request.candidate_views
            ]
            return camera_selection_result_from_value(
                {
                    "outcome": "no_feasible_candidate",
                    "attempted_candidate_ids": available_ids,
                    "rejected_candidates": [
                        {
                            "candidate_id": candidate_id,
                            "reason_codes": ["candidate_already_attempted"],
                        }
                        for candidate_id in available_ids
                    ],
                    "reason_codes": [
                        (
                            "candidate_ranking_exhausted"
                            if available_ids
                            else "candidate_bank_empty"
                        )
                    ],
                    "reason": (
                        "all trusted candidates were already attempted"
                        if available_ids
                        else "the trusted candidate bank is empty"
                    ),
                    "provenance": {
                        "strategy": "stable_input_order",
                        "test_fallback": True,
                    },
                },
                request=request,
                backend="deterministic",
            )
        return camera_selection_result_from_value(
            {
                "outcome": "selected",
                "selected_view_ids": [str(item.get("id") or "") for item in selected],
                "selected_views": selected,
                "attempted_candidate_ids": [
                    str(item.get("id") or "") for item in selected
                ],
                "reason": "stable candidate order",
                "provenance": {
                    "strategy": "stable_input_order",
                    "test_fallback": True,
                },
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
            outcome=result.outcome,
            selected_view_ids=result.selected_view_ids,
            selected_views=result.selected_views,
            camera_proposal=deepcopy(result.camera_proposal),
            camera_actions=result.camera_actions,
            attempted_candidate_ids=result.attempted_candidate_ids,
            rejected_candidates=result.rejected_candidates,
            reason_codes=result.reason_codes,
            attempted_plan_ids=result.attempted_plan_ids,
            selected_plan_id=result.selected_plan_id,
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
    if _contains_metric_decision(raw):
        raise ValueError(
            "CameraSelector response must not contain metric verdict or score"
        )
    if _contains_scene_mutation(raw):
        raise ValueError("CameraSelector response must not contain scene mutation")
    outcome = str(raw.get("outcome") or "selected").strip()
    if outcome not in {"selected", "no_feasible_candidate"}:
        raise ValueError(
            "CameraSelector outcome must be selected or "
            "no_feasible_candidate"
        )

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
    attempted_ids = _validated_candidate_id_list(
        raw.get("attempted_candidate_ids"),
        available_ids=available_ids,
        label="attempted_candidate_ids",
    )
    rejected_candidates = _validated_rejected_candidates(
        raw.get("rejected_candidates"),
        available_ids=set(available_ids),
        attempted_ids=set(attempted_ids),
        active_constraints=_active_constraint_references(
            request.constraints
        ),
    )
    reason_codes = _validated_reason_codes(raw.get("reason_codes"))
    attempted_plan_ids = _validated_plan_id_list(
        raw.get("attempted_plan_ids"),
        request=request,
        label="attempted_plan_ids",
    )
    selected_plan_id = _validated_selected_plan_id(
        raw.get("selected_plan_id"),
        request=request,
    )
    reason = str(raw.get("reason") or "").strip()
    if not reason:
        raise ValueError("CameraSelector response must include a reason")

    if outcome == "no_feasible_candidate":
        forbidden_selection = any(
            raw.get(key)
            for key in (
                "selected_view_ids",
                "selected_views",
                "camera_proposal",
                "camera_actions",
                "action",
                "selected_plan_id",
            )
        )
        if forbidden_selection:
            raise ValueError(
                "no_feasible_candidate must not contain a selected view, "
                "plan, proposal, or camera action"
            )
        if available_ids and not attempted_ids:
            raise ValueError(
                "no_feasible_candidate must record attempted_candidate_ids"
            )
        if attempted_ids and not rejected_candidates:
            raise ValueError(
                "no_feasible_candidate must record rejected_candidates"
            )
        if not reason_codes:
            raise ValueError(
                "no_feasible_candidate must include reason_codes"
            )
        provenance = _selection_provenance(raw)
        return CameraSelectionResult(
            outcome=outcome,
            attempted_candidate_ids=tuple(attempted_ids),
            rejected_candidates=tuple(
                deepcopy(rejected_candidates)
            ),
            reason_codes=tuple(reason_codes),
            attempted_plan_ids=tuple(attempted_plan_ids),
            reason=reason,
            backend=str(backend),
            evidence_round=_nonnegative_int(
                request.evidence_round,
                "CameraSelector evidence_round",
            ),
            provenance=provenance,
        )

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
            if not (
                typed_result
                and _has_independent_pose_validation(
                    raw,
                    proposal=proposal,
                )
            ):
                pose_validator = request.context.get("pose_validator")
                if not callable(pose_validator):
                    raise ValueError(
                        "free-form CameraSelector proposal requires an "
                        "injected pose validator"
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
    provenance = _selection_provenance(raw)
    return CameraSelectionResult(
        outcome=outcome,
        selected_view_ids=tuple(selected_ids),
        selected_views=tuple(deepcopy(selected_views)),
        camera_proposal=resolved_proposal,
        camera_actions=tuple(resolved_actions),
        attempted_candidate_ids=tuple(
            attempted_ids or selected_ids
        ),
        rejected_candidates=tuple(deepcopy(rejected_candidates)),
        reason_codes=tuple(reason_codes),
        attempted_plan_ids=tuple(attempted_plan_ids),
        selected_plan_id=selected_plan_id,
        reason=reason,
        backend=str(backend),
        evidence_round=_nonnegative_int(
            request.evidence_round,
            "CameraSelector evidence_round",
        ),
        provenance=deepcopy(provenance),
    )


def _has_independent_pose_validation(
    value: dict[str, Any],
    *,
    proposal: dict[str, Any],
) -> bool:
    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        return False
    validation = provenance.get("pose_validation")
    if not isinstance(validation, dict) or validation.get("valid") is not True:
        return False
    checks = validation.get("checks")
    required = {
        "camera_scene_boundary_feasibility",
        "frustum_validation",
        "collision_avoidance",
        "target_visibility_prediction",
        "pose_diversity_validation",
    }
    if not isinstance(checks, dict) or any(
        checks.get(name) is not True for name in required
    ):
        return False
    validated_pose = validation.get("pose")
    return isinstance(validated_pose, dict) and validated_pose == proposal


def _validated_candidate_id_list(
    value: Any,
    *,
    available_ids: list[str],
    label: str,
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"CameraSelector {label} must be a JSON list")
    result = [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]
    if len(result) != len(value) or len(result) != len(set(result)):
        raise ValueError(
            f"CameraSelector {label} must contain unique non-empty IDs"
        )
    unknown = [item for item in result if item not in available_ids]
    if unknown:
        raise ValueError(
            f"CameraSelector {label} references unknown candidates: "
            f"{unknown}"
        )
    return result


def _validated_rejected_candidates(
    value: Any,
    *,
    available_ids: set[str],
    attempted_ids: set[str],
    active_constraints: set[str],
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, dict) for item in value
    ):
        raise ValueError(
            "CameraSelector rejected_candidates must be a JSON list"
        )
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        candidate_id = str(
            item.get("candidate_id") or item.get("id") or ""
        ).strip()
        if (
            not candidate_id
            or candidate_id not in available_ids
            or candidate_id in seen
        ):
            raise ValueError(
                "CameraSelector rejected candidate must reference one "
                "unique trusted candidate"
            )
        if attempted_ids and candidate_id not in attempted_ids:
            raise ValueError(
                "CameraSelector rejected candidate was not attempted"
            )
        codes = _validated_reason_codes(item.get("reason_codes"))
        if not codes:
            raise ValueError(
                "CameraSelector rejected candidate requires reason_codes"
            )
        normalized = deepcopy(item)
        normalized["candidate_id"] = candidate_id
        normalized["reason_codes"] = codes
        failed = item.get("failed_constraints")
        if failed is not None:
            if not isinstance(failed, (list, tuple)):
                raise ValueError(
                    "CameraSelector rejected failed_constraints must be "
                    "a JSON list"
                )
            failed_values = [
                str(value).strip()
                for value in failed
                if str(value).strip()
            ]
            if (
                len(failed_values) != len(failed)
                or len(failed_values) != len(set(failed_values))
            ):
                raise ValueError(
                    "CameraSelector rejected failed_constraints must contain "
                    "unique non-empty references"
                )
            unknown = set(failed_values) - CAMERA_CONSTRAINT_REFERENCES
            inactive = set(failed_values) - active_constraints
            if unknown or inactive:
                raise ValueError(
                    "CameraSelector rejected failed_constraints must "
                    "reference active Camera DSL constraints"
                )
            normalized["failed_constraints"] = failed_values
        result.append(normalized)
        seen.add(candidate_id)
    return result


def _active_constraint_references(
    value: dict[str, Any],
) -> set[str]:
    if not isinstance(value, dict):
        raise ValueError("CameraSelector camera constraints must be a mapping")
    active: set[str] = set()
    for key in ("required_observations", "preserved_observations"):
        observations = value.get(key, ())
        if observations is None:
            observations = ()
        if not isinstance(observations, (list, tuple)):
            raise ValueError(
                f"CameraSelector {key} must be a JSON list"
            )
        active.update(
            str(observation).strip()
            for observation in observations
            if str(observation).strip()
        )
    if value.get("min_projected_coverage") is not None:
        active.add("min_projected_coverage")
    if value.get("require_joint_visibility") is True:
        active.add("require_joint_visibility")
    if value.get("require_global_anchor") is True:
        active.add("require_global_anchor")
    return active


def _validated_reason_codes(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("CameraSelector reason_codes must be a JSON list")
    result = [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]
    if len(result) != len(value):
        raise ValueError(
            "CameraSelector reason_codes must be non-empty strings"
        )
    return list(dict.fromkeys(result))


def _validated_selected_plan_id(
    value: Any,
    *,
    request: CameraSelectionRequest,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "CameraSelector selected_plan_id must be a non-empty string"
        )
    plan_id = value.strip()
    trusted_ids = _trusted_plan_ids(request)
    if plan_id not in trusted_ids:
        raise ValueError(
            "CameraSelector selected_plan_id is not a trusted repair plan"
        )
    return plan_id


def _validated_plan_id_list(
    value: Any,
    *,
    request: CameraSelectionRequest,
    label: str,
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"CameraSelector {label} must be a JSON list")
    result = [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]
    if len(result) != len(value) or len(result) != len(set(result)):
        raise ValueError(
            f"CameraSelector {label} must contain unique non-empty IDs"
        )
    if not result:
        return []
    unknown = set(result) - _trusted_plan_ids(request)
    if unknown:
        raise ValueError(
            f"CameraSelector {label} references untrusted repair plans"
        )
    return result


def _trusted_plan_ids(
    request: CameraSelectionRequest,
) -> set[str]:
    plans = request.context.get("camera_repair_plans")
    if not isinstance(plans, (list, tuple)):
        raise ValueError(
            "CameraSelector plan IDs require trusted repair plans"
        )
    return {
        str(item.get("plan_id") or "")
        for item in plans
        if isinstance(item, dict)
        and str(item.get("plan_id") or "").strip()
    }


def _selection_provenance(raw: dict[str, Any]) -> dict[str, Any]:
    provenance = raw.get("provenance")
    if provenance is None:
        provenance = {}
    if not isinstance(provenance, dict):
        raise ValueError("CameraSelector provenance must be a JSON object")
    provenance = _validated_json_value(
        provenance,
        path="CameraSelector provenance",
    )
    for key in (
        "candidate_count",
        "filtered_candidate_count",
        "vlm_call_count",
    ):
        value = provenance.get(key)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise ValueError(
                f"CameraSelector provenance.{key} must be a "
                "non-negative integer"
            )
    generation_time = provenance.get(
        "candidate_generation_time_seconds"
    )
    if generation_time is not None and (
        isinstance(generation_time, bool)
        or not isinstance(generation_time, (int, float))
        or not math.isfinite(float(generation_time))
        or float(generation_time) < 0.0
    ):
        raise ValueError(
            "CameraSelector provenance.candidate_generation_time_seconds "
            "must be finite and non-negative"
        )
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
    return _validated_json_value(
        provenance,
        path="CameraSelector provenance",
    )


def _validated_json_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite numbers")
        return value
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} keys must be strings")
            result[key] = _validated_json_value(
                item,
                path=f"{path}.{key}",
            )
        return result
    if isinstance(value, (list, tuple)):
        return [
            _validated_json_value(
                item,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
    raise ValueError(
        f"{path} must contain only JSON-compatible values"
    )


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
            str(key).strip().lower()
            in {
                "allow_scene_mutation",
                "scene_mutated",
                "scene_mutation",
                "scene_patch",
            }
            for key in value
        ):
            return True
        return any(_contains_scene_mutation(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_scene_mutation(item) for item in value)
    return False


def _contains_metric_decision(value: Any) -> bool:
    if isinstance(value, dict):
        if {"verdict", "score"} & {
            str(key).strip().lower() for key in value
        }:
            return True
        return any(
            _contains_metric_decision(item)
            for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_metric_decision(item) for item in value)
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
        outcome=result.outcome,
        selected_view_ids=result.selected_view_ids,
        selected_views=result.selected_views,
        camera_proposal=deepcopy(result.camera_proposal),
        camera_actions=result.camera_actions,
        attempted_candidate_ids=result.attempted_candidate_ids,
        rejected_candidates=result.rejected_candidates,
        reason_codes=result.reason_codes,
        attempted_plan_ids=result.attempted_plan_ids,
        selected_plan_id=result.selected_plan_id,
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


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value

_EXISTING_PROVIDER_CANDIDATE_ID = "existing_provider_acquisition"


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
        full_artifacts_per_view = (
            _provider_full_artifacts_per_selected_view(
                self.provider,
                metric=request.metric,
                request=request,
            )
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
                "full_artifacts_per_selected_view": (
                    full_artifacts_per_view
                ),
            },
        )


def _provider_policy(provider: Any) -> dict[str, Any] | None:
    value = getattr(provider, "policy_config", None)
    return deepcopy(value) if isinstance(value, dict) else None


def _provider_full_artifacts_per_selected_view(
    provider: Any,
    *,
    metric: str,
    request: CameraSelectionRequest,
) -> int:
    estimate = getattr(
        provider,
        "max_full_artifacts_for_controller_request",
        None,
    )
    if callable(estimate):
        value = estimate(
            {
                "task": request.task,
                "metric": request.metric,
                "camera_constraints": deepcopy(request.constraints),
                "budget": deepcopy(request.budget),
                "context": deepcopy(request.context),
            }
        )
    elif str(metric) == "functional_consistency":
        value = getattr(
            provider,
            "functional_probe_full_artifacts_per_selected_view",
            1,
        )
    else:
        value = 1
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(
            "provider functional full-artifact reservation must be a "
            "positive integer"
        )
    return value


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
    acquired_artifact_paths = usage.get(
        "acquired_artifact_paths"
    )
    if acquired_artifact_paths is not None and (
        not isinstance(acquired_artifact_paths, list)
        or not all(
            isinstance(item, str) and item.strip()
            for item in acquired_artifact_paths
        )
    ):
        raise RuntimeError(
            "camera provider last_call_usage.acquired_artifact_paths "
            "must be a list of non-empty strings"
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


def _qualified_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


EXISTING_PROVIDER_CANDIDATE_ID = _EXISTING_PROVIDER_CANDIDATE_ID
ExistingProviderCameraSelector = _ExistingProviderCameraSelector
bind_provider_selector_backend = _bind_provider_selector_backend
provider_observed_usage = _provider_observed_usage
provider_policy = _provider_policy
qualified_name = _qualified_name
