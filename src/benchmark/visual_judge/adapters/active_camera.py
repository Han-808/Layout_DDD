from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from typing import Any, Callable, Mapping, Protocol

from benchmark.visual_judge.adapters.legacy_camera import (
    camera_selection_result_from_value,
)
from benchmark.visual_judge.camera_dsl import (
    CameraConstraintSet,
    canonical_camera_metric,
)
from benchmark.visual_judge.camera_repair import (
    DEFAULT_VLM_SELECTION_MODE,
    CameraRepairPlan,
    VLMSelectionMode,
    diagnose_camera_constraint_conflicts,
    generate_camera_repair_plans,
    validate_trusted_repair_plan_selection,
    validate_vlm_selection_mode,
)
from benchmark.visual_judge.camera_targets import (
    merge_authoritative_target_ids,
)
from benchmark.visual_judge.interfaces.camera import (
    CameraSelectionRequest,
    CameraSelectionResult,
)


ACTIVE_VLM_CAMERA_SELECTOR_VERSION = "active_vlm_camera_selector_v1"
REQUIRED_POSE_VALIDATION_CHECKS = frozenset(
    {
        "camera_scene_boundary_feasibility",
        "frustum_validation",
        "collision_avoidance",
        "target_visibility_prediction",
        "pose_diversity_validation",
    }
)

_CANDIDATE_RESPONSE_FIELDS = frozenset(
    {"selected_view_ids", "reason", "provenance"}
)
_PLAN_RESPONSE_FIELDS = frozenset(
    {"selected_plan_id", "plan_id", "reason", "provenance"}
)
_POSE_RESPONSE_FIELDS = frozenset(
    {"camera_proposal", "reason", "provenance"}
)
_POSE_FIELDS = frozenset(
    {"location", "target", "lens_mm", "camera_type", "up"}
)
_POSE_VALIDATION_FIELDS = frozenset(
    {"valid", "checks", "pose", "reason_codes", "provenance"}
)
_MUTATION_FIELDS = frozenset(
    {
        "allow_scene_mutation",
        "mutated_scene",
        "scene_mutation",
        "scene_patch",
        "scene_updates",
    }
)


class CameraRepairSolver(Protocol):
    def realize(
        self,
        request: CameraSelectionRequest,
        plan: CameraRepairPlan,
    ) -> CameraSelectionResult | dict[str, Any]: ...


class CameraPoseValidator(Protocol):
    def validate(
        self,
        proposal: dict[str, Any],
        request: CameraSelectionRequest,
    ) -> Mapping[str, Any]: ...


class ActiveVLMCameraSelector:
    """Execute one VLM camera-selection policy, never a cascade or render."""

    backend = "vlm_active"

    def __init__(
        self,
        selector: Any,
        *,
        selection_mode: VLMSelectionMode = DEFAULT_VLM_SELECTION_MODE,
        repair_solver: CameraRepairSolver | None = None,
        pose_validator: CameraPoseValidator | None = None,
        repair_planner: (
            Callable[
                [CameraConstraintSet, CameraSelectionRequest],
                tuple[CameraRepairPlan, ...],
            ]
            | None
        ) = None,
        allow_freeform_pose: bool = False,
        max_repair_plans: int = 3,
        max_repair_plans_source: str | None = None,
    ) -> None:
        self._call, self.method_name = _selector_call(selector)
        self.selector = selector
        self.selection_mode = validate_vlm_selection_mode(
            selection_mode,
            allow_freeform_pose=allow_freeform_pose,
        )
        self.repair_solver = repair_solver
        self.pose_validator = pose_validator
        self.repair_planner = repair_planner
        self.allow_freeform_pose = bool(allow_freeform_pose)
        if (
            isinstance(max_repair_plans, bool)
            or not isinstance(max_repair_plans, int)
            or max_repair_plans <= 0
        ):
            raise ValueError(
                "ActiveVLMCameraSelector max_repair_plans must be positive"
            )
        self.max_repair_plans = max_repair_plans
        self.max_repair_plans_source = str(
            max_repair_plans_source
            or (
                "dependency_injection"
                if max_repair_plans != 3
                else "default"
            )
        )
        self.last_call_usage: dict[str, int] = {
            "vlm_call_count": 0
        }
        if self.selection_mode == "repair_plan" and not callable(
            getattr(repair_solver, "realize", None)
        ):
            raise TypeError(
                "repair_plan mode requires CameraRepairSolver.realize"
            )
        if self.selection_mode == "freeform_pose" and not callable(
            getattr(pose_validator, "validate", None)
        ):
            raise TypeError(
                "freeform_pose mode requires CameraPoseValidator.validate"
            )

    def select(
        self, request: CameraSelectionRequest
    ) -> CameraSelectionResult:
        self.last_call_usage = {"vlm_call_count": 0}
        if not isinstance(request, CameraSelectionRequest):
            raise TypeError(
                "ActiveVLMCameraSelector requires CameraSelectionRequest"
            )
        constraints = CameraConstraintSet.from_value(
            request.constraints,
            known_target_ids=_known_target_ids(request),
        )
        if constraints.metric != _request_metric(
            request, fallback=constraints.metric
        ):
            raise ValueError(
                "Camera DSL metric conflicts with selector request"
            )
        if constraints.target_ids != request.target_ids:
            raise ValueError(
                "Camera DSL targets conflict with selector request"
            )
        if self.selection_mode == "candidate_only":
            return self._select_candidate(request, constraints)
        if self.selection_mode == "repair_plan":
            return self._select_plan(request, constraints)
        return self._select_pose(request, constraints)

    def _select_candidate(
        self,
        request: CameraSelectionRequest,
        constraints: CameraConstraintSet,
    ) -> CameraSelectionResult:
        candidates = _candidate_index(request.candidate_views)
        attempted = _attempted_ids(request)
        available = tuple(
            candidate
            for candidate_id, candidate in candidates.items()
            if candidate_id not in attempted
        )
        if not available:
            return _no_candidate_result(
                request,
                attempted_ids=tuple(candidates),
                reason_code=(
                    "candidate_ranking_exhausted"
                    if candidates
                    else "candidate_bank_empty"
                ),
            )
        selection_request = replace(
            request,
            candidate_views=available,
        )
        payload = _base_payload(request, constraints)
        payload["selection_mode"] = "candidate_only"
        payload["candidate_views"] = list(deepcopy(available))
        raw = self._invoke(payload)
        response = _strict_response(
            raw, allowed=_CANDIDATE_RESPONSE_FIELDS
        )
        selected_ids = _text_list(
            response.get("selected_view_ids"),
            "selected_view_ids",
        )
        if not selected_ids:
            raise ValueError(
                "candidate_only must select at least one trusted candidate"
            )
        result = camera_selection_result_from_value(
            {
                "outcome": "selected",
                "selected_view_ids": selected_ids,
                "reason": _reason(response),
                "provenance": _vlm_provenance(
                    self,
                    response,
                    selection_mode="candidate_only",
                ),
            },
            request=selection_request,
            backend=self.backend,
        )
        return result

    def _select_plan(
        self,
        request: CameraSelectionRequest,
        constraints: CameraConstraintSet,
    ) -> CameraSelectionResult:
        plans = self._repair_plans(constraints, request)
        if not plans:
            return _no_candidate_result(
                request,
                attempted_ids=tuple(_candidate_index(request.candidate_views)),
                reason_code="no_trusted_repair_plan",
            )
        payload = _base_payload(request, constraints)
        payload["selection_mode"] = "repair_plan"
        payload["trusted_repair_plans"] = [
            plan.to_dict() for plan in plans
        ]
        response = _strict_response(
            self._invoke(payload), allowed=_PLAN_RESPONSE_FIELDS
        )
        selection = validate_trusted_repair_plan_selection(
            response,
            trusted_plans=plans,
            constraints=constraints,
            default_backend=self.backend,
        )
        trusted_context = {
            **deepcopy(request.context),
            "camera_repair_plans": [
                plan.to_dict() for plan in plans
            ],
        }
        solver_request = replace(request, context=trusted_context)
        raw_realization = self.repair_solver.realize(
            solver_request,
            selection.plan,
        )
        raw = (
            raw_realization.to_dict()
            if isinstance(raw_realization, CameraSelectionResult)
            else deepcopy(raw_realization)
        )
        if not isinstance(raw, dict):
            raise ValueError(
                "CameraRepairSolver must return CameraSelectionResult or mapping"
            )
        if raw.get("selected_plan_id") not in {
            None,
            selection.plan.plan_id,
        }:
            raise ValueError(
                "CameraRepairSolver changed the selected trusted plan"
            )
        if str(raw.get("outcome") or "selected") == "selected":
            raw["selected_plan_id"] = selection.plan.plan_id
        result = camera_selection_result_from_value(
            raw,
            request=solver_request,
            backend=self.backend,
        )
        provenance = deepcopy(result.provenance)
        provenance.update(
            {
                "adapter_version": ACTIVE_VLM_CAMERA_SELECTOR_VERSION,
                "selection_mode": "repair_plan",
                "max_repair_plans": self.max_repair_plans,
                "max_repair_plans_source": (
                    self.max_repair_plans_source
                ),
                "selected_plan_id": selection.plan.plan_id,
                "vlm_selection": selection.to_dict(),
                "deterministic_solver": _qualified_name(
                    self.repair_solver
                ),
            }
        )
        return replace(
            result,
            attempted_plan_ids=tuple(
                dict.fromkeys(
                    (
                        *result.attempted_plan_ids,
                        selection.plan.plan_id,
                    )
                )
            ),
            selected_plan_id=(
                selection.plan.plan_id
                if result.outcome == "selected"
                else None
            ),
            backend=self.backend,
            provenance=provenance,
        )

    def _select_pose(
        self,
        request: CameraSelectionRequest,
        constraints: CameraConstraintSet,
    ) -> CameraSelectionResult:
        if not self.allow_freeform_pose or not request.allow_freeform_pose:
            raise ValueError(
                "freeform_pose requires explicit adapter and request opt-in"
            )
        payload = _base_payload(request, constraints)
        payload["selection_mode"] = "freeform_pose"
        response = _strict_response(
            self._invoke(payload),
            allowed=_POSE_RESPONSE_FIELDS,
        )
        proposal = _validate_pose_schema(
            response.get("camera_proposal")
        )
        validation = _validate_pose_validation_result(
            self.pose_validator.validate(deepcopy(proposal), request),
            proposal=proposal,
        )
        validated_pose = validation.get("pose") or proposal
        return CameraSelectionResult(
            outcome="selected",
            camera_proposal=deepcopy(validated_pose),
            reason=_reason(response),
            backend=self.backend,
            evidence_round=request.evidence_round,
            provenance={
                **_vlm_provenance(
                    self,
                    response,
                    selection_mode="freeform_pose",
                ),
                "pose_validation": deepcopy(validation),
                "pose_validator": _qualified_name(
                    self.pose_validator
                ),
            },
        )

    def _repair_plans(
        self,
        constraints: CameraConstraintSet,
        request: CameraSelectionRequest,
    ) -> tuple[CameraRepairPlan, ...]:
        controller_plans = request.context.get("camera_repair_plans")
        if controller_plans:
            if not isinstance(controller_plans, (list, tuple)):
                raise ValueError(
                    "controller camera_repair_plans must be a JSON list"
                )
            plans = tuple(
                CameraRepairPlan.from_value(
                    value,
                    constraints=constraints,
                )
                for value in controller_plans
            )
        elif self.repair_planner is not None:
            plans = tuple(self.repair_planner(constraints, request))
        else:
            evaluations = request.context.get(
                "deterministic_candidate_evaluations",
                (),
            )
            explicit = request.context.get(
                "camera_constraint_conflicts"
            )
            conflicts = diagnose_camera_constraint_conflicts(
                constraints,
                candidate_evaluations=evaluations,
                explicit_conflicts=explicit,
            )
            plans = generate_camera_repair_plans(
                constraints,
                conflicts=conflicts,
                max_plans=self.max_repair_plans,
            )
        if not all(isinstance(plan, CameraRepairPlan) for plan in plans):
            raise TypeError(
                "repair planner must return CameraRepairPlan values"
            )
        ids = [plan.plan_id for plan in plans]
        if len(ids) != len(set(ids)):
            raise ValueError("repair planner returned duplicate plan IDs")
        for plan in plans:
            plan.validate_against(constraints)
        return plans

    def _invoke(self, payload: dict[str, Any]) -> Any:
        self.last_call_usage = {"vlm_call_count": 1}
        result = self._call(payload)
        if isinstance(result, CameraSelectionResult):
            result = result.to_dict()
        return result


def _base_payload(
    request: CameraSelectionRequest,
    constraints: CameraConstraintSet,
) -> dict[str, Any]:
    payload = {
        "vlm_role": "vlm_camera_selector",
        "decision_contract": "camera_selection_v1",
        "selection_mode": (
            request.context.get("vlm_selection_mode")
            or DEFAULT_VLM_SELECTION_MODE
        ),
        "task": request.task,
        "metric": request.metric,
        "target_ids": list(request.target_ids),
        "evidence_goal": deepcopy(request.evidence_goal),
        "scene_context": deepcopy(request.scene),
        "camera_constraints": constraints.to_dict(),
        "candidate_views": list(deepcopy(request.candidate_views)),
        "deterministic_rejected_candidates": deepcopy(
            request.context.get("deterministic_rejected_candidates")
            or request.context.get("rejected_candidates")
            or []
        ),
        "attempted_candidate_ids": list(_attempted_ids(request)),
        "previous_evidence_summary": _evidence_summary(
            request.existing_visual_evidence
        ),
        "evidence_budget": deepcopy(request.budget),
        "evidence_round": request.evidence_round,
        "scene_access": "read_only",
    }
    # Group geometry is authoritative controller context.  Preserve it across
    # deterministic-to-VLM escalation instead of asking the VLM to infer the
    # target scope again from the full scene.
    for field in (
        "group_scope",
        "member_ids",
        "target_bounds",
        "focus_center",
        "target_extent",
        "grouping_role",
    ):
        if field in request.context:
            payload[field] = deepcopy(request.context[field])
    return payload


def _strict_response(
    value: Any, *, allowed: frozenset[str]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("VLM camera response must be a JSON object")
    result = deepcopy(dict(value))
    unknown = set(result) - allowed
    if unknown:
        raise ValueError(
            "VLM camera response contains forbidden or unknown fields: "
            f"{sorted(unknown)}"
        )
    if _contains_mutation(result):
        raise ValueError("VLM camera response cannot mutate the scene")
    if _contains_decision(result):
        raise ValueError(
            "VLM camera response cannot contain verdict or score"
        )
    provenance = result.get("provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("VLM camera provenance must be a JSON object")
    return result


def _no_candidate_result(
    request: CameraSelectionRequest,
    *,
    attempted_ids: tuple[str, ...],
    reason_code: str,
) -> CameraSelectionResult:
    rejected = [
        {
            "candidate_id": candidate_id,
            "reason_codes": [reason_code],
        }
        for candidate_id in attempted_ids
    ]
    return camera_selection_result_from_value(
        {
            "outcome": "no_feasible_candidate",
            "attempted_candidate_ids": list(attempted_ids),
            "rejected_candidates": rejected,
            "reason_codes": [reason_code],
            "reason": "no trusted selection is available for this policy",
            "provenance": {
                "adapter_version": ACTIVE_VLM_CAMERA_SELECTOR_VERSION,
                "normal_selection_outcome": True,
                "vlm_call_count": 0,
            },
        },
        request=request,
        backend="vlm_active",
    )


def _validate_pose_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("freeform_pose requires camera_proposal")
    proposal = deepcopy(dict(value))
    unknown = set(proposal) - _POSE_FIELDS
    if unknown:
        raise ValueError(
            f"camera pose has unknown fields: {sorted(unknown)}"
        )
    proposal["location"] = list(_vector3(proposal.get("location"), "location"))
    proposal["target"] = list(_vector3(proposal.get("target"), "target"))
    if proposal["location"] == proposal["target"]:
        raise ValueError("camera location and target cannot be identical")
    lens = proposal.get("lens_mm")
    if (
        isinstance(lens, bool)
        or not isinstance(lens, (int, float))
        or not math.isfinite(float(lens))
        or float(lens) <= 0.0
    ):
        raise ValueError("camera lens_mm must be finite and positive")
    proposal["lens_mm"] = float(lens)
    camera_type = str(proposal.get("camera_type") or "PERSP")
    if camera_type != "PERSP":
        raise ValueError("freeform_pose currently requires camera_type PERSP")
    proposal["camera_type"] = camera_type
    if "up" in proposal:
        proposal["up"] = list(_vector3(proposal["up"], "up"))
    return proposal


def _validate_pose_validation_result(
    value: Any,
    *,
    proposal: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("pose validator must return a structured result")
    result = deepcopy(dict(value))
    unknown = set(result) - _POSE_VALIDATION_FIELDS
    if unknown:
        raise ValueError(
            f"pose validator returned unknown fields: {sorted(unknown)}"
        )
    if result.get("valid") is not True:
        raise ValueError("free-form camera proposal failed pose validation")
    checks = result.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("pose validator must report validation checks")
    missing = REQUIRED_POSE_VALIDATION_CHECKS - set(checks)
    failed = {
        name
        for name in REQUIRED_POSE_VALIDATION_CHECKS
        if checks.get(name) is not True
    }
    if missing or failed:
        raise ValueError(
            "pose validator did not pass all required checks: "
            f"missing={sorted(missing)}, failed={sorted(failed)}"
        )
    if _contains_mutation(result):
        raise ValueError("pose validator cannot mutate the scene")
    pose = result.get("pose")
    if pose is not None:
        result["pose"] = _validate_pose_schema(pose)
    else:
        result["pose"] = deepcopy(proposal)
    reason_codes = result.get("reason_codes", [])
    result["reason_codes"] = _text_list(reason_codes, "reason_codes")
    provenance = result.get("provenance", {})
    if not isinstance(provenance, dict):
        raise ValueError("pose validation provenance must be a JSON object")
    result["provenance"] = deepcopy(provenance)
    result["checks"] = deepcopy(dict(checks))
    return result


def _vlm_provenance(
    adapter: ActiveVLMCameraSelector,
    response: Mapping[str, Any],
    *,
    selection_mode: str,
) -> dict[str, Any]:
    model = getattr(adapter.selector, "model", None)
    return {
        "adapter_version": ACTIVE_VLM_CAMERA_SELECTOR_VERSION,
        "selection_mode": selection_mode,
        "selector_method": adapter.method_name,
        "vlm_call_count": int(
            adapter.last_call_usage.get("vlm_call_count", 1)
        ),
        "model": (
            getattr(model, "model_id", None)
            or getattr(adapter.selector, "model_id", None)
        ),
        "endpoint": (
            getattr(model, "endpoint", None)
            or getattr(adapter.selector, "endpoint", None)
        ),
        "vlm_response_provenance": deepcopy(
            response.get("provenance") or {}
        ),
    }


def _selector_call(value: Any) -> tuple[Callable[[dict[str, Any]], Any], str]:
    for name in ("select", "select_camera_views"):
        call = getattr(value, name, None)
        if callable(call):
            return call, name
    if callable(value):
        return value, "__call__"
    raise TypeError(
        "active VLM selector requires select(payload), "
        "select_camera_views(payload), or a callable"
    )


def _candidate_index(
    values: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for candidate in values:
        if not isinstance(candidate, dict):
            raise ValueError("candidate views must be JSON objects")
        candidate_id = str(candidate.get("id") or "").strip()
        if not candidate_id or candidate_id in result:
            raise ValueError(
                "candidate view IDs must be non-empty and unique"
            )
        result[candidate_id] = candidate
    return result


def _known_target_ids(
    request: CameraSelectionRequest,
) -> tuple[str, ...]:
    raw_known = request.context.get("known_target_ids")
    if raw_known is not None and not isinstance(
        raw_known, (list, tuple)
    ):
        raise ValueError("known_target_ids must be a JSON list")
    values = [
        str(value)
        for value in (raw_known or ())
        if str(value).strip()
    ]
    return merge_authoritative_target_ids(values, request.scene)


def _request_metric(
    request: CameraSelectionRequest, *, fallback: str
) -> str:
    relation_type = request.context.get(
        "relation_type"
    ) or request.context.get("event_type")
    if (
        str(request.metric).strip().lower()
        in {"relation", "spatial_fidelity"}
        and relation_type is None
    ):
        return fallback
    return canonical_camera_metric(
        request.metric,
        relation_type=(
            str(relation_type) if relation_type is not None else None
        ),
    )


def _attempted_ids(request: CameraSelectionRequest) -> tuple[str, ...]:
    values = request.context.get("attempted_view_ids", ())
    if not isinstance(values, (list, tuple)):
        raise ValueError("attempted_view_ids must be a JSON list")
    normalized = [
        str(value).strip()
        for value in values
        if str(value).strip()
    ]
    if len(normalized) != len(values):
        raise ValueError(
            "attempted_view_ids must contain non-empty strings"
        )
    # Attempt history is global across repair episodes, while a renderer may
    # legitimately replace the current candidate bank. Keep the full history
    # for audit and filter only its intersection with the current bank.
    return tuple(dict.fromkeys(normalized))


def _evidence_summary(values: tuple[Any, ...]) -> list[Any]:
    result: list[Any] = []
    for item in values:
        if not isinstance(item, dict):
            result.append(str(item))
            continue
        result.append(
            {
                key: deepcopy(item[key])
                for key in (
                    "path",
                    "image_path",
                    "view_id",
                    "role",
                    "representation",
                    "visibility",
                )
                if key in item
            }
        )
    return result


def _reason(value: Mapping[str, Any]) -> str:
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError("VLM camera response requires a reason")
    return reason


def _text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a JSON list")
    result = [str(item).strip() for item in value]
    if any(not item for item in result) or len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique non-empty strings")
    return result


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"camera {label} must be a numeric vector3")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"camera {label} must contain finite numbers")
    return tuple(float(item) for item in value)  # type: ignore[return-value]


def _contains_mutation(value: Any) -> bool:
    if isinstance(value, Mapping):
        if set(str(key) for key in value) & _MUTATION_FIELDS:
            return True
        return any(_contains_mutation(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_mutation(item) for item in value)
    return False


def _contains_decision(value: Any) -> bool:
    if isinstance(value, Mapping):
        if {"verdict", "score"} & {str(key) for key in value}:
            return True
        return any(_contains_decision(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_decision(item) for item in value)
    return False


def _qualified_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"
