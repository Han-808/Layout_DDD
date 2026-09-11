from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Iterable, Literal, Mapping, get_args

from benchmark.visual_judge.camera_dsl import (
    CAMERA_CONSTRAINT_REFERENCES,
    CAMERA_VIEW_FAMILIES,
    CameraConstraintSet,
    active_constraint_references,
)


VLMSelectionMode = Literal[
    "candidate_only",
    "repair_plan",
    "freeform_pose",
]
VLM_SELECTION_MODES = frozenset(get_args(VLMSelectionMode))
DEFAULT_VLM_SELECTION_MODE: VLMSelectionMode = "repair_plan"

CAMERA_REPAIR_PLAN_VERSION = "camera_repair_plan_v1"
CAMERA_CONSTRAINT_CONFLICT_VERSION = "camera_constraint_conflict_v1"

_PLAN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_ESTIMATED_COST_KEYS = frozenset(
    {
        "preview_renders",
        "full_renders",
        "camera_actions",
        "selected_views",
        "estimated_gpu_seconds",
    }
)
_SELECTION_KEYS = frozenset(
    {
        "selected_plan_id",
        "plan_id",
        "reason",
        "backend",
        "provenance",
    }
)
_SCENE_MUTATION_MARKERS = frozenset(
    {
        "allow_scene_mutation",
        "mutated_scene",
        "scene_mutation",
        "scene_patch",
        "scene_updates",
    }
)


@dataclass(frozen=True)
class CameraConstraintConflict:
    constraint_ids: tuple[str, ...]
    reason_code: str
    candidate_ids: tuple[str, ...] = ()
    rejection_reasons: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _unique_texts(self.constraint_ids, "conflict constraint_ids")
        if len(self.constraint_ids) < 2:
            raise ValueError(
                "camera constraint conflict requires at least two constraints"
            )
        unknown = set(self.constraint_ids) - CAMERA_CONSTRAINT_REFERENCES
        if unknown:
            raise ValueError(
                f"camera constraint conflict has unknown references: "
                f"{sorted(unknown)}"
            )
        if not str(self.reason_code).strip():
            raise ValueError("camera constraint conflict needs a reason_code")
        _unique_texts(self.candidate_ids, "conflict candidate_ids")
        if not isinstance(self.rejection_reasons, tuple) or not all(
            isinstance(value, dict) for value in self.rejection_reasons
        ):
            raise TypeError(
                "conflict rejection_reasons must be a tuple of mappings"
            )
        if not isinstance(self.provenance, dict):
            raise TypeError("conflict provenance must be a mapping")

    def validate_against(
        self, constraints: CameraConstraintSet
    ) -> CameraConstraintConflict:
        inactive = set(self.constraint_ids) - active_constraint_references(
            constraints
        )
        if inactive:
            raise ValueError(
                "conflict references inactive camera constraints: "
                f"{sorted(inactive)}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAMERA_CONSTRAINT_CONFLICT_VERSION,
            "constraint_ids": list(self.constraint_ids),
            "reason_code": self.reason_code,
            "candidate_ids": list(self.candidate_ids),
            "rejection_reasons": list(
                deepcopy(self.rejection_reasons)
            ),
            "provenance": deepcopy(self.provenance),
        }


@dataclass(frozen=True)
class CameraRepairPlan:
    plan_id: str
    objective: str
    preserved_constraints: tuple[str, ...]
    relaxed_constraints: tuple[str, ...]
    preferred_view_families: tuple[str, ...]
    required_view_count: int
    estimated_cost: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _PLAN_ID.fullmatch(str(self.plan_id)):
            raise ValueError("CameraRepairPlan plan_id is invalid")
        if not str(self.objective).strip():
            raise ValueError("CameraRepairPlan objective cannot be empty")
        _constraint_references(
            self.preserved_constraints, "preserved_constraints"
        )
        _constraint_references(
            self.relaxed_constraints, "relaxed_constraints"
        )
        if not self.preserved_constraints:
            raise ValueError(
                "CameraRepairPlan must preserve at least one constraint"
            )
        overlap = set(self.preserved_constraints) & set(
            self.relaxed_constraints
        )
        if overlap:
            raise ValueError(
                "CameraRepairPlan cannot preserve and relax the same "
                f"constraints: {sorted(overlap)}"
            )
        _view_families(self.preferred_view_families)
        if (
            isinstance(self.required_view_count, bool)
            or not isinstance(self.required_view_count, int)
            or self.required_view_count <= 0
        ):
            raise ValueError(
                "CameraRepairPlan required_view_count must be positive"
            )
        _estimated_cost(self.estimated_cost)
        if not isinstance(self.provenance, dict):
            raise TypeError("CameraRepairPlan provenance must be a mapping")
        if _contains_scene_mutation(self.provenance):
            raise ValueError(
                "CameraRepairPlan provenance cannot request scene mutation"
            )

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        constraints: CameraConstraintSet | None = None,
    ) -> CameraRepairPlan:
        if isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            allowed = {
                "schema_version",
                "plan_id",
                "objective",
                "preserved_constraints",
                "relaxed_constraints",
                "preferred_view_families",
                "required_view_count",
                "estimated_cost",
                "provenance",
            }
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(
                    "CameraRepairPlan contains unknown fields: "
                    f"{sorted(unknown)}"
                )
            schema_version = value.get("schema_version")
            if (
                schema_version is not None
                and schema_version != CAMERA_REPAIR_PLAN_VERSION
            ):
                raise ValueError(
                    "unsupported CameraRepairPlan schema_version"
                )
            result = cls(
                plan_id=str(value.get("plan_id") or ""),
                objective=str(value.get("objective") or "").strip(),
                preserved_constraints=_text_tuple(
                    value.get("preserved_constraints")
                ),
                relaxed_constraints=_text_tuple(
                    value.get("relaxed_constraints")
                ),
                preferred_view_families=_text_tuple(
                    value.get("preferred_view_families")
                ),
                required_view_count=value.get("required_view_count"),
                estimated_cost=_mapping(
                    value.get("estimated_cost"), "estimated_cost"
                ),
                provenance=_mapping(
                    value.get("provenance"), "plan provenance"
                ),
            )
        else:
            raise TypeError("CameraRepairPlan must be a mapping")
        if constraints is not None:
            result.validate_against(constraints)
        return result

    def validate_against(
        self, constraints: CameraConstraintSet
    ) -> CameraRepairPlan:
        active = active_constraint_references(constraints)
        referenced = set(self.preserved_constraints) | set(
            self.relaxed_constraints
        )
        if referenced != active:
            missing = active - referenced
            extra = referenced - active
            raise ValueError(
                "CameraRepairPlan must partition active constraints; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        unapproved = set(self.relaxed_constraints) - set(
            constraints.relaxable_constraints
        )
        if unapproved:
            raise ValueError(
                "CameraRepairPlan relaxes non-relaxable constraints: "
                f"{sorted(unapproved)}"
            )
        forbidden = set(self.preferred_view_families) & set(
            constraints.forbidden_view_families
        )
        if forbidden:
            raise ValueError(
                "CameraRepairPlan requests forbidden view families: "
                f"{sorted(forbidden)}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAMERA_REPAIR_PLAN_VERSION,
            "plan_id": self.plan_id,
            "objective": self.objective,
            "preserved_constraints": list(self.preserved_constraints),
            "relaxed_constraints": list(self.relaxed_constraints),
            "preferred_view_families": list(
                self.preferred_view_families
            ),
            "required_view_count": self.required_view_count,
            "estimated_cost": deepcopy(self.estimated_cost),
            "provenance": deepcopy(self.provenance),
        }


@dataclass(frozen=True)
class CameraRepairPlanSelection:
    plan: CameraRepairPlan
    reason: str
    backend: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, CameraRepairPlan):
            raise TypeError("repair-plan selection needs a trusted plan")
        if not str(self.reason).strip():
            raise ValueError("repair-plan selection reason cannot be empty")
        if not str(self.backend).strip():
            raise ValueError("repair-plan selection backend cannot be empty")
        if not isinstance(self.provenance, dict):
            raise TypeError(
                "repair-plan selection provenance must be a mapping"
            )
        if _contains_scene_mutation(self.provenance):
            raise ValueError(
                "repair-plan selection cannot request scene mutation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_plan_id": self.plan.plan_id,
            "reason": self.reason,
            "backend": self.backend,
            "provenance": deepcopy(self.provenance),
        }


def validate_vlm_selection_mode(
    value: Any, *, allow_freeform_pose: bool = False
) -> VLMSelectionMode:
    mode = str(value or "").strip()
    if mode not in VLM_SELECTION_MODES:
        raise ValueError(f"unknown VLM camera selection mode {mode!r}")
    if mode == "freeform_pose" and not allow_freeform_pose:
        raise ValueError("freeform_pose is disabled")
    return mode  # type: ignore[return-value]


def diagnose_camera_constraint_conflicts(
    constraints: CameraConstraintSet,
    *,
    candidate_evaluations: Iterable[Mapping[str, Any]] = (),
    explicit_conflicts: Iterable[Any] | None = None,
) -> tuple[CameraConstraintConflict, ...]:
    """Validate deterministic infeasibility evidence and expose conflicts.

    The function never infers a scientific requirement. It only reports
    conflicts among constraints already active in the supplied DSL.
    """

    if not isinstance(constraints, CameraConstraintSet):
        raise TypeError("conflict diagnosis requires CameraConstraintSet")
    explicit = (
        explicit_conflicts
        if explicit_conflicts is not None
        else constraints.metadata.get("constraint_conflicts", ())
    )
    if explicit:
        return _explicit_conflicts(constraints, explicit)

    evaluations = list(candidate_evaluations)
    if not evaluations:
        return ()
    candidate_ids: list[str] = []
    rejection_reasons: list[dict[str, Any]] = []
    failed_union: list[str] = []
    for raw in evaluations:
        if not isinstance(raw, Mapping):
            raise ValueError("candidate evaluation must be a mapping")
        candidate_id = str(
            raw.get("candidate_id") or raw.get("id") or ""
        ).strip()
        if not candidate_id:
            raise ValueError("candidate evaluation requires candidate_id")
        if candidate_id in candidate_ids:
            raise ValueError("candidate evaluation IDs must be unique")
        candidate_ids.append(candidate_id)
        feasible = raw.get("feasible")
        if not isinstance(feasible, bool):
            raise ValueError("candidate feasibility must be boolean")
        if feasible:
            return ()
        failed = _text_tuple(raw.get("failed_constraints"))
        if not failed:
            raise ValueError(
                "rejected candidate must identify failed_constraints"
            )
        _constraint_references(failed, "failed_constraints")
        inactive = set(failed) - active_constraint_references(constraints)
        if inactive:
            raise ValueError(
                "candidate failed inactive constraints: "
                f"{sorted(inactive)}"
            )
        failed_union.extend(
            value for value in failed if value not in failed_union
        )
        reason_codes = _text_tuple(raw.get("reason_codes"))
        rejection_reasons.append(
            {
                "candidate_id": candidate_id,
                "failed_constraints": list(failed),
                "reason_codes": list(reason_codes),
            }
        )
    if len(failed_union) < 2:
        return ()
    conflict = CameraConstraintConflict(
        constraint_ids=tuple(failed_union),
        reason_code="candidate_bank_constraint_conflict",
        candidate_ids=tuple(candidate_ids),
        rejection_reasons=tuple(rejection_reasons),
        provenance={
            "schema_version": CAMERA_CONSTRAINT_CONFLICT_VERSION,
            "source": "deterministic_candidate_evaluations",
        },
    )
    return (conflict.validate_against(constraints),)


def generate_camera_repair_plans(
    constraints: CameraConstraintSet,
    *,
    conflicts: Iterable[CameraConstraintConflict] | None = None,
    max_plans: int = 3,
) -> tuple[CameraRepairPlan, ...]:
    if not isinstance(constraints, CameraConstraintSet):
        raise TypeError("repair planning requires CameraConstraintSet")
    if (
        isinstance(max_plans, bool)
        or not isinstance(max_plans, int)
        or max_plans <= 0
    ):
        raise ValueError("max_plans must be a positive integer")
    resolved_conflicts = tuple(
        conflicts
        if conflicts is not None
        else diagnose_camera_constraint_conflicts(constraints)
    )
    active = active_constraint_references(constraints)
    allowed_relaxations = set(constraints.relaxable_constraints)
    plans: list[CameraRepairPlan] = []
    if not resolved_conflicts:
        plan = CameraRepairPlan(
            plan_id="preserve_all_constraints",
            objective=(
                "search an unattempted camera realization while preserving "
                "all metric-scoped constraints"
            ),
            preserved_constraints=tuple(sorted(active)),
            relaxed_constraints=(),
            preferred_view_families=(
                constraints.preferred_view_families
            ),
            required_view_count=1,
            estimated_cost={
                "selected_views": 1,
                "full_renders": 1,
            },
            provenance={
                "schema_version": CAMERA_REPAIR_PLAN_VERSION,
                "source": "deterministic_constraint_planner",
                "metric": constraints.metric,
                "constraint_fingerprint": _constraint_fingerprint(
                    constraints
                ),
                "constraint_conflict": None,
            },
        ).validate_against(constraints)
        return (plan,)
    for conflict_index, conflict in enumerate(resolved_conflicts):
        if not isinstance(conflict, CameraConstraintConflict):
            raise TypeError(
                "repair planning conflicts must be CameraConstraintConflict"
            )
        conflict.validate_against(constraints)
        for relaxed in conflict.constraint_ids:
            if relaxed not in allowed_relaxations:
                continue
            plan = CameraRepairPlan(
                plan_id=_plan_id(conflict_index, relaxed),
                objective=(
                    f"resolve {conflict.reason_code} while relaxing only "
                    f"{relaxed}"
                ),
                preserved_constraints=tuple(
                    sorted(active - {relaxed})
                ),
                relaxed_constraints=(relaxed,),
                preferred_view_families=(
                    constraints.preferred_view_families
                ),
                required_view_count=1,
                estimated_cost={
                    "selected_views": 1,
                    "full_renders": 1,
                },
                provenance={
                    "schema_version": CAMERA_REPAIR_PLAN_VERSION,
                    "source": "deterministic_constraint_planner",
                    "metric": constraints.metric,
                    "conflict": conflict.to_dict(),
                    "constraint_fingerprint": _constraint_fingerprint(
                        constraints
                    ),
                },
            ).validate_against(constraints)
            plans.append(plan)
            if len(plans) >= max_plans:
                return tuple(plans)
    return tuple(plans)


def validate_trusted_repair_plan_selection(
    value: Any,
    *,
    trusted_plans: Iterable[CameraRepairPlan],
    constraints: CameraConstraintSet | None = None,
    default_backend: str = "vlm",
) -> CameraRepairPlanSelection:
    if not isinstance(value, Mapping):
        raise ValueError("repair-plan selection must be a mapping")
    unknown = set(value) - _SELECTION_KEYS
    if unknown:
        raise ValueError(
            "repair-plan selection contains forbidden or unknown fields: "
            f"{sorted(unknown)}"
        )
    if _contains_scene_mutation(value):
        raise ValueError(
            "repair-plan selection cannot request scene mutation"
        )
    plan_id = str(
        value.get("selected_plan_id") or value.get("plan_id") or ""
    ).strip()
    alternate = value.get("plan_id")
    selected = value.get("selected_plan_id")
    if (
        alternate is not None
        and selected is not None
        and str(alternate) != str(selected)
    ):
        raise ValueError("repair-plan selection IDs conflict")
    if not plan_id:
        raise ValueError("repair-plan selection requires selected_plan_id")
    plans = tuple(trusted_plans)
    if not all(isinstance(plan, CameraRepairPlan) for plan in plans):
        raise TypeError("trusted plans must be CameraRepairPlan values")
    by_id = {plan.plan_id: plan for plan in plans}
    if len(by_id) != len(plans):
        raise ValueError("trusted repair plan IDs must be unique")
    plan = by_id.get(plan_id)
    if plan is None:
        raise ValueError(f"unknown trusted repair plan ID {plan_id!r}")
    if constraints is not None:
        plan.validate_against(constraints)
    reason = str(value.get("reason") or "").strip()
    if not reason:
        raise ValueError(
            "repair-plan selection requires a non-empty reason"
        )
    backend = str(value.get("backend") or default_backend).strip()
    provenance = _mapping(
        value.get("provenance"), "repair-plan selection provenance"
    )
    return CameraRepairPlanSelection(
        plan=plan,
        reason=reason,
        backend=backend,
        provenance=provenance,
    )


def _explicit_conflicts(
    constraints: CameraConstraintSet,
    values: Iterable[Any],
) -> tuple[CameraConstraintConflict, ...]:
    conflicts: list[CameraConstraintConflict] = []
    for index, raw in enumerate(values):
        if isinstance(raw, Mapping):
            refs = _text_tuple(
                raw.get("constraint_ids") or raw.get("constraints")
            )
            reason = str(
                raw.get("reason_code") or "camera_constraint_conflict"
            )
        elif isinstance(raw, (list, tuple)):
            refs = _text_tuple(raw)
            reason = "camera_constraint_conflict"
        else:
            raise ValueError("explicit camera conflict must be a mapping/list")
        conflict = CameraConstraintConflict(
            constraint_ids=refs,
            reason_code=reason,
            provenance={
                "schema_version": CAMERA_CONSTRAINT_CONFLICT_VERSION,
                "source": "deterministic_explicit_conflict",
                "conflict_index": index,
            },
        ).validate_against(constraints)
        signature = (conflict.constraint_ids, conflict.reason_code)
        if any(
            (item.constraint_ids, item.reason_code) == signature
            for item in conflicts
        ):
            continue
        conflicts.append(conflict)
    return tuple(conflicts)


def _plan_id(index: int, relaxed: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", relaxed)
    return f"repair_{index:02d}_relax_{safe}"[:128]


def _constraint_fingerprint(constraints: CameraConstraintSet) -> str:
    payload = json.dumps(
        constraints.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _constraint_references(
    values: tuple[str, ...], label: str
) -> None:
    _unique_texts(values, label)
    unknown = set(values) - CAMERA_CONSTRAINT_REFERENCES
    if unknown:
        raise ValueError(
            f"{label} contains unknown constraint references: "
            f"{sorted(unknown)}"
        )


def _view_families(values: tuple[str, ...]) -> None:
    _unique_texts(values, "preferred_view_families")
    unknown = set(values) - CAMERA_VIEW_FAMILIES
    if unknown:
        raise ValueError(
            "CameraRepairPlan has unknown preferred view families: "
            f"{sorted(unknown)}"
        )


def _estimated_cost(value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise TypeError("CameraRepairPlan estimated_cost must be a mapping")
    unknown = set(value) - _ESTIMATED_COST_KEYS
    if unknown:
        raise ValueError(
            f"estimated_cost contains unknown fields: {sorted(unknown)}"
        )
    for key, raw in value.items():
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0.0
        ):
            raise ValueError(
                f"estimated_cost {key} must be finite and non-negative"
            )


def _unique_texts(values: tuple[str, ...], label: str) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in values
    ):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} cannot contain duplicates")


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("structured repair-plan fields must be JSON lists")
    return tuple(str(item).strip() for item in value)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return deepcopy(dict(value))


def _contains_scene_mutation(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) in _SCENE_MUTATION_MARKERS:
                return True
            if _contains_scene_mutation(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_scene_mutation(child) for child in value)
    return False
