from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Literal, Mapping, get_args

from benchmark.visual_judge.interfaces.evidence import EvidenceGateResult
from benchmark.visual_judge.interfaces.judge import EvidenceRequest


CameraObservation = Literal[
    "target_visible",
    "joint_visibility",
    "contact_surface_visible",
    "support_chain_visible",
    "architecture_plane_visible",
    "front_back_disambiguated",
    "depth_baseline_available",
    "group_context_visible",
    "global_context_preserved",
    "occluder_avoided",
]

CAMERA_OBSERVATIONS = frozenset(get_args(CameraObservation))
CAMERA_DSL_VERSION = "camera_constraint_set_v1"

# This is deliberately finite. Adding a new family is an explicit schema change,
# rather than an opportunity for a selector to smuggle in a pose or action.
CAMERA_VIEW_FAMILIES = frozenset(
    {
        "global_top",
        "global_perspective",
        "metric_local",
        "visibility_ranked",
        "support_contact_plane",
        "query_cov",
        "contact_plane_oblique",
        "pair_side_orbit_left",
        "pair_side_orbit_right",
        "contact_oblique_high",
        "contact_oblique_low",
        "boundary_focus_dolly",
        "plane_tangent_left",
        "plane_tangent_right",
        "interior_normal_retreat",
        "base_support_focus_dolly",
        "low_angle_orbit_left",
        "low_angle_orbit_right",
        "lower_contact_view",
        "medium_oblique_topology",
        "separation_side",
        "pair_axis_oblique",
        "high_oblique",
        "low_oblique",
        "plane_oblique",
        "oblique_ne",
        "oblique_sw",
        "canonical_high_oblique_pair_v1",
        "canonical_style_local_v1",
    }
)

SCALAR_CONSTRAINT_REFERENCES = frozenset(
    {
        "min_projected_coverage",
        "require_joint_visibility",
        "require_global_anchor",
    }
)
CAMERA_CONSTRAINT_REFERENCES = (
    CAMERA_OBSERVATIONS | SCALAR_CONSTRAINT_REFERENCES
)


@dataclass(frozen=True)
class MetricCameraRequirement:
    """Metric-scoped camera vocabulary; this does not define metric semantics."""

    baseline_observations: tuple[CameraObservation, ...]
    allowed_observations: frozenset[CameraObservation]


def _requirement(
    baseline: tuple[CameraObservation, ...],
    *,
    extra: tuple[CameraObservation, ...] = (),
) -> MetricCameraRequirement:
    allowed = frozenset(
        (
            *baseline,
            *extra,
            "target_visible",
            "global_context_preserved",
            "occluder_avoided",
        )
    )
    return MetricCameraRequirement(
        baseline_observations=baseline,
        allowed_observations=allowed,
    )


# These entries only decompose visual observations already required by the
# corresponding metric. They do not add a score, threshold, or decision rule.
METRIC_CAMERA_REQUIREMENTS: dict[str, MetricCameraRequirement] = {
    "collision": _requirement(
        (
            "contact_surface_visible",
            "joint_visibility",
            "occluder_avoided",
        ),
        extra=("depth_baseline_available",),
    ),
    "support": _requirement(
        (
            "contact_surface_visible",
            "support_chain_visible",
            "joint_visibility",
        ),
    ),
    "oob": _requirement(
        (
            "target_visible",
            "architecture_plane_visible",
            "global_context_preserved",
        ),
    ),
    "facing": _requirement(
        (
            "front_back_disambiguated",
            "joint_visibility",
            "group_context_visible",
        ),
    ),
    "depth_relation": _requirement(
        (
            "depth_baseline_available",
            "joint_visibility",
        ),
        extra=("group_context_visible",),
    ),
    "directional_relation": _requirement(
        ("joint_visibility", "global_context_preserved"),
        extra=("group_context_visible",),
    ),
    "proximity_relation": _requirement(
        ("joint_visibility", "group_context_visible"),
    ),
    "contact_relation": _requirement(
        ("contact_surface_visible", "joint_visibility"),
        extra=("support_chain_visible",),
    ),
    "containment_relation": _requirement(
        (
            "target_visible",
            "joint_visibility",
            "global_context_preserved",
        ),
    ),
    "architecture_relation": _requirement(
        (
            "target_visible",
            "architecture_plane_visible",
            "global_context_preserved",
        ),
        extra=("contact_surface_visible", "support_chain_visible"),
    ),
    "functional_semantic_fidelity": _requirement(
        ("group_context_visible", "global_context_preserved"),
        extra=(
            "joint_visibility",
            "contact_surface_visible",
            "support_chain_visible",
        ),
    ),
    "scale_consistency": _requirement(
        ("target_visible",),
        extra=(
            "joint_visibility",
            "group_context_visible",
            "global_context_preserved",
        ),
    ),
    "object_pairing_consistency": _requirement(
        (
            "joint_visibility",
            "group_context_visible",
            "global_context_preserved",
        ),
    ),
    "style_consistency": _requirement(
        ("group_context_visible", "global_context_preserved"),
    ),
}

_METRIC_ALIASES = {
    "object_architecture_penetration": "oob",
    "out_of_bounds": "oob",
    "orientation": "facing",
    "face_to": "facing",
    "functional_semantic": "functional_semantic_fidelity",
    "scale": "scale_consistency",
    "object_pairing": "object_pairing_consistency",
    "style": "style_consistency",
    "group_level_semantic_judgment": "functional_semantic_fidelity",
    "group_semantic": "functional_semantic_fidelity",
}

_RELATION_METRICS = {
    "facing": "facing",
    "face_to": "facing",
    "in_front": "depth_relation",
    "in_front_of": "depth_relation",
    "behind": "depth_relation",
    "left": "directional_relation",
    "left_of": "directional_relation",
    "right": "directional_relation",
    "right_of": "directional_relation",
    "above": "directional_relation",
    "below": "directional_relation",
    "aligned": "directional_relation",
    "aligned_with": "directional_relation",
    "parallel": "directional_relation",
    "perpendicular": "directional_relation",
    "between": "directional_relation",
    "ordered": "directional_relation",
    "around": "directional_relation",
    "near": "proximity_relation",
    "far": "proximity_relation",
    "contact": "contact_relation",
    "on_top_of": "contact_relation",
    "within": "containment_relation",
    "contains": "containment_relation",
    "on_floor": "architecture_relation",
    "against_wall": "architecture_relation",
    "near_wall": "architecture_relation",
    "at_corner": "architecture_relation",
    "near_corner": "architecture_relation",
    "room_center": "architecture_relation",
    "room_region": "architecture_relation",
    "along_wall": "architecture_relation",
    "mounted_on_wall": "architecture_relation",
    "attached_to_ceiling": "architecture_relation",
    "hung_from_ceiling": "architecture_relation",
}

_OBSERVATION_ALIASES: dict[str, tuple[CameraObservation, ...]] = {
    observation: (observation,) for observation in CAMERA_OBSERVATIONS
}
_OBSERVATION_ALIASES.update(
    {
        "target_visibility": ("target_visible",),
        "jointly_visible": ("joint_visibility",),
        "required_entities_jointly_visible": ("joint_visibility",),
        "contact_region": ("contact_surface_visible",),
        "contact_surface": ("contact_surface_visible",),
        "support_chain": ("support_chain_visible",),
        "architecture_plane": ("architecture_plane_visible",),
        "boundary_plane": ("architecture_plane_visible",),
        "front_back": ("front_back_disambiguated",),
        "depth_baseline": ("depth_baseline_available",),
        "group_context": ("group_context_visible",),
        "global_context": ("global_context_preserved",),
        "avoid_occluder": ("occluder_avoided",),
        "global_composition": (
            "group_context_visible",
            "global_context_preserved",
        ),
        "contact": ("contact_surface_visible",),
        "local_view": ("target_visible",),
        "closer_view": ("target_visible",),
        "claim_scoped_local_view": ("target_visible",),
        "metric_scoped_render_evidence": ("target_visible",),
        "an unobstructed local view": (
            "target_visible",
            "occluder_avoided",
        ),
        "complete target silhouettes": ("target_visible",),
    }
)

_METRIC_OBSERVATION_ALIASES = {
    "collision": {
        "support_contact_region": ("contact_surface_visible",),
        "collision_contact_region": ("contact_surface_visible",),
    },
    "support": {
        "support_contact_region": (
            "contact_surface_visible",
            "support_chain_visible",
        ),
        "support_region": ("support_chain_visible",),
    },
    "functional_semantic_fidelity": {
        "support_contact_region": (
            "contact_surface_visible",
            "support_chain_visible",
        ),
        "contact_region": ("contact_surface_visible",),
    },
    "oob": {
        "boundary_region": ("architecture_plane_visible",),
    },
}

_DEFICIENCY_OBSERVATIONS: dict[
    str, tuple[CameraObservation, ...]
] = {
    # A missing packet requires the metric's existing baseline observations;
    # it does not introduce a new camera-domain requirement.
    "visual_evidence_missing": (),
    "target_not_visible": ("target_visible", "occluder_avoided"),
    "target_visibility_metadata_missing": ("target_visible",),
    "target_visibility_not_established": (
        "target_visible",
        "occluder_avoided",
    ),
    "projected_coverage_metadata_missing": ("target_visible",),
    "projected_coverage_insufficient": ("target_visible",),
    "required_entities_not_jointly_visible": ("joint_visibility",),
    "required_targets_not_jointly_visible": ("joint_visibility",),
    "joint_visibility_metadata_missing": ("joint_visibility",),
    "focus_region_out_of_frame": (),
    "focus_region_too_small": (),
    "contact_region_out_of_frame": ("contact_surface_visible",),
    "support_region_out_of_frame": ("support_chain_visible",),
    "boundary_region_out_of_frame": ("architecture_plane_visible",),
    "target_occluded_or_too_small": (
        "target_visible",
        "occluder_avoided",
    ),
    "architecture_plane_not_visible": ("architecture_plane_visible",),
    "required_global_evidence_missing": ("global_context_preserved",),
    "required_global_evidence_metadata_missing": (
        "global_context_preserved",
    ),
    "required_local_evidence_missing": (),
    "required_local_evidence_metadata_missing": (),
    "required_local_view_count_missing": (),
    "measured_local_visibility_insufficient": (),
    "redundant_view": (),
    "redundant_local_views": (),
    "view_redundancy_metadata_missing": (),
}

_METRIC_FOCUS_OBSERVATIONS = {
    "collision": ("contact_surface_visible", "joint_visibility"),
    "support": ("contact_surface_visible", "support_chain_visible"),
    "oob": ("architecture_plane_visible", "target_visible"),
    "facing": ("front_back_disambiguated", "joint_visibility"),
    "depth_relation": ("depth_baseline_available", "joint_visibility"),
    "directional_relation": ("joint_visibility",),
    "proximity_relation": ("joint_visibility", "group_context_visible"),
    "contact_relation": ("contact_surface_visible", "joint_visibility"),
    "containment_relation": ("target_visible", "joint_visibility"),
    "architecture_relation": (
        "target_visible",
        "architecture_plane_visible",
    ),
    "functional_semantic_fidelity": (
        "group_context_visible",
        "target_visible",
    ),
    "scale_consistency": ("target_visible", "joint_visibility"),
    "object_pairing_consistency": (
        "joint_visibility",
        "group_context_visible",
    ),
    "style_consistency": (
        "group_context_visible",
        "global_context_preserved",
    ),
}


@dataclass(frozen=True)
class CameraConstraintSet:
    target_ids: tuple[str, ...]
    required_observations: tuple[CameraObservation, ...]
    preserved_observations: tuple[CameraObservation, ...] = ()
    preferred_view_families: tuple[str, ...] = ()
    forbidden_view_families: tuple[str, ...] = ()
    min_projected_coverage: float | None = None
    require_joint_visibility: bool = False
    require_global_anchor: bool = False
    relaxable_constraints: tuple[str, ...] = ()
    metric: str = ""
    view_goal: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_unique_text_tuple(self.target_ids, "target_ids")
        if not self.target_ids:
            raise ValueError("CameraConstraintSet target_ids cannot be empty")
        _validate_observations(
            self.required_observations, "required_observations"
        )
        if not self.required_observations:
            raise ValueError(
                "CameraConstraintSet required_observations cannot be empty"
            )
        _validate_observations(
            self.preserved_observations, "preserved_observations"
        )
        overlap = set(self.required_observations) & set(
            self.preserved_observations
        )
        if overlap:
            raise ValueError(
                "required and preserved observations must be disjoint: "
                f"{sorted(overlap)}"
            )
        _validate_view_families(
            self.preferred_view_families, "preferred_view_families"
        )
        _validate_view_families(
            self.forbidden_view_families, "forbidden_view_families"
        )
        family_overlap = set(self.preferred_view_families) & set(
            self.forbidden_view_families
        )
        if family_overlap:
            raise ValueError(
                "preferred and forbidden view families must be disjoint: "
                f"{sorted(family_overlap)}"
            )
        if self.min_projected_coverage is not None:
            value = self.min_projected_coverage
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(
                    "min_projected_coverage must be finite and between 0 and 1"
                )
        if not isinstance(self.require_joint_visibility, bool):
            raise TypeError("require_joint_visibility must be boolean")
        if not isinstance(self.require_global_anchor, bool):
            raise TypeError("require_global_anchor must be boolean")
        global_context_active = "global_context_preserved" in (
            set(self.required_observations)
            | set(self.preserved_observations)
        )
        if global_context_active and not self.require_global_anchor:
            # Keeping global context is a packet-composition invariant, not a
            # selector preference. Normalizing here prevents a local
            # corrective render with merge_policy=replace from silently
            # deleting the global anchor.
            object.__setattr__(self, "require_global_anchor", True)
        metric = canonical_camera_metric(self.metric)
        if metric != self.metric:
            raise ValueError(
                f"CameraConstraintSet metric must be canonical {metric!r}"
            )
        if not str(self.view_goal).strip():
            raise ValueError("CameraConstraintSet view_goal cannot be empty")
        if not isinstance(self.metadata, dict):
            raise TypeError("CameraConstraintSet metadata must be a mapping")
        allowed = METRIC_CAMERA_REQUIREMENTS[metric].allowed_observations
        incompatible = (
            set(self.required_observations)
            | set(self.preserved_observations)
        ) - allowed
        if incompatible:
            raise ValueError(
                f"camera observations {sorted(incompatible)} are incompatible "
                f"with metric {metric!r}"
            )
        if (
            self.require_joint_visibility
            and "joint_visibility" not in self.required_observations
        ):
            raise ValueError(
                "require_joint_visibility requires joint_visibility"
            )
        if self.require_global_anchor and (
            "global_context_preserved"
            not in (
                set(self.required_observations)
                | set(self.preserved_observations)
            )
        ):
            raise ValueError(
                "require_global_anchor requires global_context_preserved"
            )
        _validate_unique_text_tuple(
            self.relaxable_constraints, "relaxable_constraints"
        )
        active = active_constraint_references(self)
        invalid_refs = set(self.relaxable_constraints) - active
        if invalid_refs:
            raise ValueError(
                "relaxable constraints must reference active constraints: "
                f"{sorted(invalid_refs)}"
            )

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        known_target_ids: Iterable[str],
        relation_type: str | None = None,
    ) -> CameraConstraintSet:
        if isinstance(value, cls):
            result = value
        elif isinstance(value, Mapping):
            _reject_unknown_keys(
                value,
                {
                    "schema_version",
                    "target_ids",
                    "required_observations",
                    "preserved_observations",
                    "preferred_view_families",
                    "forbidden_view_families",
                    "min_projected_coverage",
                    "require_joint_visibility",
                    "require_global_anchor",
                    "relaxable_constraints",
                    "metric",
                    "view_goal",
                    "metadata",
                },
                "CameraConstraintSet",
            )
            schema_version = value.get("schema_version")
            if (
                schema_version is not None
                and schema_version != CAMERA_DSL_VERSION
            ):
                raise ValueError(
                    "unsupported CameraConstraintSet schema_version"
                )
            metric = canonical_camera_metric(
                value.get("metric"), relation_type=relation_type
            )
            result = cls(
                target_ids=_text_tuple(value.get("target_ids")),
                required_observations=_observation_tuple(
                    value.get("required_observations"), metric=metric
                ),
                preserved_observations=_observation_tuple(
                    value.get("preserved_observations"), metric=metric
                ),
                preferred_view_families=_text_tuple(
                    value.get("preferred_view_families")
                ),
                forbidden_view_families=_text_tuple(
                    value.get("forbidden_view_families")
                ),
                min_projected_coverage=value.get(
                    "min_projected_coverage"
                ),
                require_joint_visibility=_strict_bool(
                    value.get("require_joint_visibility", False),
                    "require_joint_visibility",
                ),
                require_global_anchor=_strict_bool(
                    value.get("require_global_anchor", False),
                    "require_global_anchor",
                ),
                relaxable_constraints=_text_tuple(
                    value.get("relaxable_constraints")
                ),
                metric=metric,
                view_goal=str(value.get("view_goal") or "").strip(),
                metadata=_mapping(
                    value.get("metadata"), "CameraConstraintSet metadata"
                ),
            )
        else:
            raise TypeError("CameraConstraintSet must be a mapping")
        result.validate_targets(known_target_ids)
        return result

    def validate_targets(
        self, known_target_ids: Iterable[str]
    ) -> CameraConstraintSet:
        if isinstance(known_target_ids, (str, bytes)):
            raise TypeError("known_target_ids must be an iterable of IDs")
        known = {
            str(value).strip()
            for value in known_target_ids
            if str(value).strip()
        }
        unknown = {
            target
            for target in self.target_ids
            if target != "scene" and target not in known
        }
        if unknown:
            raise ValueError(
                f"CameraConstraintSet contains unknown target IDs: "
                f"{sorted(unknown)}"
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CAMERA_DSL_VERSION,
            "target_ids": list(self.target_ids),
            "required_observations": list(self.required_observations),
            "preserved_observations": list(self.preserved_observations),
            "preferred_view_families": list(
                self.preferred_view_families
            ),
            "forbidden_view_families": list(
                self.forbidden_view_families
            ),
            "min_projected_coverage": self.min_projected_coverage,
            "require_joint_visibility": self.require_joint_visibility,
            "require_global_anchor": self.require_global_anchor,
            "relaxable_constraints": list(self.relaxable_constraints),
            "metric": self.metric,
            "view_goal": self.view_goal,
            "metadata": deepcopy(self.metadata),
        }


def canonical_camera_metric(
    metric: Any, *, relation_type: str | None = None
) -> str:
    value = str(metric or "").strip().lower()
    value = _METRIC_ALIASES.get(value, value)
    if value in {"relation", "spatial_fidelity"}:
        subtype = str(relation_type or "").strip().lower()
        if not subtype:
            raise ValueError(
                f"{value} camera constraints require a relation/event subtype"
            )
        value = _RELATION_METRICS.get(subtype, _METRIC_ALIASES.get(subtype, subtype))
    else:
        value = _RELATION_METRICS.get(value, value)
    if value not in METRIC_CAMERA_REQUIREMENTS:
        raise ValueError(f"unsupported camera-constraint metric {metric!r}")
    return value


def camera_constraints_from_judge_request(
    request: EvidenceRequest | Mapping[str, Any],
    *,
    metric: str,
    known_target_ids: Iterable[str],
    relation_type: str | None = None,
) -> CameraConstraintSet:
    evidence_request = EvidenceRequest.from_value(request)
    metadata = deepcopy(evidence_request.metadata)
    metric_name = canonical_camera_metric(
        metric,
        # The relation subtype is authoritative only when it comes from the
        # original claim/context. A Judge evidence request cannot establish or
        # change metric scope through its metadata.
        relation_type=relation_type,
    )
    if "camera_constraints" in metadata:
        raise ValueError(
            "Judge evidence_request metadata cannot define camera_constraints"
        )
    forbidden_authority = {
        "constraint_conflicts",
        "camera_repair_plans",
        "relaxable_constraints",
        "selected_plan_id",
        "relation_type",
        "event_type",
    } & set(metadata)
    if forbidden_authority:
        raise ValueError(
            "Judge evidence_request metadata contains controller-owned "
            f"camera authority: {sorted(forbidden_authority)}"
        )
    requested = _mapped_observations(
        evidence_request.missing_observations, metric_name
    )
    required = _ordered_observations(
        (
            *METRIC_CAMERA_REQUIREMENTS[
                metric_name
            ].baseline_observations,
            *requested,
        )
    )
    preserved: tuple[CameraObservation, ...] = ()
    require_joint = "joint_visibility" in required
    require_global = "global_context_preserved" in required
    result = CameraConstraintSet(
        target_ids=evidence_request.target_ids,
        required_observations=required,
        preserved_observations=preserved,
        preferred_view_families=(),
        forbidden_view_families=(),
        min_projected_coverage=None,
        require_joint_visibility=require_joint,
        require_global_anchor=require_global,
        relaxable_constraints=(),
        metric=metric_name,
        view_goal=evidence_request.view_goal,
        metadata={
            "source": "judge_evidence_request",
            "source_missing_observations": list(
                evidence_request.missing_observations
            ),
            "source_metadata_keys": sorted(metadata),
        },
    )
    return result.validate_targets(known_target_ids)


def camera_constraints_from_gate_result(
    result: EvidenceGateResult | Mapping[str, Any],
    *,
    metric: str,
    target_ids: Iterable[str],
    known_target_ids: Iterable[str],
    view_goal: str,
    evidence_goal: Mapping[str, Any] | None = None,
    relation_type: str | None = None,
) -> CameraConstraintSet:
    gate_result = EvidenceGateResult.from_value(result)
    if gate_result.ready:
        raise ValueError("ready evidence does not require camera constraints")
    if not gate_result.camera_repairable:
        raise ValueError(
            "non-camera-repairable evidence cannot become a camera request"
        )
    goal = deepcopy(dict(evidence_goal or {}))
    metric_name = canonical_camera_metric(
        metric,
        relation_type=relation_type or _relation_type(goal),
    )
    mapped: list[CameraObservation] = []
    for deficiency in gate_result.deficiencies:
        code = str(deficiency.get("code") or "").strip()
        mapped.extend(_observations_for_deficiency(code, metric_name))
    required = _ordered_observations(
        (
            *METRIC_CAMERA_REQUIREMENTS[
                metric_name
            ].baseline_observations,
            *mapped,
        )
    )
    required_roles = goal.get("required_roles")
    global_required = _positive_role_count(
        goal.get("required_global_view_count")
    ) or (
        isinstance(required_roles, list)
        and any("global" in str(role).lower() for role in required_roles)
    )
    if global_required:
        required = _ordered_observations(
            (*required, "global_context_preserved")
        )
    technical = goal.get("technical_requirements")
    if technical is None:
        technical = {}
    if not isinstance(technical, Mapping):
        raise ValueError("technical_requirements must be a mapping")
    min_coverage = goal.get(
        "min_projected_coverage",
        technical.get("min_projected_coverage"),
    )
    require_joint = _strict_bool(
        technical.get(
            "require_joint_visibility",
            "joint_visibility" in required,
        ),
        "require_joint_visibility",
    )
    if require_joint and "joint_visibility" not in required:
        required = _ordered_observations(
            (*required, "joint_visibility")
        )
    require_global = bool(
        global_required
        or "global_context_preserved" in required
    )
    constraints = CameraConstraintSet(
        target_ids=_text_tuple(tuple(target_ids)),
        required_observations=required,
        min_projected_coverage=min_coverage,
        require_joint_visibility=require_joint,
        require_global_anchor=require_global,
        metric=metric_name,
        view_goal=str(view_goal or "").strip(),
        metadata={
            "source": "evidence_gate",
            "gate_reason_codes": list(gate_result.reason_codes),
            "gate_deficiencies": list(deepcopy(gate_result.deficiencies)),
            "gate_backend": gate_result.backend,
            "evidence_goal": goal,
        },
    )
    return constraints.validate_targets(known_target_ids)


def selector_request_from_constraints(
    constraints: CameraConstraintSet,
    *,
    base_request: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(constraints, CameraConstraintSet):
        raise TypeError(
            "selector request requires a CameraConstraintSet"
        )
    result = deepcopy(dict(base_request or {}))
    existing_metric = result.get("metric")
    if existing_metric is not None and canonical_camera_metric(
        existing_metric,
        relation_type=_relation_type(result),
    ) != constraints.metric:
        raise ValueError("selector request metric conflicts with Camera DSL")
    existing_targets = result.get("target_ids") or result.get("object_ids")
    if existing_targets is not None and _text_tuple(
        existing_targets
    ) != constraints.target_ids:
        raise ValueError("selector request targets conflict with Camera DSL")
    result["metric"] = constraints.metric
    result["target_ids"] = list(constraints.target_ids)
    result["object_ids"] = list(constraints.target_ids)
    result["evidence_goal"] = {
        **deepcopy(
            result.get("evidence_goal")
            if isinstance(result.get("evidence_goal"), dict)
            else {}
        ),
        "view_goal": constraints.view_goal,
        "missing_observations": list(
            constraints.required_observations
        ),
        "camera_constraints": constraints.to_dict(),
    }
    result["camera_constraints"] = constraints.to_dict()
    return result


def active_constraint_references(
    constraints: CameraConstraintSet,
) -> set[str]:
    active = set(constraints.required_observations) | set(
        constraints.preserved_observations
    )
    if constraints.min_projected_coverage is not None:
        active.add("min_projected_coverage")
    if constraints.require_joint_visibility:
        active.add("require_joint_visibility")
    if constraints.require_global_anchor:
        active.add("require_global_anchor")
    return active


def _mapped_observations(
    values: Iterable[str], metric: str
) -> tuple[CameraObservation, ...]:
    observations: list[CameraObservation] = []
    aliases = _METRIC_OBSERVATION_ALIASES.get(metric, {})
    for raw in values:
        token = str(raw or "").strip()
        mapped = aliases.get(token) or _OBSERVATION_ALIASES.get(token)
        if mapped is None:
            raise ValueError(f"unknown camera observation {token!r}")
        observations.extend(mapped)
    return _validate_metric_observations(observations, metric)


def _observations_for_deficiency(
    code: str, metric: str
) -> tuple[CameraObservation, ...]:
    if code.startswith("required_role_missing:"):
        role = code.split(":", 1)[1].strip().lower()
        if not role:
            raise ValueError("empty required evidence role")
        if "global" in role:
            values: tuple[CameraObservation, ...] = (
                "global_context_preserved",
            )
        elif any(
            token in role
            for token in ("local", "focus", "pair", "group")
        ):
            values = _METRIC_FOCUS_OBSERVATIONS.get(metric, ())
        else:
            raise ValueError(
                f"unmapped required evidence role {role!r}"
            )
    elif code in {"focus_region_out_of_frame", "focus_region_too_small"}:
        values = _METRIC_FOCUS_OBSERVATIONS.get(metric, ())
        if not values:
            raise ValueError(
                f"deficiency {code!r} has no mapping for metric {metric!r}"
            )
    else:
        values = _DEFICIENCY_OBSERVATIONS.get(code)
        if values is None:
            raise ValueError(
                f"unknown camera-repairable deficiency {code!r}"
            )
    return _validate_metric_observations(values, metric)


def _validate_metric_observations(
    values: Iterable[CameraObservation], metric: str
) -> tuple[CameraObservation, ...]:
    result = _ordered_observations(values)
    incompatible = set(result) - METRIC_CAMERA_REQUIREMENTS[
        metric
    ].allowed_observations
    if incompatible:
        raise ValueError(
            f"camera observations {sorted(incompatible)} are incompatible "
            f"with metric {metric!r}"
        )
    return result


def _observation_tuple(
    value: Any, *, metric: str
) -> tuple[CameraObservation, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("camera observations must be a JSON list")
    return _mapped_observations(value, metric)


def _ordered_observations(
    values: Iterable[CameraObservation],
) -> tuple[CameraObservation, ...]:
    result: list[CameraObservation] = []
    for value in values:
        if value not in CAMERA_OBSERVATIONS:
            raise ValueError(f"unknown camera observation {value!r}")
        if value not in result:
            result.append(value)
    return tuple(result)


def _validate_observations(
    values: tuple[CameraObservation, ...], label: str
) -> None:
    _validate_unique_text_tuple(values, label)
    unknown = set(values) - CAMERA_OBSERVATIONS
    if unknown:
        raise ValueError(f"{label} contains unknown values: {sorted(unknown)}")


def _validate_view_families(values: tuple[str, ...], label: str) -> None:
    _validate_unique_text_tuple(values, label)
    unknown = set(values) - CAMERA_VIEW_FAMILIES
    if unknown:
        raise ValueError(
            f"{label} contains unknown view families: {sorted(unknown)}"
        )


def _validate_unique_text_tuple(
    values: tuple[str, ...], label: str
) -> None:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    if any(
        not isinstance(value, str) or not value.strip()
        for value in values
    ):
        raise ValueError(f"{label} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} cannot contain duplicates")


def _text_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("structured camera fields must be JSON lists")
    return tuple(str(item).strip() for item in value)


def _strict_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return deepcopy(dict(value))


def _relation_type(value: Mapping[str, Any]) -> str | None:
    for key in ("relation_type", "event_type", "type", "family"):
        candidate = value.get(key)
        if candidate:
            return str(candidate)
    relation = value.get("relation")
    if isinstance(relation, Mapping):
        for key in ("type", "relation", "predicate"):
            candidate = relation.get(key)
            if candidate:
                return str(candidate)
    return None


def _positive_role_count(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            "required evidence role counts must be non-negative integers"
        )
    return value > 0


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {sorted(unknown)}")
