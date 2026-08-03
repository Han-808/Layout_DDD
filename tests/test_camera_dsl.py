from __future__ import annotations

import pytest

from benchmark.visual_judge.acquisition_planner import (
    MetricAcquisitionPlanningRequest,
    MetricSpecificAcquisitionPlanner,
)
from benchmark.visual_judge.camera_dsl import (
    CAMERA_OBSERVATIONS,
    METRIC_CAMERA_REQUIREMENTS,
    CameraConstraintSet,
    camera_constraints_from_judge_request,
    canonical_camera_metric,
    selector_request_from_constraints,
)
from benchmark.visual_judge.camera_repair import (
    DEFAULT_VLM_SELECTION_MODE,
    CameraConstraintConflict,
    CameraRepairPlan,
    diagnose_camera_constraint_conflicts,
    generate_camera_repair_plans,
    validate_trusted_repair_plan_selection,
    validate_vlm_selection_mode,
)
from benchmark.visual_judge.interfaces.judge import EvidenceRequest


_OBSERVATION_METRIC = {
    "target_visible": "oob",
    "joint_visibility": "collision",
    "contact_surface_visible": "collision",
    "support_chain_visible": "support",
    "architecture_plane_visible": "oob",
    "front_back_disambiguated": "facing",
    "depth_baseline_available": "depth_relation",
    "group_context_visible": "functional_semantic_fidelity",
    "interaction_side_visible": "functional_semantic_fidelity",
    "limited_local_context": "object_pairing_consistency",
    "global_context_preserved": "style_consistency",
    "occluder_avoided": "collision",
}


def _constraints(
    *,
    metric: str = "collision",
    target_ids: tuple[str, ...] = ("a", "b"),
    relaxable: tuple[str, ...] = ("global_context_preserved",),
) -> CameraConstraintSet:
    required = METRIC_CAMERA_REQUIREMENTS[
        metric
    ].baseline_observations
    preserved = (
        ("global_context_preserved",)
        if "global_context_preserved" not in required
        else ()
    )
    return CameraConstraintSet(
        target_ids=target_ids,
        required_observations=required,
        preserved_observations=preserved,
        preferred_view_families=("contact_plane_oblique",),
        min_projected_coverage=0.01,
        require_joint_visibility="joint_visibility" in required,
        require_global_anchor=True,
        relaxable_constraints=relaxable,
        metric=metric,
        view_goal="resolve the scoped visual ambiguity",
    )


@pytest.mark.parametrize("observation", sorted(CAMERA_OBSERVATIONS))
def test_all_camera_observation_vocabulary_tokens_validate(
    observation: str,
) -> None:
    metric = _OBSERVATION_METRIC[observation]
    constraints = CameraConstraintSet.from_value(
        {
            "target_ids": ["a", "b"],
            "required_observations": [observation],
            "metric": metric,
            "view_goal": "obtain the requested technical observation",
        },
        known_target_ids=("a", "b"),
    )

    assert constraints.required_observations == (observation,)


def test_camera_constraint_set_rejects_unknown_observation() -> None:
    with pytest.raises(ValueError, match="unknown camera observation"):
        CameraConstraintSet.from_value(
            {
                "target_ids": ["a"],
                "required_observations": ["better_angle"],
                "metric": "collision",
                "view_goal": "show the contact",
            },
            known_target_ids=("a",),
        )


def test_camera_constraint_set_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="unknown target IDs"):
        CameraConstraintSet.from_value(
            {
                "target_ids": ["invented"],
                "required_observations": ["target_visible"],
                "metric": "oob",
                "view_goal": "show the target",
            },
            known_target_ids=("a",),
        )


def test_metric_acquisition_planner_translates_judge_request() -> None:
    result = MetricSpecificAcquisitionPlanner().plan(
        MetricAcquisitionPlanningRequest(
            metric="support",
            evidence_request=EvidenceRequest(
                target_ids=("top", "base"),
                missing_observations=("support_contact_region",),
                view_goal="show the support contact",
            ),
            known_target_ids=("top", "base"),
        )
    )

    assert result.metric == "support"
    assert result.target_ids == ("top", "base")
    assert "contact_surface_visible" in result.required_observations
    assert "support_chain_visible" in result.required_observations
    assert result.metadata["planner_backend"] == (
        "metric_specific_camera_dsl"
    )


def test_functional_consistency_maps_to_local_usability_observations() -> None:
    result = MetricSpecificAcquisitionPlanner().plan(
        MetricAcquisitionPlanningRequest(
            metric="functional_consistency",
            evidence_request=EvidenceRequest(
                target_ids=("chair", "desk"),
                missing_observations=(
                    "interaction_side_visible",
                ),
                view_goal="show whether the workstation can be used",
            ),
            known_target_ids=("chair", "desk"),
        )
    )

    assert result.target_ids == ("chair", "desk")
    assert result.required_observations == (
        "interaction_side_visible",
    )
    assert result.metric == "functional_consistency"


def test_semantic_placement_maps_to_context_not_physical_support() -> None:
    result = MetricSpecificAcquisitionPlanner().plan(
        MetricAcquisitionPlanningRequest(
            metric="semantic_placement_consistency",
            evidence_request=EvidenceRequest(
                target_ids=("phone", "side_table"),
                missing_observations=(
                    "target_visible",
                    "group_context_visible",
                ),
                view_goal="show whether the phone location makes sense",
            ),
            known_target_ids=("phone", "side_table"),
        )
    )

    assert result.metric == "semantic_placement_consistency"
    assert result.required_observations == (
        "target_visible",
        "group_context_visible",
    )
    assert "support_chain_visible" not in result.required_observations


def test_metric_acquisition_planner_rejects_unknown_target() -> None:
    with pytest.raises(ValueError, match="unknown target IDs"):
        MetricSpecificAcquisitionPlanner().plan(
            MetricAcquisitionPlanningRequest(
                metric="collision",
                evidence_request=EvidenceRequest(
                    target_ids=("invented",),
                    missing_observations=("contact_surface_visible",),
                    view_goal="show the collision contact",
                ),
                known_target_ids=("a", "b"),
            )
        )


def test_metric_acquisition_planner_accepts_scene_scope_without_object_ids():
    result = MetricSpecificAcquisitionPlanner().plan(
        MetricAcquisitionPlanningRequest(
            metric="style_consistency",
            evidence_request=EvidenceRequest(
                target_ids=("scene",),
                missing_observations=("global_context_preserved",),
                view_goal="show the global scene context",
            ),
            known_target_ids=(),
        )
    )

    assert result.target_ids == ("scene",)
    assert result.required_observations == (
        "global_context_preserved",
    )


def test_scene_is_a_valid_global_target_without_scene_object_id() -> None:
    constraints = CameraConstraintSet.from_value(
        {
            "target_ids": ["scene"],
            "required_observations": ["global_context_preserved"],
            "metric": "style_consistency",
            "view_goal": "preserve the global scene context",
        },
        known_target_ids=(),
    )

    assert constraints.target_ids == ("scene",)


def test_camera_constraint_set_rejects_empty_view_goal() -> None:
    with pytest.raises(ValueError, match="view_goal cannot be empty"):
        CameraConstraintSet.from_value(
            {
                "target_ids": ["a"],
                "required_observations": ["target_visible"],
                "metric": "oob",
                "view_goal": "",
            },
            known_target_ids=("a",),
        )


def test_camera_constraint_set_rejects_metric_incompatible_observation() -> None:
    with pytest.raises(ValueError, match="incompatible"):
        CameraConstraintSet.from_value(
            {
                "target_ids": ["a", "b"],
                "required_observations": [
                    "architecture_plane_visible",
                ],
                "metric": "collision",
                "view_goal": "show an unrelated plane",
            },
            known_target_ids=("a", "b"),
        )


def test_camera_constraint_set_rejects_unknown_view_family() -> None:
    with pytest.raises(ValueError, match="unknown view families"):
        CameraConstraintSet.from_value(
            {
                "target_ids": ["a"],
                "required_observations": ["target_visible"],
                "preferred_view_families": ["teleport_camera"],
                "metric": "oob",
                "view_goal": "show the target",
            },
            known_target_ids=("a",),
        )


def test_camera_constraint_set_rejects_unknown_schema_field() -> None:
    with pytest.raises(ValueError, match="unknown fields"):
        CameraConstraintSet.from_value(
            {
                "target_ids": ["a"],
                "required_observations": ["target_visible"],
                "metric": "oob",
                "view_goal": "show the target",
                "free_text_camera_instruction": "move somewhere",
            },
            known_target_ids=("a",),
        )


def test_camera_constraint_set_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        CameraConstraintSet.from_value(
            {
                "schema_version": "future",
                "target_ids": ["a"],
                "required_observations": ["target_visible"],
                "metric": "oob",
                "view_goal": "show the target",
            },
            known_target_ids=("a",),
        )


def test_camera_constraint_set_rejects_inactive_constraint_reference() -> None:
    with pytest.raises(ValueError, match="active constraints"):
        CameraConstraintSet.from_value(
            {
                "target_ids": ["a"],
                "required_observations": ["target_visible"],
                "relaxable_constraints": [
                    "min_projected_coverage",
                ],
                "metric": "oob",
                "view_goal": "show the target",
            },
            known_target_ids=("a",),
        )


def test_judge_evidence_request_maps_stably_to_metric_scoped_dsl() -> None:
    constraints = camera_constraints_from_judge_request(
        EvidenceRequest(
            target_ids=("object", "support"),
            missing_observations=("support_contact_region",),
            view_goal="show the object/support contact",
        ),
        metric="support",
        known_target_ids=("object", "support"),
    )

    assert constraints.metric == "support"
    assert constraints.required_observations == (
        "contact_surface_visible",
        "support_chain_visible",
    )
    assert constraints.preserved_observations == ()
    assert constraints.require_global_anchor is False


def test_judge_evidence_request_cannot_inject_camera_authority() -> None:
    with pytest.raises(ValueError, match="cannot define camera_constraints"):
        camera_constraints_from_judge_request(
            {
                "target_ids": ["a"],
                "missing_observations": ["target_visible"],
                "view_goal": "show a",
                "metadata": {
                    "camera_constraints": {
                        "new_metric_requirement": True,
                    }
                },
            },
            metric="oob",
            known_target_ids=("a",),
        )


def test_judge_metadata_cannot_authorize_relation_subtype_or_conflict() -> None:
    with pytest.raises(ValueError, match="require.*subtype"):
        camera_constraints_from_judge_request(
            {
                "target_ids": ["a", "b"],
                "missing_observations": ["front_back_disambiguated"],
                "view_goal": "show which object faces the other",
                "metadata": {"relation_type": "facing"},
            },
            metric="relation",
            known_target_ids=("a", "b"),
        )

    with pytest.raises(ValueError, match="controller-owned camera authority"):
        camera_constraints_from_judge_request(
            {
                "target_ids": ["a", "b"],
                "missing_observations": ["joint_visibility"],
                "view_goal": "show both targets",
                "metadata": {
                    "constraint_conflicts": [
                        ["joint_visibility", "global_context_preserved"]
                    ]
                },
            },
            metric="object_pairing_consistency",
            known_target_ids=("a", "b"),
        )


def test_relation_camera_metric_requires_known_relation_subtype() -> None:
    with pytest.raises(ValueError, match="require.*subtype"):
        canonical_camera_metric("relation")

    assert (
        canonical_camera_metric("relation", relation_type="facing")
        == "facing"
    )


def test_global_context_observation_always_preserves_packet_anchor() -> None:
    constraints = CameraConstraintSet.from_value(
        {
            "target_ids": ["scene"],
            "required_observations": [
                "group_context_visible",
                "global_context_preserved",
            ],
            "require_global_anchor": False,
            "metric": "style_consistency",
            "view_goal": "retain the global style context",
        },
        known_target_ids=(),
    )

    assert constraints.require_global_anchor is True


def test_dsl_conversion_never_expands_beyond_metric_registry() -> None:
    constraints = camera_constraints_from_judge_request(
        {
            "target_ids": ["a", "b"],
            "missing_observations": ["support_contact_region"],
            "view_goal": "show support",
        },
        metric="support",
        known_target_ids=("a", "b"),
    )

    assert set(constraints.required_observations) <= set(
        METRIC_CAMERA_REQUIREMENTS[
            "support"
        ].allowed_observations
    )


def test_constraints_translate_to_selector_request_without_free_text_schema() -> None:
    constraints = _constraints()

    request = selector_request_from_constraints(
        constraints,
        base_request={"candidate_views": [{"id": "candidate-1"}]},
    )

    assert request["target_ids"] == ["a", "b"]
    assert request["camera_constraints"] == constraints.to_dict()
    assert request["evidence_goal"]["camera_constraints"] == (
        constraints.to_dict()
    )
    assert request["candidate_views"] == [{"id": "candidate-1"}]


def test_selector_request_rejects_conflicting_targets() -> None:
    with pytest.raises(ValueError, match="targets conflict"):
        selector_request_from_constraints(
            _constraints(),
            base_request={"target_ids": ["different"]},
        )


def test_conflict_diagnosis_uses_deterministic_candidate_failures() -> None:
    constraints = _constraints()

    conflicts = diagnose_camera_constraint_conflicts(
        constraints,
        candidate_evaluations=(
            {
                "candidate_id": "local",
                "feasible": False,
                "failed_constraints": [
                    "global_context_preserved"
                ],
                "reason_codes": ["global_anchor_lost"],
            },
            {
                "candidate_id": "global",
                "feasible": False,
                "failed_constraints": [
                    "contact_surface_visible"
                ],
                "reason_codes": ["contact_too_small"],
            },
        ),
    )

    assert len(conflicts) == 1
    assert conflicts[0].constraint_ids == (
        "global_context_preserved",
        "contact_surface_visible",
    )
    assert conflicts[0].candidate_ids == ("local", "global")


def test_feasible_candidate_means_no_constraint_conflict() -> None:
    constraints = _constraints()

    conflicts = diagnose_camera_constraint_conflicts(
        constraints,
        candidate_evaluations=(
            {
                "candidate_id": "usable",
                "feasible": True,
                "failed_constraints": [],
            },
        ),
    )

    assert conflicts == ()


def test_conflict_diagnosis_rejects_inactive_constraint_reference() -> None:
    with pytest.raises(ValueError, match="inactive"):
        diagnose_camera_constraint_conflicts(
            _constraints(),
            candidate_evaluations=(
                {
                    "candidate_id": "bad",
                    "feasible": False,
                    "failed_constraints": [
                        "support_chain_visible",
                        "global_context_preserved",
                    ],
                },
            ),
        )


def test_repair_plans_only_relax_explicitly_relaxable_constraints() -> None:
    constraints = _constraints()
    conflict = CameraConstraintConflict(
        constraint_ids=(
            "contact_surface_visible",
            "global_context_preserved",
        ),
        reason_code="local_coverage_vs_global_context",
    )

    plans = generate_camera_repair_plans(
        constraints,
        conflicts=(conflict,),
    )

    assert len(plans) == 1
    assert plans[0].relaxed_constraints == (
        "global_context_preserved",
    )
    assert "contact_surface_visible" in (
        plans[0].preserved_constraints
    )
    assert plans[0].provenance["source"] == (
        "deterministic_constraint_planner"
    )


def test_repair_plan_must_partition_active_constraints() -> None:
    constraints = _constraints()
    with pytest.raises(ValueError, match="partition active constraints"):
        CameraRepairPlan(
            plan_id="incomplete",
            objective="invalid incomplete plan",
            preserved_constraints=("contact_surface_visible",),
            relaxed_constraints=("global_context_preserved",),
            preferred_view_families=("contact_plane_oblique",),
            required_view_count=1,
            estimated_cost={"full_renders": 1},
        ).validate_against(constraints)


def test_trusted_plan_selection_returns_exact_registered_plan() -> None:
    constraints = _constraints()
    conflict = CameraConstraintConflict(
        constraint_ids=(
            "contact_surface_visible",
            "global_context_preserved",
        ),
        reason_code="local_coverage_vs_global_context",
    )
    plan = generate_camera_repair_plans(
        constraints, conflicts=(conflict,)
    )[0]

    selection = validate_trusted_repair_plan_selection(
        {
            "selected_plan_id": plan.plan_id,
            "reason": "preserves the collision-local observation",
            "backend": "vlm",
        },
        trusted_plans=(plan,),
        constraints=constraints,
    )

    assert selection.plan is plan
    assert selection.to_dict()["selected_plan_id"] == plan.plan_id


def test_trusted_plan_selection_rejects_unknown_plan_id() -> None:
    constraints = _constraints()
    conflict = CameraConstraintConflict(
        constraint_ids=(
            "contact_surface_visible",
            "global_context_preserved",
        ),
        reason_code="conflict",
    )
    plan = generate_camera_repair_plans(
        constraints, conflicts=(conflict,)
    )[0]

    with pytest.raises(ValueError, match="unknown trusted"):
        validate_trusted_repair_plan_selection(
            {
                "selected_plan_id": "invented",
                "reason": "try an unknown plan",
            },
            trusted_plans=(plan,),
        )


@pytest.mark.parametrize(
    "forbidden",
    [
        {"objective": "rewrite trusted plan"},
        {"relaxed_constraints": ["contact_surface_visible"]},
        {"camera_proposal": {"location": [0, 0, 0]}},
        {"verdict": "valid"},
        {"score": 1.0},
    ],
)
def test_trusted_plan_selection_cannot_modify_plan_or_judge(
    forbidden: dict,
) -> None:
    constraints = _constraints()
    conflict = CameraConstraintConflict(
        constraint_ids=(
            "contact_surface_visible",
            "global_context_preserved",
        ),
        reason_code="conflict",
    )
    plan = generate_camera_repair_plans(
        constraints, conflicts=(conflict,)
    )[0]
    response = {
        "selected_plan_id": plan.plan_id,
        "reason": "choose trusted plan",
        **forbidden,
    }

    with pytest.raises(ValueError, match="forbidden or unknown"):
        validate_trusted_repair_plan_selection(
            response,
            trusted_plans=(plan,),
        )


def test_repair_plan_selection_rejects_scene_mutation() -> None:
    constraints = _constraints()
    conflict = CameraConstraintConflict(
        constraint_ids=(
            "contact_surface_visible",
            "global_context_preserved",
        ),
        reason_code="conflict",
    )
    plan = generate_camera_repair_plans(
        constraints, conflicts=(conflict,)
    )[0]

    with pytest.raises(ValueError, match="scene mutation"):
        validate_trusted_repair_plan_selection(
            {
                "selected_plan_id": plan.plan_id,
                "reason": "mutate",
                "provenance": {"scene_patch": {"objects": []}},
            },
            trusted_plans=(plan,),
        )


def test_repair_plan_is_default_vlm_selection_mode() -> None:
    assert DEFAULT_VLM_SELECTION_MODE == "repair_plan"
    assert validate_vlm_selection_mode("repair_plan") == "repair_plan"


def test_freeform_pose_mode_is_fail_closed_by_default() -> None:
    with pytest.raises(ValueError, match="disabled"):
        validate_vlm_selection_mode("freeform_pose")

    assert (
        validate_vlm_selection_mode(
            "freeform_pose", allow_freeform_pose=True
        )
        == "freeform_pose"
    )
