from __future__ import annotations

from dataclasses import replace

import pytest

from benchmark.visual_judge.adapters.deterministic_camera import (
    DeterministicCameraRepairSolver,
    DeterministicLocalCameraSelector,
)
from benchmark.visual_judge.camera_dsl import CameraConstraintSet
from benchmark.visual_judge.camera_repair import (
    generate_camera_repair_plans,
)
from benchmark.visual_judge.interfaces.camera import (
    CameraSelectionRequest,
)
from benchmark.visual_judge.orchestration.camera_acquisition import (
    repair_plans_for_vlm,
)


def _constraints() -> CameraConstraintSet:
    return CameraConstraintSet(
        target_ids=("a", "b"),
        required_observations=(
            "target_visible",
            "joint_visibility",
        ),
        min_projected_coverage=0.02,
        require_joint_visibility=True,
        metric="collision",
        view_goal="show the collision contact",
    )


def _request(candidates, *, attempted=()):
    constraints = _constraints()
    return CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={"objects": [{"id": "a"}, {"id": "b"}]},
        evidence_goal={},
        existing_visual_evidence=(
            {"pose": {"location": [0.0, 0.0, 1.0]}},
        ),
        budget={
            "max_views_per_round": 1,
            "candidate_budget": 8,
        },
        constraints=constraints.to_dict(),
        candidate_views=tuple(candidates),
        context={"attempted_view_ids": list(attempted)},
    )


def _candidate(
    candidate_id: str,
    *,
    feasible: bool = True,
    target_visible: bool = True,
    jointly_visible: bool = True,
    projected_coverage: float = 0.1,
    **extra,
):
    return {
        "id": candidate_id,
        "feasible": feasible,
        "location": [2.0, 2.0, 1.5],
        "target": [0.0, 0.0, 0.5],
        "lens_mm": 52.0,
        "target_ids": ["a", "b"],
        "visibility": {
            "target_visible": target_visible,
            "jointly_visible": jointly_visible,
            "projected_coverage": projected_coverage,
        },
        **extra,
    }


def test_deterministic_selector_filters_then_ranks_a_verifiable_candidate():
    selector = DeterministicLocalCameraSelector()
    candidates = (
        _candidate("infeasible", feasible=False),
        _candidate("low-coverage", projected_coverage=0.001),
        _candidate(
            "contact-best",
            projected_coverage=0.08,
            support_contact_focus=True,
            visibility={
                "target_visible": True,
                "jointly_visible": True,
                "projected_coverage": 0.08,
                "predicted_occlusion": False,
            },
        ),
    )

    result = selector.select(_request(candidates))

    assert result.outcome == "selected"
    assert result.selected_view_ids == ("contact-best",)
    assert result.selected_views == (candidates[2],)
    rejected = {
        item["candidate_id"]: item["reason_codes"]
        for item in result.rejected_candidates
    }
    assert "geometry_infeasible" in rejected["infeasible"]
    assert (
        "projected_coverage_insufficient"
        in rejected["low-coverage"]
    )
    features = result.provenance["candidate_features"]["contact-best"]
    assert features["joint_visibility_estimate"] is True
    assert features["projected_coverage_estimate"] == 0.08
    assert features["contact_support_boundary_cue_estimate"] == 1.0
    assert result.provenance["strategy"] == (
        "geometry_visibility_diversity_v1"
    )


def test_ranking_parameters_are_injectable_and_audited_with_sources():
    selector = DeterministicLocalCameraSelector(
        ranking_config={"target_visibility_bonus": 3.5},
        ranking_config_sources={
            "target_visibility_bonus": "dependency_injection",
        },
    )

    result = selector.select(_request((_candidate("view"),)))

    assert result.provenance["ranking_parameters"][
        "target_visibility_bonus"
    ] == 3.5
    assert result.provenance["ranking_parameters"][
        "projected_coverage_weight"
    ] == 2.0
    assert result.provenance["ranking_parameter_sources"][
        "target_visibility_bonus"
    ] == "dependency_injection"
    assert result.provenance["ranking_parameter_sources"][
        "projected_coverage_weight"
    ] == "default"


def test_attempted_candidate_is_never_selected_again() -> None:
    selector = DeterministicLocalCameraSelector()
    candidates = (
        _candidate("already-used", projected_coverage=0.2),
        _candidate("new-view", projected_coverage=0.04),
    )

    result = selector.select(
        _request(candidates, attempted=("already-used",))
    )

    assert result.selected_view_ids == ("new-view",)
    rejected = result.rejected_candidates[0]
    assert rejected["candidate_id"] == "already-used"
    assert "candidate_already_attempted" in rejected["reason_codes"]


def test_candidate_exhaustion_is_structured_not_an_exception() -> None:
    selector = DeterministicLocalCameraSelector()
    candidate = _candidate("only-view")

    result = selector.select(
        _request((candidate,), attempted=("only-view",))
    )

    assert result.outcome == "no_feasible_candidate"
    assert result.selected_view_ids == ()
    assert result.attempted_candidate_ids == ("only-view",)
    assert result.rejected_candidates[0]["candidate_id"] == "only-view"
    assert "candidate_ranking_exhausted" in result.reason_codes


def test_preserve_all_repair_plan_is_realized_by_deterministic_solver():
    constraints = _constraints()
    plan = generate_camera_repair_plans(constraints)[0]
    candidate = _candidate("repair-view")

    result = DeterministicCameraRepairSolver().realize(
        _request((candidate,)),
        plan,
    )

    assert plan.plan_id == "preserve_all_constraints"
    assert plan.relaxed_constraints == ()
    assert result.outcome == "selected"
    assert result.selected_view_ids == ("repair-view",)


def test_unsupported_observation_returns_structured_no_feasible() -> None:
    constraints = CameraConstraintSet(
        target_ids=("a", "b"),
        required_observations=("interaction_side_visible",),
        metric="functional_consistency",
        view_goal="show the usable interaction side",
    )
    request = CameraSelectionRequest(
        task="functional_consistency",
        metric="functional_consistency",
        target_ids=("a", "b"),
        scene={"objects": [{"id": "a"}, {"id": "b"}]},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={
            "max_views_per_round": 1,
            "candidate_budget": 8,
        },
        constraints=constraints.to_dict(),
        candidate_views=(_candidate("candidate"),),
    )

    result = DeterministicLocalCameraSelector().select(request)

    assert result.outcome == "no_feasible_candidate"
    assert result.attempted_candidate_ids == ("candidate",)
    assert result.reason_codes == (
        "semantic_selection_required",
    )
    assert result.rejected_candidates[0]["failed_constraints"] == [
        "interaction_side_visible"
    ]
    assert (
        result.provenance["candidate_generation_skipped"] is False
    )


def test_opaque_candidate_fails_closed_as_structured_no_feasible() -> None:
    result = DeterministicLocalCameraSelector().select(
        _request(({"id": "opaque"},))
    )

    assert result.outcome == "no_feasible_candidate"
    rejected = result.rejected_candidates[0]
    assert rejected["candidate_id"] == "opaque"
    assert {
        "camera_pose_unverifiable",
        "geometry_feasibility_unverified",
        "target_visibility_unverified",
        "projected_coverage_unverified",
        "joint_visibility_unverified",
    } <= set(rejected["reason_codes"])
    assert set(rejected["failed_constraints"]) == {
        "target_visible",
        "min_projected_coverage",
        "joint_visibility",
    }


def test_candidate_generator_type_error_propagates_as_engineering_failure() -> None:
    def broken_generator(*args, **kwargs):
        del args, kwargs
        raise TypeError("candidate generator signature broke")

    selector = DeterministicLocalCameraSelector(
        candidate_generator=broken_generator
    )

    with pytest.raises(TypeError, match="signature broke"):
        selector.select(_request(()))


def test_unrelated_candidate_generator_value_error_propagates() -> None:
    def broken_generator(*args, **kwargs):
        del args, kwargs
        raise ValueError("scene contract is malformed")

    selector = DeterministicLocalCameraSelector(
        candidate_generator=broken_generator
    )

    with pytest.raises(ValueError, match="scene contract"):
        selector.select(_request(()))


def test_explicit_geometry_exhaustion_is_structured_no_feasible() -> None:
    def exhausted_generator(*args, **kwargs):
        del args, kwargs
        raise ValueError(
            "feasible camera candidate generation could not satisfy the exact "
            "bank size: requested=8, generated=0, metric=collision"
        )

    result = DeterministicLocalCameraSelector(
        candidate_generator=exhausted_generator
    ).select(_request(()))

    assert result.outcome == "no_feasible_candidate"
    assert "candidate_generation_infeasible" in result.reason_codes
    assert result.provenance["generation_outcome"] == (
        "no_feasible_candidate"
    )


def test_legacy_geometry_exhaustion_is_structured_no_feasible() -> None:
    def exhausted_generator(*args, **kwargs):
        del args, kwargs
        raise ValueError(
            "camera candidate generation produced no valid poses"
        )

    result = DeterministicLocalCameraSelector(
        candidate_generator=exhausted_generator,
        candidate_policy="legacy",
    ).select(_request(()))

    assert result.outcome == "no_feasible_candidate"
    assert "candidate_generation_infeasible" in result.reason_codes
    assert result.provenance["generation_outcome"] == (
        "no_feasible_candidate"
    )


def test_generated_candidate_bank_is_validated_and_selectable() -> None:
    constraints = CameraConstraintSet(
        target_ids=("a", "b"),
        required_observations=("target_visible", "joint_visibility"),
        require_joint_visibility=True,
        metric="collision",
        view_goal="show both targets",
    )
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={
            "boundary": [[0, 0], [7, 0], [7, 5], [0, 5]],
            "scene_height": 3.0,
            "objects": [
                {
                    "id": "a",
                    "center": [3.0, 2.5, 0.5],
                    "size": [2.0, 1.5, 1.0],
                    "rotation": [0.0, 0.0, 20.0],
                },
                {
                    "id": "b",
                    "center": [4.0, 2.5, 0.55],
                    "size": [0.8, 0.8, 1.1],
                    "rotation": [0.0, 0.0, 0.0],
                },
            ],
        },
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 2, "candidate_budget": 8},
        constraints=constraints.to_dict(),
    )

    result = DeterministicLocalCameraSelector().select(request)

    assert result.outcome == "selected"
    assert len(result.selected_view_ids) == 2
    assert all(view["candidate_policy"] == "local" for view in result.selected_views)
    assert result.provenance["generation_outcome"] == "generated"
    assert result.provenance["candidate_count"] == 8


def test_generated_camera_bank_uses_exact_group_scope() -> None:
    captured: list[dict] = []

    def generator(request, *, max_candidates, policy):
        captured.append(
            {
                "request": request,
                "max_candidates": max_candidates,
                "policy": policy,
            }
        )
        return [_candidate("group-view")]

    constraints = CameraConstraintSet(
        target_ids=("a", "b"),
        required_observations=(
            "target_visible",
            "joint_visibility",
        ),
        require_joint_visibility=True,
        metric="object_pairing_consistency",
        view_goal="observe one bounded group",
    )
    group_scope = {
        "group_id": "group_001",
        "member_ids": ["a", "b"],
        "target_bounds": {
            "min": [-0.5, -0.5, 0.0],
            "max": [1.5, 0.5, 1.0],
        },
        "focus_center": [0.5, 0.0, 0.5],
        "extent": [2.0, 1.0, 1.0],
    }
    request = CameraSelectionRequest(
        task="object_pairing_consistency",
        metric="object_pairing_consistency",
        target_ids=("a", "b"),
        scene={"objects": [{"id": "a"}, {"id": "b"}]},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1, "candidate_budget": 8},
        constraints=constraints.to_dict(),
        context={
            "group_scope": group_scope,
            "target_bounds": group_scope["target_bounds"],
            "focus_center": group_scope["focus_center"],
            "target_extent": group_scope["extent"],
        },
    )

    selector = DeterministicLocalCameraSelector(
        candidate_generator=generator,
    )
    bank = selector.build_candidate_bank(
        request,
        constraints=constraints,
    )
    result = selector.select(
        replace(request, candidate_views=bank.candidates)
    )

    assert result.outcome == "selected"
    selected = bank.candidates[0]
    assert selected["id"] == "group-view"
    assert selected["group_id"] == "group_001"
    assert selected["target_ids"] == ["a", "b"]
    assert selected["technical_feasibility"] is True
    assert set(selected["pose"]) >= {
        "location",
        "target",
        "lens_mm",
    }
    assert selected["target_visibility_estimate"] is True
    assert selected["joint_visibility_estimate"] is True
    assert selected["projected_coverage_estimate"] == 0.1
    assert selected["view_family"]
    generated = captured[0]["request"]
    assert generated["object_ids"] == ["a", "b"]
    assert generated["target_ids"] == ["a", "b"]
    assert generated["group_scope"] == group_scope
    assert generated["target_bounds"] == group_scope["target_bounds"]
    assert generated["focus_center"] == group_scope["focus_center"]
    assert generated["target_extent"] == group_scope["extent"]
    assert generated["event"]["group_id"] == "group_001"
    assert generated["event"]["object_ids"] == ["a", "b"]
    assert (
        generated["event"]["focus_region"]
        == group_scope["target_bounds"]
    )


def test_trusted_bank_keeps_valid_poses_for_real_constraint_conflict():
    constraints = CameraConstraintSet(
        target_ids=("a", "b"),
        required_observations=(
            "target_visible",
            "joint_visibility",
        ),
        require_joint_visibility=True,
        relaxable_constraints=(
            "target_visible",
            "joint_visibility",
        ),
        metric="collision",
        view_goal="show both targets together",
    )
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={"objects": [{"id": "a"}, {"id": "b"}]},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1, "candidate_budget": 8},
        constraints=constraints.to_dict(),
        candidate_views=(
            _candidate(
                "target-conflict",
                target_visible=False,
                jointly_visible=True,
            ),
            _candidate(
                "joint-conflict",
                target_visible=True,
                jointly_visible=False,
            ),
        ),
    )
    selector = DeterministicLocalCameraSelector()

    bank = selector.build_candidate_bank(
        request,
        constraints=constraints,
    )
    selection = selector.select(
        replace(request, candidate_views=bank.candidates)
    )
    plans = repair_plans_for_vlm(
        constraints=constraints,
        deterministic_selection=selection,
    )

    assert {item["id"] for item in bank.candidates} == {
        "target-conflict",
        "joint-conflict",
    }
    assert all(
        item["technical_feasibility"] is True
        for item in bank.candidates
    )
    estimates = {
        item["id"]: (
            item["target_visibility_estimate"],
            item["joint_visibility_estimate"],
        )
        for item in bank.candidates
    }
    assert estimates == {
        "target-conflict": (False, True),
        "joint-conflict": (True, False),
    }
    assert selection.outcome == "no_feasible_candidate"
    assert {
        item["candidate_id"]: item["reason_codes"]
        for item in selection.rejected_candidates
    } == {
        "target-conflict": ["target_not_visible"],
        "joint-conflict": ["joint_visibility_unavailable"],
    }
    assert {plan.relaxed_constraints for plan in plans} == {
        ("target_visible",),
        ("joint_visibility",),
    }


def test_trusted_bank_keeps_renderable_pose_with_unresolved_evidence_proxy():
    candidate = _candidate(
        "proxy-unresolved",
        proxy_framing={
            "proxy_bounds_fit": False,
            "all_corners_in_front": True,
        },
    )
    candidate.pop("visibility")
    selector = DeterministicLocalCameraSelector()

    bank = selector.build_candidate_bank(
        _request((candidate,)),
        constraints=_constraints(),
    )

    assert [item["id"] for item in bank.candidates] == [
        "proxy-unresolved"
    ]
    trusted = bank.candidates[0]
    assert trusted["technical_feasibility"] is True
    assert trusted["target_visibility_estimate"] is None
    assert trusted["joint_visibility_estimate"] is None
    assert trusted["technical_features"]["proxy_bounds_fit"] is False
    assert bank.rejected_candidates == ()


def test_rejected_constraints_drive_conflict_repair_plans() -> None:
    constraints = CameraConstraintSet(
        target_ids=("a", "b"),
        required_observations=("target_visible", "joint_visibility"),
        relaxable_constraints=("target_visible", "joint_visibility"),
        metric="collision",
        view_goal="make both targets jointly visible",
    )
    request = CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={"objects": [{"id": "a"}, {"id": "b"}]},
        evidence_goal={},
        existing_visual_evidence=(),
        budget={"max_views_per_round": 1, "candidate_budget": 8},
        constraints=constraints.to_dict(),
        candidate_views=(
            _candidate(
                "target-failure",
                target_visible=False,
                jointly_visible=True,
            ),
            _candidate(
                "joint-failure",
                target_visible=True,
                jointly_visible=False,
            ),
        ),
    )

    selection = DeterministicLocalCameraSelector().select(request)
    plans = repair_plans_for_vlm(
        constraints=constraints,
        deterministic_selection=selection,
    )

    assert selection.outcome == "no_feasible_candidate"
    assert {
        item["candidate_id"]: tuple(item["failed_constraints"])
        for item in selection.rejected_candidates
    } == {
        "target-failure": ("target_visible",),
        "joint-failure": ("joint_visibility",),
    }
    assert {plan.relaxed_constraints for plan in plans} == {
        ("target_visible",),
        ("joint_visibility",),
    }
