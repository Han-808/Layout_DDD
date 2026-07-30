from __future__ import annotations

from copy import deepcopy

import pytest

from benchmark.visual_judge.adapters.active_camera import (
    REQUIRED_POSE_VALIDATION_CHECKS,
    ActiveVLMCameraSelector,
)
from benchmark.visual_judge.camera_dsl import (
    METRIC_CAMERA_REQUIREMENTS,
    CameraConstraintSet,
)
from benchmark.visual_judge.interfaces.camera import CameraSelectionRequest


class _VLM:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.model_id = "selector-model"
        self.endpoint = "http://selector.test/v1"

    def select(self, payload):
        self.calls.append(deepcopy(payload))
        if isinstance(self.response, Exception):
            raise self.response
        if callable(self.response):
            return self.response(payload)
        return deepcopy(self.response)


class _Solver:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def realize(self, request, plan):
        self.calls.append((request, plan))
        if self.result is not None:
            return deepcopy(self.result)
        return {
            "outcome": "selected",
            "selected_view_ids": ["candidate-b"],
            "reason": "deterministic plan realization",
            "provenance": {"solver": "test"},
        }


class _PoseValidator:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def validate(self, proposal, request):
        self.calls.append((deepcopy(proposal), request))
        if self.result is not None:
            return deepcopy(self.result)
        return {
            "valid": True,
            "checks": {
                name: True for name in REQUIRED_POSE_VALIDATION_CHECKS
            },
            "pose": deepcopy(proposal),
            "reason_codes": [],
            "provenance": {"validator": "test"},
        }


def _constraints(
    *,
    relaxable: tuple[str, ...] = ("global_context_preserved",),
) -> CameraConstraintSet:
    return CameraConstraintSet(
        target_ids=("a", "b"),
        required_observations=METRIC_CAMERA_REQUIREMENTS[
            "collision"
        ].baseline_observations,
        preserved_observations=("global_context_preserved",),
        preferred_view_families=("contact_plane_oblique",),
        require_joint_visibility=True,
        require_global_anchor=True,
        relaxable_constraints=relaxable,
        metric="collision",
        view_goal="show the collision contact boundary",
    )


def _request(
    *,
    constraints: CameraConstraintSet | None = None,
    candidates=(
        {"id": "candidate-a", "view_family": "contact_plane_oblique"},
        {"id": "candidate-b", "view_family": "contact_plane_oblique"},
    ),
    attempted=(),
    allow_freeform_pose=False,
    context=None,
) -> CameraSelectionRequest:
    return CameraSelectionRequest(
        task="collision",
        metric="collision",
        target_ids=("a", "b"),
        scene={
            "objects": [
                {"id": "a"},
                {"id": "b"},
            ]
        },
        evidence_goal={"view_goal": "show collision contact"},
        existing_visual_evidence=(
            {"path": "initial.png", "view_id": "initial"},
        ),
        budget={
            "max_views_per_round": 2,
            "candidate_budget": 8,
        },
        constraints=(constraints or _constraints()).to_dict(),
        candidate_views=tuple(deepcopy(candidates)),
        evidence_round=1,
        allow_freeform_pose=allow_freeform_pose,
        context={
            "attempted_view_ids": list(attempted),
            **deepcopy(context or {}),
        },
    )


def _repair_context():
    return {
        "camera_constraint_conflicts": [
            {
                "constraint_ids": [
                    "contact_surface_visible",
                    "global_context_preserved",
                ],
                "reason_code": "local_coverage_vs_global_context",
            }
        ]
    }


def _pose():
    return {
        "location": [1.0, 2.0, 3.0],
        "target": [0.0, 0.0, 0.5],
        "lens_mm": 45.0,
        "camera_type": "PERSP",
    }


def test_candidate_only_selects_one_trusted_candidate_once() -> None:
    vlm = _VLM(
        {
            "selected_view_ids": ["candidate-b"],
            "reason": "best trusted preview",
        }
    )
    selector = ActiveVLMCameraSelector(
        vlm,
        selection_mode="candidate_only",
    )

    result = selector.select(_request())

    assert result.selected_view_ids == ("candidate-b",)
    assert result.selected_views[0]["id"] == "candidate-b"
    assert result.backend == "vlm_active"
    assert result.provenance["selection_mode"] == "candidate_only"
    assert len(vlm.calls) == 1
    assert vlm.calls[0]["selection_mode"] == "candidate_only"
    assert vlm.calls[0]["scene_context"]["objects"][0]["id"] == "a"
    assert vlm.calls[0]["scene_access"] == "read_only"


def test_candidate_only_omits_attempted_candidates() -> None:
    vlm = _VLM(
        {
            "selected_view_ids": ["candidate-b"],
            "reason": "only unattempted candidate",
        }
    )
    selector = ActiveVLMCameraSelector(
        vlm,
        selection_mode="candidate_only",
    )

    result = selector.select(_request(attempted=("candidate-a",)))

    assert result.selected_view_ids == ("candidate-b",)
    assert [
        value["id"] for value in vlm.calls[0]["candidate_views"]
    ] == ["candidate-b"]
    assert vlm.calls[0]["attempted_candidate_ids"] == ["candidate-a"]


def test_candidate_only_keeps_history_when_candidate_bank_is_replaced() -> None:
    vlm = _VLM(
        {
            "selected_view_ids": ["candidate-b"],
            "reason": "select from the replacement bank",
        }
    )
    selector = ActiveVLMCameraSelector(
        vlm,
        selection_mode="candidate_only",
    )

    result = selector.select(
        _request(
            candidates=(
                {
                    "id": "candidate-b",
                    "view_family": "contact_plane_oblique",
                },
            ),
            attempted=("retired-candidate",),
        )
    )

    assert result.selected_view_ids == ("candidate-b",)
    assert vlm.calls[0]["attempted_candidate_ids"] == [
        "retired-candidate"
    ]


def test_candidate_only_rejects_unknown_candidate_id() -> None:
    selector = ActiveVLMCameraSelector(
        _VLM(
            {
                "selected_view_ids": ["invented"],
                "reason": "invented candidate",
            }
        ),
        selection_mode="candidate_only",
    )

    with pytest.raises(ValueError, match="invalid selected_view_ids"):
        selector.select(_request())


def test_candidate_only_rejects_reselecting_attempted_candidate() -> None:
    selector = ActiveVLMCameraSelector(
        _VLM(
            {
                "selected_view_ids": ["candidate-a"],
                "reason": "repeat prior view",
            }
        ),
        selection_mode="candidate_only",
    )

    with pytest.raises(ValueError, match="invalid selected_view_ids"):
        selector.select(_request(attempted=("candidate-a",)))


def test_candidate_only_empty_bank_returns_structured_outcome_without_vlm() -> None:
    vlm = _VLM({})
    selector = ActiveVLMCameraSelector(
        vlm,
        selection_mode="candidate_only",
    )

    result = selector.select(_request(candidates=()))

    assert result.outcome == "no_feasible_candidate"
    assert result.reason_codes == ("candidate_bank_empty",)
    assert vlm.calls == []


def test_vlm_selector_exception_propagates_without_policy_fallback() -> None:
    selector = ActiveVLMCameraSelector(
        _VLM(RuntimeError("selector transport failed")),
        selection_mode="candidate_only",
    )

    with pytest.raises(RuntimeError, match="transport failed"):
        selector.select(_request())


@pytest.mark.parametrize(
    "forbidden",
    [
        {"verdict": "valid"},
        {"score": 1.0},
        {"action": {"type": "orbit"}},
        {"camera_proposal": _pose()},
        {"scene_patch": {"objects": []}},
        {"provenance": {"verdict": "invalid"}},
        {"provenance": {"scene_mutation": {}}},
    ],
)
def test_candidate_only_rejects_non_selection_output(
    forbidden,
) -> None:
    selector = ActiveVLMCameraSelector(
        _VLM(
            {
                "selected_view_ids": ["candidate-a"],
                "reason": "bad extra output",
                **forbidden,
            }
        ),
        selection_mode="candidate_only",
    )

    with pytest.raises(ValueError):
        selector.select(_request())


def test_repair_plan_mode_requires_deterministic_solver() -> None:
    with pytest.raises(TypeError, match="CameraRepairSolver"):
        ActiveVLMCameraSelector(_VLM({}))


def test_repair_plan_vlm_only_selects_plan_then_solver_realizes() -> None:
    def choose_first(payload):
        return {
            "selected_plan_id": payload["trusted_repair_plans"][0][
                "plan_id"
            ],
            "reason": "minimum scoped relaxation",
        }

    vlm = _VLM(choose_first)
    solver = _Solver()
    selector = ActiveVLMCameraSelector(
        vlm,
        selection_mode="repair_plan",
        repair_solver=solver,
    )

    result = selector.select(_request(context=_repair_context()))

    assert result.outcome == "selected"
    assert result.selected_view_ids == ("candidate-b",)
    assert result.selected_plan_id is not None
    assert result.provenance["selection_mode"] == "repair_plan"
    assert len(vlm.calls) == 1
    assert len(solver.calls) == 1
    _, trusted_plan = solver.calls[0]
    assert trusted_plan.plan_id == result.selected_plan_id
    assert "camera_proposal" not in vlm.calls[0]


def test_repair_plan_rejects_unknown_plan_id() -> None:
    selector = ActiveVLMCameraSelector(
        _VLM(
            {
                "selected_plan_id": "invented",
                "reason": "unknown plan",
            }
        ),
        selection_mode="repair_plan",
        repair_solver=_Solver(),
    )

    with pytest.raises(ValueError, match="unknown trusted"):
        selector.select(_request(context=_repair_context()))


def test_repair_plan_requires_vlm_explanation() -> None:
    def choose_without_reason(payload):
        return {
            "selected_plan_id": payload["trusted_repair_plans"][0][
                "plan_id"
            ]
        }

    selector = ActiveVLMCameraSelector(
        _VLM(choose_without_reason),
        selection_mode="repair_plan",
        repair_solver=_Solver(),
    )

    with pytest.raises(ValueError, match="non-empty reason"):
        selector.select(_request(context=_repair_context()))


def test_repair_plan_accepts_exact_plan_id_field() -> None:
    def choose_first(payload):
        return {
            "plan_id": payload["trusted_repair_plans"][0]["plan_id"],
            "reason": "choose one trusted plan id",
        }

    selector = ActiveVLMCameraSelector(
        _VLM(choose_first),
        selection_mode="repair_plan",
        repair_solver=_Solver(),
    )

    result = selector.select(_request(context=_repair_context()))

    assert result.selected_plan_id is not None


def test_repair_plan_rejects_vlm_plan_modification() -> None:
    def modify_plan(payload):
        return {
            "selected_plan_id": payload["trusted_repair_plans"][0][
                "plan_id"
            ],
            "reason": "modify plan",
            "objective": "new objective",
        }

    selector = ActiveVLMCameraSelector(
        _VLM(modify_plan),
        selection_mode="repair_plan",
        repair_solver=_Solver(),
    )

    with pytest.raises(ValueError, match="forbidden or unknown"):
        selector.select(_request(context=_repair_context()))


def test_repair_plan_rejects_solver_plan_substitution() -> None:
    solver = _Solver(
        {
            "outcome": "selected",
            "selected_view_ids": ["candidate-b"],
            "selected_plan_id": "different-plan",
            "reason": "substitute",
        }
    )

    def choose_first(payload):
        return {
            "selected_plan_id": payload["trusted_repair_plans"][0][
                "plan_id"
            ],
            "reason": "choose",
        }

    selector = ActiveVLMCameraSelector(
        _VLM(choose_first),
        selection_mode="repair_plan",
        repair_solver=solver,
    )

    with pytest.raises(ValueError, match="changed.*trusted plan"):
        selector.select(_request(context=_repair_context()))


def test_repair_plan_realization_failure_records_attempted_plan() -> None:
    def choose_first(payload):
        return {
            "selected_plan_id": payload["trusted_repair_plans"][0][
                "plan_id"
            ],
            "reason": "try the least-loss trusted plan",
        }

    solver = _Solver(
        {
            "outcome": "no_feasible_candidate",
            "attempted_candidate_ids": [
                "candidate-a",
                "candidate-b",
            ],
            "rejected_candidates": [
                {
                    "candidate_id": "candidate-a",
                    "reason_codes": ["target_not_visible"],
                },
                {
                    "candidate_id": "candidate-b",
                    "reason_codes": ["target_not_visible"],
                },
            ],
            "reason_codes": ["no_feasible_candidate"],
            "reason": "the trusted plan has no realizable pose",
        }
    )
    selector = ActiveVLMCameraSelector(
        _VLM(choose_first),
        selection_mode="repair_plan",
        repair_solver=solver,
    )

    result = selector.select(_request(context=_repair_context()))

    assert result.outcome == "no_feasible_candidate"
    assert result.selected_plan_id is None
    assert len(result.attempted_plan_ids) == 1
    assert result.provenance["selected_plan_id"] == (
        result.attempted_plan_ids[0]
    )


def test_repair_plan_without_relaxable_constraint_is_no_feasible() -> None:
    vlm = _VLM({})
    selector = ActiveVLMCameraSelector(
        vlm,
        selection_mode="repair_plan",
        repair_solver=_Solver(),
    )

    result = selector.select(
        _request(
            constraints=_constraints(relaxable=()),
            context=_repair_context(),
        )
    )

    assert result.outcome == "no_feasible_candidate"
    assert result.reason_codes == ("no_trusted_repair_plan",)
    assert vlm.calls == []


def test_repair_solver_cannot_return_metric_decision() -> None:
    def choose_first(payload):
        return {
            "selected_plan_id": payload["trusted_repair_plans"][0][
                "plan_id"
            ],
            "reason": "choose",
        }

    selector = ActiveVLMCameraSelector(
        _VLM(choose_first),
        selection_mode="repair_plan",
        repair_solver=_Solver(
            {
                "selected_view_ids": ["candidate-a"],
                "reason": "invalid solver result",
                "verdict": "valid",
            }
        ),
    )

    with pytest.raises(ValueError, match="verdict or score"):
        selector.select(_request(context=_repair_context()))


def test_freeform_pose_is_disabled_by_default() -> None:
    with pytest.raises(ValueError, match="disabled"):
        ActiveVLMCameraSelector(
            _VLM({}),
            selection_mode="freeform_pose",
            pose_validator=_PoseValidator(),
        )


def test_freeform_pose_requires_independent_validator() -> None:
    with pytest.raises(TypeError, match="CameraPoseValidator"):
        ActiveVLMCameraSelector(
            _VLM({}),
            selection_mode="freeform_pose",
            allow_freeform_pose=True,
        )


def test_freeform_pose_requires_request_opt_in() -> None:
    vlm = _VLM(
        {"camera_proposal": _pose(), "reason": "new pose"}
    )
    selector = ActiveVLMCameraSelector(
        vlm,
        selection_mode="freeform_pose",
        allow_freeform_pose=True,
        pose_validator=_PoseValidator(),
    )

    with pytest.raises(ValueError, match="request opt-in"):
        selector.select(_request())
    assert vlm.calls == []


def test_freeform_pose_runs_all_validation_checks_once() -> None:
    vlm = _VLM(
        {"camera_proposal": _pose(), "reason": "new pose"}
    )
    validator = _PoseValidator()
    selector = ActiveVLMCameraSelector(
        vlm,
        selection_mode="freeform_pose",
        allow_freeform_pose=True,
        pose_validator=validator,
    )

    result = selector.select(
        _request(allow_freeform_pose=True)
    )

    assert result.camera_proposal == _pose()
    assert len(vlm.calls) == 1
    assert len(validator.calls) == 1
    assert result.provenance["pose_validation"]["checks"] == {
        name: True for name in REQUIRED_POSE_VALIDATION_CHECKS
    }


@pytest.mark.parametrize(
    "proposal",
    [
        {**_pose(), "validated": True},
        {**_pose(), "location": [float("nan"), 0.0, 1.0]},
        {**_pose(), "location": [0.0, 0.0, 0.5]},
        {**_pose(), "lens_mm": -1.0},
        {**_pose(), "camera_type": "UNKNOWN"},
    ],
)
def test_freeform_pose_rejects_invalid_or_self_validated_pose(
    proposal,
) -> None:
    selector = ActiveVLMCameraSelector(
        _VLM(
            {"camera_proposal": proposal, "reason": "bad pose"}
        ),
        selection_mode="freeform_pose",
        allow_freeform_pose=True,
        pose_validator=_PoseValidator(),
    )

    with pytest.raises(ValueError):
        selector.select(_request(allow_freeform_pose=True))


def test_freeform_pose_rejects_validator_failure() -> None:
    selector = ActiveVLMCameraSelector(
        _VLM(
            {"camera_proposal": _pose(), "reason": "pose"}
        ),
        selection_mode="freeform_pose",
        allow_freeform_pose=True,
        pose_validator=_PoseValidator(
            {
                "valid": False,
                "checks": {},
                "reason_codes": ["outside_room"],
            }
        ),
    )

    with pytest.raises(ValueError, match="failed pose validation"):
        selector.select(_request(allow_freeform_pose=True))


def test_freeform_pose_rejects_incomplete_validator_checks() -> None:
    selector = ActiveVLMCameraSelector(
        _VLM(
            {"camera_proposal": _pose(), "reason": "pose"}
        ),
        selection_mode="freeform_pose",
        allow_freeform_pose=True,
        pose_validator=_PoseValidator(
            {
                "valid": True,
                "checks": {
                    "frustum_validation": True,
                },
            }
        ),
    )

    with pytest.raises(ValueError, match="required checks"):
        selector.select(_request(allow_freeform_pose=True))


def test_freeform_pose_rejects_scene_mutation_from_validator() -> None:
    selector = ActiveVLMCameraSelector(
        _VLM(
            {"camera_proposal": _pose(), "reason": "pose"}
        ),
        selection_mode="freeform_pose",
        allow_freeform_pose=True,
        pose_validator=_PoseValidator(
            {
                "valid": True,
                "checks": {
                    name: True
                    for name in REQUIRED_POSE_VALIDATION_CHECKS
                },
                "scene_patch": {},
            }
        ),
    )

    with pytest.raises(ValueError):
        selector.select(_request(allow_freeform_pose=True))
