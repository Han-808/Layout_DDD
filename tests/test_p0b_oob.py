from __future__ import annotations

import pytest

from benchmark.evaluator import evaluate_generic_validity
from benchmark.evaluator.generic_validity.oob import (
    OOB_EVALUATOR_VERSION,
    OOBEvaluationError,
    check_oob,
)
from benchmark.evaluator.profile import resolve_evaluation_profile
from benchmark.scene_io.validate import ArtifactValidationError, validate_generated_scene


def _scene(objects: list[dict], *, boundary: list[list[float]] | None = None, height: float = 4.0) -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "oob_scene",
        "request_id": "oob_case",
        "scene_type": "room",
        "boundary": boundary or [[0, 0], [10, 0], [10, 10], [0, 10]],
        "scene_height": height,
        "objects": objects,
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            }
        },
    }


def _obj(
    object_id: str,
    center: list[float],
    size: list[float] | None = None,
    *,
    rotation: list[float] | None = None,
    category: str = "box",
    description: str = "box",
) -> dict:
    return {
        "id": object_id,
        "category": category,
        "description": description,
        "center": center,
        "size": size or [1.0, 1.0, 1.0],
        "rotation": rotation or [0.0, 0.0, 0.0],
        "geometry_provenance": "bbox_proxy",
        "metadata": {"interactive": False},
    }


class _Judge:
    vlm_control_enabled = False

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.calls = 0
        self.requests: list[dict] = []

    def adjudicate_p0b(self, request: dict) -> dict:
        self.calls += 1
        self.requests.append(request)
        return {"verdict": self.verdict, "confidence": 0.9, "reason": "test"}


ALL_PLANES = ("west_oob", "east_oob", "south_oob", "north_oob", "floor_oob", "ceiling_oob")


# 1. A rotated OBB fully inside is direct valid and does not call VLM.
def test_rotated_obb_inside_is_direct_valid_without_vlm() -> None:
    scene = _scene([_obj("rot", [5.0, 5.0, 1.0], [1.0, 1.0, 1.0], rotation=[30.0, 20.0, 45.0])])
    judge = _Judge("invalid")
    report = check_oob(scene, vlm_judge=judge)
    record = report["objects"][0]

    assert report["evaluator_version"] == OOB_EVALUATOR_VERSION
    assert record["candidate_oob"] is False
    assert record["route"] == "direct_valid_inside"
    assert record["final_verdict"] == "valid"
    assert record["requires_vlm"] is False
    assert judge.calls == 0
    assert report["score"] == 1.0
    assert report["status"] == "checked"


# 2. Parametrized: each of the six planes is flagged in isolation.
@pytest.mark.parametrize(
    "flag,center,size",
    [
        ("west_oob", [0.3, 5.0, 0.5], [1.0, 1.0, 1.0]),
        ("east_oob", [9.7, 5.0, 0.5], [1.0, 1.0, 1.0]),
        ("south_oob", [5.0, 0.3, 0.5], [1.0, 1.0, 1.0]),
        ("north_oob", [5.0, 9.7, 0.5], [1.0, 1.0, 1.0]),
        ("floor_oob", [5.0, 5.0, -0.2], [1.0, 1.0, 1.0]),
        ("ceiling_oob", [5.0, 5.0, 3.8], [1.0, 1.0, 1.0]),
    ],
)
def test_six_plane_flags(flag: str, center: list[float], size: list[float]) -> None:
    scene = _scene([_obj("o", center, size)])
    report = check_oob(scene, {"detector_only": True})
    record = report["objects"][0]

    assert record["plane_flags"][flag] is True
    for other in ALL_PLANES:
        if other != flag:
            assert record["plane_flags"][other] is False
    assert record["candidate_oob"] is True
    assert record["requires_vlm"] is True
    assert report["status"] == "detector_only"
    assert report["score"] is None


# 3. A flagged object judged valid does not reduce the score.
def test_flagged_object_judged_valid_does_not_reduce_score() -> None:
    scene = _scene([_obj("inside", [5.0, 5.0, 0.5]), _obj("edge", [9.7, 5.0, 0.5])])
    judge = _Judge("valid")
    report = check_oob(scene, vlm_judge=judge)

    assert judge.calls == 1
    assert report["candidate_oob_count"] == 1
    assert report["invalid_object_count"] == 0
    assert report["oob_count"] == 0
    assert report["score"] == 1.0
    edge = next(record for record in report["objects"] if record["object_id"] == "edge")
    assert edge["route"] == "vlm_adjudicated"
    assert edge["final_verdict"] == "valid"
    assert edge["affects_oob_score"] is True


# 4. A flagged object judged invalid reduces the score.
def test_flagged_object_judged_invalid_reduces_score() -> None:
    scene = _scene([_obj("inside", [5.0, 5.0, 0.5]), _obj("edge", [9.7, 5.0, 0.5])])
    judge = _Judge("invalid")
    report = check_oob(scene, vlm_judge=judge)

    assert judge.calls == 1
    assert report["candidate_oob_count"] == 1
    assert report["invalid_object_count"] == 1
    assert report["oob_count"] == 1
    assert report["score"] == 0.5
    assert report["oob_rate"] == 0.5


# 5. Official mode fails when adjudication is required but unavailable.
def test_official_mode_without_judge_raises() -> None:
    scene = _scene([_obj("edge", [9.7, 5.0, 0.5])])
    with pytest.raises(OOBEvaluationError, match="no judge"):
        check_oob(scene, {"official_mode": True}, vlm_judge=None)


def test_official_mode_with_bad_verdict_raises() -> None:
    class _BadJudge:
        vlm_control_enabled = False

        def adjudicate_p0b(self, request: dict) -> dict:
            return {"verdict": "insufficient_evidence", "confidence": 0.3, "reason": "occluded"}

    scene = _scene([_obj("edge", [9.7, 5.0, 0.5])])
    with pytest.raises(OOBEvaluationError):
        check_oob(scene, {"official_mode": True}, vlm_judge=_BadJudge())


# 6. Generic evaluator passes prompt, relationships, renders, and local-view provider to OOB.
def test_generic_evaluator_forwards_context_to_oob(tmp_path) -> None:
    overview = tmp_path / "overview.png"
    overview.write_bytes(b"png")
    local_view = tmp_path / "edge_local.png"
    local_view.write_bytes(b"png")
    provider_calls: list[dict] = []

    def local_view_provider(request: dict) -> list[str]:
        provider_calls.append(request)
        return [str(local_view)]

    judge = _Judge("valid")
    scene = _scene([_obj("edge", [9.7, 5.0, 0.5])])
    report = evaluate_generic_validity(
        scene,
        {
            "collision": {"enabled": False},
            "navigability": {"enabled": False},
            "accessibility": {"enabled": False},
            "support": {"enabled": False},
        },
        prompt="Place a box against the east wall.",
        relationships=[{"subject": "edge", "predicate": "against", "object": "east_wall"}],
        render_evidence=[str(overview)],
        vlm_judge=judge,
        local_view_provider=local_view_provider,
    )

    assert judge.calls == 1
    request = judge.requests[0]
    assert request["metric"] == "oob"
    assert request["natural_language_prompt"].startswith("Place a box")
    assert request["extracted_relationships"][0]["predicate"] == "against"
    assert request["detector_evidence"]["plane_flags"]["east_oob"] is True
    assert str(local_view) in request["render_evidence"]
    assert str(overview) in request["render_evidence"]
    assert provider_calls[0]["object_ids"] == ["edge"]
    assert provider_calls[0]["metric"] == "oob"
    assert report["metrics"]["oob"]["score"] == 1.0


# 7. Unresolved score=None is preserved and not coerced into a numeric aggregate.
def test_unresolved_oob_score_none_not_coerced() -> None:
    scene = _scene([_obj("edge", [9.7, 5.0, 0.5])])
    report = evaluate_generic_validity(
        scene,
        {
            "navigability": {"enabled": False},
            "accessibility": {"enabled": False},
            "support": {"enabled": False},
        },
    )

    assert report["metrics"]["oob"]["status"] == "requires_vlm"
    assert report["metrics"]["oob"]["score"] is None
    # None is preserved in per-metric scores rather than coerced to 0.0.
    assert report["metric_scores"]["oob"] is None
    # Numeric metrics remain available as diagnostics, but unresolved OOB blocks
    # the official aggregate instead of being silently removed from its denominator.
    assert report["metrics"]["collision"]["score"] == 1.0
    assert report["active_metric_count"] == 1
    assert report["partial_score"] == 1.0
    assert report["score"] is None
    assert report["status"] == "incomplete"
    assert report["unresolved_metrics"] == ["oob"]


# 8. Semantic threshold is 0: face contact within numerical epsilon is not OOB,
#    but any protrusion beyond the epsilon band is flagged.
def test_face_contact_within_epsilon_is_not_oob() -> None:
    # Flush against the east wall (max_x == 10) and resting exactly on the floor (min_z == 0).
    flush = _scene([_obj("flush", [9.5, 5.0, 0.5], [1.0, 1.0, 1.0])])
    judge = _Judge("invalid")
    report = check_oob(flush, vlm_judge=judge)
    record = report["objects"][0]

    assert record["plane_flags"] == {
        "west_oob": False,
        "east_oob": False,
        "south_oob": False,
        "north_oob": False,
        "floor_oob": False,
        "ceiling_oob": False,
    }
    assert record["candidate_oob"] is False
    assert record["route"] == "direct_valid_inside"
    assert judge.calls == 0
    assert report["score"] == 1.0

    # A protrusion larger than the numerical epsilon is out of bounds.
    poking = _scene([_obj("poke", [9.5 + 1.0e-3, 5.0, 0.5], [1.0, 1.0, 1.0])])
    poke_record = check_oob(poking, {"detector_only": True})["objects"][0]
    assert poke_record["plane_flags"]["east_oob"] is True
    assert poke_record["candidate_oob"] is True


# 9. A wall-attachment category plus a relation claim never auto-exempts a flagged object.
def test_wall_attachment_category_and_relation_claim_still_requires_vlm() -> None:
    scene = _scene(
        [_obj("art", [9.85, 5.0, 2.0], [0.6, 0.1, 0.8], category="painting", description="framed painting on wall")]
    )
    judge = _Judge("valid")
    report = check_oob(
        scene,
        prompt="Hang a painting on the east wall.",
        relationships=[{"subject": "art", "predicate": "on", "object": "east_wall"}],
        vlm_judge=judge,
    )
    record = report["objects"][0]

    assert record["candidate_oob"] is True
    assert record["requires_vlm"] is True
    assert record["route"] == "vlm_adjudicated"
    assert judge.calls == 1


# 10. A failed/third-verdict judge in non-official mode never silently passes.
def test_non_official_failed_judge_stays_unresolved() -> None:
    class _BadJudge:
        vlm_control_enabled = False

        def __init__(self) -> None:
            self.calls = 0

        def adjudicate_p0b(self, request: dict) -> dict:
            self.calls += 1
            return {"verdict": "insufficient_evidence", "confidence": 0.2, "reason": "occluded"}

    scene = _scene([_obj("edge", [9.7, 5.0, 0.5])])
    judge = _BadJudge()
    report = check_oob(scene, {"official_mode": False}, vlm_judge=judge)
    record = report["objects"][0]

    assert judge.calls == 1
    assert record["route"] == "vlm_adjudication_failed"
    assert record["final_verdict"] is None
    assert record["requires_vlm"] is True
    assert record["adjudication_error"]
    assert report["invalid_object_count"] == 0
    assert report["status"] == "requires_vlm"
    assert report["score"] is None
    assert report["coverage"]["vlm_adjudicated_objects"] == 0


# 11. The evaluation profile no longer advertises OOB as a never-VLM module.
def test_profile_does_not_mark_oob_as_never_vlm() -> None:
    profile = resolve_evaluation_profile()
    physical = profile["l1_physical_plausibility"]
    never_vlm = physical["never_vlm_metrics"]

    assert physical["metrics"]["oob"]["enabled"] is True
    assert "oob" not in never_vlm
    assert "collision" not in never_vlm


def test_non_rectangular_room_is_rejected_instead_of_using_its_aabb() -> None:
    scene = _scene(
        [_obj("missing_corner", [3.0, 3.0, 0.5], [0.5, 0.5, 1.0])],
        boundary=[[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]],
    )

    report = check_oob(scene)
    assert report["status"] == "invalid_input"
    assert report["score"] == 0.0
    with pytest.raises(ArtifactValidationError, match="four corners"):
        validate_generated_scene(scene)


def test_direct_evaluator_enforces_resolved_input_room_contract(tmp_path) -> None:
    from evaluate import run_evaluate

    scene = {
        **_scene([_obj("box", [1.0, 1.0, 0.5])]),
        "request_id": "room_contract_case",
    }
    with pytest.raises(ValueError, match="conflicts with the resolved benchmark room"):
        run_evaluate(
            scene=scene,
            out=tmp_path / "report.json",
            scene_request={
                "request_id": "room_contract_case",
                "instruction": "Use a 7 m by 5 m by 3 m room.",
                "room": {
                    "boundary": [[0, 0], [7, 0], [7, 5], [0, 5]],
                    "height": 3.0,
                },
            },
        )


@pytest.mark.parametrize(
    "config,match",
    [
        ({"numerical_eps": -1.0e-6}, "numerical_eps"),
        ({"numerical_eps": float("nan")}, "numerical_eps"),
        ({"floor_contact_tolerance_m": -1.0e-3}, "floor_contact_tolerance_m"),
        ({"floor_contact_tolerance_m": float("nan")}, "floor_contact_tolerance_m"),
        ({"floor_contact_tolerance_m": 1.0e-9}, "must be >= oob.numerical_eps"),
        ({"official_mode": True, "detector_only": True}, "mutually exclusive"),
    ],
)
def test_oob_rejects_invalid_configuration(config: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_oob(_scene([_obj("inside", [1.0, 1.0, 0.5])]), config)


def _floor_sink_scene(depth_m: float) -> dict:
    # Object of unit height whose bottom face sinks ``depth_m`` below the floor.
    return _scene([_obj("obj", [5.0, 5.0, 0.5 - depth_m], [1.0, 1.0, 1.0])])


# 12. The evaluator advertises the v2 contract after the floor-contact fix.
def test_oob_evaluator_version_is_v2() -> None:
    assert OOB_EVALUATOR_VERSION == "oob_p0b_v2"
    report = check_oob(_scene([_obj("inside", [5.0, 5.0, 0.5])]))
    assert report["evaluator_version"] == "oob_p0b_v2"


# 13. Exact floor contact (no penetration) is direct_valid_inside and is not
#     labelled as being within floor-contact tolerance.
def test_exact_floor_contact_is_direct_valid_inside() -> None:
    scene = _floor_sink_scene(0.0)
    judge = _Judge("invalid")
    report = check_oob(scene, vlm_judge=judge)
    record = report["objects"][0]

    assert record["plane_penetration_m"]["floor_oob"] == pytest.approx(0.0, abs=1.0e-12)
    assert record["within_floor_contact_tolerance"] is False
    assert record["plane_flags"]["floor_oob"] is False
    assert record["candidate_oob"] is False
    assert record["route"] == "direct_valid_inside"
    assert record["final_verdict"] == "valid"
    assert judge.calls == 0


# 14. Shallow floor sink (0.2, 1.0, 4.9 mm) within tolerance: raw penetration is
#     preserved, floor_oob is false, the tolerance flag is true, the new
#     direct_valid route is used, and no VLM call is made.
@pytest.mark.parametrize("depth_m", [0.0002, 0.001, 0.0049])
def test_shallow_floor_sink_within_tolerance_bypasses_vlm(depth_m: float) -> None:
    scene = _floor_sink_scene(depth_m)
    judge = _Judge("invalid")
    report = check_oob(scene, vlm_judge=judge)
    record = report["objects"][0]

    assert record["plane_penetration_m"]["floor_oob"] == pytest.approx(depth_m, abs=1.0e-9)
    assert record["floor_penetration_m"] == pytest.approx(depth_m, abs=1.0e-9)
    assert record["plane_flags"]["floor_oob"] is False
    assert record["within_floor_contact_tolerance"] is True
    assert record["candidate_oob"] is False
    assert record["requires_vlm"] is False
    assert record["route"] == "direct_valid_floor_contact_tolerance"
    assert record["final_verdict"] == "valid"
    assert record["affects_oob_score"] is True
    assert judge.calls == 0
    assert report["score"] == 1.0
    assert report["status"] == "checked"
    assert report["coverage"]["direct_valid_objects"] == 1


# 15. Exactly 5.0 mm with floating-point noise is accepted as floor contact and
#     makes no VLM call.
def test_exact_five_millimetre_floor_contact_is_accepted() -> None:
    scene = _floor_sink_scene(0.005)
    judge = _Judge("invalid")
    report = check_oob(scene, vlm_judge=judge)
    record = report["objects"][0]

    # The nominal object bottom is -0.005 with representable float noise.
    assert record["plane_penetration_m"]["floor_oob"] == pytest.approx(0.005, abs=1.0e-9)
    assert record["plane_flags"]["floor_oob"] is False
    assert record["within_floor_contact_tolerance"] is True
    assert record["route"] == "direct_valid_floor_contact_tolerance"
    assert judge.calls == 0


# 16. 5.1 mm and 6.0 mm floor penetration exceed the tolerance and route to VLM.
@pytest.mark.parametrize("depth_m", [0.0051, 0.006])
def test_floor_penetration_just_beyond_tolerance_routes_to_vlm(depth_m: float) -> None:
    scene = _floor_sink_scene(depth_m)
    detector = check_oob(scene, {"detector_only": True})["objects"][0]

    assert detector["plane_penetration_m"]["floor_oob"] == pytest.approx(depth_m, abs=1.0e-9)
    assert detector["plane_flags"]["floor_oob"] is True
    assert detector["within_floor_contact_tolerance"] is False
    assert detector["candidate_oob"] is True
    assert detector["requires_vlm"] is True

    judge = _Judge("invalid")
    routed = check_oob(scene, vlm_judge=judge)["objects"][0]
    assert judge.calls == 1
    assert routed["route"] == "vlm_adjudicated"
    assert routed["final_verdict"] == "invalid"


# 17. 20 mm floor penetration remains a candidate and routes to the VLM.
def test_deep_floor_penetration_beyond_tolerance_routes_to_vlm() -> None:
    scene = _floor_sink_scene(0.020)
    detector = check_oob(scene, {"detector_only": True})["objects"][0]

    assert detector["plane_flags"]["floor_oob"] is True
    assert detector["candidate_oob"] is True
    assert detector["requires_vlm"] is True
    assert detector["plane_penetration_m"]["floor_oob"] == pytest.approx(0.020, abs=1.0e-9)
    assert detector["crossing_depths_m"]["floor_oob"] == pytest.approx(0.020, abs=1.0e-9)
    assert detector["floor_penetration_m"] == pytest.approx(0.020, abs=1.0e-9)

    judge = _Judge("invalid")
    report = check_oob(scene, vlm_judge=judge)
    routed = report["objects"][0]
    assert judge.calls == 1
    assert routed["route"] == "vlm_adjudicated"
    assert routed["final_verdict"] == "invalid"


# 18. The floor-contact tolerance is configurable; a tightened band re-flags a
#     shallow sink that the default would accept.
def test_floor_contact_tolerance_is_configurable() -> None:
    scene = _floor_sink_scene(0.003)

    accepted = check_oob(scene, {"detector_only": True})["objects"][0]
    assert accepted["plane_flags"]["floor_oob"] is False
    assert accepted["within_floor_contact_tolerance"] is True

    tightened = check_oob(
        scene, {"detector_only": True, "floor_contact_tolerance_m": 0.001}
    )["objects"][0]
    assert tightened["plane_flags"]["floor_oob"] is True
    assert tightened["within_floor_contact_tolerance"] is False
    assert tightened["candidate_oob"] is True


# 19. numerical_eps still governs the wall planes independently of the floor
#     contact tolerance: a 1 mm wall protrusion is flagged even though 1 mm floor
#     sink is not.
def test_numerical_eps_still_governs_walls_independent_of_floor_tolerance() -> None:
    # 1 mm past the east wall and 1 mm below the floor in the same object.
    scene = _scene([_obj("box", [9.5 + 0.001, 5.0, 0.5 - 0.001], [1.0, 1.0, 1.0])])
    record = check_oob(scene, {"detector_only": True})["objects"][0]

    assert record["plane_flags"]["east_oob"] is True
    assert record["plane_flags"]["floor_oob"] is False
    assert record["candidate_oob"] is True


# 20. Multi-plane routing: a shallow floor sink plus a real wall crossing must
#     still route because of the wall; the shallow floor tolerance does not make
#     the whole object direct-valid, and the raw floor penetration is preserved.
def test_shallow_floor_plus_wall_oob_routes_to_vlm() -> None:
    # 2 mm floor sink and 10 mm past the east wall.
    scene = _scene([_obj("box", [9.5 + 0.010, 5.0, 0.5 - 0.002], [1.0, 1.0, 1.0])])
    record = check_oob(scene, {"detector_only": True})["objects"][0]

    assert record["plane_penetration_m"]["floor_oob"] == pytest.approx(0.002, abs=1.0e-9)
    assert record["plane_penetration_m"]["east_oob"] == pytest.approx(0.010, abs=1.0e-9)
    assert record["within_floor_contact_tolerance"] is True
    assert record["plane_flags"]["floor_oob"] is False
    assert record["plane_flags"]["east_oob"] is True
    assert record["candidate_oob"] is True
    assert record["requires_vlm"] is True


# 21. All six plane_penetration_m values are computed and non-negative, including
#     the planes that are not crossed.
def test_all_six_plane_penetrations_are_reported() -> None:
    # Poke west (min_x) and ceiling simultaneously.
    scene = _scene(
        [_obj("box", [0.5 - 0.02, 5.0, 4.0 - 0.5 + 0.03], [1.0, 1.0, 1.0])],
        height=4.0,
    )
    record = check_oob(scene, {"detector_only": True})["objects"][0]
    penetration = record["plane_penetration_m"]

    assert set(penetration) == {
        "west_oob",
        "east_oob",
        "south_oob",
        "north_oob",
        "floor_oob",
        "ceiling_oob",
    }
    assert all(value >= 0.0 for value in penetration.values())
    assert penetration["west_oob"] == pytest.approx(0.02, abs=1.0e-9)
    assert penetration["ceiling_oob"] == pytest.approx(0.03, abs=1.0e-9)
    assert penetration["east_oob"] == 0.0
    assert penetration["south_oob"] == 0.0
    assert penetration["north_oob"] == 0.0
    assert penetration["floor_oob"] == 0.0


# 22. Ceiling penetration uses numerical_eps, not the floor tolerance.
def test_ceiling_uses_numerical_eps_not_floor_tolerance() -> None:
    # 1 mm above the ceiling: below the 5 mm floor tolerance but well above eps.
    scene = _scene([_obj("panel", [5.0, 5.0, 4.0 - 0.5 + 0.001], [1.0, 1.0, 1.0])], height=4.0)
    record = check_oob(scene, {"detector_only": True})["objects"][0]

    assert record["plane_penetration_m"]["ceiling_oob"] == pytest.approx(0.001, abs=1.0e-9)
    assert record["plane_flags"]["ceiling_oob"] is True
    assert record["candidate_oob"] is True


# 23. The new direct_valid_floor_contact_tolerance route is counted in the
#     direct-valid coverage total.
def test_floor_contact_route_is_direct_valid_coverage() -> None:
    scene = _scene(
        [
            _obj("inside", [5.0, 5.0, 0.5], [1.0, 1.0, 1.0]),
            _obj("contact", [3.0, 3.0, 0.5 - 0.003], [1.0, 1.0, 1.0]),
        ]
    )
    report = check_oob(scene)
    routes = {record["object_id"]: record["route"] for record in report["objects"]}

    assert routes["inside"] == "direct_valid_inside"
    assert routes["contact"] == "direct_valid_floor_contact_tolerance"
    assert report["coverage"]["direct_valid_objects"] == 2
    assert report["score"] == 1.0


# 24. Official mode does not require a judge for tolerance-only floor contact,
#     because such objects never become candidates.
def test_official_mode_allows_floor_contact_without_judge() -> None:
    scene = _floor_sink_scene(0.003)
    report = check_oob(scene, {"official_mode": True}, vlm_judge=None)
    record = report["objects"][0]

    assert record["route"] == "direct_valid_floor_contact_tolerance"
    assert record["candidate_oob"] is False
    assert report["status"] == "checked"
    assert report["score"] == 1.0


# 25. Official mode still fails closed for a substantive floor candidate with no
#     judge configured.
def test_official_mode_deep_floor_candidate_without_judge_raises() -> None:
    scene = _floor_sink_scene(0.020)
    with pytest.raises(OOBEvaluationError, match="no judge"):
        check_oob(scene, {"official_mode": True}, vlm_judge=None)


# 26. VLM detector evidence carries all v2 fields and keeps the two tolerances
#     distinct while reporting raw penetration as a fact.
def test_detector_evidence_includes_all_v2_fields() -> None:
    scene = _floor_sink_scene(0.020)
    judge = _Judge("valid")
    check_oob(scene, {"floor_contact_tolerance_m": 0.005}, vlm_judge=judge)

    evidence = judge.requests[0]["detector_evidence"]
    for key in (
        "detector",
        "numerical_eps",
        "floor_contact_tolerance_m",
        "plane_penetration_m",
        "within_floor_contact_tolerance",
        "plane_flags",
        "obb_intervals",
        "room",
        "object",
    ):
        assert key in evidence
    assert evidence["detector"] == "oob_p0b_v2"
    assert evidence["numerical_eps"] == pytest.approx(1.0e-6)
    assert evidence["floor_contact_tolerance_m"] == pytest.approx(0.005)
    assert evidence["plane_penetration_m"]["floor_oob"] == pytest.approx(0.020, abs=1.0e-9)
    assert evidence["crossing_depths_m"]["floor_oob"] == pytest.approx(0.020, abs=1.0e-9)
