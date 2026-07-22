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
    never_vlm = profile["structural_validity"]["never_vlm_modules"]

    assert "oob" in profile["structural_validity"]["modules"]
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
        ({"official_mode": True, "detector_only": True}, "mutually exclusive"),
    ],
)
def test_oob_rejects_invalid_configuration(config: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        check_oob(_scene([_obj("inside", [1.0, 1.0, 0.5])]), config)
