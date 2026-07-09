from __future__ import annotations

import pytest

from benchmark.evaluator import evaluate_oor, evaluate_scene


def _box(object_id: str, center: list[float], size: list[float] | None = None, yaw: float = 0.0) -> dict:
    return {
        "id": object_id,
        "category": "box",
        "size": size or [1.0, 1.0, 1.0],
        "center": center,
        "rotation": [0.0, 0.0, yaw],
    }


def _result(scene: dict, relation: str, subject: str, anchor: str) -> dict:
    return evaluate_oor(scene, [{"subject_id": subject, "object_id": anchor, "type": relation}])


def test_proximity_near_passes_and_far_fails() -> None:
    near_scene = {"objects": [_box("a", [1.2, 0.0, 0.5]), _box("b", [0.0, 0.0, 0.5])]}
    far_scene = {"objects": [_box("a", [4.0, 0.0, 0.5]), _box("b", [0.0, 0.0, 0.5])]}

    assert _result(near_scene, "near", "a", "b")["checks"][0]["passed"]
    assert not _result(far_scene, "near", "a", "b")["checks"][0]["passed"]


def test_direction_uses_anchor_local_frame() -> None:
    scene = {"objects": [_box("a", [1.2, 0.0, 0.5]), _box("b", [0.0, 0.0, 0.5], yaw=90)]}

    report = _result(scene, "in_front", "a", "b")

    assert report["checks"][0]["passed"]
    assert report["checks"][0]["score"] >= 0.5


def test_above_requires_xy_closeness() -> None:
    base = _box("b", [0.0, 0.0, 0.5])
    far_above = {"objects": [_box("a", [10.0, 0.0, 1.5], size=[0.5, 0.5, 0.5]), base]}
    overlapping_above = {"objects": [_box("a", [0.0, 0.0, 1.5], size=[0.5, 0.5, 0.5]), base]}

    assert not _result(far_above, "above", "a", "b")["checks"][0]["passed"]
    assert _result(overlapping_above, "above", "a", "b")["checks"][0]["passed"]


def test_aligned_with_compares_yaw() -> None:
    same = {"objects": [_box("a", [0.0, 0.0, 0.5], yaw=10), _box("b", [2.0, 0.0, 0.5], yaw=0)]}
    different = {"objects": [_box("a", [0.0, 0.0, 0.5], yaw=90), _box("b", [2.0, 0.0, 0.5], yaw=0)]}

    assert _result(same, "aligned_with", "a", "b")["checks"][0]["passed"]
    assert not _result(different, "aligned_with", "a", "b")["checks"][0]["passed"]


def test_contact_touching_faces_passes_and_gap_fails() -> None:
    touching = {"objects": [_box("a", [1.0, 0.0, 0.5]), _box("b", [0.0, 0.0, 0.5])]}
    gapped = {"objects": [_box("a", [1.06, 0.0, 0.5]), _box("b", [0.0, 0.0, 0.5])]}

    assert _result(touching, "contact", "a", "b")["checks"][0]["passed"]
    assert not _result(gapped, "contact", "a", "b")["checks"][0]["passed"]


def test_face_to_uses_front_direction() -> None:
    facing = {"objects": [_box("a", [0.0, 0.0, 0.5], yaw=0), _box("b", [0.0, -2.0, 0.5])]}
    away = {"objects": [_box("a", [0.0, 0.0, 0.5], yaw=180), _box("b", [0.0, -2.0, 0.5])]}

    assert _result(facing, "face_to", "a", "b")["checks"][0]["passed"]
    assert not _result(away, "face_to", "a", "b")["checks"][0]["passed"]


def test_within_and_out_of() -> None:
    large = _box("b", [0.0, 0.0, 2.0], size=[4.0, 4.0, 4.0])
    inside_scene = {"objects": [_box("a", [0.0, 0.0, 2.0], size=[1.0, 1.0, 1.0]), large]}
    outside_scene = {"objects": [_box("a", [10.0, 0.0, 2.0], size=[1.0, 1.0, 1.0]), large]}

    assert _result(inside_scene, "within", "a", "b")["checks"][0]["passed"]
    assert _result(outside_scene, "out_of", "a", "b")["checks"][0]["passed"]


def test_evaluator_average_over_called_checks() -> None:
    scene = {
        "objects": [
            _box("a", [0.0, 0.0, 0.5], yaw=0),
            _box("b", [0.0, -1.5, 0.5], yaw=0),
            _box("c", [5.0, 5.0, 0.5]),
            _box("d", [10.0, 10.0, 1.0], size=[2.0, 2.0, 2.0]),
            _box("e", [3.0, 0.0, 0.5], yaw=15),
            _box("f", [4.5, 0.0, 0.5], yaw=0),
        ],
        "oor_relations": [
            {"subject_id": "a", "object_id": "b", "type": "near"},
            {"subject_id": "a", "object_id": "b", "type": "face_to"},
            {"subject_id": "c", "object_id": "d", "type": "within"},
            {"subject_id": "e", "object_id": "f", "type": "aligned_with"},
        ],
    }

    report = evaluate_scene(scene)

    assert report["num_checks_called"] == 4
    assert report["num_passed"] == 3
    assert report["num_failed"] == 1
    assert report["overall_score"] == pytest.approx(0.75)


def test_unsupported_relation_is_skipped_and_excluded_from_average() -> None:
    scene = {"objects": [_box("a", [0.0, 0.0, 0.5]), _box("b", [1.0, 0.0, 0.5])]}
    report = evaluate_oor(
        scene,
        [
            {"subject": "a", "object": "b", "relation": "near"},
            {"subject": "a", "object": "b", "relation": "on"},
        ],
    )

    assert report["num_checks_called"] == 1
    assert report["overall_score"] == pytest.approx(1.0)
    assert report["skipped"][0]["relation"] == "on"
    assert report["skipped"][0]["reason"] == "unsupported_relation_in_oor_v0"


def test_no_checks_called() -> None:
    report = evaluate_oor({"objects": [_box("a", [0.0, 0.0, 0.5])]})

    assert report["status"] == "no_checks_called"
    assert report["overall_score"] == 0.0
    assert report["num_checks_called"] == 0
    assert report["checks"] == []


def test_vlm_fallback_request_is_recorded_but_not_executed() -> None:
    scene = {"objects": [_box("a", [0.0, 0.0, 0.5]), _box("b", [1.0, 0.0, 0.5])]}
    report = evaluate_oor(
        scene,
        [{"subject_id": "a", "object_id": "b", "type": "near"}],
        config={"runtime": {"mode": "deterministic_with_vlm_fallback", "vlm_fallback": {"enabled": True}}},
    )

    assert report["evaluation_mode"] == "deterministic"
    assert report["runtime"]["deterministic_only"] is True
    assert report["runtime"]["vlm_fallback"] == {
        "available": False,
        "requested": True,
        "status": "not_implemented",
    }
    assert report["num_checks_called"] == 1
    assert report["overall_score"] == pytest.approx(1.0)
    assert any("not implemented or executed" in note for note in report["notes"])
