from __future__ import annotations

import pytest

from benchmark.evaluator import evaluate_oar


def _scene(objects: list[dict], extra: dict | None = None) -> dict:
    scene = {
        "scene_id": "oar_test",
        "scene_type": "room",
        "boundary": [[0, 0], [4, 0], [4, 3], [0, 3]],
        "scene_height": 2.8,
        "objects": objects,
    }
    if extra:
        scene.update(extra)
    return scene


def _obj(object_id: str, center: list[float], size: list[float] | None = None, yaw: float = 0.0) -> dict:
    return {
        "id": object_id,
        "category": "box",
        "center": center,
        "size": size or [0.5, 0.5, 1.0],
        "rotation": [0, 0, yaw],
    }


def test_floor_wall_and_corner_predicates() -> None:
    scene = _scene([_obj("box", [0.25, 0.25, 0.5])])
    report = evaluate_oar(
        scene,
        [
            {"subject_id": "box", "type": "on_floor"},
            {"subject_id": "box", "type": "against_wall", "wall": "west"},
            {"subject_id": "box", "type": "near_wall", "wall": "south"},
            {"subject_id": "box", "type": "at_corner", "corner": "southwest"},
            {"subject_id": "box", "type": "near_corner", "corner": "southwest"},
        ],
    )

    assert report["num_checks_called"] == 5
    assert report["num_passed"] == 5


def test_room_center_and_named_region() -> None:
    scene = _scene([_obj("center", [2.0, 1.5, 0.5]), _obj("east", [3.6, 1.5, 0.5])])
    center = evaluate_oar(scene, [{"subject_id": "center", "type": "room_center"}])
    east = evaluate_oar(scene, [{"subject_id": "east", "type": "room_region", "region": "east"}])

    assert center["checks"][0]["passed"]
    assert east["checks"][0]["passed"]
    assert east["checks"][0]["evidence"]["coordinate_frame"].startswith("west=-x")


def test_along_wall_checks_distance_and_long_axis_orientation() -> None:
    parallel = _obj("parallel", [2.0, 0.25, 0.5], [1.5, 0.4, 1.0], yaw=0)
    perpendicular = _obj("perpendicular", [2.0, 0.25, 0.5], [1.5, 0.4, 1.0], yaw=90)
    scene = _scene([parallel, perpendicular])

    passing = evaluate_oar(scene, [{"subject_id": "parallel", "type": "along_wall", "wall": "south"}])
    failing = evaluate_oar(scene, [{"subject_id": "perpendicular", "type": "along_wall", "wall": "south"}])

    assert passing["checks"][0]["passed"]
    assert not failing["checks"][0]["passed"]


def test_attachment_proxies_are_pending_without_a_vlm_judge() -> None:
    scene = _scene(
        [
            _obj("picture", [0.1, 1.5, 1.4], [0.2, 1.0, 1.0]),
            _obj("far_picture", [2.0, 1.5, 1.4], [0.2, 1.0, 1.0]),
            _obj("lamp", [2.0, 1.5, 2.3], [0.5, 0.5, 1.0]),
            _obj("low_lamp", [2.0, 1.5, 0.5], [0.5, 0.5, 1.0]),
        ]
    )
    report = evaluate_oar(
        scene,
        [
            {"relation_id": "mounted_near", "subject_id": "picture", "type": "mounted_on_wall", "wall": "west"},
            {"relation_id": "mounted_far", "subject_id": "far_picture", "type": "mounted_on_wall", "wall": "west"},
            {"relation_id": "ceiling_near", "subject_id": "lamp", "type": "attached_to_ceiling"},
            {"relation_id": "ceiling_far", "subject_id": "low_lamp", "type": "hung_from_ceiling"},
        ],
    )

    assert report["status"] == "incomplete"
    assert report["score"] is None
    assert report["num_checks_called"] == 0
    assert report["coverage"]["vlm_pending_count"] == 4
    assert {item["relation_id"] for item in report["unresolved"]} == {
        "mounted_near",
        "mounted_far",
        "ceiling_near",
        "ceiling_far",
    }
    assert all(item["status"] == "requires_vlm" for item in report["checks"])
    assert all(item["passed"] is None for item in report["checks"])
    assert all(item["evidence"]["reason"] == "vlm_judge_not_configured" for item in report["checks"])
    proxy_outcomes = [item["evidence"]["detector_evidence"]["proxy_checks_passed"] for item in report["checks"]]
    assert proxy_outcomes == [True, False, True, False]


def test_attachment_proxy_outcome_never_replaces_binary_vlm_verdict() -> None:
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        proxy_passed = request["detector_evidence"]["proxy_checks_passed"]
        return {
            "verdict": "invalid" if proxy_passed else "valid",
            "confidence": 0.95,
            "reason": "visual attachment judgement",
        }

    scene = _scene(
        [
            _obj("picture", [0.1, 1.5, 1.4], [0.2, 1.0, 1.0]),
            _obj("far_picture", [2.0, 1.5, 1.4], [0.2, 1.0, 1.0]),
            _obj("lamp", [2.0, 1.5, 2.3], [0.5, 0.5, 1.0]),
            _obj("low_lamp", [2.0, 1.5, 0.5], [0.5, 0.5, 1.0]),
        ]
    )
    report = evaluate_oar(
        scene,
        [
            {"subject_id": "picture", "type": "mounted_on_wall", "wall": "west"},
            {"subject_id": "far_picture", "type": "mounted_on_wall", "wall": "west"},
            {"subject_id": "lamp", "type": "attached_to_ceiling"},
            {"subject_id": "low_lamp", "type": "hung_from_ceiling"},
        ],
        prompt="Place the pictures on the wall and hang the lamps from the ceiling.",
        render_evidence=["perspective.png"],
        vlm_judge=judge,
    )

    assert report["status"] == "ok"
    assert report["score"] == pytest.approx(0.5)
    assert report["num_checks_called"] == 4
    assert [item["passed"] for item in report["checks"]] == [False, True, False, True]
    assert all(item["backend"] == "vlm" for item in report["checks"])
    assert all(item["route"] == "vlm_adjudicated" for item in report["checks"])
    assert [call["detector_evidence"]["proxy_checks_passed"] for call in calls] == [True, False, True, False]
    assert all(call["vlm_role"] == "judge" for call in calls)
    assert all(call["decision_contract"] == "relation_binary_v1" for call in calls)
    assert all(call["judge_method"] == "adjudicate_relation" for call in calls)


def test_attachment_with_judge_but_no_render_is_pending() -> None:
    scene = _scene([_obj("picture", [0.1, 1.5, 1.4], [0.2, 1.0, 1.0])])
    report = evaluate_oar(
        scene,
        [{"subject_id": "picture", "type": "mounted_on_wall", "wall": "west"}],
        vlm_judge=lambda request: {"verdict": "valid", "confidence": 1.0},
    )

    check = report["checks"][0]
    assert report["status"] == "incomplete"
    assert check["status"] == "requires_vlm"
    assert check["evidence"]["reason"] == "render_evidence_not_available"
    assert check["evidence"]["detector_evidence"]["proxy"] == "obb_to_wall_attachment"
    assert check["evidence"]["detector_evidence"]["detector"] == "oar_attachment_proxy_v2"
    assert check["evidence"]["detector_evidence"]["routing_has_invalid_prior"] is False


def test_ceiling_attachment_without_numeric_scene_height_still_routes_to_vlm() -> None:
    scene = _scene([_obj("lamp", [2.0, 1.5, 2.3], [0.5, 0.5, 1.0])])
    scene.pop("scene_height")
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"verdict": "valid", "confidence": 0.9, "reason": "visibly attached"}

    report = evaluate_oar(
        scene,
        [{"subject_id": "lamp", "type": "hung_from_ceiling"}],
        render_evidence=["perspective.png"],
        vlm_judge=judge,
    )

    assert report["status"] == "ok"
    assert report["checks"][0]["passed"] is True
    assert calls[0]["detector_evidence"]["reason"] == "scene_height_unavailable"
    assert calls[0]["detector_evidence"]["proxy_checks_passed"] is None


def test_string_relations_are_normalized_from_placement_intent_and_samples() -> None:
    scene = _scene(
        [
            {
                **_obj("cabinet", [2.0, 0.25, 0.5]),
                "placement_intent": {"absolute_relations": ["against south wall"]},
            },
            _obj("plant", [3.75, 2.75, 0.5]),
        ],
        {
            "samples": [
                {
                    "id": "plant",
                    "expected_relations": {"absolute_relations": ["at northeast corner"]},
                }
            ]
        },
    )

    report = evaluate_oar(scene)

    assert report["num_checks_called"] == 2
    assert {item["relation"] for item in report["checks"]} == {"against_wall", "at_corner"}
    corner_check = next(item for item in report["checks"] if item["relation"] == "at_corner")
    assert corner_check["passed"] is True
    assert corner_check["evidence"]["requested_corner"] == "northeast"


def test_unknown_explicit_oar_relation_uses_binary_vlm_fallback() -> None:
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"verdict": "invalid", "confidence": 0.9, "reason": "not beneath the window"}

    scene = _scene([_obj("box", [2.0, 1.5, 0.5])])
    report = evaluate_oar(
        scene,
        [{"subject_id": "box", "type": "under_window", "target": "north_window"}],
        prompt="Put the box under the north window.",
        render_evidence=["perspective.png"],
        vlm_judge=judge,
    )

    assert report["status"] == "ok"
    assert report["score"] == 0.0
    assert report["checks"][0]["backend"] == "vlm"
    assert report["skipped"] == []
    assert calls[0]["family"] == "oar"


def test_unknown_relation_without_vlm_is_incomplete() -> None:
    scene = _scene([_obj("box", [2.0, 1.5, 0.5])])
    report = evaluate_oar(scene, [{"subject_id": "box", "type": "under_window"}])

    assert report["status"] == "incomplete"
    assert report["score"] is None
    assert report["coverage"]["vlm_pending_count"] == 1


def test_relation_identity_survives_deterministic_and_pending_oar_checks() -> None:
    scene = _scene([_obj("box", [2.0, 1.5, 0.5])])
    report = evaluate_oar(
        scene,
        [
            {"relation_id": "oar_floor", "subject_id": "box", "type": "on_floor"},
            {"relation_id": "oar_window", "subject_id": "box", "type": "under_window"},
        ],
    )

    assert report["checks"][0]["relation_id"] == "oar_floor"
    assert report["checks"][1]["relation_id"] == "oar_window"
    assert report["unresolved"][0]["relation_id"] == "oar_window"


def test_unresolved_subject_is_excluded_from_denominator() -> None:
    scene = _scene([_obj("box", [2.0, 0.25, 0.5])])
    report = evaluate_oar(
        scene,
        [
            {"subject_id": "box", "type": "against_wall", "wall": "south"},
            {"subject_id": "ghost", "type": "on_floor"},
        ],
    )

    assert report["status"] == "ok"
    assert report["score"] == pytest.approx(1.0)
    assert report["coverage"]["eligible_count"] == 2
    assert report["coverage"]["resolved_count"] == 1
    assert report["coverage"]["alignment_unresolved_count"] == 1


def test_no_checks_called_report() -> None:
    report = evaluate_oar(_scene([_obj("box", [2.0, 1.5, 0.5])]))

    assert report["status"] == "no_checks_called"
    assert report["score"] is None
    assert report["coverage"]["eligible_count"] == 0
