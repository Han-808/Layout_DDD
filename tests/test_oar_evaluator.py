from __future__ import annotations

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


def _obj(object_id: str, center: list[float], size: list[float] | None = None) -> dict:
    return {
        "id": object_id,
        "category": "box",
        "center": center,
        "size": size or [0.5, 0.5, 1.0],
        "rotation": [0, 0, 0],
    }


def test_on_floor_passes_and_fails() -> None:
    pass_report = evaluate_oar(_scene([_obj("box", [2.0, 1.0, 0.5])]), relation_specs=[{"subject_id": "box", "type": "on_floor"}])
    fail_report = evaluate_oar(_scene([_obj("box", [2.0, 1.0, 0.8])]), relation_specs=[{"subject_id": "box", "type": "on_floor"}])

    assert pass_report["checks"][0]["passed"] is True
    assert fail_report["checks"][0]["passed"] is False


def test_against_wall_checks_named_wall() -> None:
    scene = _scene([_obj("box", [2.0, 0.25, 0.5])])

    south = evaluate_oar(scene, relation_specs=[{"subject_id": "box", "type": "against_wall", "wall": "south"}])
    north = evaluate_oar(scene, relation_specs=[{"subject_id": "box", "type": "against_wall", "wall": "north"}])

    assert south["checks"][0]["passed"] is True
    assert north["checks"][0]["passed"] is False


def test_near_wall_passes_near_boundary_and_fails_in_center() -> None:
    near = evaluate_oar(_scene([_obj("box", [2.0, 0.60, 0.5])]), relation_specs=[{"subject_id": "box", "type": "near_wall", "wall": "south"}])
    center = evaluate_oar(_scene([_obj("box", [2.0, 1.50, 0.5])]), relation_specs=[{"subject_id": "box", "type": "near_wall", "wall": "south"}])

    assert near["checks"][0]["passed"] is True
    assert center["checks"][0]["passed"] is False


def test_below_wall_requires_near_wall_and_height_proxy() -> None:
    low = _obj("shelf", [2.0, 0.25, 1.0], [0.5, 0.5, 2.0])
    tall = _obj("shelf", [2.0, 0.25, 1.8], [0.5, 0.5, 2.6])

    low_report = evaluate_oar(_scene([low]), relation_specs=[{"subject_id": "shelf", "type": "below_wall", "wall": "south"}])
    tall_report = evaluate_oar(_scene([tall]), relation_specs=[{"subject_id": "shelf", "type": "below_wall", "wall": "south"}])

    assert low_report["checks"][0]["passed"] is True
    assert tall_report["checks"][0]["passed"] is False
    assert tall_report["checks"][0]["evidence"]["height_ok"] is False


def test_at_corner_passes_near_corner_and_fails_in_center() -> None:
    corner = evaluate_oar(_scene([_obj("plant", [3.8, 2.8, 0.25], [0.4, 0.4, 0.5])]), relation_specs=[{"subject_id": "plant", "type": "at_corner", "corner": "northeast"}])
    center = evaluate_oar(_scene([_obj("plant", [2.0, 1.5, 0.25], [0.4, 0.4, 0.5])]), relation_specs=[{"subject_id": "plant", "type": "at_corner", "corner": "northeast"}])

    assert corner["checks"][0]["passed"] is True
    assert center["checks"][0]["passed"] is False


def test_extracts_absolute_relations_from_placement_intent() -> None:
    scene = _scene(
        [
            {
                **_obj("cabinet", [2.0, 0.25, 0.5]),
                "placement_intent": {"absolute_relations": ["against south wall"]},
            },
            {
                **_obj("plant", [3.8, 2.8, 0.25], [0.4, 0.4, 0.5]),
                "placement_intent": {"absolute_relations": ["at northeast corner"]},
            },
        ]
    )

    report = evaluate_oar(scene)

    assert report["num_checks_called"] == 2
    assert {check["relation"] for check in report["checks"]} == {"against_wall", "at_corner"}
    assert report["num_passed"] == 2


def test_extracts_absolute_relations_from_autoregressive_sample() -> None:
    scene = _scene(
        [_obj("table", [2.0, 0.25, 0.5])],
        {
            "samples": [
                {
                    "id": "table",
                    "expected_relations": {"absolute_relations": ["against south wall"]},
                }
            ]
        },
    )

    report = evaluate_oar(scene)

    assert report["num_checks_called"] == 1
    assert report["checks"][0]["relation"] == "against_wall"
    assert report["checks"][0]["passed"] is True


def test_unsupported_relation_is_skipped_and_not_averaged() -> None:
    report = evaluate_oar(
        _scene([_obj("box", [2.0, 1.5, 0.5])]),
        relation_specs=[
            {"subject_id": "box", "relation": "in center region"},
            {"subject_id": "box", "type": "on_floor"},
        ],
    )

    assert report["num_checks_called"] == 1
    assert report["overall_score"] == 1.0
    assert report["skipped"] == [{"relation": "in center region", "subject_id": "box", "reason": "unsupported_relation_in_oar_v0"}]


def test_evaluator_average_uses_all_called_checks() -> None:
    scene = _scene([_obj("box", [2.0, 0.25, 0.5])])
    report = evaluate_oar(
        scene,
        relation_specs=[
            {"subject_id": "box", "type": "on_floor"},
            {"subject_id": "box", "type": "against_wall", "wall": "south"},
            {"subject_id": "box", "type": "near_wall", "wall": "south"},
            {"subject_id": "box", "type": "against_wall", "wall": "north"},
        ],
    )

    assert report["num_checks_called"] == 4
    assert report["num_passed"] == 3
    assert report["num_failed"] == 1
    assert report["overall_score"] == 0.75


def test_no_checks_called_report() -> None:
    report = evaluate_oar(_scene([_obj("box", [2.0, 1.5, 0.5])]))

    assert report["status"] == "no_checks_called"
    assert report["overall_score"] == 0.0
    assert report["num_checks_called"] == 0
