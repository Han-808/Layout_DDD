from __future__ import annotations

import pytest

from benchmark.evaluator import evaluate_oor


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


def test_binary_direction_proximity_and_contact_predicates() -> None:
    scene = {"objects": [_box("a", [1.0, 0.0, 0.5]), _box("b", [0.0, 0.0, 0.5])]}

    assert _result(scene, "right", "a", "b")["checks"][0]["passed"]
    assert _result(scene, "near", "a", "b")["checks"][0]["passed"]
    assert _result(scene, "contact", "a", "b")["checks"][0]["passed"]
    assert not _result(scene, "far", "a", "b")["checks"][0]["passed"]


@pytest.mark.parametrize(
    ("relation", "subject_center"),
    [
        ("left", [-1.0, 0.0, 0.5]),
        ("right", [1.0, 0.0, 0.5]),
        ("in_front", [0.0, -1.0, 0.5]),
        ("behind", [0.0, 1.0, 0.5]),
    ],
)
def test_planar_directions_use_room_frame_pairwise_ordering(relation: str, subject_center: list[float]) -> None:
    scene = {"objects": [_box("subject", subject_center), _box("anchor", [0.0, 0.0, 0.5], yaw=90)]}
    check = _result(scene, relation, "subject", "anchor")["checks"][0]

    assert check["passed"] is True
    assert check["backend"] == "deterministic"
    assert check["evidence"]["reference_frame"] == "room"
    assert check["evidence"]["ordering_state"] == "valid"
    assert check["evidence"]["ordering_score"] >= 0.60


def test_planar_direction_allows_overlap_and_does_not_impose_distance_or_vertical_gates() -> None:
    scene = {
        "objects": [
            _box("nightstand", [4.5, 0.0, 2.5], [0.6, 0.5, 0.6]),
            _box("bed", [3.5, 0.0, 0.85], [2.0, 1.0, 0.8]),
            _box("mug", [-10.0, 4.0, 6.0], [0.1, 0.1, 0.2]),
            _box("bowl", [10.0, 4.0, 0.2], [0.3, 0.3, 0.2]),
        ]
    }

    overlapping = _result(scene, "right", "nightstand", "bed")["checks"][0]
    distant = _result(scene, "left", "mug", "bowl")["checks"][0]

    assert overlapping["passed"] is True
    assert distant["passed"] is True
    assert "max_side_distance" not in overlapping["evidence"]
    assert "vertical_margin" not in distant["evidence"]


def test_planar_direction_reports_boundary_and_never_calls_vlm() -> None:
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"verdict": "valid", "confidence": 1.0, "reason": "should not be called"}

    scene = {"objects": [_box("subject", [0.0, 0.0, 0.5]), _box("anchor", [0.0, 0.0, 0.5])]}
    report = evaluate_oor(
        scene,
        [{"subject_id": "subject", "object_id": "anchor", "type": "right"}],
        render_evidence=["unused.png"],
        vlm_judge=judge,
    )
    check = report["checks"][0]

    assert check["passed"] is False
    assert check["backend"] == "deterministic"
    assert check["evidence"]["ordering_state"] == "boundary"
    assert check["evidence"]["ordering_score"] == pytest.approx(0.5)
    assert calls == []


def test_planar_direction_classifies_opposite_order_as_invalid() -> None:
    scene = {"objects": [_box("subject", [-1.0, 0.0, 0.5]), _box("anchor", [0.0, 0.0, 0.5])]}
    check = _result(scene, "right", "subject", "anchor")["checks"][0]

    assert check["passed"] is False
    assert check["evidence"]["ordering_state"] == "invalid"
    assert check["evidence"]["ordering_score"] <= 0.40


def test_above_requires_xy_closeness() -> None:
    base = _box("b", [0.0, 0.0, 0.5])
    far = {"objects": [_box("a", [10.0, 0.0, 1.5], [0.5, 0.5, 0.5]), base]}
    close = {"objects": [_box("a", [0.0, 0.0, 1.5], [0.5, 0.5, 0.5]), base]}

    assert not _result(far, "above", "a", "b")["checks"][0]["passed"]
    assert _result(close, "above", "a", "b")["checks"][0]["passed"]


def test_aligned_is_positional_while_parallel_is_orientational() -> None:
    scene = {"objects": [_box("a", [0.0, 0.0, 0.5], yaw=90), _box("b", [2.0, 0.0, 0.5], yaw=0)]}

    aligned = _result(scene, "aligned_with", "a", "b")["checks"][0]
    parallel = _result(scene, "parallel", "a", "b")["checks"][0]

    assert aligned["relation"] == "aligned"
    assert aligned["passed"] is True
    assert aligned["evidence"]["alignment_axis"] == "x"
    assert parallel["passed"] is False


def test_inside_and_contains_are_inverse_views_of_same_containment() -> None:
    scene = {
        "objects": [
            _box("small", [0.0, 0.0, 2.0], [1.0, 1.0, 1.0]),
            _box("large", [0.0, 0.0, 2.0], [4.0, 4.0, 4.0]),
        ]
    }

    assert _result(scene, "inside", "small", "large")["checks"][0]["passed"]
    contains = _result(scene, "contains", "large", "small")["checks"][0]
    assert contains["passed"]
    assert contains["evidence"]["inverted_containment_check"] is True


def test_between_ordered_and_around_group_predicates() -> None:
    scene = {
        "objects": [
            _box("left", [-2.0, 0.0, 0.5]),
            _box("middle", [0.0, 0.0, 0.5]),
            _box("right", [2.0, 0.0, 0.5]),
            _box("north", [0.0, 2.0, 0.5]),
            _box("south", [0.0, -2.0, 0.5]),
        ]
    }
    report = evaluate_oor(
        scene,
        [
            {"type": "between", "subject_id": "middle", "object_ids": ["left", "right"]},
            {"type": "ordered", "object_ids": ["left", "middle", "right"], "direction": "left_to_right"},
            {"type": "around", "subject_ids": ["left", "right", "north", "south"], "object_id": "middle"},
        ],
    )

    assert report["num_checks_called"] == 3
    assert report["num_passed"] == 3
    assert report["score"] == 1.0


def test_direction_detector_score_is_evidence_but_relation_score_is_binary() -> None:
    scene = {"objects": [_box("a", [1.2, 0.0, 0.5]), _box("b", [0.0, 0.0, 0.5])]}
    check = _result(scene, "right", "a", "b")["checks"][0]

    assert check["score"] in {0.0, 1.0}
    if "detector_score" in check["evidence"]:
        assert 0.0 <= check["evidence"]["detector_score"] <= 1.0


def test_unknown_explicit_relation_is_binary_vlm_fallback_not_skipped() -> None:
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"verdict": "valid", "confidence": 0.8, "reason": "visible symmetry"}

    scene = {"scene_id": "s", "objects": [_box("a", [0, 0, 0.5]), _box("b", [1, 0, 0.5])]}
    report = evaluate_oor(
        scene,
        [{"subject_id": "a", "object_id": "b", "type": "mirrors", "raw_relation": "A mirrors B."}],
        prompt="A mirrors B.",
        render_evidence=["top.png"],
        vlm_judge=judge,
    )

    assert report["status"] == "ok"
    assert report["score"] == 1.0
    assert report["checks"][0]["backend"] == "vlm"
    assert report["skipped"] == []
    assert calls[0]["natural_language_prompt"] == "A mirrors B."
    assert calls[0]["relation"]["raw_relation"] == "A mirrors B."
    assert calls[0]["render_evidence"] == ["top.png"]


def test_unknown_relation_without_judge_is_incomplete_not_silently_excluded() -> None:
    scene = {"objects": [_box("a", [0, 0, 0.5]), _box("b", [1, 0, 0.5])]}
    report = evaluate_oor(scene, [{"subject_id": "a", "object_id": "b", "type": "mirrors"}])

    assert report["status"] == "incomplete"
    assert report["score"] is None
    assert report["coverage"]["vlm_pending_count"] == 1
    assert report["checks"][0]["status"] == "requires_vlm"


def test_on_top_of_direct_valid_uses_claimed_support_target() -> None:
    scene = {
        "boundary": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
        "scene_height": 3.0,
        "objects": [
            _box("table", [0.0, 0.0, 0.375], [2.0, 2.0, 0.75]),
            _box("clock", [0.0, 0.0, 0.85], [0.2, 0.2, 0.2]),
        ],
    }
    report = evaluate_oor(
        scene,
        [{"relation_id": "oor_clock_table", "subject_id": "clock", "object_id": "table", "type": "on_top_of"}],
    )
    check = report["checks"][0]

    assert report["status"] == "ok"
    assert check["relation_id"] == "oor_clock_table"
    assert check["passed"] is True
    assert check["route"] == "direct_valid"
    assert check["evidence"]["claimed_anchor_first_support_hit_count"] > 0


def test_object_object_on_alias_uses_target_specific_support_check() -> None:
    scene = {
        "boundary": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
        "scene_height": 3.0,
        "objects": [
            _box("table", [0.0, 0.0, 0.375], [2.0, 2.0, 0.75]),
            _box("clock", [0.0, 0.0, 0.85], [0.2, 0.2, 0.2]),
        ],
    }

    check = evaluate_oor(
        scene,
        [{"subject_id": "clock", "object_id": "table", "type": "on"}],
    )["checks"][0]

    assert check["relation"] == "on_top_of"
    assert check["category"] == "target_support"
    assert check["passed"] is True


def test_on_top_of_clear_gap_routes_to_vlm_without_an_invalid_prior() -> None:
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"verdict": "invalid", "confidence": 1.0, "reason": "clear gap"}

    scene = {
        "boundary": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
        "scene_height": 3.0,
        "objects": [
            _box("table", [0.0, 0.0, 0.375], [2.0, 2.0, 0.75]),
            _box("clock", [0.0, 0.0, 1.20], [0.2, 0.2, 0.2]),
        ],
    }
    report = evaluate_oor(
        scene,
        [{"relation_id": "oor_gap", "subject_id": "clock", "object_id": "table", "type": "on_top"}],
        render_evidence=["unused.png"],
        vlm_judge=judge,
    )
    check = report["checks"][0]

    assert check["passed"] is False
    assert check["route"] == "vlm_adjudicated"
    assert check["backend"] == "vlm"
    detector_evidence = check["evidence"]["request"]["detector_evidence"]
    assert "clear_positive_gap_above_claimed_anchor" in detector_evidence["direct_invalid_reasons"]
    assert detector_evidence["candidate_invalid_reasons"] == detector_evidence["direct_invalid_reasons"]
    assert calls[0]["detector_evidence"]["routing_has_invalid_prior"] is False


def test_on_top_of_boundary_case_routes_detector_packet_to_binary_vlm() -> None:
    calls = []

    def judge(request: dict) -> dict:
        calls.append(request)
        return {"verdict": "valid", "confidence": 0.8, "reason": "visual contact"}

    scene = {
        "scene_id": "on_top_boundary",
        "boundary": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
        "scene_height": 3.0,
        "objects": [
            _box("table", [0.0, 0.0, 0.375], [2.0, 2.0, 0.75]),
            # Mild OBB penetration is intentionally ambiguous rather than an
            # automatic fidelity pass or failure.
            _box("clock", [0.0, 0.0, 0.82], [0.2, 0.2, 0.2]),
        ],
    }
    report = evaluate_oor(
        scene,
        [{"relation_id": "oor_boundary", "subject_id": "clock", "object_id": "table", "type": "on_top_of"}],
        prompt="Put the clock on top of the table.",
        render_evidence=["focused.png", "global.png"],
        vlm_judge=judge,
    )
    check = report["checks"][0]

    assert report["status"] == "ok"
    assert check["backend"] == "vlm"
    assert check["route"] == "vlm_adjudicated"
    assert check["relation_id"] == "oor_boundary"
    assert calls[0]["relation"]["relation_id"] == "oor_boundary"
    assert calls[0]["detector_evidence"]["detector"] == "oor_on_top_of_v2"
    assert calls[0]["detector_evidence"]["routing_has_invalid_prior"] is False


def test_relation_identity_survives_pending_vlm_result() -> None:
    scene = {"objects": [_box("a", [0, 0, 0.5]), _box("b", [1, 0, 0.5])]}
    report = evaluate_oor(
        scene,
        [{"relation_id": "oor_mirror", "subject_id": "a", "object_id": "b", "type": "mirrors"}],
    )

    assert report["checks"][0]["relation_id"] == "oor_mirror"
    assert report["unresolved"][0]["relation_id"] == "oor_mirror"


def test_pending_on_top_keeps_target_support_route() -> None:
    scene = {
        "boundary": [[-2, -2], [2, -2], [2, 2], [-2, 2]],
        "scene_height": 3.0,
        "objects": [
            _box("table", [0.0, 0.0, 0.375], [2.0, 2.0, 0.75]),
            _box("clock", [0.0, 0.0, 0.82], [0.2, 0.2, 0.2]),
        ],
    }

    check = evaluate_oor(
        scene,
        [{"relation_id": "oor_pending", "subject_id": "clock", "object_id": "table", "type": "on_top_of"}],
    )["checks"][0]

    assert check["status"] == "requires_vlm"
    assert check["category"] == "target_support"
    assert check["route"] == "requires_vlm"
    assert check["relation_id"] == "oor_pending"


def test_unresolved_identity_is_excluded_from_relation_denominator() -> None:
    scene = {"objects": [_box("a", [1.2, 0.0, 0.5]), _box("b", [0.0, 0.0, 0.5])]}
    report = evaluate_oor(
        scene,
        [
            {"subject_id": "a", "object_id": "b", "type": "near"},
            {"subject_id": "a", "object_id": "ghost", "type": "near"},
        ],
    )

    assert report["status"] == "ok"
    assert report["score"] == pytest.approx(1.0)
    assert report["coverage"]["eligible_count"] == 2
    assert report["coverage"]["resolved_count"] == 1
    assert report["coverage"]["alignment_unresolved_count"] == 1


def test_no_checks_called() -> None:
    report = evaluate_oor({"objects": [_box("a", [0.0, 0.0, 0.5])]})

    assert report["status"] == "no_checks_called"
    assert report["score"] is None
    assert report["checks"] == []
    assert report["coverage"]["eligible_count"] == 0
