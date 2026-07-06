from __future__ import annotations

from benchmark.workflow.nodes import _apply_deterministic_repair_actions, _layout_change_summary


def test_layout_change_summary_ignores_rounding_noise() -> None:
    previous = {
        "objects": [
            {"object_id": "chair_1", "center": [1.0, 2.0, 0.5], "size": [1.0, 1.0, 1.0], "yaw": 0},
        ]
    }
    current = {
        "objects": [
            {"object_id": "chair_1", "center": [1.0005, 2.0004, 0.5003], "size": [1.0, 1.0, 1.0004], "yaw": 0.2},
        ]
    }

    summary = _layout_change_summary(previous, current, ["chair_1"])

    assert summary["changed_object_ids"] == []
    assert summary["changed_repair_targets"] == []
    assert summary["repair_noop"] is True


def test_layout_change_summary_counts_meaningful_target_motion() -> None:
    previous = {
        "objects": [
            {"object_id": "chair_1", "center": [1.0, 2.0, 0.5], "size": [1.0, 1.0, 1.0], "yaw": 0},
        ]
    }
    current = {
        "objects": [
            {"object_id": "chair_1", "center": [1.03, 2.0, 0.5], "size": [1.0, 1.0, 1.0], "yaw": 0},
        ]
    }

    summary = _layout_change_summary(previous, current, ["chair_1"])

    assert summary["changed_object_ids"] == ["chair_1"]
    assert summary["changed_repair_targets"] == ["chair_1"]
    assert summary["repair_noop"] is False


def test_deterministic_repair_actions_are_advisory_noop() -> None:
    previous = {
        "objects": [
            {"object_id": "cabinet_1", "center": [0.0, 0.0, 2.4], "size": [1.0, 1.0, 1.4], "yaw": 0},
        ]
    }
    feedback = {
        "repair_actions": [
            {
                "action": "lower_below_wall_height",
                "object_id": "cabinet_1",
                "suggested_center": [0.0, 0.0, 2.0],
            }
        ]
    }

    repaired, summary = _apply_deterministic_repair_actions(previous, feedback)

    assert repaired is previous
    assert repaired["objects"][0]["center"] == [0.0, 0.0, 2.4]
    assert summary["enabled"] is False
    assert summary["changed_object_ids"] == []
    assert summary["num_changed_objects"] == 0
    assert summary["candidate_action_count"] == 1
    assert previous["objects"][0]["center"] == [0.0, 0.0, 2.4]
