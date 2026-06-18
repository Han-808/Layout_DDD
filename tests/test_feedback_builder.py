from __future__ import annotations

from benchmark.feedback import build_feedback


def test_feedback_builder_is_deterministic_from_report() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": ["desk_1", "bed_1"],
        "schema_failures": [],
        "physical_failures": [
            {
                "type": "collision",
                "objects": ["bed_1", "desk_1"],
                "message": "bed_1 collides with desk_1.",
            }
        ],
        "spatial_relation_failures": [],
    }
    layout = {"objects": [{"object_id": "bed_1"}, {"object_id": "desk_1"}, {"object_id": "chair_1"}]}

    feedback = build_feedback(report, layout, {"task_id": "case_1"})

    assert feedback["repair_targets"] == ["bed_1", "desk_1"]
    assert feedback["locked_objects"] == ["chair_1"]
    assert feedback["violations"] == [
        {
            "category": "physical",
            "type": "collision",
            "message": "bed_1 collides with desk_1.",
            "objects": ["bed_1", "desk_1"],
        }
    ]


def test_feedback_builder_uses_spatial_relation_category() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": ["chair_1"],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [
            {
                "type": "facing",
                "objects": ["chair_1", "desk_1"],
                "message": "chair_1 is not facing desk_1.",
            }
        ],
    }
    layout = {"objects": [{"object_id": "chair_1"}, {"object_id": "desk_1"}]}

    feedback = build_feedback(report, layout, {"task_id": "case_1"})

    assert feedback["violations"][0]["category"] == "spatial_relation"
