from __future__ import annotations

from benchmark.feedback import build_feedback
from benchmark.feedback.feedback_builder import DEFAULT_COLLISION_AVOIDANCE_CONFIG


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


def test_feedback_builder_summarizes_debug_evidence_for_repair_prompt() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": ["chair_1"],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {
            "physical_flags": [
                {
                    "type": "below_floor",
                    "objects": ["chair_1"],
                    "message": "chair_1 extends below floor.",
                    "diagnostics": {"huge": "x" * 1000},
                }
            ],
            "group_view_artifacts": [{"diagnostics": {"object_pixel_counts": {"chair_1": 100}}}],
            "judge_input_manifest": {
                "selected_groups": [{"group_id": "group_001", "object_ids": ["chair_1"], "views_sent": {"xy": "view.png"}}],
            },
        },
    }

    feedback = build_feedback(report, {"objects": [{"object_id": "chair_1"}]}, {"task_id": "case_1"})

    assert "debug_evidence" not in feedback
    assert feedback["debug_evidence_summary"]["physical_flags"] == [
        {"type": "below_floor", "objects": ["chair_1"], "message": "chair_1 extends below floor."}
    ]
    assert feedback["debug_evidence_summary"]["selected_groups"] == [
        {"group_id": "group_001", "object_ids": ["chair_1"]}
    ]


def test_feedback_builder_uses_debug_physical_flags_as_repair_targets() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": [],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {
            "physical_flags": [
                {"type": "below_floor", "objects": ["chair_1"], "message": "chair_1 extends below floor."},
                {"type": "room_boundary", "objects": ["table_1"], "message": "table_1 is outside the room."},
            ]
        },
    }
    layout = {"objects": [{"object_id": "chair_1"}, {"object_id": "table_1"}, {"object_id": "lamp_1"}]}

    feedback = build_feedback(report, layout, {"task_id": "case_1"})

    assert feedback["repair_targets"] == ["chair_1", "table_1"]
    assert feedback["locked_objects"] == ["lamp_1"]
    assert [item["category"] for item in feedback["violations"]] == ["physical_debug_flag", "physical_debug_flag"]


def test_feedback_builder_adds_geometry_repair_actions() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": [],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {
            "physical_flags": [
                {"type": "room_boundary", "objects": ["table_1"], "message": "table_1 is outside."},
                {
                    "type": "serious_collision",
                    "objects": ["chair_1", "sofa_1"],
                    "message": "chair_1 overlaps sofa_1.",
                    "overlap_ratio": 0.9,
                },
            ]
        },
        "vlm_judgement": {
            "issues": [
                {
                    "issue_type": "boundary",
                    "severity": "critical",
                    "object_ids": ["table_1"],
                    "evidence": "outside",
                    "repair_hint": "move inside",
                }
            ]
        },
    }
    layout = {
        "objects": [
            {"object_id": "table_1", "category": "table", "center": [2.2, 0.5, 0.5], "size": [0.8, 0.8, 1.0], "region_id": "room_a"},
            {"object_id": "chair_1", "category": "chair", "center": [0.0, 0.0, 0.5], "size": [1.0, 1.0, 1.0]},
            {"object_id": "sofa_1", "category": "sofa", "center": [0.1, 0.0, 0.5], "size": [1.0, 1.0, 1.0]},
        ]
    }
    case = {
        "task_id": "case_1",
        "room": {
            "floor_plan": {
                "regions": [
                    {"id": "room_a", "label": "room_a", "floor_polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}
                ]
            }
        },
    }

    feedback = build_feedback(report, layout, case)

    actions = feedback["repair_actions"]
    assert any(action["action"] == "move_inside_floor_plan" and action["object_id"] == "table_1" for action in actions)
    collision_actions = [action for action in actions if action["action"] == "separate_collision_pair"]
    assert collision_actions
    assert collision_actions[0]["must_remain_inside_floor_plan"] is True
    assert "candidate_floor_plan_outside_penalty" in collision_actions[0]
    assert "candidate_total_overlap_volume_m3" in collision_actions[0]
    assert "candidate_overlap_pairs" in collision_actions[0]
    assert collision_actions[0]["cost_mode"] == "normalized_dimensionless"
    assert 0.0 <= collision_actions[0]["outside_cost"] <= 1.0
    assert 0.0 <= collision_actions[0]["overlap_cost"] <= 1.0
    assert 0.0 <= collision_actions[0]["movement_cost"] <= 1.0
    assert collision_actions[0]["total_cost"] >= 0.0
    assert any(action["category"] == "vlm_judge_issue" for action in feedback["violations"])


def test_feedback_builder_adds_soft_collision_pressure_from_layout_overlap() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": [],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {"physical_flags": []},
    }
    layout = {
        "objects": [
            {"object_id": "chair_1", "category": "chair", "center": [0.0, 0.0, 0.5], "size": [1.0, 1.0, 1.0]},
            {"object_id": "sofa_1", "category": "sofa", "center": [0.8, 0.0, 0.5], "size": [1.0, 1.0, 1.0]},
            {"object_id": "lamp_1", "category": "lamp", "center": [3.0, 0.0, 0.5], "size": [0.5, 0.5, 1.0]},
        ]
    }
    config = {
        "repair": {
            "collision_avoidance": {
                "soft_overlap_ratio": 0.15,
                "soft_min_volume": {
                    "abs_min_volume_m3": 0.001,
                    "object_volume_ratio": 0.005,
                    "scene_volume_ratio": 0.00005,
                    "min_cap_m3": 0.001,
                    "max_cap_m3": 0.03,
                },
            }
        }
    }

    feedback = build_feedback(report, layout, {"task_id": "case_1"}, benchmark_config=config)

    assert feedback["repair_targets"] == ["chair_1", "sofa_1"]
    soft_actions = [action for action in feedback["repair_actions"] if action.get("soft_collision")]
    assert soft_actions
    assert soft_actions[0]["collision_pressure"] == "moderate"
    assert soft_actions[0]["effective_soft_min_volume_m3"] > 0


def test_feedback_soft_collision_uses_scale_aware_min_volume_below_old_fixed_threshold() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": [],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {"physical_flags": []},
    }
    layout = {
        "objects": [
            {"object_id": "tiny_a", "category": "box", "center": [0.0, 0.0, 0.1], "size": [0.2, 0.2, 0.2]},
            {"object_id": "tiny_b", "category": "box", "center": [0.05, 0.0, 0.1], "size": [0.2, 0.2, 0.2]},
        ]
    }

    feedback = build_feedback(report, layout, {"task_id": "case_1"})

    assert feedback["repair_targets"] == ["tiny_a", "tiny_b"]
    soft_actions = [action for action in feedback["repair_actions"] if action.get("soft_collision")]
    assert soft_actions
    assert soft_actions[0]["effective_soft_min_volume_m3"] == 0.001


def test_feedback_soft_collision_uses_scene_scale_term() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": [],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {"physical_flags": []},
    }
    layout = {
        "objects": [
            {"object_id": "chair_1", "category": "chair", "center": [0.0, 0.0, 0.5], "size": [1.0, 1.0, 1.0]},
            {"object_id": "sofa_1", "category": "sofa", "center": [0.8, 0.0, 0.5], "size": [1.0, 1.0, 1.0]},
        ]
    }
    case = {
        "task_id": "case_1",
        "room": {"boundary": [[0, 0], [10, 0], [10, 10], [0, 10]], "wall_height": 3.0},
    }

    feedback = build_feedback(report, layout, case)
    soft_actions = [action for action in feedback["repair_actions"] if action.get("soft_collision")]

    assert soft_actions
    assert soft_actions[0]["effective_soft_min_volume_m3"] == 0.015


def test_feedback_default_collision_costs_are_dimensionless() -> None:
    assert DEFAULT_COLLISION_AVOIDANCE_CONFIG["cost_mode"] == "normalized_dimensionless"
    assert DEFAULT_COLLISION_AVOIDANCE_CONFIG["weights"] == {"outside": 1.0, "overlap": 1.0, "movement": 0.25}
    assert "candidate_outside_cost_weight" not in DEFAULT_COLLISION_AVOIDANCE_CONFIG
    assert "candidate_overlap_cost_weight" not in DEFAULT_COLLISION_AVOIDANCE_CONFIG
    assert "candidate_movement_cost_weight" not in DEFAULT_COLLISION_AVOIDANCE_CONFIG


def test_feedback_builder_can_disable_soft_collision_pressure() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": [],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {"physical_flags": []},
    }
    layout = {
        "objects": [
            {"object_id": "chair_1", "category": "chair", "center": [0.0, 0.0, 0.5], "size": [1.0, 1.0, 1.0]},
            {"object_id": "sofa_1", "category": "sofa", "center": [0.8, 0.0, 0.5], "size": [1.0, 1.0, 1.0]},
        ]
    }
    config = {"repair": {"collision_avoidance": {"enabled": False}}}

    feedback = build_feedback(report, layout, {"task_id": "case_1"}, benchmark_config=config)

    assert feedback["repair_targets"] == []
    assert feedback["repair_actions"] == []
