from __future__ import annotations

from benchmark.feedback import build_feedback
from benchmark.feedback.feedback_builder import (
    DEFAULT_COLLISION_AVOIDANCE_CONFIG,
    _floor_plan_regions,
    _outside_floor_plan_penalty,
)


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
    boundary_actions = [action for action in actions if action["action"] == "move_inside_boundary" and action["object_id"] == "table_1"]
    assert boundary_actions
    assert boundary_actions[0]["suggested_delta_xy"]
    assert boundary_actions[0]["advisory"] is True
    collision_actions = [action for action in actions if action["action"] == "separate_collision_pair"]
    assert collision_actions
    assert collision_actions[0]["must_remain_inside_floor_plan"] is True
    assert collision_actions[0]["suggested_delta_xy"]
    assert collision_actions[0]["move_object"]
    assert collision_actions[0]["anchor_object"]
    assert collision_actions[0]["reason_code"] == "serious_collision"
    assert "candidate_floor_plan_outside_penalty" in collision_actions[0]
    assert "candidate_total_overlap_volume_m3" in collision_actions[0]
    assert "candidate_overlap_pairs" in collision_actions[0]
    assert collision_actions[0]["cost_mode"] == "normalized_dimensionless"
    assert 0.0 <= collision_actions[0]["outside_cost"] <= 1.0
    assert 0.0 <= collision_actions[0]["overlap_cost"] <= 1.0
    assert 0.0 <= collision_actions[0]["movement_cost"] <= 1.0
    assert collision_actions[0]["total_cost"] >= 0.0
    assert any(action["category"] == "vlm_judge_issue" for action in feedback["violations"])


def test_feedback_builder_uses_aggregate_boundary_when_regions_are_missing() -> None:
    case = {
        "task_id": "case_1",
        "room": {
            "floor_plan": {
                "aggregate_boundary": [[-2, -1], [2, -1], [2, 1], [-2, 1]],
                "geometry_fidelity": "proxy_rectangle",
                "source_kind": "object_position_extent_fallback",
            },
            "boundary": [[-2, -1], [2, -1], [2, 1], [-2, 1]],
            "geometry_fidelity": "proxy_rectangle",
        },
    }
    obj = {"object_id": "chair_1", "center": [0, -0.5, 0.5], "size": [0.8, 0.8, 1.0]}

    regions = _floor_plan_regions(case)

    assert regions
    assert regions[0]["id"] == "__aggregate_floor_plan__"
    assert _outside_floor_plan_penalty(obj, [0, -1.2, 0.5], regions) > 0.0
    assert _outside_floor_plan_penalty(obj, [0, -0.5, 0.5], regions) == 0.0


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


def test_feedback_builder_exposes_general_purpose_advisory_feedback() -> None:
    report = {
        "task_id": "case_1",
        "case_id": "scene_1",
        "iteration": 0,
        "overall_valid": False,
        "bbox_available_rate": 0.5,
        "render_evidence_used": False,
        "json_scene_used": True,
        "repair_targets": ["chair_1"],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {
            "physical_flags": [
                {
                    "type": "room_boundary",
                    "objects": ["chair_1"],
                    "message": "chair_1 is outside.",
                    "confidence": "medium",
                }
            ],
            "bbox_missing_assets": [{"type": "bbox_missing_asset", "asset_id": "plant_1", "message": "missing bbox"}],
        },
        "vlm_judgement": {
            "valid": False,
            "score": 1,
            "score_norm": 0.25,
            "confidence": "medium",
            "judgement_status": "valid_judgement",
            "brief_reasoning": "Object placement needs attention.",
            "issues": [
                {
                    "issue_type": "boundary",
                    "severity": "high",
                    "object_ids": ["chair_1"],
                    "evidence": "outside",
                    "repair_hint": "Move chair_1 inside the floor plan.",
                }
            ],
        },
    }
    layout = {
        "objects": [
            {"object_id": "chair_1", "category": "chair", "center": [3.5, 0.5, 0.45], "size": [0.6, 0.6, 0.9]}
        ]
    }
    case = {"task_id": "case_1", "room": {"boundary": [[0, 0], [3, 0], [3, 3], [0, 3]]}}

    feedback = build_feedback(report, layout, case)

    assert feedback["scene_id"] == "scene_1"
    assert feedback["overall_valid"] is False
    assert feedback["score"] == 1
    assert feedback["score_norm"] == 0.25
    assert feedback["advisory"] is True
    assert feedback["issues"]
    assert feedback["repair_hints"]
    assert feedback["physical_evidence"]["bbox_available_rate"] == 0.5
    assert feedback["vlm_judge_feedback"]["brief_reasoning"] == "Object placement needs attention."
    assert feedback["suggested_actions"]
    assert all(action["advisory"] is True for action in feedback["suggested_actions"])


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


def test_feedback_builder_marks_fallback_boundary_cue_low_confidence() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": [],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {
            "physical_flags": [
                {
                    "type": "room_boundary",
                    "code": "room_boundary_low_confidence",
                    "objects": ["chair_1"],
                    "source_kind": "object_position_extent_fallback",
                    "confidence": "low",
                    "message": "chair_1 is outside fallback boundary.",
                }
            ]
        },
    }
    layout = {"objects": [{"object_id": "chair_1", "center": [2.4, 1.0, 0.5], "size": [1.0, 1.0, 1.0]}]}
    case = {"task_id": "case_1", "room": {"boundary": [[0, 0], [2, 0], [2, 2], [0, 2]], "boundary_source_kind": "object_position_extent_fallback"}}

    feedback = build_feedback(report, layout, case)

    action = next(item for item in feedback["repair_actions"] if item["action"] == "move_inside_boundary")
    assert action["confidence"] == "low"
    assert "fallback-derived" in action["fallback_note"]
    assert action["suggested_center"][0] < 2.4


def test_feedback_builder_keeps_suppressed_fallback_out_of_repair_cues() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": [],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {
            "physical_flags": [
                {
                    "type": "room_boundary",
                    "code": "room_boundary_suppressed",
                    "objects": ["chair_1"],
                    "source_kind": "object_position_extent_fallback",
                    "confidence": "low",
                    "severity": "info",
                    "repair_relevant": False,
                    "message": "chair_1 is outside suppressed fallback boundary.",
                }
            ]
        },
    }
    layout = {"objects": [{"object_id": "chair_1", "center": [2.4, 1.0, 0.5], "size": [1.0, 1.0, 1.0]}]}

    feedback = build_feedback(report, layout, {"task_id": "case_1"})

    assert feedback["repair_targets"] == []
    assert not any(item["action"] == "move_inside_boundary" for item in feedback["repair_actions"])


def test_feedback_builder_aggregates_repeated_collision_moves() -> None:
    flags = [
        {"type": "serious_collision", "objects": ["small", "anchor_a"], "overlap_ratio": 0.9, "message": "collision"},
        {"type": "serious_collision", "objects": ["small", "anchor_b"], "overlap_ratio": 0.8, "message": "collision"},
    ]
    report = {"task_id": "case_1", "iteration": 0, "repair_targets": [], "schema_failures": [], "physical_failures": [], "spatial_relation_failures": [], "debug_evidence": {"physical_flags": flags}}
    layout = {
        "objects": [
            {"object_id": "small", "center": [0.0, 0.0, 0.5], "size": [0.5, 0.5, 1.0]},
            {"object_id": "anchor_a", "center": [0.1, 0.0, 0.5], "size": [1.0, 1.0, 1.0]},
            {"object_id": "anchor_b", "center": [-0.1, 0.0, 0.5], "size": [1.0, 1.0, 1.0]},
        ]
    }

    feedback = build_feedback(report, layout, {"task_id": "case_1"})

    aggregate = next(item for item in feedback["repair_actions"] if item["action"] == "move_object_to_reduce_collisions")
    assert aggregate["object_id"] == "small"
    assert aggregate["collision_count"] == 2
    assert aggregate["contributing_pair_count"] == 2
    assert aggregate["omitted_pair_count"] == 0
    assert aggregate["advisory"] is True


def test_feedback_builder_detects_dense_collision_cluster() -> None:
    ids = ["a", "b", "c", "d"]
    flags = [
        {"type": "serious_collision", "objects": [ids[i], ids[j]], "overlap_ratio": 0.9, "message": "collision"}
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
    ]
    report = {"task_id": "case_1", "iteration": 0, "repair_targets": [], "schema_failures": [], "physical_failures": [], "spatial_relation_failures": [], "debug_evidence": {"physical_flags": flags}}
    layout = {"objects": [{"object_id": object_id, "center": [0.0, 0.0, 0.5], "size": [1.0, 1.0, 1.0]} for object_id in ids]}

    feedback = build_feedback(
        report,
        layout,
        {"task_id": "case_1"},
        benchmark_config={"repair": {"collision_repair": {"max_pair_cues_per_cluster": 3}}},
    )

    cluster = next(item for item in feedback["repair_actions"] if item["action"] == "spread_dense_collision_cluster")
    assert cluster["cluster_id"] == "collision_cluster_0"
    assert len(cluster["objects"]) == 4
    assert cluster["top_pair_count"] == 3
    assert cluster["omitted_pair_count"] == 3
    assert "Do not move the whole cluster together" in cluster["message"]


def test_feedback_height_interval_avoids_below_floor_target() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": [],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {
            "physical_flags": [
                {
                    "type": "above_wall_height",
                    "objects": ["cabinet"],
                    "wall_height": 3.0,
                    "effective_above_wall_tolerance_m": 0.0,
                    "confidence": "high",
                    "message": "too high",
                }
            ]
        },
    }
    layout = {"objects": [{"object_id": "cabinet", "center": [0.0, 0.0, 3.0], "size": [1.0, 1.0, 1.0]}]}

    feedback = build_feedback(report, layout, {"task_id": "case_1", "room": {"floor_z": 0.0, "wall_height": 3.0}})

    action = next(item for item in feedback["repair_actions"] if item["action"] == "adjust_height_within_floor_wall_interval")
    assert action["target_center_z"] >= action["min_center_z"]
    assert action["target_center_z"] <= action["max_center_z"]


def test_feedback_impossible_height_does_not_emit_naive_lowering() -> None:
    report = {
        "task_id": "case_1",
        "iteration": 0,
        "repair_targets": [],
        "schema_failures": [],
        "physical_failures": [],
        "spatial_relation_failures": [],
        "debug_evidence": {
            "physical_flags": [
                {"type": "above_wall_height", "objects": ["tall"], "confidence": "low", "source_kind": "fallback_default", "message": "too high"},
                {
                    "type": "impossible_height_constraint",
                    "code": "fallback_metadata_conflict",
                    "objects": ["tall"],
                    "confidence": "low",
                    "source_kind": "fallback_default",
                    "message": "impossible",
                },
            ]
        },
    }
    layout = {"objects": [{"object_id": "tall", "center": [0.0, 0.0, 2.5], "size": [1.0, 1.0, 4.0]}]}

    feedback = build_feedback(report, layout, {"task_id": "case_1", "room": {"floor_z": 0.0, "wall_height": 3.0}})

    actions = feedback["repair_actions"]
    impossible = next(item for item in actions if item["action"] == "impossible_height_constraint")
    assert impossible["code"] == "fallback_metadata_conflict"
    assert not any(item["action"] == "lower_below_wall_height" for item in actions)
