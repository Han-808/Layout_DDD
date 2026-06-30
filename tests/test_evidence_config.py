from __future__ import annotations

from benchmark.evidence_config import resolve_runtime_evidence_config


def test_runtime_evidence_config_resolves_scene_and_group_scale() -> None:
    case = {"room": {"boundary": [[0, 0], [6, 0], [6, 4], [0, 4]], "wall_height": 3.0}}
    layout = {
        "objects": [
            {"object_id": "desk_1", "center": [1, 1, 0.4], "size": [1.0, 0.5, 0.8]},
            {"object_id": "chair_1", "center": [2, 1, 0.45], "size": [0.6, 0.6, 0.9]},
            {"object_id": "bed_1", "center": [5, 3, 0.5], "size": [2.0, 1.4, 1.0]},
        ]
    }
    config = {"render": {"distance_scale": 2.0, "far": 20.0}, "physical_flags": {"floor_contact_tolerance_rel_height": 0.1}}

    resolved = resolve_runtime_evidence_config(config, case, layout, {"object_ids": ["desk_1", "chair_1"]})

    assert resolved["room_extent_m"] == 6.0
    assert resolved["room_height_m"] == 3.0
    assert resolved["group_extent_m"] < resolved["scene_bbox_extent_m"]
    assert resolved["render"]["effective_camera_distance_m"] >= resolved["group_extent_m"] * 2.0
    assert resolved["render"]["far"] >= 20.0
    assert resolved["physical_flags"]["effective_floor_contact_tolerance_m"] >= 0.05
