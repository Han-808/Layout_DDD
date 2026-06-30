from __future__ import annotations

from pathlib import Path

from benchmark.visualization.view_renderer import SimpleBBoxRenderer


CASE = {"room": {"boundary": [[0, 0], [4, 0], [4, 4], [0, 4]], "floor_z": 0.0, "wall_height": 3.0}}
CASE_WITH_REGIONS = {
    "room": {
        "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
        "floor_plan": {
            "regions": [
                {"id": "left", "floor_polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]},
                {"id": "right", "floor_polygon": [[2, 0], [4, 0], [4, 2], [2, 2]]},
            ]
        },
        "floor_z": 0.0,
        "wall_height": 3.0,
    }
}
LAYOUT = {
    "objects": [
        {"object_id": "desk_1", "category": "desk", "center": [1, 1, 0.4], "size": [1, 1, 0.8], "yaw": 0},
        {"object_id": "chair_1", "category": "chair", "center": [2, 1, 0.45], "size": [0.5, 0.5, 0.9], "yaw": 0},
    ]
}


def test_global_and_group_triviews_are_rendered(tmp_path: Path) -> None:
    renderer = SimpleBBoxRenderer(tmp_path)
    group = {"group_id": "group_001", "object_ids": ["desk_1", "chair_1"]}

    global_artifacts = renderer.render_global_top_view(CASE, LAYOUT)
    group_artifacts, flags = renderer.render_group_views(CASE, LAYOUT, group)

    assert not flags
    assert any(item["id"] == "topdown_global_xy" for item in global_artifacts)
    assert (tmp_path / "views" / "global" / "topdown_global_xy.png").exists()
    assert (tmp_path / "views" / "groups" / "group_001" / "group_001_xy.png").exists()
    assert (tmp_path / "views" / "groups" / "group_001" / "group_001_yz.png").exists()
    assert (tmp_path / "views" / "groups" / "group_001" / "group_001_xz.png").exists()
    assert all(item.get("diagnostics", {}).get("camera_candidate") == 0 for item in group_artifacts if item["id"] != "camera_policy")
    diagnostics = next(item["diagnostics"] for item in group_artifacts if item["id"] == "group_001_xy")
    assert diagnostics["render_backend"] == "perspective_bbox_zbuffer"
    assert diagnostics["projection_type"] == "perspective"
    assert diagnostics["camera_position"]
    assert diagnostics["camera_look_at"]
    assert diagnostics["camera_up"]
    assert diagnostics["visible_object_ratio"] > 0
    assert diagnostics["object_pixel_counts"]


def test_invalid_group_view_emits_view_flag_and_records_retry(tmp_path: Path) -> None:
    renderer = SimpleBBoxRenderer(tmp_path)
    group = {"group_id": "group_404", "object_ids": ["missing_1"]}

    artifacts, flags = renderer.render_group_views(
        CASE,
        LAYOUT,
        group,
        {"min_foreground_ratio": 0.01, "min_visible_object_ratio": 0.5, "max_camera_retries": 1},
    )

    assert flags
    assert flags[0]["type"] == "view_invalid"
    assert flags[0]["diagnostics"]["camera_candidate"] == 1
    assert flags[0]["diagnostics"]["render_backend"] == "perspective_bbox_zbuffer"
    assert any(item["id"] == "group_404_xy" for item in artifacts)


def test_renderer_uses_configured_camera_values_and_candidates(tmp_path: Path) -> None:
    config = {
        "render": {
            "fov_degrees": 35.0,
            "camera_candidates": [[0.0, 0.0], [0.5, 0.0]],
            "min_visible_pixel_area": 12.0,
            "distance_scale": 3.0,
        }
    }
    renderer = SimpleBBoxRenderer(tmp_path, benchmark_config=config)
    group = {"group_id": "group_custom", "object_ids": ["missing_1"]}

    _, flags = renderer.render_group_views(
        CASE,
        LAYOUT,
        group,
        {"min_foreground_ratio": 0.01, "min_visible_object_ratio": 0.5, "max_camera_retries": 1},
    )

    diagnostics = flags[0]["diagnostics"]
    assert diagnostics["fov_degrees"] == 35.0
    assert diagnostics["selected_camera_candidate"] == [0.5, 0.0]
    assert diagnostics["resolved_config"]["render"]["min_visible_pixel_area"] == 12.0
    assert diagnostics["resolved_config"]["render"]["distance_scale"] == 3.0


def test_perspective_renderer_records_zbuffer_pixel_ownership(tmp_path: Path) -> None:
    renderer = SimpleBBoxRenderer(tmp_path)
    overlap_layout = {
        "objects": [
            {"object_id": "box_low", "category": "box", "center": [1, 1, 0.5], "size": [1, 1, 1], "yaw": 0},
            {"object_id": "box_high", "category": "box", "center": [1, 1, 1.1], "size": [1, 1, 1], "yaw": 0},
        ]
    }
    group = {"group_id": "group_overlap", "object_ids": ["box_low", "box_high"]}

    artifacts, _ = renderer.render_group_views(CASE, overlap_layout, group)
    diagnostics = next(item["diagnostics"] for item in artifacts if item["id"] == "group_overlap_xy")

    assert diagnostics["object_pixel_counts"]
    assert set(diagnostics["object_pixel_counts"]).issubset({"box_low", "box_high"})


def test_renderer_accepts_multi_region_floor_plan(tmp_path: Path) -> None:
    renderer = SimpleBBoxRenderer(tmp_path)

    artifacts = renderer.render_global_top_view(CASE_WITH_REGIONS, LAYOUT)

    assert (tmp_path / "views" / "global" / "topdown_global_xy.png").exists()
    diagnostics = next(item["diagnostics"] for item in artifacts if item["id"] == "topdown_global_xy")
    assert diagnostics["render_backend"] == "perspective_bbox_zbuffer"
