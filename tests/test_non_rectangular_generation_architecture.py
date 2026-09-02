from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from benchmark.scene_generation.non_rectangular_multi_room.architecture import (
    COMPILED_ARCHITECTURE_SCHEMA_PATH,
    NonRectangularArchitectureError,
    build_polygon_architecture,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCHEMA = (
    ROOT / "schemas/non_rectangular/compiled_architecture_v1.schema.json"
)


def _wall(
    room_id: str,
    index: int,
    start: list[float],
    end: list[float],
    normal: list[float],
) -> dict:
    return {
        "wall_id": f"{room_id}.wall_{index:03d}",
        "start_xy": start,
        "end_xy": end,
        "inward_normal_xy": normal,
        "height_m": 2.8,
        "thickness_m": 0.1,
    }


def _adjacent_layout() -> dict:
    return {
        "schema_version": "non_rectangular_room_layout_v1",
        "layout_id": "adjacent_fixture",
        "coordinate_frame": {
            "origin": "shared_scene_global",
            "axes": "x_width_y_depth_z_up",
            "handedness": "right_handed",
            "unit": "meter",
            "rotation_unit": "degree",
        },
        "geometry_conventions": {
            "polygon_winding": "counter_clockwise",
            "polygon_closure": "implicit_last_to_first",
            "wall_segment_order": "matches_floor_polygon_edges",
            "inward_normal": "left_of_directed_wall_segment",
        },
        "geometry_tolerance_m": 0.000001,
        "room_count": 2,
        "room_order": ["room_000", "room_001"],
        "rooms": [
            {
                "room_id": "room_000",
                "floor_z_m": 0.0,
                "floor_polygon_xy": [[0, 0], [2, 0], [2, 2], [0, 2]],
                "wall_segments": [
                    _wall("room_000", 0, [0, 0], [2, 0], [0, 1]),
                    _wall("room_000", 1, [2, 0], [2, 2], [-1, 0]),
                    _wall("room_000", 2, [2, 2], [0, 2], [0, -1]),
                    _wall("room_000", 3, [0, 2], [0, 0], [1, 0]),
                ],
            },
            {
                "room_id": "room_001",
                "floor_z_m": 0.0,
                "floor_polygon_xy": [[2, 0], [4, 0], [4, 2], [2, 2]],
                "wall_segments": [
                    _wall("room_001", 0, [2, 0], [4, 0], [0, 1]),
                    _wall("room_001", 1, [4, 0], [4, 2], [-1, 0]),
                    _wall("room_001", 2, [4, 2], [2, 2], [0, -1]),
                    _wall("room_001", 3, [2, 2], [2, 0], [1, 0]),
                ],
            },
        ],
    }


def test_compiled_architecture_schema_is_packaged_identically() -> None:
    source = json.loads(SOURCE_SCHEMA.read_text(encoding="utf-8"))
    packaged = json.loads(
        COMPILED_ARCHITECTURE_SCHEMA_PATH.read_text(encoding="utf-8")
    )

    Draft202012Validator.check_schema(source)
    assert source == packaged


def test_architecture_deduplicates_one_exact_shared_wall() -> None:
    artifact = build_polygon_architecture(_adjacent_layout())

    assert len(artifact["floors"]) == 2
    assert len(artifact["logical_walls"]) == 8
    assert len(artifact["physical_walls"]) == 7
    shared = [item for item in artifact["physical_walls"] if item["shared"]]
    assert len(shared) == 1
    assert shared[0]["segment_global_m"] == [[2.0, 0.0], [2.0, 2.0]]
    assert shared[0]["logical_endpoints"] == [
        {"room_id": "room_000", "wall_id": "room_000.wall_001"},
        {"room_id": "room_001", "wall_id": "room_001.wall_003"},
    ]


def test_architecture_contains_only_benchmark_owned_floor_and_walls() -> None:
    artifact = build_polygon_architecture(_adjacent_layout())

    assert artifact["excluded_architecture"] == [
        "ceiling",
        "doors",
        "windows",
    ]
    assert "objects" not in artifact
    assert all(item["floor_z_m"] == 0.0 for item in artifact["floors"])


def test_coincident_wall_property_drift_fails_closed() -> None:
    value = deepcopy(_adjacent_layout())
    value["rooms"][1]["wall_segments"][3]["thickness_m"] = 0.2

    with pytest.raises(
        NonRectangularArchitectureError,
        match="height/thickness",
    ):
        build_polygon_architecture(value)
