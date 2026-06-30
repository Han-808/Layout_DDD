from __future__ import annotations

import pytest

from benchmark.workflow.physical_flags import collect_physical_flags


ROOM = {
    "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
    "floor_z": 0.0,
    "wall_height": 3.0,
}


def _obj(object_id: str, center: list[float], size: list[float]) -> dict:
    return {"object_id": object_id, "category": "box", "center": center, "size": size, "yaw": 0}


def test_physical_flags_boundary_floor_and_wall() -> None:
    layout = {
        "objects": [
            _obj("outside", [-0.2, 1.0, 0.5], [1.0, 1.0, 1.0]),
            _obj("below", [1.5, 1.0, -0.1], [0.5, 0.5, 0.5]),
            _obj("above", [2.5, 1.0, 3.2], [0.5, 0.5, 0.5]),
        ]
    }

    flags = collect_physical_flags(layout, {"room": ROOM})
    flag_types = {flag["type"] for flag in flags}

    assert {"room_boundary", "below_floor", "above_wall_height"}.issubset(flag_types)


def test_serious_collision_threshold() -> None:
    base = _obj("base", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0])
    overlap_59 = _obj("overlap_59", [1.41, 1.0, 0.5], [1.0, 1.0, 1.0])
    overlap_61 = _obj("overlap_61", [1.39, 1.0, 0.5], [1.0, 1.0, 1.0])

    flags_59 = collect_physical_flags({"objects": [base, overlap_59]}, {"room": ROOM})
    flags_61 = collect_physical_flags({"objects": [base, overlap_61]}, {"room": ROOM})

    assert not any(flag["type"] == "serious_collision" for flag in flags_59)
    assert any(flag["type"] == "serious_collision" for flag in flags_61)


def test_serious_collision_effective_min_volume_uses_object_volume_term() -> None:
    base = _obj("base", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0])
    overlap_61 = _obj("overlap_61", [1.39, 1.0, 0.5], [1.0, 1.0, 1.0])
    config = {
        "physical_flags": {
            "serious_collision_min_volume": {
                "abs_min_volume_m3": 0.002,
                "object_volume_ratio": 0.5,
                "scene_volume_ratio": 0.0,
                "min_cap_m3": 0.002,
                "max_cap_m3": 1.0,
            }
        }
    }

    flags = collect_physical_flags({"objects": [base, overlap_61]}, {"room": ROOM}, config)
    collision = next(flag for flag in flags if flag["type"] == "serious_collision")

    assert collision["effective_min_collision_volume_m3"] == pytest.approx(0.5)
    assert collision["threshold_source"] == "scale_aware"


def test_serious_collision_effective_min_volume_uses_scene_scale_term() -> None:
    base = _obj("base", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0])
    overlap_61 = _obj("overlap_61", [1.39, 1.0, 0.5], [1.0, 1.0, 1.0])
    config = {
        "physical_flags": {
            "serious_collision_min_volume": {
                "abs_min_volume_m3": 0.002,
                "object_volume_ratio": 0.0,
                "scene_volume_ratio": 0.01,
                "min_cap_m3": 0.002,
                "max_cap_m3": 1.0,
            }
        }
    }

    flags = collect_physical_flags({"objects": [base, overlap_61]}, {"room": ROOM}, config)
    collision = next(flag for flag in flags if flag["type"] == "serious_collision")

    assert collision["effective_min_collision_volume_m3"] == pytest.approx(0.48)


def test_serious_collision_effective_min_volume_falls_back_without_room_height() -> None:
    base = _obj("base", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0])
    overlap_61 = _obj("overlap_61", [1.39, 1.0, 0.5], [1.0, 1.0, 1.0])
    room_without_height = {"boundary": [[0, 0], [4, 0], [4, 4], [0, 4]], "floor_z": 0.0}
    config = {
        "physical_flags": {
            "serious_collision_min_volume": {
                "abs_min_volume_m3": 0.002,
                "object_volume_ratio": 0.0,
                "scene_volume_ratio": 1.0,
                "min_cap_m3": 0.002,
                "max_cap_m3": 1.0,
            }
        }
    }

    flags = collect_physical_flags({"objects": [base, overlap_61]}, {"room": room_without_height}, config)
    collision = next(flag for flag in flags if flag["type"] == "serious_collision")

    assert collision["effective_min_collision_volume_m3"] == pytest.approx(0.002)
    assert collision["smaller_object_volume_m3"] == pytest.approx(1.0)


def test_physical_flags_use_physical_flags_config_and_relative_floor_tolerance() -> None:
    tall = _obj("tall", [1.0, 1.0, 1.92], [0.5, 0.5, 4.0])
    strict_config = {"physical_flags": {"floor_contact_tolerance_m": 0.01, "floor_contact_tolerance_rel_height": 0.0}}
    relative_config = {"physical_flags": {"floor_contact_tolerance_m": 0.01, "floor_contact_tolerance_rel_height": 0.05}}

    strict_flags = collect_physical_flags({"objects": [tall]}, {"room": ROOM}, strict_config)
    relative_flags = collect_physical_flags({"objects": [tall]}, {"room": ROOM}, relative_config)

    assert any(flag["type"] == "below_floor" for flag in strict_flags)
    assert not any(flag["type"] == "below_floor" for flag in relative_flags)


def test_physical_flags_use_floor_plan_region_union() -> None:
    room = {
        "boundary": [[0, 0], [6, 0], [6, 6], [0, 6]],
        "floor_plan": {
            "regions": [
                {"id": "left", "floor_polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]},
                {"id": "right", "floor_polygon": [[4, 0], [6, 0], [6, 2], [4, 2]]},
            ]
        },
        "floor_z": 0.0,
        "wall_height": 3.0,
    }
    in_region = _obj("in_region", [1.0, 1.0, 0.5], [0.5, 0.5, 1.0])
    in_gap_inside_aabb = _obj("in_gap", [3.0, 1.0, 0.5], [0.5, 0.5, 1.0])

    flags = collect_physical_flags({"objects": [in_region, in_gap_inside_aabb]}, {"room": room})

    boundary_objects = {flag["objects"][0] for flag in flags if flag["type"] == "room_boundary"}
    assert "in_region" not in boundary_objects
    assert "in_gap" in boundary_objects
