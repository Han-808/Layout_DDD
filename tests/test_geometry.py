from __future__ import annotations

from benchmark.evaluator.geometry import bbox_z_bounds, footprint_inside_room, footprints_intersect
from benchmark.evaluator.physical_check import check_physical_validity


ROOM = {
    "unit": "meter",
    "floor_polygon": [[0, 0], [4, 0], [4, 4], [0, 4]],
    "floor_z": 0.0,
    "wall_height": 3.0,
}


def _obj(object_id: str, center: list[float], size: list[float], support_parent: str = "floor") -> dict:
    return {
        "object_id": object_id,
        "category": object_id.split("_")[0],
        "center": center,
        "size": size,
        "yaw": 0.0,
        "support_parent": support_parent,
    }


def test_bbox_z_min_z_max() -> None:
    assert bbox_z_bounds(_obj("box_1", [1, 1, 0.5], [1, 1, 1])) == (0.0, 1.0)


def test_room_boundary_check() -> None:
    assert footprint_inside_room(_obj("box_1", [1, 1, 0.5], [1, 1, 1]), ROOM)
    assert not footprint_inside_room(_obj("box_2", [-0.2, 1, 0.5], [1, 1, 1]), ROOM)


def test_simple_collision_check() -> None:
    assert footprints_intersect(
        _obj("box_1", [1, 1, 0.5], [1, 1, 1]),
        _obj("box_2", [1.2, 1.0, 0.5], [1, 1, 1]),
    )


def test_support_check_valid_and_floating() -> None:
    table = _obj("table_1", [1, 1, 0.5], [1, 1, 1])
    lamp = _obj("lamp_1", [1, 1, 1.1], [0.3, 0.3, 0.2], support_parent="table_1")
    valid, failures, _ = check_physical_validity({"objects": [table, lamp]}, {"room": ROOM}, {})
    assert valid
    assert failures == []

    floating = _obj("box_3", [2, 2, 1.0], [0.5, 0.5, 0.5])
    valid, failures, _ = check_physical_validity({"objects": [floating]}, {"room": ROOM}, {})
    assert not valid
    assert any(failure["type"] == "floating" for failure in failures)
