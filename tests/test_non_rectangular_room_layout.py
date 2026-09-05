from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from benchmark.non_rectangular import (
    RoomLayoutValidationError,
    validate_room_layout,
)
from benchmark.non_rectangular.room_layout import ROOM_LAYOUT_SCHEMA_PATH


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"
SCHEMA = ROOT / "schemas/non_rectangular/room_layout_v1.schema.json"


def _fixture(name: str) -> dict:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("name", "room_count", "wall_count"),
    [
        ("l_shape_single.json", 1, 6),
        ("angled_single.json", 1, 4),
        ("simple_multi_room.json", 2, 10),
    ],
)
def test_room_layout_fixtures_are_valid(
    name: str,
    room_count: int,
    wall_count: int,
) -> None:
    report = validate_room_layout(_fixture(name))

    assert report["valid"] is True
    assert report["room_count"] == room_count
    assert report["wall_segment_count"] == wall_count
    assert report["coordinate_frame_shared"] is True
    assert report["room_interiors_disjoint"] is True


def test_schema_is_packaged_and_has_no_ceiling_contract() -> None:
    source = json.loads(SCHEMA.read_text(encoding="utf-8"))
    packaged = json.loads(ROOM_LAYOUT_SCHEMA_PATH.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(source)
    assert source == packaged
    assert "ceiling_z_m" not in source["$defs"]["room"]["properties"]
    assert source["$defs"]["room"]["properties"]["floor_z_m"] == {
        "const": 0.0
    }


def test_validation_does_not_mutate_input() -> None:
    value = _fixture("angled_single.json")
    before = deepcopy(value)

    validate_room_layout(value)

    assert value == before


def test_wall_loop_must_match_polygon_order() -> None:
    value = _fixture("l_shape_single.json")
    value["rooms"][0]["wall_segments"][1]["start_xy"] = [3.9, 0.0]

    with pytest.raises(RoomLayoutValidationError, match="ordered polygon geometry"):
        validate_room_layout(value)


def test_inward_normal_must_match_ccw_wall_direction() -> None:
    value = _fixture("angled_single.json")
    value["rooms"][0]["wall_segments"][1]["inward_normal_xy"] = [1.0, 0.0]

    with pytest.raises(RoomLayoutValidationError, match="inward_normal_xy"):
        validate_room_layout(value)


def test_clockwise_polygon_is_rejected() -> None:
    value = _fixture("l_shape_single.json")
    value["rooms"][0]["floor_polygon_xy"].reverse()

    with pytest.raises(RoomLayoutValidationError, match="counter-clockwise"):
        validate_room_layout(value)


def test_nonzero_floor_is_rejected() -> None:
    value = _fixture("l_shape_single.json")
    value["rooms"][0]["floor_z_m"] = 0.1

    with pytest.raises(RoomLayoutValidationError, match="schema validation failed"):
        validate_room_layout(value)


def test_room_interiors_must_not_overlap() -> None:
    value = _fixture("l_shape_single.json")
    duplicate = deepcopy(value["rooms"][0])
    duplicate["room_id"] = "room_001"
    for wall in duplicate["wall_segments"]:
        wall["wall_id"] = wall["wall_id"].replace("room_000.", "room_001.")
    value["rooms"].append(duplicate)
    value["room_count"] = 2
    value["room_order"] = ["room_000", "room_001"]

    with pytest.raises(RoomLayoutValidationError, match="interiors must not overlap"):
        validate_room_layout(value)


@pytest.mark.parametrize("forbidden", ["doors", "windows", "bboxes", "point_cloud"])
def test_excluded_spatiallm_entities_are_not_in_room_layout(
    forbidden: str,
) -> None:
    value = _fixture("l_shape_single.json")
    value[forbidden] = []

    with pytest.raises(RoomLayoutValidationError, match="schema validation failed"):
        validate_room_layout(value)
