from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.non_rectangular import (
    NonRectangularEvaluationInput,
    RoomEvaluationUnitError,
    build_room_evaluation_units,
    prepare_non_rectangular_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"


def _fixture(name: str) -> dict:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _input() -> NonRectangularEvaluationInput:
    return NonRectangularEvaluationInput.from_artifacts(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
        object_plan=_fixture("simple_multi_room_object_plan.json"),
        generated_scene=_fixture("simple_multi_room_scene.json"),
    )


def test_room_units_follow_authoritative_room_order() -> None:
    preflight = prepare_non_rectangular_evaluation(_input())

    units = build_room_evaluation_units(preflight)

    assert [unit.room_id for unit in units] == ["room_000", "room_001"]
    assert [unit.room_index for unit in units] == [0, 1]
    assert [unit.room_type for unit in units] == ["kitchen", "living_room"]
    assert [unit.planned_instance_count for unit in units] == [2, 2]
    assert [unit.generated_object_count for unit in units] == [2, 2]


def test_room_units_project_simplified_v2_plan_without_legacy_placeholders() -> None:
    source = _input()
    value = NonRectangularEvaluationInput.from_artifacts(
        room_layout=source.room_layout,
        room_program=source.room_program,
        object_plan=_fixture("simple_multi_room_object_plan_v2.json"),
        generated_scene=source.generated_scene,
    )
    units = build_room_evaluation_units(
        prepare_non_rectangular_evaluation(value)
    )

    assert units[0].prompt_granularity == "simplified"
    assert units[0].scene_description == "kitchen"
    assert units[0].zones == ()
    assert units[0].relations == ()
    assert units[0].planned_objects[0]["facing_target"] == "room_interior"


def test_room_unit_preserves_global_coordinates_and_ids() -> None:
    source = _input()
    expected_objects = source.generated_scene["rooms"][1]["objects"]
    preflight = prepare_non_rectangular_evaluation(source)

    unit = build_room_evaluation_units(preflight)[1]

    assert list(unit.generated_objects) == expected_objects
    assert unit.object_ids == (
        "fixture.object_002",
        "fixture.object_003",
    )
    assert unit.coordinate_frame["origin"] == "shared_scene_global"
    assert unit.public_dict()["coordinates_transformed"] is False


def test_room_unit_contains_exact_polygon_and_walls_without_ceiling() -> None:
    source = _input()
    expected_geometry = source.room_layout["rooms"][0]
    unit = build_room_evaluation_units(
        prepare_non_rectangular_evaluation(source)
    )[0]

    public = unit.public_dict()
    assert public["geometry"]["floor_polygon_xy"] == expected_geometry[
        "floor_polygon_xy"
    ]
    assert public["geometry"]["wall_segments"] == expected_geometry[
        "wall_segments"
    ]
    assert public["geometry"]["floor_z_m"] == 0.0
    assert "ceiling_z_m" not in public["geometry"]


def test_room_unit_is_detached_from_preflight_artifacts() -> None:
    preflight = prepare_non_rectangular_evaluation(_input())
    units = build_room_evaluation_units(preflight)

    preflight.evaluation_input.generated_scene["rooms"][0]["objects"][0][
        "center"
    ][0] = 999.0
    public = units[0].public_dict()
    public["generated_objects"][0]["center"][0] = -999.0

    assert units[0].generated_objects[0]["center"][0] == 1.0


def test_terminal_mapping_zero_cannot_build_room_units() -> None:
    value = _input()
    for artifact in (value.object_plan, value.generated_scene):
        artifact["rooms"][0].pop("program_id")
        artifact["rooms"][0].pop("room_type")

    preflight = prepare_non_rectangular_evaluation(value)

    assert preflight.program_mapping["rooms"]["room_000"][
        "functional_score_override"
    ] is None
    with pytest.raises(RoomEvaluationUnitError, match="failed preflight"):
        build_room_evaluation_units(preflight)


def test_unit_provenance_uses_cross_artifact_hashes() -> None:
    preflight = prepare_non_rectangular_evaluation(_input())

    units = build_room_evaluation_units(preflight)

    assert all(unit.artifact_sha256 == preflight.artifact_sha256 for unit in units)


def test_count_gate_failed_case_cannot_build_units() -> None:
    value = _input()
    value.room_program["target_total_instances"] = {"min": 7, "max": 8}
    preflight = prepare_non_rectangular_evaluation(value)

    with pytest.raises(
        RoomEvaluationUnitError,
        match="failed preflight",
    ):
        build_room_evaluation_units(preflight)
