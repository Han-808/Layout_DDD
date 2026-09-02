from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from benchmark.non_rectangular import (
    NonRectangularContractError,
    validate_multi_room_object_plan,
    validate_multi_room_scene,
    validate_room_program,
)
from benchmark.non_rectangular.contracts import (
    MULTI_ROOM_SCENE_SCHEMA_PATH,
    OBJECT_PLAN_SCHEMA_PATH,
    OBJECT_PLAN_V2_SCHEMA_PATH,
    ROOM_PROGRAM_SCHEMA_PATH,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"
SOURCE_SCHEMAS = ROOT / "schemas/non_rectangular"


def _fixture(name: str) -> dict:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


@pytest.mark.parametrize(
    ("source_name", "packaged_path"),
    [
        ("room_program_v1.schema.json", ROOM_PROGRAM_SCHEMA_PATH),
        ("object_plan_v1.schema.json", OBJECT_PLAN_SCHEMA_PATH),
        ("object_plan_v2.schema.json", OBJECT_PLAN_V2_SCHEMA_PATH),
        ("scene_v1.schema.json", MULTI_ROOM_SCENE_SCHEMA_PATH),
    ],
)
def test_contract_schemas_are_valid_and_packaged_identically(
    source_name: str,
    packaged_path: Path,
) -> None:
    source = json.loads((SOURCE_SCHEMAS / source_name).read_text(encoding="utf-8"))
    packaged = json.loads(packaged_path.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(source)
    assert source == packaged


def test_room_program_fixture_is_valid() -> None:
    report = validate_room_program(
        _fixture("simple_multi_room_program.json")
    )

    assert report == {
        "schema_version": "non_rectangular_room_program_v1",
        "layout_id": "fixture_simple_multi_room",
        "valid": True,
        "program_count": 2,
        "program_ids": ["kitchen_01", "living_room_01"],
        "room_types": ["kitchen", "living_room"],
        "target_total_instances": {"min": 4, "max": 6},
    }


def test_object_plan_fixture_is_valid() -> None:
    report = validate_multi_room_object_plan(
        _fixture("simple_multi_room_object_plan.json")
    )

    assert report["valid"] is True
    assert report["room_ids"] == ["room_000", "room_001"]
    assert report["room_instance_counts"] == {
        "room_000": 2,
        "room_001": 2,
    }
    assert report["planned_instance_count"] == 4
    assert report["mapping_complete"] is True
    assert report["plan_contract_version"] == "v1"


def test_simplified_object_plan_v2_fixture_is_valid() -> None:
    report = validate_multi_room_object_plan(
        _fixture("simple_multi_room_object_plan_v2.json")
    )

    assert report["schema_version"] == (
        "non_rectangular_multi_room_object_plan_v2"
    )
    assert report["plan_contract_version"] == "v2"
    assert report["room_ids"] == ["room_000", "room_001"]
    assert report["planned_instance_count"] == 4
    assert report["mapping_complete"] is True


def test_scene_fixture_is_valid_and_keeps_global_ids() -> None:
    report = validate_multi_room_scene(
        _fixture("simple_multi_room_scene.json")
    )

    assert report["valid"] is True
    assert report["room_ids"] == ["room_000", "room_001"]
    assert report["room_object_counts"] == {
        "room_000": 2,
        "room_001": 2,
    }
    assert report["generated_object_count"] == 4
    assert report["coordinate_frame_shared"] is True


@pytest.mark.parametrize(
    ("name", "validator"),
    [
        ("simple_multi_room_program.json", validate_room_program),
        (
            "simple_multi_room_object_plan.json",
            validate_multi_room_object_plan,
        ),
        (
            "simple_multi_room_object_plan_v2.json",
            validate_multi_room_object_plan,
        ),
        ("simple_multi_room_scene.json", validate_multi_room_scene),
    ],
)
def test_contract_validation_does_not_mutate_input(name: str, validator) -> None:
    value = _fixture(name)
    before = deepcopy(value)

    validator(value)

    assert value == before


def test_room_program_rejects_inverted_instance_range() -> None:
    value = _fixture("simple_multi_room_program.json")
    value["target_total_instances"] = {"min": 7, "max": 4}

    with pytest.raises(
        NonRectangularContractError,
        match="min must be <= max",
    ):
        validate_room_program(value)


def test_room_program_rejects_whitespace_only_room_type() -> None:
    value = _fixture("simple_multi_room_program.json")
    value["programs"][0]["room_type"] = "   "

    with pytest.raises(
        NonRectangularContractError,
        match="non-whitespace text",
    ):
        validate_room_program(value)


def test_object_plan_allows_missing_program_mapping_for_scored_fallback() -> None:
    value = _fixture("simple_multi_room_object_plan.json")
    value["rooms"][0].pop("program_id")
    value["rooms"][0].pop("room_type")

    report = validate_multi_room_object_plan(value)

    assert report["valid"] is True
    assert report["mapping_complete"] is False


def test_object_plan_v2_allows_missing_mapping_but_rejects_v1_fields() -> None:
    value = _fixture("simple_multi_room_object_plan_v2.json")
    value["rooms"][0].pop("program_id")
    value["rooms"][0].pop("room_type")

    report = validate_multi_room_object_plan(value)

    assert report["mapping_complete"] is False
    value["rooms"][0]["objects"][0]["role"] = "duplicate_field"
    with pytest.raises(
        NonRectangularContractError,
        match="object-plan schema validation failed",
    ):
        validate_multi_room_object_plan(value)


def test_object_plan_rejects_unknown_zone_reference() -> None:
    value = _fixture("simple_multi_room_object_plan.json")
    value["rooms"][0]["objects"][0]["metadata"]["zone"] = "missing"

    with pytest.raises(
        NonRectangularContractError,
        match="references unknown zone",
    ):
        validate_multi_room_object_plan(value)


def test_object_plan_rejects_requested_count_drift() -> None:
    value = _fixture("simple_multi_room_object_plan.json")
    value["rooms"][0]["objects"][0]["metadata"]["requested_count"] = 2

    with pytest.raises(
        NonRectangularContractError,
        match="requested_count must equal count",
    ):
        validate_multi_room_object_plan(value)


def test_scene_allows_missing_program_mapping_for_scored_fallback() -> None:
    value = _fixture("simple_multi_room_scene.json")
    value["rooms"][0].pop("program_id")
    value["rooms"][0].pop("room_type")

    report = validate_multi_room_scene(value)

    assert report["valid"] is True
    assert report["mapping_complete"] is False


def test_scene_rejects_duplicate_global_object_id() -> None:
    value = _fixture("simple_multi_room_scene.json")
    value["rooms"][1]["objects"][0]["id"] = value["rooms"][0]["objects"][0]["id"]

    with pytest.raises(
        NonRectangularContractError,
        match="duplicate scene-global object ID",
    ):
        validate_multi_room_scene(value)


def test_scene_rejects_nonfinite_object_geometry() -> None:
    value = _fixture("simple_multi_room_scene.json")
    value["rooms"][0]["objects"][0]["center"][0] = float("nan")

    with pytest.raises(NonRectangularContractError, match="must be finite"):
        validate_multi_room_scene(value)


def test_scene_requires_slot_id() -> None:
    value = _fixture("simple_multi_room_scene.json")
    value["rooms"][0]["objects"][0].pop("slot_id")

    with pytest.raises(
        NonRectangularContractError,
        match="scene schema validation failed",
    ):
        validate_multi_room_scene(value)


def test_scene_object_must_not_repeat_room_id() -> None:
    value = _fixture("simple_multi_room_scene.json")
    value["rooms"][0]["objects"][0]["room_id"] = "room_000"

    with pytest.raises(
        NonRectangularContractError,
        match="scene schema validation failed",
    ):
        validate_multi_room_scene(value)


def test_scene_accepts_additional_canonical_object_payload() -> None:
    value = _fixture("simple_multi_room_scene.json")
    value["rooms"][0]["objects"][0]["task_slot"] = {
        "intended_category": "counter"
    }

    assert validate_multi_room_scene(value)["valid"] is True
