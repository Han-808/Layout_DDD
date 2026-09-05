from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest
from jsonschema import Draft202012Validator

from benchmark.scene_generation.non_rectangular_multi_room import (
    NonRectangularGenerationContractError,
    build_global_retrieval_plan,
    build_stage_a_user_value,
    build_stage_a_user_value_v2,
    build_stage_c_user_value,
    build_stage_c_user_value_v2,
    group_asset_selection,
    materialize_generated_scene,
    validate_global_placement,
    validate_stage_a_artifacts,
)
from benchmark.scene_generation.non_rectangular_multi_room.contracts import (
    GLOBAL_PLACEMENT_SCHEMA_PATH,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"


def _fixture(name: str) -> dict:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _frozen_request(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "hy34_frozen_top1_requests_v1",
        "retrieval_policy": {
            "category_argument": None,
            "size_constraint_used": False,
            "top_k": 1,
            "min_score": 0.3,
            "query_rewrite_allowed": False,
            "retry_allowed": False,
            "asset_replacement_allowed": False,
        },
        "requests": [
            {
                "slot_id": item["id"],
                "retrieval_query": item["metadata"]["retrieval_query"],
                "estimated_size": item["estimated_size"],
                "size_constraint": None,
            }
            for item in plan["objects"]
        ],
    }


def _retrieval_results(request: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for order, item in enumerate(request["requests"]):
        slot_id = str(item["slot_id"])
        rows.append(
            {
                "order": order,
                "slot_id": slot_id,
                "retrieval_query": item["retrieval_query"],
                "size_constraint": None,
                "invocation_count": 1,
                "rank1": {
                    "rank": 1,
                    "jid": f"asset_{order:03d}",
                    "category": "fixture_asset",
                    "description": "Fixture asset.",
                    "short_desc": "Fixture.",
                    "size": item["estimated_size"],
                    "score": 0.9,
                    "index_row": order,
                },
                "accepted_as_frozen_outcome": True,
            }
        )
    return {
        "schema_version": "hy34_frozen_top1_results_v1",
        "total_invocations": len(rows),
        "retry_count": 0,
        "asset_replacement_count": 0,
        "results": rows,
    }


def _prepared_assets():
    plan = _fixture("simple_multi_room_object_plan.json")
    flat, request, bindings = build_global_retrieval_plan(
        plan,
        frozen_build_retrieval_request=_frozen_request,
    )
    results, selection = group_asset_selection(
        object_plan=plan,
        flat_plan=flat,
        raw_retrieval_results=_retrieval_results(request),
        bindings=bindings,
    )
    return plan, flat, request, bindings, results, selection


def _prepared_assets_v2():
    plan = _fixture("simple_multi_room_object_plan_v2.json")
    flat, request, bindings = build_global_retrieval_plan(
        plan,
        frozen_build_retrieval_request=_frozen_request,
    )
    results, selection = group_asset_selection(
        object_plan=plan,
        flat_plan=flat,
        raw_retrieval_results=_retrieval_results(request),
        bindings=bindings,
    )
    return plan, flat, request, bindings, results, selection


def test_global_placement_schema_is_packaged_identically() -> None:
    source = json.loads(
        (ROOT / "schemas/non_rectangular/global_placement_v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    packaged = json.loads(
        GLOBAL_PLACEMENT_SCHEMA_PATH.read_text(encoding="utf-8")
    )

    Draft202012Validator.check_schema(source)
    assert source == packaged


def _placement(selection: Mapping[str, Any]) -> dict[str, Any]:
    asset_by_key = {
        (room["room_id"], item["slot_id"]): item["selected_asset"]["jid"]
        for room in selection["rooms"]
        for item in room["objects"]
    }
    return {
        "schema_version": "non_rectangular_global_catalog_placement_v1",
        "layout_id": "fixture_simple_multi_room",
        "coordinate_frame": "shared_scene_global_x_width_y_depth_z_up_meters",
        "room_order": ["room_000", "room_001"],
        "rooms": [
            {
                "room_id": "room_000",
                "program_id": "kitchen_01",
                "room_type": "kitchen",
                "instances": [
                    {
                        "instance_id": "fixture.global_000",
                        "asset_id": asset_by_key[("room_000", "counter")],
                        "slot_id": "counter",
                        "center_m": [1.0, 1.0, 0.45],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 0.0]
                    },
                    {
                        "instance_id": "fixture.global_001",
                        "asset_id": asset_by_key[("room_000", "stool")],
                        "slot_id": "stool",
                        "center_m": [2.0, 1.0, 0.45],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 0.0]
                    }
                ]
            },
            {
                "room_id": "room_001",
                "program_id": "living_room_01",
                "room_type": "living_room",
                "instances": [
                    {
                        "instance_id": "fixture.global_002",
                        "asset_id": asset_by_key[("room_001", "sofa")],
                        "slot_id": "sofa",
                        "center_m": [5.7, 1.6, 0.4],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 90.0]
                    },
                    {
                        "instance_id": "fixture.global_003",
                        "asset_id": asset_by_key[("room_001", "coffee_table")],
                        "slot_id": "coffee_table",
                        "center_m": [4.2, 0.6, 0.2],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 0.0]
                    }
                ]
            }
        ]
    }


def test_stage_a_brief_exposes_complete_layout_and_program_once() -> None:
    value = build_stage_a_user_value(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
    )

    assert value["generation_contract"]["one_global_emission"] is True
    assert value["generation_contract"]["source_room_type_labels_provided"] is False
    assert value["generation_contract"][
        "scene_allocation_plausibility_judge_enabled"
    ] is False
    assert value["room_layout"]["room_count"] == 2
    assert len(value["room_program"]["programs"]) == 2
    hints = value["planning_hints"]
    assert hints["policy"] == "area_proportional_object_instance_guidance_v1"
    assert hints["target_total_instances"] == {"min": 4, "max": 6}
    assert hints["per_room_quotas_are_hard_constraints"] is False
    assert [item["room_id"] for item in hints["rooms"]] == [
        "room_000",
        "room_001",
    ]
    assert sum(item["area_share"] for item in hints["rooms"]) == pytest.approx(1.0)
    assert sum(
        item["proportional_instance_quota"]["at_target_min"]
        for item in hints["rooms"]
    ) == pytest.approx(4.0)
    assert sum(
        item["proportional_instance_quota"]["at_target_max"]
        for item in hints["rooms"]
    ) == pytest.approx(6.0)


def test_stage_a_v2_brief_selects_simplified_plan_and_global_facing() -> None:
    value = build_stage_a_user_value_v2(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
    )

    assert value["schema_version"] == "non_rectangular_stage_a_brief_v2"
    assert value["generation_mode"] == "non_rectangular_multi_room_global_v2"
    assert value["generation_contract"]["output_schema_version"] == (
        "non_rectangular_multi_room_object_plan_v2"
    )
    assert value["generation_contract"]["catalog_facing_prior"] == (
        "directed_local_neg_y"
    )
    assert value["planning_hints"]["rooms"][0]["area_share"] > 0.0


def test_stage_a_validation_applies_count_gate_before_stage_c() -> None:
    program = _fixture("simple_multi_room_program.json")
    program["target_total_instances"] = {"min": 7, "max": 8}

    report = validate_stage_a_artifacts(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=program,
        object_plan=_fixture("simple_multi_room_object_plan.json"),
    )

    assert report["terminal_status"] == "failed"
    assert report["failure_reason"] == "object_count_contract_failed"
    assert report["count_compliance"]["factor"] == pytest.approx((4 / 7) ** 2)


def test_stage_a_mapping_cutoff_is_terminal_zero_before_retrieval() -> None:
    plan = _fixture("simple_multi_room_object_plan.json")
    plan["rooms"][0].pop("program_id")
    plan["rooms"][0].pop("room_type")

    report = validate_stage_a_artifacts(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
        object_plan=plan,
    )

    compliance = report["program_mapping"]["coverage_compliance"]
    assert report["terminal_status"] == "failed"
    assert report["failure_reason"] == "program_mapping_contract_failed"
    assert compliance["invalid_room_count"] == 1
    assert compliance["failure_boundary_invalid_room_count"] == 1
    assert compliance["factor"] == 0.0
    assert compliance["terminal_case_score"] == 0.0


def test_stage_a_rejects_cross_room_or_cyclic_support() -> None:
    layout = _fixture("simple_multi_room.json")
    program = _fixture("simple_multi_room_program.json")
    plan = _fixture("simple_multi_room_object_plan.json")
    plan["rooms"][0]["objects"][0]["metadata"]["support"] = (
        "room_001.wall_000"
    )
    with pytest.raises(
        NonRectangularGenerationContractError,
        match="unknown same-room target",
    ):
        validate_stage_a_artifacts(
            room_layout=layout,
            room_program=program,
            object_plan=plan,
        )


def test_stage_a_v2_validates_simplified_support_and_facing() -> None:
    layout = _fixture("simple_multi_room.json")
    program = _fixture("simple_multi_room_program.json")
    valid = _fixture("simple_multi_room_object_plan_v2.json")

    report = validate_stage_a_artifacts(
        room_layout=layout,
        room_program=program,
        object_plan=valid,
        expected_plan_contract_version="v2",
    )
    assert report["plan_contract_version"] == "v2"
    assert report["terminal_status"] == "ready"

    invalid = deepcopy(valid)
    invalid["rooms"][0]["objects"][0]["support"] = "room_001.wall_000"
    with pytest.raises(
        NonRectangularGenerationContractError,
        match="unknown same-room target",
    ):
        validate_stage_a_artifacts(
            room_layout=layout,
            room_program=program,
            object_plan=invalid,
            expected_plan_contract_version="v2",
        )

    self_facing = deepcopy(valid)
    self_facing["rooms"][0]["objects"][0]["facing_target"] = "counter"
    with pytest.raises(
        NonRectangularGenerationContractError,
        match="cannot face itself",
    ):
        validate_stage_a_artifacts(
            room_layout=layout,
            room_program=program,
            object_plan=self_facing,
            expected_plan_contract_version="v2",
        )

    cycle = _fixture("simple_multi_room_object_plan.json")
    cycle["rooms"][0]["objects"][0]["metadata"]["support"] = "stool"
    cycle["rooms"][0]["objects"][1]["metadata"]["support"] = "counter"
    with pytest.raises(
        NonRectangularGenerationContractError,
        match="support graph contains a cycle",
    ):
        validate_stage_a_artifacts(
            room_layout=layout,
            room_program=program,
            object_plan=cycle,
        )


def test_retrieval_slots_are_namespaced_and_invoked_once() -> None:
    _, _, request, bindings, results, selection = _prepared_assets()

    assert [item["slot_id"] for item in request["requests"]] == [
        "room_000::counter",
        "room_000::stool",
        "room_001::sofa",
        "room_001::coffee_table",
    ]
    assert len(bindings) == 4
    assert results["total_invocations"] == 4
    assert [room["room_id"] for room in selection["rooms"]] == [
        "room_000",
        "room_001",
    ]


def test_v2_retrieval_adapter_is_deterministic_and_internal_only() -> None:
    plan, flat, request, bindings, results, selection = _prepared_assets_v2()

    assert plan["rooms"][0]["objects"][0].keys() == {
        "id",
        "category",
        "description",
        "count",
        "estimated_size",
        "retrieval_query",
        "support",
        "facing_target",
        "placement_hints",
    }
    adapted = flat["objects"][0]
    assert adapted["metadata"]["retrieval_query"] == (
        plan["rooms"][0]["objects"][0]["retrieval_query"]
    )
    assert adapted["metadata"]["functional_side"] == "local_neg_y"
    assert adapted["metadata"]["requested_count"] == adapted["count"]
    assert request["requests"][0]["slot_id"] == "room_000::counter"
    assert selection["rooms"][0]["objects"][0]["planned_object"] == (
        plan["rooms"][0]["objects"][0]
    )
    assert len(bindings) == 4
    assert results["total_invocations"] == 4
    assert [room["room_id"] for room in selection["rooms"]] == [
        "room_000",
        "room_001",
    ]


def test_stage_c_payload_keeps_global_geometry_and_all_rooms() -> None:
    plan, _, _, _, _, selection = _prepared_assets()

    value = build_stage_c_user_value(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
        object_plan=plan,
        asset_selection=selection,
    )

    assert value["generation_contract"]["one_global_emission"] is True
    assert value["generation_contract"]["coordinate_frame"].startswith(
        "shared_scene_global"
    )
    assert "local_to_global_offset_m" not in json.dumps(value)


def test_stage_c_v2_payload_freezes_simplified_plan_and_facing_prior() -> None:
    plan, _, _, _, _, selection = _prepared_assets_v2()

    value = build_stage_c_user_value_v2(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
        object_plan=plan,
        asset_selection=selection,
    )

    assert value["schema_version"] == "non_rectangular_stage_c_input_v2"
    assert value["generation_mode"] == "non_rectangular_multi_room_global_v2"
    assert value["generation_contract"]["catalog_facing_prior"] == (
        "directed_local_neg_y"
    )
    assert value["object_plan"] == plan


def test_global_placement_rejects_mapping_slot_and_asset_drift() -> None:
    plan, _, _, _, _, selection = _prepared_assets()
    valid = _placement(selection)
    assert validate_global_placement(
        valid,
        object_plan=plan,
        asset_selection=selection,
    ) == valid

    mapping = deepcopy(valid)
    mapping["rooms"][0]["room_type"] = "bedroom"
    with pytest.raises(NonRectangularGenerationContractError, match="changed"):
        validate_global_placement(
            mapping,
            object_plan=plan,
            asset_selection=selection,
        )

    slot = deepcopy(valid)
    slot["rooms"][0]["instances"].pop()
    with pytest.raises(NonRectangularGenerationContractError, match="slot coverage"):
        validate_global_placement(
            slot,
            object_plan=plan,
            asset_selection=selection,
        )

    asset = deepcopy(valid)
    asset["rooms"][0]["instances"][0]["asset_id"] = "replacement"
    with pytest.raises(NonRectangularGenerationContractError, match="Top-1"):
        validate_global_placement(
            asset,
            object_plan=plan,
            asset_selection=selection,
        )


def test_materializer_builds_nested_canonical_scene_without_translation() -> None:
    plan, _, _, _, _, selection = _prepared_assets()
    placement = _placement(selection)

    scene, preflight = materialize_generated_scene(
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
        object_plan=plan,
        asset_selection=selection,
        placement=placement,
    )

    assert scene["room_order"] == ["room_000", "room_001"]
    assert scene["rooms"][1]["objects"][0]["center"] == [5.7, 1.6, 0.4]
    assert scene["rooms"][1]["objects"][0]["id"] == "fixture.global_002"
    assert "room_id" not in scene["rooms"][1]["objects"][0]
    assert scene["provenance"]["coordinates_transformed"] is False
    assert preflight["terminal_status"] == "ready"
