"""Pure contracts for one global Stage A and one global Stage C."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Mapping

from jsonschema import Draft202012Validator
from shapely.geometry import Polygon

from benchmark.non_rectangular.contracts import (
    OBJECT_PLAN_SCHEMA_VERSION,
    OBJECT_PLAN_V2_SCHEMA_VERSION,
    validate_multi_room_object_plan,
    validate_multi_room_scene,
    validate_room_program,
)
from benchmark.non_rectangular.preflight import (
    NonRectangularEvaluationInput,
    object_count_compliance,
    prepare_non_rectangular_evaluation,
    program_mapping_report,
)
from benchmark.non_rectangular.room_layout import (
    validate_room_layout,
)
from benchmark.resources import runtime_resource_path
from benchmark.scene_generation.multi_room.contracts import (
    build_asset_selection,
    validate_retrieval_results,
)


GENERATION_MODE = "non_rectangular_multi_room_global_v1"
GENERATION_MODE_V2 = "non_rectangular_multi_room_global_v2"
GLOBAL_PLACEMENT_SCHEMA_VERSION = (
    "non_rectangular_global_catalog_placement_v1"
)
GLOBAL_PLACEMENT_SCHEMA_PATH = runtime_resource_path(
    "schemas/non_rectangular/global_placement_v1.schema.json"
)
STAGE_A_BRIEF_SCHEMA_VERSION = "non_rectangular_stage_a_brief_v1"
STAGE_A_BRIEF_V2_SCHEMA_VERSION = "non_rectangular_stage_a_brief_v2"
STAGE_C_INPUT_SCHEMA_VERSION = "non_rectangular_stage_c_input_v1"
STAGE_C_INPUT_V2_SCHEMA_VERSION = "non_rectangular_stage_c_input_v2"
ASSET_SELECTION_SCHEMA_VERSION = "non_rectangular_asset_selection_v1"
RETRIEVAL_BINDING_POLICY = "room_id_double_colon_slot_id_v1"


class NonRectangularGenerationContractError(ValueError):
    """Raised when a global generation artifact violates the new mode."""


def build_stage_a_user_value(
    *,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserved v1 Stage-A payload."""

    return _build_stage_a_user_value(
        room_layout=room_layout,
        room_program=room_program,
        schema_version=STAGE_A_BRIEF_SCHEMA_VERSION,
        generation_mode=GENERATION_MODE,
        output_schema_version=OBJECT_PLAN_SCHEMA_VERSION,
        include_catalog_facing_prior=False,
    )


def build_stage_a_user_value_v2(
    *,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
) -> dict[str, Any]:
    """Simplified v2 Stage-A payload with the same immutable geometry/program."""

    return _build_stage_a_user_value(
        room_layout=room_layout,
        room_program=room_program,
        schema_version=STAGE_A_BRIEF_V2_SCHEMA_VERSION,
        generation_mode=GENERATION_MODE_V2,
        output_schema_version=OBJECT_PLAN_V2_SCHEMA_VERSION,
        include_catalog_facing_prior=True,
    )


def _build_stage_a_user_value(
    *,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
    schema_version: str,
    generation_mode: str,
    output_schema_version: str,
    include_catalog_facing_prior: bool,
) -> dict[str, Any]:
    """Expose the complete layout/program once, without external source data."""

    layout_report = validate_room_layout(room_layout)
    program_report = validate_room_program(room_program)
    if layout_report["layout_id"] != program_report["layout_id"]:
        raise NonRectangularGenerationContractError(
            "room_layout and room_program layout_id mismatch"
        )
    if layout_report["room_count"] != program_report["program_count"]:
        raise NonRectangularGenerationContractError(
            "room program count must equal layout room count"
        )
    planning_hints = _stage_a_area_planning_hints(
        room_layout=room_layout,
        minimum_instances=int(
            program_report["target_total_instances"]["min"]
        ),
        maximum_instances=int(
            program_report["target_total_instances"]["max"]
        ),
    )
    return {
        "schema_version": schema_version,
        "generation_mode": generation_mode,
        "room_layout": deepcopy(dict(room_layout)),
        "room_program": deepcopy(dict(room_program)),
        "planning_hints": planning_hints,
        "generation_contract": {
            "one_global_emission": True,
            "model_selects_room_program_mapping": True,
            "source_room_type_labels_provided": False,
            "scene_allocation_plausibility_judge_enabled": False,
            "model_selects_object_distribution": True,
            "architecture_is_benchmark_owned": True,
            "doors_windows_ceiling_point_cloud_excluded": True,
            "output_schema_version": output_schema_version,
            **(
                {"catalog_facing_prior": "directed_local_neg_y"}
                if include_catalog_facing_prior
                else {}
            ),
        },
    }


def _stage_a_area_planning_hints(
    *,
    room_layout: Mapping[str, Any],
    minimum_instances: int,
    maximum_instances: int,
) -> dict[str, Any]:
    rooms: list[dict[str, Any]] = []
    areas = [
        float(Polygon(room["floor_polygon_xy"]).area)
        for room in room_layout["rooms"]
    ]
    total_area = sum(areas)
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise NonRectangularGenerationContractError(
            "room layout total floor area must be finite and positive"
        )
    for room, area in zip(room_layout["rooms"], areas):
        share = area / total_area
        rooms.append(
            {
                "room_id": str(room["room_id"]),
                "floor_area_m2": area,
                "area_share": share,
                "proportional_instance_quota": {
                    "at_target_min": share * minimum_instances,
                    "at_target_max": share * maximum_instances,
                },
            }
        )
    return {
        "policy": "area_proportional_object_instance_guidance_v1",
        "counting_rule": "sum of every rooms[].objects[].count",
        "scene_total_floor_area_m2": total_area,
        "target_total_instances": {
            "min": minimum_instances,
            "max": maximum_instances,
        },
        "rooms": rooms,
        "per_room_quotas_are_hard_constraints": False,
        "functional_completeness_adjustments_allowed": True,
    }


def validate_stage_a_artifacts(
    *,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
    object_plan: Mapping[str, Any],
    expected_plan_contract_version: str | None = None,
) -> dict[str, Any]:
    """Validate Stage A and return mapping/count early-stop decisions."""

    layout = validate_room_layout(room_layout)
    program = validate_room_program(room_program)
    plan = validate_multi_room_object_plan(object_plan)
    if (
        expected_plan_contract_version is not None
        and plan["plan_contract_version"] != expected_plan_contract_version
    ):
        raise NonRectangularGenerationContractError(
            "Stage-A object-plan contract version mismatch: "
            f"expected={expected_plan_contract_version!r}, "
            f"actual={plan['plan_contract_version']!r}"
        )
    layout_ids = {
        layout["layout_id"],
        program["layout_id"],
        plan["layout_id"],
    }
    if len(layout_ids) != 1:
        raise NonRectangularGenerationContractError(
            "Stage-A artifact layout_id mismatch"
        )
    room_order = list(layout["room_ids"])
    if list(plan["room_ids"]) != room_order:
        raise NonRectangularGenerationContractError(
            "Stage-A plan room coverage/order differs from room layout"
        )
    if int(program["program_count"]) != len(room_order):
        raise NonRectangularGenerationContractError(
            "room program count must equal room layout room count"
        )
    _validate_room_plan_semantics(
        room_layout=room_layout,
        object_plan=object_plan,
    )
    mapping = program_mapping_report(
        room_order=tuple(room_order),
        programs=list(room_program["programs"]),
        plan_rooms={
            str(item["room_id"]): item for item in object_plan["rooms"]
        },
    )
    target = program["target_total_instances"]
    count = object_count_compliance(
        planned_count=int(plan["planned_instance_count"]),
        minimum=int(target["min"]),
        maximum=int(target["max"]),
    )
    mapping_failed = bool(mapping["coverage_compliance"]["failed"])
    count_failed = bool(count["failed"])
    if mapping_failed and count_failed:
        failure_reason = "program_mapping_and_object_count_contract_failed"
    elif mapping_failed:
        failure_reason = "program_mapping_contract_failed"
    elif count_failed:
        failure_reason = "object_count_contract_failed"
    else:
        failure_reason = None
    return {
        "schema_version": (
            "non_rectangular_stage_a_validation_v2"
            if plan["plan_contract_version"] == "v2"
            else "non_rectangular_stage_a_validation_v1"
        ),
        "plan_contract_version": plan["plan_contract_version"],
        "valid": True,
        "layout_id": layout["layout_id"],
        "room_order": room_order,
        "planned_instance_count": int(plan["planned_instance_count"]),
        "room_instance_counts": deepcopy(plan["room_instance_counts"]),
        "mapping_complete": bool(plan["mapping_complete"]),
        "program_mapping": mapping,
        "count_compliance": count,
        "terminal_status": "failed" if failure_reason is not None else "ready",
        "failure_reason": failure_reason,
    }


def build_global_retrieval_plan(
    object_plan: Mapping[str, Any],
    *,
    frozen_build_retrieval_request: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, str]]]:
    """Namespace every room-local slot and build one deterministic batch."""

    plan_report = validate_multi_room_object_plan(object_plan)
    flattened: list[dict[str, Any]] = []
    bindings: dict[str, dict[str, str]] = {}
    for room in object_plan["rooms"]:
        room_id = str(room["room_id"])
        for item in room["objects"]:
            slot_id = str(item["id"])
            retrieval_slot_id = _retrieval_slot_id(room_id, slot_id)
            if retrieval_slot_id in bindings:
                raise NonRectangularGenerationContractError(
                    f"duplicate retrieval slot ID: {retrieval_slot_id!r}"
                )
            copied = _retrieval_compatible_object(
                item,
                plan_contract_version=str(
                    plan_report["plan_contract_version"]
                ),
            )
            copied["id"] = retrieval_slot_id
            flattened.append(copied)
            bindings[retrieval_slot_id] = {
                "room_id": room_id,
                "slot_id": slot_id,
            }
    flat_plan = {"objects": flattened}
    request = frozen_build_retrieval_request(flat_plan)
    expected = [item["id"] for item in flattened]
    actual = [item.get("slot_id") for item in request.get("requests", [])]
    if actual != expected:
        raise NonRectangularGenerationContractError(
            "frozen retrieval request changed global slot identity/order"
        )
    return flat_plan, request, bindings


def _retrieval_compatible_object(
    item: Mapping[str, Any],
    *,
    plan_contract_version: str,
) -> dict[str, Any]:
    """Project simplified v2 slots into the frozen retrieval primitive shape."""

    if plan_contract_version == "v1":
        return deepcopy(dict(item))
    if plan_contract_version != "v2":
        raise NonRectangularGenerationContractError(
            f"unsupported retrieval plan contract: {plan_contract_version!r}"
        )
    facing_target = item.get("facing_target")
    directed = facing_target is not None
    category = str(item["category"])
    hints = [str(value) for value in item["placement_hints"]]
    return {
        "id": str(item["id"]),
        "category": category,
        "role": category,
        "description": str(item["description"]),
        "count": int(item["count"]),
        "estimated_size": [float(value) for value in item["estimated_size"]],
        "metadata": {
            "intended_role": category,
            "zone": "room_zone",
            "support": str(item["support"]),
            "directed": directed,
            "functional_side": "local_neg_y" if directed else None,
            "facing_intent": facing_target,
            "retrieval_query": str(item["retrieval_query"]),
            "requested_count": int(item["count"]),
        },
        "placement_intent": {
            "absolute_relations": hints,
            "relative_relations": (
                [f"faces {facing_target}"] if directed else []
            ),
        },
    }


def group_asset_selection(
    *,
    object_plan: Mapping[str, Any],
    flat_plan: Mapping[str, Any],
    raw_retrieval_results: Mapping[str, Any],
    bindings: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one retrieval batch and project selected assets back to rooms."""

    validated_results = validate_retrieval_results(
        raw_retrieval_results,
        plan=flat_plan,
    )
    flat_selection = build_asset_selection(flat_plan, validated_results)
    selected_by_id = {
        str(item["object_id"]): item for item in flat_selection["objects"]
    }
    rooms: list[dict[str, Any]] = []
    for room in object_plan["rooms"]:
        room_id = str(room["room_id"])
        objects: list[dict[str, Any]] = []
        for planned in room["objects"]:
            slot_id = str(planned["id"])
            retrieval_slot_id = _retrieval_slot_id(room_id, slot_id)
            binding = bindings.get(retrieval_slot_id)
            if binding != {"room_id": room_id, "slot_id": slot_id}:
                raise NonRectangularGenerationContractError(
                    "retrieval binding coverage changed"
                )
            selected = selected_by_id.get(retrieval_slot_id)
            if selected is None:
                raise NonRectangularGenerationContractError(
                    f"missing selected asset for {retrieval_slot_id!r}"
                )
            objects.append(
                {
                    "slot_id": slot_id,
                    "retrieval_slot_id": retrieval_slot_id,
                    "planned_object": deepcopy(planned),
                    "selected_asset": deepcopy(selected["selected_asset"]),
                    "retrieval_query": deepcopy(selected["retrieval_query"]),
                }
            )
        rooms.append({"room_id": room_id, "objects": objects})
    return validated_results, {
        "schema_version": ASSET_SELECTION_SCHEMA_VERSION,
        "layout_id": str(object_plan["layout_id"]),
        "binding_policy": RETRIEVAL_BINDING_POLICY,
        "rooms": rooms,
    }


def build_stage_c_user_value(
    *,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
    object_plan: Mapping[str, Any],
    asset_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserved v1 Stage-C payload."""

    return _build_stage_c_user_value(
        room_layout=room_layout,
        room_program=room_program,
        object_plan=object_plan,
        asset_selection=asset_selection,
        expected_plan_contract_version="v1",
        schema_version=STAGE_C_INPUT_SCHEMA_VERSION,
        generation_mode=GENERATION_MODE,
        include_catalog_facing_prior=False,
    )


def build_stage_c_user_value_v2(
    *,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
    object_plan: Mapping[str, Any],
    asset_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the v2 Stage-C payload around a frozen simplified plan."""

    return _build_stage_c_user_value(
        room_layout=room_layout,
        room_program=room_program,
        object_plan=object_plan,
        asset_selection=asset_selection,
        expected_plan_contract_version="v2",
        schema_version=STAGE_C_INPUT_V2_SCHEMA_VERSION,
        generation_mode=GENERATION_MODE_V2,
        include_catalog_facing_prior=True,
    )


def _build_stage_c_user_value(
    *,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
    object_plan: Mapping[str, Any],
    asset_selection: Mapping[str, Any],
    expected_plan_contract_version: str,
    schema_version: str,
    generation_mode: str,
    include_catalog_facing_prior: bool,
) -> dict[str, Any]:
    """Build one global Stage-C payload without room-local projections."""

    validate_stage_a_artifacts(
        room_layout=room_layout,
        room_program=room_program,
        object_plan=object_plan,
        expected_plan_contract_version=expected_plan_contract_version,
    )
    if asset_selection.get("schema_version") != ASSET_SELECTION_SCHEMA_VERSION:
        raise NonRectangularGenerationContractError(
            "unsupported non-rectangular asset selection"
        )
    return {
        "schema_version": schema_version,
        "generation_mode": generation_mode,
        "room_layout": deepcopy(dict(room_layout)),
        "room_program": deepcopy(dict(room_program)),
        "object_plan": deepcopy(dict(object_plan)),
        "asset_selection": deepcopy(dict(asset_selection)),
        "generation_contract": {
            "one_global_emission": True,
            "coordinate_frame": (
                "shared_scene_global_x_width_y_depth_z_up_meters"
            ),
            "post_placement_edits_allowed": 0,
            "mapping_and_slot_counts_immutable": True,
            "architecture_is_benchmark_owned": True,
            "output_schema_version": GLOBAL_PLACEMENT_SCHEMA_VERSION,
            **(
                {"catalog_facing_prior": "directed_local_neg_y"}
                if include_catalog_facing_prior
                else {}
            ),
        },
    }


def validate_global_placement(
    value: Mapping[str, Any],
    *,
    object_plan: Mapping[str, Any],
    asset_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate global room/mapping/slot/asset/pose identity without repair."""

    if not isinstance(value, Mapping):
        raise NonRectangularGenerationContractError(
            "global placement must be a JSON object"
        )
    _validate_placement_schema(value)
    if value.get("layout_id") != object_plan.get("layout_id"):
        raise NonRectangularGenerationContractError(
            "global placement layout_id mismatch"
        )
    plan_rooms = {str(item["room_id"]): item for item in object_plan["rooms"]}
    placement_rooms = {str(item["room_id"]): item for item in value["rooms"]}
    room_order = [str(item) for item in object_plan["room_order"]]
    if list(value["room_order"]) != room_order or list(placement_rooms) != room_order:
        raise NonRectangularGenerationContractError(
            "global placement room coverage/order differs from Stage A"
        )
    selection = _selection_by_room_and_slot(asset_selection)
    global_instance_ids: set[str] = set()
    for room_id in room_order:
        plan_room = plan_rooms[room_id]
        placed_room = placement_rooms[room_id]
        for field in ("program_id", "room_type"):
            if placed_room.get(field) != plan_room.get(field):
                raise NonRectangularGenerationContractError(
                    f"Stage C changed {room_id}.{field}"
                )
        expected_counts = {
            str(item["id"]): int(item["count"])
            for item in plan_room["objects"]
        }
        actual_counts: dict[str, int] = {}
        for index, instance in enumerate(placed_room["instances"]):
            path = f"rooms[{room_id}].instances[{index}]"
            instance_id = str(instance["instance_id"])
            if instance_id in global_instance_ids:
                raise NonRectangularGenerationContractError(
                    f"duplicate scene-global instance ID: {instance_id!r}"
                )
            global_instance_ids.add(instance_id)
            slot_id = str(instance["slot_id"])
            actual_counts[slot_id] = actual_counts.get(slot_id, 0) + 1
            selected = selection.get((room_id, slot_id))
            if selected is None:
                raise NonRectangularGenerationContractError(
                    f"{path}.slot_id references unknown frozen slot"
                )
            if str(instance["asset_id"]) != str(selected["jid"]):
                raise NonRectangularGenerationContractError(
                    f"{path}.asset_id differs from frozen Top-1 asset"
                )
            _finite_vec3(instance["center_m"], f"{path}.center_m")
            _finite_vec3(
                instance["rotation_euler_xyz_deg"],
                f"{path}.rotation_euler_xyz_deg",
            )
            scale = instance["uniform_scale"]
            if (
                isinstance(scale, bool)
                or not isinstance(scale, (int, float))
                or not math.isfinite(float(scale))
                or float(scale) <= 0.0
            ):
                raise NonRectangularGenerationContractError(
                    f"{path}.uniform_scale must be finite and positive"
                )
        if actual_counts != expected_counts:
            raise NonRectangularGenerationContractError(
                f"Stage C slot coverage differs for {room_id}: "
                f"expected={expected_counts!r}, actual={actual_counts!r}"
            )
    return deepcopy(dict(value))


def materialize_generated_scene(
    *,
    room_layout: Mapping[str, Any],
    room_program: Mapping[str, Any],
    object_plan: Mapping[str, Any],
    asset_selection: Mapping[str, Any],
    placement: Mapping[str, Any],
    generation_mode: str = GENERATION_MODE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Deterministically add canonical object payload around model-owned poses."""

    validated = validate_global_placement(
        placement,
        object_plan=object_plan,
        asset_selection=asset_selection,
    )
    selection = _selection_by_room_and_slot(asset_selection)
    plan_rooms = {str(item["room_id"]): item for item in object_plan["rooms"]}
    placed_rooms = {str(item["room_id"]): item for item in validated["rooms"]}
    rooms: list[dict[str, Any]] = []
    for room_id in object_plan["room_order"]:
        plan_room = plan_rooms[str(room_id)]
        plan_objects = {str(item["id"]): item for item in plan_room["objects"]}
        generated_objects: list[dict[str, Any]] = []
        for instance in placed_rooms[str(room_id)]["instances"]:
            slot_id = str(instance["slot_id"])
            spec = plan_objects[slot_id]
            selected = selection[(str(room_id), slot_id)]
            scale = float(instance["uniform_scale"])
            selected_size = [float(item) for item in selected["size"]]
            size = [item * scale for item in selected_size]
            generated_objects.append(
                {
                    "id": str(instance["instance_id"]),
                    "slot_id": slot_id,
                    "category": str(spec["category"]),
                    "description": str(spec["description"]),
                    "size": size,
                    "center": [float(item) for item in instance["center_m"]],
                    "rotation": [
                        float(item)
                        for item in instance["rotation_euler_xyz_deg"]
                    ],
                    "jid": str(selected["jid"]),
                    "geometry_provenance": "asset_mesh",
                    "asset_ref": deepcopy(selected.get("asset_ref") or {}),
                    "asset_proxy": deepcopy(selected.get("asset_proxy") or {}),
                    "metadata": {
                        "uniform_scale": scale,
                        "task_slot": deepcopy(spec),
                        "retrieval_binding_policy": RETRIEVAL_BINDING_POLICY,
                    },
                }
            )
        rooms.append(
            {
                "room_id": str(room_id),
                **(
                    {"program_id": plan_room.get("program_id")}
                    if "program_id" in plan_room
                    else {}
                ),
                **(
                    {"room_type": plan_room.get("room_type")}
                    if "room_type" in plan_room
                    else {}
                ),
                "objects": generated_objects,
            }
        )
    scene = {
        "schema_version": "non_rectangular_multi_room_scene_v1",
        "layout_id": str(object_plan["layout_id"]),
        "coordinate_frame": deepcopy(room_layout["coordinate_frame"]),
        "room_order": [str(item) for item in object_plan["room_order"]],
        "rooms": rooms,
        "provenance": {
            "generation_mode": generation_mode,
            "stage_c_contract": GLOBAL_PLACEMENT_SCHEMA_VERSION,
            "coordinates_transformed": False,
            "architecture_generated_by_model": False,
        },
    }
    validate_multi_room_scene(scene)
    preflight = prepare_non_rectangular_evaluation(
        NonRectangularEvaluationInput.from_artifacts(
            room_layout=room_layout,
            room_program=room_program,
            object_plan=object_plan,
            generated_scene=scene,
        )
    )
    if not preflight.should_run_room_evaluation:
        raise NonRectangularGenerationContractError(
            "materialized scene unexpectedly failed preflight: "
            f"{preflight.failure_reason}"
        )
    return scene, preflight.public_dict()


def _validate_placement_schema(value: Mapping[str, Any]) -> None:
    try:
        schema = json.loads(
            GLOBAL_PLACEMENT_SCHEMA_PATH.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NonRectangularGenerationContractError(
            "cannot load packaged global-placement schema"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise NonRectangularGenerationContractError(
            f"global-placement schema failed at {path}: {error.message}"
        )


def _selection_by_room_and_slot(
    asset_selection: Mapping[str, Any],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    if asset_selection.get("schema_version") != ASSET_SELECTION_SCHEMA_VERSION:
        raise NonRectangularGenerationContractError(
            "unsupported non-rectangular asset selection"
        )
    output: dict[tuple[str, str], Mapping[str, Any]] = {}
    for room in asset_selection.get("rooms") or []:
        room_id = str(room.get("room_id") or "")
        for item in room.get("objects") or []:
            key = (room_id, str(item.get("slot_id") or ""))
            if not all(key) or key in output:
                raise NonRectangularGenerationContractError(
                    "asset selection room/slot identity must be unique"
                )
            selected = item.get("selected_asset")
            if not isinstance(selected, Mapping):
                raise NonRectangularGenerationContractError(
                    "asset selection entry lacks selected_asset"
                )
            output[key] = selected
    return output


def _validate_room_plan_semantics(
    *,
    room_layout: Mapping[str, Any],
    object_plan: Mapping[str, Any],
) -> None:
    plan_contract_version = validate_multi_room_object_plan(object_plan)[
        "plan_contract_version"
    ]
    layout_rooms = {
        str(room["room_id"]): room for room in room_layout["rooms"]
    }
    for room in object_plan["rooms"]:
        room_id = str(room["room_id"])
        slots = {str(item["id"]): item for item in room["objects"]}
        walls = {
            str(item["wall_id"])
            for item in layout_rooms[room_id]["wall_segments"]
        }
        support_edges: dict[str, str] = {}
        for slot_id, item in slots.items():
            if plan_contract_version == "v1":
                metadata = item["metadata"]
                if str(metadata["intended_role"]) != str(item["role"]):
                    raise NonRectangularGenerationContractError(
                        f"{room_id}.{slot_id} intended_role must equal role"
                    )
                directed = bool(metadata["directed"])
                functional_side = metadata["functional_side"]
                if directed != (functional_side == "local_neg_y"):
                    raise NonRectangularGenerationContractError(
                        f"{room_id}.{slot_id} directed/functional_side mismatch"
                    )
                support = str(metadata["support"])
            else:
                support = str(item["support"])
                if item.get("facing_target") == slot_id:
                    raise NonRectangularGenerationContractError(
                        f"{room_id}.{slot_id} cannot face itself"
                    )
            if support == slot_id:
                raise NonRectangularGenerationContractError(
                    f"{room_id}.{slot_id} cannot support itself"
                )
            if support == "floor" or support in walls:
                continue
            if support not in slots:
                raise NonRectangularGenerationContractError(
                    f"{room_id}.{slot_id} support references unknown same-room target"
                )
            support_edges[slot_id] = support
        _require_acyclic_support(room_id, support_edges)


def _require_acyclic_support(
    room_id: str,
    edges: Mapping[str, str],
) -> None:
    complete: set[str] = set()
    for start in edges:
        if start in complete:
            continue
        path: set[str] = set()
        current = start
        while current in edges:
            if current in path:
                raise NonRectangularGenerationContractError(
                    f"{room_id} object support graph contains a cycle"
                )
            if current in complete:
                break
            path.add(current)
            current = edges[current]
        complete.update(path)


def _retrieval_slot_id(room_id: str, slot_id: str) -> str:
    return f"{room_id}::{slot_id}"


def _finite_vec3(value: Any, path: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise NonRectangularGenerationContractError(f"{path} must be a 3-vector")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise NonRectangularGenerationContractError(
            f"{path} must contain finite numbers"
        )


__all__ = [
    "ASSET_SELECTION_SCHEMA_VERSION",
    "GENERATION_MODE",
    "GENERATION_MODE_V2",
    "GLOBAL_PLACEMENT_SCHEMA_PATH",
    "GLOBAL_PLACEMENT_SCHEMA_VERSION",
    "NonRectangularGenerationContractError",
    "build_global_retrieval_plan",
    "build_stage_a_user_value",
    "build_stage_a_user_value_v2",
    "build_stage_c_user_value",
    "build_stage_c_user_value_v2",
    "group_asset_selection",
    "materialize_generated_scene",
    "validate_global_placement",
    "validate_stage_a_artifacts",
]
