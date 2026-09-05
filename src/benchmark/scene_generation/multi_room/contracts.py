"""Room-wise model contracts layered over the frozen two-stage primitives."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from typing import Any, Mapping

from jsonschema import Draft202012Validator

from benchmark.resources import runtime_resource_path


OBJECT_PLAN_SCHEMA_VERSION = "multi_room_object_plan_v1"
PLACEMENT_SCHEMA_VERSION = "catalog_placement_v1"
OBJECT_PLAN_SCHEMA_PATH = runtime_resource_path(
    "schemas/multi_room/object_plan_v1.schema.json"
)
_WALL_IDS = frozenset(
    {"north_wall", "south_wall", "east_wall", "west_wall"}
)


class MultiRoomContractError(ValueError):
    """Raised when one model-owned room emission violates the new mode."""


def _schema() -> Mapping[str, Any]:
    try:
        value = json.loads(OBJECT_PLAN_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MultiRoomContractError(
            f"cannot load packaged object-plan schema: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise MultiRoomContractError("packaged object-plan schema must be an object")
    return value


def _validate_schema(value: Mapping[str, Any]) -> None:
    errors = sorted(
        Draft202012Validator(_schema()).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        error = errors[0]
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise MultiRoomContractError(
            f"multi-room object-plan schema failed at {path}: {error.message}"
        )


def validate_room_object_plan(
    value: Mapping[str, Any],
    *,
    room_brief: Mapping[str, Any],
    frozen_validate_object_plan: Any,
) -> dict[str, Any]:
    """Validate a new plan while reusing all unchanged frozen plan semantics.

    The frozen validator remains authoritative for exact keys, IDs, counts,
    relations, facing, retrieval-query, and support-graph semantics.  This
    compatibility validator changes only the version tag and permits a
    declared active wall as an architecture support target.
    """

    if not isinstance(value, Mapping):
        raise MultiRoomContractError("multi-room object plan must be an object")
    plan = deepcopy(dict(value))
    _validate_schema(plan)
    if plan.get("schema_version") != OBJECT_PLAN_SCHEMA_VERSION:
        raise MultiRoomContractError("unsupported multi-room object-plan version")
    if plan.get("scene_type") != room_brief.get("room_type"):
        raise MultiRoomContractError(
            "object_plan.scene_type must equal the current room_type"
        )
    architecture = room_brief.get("architecture")
    if not isinstance(architecture, Mapping):
        raise MultiRoomContractError("room brief has no architecture contract")
    active_walls = frozenset(str(item) for item in architecture["active_wall_ids"])
    if not active_walls <= _WALL_IDS:
        raise MultiRoomContractError("room brief contains an unsupported wall ID")

    original_support: dict[str, str] = {}
    normalized = deepcopy(plan)
    normalized["schema_version"] = "hy34_object_plan_v2"
    for index, item in enumerate(normalized["objects"]):
        slot_id = str(item.get("id") or "")
        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            raise MultiRoomContractError(
                f"object_plan.objects[{index}].metadata must be an object"
            )
        support = str(metadata.get("support") or "")
        original_support[slot_id] = support
        if support in _WALL_IDS:
            if support not in active_walls:
                raise MultiRoomContractError(
                    f"object slot {slot_id!r} targets inactive wall {support!r}"
                )
            # The frozen validator knows floor and public-slot support only.
            # Mapping a declared architecture support to floor for that one
            # validation call preserves every other frozen check; the original
            # wall support is restored immediately below.
            metadata["support"] = "floor"
    legacy_brief = {
        "target_instances": deepcopy(room_brief["target_instances"]),
    }
    try:
        validated = frozen_validate_object_plan(normalized, brief=legacy_brief)
    except Exception as exc:
        raise MultiRoomContractError(str(exc)) from exc
    validated["schema_version"] = OBJECT_PLAN_SCHEMA_VERSION
    attached_count = 0
    for item in validated["objects"]:
        support = original_support[item["id"]]
        item["metadata"]["support"] = support
        if support in active_walls:
            attached_count += int(item["count"])
    required = room_brief["wall_attachment_requirement"]
    minimum = int(required["minimum_count"])
    maximum = int(required["maximum_count"])
    if not minimum <= attached_count <= maximum:
        raise MultiRoomContractError(
            "expanded wall-attached instance count "
            f"{attached_count} is outside [{minimum}, {maximum}]"
        )
    return validated


def build_retrieval_request(
    plan: Mapping[str, Any], *, frozen_build_retrieval_request: Any
) -> dict[str, Any]:
    """Build the unchanged one-request-per-public-slot Top-1 payload."""

    request = frozen_build_retrieval_request(plan)
    if len(request.get("requests", [])) != len(plan.get("objects", [])):
        raise MultiRoomContractError(
            "retrieval request must contain exactly one entry per public slot"
        )
    slot_ids = [str(item["slot_id"]) for item in request["requests"]]
    if len(slot_ids) != len(set(slot_ids)):
        raise MultiRoomContractError("retrieval request contains duplicate slots")
    return request


def validate_retrieval_results(
    value: Mapping[str, Any], *, plan: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed if the shared retriever returns anything but one rank-1/slot."""

    if not isinstance(value, Mapping) or not isinstance(value.get("results"), list):
        raise MultiRoomContractError("retrieval results must contain a results array")
    expected_top_keys = {
        "schema_version",
        "total_invocations",
        "retry_count",
        "asset_replacement_count",
        "results",
    }
    if set(value) != expected_top_keys:
        raise MultiRoomContractError("retrieval results keys differ from frozen contract")
    if value.get("schema_version") != "hy34_frozen_top1_results_v1":
        raise MultiRoomContractError("unsupported retrieval results schema")
    expected = [str(item["id"]) for item in plan["objects"]]
    query_by_slot = {
        str(item["id"]): str(item["metadata"]["retrieval_query"])
        for item in plan["objects"]
    }
    for field, expected_value in (
        ("total_invocations", len(expected)),
        ("retry_count", 0),
        ("asset_replacement_count", 0),
    ):
        actual = value.get(field)
        if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected_value:
            raise MultiRoomContractError(
                f"retrieval results {field} must equal {expected_value}"
            )
    actual: list[str] = []
    normalized = deepcopy(dict(value))
    for index, row in enumerate(normalized["results"]):
        if not isinstance(row, dict):
            raise MultiRoomContractError(f"retrieval results[{index}] must be an object")
        if set(row) != {
            "order",
            "slot_id",
            "retrieval_query",
            "size_constraint",
            "invocation_count",
            "rank1",
            "accepted_as_frozen_outcome",
        }:
            raise MultiRoomContractError(
                f"retrieval results[{index}] keys differ from frozen contract"
            )
        slot_id = row.get("slot_id")
        query = row.get("retrieval_query")
        rank1 = row.get("rank1")
        if not isinstance(slot_id, str) or not slot_id:
            raise MultiRoomContractError(f"retrieval results[{index}] has no slot_id")
        if not isinstance(query, str) or not query.strip():
            raise MultiRoomContractError(
                f"retrieval results[{index}] has no retrieval_query"
            )
        if not isinstance(rank1, dict):
            raise MultiRoomContractError(f"retrieval results[{index}] has no rank1")
        if (
            isinstance(row.get("order"), bool)
            or not isinstance(row.get("order"), int)
            or row.get("order") != index
        ):
            raise MultiRoomContractError(
                f"retrieval results[{index}].order must equal {index}"
            )
        invocation_count = row.get("invocation_count")
        if (
            isinstance(invocation_count, bool)
            or not isinstance(invocation_count, int)
            or invocation_count != 1
        ):
            raise MultiRoomContractError(
                f"retrieval results[{index}].invocation_count must equal 1"
            )
        if row.get("accepted_as_frozen_outcome") is not True:
            raise MultiRoomContractError(
                f"retrieval results[{index}] is not an accepted frozen outcome"
            )
        if row.get("size_constraint") is not None:
            raise MultiRoomContractError(
                f"retrieval results[{index}] unexpectedly used a size constraint"
            )
        if query != query_by_slot.get(slot_id):
            raise MultiRoomContractError(
                f"retrieval results[{index}] rewrote the frozen retrieval query"
            )
        rank = rank1.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank != 1:
            raise MultiRoomContractError(
                f"retrieval results[{index}].rank1.rank must equal 1"
            )
        if not isinstance(rank1.get("jid"), str) or not rank1["jid"].strip():
            raise MultiRoomContractError(
                f"retrieval results[{index}].rank1.jid is invalid"
            )
        # The frozen Imaginarium retriever preserves catalog text fields exactly,
        # including empty strings.  Asset identity is carried by ``jid``; empty
        # display metadata must not turn an accepted frozen Top-1 result into a
        # retrieval failure.  The canonical projection layer deterministically
        # falls back to the benchmark-owned public task-slot semantics when one
        # of these optional catalog strings is blank.
        for field in ("category", "description", "short_desc"):
            if not isinstance(rank1.get(field), str):
                raise MultiRoomContractError(
                    f"retrieval results[{index}].rank1.{field} is invalid"
                )
        index_row = rank1.get("index_row")
        if (
            isinstance(index_row, bool)
            or not isinstance(index_row, int)
            or index_row < 0
        ):
            raise MultiRoomContractError(
                f"retrieval results[{index}].rank1.index_row is invalid"
            )
        size = rank1.get("size")
        if not isinstance(size, list) or len(size) != 3:
            raise MultiRoomContractError(
                f"retrieval results[{index}].rank1.size must be a 3-vector"
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0
            for item in size
        ):
            raise MultiRoomContractError(
                f"retrieval results[{index}].rank1.size must be finite and positive"
            )
        score = rank1.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise MultiRoomContractError(
                f"retrieval results[{index}].rank1.score must be finite"
            )
        actual.append(slot_id)
    if actual != expected:
        raise MultiRoomContractError(
            f"retrieval slot order/identity differs: expected={expected} actual={actual}"
        )
    return normalized


def build_asset_selection(
    plan: Mapping[str, Any], retrieval_results: Mapping[str, Any]
) -> dict[str, Any]:
    """Build the same deterministic selected-asset projection as the frozen core."""

    plan_by_id = {str(item["id"]): item for item in plan["objects"]}
    selection_reason = (
        "frozen deterministic semantic Top-1 retrieval result; "
        "no size soft score, rewrite, retry, or replacement"
    )
    objects: list[dict[str, Any]] = []
    for row in retrieval_results["results"]:
        slot_id = str(row["slot_id"])
        rank1 = row["rank1"]
        size = [float(value) for value in rank1["size"]]
        category = str(rank1.get("category") or "")
        description = str(rank1.get("description") or "")
        short_description = str(rank1.get("short_desc") or "")
        selected_asset = {
            "jid": str(rank1["jid"]),
            "category": category,
            "desc": description,
            "short_desc": short_description,
            "size": size,
            "asset_ref": {
                "source_db": "imaginarium",
                "asset_key": str(rank1["jid"]),
            },
            "asset_proxy": {
                "type": "canonical_catalog_bbox",
                "bbox_center_local": [0.0, 0.0, 0.0],
                "bbox_size": size,
            },
            "metadata": {
                "catalog_facing_contract_version": "imaginarium_catalog_facing_v1",
                "default_directed_functional_side": "local_neg_y",
            },
        }
        objects.append(
            {
                "object_id": slot_id,
                "object_spec": deepcopy(plan_by_id[slot_id]),
                "retrieval_query": {
                    "description": str(row["retrieval_query"]),
                    "category": None,
                    "size_constraint": None,
                    "top_k": 1,
                },
                "selected_asset": selected_asset,
                "candidates": [
                    {
                        "rank": 1,
                        "jid": str(rank1["jid"]),
                        "category": category,
                        "short_desc": short_description,
                        "description": description,
                        "size": size,
                        "score": float(rank1["score"]),
                    }
                ],
                "selection_action": "select",
                "selection_decision": {
                    "action": "select",
                    "selected_jid": str(rank1["jid"]),
                    "reason": selection_reason,
                    "generation_request": None,
                },
                "selection_reason": selection_reason,
            }
        )
    return {"schema_version": "multi_room_frozen_asset_selection_v1", "objects": objects}


def build_generation_input(
    *,
    room_brief: Mapping[str, Any],
    plan: Mapping[str, Any],
    asset_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one room-isolated Stage C payload with explicit architecture."""

    local_room = deepcopy(room_brief["local_room"])
    width, depth, height = [float(value) for value in room_brief["room_dimensions_m"]]
    room = {
        "boundary": deepcopy(local_room["boundary"]),
        "height": float(local_room["height"]),
        "unit": "meter",
        "dimensions": {"width": width, "depth": depth, "height": height},
        "floor_z": 0.0,
        "topology": "rectangular_logical_boundary",
    }
    return {
        "schema_version": "multi_room_generation_input_v1",
        "generation_mode": "multi_room_with_architecture_v1",
        "layout_id": room_brief["layout_id"],
        "room_id": room_brief["room_id"],
        "scene_request": {
            "instruction": room_brief["instruction"],
            "theme": room_brief["theme"],
            "scene_type": plan["scene_type"],
            "structure": True,
            "prompt_granularity": plan["prompt_granularity"],
            "room_type": room_brief["room_type"],
            "target_instances": deepcopy(room_brief["target_instances"]),
            "room": room,
        },
        "generation_contract": {
            "output_format": PLACEMENT_SCHEMA_VERSION,
            "requires_pose": True,
            "input_mode": "structured_assets",
            "requires_asset_selection": True,
            "coordinate_frame": "room_min_corner_x_width_y_depth_z_up_meters",
            "architecture": deepcopy(room_brief["architecture"]),
            "wall_attachment_requirement": deepcopy(
                room_brief["wall_attachment_requirement"]
            ),
            "catalog_facing_prior": "directed_local_neg_y",
            "one_shot": True,
        },
        "object_plan": deepcopy(plan),
        "asset_selection": deepcopy(asset_selection),
    }


def validate_room_placement(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    retrieval_results: Mapping[str, Any],
    room_brief: Mapping[str, Any],
    frozen_validate_placement: Any,
) -> dict[str, Any]:
    """Validate unchanged slot/count/asset/pose semantics for one room."""

    if not isinstance(value, Mapping):
        raise MultiRoomContractError("room placement must be an object")
    try:
        validated = frozen_validate_placement(
            value,
            plan=plan,
            retrieval_results=retrieval_results,
            brief={"target_instances": room_brief["target_instances"]},
        )
    except Exception as exc:
        raise MultiRoomContractError(str(exc)) from exc
    if validated.get("schema_version") != PLACEMENT_SCHEMA_VERSION:
        raise MultiRoomContractError("unsupported placement schema version")
    offset = [float(item) for item in room_brief["local_to_global_offset_m"]]
    for index, instance in enumerate(validated["instances"]):
        local = [float(item) for item in instance["center_m"]]
        global_center = [local[axis] + offset[axis] for axis in range(3)]
        if not all(math.isfinite(item) for item in global_center) or any(
            not math.isclose(
                global_center[axis] - offset[axis],
                local[axis],
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for axis in range(3)
        ):
            raise MultiRoomContractError(
                "placement coordinate cannot preserve local/global translation "
                f"at instances[{index}]"
            )
    return validated
