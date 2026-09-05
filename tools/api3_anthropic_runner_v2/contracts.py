"""Scientific contracts for the two model-owned generation stages."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence


class ContractError(ValueError):
    pass


_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_PLAN_KEYS = {
    "schema_version",
    "scene_type",
    "scene_description",
    "prompt_granularity",
    "global_constraints",
    "zones",
    "relations",
    "objects",
}
_ZONE_KEYS = {"id", "description", "extent_hint"}
_OBJECT_KEYS = {
    "id",
    "category",
    "role",
    "description",
    "count",
    "estimated_size",
    "metadata",
    "placement_intent",
}
_METADATA_KEYS = {
    "intended_role",
    "zone",
    "support",
    "directed",
    "functional_side",
    "facing_intent",
    "retrieval_query",
    "requested_count",
}
_PLACEMENT_INTENT_KEYS = {"absolute_relations", "relative_relations"}
_RELATION_KEYS = {"family", "type", "subject_id", "object_id"}
_PLACEMENT_KEYS = {"schema_version", "instances"}
_INSTANCE_KEYS = {
    "instance_id",
    "asset_id",
    "slot_id",
    "center_m",
    "uniform_scale",
    "rotation_euler_xyz_deg",
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ContractError(
            f"{path} keys mismatch: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{path} must be an object")
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{path} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, path: str) -> str:
    text = _text(value, path)
    if not _ID_RE.fullmatch(text):
        raise ContractError(f"{path} must match {_ID_RE.pattern}")
    return text


def _string_list(value: Any, path: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise ContractError(f"{path} must be a list of strings")
    return [_text(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _finite_number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{path} must be finite")
    if positive and result <= 0:
        raise ContractError(f"{path} must be positive")
    return result


def _vec3(value: Any, path: str, *, positive: bool = False, nonnegative: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ContractError(f"{path} must be a 3-vector")
    result = [
        _finite_number(item, f"{path}[{index}]", positive=positive)
        for index, item in enumerate(value)
    ]
    if nonnegative and any(item < 0 for item in result):
        raise ContractError(f"{path} components must be non-negative")
    return result


def validate_brief(value: Any) -> dict[str, Any]:
    brief = dict(_mapping(value, "brief"))
    expected = {
        "brief_id",
        "room_type",
        "room_dimensions_m",
        "target_instances",
        "instruction",
        "physical_wall_policy",
        "active_wall_ids",
    }
    _exact_keys(brief, expected, "brief")
    brief_id = _text(brief["brief_id"], "brief.brief_id")
    if not re.fullmatch(r"brief_\d{2}", brief_id):
        raise ContractError("brief.brief_id must use brief_NN format")
    dimensions = _vec3(brief["room_dimensions_m"], "brief.room_dimensions_m", positive=True)
    target = _mapping(brief["target_instances"], "brief.target_instances")
    _exact_keys(target, {"min", "max"}, "brief.target_instances")
    minimum = target["min"]
    maximum = target["max"]
    if (
        isinstance(minimum, bool)
        or not isinstance(minimum, int)
        or isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or minimum < 1
        or maximum < minimum
    ):
        raise ContractError("brief.target_instances must be a valid positive integer range")
    if brief["physical_wall_policy"] != "explicit_only" or brief["active_wall_ids"] != []:
        raise ContractError("brief must preserve explicit_only with no active wall IDs")
    return {
        **brief,
        "brief_id": brief_id,
        "room_type": _text(brief["room_type"], "brief.room_type"),
        "room_dimensions_m": dimensions,
        "instruction": _text(brief["instruction"], "brief.instruction"),
        "target_instances": {"min": minimum, "max": maximum},
    }


def validate_object_plan(value: Any, *, brief: Mapping[str, Any]) -> dict[str, Any]:
    plan = dict(_mapping(value, "object_plan"))
    _exact_keys(plan, _PLAN_KEYS, "object_plan")
    if plan["schema_version"] != "hy34_object_plan_v2":
        raise ContractError("object_plan.schema_version must be hy34_object_plan_v2")
    zones_raw = plan["zones"]
    if not isinstance(zones_raw, list) or not zones_raw:
        raise ContractError("object_plan.zones must be non-empty")
    zones: list[dict[str, Any]] = []
    zone_ids: set[str] = set()
    for index, raw in enumerate(zones_raw):
        zone = dict(_mapping(raw, f"object_plan.zones[{index}]"))
        _exact_keys(zone, _ZONE_KEYS, f"object_plan.zones[{index}]")
        zone_id = _identifier(zone["id"], f"object_plan.zones[{index}].id")
        if zone_id in zone_ids:
            raise ContractError(f"duplicate zone ID: {zone_id}")
        zone_ids.add(zone_id)
        zones.append(
            {
                "id": zone_id,
                "description": _text(zone["description"], f"object_plan.zones[{index}].description"),
                "extent_hint": _text(zone["extent_hint"], f"object_plan.zones[{index}].extent_hint"),
            }
        )
    objects_raw = plan["objects"]
    if not isinstance(objects_raw, list) or not objects_raw:
        raise ContractError("object_plan.objects must be non-empty")
    objects: list[dict[str, Any]] = []
    object_ids: set[str] = set()
    expanded_count = 0
    for index, raw in enumerate(objects_raw):
        item = dict(_mapping(raw, f"object_plan.objects[{index}]"))
        _exact_keys(item, _OBJECT_KEYS, f"object_plan.objects[{index}]")
        object_id = _identifier(item["id"], f"object_plan.objects[{index}].id")
        if object_id in object_ids:
            raise ContractError(f"duplicate object slot ID: {object_id}")
        object_ids.add(object_id)
        count = item["count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ContractError(f"object_plan.objects[{index}].count must be positive integer")
        expanded_count += count
        metadata = dict(
            _mapping(item["metadata"], f"object_plan.objects[{index}].metadata")
        )
        _exact_keys(
            metadata,
            _METADATA_KEYS,
            f"object_plan.objects[{index}].metadata",
        )
        placement_intent = dict(
            _mapping(
                item["placement_intent"],
                f"object_plan.objects[{index}].placement_intent",
            )
        )
        _exact_keys(
            placement_intent,
            _PLACEMENT_INTENT_KEYS,
            f"object_plan.objects[{index}].placement_intent",
        )
        zone = _identifier(
            metadata["zone"], f"object_plan.objects[{index}].metadata.zone"
        )
        if zone not in zone_ids:
            raise ContractError(f"object slot {object_id} references unknown zone {zone}")
        directed = metadata["directed"]
        if not isinstance(directed, bool):
            raise ContractError(
                f"object_plan.objects[{index}].metadata.directed must be boolean"
            )
        facing_intent = metadata["facing_intent"]
        functional_side = metadata["functional_side"]
        if directed:
            _text(
                facing_intent,
                f"object_plan.objects[{index}].metadata.facing_intent",
            )
            if functional_side != "local_neg_y":
                raise ContractError("directed object functional_side must be local_neg_y")
        elif facing_intent is not None and not isinstance(facing_intent, str):
            raise ContractError(
                f"object_plan.objects[{index}].metadata.facing_intent must be string or null"
            )
        elif functional_side is not None:
            raise ContractError("non-directed object functional_side must be null")
        if metadata["requested_count"] != count:
            raise ContractError("metadata.requested_count must equal object count")
        role = _text(item["role"], f"object_plan.objects[{index}].role")
        if metadata["intended_role"] != role:
            raise ContractError("metadata.intended_role must equal object role")
        objects.append(
            {
                "id": object_id,
                "category": _text(item["category"], f"object_plan.objects[{index}].category"),
                "role": role,
                "description": _text(item["description"], f"object_plan.objects[{index}].description"),
                "count": count,
                "estimated_size": _vec3(
                    item["estimated_size"],
                    f"object_plan.objects[{index}].estimated_size",
                    positive=True,
                ),
                "metadata": {
                    "intended_role": role,
                    "zone": zone,
                    "support": _text(
                        metadata["support"],
                        f"object_plan.objects[{index}].metadata.support",
                    ),
                    "directed": directed,
                    "functional_side": functional_side,
                    "facing_intent": facing_intent,
                    "retrieval_query": _text(
                        metadata["retrieval_query"],
                        f"object_plan.objects[{index}].metadata.retrieval_query",
                    ),
                    "requested_count": count,
                },
                "placement_intent": {
                    "absolute_relations": _string_list(
                        placement_intent["absolute_relations"],
                        f"object_plan.objects[{index}].placement_intent.absolute_relations",
                    ),
                    "relative_relations": _string_list(
                        placement_intent["relative_relations"],
                        f"object_plan.objects[{index}].placement_intent.relative_relations",
                    ),
                },
            }
        )
    for item in objects:
        support = item["metadata"]["support"]
        if support != "floor" and support not in object_ids:
            raise ContractError(
                f"object slot {item['id']} references unknown support {support}"
            )
    relations_raw = plan["relations"]
    if not isinstance(relations_raw, list):
        raise ContractError("object_plan.relations must be a list")
    relations: list[dict[str, str]] = []
    for index, raw in enumerate(relations_raw):
        relation = dict(_mapping(raw, f"object_plan.relations[{index}]"))
        _exact_keys(relation, _RELATION_KEYS, f"object_plan.relations[{index}]")
        subject_id = _identifier(
            relation["subject_id"], f"object_plan.relations[{index}].subject_id"
        )
        object_id = _identifier(
            relation["object_id"], f"object_plan.relations[{index}].object_id"
        )
        if subject_id not in object_ids or object_id not in object_ids:
            raise ContractError("relation endpoints must reference public object slots")
        relations.append(
            {
                "family": _text(
                    relation["family"], f"object_plan.relations[{index}].family"
                ),
                "type": _text(
                    relation["type"], f"object_plan.relations[{index}].type"
                ),
                "subject_id": subject_id,
                "object_id": object_id,
            }
        )
    target = brief["target_instances"]
    if not int(target["min"]) <= expanded_count <= int(target["max"]):
        raise ContractError(
            f"expanded instance count {expanded_count} outside "
            f"[{target['min']}, {target['max']}]"
        )
    prompt_granularity = _text(
        plan["prompt_granularity"], "object_plan.prompt_granularity"
    )
    if prompt_granularity != "fine_grained":
        raise ContractError("object_plan.prompt_granularity must be fine_grained")
    return {
        "schema_version": "hy34_object_plan_v2",
        "scene_type": _text(plan["scene_type"], "object_plan.scene_type"),
        "scene_description": _text(plan["scene_description"], "object_plan.scene_description"),
        "prompt_granularity": prompt_granularity,
        "global_constraints": _string_list(
            plan["global_constraints"], "object_plan.global_constraints"
        ),
        "zones": zones,
        "relations": relations,
        "objects": objects,
    }


def build_retrieval_request(plan: Mapping[str, Any]) -> dict[str, Any]:
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


def validate_placement(
    value: Any,
    *,
    plan: Mapping[str, Any],
    retrieval_results: Mapping[str, Any],
    brief: Mapping[str, Any],
) -> dict[str, Any]:
    placement = dict(_mapping(value, "placement"))
    _exact_keys(placement, _PLACEMENT_KEYS, "placement")
    if placement["schema_version"] != "catalog_placement_v1":
        raise ContractError("placement.schema_version must be catalog_placement_v1")
    slot_counts = {str(item["id"]): int(item["count"]) for item in plan["objects"]}
    assets_by_slot = {
        str(item["slot_id"]): str(item["rank1"]["jid"])
        for item in retrieval_results["results"]
    }
    instances_raw = placement["instances"]
    if not isinstance(instances_raw, list):
        raise ContractError("placement.instances must be a list")
    target = brief["target_instances"]
    if not int(target["min"]) <= len(instances_raw) <= int(target["max"]):
        raise ContractError("placement instance count is outside the frozen target range")
    instances: list[dict[str, Any]] = []
    observed_counts = {slot_id: 0 for slot_id in slot_counts}
    instance_ids: set[str] = set()
    for index, raw in enumerate(instances_raw):
        item = dict(_mapping(raw, f"placement.instances[{index}]"))
        _exact_keys(item, _INSTANCE_KEYS, f"placement.instances[{index}]")
        instance_id = _identifier(item["instance_id"], f"placement.instances[{index}].instance_id")
        if instance_id in instance_ids:
            raise ContractError(f"duplicate instance_id: {instance_id}")
        instance_ids.add(instance_id)
        slot_id = _identifier(item["slot_id"], f"placement.instances[{index}].slot_id")
        if slot_id not in slot_counts:
            raise ContractError(f"placement references unknown slot {slot_id}")
        observed_counts[slot_id] += 1
        asset_id = _text(item["asset_id"], f"placement.instances[{index}].asset_id")
        if asset_id != assets_by_slot.get(slot_id):
            raise ContractError(
                f"slot {slot_id} must use frozen asset {assets_by_slot.get(slot_id)!r}, "
                f"got {asset_id!r}"
            )
        rotation = _vec3(
            item["rotation_euler_xyz_deg"],
            f"placement.instances[{index}].rotation_euler_xyz_deg",
        )
        if abs(rotation[0]) > 1e-9 or abs(rotation[1]) > 1e-9:
            raise ContractError("placement may use yaw only; X/Y rotation must be zero")
        instances.append(
            {
                "instance_id": instance_id,
                "asset_id": asset_id,
                "slot_id": slot_id,
                "center_m": _vec3(
                    item["center_m"],
                    f"placement.instances[{index}].center_m",
                    nonnegative=True,
                ),
                "uniform_scale": _finite_number(
                    item["uniform_scale"],
                    f"placement.instances[{index}].uniform_scale",
                    positive=True,
                ),
                "rotation_euler_xyz_deg": rotation,
            }
        )
    if observed_counts != slot_counts:
        raise ContractError(
            f"placement slot counts differ: expected={slot_counts} actual={observed_counts}"
        )
    return {"schema_version": "catalog_placement_v1", "instances": instances}
