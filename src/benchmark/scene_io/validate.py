from __future__ import annotations

import math
from typing import Any

from shapely.geometry import Polygon

from benchmark.io_contracts import (
    EVALUATOR_OUTPUT_TYPES,
    input_type_for_mode,
)
from benchmark.nl_scene.converter import PROMPT_GRANULARITIES


class ArtifactValidationError(ValueError):
    """Raised when a canonical harness artifact is malformed."""


STRUCTURED_ASSETS_INPUT_MODE = "structured_assets"
STRUCTURED_NATURAL_LANGUAGE_INPUT_MODE = "natural_language_structured"
DIRECT_NATURAL_LANGUAGE_INPUT_MODE = "natural_language_direct"


def validate_scene_request(scene_request: dict) -> dict:
    _require_mapping(scene_request, "scene_request")
    _require_string(scene_request, "request_id", "scene_request")
    _require_string(scene_request, "instruction", "scene_request")
    if "structure" in scene_request:
        _require_bool(scene_request.get("structure"), "scene_request.structure")
    if (
        scene_request.get("prompt_granularity") is not None
        and scene_request.get("prompt_granularity") not in PROMPT_GRANULARITIES
    ):
        raise ArtifactValidationError(
            f"scene_request.prompt_granularity must be one of {sorted(PROMPT_GRANULARITIES)}"
        )
    if scene_request.get("room") is not None:
        room = _require_mapping(scene_request.get("room"), "scene_request.room")
        _require_boundary(room.get("boundary"), "scene_request.room.boundary")
        if "height" in room:
            _require_positive_number(room.get("height"), "scene_request.room.height")
        _validate_resolved_room_metadata(room)
    return scene_request


def validate_object_plan(object_plan: dict) -> dict:
    _require_mapping(object_plan, "object_plan")
    _require_string(object_plan, "request_id", "object_plan")
    _require_string(object_plan, "scene_type", "object_plan")
    _require_string(object_plan, "scene_description", "object_plan")
    if (
        object_plan.get("prompt_granularity") is not None
        and object_plan.get("prompt_granularity") not in PROMPT_GRANULARITIES
    ):
        raise ArtifactValidationError(
            f"object_plan.prompt_granularity must be one of {sorted(PROMPT_GRANULARITIES)}"
        )
    if "explicit_claims" in object_plan:
        _require_list(object_plan.get("explicit_claims"), "object_plan.explicit_claims")
    _require_list(object_plan.get("global_constraints"), "object_plan.global_constraints")
    relations = _require_list(object_plan.get("relations"), "object_plan.relations")
    objects = _require_list(object_plan.get("objects"), "object_plan.objects")
    object_ids: set[str] = set()
    for index, obj in enumerate(objects):
        path = f"object_plan.objects[{index}]"
        _require_mapping(obj, path)
        object_id = _require_string(obj, "id", path)
        if object_id in object_ids:
            raise ArtifactValidationError(f"{path}.id duplicates object_plan object id {object_id!r}")
        object_ids.add(object_id)
        _require_string(obj, "category", path)
        _require_string(obj, "description", path)
        _forbid_keys(obj, {"center", "position", "rotation", "target_pose", "pose", "jid", "asset_jid", "asset_id", "asset_ref", "expected_relations"}, path)
        if "estimated_size" in obj:
            _require_vector3(obj["estimated_size"], f"{path}.estimated_size", positive=True)
        if "count" in obj:
            _require_positive_integer(obj["count"], f"{path}.count")
        _require_mapping(obj.get("metadata"), f"{path}.metadata")
        placement = _require_mapping(obj.get("placement_intent"), f"{path}.placement_intent")
        for key in ["absolute_relations", "relative_relations"]:
            _require_list(placement.get(key), f"{path}.placement_intent.{key}")
    for index, relation in enumerate(relations):
        _validate_object_plan_relation(relation, object_ids, f"object_plan.relations[{index}]")
    return object_plan


def validate_asset_selection(asset_selection: dict) -> dict:
    _require_mapping(asset_selection, "asset_selection")
    _require_string(asset_selection, "request_id", "asset_selection")
    objects = _require_list(asset_selection.get("objects"), "asset_selection.objects")
    for index, item in enumerate(objects):
        path = f"asset_selection.objects[{index}]"
        _require_mapping(item, path)
        _require_string(item, "object_id", path)
        _require_mapping(item.get("object_spec"), f"{path}.object_spec")
        selected = _require_mapping(item.get("selected_asset"), f"{path}.selected_asset")
        _require_string(selected, "jid", f"{path}.selected_asset")
        if "size" in selected:
            _require_vector3(selected["size"], f"{path}.selected_asset.size", positive=True)
        _require_mapping(selected.get("asset_ref"), f"{path}.selected_asset.asset_ref")
        asset_proxy = _require_mapping(selected.get("asset_proxy"), f"{path}.selected_asset.asset_proxy")
        if "bbox_size" in asset_proxy:
            _require_vector3(asset_proxy["bbox_size"], f"{path}.selected_asset.asset_proxy.bbox_size", positive=True)
        selected_metadata = _require_mapping(selected.get("metadata"), f"{path}.selected_asset.metadata")
        _require_list(item.get("candidates"), f"{path}.candidates")
        selection_action = item.get("selection_action")
        if selection_action is not None:
            if selection_action not in {"select", "generate"}:
                raise ArtifactValidationError(f"{path}.selection_action must be select or generate")
            decision = _require_mapping(item.get("selection_decision"), f"{path}.selection_decision")
            if decision.get("action") != selection_action:
                raise ArtifactValidationError(f"{path}.selection_decision.action must match selection_action")
            if selection_action == "generate":
                if selected.get("asset_ref", {}).get("source_db") != "generated":
                    raise ArtifactValidationError(f"{path}.selected_asset.asset_ref.source_db must be generated")
                if selected_metadata.get("generated") is not True:
                    raise ArtifactValidationError(f"{path}.selected_asset.metadata.generated must be true")
    return asset_selection


def validate_generation_input(generation_input: dict) -> dict:
    _require_mapping(generation_input, "generation_input")
    _forbid_keys(
        generation_input,
        {
            "scene_spec",
            "asset_retrieval",
            "selected_assets",
            "original_instruction",
            "reference_annotation",
            "evaluation_context",
        },
        "generation_input",
    )
    _require_string(generation_input, "request_id", "generation_input")
    validate_scene_request(_require_mapping(generation_input.get("scene_request"), "generation_input.scene_request"))
    contract = _require_mapping(generation_input.get("generation_contract"), "generation_input.generation_contract")
    if contract.get("output_format") != "canonical_generated_scene_v1":
        raise ArtifactValidationError("generation_input.generation_contract.output_format must be canonical_generated_scene_v1")
    if contract.get("requires_pose") is not True:
        raise ArtifactValidationError("generation_input.generation_contract.requires_pose must be true")
    input_mode = str(contract.get("input_mode") or STRUCTURED_ASSETS_INPUT_MODE)
    try:
        expected_input_type = input_type_for_mode(input_mode)
    except ValueError as exc:
        raise ArtifactValidationError(str(exc)) from exc
    input_type = str(contract.get("input_type") or expected_input_type)
    if input_type != expected_input_type:
        raise ArtifactValidationError(
            f"generation_input.generation_contract.input_type must be {expected_input_type} for {input_mode}"
        )
    evaluator_output_type = str(contract.get("evaluator_output_type") or "o1_object_state")
    if evaluator_output_type not in EVALUATOR_OUTPUT_TYPES:
        raise ArtifactValidationError(
            "generation_input.generation_contract.evaluator_output_type must be "
            "o1_object_state or o3_scene_package"
        )
    if input_mode == STRUCTURED_ASSETS_INPUT_MODE:
        _validate_requires_asset_selection(contract, True)
        validate_object_plan(
            _require_mapping(generation_input.get("object_plan"), "generation_input.object_plan")
        )
        validate_asset_selection(_require_mapping(generation_input.get("asset_selection"), "generation_input.asset_selection"))
    elif input_mode == STRUCTURED_NATURAL_LANGUAGE_INPUT_MODE:
        _validate_requires_asset_selection(contract, False)
        public_plan = _require_mapping(generation_input.get("object_plan"), "generation_input.object_plan")
        validate_object_plan(public_plan)
        if generation_input.get("asset_selection") is not None:
            raise ArtifactValidationError(
                "generation_input.asset_selection must be omitted for natural_language_structured"
            )
        generator_input = _require_mapping(generation_input.get("generator_input"), "generation_input.generator_input")
        if generator_input.get("input_mode") != STRUCTURED_NATURAL_LANGUAGE_INPUT_MODE:
            raise ArtifactValidationError(
                "generation_input.generator_input.input_mode must be natural_language_structured"
            )
        _require_string(generator_input, "instruction", "generation_input.generator_input")
        validate_object_plan(
            _require_mapping(generator_input.get("object_plan"), "generation_input.generator_input.object_plan")
        )
        if generator_input.get("object_plan") != public_plan:
            raise ArtifactValidationError(
                "generation_input.generator_input.object_plan must match the public top-level object_plan"
            )
    elif input_mode == DIRECT_NATURAL_LANGUAGE_INPUT_MODE:
        _validate_requires_asset_selection(contract, False)
        if generation_input.get("object_plan") is not None:
            raise ArtifactValidationError(
                "generation_input.object_plan must be omitted for natural_language_direct"
            )
        generator_input = _require_mapping(generation_input.get("generator_input"), "generation_input.generator_input")
        if generator_input.get("input_mode") != DIRECT_NATURAL_LANGUAGE_INPUT_MODE:
            raise ArtifactValidationError("generation_input.generator_input.input_mode must be natural_language_direct")
        _require_string(generator_input, "instruction", "generation_input.generator_input")
        if generator_input.get("room") is not None:
            room = _require_mapping(generator_input.get("room"), "generation_input.generator_input.room")
            _require_boundary(room.get("boundary"), "generation_input.generator_input.room.boundary")
        if generation_input.get("asset_selection") is not None:
            raise ArtifactValidationError(
                "generation_input.asset_selection must be omitted for natural_language_direct"
            )
    else:
        raise ArtifactValidationError(
            "generation_input.generation_contract.input_mode must be structured_assets, "
            "natural_language_structured, or natural_language_direct"
        )
    return generation_input


CANONICAL_SCENE_SCHEMA_VERSION = "canonical_scene_v1"
BBOX_PROXY_GEOMETRY = "bbox_proxy"
ASSET_MESH_GEOMETRY = "asset_mesh"
GENERATED_MESH_GEOMETRY = "generated_mesh"
GEOMETRY_PROVENANCE_VALUES = (BBOX_PROXY_GEOMETRY, ASSET_MESH_GEOMETRY, GENERATED_MESH_GEOMETRY)


def validate_generated_scene(scene: dict) -> dict:
    _require_mapping(scene, "generated_scene")
    if scene.get("schema_version") != CANONICAL_SCENE_SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"generated_scene.schema_version must be {CANONICAL_SCENE_SCHEMA_VERSION!r}"
        )
    if "assets" in scene:
        raise ArtifactValidationError("generated_scene must use objects, not assets")
    _require_string(scene, "scene_id", "generated_scene")
    _require_string(scene, "request_id", "generated_scene")
    _require_string(scene, "scene_type", "generated_scene")
    _require_boundary(scene.get("boundary"), "generated_scene.boundary")
    _require_axis_aligned_rectangular_boundary(scene["boundary"], "generated_scene.boundary")
    _require_min_corner_boundary(scene["boundary"], "generated_scene.boundary")
    _require_positive_number(scene.get("scene_height"), "generated_scene.scene_height")
    metadata = _require_mapping(scene.get("metadata"), "generated_scene.metadata")
    _validate_coordinate_frame(metadata.get("coordinate_frame"), "generated_scene.metadata.coordinate_frame")
    objects = _require_list(scene.get("objects"), "generated_scene.objects")
    object_ids: set[str] = set()
    for index, obj in enumerate(objects):
        path = f"generated_scene.objects[{index}]"
        _require_mapping(obj, path)
        object_id = _require_string(obj, "id", path)
        if object_id in object_ids:
            raise ArtifactValidationError(f"{path}.id duplicates generated object id {object_id!r}")
        object_ids.add(object_id)
        _require_string(obj, "category", path)
        if "description" in obj:
            _require_string(obj, "description", path)
        _require_vector3(obj.get("size"), f"{path}.size", positive=True)
        _require_vector3(obj.get("center"), f"{path}.center")
        _require_vector3(obj.get("rotation"), f"{path}.rotation")
        if "geometry_provenance" in obj and obj.get("geometry_provenance") not in GEOMETRY_PROVENANCE_VALUES:
            raise ArtifactValidationError(
                f"{path}.geometry_provenance must be one of {list(GEOMETRY_PROVENANCE_VALUES)}"
            )
        if "jid" in obj:
            _require_string(obj, "jid", path)
        if "asset_proxy" in obj:
            asset_proxy = _require_mapping(obj.get("asset_proxy"), f"{path}.asset_proxy")
            _require_string(asset_proxy, "type", f"{path}.asset_proxy")
            _require_vector3(asset_proxy.get("bbox_size"), f"{path}.asset_proxy.bbox_size", positive=True)
            if "bbox_center_local" in asset_proxy:
                _require_vector3(asset_proxy.get("bbox_center_local"), f"{path}.asset_proxy.bbox_center_local")
        if "asset_ref" in obj:
            asset_ref = _require_mapping(obj.get("asset_ref"), f"{path}.asset_ref")
            _require_string(asset_ref, "source_db", f"{path}.asset_ref")
            _require_string(asset_ref, "asset_key", f"{path}.asset_ref")
        _require_mapping(obj.get("metadata"), f"{path}.metadata")
    for key in ["relations", "oor_relations"]:
        if key not in scene:
            continue
        relations = _require_list(scene.get(key), f"generated_scene.{key}")
        for index, relation in enumerate(relations):
            _validate_generated_relation(
                relation,
                object_ids,
                f"generated_scene.{key}[{index}]",
                architecture_allowed=False,
            )
    if "oar_relations" in scene:
        relations = _require_list(scene.get("oar_relations"), "generated_scene.oar_relations")
        for index, relation in enumerate(relations):
            _validate_generated_oar_relation(
                relation,
                object_ids,
                f"generated_scene.oar_relations[{index}]",
            )
    return scene


def validate_scene_package(
    scene: dict,
    *,
    allowed_asset_ids: list[str] | tuple[str, ...] | set[str] | None = None,
    require_fixed_catalog: bool = False,
) -> dict:
    """Validate the stricter O3 profile over canonical generated_scene JSON."""

    validate_generated_scene(scene)
    if allowed_asset_ids is not None and not isinstance(allowed_asset_ids, (list, tuple, set)):
        raise ArtifactValidationError("allowed_asset_ids must be a list, tuple, or set")
    allowed = {str(value) for value in allowed_asset_ids} if allowed_asset_ids is not None else None
    if require_fixed_catalog and not allowed:
        raise ArtifactValidationError(
            "official O3 scene packages require a non-empty fixed-catalog allowed_asset_ids"
        )
    for index, obj in enumerate(scene["objects"]):
        path = f"generated_scene.objects[{index}]"
        _require_string(obj, "jid", path)
        asset_ref = _require_mapping(obj.get("asset_ref"), f"{path}.asset_ref")
        _require_string(asset_ref, "source_db", f"{path}.asset_ref")
        _require_string(asset_ref, "asset_key", f"{path}.asset_ref")
        asset_key = str(asset_ref.get("asset_key") or "")
        source_db = str(asset_ref.get("source_db") or "")
        if not source_db or source_db == "layout_json_proxy" or asset_key.startswith("layout_json_proxy:"):
            raise ArtifactValidationError(f"{path} must resolve to a real O3 asset, not a layout proxy")
        if allowed is not None and asset_key not in allowed:
            raise ArtifactValidationError(f"{path}.asset_ref.asset_key {asset_key!r} is outside the fixed catalog")
    return scene


def _require_mapping(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"{path} must be a JSON object")
    return value


def _require_list(value: Any, path: str) -> list:
    if not isinstance(value, list):
        raise ArtifactValidationError(f"{path} must be a JSON list")
    return value


def _require_string(mapping: dict, key: str, path: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{path}.{key} must be a non-empty string")
    return value


def _require_boundary(value: Any, path: str) -> None:
    points = _require_list(value, path)
    if len(points) < 3:
        raise ArtifactValidationError(f"{path} must contain at least three points")
    for index, point in enumerate(points):
        if not isinstance(point, list) or len(point) != 2:
            raise ArtifactValidationError(f"{path}[{index}] must be [x, y]")
        _number(point[0], f"{path}[{index}][0]")
        _number(point[1], f"{path}[{index}][1]")
    numeric_points = [(float(point[0]), float(point[1])) for point in points]
    if len(set(numeric_points)) < 3:
        raise ArtifactValidationError(f"{path} must contain at least three distinct points")
    polygon = Polygon(numeric_points)
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
        raise ArtifactValidationError(f"{path} must be a valid non-self-intersecting polygon with positive area")


def _require_vector3(value: Any, path: str, *, positive: bool = False) -> None:
    if not isinstance(value, list) or len(value) != 3:
        raise ArtifactValidationError(f"{path} must be a 3-vector")
    for index in range(3):
        number = _number(value[index], f"{path}[{index}]")
        if positive and number <= 0:
            raise ArtifactValidationError(f"{path}[{index}] must be positive")


def _validate_resolved_room_metadata(room: dict) -> None:
    dimensions = room.get("dimensions")
    if dimensions is None:
        return
    dimensions = _require_mapping(dimensions, "scene_request.room.dimensions")
    resolved = {}
    for axis in ("width", "depth", "height"):
        if axis not in dimensions:
            raise ArtifactValidationError(f"scene_request.room.dimensions.{axis} is required")
        _require_positive_number(dimensions[axis], f"scene_request.room.dimensions.{axis}")
        resolved[axis] = float(dimensions[axis])
    boundary = room["boundary"]
    width = max(float(point[0]) for point in boundary) - min(float(point[0]) for point in boundary)
    depth = max(float(point[1]) for point in boundary) - min(float(point[1]) for point in boundary)
    tolerance = 1.0e-6
    for axis, actual in (("width", width), ("depth", depth), ("height", float(room.get("height")))):
        if abs(actual - resolved[axis]) > tolerance:
            raise ArtifactValidationError(
                f"scene_request.room.dimensions.{axis} conflicts with room geometry"
            )
    explicit = room.get("explicit_dimensions")
    if explicit is not None:
        explicit = _require_mapping(explicit, "scene_request.room.explicit_dimensions")
        for axis, value in explicit.items():
            if axis not in {"width", "depth", "height"}:
                raise ArtifactValidationError(
                    f"scene_request.room.explicit_dimensions contains unknown axis {axis!r}"
                )
            _require_positive_number(value, f"scene_request.room.explicit_dimensions.{axis}")
            if abs(float(value) - resolved[axis]) > tolerance:
                raise ArtifactValidationError(
                    f"scene_request.room.explicit_dimensions.{axis} conflicts with resolved dimensions"
                )
    provenance = _require_mapping(
        room.get("dimension_provenance"),
        "scene_request.room.dimension_provenance",
    )
    for axis in ("width", "depth", "height"):
        _require_string(provenance, axis, "scene_request.room.dimension_provenance")
    if room.get("resolution_policy") != "room_dimension_policy_v1":
        raise ArtifactValidationError(
            "scene_request.room.resolution_policy must be 'room_dimension_policy_v1'"
        )
    if room.get("topology") != "rectangular_enclosed_room":
        raise ArtifactValidationError(
            "scene_request.room.topology must be 'rectangular_enclosed_room'"
        )
    if float(room.get("floor_z", 0.0)) != 0.0:
        raise ArtifactValidationError("scene_request.room.floor_z must be 0")


def _require_positive_number(value: Any, path: str) -> None:
    number = _number(value, path)
    if number <= 0:
        raise ArtifactValidationError(f"{path} must be positive")


def _require_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ArtifactValidationError(f"{path} must be boolean")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactValidationError(f"{path} must be a JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactValidationError(f"{path} must be finite")
    return number


def _require_positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactValidationError(f"{path} must be a positive integer")
    return value


def _validate_coordinate_frame(value: Any, path: str) -> None:
    frame = _require_mapping(value, path)
    expected = {
        "origin": "room_min_corner_floor",
        "axes": "x_width_y_depth_z_up",
        "unit": "meter",
        "rotation_unit": "degree",
    }
    for key, required_value in expected.items():
        if frame.get(key) != required_value:
            raise ArtifactValidationError(f"{path}.{key} must be {required_value!r}")
    extra_keys = sorted(set(frame) - set(expected))
    if extra_keys:
        raise ArtifactValidationError(f"{path} contains unknown fields: {extra_keys}")


def _validate_object_plan_relation(relation: Any, object_ids: set[str], path: str) -> None:
    relation = _require_mapping(relation, path)
    family = _non_empty_string(relation.get("family"), f"{path}.family").lower()
    if family not in {"oor", "oar"}:
        raise ArtifactValidationError(f"{path}.family must be 'oor' or 'oar'")
    relation_type = _non_empty_string(
        _first_value(relation, ["type", "predicate", "relation"]),
        f"{path}.type",
    ).strip().lower()

    subject = _first_value(relation, ["subject_id", "subject"])
    subject_ids = _object_id_list(relation.get("subject_ids"), f"{path}.subject_ids", object_ids)
    anchor = _first_value(relation, ["object_id", "anchor_id", "target_id", "object"])
    anchor_ids = _object_id_list(relation.get("object_ids"), f"{path}.object_ids", object_ids)

    if family == "oar":
        subject_id = _non_empty_string(subject, f"{path}.subject_id")
        if subject_id not in object_ids:
            raise ArtifactValidationError(f"{path}.subject_id references unknown object id {subject_id!r}")
        return

    if relation_type == "between":
        subject_id = _non_empty_string(subject, f"{path}.subject_id")
        if subject_id not in object_ids:
            raise ArtifactValidationError(f"{path}.subject_id references unknown object id {subject_id!r}")
        if len(anchor_ids) != 2:
            raise ArtifactValidationError(f"{path}.object_ids must contain exactly two IDs for between")
        return
    if relation_type == "ordered":
        if len(anchor_ids) < 2:
            raise ArtifactValidationError(f"{path}.object_ids must contain at least two IDs for ordered")
        if not str(relation.get("axis") or relation.get("direction") or "").strip():
            raise ArtifactValidationError(f"{path} requires axis or direction for ordered")
        return
    if relation_type == "around":
        if len(subject_ids) < 1:
            raise ArtifactValidationError(
                f"{path}.subject_ids must contain at least one plan ID for around; count expansion may provide multiple instances"
            )
        anchor_id = _non_empty_string(anchor, f"{path}.object_id")
        if anchor_id not in object_ids:
            raise ArtifactValidationError(f"{path}.object_id references unknown object id {anchor_id!r}")
        return

    # Binary frozen predicates and unknown VLM-routed predicates share this
    # minimum identity contract. Unknown group predicates may use explicit ID
    # lists as long as every referenced ID is valid.
    if subject is not None and anchor is not None:
        subject_id = _non_empty_string(subject, f"{path}.subject_id")
        anchor_id = _non_empty_string(anchor, f"{path}.object_id")
        if subject_id not in object_ids:
            raise ArtifactValidationError(f"{path}.subject_id references unknown object id {subject_id!r}")
        if anchor_id not in object_ids:
            raise ArtifactValidationError(f"{path}.object_id references unknown object id {anchor_id!r}")
        return
    if len(subject_ids) + len(anchor_ids) < 2:
        raise ArtifactValidationError(f"{path} must reference at least two object IDs")


def _object_id_list(value: Any, path: str, valid_ids: set[str]) -> list[str]:
    if value is None:
        return []
    values = _require_list(value, path)
    result: list[str] = []
    for index, item in enumerate(values):
        object_id = _non_empty_string(item, f"{path}[{index}]")
        if object_id not in valid_ids:
            raise ArtifactValidationError(f"{path}[{index}] references unknown object id {object_id!r}")
        result.append(object_id)
    return result


def _validate_generated_relation(
    relation: Any,
    object_ids: set[str],
    path: str,
    *,
    architecture_allowed: bool,
) -> None:
    relation = _require_mapping(relation, path)
    _forbid_keys(
        relation,
        {"subject", "predicate", "relation", "object", "anchor_id", "target_id"},
        path,
    )
    relation_type = _require_string(relation, "type", path)
    family = str(relation.get("family") or "").strip().lower()
    if architecture_allowed and family == "oar":
        return
    if relation.get("family") is not None and family != "oor":
        raise ArtifactValidationError(f"{path}.family must be 'oor' when provided")

    subject = relation.get("subject_id")
    anchor = relation.get("object_id")
    subject_ids = _object_id_list(relation.get("subject_ids"), f"{path}.subject_ids", object_ids)
    anchor_ids = _object_id_list(relation.get("object_ids"), f"{path}.object_ids", object_ids)

    if relation_type == "between":
        subject_id = _non_empty_string(subject, f"{path}.subject_id")
        if subject_id not in object_ids:
            raise ArtifactValidationError(f"{path}.subject_id references unknown object id {subject_id!r}")
        if len(anchor_ids) != 2:
            raise ArtifactValidationError(f"{path}.object_ids must contain exactly two IDs for between")
        return
    if relation_type == "ordered":
        if len(anchor_ids) < 2:
            raise ArtifactValidationError(f"{path}.object_ids must contain at least two IDs for ordered")
        if not str(relation.get("axis") or relation.get("direction") or "").strip():
            raise ArtifactValidationError(f"{path} requires axis or direction for ordered")
        return
    if relation_type == "around":
        if len(subject_ids) < 2:
            raise ArtifactValidationError(f"{path}.subject_ids must contain at least two IDs for around")
        anchor_id = _non_empty_string(anchor, f"{path}.object_id")
        if anchor_id not in object_ids:
            raise ArtifactValidationError(f"{path}.object_id references unknown object id {anchor_id!r}")
        return

    if subject is not None and anchor is not None:
        subject_id = _non_empty_string(subject, f"{path}.subject_id")
        anchor_id = _non_empty_string(anchor, f"{path}.object_id")
        if subject_id not in object_ids:
            raise ArtifactValidationError(f"{path}.subject_id references unknown object id {subject_id!r}")
        if anchor_id not in object_ids:
            raise ArtifactValidationError(f"{path}.object_id references unknown object id {anchor_id!r}")
        return
    if len(subject_ids) + len(anchor_ids) < 2:
        raise ArtifactValidationError(f"{path} must reference at least two object IDs")


def _validate_generated_oar_relation(relation: Any, object_ids: set[str], path: str) -> None:
    relation = _require_mapping(relation, path)
    _forbid_keys(
        relation,
        {"subject", "predicate", "relation", "object", "anchor_id", "target_id", "target"},
        path,
    )
    subject_id = _require_string(relation, "subject_id", path)
    if subject_id not in object_ids:
        raise ArtifactValidationError(f"{path}.subject_id references unknown object id {subject_id!r}")
    _require_string(relation, "type", path)
    _require_string(relation, "architectural_element", path)
    family = str(relation.get("family") or "").strip().lower()
    if relation.get("family") is not None and family != "oar":
        raise ArtifactValidationError(f"{path}.family must be 'oar' when provided")


def _require_min_corner_boundary(value: list, path: str) -> None:
    xs = [float(point[0]) for point in value]
    ys = [float(point[1]) for point in value]
    tolerance = 1.0e-6
    if abs(min(xs)) > tolerance or abs(min(ys)) > tolerance:
        raise ArtifactValidationError(
            f"{path} must have minimum x=0 and y=0 for room_min_corner_floor"
        )


def _require_axis_aligned_rectangular_boundary(value: list, path: str) -> None:
    if len(value) != 4:
        raise ArtifactValidationError(f"{path} must contain four corners for rectangular_enclosed_room_v1")
    points = [(float(point[0]), float(point[1])) for point in value]
    xs = sorted({point[0] for point in points})
    ys = sorted({point[1] for point in points})
    if len(xs) != 2 or len(ys) != 2:
        raise ArtifactValidationError(f"{path} must be an axis-aligned rectangle")
    expected = {(x, y) for x in xs for y in ys}
    if set(points) != expected:
        raise ArtifactValidationError(f"{path} must contain each axis-aligned rectangle corner exactly once")


def _first_value(mapping: dict, keys: list[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _non_empty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _forbid_keys(mapping: dict, keys: set[str], path: str) -> None:
    present = sorted(key for key in keys if key in mapping)
    if present:
        raise ArtifactValidationError(f"{path} must not contain legacy/forbidden keys: {present}")


def _validate_requires_asset_selection(contract: dict, expected: bool) -> None:
    if "requires_asset_selection" not in contract:
        return
    value = contract.get("requires_asset_selection")
    if not isinstance(value, bool):
        raise ArtifactValidationError("generation_input.generation_contract.requires_asset_selection must be boolean")
    if value is not expected:
        expected_text = "true" if expected else "false"
        raise ArtifactValidationError(f"generation_input.generation_contract.requires_asset_selection must be {expected_text}")
