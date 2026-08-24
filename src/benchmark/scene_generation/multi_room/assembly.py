"""Deterministic translation-only assembly and room projection artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from benchmark.adapters.catalog_placement.converter import (
    convert_catalog_placement_to_scene,
)
from benchmark.architecture_policy import (
    build_architecture_contract,
    validate_architecture_contract,
)
from benchmark.resources import runtime_resource_path
from benchmark.scene_generation.multi_room.floor_plan import LoadedFloorPlan
from benchmark.scene_io.validate import (
    validate_asset_selection,
    validate_generated_scene,
    validate_generation_input,
    validate_object_plan,
    validate_scene_request,
)
from benchmark.task_contract import require_scene_matches_architecture


MULTI_ROOM_SCENE_SCHEMA_VERSION = "multi_room_scene_v1"
COMPILED_ARCHITECTURE_SCHEMA_VERSION = "compiled_architecture_v1"
ROOM_EVALUATION_INDEX_SCHEMA_VERSION = "room_evaluation_index_v1"
ROOM_EVALUATION_OBJECT_PLAN_SCHEMA_VERSION = "room_evaluation_object_plan_v1"
ASSEMBLY_MANIFEST_SCHEMA_VERSION = "assembly_manifest_v1"
_SCHEMA_PATHS = {
    MULTI_ROOM_SCENE_SCHEMA_VERSION: runtime_resource_path(
        "schemas/multi_room/scene_v1.schema.json"
    ),
    COMPILED_ARCHITECTURE_SCHEMA_VERSION: runtime_resource_path(
        "schemas/multi_room/compiled_architecture_v1.schema.json"
    ),
    ROOM_EVALUATION_INDEX_SCHEMA_VERSION: runtime_resource_path(
        "schemas/multi_room/room_evaluation_index_v1.schema.json"
    ),
    ROOM_EVALUATION_OBJECT_PLAN_SCHEMA_VERSION: runtime_resource_path(
        "schemas/multi_room/room_evaluation_object_plan_v1.schema.json"
    ),
    ASSEMBLY_MANIFEST_SCHEMA_VERSION: runtime_resource_path(
        "schemas/multi_room/assembly_manifest_v1.schema.json"
    ),
}


class AssemblyError(ValueError):
    """Raised when deterministic assembly cannot preserve its invariants."""


@dataclass(frozen=True, slots=True)
class RoomAssemblySource:
    room_key: str
    room_id: str
    generation_index: int
    status: str
    room_result_path: Path
    room_result_sha256: str
    object_plan_artifact_sha256: str | None
    room_brief: Mapping[str, Any]
    object_plan: Mapping[str, Any] | None
    asset_selection: Mapping[str, Any] | None
    placement: Mapping[str, Any] | None

    @property
    def complete(self) -> bool:
        return (
            self.status == "complete"
            and self.object_plan is not None
            and self.asset_selection is not None
            and self.placement is not None
        )


@dataclass(frozen=True, slots=True)
class RoomProjectionBundle:
    """Canonical room projection plus its hash-addressed evaluator inputs."""

    canonical_scene_path: str
    canonical_scene_sha256: str
    canonical_scene: Mapping[str, Any]
    evaluation_inputs: Mapping[str, Mapping[str, Any]]


ROOM_EVALUATION_INPUT_FILENAMES = {
    "scene_request": "scene_request.json",
    "object_plan": "object_plan.json",
    "asset_selection": "asset_selection.json",
    "generation_input": "generation_input.json",
    "architecture_contract": "architecture_contract.json",
}
_EVALUATOR_MULTI_ANCHOR_RELATION_TYPES = frozenset(
    {"between", "ordered", "around"}
)
_ROOM_EVALUATION_INDEX_FIELDS = {
    "canonical_scene": ("projection_path", "projection_hash"),
    "scene_request": ("scene_request_path", "scene_request_hash"),
    "object_plan": ("object_plan_path", "object_plan_hash"),
    "asset_selection": ("asset_selection_path", "asset_selection_hash"),
    "generation_input": ("generation_input_path", "generation_input_hash"),
    "architecture_contract": (
        "architecture_contract_path",
        "architecture_contract_hash",
    ),
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8") + b"\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _validate_schema(value: Mapping[str, Any]) -> None:
    version = value.get("schema_version")
    try:
        path = _SCHEMA_PATHS[str(version)]
    except KeyError as exc:
        raise AssemblyError(f"unsupported assembly schema: {version!r}") from exc
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(
            f"cannot load packaged assembly schema: {type(exc).__name__}"
        ) from exc
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        raise AssemblyError(
            f"{version} schema validation failed at {location}: {error.message}"
        )


def _room_key(index: int) -> str:
    return f"room_{index:03d}"


def room_key_for_generation_index(index: int) -> str:
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise AssemblyError("generation index must be a non-negative integer")
    return _room_key(index)


def _wall_edge(room: Mapping[str, Any], wall_id: str) -> list[list[float]]:
    boundary = room["room"]["boundary"]
    x0, y0 = float(boundary[0][0]), float(boundary[0][1])
    x1, y1 = float(boundary[2][0]), float(boundary[2][1])
    if wall_id == "south_wall":
        edge = [[x0, y0], [x1, y0]]
    elif wall_id == "north_wall":
        edge = [[x0, y1], [x1, y1]]
    elif wall_id == "west_wall":
        edge = [[x0, y0], [x0, y1]]
    elif wall_id == "east_wall":
        edge = [[x1, y0], [x1, y1]]
    else:
        raise AssemblyError(f"unknown wall ID: {wall_id!r}")
    if edge[0] == edge[1]:
        raise AssemblyError("float conversion collapsed an active wall segment")
    return edge


def _interval(segment: Sequence[Sequence[float]]) -> tuple[float, float]:
    if float(segment[0][0]) == float(segment[1][0]):
        return tuple(sorted((float(segment[0][1]), float(segment[1][1]))))
    return tuple(sorted((float(segment[0][0]), float(segment[1][0]))))


def _subtract_intervals(
    whole: tuple[float, float], cuts: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    remaining = [whole]
    for cut_start, cut_end in sorted(cuts):
        next_remaining: list[tuple[float, float]] = []
        for start, end in remaining:
            if cut_end <= start or cut_start >= end:
                next_remaining.append((start, end))
                continue
            if cut_start > start:
                next_remaining.append((start, min(cut_start, end)))
            if cut_end < end:
                next_remaining.append((max(cut_end, start), end))
        remaining = next_remaining
    return [(start, end) for start, end in remaining if end > start]


def _segment_from_interval(
    edge: Sequence[Sequence[float]], interval: tuple[float, float]
) -> list[list[float]]:
    if float(edge[0][0]) == float(edge[1][0]):
        x = float(edge[0][0])
        return [[x, interval[0]], [x, interval[1]]]
    y = float(edge[0][1])
    return [[interval[0], y], [interval[1], y]]


def build_compiled_architecture(plan: LoadedFloorPlan) -> dict[str, Any]:
    """Compile exact physical wall segments without inferring missing walls."""

    room_by_id = {room["room_id"]: room for room in plan.value["rooms"]}
    shared_by_endpoint: dict[tuple[str, str], list[list[list[float]]]] = {}
    shared_walls: list[dict[str, Any]] = []
    for shared in plan.value["shared_walls"]:
        copied = deepcopy(shared)
        copied["deduplicated"] = True
        shared_walls.append(copied)
        for endpoint in shared["rooms"]:
            key = (str(endpoint["room_id"]), str(endpoint["wall_id"]))
            shared_by_endpoint.setdefault(key, []).append(
                deepcopy(shared["segment_global_m"])
            )

    rooms: list[dict[str, Any]] = []
    active_walls: list[dict[str, Any]] = []
    exterior_walls: list[dict[str, Any]] = []
    for generation_index, room_id in enumerate(plan.generation_order):
        room = room_by_id[room_id]
        room_key = _room_key(generation_index)
        offset = [
            float(value)
            for value in room["runner_projection"]["local_to_global_offset_m"]
        ]
        room_exterior_ids: list[str] = []
        for wall_id in room["architecture"]["active_wall_ids"]:
            edge = _wall_edge(room, wall_id)
            active_walls.append(
                {
                    "room_id": room_id,
                    "wall_id": wall_id,
                    "segment_global_m": edge,
                    "height_m": float(room["room"]["height"]),
                    "wall_role": "active",
                }
            )
            cuts = [
                _interval(segment)
                for segment in shared_by_endpoint.get((room_id, wall_id), [])
            ]
            for segment_index, remaining in enumerate(
                _subtract_intervals(_interval(edge), cuts)
            ):
                if wall_id not in room_exterior_ids:
                    room_exterior_ids.append(wall_id)
                exterior_walls.append(
                    {
                        "room_id": room_id,
                        "wall_id": wall_id,
                        "segment_global_m": _segment_from_interval(edge, remaining),
                        "height_m": float(room["room"]["height"]),
                        "wall_role": "exterior",
                    }
                )
        rooms.append(
            {
                "room_id": room_id,
                "generation_index": generation_index,
                "boundary": deepcopy(room["room"]["boundary"]),
                "dimensions_m": deepcopy(room["room_dimensions_m"]),
                "height_m": float(room["room"]["height"]),
                "active_wall_ids": list(room["architecture"]["active_wall_ids"]),
                "exterior_wall_ids": room_exterior_ids,
            }
        )
    artifact = {
        "schema_version": COMPILED_ARCHITECTURE_SCHEMA_VERSION,
        "layout_id": plan.layout_id,
        "floor_plan_hash": plan.canonical_sha256,
        "wall_thickness_m": float(plan.value["wall_thickness_m"]),
        "wall_inventory_contract": {
            "logical_support_topology": "active_walls_per_room_v1",
            "physical_inventory": (
                "exterior_walls_plus_deduplicated_shared_walls_v1"
            ),
        },
        "rooms": rooms,
        "active_walls": active_walls,
        "shared_walls": shared_walls,
        "exterior_walls": exterior_walls,
        "provenance": {
            "generation_mode": "multi_room_with_architecture_v1",
            "ordered_room_ids": list(plan.generation_order),
            "wall_compilation_policy": "declared_active_edges_minus_deduplicated_shared_segments_v1",
            "openings": [],
            "room_frames": [
                {
                    "room_key": _room_key(index),
                    "room_id": room_id,
                    "local_boundary": deepcopy(
                        room_by_id[room_id]["runner_projection"]["local_room"]["boundary"]
                    ),
                    "local_to_global_offset_m": deepcopy(
                        room_by_id[room_id]["runner_projection"]["local_to_global_offset_m"]
                    ),
                }
                for index, room_id in enumerate(plan.generation_order)
            ],
        },
    }
    _validate_schema(artifact)
    return artifact


def _global_objects(
    plan: LoadedFloorPlan,
    sources: Sequence[RoomAssemblySource],
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    global_ids: set[str] = set()
    for source in sources:
        if not source.complete:
            continue
        assert source.placement is not None
        assert source.object_plan is not None
        slots = {str(item["id"]): item for item in source.object_plan["objects"]}
        offset = [
            float(value)
            for value in source.room_brief["local_to_global_offset_m"]
        ]
        for item in source.placement["instances"]:
            local_id = str(item["instance_id"])
            global_id = f"{source.room_id}__{local_id}"
            if global_id in global_ids:
                raise AssemblyError(f"duplicate global instance ID: {global_id}")
            global_ids.add(global_id)
            local_center = [float(value) for value in item["center_m"]]
            global_center = [local_center[index] + offset[index] for index in range(3)]
            if not all(math.isfinite(value) for value in global_center):
                raise AssemblyError("object translation produced a non-finite coordinate")
            if any(
                not math.isclose(
                    global_center[index] - offset[index],
                    local_center[index],
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for index in range(3)
            ):
                raise AssemblyError(
                    "object local/global translation is not numerically reversible"
                )
            slot_id = str(item["slot_id"])
            object_spec = deepcopy(slots[slot_id])
            objects.append(
                {
                    "global_instance_id": global_id,
                    "local_instance_id": local_id,
                    "room_id": source.room_id,
                    "slot_id": slot_id,
                    "asset_id": str(item["asset_id"]),
                    "local_center_m": local_center,
                    "global_center_m": global_center,
                    "uniform_scale": float(item["uniform_scale"]),
                    "rotation": [
                        float(value) for value in item["rotation_euler_xyz_deg"]
                    ],
                    "source_artifact_hash": source.room_result_sha256,
                    "support": object_spec["metadata"]["support"],
                    "facing": object_spec["metadata"]["facing_intent"],
                    "metadata": {
                        "room_key": source.room_key,
                        "object_plan_slot": object_spec,
                        "source_room_result_sha256": source.room_result_sha256,
                        "assembly_transform": {
                            "type": "translation_only",
                            "local_to_global_offset_m": offset,
                        },
                    },
                }
            )
    return objects


def build_global_scene(
    plan: LoadedFloorPlan,
    sources: Sequence[RoomAssemblySource],
    *,
    compiled_architecture_sha256: str,
) -> dict[str, Any]:
    source_by_id = {source.room_id: source for source in sources}
    statuses = {source.room_id: source.status for source in sources}
    complete = all(statuses.get(room_id) == "complete" for room_id in plan.generation_order)
    artifact = {
        "schema_version": MULTI_ROOM_SCENE_SCHEMA_VERSION,
        "layout_id": plan.layout_id,
        "floor_plan_hash": plan.canonical_sha256,
        "assembly_status": "complete" if complete else "incomplete",
        "ordered_room_ids": list(plan.generation_order),
        "compiled_architecture_hash": compiled_architecture_sha256,
        "rooms": [
            {
                "room_id": room_id,
                "generation_index": index,
                "terminal_status": (
                    "succeeded" if statuses.get(room_id) == "complete" else "failed"
                ),
                "room_type": plan.room(room_id)["room_type"],
                "theme": plan.room(room_id)["theme"],
                "boundary": deepcopy(plan.room(room_id)["room"]["boundary"]),
                "height": float(plan.room(room_id)["room"]["height"]),
                "active_wall_ids": list(
                    plan.room(room_id)["architecture"]["active_wall_ids"]
                ),
                "object_ids": [
                    obj["global_instance_id"]
                    for obj in _global_objects(plan, (source_by_id[room_id],))
                ],
                "source_result_hash": source_by_id[room_id].room_result_sha256,
                "local_to_global_offset_m": deepcopy(
                    plan.room(room_id)["runner_projection"][
                        "local_to_global_offset_m"
                    ]
                ),
                "provenance": {"room_key": _room_key(index)},
            }
            for index, room_id in enumerate(plan.generation_order)
        ],
        "objects": _global_objects(plan, sources),
        "provenance": {
            "generation_mode": "multi_room_with_architecture_v1",
            "identity_policy": "room_id_double_underscore_local_instance_id_v1",
            "assembly_transform": "translation_only",
        },
    }
    _validate_schema(artifact)
    return artifact


def _projection_generation_input(source: RoomAssemblySource) -> dict[str, Any]:
    assert source.object_plan is not None
    assert source.asset_selection is not None
    if source.object_plan_artifact_sha256 is None:
        raise AssemblyError("complete room lacks source object-plan artifact hash")
    request_id = f"{source.room_brief['layout_id']}__{source.room_key}"
    room = deepcopy(source.room_brief["local_room"])
    room["floor_z"] = 0.0
    source_object_plan = deepcopy(source.object_plan)
    source_schema_version = source_object_plan.pop("schema_version")
    # The multi-room model contract records semantic relation families such as
    # ``proximity`` and ``support``.  Every row in this array is nevertheless
    # object-to-object by schema (subject_id + object_id); architecture support
    # is represented separately in each object's metadata.  The evaluator
    # companion therefore projects these rows to its canonical ``oor`` family
    # without mutating the source artifact or its hashes.
    projected_relations: list[dict[str, Any]] = []
    for source_relation in source_object_plan.get("relations", []):
        relation = deepcopy(source_relation)
        relation["family"] = "oor"
        relation_type = str(relation.get("type") or "").strip()
        # The source schema is strictly binary (subject_id + object_id), while
        # the evaluator reserves these bare names for multi-anchor structures.
        # Namespace only the colliding binary labels so the generic relation
        # route preserves their meaning without fabricating missing anchors.
        if relation_type.lower() in _EVALUATOR_MULTI_ANCHOR_RELATION_TYPES:
            relation["type"] = f"binary_{relation_type.lower()}"
        projected_relations.append(relation)
    source_object_plan["relations"] = projected_relations
    object_plan = {
        "schema_version": ROOM_EVALUATION_OBJECT_PLAN_SCHEMA_VERSION,
        "request_id": request_id,
        "source_schema_version": source_schema_version,
        "source_object_plan_artifact_sha256": source.object_plan_artifact_sha256,
        "source_object_plan_canonical_sha256": sha256_bytes(
            canonical_json_bytes(source.object_plan)
        ),
        **source_object_plan,
    }
    asset_selection = deepcopy(source.asset_selection)
    asset_selection["request_id"] = request_id
    return {
        "request_id": request_id,
        "scene_request": {
            "request_id": request_id,
            "instruction": source.room_brief["instruction"],
            "scene_type": source.room_brief["room_type"],
            "structure": True,
            "prompt_granularity": "fine_grained",
            "room": room,
        },
        "generation_contract": {
            "output_format": "canonical_generated_scene_v1",
            "requires_pose": True,
            "input_mode": "structured_assets",
            "input_type": "i2_natural_language_structure",
            "evaluator_output_type": "o3_scene_package",
            "requires_asset_selection": True,
        },
        "object_plan": object_plan,
        "asset_selection": asset_selection,
    }


def build_room_evaluation_inputs(
    source: RoomAssemblySource,
    projection: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    """Return matching persisted companions for current room-wise evaluation."""

    generation_input = _projection_generation_input(source)
    request_id = str(generation_input["request_id"])
    scene_request = deepcopy(generation_input["scene_request"])
    object_plan = deepcopy(generation_input["object_plan"])
    asset_selection = deepcopy(generation_input["asset_selection"])
    architecture = deepcopy(projection["metadata"]["architecture_contract"])
    registry = projection["metadata"].get("instance_registry")
    identities = {
        "canonical_scene": projection.get("request_id"),
        "scene_request": scene_request.get("request_id"),
        "object_plan": object_plan.get("request_id"),
        "asset_selection": asset_selection.get("request_id"),
        "generation_input": generation_input.get("request_id"),
        "instance_registry": (
            registry.get("request_id") if isinstance(registry, Mapping) else None
        ),
    }
    if set(identities.values()) != {request_id}:
        raise AssemblyError(f"room evaluation request identity mismatch: {identities}")
    validate_scene_request(scene_request)
    validate_object_plan(object_plan)
    _validate_schema(object_plan)
    validate_asset_selection(asset_selection)
    validate_generation_input(generation_input)
    validate_architecture_contract(architecture)
    return {
        "scene_request": scene_request,
        "object_plan": object_plan,
        "asset_selection": asset_selection,
        "generation_input": generation_input,
        "architecture_contract": architecture,
    }


def build_room_projection(
    source: RoomAssemblySource,
    *,
    compiled_architecture_sha256: str,
) -> dict[str, Any]:
    """Create a current canonical scene with local transforms and global IDs."""

    if not source.complete:
        raise AssemblyError(f"cannot project incomplete room {source.room_id}")
    assert source.placement is not None
    placement = deepcopy(source.placement)
    local_by_global: dict[str, str] = {}
    for item in placement["instances"]:
        local_id = str(item["instance_id"])
        global_id = f"{source.room_id}__{local_id}"
        item["instance_id"] = global_id
        local_by_global[global_id] = local_id
    generation_input = _projection_generation_input(source)
    scene = convert_catalog_placement_to_scene(placement, generation_input)
    offset = [float(value) for value in source.room_brief["local_to_global_offset_m"]]
    local_room = deepcopy(source.room_brief["local_room"])
    local_room["floor_z"] = 0.0
    architecture = build_architecture_contract(
        local_room,
        physical_wall_policy="explicit_only",
        requested_policy="explicit_only",
        policy_source="multi_room_floor_plan_v1",
        active_wall_ids=source.room_brief["architecture"]["active_wall_ids"],
        activation_sources=("multi_room_floor_plan_v1",),
        activation_claims=(
            {
                "source": "multi_room_floor_plan_v1",
                "active_wall_ids": list(
                    source.room_brief["architecture"]["active_wall_ids"]
                ),
            },
        ),
    )
    scene["scene_id"] = f"{source.room_brief['layout_id']}__{source.room_key}"
    scene["request_id"] = f"{source.room_brief['layout_id']}__{source.room_key}"
    scene["scene_type"] = source.room_brief["room_type"]
    scene["metadata"]["architecture_contract"] = architecture
    scene["metadata"]["multi_room_projection"] = {
        "schema_version": "room_local_projection_v1",
        "layout_id": source.room_brief["layout_id"],
        "room_key": source.room_key,
        "room_id": source.room_id,
        "generation_index": source.generation_index,
        "source_room_result_sha256": source.room_result_sha256,
        "compiled_architecture_sha256": compiled_architecture_sha256,
        "local_to_global_offset_m": offset,
        "identity_policy": "room_id_double_underscore_local_instance_id_v1",
    }
    for obj in scene["objects"]:
        global_id = str(obj["id"])
        local_id = local_by_global[global_id]
        local_center = [float(value) for value in obj["center"]]
        obj["metadata"]["multi_room_provenance"] = {
            "room_key": source.room_key,
            "room_id": source.room_id,
            "local_instance_id": local_id,
            "global_instance_id": global_id,
            "local_center_m": local_center,
            "global_center_m": [
                local_center[index] + offset[index] for index in range(3)
            ],
            "local_to_global_offset_m": offset,
            "source_room_result_sha256": source.room_result_sha256,
        }
    validate_generated_scene(scene)
    validate_architecture_contract(architecture)
    require_scene_matches_architecture(scene, local_room)
    return scene


def build_evaluation_index(
    plan: LoadedFloorPlan,
    sources: Sequence[RoomAssemblySource],
    *,
    compiled_architecture_sha256: str,
    projections: Mapping[str, RoomProjectionBundle],
) -> dict[str, Any]:
    source_by_id = {source.room_id: source for source in sources}
    rooms: list[dict[str, Any]] = []
    for index, room_id in enumerate(plan.generation_order):
        source = source_by_id[room_id]
        projection = projections.get(room_id)
        room_entry = {
            "room_id": room_id,
            "order_index": index,
            "terminal_status": (
                "succeeded" if source.status == "complete" else "failed"
            ),
            "global_offset_m": deepcopy(
                source.room_brief["local_to_global_offset_m"]
            ),
            "source_room_result_hash": source.room_result_sha256,
            "provenance": {
                "room_key": source.room_key,
                "source_room_result_path": source.room_result_path.as_posix(),
                "source_terminal_status": source.status,
                "object_count": (
                    0
                    if projection is None
                    else len(projection.canonical_scene["objects"])
                ),
            },
        }
        if projection is not None:
            room_entry["projection_path"] = projection.canonical_scene_path
            room_entry["projection_hash"] = projection.canonical_scene_sha256
            for name, (path_field, hash_field) in _ROOM_EVALUATION_INDEX_FIELDS.items():
                if name == "canonical_scene":
                    continue
                companion = projection.evaluation_inputs[name]
                room_entry[path_field] = companion["path"]
                room_entry[hash_field] = companion["sha256"]
            room_entry["provenance"].update(
                {
                    "canonical_schema_version": "canonical_scene_v1",
                    "validator_status": "passed",
                }
            )
        rooms.append(room_entry)
    artifact = {
        "schema_version": ROOM_EVALUATION_INDEX_SCHEMA_VERSION,
        "layout_id": plan.layout_id,
        "floor_plan_hash": plan.canonical_sha256,
        "expected_room_ids": list(plan.generation_order),
        "all_expected_rooms": list(plan.generation_order),
        "ordered_room_ids": list(plan.generation_order),
        "rooms": rooms,
        "provenance": {
            "generation_mode": "multi_room_with_architecture_v1",
            "compiled_architecture_sha256": compiled_architecture_sha256,
            "evaluation_scope": "independent_room_projection_v1",
            "unsupported_global_scopes": [
                "cross_room_collision",
                "cross_room_functionality",
                "global_architecture_scoring",
                "multi_room_overall_score",
            ],
        },
    }
    return validate_evaluation_index(artifact)


def _validated_relative_artifact_path(value: Any, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise AssemblyError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or value.startswith("/") or ".." in path.parts:
        raise AssemblyError(f"{field} must stay within the layout root")
    if path.as_posix() in {"", "."}:
        raise AssemblyError(f"{field} must identify a file")
    return path


def validate_evaluation_index(
    value: Mapping[str, Any],
    *,
    layout_root: Path | None = None,
) -> dict[str, Any]:
    """Validate exact room coverage and, when available, artifact hashes."""

    artifact = deepcopy(dict(value))
    _validate_schema(artifact)
    ordered = artifact["ordered_room_ids"]
    if artifact["expected_room_ids"] != ordered:
        raise AssemblyError("evaluation index expected_room_ids mismatch")
    if artifact["all_expected_rooms"] != ordered:
        raise AssemblyError("evaluation index all_expected_rooms mismatch")
    rooms = artifact["rooms"]
    if len(rooms) != len(ordered):
        raise AssemblyError("evaluation index room coverage is not exact")
    observed_ids = [room["room_id"] for room in rooms]
    if observed_ids != ordered or len(set(observed_ids)) != len(observed_ids):
        raise AssemblyError("evaluation index room order or identity mismatch")

    seen_paths: set[str] = set()
    resolved_root = None if layout_root is None else layout_root.resolve()
    for expected_index, room in enumerate(rooms):
        if room["order_index"] != expected_index:
            raise AssemblyError("evaluation index order_index mismatch")
        succeeded = room["terminal_status"] == "succeeded"
        for _, (path_field, hash_field) in _ROOM_EVALUATION_INDEX_FIELDS.items():
            has_path = path_field in room
            has_hash = hash_field in room
            if succeeded and (not has_path or not has_hash):
                raise AssemblyError(
                    f"succeeded room lacks evaluator artifact: {path_field}"
                )
            if not succeeded and (has_path or has_hash):
                raise AssemblyError(
                    f"non-succeeded room exposes evaluator artifact: {path_field}"
                )
            if not has_path:
                continue
            relative = _validated_relative_artifact_path(
                room[path_field], field=path_field
            )
            relative_text = relative.as_posix()
            if relative_text in seen_paths:
                raise AssemblyError("evaluation index artifact paths must be unique")
            seen_paths.add(relative_text)
            if resolved_root is None:
                continue
            target = resolved_root.joinpath(*relative.parts)
            resolved_target = target.resolve()
            try:
                resolved_target.relative_to(resolved_root)
            except ValueError as exc:
                raise AssemblyError(
                    f"evaluation artifact escapes layout root: {path_field}"
                ) from exc
            if not target.is_file() or target.is_symlink():
                raise AssemblyError(
                    f"evaluation artifact is missing or not regular: {relative_text}"
                )
            if sha256_file(target) != room[hash_field]:
                raise AssemblyError(
                    f"evaluation artifact hash mismatch: {relative_text}"
                )
    return artifact


def build_assembly_manifest(
    plan: LoadedFloorPlan,
    sources: Sequence[RoomAssemblySource],
    *,
    compiled_architecture_path: str,
    compiled_architecture_sha256: str,
    global_scene_path: str,
    global_scene_sha256: str,
    global_scene: Mapping[str, Any],
    evaluation_index_path: str,
    evaluation_index_sha256: str,
    projections: Mapping[str, RoomProjectionBundle],
    runtime_source_manifest_sha256: str,
) -> dict[str, Any]:
    complete_rooms = [source for source in sources if source.complete]
    if [source.room_id for source in sources] != list(plan.generation_order):
        raise AssemblyError("assembly sources do not exactly follow room order")
    expected_global = build_global_scene(
        plan,
        sources,
        compiled_architecture_sha256=compiled_architecture_sha256,
    )
    if dict(global_scene) != expected_global:
        raise AssemblyError("global scene differs from deterministic assembly")
    if sha256_bytes(canonical_json_bytes(global_scene)) != global_scene_sha256:
        raise AssemblyError("global scene hash attestation mismatch")
    for field, value in (
        ("compiled_architecture_path", compiled_architecture_path),
        ("global_scene_path", global_scene_path),
        ("evaluation_index_path", evaluation_index_path),
    ):
        _validated_relative_artifact_path(value, field=field)
    source_by_id = {source.room_id: source for source in sources}
    if set(projections) != {source.room_id for source in complete_rooms}:
        raise AssemblyError("projection coverage differs from complete rooms")
    for room_id, bundle in projections.items():
        source = source_by_id[room_id]
        _validated_relative_artifact_path(
            bundle.canonical_scene_path, field="projection_path"
        )
        expected_projection = build_room_projection(
            source,
            compiled_architecture_sha256=compiled_architecture_sha256,
        )
        if dict(bundle.canonical_scene) != expected_projection:
            raise AssemblyError("room projection differs from deterministic projection")
        if (
            sha256_bytes(canonical_json_bytes(bundle.canonical_scene))
            != bundle.canonical_scene_sha256
        ):
            raise AssemblyError("room projection hash attestation mismatch")
        expected_inputs = build_room_evaluation_inputs(source, expected_projection)
        if set(bundle.evaluation_inputs) != set(ROOM_EVALUATION_INPUT_FILENAMES):
            raise AssemblyError("room evaluator input coverage is not exact")
        for name, expected_value in expected_inputs.items():
            companion = bundle.evaluation_inputs[name]
            if set(companion) != {"path", "sha256", "value"}:
                raise AssemblyError("room evaluator artifact record is malformed")
            _validated_relative_artifact_path(
                companion["path"], field=f"{name}_path"
            )
            if dict(companion["value"]) != dict(expected_value):
                raise AssemblyError(
                    f"room evaluator input differs from deterministic {name}"
                )
            if (
                sha256_bytes(canonical_json_bytes(companion["value"]))
                != companion["sha256"]
            ):
                raise AssemblyError(f"room evaluator {name} hash mismatch")
    projected_ids = {
        str(obj["id"])
        for projection in projections.values()
        for obj in projection.canonical_scene["objects"]
    }
    expected_ids = {
        f"{source.room_id}__{item['instance_id']}"
        for source in complete_rooms
        for item in (source.placement or {}).get("instances", [])
    }
    if projected_ids != expected_ids:
        raise AssemblyError("projection object identity partition is not exact")
    all_complete = len(complete_rooms) == plan.room_count
    artifact = {
        "schema_version": ASSEMBLY_MANIFEST_SCHEMA_VERSION,
        "layout_id": plan.layout_id,
        "completion_status": "complete" if all_complete else "incomplete",
        "source_hashes": {
            "floor_plan": plan.canonical_sha256,
            "room_results": [
                {"room_id": source.room_id, "sha256": source.room_result_sha256}
                for source in sources
            ],
        },
        "artifact_hashes": {
            "compiled_architecture": compiled_architecture_sha256,
            "scene": global_scene_sha256,
            "room_evaluation_index": evaluation_index_sha256,
            "room_projections": [
                {
                    "room_id": room_id,
                    "sha256": projections[room_id].canonical_scene_sha256,
                }
                for room_id in plan.generation_order if room_id in projections
            ],
        },
        "invariants": {
            "room_order_preserved": True,
            "translation_only": True,
            "object_partition_exact": projected_ids == expected_ids,
            "global_instance_ids_unique": len(expected_ids) == len(projected_ids),
            "projection_hashes_verified": True,
            "evaluation_input_hashes_verified": True,
            "current_canonical_validator_passed": True,
            "evaluator_feedback_used": False,
        },
        "provenance": {
            "generation_mode": "multi_room_with_architecture_v1",
            "runtime_source_manifest_sha256": runtime_source_manifest_sha256,
            "floor_plan_source_sha256": plan.source_sha256,
            "paths": {
                "compiled_architecture": compiled_architecture_path,
                "global_scene": global_scene_path,
                "room_evaluation_index": evaluation_index_path,
                "room_projections": [
                    {
                        "room_id": room_id,
                        "path": projections[room_id].canonical_scene_path,
                    }
                    for room_id in plan.generation_order if room_id in projections
                ],
            },
            "room_sources": [
                {
                    "room_key": source.room_key,
                    "room_id": source.room_id,
                    "generation_index": source.generation_index,
                    "terminal_status": source.status,
                    "room_result_path": source.room_result_path.as_posix(),
                }
                for source in sources
            ],
            "deferred_evaluation_policy": [
                "multi_room_score_aggregation",
                "global_architecture_metrics",
                "cross_room_functionality",
            ],
        },
    }
    _validate_schema(artifact)
    return artifact
