from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from benchmark.evaluation_campaign.dataset_identity import (
    inspect_evaluation_dataset,
    prepare_portable_dataset_view,
)
from benchmark.evaluation_campaign.config import load_campaign
from benchmark.camera_cal_scene_level.discovery import discover_cases
from benchmark.evaluator.generic_validity.mesh_geometry import (
    load_collision_geometry_manifest,
)
from benchmark.architecture_policy import build_architecture_contract
from benchmark.multi_room_evaluation import (
    OFFICIAL_RENDER_PROFILE,
    build_existing_evaluation_campaign_config,
    discover_multi_room_evaluation_inventory,
    evaluation_campaign_command,
    materialize_multi_room_evaluation_dataset,
)
from benchmark.scene_generation.multi_room.floor_plan import load_floor_plan
from benchmark.scene_generation.multi_room.assembly import canonical_json_bytes
from scripts import prepare_multi_room_evaluation_dataset as materializer_cli


ROOT = Path(__file__).resolve().parents[1]
MODEL = "model-safe.v1"
LOCAL_ROOM = {
    "boundary": [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0]],
    "height": 3.0,
    "floor_z": 0.0,
    "unit": "meter",
}
ARCHITECTURE = build_architecture_contract(
    LOCAL_ROOM,
    physical_wall_policy="explicit_only",
    requested_policy="explicit_only",
    policy_source="test_multi_room_floor_plan_v1",
    active_wall_ids=("north_wall", "south_wall", "east_wall", "west_wall"),
    activation_sources=("test_multi_room_floor_plan_v1",),
    activation_claims=(
        {
            "source": "test_multi_room_floor_plan_v1",
            "active_wall_ids": [
                "north_wall",
                "south_wall",
                "east_wall",
                "west_wall",
            ],
        },
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _refresh_collection_selection_hash(collection: Path) -> None:
    manifest_path = collection / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    selection_path = collection / MODEL / "selection_manifest.json"
    manifest["model_manifest_sha256"][MODEL] = _sha256(selection_path)
    _write_json(manifest_path, manifest)


def _room(index: int, count: int) -> dict[str, Any]:
    x0 = 3.0 * index
    x1 = x0 + 3.0
    return {
        "room_id": f"room_{index + 1:02d}",
        "generation_index": index,
        "room_type": f"type_{index}",
        "theme": f"theme_{index}",
        "object_count_tier": "compact_7_10",
        "instruction": f"Generate room {index}.",
        "target_instances": {"min": 7, "max": 10},
        "room_dimensions_m": [3.0, 4.0, 3.0],
        "room": {
            "boundary": [[x0, 0.0], [x1, 0.0], [x1, 4.0], [x0, 4.0]],
            "height": 3.0,
            "unit": "meter",
            "topology": "axis_aligned_rectangle",
        },
        "runner_projection": {
            "local_to_global_offset_m": [x0, 0.0, 0.0],
            "local_room": {
                "boundary": [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0]],
                "height": 3.0,
                "unit": "meter",
                "topology": "axis_aligned_rectangle",
            },
        },
        "architecture": {
            "schema_version": "rectangular_room_walls_v1",
            "physical_wall_policy": "explicit_only",
            "wall_thickness_m": 0.08,
            "active_wall_ids": [
                "north_wall",
                "south_wall",
                "east_wall",
                "west_wall",
            ],
        },
        "wall_attachment_requirement": {
            "minimum_count": 1,
            "maximum_count": 1,
            "selection_policy": "model_selects_functionally_appropriate_objects",
        },
    }


def _floor_plan(layout_id: str, count: int) -> dict[str, Any]:
    rooms = [_room(index, count) for index in range(count)]
    room_ids = [room["room_id"] for room in rooms]
    return {
        "schema_version": "multi_room_floor_plan_v1",
        "generation_mode": "multi_room_with_architecture_v1",
        "room_prompt_version": "room_generation_prompt_v1",
        "layout_id": layout_id,
        "room_count": count,
        "aggregate_shape": "rectangular",
        "coordinate_frame": {
            "origin": "shared_global_floor_plan_min_corner",
            "axes": "x_width_y_depth_z_up",
            "unit": "meter",
            "rotation_unit": "degree",
        },
        "global_envelope": {
            "boundary": [
                [0.0, 0.0],
                [3.0 * count, 0.0],
                [3.0 * count, 4.0],
                [0.0, 4.0],
            ],
            "dimensions_m": [3.0 * count, 4.0, 3.0],
        },
        "wall_thickness_m": 0.08,
        "generation_order": room_ids,
        "shared_walls": [
            {
                "shared_wall_id": f"shared_wall_{index + 1}",
                "segment_global_m": [
                    [3.0 * (index + 1), 0.0],
                    [3.0 * (index + 1), 4.0],
                ],
                "rooms": [
                    {"room_id": room_ids[index], "wall_id": "east_wall"},
                    {"room_id": room_ids[index + 1], "wall_id": "west_wall"},
                ],
                "opaque": True,
                "full_height": True,
            }
            for index in range(count - 1)
        ],
        "rooms": rooms,
    }


def _canonical_scene(
    *,
    layout_id: str,
    room_id: str,
    room_key: str,
    order_index: int,
    source_room_result_hash: str,
    compiled_architecture_sha256: str,
) -> dict[str, Any]:
    request_id = f"{layout_id}__{room_key}"
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": request_id,
        "request_id": request_id,
        "scene_type": "test_room",
        "boundary": [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": f"{room_id}__object_1",
                "category": "test_object",
                "description": "A test object.",
                "jid": "asset_1",
                "asset_ref": {
                    "source_db": "imaginarium",
                    "asset_key": "asset_1",
                },
                "asset_proxy": {
                    "type": "canonical_catalog_bbox",
                    "bbox_size": [1.0, 1.0, 1.0],
                    "bbox_center_local": [0.0, 0.0, 0.0],
                },
                "geometry_provenance": "asset_mesh",
                "center": [1.0, 1.0, 0.5],
                "size": [1.0, 1.0, 1.0],
                "rotation": [0.0, 0.0, 0.0],
                "metadata": {},
            }
        ],
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            },
            "architecture_contract": ARCHITECTURE,
            "instance_registry": {"request_id": request_id},
            "multi_room_projection": {
                "schema_version": "room_local_projection_v1",
                "layout_id": layout_id,
                "room_key": room_key,
                "room_id": room_id,
                "generation_index": order_index,
                "source_room_result_sha256": source_room_result_hash,
                "compiled_architecture_sha256": compiled_architecture_sha256,
                "local_to_global_offset_m": [3.0 * order_index, 0.0, 0.0],
                "identity_policy": "room_id_double_underscore_local_instance_id_v1",
            },
        },
    }


def _evaluation_values(
    *,
    layout_id: str,
    room_id: str,
    room_key: str,
    order_index: int,
    source_room_result_hash: str,
    compiled_architecture_sha256: str,
    source_object_plan_path: Path,
) -> dict[str, dict[str, Any]]:
    request_id = f"{layout_id}__{room_key}"
    scene_request = {
        "request_id": request_id,
        "instruction": "Generate a test room.",
        "scene_type": "test_room",
        "structure": True,
        "prompt_granularity": "fine_grained",
        "room": deepcopy(LOCAL_ROOM),
    }
    object_spec = {
        "id": "object_1",
        "category": "test_object",
        "role": "test role",
        "description": "A test object.",
        "count": 1,
        "estimated_size": [1.0, 1.0, 1.0],
        "metadata": {
            "intended_role": "test role",
            "zone": "main_zone",
            "support": "floor",
            "directed": False,
            "functional_side": None,
            "facing_intent": None,
            "retrieval_query": "test object",
            "requested_count": 1,
        },
        "placement_intent": {
            "absolute_relations": [],
            "relative_relations": [],
        },
    }
    source_plan = json.loads(source_object_plan_path.read_text())
    object_plan = {
        "schema_version": "room_evaluation_object_plan_v1",
        "request_id": request_id,
        "source_schema_version": "multi_room_object_plan_v1",
        "source_object_plan_artifact_sha256": _sha256(source_object_plan_path),
        "source_object_plan_canonical_sha256": hashlib.sha256(
            canonical_json_bytes(source_plan)
        ).hexdigest(),
        "scene_type": "test_room",
        "scene_description": "A test room.",
        "prompt_granularity": "fine_grained",
        "global_constraints": [],
        "zones": [
            {
                "id": "main_zone",
                "description": "Main zone.",
                "extent_hint": "whole room",
            }
        ],
        "relations": [],
        "objects": [object_spec],
    }
    asset_selection = {
        "request_id": request_id,
        "objects": [
            {
                "object_id": "object_1",
                "object_spec": deepcopy(object_spec),
                "selected_asset": {
                    "jid": "asset_1",
                    "size": [1.0, 1.0, 1.0],
                    "asset_ref": {
                        "source_db": "imaginarium",
                        "asset_key": "asset_1",
                    },
                    "asset_proxy": {
                        "type": "canonical_catalog_bbox",
                        "bbox_size": [1.0, 1.0, 1.0],
                    },
                    "metadata": {},
                },
                "candidates": [],
            }
        ],
    }
    generation_input = {
        "request_id": request_id,
        "scene_request": deepcopy(scene_request),
        "generation_contract": {
            "output_format": "canonical_generated_scene_v1",
            "requires_pose": True,
            "input_mode": "structured_assets",
            "input_type": "i2_natural_language_structure",
            "evaluator_output_type": "o3_scene_package",
            "requires_asset_selection": True,
        },
        "object_plan": deepcopy(object_plan),
        "asset_selection": deepcopy(asset_selection),
    }
    return {
        "canonical_scene": _canonical_scene(
            layout_id=layout_id,
            room_id=room_id,
            room_key=room_key,
            order_index=order_index,
            source_room_result_hash=source_room_result_hash,
            compiled_architecture_sha256=compiled_architecture_sha256,
        ),
        "scene_request": scene_request,
        "object_plan": object_plan,
        "asset_selection": asset_selection,
        "generation_input": generation_input,
        "architecture_contract": ARCHITECTURE,
    }


def _build_collection(
    tmp_path: Path,
    layouts: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("layout_01", ("succeeded",)),
    ),
) -> Path:
    collection = tmp_path / "collection"
    model_root = collection / MODEL
    selected = model_root / "selected"
    selected.mkdir(parents=True)
    source_base = tmp_path / "source" / MODEL
    selected_rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    totals = {"succeeded": 0, "failed": 0, "missing": 0}
    complete_layouts = 0
    incomplete_layouts = 0
    missing_layouts = 0
    for layout_id, room_statuses in layouts:
        count = len(room_statuses)
        if set(room_statuses) == {"missing"}:
            missing_layouts += 1
            totals["missing"] += count
            selected_rows.append(
                {
                    "layout_id": layout_id,
                    "selection_status": "missing",
                    "expected_rooms": count,
                }
            )
            for index in range(count):
                unresolved.append(
                    {
                        "layout_id": layout_id,
                        "room_key": f"room_{index:03d}",
                        "status": "missing_layout",
                    }
                )
            continue
        run_root = source_base / layout_id
        layout_root = run_root / layout_id
        layout_root.mkdir(parents=True)
        plan_path = _write_json(layout_root / "floor_plan.json", _floor_plan(layout_id, count))
        plan = load_floor_plan(plan_path)
        compiled = _write_json(layout_root / "compiled_architecture.json", {"layout_id": layout_id})
        room_rows: list[dict[str, Any]] = []
        for index, status in enumerate(room_statuses):
            room_id = f"room_{index + 1:02d}"
            room_key = f"room_{index:03d}"
            source_object_plan = None
            room_result_value: dict[str, Any] = {
                "schema_version": "multi_room_room_result_v1",
                "layout_id": layout_id,
                "room_id": room_id,
                "room_key": room_key,
                "generation_index": index,
                "status": "complete" if status == "succeeded" else "stage_a_failed",
                "eligible_for_room_projection": status == "succeeded",
                "artifact_hashes": {},
            }
            if status == "succeeded":
                source_object_plan = _write_json(
                    layout_root / f"rooms/{room_key}/object_plan.json",
                    {
                        "schema_version": "multi_room_object_plan_v1",
                        "scene_type": "test_room",
                        "objects": ["object_1"],
                    },
                )
                room_result_value["artifact_hashes"] = {
                    "object_plan.json": _sha256(source_object_plan)
                }
            room_result = _write_json(
                layout_root / f"rooms/{room_key}/room_result.json",
                room_result_value,
            )
            row: dict[str, Any] = {
                "room_id": room_id,
                "order_index": index,
                "terminal_status": status,
                "global_offset_m": [3.0 * index, 0.0, 0.0],
                "source_room_result_hash": _sha256(room_result),
                "provenance": {
                    "room_key": room_key,
                    "source_room_result_path": f"rooms/{room_key}/room_result.json",
                    "source_terminal_status": "complete" if status == "succeeded" else "stage_a_failed",
                },
            }
            if status == "succeeded":
                assert source_object_plan is not None
                values = _evaluation_values(
                    layout_id=layout_id,
                    room_id=room_id,
                    room_key=room_key,
                    order_index=index,
                    source_room_result_hash=_sha256(room_result),
                    compiled_architecture_sha256=_sha256(compiled),
                    source_object_plan_path=source_object_plan,
                )
                fields = {
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
                for name, value in values.items():
                    path = _write_json(
                        layout_root / f"evaluation_rooms/{room_key}/{name}.json",
                        value,
                    )
                    path_field, hash_field = fields[name]
                    row[path_field] = f"evaluation_rooms/{room_key}/{name}.json"
                    row[hash_field] = _sha256(path)
                totals["succeeded"] += 1
            else:
                totals["failed"] += 1
                unresolved.append(
                    {
                        "layout_id": layout_id,
                        "room_key": room_key,
                        "status": "stage_a_failed",
                    }
                )
            room_rows.append(row)
        index = {
            "schema_version": "room_evaluation_index_v1",
            "layout_id": layout_id,
            "floor_plan_hash": plan.canonical_sha256,
            "expected_room_ids": [row["room_id"] for row in room_rows],
            "all_expected_rooms": [row["room_id"] for row in room_rows],
            "ordered_room_ids": [row["room_id"] for row in room_rows],
            "rooms": room_rows,
            "provenance": {
                "generation_mode": "multi_room_with_architecture_v1",
                "compiled_architecture_sha256": _sha256(compiled),
                "evaluation_scope": "independent_room_projection_v1",
                "unsupported_global_scopes": [
                    "cross_room_collision",
                    "cross_room_functionality",
                    "global_architecture_scoring",
                    "multi_room_overall_score",
                ],
            },
        }
        index_path = _write_json(
            layout_root / "room_evaluation_index.json", index
        )
        global_scene = _write_json(
            layout_root / "assembled_multi_room_scene.json",
            {"schema_version": "multi_room_scene_v1", "layout_id": layout_id},
        )
        successes = room_statuses.count("succeeded")
        failures = count - successes
        assembly_status = "complete" if failures == 0 else "incomplete"
        _write_json(
            layout_root / "assembly_manifest.json",
            {
                "schema_version": "assembly_manifest_v1",
                "layout_id": layout_id,
                "completion_status": assembly_status,
                "source_hashes": {
                    "floor_plan": plan.canonical_sha256,
                    "room_results": [
                        {
                            "room_id": row["room_id"],
                            "sha256": row["source_room_result_hash"],
                        }
                        for row in room_rows
                    ],
                },
                "artifact_hashes": {
                    "compiled_architecture": _sha256(compiled),
                    "scene": _sha256(global_scene),
                    "room_evaluation_index": _sha256(index_path),
                    "room_projections": [
                        {
                            "room_id": row["room_id"],
                            "sha256": row["projection_hash"],
                        }
                        for row in room_rows
                        if row["terminal_status"] == "succeeded"
                    ],
                },
                "invariants": {
                    "room_order_preserved": True,
                    "translation_only": True,
                    "object_partition_exact": True,
                    "global_instance_ids_unique": True,
                    "projection_hashes_verified": True,
                    "evaluation_input_hashes_verified": True,
                    "current_canonical_validator_passed": True,
                    "evaluator_feedback_used": False,
                },
                "provenance": {
                    "generation_mode": "multi_room_with_architecture_v1",
                    "runtime_source_manifest_sha256": "0" * 64,
                    "floor_plan_source_sha256": plan.source_sha256,
                    "paths": {
                        "compiled_architecture": "compiled_architecture.json",
                        "global_scene": "assembled_multi_room_scene.json",
                        "room_evaluation_index": "room_evaluation_index.json",
                        "room_projections": [
                            {
                                "room_id": row["room_id"],
                                "path": row["projection_path"],
                            }
                            for row in room_rows
                            if row["terminal_status"] == "succeeded"
                        ],
                    },
                    "room_sources": [
                        {
                            "room_key": row["provenance"]["room_key"],
                            "room_id": row["room_id"],
                            "generation_index": row["order_index"],
                            "terminal_status": row["provenance"][
                                "source_terminal_status"
                            ],
                            "room_result_path": row["provenance"][
                                "source_room_result_path"
                            ],
                        }
                        for row in room_rows
                    ],
                },
            },
        )
        summary = _write_json(
            run_root / "summary.json", {"assembly_status": assembly_status}
        )
        link = selected / layout_id
        link.symlink_to(os.path.relpath(run_root, start=link.parent))
        selected_rows.append(
            {
                "layout_id": layout_id,
                "selection_status": "complete" if failures == 0 else "incomplete",
                "source_kind": "fixture",
                "source_path": run_root.relative_to(tmp_path).as_posix(),
                "summary_sha256": _sha256(summary),
                "room_evaluation_index_sha256": _sha256(
                    layout_root / "room_evaluation_index.json"
                ),
                "assembly_manifest_sha256": _sha256(
                    layout_root / "assembly_manifest.json"
                ),
                "expected_rooms": count,
                "successful_rooms": successes,
                "failed_rooms": failures,
                "missing_rooms": 0,
            }
        )
        complete_layouts += int(failures == 0)
        incomplete_layouts += int(failures != 0)
    expected_rooms = sum(len(statuses) for _, statuses in layouts)
    selection_manifest = {
        "schema_version": "multi_room_model_selection_manifest_v1",
        "model": MODEL,
        "expected_layouts": len(layouts),
        "terminal_layouts": len(layouts) - missing_layouts,
        "complete_layouts": complete_layouts,
        "incomplete_layouts": incomplete_layouts,
        "missing_layouts": missing_layouts,
        "expected_rooms": expected_rooms,
        "successful_rooms": totals["succeeded"],
        "failed_rooms": totals["failed"],
        "missing_rooms": totals["missing"],
        "selected_layouts": selected_rows,
        "unresolved_rooms": unresolved,
        "selection_policy": "fixture",
    }
    _write_json(model_root / "selection_manifest.json", selection_manifest)
    selection_hash = _sha256(model_root / "selection_manifest.json")
    _write_json(
        collection / "collection_manifest.json",
        {
            "schema_version": "multi_room_unified_collection_v1",
            "collection_id": "fixture-collection",
            "model_count": 1,
            "models": [MODEL],
            "expected_rooms": expected_rooms,
            "successful_rooms": totals["succeeded"],
            "failed_rooms": totals["failed"],
            "missing_rooms": totals["missing"],
            "model_manifests": {
                MODEL: f"{MODEL}/selection_manifest.json"
            },
            "model_manifest_sha256": {MODEL: selection_hash},
        },
    )
    return collection


class FakeRenderer:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls = 0
        self.fail_on_call = fail_on_call
        for field, value in OFFICIAL_RENDER_PROFILE.items():
            setattr(self, field, value)

    def render_scene(
        self,
        *,
        scene_path: str | Path,
        out_dir: str | Path,
        asset_root: str | Path | None = None,
    ) -> dict[str, Any]:
        del asset_root
        self.calls += 1
        if self.fail_on_call == self.calls:
            raise RuntimeError("injected render interruption")
        scene_path = Path(scene_path).resolve()
        scene = json.loads(scene_path.read_text())
        out = Path(out_dir).resolve()
        out.mkdir(parents=True)
        (out / "scene.blend").write_bytes(b"blend")
        for name in (
            "standardized_perspective.png",
            "standardized_top.png",
            "standardized_identity_map.png",
        ):
            (out / name).write_bytes(f"png:{name}".encode())
        _write_json(out / "architecture_contract.json", ARCHITECTURE)
        geometry = out / "collision_geometry/object_1.ply"
        geometry.parent.mkdir()
        geometry.write_text(
            "ply\nformat ascii 1.0\n"
            "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
            "element face 1\nproperty list uchar int vertex_indices\n"
            "end_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n",
            encoding="utf-8",
        )
        collision = {
            "schema_version": "collision_geometry_v1",
            "units": "meter",
            "up_axis": "z",
            "objects": {
                "object_1": {
                    "representation": "triangle_mesh",
                    "complete": True,
                    "transform_baked": True,
                    "geometry_path": str(geometry),
                    "source_uri": "/private/source/object.fbx",
                }
            },
        }
        _write_json(out / "collision_geometry_manifest.json", collision)
        views = [
            {"name": "perspective", "path": str(out / "standardized_perspective.png")},
            {"name": "top", "path": str(out / "standardized_top.png")},
            {"name": "identity_map", "path": str(out / "standardized_identity_map.png")},
        ]
        manifest = {
            "backend": "fake",
            "blender_version": "fake-1",
            "render_engine": OFFICIAL_RENDER_PROFILE["render_engine"],
            "render_config": {
                "width": OFFICIAL_RENDER_PROFILE["width"],
                "height": OFFICIAL_RENDER_PROFILE["height"],
                "render_engine_requested": OFFICIAL_RENDER_PROFILE[
                    "render_engine"
                ],
                "cycles_device_requested": OFFICIAL_RENDER_PROFILE[
                    "cycles_device"
                ],
            },
            "architecture_policy_version": "fixture",
            "architecture": ARCHITECTURE,
            "identity_legend": {"#FFFFFF": "object_1"},
            "objects": [
                {
                    "id": item["id"],
                    "root_object": f"asset_{item['id']}",
                    "representation": "asset_mesh",
                }
                for item in scene["objects"]
            ],
            "views": views,
            "blend_file": str(out / "scene.blend"),
            "scene_json": str(scene_path),
            "collision_geometry_manifest": str(out / "collision_geometry_manifest.json"),
            "collision_geometry": {
                **collision,
                "manifest_path": str(out / "collision_geometry_manifest.json"),
            },
            "collision_geometry_export": {
                "status": "completed",
                "limits": {
                    "max_vertices_per_object": OFFICIAL_RENDER_PROFILE[
                        "collision_max_vertices_per_object"
                    ],
                    "max_faces_per_object": OFFICIAL_RENDER_PROFILE[
                        "collision_max_faces_per_object"
                    ],
                    "max_total_vertices": OFFICIAL_RENDER_PROFILE[
                        "collision_max_total_vertices"
                    ],
                    "max_total_faces": OFFICIAL_RENDER_PROFILE[
                        "collision_max_total_faces"
                    ],
                },
            },
            "asset_coverage": {
                "object_count": len(scene["objects"]),
                "asset_mesh_count": len(scene["objects"]),
                "bbox_proxy_count": 0,
                "required": True,
            },
        }
        _write_json(out / "render_manifest.json", manifest)
        return manifest


def _asset_root(tmp_path: Path) -> Path:
    path = tmp_path / "assets"
    path.mkdir()
    (path / "imaginarium_asset_info.csv").write_text(
        "jid,category\nasset_1,test\n", encoding="utf-8"
    )
    asset = path / "asset_1"
    asset.mkdir()
    (asset / "asset_1.fbx").write_bytes(b"fixture-asset")
    return path


def test_reader_discovers_exact_deterministic_ledgers(tmp_path: Path) -> None:
    root = _build_collection(
        tmp_path,
        layouts=(
            ("layout_02", ("succeeded", "failed")),
            ("layout_01", ("succeeded",)),
            ("layout_03", ("missing", "missing")),
        ),
    )
    inventory = discover_multi_room_evaluation_inventory(root)

    assert [room.case_id for room in inventory.succeeded] == [
        f"mr.{MODEL}.layout_01.room_000",
        f"mr.{MODEL}.layout_02.room_000",
    ]
    assert [(room.layout_id, room.room_key) for room in inventory.failed] == [
        ("layout_02", "room_001")
    ]
    assert [(room.layout_id, room.room_key, room.room_id) for room in inventory.missing] == [
        ("layout_03", "room_000", None),
        ("layout_03", "room_001", None),
    ]
    assert inventory.expected_room_count == 5
    assert inventory.complete is False


def test_reader_rejects_source_hash_tampering(tmp_path: Path) -> None:
    root = _build_collection(tmp_path)
    canonical = next(root.glob(f"{MODEL}/selected/layout_01/layout_01/evaluation_rooms/*/canonical_scene.json"))
    canonical.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash|artifact"):
        discover_multi_room_evaluation_inventory(root)


def test_reader_rejects_file_and_index_coordinated_rewrite(tmp_path: Path) -> None:
    root = _build_collection(tmp_path)
    layout = root / MODEL / "selected/layout_01/layout_01"
    request_path = layout / "evaluation_rooms/room_000/scene_request.json"
    request = json.loads(request_path.read_text())
    request["instruction"] = "valid but unauthorized modified instruction"
    _write_json(request_path, request)
    index_path = layout / "room_evaluation_index.json"
    index = json.loads(index_path.read_text())
    index["rooms"][0]["scene_request_hash"] = _sha256(request_path)
    _write_json(index_path, index)

    with pytest.raises(ValueError, match="selection manifest.*index"):
        discover_multi_room_evaluation_inventory(root)


def test_reader_rejects_index_rewrite_even_if_selection_hash_is_updated(
    tmp_path: Path,
) -> None:
    root = _build_collection(tmp_path)
    layout = root / MODEL / "selected/layout_01/layout_01"
    request_path = layout / "evaluation_rooms/room_000/scene_request.json"
    request = json.loads(request_path.read_text())
    request["instruction"] = "valid but unauthorized modified instruction"
    _write_json(request_path, request)
    index_path = layout / "room_evaluation_index.json"
    index = json.loads(index_path.read_text())
    index["rooms"][0]["scene_request_hash"] = _sha256(request_path)
    _write_json(index_path, index)
    selection_path = root / MODEL / "selection_manifest.json"
    selection = json.loads(selection_path.read_text())
    selection["selected_layouts"][0]["room_evaluation_index_sha256"] = _sha256(
        index_path
    )
    _write_json(selection_path, selection)
    _refresh_collection_selection_hash(root)

    with pytest.raises(ValueError, match="assembly manifest index hash"):
        discover_multi_room_evaluation_inventory(root)


def test_reader_rejects_index_traversal_even_with_matching_hash(tmp_path: Path) -> None:
    root = _build_collection(tmp_path)
    index_path = next(
        root.glob(f"{MODEL}/selected/layout_01/layout_01/room_evaluation_index.json")
    )
    index = json.loads(index_path.read_text())
    source = index_path.parent / index["rooms"][0]["projection_path"]
    escaped = index_path.parent / "escaped.json"
    escaped.write_bytes(source.read_bytes())
    index["rooms"][0]["projection_path"] = "../layout_01/escaped.json"
    index["rooms"][0]["projection_hash"] = _sha256(escaped)
    _write_json(index_path, index)

    with pytest.raises(ValueError, match="layout root|traversal|valid"):
        discover_multi_room_evaluation_inventory(root)


def test_reader_rejects_unexpected_artifact_symlink(tmp_path: Path) -> None:
    root = _build_collection(tmp_path)
    canonical = next(root.glob(f"{MODEL}/selected/layout_01/layout_01/evaluation_rooms/*/canonical_scene.json"))
    replacement = canonical.with_name("copy.json")
    replacement.write_bytes(canonical.read_bytes())
    canonical.unlink()
    canonical.symlink_to(replacement.name)

    with pytest.raises(ValueError, match="symlink|regular"):
        discover_multi_room_evaluation_inventory(root)


def test_reader_rejects_selection_target_drift_and_duplicate_layouts(tmp_path: Path) -> None:
    root = _build_collection(tmp_path)
    selection_path = root / MODEL / "selection_manifest.json"
    selection = json.loads(selection_path.read_text())
    selection["selected_layouts"][0]["source_path"] = "source/other"
    _write_json(selection_path, selection)
    _refresh_collection_selection_hash(root)
    with pytest.raises(ValueError, match="target drift"):
        discover_multi_room_evaluation_inventory(root)

    root = _build_collection(tmp_path / "duplicate")
    selection_path = root / MODEL / "selection_manifest.json"
    selection = json.loads(selection_path.read_text())
    selection["selected_layouts"].append(deepcopy(selection["selected_layouts"][0]))
    selection["expected_layouts"] += 1
    _write_json(selection_path, selection)
    _refresh_collection_selection_hash(root)
    with pytest.raises(ValueError, match="duplicate layout"):
        discover_multi_room_evaluation_inventory(root)


def test_require_complete_fails_before_renderer_or_output(tmp_path: Path) -> None:
    inventory = discover_multi_room_evaluation_inventory(
        _build_collection(
            tmp_path, layouts=(("layout_01", ("succeeded", "failed")),)
        )
    )
    renderer = FakeRenderer()
    output = tmp_path / "dataset"

    with pytest.raises(ValueError, match="requires every room"):
        materialize_multi_room_evaluation_dataset(
            inventory,
            output_root=output,
            renderer=renderer,
            asset_root=_asset_root(tmp_path),
        )
    assert renderer.calls == 0
    assert not output.exists()
    assert not output.with_name(".dataset.building").exists()


def test_cli_complete_gate_precedes_renderer_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    collection = _build_collection(
        tmp_path, layouts=(("layout_01", ("succeeded", "failed")),)
    )

    def forbidden_renderer(**_: Any) -> Any:
        raise AssertionError("renderer must not be constructed")

    monkeypatch.setattr(materializer_cli, "BlenderRenderer", forbidden_renderer)
    assert materializer_cli.main(
        [
            "--collection-root",
            str(collection),
            "--model",
            MODEL,
            "--output-root",
            str(tmp_path / "dataset"),
            "--blender-bin",
            str(tmp_path / "blender"),
            "--asset-root",
            str(tmp_path / "assets"),
            "--require-complete",
        ]
    ) == 3


def test_cli_defaults_pin_existing_official_render_profile(tmp_path: Path) -> None:
    args = materializer_cli.build_parser().parse_args(
        [
            "--collection-root",
            str(tmp_path / "collection"),
            "--model",
            MODEL,
            "--output-root",
            str(tmp_path / "dataset"),
            "--blender-bin",
            str(tmp_path / "blender"),
            "--asset-root",
            str(tmp_path / "assets"),
        ]
    )
    for field, expected in OFFICIAL_RENDER_PROFILE.items():
        assert getattr(args, field) == expected


def test_official_materialization_rejects_renderer_profile_drift(
    tmp_path: Path,
) -> None:
    inventory = discover_multi_room_evaluation_inventory(_build_collection(tmp_path))
    renderer = FakeRenderer()
    renderer.width = 512
    with pytest.raises(ValueError, match="official.*profile mismatch"):
        materialize_multi_room_evaluation_dataset(
            inventory,
            output_root=tmp_path / "dataset",
            renderer=renderer,
            asset_root=_asset_root(tmp_path),
        )
    assert renderer.calls == 0


@pytest.mark.parametrize(
    "extra",
    (
        ("--evaluation-bindings", "private.json"),
        (
            "--evaluation-bindings",
            "private.json",
            "--campaign-template",
            "template.json",
        ),
    ),
)
def test_cli_bindings_without_complete_campaign_args_fail_before_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra: tuple[str, ...],
) -> None:
    collection = _build_collection(tmp_path)

    def forbidden_renderer(**_: Any) -> Any:
        raise AssertionError("renderer must not be constructed")

    monkeypatch.setattr(materializer_cli, "BlenderRenderer", forbidden_renderer)
    assert materializer_cli.main(
        [
            "--collection-root",
            str(collection),
            "--model",
            MODEL,
            "--output-root",
            str(tmp_path / "dataset"),
            "--blender-bin",
            str(tmp_path / "blender"),
            "--asset-root",
            str(tmp_path / "assets"),
            *extra,
        ]
    ) == 3


def test_cli_full_campaign_and_binding_emit_existing_run_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    collection = _build_collection(tmp_path)
    monkeypatch.setattr(
        materializer_cli,
        "BlenderRenderer",
        lambda **_: FakeRenderer(),
    )
    monkeypatch.setattr(materializer_cli, "REPO_ROOT", tmp_path)
    binding = tmp_path / "private.json"
    rc = materializer_cli.main(
        [
            "--collection-root",
            str(collection),
            "--model",
            MODEL,
            "--output-root",
            str(tmp_path / "dataset"),
            "--blender-bin",
            str(tmp_path / "blender"),
            "--asset-root",
            str(_asset_root(tmp_path)),
            "--campaign-template",
            str(ROOT / "configs/evaluation/campaigns/glm53_api1_full10_v1.json"),
            "--campaign-config-out",
            str(tmp_path / "campaign.json"),
            "--campaign-id",
            "mr-cli-binding-test-v1",
            "--attempt-parent",
            str(tmp_path / "attempts"),
            "--final-selection-root",
            str(tmp_path / "final"),
            "--evaluation-bindings",
            str(binding),
        ]
    )
    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    run_command = output["campaign"]["run_command"]
    assert run_command is not None
    assert run_command[-2:] == ["--bindings", str(binding.resolve())]


@pytest.mark.parametrize("target_kind", ("collection", "generation", "asset"))
def test_materializer_rejects_output_overlapping_read_only_sources(
    tmp_path: Path, target_kind: str
) -> None:
    collection = _build_collection(tmp_path)
    inventory = discover_multi_room_evaluation_inventory(collection)
    assets = _asset_root(tmp_path)
    targets = {
        "collection": collection / "derived",
        "generation": inventory.succeeded[0].source_index_path.parent / "derived",
        "asset": assets / "derived",
    }
    renderer = FakeRenderer()
    with pytest.raises(ValueError, match="disjoint"):
        materialize_multi_room_evaluation_dataset(
            inventory,
            output_root=targets[target_kind],
            renderer=renderer,
            asset_root=assets,
            materialization_config={"renderer": "fake-v1"},
        )
    assert renderer.calls == 0


def test_materializer_rechecks_source_index_before_render(tmp_path: Path) -> None:
    inventory = discover_multi_room_evaluation_inventory(_build_collection(tmp_path))
    inventory.succeeded[0].source_index_path.write_text(
        inventory.succeeded[0].source_index_path.read_text() + "\n",
        encoding="utf-8",
    )
    renderer = FakeRenderer()
    with pytest.raises(ValueError, match="index (changed|drift)"):
        materialize_multi_room_evaluation_dataset(
            inventory,
            output_root=tmp_path / "dataset",
            renderer=renderer,
            asset_root=_asset_root(tmp_path),
            materialization_config={"renderer": "fake-v1"},
        )
    assert renderer.calls == 0


def test_materializer_builds_self_contained_existing_case_layout(tmp_path: Path) -> None:
    inventory = discover_multi_room_evaluation_inventory(_build_collection(tmp_path))
    renderer = FakeRenderer()
    output = tmp_path / "dataset"
    result = materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=output,
        renderer=renderer,
        asset_root=_asset_root(tmp_path),
        materialization_config={"renderer": "fake-v1"},
    )
    case_id = result.case_ids[0]
    case = output / case_id

    assert renderer.calls == 1
    assert result.dataset_manifest["all_cases_ready"] is True
    assert result.dataset_manifest["all_expected_rooms_ready"] is True
    inspect_evaluation_dataset(output, expected_case_ids=result.case_ids)
    for relative in (
        "case_manifest.json",
        "scene/canonical_scene.json",
        "prepared/evaluation.blend",
        "annotation.json",
        "evidence/standardized_perspective.png",
        "evidence/standardized_top.png",
        "evidence/standardized_identity_map.png",
        "evidence/prepared_render_manifest.json",
        "evidence/collision_geometry_manifest.json",
        "provenance/source_inputs/canonical_scene.json",
        "provenance/source_inputs/scene_request.json",
        "provenance/source_inputs/object_plan.json",
        "provenance/source_inputs/asset_selection.json",
        "provenance/source_inputs/generation_input.json",
        "provenance/source_inputs/architecture_contract.json",
    ):
        assert (case / relative).is_file()
    assert (case / "evidence/collision_geometry").is_dir()
    assert (
        inventory.succeeded[0].artifacts["generation_input"].path.stat().st_ino
        != (case / "provenance/source_inputs/generation_input.json").stat().st_ino
    )
    load_collision_geometry_manifest(case / "evidence/collision_geometry_manifest.json")
    render_manifest_text = (case / "evidence/prepared_render_manifest.json").read_text()
    assert ".building" not in render_manifest_text
    assert str(tmp_path) not in render_manifest_text
    annotation = json.loads((case / "annotation.json").read_text())
    assert annotation["reviewed"] is False
    assert annotation["included_in_human_accuracy"] is False
    assert all(
        value["anomaly"] is False and value["unclear"] is True
        for value in annotation["metrics"].values()
    )
    case_manifest = json.loads((case / "case_manifest.json").read_text())
    assert case_manifest["source_prompt_used"] is False
    assert case_manifest["generation_prompt_withheld_from_evaluator"] is True
    assert "generation_input" not in json.dumps(case_manifest["paths"])


def test_materializer_resume_skips_hash_verified_case(tmp_path: Path) -> None:
    inventory = discover_multi_room_evaluation_inventory(_build_collection(tmp_path))
    output = tmp_path / "dataset"
    config = {"renderer": "fake-v1"}
    first = FakeRenderer()
    materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=output,
        renderer=first,
        asset_root=_asset_root(tmp_path),
        materialization_config=config,
    )
    second = FakeRenderer()
    result = materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=output,
        renderer=second,
        asset_root=tmp_path / "assets",
        materialization_config=config,
    )
    assert first.calls == 1
    assert second.calls == 0
    assert result.already_final is True
    assert result.resumed_cases == 1


def test_materializer_resume_rejects_renderer_identity_drift(tmp_path: Path) -> None:
    inventory = discover_multi_room_evaluation_inventory(_build_collection(tmp_path))
    output = tmp_path / "dataset"
    assets = _asset_root(tmp_path)
    materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=output,
        renderer=FakeRenderer(),
        asset_root=assets,
        materialization_config={"renderer": "fake-v1"},
    )
    renderer = FakeRenderer()
    with pytest.raises(ValueError, match="build identity"):
        materialize_multi_room_evaluation_dataset(
            inventory,
            output_root=output,
            renderer=renderer,
            asset_root=assets,
            materialization_config={"renderer": "fake-v2"},
        )
    assert renderer.calls == 0


def test_materializer_resume_rejects_extra_broken_symlink(tmp_path: Path) -> None:
    inventory = discover_multi_room_evaluation_inventory(_build_collection(tmp_path))
    output = tmp_path / "dataset"
    assets = _asset_root(tmp_path)
    config = {"renderer": "fake-v1"}
    result = materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=output,
        renderer=FakeRenderer(),
        asset_root=assets,
        materialization_config=config,
    )
    (output / result.case_ids[0] / "provenance/broken-link").symlink_to(
        "missing-target"
    )
    renderer = FakeRenderer()
    with pytest.raises(ValueError, match="symlink"):
        materialize_multi_room_evaluation_dataset(
            inventory,
            output_root=output,
            renderer=renderer,
            asset_root=assets,
            materialization_config=config,
        )
    assert renderer.calls == 0


def test_materializer_resume_rejects_referenced_asset_drift(tmp_path: Path) -> None:
    inventory = discover_multi_room_evaluation_inventory(_build_collection(tmp_path))
    output = tmp_path / "dataset"
    assets = _asset_root(tmp_path)
    config = {"renderer": "fake-v1"}
    materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=output,
        renderer=FakeRenderer(),
        asset_root=assets,
        materialization_config=config,
    )
    (assets / "asset_1/asset_1.fbx").write_bytes(b"changed")
    renderer = FakeRenderer()
    with pytest.raises(ValueError, match="build identity"):
        materialize_multi_room_evaluation_dataset(
            inventory,
            output_root=output,
            renderer=renderer,
            asset_root=assets,
            materialization_config=config,
        )
    assert renderer.calls == 0


def test_interrupted_materialization_is_hidden_and_resumable(tmp_path: Path) -> None:
    inventory = discover_multi_room_evaluation_inventory(
        _build_collection(
            tmp_path, layouts=(("layout_01", ("succeeded", "succeeded")),)
        )
    )
    output = tmp_path / "dataset"
    assets = _asset_root(tmp_path)
    config = {"renderer": "fake-v1"}
    with pytest.raises(RuntimeError, match="interruption"):
        materialize_multi_room_evaluation_dataset(
            inventory,
            output_root=output,
            renderer=FakeRenderer(fail_on_call=2),
            asset_root=assets,
            materialization_config=config,
        )
    building = output.with_name(".dataset.building")
    assert not output.exists()
    assert building.is_dir()
    assert not (building / "dataset_manifest.json").exists()
    with pytest.raises(ValueError, match="not finalized"):
        discover_cases(building)

    resumed_renderer = FakeRenderer()
    result = materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=output,
        renderer=resumed_renderer,
        asset_root=assets,
        materialization_config=config,
    )
    assert resumed_renderer.calls == 1
    assert result.resumed_cases == 1
    assert result.rendered_cases == 1
    assert output.is_dir() and not building.exists()
    assert len(discover_cases(output)) == 2


def test_allow_incomplete_keeps_explicit_diagnostic_ledger(tmp_path: Path) -> None:
    inventory = discover_multi_room_evaluation_inventory(
        _build_collection(
            tmp_path,
            layouts=(
                ("layout_01", ("succeeded", "failed")),
                ("layout_02", ("missing",)),
            ),
        )
    )
    output = tmp_path / "dataset"
    result = materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=output,
        renderer=FakeRenderer(),
        asset_root=_asset_root(tmp_path),
        require_complete=False,
        materialization_config={"renderer": "fake-v1"},
    )
    manifest = result.dataset_manifest
    assert manifest["all_cases_ready"] is True
    assert manifest["all_expected_rooms_ready"] is False
    assert manifest["official_full_model_score_eligible"] is False
    assert manifest["diagnostic_incomplete"] is True
    assert manifest["failed_room_count"] == 1
    assert manifest["missing_room_count"] == 1
    inspect_evaluation_dataset(output, expected_case_ids=result.case_ids)


def test_portable_collision_projection_loads_after_copy(tmp_path: Path) -> None:
    inventory = discover_multi_room_evaluation_inventory(_build_collection(tmp_path))
    dataset = tmp_path / "dataset"
    result = materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=dataset,
        renderer=FakeRenderer(),
        asset_root=_asset_root(tmp_path),
        materialization_config={"renderer": "fake-v1"},
    )
    projected = prepare_portable_dataset_view(dataset, tmp_path / "portable")
    case = projected / result.case_ids[0]
    load_collision_geometry_manifest(case / "evidence/collision_geometry_manifest.json")
    inspect_evaluation_dataset(projected, expected_case_ids=result.case_ids)


def test_portable_identity_commits_full_multi_room_source_inventory(
    tmp_path: Path,
) -> None:
    inventory = discover_multi_room_evaluation_inventory(_build_collection(tmp_path))
    dataset = tmp_path / "dataset"
    result = materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=dataset,
        renderer=FakeRenderer(),
        asset_root=_asset_root(tmp_path),
        materialization_config={"renderer": "fake-v1"},
    )
    before = inspect_evaluation_dataset(dataset, expected_case_ids=result.case_ids)
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_inventory"]["collection_manifest_sha256"] = "f" * 64
    manifest["source_collection_manifest_sha256"] = "f" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="fingerprint"):
        inspect_evaluation_dataset(dataset, expected_case_ids=result.case_ids)

    manifest["source_fingerprint_sha256"] = _json_sha256(
        manifest["source_inventory"]
    )
    _write_json(manifest_path, manifest)
    after = inspect_evaluation_dataset(dataset, expected_case_ids=result.case_ids)
    assert after.portable_fingerprint_sha256 != before.portable_fingerprint_sha256


def test_campaign_builder_reuses_template_kernel_and_existing_entrypoint(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    collection = _build_collection(repo)
    inventory = discover_multi_room_evaluation_inventory(collection)
    dataset = repo / "datasets/model"
    result = materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=dataset,
        renderer=FakeRenderer(),
        asset_root=_asset_root(repo),
        materialization_config={"renderer": "fake-v1"},
    )
    template_path = ROOT / "configs/evaluation/campaigns/glm53_api1_full10_v1.json"
    template = json.loads(template_path.read_text())
    config = build_existing_evaluation_campaign_config(
        repo_root=repo,
        template_path=template_path,
        dataset_root=dataset,
        campaign_id="mr-model-safe-evaluation-v1",
        model_label=MODEL,
        attempt_parent=repo / "outputs/attempts",
        final_selection_root=repo / "outputs/final",
    )

    assert config["kernel"] == template["kernel"]
    assert config["attempt_policy"] == template["attempt_policy"]
    assert config["selection"] == template["selection"]
    assert config["judge_profile_id"] == template["judge_profile_id"]
    assert config["dataset"]["expected_case_ids"] == list(result.case_ids)
    loaded = load_campaign(_write_json(repo / "campaign.json", config), repo_root=repo)
    assert loaded.dataset.expected_case_ids == result.case_ids
    command = evaluation_campaign_command(
        config_path=repo / "campaign.json",
        python_executable=repo / ".venv/bin/python",
        run=True,
        bindings_path=repo / ".runtime/evaluation.json",
    )
    assert command[1:4] == (
        "-m",
        "benchmark.evaluation_campaign",
        "run",
    )


def test_official_campaign_builder_rejects_incomplete_source(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    inventory = discover_multi_room_evaluation_inventory(
        _build_collection(
            repo, layouts=(("layout_01", ("succeeded", "failed")),)
        )
    )
    dataset = repo / "datasets/model"
    materialize_multi_room_evaluation_dataset(
        inventory,
        output_root=dataset,
        renderer=FakeRenderer(),
        asset_root=_asset_root(repo),
        require_complete=False,
        materialization_config={"renderer": "fake-v1"},
    )
    with pytest.raises(ValueError, match="diagnostic-only"):
        build_existing_evaluation_campaign_config(
            repo_root=repo,
            template_path=ROOT / "configs/evaluation/campaigns/glm53_api1_full10_v1.json",
            dataset_root=dataset,
            campaign_id="mr-model-safe-evaluation-v1",
            model_label=MODEL,
            attempt_parent=repo / "outputs/attempts",
            final_selection_root=repo / "outputs/final",
        )
