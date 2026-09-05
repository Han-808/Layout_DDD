from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from benchmark.scene_generation.campaign.api import (
    preflight_generation_campaign,
    prepare_generation_campaign,
    resolve_generation_campaign,
    resource_gate_generation_campaign,
    run_generation_campaign,
)
from benchmark.scene_generation.campaign.cli import main as campaign_main
from benchmark.scene_generation.frozen_two_stage.compatibility.loader import (
    load_frozen_core,
)
from benchmark.scene_generation.multi_room.assembly import (
    AssemblyError,
    RoomAssemblySource,
    build_compiled_architecture,
    build_global_scene,
    build_room_evaluation_inputs,
    build_room_projection,
    canonical_json_bytes,
    sha256_file,
    sha256_bytes,
    validate_evaluation_index,
)
from benchmark.scene_generation.multi_room.contracts import (
    MultiRoomContractError,
    validate_retrieval_results,
    validate_room_object_plan,
)
from benchmark.scene_generation.multi_room.floor_plan import (
    FloorPlanValidationError,
    LoadedFloorPlan,
    compile_room_brief,
    load_floor_plan,
    validate_floor_plan,
)
from benchmark.scene_generation.multi_room.provenance import (
    compatibility_source_manifest,
)
from benchmark.scene_generation.campaign.multi_room_profiles import (
    load_multi_room_profile_registry,
)
from benchmark.scene_generation.campaign.loader import load_campaign_profile_bundle
from tools.api3_anthropic_runner_v2.transport import TransportResult
from benchmark.architecture_policy import validate_architecture_contract
from benchmark.scene_io.validate import (
    validate_asset_selection,
    validate_generated_scene,
    validate_generation_input,
    validate_object_plan,
    validate_scene_request,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "configs/generation/campaign_v2"
RETRIEVAL_PROFILE_ID = "imaginarium-qwen3-embedding-0.6b-stable-top1-v2"


def _room(
    index: int,
    *,
    room_id: str,
    x0: float,
    width: float = 3.0,
    depth: float = 4.0,
    attachment_min: int = 1,
    attachment_max: int = 1,
) -> dict[str, Any]:
    x1 = x0 + width
    return {
        "room_id": room_id,
        "generation_index": index,
        "room_type": f"type_{index}",
        "theme": f"theme_{index}",
        "object_count_tier": "compact_7_10",
        "instruction": f"Generate only room {index} with its own function.",
        "target_instances": {"min": 7, "max": 10},
        "room_dimensions_m": [width, depth, 3.0],
        "room": {
            "boundary": [[x0, 0.0], [x1, 0.0], [x1, depth], [x0, depth]],
            "height": 3.0,
            "unit": "meter",
            "topology": "axis_aligned_rectangle",
        },
        "runner_projection": {
            "local_to_global_offset_m": [x0, 0.0, 0.0],
            "local_room": {
                "boundary": [[0.0, 0.0], [width, 0.0], [width, depth], [0.0, depth]],
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
            "minimum_count": attachment_min,
            "maximum_count": attachment_max,
            "selection_policy": "model_selects_functionally_appropriate_objects",
        },
    }


def _plan(
    count: int,
    *,
    room_ids: list[str] | None = None,
    attachment_min: int = 1,
    attachment_max: int = 1,
) -> dict[str, Any]:
    ids = room_ids or [f"room_{index + 1}" for index in range(count)]
    rooms = [
        _room(
            index,
            room_id=ids[index],
            x0=3.0 * index,
            attachment_min=attachment_min,
            attachment_max=attachment_max,
        )
        for index in range(count)
    ]
    shared = [
        {
            "shared_wall_id": f"shared_wall_{index + 1}",
            "segment_global_m": [
                [3.0 * (index + 1), 0.0],
                [3.0 * (index + 1), 4.0],
            ],
            "rooms": [
                {"room_id": ids[index], "wall_id": "east_wall"},
                {"room_id": ids[index + 1], "wall_id": "west_wall"},
            ],
            "opaque": True,
            "full_height": True,
        }
        for index in range(count - 1)
    ]
    return {
        "schema_version": "multi_room_floor_plan_v1",
        "generation_mode": "multi_room_with_architecture_v1",
        "room_prompt_version": "room_generation_prompt_v1",
        "layout_id": "layout_42",
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
        "generation_order": ids,
        "shared_walls": shared,
        "rooms": rooms,
    }


def _write_plan(path: Path, value: Mapping[str, Any]) -> Path:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "room_ids",
    [
        ["room_1"],
        ["room_1", "room_10"],
        ["room_1", "room_2", "room_3", "room_4"],
        ["room_1", "room_2", "room_3", "room_4", "room_100"],
    ],
)
def test_floor_plan_accepts_generic_declared_cardinality(room_ids: list[str]) -> None:
    report = validate_floor_plan(_plan(len(room_ids), room_ids=room_ids))
    assert report["room_count"] == len(room_ids)
    assert report["generation_order"] == room_ids


def test_floor_plan_rejects_repeated_room_types_per_authoritative_contract() -> None:
    value = _plan(2)
    value["rooms"][0]["room_type"] = "bedroom"
    value["rooms"][1]["room_type"] = "bedroom"
    with pytest.raises(FloorPlanValidationError, match="room_type.*unique"):
        validate_floor_plan(value)


def test_floor_plan_rejects_coordinates_that_collapse_float_translation() -> None:
    value = _plan(1)
    width = 10**16 + 1
    boundary = [[0, 0], [width, 0], [width, 4], [0, 4]]
    value["rooms"][0]["room"]["boundary"] = deepcopy(boundary)
    value["rooms"][0]["runner_projection"]["local_room"]["boundary"] = deepcopy(
        boundary
    )
    value["rooms"][0]["room_dimensions_m"][0] = width
    value["global_envelope"]["boundary"] = deepcopy(boundary)
    value["global_envelope"]["dimensions_m"][0] = width
    with pytest.raises(FloorPlanValidationError, match="round-trip|precision"):
        validate_floor_plan(value)


def test_floor_plan_loader_rejects_numeric_literal_precision_loss(
    tmp_path: Path,
) -> None:
    path = _write_plan(tmp_path / "floor.json", _plan(1))
    text = path.read_text(encoding="utf-8").replace(
        "0.08", "0.0800000000000000000001"
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(FloorPlanValidationError, match="loses precision"):
        load_floor_plan(path)


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda value: value.update(room_count=3), "room_count"),
        (lambda value: value.update(unknown=True), "Additional properties"),
        (
            lambda value: value["rooms"][1]["room"].update(
                boundary=[[2.0, 0.0], [5.0, 0.0], [5.0, 4.0], [2.0, 4.0]]
            ),
            "offset x|overlap",
        ),
        (lambda value: value.update(shared_walls=[]), "shared_walls|connected"),
        (
            lambda value: value["rooms"][0].update(
                object_count_tier="standard_11_14",
                target_instances={"min": 11, "max": 14},
            ),
            "object_count_tier",
        ),
        (
            lambda value: value["rooms"][0]["wall_attachment_requirement"].update(
                maximum_count=11
            ),
            "wall attachment",
        ),
        (
            lambda value: value["shared_walls"][0]["rooms"][1].update(
                wall_id="east_wall"
            ),
            "opposing|exact",
        ),
        (
            lambda value: value["shared_walls"][0].update(opaque=False),
            "True was expected",
        ),
    ],
)
def test_floor_plan_rejects_malformed_semantics(mutator: Any, match: str) -> None:
    value = _plan(2)
    mutator(value)
    with pytest.raises(FloorPlanValidationError, match=match):
        validate_floor_plan(value)


def test_room_brief_is_isolated_from_peer_room_content(tmp_path: Path) -> None:
    plan = load_floor_plan(_write_plan(tmp_path / "floor.json", _plan(2)))
    brief = compile_room_brief(plan, "room_1")
    serialized = json.dumps(brief, sort_keys=True)
    assert brief["room_id"] == "room_1"
    assert "Generate only room 0" in serialized
    assert "Generate only room 1" not in serialized
    assert "room_2" not in serialized


def _object_plan(room_type: str, *, wall_count: int = 1) -> dict[str, Any]:
    objects = [
        {
            "id": "wall_art",
            "category": "wall_art",
            "role": "decoration",
            "description": "one wall artwork",
            "count": wall_count,
            "estimated_size": [0.8, 0.1, 0.6],
            "metadata": {
                "intended_role": "decoration",
                "zone": "main_zone",
                "support": "north_wall",
                "directed": True,
                "functional_side": "local_neg_y",
                "facing_intent": "face into the room",
                "retrieval_query": "framed wall artwork",
                "requested_count": wall_count,
            },
            "placement_intent": {
                "absolute_relations": ["attached to north wall"],
                "relative_relations": [],
            },
        },
        {
            "id": "chair",
            "category": "chair",
            "role": "seating",
            "description": "six simple chairs",
            "count": 7 - wall_count,
            "estimated_size": [0.5, 0.5, 0.9],
            "metadata": {
                "intended_role": "seating",
                "zone": "main_zone",
                "support": "floor",
                "directed": True,
                "functional_side": "local_neg_y",
                "facing_intent": "face room center",
                "retrieval_query": "simple chair with backrest",
                "requested_count": 7 - wall_count,
            },
            "placement_intent": {
                "absolute_relations": ["inside room"],
                "relative_relations": [],
            },
        },
    ]
    return {
        "schema_version": "multi_room_object_plan_v1",
        "scene_type": room_type,
        "scene_description": f"A coherent {room_type}.",
        "prompt_granularity": "fine_grained",
        "global_constraints": ["keep circulation usable"],
        "zones": [
            {"id": "main_zone", "description": "main area", "extent_hint": "center"}
        ],
        "relations": [],
        "objects": objects,
    }


def test_wall_attachment_range_is_declared_not_hardcoded(tmp_path: Path) -> None:
    core = load_frozen_core(ROOT / "tools/api3_anthropic_runner_v2")
    plan = load_floor_plan(
        _write_plan(
            tmp_path / "floor.json",
            _plan(1, attachment_min=0, attachment_max=3),
        )
    )
    brief = compile_room_brief(plan, "room_1")
    assert validate_room_object_plan(
        _object_plan("type_0", wall_count=3),
        room_brief=brief,
        frozen_validate_object_plan=core.validate_object_plan,
    )["objects"][0]["metadata"]["support"] == "north_wall"
    with pytest.raises(MultiRoomContractError, match="outside"):
        validate_room_object_plan(
            _object_plan("type_0", wall_count=4),
            room_brief=brief,
            frozen_validate_object_plan=core.validate_object_plan,
        )


def test_dispatch_is_explicit_not_inferred_from_floor_plan(tmp_path: Path) -> None:
    path = _write_plan(tmp_path / "floor.json", _plan(1))
    with pytest.raises(ValueError, match="single-room campaigns"):
        prepare_generation_campaign(
            "api2-kimi-k3-scene10-v2", floor_plan_path=path
        )
    with pytest.raises(ValueError, match="require an explicit floor-plan"):
        prepare_generation_campaign("api2-kimi-k3-multi-room-v1")


def test_run_rejects_floor_plan_file_toctou_before_external_runtime(
    tmp_path: Path,
) -> None:
    path = _write_plan(tmp_path / "floor.json", _plan(1))
    prepared = prepare_generation_campaign(
        "api2-kimi-k3-multi-room-v1", floor_plan_path=path
    )
    changed = _plan(1)
    changed["rooms"][0]["theme"] = "changed after prepare"
    _write_plan(path, changed)
    external_calls: list[bool] = []

    def runtime_factory(**_: Any) -> Any:
        external_calls.append(True)
        raise AssertionError("external runtime must not be constructed")

    with pytest.raises(ValueError, match="floor-plan identity changed"):
        run_generation_campaign(
            prepared,
            output_root=tmp_path / "output",
            runtime_factory=runtime_factory,
        )
    assert external_calls == []


def test_run_rejects_mutated_prepared_floor_plan_before_external_runtime(
    tmp_path: Path,
) -> None:
    path = _write_plan(tmp_path / "floor.json", _plan(1))
    prepared = prepare_generation_campaign(
        "api2-kimi-k3-multi-room-v1", floor_plan_path=path
    )
    prepared.floor_plan.value["rooms"][0]["theme"] = "mutated mapping"
    external_calls: list[bool] = []

    def runtime_factory(**_: Any) -> Any:
        external_calls.append(True)
        raise AssertionError("external runtime must not be constructed")

    with pytest.raises(ValueError, match="floor-plan identity changed"):
        run_generation_campaign(
            prepared,
            output_root=tmp_path / "output",
            runtime_factory=runtime_factory,
        )
    assert external_calls == []


def test_canonical_cli_check_accepts_explicit_multi_room_floor_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_plan(tmp_path / "floor.json", _plan(1))
    assert campaign_main(
        [
            "check",
            "--campaign",
            "api2-kimi-k3-multi-room-v1",
            "--floor-plan",
            str(path),
        ]
    ) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["valid"] is True
    assert value["campaign"]["generation_mode"] == (
        "multi_room_with_architecture_v1"
    )
    assert value["credential_loaded"] is False
    assert value["network_used"] is False


def test_resolve_resource_gate_and_preflight_share_canonical_boundaries(
    tmp_path: Path,
) -> None:
    prepared = prepare_generation_campaign(
        "api2-kimi-k3-multi-room-v1",
        floor_plan_path=_write_plan(tmp_path / "floor.json", _plan(1)),
    )
    generation_binding = _binding(tmp_path / "binding.json")
    resource_binding = tmp_path / "resources.json"
    resource_binding.write_text(
        json.dumps(
            {
                "schema_version": "generation_resource_bindings_v2",
                "bindings": {
                    "asset-index-json:imaginarium-qwen3-0.6b-v2": {
                        "path": "/nonexistent/index.json"
                    },
                    "asset-index-matrix:imaginarium-qwen3-0.6b-v2": {
                        "path": "/nonexistent/index.npy"
                    },
                    "encoder-snapshot:qwen3-embedding-0.6b-97b0c614-v2": {
                        "path": "/nonexistent/encoder"
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    _, _, resolved = resolve_generation_campaign(
        prepared,
        generation_bindings_path=generation_binding,
        resource_bindings_path=resource_binding,
    )
    assert resolved["credential_loaded"] is False
    runtime = _FakeRetrievalRuntime()
    _, gate = resource_gate_generation_campaign(
        prepared, runtime_factory=lambda **_: runtime
    )
    assert gate["status"] == "ready"
    preflight, _ = preflight_generation_campaign(
        prepared,
        generation_bindings_path=generation_binding,
        environ={"PRIVATE_MULTI_ROOM_KEY": "app:test-secret"},
        runtime_factory=lambda **_: runtime,
        transport=lambda *args, **kwargs: _transport_response({"ok": True}),
    )
    assert preflight["ok"] is True


def test_additive_registry_reuses_every_reviewed_model_profile() -> None:
    base = load_campaign_profile_bundle(PROFILE_ROOT)
    registry = load_multi_room_profile_registry(ROOT, base)
    assert {item.model_profile_id for item in registry.campaigns.values()} == set(
        base.models.by_id
    )
    assert all(
        item.retrieval_profile_id == RETRIEVAL_PROFILE_ID
        for item in registry.campaigns.values()
    )


def test_compatibility_source_manifest_closes_projection_dependencies() -> None:
    paths = {
        item["path"] for item in compatibility_source_manifest()["files"]
    }
    assert {
        "benchmark/adapters/catalog_placement/converter.py",
        "benchmark/adapters/catalog_placement/prompt.py",
        "benchmark/architecture_policy.py",
        "benchmark/assets/facing.py",
        "benchmark/io_contracts.py",
        "benchmark/nl_scene/converter.py",
        "benchmark/nl_scene/generation_input.py",
        "benchmark/resources.py",
        "benchmark/scene_io/validate.py",
        "benchmark/task_contract.py",
        "benchmark/utils/io.py",
        "benchmark._resources/schemas/generator_catalog_placement_v1.schema.json",
        "benchmark._resources/schemas/multi_room/room_evaluation_object_plan_v1.schema.json",
    } <= paths


def _asset_selection(plan: Mapping[str, Any]) -> dict[str, Any]:
    objects = []
    for item in plan["objects"]:
        slot = str(item["id"])
        size = [0.8, 0.1, 0.6] if slot == "wall_art" else [0.5, 0.5, 0.9]
        objects.append(
            {
                "object_id": slot,
                "object_spec": deepcopy(item),
                "retrieval_query": {
                    "description": item["metadata"]["retrieval_query"],
                    "category": None,
                    "size_constraint": None,
                    "top_k": 1,
                },
                "selected_asset": {
                    "jid": f"asset_{slot}",
                    "category": item["category"],
                    "desc": item["description"],
                    "short_desc": item["description"],
                    "size": size,
                    "asset_ref": {
                        "source_db": "imaginarium",
                        "asset_key": f"asset_{slot}",
                    },
                    "asset_proxy": {
                        "type": "canonical_catalog_bbox",
                        "bbox_center_local": [0.0, 0.0, 0.0],
                        "bbox_size": size,
                    },
                    "metadata": {},
                },
                "candidates": [],
                "selection_action": "select",
                "selection_decision": {
                    "action": "select",
                    "selected_jid": f"asset_{slot}",
                    "reason": "frozen top1",
                    "generation_request": None,
                },
                "selection_reason": "frozen top1",
            }
        )
    return {"schema_version": "multi_room_frozen_asset_selection_v1", "objects": objects}


def _placement() -> dict[str, Any]:
    instances = [
        {
            "instance_id": "wall_art_1",
            "asset_id": "asset_wall_art",
            "slot_id": "wall_art",
            "center_m": [1.5, 3.8, 1.5],
            "uniform_scale": 1.0,
            "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
        }
    ]
    instances.extend(
        {
            "instance_id": f"chair_{index + 1}",
            "asset_id": "asset_chair",
            "slot_id": "chair",
            "center_m": [0.4 + (index % 3), 0.5 + (index // 3), 0.45],
            "uniform_scale": 1.0,
            "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
        }
        for index in range(6)
    )
    return {"schema_version": "catalog_placement_v1", "instances": instances}


def test_assembly_is_translation_only_and_global_is_not_canonical(tmp_path: Path) -> None:
    plan = load_floor_plan(_write_plan(tmp_path / "floor.json", _plan(2)))
    sources = []
    for index, room_id in enumerate(plan.generation_order):
        brief = compile_room_brief(plan, room_id)
        object_plan = _object_plan(brief["room_type"])
        sources.append(
            RoomAssemblySource(
                room_key=f"room_{index:03d}",
                room_id=room_id,
                generation_index=index,
                status="complete",
                room_result_path=Path(f"rooms/room_{index:03d}/room_result.json"),
                room_result_sha256=f"{index + 1}" * 64,
                object_plan_artifact_sha256=f"{index + 3}" * 64,
                room_brief=brief,
                object_plan=object_plan,
                asset_selection=_asset_selection(object_plan),
                placement=_placement(),
            )
        )
    compiled = build_compiled_architecture(plan)
    assert compiled["wall_inventory_contract"] == {
        "logical_support_topology": "active_walls_per_room_v1",
        "physical_inventory": (
            "exterior_walls_plus_deduplicated_shared_walls_v1"
        ),
    }
    assert all(item["wall_role"] == "active" for item in compiled["active_walls"])
    assert all(
        item["wall_role"] == "exterior" for item in compiled["exterior_walls"]
    )
    assert all(item["deduplicated"] is True for item in compiled["shared_walls"])
    compiled_sha = sha256_bytes(canonical_json_bytes(compiled))
    global_scene = build_global_scene(
        plan, sources, compiled_architecture_sha256=compiled_sha
    )
    assert global_scene["schema_version"] == "multi_room_scene_v1"
    assert "boundary" not in global_scene
    second = next(
        item for item in global_scene["objects"] if item["room_id"] == "room_2"
    )
    assert second["global_center_m"][0] == second["local_center_m"][0] + 3.0
    projection = build_room_projection(
        sources[1], compiled_architecture_sha256=compiled_sha
    )
    assert projection["schema_version"] == "canonical_scene_v1"
    assert projection["boundary"] == [[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0]]
    assert projection["objects"][0]["id"].startswith("room_2__")
    companions = build_room_evaluation_inputs(sources[1], projection)
    request_id = projection["request_id"]
    assert request_id == "layout_42__room_001"
    assert companions["object_plan"]["schema_version"] == (
        "room_evaluation_object_plan_v1"
    )
    assert companions["object_plan"]["source_object_plan_artifact_sha256"] == (
        sources[1].object_plan_artifact_sha256
    )
    for name in ("scene_request", "object_plan", "asset_selection", "generation_input"):
        assert companions[name]["request_id"] == request_id
    assert projection["metadata"]["instance_registry"]["request_id"] == request_id
    validate_scene_request(companions["scene_request"])
    validate_object_plan(companions["object_plan"])
    validate_asset_selection(companions["asset_selection"])
    validate_generation_input(companions["generation_input"])
    validate_architecture_contract(companions["architecture_contract"])


class _FakeRetrievalRuntime:
    embedding_model_name = "Qwen/Qwen3-Embedding-0.6B"
    profile_id = RETRIEVAL_PROFILE_ID

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def gate(self, **_: Any) -> dict[str, Any]:
        return {
            "schema_version": "generation_retrieval_gate_report_v2",
            "status": "ready",
            "strict": True,
            "errors": [],
            "warnings": [],
            "observed": self.public_provenance(),
            "golden_results": [],
        }

    def public_provenance(self) -> dict[str, Any]:
        return {
            "schema_version": "generation_retrieval_provenance_v2",
            "retrieval_profile_id": self.profile_id,
            "catalog_sha256": "0" * 64,
            "profile_sha256": "1" * 64,
            "dataset_id": "imaginarium-assets-v1",
            "asset_namespace": "imaginarium-jid-v1",
        }

    def retrieve_batch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.requests.append(deepcopy(dict(request)))
        results = []
        for order, row in enumerate(request["requests"]):
            slot = str(row["slot_id"])
            size = [0.8, 0.1, 0.6] if slot == "wall_art" else [0.5, 0.5, 0.9]
            results.append(
                {
                    "order": order,
                    "slot_id": slot,
                    "retrieval_query": row["retrieval_query"],
                    "size_constraint": None,
                    "invocation_count": 1,
                    "rank1": {
                        "rank": 1,
                        "jid": f"asset_{slot}",
                        "short_desc": slot,
                        "size": size,
                        "category": slot,
                        "description": slot,
                        "score": 0.75,
                        "index_row": order,
                    },
                    "accepted_as_frozen_outcome": True,
                }
            )
        return {
            "schema_version": "hy34_frozen_top1_results_v1",
            "total_invocations": len(results),
            "retry_count": 0,
            "asset_replacement_count": 0,
            "results": results,
        }


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["results"][0]["rank1"].update(rank=2),
        lambda value: value["results"][0].update(retrieval_query="rewritten"),
        lambda value: value["results"][0].update(invocation_count=7),
        lambda value: value.update(retry_count=1),
        lambda value: value["results"][0]["rank1"].update(size=[1.0, 0.0, 1.0]),
        lambda value: value["results"][0]["rank1"].update(score=float("nan")),
        lambda value: value["results"][0]["rank1"].update(category=None),
    ],
)
def test_retrieval_result_gate_preserves_exact_frozen_top1(mutator: Any) -> None:
    plan = _object_plan("bedroom")
    request = {
        "requests": [
            {
                "slot_id": item["id"],
                "retrieval_query": item["metadata"]["retrieval_query"],
                "size_constraint": None,
            }
            for item in plan["objects"]
        ]
    }
    value = _FakeRetrievalRuntime().retrieve_batch(request)
    mutator(value)
    with pytest.raises(MultiRoomContractError):
        validate_retrieval_results(value, plan=plan)


def _chat_response(content: Mapping[str, Any] | str) -> bytes:
    rendered = (
        content
        if isinstance(content, str)
        else json.dumps(content, separators=(",", ":"))
    )
    return json.dumps(
        {
            "model": "kimi-k3",
            "choices": [
                {"message": {"content": rendered}}
            ],
            "usage": {"completion_tokens": 1},
        },
        separators=(",", ":"),
    ).encode()


def _transport_response(content: Mapping[str, Any]) -> TransportResult:
    return TransportResult(
        status="response",
        elapsed_seconds=0.01,
        stage="complete",
        http_status=200,
        response_body=_chat_response(content),
    )


def _binding(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "generation_route_bindings_v2",
                "bindings": {
                    "api2-chat-top-level-reasoning-v1": {
                        "endpoint": "https://runtime.example.invalid/v1/chat/completions",
                        "credential_env": "PRIVATE_MULTI_ROOM_KEY",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _run_fake_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    progress: Any = None,
    resume: bool = False,
    shared: dict[str, Any] | None = None,
) -> tuple[Any, _FakeRetrievalRuntime, list[bytes], Path]:
    shared = shared if shared is not None else {}
    if "plan_path" not in shared:
        shared["plan_path"] = _write_plan(tmp_path / "floor.json", _plan(2))
    plan_path = shared["plan_path"]
    prepared = prepare_generation_campaign(
        "api2-kimi-k3-multi-room-v1", floor_plan_path=plan_path
    )
    core = load_frozen_core(prepared.core_root)
    responses = shared.setdefault(
        "responses",
        [
            _chat_response(_object_plan("type_0")),
            _chat_response(_placement()),
            _chat_response(_object_plan("type_1")),
            _chat_response(_placement()),
        ],
    )
    calls = shared.setdefault("calls", [])
    preflight_calls = shared.setdefault("preflight_calls", [])
    request_bodies = shared.setdefault("request_bodies", [])

    def post_once(*args: Any, **__: Any) -> TransportResult:
        request_bodies.append(bytes(args[1]))
        body = responses.pop(0)
        calls.append(body)
        return TransportResult(
            status="response",
            elapsed_seconds=0.01,
            stage="complete",
            http_status=200,
            response_body=body,
        )

    monkeypatch.setattr(core, "post_once", post_once)
    fail_write_name = shared.get("fail_write_name")
    if isinstance(fail_write_name, str):
        original_write_json = core.write_json_exclusive

        def write_json_with_crash(path: Path, value: Mapping[str, Any]) -> None:
            original_write_json(path, value)
            if (
                path.name == fail_write_name
                and not shared.get("write_crash_triggered", False)
            ):
                shared["write_crash_triggered"] = True
                raise RuntimeError("injected finalization crash")

        monkeypatch.setattr(core, "write_json_exclusive", write_json_with_crash)
    runtime = shared.setdefault("retriever", _FakeRetrievalRuntime())
    output = tmp_path / "output"
    def preflight_transport(*args: Any, **kwargs: Any) -> TransportResult:
        preflight_calls.append((args, kwargs))
        return _transport_response({"ok": True})

    if "binding" not in shared:
        shared["binding"] = _binding(tmp_path / "binding.json")
    result = run_generation_campaign(
        prepared,
        output_root=output,
        generation_bindings_path=shared["binding"],
        environ=(
            {}
            if shared.get("omit_credential")
            else {"PRIVATE_MULTI_ROOM_KEY": "app:secret-test-value"}
        ),
        runtime_factory=lambda **_: runtime,
        transport=preflight_transport,
        progress=progress,
        resume=resume,
    )
    return result, runtime, calls, output


def test_fake_backed_canonical_api_runs_complete_multi_room_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (summary, stopped, preflight), runtime, calls, output = _run_fake_campaign(
        tmp_path, monkeypatch
    )
    assert preflight["ok"] is True
    assert stopped is False
    assert summary["complete_rooms"] == 2
    assert summary["failed_rooms"] == 0
    assert summary["projected_rooms"] == 2
    assert len(calls) == 4
    assert len(runtime.requests) == 2
    index = json.loads(
        (output / "layout_42/room_evaluation_index.json").read_text()
    )
    assert [item["room_id"] for item in index["rooms"]] == ["room_1", "room_2"]
    assert all(not Path(item["projection_path"]).is_absolute() for item in index["rooms"])
    layout_root = output / "layout_42"
    validate_evaluation_index(index, layout_root=layout_root)
    artifact_fields = {
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
    for room in index["rooms"]:
        loaded: dict[str, Any] = {}
        for name, (path_field, hash_field) in artifact_fields.items():
            path = layout_root / room[path_field]
            assert path.is_file() and not path.is_symlink()
            assert sha256_file(path) == room[hash_field]
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        request_id = loaded["canonical_scene"]["request_id"]
        assert loaded["scene_request"]["request_id"] == request_id
        assert loaded["object_plan"]["request_id"] == request_id
        assert loaded["asset_selection"]["request_id"] == request_id
        assert loaded["generation_input"]["request_id"] == request_id
        assert (
            loaded["canonical_scene"]["metadata"]["instance_registry"][
                "request_id"
            ]
            == request_id
        )
        assert loaded["object_plan"]["schema_version"] == (
            "room_evaluation_object_plan_v1"
        )
        validate_generated_scene(loaded["canonical_scene"])
        validate_scene_request(loaded["scene_request"])
        validate_object_plan(loaded["object_plan"])
        validate_asset_selection(loaded["asset_selection"])
        validate_generation_input(loaded["generation_input"])
        validate_architecture_contract(loaded["architecture_contract"])
    scene = json.loads(
        (output / "layout_42/assembled_multi_room_scene.json").read_text()
    )
    assert scene["assembly_status"] == "complete"
    assert len(scene["objects"]) == 14
    public = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            output / "run_manifest.json",
            output / "execution_policy.json",
            output / "summary.json",
            output / "layout_42/assembly_manifest.json",
        )
    )
    assert "secret-test-value" not in public
    assert "runtime.example.invalid" not in public


def test_blank_catalog_text_uses_public_slot_fallback_only_for_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BlankCatalogTextRuntime(_FakeRetrievalRuntime):
        def retrieve_batch(self, request: Mapping[str, Any]) -> dict[str, Any]:
            result = super().retrieve_batch(request)
            for row in result["results"]:
                row["rank1"]["category"] = ""
                row["rank1"]["description"] = ""
                row["rank1"]["short_desc"] = ""
            return result

    shared: dict[str, Any] = {"retriever": BlankCatalogTextRuntime()}
    (summary, stopped, _), _, _, output = _run_fake_campaign(
        tmp_path, monkeypatch, shared=shared
    )

    assert stopped is False
    assert summary["complete_rooms"] == 2
    assert summary["failed_rooms"] == 0
    room_root = output / "layout_42/rooms/room_000"
    retrieval = json.loads((room_root / "retrieval_results.json").read_text())
    selection = json.loads((room_root / "asset_selection.json").read_text())
    assert retrieval["results"][0]["rank1"]["category"] == ""
    assert selection["objects"][0]["selected_asset"]["category"] == ""

    index = json.loads(
        (output / "layout_42/room_evaluation_index.json").read_text()
    )
    projection_path = output / "layout_42" / index["rooms"][0]["projection_path"]
    projection = json.loads(projection_path.read_text())
    categories = {item["category"] for item in projection["objects"]}
    assert categories == {"wall_art", "chair"}


def test_exact_json_fence_is_normalized_and_relation_family_is_projected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_plan = _object_plan("type_0")
    first_plan["relations"] = [
        {
            "family": "proximity",
            "type": "between",
            "subject_id": "chair",
            "object_id": "wall_art",
        }
    ]
    fenced_plan = "```json\n" + json.dumps(first_plan) + "\n```"
    fenced_placement = "```json\n" + json.dumps(_placement()) + "\n```"
    shared = {
        "responses": [
            _chat_response(fenced_plan),
            _chat_response(fenced_placement),
            _chat_response(_object_plan("type_1")),
            _chat_response(_placement()),
        ]
    }
    (summary, stopped, _), _, _, output = _run_fake_campaign(
        tmp_path, monkeypatch, shared=shared
    )

    assert stopped is False
    assert summary["complete_rooms"] == 2
    room_root = output / "layout_42/rooms/room_000"
    assert (room_root / "object_plan_first_emission.json").read_text().startswith(
        "```json"
    )
    assert (room_root / "catalog_placement_first_emission.json").read_text().startswith(
        "```json"
    )
    assert json.loads((room_root / "object_plan.json").read_text())["relations"][0][
        "family"
    ] == "proximity"
    assert json.loads((room_root / "object_plan_validation.json").read_text()) == {
        "response_envelope": "single_json_code_fence_v1",
        "syntactic_normalization": True,
        "valid": True,
    }
    assert json.loads((room_root / "placement_validation.json").read_text()) == {
        "response_envelope": "single_json_code_fence_v1",
        "syntactic_normalization": True,
        "valid": True,
    }

    index = json.loads(
        (output / "layout_42/room_evaluation_index.json").read_text()
    )
    projected_plan_path = (
        output / "layout_42" / index["rooms"][0]["object_plan_path"]
    )
    projected_plan = json.loads(projected_plan_path.read_text())
    assert projected_plan["relations"][0] == {
        "family": "oor",
        "type": "binary_between",
        "subject_id": "chair",
        "object_id": "wall_art",
    }
    assert projected_plan["source_object_plan_canonical_sha256"] == sha256_bytes(
        canonical_json_bytes(first_plan)
    )


def test_json_fence_with_surrounding_prose_remains_contract_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrapped_with_prose = (
        "Here is the requested object plan:\n```json\n"
        + json.dumps(_object_plan("type_0"))
        + "\n```"
    )
    shared = {
        "responses": [
            _chat_response(wrapped_with_prose),
            _chat_response(_object_plan("type_1")),
            _chat_response(_placement()),
        ]
    }
    (summary, stopped, _), _, _, output = _run_fake_campaign(
        tmp_path, monkeypatch, shared=shared
    )

    assert stopped is False
    assert summary["complete_rooms"] == 1
    assert summary["failed_rooms"] == 1
    first = json.loads(
        (output / "layout_42/rooms/room_000/room_result.json").read_text()
    )
    assert first["status"] == "stage_a_schema_invalid"
    assert first["error_type"] == "StrictJSONError"


def test_stage_a_receives_only_room_local_model_brief(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared: dict[str, Any] = {}
    _run_fake_campaign(tmp_path, monkeypatch, shared=shared)
    assert len(shared["request_bodies"]) == 4
    for request_body in shared["request_bodies"][::2]:
        text = request_body.decode("utf-8")
        for forbidden in (
            "floor_plan_sha256",
            "generation_index",
            "local_to_global_offset_m",
            "layout_id",
        ):
            assert forbidden not in text


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.pop("expected_room_ids"),
        lambda value: value["rooms"].__setitem__(1, deepcopy(value["rooms"][0])),
        lambda value: value["rooms"][0].pop("projection_path"),
        lambda value: value["rooms"][0].update(
            projection_path="../../outside.json"
        ),
        lambda value: value["rooms"][0].update(projection_hash="0" * 64),
    ],
)
def test_evaluation_index_fails_closed_on_coverage_path_and_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Any,
) -> None:
    (_, _, _), _, _, output = _run_fake_campaign(tmp_path, monkeypatch)
    layout_root = output / "layout_42"
    value = json.loads(
        (layout_root / "room_evaluation_index.json").read_text(encoding="utf-8")
    )
    mutator(value)
    with pytest.raises(AssemblyError):
        validate_evaluation_index(value, layout_root=layout_root)


def test_room_schema_failure_is_terminal_and_later_room_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared = {
        "responses": [
            _chat_response({"schema_version": "invalid"}),
            _chat_response(_object_plan("type_1")),
            _chat_response(_placement()),
        ]
    }
    (summary, stopped, _), runtime, calls, output = _run_fake_campaign(
        tmp_path, monkeypatch, shared=shared
    )
    assert stopped is False
    assert summary["complete_rooms"] == 1
    assert summary["failed_rooms"] == 1
    assert summary["projected_rooms"] == 1
    assert len(calls) == 3
    assert len(runtime.requests) == 1
    scene = json.loads(
        (output / "layout_42/assembled_multi_room_scene.json").read_text()
    )
    assert scene["assembly_status"] == "incomplete"
    index = json.loads(
        (output / "layout_42/room_evaluation_index.json").read_text()
    )
    assert [item["terminal_status"] for item in index["rooms"]] == [
        "failed",
        "succeeded",
    ]
    assert "projection_path" not in index["rooms"][0]
    assert "projection_path" in index["rooms"][1]


def test_resume_skips_hash_verified_terminal_room_without_resending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared: dict[str, Any] = {}

    class Interrupted(RuntimeError):
        pass

    def progress(event: Mapping[str, Any]) -> None:
        if event.get("event") == "room_terminal" and event.get("room_id") == "room_1":
            raise Interrupted

    with pytest.raises(Interrupted):
        _run_fake_campaign(
            tmp_path,
            monkeypatch,
            progress=progress,
            shared=shared,
        )
    assert len(shared["calls"]) == 2
    assert len(shared["preflight_calls"]) == 1
    room_root = tmp_path / "output/layout_42/rooms/room_000"
    room_result_path = room_root / "room_result.json"
    fixed_path = room_root / "fixed_instruction.json"
    original_result = room_result_path.read_bytes()
    original_fixed = fixed_path.read_bytes()
    transplanted_fixed = json.loads(original_fixed)
    transplanted_fixed["run_identity"]["run_input_fingerprint_sha256"] = "0" * 64
    fixed_path.write_text(
        json.dumps(transplanted_fixed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    transplanted_result = json.loads(original_result)
    transplanted_result["run_input_fingerprint_sha256"] = "0" * 64
    transplanted_result["artifact_hashes"]["fixed_instruction.json"] = sha256_file(
        fixed_path
    )
    room_result_path.write_text(
        json.dumps(transplanted_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="immutable identity|fixed run identity"):
        _run_fake_campaign(
            tmp_path,
            monkeypatch,
            resume=True,
            shared=shared,
        )
    assert len(shared["preflight_calls"]) == 1
    fixed_path.write_bytes(original_fixed)
    room_result_path.write_bytes(original_result)
    execution_path = tmp_path / "output/execution_policy.json"
    original_execution = execution_path.read_bytes()
    altered_execution = json.loads(original_execution)
    altered_execution["continue_after_terminal_room"] = False
    execution_path.write_text(
        json.dumps(altered_execution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="execution policy hash mismatch"):
        _run_fake_campaign(
            tmp_path,
            monkeypatch,
            resume=True,
            shared=shared,
        )
    assert len(shared["preflight_calls"]) == 1
    execution_path.write_bytes(original_execution)
    shared["binding"].write_text(
        json.dumps(
            {
                "schema_version": "generation_route_bindings_v2",
                "bindings": {
                    "api2-chat-top-level-reasoning-v1": {
                        "endpoint": "https://changed.example.invalid/v1/chat/completions",
                        "credential_env": "PRIVATE_MULTI_ROOM_KEY",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="route-binding identity mismatch"):
        _run_fake_campaign(
            tmp_path,
            monkeypatch,
            resume=True,
            shared=shared,
        )
    assert len(shared["preflight_calls"]) == 1
    _binding(shared["binding"])
    original_result = room_result_path.read_bytes()
    altered_result = json.loads(original_result)
    altered_result.update(
        status="stage_c_failed",
        eligible_for_room_projection=False,
        reason_code="altered",
    )
    room_result_path.write_text(
        json.dumps(altered_result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="status|terminal artifacts"):
        _run_fake_campaign(
            tmp_path,
            monkeypatch,
            resume=True,
            shared=shared,
        )
    assert len(shared["preflight_calls"]) == 1
    room_result_path.write_bytes(original_result)
    artifact = tmp_path / "output/layout_42/rooms/room_000/object_plan.json"
    original = artifact.read_bytes()
    artifact.write_bytes(b"{}")
    with pytest.raises(Exception, match="hash.*mismatch"):
        _run_fake_campaign(
            tmp_path,
            monkeypatch,
            resume=True,
            shared=shared,
        )
    assert len(shared["calls"]) == 2
    assert len(shared["preflight_calls"]) == 1
    artifact.write_bytes(original)
    (summary, _, _), runtime, calls, output = _run_fake_campaign(
        tmp_path,
        monkeypatch,
        resume=True,
        shared=shared,
    )
    assert summary["complete_rooms"] == 2
    assert len(calls) == 4
    assert len(runtime.requests) == 2
    assert len(shared["preflight_calls"]) == 2
    assert (output / "summary.json").is_file()
    with pytest.raises(FileExistsError):
        _run_fake_campaign(
            tmp_path,
            monkeypatch,
            resume=False,
            shared=shared,
        )
    assert len(shared["preflight_calls"]) == 2
    with pytest.raises(Exception, match="terminal.*cannot be resumed|cannot be resumed"):
        _run_fake_campaign(
            tmp_path,
            monkeypatch,
            resume=True,
            shared=shared,
        )


def test_resume_completes_hash_matching_partial_finalization_without_resending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared: dict[str, Any] = {"fail_write_name": "compiled_architecture.json"}
    with pytest.raises(RuntimeError, match="injected finalization crash"):
        _run_fake_campaign(tmp_path, monkeypatch, shared=shared)
    output = tmp_path / "output"
    assert (output / "layout_42/compiled_architecture.json").is_file()
    assert not (output / "summary.json").exists()
    assert len(shared["calls"]) == 4
    assert len(shared["preflight_calls"]) == 1
    shared["omit_credential"] = True
    (summary, stopped, _), _, calls, _ = _run_fake_campaign(
        tmp_path,
        monkeypatch,
        resume=True,
        shared=shared,
    )
    assert stopped is False
    assert summary["complete_rooms"] == 2
    assert len(calls) == 4
    assert len(shared["preflight_calls"]) == 1
    assert (output / "summary.json").is_file()
