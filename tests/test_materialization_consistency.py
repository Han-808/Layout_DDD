from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.materialization.consistency import run_consistency_gate
from benchmark.materialization.catalog import FrozenCatalog
from benchmark.materialization.geometry import uniform_fit, world_bounds
from benchmark.materialization.contracts import MaterializationError
from benchmark.materialization.native_registry import (
    NativeRegistryAuthority,
    write_benchmark_native_registry,
)
from benchmark.materialization.preparation import prepare_catalog_submission
from benchmark.io_contracts import O3_SCENE_PACKAGE
from benchmark.nl_scene.generation_input import (
    build_generation_input,
    build_scene_request,
)
from benchmark.utils.io import read_json, write_json


def _representations() -> tuple[dict, dict, dict, dict, dict[str, str]]:
    center = [1.0, 2.0, 0.75]
    rotation = [10.0, 20.0, 30.0]
    target = [1.0, 2.0, 3.0]
    fit = uniform_fit([2.0, 2.0, 2.0], target)
    bounds = world_bounds(center, fit["local_bbox_size_m"], rotation)
    expected = {
        "instance_id": "chair_left",
        "evaluator_object_id": "chair_left",
        "asset_id": "asset_chair",
        "slot_id": "chair_slot",
        "center_m": center,
        "target_size_m": target,
        "rotation_euler_xyz_deg": rotation,
        "uniform_scale": fit["uniform_scale"],
        "local_bbox_size_m": fit["local_bbox_size_m"],
        "world_bounds": bounds,
    }
    plan = {"instances": [deepcopy(expected)]}
    registry = {
        "instances": [
            {
                "instance_id": expected["instance_id"],
                "evaluator_object_id": expected["evaluator_object_id"],
                "asset_id": expected["asset_id"],
                "slot_id": expected["slot_id"],
                "transform": {
                    "center_m": center,
                    "target_size_m": target,
                    "rotation_euler_xyz_deg": rotation,
                    "uniform_scale": fit["uniform_scale"],
                },
                "local_bbox": {"size_m": fit["local_bbox_size_m"]},
                "world_bounds": bounds,
            }
        ]
    }
    scene = {
        "objects": [
            {
                "id": expected["evaluator_object_id"],
                "asset_ref": {"asset_key": expected["asset_id"]},
                "center": center,
                "rotation": rotation,
                "size": fit["local_bbox_size_m"],
                "metadata": {
                    "materialization": {
                        "instance_id": expected["instance_id"],
                        "slot_id": expected["slot_id"],
                        "target_size_m": target,
                        "uniform_scale": fit["uniform_scale"],
                        "world_bounds": bounds,
                    }
                },
            }
        ]
    }
    inspection = {
        "instances": [deepcopy(expected)],
        "technical_state": {
            "all_instances_render_enabled": True,
            "extra_renderable_instance_count": 0,
        },
    }
    hashes = {
        key: "a" * 64
        for key in (
            "source_artifact_sha256",
            "normalized_scene_sha256",
            "instance_registry_sha256",
            "trusted_render_source_sha256",
            "materialization_plan_sha256",
            "trusted_blend_inspection_sha256",
            "provenance_core_sha256",
            "adapter_contract_revision_sha256",
        )
    }
    return plan, registry, scene, inspection, hashes


def test_consistency_gate_accepts_exact_four_way_identity_and_geometry() -> None:
    plan, registry, scene, inspection, hashes = _representations()

    report = run_consistency_gate(
        plan=plan,
        normalized_scene=scene,
        instance_registry=registry,
        blend_inspection=inspection,
        hashes=hashes,
    )

    assert report["status"] == "passed"
    assert report["mismatches"] == []


@pytest.mark.parametrize("representation", ["registry", "scene", "trusted_blend"])
def test_consistency_gate_rejects_a_difference_in_every_trusted_representation(
    representation: str,
) -> None:
    plan, registry, scene, inspection, hashes = _representations()
    if representation == "registry":
        registry["instances"][0]["transform"]["center_m"][0] += 0.25
    elif representation == "scene":
        scene["objects"][0]["size"][1] += 0.25
    else:
        inspection["instances"][0]["asset_id"] = "different_asset"

    report = run_consistency_gate(
        plan=plan,
        normalized_scene=scene,
        instance_registry=registry,
        blend_inspection=inspection,
        hashes=hashes,
    )

    assert report["status"] == "failed"
    assert any(
        mismatch["code"] == "representation_mismatch"
        for mismatch in report["mismatches"]
    )


def test_consistency_gate_requires_all_boundary_hashes() -> None:
    plan, registry, scene, inspection, hashes = _representations()
    hashes.pop("source_artifact_sha256")

    report = run_consistency_gate(
        plan=plan,
        normalized_scene=scene,
        instance_registry=registry,
        blend_inspection=inspection,
        hashes=hashes,
    )

    assert report["status"] == "failed"
    assert report["checks"]["hash_coverage"]["missing"] == [
        "source_artifact_sha256"
    ]


def test_malformed_raw_generator_artifact_is_preserved_before_failure(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "malformed.json"
    raw = b'{"instances": [this is not valid JSON]}\n'
    artifact.write_bytes(raw)
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    asset_csv = tmp_path / "assets.csv"
    asset_csv.write_text(
        "name_en,category,class_en,retrieval_class_en,caption_en,short_desc,bbx\n"
        'asset,chair,chair,chair,a chair,a chair,"[1,1,1]"\n',
        encoding="utf-8",
    )
    bundle = SimpleNamespace(
        case_id="raw_failure",
        manifest_sha256="d" * 64,
        catalog_snapshot_id="catalog_v1",
        allowed_asset_ids=("asset",),
        scene_request={
            "request_id": "raw_failure",
            "scene_type": "room",
            "room": {
                "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                "height": 3.0,
            },
        },
    )

    result = prepare_catalog_submission(
        artifact=artifact,
        case_bundle=bundle,
        out_dir=tmp_path / "prepared",
        asset_root=asset_root,
        asset_csv=asset_csv,
        blender_bin=tmp_path / "unused_blender",
    )

    preserved = tmp_path / "prepared" / "raw_generator_artifact.json"
    assert preserved.read_bytes() == raw
    assert result.hashes["source_artifact_sha256"]
    assert read_json(result.provenance_path)["source"]["preserved_path"] == (
        preserved.as_posix()
    )
    assert read_json(result.readiness_report_path)["status"] == "not_evaluable"


def test_preparation_rejects_slots_without_generator_visible_input(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    asset_csv = tmp_path / "assets.csv"
    asset_csv.write_text(
        "name_en,category,class_en,retrieval_class_en,caption_en,short_desc,bbx\n"
        'asset,chair,chair,chair,a chair,a chair,"[1,1,1]"\n',
        encoding="utf-8",
    )
    bundle = SimpleNamespace(
        case_id="slot_failure",
        manifest_sha256="e" * 64,
        catalog_snapshot_id="catalog_v1",
        allowed_asset_ids=("asset",),
        scene_request={
            "request_id": "slot_failure",
            "scene_type": "room",
            "room": {
                "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                "height": 3.0,
            },
        },
    )

    result = prepare_catalog_submission(
        artifact={
            "schema_version": "catalog_placement_v1",
            "instances": [
                {
                    "instance_id": "chair",
                    "asset_id": "asset",
                    "center_m": [1.0, 1.0, 0.5],
                    "target_size_m": [1.0, 1.0, 1.0],
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    "slot_id": "private_slot",
                }
            ],
        },
        case_bundle=bundle,
        out_dir=tmp_path / "prepared",
        asset_root=asset_root,
        asset_csv=asset_csv,
        blender_bin=tmp_path / "unused_blender",
        public_slot_ids={"private_slot"},
    )

    readiness = read_json(result.readiness_report_path)
    provenance = read_json(result.provenance_path)
    assert readiness["status"] == "not_evaluable"
    assert "generator-visible structured input" in provenance["failure"]["message"]
    assert not result.trusted_render_source_path.exists()


def test_preparation_rejects_allowlisted_asset_not_visible_to_generator(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    selected_dir = asset_root / "selected_asset"
    selected_dir.mkdir(parents=True)
    (selected_dir / "selected_asset.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    asset_csv = tmp_path / "assets.csv"
    asset_csv.write_text(
        "name_en,category,class_en,retrieval_class_en,caption_en,short_desc,bbx\n"
        'selected_asset,chair,chair,chair,a chair,a chair,"[1,1,1]"\n'
        'unselected_asset,table,table,table,a table,a table,"[1,1,1]"\n',
        encoding="utf-8",
    )
    scene_request = build_scene_request(
        request_id="visible_catalog",
        instruction="Place the selected chair.",
        scene_type="room",
        room={
            "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
            "height": 3.0,
            "unit": "meter",
        },
        structure=True,
    )
    generation_input = build_generation_input(
        scene_request=scene_request,
        object_plan={
            "request_id": "visible_catalog",
            "scene_type": "room",
            "scene_description": "one chair",
            "objects": [
                {
                    "id": "chair_slot",
                    "role": "seating",
                    "category": "chair",
                    "description": "selected chair",
                    "count": 1,
                    "placement_intent": {
                        "absolute_relations": [],
                        "relative_relations": [],
                    },
                    "metadata": {},
                }
            ],
            "global_constraints": [],
            "relations": [],
        },
        asset_selection={
            "request_id": "visible_catalog",
            "objects": [
                {
                    "object_id": "chair_slot",
                    "object_spec": {
                        "role": "seating",
                        "category": "chair",
                        "description": "selected chair",
                        "estimated_size": [1.0, 1.0, 1.0],
                        "count": 1,
                    },
                    "retrieval_query": {
                        "description": "selected chair",
                        "category": "chair",
                        "size_constraint": [1.0, 1.0, 1.0],
                    },
                    "selected_asset": {
                        "jid": "selected_asset",
                        "category": "chair",
                        "retrieval_category": "chair",
                        "desc": "a selected frozen chair",
                        "short_desc": "selected chair",
                        "size": [1.0, 1.0, 1.0],
                        "asset_ref": {
                            "source_db": "frozen_test_catalog",
                            "asset_key": "selected_asset",
                            "mesh_uri": None,
                            "pointcloud_uri": None,
                            "metadata_uri": None,
                        },
                        "asset_proxy": {
                            "type": "canonical_catalog_bbox",
                            "bbox_center_local": [0.0, 0.0, 0.0],
                            "bbox_size": [1.0, 1.0, 1.0],
                        },
                        "metadata": {},
                    },
                    "candidates": [],
                    "selection_action": "select",
                    "selection_decision": {
                        "action": "select",
                        "selected_jid": "selected_asset",
                        "reason": "fixture",
                        "generation_request": None,
                    },
                    "selection_reason": "fixture",
                }
            ],
        },
        evaluator_output_type=O3_SCENE_PACKAGE,
    )
    bundle = SimpleNamespace(
        case_id="visible_catalog",
        manifest_sha256="7" * 64,
        catalog_snapshot_id="catalog_v1",
        allowed_asset_ids=("selected_asset", "unselected_asset"),
        scene_request=scene_request,
    )

    result = prepare_catalog_submission(
        artifact={
            "schema_version": "catalog_placement_v1",
            "instances": [
                {
                    "instance_id": "table",
                    "asset_id": "unselected_asset",
                    "center_m": [1.0, 1.0, 0.5],
                    "target_size_m": [1.0, 1.0, 1.0],
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                }
            ],
        },
        case_bundle=bundle,
        out_dir=tmp_path / "prepared",
        asset_root=asset_root,
        asset_csv=asset_csv,
        blender_bin=tmp_path / "must_not_run",
        generation_input=generation_input,
    )

    readiness = read_json(result.readiness_report_path)
    provenance = read_json(result.provenance_path)
    assert readiness["status"] == "not_evaluable"
    assert "not in the exact generator-visible selected" in (
        provenance["failure"]["message"]
    )
    assert provenance["generator_visible_input"]["selected_asset_ids"] == [
        "selected_asset"
    ]
    assert not result.trusted_render_source_path.exists()


def test_preparation_rejects_unbound_native_registry_before_blender(
    tmp_path: Path,
) -> None:
    source_blend = tmp_path / "submitted.blend"
    source_blend.write_bytes(b"not opened because registry authority fails")
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    asset_csv = tmp_path / "assets.csv"
    asset_csv.write_text(
        "name_en,category,class_en,retrieval_class_en,caption_en,short_desc,bbx\n"
        'asset,chair,chair,chair,a chair,a chair,"[1,1,1]"\n',
        encoding="utf-8",
    )
    registry = write_json(
        tmp_path / "untrusted_registry.json",
        {
            "schema_version": "benchmark_owned_native_registry_v1",
            "instances": [],
        },
    )
    bundle = SimpleNamespace(
        case_id="native_registry_failure",
        manifest_sha256="f" * 64,
        catalog_snapshot_id="catalog_v1",
        allowed_asset_ids=("asset",),
        scene_request={
            "request_id": "native_registry_failure",
            "scene_type": "room",
            "room": {
                "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                "height": 3.0,
            },
        },
    )

    result = prepare_catalog_submission(
        artifact=source_blend,
        case_bundle=bundle,
        out_dir=tmp_path / "prepared",
        asset_root=asset_root,
        asset_csv=asset_csv,
        blender_bin=tmp_path / "must_not_run",
        native_registry_path=registry,
        native_registry_authority=NativeRegistryAuthority.from_secret(
            key_id="test-authority",
            secret=b"catalog-native-registry-test-secret-0002",
        ),
    )

    readiness = read_json(result.readiness_report_path)
    provenance = read_json(result.provenance_path)
    assert readiness["status"] == "not_evaluable"
    assert "invalid root fields" in provenance["failure"]["message"]
    assert not result.trusted_render_source_path.exists()


def test_frozen_catalog_hash_binds_mesh_dependency_tree(tmp_path: Path) -> None:
    asset_root = tmp_path / "assets"
    asset_dir = asset_root / "asset"
    asset_dir.mkdir(parents=True)
    (asset_dir / "asset.obj").write_text(
        "mtllib asset.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    material = asset_dir / "asset.mtl"
    material.write_text("newmtl frozen\nKd 0.1 0.2 0.3\n", encoding="utf-8")
    asset_csv = tmp_path / "assets.csv"
    asset_csv.write_text(
        "name_en,category,class_en,retrieval_class_en,caption_en,short_desc,bbx\n"
        'asset,chair,chair,chair,a chair,a chair,"[1,1,1]"\n',
        encoding="utf-8",
    )
    catalog = FrozenCatalog(
        asset_csv=asset_csv,
        asset_root=asset_root,
        allowed_asset_ids=("asset",),
        snapshot_id="catalog_v1",
    )
    before = catalog.resolve("asset").hashes["asset_tree_sha256"]

    material.write_text("newmtl frozen\nKd 0.9 0.2 0.3\n", encoding="utf-8")
    after = catalog.resolve("asset").hashes["asset_tree_sha256"]

    assert after != before


def test_native_registry_authority_seal_rejects_self_authored_tampering(
    tmp_path: Path,
) -> None:
    source = tmp_path / "native.blend"
    source.write_bytes(b"native placement")
    authority = NativeRegistryAuthority.from_secret(
        key_id="benchmark-placement-tool-test",
        secret=b"catalog-native-registry-test-secret-0003",
    )
    registry_path = write_benchmark_native_registry(
        tmp_path / "native_registry.json",
        authority=authority,
        source_blend_path=source,
        case_bundle_manifest_sha256="a" * 64,
        catalog_snapshot_id="catalog_v1",
        instances=[
            {
                "instance_id": "chair",
                "evaluator_object_id": "chair",
                "asset_id": "asset",
                "native_root_name": "benchmark_instance_chair",
                "center_m": [1.0, 1.0, 0.5],
                "target_size_m": [1.0, 1.0, 1.0],
                "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                "geometry_sha256": "b" * 64,
                "material_sha256": "c" * 64,
            }
        ],
    )
    sealed = read_json(registry_path)
    authority.verify(sealed)
    assert sealed["authority"]["key_id"] == (
        "benchmark-placement-tool-test"
    )

    sealed["instances"][0]["center_m"][0] = 2.0
    with pytest.raises(MaterializationError, match="seal is invalid"):
        authority.verify(sealed)


def test_non_finite_in_memory_artifact_is_preserved_before_rejection(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    asset_csv = tmp_path / "assets.csv"
    asset_csv.write_text(
        "name_en,category,class_en,retrieval_class_en,caption_en,short_desc,bbx\n"
        'asset,chair,chair,chair,a chair,a chair,"[1,1,1]"\n',
        encoding="utf-8",
    )
    bundle = SimpleNamespace(
        case_id="non_finite",
        manifest_sha256="9" * 64,
        catalog_snapshot_id="catalog_v1",
        allowed_asset_ids=("asset",),
        scene_request={
            "request_id": "non_finite",
            "scene_type": "room",
            "room": {
                "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                "height": 3.0,
            },
        },
    )
    result = prepare_catalog_submission(
        artifact={
            "instances": [
                {
                    "instance_id": "chair",
                    "asset_id": "asset",
                    "center_m": [float("nan"), 1.0, 0.5],
                    "target_size_m": [1.0, 1.0, 1.0],
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                }
            ]
        },
        case_bundle=bundle,
        out_dir=tmp_path / "prepared",
        asset_root=asset_root,
        asset_csv=asset_csv,
        blender_bin=tmp_path / "must_not_run",
    )

    raw = tmp_path / "prepared" / "raw_generator_artifact.json"
    assert raw.is_file()
    assert "NaN" in raw.read_text(encoding="utf-8")
    assert result.hashes["source_artifact_sha256"]
    assert read_json(result.readiness_report_path)["status"] == "not_evaluable"
