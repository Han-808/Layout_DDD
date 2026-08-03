from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.api import submission as submission_module
from benchmark.io_contracts import O3_SCENE_PACKAGE
from benchmark.materialization import (
    MaterializationError,
    NativeRegistryAuthority,
    validate_public_native_instance_mapping,
    verify_prepared_submission,
)
from benchmark.materialization import preparation as preparation_module
from benchmark.utils.io import read_json, write_json


def _mapping() -> dict:
    return {
        "schema_version": "public_native_instance_mapping_v1",
        "instances": [
            {
                "instance_id": "chair-left",
                "asset_id": "asset",
                "native_root_name": "Chair.Left",
                "center_m": [1.0, 1.5, 0.5],
                "uniform_scale": 1.25,
                "rotation_euler_xyz_deg": [0.0, 0.0, 30.0],
                "slot_id": "seat_slot",
            }
        ],
    }


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        case_id="public-native",
        manifest_sha256="b" * 64,
        catalog_snapshot_id="catalog_v1",
        allowed_asset_ids=("asset",),
        evaluator_output_type=O3_SCENE_PACKAGE,
        scene_request={
            "request_id": "public-native",
            "scene_type": "room",
            "instruction": "Place one chair.",
            "room": {
                "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                "height": 3.0,
            },
        },
    )


def _catalog(tmp_path: Path) -> tuple[Path, Path]:
    asset_root = tmp_path / "assets"
    asset_dir = asset_root / "asset"
    asset_dir.mkdir(parents=True)
    (asset_dir / "asset.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    asset_csv = tmp_path / "assets.csv"
    asset_csv.write_text(
        "name_en,category,class_en,retrieval_class_en,caption_en,"
        "short_desc,bbx\n"
        'asset,chair,chair,chair,a chair,a chair,"[1,1,1]"\n',
        encoding="utf-8",
    )
    return asset_root, asset_csv


def test_public_native_mapping_has_no_trusted_or_derived_fields() -> None:
    validated = validate_public_native_instance_mapping(_mapping())
    assert validated["instances"][0]["uniform_scale"] == 1.25

    for forbidden in (
        "authority",
        "source_blend_sha256",
        "geometry_sha256",
        "material_sha256",
        "evaluator_object_id",
    ):
        invalid = _mapping()
        invalid["instances"][0][forbidden] = "x"
        with pytest.raises(
            MaterializationError,
            match="invalid fields",
        ):
            validate_public_native_instance_mapping(invalid)

    duplicate = _mapping()
    duplicate["instances"].append(
        {
            **deepcopy(duplicate["instances"][0]),
            "native_root_name": "Chair.Right",
        }
    )
    with pytest.raises(MaterializationError, match="duplicate instance_id"):
        validate_public_native_instance_mapping(duplicate)


def test_public_mapping_inspection_hash_must_bind_registry_sealing(
    tmp_path: Path,
) -> None:
    mapping_path = write_json(tmp_path / "mapping.json", _mapping())
    mapping_hash = hashlib.sha256(mapping_path.read_bytes()).hexdigest()

    with pytest.raises(
        MaterializationError,
        match="changed between trusted inspection and registry sealing",
    ):
        preparation_module._reload_public_mapping_after_inspection(
            mapping_path,
            expected_sha256=mapping_hash,
            inspection={
                "expected_registry_sha256_before": "f" * 64,
                "expected_registry_sha256_after": "f" * 64,
            },
        )


def test_prepare_submission_derives_and_seals_public_native_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_blend = tmp_path / "submission.blend"
    source_blend.write_bytes(b"read-only submitted blend")
    source_hash = hashlib.sha256(source_blend.read_bytes()).hexdigest()
    mapping_path = write_json(tmp_path / "mapping.json", _mapping())
    mapping_hash = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    asset_root, asset_csv = _catalog(tmp_path)
    bundle = _bundle()
    authority = NativeRegistryAuthority.from_secret(
        key_id="benchmark-public-native-test",
        secret=b"public-native-benchmark-secret-000001",
    )
    inspection_calls: list[dict] = []

    def fake_inspect_native_source(**kwargs):
        inspection_calls.append(kwargs)
        instance = _mapping()["instances"][0]
        observed = {
            "instance_id": instance["instance_id"],
            "asset_id": instance["asset_id"],
            "root_object_name": instance["native_root_name"],
            "center_m": instance["center_m"],
            "uniform_scale": instance["uniform_scale"],
            "rotation_euler_xyz_deg": instance[
                "rotation_euler_xyz_deg"
            ],
            "slot_id": instance["slot_id"],
            "geometry_sha256": "a" * 64,
            "material_sha256": "c" * 64,
            "asset_assembly_sha256": "d" * 64,
        }
        return (
            {
                "schema_version": "catalog_placement_v1",
                "instances": [
                    {
                        "instance_id": instance["instance_id"],
                        "asset_id": instance["asset_id"],
                        "center_m": instance["center_m"],
                        "uniform_scale": instance["uniform_scale"],
                        "rotation_euler_xyz_deg": instance[
                            "rotation_euler_xyz_deg"
                        ],
                        "slot_id": instance["slot_id"],
                    }
                ],
            },
            {
                "status": "passed",
                "reason_codes": [],
                "inspection_mode": "public_native",
                "instances": [observed],
                "instance_fingerprints": [observed],
                "source_sha256_before": source_hash,
                "source_sha256_after": source_hash,
                "source_modified": False,
                "auto_execution_disabled": True,
                "source_scene_saved": False,
                "expected_registry_sha256_before": mapping_hash,
                "expected_registry_sha256_after": mapping_hash,
            },
        )

    def fake_materialize_catalog_scene(
        *,
        plan_path,
        out_blend_path,
        inspection_path,
        blender_bin,
        timeout_seconds,
    ):
        del blender_bin, timeout_seconds
        plan = read_json(plan_path)
        Path(out_blend_path).write_bytes(b"sanitized benchmark blend")
        rows = []
        for item in plan["instances"]:
            rows.append(
                {
                    **deepcopy(item),
                    "root_object_name": (
                        f"benchmark_instance_{item['instance_id']}"
                    ),
                    "render_enabled": True,
                    "geometry_sha256": "e" * 64,
                    "material_sha256": "f" * 64,
                    "asset_assembly_sha256": "d" * 64,
                }
            )
        inspection = {
            "status": "passed",
            "instances": rows,
            "technical_state": {
                "all_instances_render_enabled": True,
                "extra_renderable_instance_count": 0,
            },
        }
        write_json(inspection_path, inspection)
        return inspection

    monkeypatch.setattr(
        preparation_module,
        "_inspect_native_source",
        fake_inspect_native_source,
    )
    monkeypatch.setattr(
        "benchmark.materialization.blender.materialize_catalog_scene",
        fake_materialize_catalog_scene,
    )
    monkeypatch.setattr(
        submission_module,
        "load_case_bundle",
        lambda _: bundle,
    )

    result = submission_module.prepare_submission(
        artifact=source_blend,
        case_bundle=tmp_path / "case_bundle",
        out_dir=tmp_path / "prepared",
        asset_root=asset_root,
        asset_csv=asset_csv,
        blender_bin=tmp_path / "unused-blender",
        native_instance_mapping_path=mapping_path,
        native_registry_authority=authority,
    )

    assert read_json(result.readiness_report_path)["status"] == "ready"
    assert verify_prepared_submission(
        result,
        case_bundle=bundle,
    )["status"] == "ready"
    assert source_blend.read_bytes() == b"read-only submitted blend"
    assert inspection_calls[0]["inspection_mode"] == "public_native"
    derived_path = (
        tmp_path
        / "prepared"
        / "benchmark_derived_native_registry.json"
    )
    derived = read_json(derived_path)
    authority.verify(derived)
    assert derived["source_blend_sha256"] == result.hashes[
        "source_artifact_sha256"
    ]
    assert derived["instances"][0]["geometry_sha256"] == "a" * 64
    assert derived["instances"][0]["material_sha256"] == "c" * 64
    assert derived["instances"][0]["slot_id"] == "seat_slot"
    assert "geometry_sha256" not in read_json(mapping_path)["instances"][0]
    provenance = read_json(result.provenance_path)
    assert provenance["native_registry"]["origin"] == (
        "benchmark_derived_from_public_native_mapping"
    )
    assert provenance["public_native_mapping"]["sha256"] == mapping_hash
    preserved_mapping = (
        tmp_path
        / "prepared"
        / "public_native_instance_mapping.json"
    )
    assert provenance["public_native_mapping"]["path"] == (
        preserved_mapping.resolve().as_posix()
    )
    assert provenance["public_native_mapping"]["source_path"] == (
        mapping_path.resolve().as_posix()
    )
    assert preserved_mapping.read_bytes() == mapping_path.read_bytes()
    assert result.hashes["native_instance_mapping_sha256"] == mapping_hash
    assert result.trusted_render_source_path != source_blend


def test_evaluate_artifact_submission_threads_public_mapping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    mapping_path = tmp_path / "mapping.json"
    prepared = object()
    calls: list[tuple[str, object]] = []

    def fake_prepare_submission(**kwargs):
        calls.append(
            ("prepare", kwargs.get("native_instance_mapping_path"))
        )
        return prepared

    def fake_evaluate_prepared_submission(**kwargs):
        assert kwargs["prepared_submission"] is prepared
        calls.append(
            ("evaluate", kwargs.get("native_instance_mapping_path"))
        )
        return {"status": "complete"}

    monkeypatch.setattr(
        submission_module,
        "prepare_submission",
        fake_prepare_submission,
    )
    monkeypatch.setattr(
        submission_module,
        "evaluate_prepared_submission",
        fake_evaluate_prepared_submission,
    )

    report = submission_module.evaluate_artifact_submission(
        artifact=tmp_path / "scene.blend",
        case_bundle=tmp_path / "case",
        out_dir=tmp_path / "output",
        asset_root=tmp_path / "assets",
        asset_csv=tmp_path / "assets.csv",
        blender_bin=tmp_path / "blender",
        native_instance_mapping_path=mapping_path,
    )

    assert report == {"status": "complete"}
    assert calls == [
        ("prepare", mapping_path),
        ("evaluate", None),
    ]
