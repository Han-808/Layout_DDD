from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from benchmark.generation_comparison.backends import load_3d_future_subset_catalog
from benchmark.generation_comparison.catalog import (
    CanonicalAssetCatalog,
    converter_asset_manifest,
)
from benchmark.generation_comparison.eligibility import check_method_eligibility
from benchmark.generation_comparison.execution import (
    ComparisonRunError,
    run_controlled_generation,
)
from benchmark.generation_comparison.identity import (
    architecture_sha256,
)
from benchmark.generation_comparison.materializers import materialize_method_catalog
from benchmark.generation_comparison.protocol import ComparisonProtocol
from benchmark.generation_comparison.validation import validate_comparison_run
from benchmark.io_contracts import O1_OBJECT_STATE
from benchmark.nl_scene.generation_input import (
    build_direct_natural_language_generation_input,
    build_generator_visible_payload,
)
from benchmark.utils.io import read_json, write_json


METHODS = (
    "layout_gpt",
    "direct_layout",
    "layout_vlm",
    "respace",
    "scene_weaver",
)
ALL_CONTROLS = {
    "shared_catalog": True,
    "fixed_object_inventory": True,
    "exact_asset_ids": True,
    "fixed_native_scale": True,
    "frozen_iteration_bindings": True,
    "no_object_insertion_removal": True,
}
PRIVATE_SENTINEL = "PRIVATE_EVALUATOR_SENTINEL"


def test_canonical_catalog_hash_is_stable_order_independent_and_immutable() -> None:
    first = _catalog_mapping()
    second = deepcopy(first)
    second["assets"] = list(reversed(second["assets"]))
    one = CanonicalAssetCatalog.from_mapping(first)
    two = CanonicalAssetCatalog.from_mapping(second)

    assert one.sha256 == two.sha256
    assert one.as_dict() == two.as_dict()
    first["assets"][0]["category"] = "mutated"
    assert one.get("chair.asset.001")["category"] == "chair"


def test_supplied_catalog_hash_mismatch_fails_closed() -> None:
    payload = _catalog_mapping()
    payload["catalog_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="catalog_sha256"):
        CanonicalAssetCatalog.from_mapping(payload)


def test_catalog_rejects_conflated_local_bbox_scale_and_physical_dimensions() -> None:
    payload = _catalog_mapping()
    payload["assets"][0]["physical_dimensions"] = [8.0, 7.0, 10.0]
    with pytest.raises(ValueError, match=r"bbox_size_local \* native_scale"):
        CanonicalAssetCatalog.from_mapping(payload)


def test_3d_future_pilot_backend_normalizes_only_frozen_records(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "chair.obj"
    mesh.write_text("o chair\n", encoding="utf-8")
    source = write_json(
        tmp_path / "subset.json",
        {
            "assets": [
                {
                    "jid": "future-chair",
                    "category": "chair",
                    "description": "pilot chair",
                    "bbox_size": [0.8, 0.7, 1.0],
                    "bbox_center_local": [0.0, 0.0, 0.5],
                    "mesh_path": mesh.name,
                    "canonical_front": [0.0, -1.0, 0.0],
                }
            ]
        },
    )
    catalog = load_3d_future_subset_catalog(
        source,
        catalog_id="future-pilot",
        catalog_version="2026-09-test",
    )

    record = catalog.get("future-chair")
    assert record["source_db"] == "3d_future"
    assert len(record["content"]["mesh_sha256"]) == 64
    assert record["canonical_front"] == [0.0, -1.0, 0.0]


def test_catalog_logical_hash_is_independent_of_content_identical_cache_path(
    tmp_path: Path,
) -> None:
    first_mesh = tmp_path / "cache_a" / "chair.glb"
    second_mesh = tmp_path / "cache_b" / "renamed.glb"
    first_mesh.parent.mkdir()
    second_mesh.parent.mkdir()
    first_mesh.write_bytes(b"same mesh bytes")
    second_mesh.write_bytes(b"same mesh bytes")
    first = _catalog_mapping()
    second = deepcopy(first)
    first["assets"][0]["mesh_uri"] = first_mesh.as_posix()
    second["assets"][0]["mesh_uri"] = second_mesh.as_posix()
    digest = hashlib.sha256(b"same mesh bytes").hexdigest()
    first["assets"][0]["content"]["mesh_sha256"] = digest
    second["assets"][0]["content"]["mesh_sha256"] = digest

    assert CanonicalAssetCatalog.from_mapping(first).sha256 == (
        CanonicalAssetCatalog.from_mapping(second).sha256
    )


def test_local_mesh_hash_is_verified_when_requested(tmp_path: Path) -> None:
    mesh = tmp_path / "asset.glb"
    mesh.write_bytes(b"actual mesh")
    payload = _catalog_mapping()
    payload["assets"] = [deepcopy(payload["assets"][0])]
    payload["assets"][0]["mesh_uri"] = mesh.as_posix()
    payload["assets"][0]["content"] = {}
    payload["assets"][0]["content"]["mesh_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match local mesh"):
        CanonicalAssetCatalog.from_mapping(payload, hash_local_meshes=True)


@pytest.mark.parametrize("adapter_name", METHODS)
def test_same_logical_catalog_materializes_deterministically_for_methods(
    adapter_name: str,
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="frozen_assets")
    first = materialize_method_catalog(
        adapter_name=adapter_name,
        catalog=catalog,
        protocol=protocol,
        out_dir=tmp_path / "first" / adapter_name,
    )
    second = materialize_method_catalog(
        adapter_name=adapter_name,
        catalog=catalog,
        protocol=protocol,
        out_dir=tmp_path / "second" / adapter_name,
    )

    assert first["materialized_catalog_sha256"] == catalog.sha256
    assert second["materialized_catalog_sha256"] == catalog.sha256
    assert first["method_payload_sha256"] == second["method_payload_sha256"]
    assert read_json(first["catalog_path"])["catalog_sha256"] == catalog.sha256
    payload = read_json(first["method_catalog_path"])
    expected_key = {
        "layout_gpt": "dataset_asset_index",
        "direct_layout": "asset_library",
        "layout_vlm": "frozen_assets",
        "respace": "frozen_scene_objects",
        "scene_weaver": "frozen_asset_bindings",
    }[adapter_name]
    assert expected_key in payload


def test_direct_layout_materializer_builds_nonmutating_asset_library_symlink(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "source.glb"
    mesh.write_bytes(b"synthetic-mesh")
    payload = _catalog_mapping()
    payload["assets"] = [deepcopy(payload["assets"][0])]
    payload["assets"][0]["mesh_uri"] = mesh.as_posix()
    payload["assets"][0]["content"] = {}
    catalog = CanonicalAssetCatalog.from_mapping(payload, hash_local_meshes=True)
    materialized = materialize_method_catalog(
        adapter_name="direct_layout",
        catalog=catalog,
        protocol=_protocol(catalog, mode="frozen_assets"),
        out_dir=tmp_path / "direct_materialization",
    )
    asset_root = Path(materialized["method_asset_root"])
    links = list(asset_root.iterdir())
    assert len(links) == 1
    assert links[0].is_symlink()
    assert links[0].resolve() == mesh.resolve()
    assert mesh.read_bytes() == b"synthetic-mesh"
    asset = read_json(materialized["method_catalog_path"])["asset_library"][0]
    assert asset["materialized_mesh_path"].startswith(
        "direct_layout_asset_library/"
    )


def test_respace_materializer_converts_local_bbox_and_scale_axes(
    tmp_path: Path,
) -> None:
    payload = _catalog_mapping()
    payload["assets"] = [deepcopy(payload["assets"][0])]
    payload["assets"][0]["native_scale"] = [1.0, 2.0, 3.0]
    catalog = CanonicalAssetCatalog.from_mapping(payload)
    protocol = _protocol(catalog, mode="frozen_assets")
    # The helper binds the same exact ID used by the reduced catalog.
    materialized = materialize_method_catalog(
        adapter_name="respace",
        catalog=catalog,
        protocol=protocol,
        out_dir=tmp_path / "respace_materializer",
    )
    method = read_json(materialized["method_catalog_path"])
    asset = method["asset_metadata"]["chair.asset.001"]
    frozen = method["frozen_scene_objects"][0]
    assert asset["sampled_asset_size"] == [0.8, 1.0, 0.7]
    assert asset["scale"] == [1.0, 3.0, 2.0]
    assert frozen["size"] == pytest.approx([0.8, 3.0, 1.4])


def test_architecture_hash_normalizes_translation_cycle_and_winding() -> None:
    first = _architecture()
    second = {
        "room_model": "single_room",
        "boundary_model": "axis_aligned_rectangle",
        "room": {
            "boundary": [[14, 25], [10, 25], [10, 20], [14, 20]],
            "height": 3.0,
            "unit": "m",
        },
    }
    assert architecture_sha256(first) == architecture_sha256(second)


def test_arbitrary_evaluator_private_comparison_fields_are_not_projected() -> None:
    generation_input = _generation_input()
    generation_input["generation_comparison"] = {
        "protocol_id": "generation_comparison_v1",
        "evaluation_report": {"benchmark_score": 100.0},
    }
    with pytest.raises(ValueError, match="non-public fields"):
        build_generator_visible_payload(generation_input)


def test_eligibility_is_explicit_and_fail_closed() -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="frozen_assets")
    layout_vlm = check_method_eligibility(
        adapter_name="layout_vlm",
        protocol=protocol,
        catalog=catalog,
    )
    layout_gpt = check_method_eligibility(
        adapter_name="layout_gpt",
        protocol=protocol,
        catalog=catalog,
    )
    scene_weaver = check_method_eligibility(
        adapter_name="scene_weaver",
        protocol=protocol,
        catalog=catalog,
        adapter_config={"comparison_support": ALL_CONTROLS},
    )

    assert layout_vlm["status"] == "ELIGIBLE"
    assert layout_gpt["status"] == "INELIGIBLE"
    assert {item["code"] for item in layout_gpt["reasons"]} >= {
        "cannot_accept_fixed_object_inventory",
        "cannot_accept_fixed_asset_ids",
        "incompatible_scale_policy",
    }
    assert scene_weaver["status"] == "ELIGIBLE"
    for report in (layout_vlm, layout_gpt, scene_weaver):
        assert report["control_evidence"] == {
            "pre_run_basis": "capability_declarations",
            "real_upstream_smoke_test_verified": False,
            "post_run_validation_required": True,
        }


def test_shared_db_requires_explicit_runner_catalog_control() -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="shared_db")
    denied = check_method_eligibility(
        adapter_name="direct_layout",
        protocol=protocol,
        catalog=catalog,
    )
    allowed = check_method_eligibility(
        adapter_name="direct_layout",
        protocol=protocol,
        catalog=catalog,
        adapter_config={
            "comparison_support": {
                "shared_catalog": True,
                "fixed_object_inventory": True,
            }
        },
    )
    assert denied["status"] == "INELIGIBLE"
    assert allowed["status"] == "ELIGIBLE"


def test_native_protocol_runs_without_shared_catalog_or_control_attestation(
    tmp_path: Path,
) -> None:
    protocol = _protocol(_catalog(), mode="native")

    def runner(*, method_input_path: Path, out_dir: Path, config: dict) -> Path:
        method_input = read_json(method_input_path)
        assert method_input["generator_input"]["generation_comparison"]["mode"] == (
            "native"
        )
        return _native_output("direct_layout", out_dir, config)

    result = run_controlled_generation(
        generation_input=_generation_input(),
        adapter_name="direct_layout",
        protocol=protocol,
        out_dir=tmp_path / "native",
        adapter_config={"runner": runner},
    )
    assert result["protocol_mode"] == "native"
    assert result["catalog_sha256"] is None
    assert result["valid_comparison_run"] is True


def test_shared_db_execution_uses_catalog_identity_with_method_selection(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="shared_db")

    def runner(*, method_input_path: Path, out_dir: Path, config: dict) -> Path:
        comparison = read_json(method_input_path)["generator_input"][
            "generation_comparison"
        ]
        assert comparison["catalog"] == catalog.identity
        return _native_output("direct_layout", out_dir, config)

    result = run_controlled_generation(
        generation_input=_generation_input(),
        adapter_name="direct_layout",
        protocol=protocol,
        asset_catalog=catalog,
        out_dir=tmp_path / "shared",
        adapter_config={
            "runner": runner,
            "comparison_support": {
                "shared_catalog": True,
                "fixed_object_inventory": True,
            },
        },
    )
    assert result["protocol_mode"] == "shared_db"
    assert result["catalog_sha256"] == catalog.sha256
    assert result["selected_asset_ids"] == {"chair_0": "chair.asset.001"}
    assert result["observed_object_inventory_sha256"] == (
        result["object_inventory_sha256"]
    )
    assert result["asset_binding_sha256"] is None
    assert result["observed_asset_binding_sha256"] is not None
    assert result["valid_comparison_run"] is True


@pytest.mark.parametrize("adapter_name", METHODS)
def test_frozen_assets_runs_preserve_exact_controls_and_use_same_evaluator(
    adapter_name: str,
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="frozen_assets")
    seen: dict[str, Any] = {}

    def runner(*, method_input_path: Path, out_dir: Path, config: dict) -> Path:
        method_input = read_json(method_input_path)
        seen["method_input"] = method_input
        assert PRIVATE_SENTINEL not in json.dumps(method_input)
        comparison = method_input["generator_input"]["generation_comparison"]
        assert comparison["catalog"]["catalog_sha256"] == catalog.sha256
        assert comparison["mode"] == "frozen_assets"
        return _native_output(adapter_name, out_dir, config)

    config: dict[str, Any] = {
        "runner": runner,
        "comparison_support": ALL_CONTROLS,
    }
    if adapter_name == "scene_weaver":
        config["selected_iteration"] = 1
    evaluator_scene_request = deepcopy(_generation_input()["scene_request"])
    evaluator_scene_request["metadata"] = {"private_marker": PRIVATE_SENTINEL}
    result = run_controlled_generation(
        generation_input=_generation_input(),
        adapter_name=adapter_name,
        protocol=protocol,
        asset_catalog=catalog,
        out_dir=tmp_path / adapter_name,
        adapter_config=config,
        evaluation_kwargs={"scene_request": evaluator_scene_request},
    )

    assert result["status"] == "COMPLETED"
    assert result["valid_comparison_run"] is True
    assert result["catalog_sha256"] == catalog.sha256
    assert result["selected_asset_ids"] == {"chair_0": "chair.asset.001"}
    assert result["observed_object_inventory_sha256"] == (
        result["object_inventory_sha256"]
    )
    assert result["observed_asset_binding_sha256"] == result["asset_binding_sha256"]
    assert result["architecture_hashes"] == {
        "comparison_case_sha256": protocol.architecture_hash,
        "method_input_sha256": protocol.architecture_hash,
        "canonical_output_sha256": protocol.architecture_hash,
    }
    assert result["evaluator"]["entrypoint"] == (
        "benchmark.api.evaluation.run_evaluate"
    )
    assert result["evaluator"]["workflow"] == "canonical_l0_l4"
    assert Path(result["native_artifact"]).exists()
    assert len(result["native_artifact_sha256"]) == 64
    assert Path(result["execution_result"]).is_file()
    assert result["runner"]["kind"] == "callback"
    assert read_json(result["manifest_path"])["valid_comparison_run"] is True
    method_input = seen["method_input"]
    assert "reference_annotation" not in json.dumps(method_input)
    assert "evaluation_context" not in json.dumps(method_input)
    assert PRIVATE_SENTINEL not in json.dumps(method_input)
    scene = read_json(result["canonical_scene"])
    assert scene["metadata"]["harness_compatibility"][
        "asset_resolution_policy"
    ] == "exact_only"

    if adapter_name == "scene_weaver":
        trajectory = result["sceneweaver_trajectory"]
        assert trajectory["valid_comparison_trajectory"] is True
        assert trajectory["benchmark_feedback_used_by_native_loop"] is False
        assert [item["iteration"] for item in trajectory["iterations"]] == [0, 1]
        assert all(
            item["selected_asset_ids"] == {"chair_0": "chair.asset.001"}
            for item in trajectory["iterations"]
        )
        assert all(
            Path(item["native_artifact"]).is_file()
            and Path(item["canonical_scene"]).is_file()
            and item["evaluation_workflow"] == "canonical_l0_l4"
            for item in trajectory["iterations"]
        )


def test_controlled_conversion_never_invokes_semantic_retrieval(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="frozen_assets")
    calls = {"resolve": 0, "retrieve": 0}

    class Provider:
        def resolve(self, asset_key, *, source_db=None, hint=None):
            del source_db, hint
            calls["resolve"] += 1
            return converter_asset_manifest(catalog)["assets"][asset_key]

        def retrieve(self, query, *, category=None, size=None, hint=None):
            del query, category, size, hint
            calls["retrieve"] += 1
            raise AssertionError("strict controlled conversion attempted retrieval")

    def runner(*, method_input_path: Path, out_dir: Path, config: dict) -> Path:
        del method_input_path
        return _native_output("direct_layout", out_dir, config)

    result = run_controlled_generation(
        generation_input=_generation_input(),
        adapter_name="direct_layout",
        protocol=protocol,
        asset_catalog=catalog,
        out_dir=tmp_path / "no_retrieval",
        adapter_config={
            "runner": runner,
            "asset_provider": Provider(),
            "comparison_support": ALL_CONTROLS,
        },
    )
    assert result["valid_comparison_run"] is True
    assert calls == {"resolve": 1, "retrieve": 0}


def test_comparison_manifest_preserves_reported_cost_and_retrieval_provenance(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="frozen_assets")

    def runner(*, method_input_path: Path, out_dir: Path, config: dict) -> dict:
        del method_input_path
        native = _native_output("direct_layout", out_dir, config)
        return {
            "native_artifact_path": native,
            "resource_usage": {
                "model": "fake-model",
                "generation_calls": 2,
                "tokens": 123,
                "tool_calls": 3,
                "retrieval_calls": 1,
                "rendering_calls": 0,
            },
            "retrieval_provenance": {
                "query": "wood dining chair",
                "candidate_ids": ["chair.asset.001", "table.asset.001"],
                "selected_asset_id": "chair.asset.001",
            },
        }

    result = run_controlled_generation(
        generation_input=_generation_input(),
        adapter_name="direct_layout",
        protocol=protocol,
        asset_catalog=catalog,
        out_dir=tmp_path / "resource_metadata",
        adapter_config={
            "runner": runner,
            "comparison_support": ALL_CONTROLS,
        },
    )

    assert result["generation_resources"]["model"] == "fake-model"
    assert result["generation_resources"]["tokens"] == 123
    provenance = result["runner"]["source_provenance"]
    assert provenance["source_path"] == Path(__file__).resolve().as_posix()
    assert len(provenance["source_sha256"]) == 64
    assert provenance["control_verification"] == "NOT_VERIFIED"
    assert result["control_evidence"]["real_upstream_smoke_test_verified"] is False
    assert result["retrieval_selection_provenance"]["upstream_reported"][
        "candidate_ids"
    ] == ["chair.asset.001", "table.asset.001"]


def test_controlled_offline_native_artifact_mode_remains_available(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="frozen_assets")
    native = _native_output("direct_layout", tmp_path / "upstream", {})
    before = native.read_bytes()
    result = run_controlled_generation(
        generation_input=_generation_input(),
        adapter_name="direct_layout",
        protocol=protocol,
        asset_catalog=catalog,
        out_dir=tmp_path / "offline",
        method_output=native,
        run_generation=False,
        adapter_config={"comparison_support": ALL_CONTROLS},
    )

    assert result["valid_comparison_run"] is True
    assert native.read_bytes() == before
    assert Path(result["native_artifact"]).resolve() != native.resolve()


def test_current_catalog_placement_method_has_frozen_assets_offline_route(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="frozen_assets")
    native = write_json(
        tmp_path / "catalog_placement_v1.json",
        {
            "schema_version": "catalog_placement_v1",
            "instances": [
                {
                    "instance_id": "current_method_chair",
                    "asset_id": "chair.asset.001",
                    "center_m": [2.0, 2.0, 0.5],
                    "uniform_scale": 1.0,
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    "slot_id": "chair_0",
                }
            ],
        },
    )
    result = run_controlled_generation(
        generation_input=_generation_input(),
        adapter_name="catalog_placement",
        protocol=protocol,
        asset_catalog=catalog,
        out_dir=tmp_path / "catalog_placement",
        method_output=native,
        run_generation=False,
    )

    assert result["valid_comparison_run"] is True
    assert result["eligibility_status"] == "ELIGIBLE"
    assert result["selected_asset_ids"] == {"chair_0": "chair.asset.001"}
    assert result["runner"]["kind"] is None
    assert Path(result["native_artifact"]).read_bytes() == native.read_bytes()
    method_input = read_json(result["method_input"])
    assert method_input["generation_comparison"]["scale_policy"] == (
        "fixed_native_scale"
    )
    assert "fixed_native_scale" in method_input["messages"][0]["content"]
    model_messages = json.dumps(method_input["messages"])
    assert "/synthetic/chair.glb" not in model_messages
    assert method_input["generation_comparison"]["method_materialization"][
        "method_catalog_path"
    ] not in model_messages
    assert "chair.asset.001" in model_messages


def test_one_frozen_asset_can_bind_multiple_slots_without_metadata_conflict(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol_value = _protocol(catalog, mode="frozen_assets").as_dict()
    protocol_value["objects"] = [
        {
            "slot_id": f"chair_{index}",
            "category": "chair",
            "description": "wood dining chair",
            "asset_id": "chair.asset.001",
        }
        for index in range(2)
    ]
    protocol = ComparisonProtocol.from_mapping(protocol_value)
    native = write_json(
        tmp_path / "two_chairs.json",
        {
            "schema_version": "catalog_placement_v1",
            "instances": [
                {
                    "instance_id": f"chair_instance_{index}",
                    "asset_id": "chair.asset.001",
                    "center_m": [1.5 + index, 2.0, 0.5],
                    "uniform_scale": 1.0,
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    "slot_id": f"chair_{index}",
                }
                for index in range(2)
            ],
        },
    )
    result = run_controlled_generation(
        generation_input=_generation_input(),
        adapter_name="catalog_placement",
        protocol=protocol,
        asset_catalog=catalog,
        out_dir=tmp_path / "two_chairs",
        method_output=native,
        run_generation=False,
    )

    assert result["valid_comparison_run"] is True
    assert result["selected_asset_ids"] == {
        "chair_0": "chair.asset.001",
        "chair_1": "chair.asset.001",
    }


def test_catalog_placement_rejects_duplicate_native_slot_bindings(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="frozen_assets")
    native = write_json(
        tmp_path / "duplicate_slot.json",
        {
            "schema_version": "catalog_placement_v1",
            "instances": [
                {
                    "instance_id": f"chair_instance_{index}",
                    "asset_id": "chair.asset.001",
                    "center_m": [1.5 + index, 2.0, 0.5],
                    "uniform_scale": 1.0,
                    "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    "slot_id": "chair_0",
                }
                for index in range(2)
            ],
        },
    )
    with pytest.raises(ComparisonRunError, match="fairness validation failed"):
        run_controlled_generation(
            generation_input=_generation_input(),
            adapter_name="catalog_placement",
            protocol=protocol,
            asset_catalog=catalog,
            out_dir=tmp_path / "duplicate_slot_run",
            method_output=native,
            run_generation=False,
        )
    validation = read_json(
        tmp_path / "duplicate_slot_run" / "comparison" / "validation.json"
    )
    assert any(
        item["code"] == "object_inventory_mismatch"
        and "duplicate_slot_bindings" in item["details"]
        for item in validation["violations"]
    )


def test_pre_run_architecture_mismatch_is_rejected_before_runner(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="frozen_assets")
    called = False

    def runner(**kwargs: Any) -> Path:
        nonlocal called
        called = True
        raise AssertionError(kwargs)

    generation_input = _generation_input()
    generation_input["scene_request"]["room"]["boundary"] = [
        [0, 0],
        [6, 0],
        [6, 5],
        [0, 5],
    ]
    generation_input["generator_input"]["room"] = deepcopy(
        generation_input["scene_request"]["room"]
    )
    generation_input["generation_contract"]["architecture"]["room"][
        "boundary"
    ] = deepcopy(generation_input["scene_request"]["room"]["boundary"])
    generation_input["generation_contract"]["architecture"]["logical_boundary"][
        "boundary"
    ] = deepcopy(generation_input["scene_request"]["room"]["boundary"])
    with pytest.raises(ComparisonRunError, match="eligibility"):
        run_controlled_generation(
            generation_input=generation_input,
            adapter_name="layout_vlm",
            protocol=protocol,
            asset_catalog=catalog,
            out_dir=tmp_path / "mismatch",
            adapter_config={"runner": runner},
        )
    assert called is False
    manifest = read_json(tmp_path / "mismatch" / "comparison" / "run_manifest.json")
    assert manifest["status"] == "INELIGIBLE"


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("inventory", "object_inventory_mismatch"),
        ("insert", "unexpected_object_insertion"),
        ("asset", "exact_asset_replacement"),
        ("scale", "scale_policy_violation"),
        ("architecture", "architecture_mismatch"),
        ("catalog", "catalog_mismatch"),
    ],
)
def test_fairness_validation_detects_protocol_violations(
    mutation: str,
    expected_code: str,
) -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="frozen_assets")
    materialization = {
        "logical_to_native_slot": {"chair_0": "chair_0"},
        "materialized_catalog_sha256": catalog.sha256,
    }
    scene = _canonical_scene()
    native = {
        "iterations": [
            {
                "iteration": None,
                "objects": [
                    {
                        "native_object_id": "chair_0",
                        "asset_id": "chair.asset.001",
                    }
                ],
            }
        ]
    }
    if mutation == "inventory":
        scene["objects"] = []
    elif mutation == "insert":
        scene["objects"].append(
            {
                **deepcopy(scene["objects"][0]),
                "id": "extra_0",
            }
        )
    elif mutation == "asset":
        scene["objects"][0]["asset_ref"]["asset_key"] = "table.asset.001"
        scene["objects"][0]["jid"] = "table.asset.001"
        native["iterations"][0]["objects"][0]["asset_id"] = "table.asset.001"
    elif mutation == "scale":
        scene["objects"][0]["size"] = [1.6, 0.7, 1.0]
    elif mutation == "architecture":
        scene["boundary"] = [[0, 0], [6, 0], [6, 5], [0, 5]]
    elif mutation == "catalog":
        materialization["materialized_catalog_sha256"] = "0" * 64
    report = validate_comparison_run(
        adapter_name="direct_layout",
        protocol=protocol,
        catalog=catalog,
        canonical_scene=scene,
        materialization=materialization,
        native_selection=native,
        method_input_architecture_sha256=protocol.architecture_hash,
        eligibility={"eligible": True},
    )
    codes = {item["code"] for item in report["violations"]}
    assert expected_code in codes
    assert report["valid_comparison_run"] is False


def test_shared_db_accepts_method_selection_only_inside_same_catalog() -> None:
    catalog = _catalog()
    protocol = _protocol(catalog, mode="shared_db")
    materialization = {
        "logical_to_native_slot": {"chair_0": "chair_0"},
        "materialized_catalog_sha256": catalog.sha256,
    }
    scene = _canonical_scene(asset_id="table.asset.001", size=[1.2, 0.8, 0.75])
    native = {
        "iterations": [
            {
                "iteration": None,
                "objects": [
                    {"native_object_id": "chair_0", "asset_id": "table.asset.001"}
                ],
            }
        ]
    }
    report = validate_comparison_run(
        adapter_name="direct_layout",
        protocol=protocol,
        catalog=catalog,
        canonical_scene=scene,
        materialization=materialization,
        native_selection=native,
        method_input_architecture_sha256=protocol.architecture_hash,
        eligibility={"eligible": True},
    )
    assert report["valid_comparison_run"] is True
    assert report["selected_asset_ids"] == {"chair_0": "table.asset.001"}


def _catalog() -> CanonicalAssetCatalog:
    return CanonicalAssetCatalog.from_mapping(_catalog_mapping())


def _catalog_mapping() -> dict[str, Any]:
    return {
        "schema_version": "canonical_asset_catalog_v1",
        "catalog_id": "synthetic-furniture-v1",
        "catalog_version": "1",
        "assets": [
            {
                "asset_id": "chair.asset.001",
                "source_db": "synthetic",
                "category": "chair",
                "description": "wood dining chair",
                "mesh_uri": "/synthetic/chair.glb",
                "bbox_size_local": [0.8, 0.7, 1.0],
                "bbox_center_local": [0.0, 0.0, 0.5],
                "canonical_front": [0.0, -1.0, 0.0],
                "native_scale": [1.0, 1.0, 1.0],
                "content": {"mesh_sha256": "1" * 64},
                "metadata": {},
            },
            {
                "asset_id": "table.asset.001",
                "source_db": "synthetic",
                "category": "table",
                "description": "small dining table",
                "mesh_uri": "/synthetic/table.glb",
                "bbox_size_local": [1.2, 0.8, 0.75],
                "bbox_center_local": [0.0, 0.0, 0.375],
                "native_scale": 1.0,
                "content": {"mesh_sha256": "2" * 64},
                "metadata": {},
            },
        ],
        "metadata": {"fixture": True},
    }


def _protocol(
    catalog: CanonicalAssetCatalog,
    *,
    mode: str,
) -> ComparisonProtocol:
    if mode == "native":
        asset_policy = "native"
        retrieval_policy = "method_native"
        inventory_policy = "method_native"
        scale_policy = "method_native"
        assets = None
        objects = []
    elif mode == "shared_db":
        asset_policy = "shared_catalog"
        retrieval_policy = "method_native_shared_catalog"
        inventory_policy = "frozen"
        scale_policy = "method_native"
        assets = catalog.identity
        objects = [
            {
                "slot_id": "chair_0",
                "category": "chair",
                "description": "wood dining chair",
            }
        ]
    else:
        asset_policy = "frozen_exact"
        retrieval_policy = "disabled_exact_bindings"
        inventory_policy = "frozen"
        scale_policy = "fixed_native_scale"
        assets = catalog.identity
        objects = [
            {
                "slot_id": "chair_0",
                "category": "chair",
                "description": "wood dining chair",
                "asset_id": "chair.asset.001",
            }
        ]
    return ComparisonProtocol.from_mapping(
        {
            "protocol_id": "generation_comparison_v1",
            "protocol_version": 1,
            "mode": mode,
            "case_id": "rect-room-chair",
            "architecture": _architecture(),
            "object_inventory_policy": inventory_policy,
            "objects": objects,
            "asset_policy": asset_policy,
            "assets": assets,
            "scale_policy": scale_policy,
            "retrieval_policy": retrieval_policy,
            "generation": {"budget_policy": "method_native_recorded"},
            "evaluator": {"policy": "same_canonical_run_evaluate"},
        }
    )


def _architecture() -> dict[str, Any]:
    return {
        "room_model": "single_room",
        "boundary_model": "axis_aligned_rectangle",
        "room": {
            "boundary": [[0, 0], [4, 0], [4, 5], [0, 5]],
            "height": 3.0,
            "unit": "meter",
        },
    }


def _generation_input() -> dict[str, Any]:
    return build_direct_natural_language_generation_input(
        request_id="comparison-case",
        instruction="Place exactly one wood dining chair.",
        scene_type="room",
        room={
            "boundary": [[0, 0], [4, 0], [4, 5], [0, 5]],
            "height": 3.0,
            "unit": "meter",
            "metadata": {"public_marker": "PUBLIC_ONLY"},
        },
        evaluator_output_type=O1_OBJECT_STATE,
    )


def _native_output(adapter_name: str, out_dir: Path, config: dict) -> Path:
    del config
    asset_id = "chair.asset.001"
    if adapter_name == "layout_gpt":
        return write_json(
            out_dir / "layoutgpt.json",
            {
                "unit": "m",
                "object_list": [
                    [
                        "chair",
                        {
                            "length": 0.8,
                            "width": 0.7,
                            "height": 1.0,
                            "left": 2.0,
                            "top": 2.0,
                            "depth": 0.5,
                            "orientation": 0.0,
                            "asset": {
                                "asset_key": asset_id,
                                "category": "chair",
                            },
                        },
                    ]
                ],
            },
        )
    if adapter_name == "direct_layout":
        return write_json(
            out_dir / "direct.json",
            [
                {
                    "new_object_id": "chair_0",
                    "asset_id": asset_id,
                    "category": "chair",
                    "description": "wood dining chair",
                    "rotation": {"z_angle": 0.0},
                    "size_in_meters": {
                        "length": 0.8,
                        "width": 0.7,
                        "height": 1.0,
                    },
                    "position": {"x": 2.0, "y": 2.0, "z": 0.5},
                }
            ],
        )
    if adapter_name == "layout_vlm":
        return write_json(
            out_dir / "layout.json",
            {
                "chair_0": {
                    "position": [2.0, 2.0, 0.5],
                    "rotation": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                }
            },
        )
    if adapter_name == "respace":
        return write_json(
            out_dir / "scene.json",
            {
                "bounds_bottom": [
                    [0.0, 0.0, 0.0],
                    [4.0, 0.0, 0.0],
                    [4.0, 0.0, -5.0],
                    [0.0, 0.0, -5.0],
                ],
                "bounds_top": [
                    [0.0, 3.0, 0.0],
                    [4.0, 3.0, 0.0],
                    [4.0, 3.0, -5.0],
                    [0.0, 3.0, -5.0],
                ],
                "objects": [
                    {
                        "id": "chair_0",
                        "category": "chair",
                        "desc": "wood dining chair",
                        "sampled_asset_jid": asset_id,
                        "pos": [2.0, 0.0, -2.0],
                        "rot": [0.0, 0.0, 0.0, 1.0],
                        "size": [0.8, 1.0, 0.7],
                        "sampled_asset_size": [0.8, 1.0, 0.7],
                        "scale": [1.0, 1.0, 1.0],
                    }
                ],
            },
        )
    root = out_dir / "sceneweaver_native" / "record_scene"
    for iteration in (0, 1):
        write_json(
            root / f"layout_{iteration}.json",
            {
                "roomsize": [4.0, 5.0],
                "structure": {},
                "objects": {
                    "chair_0": {
                        "asset_id": asset_id,
                        "category": "chair",
                        "description": "wood dining chair",
                        "location": [1.5 + iteration * 0.5, 2.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                        "size": [0.8, 0.7, 1.0],
                        "parent": [],
                    }
                },
            },
        )
    return root.parent


def _canonical_scene(
    *,
    asset_id: str = "chair.asset.001",
    size: list[float] | None = None,
) -> dict[str, Any]:
    catalog = _catalog()
    asset = catalog.get(asset_id)
    provenance = {
        **catalog.identity,
        "asset_id": asset["asset_id"],
        "bbox_size_local": asset["bbox_size_local"],
        "bbox_center_local": asset["bbox_center_local"],
        "native_scale": asset["native_scale"],
        "physical_dimensions": asset["physical_dimensions"],
        "canonical_front": asset.get("canonical_front"),
    }
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "comparison-scene",
        "request_id": "comparison-case",
        "scene_type": "room",
        "boundary": [[0.0, 0.0], [4.0, 0.0], [4.0, 5.0], [0.0, 5.0]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "chair_0",
                "jid": asset_id,
                "category": "chair",
                "description": "wood dining chair",
                "size": size or [0.8, 0.7, 1.0],
                "center": [2.0, 2.0, 0.5],
                "rotation": [0.0, 0.0, 0.0],
                "geometry_provenance": "bbox_proxy",
                "asset_ref": {
                    "source_db": "synthetic",
                    "asset_key": asset_id,
                    "mesh_uri": asset["mesh_uri"],
                },
                "asset_proxy": {
                    "type": "harness_evaluated_obb",
                    "bbox_center_local": [0.0, 0.0, 0.0],
                    "bbox_size": size or [0.8, 0.7, 1.0],
                },
                "metadata": {
                    "comparison_catalog_provenance": provenance,
                    **(
                        {"canonical_front": asset["canonical_front"]}
                        if asset.get("canonical_front") is not None
                        else {}
                    ),
                },
            }
        ],
        "metadata": {
            "coordinate_frame": {
                "handedness": "right",
                "up_axis": "z",
                "linear_unit": "meter",
                "rotation_unit": "degree",
                "object_position": "bbox_center",
            },
            "harness_compatibility": {"asset_resolution_policy": "exact_only"},
        },
    }
