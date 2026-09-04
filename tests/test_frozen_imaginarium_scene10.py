from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.adapters.direct_layout.converter import convert_direct_layout
from benchmark.adapters.layout_vlm.converter import convert_layout_vlm
from benchmark.adapters.scene_weaver.converter import (
    convert_scene_weaver,
    discover_layout_iterations,
)
from benchmark.adapters.scene_weaver.adapter import (
    SceneWeaverAdapter,
    _without_asset_locators,
)
from benchmark.adapters.common.execution import (
    ExternalExecutionError,
    bridge_bundle_identity,
)
from benchmark.generation_comparison.catalog import CanonicalAssetCatalog
from benchmark.generation_comparison.imaginarium_bundle import (
    build_imaginarium_glb_bundle_plan,
    file_sha256,
    validate_imaginarium_glb_bundle,
)
from benchmark.generation_comparison.execution import (
    ComparisonRunError,
    run_controlled_generation,
)
from benchmark.generation_comparison.inputs import build_controlled_generation_input
from benchmark.generation_comparison.materializers import materialize_method_catalog
from benchmark.generation_comparison.model_policy import (
    api_base_sha256,
    configured_model_policy_report,
    normalize_model_policy,
    reported_model_policy_report,
)
from benchmark.generation_comparison.protocol import ComparisonProtocol
from benchmark.nl_scene.generation_input import build_generation_input, build_scene_request
from benchmark.utils.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]
BRIDGES = ROOT / "scripts" / "external_harness_bridges"


def test_checked_in_scene10_candidate_is_exact_and_excludes_respace() -> None:
    value = read_json(
        ROOT
        / "configs"
        / "generation_comparison"
        / "frozen_imaginarium_scene10_v1.json"
    )
    assert value["methods"] == [
        "catalog_placement",
        "layout_gpt",
        "direct_layout",
        "layout_vlm",
        "scene_weaver",
    ]
    assert value["asset_selection_status"] == (
        "candidate_pending_human_approval"
    )
    assert [case["case_id"] for case in value["cases"]] == [
        f"S{number}" for number in range(100, 110)
    ]
    assert [len(case["objects"]) for case in value["cases"]] == [
        22,
        25,
        24,
        21,
        27,
        21,
        23,
        24,
        21,
        27,
    ]
    assert sum(len(case["objects"]) for case in value["cases"]) == 235
    assert len(value["catalog"]["assets"]) == 125
    assets_with_front = [
        asset
        for asset in value["catalog"]["assets"]
        if asset.get("canonical_front") is not None
    ]
    assert assets_with_front
    assert all(
        asset.get("canonical_front_source")
        == "imaginarium_catalog_facing_v1"
        for asset in assets_with_front
    )
    policy = value["generation"]["model_policy"]
    assert value["generation"]["require_local_asset_bytes"] is True
    assert value["generation"]["asset_geometry_tolerance_m"] == 1.0e-4
    assert value["generation"]["require_pinned_execution_identity"] is True
    assert len(policy["required_api_base_sha256"]) == 64
    assert value["generation"]["harness_inputs"]["layout_gpt"]["status"] == (
        "candidate_pending_human_approval"
    )
    assert value["generation"]["harness_inputs"]["layout_gpt"][
        "hidden_evaluator_data_used"
    ] is False
    assert policy["comparison_group"] == [
        "layout_gpt",
        "direct_layout",
        "layout_vlm",
        "scene_weaver",
    ]
    assert policy["excluded_baselines"] == ["catalog_placement"]
    assert value["evaluator"]["static_kwargs"] == {}
    assert "reference_annotation" not in value["evaluator"]
    assert "specification_contract" not in value["evaluator"]
    example = read_json(
        ROOT
        / "configs"
        / "generation_comparison"
        / "frozen_imaginarium_scene10_methods.example.json"
    )
    for method in ("layout_gpt", "direct_layout", "layout_vlm", "scene_weaver"):
        execution = example["methods"][method]["adapter_config"]["execution"]
        bridge = BRIDGES / f"{method}_frozen.py"
        assert execution["expected_entrypoint_sha256"] == file_sha256(bridge)
        assert execution["expected_bridge_bundle_sha256"] == (
            bridge_bundle_identity(bridge)["bridge_bundle_sha256"]
        )
    for case in value["cases"]:
        frozen_ids = [item["slot_id"] for item in case["objects"]]
        plan_ids = [item["id"] for item in case["object_plan"]["objects"]]
        assert len(frozen_ids) == len(set(frozen_ids))
        assert set(frozen_ids) == set(plan_ids)
        assert all(item["count"] == 1 for item in case["object_plan"]["objects"])
        assert all(
            item["placement_intent"]["absolute_relations"] == []
            for item in case["object_plan"]["objects"]
        )
        assert case["source_provenance"]["pose_reused"] is False
        assert case["source_provenance"]["evaluation_data_reused"] is False
        assert all(
            relation.get("subject_id") != relation.get("object_id")
            for relation in case["object_plan"]["relations"]
            if relation.get("subject_id") is not None
            and relation.get("object_id") is not None
        )


def test_same_model_policy_checks_config_and_preserved_runner_report(
    tmp_path: Path,
) -> None:
    endpoint_hash = api_base_sha256("https://models.example/v1")
    policy = normalize_model_policy(
        {
            "policy": "same_backing_model",
            "comparison_group": ["layout_gpt", "scene_weaver"],
            "excluded_baselines": ["catalog_placement"],
            "required_identity": {
                "provider": "openai_compatible",
                "model_id": "model-x",
            },
            "required_deployment_id": "shared-deployment-v1",
            "required_api_base_sha256": endpoint_hash,
        }
    )
    assert policy is not None
    assert len(policy["required_identity_sha256"]) == 64
    valid = configured_model_policy_report(
        adapter_name="layout_gpt",
        policy=policy,
        adapter_config={
            "model_identity": {
                "provider": "openai_compatible",
                "model_id": "model-x",
            },
            "model_deployment_id": "shared-deployment-v1",
        },
    )
    assert valid["valid"] is True
    mismatch = configured_model_policy_report(
        adapter_name="layout_gpt",
        policy=policy,
        adapter_config={
            "model_identity": {
                "provider": "openai_compatible",
                "model_id": "model-y",
            },
            "model_deployment_id": "shared-deployment-v1",
        },
    )
    assert mismatch["valid"] is False
    deployment_mismatch = configured_model_policy_report(
        adapter_name="layout_gpt",
        policy=policy,
        adapter_config={
            "model_identity": {
                "provider": "openai_compatible",
                "model_id": "model-x",
            },
            "model_deployment_id": "different-deployment",
        },
    )
    assert deployment_mismatch["valid"] is False
    assert "configured_model_deployment_mismatch" in deployment_mismatch[
        "reasons"
    ]
    baseline = configured_model_policy_report(
        adapter_name="catalog_placement", policy=policy, adapter_config={}
    )
    assert baseline["status"] == "EXCLUDED_BASELINE"
    report = write_json(
        tmp_path / "runner_report.json",
        {
            "resource_usage": {
                "model_identity_evidence": "observed_response",
                "model_identities": [
                    {"provider": "openai_compatible", "model_id": "model-x"}
                ],
                "model_deployment_id": "shared-deployment-v1",
                "model_endpoint_sha256": endpoint_hash,
            }
        },
    )
    observed = reported_model_policy_report(
        adapter_name="layout_gpt",
        policy=policy,
        execution_metadata={
            "preserved_auxiliary_artifacts": {
                "runner_report": {"path": report.as_posix()}
            }
        },
    )
    assert observed["valid"] is True


def test_agent_endpoint_forms_resolve_to_one_frozen_base_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common = _bridge("_common")
    digest = api_base_sha256("https://Models.Example/v1/")
    monkeypatch.setenv("LAYOUT_DDD_REQUIRED_API_BASE_SHA256", digest)
    assert common.verify_api_endpoint_contract(
        "https://models.example/v1", completion_endpoint=False
    ) == digest
    assert common.verify_api_endpoint_contract(
        "https://models.example/v1/chat/completions",
        completion_endpoint=True,
    ) == digest
    with pytest.raises(RuntimeError, match="differs"):
        common.verify_api_endpoint_contract(
            "https://different.example/v1", completion_endpoint=False
        )
    with pytest.raises(ValueError, match="must use HTTPS"):
        api_base_sha256("http://remote.example/v1")
    assert len(api_base_sha256("http://127.0.0.1:8080/v1")) == 64


def test_layoutgpt_transport_error_detail_redacts_and_caps_secrets() -> None:
    common = _bridge("_common")
    secret = "private-test-token"
    detail = common._redacted_error_detail(
        (
            f'Authorization: Bearer {secret} api_key={secret} '
            + "x" * 5000
        ),
        secret=secret,
        truncated=False,
    )
    assert secret not in detail
    assert "Authorization: <redacted>" in detail
    assert detail.endswith("...<truncated>")
    assert len(detail) < 4200


def test_same_model_policy_is_enforced_after_controlled_generation(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = ComparisonProtocol.from_mapping(
        {
            **_protocol(catalog).as_dict(),
            "generation": {
                "model_policy": {
                    "policy": "same_backing_model",
                    "comparison_group": ["direct_layout"],
                    "excluded_baselines": ["catalog_placement"],
                    "required_identity": {
                        "provider": "openai_compatible",
                        "model_id": "model-x",
                    },
                    "required_deployment_id": "shared-deployment-v1",
                }
            },
        }
    )

    def runner(*, method_input_path: Path, out_dir: Path, config: dict) -> dict:
        del method_input_path, config
        native = write_json(
            out_dir / "direct.json",
            [
                {
                    "new_object_id": "reading_chair_1",
                    "asset_id": "chair.asset",
                    "category": "Single_sofa_chair",
                    "description": "fixture chair",
                    "canonical_front": [0.0, -1.0, 0.0],
                    "size_in_meters": {
                        "length": 0.8,
                        "width": 0.7,
                        "height": 1.0,
                    },
                    "position": {"x": 2.0, "y": 2.0, "z": 0.5},
                    "rotation": {"z_angle": 0.0},
                }
            ],
        )
        return {
            "native_artifact_path": native,
            "resource_usage": {
                "model_identity_evidence": "observed_response",
                "model_identities": [
                    {"provider": "openai_compatible", "model_id": "model-x"}
                ],
                "model_deployment_id": "shared-deployment-v1",
            },
        }

    config = {
        "runner": runner,
        "model_identity": {
            "provider": "openai_compatible",
            "model_id": "model-x",
        },
        "model_deployment_id": "shared-deployment-v1",
        "comparison_support": {
            "fixed_object_inventory": True,
            "exact_asset_ids": True,
            "fixed_native_scale": True,
        },
    }
    result = run_controlled_generation(
        generation_input=_direct_generation_input(),
        adapter_name="direct_layout",
        protocol=protocol,
        asset_catalog=catalog,
        out_dir=tmp_path / "valid",
        adapter_config=config,
    )
    assert result["valid_comparison_run"] is True
    assert result["model_policy"]["reported"]["status"] == "VALID"

    def mismatched_runner(
        *, method_input_path: Path, out_dir: Path, config: dict
    ) -> dict:
        result = runner(
            method_input_path=method_input_path,
            out_dir=out_dir,
            config=config,
        )
        result["resource_usage"]["model_identities"][0]["model_id"] = "model-y"
        return result

    invalid_config = {**config, "runner": mismatched_runner}
    with pytest.raises(ComparisonRunError, match="fairness validation failed"):
        run_controlled_generation(
            generation_input=_direct_generation_input(),
            adapter_name="direct_layout",
            protocol=protocol,
            asset_catalog=catalog,
            out_dir=tmp_path / "invalid",
            adapter_config=invalid_config,
        )
    validation = read_json(
        tmp_path / "invalid" / "comparison" / "validation.json"
    )
    assert validation["valid_comparison_run"] is False
    assert any(
        item["code"] == "model_policy_mismatch"
        for item in validation["violations"]
    )


def test_candidate_approval_gate_cannot_be_bypassed_via_core_runner(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = ComparisonProtocol.from_mapping(
        {
            **_protocol(catalog).as_dict(),
            "generation": {
                "asset_selection_status": "candidate_pending_human_approval"
            },
        }
    )
    called = False

    def runner(**_kwargs: object) -> dict[str, object]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(ComparisonRunError, match="pre-run eligibility failed"):
        run_controlled_generation(
            generation_input=_direct_generation_input(),
            adapter_name="direct_layout",
            protocol=protocol,
            asset_catalog=catalog,
            out_dir=tmp_path / "candidate",
            adapter_config={
                "runner": runner,
                "comparison_support": {
                    "fixed_object_inventory": True,
                    "exact_asset_ids": True,
                    "fixed_native_scale": True,
                },
            },
        )
    assert called is False


def test_core_runner_detects_frozen_mesh_byte_mutation(tmp_path: Path) -> None:
    mesh = tmp_path / "chair.glb"
    mesh.write_bytes(b"original frozen mesh")
    catalog = CanonicalAssetCatalog.from_mapping(
        {
            "catalog_id": "fixture",
            "catalog_version": "mesh-v1",
            "assets": [
                {
                    "asset_id": "chair.asset",
                    "source_db": "imaginarium",
                    "category": "Single_sofa_chair",
                    "description": "fixture chair",
                    "mesh_uri": mesh.as_posix(),
                    "bbox_size_local": [0.8, 0.7, 1.0],
                    "bbox_center_local": [0.0, 0.0, 0.5],
                    "native_scale": [1.0, 1.0, 1.0],
                }
            ],
        },
        hash_local_meshes=True,
    )
    protocol = ComparisonProtocol.from_mapping(
        {
            **_protocol(catalog).as_dict(),
            "generation": {"require_local_asset_bytes": True},
        }
    )

    def runner(*, out_dir: Path, **_kwargs: object) -> dict[str, object]:
        mesh.write_bytes(b"mutated mesh")
        native = write_json(
            out_dir / "direct.json",
            [
                {
                    "new_object_id": "reading_chair_1",
                    "asset_id": "chair.asset",
                    "category": "Single_sofa_chair",
                    "description": "fixture chair",
                    "size_in_meters": [0.8, 0.7, 1.0],
                    "position": [2.0, 2.0, 0.5],
                    "rotation": {"z_angle": 0.0},
                }
            ],
        )
        return {"native_artifact_path": native}

    with pytest.raises(ComparisonRunError, match="fairness validation failed"):
        run_controlled_generation(
            generation_input=_direct_generation_input(),
            adapter_name="direct_layout",
            protocol=protocol,
            asset_catalog=catalog,
            out_dir=tmp_path / "mutated",
            adapter_config={
                "runner": runner,
                "comparison_support": {
                    "fixed_object_inventory": True,
                    "exact_asset_ids": True,
                    "fixed_native_scale": True,
                },
            },
        )
    validation = read_json(tmp_path / "mutated" / "comparison" / "validation.json")
    assert any(
        item["code"] == "asset_bytes_changed"
        for item in validation["violations"]
    )

    mesh.write_bytes(b"original frozen mesh")

    def failing_runner(**_kwargs: object) -> dict[str, object]:
        mesh.write_bytes(b"mutated before runner failure")
        raise RuntimeError("upstream failed after touching the mesh")

    failed_root = tmp_path / "failed_after_mutation"
    with pytest.raises(ExternalExecutionError, match="upstream failed"):
        run_controlled_generation(
            generation_input=_direct_generation_input(),
            adapter_name="direct_layout",
            protocol=protocol,
            asset_catalog=catalog,
            out_dir=failed_root,
            adapter_config={
                "runner": failing_runner,
                "comparison_support": {
                    "fixed_object_inventory": True,
                    "exact_asset_ids": True,
                    "fixed_native_scale": True,
                },
            },
        )
    failed_snapshot = read_json(
        failed_root / "comparison" / "asset_bytes_after_generation.json"
    )
    assert failed_snapshot["valid"] is False
    assert failed_snapshot["errors"][0]["code"] == "mesh_hash_mismatch"
    failed_manifest = read_json(
        failed_root / "comparison" / "run_manifest.json"
    )
    assert failed_manifest["status"] == "GENERATION_FAILED"


def test_controlled_input_preserves_public_semantics_separately_from_asset_category() -> None:
    catalog = _catalog()
    protocol = _protocol(catalog)
    object_plan = {
        "request_id": "case",
        "scene_type": "reading_room",
        "scene_description": "Arrange a reading chair.",
        "objects": [
            {
                "id": "reading_chair_1",
                "category": "lounge chair",
                "description": "A comfortable chair for reading.",
                "count": 1,
                "estimated_size": [2.0, 2.0, 2.0],
                "metadata": {"zone": "reading"},
                "placement_intent": {
                    "absolute_relations": [],
                    "relative_relations": ["Face the bookcase."],
                },
            }
        ],
        "global_constraints": ["Preserve circulation."],
        "relations": [],
        "zones": [{"id": "reading", "description": "Reading area"}],
    }
    source = build_generation_input(
        scene_request=build_scene_request(
            request_id="case",
            instruction="Arrange a reading chair.",
            scene_type="reading_room",
            room={
                "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                "height": 3,
                "unit": "meter",
            },
            structure=True,
        ),
        object_plan=object_plan,
    )
    controlled = build_controlled_generation_input(
        source,
        protocol=protocol,
        catalog=catalog,
        materialization=None,
    )
    assert controlled["object_plan"]["relations"] == object_plan["relations"]
    assert controlled["object_plan"]["global_constraints"] == object_plan[
        "global_constraints"
    ]
    assert controlled["object_plan"]["objects"][0]["category"] == "lounge chair"
    assert controlled["object_plan"]["objects"][0]["placement_intent"] == (
        object_plan["objects"][0]["placement_intent"]
    )
    assert controlled["object_plan"]["objects"][0]["metadata"] == {
        "zone": "reading",
        "comparison_slot_id": "reading_chair_1",
        "comparison_scale_policy": "fixed_native_scale",
    }
    assert controlled["object_plan"]["objects"][0]["estimated_size"] == [
        0.8,
        0.7,
        1.0,
    ]
    selection = controlled["asset_selection"]["objects"][0]
    assert selection["object_spec"]["category"] == "lounge chair"
    assert selection["selected_asset"]["category"] == "Single_sofa_chair"


def test_materialized_control_exposes_frozen_rows_and_model_contract(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = ComparisonProtocol.from_mapping(
        {
            **_protocol(catalog).as_dict(),
            "generation": {
                "model_policy": {
                    "policy": "same_backing_model",
                    "comparison_group": ["layout_gpt"],
                    "excluded_baselines": ["catalog_placement"],
                    "required_identity": {
                        "provider": "openai_compatible",
                        "model_id": "model-x",
                    },
                }
            },
        }
    )
    materialized = materialize_method_catalog(
        adapter_name="layout_gpt",
        catalog=catalog,
        protocol=protocol,
        out_dir=tmp_path / "materialized",
    )
    control = read_json(materialized["comparison_control_path"])

    assert control["objects"] == [
        {
            "slot_id": "reading_chair_1",
            "category": "Single_sofa_chair",
            "description": "fixture chair",
            "asset_id": "chair.asset",
        }
    ]
    assert control["generation"]["model_policy"]["required_identity"] == {
        "provider": "openai_compatible",
        "model_id": "model-x",
    }


def test_sceneweaver_frozen_materialization_exposes_only_bound_assets(
    tmp_path: Path,
) -> None:
    catalog = CanonicalAssetCatalog.from_mapping(
        {
            "catalog_id": "fixture",
            "catalog_version": "two-assets",
            "assets": [
                {
                    "asset_id": asset_id,
                    "source_db": "imaginarium",
                    "category": "Single_sofa_chair",
                    "description": asset_id,
                    "bbox_size_local": [0.8, 0.7, 1.0],
                    "bbox_center_local": [0.0, 0.0, 0.5],
                    "native_scale": [1.0, 1.0, 1.0],
                }
                for asset_id in ("chair.asset", "unbound.asset")
            ],
        }
    )
    materialized = materialize_method_catalog(
        adapter_name="scene_weaver",
        catalog=catalog,
        protocol=_protocol(catalog),
        out_dir=tmp_path / "sceneweaver",
    )
    payload = read_json(materialized["method_catalog_path"])
    assert list(payload["asset_source"]) == ["chair.asset"]
    assert payload["frozen_asset_bindings"]["reading_chair_1"]["asset_key"] == (
        "chair.asset"
    )


def test_directlayout_exact_binding_sidecar_does_not_mutate_native_artifact(
    tmp_path: Path,
) -> None:
    native = write_json(
        tmp_path / "direct.json",
        [
            {
                "new_object_id": "reading_chair_1",
                "rotation": {"z_angle": 20.0},
                "size_in_meters": {
                    "length": 0.8,
                    "width": 0.7,
                    "height": 1.0,
                },
                "position": {"x": 2.0, "y": 2.0, "z": 0.5},
            }
        ],
    )
    before = native.read_bytes()
    scene = convert_direct_layout(
        native,
        _direct_generation_input(),
        {
            "asset_bindings": {
                "reading_chair_1": {
                    "asset_key": "chair.asset",
                    "source_db": "imaginarium",
                    "category": "Single_sofa_chair",
                    "description": "fixture chair",
                    "canonical_front": [0.0, -1.0, 0.0],
                }
            }
        },
        None,
    )
    assert native.read_bytes() == before
    assert scene["objects"][0]["asset_ref"]["asset_key"] == "chair.asset"
    assert scene["objects"][0]["metadata"]["native_asset_binding_source"] == (
        "preserved_asset_bindings_sidecar"
    )
    rotation_z = scene["objects"][0]["rotation"][2]
    assert ((rotation_z - 200.0 + 180.0) % 360.0) - 180.0 == pytest.approx(0.0)


def test_layoutvlm_processed_bbox_and_front_are_restored_to_canonical_mesh_frame(
    tmp_path: Path,
) -> None:
    native = write_json(
        tmp_path / "layout.json",
        {
            "reading_chair_1-0": {
                "position": [2.0, 2.0, 0.5],
                "rotation": [0.0, 0.0, 0.0],
                "scale": [1.0, 1.0, 1.0],
            }
        },
    )
    scene_config = {
        "boundary": {
            "floor_vertices": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]],
            "wall_height": 3.0,
        },
        "assets": {
            "reading_chair_1-0": {
                "uid": "chair.asset",
                "category": "Single_sofa_chair",
                "description": "fixture chair",
                "canonical_front": [0.0, -1.0, 0.0],
                "assetMetadata": {
                    "boundingBox": {"x": 0.7, "y": 0.8, "z": 1.0},
                    "canonicalBoundingBoxBeforeLayoutVLMSwap": {
                        "x": 0.8,
                        "y": 0.7,
                        "z": 1.0,
                    },
                    "axisTransform": (
                        "swap_xy_for_layoutvlm_processed_positive_x_frame"
                    ),
                },
            }
        },
    }
    scene = convert_layout_vlm(
        native,
        _direct_generation_input(),
        {"scene_config": scene_config},
        None,
    )
    assert scene["objects"][0]["size"] == pytest.approx([0.8, 0.7, 1.0])
    assert scene["objects"][0]["rotation"][2] == pytest.approx(90.0)
    assert scene["objects"][0]["metadata"]["geometry_audit"][
        "native_bbox_axis_transform"
    ] == "swap_xy_for_layoutvlm_processed_positive_x_frame"


def test_sceneweaver_released_world_aabb_restores_local_bbox_and_front_basis(
    tmp_path: Path,
) -> None:
    native = write_json(
        tmp_path / "layout_0.json",
        {
            "roomsize": [4.0, 4.0],
            "objects": {
                "reading_chair_1": {
                    "asset_id": "chair.asset",
                    "location": [2.0, 2.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "size": [0.7, 0.8, 1.0],
                }
            },
        },
    )
    scene = convert_scene_weaver(
        native,
        _direct_generation_input(),
        {
            "rotation_unit": "radian",
            "sceneweaver_native_size_semantics": (
                "released_world_aabb_rounded_2dp"
            ),
            "sceneweaver_world_aabb_tolerance": 1.0e-6,
            "sceneweaver_orientation_basis": (
                "bake_catalog_front_to_sceneweaver_positive_x"
            ),
            "sceneweaver_anchor_basis": (
                "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin"
            ),
            "asset_bindings": {
                "reading_chair_1": {
                    "asset_key": "chair.asset",
                    "source_db": "imaginarium",
                    "category": "Single_sofa_chair",
                    "description": "fixture chair",
                    "bbox_size_local": [0.8, 0.7, 1.0],
                    "physical_dimensions": [0.8, 0.7, 1.0],
                    "canonical_front": [0.0, -1.0, 0.0],
                    "full_precision_native_euler_xyz_by_iteration": {
                        "0": [0.0, 0.0, 0.0]
                    },
                    "full_precision_native_local_bbox_size_by_iteration": {
                        "0": [0.8, 0.7, 1.0]
                    },
                    "anchor_basis": {
                        "policy": (
                            "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin"
                        ),
                        "native_origin_semantics": "bbox_bottom_center",
                        "applied": True,
                    },
                }
            },
            "sceneweaver_asset_geometry_tolerance_m": 1.0e-4,
        },
        None,
    )
    obj = scene["objects"][0]
    assert obj["size"] == pytest.approx([0.8, 0.7, 1.0])
    assert obj["rotation"][2] == pytest.approx(90.0)
    assert obj["center"] == pytest.approx([2.0, 2.0, 0.5])
    audit = obj["metadata"]["geometry_audit"]
    assert audit["native_size"] == [0.7, 0.8, 1.0]
    assert audit["released_world_aabb_verified"] is True


def test_sceneweaver_uses_full_precision_pose_before_released_rounding(
    tmp_path: Path,
) -> None:
    native = write_json(
        tmp_path / "layout_3.json",
        {
            "roomsize": [8.0, 6.0],
            "objects": {
                "commode_1": {
                    "asset_id": "commode.asset",
                    "location": [3.0, 2.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "size": [4.5, 1.19, 1.0],
                }
            },
        },
    )
    generation_input = build_generation_input(
        scene_request=build_scene_request(
            request_id="case",
            instruction="Arrange the commode.",
            scene_type="living_room",
            room={
                "boundary": [[0, 0], [8, 0], [8, 6], [0, 6]],
                "height": 3,
                "unit": "meter",
            },
            structure=False,
        ),
        object_plan=None,
    )
    exact_size = [4.490486, 1.1699, 1.0]
    precise_rotation = [0.0, 0.0, 0.0039]
    scene = convert_scene_weaver(
        native,
        generation_input,
        {
            "rotation_unit": "radian",
            "sceneweaver_native_size_semantics": (
                "released_world_aabb_rounded_2dp"
            ),
            "sceneweaver_world_aabb_tolerance": 1.0e-6,
            "sceneweaver_asset_geometry_tolerance_m": 1.0e-4,
            "sceneweaver_orientation_basis": (
                "bake_catalog_front_to_sceneweaver_positive_x"
            ),
            "sceneweaver_anchor_basis": (
                "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin"
            ),
            "asset_bindings": {
                "commode_1": {
                    "asset_key": "commode.asset",
                    "source_db": "imaginarium",
                    "category": "commode",
                    "description": "fixture commode",
                    "bbox_size_local": exact_size,
                    "physical_dimensions": exact_size,
                    "canonical_front": [1.0, 0.0, 0.0],
                    "full_precision_native_euler_xyz_by_iteration": {
                        "3": precise_rotation
                    },
                    "full_precision_native_local_bbox_size_by_iteration": {
                        "3": exact_size
                    },
                    "anchor_basis": {
                        "policy": (
                            "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin"
                        ),
                        "native_origin_semantics": "bbox_bottom_center",
                        "applied": True,
                    },
                }
            },
        },
        None,
    )
    obj = scene["objects"][0]
    assert obj["rotation"][2] == pytest.approx(0.2234535401)
    assert obj["size"] == pytest.approx(exact_size)
    audit = obj["metadata"]["geometry_audit"]
    assert audit["native_serialized_rotation"] == [0.0, 0.0, 0.0]
    assert audit["full_precision_native_rotation"] == precise_rotation
    assert audit["expected_released_world_aabb"] == [4.5, 1.19, 1.0]


def test_sceneweaver_released_zero_height_aabb_retains_positive_catalog_size(
    tmp_path: Path,
) -> None:
    native = write_json(
        tmp_path / "layout_0.json",
        {
            "roomsize": [4.0, 4.0],
            "objects": {
                "rug_1": {
                    "asset_id": "rug.asset",
                    "location": [2.0, 2.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "size": [0.8, 0.7, 0.0],
                }
            },
        },
    )
    exact_size = [0.8, 0.7, 0.000076]
    config = {
        "rotation_unit": "radian",
        "sceneweaver_native_size_semantics": "released_world_aabb_rounded_2dp",
        "sceneweaver_world_aabb_tolerance": 1.0e-6,
        "sceneweaver_asset_geometry_tolerance_m": 1.0e-4,
        "sceneweaver_orientation_basis": (
            "bake_catalog_front_to_sceneweaver_positive_x"
        ),
        "sceneweaver_anchor_basis": (
            "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin"
        ),
        "asset_bindings": {
            "rug_1": {
                "asset_key": "rug.asset",
                "source_db": "imaginarium",
                "category": "rug",
                "description": "thin floor rug",
                "bbox_size_local": exact_size,
                "physical_dimensions": exact_size,
                "canonical_front": [1.0, 0.0, 0.0],
                "full_precision_native_euler_xyz_by_iteration": {
                    "0": [0.0, 0.0, 0.0]
                },
                "full_precision_native_local_bbox_size_by_iteration": {
                    "0": exact_size
                },
                "anchor_basis": {
                    "policy": (
                        "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin"
                    ),
                    "native_origin_semantics": "bbox_bottom_center",
                    "applied": True,
                },
            }
        },
    }
    scene = convert_scene_weaver(
        native,
        _direct_generation_input(),
        config,
        None,
    )
    obj = scene["objects"][0]
    assert obj["size"] == pytest.approx(exact_size)
    assert obj["center"][2] == pytest.approx(exact_size[2] / 2.0)
    assert obj["metadata"]["geometry_audit"][
        "expected_released_world_aabb"
    ] == [0.8, 0.7, 0.0]


def test_sceneweaver_model_input_withholds_host_asset_locators() -> None:
    sanitized = _without_asset_locators(
        {
            "objects": [
                {
                    "selected_asset": {
                        "jid": "chair.asset",
                        "asset_ref": {
                            "source_db": "imaginarium",
                            "asset_key": "chair.asset",
                            "mesh_uri": "/private/cache/chair.glb",
                        },
                    }
                }
            ]
        }
    )
    text = json.dumps(sanitized)
    assert "chair.asset" in text
    assert "/private/cache" not in text
    assert "mesh_uri" not in text

    method_input = {
        "request_id": "case",
        "public_request": {"scene_type": "room"},
        "generator_input": {
            "natural_language": "Arrange the reading room.",
            "benchmark_environment": {
                "architecture": {
                    "room": {
                        "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                        "dimensions": {"width": 4, "depth": 4, "height": 3},
                        "height": 3,
                        "unit": "meter",
                    }
                }
            },
            "structure": {
                "object_plan": {
                    "objects": [{"id": "chair_1", "category": "chair"}],
                    "relations": [{"subject_id": "chair_1", "type": "near_wall"}],
                    "global_constraints": ["Keep a clear aisle."],
                }
            },
            "assistance": {"asset_selection": sanitized},
        },
    }
    native_input = SceneWeaverAdapter().build_native_input(method_input, {})
    assert native_input["public_object_plan"]["global_constraints"] == [
        "Keep a clear aisle."
    ]
    assert len(native_input["public_object_plan_sha256"]) == 64
    assert "/private/cache" not in json.dumps(native_input)


def test_sceneweaver_discovers_released_nested_output_without_flattening(
    tmp_path: Path,
) -> None:
    root = tmp_path / "native_root"
    first = write_json(root / "scene_name" / "record_scene" / "layout_0.json", {})
    second = write_json(root / "scene_name" / "record_scene" / "layout_1.json", {})
    assert discover_layout_iterations(root) == {0: first, 1: second}

    write_json(root / "another_scene" / "record_scene" / "layout_1.json", {})
    with pytest.raises(ValueError, match="ambiguous duplicate iteration 1"):
        discover_layout_iterations(root)


def test_sceneweaver_controlled_trajectory_reuses_preserved_binding_sidecar(
    tmp_path: Path,
) -> None:
    catalog = _catalog()
    protocol = _protocol(catalog)

    def runner(*, out_dir: Path, **_kwargs: object) -> dict[str, object]:
        native_root = out_dir / "sceneweaver_native" / "record_scene"
        for iteration, x_position in enumerate((1.5, 2.0)):
            write_json(
                native_root / f"layout_{iteration}.json",
                {
                    "roomsize": [4.0, 4.0],
                    "objects": {
                        "reading_chair_1": {
                            "location": [x_position, 2.0, 0.0],
                            "rotation": [0.0, 0.0, 0.0],
                            "size": [0.8, 0.7, 1.0],
                        }
                    },
                },
            )
        sidecar = out_dir / "upstream_output" / "asset_bindings.json"
        write_json(
            sidecar,
            {
                "asset_bindings": {
                    "reading_chair_1": {
                        "asset_key": "chair.asset",
                        "source_db": "imaginarium",
                        "category": "Single_sofa_chair",
                        "description": "fixture chair",
                        "bbox_size_local": [0.8, 0.7, 1.0],
                        "physical_dimensions": [0.8, 0.7, 1.0],
                        "anchor_basis": {
                            "policy": (
                                "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin"
                            ),
                            "native_origin_semantics": "bbox_bottom_center",
                            "applied": True,
                        },
                        "full_precision_native_euler_xyz_by_iteration": {
                            "0": [0.0, 0.0, 0.0],
                            "1": [0.0, 0.0, 0.0],
                        },
                        "full_precision_native_local_bbox_size_by_iteration": {
                            "0": [0.8, 0.7, 1.0],
                            "1": [0.8, 0.7, 1.0],
                        },
                    }
                }
            },
        )
        return {"native_artifact_path": native_root.parent}

    result = run_controlled_generation(
        generation_input=_direct_generation_input(),
        adapter_name="scene_weaver",
        protocol=protocol,
        asset_catalog=catalog,
        out_dir=tmp_path / "run",
        adapter_config={
            "runner": runner,
            "selected_iteration": 1,
            "comparison_support": {
                "fixed_object_inventory": True,
                "exact_asset_ids": True,
                "fixed_native_scale": True,
                "frozen_iteration_bindings": True,
                "no_object_insertion_removal": True,
                "sceneweaver_released_export_contract": True,
            },
            "execution": {
                "auxiliary_artifacts": {
                    "asset_bindings": (
                        "{upstream_output_dir}/asset_bindings.json"
                    )
                }
            },
        },
    )
    trajectory = result["sceneweaver_trajectory"]
    assert trajectory["valid_comparison_trajectory"] is True
    assert [row["iteration"] for row in trajectory["iterations"]] == [0, 1]
    for row in trajectory["iterations"]:
        scene = read_json(row["canonical_scene"])
        assert scene["objects"][0]["asset_ref"]["asset_key"] == "chair.asset"


def test_asset_review_flags_assets_taller_than_the_room(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "audit_frozen_imaginarium_scene10_assets.py"
    spec = importlib.util.spec_from_file_location("scene10_asset_audit", script)
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    asset_root = tmp_path / "assets"
    metadata = asset_root / "tall_shelf" / "tall_shelf_metadata.json"
    write_json(
        metadata,
        {
            "transformed_size": [1.0, 0.4, 3.2],
            "transformed_bbox_center": [0.0, 0.0, 1.6],
        },
    )
    rows = audit._rows(
        {
            "catalog": {
                "assets": [
                    {
                        "asset_id": "tall_shelf",
                        "category": "shelf",
                        "description": "tall shelf",
                    }
                ]
            },
            "cases": [
                {
                    "case_id": "S-test",
                    "room": {
                        "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                        "height": 3.0,
                    },
                    "objects": [
                        {
                            "slot_id": "shelf_1",
                            "asset_id": "tall_shelf",
                            "metadata": {"requested_category": "shelf"},
                        }
                    ],
                    "object_plan": {
                        "objects": [
                            {
                                "id": "shelf_1",
                                "category": "shelf",
                                "description": "storage shelf",
                                "estimated_size": [1.0, 0.4, 3.0],
                                "metadata": {"support": "floor"},
                            }
                        ]
                    },
                }
            ],
        },
        asset_root,
    )
    assert rows[0]["room_height_m"] == 3.0
    assert "exceeds_room_height" in rows[0]["review_flags"]
    assert rows[0]["review_priority"] == "HIGH"


def test_imaginarium_glb_bundle_plan_and_validation_are_content_addressed(
    tmp_path: Path,
) -> None:
    source = tmp_path / "assets" / "chair.asset"
    source.mkdir(parents=True)
    fbx = source / "chair.asset.fbx"
    fbx.write_bytes(b"fbx")
    write_json(
        source / "chair.asset_metadata.json",
        {
            "transformed_size": [0.8, 0.7, 1.0],
            "transformed_bbox_center": [0.0, 0.0, 0.5],
        },
    )
    plan = build_imaginarium_glb_bundle_plan(
        catalog_spec={
            "catalog_id": "fixture",
            "catalog_version": "1",
            "source_db": "imaginarium",
            "assets": [{"asset_id": "chair.asset"}],
        },
        asset_root=tmp_path / "assets",
        bundle_root=tmp_path / "bundle",
    )
    target = Path(plan["assets"][0]["target_glb"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"glb")
    report = write_json(
        tmp_path / "bundle" / "bundle_report.json",
        {
            "schema_version": "imaginarium_glb_bundle_report_v1",
            "plan": Path(plan["plan_path"]).resolve().as_posix(),
            "geometry_tolerance_m": 1.0e-4,
            "assets": [
                {
                    "asset_id": "chair.asset",
                    "status": "passed",
                    "source_fbx_sha256": file_sha256(fbx),
                    "source_metadata_sha256": plan["assets"][0][
                        "source_metadata_sha256"
                    ],
                    "target_glb_sha256": file_sha256(target),
                    "target_glb": target.resolve().as_posix(),
                    "source_bbox_size": [0.8, 0.7, 1.0],
                    "source_bbox_center": [0.0, 0.0, 0.5],
                    "roundtrip_bbox_size": [0.8, 0.7, 1.0],
                    "roundtrip_bbox_center": [0.0, 0.0, 0.5],
                    "geometry_verified": True,
                }
            ],
        },
    )
    validation = validate_imaginarium_glb_bundle(
        plan=plan["plan_path"], report=report
    )
    assert validation == {
        "schema_version": "imaginarium_glb_bundle_validation_v1",
        "valid": True,
        "asset_count": 1,
        "errors": [],
    }
    metadata_path = source / "chair.asset_metadata.json"
    metadata_bytes = metadata_path.read_bytes()
    metadata_path.write_text("{}\n", encoding="utf-8")
    metadata_validation = validate_imaginarium_glb_bundle(
        plan=plan["plan_path"], report=report
    )
    assert any(
        item["code"] == "source_metadata_hash_mismatch"
        for item in metadata_validation["errors"]
    )
    metadata_path.write_bytes(metadata_bytes)
    target.write_bytes(b"changed")
    assert validate_imaginarium_glb_bundle(
        plan=plan["plan_path"], report=report
    )["valid"] is False

    target.write_bytes(b"glb")
    copied_root = tmp_path / "copied_bundle"
    copied_root.mkdir()
    copied_plan = write_json(copied_root / "bundle_plan.json", read_json(plan["plan_path"]))
    copied_report = write_json(copied_root / "bundle_report.json", read_json(report))
    root_validation = validate_imaginarium_glb_bundle(
        plan=copied_plan,
        report=copied_report,
        expected_asset_root=tmp_path / "assets",
        expected_bundle_root=copied_root,
    )
    assert any(
        item["code"] in {"target_root_mismatch", "report_plan_mismatch"}
        for item in root_validation["errors"]
    )


def test_bridge_input_builders_preserve_frozen_ids_and_geometry(
    tmp_path: Path,
) -> None:
    layout_gpt = _bridge("layout_gpt_frozen")
    direct = _bridge("direct_layout_frozen")
    layout_vlm = _bridge("layout_vlm_frozen")
    expected = [
        {
            "slot_id": "reading_chair_1",
            "native_object_id": "single_sofa_chair_1",
            "selector": "single_sofa_chair",
            "description": "fixture chair",
            "canonical_front": [0.0, -1.0, 0.0],
            "size": [0.8, 0.7, 1.0],
        }
    ]
    parsed = layout_gpt._parse_and_validate(
        "single_sofa_chair {length: 0.8m; width: 0.7m; height: 1.0m; "
        "orientation: 0 degrees; left: 2m; top: 2m; depth: 0.5m;}",
        expected,
        size_tolerance=1.0e-6,
    )
    assert parsed[0][0] == "single_sofa_chair"
    with pytest.raises(RuntimeError, match="fields differ|invalid controlled units"):
        layout_gpt._parse_and_validate(
            "single_sofa_chair {length: 0.8cm; width: 0.7m; height: 1.0m; "
            "orientation: 0 degrees; left: 2m; top: 2m; depth: 0.5m;}",
            expected,
            size_tolerance=1.0e-6,
        )
    icl = write_json(
        tmp_path / "layoutgpt_icl.json",
        [
            {"role": "user", "content": "Condition:\nRoom Size: 4m x 4m"},
            {
                "role": "assistant",
                "content": "chair {length: 1m; width: 1m; height: 1m; "
                "orientation: 0 degrees; left: 1m; top: 1m; depth: 0.5m;}",
            },
        ],
    )
    messages = layout_gpt._messages(
        {"prompt": "Arrange a chair.", "room_dimensions_m": [4.0, 4.0, 3.0]},
        {"objects": [], "global_constraints": ["Keep a clear aisle."]},
        expected,
        icl,
    )
    assert [item["role"] for item in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert "0 degrees faces world -Y" in messages[0]["content"]
    assert '"canonical_front":[0.0,-1.0,0.0]' in messages[-1]["content"]
    invalid_icl = write_json(
        tmp_path / "layoutgpt_invalid_icl.json",
        [
            {"role": "assistant", "content": "wrong first role"},
            {"role": "user", "content": "wrong second role"},
        ],
    )
    with pytest.raises(ValueError, match="alternating user"):
        layout_gpt._messages(
            {"prompt": "Arrange a chair.", "room_dimensions_m": [4.0, 4.0, 3.0]},
            {"objects": []},
            expected,
            invalid_icl,
        )

    control = {"objects": [{"slot_id": "reading_chair_1"}]}
    direct_catalog = {
        "native_selector_by_slot": {"reading_chair_1": "reading_chair"},
        "frozen_asset_bindings": {"reading_chair_1": "chair.asset"},
        "asset_library": [
            {
                "new_object_id": "chair.asset",
                "category": "Single_sofa_chair",
                "description": "fixture chair",
                "physical_dimensions": [0.8, 0.7, 1.0],
            }
        ],
    }
    prompt, rows = direct._controlled_prompt(
        original_prompt="Arrange a chair.",
        plan={"objects": []},
        control=control,
        catalog=direct_catalog,
    )
    assert "reading_chair" in prompt
    assert "80.0" in prompt
    assert "fixed_size_in_native_pixels" in prompt
    assert "canonical functional front local -Y" in prompt
    assert rows[0]["native_object_id"] == "reading_chair_1"

    calls = []

    class FakeRendering:
        def render_views(self, **kwargs: object) -> None:
            calls.append(("render", kwargs["layout_path"]))

    class FakeOptimization:
        def __init__(self, rendering_service: FakeRendering) -> None:
            self.rendering_service = rendering_service

        def _request_reasoning_feedback(self, *, prompt: str) -> None:
            calls.append(("reasoning", prompt))

        def _request_vlm_feedback(self, *, prompt: str, room: str) -> None:
            calls.append(("vlm", prompt, room))

        def optimize(self, **kwargs: object) -> None:
            calls.append(("path_prompt", kwargs["prompt"]))
            self._request_reasoning_feedback(prompt=str(kwargs["prompt"]))
            self._request_vlm_feedback(
                prompt=str(kwargs["prompt"]), room=str(kwargs["prompt"])
            )

    class FakePipeline:
        def __init__(self) -> None:
            self.rendering_service = FakeRendering()
            self.optimization_service = FakeOptimization(self.rendering_service)
            self.settings = SimpleNamespace(
                runtime=SimpleNamespace(
                    max_retries=2,
                    length=400,
                    width=400,
                    height=300,
                    views=[0, 1],
                    max_iterations=1,
                ),
                paths=SimpleNamespace(
                    render_dir="render",
                    output_dir="output",
                    assets_dir="assets",
                ),
            )

    fake_pipeline = FakePipeline()
    tracking = direct._install_stable_room_bridge(
        fake_pipeline,
        semantic_prompt="Full controlled semantic prompt",
        room_name="controlled_scene",
        expected=[],
        tolerance=1.0e-6,
        audit_dir=tmp_path / "direct_audit",
    )
    fake_pipeline._run_optimization_with_retry("ignored", object())
    assert calls == [
        ("path_prompt", "controlled_scene"),
        ("reasoning", "Full controlled semantic prompt"),
        ("vlm", "Full controlled semantic prompt", "controlled_scene"),
    ]
    assert tracking["optimization_completed"] is True
    assert tracking["feedback_rounds"] == 1

    invalid_layout = write_json(
        tmp_path / "invalid_direct.json",
        [
            {
                "new_object_id": "unexpected_1",
                "size_in_meters": [1.0, 1.0, 1.0],
            }
        ],
    )
    with pytest.raises(
        direct.FrozenContractViolation,
        match="violates frozen controls before render",
    ):
        fake_pipeline.rendering_service.render_views(
            scene_bound=[4.0, 4.0, 3.0],
            layout_path=invalid_layout.as_posix(),
            assets_path="assets",
            render_root="render",
            room="controlled_scene",
            views=[0, 1],
        )
    assert calls[-1][0] != "render"

    class FailingOptimization(FakeOptimization):
        def optimize(self, **_kwargs: object) -> None:
            raise RuntimeError("optimizer exploded")

    failing_pipeline = FakePipeline()
    failing_pipeline.optimization_service = FailingOptimization(
        failing_pipeline.rendering_service
    )
    failing_tracking = direct._install_stable_room_bridge(
        failing_pipeline,
        semantic_prompt="Full controlled semantic prompt",
        room_name="controlled_scene",
        expected=[],
        tolerance=1.0e-6,
        audit_dir=tmp_path / "direct_failure_audit",
    )
    with pytest.raises(RuntimeError, match="no base-layout fallback"):
        failing_pipeline._run_optimization_with_retry("ignored", object())
    assert failing_tracking["optimization_attempts"] == 2
    assert failing_tracking["optimization_completed"] is False
    selected_state = direct._select_completed_layout_state(
        {
            "completed_attempt": 2,
            "layout_states": [
                {
                    "optimization_attempt": 0,
                    "rendered": True,
                    "snapshot_path": "base.json",
                },
                {
                    "optimization_attempt": 1,
                    "rendered": True,
                    "snapshot_path": "stale_refined.json",
                },
            ],
        }
    )
    assert selected_state["snapshot_path"] == "base.json"

    mesh = tmp_path / "chair.glb"
    mesh.write_bytes(b"glb")
    layout_catalog_path = write_json(tmp_path / "layout_catalog.json", {})
    layout_catalog = {
        "logical_to_native_slot": {"reading_chair_1": "reading_chair_1-0"},
        "frozen_assets": {
            "reading_chair_1-0": {
                "uid": "chair.asset",
                "category": "Single_sofa_chair",
                "description": "fixture chair",
                "path": mesh.as_posix(),
                "canonical_front": [0.0, -1.0, 0.0],
                "assetMetadata": {
                    "boundingBox": {"x": 0.8, "y": 0.7, "z": 1.0}
                },
            }
        },
    }
    task = layout_vlm._prepare_task(
        {
            "task_description": "Arrange a chair.",
            "boundary": {
                "floor_vertices": [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]],
                "wall_height": 3,
            },
        },
        {
            "objects": [
                {
                    "id": "reading_chair_1",
                    "category": "lounge chair",
                    "role": "primary reading seat",
                    "description": "reading chair",
                    "estimated_size": [0.8, 0.7, 1.0],
                    "metadata": {
                        "support": "floor",
                        "zone": "reading_zone",
                        "directed": True,
                    },
                }
            ],
            "global_constraints": ["Keep a clear aisle."],
        },
        layout_catalog,
        layout_catalog_path,
    )
    assert list(task["assets"]) == ["reading_chair_1-0"]
    assert task["assets"]["reading_chair_1-0"]["uid"] == "chair.asset"
    assert task["assets"]["reading_chair_1-0"]["onFloor"] is True
    assert task["assets"]["reading_chair_1-0"]["assetMetadata"]["boundingBox"] == {
        "x": 0.7,
        "y": 0.8,
        "z": 1.0,
    }
    assert task["assets"]["reading_chair_1-0"]["assetMetadata"][
        "canonicalBoundingBoxBeforeLayoutVLMSwap"
    ] == {"x": 0.8, "y": 0.7, "z": 1.0}
    assert "Keep a clear aisle." in task["layout_criteria"]
    assert "primary reading seat" in task["layout_criteria"]
    assert "reading_zone" in task["layout_criteria"]
    assert "LayoutVLM zero rotation faces native +X" in task["layout_criteria"]
    assert "estimated_size" not in task["layout_criteria"]

    class FakeProgramSolver:
        def get_task_program(self, *_args: object, **_kwargs: object) -> str:
            return (
                'reading_chair_1 = Assets(description="chair", '
                "size=[0.71, 0.80, 1.00], placements=[])\n"
            )

    fake_program_solver = FakeProgramSolver()
    precise_task = deepcopy(task)
    precise_task["assets"]["reading_chair_1-0"]["assetMetadata"]["boundingBox"][
        "x"
    ] = 0.712345
    size_tracking = layout_vlm._install_exact_asset_size_literals(
        fake_program_solver, precise_task
    )
    exact_program = fake_program_solver.get_task_program([], precise_task)
    assert "size=[0.712345,0.8,1.0]" in exact_program
    assert size_tracking["calls"] == 1

    subset_task = deepcopy(precise_task)
    subset_task["assets"]["table_1-0"] = {
        **deepcopy(subset_task["assets"]["reading_chair_1-0"]),
        "asset_var_name": "table_1",
        "instance_var_name": "table_1",
        "uid": "table.asset",
    }

    class FakeSubsetSolver:
        def get_task_program(self, grouped_assets: list[str], *_args: object) -> str:
            lines = []
            if "reading_chair_1-0" in grouped_assets:
                lines.append(
                    'reading_chair_1 = Assets(description="chair", '
                    "size=[0.71, 0.80, 1.00], placements=[])"
                )
            if "table_1-0" in grouped_assets:
                lines.append(
                    'table_1 = Assets(description="table", '
                    "size=[0.71, 0.80, 1.00], placements=[])"
                )
            return "\n".join(lines)

    subset_solver = FakeSubsetSolver()
    layout_vlm._install_exact_asset_size_literals(subset_solver, subset_task)
    subset_program = subset_solver.get_task_program(
        ["reading_chair_1-0"], subset_task
    )
    assert "reading_chair_1" in subset_program
    assert "table_1" not in subset_program

    class FakeSandbox:
        def __init__(self) -> None:
            self.local_vars = {
                "reading_chair_1": SimpleNamespace(size=[0.7, 0.8, 1.0])
            }

        def export_layout(self, **_kwargs: object) -> dict[str, object]:
            return {"reading_chair_1-0": {"position": [1, 1, 0.5]}}

    sandbox = FakeSandbox()
    solver_observation = layout_vlm._observe_solver_state(
        SimpleNamespace(sandbox=sandbox), task, tolerance=1.0e-6
    )
    assert solver_observation["valid"] is True
    sandbox.local_vars["reading_chair_1"].size = [9.0, 9.0, 9.0]
    assert layout_vlm._observe_solver_state(
        SimpleNamespace(sandbox=sandbox), task, tolerance=1.0e-6
    )["valid"] is False


def test_sceneweaver_bridge_validates_every_frozen_iteration(
    tmp_path: Path,
) -> None:
    bridge = _bridge("scene_weaver_frozen")
    mesh = tmp_path / "chair.glb"
    mesh.write_bytes(b"frozen chair mesh")
    mesh_sha256 = file_sha256(mesh)
    catalog = {
        "logical_to_native_slot": {"reading_chair_1": "reading_chair_1"},
        "frozen_asset_bindings": {
            "reading_chair_1": {
                "asset_key": "chair.asset",
                "category": "Single_sofa_chair",
                "description": "fixture chair",
                "mesh_uri": mesh.as_posix(),
                "mesh_sha256": mesh_sha256,
                "bbox_center_local": [0.0, 0.0, 0.0],
                "native_scale": [1.0, 1.0, 1.0],
                "physical_dimensions": [0.8, 0.7, 1.0],
            }
        },
    }
    orientation_basis = bridge._orientation_basis(
        catalog["frozen_asset_bindings"]["reading_chair_1"]
    )
    anchor_basis = bridge._anchor_basis(
        catalog["frozen_asset_bindings"]["reading_chair_1"]
    )
    control = {"generation": {"asset_geometry_tolerance_m": 1.0e-4}}
    layouts = []
    for iteration, x_position in enumerate((1.0, 1.5)):
        path = write_json(
            tmp_path / f"layout_{iteration}.json",
            {
                "roomsize": [4.0, 4.0],
                "objects": {
                    "reading_chair_1": {
                        "asset_id": "chair.asset",
                        "size": [0.8, 0.7, 1.0],
                        "location": [x_position, 2.0, 0.0],
                        "rotation": [0.0, 0.0, 0.0],
                    }
                },
            },
        )
        layouts.append((iteration, path))

    observation, bindings = bridge._observe_trajectory(
        layouts=layouts,
        control=control,
        catalog=catalog,
        request={
            "benchmark_room": {
                "roomsize": [4.0, 4.0],
                "height": 3.0,
                "unit": "meter",
            }
        },
        plugin_report={
            "native_room_observation": {
                "roomsize": [4.0, 4.0],
                "height": 3.0,
                "unit": "meter",
            },
            "iteration_asset_observations": [
                {
                    "iteration": iteration,
                    "objects": {
                        "reading_chair_1": {
                            "asset_id": "chair.asset",
                            "mesh_path": mesh.as_posix(),
                            "mesh_sha256": mesh_sha256,
                            "canonical_local_bbox_size": [
                                0.80005 if iteration == 1 else 0.8,
                                0.7,
                                1.0,
                            ],
                            "orientation_basis": orientation_basis,
                            "anchor_basis": anchor_basis,
                            "full_precision_native_euler_xyz": [
                                0.0,
                                0.0,
                                0.0039 if iteration == 1 else 0.0,
                            ],
                        }
                    },
                }
                for iteration in range(2)
            ]
        },
        tolerance=1.0e-6,
    )

    assert observation["valid"] is True
    assert [row["iteration"] for row in observation["iterations"]] == [0, 1]
    assert bindings["reading_chair_1"]["asset_key"] == "chair.asset"
    assert bindings["reading_chair_1"][
        "full_precision_native_euler_xyz_by_iteration"
    ]["1"] == [0.0, 0.0, 0.0039]
    assert bindings["reading_chair_1"][
        "full_precision_native_local_bbox_size_by_iteration"
    ]["1"] == [0.80005, 0.7, 1.0]
    wrong_room_report = deepcopy(
        {
            "native_room_observation": {
                "roomsize": [4.0, 4.0],
                "height": 2.9,
                "unit": "meter",
            },
            "iteration_asset_observations": [
                {
                    "iteration": iteration,
                    "objects": {
                        "reading_chair_1": {
                            "asset_id": "chair.asset",
                            "mesh_path": mesh.as_posix(),
                            "mesh_sha256": mesh_sha256,
                            "canonical_local_bbox_size": [0.8, 0.7, 1.0],
                            "orientation_basis": orientation_basis,
                            "anchor_basis": anchor_basis,
                            "full_precision_native_euler_xyz": [0.0, 0.0, 0.0],
                        }
                    },
                }
                for iteration in range(2)
            ],
        }
    )
    wrong_room, _ = bridge._observe_trajectory(
        layouts=layouts,
        control=control,
        catalog=catalog,
        request={
            "benchmark_room": {
                "roomsize": [4.0, 4.0],
                "height": 3.0,
                "unit": "meter",
            }
        },
        plugin_report=wrong_room_report,
        tolerance=1.0e-6,
    )
    assert "native_solver_room_height_mismatch" in wrong_room["violations"]
    changed = read_json(layouts[1][1])
    changed["objects"]["reading_chair_1"]["asset_id"] = "replacement.asset"
    write_json(layouts[1][1], changed)
    observation, _ = bridge._observe_trajectory(
        layouts=layouts,
        control=control,
        catalog=catalog,
        request={
            "benchmark_room": {
                "roomsize": [4.0, 4.0],
                "height": 3.0,
                "unit": "meter",
            }
        },
        plugin_report={
            "native_room_observation": {
                "roomsize": [4.0, 4.0],
                "height": 3.0,
                "unit": "meter",
            },
            "iteration_asset_observations": [
                {
                    "iteration": iteration,
                    "objects": {
                        "reading_chair_1": {
                            "asset_id": "chair.asset",
                            "mesh_path": mesh.as_posix(),
                            "mesh_sha256": mesh_sha256,
                            "canonical_local_bbox_size": [0.8, 0.7, 1.0],
                            "orientation_basis": orientation_basis,
                            "anchor_basis": anchor_basis,
                            "full_precision_native_euler_xyz": [0.0, 0.0, 0.0],
                        }
                    },
                }
                for iteration in range(2)
            ]
        },
        tolerance=1.0e-6,
    )
    assert observation["valid"] is False
    assert "iteration_1:asset_replaced:reading_chair_1" in observation[
        "violations"
    ]

    replacement_mesh = tmp_path / "same_size_replacement.glb"
    replacement_mesh.write_bytes(b"different mesh bytes")
    unchanged = read_json(layouts[1][1])
    unchanged["objects"]["reading_chair_1"]["asset_id"] = "chair.asset"
    write_json(layouts[1][1], unchanged)
    replacement_report = {
        "native_room_observation": {
            "roomsize": [4.0, 4.0],
            "height": 3.0,
            "unit": "meter",
        },
        "iteration_asset_observations": [
            {
                "iteration": iteration,
                "objects": {
                    "reading_chair_1": {
                        "asset_id": "chair.asset",
                        "mesh_path": (
                            replacement_mesh.as_posix()
                            if iteration == 1
                            else mesh.as_posix()
                        ),
                        "mesh_sha256": (
                            file_sha256(replacement_mesh)
                            if iteration == 1
                            else mesh_sha256
                        ),
                        "canonical_local_bbox_size": [0.8, 0.7, 1.0],
                        "orientation_basis": orientation_basis,
                        "anchor_basis": anchor_basis,
                        "full_precision_native_euler_xyz": [0.0, 0.0, 0.0],
                    }
                },
            }
            for iteration in range(2)
        ]
    }
    observation, _ = bridge._observe_trajectory(
        layouts=layouts,
        control=control,
        catalog=catalog,
        request={
            "benchmark_room": {
                "roomsize": [4.0, 4.0],
                "height": 3.0,
                "unit": "meter",
            }
        },
        plugin_report=replacement_report,
        tolerance=1.0e-6,
    )
    assert observation["valid"] is False
    assert "iteration_1:frozen_mesh_mismatch:reading_chair_1" in observation[
        "violations"
    ]


def _bridge(name: str):
    if BRIDGES.as_posix() not in sys.path:
        sys.path.insert(0, BRIDGES.as_posix())
    path = BRIDGES / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _catalog() -> CanonicalAssetCatalog:
    return CanonicalAssetCatalog.from_mapping(
        {
            "catalog_id": "fixture",
            "catalog_version": "1",
            "assets": [
                {
                    "asset_id": "chair.asset",
                    "source_db": "imaginarium",
                    "category": "Single_sofa_chair",
                    "description": "fixture chair",
                    "bbox_size_local": [0.8, 0.7, 1.0],
                    "bbox_center_local": [0.0, 0.0, 0.5],
                    "native_scale": [1.0, 1.0, 1.0],
                }
            ],
        }
    )


def _protocol(catalog: CanonicalAssetCatalog) -> ComparisonProtocol:
    return ComparisonProtocol.from_mapping(
        {
            "protocol_id": "generation_comparison_v1",
            "protocol_version": 1,
            "mode": "frozen_assets",
            "case_id": "case",
            "architecture": {
                "room_model": "single_room",
                "boundary_model": "axis_aligned_rectangle",
                "room": {
                    "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                    "height": 3,
                    "unit": "meter",
                },
            },
            "object_inventory_policy": "frozen",
            "objects": [
                {
                    "slot_id": "reading_chair_1",
                    "category": "Single_sofa_chair",
                    "description": "fixture chair",
                    "asset_id": "chair.asset",
                }
            ],
            "asset_policy": "frozen_exact",
            "assets": catalog.identity,
            "scale_policy": "fixed_native_scale",
            "retrieval_policy": "disabled_exact_bindings",
            "generation": {"asset_geometry_tolerance_m": 1.0e-4},
            "evaluator": {"policy": "same_canonical_run_evaluate"},
        }
    )


def _direct_generation_input() -> dict:
    return build_generation_input(
        scene_request=build_scene_request(
            request_id="case",
            instruction="Arrange a chair.",
            scene_type="reading_room",
            room={
                "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
                "height": 3,
                "unit": "meter",
            },
            structure=False,
        )
    )
