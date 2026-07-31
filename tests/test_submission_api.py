from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from PIL import Image

from benchmark.api.submission import (
    CaseBundleError,
    SubmissionEvaluationError,
    evaluate_submission,
    load_case_bundle,
)
from benchmark.evaluator.profile import L1, L2, L3, L4, resolve_evaluation_profile
from benchmark.utils.io import read_json, write_json


ROOT = Path(__file__).resolve().parents[1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene_request(*, granularity: str = "fine_grained") -> dict:
    return {
        "request_id": "trusted_request",
        "instruction": "Place one blue bed in the room.",
        "scene_type": "bedroom",
        "structure": False,
        "prompt_granularity": granularity,
        "room": {
            "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "height": 3.0,
            "unit": "meter",
        },
    }


def _reference_annotation() -> dict:
    return {
        "annotation_version": "reference_annotation_v1",
        "validation_status": "confirmed",
        "source": "manual",
        "request_id": "trusted_request",
        "scene_type": "bedroom",
        "inventory_policy": "closed_world",
        "objects": [
            {
                "id": "bed_claim",
                "category": "bed",
                "description": "blue bed",
                "count": 1,
                "claim_state": "confirmed",
            }
        ],
        "oor_relations": [],
        "oar_relations": [],
        "room_constraints": {"claim_state": "not_mentioned"},
    }


def _empty_contract() -> dict:
    return {
        "contract_version": "specification_contract_v1",
        "source": "trusted_case_bundle",
        "frozen": True,
        "request_id": "trusted_request",
        "claims": {
            "oor": [],
            "oar": [],
            "functional_semantic_fidelity": [],
        },
    }


def _scene(*, generated_mesh: bool = False) -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "trusted_scene",
        "request_id": "trusted_request",
        "scene_type": "bedroom",
        "boundary": [[0, 0], [5, 0], [5, 5], [0, 5]],
        "scene_height": 3.0,
        "objects": [
            {
                "id": "generated_bed",
                "category": "bed",
                "description": "blue bed",
                "size": [2.0, 1.6, 0.6],
                "center": [2.5, 2.5, 0.3],
                "rotation": [0, 0, 0],
                "geometry_provenance": (
                    "generated_mesh" if generated_mesh else "bbox_proxy"
                ),
                "metadata": {"interactive": False},
            }
        ],
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            }
        },
    }


def _proxy_profile() -> dict:
    profile = resolve_evaluation_profile()
    profile["layer_weights"] = {L1: 1.0, L2: 0.0, L3: 0.0, L4: 0.0}
    profile[L3]["enabled"] = False
    for metric in profile[L3]["metrics"].values():
        metric["enabled"] = False
        metric["weight"] = 0.0
    return resolve_evaluation_profile(profile)


def _generator_asset_policy() -> dict:
    return {
        "mode": "generated_or_open_assets",
        "identity_owner": "generator",
        "category_selection_owner": "generator",
        "scale_owner": "generator",
        "appearance_owner": "generator",
        "arrangement_owner": "generator",
    }


def _write_bundle(
    tmp_path: Path,
    *,
    output_type: str = "o1_object_state",
    profile: dict | None = None,
    include_contract: bool = True,
    granularity: str = "fine_grained",
) -> Path:
    root = tmp_path / "case"
    root.mkdir(parents=True)
    paths = {
        "scene_request": write_json(
            root / "scene_request.json",
            _scene_request(granularity=granularity),
        ),
        "reference_annotation": write_json(
            root / "reference_annotation.json", _reference_annotation()
        ),
        "evaluation_profile": write_json(
            root / "evaluation_profile.json",
            profile or resolve_evaluation_profile(),
        ),
    }
    if include_contract:
        paths["specification_contract"] = write_json(
            root / "specification_contract.json", _empty_contract()
        )
    manifest = {
        "bundle_version": "benchmark_case_bundle_v1",
        "case_id": "trusted_case",
        "task": {"evaluator_output_type": output_type},
        "artifacts": {
            name: {"path": path.name, "sha256": _digest(path)}
            for name, path in paths.items()
        },
        "evaluation": {
            "workflow": "canonical_l0_l4",
            "p0b_official_mode": True,
            "camera_evidence": {
                "mode": None,
                "metric_modes": {},
                "max_views": 2,
                "max_steps": 0,
                "collision_overlay": True,
                "collision_contour": True,
            },
        },
    }
    write_json(root / "case_bundle.json", manifest)
    return root


def _add_artifact(root: Path, name: str, value: object) -> Path:
    path = write_json(root / f"{name}.json", value)
    manifest_path = root / "case_bundle.json"
    manifest = read_json(manifest_path)
    manifest["artifacts"][name] = {"path": path.name, "sha256": _digest(path)}
    write_json(manifest_path, manifest)
    return path


def _add_fixed_catalog(root: Path, asset_ids: list[str]) -> None:
    allowed = _add_artifact(root, "allowed_asset_ids", {"asset_ids": asset_ids})
    assert allowed.is_file()
    manifest_path = root / "case_bundle.json"
    manifest = read_json(manifest_path)
    manifest["asset_catalog"] = {"snapshot_id": "test_catalog_v1"}
    write_json(manifest_path, manifest)


class _FakeTrustedRenderer:
    def render_scene(self, *, scene_path: Path, out_dir: Path, asset_root=None) -> dict:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        views = []
        for name in ("top", "perspective"):
            path = destination / f"standardized_{name}.png"
            image = Image.new("RGB", (2, 2), (40, 60, 80))
            image.putpixel((0, 0), (180, 120, 60))
            image.save(path)
            views.append({"name": name, "path": path.as_posix()})
        blend = destination / "scene.blend"
        blend.write_bytes(b"trusted blend")
        manifest = {
            "backend": "fake_trusted_renderer",
            "views": views,
            "objects": [{"id": "generated_bed", "representation": "test"}],
            "blend_file": blend.as_posix(),
        }
        write_json(destination / "render_manifest.json", manifest)
        return manifest


class _FakeJudge:
    def __init__(self) -> None:
        self.relation_requests: list[dict] = []
        self.scene_quality_requests: list[dict] = []

    def adjudicate_p0b(self, request: dict) -> dict:
        return {"verdict": "valid", "confidence": 1.0, "reason": "test"}

    def adjudicate_relation(self, request: dict) -> dict:
        self.relation_requests.append(request)
        return {"verdict": "valid", "confidence": 1.0, "reason": "test"}

    def adjudicate_scene_quality(self, request: dict) -> dict:
        self.scene_quality_requests.append(request)
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "no significant defect",
            "defects": [],
        }

    def adjudicate_functional_semantic(self, request: dict) -> dict:
        return {
            "evidence_status": "sufficient",
            "verdict": "valid",
            "confidence": 1.0,
            "reason": "prompt requirement satisfied",
            "defects": [],
        }


def test_case_bundle_schema_accepts_canonical_workflow() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "benchmark_case_bundle.schema.json").read_text()
    )
    Draft202012Validator.check_schema(schema)
    manifest = {
        "bundle_version": "benchmark_case_bundle_v1",
        "case_id": "case",
        "task": {"evaluator_output_type": "o1_object_state"},
        "artifacts": {
            "scene_request": {"path": "request.json", "sha256": "a" * 64},
            "evaluation_profile": {"path": "profile.json", "sha256": "b" * 64},
        },
        "evaluation": {
            "workflow": "canonical_l0_l4",
            "p0b_official_mode": True,
        },
    }
    validator = Draft202012Validator(schema)
    assert list(validator.iter_errors(manifest)) == []
    manifest["evaluation"]["enabled_evaluators"] = {
        "oor": True,
        "oar": True,
        "generic_validity": True,
    }
    assert list(validator.iter_errors(manifest))


def test_case_bundle_rejects_hash_drift(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    (root / "scene_request.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(CaseBundleError, match="hash mismatch"):
        load_case_bundle(root)


def test_canonical_bundle_rejects_duplicate_runtime_routing_flags(
    tmp_path: Path,
) -> None:
    root = _write_bundle(tmp_path)
    manifest_path = root / "case_bundle.json"
    manifest = read_json(manifest_path)
    manifest["evaluation"]["enabled_evaluators"] = {
        "oor": True,
        "oar": True,
        "generic_validity": True,
    }
    write_json(manifest_path, manifest)
    with pytest.raises(CaseBundleError, match="must not declare enabled_evaluators"):
        load_case_bundle(root)


def test_prompt_granularity_does_not_select_a_second_workflow(tmp_path: Path) -> None:
    fine = load_case_bundle(_write_bundle(tmp_path / "fine", granularity="fine_grained"))
    coarse = load_case_bundle(
        _write_bundle(tmp_path / "coarse", granularity="coarse_grained")
    )
    assert fine.workflow == coarse.workflow == "canonical_l0_l4"
    assert fine.specification_activation_mode == "specification_contract"
    assert coarse.specification_activation_mode == "specification_contract"


def test_canonical_bundle_compiles_contract_from_confirmed_annotation(
    tmp_path: Path,
) -> None:
    bundle = load_case_bundle(_write_bundle(tmp_path, include_contract=False))
    assert bundle.specification_contract is not None
    assert bundle.specification_contract["source"] == "benchmark_annotation"
    assert bundle.specification_contract["frozen"] is True


def test_canonical_bundle_rejects_untrusted_contract(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path, include_contract=False)
    contract = _empty_contract()
    contract["source"] = "programmatic"
    _add_artifact(root, "specification_contract", contract)
    with pytest.raises(CaseBundleError, match="benchmark-owned"):
        load_case_bundle(root)


def test_canonical_bundle_rejects_retired_spatial_ontology(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    _add_artifact(root, "spatial_fidelity_ontology", {"categories": {}})
    with pytest.raises(CaseBundleError, match="retired non-game workflow"):
        load_case_bundle(root)


def test_canonical_bundle_loads_hashed_module_inputs(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    _add_artifact(root, "functional_semantic_config", {"enabled": True})
    _add_artifact(root, "scene_quality_config", {"enabled": True})
    _add_artifact(root, "object_grouping_report", {"groups": []})
    _add_artifact(root, "asset_policy", _generator_asset_policy())
    _add_artifact(root, "authorized_deviations", [])
    bundle = load_case_bundle(root)
    assert bundle.functional_semantic_config == {"enabled": True}
    assert bundle.scene_quality_config == {"enabled": True}
    assert bundle.object_grouping_report == {"groups": []}
    assert bundle.asset_policy["mode"] == "generated_or_open_assets"
    assert bundle.authorized_deviations == []


def test_official_proxy_case_uses_one_canonical_report(tmp_path: Path) -> None:
    bundle = load_case_bundle(_write_bundle(tmp_path, profile=_proxy_profile()))
    result = evaluate_submission(
        scene=_scene(),
        case_bundle=bundle,
        out_dir=tmp_path / "run",
        renderer=_FakeTrustedRenderer(),
        vlm_judge=_FakeJudge(),
        official_mode=True,
    )
    report = result["evaluation_report"]
    assert report["workflow"] == "canonical_l0_l4"
    assert set(report["layer_reports"]) == {
        "l0_structural_validity",
        "l1_physical_plausibility",
        "l2_specification_fidelity",
        "l3_scene_quality",
        "l4_downstream_task_functionality",
    }
    assert not {
        "prompt_fidelity",
        "spatial_fidelity",
        "structural_validity",
        "visual_quality",
    } & set(report["layer_reports"])
    assert report["evidence_provenance"]["render_input_policy"] == (
        "trusted_bbox_proxy_projection"
    )
    assert result["manifest"]["generator"]["invoked"] is False
    schema = read_json(ROOT / "schemas" / "evaluation_report.schema.json")
    Draft202012Validator(schema).validate(report)


def test_official_proxy_case_rejects_active_l3(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    _add_artifact(root, "asset_policy", _generator_asset_policy())
    bundle = load_case_bundle(root)
    with pytest.raises(SubmissionEvaluationError, match="cannot score active L3"):
        evaluate_submission(
            scene=_scene(),
            case_bundle=bundle,
            out_dir=tmp_path / "proxy_l3",
            renderer=_FakeTrustedRenderer(),
            vlm_judge=_FakeJudge(),
            official_mode=True,
        )


def test_official_proxy_case_allows_no_applicable_l3(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    _add_artifact(
        root,
        "asset_policy",
        {
            "mode": "benchmark_provided",
            "identity_owner": "benchmark",
            "category_selection_owner": "benchmark",
            "scale_owner": "benchmark",
            "appearance_owner": "benchmark",
            "arrangement_owner": "benchmark",
        },
    )
    bundle = load_case_bundle(root)
    result = evaluate_submission(
        scene=_scene(),
        case_bundle=bundle,
        out_dir=tmp_path / "proxy_no_applicable_l3",
        renderer=_FakeTrustedRenderer(),
        vlm_judge=_FakeJudge(),
        official_mode=True,
    )
    assert result["evaluation_report"]["layer_reports"][L3]["status"] == (
        "not_applicable"
    )


def test_generator_authored_geometry_preserves_l3_signal(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    _add_artifact(root, "asset_policy", _generator_asset_policy())
    bundle = load_case_bundle(root)
    judge = _FakeJudge()
    result = evaluate_submission(
        scene=_scene(generated_mesh=True),
        case_bundle=bundle,
        out_dir=tmp_path / "mesh_l3",
        renderer=_FakeTrustedRenderer(),
        vlm_judge=judge,
        official_mode=False,
    )
    assert result["evaluation_report"]["workflow"] == "canonical_l0_l4"
    assert result["manifest"]["rendering"]["input_policy"] == (
        "trusted_generator_authored_geometry"
    )
    assert judge.scene_quality_requests


def test_o3_case_bundle_still_requires_fixed_catalog(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path, output_type="o3_scene_package")
    with pytest.raises(CaseBundleError, match="fixed asset catalog"):
        load_case_bundle(root)


def test_o3_submission_discards_untrusted_asset_paths(tmp_path: Path) -> None:
    root = _write_bundle(
        tmp_path,
        output_type="o3_scene_package",
        profile=_proxy_profile(),
    )
    _add_fixed_catalog(root, ["catalog_bed"])
    bundle = load_case_bundle(root)
    scene = deepcopy(_scene())
    scene["objects"][0].update(
        {
            "jid": "substituted_mesh",
            "category": "spoofed_sofa",
            "description": "spoofed sofa",
            "geometry_provenance": "generated_mesh",
            "asset_ref": {
                "source_db": "imaginarium",
                "asset_key": "catalog_bed",
                "mesh_uri": "/tmp/untrusted_mesh.fbx",
                "metadata_uri": "/tmp/untrusted_metadata.json",
            },
        }
    )
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    asset_csv = tmp_path / "assets.csv"
    asset_csv.write_text(
        "name_en,class_en,retrieval_class_en,caption_en,short_desc\n"
        "catalog_bed,bed,bed,a blue catalog bed,blue bed\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "o3"
    result = evaluate_submission(
        scene=scene,
        case_bundle=bundle,
        out_dir=out_dir,
        renderer=_FakeTrustedRenderer(),
        vlm_judge=_FakeJudge(),
        asset_root=asset_root,
        asset_csv=asset_csv,
        official_mode=True,
    )
    obj = read_json(out_dir / "render_input_scene.json")["objects"][0]
    assert obj["jid"] == "catalog_bed"
    assert obj["asset_ref"] == {
        "source_db": "fixed_catalog",
        "asset_key": "catalog_bed",
    }
    assert result["manifest"]["submission"]["normalization_policy"] == (
        "validated_fixed_catalog_scene_package"
    )
