from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmark.api.submission import (
    CaseBundleError,
    SubmissionEvaluationError,
    evaluate_submission,
    load_case_bundle,
)
from benchmark.evaluator.profile import resolve_evaluation_profile
from benchmark.utils.io import read_json, write_json


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene_request() -> dict:
    return {
        "request_id": "trusted_request",
        "instruction": "Place one blue bed in the room.",
        "scene_type": "bedroom",
        "structure": False,
        "prompt_granularity": "fine_grained",
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


def _scene() -> dict:
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
                "geometry_provenance": "bbox_proxy",
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


def _write_bundle(tmp_path: Path, *, output_type: str = "o1_object_state") -> Path:
    root = tmp_path / "case"
    root.mkdir()
    profile = resolve_evaluation_profile()
    if output_type == "o1_object_state":
        profile["weights"] = {
            "prompt_fidelity": 0.4,
            "spatial_fidelity": 0.4,
            "structural_validity": 0.6,
            "visual_quality": 0.0,
        }
    paths = {
        "scene_request": write_json(root / "scene_request.json", _scene_request()),
        "reference_annotation": write_json(root / "reference_annotation.json", _reference_annotation()),
        "evaluation_profile": write_json(
            root / "evaluation_profile.json",
            profile,
        ),
    }
    artifacts = {
        name: {"path": path.name, "sha256": _digest(path)}
        for name, path in paths.items()
    }
    manifest = {
        "bundle_version": "benchmark_case_bundle_v1",
        "case_id": "trusted_case",
        "task": {"evaluator_output_type": output_type},
        "artifacts": artifacts,
        "evaluation": {
            "enabled_evaluators": {
                "oor": True,
                "oar": True,
                "generic_validity": True,
            },
            "p0b_official_mode": True,
            "camera_evidence": {
                "mode": None,
                "metric_modes": {},
                "max_views": 2,
                "max_steps": 0,
                "collision_overlay": True,
            },
        },
    }
    write_json(root / "case_bundle.json", manifest)
    return root


def _spatial_ontology() -> dict:
    return {
        "schema_version": "sceneonto_test_v1",
        "categories": {
            "bed": {
                "count": 500,
                "dimensions": {
                    "width": {"p5": 1.8, "p95": 2.2, "median": 2.0, "n": 500},
                    "depth": {"p5": 1.4, "p95": 1.8, "median": 1.6, "n": 500},
                    "height": {"p5": 0.4, "p95": 0.8, "median": 0.6, "n": 500},
                },
                "cooccurrence": {"global": {}},
            }
        },
    }


def _write_coarse_bundle(tmp_path: Path, *, include_ontology: bool = True) -> Path:
    root = tmp_path / "case"
    root.mkdir(parents=True)
    request = _scene_request()
    request.update(
        {
            "instruction": "Create a comfortable bedroom.",
            "prompt_granularity": "coarse_grained",
        }
    )
    profile = resolve_evaluation_profile()
    profile["weights"] = {
        "prompt_fidelity": 0.4,
        "spatial_fidelity": 0.4,
        "structural_validity": 0.6,
        "visual_quality": 0.0,
    }
    paths = {
        "scene_request": write_json(root / "scene_request.json", request),
        "evaluation_profile": write_json(root / "evaluation_profile.json", profile),
    }
    if include_ontology:
        paths["spatial_fidelity_ontology"] = write_json(
            root / "spatial_fidelity_ontology.json",
            _spatial_ontology(),
        )
    manifest = {
        "bundle_version": "benchmark_case_bundle_v1",
        "case_id": "trusted_coarse_case",
        "task": {"evaluator_output_type": "o1_object_state"},
        "artifacts": {
            name: {"path": path.name, "sha256": _digest(path)}
            for name, path in paths.items()
        },
        "evaluation": {
            "enabled_evaluators": {
                "oor": False,
                "oar": False,
                "generic_validity": True,
            },
            "p0b_official_mode": True,
            "camera_evidence": {
                "mode": None,
                "metric_modes": {},
                "max_views": 2,
                "max_steps": 0,
                "collision_overlay": True,
            },
        },
    }
    write_json(root / "case_bundle.json", manifest)
    return root


def _add_fixed_catalog(root: Path, asset_ids: list[str]) -> None:
    allowed_path = write_json(root / "allowed_asset_ids.json", {"asset_ids": asset_ids})
    manifest_path = root / "case_bundle.json"
    manifest = read_json(manifest_path)
    manifest["artifacts"]["allowed_asset_ids"] = {
        "path": allowed_path.name,
        "sha256": _digest(allowed_path),
    }
    manifest["asset_catalog"] = {"snapshot_id": "test_catalog_v1"}
    write_json(manifest_path, manifest)


class _FakeTrustedRenderer:
    def render_scene(self, *, scene_path: Path, out_dir: Path, asset_root=None) -> dict:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        top = destination / "standardized_top.png"
        perspective = destination / "standardized_perspective.png"
        top.write_bytes(b"trusted top evidence")
        perspective.write_bytes(b"trusted perspective evidence")
        blend = destination / "scene.blend"
        blend.write_bytes(b"trusted blend")
        manifest = {
            "backend": "fake_trusted_renderer",
            "views": [
                {"name": "top", "path": top.as_posix()},
                {"name": "perspective", "path": perspective.as_posix()},
            ],
            "objects": [{"id": "generated_bed", "representation": "bbox_proxy"}],
            "blend_file": blend.as_posix(),
        }
        write_json(destination / "render_manifest.json", manifest)
        return manifest


class _FakeJudge:
    def __init__(self) -> None:
        self.p0b_requests: list[dict] = []
        self.relation_requests: list[dict] = []

    def evaluate(self, request: dict) -> dict:
        return {
            "applicable": True,
            "score": 0.8,
            "confidence": 1.0,
            "summary": "test judgement",
        }

    def adjudicate_p0b(self, request: dict) -> dict:
        self.p0b_requests.append(request)
        return {"verdict": "valid", "confidence": 1.0, "reason": "test judgement"}

    def adjudicate_relation(self, request: dict) -> dict:
        self.relation_requests.append(request)
        return {"verdict": "valid", "confidence": 1.0, "reason": "test judgement"}


def test_case_bundle_rejects_hash_drift(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    (root / "scene_request.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(CaseBundleError, match="hash mismatch"):
        load_case_bundle(root)


def test_coarse_case_bundle_requires_hash_verified_spatial_ontology(tmp_path: Path) -> None:
    missing_root = _write_coarse_bundle(tmp_path / "missing", include_ontology=False)
    with pytest.raises(CaseBundleError, match="spatial_fidelity_ontology artifact"):
        load_case_bundle(missing_root)

    drift_root = _write_coarse_bundle(tmp_path / "drift")
    ontology_path = drift_root / "spatial_fidelity_ontology.json"
    ontology = read_json(ontology_path)
    ontology["categories"]["bed"]["count"] = 501
    write_json(ontology_path, ontology)
    with pytest.raises(CaseBundleError, match="hash mismatch"):
        load_case_bundle(drift_root)


def test_official_coarse_submission_uses_only_hash_verified_spatial_fidelity(
    tmp_path: Path,
) -> None:
    root = _write_coarse_bundle(tmp_path)
    ontology_path = root / "spatial_fidelity_ontology.json"
    ontology_sha256 = _digest(ontology_path)
    bundle = load_case_bundle(root)

    assert bundle.reference_annotation is None
    assert bundle.spatial_fidelity_ontology_path == ontology_path
    assert bundle.artifact_records["spatial_fidelity_ontology"] == {
        "path": "spatial_fidelity_ontology.json",
        "sha256": ontology_sha256,
    }

    out_dir = tmp_path / "coarse_run"
    result = evaluate_submission(
        scene=_scene(),
        case_bundle=bundle,
        out_dir=out_dir,
        renderer=_FakeTrustedRenderer(),
        vlm_judge=_FakeJudge(),
        official_mode=True,
    )

    report = result["evaluation_report"]
    assert report["prompt_granularity"] == "coarse_grained"
    assert report["evaluation_mode"] == "coarse_grained_mode"
    assert set(report["category_reports"]) == {
        "spatial_fidelity",
        "structural_validity",
        "visual_quality",
    }
    assert "prompt_fidelity" not in report["category_reports"]
    spatial = report["category_reports"]["spatial_fidelity"]
    assert spatial["status"] == "checked"
    assert spatial["score"] == 1.0
    assert spatial["ontology_identity"]["sha256"] == ontology_sha256
    assert spatial["ontology_identity"]["source"] == ontology_path.resolve().as_posix()
    assert report["reports"]["spatial_fidelity"] == spatial
    assert report["case_bundle"]["spatial_fidelity_ontology_sha256"] == ontology_sha256
    assert report["case_bundle"]["artifact_records"]["spatial_fidelity_ontology"] == {
        "path": "spatial_fidelity_ontology.json",
        "sha256": ontology_sha256,
    }
    assert report["evidence_provenance"]["spatial_fidelity_ontology"] == (
        "benchmark_hash_verified"
    )
    assert result["manifest"]["case_bundle"]["spatial_fidelity_ontology_sha256"] == (
        ontology_sha256
    )
    assert read_json(out_dir / "evaluation_report.json")["case_bundle"] == report[
        "case_bundle"
    ]


def test_official_submission_skips_generator_and_owns_evidence(tmp_path: Path) -> None:
    bundle = load_case_bundle(_write_bundle(tmp_path))
    out_dir = tmp_path / "run"

    result = evaluate_submission(
        scene=_scene(),
        case_bundle=bundle,
        out_dir=out_dir,
        renderer=_FakeTrustedRenderer(),
        vlm_judge=_FakeJudge(),
        official_mode=True,
    )

    report = result["evaluation_report"]
    manifest = result["manifest"]
    assert report["benchmark_score_status"] == "complete"
    assert report["protocol_scope"] == "official_submission"
    assert report["official_submission"] is True
    assert report["category_reports"]["prompt_fidelity"]["score"] == 1.0
    assert report["evidence_provenance"] == {
        "render_evidence": "benchmark_generated",
        "collision_geometry": "not_available",
        "submitted_evidence_accepted": False,
        "spatial_fidelity_ontology": "not_applicable",
    }
    assert manifest["generator"] == {
        "invoked": False,
        "stage": "skipped_by_submission_protocol",
    }
    render_input = read_json(out_dir / "render_input_scene.json")
    assert "asset_ref" not in render_input["objects"][0]
    assert "jid" not in render_input["objects"][0]
    assert render_input["objects"][0]["geometry_provenance"] == "bbox_proxy"
    assert manifest["rendering"]["input_policy"] == "trusted_bbox_proxy_projection"
    assert all(str(out_dir / "renders") in path for path in manifest["rendering"]["overview_views"])
    persisted = read_json(out_dir / "evaluation_report.json")
    assert persisted["case_bundle"]["manifest_sha256"] == bundle.manifest_sha256


def test_official_submission_routes_attachment_relation_with_detector_evidence(
    tmp_path: Path,
) -> None:
    root = _write_bundle(tmp_path)
    annotation_path = root / "reference_annotation.json"
    annotation = read_json(annotation_path)
    annotation["oar_relations"] = [
        {
            "relation_id": "oar_bed_wall",
            "subject_id": "bed_claim",
            "type": "mounted_on_wall",
            "architectural_element": "west_wall",
            "claim_state": "confirmed",
        }
    ]
    write_json(annotation_path, annotation)
    manifest_path = root / "case_bundle.json"
    manifest = read_json(manifest_path)
    manifest["artifacts"]["reference_annotation"]["sha256"] = _digest(annotation_path)
    write_json(manifest_path, manifest)
    bundle = load_case_bundle(root)
    judge = _FakeJudge()

    result = evaluate_submission(
        scene=_scene(),
        case_bundle=bundle,
        out_dir=tmp_path / "relation_run",
        renderer=_FakeTrustedRenderer(),
        vlm_judge=judge,
        official_mode=True,
    )

    assert result["evaluation_report"]["benchmark_score_status"] == "complete"
    assert len(judge.relation_requests) == 1
    request = judge.relation_requests[0]
    assert request["family"] == "oar"
    assert request["relation"]["relation_id"] == "oar_bed_wall"
    assert request["detector_evidence"]["proxy"] == "obb_to_wall_attachment"
    check = result["evaluation_report"]["reports"]["oar"]["checks"][0]
    assert check["route"] == "vlm_adjudicated"
    assert check["relation_id"] == "oar_bed_wall"


def test_official_submission_requires_trusted_renderer_and_judge(tmp_path: Path) -> None:
    bundle = load_case_bundle(_write_bundle(tmp_path))

    with pytest.raises(SubmissionEvaluationError, match="trusted renderer"):
        evaluate_submission(
            scene=_scene(),
            case_bundle=bundle,
            out_dir=tmp_path / "no_renderer",
            official_mode=True,
        )


def test_official_o1_rejects_proxy_visual_quality_weight(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path)
    profile_path = root / "evaluation_profile.json"
    profile = read_json(profile_path)
    profile["weights"] = {
        "prompt_fidelity": 0.4,
        "spatial_fidelity": 0.4,
        "structural_validity": 0.5,
        "visual_quality": 0.1,
    }
    write_json(profile_path, profile)
    manifest_path = root / "case_bundle.json"
    manifest = read_json(manifest_path)
    manifest["artifacts"]["evaluation_profile"]["sha256"] = _digest(profile_path)
    write_json(manifest_path, manifest)
    bundle = load_case_bundle(root)

    with pytest.raises(SubmissionEvaluationError, match="visual_quality weight to 0"):
        evaluate_submission(
            scene=_scene(),
            case_bundle=bundle,
            out_dir=tmp_path / "invalid_o1_visual_weight",
            renderer=_FakeTrustedRenderer(),
            vlm_judge=_FakeJudge(),
            official_mode=True,
        )


def test_o3_case_bundle_requires_fixed_catalog(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path, output_type="o3_scene_package")

    with pytest.raises(CaseBundleError, match="fixed asset catalog"):
        load_case_bundle(root)


def test_official_o3_ignores_submitted_paths_and_binds_asset_key(tmp_path: Path) -> None:
    root = _write_bundle(tmp_path, output_type="o3_scene_package")
    _add_fixed_catalog(root, ["catalog_bed"])
    bundle = load_case_bundle(root)
    scene = _scene()
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
    asset_csv = tmp_path / "imaginarium_asset_info.csv"
    asset_csv.write_text(
        "name_en,class_en,retrieval_class_en,caption_en,short_desc\n"
        "catalog_bed,bed,bed,a blue catalog bed,blue bed\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "o3_run"

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

    canonical_obj = read_json(out_dir / "generated_scene.json")["objects"][0]
    render_obj = read_json(out_dir / "render_input_scene.json")["objects"][0]
    for obj in (canonical_obj, render_obj):
        assert obj["jid"] == "catalog_bed"
        assert obj["asset_ref"] == {
            "source_db": "fixed_catalog",
            "asset_key": "catalog_bed",
        }
        assert obj["geometry_provenance"] == "asset_mesh"
        assert obj["category"] == "bed"
        assert obj["description"] == "a blue catalog bed"
    assert result["manifest"]["submission"]["normalization_policy"] == (
        "fixed_catalog_asset_key_projection"
    )
