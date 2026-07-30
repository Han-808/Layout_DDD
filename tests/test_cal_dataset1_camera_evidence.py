from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_cal_dataset1_camera_evidence",
    ROOT / "scripts" / "run_cal_dataset1_camera_evidence.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_distorted_evidence_plan_is_balanced_by_metric_and_severity() -> None:
    cases = MODULE._selected_events(
        ROOT / "Support" / "datasets" / "cal_dataset1",
        splits=set(MODULE.DISTORTION_SPLITS),
        metrics=set(MODULE.METRICS),
    )
    events = [event for case in cases for event in case["events"]]

    assert len(cases) == 16
    assert len(events) == 24
    assert {metric: sum(event["metric"] == metric for event in events) for metric in MODULE.METRICS} == {
        "collision": 8,
        "oob": 8,
        "support": 8,
    }
    assert {severity: sum(event["severity_class"] == severity for event in events) for severity in ("obvious", "subtle")} == {
        "obvious": 12,
        "subtle": 12,
    }


def test_event_context_preserves_metric_specific_detector_fields() -> None:
    collision_gt = {
        "metric": "collision",
        "object_ids": ["a", "b"],
    }
    collision_record = {
        "evidence_level": "mesh",
        "obb_evidence": {"intersects": True},
        "mesh_evidence": {"focus_region": {"center": [0, 0, 0]}},
        "diagnostics": {"xy_overlap_area": 1.0},
    }
    detector, event = MODULE._event_context("collision", collision_record, collision_gt)
    assert event["object_a"] == "a"
    assert event["object_b"] == "b"
    assert detector["obb"]["intersects"] is True
    assert detector["focus_region"]["center"] == [0, 0, 0]

    support_gt = {"metric": "support", "object_ids": ["subject"]}
    support_record = {
        "representative_samples": [{"position": [1, 2, 0], "gap_m": 0.2}],
        "candidate_support_object_ids": ["table"],
    }
    detector, event = MODULE._event_context("support", support_record, support_gt)
    assert detector["representative_ray_hits"] == support_record["representative_samples"]
    assert event["object_ids"] == ["subject", "table"]


def test_experiment_arms_hold_budget_and_presentation_fixed() -> None:
    assert MODULE.ARMS == ("fixed_global_highlight", "metric_local_highlight")
    assert MODULE.METRIC_MODES == {
        "collision": "visibility_ranked",
        "oob": "visibility_ranked",
        "support": "support_contact_plane",
    }


def test_comparison_resume_requires_source_and_image_content_identity(tmp_path: Path) -> None:
    source = {"scene": "scene-hash", "camera_policy_config": "policy-hash"}
    images = []
    for index in range(8):
        path = tmp_path / f"image_{index}.png"
        path.write_bytes(f"image-{index}".encode())
        images.append(
            {
                "path": str(path),
                "sha256": MODULE._file_sha256(path),
            }
        )
    camera_manifest = tmp_path / "camera_evidence_manifest.json"
    camera_manifest.write_text('{"selection":{}}', encoding="utf-8")
    comparison = {
        "schema_version": MODULE.COMPARISON_SCHEMA_VERSION,
        "source_sha256": source,
        "arms": {
            "fixed_global_highlight": {
                "image_count": 4,
                "items": images[:4],
            },
            "metric_local_highlight": {
                "image_count": 4,
                "items": images[4:],
                "camera_evidence_manifest": str(camera_manifest),
                "camera_evidence_manifest_sha256": MODULE._file_sha256(camera_manifest),
            },
        },
    }
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    assert MODULE._comparison_ready(
        comparison_path,
        expected_source_sha256=source,
    )
    assert not MODULE._comparison_ready(
        comparison_path,
        expected_source_sha256={**source, "scene": "different"},
    )
    Path(images[0]["path"]).write_bytes(b"drift")
    assert not MODULE._comparison_ready(
        comparison_path,
        expected_source_sha256=source,
    )


def test_materialization_resume_detects_blend_manifest_and_asset_drift(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    blend = scene_dir / "scene.blend"
    blend.write_bytes(b"blend")
    asset = tmp_path / "asset.fbx"
    asset.write_bytes(b"asset")
    render_manifest = scene_dir / "render_manifest.json"
    render_manifest.write_text(
        json.dumps(
            {
                "blender_version": "test",
                "objects": [{"id": "object", "mesh_path": str(asset)}],
            }
        ),
        encoding="utf-8",
    )
    base = {
        "schema_version": MODULE.MATERIALIZATION_SCHEMA_VERSION,
        "scene_sha256": "scene",
    }
    provenance_path = scene_dir / "materialization_provenance.json"
    provenance = MODULE._complete_materialization_provenance(
        base,
        scene_dir=scene_dir,
        blend_file=blend,
    )
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    assert MODULE._materialization_ready(
        blend_file=blend,
        provenance_path=provenance_path,
        expected_base=base,
    )
    asset.write_bytes(b"asset-drift")
    assert not MODULE._materialization_ready(
        blend_file=blend,
        provenance_path=provenance_path,
        expected_base=base,
    )
