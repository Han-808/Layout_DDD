from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.evaluator import evaluate_generic_validity
from benchmark.evaluator.generic_validity.accessibility import check_accessibility
from benchmark.evaluator.generic_validity.asset_resolver import resolve_asset_metadata
from benchmark.evaluator.generic_validity.collision import check_collision
from benchmark.evaluator.generic_validity.navigability import check_navigability
from benchmark.evaluator.generic_validity.oob import check_oob
from benchmark.evaluator.generic_validity.geometry import normalize_object
from benchmark.evaluator.generic_validity.support import check_support
from benchmark.scene_io.normalize import normalize_scene
from benchmark.utils.io import read_json


ROOT = Path(__file__).resolve().parents[1]


def _scene(objects: list[dict], boundary: list[list[float]] | None = None, height: float = 2.8) -> dict:
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "generic_validity_test",
        "scene_type": "room",
        "boundary": boundary or [[0, 0], [4, 0], [4, 3], [0, 3]],
        "scene_height": height,
        "objects": objects,
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            }
        },
    }


def _obj(
    object_id: str,
    center: list[float],
    size: list[float] | None = None,
    *,
    category: str = "box",
    interactive: bool = False,
) -> dict:
    result = {
        "id": object_id,
        "category": category,
        "center": center,
        "size": size or [0.5, 0.5, 1.0],
        "rotation": [0, 0, 0],
    }
    if interactive:
        result["interactive"] = True
    return result



def _canonical_obj(object_id: str, center: list[float], size: list[float] | None = None) -> dict:
    jid = f"{object_id}_asset"
    resolved_size = size or [0.5, 0.5, 1.0]
    return {
        "id": object_id,
        "jid": jid,
        "category": "box",
        "description": "box",
        "retrieval_category": "box",
        "desc": "box",
        "short_desc": "box",
        "center": center,
        "size": resolved_size,
        "rotation": [0, 0, 0],
        "asset_ref": {"source_db": "imaginarium", "asset_key": jid, "mesh_uri": None, "pointcloud_uri": None, "metadata_uri": None},
        "asset_proxy": {"type": "obb_from_metadata_or_csv", "bbox_center_local": [0, 0, 0], "bbox_size": resolved_size},
        "metadata": {"interactive": False},
    }

def test_collision_counts_overlap_and_routes_separated_pairs_without_vlm() -> None:
    separated = _scene([_obj("a", [1.0, 1.0, 0.5], [0.5, 0.5, 0.5]), _obj("b", [4.0, 1.0, 0.5], [0.5, 0.5, 0.5])])
    overlapping = _scene([_obj("a", [1.0, 1.0, 0.5], [1.0, 1.0, 1.0]), _obj("b", [1.2, 1.0, 0.5], [1.0, 1.0, 1.0])])

    separated_report = check_collision(separated)
    overlapping_report = check_collision(overlapping, {"detector_only": True})

    assert separated_report["pairs"][0]["route"] == "direct_valid_obb_separated"
    assert separated_report["collision_count"] == 0
    assert separated_report["score"] == 1.0
    assert overlapping_report["requires_vlm_count"] == 1
    assert overlapping_report["score"] is None


def test_collision_no_longer_auto_exempts_supported_or_contained_pairs() -> None:
    scene = _scene(
        [
            _obj("container", [1.0, 1.0, 0.5], [1.5, 1.5, 1.0]),
            _obj("stored", [1.0, 1.0, 0.5], [0.3, 0.3, 0.3]),
        ]
    )

    report = check_collision(scene, {"detector_only": True})

    assert report["requires_vlm_count"] == 1
    assert report["pairs"][0].get("exempted") is not True


def test_oob_flags_planes_and_routes_inside_without_vlm() -> None:
    inside = check_oob(_scene([_obj("inside", [1.0, 1.0, 0.5])]))
    outside = check_oob(_scene([_obj("outside", [-0.1, 1.0, 0.5])]), {"detector_only": True})
    below = check_oob(_scene([_obj("below", [1.0, 1.0, -0.2])]), {"detector_only": True})
    above = check_oob(_scene([_obj("above", [1.0, 1.0, 2.7], [0.5, 0.5, 0.5])], height=2.8), {"detector_only": True})

    assert inside["candidate_oob_count"] == 0
    assert inside["objects"][0]["route"] == "direct_valid_inside"
    assert inside["objects"][0]["final_verdict"] == "valid"
    assert inside["score"] == 1.0
    assert outside["objects"][0]["plane_flags"]["west_oob"] is True
    assert below["objects"][0]["plane_flags"]["floor_oob"] is True
    assert above["objects"][0]["plane_flags"]["ceiling_oob"] is True


def test_navigability_empty_room_splitter_and_non_blocking_objects() -> None:
    empty = check_navigability(_scene([]))
    split = check_navigability(_scene([_obj("divider", [2.0, 1.5, 0.5], [0.2, 3.0, 1.0])]), {"grid_resolution": 0.10, "agent_radius": 0.0})
    rug = check_navigability(_scene([_obj("rug", [2.0, 1.5, 0.05], [2.0, 1.0, 0.10], category="rug")]))
    elevated = check_navigability(_scene([_obj("ceiling_panel", [2.0, 1.5, 2.1], [2.0, 1.0, 0.3])]))

    assert empty["score"] == 1.0
    assert split["score"] < 1.0
    assert split["num_components"] == 2
    assert rug["score"] == 1.0
    assert rug["blocking_object_count"] == 0
    assert elevated["score"] == 1.0
    assert elevated["blocking_object_count"] == 0


def test_game_navigability_uses_non_flat_structure_not_oversized_ground() -> None:
    scene = _scene(
        [
            _obj(
                "decorative_ground",
                [50.0, 50.0, 0.005],
                [100.0, 100.0, 0.01],
            ),
            _obj("west_wall", [10.1, 15.0, 1.0], [0.2, 10.0, 2.0]),
            _obj("east_wall", [19.9, 15.0, 1.0], [0.2, 10.0, 2.0]),
            _obj("south_wall", [15.0, 10.1, 1.0], [10.0, 0.2, 2.0]),
            _obj("north_wall", [15.0, 19.9, 1.0], [10.0, 0.2, 2.0]),
        ],
        boundary=[[0, 0], [100, 0], [100, 100], [0, 100]],
        height=2.0,
    )
    report = check_navigability(
        scene,
        {
            "boundary_source": "non_flat_structure_envelope",
            "grid_resolution": 0.2,
            "agent_radius": 0.0,
            "max_grid_cells": 250000,
        },
    )

    assert report["boundary_source"] == "non_flat_structure_envelope"
    assert report["score_definition"] == (
        "largest_connected_free_area / total_free_area"
    )
    assert report["projection"] == "single_floor_2d"
    assert report["total_free_area"] < 150.0


def test_accessibility_not_applicable_passes_and_surrounded_object_fails() -> None:
    no_interactive = check_accessibility(_scene([_obj("box", [1.0, 1.0, 0.5])]))
    reachable = check_accessibility(_scene([_obj("desk", [1.0, 1.0, 0.5], interactive=True)]))
    surrounded = check_accessibility(
        _scene(
            [
                _obj("target", [2.0, 1.5, 0.5], [0.4, 0.4, 1.0], interactive=True),
                _obj("north", [2.0, 2.05, 0.5], [1.4, 0.3, 1.0]),
                _obj("south", [2.0, 0.95, 0.5], [1.4, 0.3, 1.0]),
                _obj("east", [2.55, 1.5, 0.5], [0.3, 1.4, 1.0]),
                _obj("west", [1.45, 1.5, 0.5], [0.3, 1.4, 1.0]),
            ]
        ),
        {"grid_resolution": 0.10, "agent_radius": 0.05, "access_radius": 0.25},
    )

    assert no_interactive["status"] == "not_applicable"
    assert reachable["score"] == 1.0
    assert surrounded["score"] == 0.0
    assert surrounded["objects"][0]["accessible"] is False


def test_support_probe_routes_floor_object_and_floating_cases() -> None:
    on_floor = check_support(_scene([_obj("box", [1.0, 1.0, 0.5])]))
    floating = check_support(_scene([_obj("floating", [1.0, 1.0, 1.0], [0.5, 0.5, 0.5])]), {"detector_only": True})
    on_table = check_support(_scene([_obj("table", [1.0, 1.0, 0.5]), _obj("book", [1.0, 1.0, 1.1], [0.3, 0.3, 0.2])]))

    assert on_floor["objects"][0]["route"] == "direct_valid_contact"
    assert on_floor["objects"][0]["final_verdict"] == "valid"
    assert on_floor["score"] == 1.0

    floating_obj = floating["objects"][0]
    assert floating_obj["contact_fraction"] == 0.0
    assert floating_obj["requires_vlm"] is True
    assert floating_obj["final_verdict"] is None
    assert floating["score"] is None

    book = next(obj for obj in on_table["objects"] if obj["object_id"] == "book")
    assert book["route"] == "direct_valid_contact"
    assert book["support_targets"] == ["table"]


def test_evaluator_aggregation_averages_active_metrics_only() -> None:
    scene = _scene([_obj("a", [1.0, 1.0, 0.5], [0.5, 0.5, 0.5]), _obj("b", [2.5, 1.0, 0.5], [0.5, 0.5, 0.5])])
    report = evaluate_generic_validity(
        scene,
        {
            "collision": {"detector_only": False},
            "navigability": {"enabled": False},
            "support": {"enabled": False},
        },
    )

    assert report["metrics"]["collision"]["score"] == 1.0
    assert report["metrics"]["oob"]["score"] == 1.0
    assert report["metrics"]["accessibility"]["status"] == "not_applicable"
    assert report["active_metric_count"] == 2
    assert report["score"] == 1.0


def test_evaluate_cli_writes_generic_validity_report_by_default(tmp_path: Path) -> None:
    scene_path = tmp_path / "scene.json"
    out_path = tmp_path / "report.json"
    scene_path.write_text(json.dumps({**_scene([_canonical_obj("box", [1.0, 1.0, 0.5])]), "request_id": "cli"}), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluate.py"),
            "--scene",
            str(scene_path),
            "--out",
            str(out_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = read_json(out_path)
    assert report["evaluator_version"] == "scene_harness_evaluator_v2"
    assert set(report["category_reports"]) == {
        "l0_structural_validity",
        "l1_physical_plausibility",
        "l2_specification_fidelity",
        "l3_scene_quality",
        "l4_downstream_task_functionality",
    }
    assert "generic_validity" in report["reports"]
    assert "benchmark_score:" in completed.stdout



def test_asset_metadata_resolution_uses_transformed_size(tmp_path: Path) -> None:
    jid = "0_alarm_clock_01_2k_packed"
    asset_dir = tmp_path / jid
    asset_dir.mkdir()
    (asset_dir / f"{jid}_metadata.json").write_text(
        json.dumps(
            {
                "transformed_bbox_center": [0.0, 0.0, 0.0],
                "transformed_size": [0.2, 0.1, 0.4],
                "actual_points": 8192,
                "has_color": True,
                "has_normal": True,
                "is_centered": True,
                "is_coordinate_transformed": True,
            }
        ),
        encoding="utf-8",
    )
    (asset_dir / f"{jid}.fbx").write_text("fake mesh ref only", encoding="utf-8")
    (asset_dir / f"{jid}.ply").write_text("fake pointcloud ref only", encoding="utf-8")
    obj = {"id": "clock", "jid": jid, "center": [1.0, 1.0, 0.2], "rotation": [0, 0, 0]}

    normalized = normalize_object(obj, asset_root=str(tmp_path))

    assert normalized.size.tolist() == [0.2, 0.1, 0.4]
    assert normalized.asset_ref["pointcloud_uri"] == f"{jid}/{jid}.ply"
    assert normalized.asset_proxy["point_count"] == 8192


def test_asset_csv_fallback_parses_bbx_and_populates_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "asset_info.csv"
    csv_path.write_text(
        "id,name_en,bbx,caption_en,short_desc,class_en,retrieval_class_en\n"
        '3,csv_asset,"[0.3, 0.2, 0.5]",long caption,short caption,decor,clock\n',
        encoding="utf-8",
    )
    obj = {"id": "csv_obj", "jid": "csv_asset", "center": [1.0, 1.0, 0.25], "rotation": [0, 0, 0]}

    enriched = resolve_asset_metadata(obj, asset_csv_path=csv_path)

    assert enriched["size"] == [0.3, 0.2, 0.5]
    assert enriched["desc"] == "long caption"
    assert enriched["short_desc"] == "short caption"
    assert enriched["category"] == "decor"
    assert enriched["retrieval_category"] == "clock"
    assert enriched["jid"] == "csv_asset"


def test_self_contained_scene_evaluates_without_asset_resolution() -> None:
    report = evaluate_generic_validity(_scene([_obj("box", [1.0, 1.0, 0.5])]))

    assert report["status"] == "ok"
    assert report["metrics"]["oob"]["score"] == 1.0


def test_mesh_and_pointcloud_uris_are_not_loaded_when_size_exists() -> None:
    scene = _scene(
        [
            {
                "id": "asset_obj",
                "jid": "fake_asset",
                "category": "decor",
                "center": [1.0, 1.0, 0.5],
                "size": [0.5, 0.5, 1.0],
                "rotation": [0, 0, 0],
                "asset_ref": {
                    "asset_key": "fake_asset",
                    "mesh_uri": "/path/that/does/not/exist/fake_asset.fbx",
                    "pointcloud_uri": "/path/that/does/not/exist/fake_asset.ply",
                },
            }
        ]
    )

    report = evaluate_generic_validity(scene)

    assert report["status"] == "ok"
    assert report["metrics"]["collision"]["score"] == 1.0


def test_interactive_defaults_false_without_manual_annotation() -> None:
    normalized = normalize_object({"id": "plain", "center": [1.0, 1.0, 0.5], "size": [0.5, 0.5, 1.0], "rotation": [0, 0, 0]})

    assert normalized.interactive is False


@pytest.mark.parametrize("yaw", [1.0, 5.0, 90.0, 180.0])
def test_canonical_rotation_is_always_interpreted_as_degrees(yaw: float) -> None:
    normalized = normalize_object(
        {"id": "rotated", "center": [1.0, 1.0, 0.5], "size": [0.5, 0.5, 1.0], "rotation": [0, 0, yaw]}
    )

    assert normalized.rotation.tolist() == [0.0, 0.0, yaw]
    assert normalized.yaw_degrees == yaw


def test_scene_normalization_rejects_non_degree_rotation_contract() -> None:
    scene = _scene([])
    scene["request_id"] = "generic_validity_test"
    scene["metadata"] = {
        "coordinate_frame": {
            "origin": "room_min_corner_floor",
            "axes": "x_width_y_depth_z_up",
            "unit": "meter",
            "rotation_unit": "radian",
        }
    }

    with pytest.raises(ValueError, match="rotation_unit must be 'degree'"):
        normalize_scene(scene)


def test_metadata_interactive_true_drives_accessibility() -> None:
    scene = _scene(
        [
            {
                "id": "desk",
                "category": "desk",
                "center": [1.0, 1.0, 0.5],
                "size": [0.5, 0.5, 1.0],
                "rotation": [0, 0, 0],
                "metadata": {"interactive": True},
            }
        ]
    )

    report = check_accessibility(scene)

    assert report["status"] == "checked"
    assert report["interactive_count"] == 1
    assert report["objects"][0]["accessible"] is True


def test_evaluate_cli_can_enrich_assets_from_metadata_json(tmp_path: Path) -> None:
    jid = "cli_asset"
    asset_dir = tmp_path / "assets" / jid
    asset_dir.mkdir(parents=True)
    (asset_dir / f"{jid}_metadata.json").write_text(json.dumps({"transformed_size": [0.4, 0.2, 0.6]}), encoding="utf-8")
    (asset_dir / f"{jid}.ply").write_text("reference only", encoding="utf-8")
    scene_path = tmp_path / "scene.json"
    out_path = tmp_path / "report.json"
    scene = {
        **_scene(
            [
                    {
                        "id": "obj",
                        "jid": jid,
                        "category": "box",
                        "description": "box",
                        "size": [0.4, 0.2, 0.6],
                        "center": [1.0, 1.0, 0.3],
                        "rotation": [0, 0, 0],
                        "metadata": {},
                    }
            ]
        ),
        "request_id": "cli",
    }
    scene_path.write_text(json.dumps(scene), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "evaluate.py"),
            "--scene",
            str(scene_path),
            "--asset-root",
            str(tmp_path / "assets"),
            "--enrich-assets",
            "--out",
            str(out_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = read_json(out_path)
    assert report["evaluator_version"] == "scene_harness_evaluator_v2"
    assert report["reports"]["generic_validity"]["status"] == "ok"
