#!/usr/bin/env python3
"""Audit controlled P0b GT against canonical coordinates and rendered asset bounds.

This is an experiment-data audit, not an evaluator. It keeps three facts separate:

* the stored GT label;
* the classification implied by canonical JSON/OBB coordinates; and
* the classification implied by the uniformly fitted asset geometry shown to the VLM.

The distinction matters for Support because a uniformly scaled asset can be centered
inside a taller canonical bbox and therefore float above its canonical support plane.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.evaluator.generic_validity.geometry import footprint_overlap_area
from benchmark.scene_io.object_normalization import (
    get_room_boundary,
    get_scene_height,
    normalize_object,
    normalize_objects,
)


COLLISION_REVIEW_OVERRIDES = {
    ("source_clean_001__collision", "obj_004|obj_005"): {
        "classification": "invalid",
        "basis": "same projector asset and same center; 90-degree point-cloud comparison has 0.000080 m minimum surface distance",
        "pointcloud_minimum_surface_distance_m": 0.00008015516372745689,
    },
    ("source_clean_005__collision", "obj_004|obj_005"): {
        "classification": "invalid",
        "basis": "co-centered plant and wardrobe; 94.128% of plant samples lie inside the wardrobe OBB and point-cloud surfaces approach within 0.001184 m",
        "pointcloud_minimum_surface_distance_m": 0.0011842933115219542,
        "plant_sample_fraction_inside_wardrobe_obb": 0.9412841796875,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=PROJECT_ROOT / "Support" / "artifacts" / "result" / "benchmark_metric_analysis" / "source_distortion5" / "gt",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiments" / "p0b_source_distortion5" / "fixtures",
    )
    parser.add_argument(
        "--report-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "Support"
            / "artifacts"
            / "result"
            / "benchmark_metric_analysis"
            / "source_distortion5"
            / "source_reports"
        ),
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=PROJECT_ROOT / "Support" / "Assets" / "imaginarium_assets",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "Support" / "artifacts" / "result" / "benchmark_metric_analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    for gt_path in sorted(args.gt_root.glob("*.json")):
        gt = _read_json(gt_path)
        case_id = str(gt["case_id"])
        fixture_dir = args.fixture_root / case_id
        scene = _read_json(fixture_dir / "generated_scene.json")
        report = _read_json(args.report_root / case_id / "evaluation_report.json")
        geometry_manifest = _read_json(
            args.report_root / case_id / "renders" / "collision_geometry_manifest.json"
        )
        raw_objects = {str(item["id"]): item for item in scene.get("objects", [])}
        objects, errors = normalize_objects(scene)
        if errors:
            raise RuntimeError(f"{case_id}: object normalization failed: {errors}")
        normalized = {item.id: item for item in objects}
        rendered = {
            object_id: _rendered_object(raw, args.asset_root)
            for object_id, raw in raw_objects.items()
        }

        for event in gt.get("events", []):
            metric = str(event["metric"])
            if metric == "collision":
                row = _audit_collision(case_id, event, report, raw_objects)
            elif metric == "oob":
                row = _audit_oob(
                    case_id,
                    event,
                    scene,
                    normalized,
                    geometry_manifest,
                    rendered,
                )
            elif metric == "support":
                row = _audit_support(
                    case_id,
                    event,
                    normalized,
                    rendered,
                )
            else:
                continue
            rows.append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = args.out_dir / "GT_COORDINATE_AUDIT.tsv"
    json_path = args.out_dir / "GT_COORDINATE_AUDIT.json"
    _write_tsv(tsv_path, rows)
    json_path.write_text(json.dumps({"events": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"events={len(rows)}")
    print(f"tsv={tsv_path}")
    print(f"json={json_path}")


def _audit_collision(
    case_id: str,
    event: dict[str, Any],
    report: dict[str, Any],
    raw_objects: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    event_id = str(event["event_id"])
    pair = _collision_pair(report, event_id)
    mesh = pair.get("mesh_evidence") if isinstance(pair.get("mesh_evidence"), dict) else {}
    override = COLLISION_REVIEW_OVERRIDES.get((case_id, event_id))
    if mesh.get("surface_intersection") is True:
        rendered_label = "invalid"
        basis = "triangle-mesh surface intersection"
    elif (
        mesh.get("surface_intersection") is False
        and mesh.get("mesh_state") == "separated"
        and float(mesh.get("minimum_surface_distance_m") or 0.0) > 0.02
    ):
        rendered_label = "valid"
        basis = "triangle meshes are positively separated"
    elif override:
        rendered_label = str(override["classification"])
        basis = str(override["basis"])
    else:
        raise RuntimeError(f"{case_id}/{event_id}: collision requires an explicit audit decision")

    ids = event_id.split("|")
    evidence = {
        "object_a": _pose(raw_objects[ids[0]]),
        "object_b": _pose(raw_objects[ids[1]]),
        "obb_minimum_overlap_depth_proxy_m": pair.get("obb_evidence", {}).get(
            "minimum_overlap_depth_proxy_m"
        ),
        "obb_xy_overlap_area_m2": pair.get("diagnostics", {}).get("xy_overlap_area"),
        "obb_z_overlap_m": pair.get("diagnostics", {}).get("z_overlap"),
        "mesh_state": mesh.get("mesh_state"),
        "mesh_surface_intersection": mesh.get("surface_intersection"),
        "mesh_minimum_surface_distance_m": mesh.get("minimum_surface_distance_m"),
        "classification_basis": basis,
        **({key: value for key, value in override.items() if key != "classification"} if override else {}),
    }
    stored = str(event["label"])
    if stored == rendered_label and event.get("reason_code") == "coordinate_mesh_audit":
        outcome = "corrected"
    else:
        outcome = "confirmed" if stored == rendered_label else "label_error"
    return _row(
        case_id,
        "collision",
        event_id,
        stored,
        canonical_label="candidate_only",
        rendered_label=rendered_label,
        evidence=evidence,
        outcome=outcome,
    )


def _audit_oob(
    case_id: str,
    event: dict[str, Any],
    scene: dict[str, Any],
    normalized: dict[str, Any],
    geometry_manifest: dict[str, Any],
    rendered: dict[str, Any],
) -> dict[str, Any]:
    object_id = str(event["event_id"])
    obj = normalized[object_id]
    room_min, room_max = _room_bounds(scene)
    radius = np.abs(np.asarray(obj.R, dtype=float)) @ np.asarray(obj.half, dtype=float)
    canonical_min = np.asarray(obj.center, dtype=float) - radius
    canonical_max = np.asarray(obj.center, dtype=float) + radius
    canonical_depths = _plane_depths(canonical_min, canonical_max, room_min, room_max)

    geometry = (geometry_manifest.get("objects") or {}).get(object_id) or {}
    world_aabb = geometry.get("world_aabb") if isinstance(geometry, dict) else None
    if isinstance(world_aabb, dict):
        rendered_min = np.asarray(world_aabb["min"], dtype=float)
        rendered_max = np.asarray(world_aabb["max"], dtype=float)
        rendered_source = "exported_triangle_mesh_world_aabb"
    else:
        actual = rendered[object_id]
        actual_radius = np.abs(np.asarray(actual.R, dtype=float)) @ np.asarray(actual.half, dtype=float)
        rendered_min = np.asarray(actual.center, dtype=float) - actual_radius
        rendered_max = np.asarray(actual.center, dtype=float) + actual_radius
        rendered_source = "asset_pointcloud_transformed_bounds"
    rendered_depths = _plane_depths(rendered_min, rendered_max, room_min, room_max)
    canonical_label = "invalid" if canonical_depths else "valid"
    rendered_label = "invalid" if rendered_depths else "valid"
    stored = str(event["label"])
    evidence = {
        "object": _pose_from_normalized(obj),
        "room_min_xyz": room_min.tolist(),
        "room_max_xyz": room_max.tolist(),
        "canonical_obb_min_xyz": canonical_min.tolist(),
        "canonical_obb_max_xyz": canonical_max.tolist(),
        "canonical_plane_penetration_m": canonical_depths,
        "rendered_bounds_source": rendered_source,
        "rendered_min_xyz": rendered_min.tolist(),
        "rendered_max_xyz": rendered_max.tolist(),
        "rendered_plane_penetration_m": rendered_depths,
    }
    outcome = "confirmed" if stored == canonical_label == rendered_label else "representation_conflict"
    return _row(
        case_id,
        "oob",
        object_id,
        stored,
        canonical_label=canonical_label,
        rendered_label=rendered_label,
        evidence=evidence,
        outcome=outcome,
    )


def _audit_support(
    case_id: str,
    event: dict[str, Any],
    normalized: dict[str, Any],
    rendered: dict[str, Any],
) -> dict[str, Any]:
    object_id = str(event["event_id"])
    request = event.get("frozen_request") or {}
    detector = request.get("detector_evidence") or {}
    near_tolerance = float(detector.get("near_support_tolerance_m") or 0.08)
    canonical_target, canonical_gap = _nearest_support(object_id, normalized)
    rendered_target, rendered_gap = _nearest_support(object_id, rendered)
    canonical_label = "valid" if canonical_gap <= near_tolerance + 1.0e-9 else "invalid"
    rendered_label = "valid" if rendered_gap <= near_tolerance + 1.0e-9 else "invalid"
    stored = str(event["label"])
    evidence = {
        "object": _pose_from_normalized(normalized[object_id]),
        "near_support_tolerance_m": near_tolerance,
        "canonical_nearest_support": canonical_target,
        "canonical_gap_m": canonical_gap,
        "rendered_nearest_support": rendered_target,
        "rendered_uniform_contain_gap_m": rendered_gap,
        "frozen_detector_minimum_positive_clearance_m": detector.get(
            "minimum_positive_clearance_m"
        ),
        "frozen_detector_gap_band": detector.get("gap_band"),
        "rendered_size_xyz": rendered[object_id].size.tolist(),
        "classification_basis": "positive clearance along gravity; no fixture claim permits suspension or intentional floating",
    }
    if canonical_label != rendered_label:
        outcome = "representation_conflict"
    elif stored != canonical_label:
        outcome = "label_error"
    else:
        outcome = "confirmed"
    return _row(
        case_id,
        "support",
        object_id,
        stored,
        canonical_label=canonical_label,
        rendered_label=rendered_label,
        evidence=evidence,
        outcome=outcome,
    )


def _rendered_object(raw: dict[str, Any], asset_root: Path):
    rendered_raw = deepcopy(raw)
    jid = str(raw.get("jid") or (raw.get("asset_ref") or {}).get("asset_key") or "")
    metadata_path = asset_root / jid / f"{jid}_metadata.json"
    if metadata_path.is_file():
        metadata = _read_json(metadata_path)
        source_size = np.asarray(metadata.get("transformed_size"), dtype=float)
        target_size = np.asarray(raw.get("size"), dtype=float)
        scale = float(np.min(target_size / source_size))
        rendered_raw["size"] = (source_size * scale).tolist()
    return normalize_object(rendered_raw)


def _nearest_support(object_id: str, objects: dict[str, Any]) -> tuple[str, float]:
    source = objects[object_id]
    candidates: list[tuple[str, float]] = [("floor", max(0.0, float(source.bottom_z)))]
    for target_id, target in objects.items():
        if target_id == object_id:
            continue
        gap = float(source.bottom_z - target.top_z)
        if gap < -1.0e-6:
            continue
        if footprint_overlap_area(source, target) <= 1.0e-8:
            continue
        candidates.append((target_id, max(0.0, gap)))
    return min(candidates, key=lambda item: item[1])


def _collision_pair(report: dict[str, Any], event_id: str) -> dict[str, Any]:
    pairs = (
        report.get("reports", {})
        .get("generic_validity", {})
        .get("metrics", {})
        .get("collision", {})
        .get("pairs", [])
    )
    for pair in pairs:
        key = "|".join(sorted([str(pair.get("object_a")), str(pair.get("object_b"))]))
        if key == event_id:
            return pair
    raise KeyError(f"collision event not found in source report: {event_id}")


def _room_bounds(scene: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    boundary = np.asarray(get_room_boundary(scene), dtype=float)
    height = float(get_scene_height(scene))
    return (
        np.asarray([boundary[:, 0].min(), boundary[:, 1].min(), 0.0], dtype=float),
        np.asarray([boundary[:, 0].max(), boundary[:, 1].max(), height], dtype=float),
    )


def _plane_depths(
    minimum: np.ndarray,
    maximum: np.ndarray,
    room_min: np.ndarray,
    room_max: np.ndarray,
) -> dict[str, float]:
    values = {
        "west": float(room_min[0] - minimum[0]),
        "east": float(maximum[0] - room_max[0]),
        "south": float(room_min[1] - minimum[1]),
        "north": float(maximum[1] - room_max[1]),
        "floor": float(room_min[2] - minimum[2]),
        "ceiling": float(maximum[2] - room_max[2]),
    }
    return {key: value for key, value in values.items() if value > 1.0e-6}


def _pose(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": raw.get("id"),
        "category": raw.get("category"),
        "center": raw.get("center"),
        "size": raw.get("size"),
        "rotation_degrees": raw.get("rotation"),
        "asset_key": raw.get("jid") or (raw.get("asset_ref") or {}).get("asset_key"),
    }


def _pose_from_normalized(obj: Any) -> dict[str, Any]:
    return {
        "id": obj.id,
        "category": obj.category,
        "center": obj.center.tolist(),
        "size": obj.size.tolist(),
        "rotation_degrees": obj.rotation.tolist(),
        "asset_key": obj.jid,
    }


def _row(
    case_id: str,
    metric: str,
    event_id: str,
    stored_label: str,
    *,
    canonical_label: str,
    rendered_label: str,
    evidence: dict[str, Any],
    outcome: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "metric": metric,
        "event_id": event_id,
        "stored_gt_label": stored_label,
        "stored_gt_violation": _label_to_violation(stored_label),
        "canonical_coordinate_classification": canonical_label,
        "canonical_coordinate_violation": _label_to_violation(canonical_label),
        "rendered_geometry_classification": rendered_label,
        "rendered_geometry_violation": _label_to_violation(rendered_label),
        "audit_outcome": outcome,
        "stored_matches_rendered": stored_label == rendered_label,
        "coordinate_evidence": evidence,
    }


def _label_to_violation(label: str) -> bool | None:
    if label == "invalid":
        return True
    if label == "valid":
        return False
    return None


def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["coordinate_evidence"] = json.dumps(
                serialized["coordinate_evidence"], separators=(",", ":")
            )
            writer.writerow(serialized)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


if __name__ == "__main__":
    main()
