#!/usr/bin/env python3
"""Build controlled P0b/OAR fixtures from five clean repository scenes.

The generated fixtures are benchmark-owned inputs.  They never enter the
generator path: each case contains a reviewed prompt annotation and a frozen
canonical scene that can be rendered and evaluated directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.evaluator.OAR import evaluate_oar
from benchmark.evaluator.OOR import evaluate_oor
from benchmark.evaluator.generic_validity import evaluate_generic_validity
from benchmark.reference_annotation import validate_reference_annotation
from benchmark.scene_io.validate import validate_generated_scene, validate_scene_request


COORDINATE_FRAME = {
    "origin": "room_min_corner_floor",
    "axes": "x_width_y_depth_z_up",
    "unit": "meter",
    "rotation_unit": "degree",
}

DETECTOR_ONLY_CONFIG = {
    "collision": {"detector_only": True},
    "oob": {"detector_only": True},
    "support": {"detector_only": True},
    "navigability": {"enabled": False},
    "accessibility": {"enabled": False},
}

VARIANT_FAMILIES = ("clean", "collision", "oob", "support", "oar")


SOURCE_CASES: list[dict[str, Any]] = [
    {
        "base_case_id": "source_clean_001",
        "source_scene_id": "scene_007379_03",
        "difficulty": "simple",
        "keep_indices": [0, 1, 2, 3, 4, 5],
        "instruction": (
            "Create a compact 2.81 m by 4.20 m master bedroom with a 2.799 m ceiling. "
            "Place a white wood wardrobe against the north wall and a light wood bunk bed near "
            "the room center. Put two gray suede nightstands near the bed, both near the east "
            "wall. Rest one light gray desktop projector on each nightstand."
        ),
        "category_overrides": {4: "projector", 5: "projector"},
        "oor_relations": [
            {"type": "above", "subject_id": "obj_004", "object_id": "obj_002"},
            {"type": "contact", "subject_id": "obj_004", "object_id": "obj_002"},
            {"type": "above", "subject_id": "obj_005", "object_id": "obj_001"},
            {"type": "contact", "subject_id": "obj_005", "object_id": "obj_001"},
            {"type": "near", "subject_id": "obj_001", "object_id": "obj_003"},
            {"type": "near", "subject_id": "obj_002", "object_id": "obj_003"},
        ],
        "oar_relations": [
            {"type": "against_wall", "subject_id": "obj_000", "architectural_element": "north_wall"},
            {"type": "room_center", "subject_id": "obj_003", "architectural_element": "center_region"},
            {"type": "near_wall", "subject_id": "obj_001", "architectural_element": "east_wall"},
            {"type": "near_wall", "subject_id": "obj_002", "architectural_element": "east_wall"},
        ],
        "collision_moves": [
            {"target": "obj_001", "anchor": "obj_002"},
            {"target": "obj_005", "anchor": "obj_004", "z_mode": "anchor"},
        ],
        "oob_moves": [("obj_000", "north")],
        "support_targets": ["obj_000"],
        "oar_moves": [(0, "obj_000", [0.90, 0.45])],
    },
    {
        "base_case_id": "source_clean_002",
        "source_scene_id": "scene_003309_01",
        "difficulty": "simple",
        "keep_indices": [0, 1, 2, 3, 4],
        "instruction": (
            "Create a compact 2.46 m by 3.51 m bedroom with a 2.45 m ceiling. Place two dark "
            "gray minimalist cabinets against the west wall. Include two dark wooden storage "
            "boxes, and place a gray wooden single bed near the center of the room."
        ),
        "oor_relations": [],
        "oar_relations": [
            {"type": "against_wall", "subject_id": "obj_000", "architectural_element": "west_wall"},
            {"type": "against_wall", "subject_id": "obj_001", "architectural_element": "west_wall"},
            {"type": "room_center", "subject_id": "obj_002", "architectural_element": "center_region"},
        ],
        "collision_moves": [{"target": "obj_000", "anchor": "obj_001"}],
        "oob_moves": [("obj_000", "west"), ("obj_003", "east")],
        "support_targets": ["obj_003"],
        "oar_moves": [(0, "obj_000", [1.80, 0.40])],
    },
    {
        "base_case_id": "source_clean_003",
        "source_scene_id": "scene_000657_03",
        "difficulty": "simple",
        "keep_indices": [0, 1, 2, 3, 4],
        "instruction": (
            "Create a 5.25 m by 3.27 m bedroom-study with a 2.881 m ceiling. Put a gray silver "
            "wardrobe against the east wall, a gray baroque bedside table against the north wall, "
            "and a dark green metal cabinet against the south wall. Place a gray wooden single bed "
            "to the right of the bedside table. Put a brown wooden coffee table near the bed, "
            "with the metal cabinet in front of the bed."
        ),
        "oor_relations": [
            {"type": "left", "subject_id": "obj_001", "object_id": "obj_002"},
            {"type": "near", "subject_id": "obj_003", "object_id": "obj_002"},
            {"type": "in_front", "subject_id": "obj_004", "object_id": "obj_002"},
        ],
        "oar_relations": [
            {"type": "against_wall", "subject_id": "obj_000", "architectural_element": "east_wall"},
            {"type": "against_wall", "subject_id": "obj_001", "architectural_element": "north_wall"},
            {"type": "against_wall", "subject_id": "obj_004", "architectural_element": "south_wall"},
            {"type": "against_wall", "subject_id": "obj_003", "architectural_element": "west_wall"},
        ],
        "collision_moves": [
            {
                "target": "obj_002",
                "anchor": "obj_000",
                "xy": [3.80, 2.00],
                # This is an intentional broad-phase hard negative: the OBBs overlap,
                # but the frozen asset meshes are separated by about 0.299 m.
                "expected_collision_label": "valid",
            }
        ],
        "oob_moves": [("obj_000", "east"), ("obj_004", "south")],
        "support_targets": ["obj_001", "obj_004"],
        "oar_moves": [
            (0, "obj_000", [4.50, 1.80]),
            (1, "obj_001", [1.90, 1.00]),
        ],
    },
    {
        "base_case_id": "source_clean_004",
        "source_scene_id": "scene_003609_02",
        "difficulty": "simple",
        "keep_indices": [0, 1, 2, 3, 4, 5],
        "instruction": (
            "Create a compact 2.64 m by 3.29 m kitchen with a 2.400 m ceiling. Put a white rounded "
            "refrigerator at the southwest corner. Place two light wood kitchen cabinets side by "
            "side in the north part of the room. "
            "Place one dark wooden kitchen cabinet against the south wall, another against the east "
            "wall, and a third dark cabinet near the south-wall cabinet."
        ),
        "oor_relations": [
            {"type": "near", "subject_id": "obj_002", "object_id": "obj_000"},
            {"type": "aligned", "subject_id": "obj_002", "object_id": "obj_000", "axis": "x"},
            {"type": "near", "subject_id": "obj_003", "object_id": "obj_005"},
        ],
        "oar_relations": [
            {"type": "at_corner", "subject_id": "obj_001", "architectural_element": "southwest_corner"},
            {"type": "against_wall", "subject_id": "obj_003", "architectural_element": "south_wall"},
            {"type": "against_wall", "subject_id": "obj_004", "architectural_element": "east_wall"},
            {"type": "room_region", "subject_id": "obj_000", "architectural_element": "north_region", "region": "north"},
            {"type": "room_region", "subject_id": "obj_002", "architectural_element": "north_region", "region": "north"},
        ],
        "collision_moves": [
            {"target": "obj_000", "anchor": "obj_002"},
            {"target": "obj_005", "anchor": "obj_003"},
        ],
        "oob_moves": [("obj_001", "west"), ("obj_003", "south"), ("obj_004", "east")],
        "support_targets": ["obj_001", "obj_003", "obj_004"],
        "oar_moves": [
            (0, "obj_001", [0.40, 1.20]),
            (1, "obj_003", [1.17, 1.23]),
            (2, "obj_004", [2.20, 2.80]),
        ],
    },
    {
        "base_case_id": "source_clean_005",
        "source_scene_id": "scene_011841_04",
        "difficulty": "simple",
        "keep_indices": [0, 1, 2, 3, 4, 5],
        "instruction": (
            "Create a simple 4.50 m by 12.87 m living room with a 2.797 m ceiling. Put a gray "
            "wooden storage locker near the north wall and a dark gray round coffee table near it. "
            "Place a black leather sofa in the south part of the room. Put one gray textured potted "
            "plant at the southwest corner and a second potted plant near the east wall and near the "
            "sofa. Place a gray open shelving wardrobe in the east part of the room."
        ),
        "oor_relations": [
            {"type": "near", "subject_id": "obj_001", "object_id": "obj_000"},
            {"type": "near", "subject_id": "obj_004", "object_id": "obj_002"},
        ],
        "oar_relations": [
            {"type": "at_corner", "subject_id": "obj_003", "architectural_element": "southwest_corner"},
            {"type": "near_wall", "subject_id": "obj_004", "architectural_element": "east_wall"},
            {"type": "near_wall", "subject_id": "obj_000", "architectural_element": "north_wall"},
            {"type": "room_region", "subject_id": "obj_001", "architectural_element": "north_region", "region": "north"},
            {"type": "room_region", "subject_id": "obj_002", "architectural_element": "south_region", "region": "south"},
            {"type": "room_region", "subject_id": "obj_005", "architectural_element": "east_region", "region": "east"},
        ],
        "collision_moves": [
            {"target": "obj_000", "anchor": "obj_001", "xy": [1.08, 11.03]},
            {"target": "obj_005", "anchor": "obj_004"},
        ],
        "oob_moves": [
            ("obj_000", "north"),
            ("obj_001", "north"),
            ("obj_003", "south"),
            ("obj_004", "east"),
        ],
        "support_targets": ["obj_000", "obj_001", "obj_003", "obj_004"],
        "oar_moves": [
            (0, "obj_003", [0.60, 4.50]),
            (2, "obj_000", [3.00, 9.50]),
            (4, "obj_002", [2.70, 6.00]),
            (5, "obj_005", [0.60, 8.50]),
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "Support" / "Scenes" / "converted_scenes",
    )
    parser.add_argument(
        "--asset-csv",
        type=Path,
        default=PROJECT_ROOT
        / "Support"
        / "Assets"
        / "imaginarium_assets"
        / "imaginarium_asset_info.csv",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=PROJECT_ROOT / "configs" / "experiments" / "p0b_source_distortion5",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_rows = _asset_rows(args.asset_csv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_cases: list[dict[str, Any]] = []

    for spec in SOURCE_CASES:
        source_path = args.source_root / f"{spec['source_scene_id']}.json"
        source_scene = _read_json(source_path)
        base_scene = _canonical_scene(source_scene, spec, asset_rows)
        base_annotation = _reference_annotation(base_scene, spec, request_id=spec["base_case_id"])
        _assert_reference_geometry(base_scene, base_annotation, context=spec["base_case_id"])

        for family in VARIANT_FAMILIES:
            variant_id = f"{spec['base_case_id']}__{family}"
            scene, distortion = _variant(base_scene, spec, family, variant_id)
            request = _scene_request(scene, spec["instruction"])
            annotation = deepcopy(base_annotation)
            annotation["request_id"] = variant_id
            annotation["provenance"] = {
                **annotation["provenance"],
                "base_case_id": spec["base_case_id"],
                "variant_family": family,
            }
            audit = _audit_variant(scene, annotation, distortion)
            distortion["deterministic_audit"] = audit

            case_dir = args.out_root / "fixtures" / variant_id
            case_dir.mkdir(parents=True, exist_ok=True)
            _write_json(case_dir / "generated_scene.json", scene)
            _write_json(case_dir / "scene_request.json", request)
            _write_json(case_dir / "reference_annotation.json", annotation)
            _write_json(case_dir / "distortion_manifest.json", distortion)
            manifest_cases.append(
                {
                    "case_id": variant_id,
                    "base_case_id": spec["base_case_id"],
                    "source_scene_id": spec["source_scene_id"],
                    "difficulty": spec["difficulty"],
                    "family": family,
                    "fixture_dir": f"fixtures/{variant_id}",
                    "object_count": len(scene["objects"]),
                    "instruction": spec["instruction"],
                    "severity": distortion["severity"],
                }
            )

    experiment = {
        "experiment_id": "p0b_source_scene_distortion5_v1",
        "generator_mode": "skipped_frozen_canonical_scene",
        "input_mode": "fine_grained_natural_language_with_private_reference_annotation",
        "source_case_count": len(SOURCE_CASES),
        "variant_families": list(VARIANT_FAMILIES),
        "total_case_count": len(manifest_cases),
        "cases": manifest_cases,
    }
    _write_json(args.out_root / "cases.json", experiment)
    print(f"Prepared {len(manifest_cases)} frozen variants under {args.out_root}")


def _asset_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {str(row.get("name_en") or ""): row for row in csv.DictReader(handle)}


def _canonical_scene(source: dict[str, Any], spec: dict[str, Any], assets: dict[str, dict[str, str]]) -> dict[str, Any]:
    objects: list[dict[str, Any]] = []
    for output_index, source_index in enumerate(spec["keep_indices"]):
        raw = source["objects"][source_index]
        asset_key = str(raw["jid"])
        asset = assets.get(asset_key, {})
        category = str(spec.get("category_overrides", {}).get(source_index) or asset.get("category") or asset.get("class_en") or "object")
        description = str(raw.get("short_desc") or asset.get("short_desc") or raw.get("desc") or category)
        size = [float(value) for value in raw["size"]]
        objects.append(
            {
                "id": f"obj_{output_index:03d}",
                "category": category.replace("_", " ").lower(),
                "description": description,
                "desc": str(raw.get("desc") or asset.get("caption_en") or description),
                "short_desc": description,
                "jid": asset_key,
                "size": size,
                "center": [float(value) for value in raw["center"]],
                "rotation": [float(value) for value in raw["rotation"]],
                "geometry_provenance": "asset_mesh",
                "asset_ref": {
                    "source_db": "imaginarium",
                    "asset_key": asset_key,
                    "mesh_uri": None,
                    "pointcloud_uri": None,
                    "metadata_uri": None,
                },
                "asset_proxy": {
                    "type": "obb_from_source_scene",
                    "bbox_center_local": [0.0, 0.0, 0.0],
                    "bbox_size": size,
                },
                "metadata": {
                    "source_scene_id": spec["source_scene_id"],
                    "source_object_index": int(source_index),
                    "fixture_role": "frozen_source_object",
                    "interactive": False,
                },
            }
        )
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": f"{spec['base_case_id']}__clean",
        "request_id": spec["base_case_id"],
        "scene_type": str(source["scene_type"]),
        "boundary": [[float(value) for value in point] for point in source["boundary"]],
        "scene_height": float(source["scene_height"]),
        "objects": objects,
        "metadata": {
            "coordinate_frame": dict(COORDINATE_FRAME),
            "source_scene_id": spec["source_scene_id"],
            "source_selection": "strict_zero_candidate_collision_oob_support_screen",
            "generator_skipped": True,
        },
    }


def _reference_annotation(scene: dict[str, Any], spec: dict[str, Any], *, request_id: str) -> dict[str, Any]:
    min_x, max_x, min_y, max_y = _room_bounds(scene)
    objects = [
        {
            "id": obj["id"],
            "category": obj["category"],
            "description": obj["description"],
            "count": 1,
            "claim_state": "confirmed",
        }
        for obj in scene["objects"]
    ]
    oor_relations = [
        {**deepcopy(relation), "claim_state": "confirmed", "relation_id": f"oor_{index:03d}"}
        for index, relation in enumerate(spec["oor_relations"])
    ]
    oar_relations = [
        {**deepcopy(relation), "claim_state": "confirmed", "relation_id": f"oar_{index:03d}"}
        for index, relation in enumerate(spec["oar_relations"])
    ]
    return {
        "annotation_version": "reference_annotation_v1",
        "validation_status": "confirmed",
        "source": "manual",
        "request_id": request_id,
        "scene_type": scene["scene_type"],
        "inventory_policy": "closed_world",
        "objects": objects,
        "oor_relations": oor_relations,
        "oar_relations": oar_relations,
        "room_constraints": {
            "claim_state": "confirmed",
            "shape": "rectangular_enclosed_room",
            "dimensions": {
                "width": max_x - min_x,
                "depth": max_y - min_y,
                "height": float(scene["scene_height"]),
            },
        },
        "review": {"status": "approved", "reviewer": "benchmark_author"},
        "provenance": {
            "origin": "manual_source_scene_fixture_conversion",
            "generator_visible": False,
            "leaderboard_ground_truth": False,
            "source_scene_id": spec["source_scene_id"],
        },
    }


def _scene_request(scene: dict[str, Any], instruction: str) -> dict[str, Any]:
    min_x, max_x, min_y, max_y = _room_bounds(scene)
    dimensions = {
        "width": max_x - min_x,
        "depth": max_y - min_y,
        "height": float(scene["scene_height"]),
    }
    return {
        "request_id": scene["request_id"],
        "instruction": instruction,
        "scene_type": scene["scene_type"],
        "structure": False,
        "prompt_granularity": "fine_grained",
        "metadata": {
            "generator_skipped": True,
            "reference_annotation_visibility": "benchmark_private",
        },
        "room": {
            "boundary": deepcopy(scene["boundary"]),
            "height": float(scene["scene_height"]),
            "unit": "meter",
            "dimensions": dimensions,
            "explicit_dimensions": dimensions,
            "dimension_provenance": {axis: "explicit_benchmark_fixture" for axis in dimensions},
            "resolution_policy": "room_dimension_policy_v1",
            "topology": "rectangular_enclosed_room",
            "floor_z": 0.0,
        },
    }


def _variant(
    base_scene: dict[str, Any],
    spec: dict[str, Any],
    family: str,
    variant_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scene = deepcopy(base_scene)
    scene["scene_id"] = variant_id
    scene["request_id"] = variant_id
    scene["metadata"] = {**scene["metadata"], "base_case_id": spec["base_case_id"], "variant_family": family}
    transforms: list[dict[str, Any]] = []
    expected = {
        "collision_candidate_pairs": [],
        "collision_invalid_pairs": [],
        "collision_valid_pairs": [],
        "oob_invalid_object_ids": [],
        "support_invalid_object_ids": [],
        "oar_invalid_relation_indices": [],
        "oar_invalid_relation_ids": [],
    }

    if family == "collision":
        for move in spec["collision_moves"]:
            target_id = str(move["target"])
            anchor_id = str(move["anchor"])
            target = _object(scene, target_id)
            anchor = _object(scene, anchor_id)
            before = deepcopy(target["center"])
            xy = move.get("xy") or anchor["center"][:2]
            z = float(anchor["center"][2]) if move.get("z_mode") == "anchor" else float(target["size"][2]) / 2.0
            target["center"] = [float(xy[0]), float(xy[1]), z]
            transforms.append(_transform(target_id, before, target["center"], reason=f"co_located_with:{anchor_id}"))
            pair = sorted([target_id, anchor_id])
            expected["collision_candidate_pairs"].append(pair)
            expected_label = str(move.get("expected_collision_label", "invalid"))
            if expected_label == "invalid":
                expected["collision_invalid_pairs"].append(pair)
            elif expected_label == "valid":
                expected["collision_valid_pairs"].append(pair)
            else:
                raise ValueError(f"unsupported expected_collision_label: {expected_label!r}")
    elif family == "oob":
        for target_id, plane in spec["oob_moves"]:
            target = _object(scene, target_id)
            before = deepcopy(target["center"])
            _move_partly_outside(scene, target, plane, penetration_fraction=0.35)
            transforms.append(_transform(target_id, before, target["center"], reason=f"35_percent_extent_outside:{plane}"))
            expected["oob_invalid_object_ids"].append(target_id)
    elif family == "support":
        for target_id in spec["support_targets"]:
            target = _object(scene, target_id)
            before = deepcopy(target["center"])
            gap = min(0.25, max(0.12, 0.10 * float(target["size"][2])))
            target["center"][2] = float(target["center"][2]) + gap
            transforms.append(_transform(target_id, before, target["center"], reason=f"positive_support_gap:{gap:.6f}m"))
            expected["support_invalid_object_ids"].append(target_id)
    elif family == "oar":
        for relation_index, target_id, xy in spec["oar_moves"]:
            target = _object(scene, target_id)
            before = deepcopy(target["center"])
            target["center"] = [float(xy[0]), float(xy[1]), float(target["size"][2]) / 2.0]
            transforms.append(_transform(target_id, before, target["center"], reason=f"violates_oar_relation:{relation_index}"))
            expected["oar_invalid_relation_indices"].append(int(relation_index))
            expected["oar_invalid_relation_ids"].append(f"oar_{int(relation_index):03d}")
    elif family != "clean":
        raise ValueError(f"unsupported variant family: {family}")

    target_object_ids = sorted({item["object_id"] for item in transforms})
    relation_denominator = len(spec["oar_relations"])
    severity = {
        "family": family,
        "target_object_count": len(target_object_ids),
        "object_denominator": len(scene["objects"]),
        "target_object_fraction": len(target_object_ids) / len(scene["objects"]),
        "target_relation_count": len(expected["oar_invalid_relation_indices"]),
        "relation_denominator": relation_denominator,
        "target_relation_fraction": (
            len(expected["oar_invalid_relation_indices"]) / relation_denominator if relation_denominator else 0.0
        ),
        "collision_pair_count": len(expected["collision_invalid_pairs"]),
        "collision_candidate_pair_count": len(expected["collision_candidate_pairs"]),
    }
    return scene, {
        "schema_version": "controlled_scene_distortion_v1",
        "case_id": variant_id,
        "base_case_id": spec["base_case_id"],
        "source_scene_id": spec["source_scene_id"],
        "generator_skipped": True,
        "family": family,
        "severity": severity,
        "transforms": transforms,
        "expected": expected,
        "gt_policy": "programmatic_from_frozen_transform_not_from_evaluator_prediction",
    }


def _audit_variant(scene: dict[str, Any], annotation: dict[str, Any], distortion: dict[str, Any]) -> dict[str, Any]:
    validate_generated_scene(scene)
    validate_scene_request(_scene_request(scene, "fixture validation instruction"))
    validate_reference_annotation(annotation)
    report = evaluate_generic_validity(scene, config=DETECTOR_ONLY_CONFIG)
    metrics = report["metrics"]
    collision_pairs = sorted(
        sorted([str(item["object_a"]), str(item["object_b"])])
        for item in metrics["collision"].get("pairs", [])
        if item.get("requires_vlm")
    )
    oob_ids = sorted(
        str(item["object_id"])
        for item in metrics["oob"].get("objects", [])
        if item.get("requires_vlm")
    )
    support_ids = sorted(
        str(item["object_id"])
        for item in metrics["support"].get("objects", [])
        if item.get("requires_vlm")
    )
    expected = distortion["expected"]
    expected_collision = sorted(
        sorted(pair)
        for pair in (
            expected.get("collision_candidate_pairs")
            or [
                *expected.get("collision_invalid_pairs", []),
                *expected.get("collision_valid_pairs", []),
            ]
        )
    )
    expected_oob = sorted(expected["oob_invalid_object_ids"])
    expected_support = sorted(expected["support_invalid_object_ids"])
    family = distortion["family"]

    if family in {"clean", "collision", "oob", "support", "oar"}:
        if collision_pairs != expected_collision:
            raise RuntimeError(f"{distortion['case_id']}: collision audit {collision_pairs} != {expected_collision}")
        if oob_ids != expected_oob:
            raise RuntimeError(f"{distortion['case_id']}: OOB audit {oob_ids} != {expected_oob}")
        if not set(expected_support).issubset(support_ids):
            raise RuntimeError(
                f"{distortion['case_id']}: Support bypassed known-invalid objects; "
                f"routed={support_ids}, required={expected_support}"
            )

    oor_report = evaluate_oor(scene, relation_specs=_relation_specs(annotation, "oor_relations"))
    oor_passes = [check.get("passed") for check in oor_report.get("checks", [])]
    if any(passed is False for passed in oor_passes):
        raise RuntimeError(f"{distortion['case_id']}: non-target OOR claim changed: {oor_passes}")

    oar_report = evaluate_oar(scene, relation_specs=_relation_specs(annotation, "oar_relations"))
    oar_passes = [check.get("passed") for check in oar_report.get("checks", [])]
    if family != "oar" and any(passed is False for passed in oar_passes):
        raise RuntimeError(f"{distortion['case_id']}: non-target OAR claim changed: {oar_passes}")
    if family == "oar":
        expected_failed = set(expected["oar_invalid_relation_ids"])
        actual_failed = {
            str(check.get("relation_id"))
            for check in oar_report.get("checks", [])
            if check.get("passed") is False
        }
        if actual_failed != expected_failed:
            raise RuntimeError(f"{distortion['case_id']}: OAR failures {actual_failed} != {expected_failed}")

    return {
        "collision_routed_pairs": collision_pairs,
        "oob_routed_object_ids": oob_ids,
        "support_routed_object_ids": support_ids,
        "support_required_invalid_object_ids": expected_support,
        "support_extra_candidate_object_ids": sorted(set(support_ids) - set(expected_support)),
        "oor_pass_vector": oor_passes,
        "oar_pass_vector": oar_passes,
        "cross_metric_isolation_verified": True,
    }


def _assert_reference_geometry(scene: dict[str, Any], annotation: dict[str, Any], *, context: str) -> None:
    validate_generated_scene(scene)
    validate_reference_annotation(annotation)
    oor = evaluate_oor(scene, relation_specs=_relation_specs(annotation, "oor_relations"))
    oar = evaluate_oar(scene, relation_specs=_relation_specs(annotation, "oar_relations"))
    failed_oor = [check for check in oor.get("checks", []) if check.get("passed") is False]
    failed_oar = [check for check in oar.get("checks", []) if check.get("passed") is False]
    if failed_oor or failed_oar:
        raise RuntimeError(f"{context}: reference relation audit failed: OOR={failed_oor}, OAR={failed_oar}")
    generic = evaluate_generic_validity(scene, config=DETECTOR_ONLY_CONFIG)
    metrics = generic["metrics"]
    routed = {
        "collision": int(metrics["collision"].get("requires_vlm_count", 0)),
        "oob": int(metrics["oob"].get("requires_vlm_count", 0)),
        "support": int(metrics["support"].get("requires_vlm_count", 0)),
    }
    # Collision/OOB candidate sets are frozen for these source fixtures.
    # Support is a one-sided high-recall router, so clean source scenes may
    # legitimately contain positive-gap or ambiguous-attachment candidates.
    if routed["collision"] or routed["oob"]:
        raise RuntimeError(f"{context}: source scene is not detector-clean: {routed}")


def _relation_specs(annotation: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [
        {field: deepcopy(value) for field, value in relation.items() if field != "claim_state"}
        for relation in annotation.get(key, [])
    ]


def _move_partly_outside(scene: dict[str, Any], obj: dict[str, Any], plane: str, *, penetration_fraction: float) -> None:
    min_x, max_x, min_y, max_y = _room_bounds(scene)
    half_x, half_y = _rotated_half_extents_xy(obj)
    inside_center_fraction = 1.0 - 2.0 * float(penetration_fraction)
    if plane == "west":
        obj["center"][0] = min_x + inside_center_fraction * half_x
    elif plane == "east":
        obj["center"][0] = max_x - inside_center_fraction * half_x
    elif plane == "south":
        obj["center"][1] = min_y + inside_center_fraction * half_y
    elif plane == "north":
        obj["center"][1] = max_y - inside_center_fraction * half_y
    else:
        raise ValueError(f"unknown OOB plane: {plane}")
    obj["center"][2] = float(obj["size"][2]) / 2.0


def _rotated_half_extents_xy(obj: dict[str, Any]) -> tuple[float, float]:
    half_x = float(obj["size"][0]) / 2.0
    half_y = float(obj["size"][1]) / 2.0
    yaw = math.radians(float(obj["rotation"][2]))
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    return c * half_x + s * half_y, s * half_x + c * half_y


def _room_bounds(scene: dict[str, Any]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in scene["boundary"]]
    ys = [float(point[1]) for point in scene["boundary"]]
    return min(xs), max(xs), min(ys), max(ys)


def _object(scene: dict[str, Any], object_id: str) -> dict[str, Any]:
    for obj in scene["objects"]:
        if obj["id"] == object_id:
            return obj
    raise KeyError(object_id)


def _transform(object_id: str, before: list[float], after: list[float], *, reason: str) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "before_center": [float(value) for value in before],
        "after_center": [float(value) for value in after],
        "reason": reason,
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
