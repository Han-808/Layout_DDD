#!/usr/bin/env python3
"""Build the frozen cal_dataset1 deterministic-calibration fixtures.

The builder is deliberately model-free. Source scenes retain their frozen
Imaginarium asset IDs; constructed scenes reuse frozen Top-1 selections or an
explicit exact catalog lookup. Semantic GT is authored from controlled
transforms, never from evaluator predictions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmark.reference_annotation import (  # noqa: E402
    build_reference_annotation_draft,
    confirm_reference_annotation,
    ensure_reference_relation_ids,
    validate_reference_annotation,
)
from benchmark.scene_io.validate import (  # noqa: E402
    validate_asset_selection,
    validate_generated_scene,
    validate_object_plan,
    validate_scene_request,
)


DATASET_ID = "cal_dataset1"
SCHEMA_VERSION = "calibration_dataset_v2"
HUMAN_REVIEWER = "human_benchmark_owner"
HUMAN_REVIEWED_AT = "2026-07-21"
FINE_CONVERTER = {
    "mode": "offline_assistant_authored",
    "agent": "Codex",
    "model_family": "GPT-5",
    "runtime_benchmark_component": False,
    "generator_visible": False,
}
COORDINATE_FRAME = {
    "origin": "room_min_corner_floor",
    "axes": "x_width_y_depth_z_up",
    "unit": "meter",
    "rotation_unit": "degree",
}
SOURCE_SCENES = [
    "scene_000098_01",
    "scene_000145_02",
    "scene_000246_02",
    "scene_000306_02",
    "scene_000386_00",
    "scene_000404_01",
    "scene_000581_06",
    "scene_001037_01",
    "scene_001315_04",
    "scene_001318_02",
]
FINE_CASES = ["easy_001", "easy_003", "easy_004", "easy_007", "prompt_010"]
FROZEN_SELECTIONS = {
    "easy_001": "Support/artifacts/result/qwen32b_camera_pose14_incremental_20260717/bbox_track/easy_001/asset_selection.json",
    "easy_003": "Support/artifacts/result/qwen32b_camera_pose14_incremental_20260717/bbox_track/easy_003/asset_selection.json",
    "easy_004": "Support/artifacts/result/qwen32b_camera_pose14_incremental_20260717/bbox_track/easy_004/asset_selection.json",
    "easy_007": "Support/artifacts/result/qwen32b_camera_pose14_incremental_20260717/bbox_track/easy_007/asset_selection.json",
    "prompt_010": "Support/artifacts/result/camera_pose14_first7_bbox/prompt_010/05_asset_selection.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "Support" / "datasets" / DATASET_ID,
    )
    parser.add_argument(
        "--ontology",
        type=Path,
        default=Path("/tmp/cal_dataset1_SceneOnto.json"),
        help="Fixed-revision SceneOnto JSON downloaded from the public dataset repo.",
    )
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    catalog_path = (
        REPO_ROOT
        / "Support"
        / "Assets"
        / "imaginarium_assets"
        / "imaginarium_asset_info.csv"
    )
    catalog = _read_catalog(catalog_path)

    ontology_source = args.ontology.resolve()
    existing_ontology = out_root / "ontology/SceneOnto.json"
    if not ontology_source.is_file() and existing_ontology.is_file():
        ontology_source = existing_ontology
    if not ontology_source.is_file():
        raise FileNotFoundError(
            "SceneOnto.json is required. Pass --ontology with the fixed-revision artifact."
        )
    ontology_target = out_root / "ontology/SceneOnto.json"
    ontology_target.parent.mkdir(parents=True, exist_ok=True)
    if ontology_source != ontology_target:
        shutil.copyfile(ontology_source, ontology_target)

    cases: list[dict[str, Any]] = []
    source_bases: dict[str, dict[str, Any]] = {}

    for index, source_id in enumerate(SOURCE_SCENES, start=1):
        case_id = f"source_valid_{index:03d}"
        source_path = REPO_ROOT / "Support" / "Scenes" / "converted_scenes" / f"{source_id}.json"
        source = _read_json(source_path)
        scene = _canonical_source_scene(source, source_id=source_id, case_id=case_id, catalog=catalog)
        instruction = f"Create a plausible {str(source['scene_type']).replace('_', ' ')} with this frozen source-scene inventory."
        request = _scene_request(scene, instruction, prompt_granularity="coarse_grained")
        plan = _object_plan_from_scene(scene, instruction, prompt_granularity="coarse_grained")
        assets = _asset_selection_from_scene(
            scene,
            catalog,
            policy="frozen_source_asset_id",
            source=f"Support/Scenes/converted_scenes/{source_id}.json",
        )
        provenance = {
            "schema_version": "calibration_case_provenance_v1",
            "case_id": case_id,
            "origin": "repository_source_scene",
            "source_path": source_path.relative_to(REPO_ROOT).as_posix(),
            "source_sha256": _sha256(source_path),
            "source_validity_policy": "source_dataset_plus_human_visual_audit",
            "selection_policy": "minimum_object_count_scene_type_stratified_independent_of_detector_output",
            "asset_policy": "preserve_frozen_source_jid",
        }
        event_gt = _event_gt(scene, status="confirmed_programmatic", intended={})
        review = {
            "schema_version": "calibration_review_v1",
            "status": "approved_human",
            "required": False,
            "reason": "Human audit approved this source scene as a shared spatial-validity control.",
            "reviewer": HUMAN_REVIEWER,
            "reviewed_at": HUMAN_REVIEWED_AT,
        }
        _write_case(
            out_root,
            case_id,
            scene=scene,
            request=request,
            plan=plan,
            assets=assets,
            provenance=provenance,
            event_gt=event_gt,
            review=review,
        )
        case = _case_record(
            case_id,
            split="source_valid",
            prompt_granularity="coarse_grained",
            evaluation_scope="deterministic_full",
            scene=scene,
            review_status=review["status"],
            base_case_id=None,
            source_scene_id=source_id,
        )
        cases.append(case)
        source_bases[case_id] = {
            "scene": scene,
            "request": request,
            "plan": plan,
            "assets": assets,
            "source_scene_id": source_id,
        }

    obvious_specs = [
        ("obvious_collision_001", "source_valid_001", [("collision", ["obj_000", "obj_001"], "obvious")]),
        ("obvious_collision_002", "source_valid_002", [("collision", ["obj_003", "obj_004"], "obvious")]),
        ("obvious_oob_001", "source_valid_003", [("oob", ["obj_000"], "north", "obvious")]),
        ("obvious_oob_002", "source_valid_004", [("oob", ["obj_001"], "east", "obvious")]),
        ("obvious_support_001", "source_valid_005", [("support", ["obj_001"], 0.18, "obvious")]),
        ("obvious_support_002", "source_valid_006", [("support", ["obj_000"], 0.18, "obvious")]),
        (
            "obvious_combined_001",
            "source_valid_007",
            [
                ("collision", ["obj_001", "obj_002"], "obvious"),
                ("oob", ["obj_004"], "north", "obvious"),
                ("support", ["obj_003"], 0.18, "obvious"),
            ],
        ),
        (
            "obvious_combined_002",
            "source_valid_008",
            [
                ("collision", ["obj_001", "obj_003"], "obvious"),
                ("oob", ["obj_002"], "west", "obvious"),
                ("support", ["obj_004"], 0.18, "obvious"),
            ],
        ),
    ]
    subtle_specs = [
        ("subtle_collision_001", "source_valid_001", [("collision", ["obj_000", "obj_001"], "subtle")]),
        ("subtle_collision_002", "source_valid_002", [("collision", ["obj_003", "obj_004"], 0.08, "subtle")]),
        ("subtle_oob_001", "source_valid_003", [("oob", ["obj_000"], "north", "subtle")]),
        ("subtle_oob_002", "source_valid_004", [("oob", ["obj_001"], "east", "subtle")]),
        ("subtle_support_001", "source_valid_005", [("support", ["obj_001"], 0.04, "subtle")]),
        ("subtle_support_002", "source_valid_006", [("support", ["obj_000"], 0.04, "subtle")]),
        (
            "subtle_combined_001",
            "source_valid_009",
            [
                ("collision", ["obj_001", "obj_004"], "subtle"),
                ("oob", ["obj_002"], "east", "subtle"),
                ("support", ["obj_003"], 0.04, "subtle"),
            ],
        ),
        (
            "subtle_combined_002",
            "source_valid_010",
            [
                ("collision", ["obj_000", "obj_001"], "subtle"),
                ("oob", ["obj_002"], "east", "subtle"),
                ("support", ["obj_004"], 0.04, "subtle"),
            ],
        ),
    ]
    for split, specs in (("obvious_distortion", obvious_specs), ("subtle_distortion", subtle_specs)):
        for case_id, base_case_id, operations in specs:
            base = source_bases[base_case_id]
            scene, distortion, intended = _distort_scene(
                base["scene"], case_id=case_id, base_case_id=base_case_id, operations=operations
            )
            instruction = str(base["request"]["instruction"])
            request = _scene_request(scene, instruction, prompt_granularity="coarse_grained")
            plan = deepcopy(base["plan"])
            plan["request_id"] = case_id
            assets = deepcopy(base["assets"])
            assets["request_id"] = case_id
            status = "confirmed_programmatic" if split == "obvious_distortion" else "confirmed_human"
            event_gt = _event_gt(scene, status=status, intended=intended)
            human_reviewed = split == "subtle_distortion"
            review = {
                "schema_version": "calibration_review_v1",
                "status": "approved_human" if human_reviewed else "approved_programmatic",
                "required": False,
                "reason": (
                    "Human audit approved the subtle controlled distortion as the intended invalid event."
                    if human_reviewed
                    else "Large controlled transform is an obvious invalid geometry event."
                ),
                "review_items": list(intended.values()),
            }
            if human_reviewed:
                review.update({"reviewer": HUMAN_REVIEWER, "reviewed_at": HUMAN_REVIEWED_AT})
            provenance = {
                "schema_version": "calibration_case_provenance_v1",
                "case_id": case_id,
                "origin": "controlled_transform_from_source_valid",
                "base_case_id": base_case_id,
                "source_scene_id": base["source_scene_id"],
                "gt_policy": "controlled_transform_not_evaluator_prediction",
            }
            _write_case(
                out_root,
                case_id,
                scene=scene,
                request=request,
                plan=plan,
                assets=assets,
                provenance=provenance,
                event_gt=event_gt,
                review=review,
                perturbation=distortion,
            )
            cases.append(
                _case_record(
                    case_id,
                    split=split,
                    prompt_granularity="coarse_grained",
                    evaluation_scope="deterministic_full",
                    scene=scene,
                    review_status=review["status"],
                    base_case_id=base_case_id,
                    source_scene_id=base["source_scene_id"],
                )
            )

    fine_prompts: list[dict[str, Any]] = []
    for source_case in FINE_CASES:
        case_id = f"fine_edge_{source_case}"
        built = _build_fine_case(source_case, case_id=case_id)
        scene = built["scene"]
        request = _scene_request(scene, built["prompt"], prompt_granularity="fine_grained")
        event_gt = _event_gt(scene, status="confirmed_human_edge_cases", intended=built["intended"])
        review = {
            "schema_version": "calibration_review_v1",
            "status": "approved_human",
            "required": False,
            "reason": "Human audit approved all three events as deliberate semantic edge cases; ambiguous labels remain excluded from binary accuracy.",
            "reviewer": HUMAN_REVIEWER,
            "reviewed_at": HUMAN_REVIEWED_AT,
            "review_items": list(built["intended"].values()),
            "prompt": built["prompt"],
            "known_asset_notes": built["asset_review_notes"],
        }
        provenance = {
            "schema_version": "calibration_case_provenance_v1",
            "case_id": case_id,
            "origin": "manual_canonical_construction_from_assistant_converted_fine_prompt",
            "prompt_inventory_source": f"configs/experiments/camera_pose14/generator_structures/{source_case}.json",
            "legacy_reference_consulted": f"configs/experiments/camera_pose14/reference_annotations/{source_case}.json",
            "converter": dict(FINE_CONVERTER),
            "asset_selection_source": FROZEN_SELECTIONS[source_case],
            "asset_policy": "reuse_frozen_deterministic_top1_selection",
            "local_embedding_retrieval_rerun": False,
        }
        _write_case(
            out_root,
            case_id,
            scene=scene,
            request=request,
            plan=built["plan"],
            assets=built["assets"],
            provenance=provenance,
            event_gt=event_gt,
            review=review,
            reference_annotation=built["reference_annotation"],
            corner_manifest=built["corner_manifest"],
        )
        cases.append(
            _case_record(
                case_id,
                split="fine_edge",
                prompt_granularity="fine_grained",
                evaluation_scope="deterministic_full",
                scene=scene,
                review_status="approved_human",
                base_case_id=source_case,
                source_scene_id=None,
            )
        )
        fine_prompts.append(
            {
                "case_id": case_id,
                "source_case_id": source_case,
                "prompt": built["prompt"],
                "corner_cases": list(built["intended"].values()),
            }
        )

    constructed_specs = [
        {
            "case_id": "scale_outlier_giant_clock",
            "split": "scale_outlier",
            "scope": "scale_only",
            "scene_type": "living room",
            "instruction": "Create a living room containing one freestanding round wall clock.",
            "objects": [
                ("obj_000", "18_SM_Wall_Clock", "clock", [3.2, 0.57, 3.2], [3.0, 3.0, 1.6]),
            ],
            "intended_metric": "scale",
            "intended_event": "obj_000",
            "semantic_label": "invalid",
        },
        {
            "case_id": "scale_outlier_tiny_bed",
            "split": "scale_outlier",
            "scope": "scale_only",
            "scene_type": "bedroom",
            "instruction": "Create a bedroom containing one unusually tiny upholstered bed.",
            "objects": [
                ("obj_000", "0_SM_Bed", "bed", [0.35, 0.37, 0.18], [3.0, 3.0, 0.09]),
            ],
            "intended_metric": "scale",
            "intended_event": "obj_000",
            "semantic_label": "invalid",
        },
        {
            "case_id": "cooccurrence_keyboard_toilet",
            "split": "cooccurrence_outlier",
            "scope": "cooccurrence_only",
            "scene_type": "home office",
            "instruction": "Create a home office containing a keyboard and a toilet.",
            "objects": [
                ("obj_000", "0_SM_Keyboard", "keyboard", None, [2.0, 2.0, None]),
                ("obj_001", "0_SM_Toilet", "toilet", None, [4.5, 4.5, None]),
            ],
            "intended_metric": "cooccurrence_plausibility",
            "intended_event": "keyboard|toilet",
            "semantic_label": "invalid",
        },
        {
            "case_id": "cooccurrence_oven_shower",
            "split": "cooccurrence_outlier",
            "scope": "cooccurrence_only",
            "scene_type": "bedroom",
            "instruction": "Create a bedroom containing a countertop oven and a shower enclosure.",
            "objects": [
                ("obj_000", "0_SM_Deco021_02", "oven", None, [2.0, 2.0, None]),
                ("obj_001", "17_SM_Shower_Cabin", "shower", None, [4.5, 4.5, None]),
            ],
            "intended_metric": "cooccurrence_plausibility",
            "intended_event": "oven|shower",
            "semantic_label": "invalid",
        },
    ]
    for spec in constructed_specs:
        case_id = str(spec["case_id"])
        scene = _constructed_scene(spec, catalog)
        request = _scene_request(scene, str(spec["instruction"]), prompt_granularity="coarse_grained")
        plan = _object_plan_from_scene(scene, str(spec["instruction"]), prompt_granularity="coarse_grained")
        assets = _asset_selection_from_scene(
            scene,
            catalog,
            policy="exact_catalog_asset_lookup",
            source="Support/Assets/imaginarium_assets/imaginarium_asset_info.csv",
        )
        event_gt = {
            "schema_version": "deterministic_event_gt_v1",
            "case_id": case_id,
            "status": (
                "confirmed_human"
                if spec["split"] == "cooccurrence_outlier"
                else "confirmed_programmatic"
            ),
            "events": [
                {
                    "metric": spec["intended_metric"],
                    "event_id": spec["intended_event"],
                    "object_ids": [obj[0] for obj in spec["objects"]],
                    "semantic_label": spec["semantic_label"],
                    "route_requirement": "must_route",
                    "gt_basis": "manual_construction_independent_of_evaluator_prediction",
                }
            ],
            "scene_metrics": {},
        }
        human_reviewed = spec["split"] == "cooccurrence_outlier"
        review = {
            "schema_version": "calibration_review_v1",
            "status": "approved_human" if human_reviewed else "approved_programmatic",
            "required": False,
            "reason": (
                "Human audit confirmed the deliberately incoherent category pair as an invalid co-occurrence control."
                if human_reviewed
                else "The canonical target dimensions are an obvious category-scale outlier."
            ),
        }
        if human_reviewed:
            review.update({"reviewer": HUMAN_REVIEWER, "reviewed_at": HUMAN_REVIEWED_AT})
        provenance = {
            "schema_version": "calibration_case_provenance_v1",
            "case_id": case_id,
            "origin": "manual_canonical_construction",
            "asset_policy": "exact_catalog_asset_lookup",
            "ontology_dependency": "ontology/SceneOnto.json",
        }
        _write_case(
            out_root,
            case_id,
            scene=scene,
            request=request,
            plan=plan,
            assets=assets,
            provenance=provenance,
            event_gt=event_gt,
            review=review,
        )
        cases.append(
            _case_record(
                case_id,
                split=spec["split"],
                prompt_granularity="coarse_grained",
                evaluation_scope=spec["scope"],
                scene=scene,
                review_status=review["status"],
                base_case_id=None,
                source_scene_id=None,
            )
        )

    config = {
        "collision": {"enabled": True, "official_mode": False, "detector_only": True},
        "oob": {"enabled": True, "official_mode": False, "detector_only": True},
        "support": {"enabled": True, "official_mode": False, "detector_only": True},
        "navigability": {"enabled": True},
        "accessibility": {"enabled": True},
    }
    _write_json(out_root / "configs/deterministic_full.json", config)
    _write_json(
        out_root / "cases.json",
        {
            "dataset_id": DATASET_ID,
            "schema_version": SCHEMA_VERSION,
            "default_prompt_granularity": "coarse_grained",
            "shared_deterministic_metrics": [
                "collision",
                "oob",
                "support",
                "navigability",
                "accessibility",
            ],
            "excluded_from_deterministic_full": [
                "scale",
                "cooccurrence_plausibility",
                "functional_grouping",
            ],
            "cases": cases,
        },
    )
    _write_json(out_root / "review/fine_prompts.json", {"cases": fine_prompts})
    _write_json(
        out_root / "review/human_audit_approval.json",
        {
            "dataset_id": DATASET_ID,
            "status": "approved",
            "reviewer": HUMAN_REVIEWER,
            "reviewed_at": HUMAN_REVIEWED_AT,
            "approved_splits": [
                "source_valid",
                "subtle_distortion",
                "fine_edge",
                "cooccurrence_outlier",
            ],
            "fine_edge_semantics": (
                "approved as routing edge cases; labels remain ambiguous rather than "
                "inventing binary validity"
            ),
        },
    )
    _write_review_queue(out_root, cases)
    ontology_sha = _sha256(ontology_target)
    _write_json(
        out_root / "dataset_manifest.json",
        {
            "dataset_id": DATASET_ID,
            "schema_version": "calibration_dataset_manifest_v2",
            "case_count": len(cases),
            "split_counts": _counts(cases, "split"),
            "evaluation_scope_counts": _counts(cases, "evaluation_scope"),
            "prompt_granularity_counts": _counts(cases, "prompt_granularity"),
            "source_path_resolution": {
                "requested": "Support/Scenes/ConcertedScenes",
                "requested_path_exists": False,
                "used": "Support/Scenes/converted_scenes",
                "reason": "ConcertedScenes is absent; existing source-distortion tooling also targets converted_scenes.",
            },
            "asset_catalog": {
                "path": catalog_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(catalog_path),
                "policy": "frozen source/frozen Top-1/exact catalog lookup; no local embedding-index rerun",
            },
            "ontology": {
                "path": "ontology/SceneOnto.json",
                "sha256": ontology_sha,
                "source": "https://huggingface.co/datasets/neurips-25121999/SceneOnto",
                "revision": "54a93fa492c93a79009b45c73567615790fd3971",
                "license": "CC BY-NC-SA 4.0",
            },
            "gt_policy": (
                "semantic GT authored independently of evaluator outputs; human-approved Fine edge "
                "events remain ambiguous and are excluded from binary accuracy"
            ),
            "human_audit": {
                "status": "approved",
                "reviewer": HUMAN_REVIEWER,
                "reviewed_at": HUMAN_REVIEWED_AT,
            },
            "experiment_entrypoint": {
                "path": "Support/legacy/scripts/run_cal_dataset1_experiment.py",
                "default_track": "deterministic",
                "tracks": ["deterministic", "spatial", "fine_fidelity"],
                "frozen_scene_input": True,
                "runtime_converter_called": False,
                "generator_called": False,
                "retrieval_called": False,
            },
        },
    )
    (out_root / "README.md").write_text(_readme(cases, fine_prompts), encoding="utf-8")
    print(json.dumps({"dataset": str(out_root), "case_count": len(cases), "split_counts": _counts(cases, "split")}, indent=2))


def _canonical_source_scene(
    source: dict[str, Any], *, source_id: str, case_id: str, catalog: dict[str, dict[str, str]]
) -> dict[str, Any]:
    objects = []
    for index, raw in enumerate(source["objects"]):
        jid = str(raw["jid"])
        row = catalog[jid]
        size = [float(value) for value in raw["size"]]
        description = str(raw.get("short_desc") or row.get("short_desc") or row.get("category") or "object")
        objects.append(
            _object(
                object_id=f"obj_{index:03d}",
                jid=jid,
                category=str(row.get("category") or row.get("class_en") or "object").lower(),
                description=description,
                size=size,
                center=[float(value) for value in raw["center"]],
                rotation=[float(value) for value in raw["rotation"]],
                metadata={
                    "interactive": False,
                    "source_scene_id": source_id,
                    "source_object_index": index,
                    "fixture_role": "frozen_source_object",
                },
            )
        )
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": case_id,
        "request_id": case_id,
        "scene_type": str(source["scene_type"]),
        "boundary": [[float(value) for value in point] for point in source["boundary"]],
        "scene_height": float(source["scene_height"]),
        "objects": objects,
        "metadata": {
            "coordinate_frame": dict(COORDINATE_FRAME),
            "source_scene_id": source_id,
            "generator_skipped": True,
            "source_selection": "minimum_object_count_stratified_independent_of_detector_output",
        },
    }


def _distort_scene(
    base_scene: dict[str, Any], *, case_id: str, base_case_id: str, operations: list[tuple[Any, ...]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    scene = deepcopy(base_scene)
    scene["scene_id"] = case_id
    scene["request_id"] = case_id
    scene["metadata"] = {**scene["metadata"], "base_case_id": base_case_id, "variant": case_id}
    changes: list[dict[str, Any]] = []
    intended: dict[str, dict[str, Any]] = {}
    for op_index, operation in enumerate(operations, start=1):
        metric = str(operation[0])
        object_ids = [str(value) for value in operation[1]]
        severity = str(operation[-1])
        if metric == "collision":
            target, anchor = (_find(scene, value) for value in object_ids)
            before = deepcopy(target["center"])
            if severity == "obvious":
                target["center"] = [
                    float(anchor["center"][0]),
                    float(anchor["center"][1]),
                    float(target["size"][2]) / 2.0,
                ]
                parameters = {"mode": "co_located_xy", "overlap_class": "obvious"}
            else:
                overlap_m = float(operation[2]) if len(operation) == 4 else 0.02
                _place_shallow_overlap(scene, target, anchor, overlap_m=overlap_m)
                parameters = {"mode": "shallow_obb_overlap", "target_overlap_m": overlap_m}
            event_id = "|".join(sorted(object_ids))
        elif metric == "oob":
            target = _find(scene, object_ids[0])
            before = deepcopy(target["center"])
            plane = str(operation[2])
            outside_m = 0.02 if severity == "subtle" else None
            _move_oob(scene, target, plane=plane, outside_m=outside_m, penetration_fraction=0.35)
            parameters = (
                {"plane": plane, "outside_m": 0.02}
                if severity == "subtle"
                else {"plane": plane, "penetration_fraction": 0.35}
            )
            event_id = object_ids[0]
        elif metric == "support":
            target = _find(scene, object_ids[0])
            before = deepcopy(target["center"])
            gap = float(operation[2])
            target["center"][2] = float(target["center"][2]) + gap
            parameters = {"positive_vertical_gap_m": gap}
            event_id = object_ids[0]
        else:
            raise ValueError(f"unsupported distortion metric: {metric}")
        after = deepcopy(_find(scene, object_ids[0])["center"])
        perturbation_id = f"p{op_index:03d}"
        changes.append(
            {
                "perturbation_id": perturbation_id,
                "metric": metric,
                "object_ids": object_ids,
                "field": "center",
                "before": before,
                "after": after,
                "parameters": parameters,
                "construction_basis": "programmatic_transform",
            }
        )
        intended[f"{metric}:{event_id}"] = {
            "metric": metric,
            "event_id": event_id,
            "object_ids": object_ids,
            "semantic_label": "invalid",
            "route_requirement": "must_route",
            "gt_basis": "controlled_transform" if severity == "obvious" else "human_reviewed_controlled_transform",
            "severity_class": severity,
            "perturbation_id": perturbation_id,
        }
    return scene, {
        "schema_version": "controlled_scene_distortion_v2",
        "case_id": case_id,
        "base_case_id": base_case_id,
        "severity_class": str(operations[0][-1]),
        "changes": changes,
        "gt_policy": "programmatic_transform_not_evaluator_prediction",
    }, intended


def _build_fine_case(source_case: str, *, case_id: str) -> dict[str, Any]:
    structure_path = REPO_ROOT / f"configs/experiments/camera_pose14/generator_structures/{source_case}.json"
    annotation_path = REPO_ROOT / f"configs/experiments/camera_pose14/reference_annotations/{source_case}.json"
    selection_path = REPO_ROOT / FROZEN_SELECTIONS[source_case]
    source_plan = _read_json(structure_path)
    prompt = str(source_plan["scene_description"])
    plan = _assistant_convert_fine_plan(source_case, source_plan, case_id=case_id)
    assets = deepcopy(_read_json(selection_path))
    assets["request_id"] = case_id
    selected = {str(item["object_id"]): item["selected_asset"] for item in assets["objects"]}
    placements = _fine_placements(source_case, selected)
    plan_objects = {str(item["id"]): item for item in plan["objects"]}
    objects: list[dict[str, Any]] = []
    for object_id, placement in placements.items():
        plan_id = str(placement.get("plan_id") or object_id)
        plan_obj = plan_objects[plan_id]
        asset = selected[plan_id]
        objects.append(
            _object(
                object_id=object_id,
                jid=str(asset["jid"]),
                category=str(plan_obj["category"]),
                description=str(plan_obj["description"]),
                size=[float(value) for value in asset["size"]],
                center=[float(value) for value in placement["center"]],
                rotation=[float(value) for value in placement.get("rotation", [0.0, 0.0, 0.0])],
                metadata={
                    "interactive": False,
                    "fixture_role": "fine_prompt_edge_case",
                    "plan_object_id": plan_id,
                    "asset_selection_source": FROZEN_SELECTIONS[source_case],
                },
            )
        )
    room = _fine_room(source_case)
    scene = {
        "schema_version": "canonical_scene_v1",
        "scene_id": case_id,
        "request_id": case_id,
        "scene_type": str(plan["scene_type"]),
        "boundary": [[0.0, 0.0], [room[0], 0.0], [room[0], room[1]], [0.0, room[1]]],
        "scene_height": room[2],
        "objects": objects,
        "metadata": {
            "coordinate_frame": dict(COORDINATE_FRAME),
            "generator_skipped": True,
            "construction": "fresh_canonical_pose_from_existing_fine_prompt",
        },
    }
    intended = _fine_intended(source_case)
    reference = build_reference_annotation_draft(
        plan,
        {
            "request_id": case_id,
            "instruction": prompt,
            "scene_type": str(plan["scene_type"]),
            "prompt_granularity": "fine_grained",
        },
        source="manual",
    )
    for index, obj in enumerate(reference["objects"]):
        obj["provenance"] = {
            "origin": "assistant_authored_conversion",
            "object_plan_index": index,
        }
    if source_case == "prompt_010":
        reference["room_constraints"] = {
            "claim_state": "confirmed",
            "dimensions": {"width": 8.0, "depth": 9.0, "height": 3.0},
            "shape": "rectangular_enclosed_room",
        }
    reference["provenance"] = {
        "origin": "assistant_authored_fine_prompt_conversion",
        "source_case_id": source_case,
        "legacy_reference_consulted": annotation_path.relative_to(REPO_ROOT).as_posix(),
        "converter": dict(FINE_CONVERTER),
        "leaderboard_ground_truth": False,
    }
    reference = confirm_reference_annotation(
        reference,
        inventory_policy="closed_world",
        reviewer=HUMAN_REVIEWER,
        reviewed_at=HUMAN_REVIEWED_AT,
    )
    reference["review"] = {
        "status": "approved",
        "reviewer": HUMAN_REVIEWER,
        "reviewed_at": HUMAN_REVIEWED_AT,
    }
    reference["provenance"]["approval_mode"] = "human_review_of_assistant_conversion"
    reference = ensure_reference_relation_ids(reference)
    validate_reference_annotation(reference)
    corner_manifest = {
        "schema_version": "fine_edge_case_manifest_v1",
        "case_id": case_id,
        "source_prompt_case_id": source_case,
        "prompt": prompt,
        "corner_cases": list(intended.values()),
        "review_required": False,
        "review": {
            "status": "approved",
            "reviewer": HUMAN_REVIEWER,
            "reviewed_at": HUMAN_REVIEWED_AT,
        },
    }
    return {
        "scene": scene,
        "prompt": prompt,
        "plan": plan,
        "assets": assets,
        "reference_annotation": reference,
        "intended": intended,
        "corner_manifest": corner_manifest,
        "asset_review_notes": _fine_asset_review_notes(source_case),
    }


def _assistant_convert_fine_plan(
    source_case: str,
    source_plan: dict[str, Any],
    *,
    case_id: str,
) -> dict[str, Any]:
    """Freeze the benchmark-authoring assistant's literal Fine prompt conversion.

    This is offline benchmark authoring, never a runtime model call.  The
    legacy camera fixtures are consulted only for object inventory; relations
    and claims below are re-authored against the current converter contract.
    """

    claims = {
        "easy_001": [
            "one grey upholstered bed against the north wall",
            "one dark brown wooden nightstand to the right of the bed",
            "one mint green vintage alarm clock on the nightstand",
            "one dark gray angular floor lamp to the left of the bed",
            "one beige geometric carpet in front of the bed",
        ],
        "easy_003": [
            "one dark angular wooden desk against the north wall",
            "one dark gray flat screen monitor on the desk",
            "one dark brown wooden chair in front of the desk",
            "one light wood five tier shelf to the left of the desk",
            "one white ceramic cup to the right of the monitor and on the desk",
        ],
        "easy_004": [
            "one beige fabric sofa against the south wall",
            "one oval walnut coffee table in front of the sofa",
            "one teal blue paperback book on the coffee table",
            "one beige linen table lamp on the coffee table",
            "one lush green philodendron to the left of the sofa",
        ],
        "easy_007": [
            "one dark gray tufted sofa against the north wall",
            "one teal storage coffee table in front of the sofa",
            "one dark gray ceramic cup on the coffee table",
            "one white veined zebra plant to the right of the sofa",
            "one beige geometric carpet near the coffee table",
        ],
        "prompt_010": [
            "an 8 m x 9 m living room with a 3 m ceiling",
            "one light-grey L-shaped sectional in the southwest corner following the south and west walls",
            "one square black coffee table inside the open side of the sectional",
            "one red bowl and one closed magazine side by side on the coffee table",
            "one round side table near the short end of the sectional",
            "one table lamp on the side table",
            "one large television mounted on the north wall opposite the long sofa section",
            "two green plants symmetrically below the television",
        ],
    }
    relations: dict[str, list[dict[str, Any]]] = {
        "easy_001": [
            {"family": "oar", "subject_id": "obj_000", "type": "against_wall", "target": "north_wall"},
            {"family": "oor", "subject_id": "obj_001", "type": "right", "object_id": "obj_000"},
            {"family": "oor", "subject_id": "obj_002", "type": "on_top_of", "object_id": "obj_001"},
            {"family": "oor", "subject_id": "obj_003", "type": "left", "object_id": "obj_000"},
            {"family": "oor", "subject_id": "obj_004", "type": "in_front", "object_id": "obj_000"},
        ],
        "easy_003": [
            {"family": "oar", "subject_id": "obj_000", "type": "against_wall", "target": "north_wall"},
            {"family": "oor", "subject_id": "obj_001", "type": "on_top_of", "object_id": "obj_000"},
            {"family": "oor", "subject_id": "obj_002", "type": "in_front", "object_id": "obj_000"},
            {"family": "oor", "subject_id": "obj_003", "type": "left", "object_id": "obj_000"},
            {"family": "oor", "subject_id": "obj_004", "type": "right", "object_id": "obj_001"},
            {"family": "oor", "subject_id": "obj_004", "type": "on_top_of", "object_id": "obj_000"},
        ],
        "easy_004": [
            {"family": "oar", "subject_id": "obj_000", "type": "against_wall", "target": "south_wall"},
            {"family": "oor", "subject_id": "obj_001", "type": "in_front", "object_id": "obj_000"},
            {"family": "oor", "subject_id": "obj_002", "type": "on_top_of", "object_id": "obj_001"},
            {"family": "oor", "subject_id": "obj_003", "type": "on_top_of", "object_id": "obj_001"},
            {"family": "oor", "subject_id": "obj_004", "type": "left", "object_id": "obj_000"},
        ],
        "easy_007": [
            {"family": "oar", "subject_id": "obj_000", "type": "against_wall", "target": "north_wall"},
            {"family": "oor", "subject_id": "obj_001", "type": "in_front", "object_id": "obj_000"},
            {"family": "oor", "subject_id": "obj_002", "type": "on_top_of", "object_id": "obj_001"},
            {"family": "oor", "subject_id": "obj_003", "type": "right", "object_id": "obj_000"},
            {"family": "oor", "subject_id": "obj_004", "type": "near", "object_id": "obj_001"},
        ],
        "prompt_010": [
            {"family": "oar", "subject_id": "obj_000", "type": "at_corner", "target": "southwest_corner"},
            {"family": "oar", "subject_id": "obj_000", "type": "along_wall", "target": "south_wall"},
            {"family": "oar", "subject_id": "obj_000", "type": "along_wall", "target": "west_wall"},
            {
                "family": "oor",
                "subject_id": "obj_001",
                "type": "inside_open_side",
                "object_id": "obj_000",
                "raw_relation": "inside the open side of the sectional",
            },
            {"family": "oor", "subject_id": "obj_002", "type": "on_top_of", "object_id": "obj_001"},
            {"family": "oor", "subject_id": "obj_003", "type": "on_top_of", "object_id": "obj_001"},
            {
                "family": "oor",
                "subject_id": "obj_002",
                "type": "side_by_side",
                "object_id": "obj_003",
                "raw_relation": "side by side",
            },
            {
                "family": "oor",
                "subject_id": "obj_004",
                "type": "near_short_end",
                "object_id": "obj_000",
                "raw_relation": "near the short end of the sectional",
            },
            {"family": "oor", "subject_id": "obj_005", "type": "on_top_of", "object_id": "obj_004"},
            {"family": "oar", "subject_id": "obj_006", "type": "mounted_on_wall", "target": "north_wall"},
            {
                "family": "oor",
                "subject_id": "obj_006",
                "type": "opposite_long_section_of",
                "object_id": "obj_000",
                "raw_relation": "opposite the long sofa section",
            },
            {
                "family": "oor",
                "subject_ids": ["obj_007"],
                "type": "symmetrically_below",
                "object_ids": ["obj_006"],
                "raw_relation": "two green plants symmetrically below the television",
            },
        ],
    }
    objects = deepcopy(source_plan["objects"])
    for obj in objects:
        obj.pop("estimated_size", None)
        obj["placement_intent"] = {"absolute_relations": [], "relative_relations": []}
        obj["metadata"] = {
            **dict(obj.get("metadata") or {}),
            "conversion_source": "literal_fine_prompt",
        }
    return {
        "request_id": case_id,
        "scene_type": str(source_plan["scene_type"]),
        "scene_description": str(source_plan["scene_description"]),
        "prompt_granularity": "fine_grained",
        "explicit_claims": claims[source_case],
        "objects": objects,
        "global_constraints": (
            ["room dimensions: width=8 m, depth=9 m, height=3 m"]
            if source_case == "prompt_010"
            else []
        ),
        "relations": relations[source_case],
        "metadata": {
            "conversion": dict(FINE_CONVERTER),
            "source_case_id": source_case,
            "literal_claims_only": True,
        },
    }
def _fine_asset_review_notes(source_case: str) -> list[str]:
    notes = {
        "easy_001": ["Frozen bed/rug colors are approximate rather than exact prompt matches."],
        "easy_003": ["Frozen chair is a tall Chinese wooden armchair; category remains compatible."],
        "easy_004": [],
        "easy_007": ["Frozen rug is black/beige rather than simply beige."],
        "prompt_010": [
            "Frozen sectional is beige rather than light grey.",
            "Frozen coffee table, bowl, and television are weak visual matches (bronze table, white bowl, vintage CRT).",
            "The two plants expand one count=2 plan object into obj_007 and obj_008 for canonical instance identity.",
        ],
    }
    return list(notes[source_case])


def _fine_room(source_case: str) -> tuple[float, float, float]:
    return (8.0, 9.0, 3.0) if source_case == "prompt_010" else ((7.0, 6.0, 3.0) if source_case == "easy_004" else (6.0, 5.0, 3.0))


def _fine_placements(source_case: str, selected: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    def size(object_id: str) -> list[float]:
        return [float(value) for value in selected[object_id]["size"]]

    if source_case == "easy_001":
        bed, nightstand, clock, _lamp, rug = (size(f"obj_{i:03d}") for i in range(5))
        bed_y = 5.0 - bed[1] / 2.0 + 0.02
        rug_y = (bed_y - bed[1] / 2.0 + 0.02) - rug[1] / 2.0
        return {
            "obj_000": {"center": [3.0, bed_y, bed[2] / 2.0]},
            "obj_001": {"center": [4.45, 4.2, nightstand[2] / 2.0]},
            "obj_002": {"center": [4.45, 4.2, nightstand[2] + 0.03 + clock[2] / 2.0]},
            "obj_003": {"center": [1.4, 4.2, size("obj_003")[2] / 2.0]},
            "obj_004": {"center": [3.0, rug_y, rug[2] / 2.0]},
        }
    if source_case == "easy_003":
        desk, monitor, chair, shelf, cup = (size(f"obj_{i:03d}") for i in range(5))
        desk_y = 5.0 - desk[1] / 2.0 + 0.02
        chair_y = desk_y - desk[1] / 2.0 + 0.02 - chair[1] / 2.0
        return {
            "obj_000": {"center": [3.0, desk_y, desk[2] / 2.0]},
            "obj_001": {"center": [2.75, desk_y, desk[2] + 0.03 + monitor[2] / 2.0]},
            "obj_002": {"center": [3.0, chair_y, chair[2] / 2.0]},
            "obj_003": {"center": [1.25, 4.25, shelf[2] / 2.0]},
            "obj_004": {"center": [3.55, desk_y, desk[2] + cup[2] / 2.0]},
        }
    if source_case == "easy_004":
        sofa, table, book, lamp, plant = (size(f"obj_{i:03d}") for i in range(5))
        sofa_y = sofa[1] / 2.0 - 0.02
        table_y = sofa_y + sofa[1] / 2.0 + table[1] / 2.0 - 0.02
        return {
            "obj_000": {"center": [3.5, sofa_y, sofa[2] / 2.0]},
            "obj_001": {"center": [3.5, table_y, table[2] / 2.0]},
            "obj_002": {"center": [3.1, table_y, table[2] + 0.03 + book[2] / 2.0]},
            "obj_003": {"center": [4.1, table_y, table[2] + lamp[2] / 2.0]},
            "obj_004": {"center": [1.4, 0.8, plant[2] / 2.0]},
        }
    if source_case == "easy_007":
        sofa, table, cup, plant, rug = (size(f"obj_{i:03d}") for i in range(5))
        sofa_y = 5.0 - sofa[1] / 2.0 + 0.02
        return {
            "obj_000": {"center": [3.0, sofa_y, sofa[2] / 2.0]},
            "obj_001": {"center": [3.0, 3.3, table[2] / 2.0]},
            "obj_002": {"center": [3.0, 3.3, table[2] + 0.03 + cup[2] / 2.0]},
            "obj_003": {"center": [4.6, 4.35, plant[2] / 2.0]},
            "obj_004": {"center": [3.0, 3.3, rug[2] / 2.0]},
        }
    if source_case == "prompt_010":
        sofa, table, bowl, magazine, side_table, lamp, tv, plant = (size(f"obj_{i:03d}") for i in range(8))
        return {
            "obj_000": {"center": [sofa[0] / 2.0 - 0.02, sofa[1] / 2.0 - 0.02, sofa[2] / 2.0]},
            "obj_001": {"center": [4.25, 3.35, table[2] / 2.0]},
            "obj_002": {"center": [4.08, 3.35, table[2] + bowl[2] / 2.0]},
            "obj_003": {"center": [4.43, 3.35, table[2] + magazine[2] / 2.0]},
            "obj_004": {"center": [1.0, 3.65, side_table[2] / 2.0]},
            "obj_005": {"center": [1.0, 3.65, side_table[2] + 0.03 + lamp[2] / 2.0]},
            "obj_006": {"center": [4.0, 9.0 - tv[1] / 2.0 - 0.01, 1.6]},
            "obj_007": {"center": [3.2, 8.5, plant[2] / 2.0], "plan_id": "obj_007"},
            "obj_008": {"center": [4.8, 8.5, plant[2] / 2.0], "plan_id": "obj_007"},
        }
    raise KeyError(source_case)


def _fine_intended(source_case: str) -> dict[str, dict[str, Any]]:
    specs = {
        "easy_001": [
            ("oob", "obj_000", ["obj_000"], "bed extends 2 cm across north wall"),
            ("support", "obj_002", ["obj_002"], "clock has a 3 cm positive gap above nightstand"),
            ("collision", "obj_000|obj_004", ["obj_000", "obj_004"], "rug has a shallow overlap under bed"),
        ],
        "easy_003": [
            ("oob", "obj_000", ["obj_000"], "desk extends 2 cm across north wall"),
            ("support", "obj_001", ["obj_001"], "monitor has a 3 cm positive gap above desk"),
            ("collision", "obj_000|obj_002", ["obj_000", "obj_002"], "chair is tucked 2 cm into desk envelope"),
        ],
        "easy_004": [
            ("oob", "obj_000", ["obj_000"], "sofa extends 2 cm across south wall"),
            ("support", "obj_002", ["obj_002"], "book has a 3 cm positive gap above table"),
            ("collision", "obj_000|obj_001", ["obj_000", "obj_001"], "coffee table has a 2 cm edge overlap with sofa"),
        ],
        "easy_007": [
            ("oob", "obj_000", ["obj_000"], "sofa extends 2 cm across north wall"),
            ("support", "obj_002", ["obj_002"], "cup has a 3 cm positive gap above table"),
            ("collision", "obj_001|obj_004", ["obj_001", "obj_004"], "table overlaps a thin rug"),
        ],
        "prompt_010": [
            ("oob", "obj_000", ["obj_000"], "sectional extends 2 cm across southwest walls"),
            ("support", "obj_005", ["obj_005"], "table lamp has a 3 cm positive gap"),
            ("support", "obj_006", ["obj_006"], "elevated television is 1 cm from north wall as a valid attachment candidate"),
        ],
    }
    result = {}
    for metric, event_id, object_ids, note in specs[source_case]:
        result[f"{metric}:{event_id}"] = {
            "metric": metric,
            "event_id": event_id,
            "object_ids": object_ids,
            "semantic_label": "ambiguous",
            "route_requirement": "must_route",
            "gt_basis": "human_reviewed_edge_case",
            "severity_class": "edge",
            "review_question": note,
        }
    return result


def _constructed_scene(spec: dict[str, Any], catalog: dict[str, dict[str, str]]) -> dict[str, Any]:
    objects = []
    for object_id, jid, category, override_size, raw_center in spec["objects"]:
        catalog_size = _catalog_size(catalog[jid])
        size = [float(value) for value in (override_size or catalog_size)]
        center = [
            float(raw_center[0]),
            float(raw_center[1]),
            size[2] / 2.0 if raw_center[2] is None else float(raw_center[2]),
        ]
        objects.append(
            _object(
                object_id=object_id,
                jid=jid,
                category=category,
                description=str(catalog[jid].get("short_desc") or category),
                size=size,
                center=center,
                rotation=[0.0, 0.0, 0.0],
                metadata={
                    "interactive": False,
                    "fixture_role": str(spec["split"]),
                    "native_asset_bbox_size": catalog_size,
                    "scale_override": override_size is not None,
                },
            )
        )
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": spec["case_id"],
        "request_id": spec["case_id"],
        "scene_type": spec["scene_type"],
        "boundary": [[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]],
        "scene_height": 4.0,
        "objects": objects,
        "metadata": {
            "coordinate_frame": dict(COORDINATE_FRAME),
            "generator_skipped": True,
            "construction": "manual_asset_grounded_calibration_scene",
        },
    }


def _event_gt(
    scene: dict[str, Any], *, status: str, intended: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    objects = scene["objects"]
    for left, right in combinations(objects, 2):
        object_ids = sorted([str(left["id"]), str(right["id"])])
        event_id = "|".join(object_ids)
        key = f"collision:{event_id}"
        events.append(
            deepcopy(
                intended.get(
                    key,
                    {
                        "metric": "collision",
                        "event_id": event_id,
                        "object_ids": object_ids,
                        "semantic_label": "valid",
                        "route_requirement": "route_allowed",
                        "gt_basis": "source_validity_or_controlled_construction",
                    },
                )
            )
        )
    for metric in ("oob", "support"):
        for obj in objects:
            object_id = str(obj["id"])
            key = f"{metric}:{object_id}"
            events.append(
                deepcopy(
                    intended.get(
                        key,
                        {
                            "metric": metric,
                            "event_id": object_id,
                            "object_ids": [object_id],
                            "semantic_label": "valid",
                            "route_requirement": "route_allowed",
                            "gt_basis": "source_validity_or_controlled_construction",
                        },
                    )
                )
            )
    return {
        "schema_version": "deterministic_event_gt_v1",
        "case_id": scene["scene_id"],
        "status": status,
        "events": events,
        "scene_metrics": {
            "navigability": {"semantic_label": "diagnostic_only", "gt_basis": "no_human_connectivity_oracle"},
            "accessibility": {"semantic_label": "not_applicable", "gt_basis": "no_explicit_interactive_targets"},
        },
    }


def _scene_request(scene: dict[str, Any], instruction: str, *, prompt_granularity: str) -> dict[str, Any]:
    min_x, max_x, min_y, max_y = _room_bounds(scene)
    dimensions = {"width": max_x - min_x, "depth": max_y - min_y, "height": float(scene["scene_height"])}
    return {
        "request_id": scene["request_id"],
        "instruction": instruction,
        "scene_type": scene["scene_type"],
        "structure": False,
        "prompt_granularity": prompt_granularity,
        "metadata": {"generator_skipped": True, "reference_annotation_visibility": "benchmark_private"},
        "room": {
            "boundary": deepcopy(scene["boundary"]),
            "height": float(scene["scene_height"]),
            "unit": "meter",
            "dimensions": dimensions,
            "explicit_dimensions": dimensions,
            "dimension_provenance": {axis: "explicit_calibration_fixture" for axis in dimensions},
            "resolution_policy": "room_dimension_policy_v1",
            "topology": "rectangular_enclosed_room",
            "floor_z": 0.0,
        },
    }


def _object_plan_from_scene(
    scene: dict[str, Any], instruction: str, *, prompt_granularity: str
) -> dict[str, Any]:
    return {
        "request_id": scene["request_id"],
        "scene_type": scene["scene_type"],
        "scene_description": instruction,
        "prompt_granularity": prompt_granularity,
        "explicit_claims": [],
        "objects": [
            {
                "id": obj["id"],
                "role": "",
                "category": obj["category"],
                "description": obj.get("description", obj["category"]),
                "estimated_size": deepcopy(obj["size"]),
                "count": 1,
                "placement_intent": {"absolute_relations": [], "relative_relations": []},
                "metadata": {"description_source": "frozen_scene_or_asset_catalog"},
            }
            for obj in scene["objects"]
        ],
        "global_constraints": [],
        "relations": [],
        "metadata": {"fixture_policy": "cal_dataset1_frozen_construction"},
    }


def _asset_selection_from_scene(
    scene: dict[str, Any],
    catalog: dict[str, dict[str, str]],
    *,
    policy: str,
    source: str,
) -> dict[str, Any]:
    items = []
    for obj in scene["objects"]:
        jid = str(obj["jid"])
        row = catalog[jid]
        selected = {
            "jid": jid,
            "category": obj["category"],
            "retrieval_category": obj["category"],
            "desc": str(row.get("caption_en") or obj.get("description") or ""),
            "short_desc": str(row.get("short_desc") or obj.get("description") or ""),
            "size": deepcopy(obj["size"]),
            "asset_ref": deepcopy(obj["asset_ref"]),
            "asset_proxy": deepcopy(obj["asset_proxy"]),
            "metadata": {"interactive": False, "selection_policy": policy, "selection_source": source},
        }
        items.append(
            {
                "object_id": obj["id"],
                "object_spec": {
                    "role": "",
                    "category": obj["category"],
                    "description": obj.get("description", obj["category"]),
                    "estimated_size": deepcopy(obj["size"]),
                    "count": 1,
                },
                "retrieval_query": {"description": obj.get("description", ""), "category": obj["category"]},
                "selected_asset": selected,
                "candidates": [{**deepcopy(selected), "score": None}],
                "selection_action": "select",
                "selection_decision": {
                    "action": "select",
                    "selected_jid": jid,
                    "reason": policy,
                    "generation_request": None,
                },
                "selection_reason": policy,
            }
        )
    return {"request_id": scene["request_id"], "objects": items}


def _object(
    *,
    object_id: str,
    jid: str,
    category: str,
    description: str,
    size: list[float],
    center: list[float],
    rotation: list[float],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": object_id,
        "category": category,
        "description": description,
        "desc": description,
        "short_desc": description,
        "jid": jid,
        "size": [float(value) for value in size],
        "center": [float(value) for value in center],
        "rotation": [float(value) for value in rotation],
        "geometry_provenance": "asset_mesh",
        "asset_ref": {
            "source_db": "imaginarium",
            "asset_key": jid,
            "mesh_uri": None,
            "pointcloud_uri": None,
            "metadata_uri": None,
        },
        "asset_proxy": {"type": "canonical_obb", "bbox_center_local": [0.0, 0.0, 0.0], "bbox_size": [float(value) for value in size]},
        "metadata": metadata,
    }


def _write_case(
    out_root: Path,
    case_id: str,
    *,
    scene: dict[str, Any],
    request: dict[str, Any],
    plan: dict[str, Any],
    assets: dict[str, Any],
    provenance: dict[str, Any],
    event_gt: dict[str, Any],
    review: dict[str, Any],
    perturbation: dict[str, Any] | None = None,
    reference_annotation: dict[str, Any] | None = None,
    corner_manifest: dict[str, Any] | None = None,
) -> None:
    validate_generated_scene(scene)
    validate_scene_request(request)
    validate_object_plan(plan)
    validate_asset_selection(assets)
    if reference_annotation is not None:
        validate_reference_annotation(reference_annotation)
    fixture = out_root / "fixtures" / case_id
    _write_json(fixture / "generated_scene.json", scene)
    _write_json(fixture / "scene_request.json", request)
    _write_json(fixture / "object_plan.json", plan)
    _write_json(fixture / "asset_selection.json", assets)
    _write_json(fixture / "provenance.json", provenance)
    _write_json(fixture / "event_gt.json", event_gt)
    _write_json(fixture / "review.json", review)
    if perturbation is not None:
        _write_json(fixture / "perturbation_manifest.json", perturbation)
    if reference_annotation is not None:
        _write_json(fixture / "reference_annotation.json", reference_annotation)
    if corner_manifest is not None:
        _write_json(fixture / "corner_case_manifest.json", corner_manifest)


def _case_record(
    case_id: str,
    *,
    split: str,
    prompt_granularity: str,
    evaluation_scope: str,
    scene: dict[str, Any],
    review_status: str,
    base_case_id: str | None,
    source_scene_id: str | None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "split": split,
        "base_case_id": base_case_id,
        "source_scene_id": source_scene_id,
        "prompt_granularity": prompt_granularity,
        "evaluation_scope": evaluation_scope,
        "fixture_dir": f"fixtures/{case_id}",
        "object_count": len(scene["objects"]),
        "review_status": review_status,
    }


def _place_shallow_overlap(
    scene: dict[str, Any], target: dict[str, Any], anchor: dict[str, Any], *, overlap_m: float
) -> None:
    target_half = _rotated_half_extents_xy(target)
    anchor_half = _rotated_half_extents_xy(anchor)
    min_x, max_x, min_y, max_y = _room_bounds(scene)
    candidates: list[tuple[float, list[float]]] = []
    for axis in (0, 1):
        for sign in (-1.0, 1.0):
            center = [float(anchor["center"][0]), float(anchor["center"][1]), float(target["size"][2]) / 2.0]
            center[axis] = float(anchor["center"][axis]) + sign * (anchor_half[axis] + target_half[axis] - overlap_m)
            margin = min(
                center[0] - target_half[0] - min_x,
                max_x - center[0] - target_half[0],
                center[1] - target_half[1] - min_y,
                max_y - center[1] - target_half[1],
            )
            candidates.append((margin, center))
    target["center"] = max(candidates, key=lambda item: item[0])[1]


def _move_oob(
    scene: dict[str, Any],
    obj: dict[str, Any],
    *,
    plane: str,
    outside_m: float | None,
    penetration_fraction: float,
) -> None:
    min_x, max_x, min_y, max_y = _room_bounds(scene)
    half_x, half_y = _rotated_half_extents_xy(obj)
    if outside_m is None:
        inside_fraction = 1.0 - 2.0 * penetration_fraction
        centers = {
            "west": min_x + inside_fraction * half_x,
            "east": max_x - inside_fraction * half_x,
            "south": min_y + inside_fraction * half_y,
            "north": max_y - inside_fraction * half_y,
        }
    else:
        centers = {
            "west": min_x + half_x - outside_m,
            "east": max_x - half_x + outside_m,
            "south": min_y + half_y - outside_m,
            "north": max_y - half_y + outside_m,
        }
    if plane in {"west", "east"}:
        obj["center"][0] = float(centers[plane])
    elif plane in {"south", "north"}:
        obj["center"][1] = float(centers[plane])
    else:
        raise ValueError(f"unknown room plane: {plane}")
    obj["center"][2] = float(obj["size"][2]) / 2.0


def _rotated_half_extents_xy(obj: dict[str, Any]) -> tuple[float, float]:
    half_x = float(obj["size"][0]) / 2.0
    half_y = float(obj["size"][1]) / 2.0
    yaw = math.radians(float(obj["rotation"][2]))
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    return c * half_x + s * half_y, s * half_x + c * half_y


def _find(scene: dict[str, Any], object_id: str) -> dict[str, Any]:
    return next(obj for obj in scene["objects"] if obj["id"] == object_id)


def _room_bounds(scene: dict[str, Any]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for point in scene["boundary"]]
    ys = [float(point[1]) for point in scene["boundary"]]
    return min(xs), max(xs), min(ys), max(ys)


def _read_catalog(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {str(row.get("name_en") or ""): row for row in csv.DictReader(handle)}
    if not rows:
        raise ValueError(f"empty asset catalog: {path}")
    return rows


def _catalog_size(row: dict[str, str]) -> list[float]:
    values = [float(value.strip()) for value in str(row.get("bbx") or "").split(",")]
    if len(values) != 3 or any(value <= 0.0 for value in values):
        raise ValueError(f"invalid catalog bbx for {row.get('name_en')}: {row.get('bbx')}")
    return values


def _write_review_queue(out_root: Path, cases: list[dict[str, Any]]) -> None:
    rows = ["case_id\tsplit\tprompt_granularity\treview_status\tfixture_dir"]
    for case in cases:
        if case["review_status"] == "pending":
            rows.append(
                "\t".join(
                    [
                        str(case["case_id"]),
                        str(case["split"]),
                        str(case["prompt_granularity"]),
                        str(case["review_status"]),
                        str(case["fixture_dir"]),
                    ]
                )
            )
    path = out_root / "validation/review_queue.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _counts(cases: list[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for case in cases:
        value = str(case[key])
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readme(cases: list[dict[str, Any]], fine_prompts: list[dict[str, Any]]) -> str:
    prompts = "\n\n".join(
        f"### {item['case_id']}\n\n{item['prompt']}\n\nCorner cases: "
        + "; ".join(str(edge["review_question"]) for edge in item["corner_cases"])
        for item in fine_prompts
    )
    return f"""# cal_dataset1

`cal_dataset1` is a model-free calibration set for deterministic routing and Spatial Fidelity priors.

- Total: {len(cases)} cases.
- Deterministic full: 31 cases; Collision, OOB, Support, Navigability and Accessibility only.
- Scale: 2 isolated cases.
- Co-occurrence Plausibility: 2 isolated cases.
- Functional Grouping remains out of scope.
- Coarse-grained is the default prompt mode. Only the five `fine_edge` cases are Fine-grained.
- Human audit is approved. Subtle distortions and Co-occurrence controls have confirmed invalid labels.
- Fine edge events are approved as deliberate edge cases but remain `ambiguous`, so they test routing/VLM behavior without inventing binary GT.
- Fine prompts were converted offline by the benchmark-authoring assistant; their confirmed reference annotations are official-scoreable.
- Detector-only reports must keep candidate scores null; `requires_vlm` is a route, not an invalid verdict.

The requested `Scenes/ConcertedScenes` path is absent. Source fixtures therefore use `Scenes/converted_scenes`, the same source used by the repository's existing distortion-fixture tooling. Selection is independent of detector output to avoid circular false-positive calibration.

## Fine-grained prompts

{prompts}

## Reproduce

```bash
.venv/bin/python scripts/build_cal_dataset1.py --ontology cal_dataset1/ontology/SceneOnto.json
.venv/bin/python scripts/run_cal_dataset1.py
.venv/bin/python cal_dataset1/validate_dataset.py
.venv/bin/python Support/legacy/scripts/run_cal_dataset1_experiment.py --track all --plan-only
```

## Model-backed experiment

The reviewed bundle can run directly from frozen `generated_scene.json` files. Generation,
runtime conversion, retrieval, and asset binding stay disabled. The default experiment track
is the 31-case deterministic full test; Scale/Co-occurrence and Fine Prompt Fidelity are
separate opt-in tracks.

```bash
# Shared deterministic calibration only (Scale and Co-occurrence excluded)
.venv/bin/python Support/legacy/scripts/run_cal_dataset1_experiment.py \
  --judge-config configs/models/qwen3vl_mnet_judge.json

# All explicit tracks: 31 deterministic + 4 isolated spatial + 5 Fine fidelity
.venv/bin/python Support/legacy/scripts/run_cal_dataset1_experiment.py \
  --track all \
  --judge-config configs/models/qwen3vl_mnet_judge.json
```

The runner verifies the exact served model ID through `/v1/models`, is resumable by default,
continues after individual case failures, and records complete per-case reports plus a summary
under `outputs/cal_dataset1_experiment/`.

## Review packets

- `review/source_valid_contact_sheet.png`: approved valid-source visual check.
- `review/subtle_review_contact_sheet.png`: eight approved subtle controls.
- `review/fine_edge_contact_sheet.png`: five approved Fine-grained edge cases.
- `review/fine_prompts.json`: verbatim Fine-grained prompts and the three intended edge events per case.
- `review/human_audit_approval.json`: frozen human-audit decision.
- `validation/review_queue.tsv`: header-only when no review remains pending.
"""


if __name__ == "__main__":
    main()
