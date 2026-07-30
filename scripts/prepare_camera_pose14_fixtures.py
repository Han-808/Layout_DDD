"""Build the fixed ``camera_pose14`` test/experiment fixtures.

Summary:
    Deterministically materializes the object plans, reference annotations, and
    a ``cases.json`` manifest for the camera-pose-14 fixture set.

Input:
    - None (no CLI args; uses in-repo source cases and constants).

Output:
    - Object-plan and reference-annotation JSON files plus ``cases.json`` under
      a fixed output root; prints prepared/moderate/easy case counts.

Function:
    One-shot fixture generator used to keep camera-pose experiments and tests
    reproducible.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.reference_annotation import (  # noqa: E402
    build_reference_annotation_draft,
    confirm_reference_annotation,
    validate_reference_annotation,
)
from benchmark.scene_io.validate import validate_object_plan, validate_scene_request  # noqa: E402
from benchmark.utils.io import write_json  # noqa: E402


SOURCE_PROMPTS = PROJECT_ROOT / "configs" / "experiments" / "qwen32b_nl50_prompts.json"
MODERATE_ROOT = PROJECT_ROOT / "Support" / "artifacts" / "result" / "p0b_first7_instances"
OUTPUT_ROOT = PROJECT_ROOT / "configs" / "experiments" / "camera_pose14"
MODERATE_IDS = (
    "prompt_001",
    "prompt_005",
    "prompt_010",
    "prompt_015",
    "prompt_018",
    "prompt_022",
    "prompt_027",
)


def main() -> None:
    source_payload = json.loads(SOURCE_PROMPTS.read_text(encoding="utf-8"))
    source_by_id = {item["case_id"]: item for item in source_payload["cases"]}
    cases: list[dict[str, Any]] = []
    plans: dict[str, dict[str, Any]] = {}
    dropped_relations: dict[str, list[dict[str, Any]]] = {}

    for case_id in MODERATE_IDS:
        case = deepcopy(source_by_id[case_id])
        case["difficulty"] = "moderate"
        case["fixture_source"] = "p0b_first7_instances"
        plan_path = MODERATE_ROOT / case_id / "02_converter_output.object_plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan, dropped = _sanitize_moderate_plan(plan, instruction=case["instruction"])
        validate_object_plan(plan)
        cases.append(case)
        plans[case_id] = plan
        dropped_relations[case_id] = dropped

    for case, plan in _easy_cases():
        validate_object_plan(plan)
        cases.append(case)
        plans[case["case_id"]] = plan
        dropped_relations[case["case_id"]] = []

    structure_dir = OUTPUT_ROOT / "generator_structures"
    reference_dir = OUTPUT_ROOT / "reference_annotations"
    structure_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        case_id = case["case_id"]
        plan = plans[case_id]
        request = _scene_request(case)
        validate_scene_request(request)
        annotation = _reference_annotation(
            plan,
            request,
            confirm_room=case["difficulty"] == "moderate",
            dropped_relations=dropped_relations[case_id],
        )
        validate_reference_annotation(annotation)
        write_json(structure_dir / f"{case_id}.json", plan)
        write_json(reference_dir / f"{case_id}.json", annotation)

    payload = {
        "experiment_id": "qwen32b_camera_pose14_twomode",
        "input_mode": "natural_language_plus_public_generator_structure",
        "prompt_granularity": "fine_grained",
        "asset_mode": "retrieve",
        "camera_pose_modes": ["bbox_track", "query_cov"],
        "notes": [
            "Seven moderate cases are frozen from the audited P0b first-seven set.",
            "Seven easy cases contain four to six objects and use exact Imaginarium short_desc phrases.",
            "Invalid legacy one-anchor between relations are omitted from the public fixture; prompts remain unchanged.",
            "References are calibration fixtures and must not be reported as leaderboard ground truth.",
        ],
        "cases": cases,
    }
    write_json(OUTPUT_ROOT / "cases.json", payload)
    print(f"prepared_cases: {len(cases)}")
    print(f"moderate_cases: {len(MODERATE_IDS)}")
    print(f"easy_cases: {len(cases) - len(MODERATE_IDS)}")
    print(f"output_root: {OUTPUT_ROOT}")
    for case_id, dropped in dropped_relations.items():
        if dropped:
            print(f"sanitized_relations[{case_id}]: {len(dropped)}")


def _sanitize_moderate_plan(
    plan: dict[str, Any],
    *,
    instruction: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sanitized = deepcopy(plan)
    sanitized["scene_description"] = instruction
    object_ids = {
        str(item.get("id"))
        for item in sanitized.get("objects", [])
        if isinstance(item, dict) and item.get("id")
    }
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for relation in sanitized.get("relations", []):
        if not isinstance(relation, dict):
            dropped.append({"reason": "non_object_relation", "relation": relation})
            continue
        relation_type = str(relation.get("type") or "").strip().lower()
        family = str(relation.get("family") or "").strip().lower()
        if relation_type == "between" and len(relation.get("object_ids") or []) != 2:
            dropped.append({"reason": "legacy_between_missing_two_anchor_ids", "relation": relation})
            continue
        if relation_type == "ordered" and len(relation.get("object_ids") or []) < 2:
            dropped.append({"reason": "legacy_ordered_missing_member_ids", "relation": relation})
            continue
        if relation_type == "around" and len(relation.get("subject_ids") or []) < 1:
            dropped.append({"reason": "legacy_around_missing_member_ids", "relation": relation})
            continue
        target = str(
            relation.get("target")
            or relation.get("architectural_element")
            or relation.get("object_id")
            or ""
        )
        if family == "oar" and target in object_ids:
            dropped.append({"reason": "object_target_mislabeled_as_architecture", "relation": relation})
            continue
        kept.append(relation)
    sanitized["relations"] = kept
    sanitized.setdefault("metadata", {})
    sanitized["metadata"] = {
        **sanitized["metadata"],
        "fixture_policy": "legacy_plan_schema_sanitization_only",
        "dropped_relation_count": len(dropped),
    }
    return sanitized, dropped


def _reference_annotation(
    plan: dict[str, Any],
    request: dict[str, Any],
    *,
    confirm_room: bool,
    dropped_relations: list[dict[str, Any]],
) -> dict[str, Any]:
    draft = build_reference_annotation_draft(plan, request, source="programmatic")
    if confirm_room:
        room = request["room"]
        dimensions = room.get("dimensions") or {
            "width": max(float(point[0]) for point in room["boundary"]),
            "depth": max(float(point[1]) for point in room["boundary"]),
            "height": float(room["height"]),
        }
        draft["room_constraints"] = {
            "claim_state": "confirmed",
            "dimensions": {
                "width": float(dimensions["width"]),
                "depth": float(dimensions["depth"]),
                "height": float(dimensions["height"]),
            },
            "shape": "rectangular_enclosed_room",
        }
    draft["provenance"] = {
        "origin": "camera_pose14_calibration_fixture",
        "generator_visible": False,
        "leaderboard_ground_truth": False,
        "dropped_legacy_relations": dropped_relations,
    }
    return confirm_reference_annotation(draft, inventory_policy="closed_world")


def _scene_request(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": case["case_id"],
        "instruction": case["instruction"],
        "scene_type": case["scene_type"],
        "room": case["room"],
        "prompt_granularity": "fine_grained",
    }


def _easy_cases() -> list[tuple[dict[str, Any], dict[str, Any]]]:
    definitions = [
        {
            "case_id": "easy_001",
            "scene_type": "bedroom",
            "instruction": (
                "Create a bedroom with a grey upholstered bed against the north wall. "
                "Place a dark brown wooden nightstand to the right of the bed and put a mint green vintage alarm clock on the nightstand. "
                "Put a dark gray angular floor lamp to the left of the bed and a beige geometric carpet in front of the bed."
            ),
            "objects": [
                ("bed", "grey upholstered bed"),
                ("nightstand", "dark brown wooden nightstand"),
                ("clock", "mint green vintage alarm clock"),
                ("lamp", "dark gray angular floor lamp"),
                ("rug", "beige geometric carpet"),
            ],
            "relations": [
                _oar("obj_000", "against_wall", "north_wall"),
                _oor("obj_001", "right_of", "obj_000"),
                _oor("obj_002", "on_top_of", "obj_001"),
                _oor("obj_003", "left_of", "obj_000"),
                _oor("obj_004", "in_front_of", "obj_000"),
            ],
        },
        {
            "case_id": "easy_002",
            "scene_type": "living_room",
            "instruction": (
                "Create a living room with a brown leather chesterfield sofa against the west wall. "
                "Place a dark wood coffee table in front of the sofa and a white ceramic vase on the table. "
                "Put a black leather lounge chair to the right of the table and a deep red floral carpet near the table."
            ),
            "objects": [
                ("sofa", "brown leather chesterfield sofa"),
                ("table", "dark wood coffee table"),
                ("vase", "white ceramic vase"),
                ("chair", "black leather lounge chair"),
                ("rug", "deep red floral carpet"),
            ],
            "relations": [
                _oar("obj_000", "against_wall", "west_wall"),
                _oor("obj_001", "in_front_of", "obj_000"),
                _oor("obj_002", "on_top_of", "obj_001"),
                _oor("obj_003", "right_of", "obj_001"),
                _oor("obj_004", "near", "obj_001"),
            ],
        },
        {
            "case_id": "easy_003",
            "scene_type": "home_office",
            "instruction": (
                "Create a home office with a dark angular wooden desk against the north wall. "
                "Put a dark gray flat screen monitor on the desk and a dark brown wooden chair in front of the desk. "
                "Place a light wood five tier shelf to the left of the desk and a white ceramic cup to the right of the monitor on the desk."
            ),
            "objects": [
                ("desk", "dark angular wooden desk"),
                ("monitor", "dark gray flat screen monitor"),
                ("chair", "dark brown wooden chair"),
                ("shelf", "light wood five tier shelf"),
                ("cup", "white ceramic cup"),
            ],
            "relations": [
                _oar("obj_000", "against_wall", "north_wall"),
                _oor("obj_001", "on_top_of", "obj_000"),
                _oor("obj_002", "in_front_of", "obj_000"),
                _oor("obj_003", "left_of", "obj_000"),
                _oor("obj_004", "right_of", "obj_001"),
                _oor("obj_004", "on_top_of", "obj_000"),
            ],
        },
        {
            "case_id": "easy_004",
            "scene_type": "reading_lounge",
            "instruction": (
                "Create a reading lounge with a beige fabric sofa against the south wall. "
                "Place an oval walnut coffee table in front of the sofa. Put a teal blue paperback book and a beige linen table lamp on the table. "
                "Place a lush green philodendron to the left of the sofa."
            ),
            "objects": [
                ("sofa", "beige fabric sofa"),
                ("table", "oval walnut coffee table"),
                ("book", "teal blue paperback book"),
                ("lamp", "beige linen table lamp"),
                ("plant", "lush green philodendron"),
            ],
            "relations": [
                _oar("obj_000", "against_wall", "south_wall"),
                _oor("obj_001", "in_front_of", "obj_000"),
                _oor("obj_002", "on_top_of", "obj_001"),
                _oor("obj_003", "on_top_of", "obj_001"),
                _oor("obj_004", "left_of", "obj_000"),
            ],
        },
        {
            "case_id": "easy_005",
            "scene_type": "small_lounge",
            "instruction": (
                "Create a small lounge with a white marble coffee table in the center of the room. "
                "Place one dark brown wooden chair to the left of the table and another dark brown wooden chair to its right. "
                "Put a white ceramic vase on the table and a burgundy geometric rug near the table."
            ),
            "objects": [
                ("table", "white marble coffee table"),
                ("chair", "dark brown wooden chair"),
                ("chair", "dark brown wooden chair"),
                ("vase", "white ceramic vase"),
                ("rug", "burgundy geometric rug"),
            ],
            "relations": [
                _oar("obj_000", "room_center", "room_center"),
                _oor("obj_001", "left_of", "obj_000"),
                _oor("obj_002", "right_of", "obj_000"),
                _oor("obj_003", "on_top_of", "obj_000"),
                _oor("obj_004", "near", "obj_000"),
            ],
        },
        {
            "case_id": "easy_006",
            "scene_type": "bedroom",
            "instruction": (
                "Create a bedroom with a grey upholstered bed against the west wall. "
                "Place a light wood nightstand to the right of the bed and a gray cylindrical alarm clock on the nightstand. "
                "Put a white cylindrical floor lamp near the bed. Place a light wood shelf against the east wall and a white magazine on the shelf."
            ),
            "objects": [
                ("bed", "grey upholstered bed"),
                ("nightstand", "light wood nightstand"),
                ("clock", "gray cylindrical alarm clock"),
                ("lamp", "white cylindrical floor lamp"),
                ("shelf", "light wood shelf"),
                ("magazine", "white magazine"),
            ],
            "relations": [
                _oar("obj_000", "against_wall", "west_wall"),
                _oor("obj_001", "right_of", "obj_000"),
                _oor("obj_002", "on_top_of", "obj_001"),
                _oor("obj_003", "near", "obj_000"),
                _oar("obj_004", "against_wall", "east_wall"),
                _oor("obj_005", "on_top_of", "obj_004"),
            ],
        },
        {
            "case_id": "easy_007",
            "scene_type": "living_room",
            "instruction": (
                "Create a living room with a dark gray tufted sofa against the north wall. "
                "Place a teal storage coffee table in front of the sofa and a dark gray ceramic cup on the table. "
                "Put a white veined zebra plant to the right of the sofa and a beige geometric carpet near the table."
            ),
            "objects": [
                ("sofa", "dark gray tufted sofa"),
                ("table", "teal storage coffee table"),
                ("cup", "dark gray ceramic cup"),
                ("plant", "white veined zebra plant"),
                ("rug", "beige geometric carpet"),
            ],
            "relations": [
                _oar("obj_000", "against_wall", "north_wall"),
                _oor("obj_001", "in_front_of", "obj_000"),
                _oor("obj_002", "on_top_of", "obj_001"),
                _oor("obj_003", "right_of", "obj_000"),
                _oor("obj_004", "near", "obj_001"),
            ],
        },
    ]
    result: list[tuple[dict[str, Any], dict[str, Any]]] = []
    room = {
        "boundary": [[0.0, 0.0], [7.0, 0.0], [7.0, 5.0], [0.0, 5.0]],
        "height": 3.0,
        "unit": "meter",
    }
    for definition in definitions:
        case_id = str(definition["case_id"])
        case = {
            "case_id": case_id,
            "scene_type": definition["scene_type"],
            "room": deepcopy(room),
            "instruction": definition["instruction"],
            "difficulty": "easy",
            "fixture_source": "imaginarium_short_desc_manual_fixture",
        }
        plan = {
            "request_id": case_id,
            "scene_type": definition["scene_type"],
            "scene_description": definition["instruction"],
            "prompt_granularity": "fine_grained",
            "explicit_claims": [],
            "objects": [
                {
                    "id": f"obj_{index:03d}",
                    "role": "",
                    "category": category,
                    "description": description,
                    "count": 1,
                    "placement_intent": {"absolute_relations": [], "relative_relations": []},
                    "metadata": {"description_source": "imaginarium_asset_info.short_desc"},
                }
                for index, (category, description) in enumerate(definition["objects"])
            ],
            "global_constraints": [],
            "relations": definition["relations"],
            "metadata": {
                "fixture_policy": "manual_low_ambiguity_public_structure",
                "room_dimensions_are_benchmark_fallback": True,
            },
        }
        result.append((case, plan))
    return result


def _oor(subject_id: str, relation_type: str, object_id: str) -> dict[str, str]:
    return {
        "family": "oor",
        "subject_id": subject_id,
        "type": relation_type,
        "object_id": object_id,
    }


def _oar(subject_id: str, relation_type: str, target: str) -> dict[str, str]:
    return {
        "family": "oar",
        "subject_id": subject_id,
        "type": relation_type,
        "target": target,
    }


if __name__ == "__main__":
    main()
