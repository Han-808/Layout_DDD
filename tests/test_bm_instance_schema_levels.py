from __future__ import annotations

from pathlib import Path

from jsonschema import Draft202012Validator

from benchmark.utils.io import read_json


ROOT = Path(__file__).resolve().parents[1]


def _validator() -> Draft202012Validator:
    return Draft202012Validator(read_json(ROOT / "schemas" / "bm_instance.schema.json"))


def _assert_valid(case: dict) -> None:
    errors = list(_validator().iter_errors(case))
    assert errors == []


def test_prompt_only_case_validates() -> None:
    _assert_valid(
        {
            "case_id": "prompt_001",
            "schema_version": "2.0",
            "input_level": "prompt_only",
            "description": {"text": "Create a compact reading room."},
        }
    )


def test_structured_basic_case_validates() -> None:
    _assert_valid(
        {
            "case_id": "basic_001",
            "schema_version": "2.0",
            "input_level": "structured_basic",
            "description": {"text": "Create a bedroom."},
            "room": {"boundary": [[0, 0], [4, 0], [4, 3], [0, 3]], "unit": "meter"},
            "objects": [{"id": "bed_001", "category": "bed"}],
        }
    )


def test_structured_relation_case_validates() -> None:
    _assert_valid(
        {
            "case_id": "relation_001",
            "schema_version": "2.0",
            "input_level": "structured_relation",
            "description": {"text": "Create a study room."},
            "room": {"boundary": [[0, 0], [4, 0], [4, 3], [0, 3]], "unit": "meter"},
            "objects": [
                {"id": "chair_001", "category": "chair"},
                {"id": "desk_001", "category": "desk"},
            ],
            "relations": [
                {
                    "id": "rel_001",
                    "type": "near",
                    "subject": "chair_001",
                    "object": "desk_001",
                    "visible_to_model": True,
                }
            ],
        }
    )


def test_structured_relation_with_spatial_cues_validates() -> None:
    _assert_valid(
        {
            "case_id": "cue_001",
            "schema_version": "2.0",
            "input_level": "structured_relation",
            "scene_representation_mode": "compact_objects_with_spatial_cues",
            "description": {"text": "Create a study room."},
            "room": {"boundary": [[0, 0], [4, 0], [4, 3], [0, 3]], "unit": "meter"},
            "objects": [
                {"id": "vase_001", "category": "vase"},
                {"id": "table_001", "category": "table"},
            ],
            "spatial_cues": [
                {
                    "id": "support_candidate__vase_001__table_001",
                    "type": "support_candidate",
                    "subject": "vase_001",
                    "target": "table_001",
                    "source": "bbox_geometry_heuristic",
                    "confidence": 0.8,
                    "hard": False,
                }
            ],
        }
    )


def test_legacy_cases_still_validate() -> None:
    for path in sorted((ROOT / "data" / "benchmark_cases").glob("*.json")):
        _assert_valid(read_json(path))
