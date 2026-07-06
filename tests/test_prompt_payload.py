from __future__ import annotations

from benchmark.models.base_model import build_generation_prompt
from benchmark.object_aliasing import build_object_alias_map
from benchmark.workflow.payloads import build_input_payloads


def _case(mode: str) -> dict:
    return {
        "case_id": "region_case",
        "input_level": "structured_relation" if "estimated" in mode else "structured_basic",
        "scene_representation_mode": mode,
        "description": {"text": "A test room."},
        "room": {
            "unit": "meter",
            "floor_z": 0,
            "boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
            "floor_plan": {
                "regions": [{"id": "kitchen_0", "label": "kitchen", "floor_polygon": [[0, 0], [2, 0], [2, 2], [0, 2]]}],
                "aggregate_boundary": [[0, 0], [4, 0], [4, 4], [0, 4]],
            },
        },
        "objects": [
            {
                "id": "20b73dd1f91dd128fb928fb7a032af2a47e79882_001",
                "category": "20b73dd1f91dd128fb928fb7a032af2a47e79882",
                "semantic_category": "chair",
                "bbox_size": [1, 1, 1],
                "layout_center_hint": [1, 1, 0.5],
                "source_position": [1, 0, 1],
                "source_floor_position": [1, 1],
                "source_region_id": "kitchen_0",
                "source_region_label": "kitchen",
            }
        ],
        "spatial_cues": [
            {
                "id": "near__chair_1__table_1",
                "type": "near",
                "subject": "20b73dd1f91dd128fb928fb7a032af2a47e79882_001",
                "object": "table_1",
                "target": "table_1",
                "source": "bbox_geometry_heuristic",
                "confidence": 0.8,
                "hard": False,
            }
        ],
    }


def test_prompt_only_payload_has_no_required_objects() -> None:
    payloads = build_input_payloads({**_case("compact_objects"), "input_level": "prompt_only", "scene_representation_mode": "prompt_only"})

    assert "objects" not in payloads["prompt_payload"]
    assert payloads["visibility_audit"]["prompt_object_list_visible"] is False


def test_alias_map_is_deterministic() -> None:
    first = build_object_alias_map(_case("compact_objects"))
    second = build_object_alias_map(_case("compact_objects"))

    assert first == second
    assert first["alias_order"] == ["o001"]
    assert first["canonical_to_alias"]["20b73dd1f91dd128fb928fb7a032af2a47e79882_001"] == "o001"


def test_compact_objects_payload_does_not_leak_regions_or_source_metadata() -> None:
    payloads = build_input_payloads(_case("compact_objects"))
    prompt_payload = payloads["prompt_payload"]
    eval_context = payloads["eval_context"]

    assert prompt_payload["objects"][0]["id"] == "o001"
    assert prompt_payload["objects"][0]["category"] == "chair"
    assert "20b73dd1f91dd128fb928fb7a032af2a47e79882_001" not in str(prompt_payload)
    assert "source_region_id" not in prompt_payload["objects"][0]
    assert "source_position" not in prompt_payload["objects"][0]
    assert "regions" not in prompt_payload["room"]["floor_plan"]
    assert payloads["visibility_audit"]["regions_visible_to_model"] is False
    assert eval_context["objects_by_id"]["20b73dd1f91dd128fb928fb7a032af2a47e79882_001"]["source_region_id"] == "kitchen_0"
    assert eval_context["object_alias_map"]["aliases"]["o001"]["canonical_object_id"] == "20b73dd1f91dd128fb928fb7a032af2a47e79882_001"
    assert eval_context["regions"]["available"] is True


def test_relation_mode_payload_includes_spatial_cues() -> None:
    payloads = build_input_payloads(_case("compact_objects_with_estimated_relations"))

    assert payloads["prompt_payload"]["spatial_cues"][0]["type"] == "near"
    assert payloads["prompt_payload"]["spatial_cues"][0]["subject"] == "o001"
    assert payloads["visibility_audit"]["spatial_cues_visible_to_model"] is True


def test_generation_prompt_uses_alias_rows_and_omits_hierarchy_contract() -> None:
    payloads = build_input_payloads(_case("compact_objects"))
    prompt = build_generation_prompt(payloads["prompt_payload"], {})

    assert '"required_object_rows"' in prompt
    assert "o001" in prompt
    assert "20b73dd1f91dd128fb928fb7a032af2a47e79882_001" not in prompt
    assert "source_object_metadata" not in prompt
    assert '"floor_objects"' not in prompt
    assert '"hierarchy"' not in prompt
