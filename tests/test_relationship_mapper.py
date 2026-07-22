from __future__ import annotations

import pytest

from benchmark.relationship_mapper import map_relationships
from benchmark.relationship_mapper.schemas import oor_relation_intent


def test_relationship_mapper_passthrough_shape() -> None:
    result = map_relationships(
        scene_request={"request_id": "demo_001"},
        object_plan={"request_id": "demo_001", "relations": []},
    )

    assert result == {
        "request_id": "demo_001",
        "status": "mapped",
        "oor_relations": [],
        "oar_relations": [],
        "unsupported_relations": [],
        "notes": [
            "The converter maps explicit text to the frozen relation registry.",
            "Unknown explicit predicates are preserved for evaluator-side VLM adjudication.",
        ],
    }


def test_relationship_mapper_vlm_mode_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        map_relationships(scene_request={"request_id": "demo_001"}, object_plan={"request_id": "demo_001"}, mode="vlm")


def test_unknown_predicate_is_preserved_inside_its_relation_family() -> None:
    relation = oor_relation_intent(
        relation_type="Mirrors With",
        subject_id="left_chair",
        anchor_id="right_chair",
        raw_relation="the left chair mirrors the right chair",
        source="explicit_text",
    )
    result = map_relationships(
        scene_request={"request_id": "demo_001"},
        object_plan={"request_id": "demo_001", "relations": [relation]},
    )

    assert relation["type"] == "mirrors_with"
    assert result["oor_relations"] == [relation]
    assert result["unsupported_relations"] == []
