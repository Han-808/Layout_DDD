from __future__ import annotations

import pytest

from benchmark.relationship_mapper import map_relationships


def test_relationship_mapper_passthrough_shape() -> None:
    result = map_relationships(
        scene_request={"request_id": "demo_001"},
        object_plan={"request_id": "demo_001", "relations": []},
    )

    assert result == {
        "request_id": "demo_001",
        "status": "tbd_passthrough",
        "oor_relations": [],
        "oar_relations": [],
        "unsupported_relations": [],
        "notes": ["Relationship mapping is a TODO. This module defines the interface only."],
    }


def test_relationship_mapper_vlm_mode_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        map_relationships(scene_request={"request_id": "demo_001"}, object_plan={"request_id": "demo_001"}, mode="vlm")
