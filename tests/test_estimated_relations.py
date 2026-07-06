from __future__ import annotations

from benchmark.datasets.estimated_relations import build_estimated_spatial_cues, compatibility_relations


def test_estimated_near_cues_are_deterministic_and_capped() -> None:
    objects = [
        {"id": "obj_c", "bbox_size": [1, 1, 1], "source_floor_position": [2.0, 0.0]},
        {"id": "obj_a", "bbox_size": [1, 1, 1], "source_floor_position": [0.0, 0.0]},
        {"id": "obj_b", "bbox_size": [1, 1, 1], "source_floor_position": [0.5, 0.0]},
    ]

    first = build_estimated_spatial_cues(objects, max_near_cues_total=1, max_near_cues_per_object=1)
    second = build_estimated_spatial_cues(list(reversed(objects)), max_near_cues_total=1, max_near_cues_per_object=1)

    assert first == second
    near = [cue for cue in first if cue["type"] == "near"]
    assert len(near) == 1
    assert near[0]["id"] == "near__obj_a__obj_b"
    assert near[0]["source"] == "bbox_geometry_heuristic"
    assert 0 <= near[0]["confidence"] <= 1
    assert near[0]["hard"] is False
    assert "horizontal_distance" in near[0]["evidence"]


def test_support_candidate_requires_vertical_contact_and_footprint_overlap() -> None:
    objects = [
        {"id": "table", "bbox_size": [2.0, 2.0, 1.0], "layout_center_hint": [0.0, 0.0, 0.5], "source_floor_position": [0.0, 0.0]},
        {"id": "vase", "bbox_size": [0.4, 0.4, 0.4], "layout_center_hint": [0.0, 0.0, 1.2], "source_floor_position": [0.0, 0.0]},
        {"id": "floating", "bbox_size": [0.4, 0.4, 0.4], "layout_center_hint": [0.0, 0.0, 2.0], "source_floor_position": [0.0, 0.0]},
        {"id": "beside", "bbox_size": [0.4, 0.4, 0.4], "layout_center_hint": [3.0, 3.0, 1.2], "source_floor_position": [3.0, 3.0]},
    ]

    cues = build_estimated_spatial_cues(objects)
    support = [cue for cue in cues if cue["type"] == "support_candidate"]

    assert [cue["id"] for cue in support] == ["support_candidate__vase__table"]
    assert support[0]["hard"] is False
    assert support[0]["evidence"]["vertical_gap"] == 0.0
    assert support[0]["evidence"]["footprint_overlap_ratio"] > 0


def test_only_relation_compatible_cues_enter_legacy_relations() -> None:
    cues = build_estimated_spatial_cues(
        [
            {"id": "table", "bbox_size": [2.0, 2.0, 1.0], "layout_center_hint": [0.0, 0.0, 0.5], "source_floor_position": [0.0, 0.0]},
            {"id": "vase", "bbox_size": [0.4, 0.4, 0.4], "layout_center_hint": [0.0, 0.0, 1.2], "source_floor_position": [0.0, 0.0]},
        ]
    )

    assert any(cue["type"] == "support_candidate" for cue in cues)
    assert all(cue["type"] != "support_candidate" for cue in compatibility_relations(cues))
