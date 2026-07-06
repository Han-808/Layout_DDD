from __future__ import annotations

from benchmark.workflow.layout_normalization import enforce_layout_object_set


CASE = {
    "objects": [
        {"id": "chair_1", "category": "chair", "bbox_size": [0.5, 0.6, 0.9], "layout_center_hint": [1.0, 1.0, 0.45]},
        {"id": "desk_1", "category": "desk", "bbox_size": [1.2, 0.7, 0.8], "layout_center_hint": [2.0, 1.0, 0.4]},
    ]
}

ALIASED_CASE = {
    "objects": [
        {
            "id": "20b73dd1f91dd128fb928fb7a032af2a47e79882_001",
            "category": "20b73dd1f91dd128fb928fb7a032af2a47e79882",
            "semantic_category": "chair",
            "bbox_size": [0.5, 0.6, 0.9],
            "layout_center_hint": [1.0, 1.0, 0.45],
        },
        {
            "id": "3bd3f3a882da9283719fe1d238ad49d88760d0c3_002",
            "category": "3bd3f3a882da9283719fe1d238ad49d88760d0c3",
            "semantic_category": "desk",
            "bbox_size": [1.2, 0.7, 0.8],
            "layout_center_hint": [2.0, 1.0, 0.4],
        },
    ],
    "object_alias_map": {
        "enabled": True,
        "alias_order": ["o001", "o002"],
        "canonical_to_alias": {
            "20b73dd1f91dd128fb928fb7a032af2a47e79882_001": "o001",
            "3bd3f3a882da9283719fe1d238ad49d88760d0c3_002": "o002",
        },
        "aliases": {
            "o001": {
                "alias": "o001",
                "canonical_object_id": "20b73dd1f91dd128fb928fb7a032af2a47e79882_001",
                "canonical_category": "20b73dd1f91dd128fb928fb7a032af2a47e79882",
                "model_visible_category": "chair",
                "bbox_size": [0.5, 0.6, 0.9],
            },
            "o002": {
                "alias": "o002",
                "canonical_object_id": "3bd3f3a882da9283719fe1d238ad49d88760d0c3_002",
                "canonical_category": "3bd3f3a882da9283719fe1d238ad49d88760d0c3",
                "model_visible_category": "desk",
                "bbox_size": [1.2, 0.7, 0.8],
            },
        },
    },
}


def test_generation_missing_and_extra_objects_are_normalized() -> None:
    layout = {
        "objects": [
            {"object_id": "chair_1", "category": "wrong", "center": [1.1, 1.0, 0.45], "size": [0.5, 0.6, 0.9]},
            {"object_id": "extra_1", "category": "extra", "center": [0, 0, 0.5], "size": [1, 1, 1]},
        ]
    }

    normalized, report = enforce_layout_object_set(layout, CASE, stage="generation")

    assert [obj["object_id"] for obj in normalized["objects"]] == ["chair_1", "desk_1"]
    assert normalized["objects"][0]["category"] == "chair"
    assert normalized["objects"][1]["center"] == [2.0, 1.0, 0.4]
    assert report["final_object_count"] == 2
    assert report["missing_restored_or_synthesized"] == ["desk_1"]
    assert report["extra_dropped"] == ["extra_1"]


def test_repair_missing_object_is_restored_from_previous_layout() -> None:
    previous = {
        "objects": [
            {"object_id": "chair_1", "category": "chair", "center": [1.0, 1.0, 0.45], "size": [0.5, 0.6, 0.9]},
            {"object_id": "desk_1", "category": "desk", "center": [2.0, 1.0, 0.4], "size": [1.2, 0.7, 0.8]},
        ]
    }
    repair = {"objects": [{"object_id": "chair_1", "category": "chair", "center": [1.2, 1.0, 0.45], "size": [0.5, 0.6, 0.9]}]}

    normalized, report = enforce_layout_object_set(repair, CASE, previous_layout=previous, stage="repair_iter_1")

    assert [obj["object_id"] for obj in normalized["objects"]] == ["chair_1", "desk_1"]
    assert normalized["objects"][1]["center"] == [2.0, 1.0, 0.4]
    assert report["missing_restored_or_synthesized"] == ["desk_1"]


def test_alias_output_remaps_to_canonical_object_ids() -> None:
    layout = {
        "objects": [
            {"object_id": "o001", "category": "chair", "center": [1.1, 1.0, 0.45], "size": [0.5, 0.6, 0.9], "yaw": 0},
            {"object_id": "o002", "category": "desk", "center": [2.1, 1.0, 0.4], "size": [1.2, 0.7, 0.8], "yaw": 0},
        ],
        "relations": [{"type": "near", "source": "o001", "target": "o002"}],
    }

    normalized, report = enforce_layout_object_set(layout, ALIASED_CASE, stage="generation")

    assert [obj["object_id"] for obj in normalized["objects"]] == [
        "20b73dd1f91dd128fb928fb7a032af2a47e79882_001",
        "3bd3f3a882da9283719fe1d238ad49d88760d0c3_002",
    ]
    assert normalized["objects"][0]["model_object_id"] == "o001"
    assert normalized["objects"][0]["model_category"] == "chair"
    assert normalized["objects"][0]["category"] == "20b73dd1f91dd128fb928fb7a032af2a47e79882"
    assert normalized["relations"][0]["source"] == "20b73dd1f91dd128fb928fb7a032af2a47e79882_001"
    assert report["alias_remap"]["missing_aliases"] == []


def test_alias_output_reports_missing_extra_and_duplicate_aliases() -> None:
    layout = {
        "objects": [
            {"object_id": "o001", "category": "chair", "center": [1, 1, 0.45], "size": [0.5, 0.6, 0.9], "yaw": 0},
            {"object_id": "o001", "category": "chair", "center": [1.2, 1, 0.45], "size": [0.5, 0.6, 0.9], "yaw": 0},
            {"object_id": "o999", "category": "extra", "center": [0, 0, 0.5], "size": [1, 1, 1], "yaw": 0},
        ]
    }

    _, report = enforce_layout_object_set(layout, ALIASED_CASE, stage="generation")

    assert report["alias_remap"]["missing_aliases"] == ["o002"]
    assert report["alias_remap"]["duplicate_aliases"] == ["o001"]
    assert any(flag["type"] == "unknown_alias" for flag in report["alias_remap"]["flags"])
