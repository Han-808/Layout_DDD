from __future__ import annotations

from benchmark.workflow.layout_normalization import enforce_layout_object_set


CASE = {
    "objects": [
        {"id": "chair_1", "category": "chair", "bbox_size": [0.5, 0.6, 0.9], "layout_center_hint": [1.0, 1.0, 0.45]},
        {"id": "desk_1", "category": "desk", "bbox_size": [1.2, 0.7, 0.8], "layout_center_hint": [2.0, 1.0, 0.4]},
    ]
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
