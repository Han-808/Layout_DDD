from __future__ import annotations

from benchmark.workflow.scoring import compute_object_presence


def test_id_based_presence() -> None:
    case = {
        "input_level": "structured_basic",
        "objects": [{"id": "bed_001", "category": "bed"}, {"id": "desk_001", "category": "desk"}],
    }
    layout = {"objects": [{"object_id": "bed_001", "category": "bed"}, {"object_id": "desk_001", "category": "desk"}]}

    result = compute_object_presence(case, layout)

    assert result.rate == 1.0
    assert result.missing_objects == []


def test_category_fallback_if_ids_absent() -> None:
    case = {"input_level": "structured_basic", "required_objects": ["chair", "chair", "desk"]}
    layout = {"objects": [{"object_id": "chair_1", "category": "chair"}, {"object_id": "desk_1", "category": "desk"}]}

    result = compute_object_presence(case, layout)

    assert result.rate == 2 / 3
    assert result.missing_objects == ["chair"]


def test_missing_object_lowers_rate() -> None:
    case = {
        "input_level": "structured_basic",
        "objects": [{"id": "bed_001", "category": "bed"}, {"id": "desk_001", "category": "desk"}],
    }
    layout = {"objects": [{"object_id": "bed_001", "category": "bed"}]}

    result = compute_object_presence(case, layout)

    assert result.rate == 0.5
    assert result.missing_objects == ["desk_001"]
