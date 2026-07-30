from __future__ import annotations

import pytest

from benchmark.task_contract import (
    architecture_contract_for_room,
    require_scene_matches_architecture,
    resolve_room_contract,
)


def _dimensions(room: dict) -> dict:
    return room["dimensions"]


def test_room_contract_uses_fallback_when_no_dimensions_are_explicit() -> None:
    room = resolve_room_contract(None)

    assert _dimensions(room) == {"width": 7.0, "depth": 5.0, "height": 3.0}
    assert room["boundary"] == [[0.0, 0.0], [7.0, 0.0], [7.0, 5.0], [0.0, 5.0]]
    assert room["explicit_dimensions"] == {}
    assert set(room["dimension_provenance"].values()) == {"benchmark_fallback"}


@pytest.mark.parametrize(
    ("prompt_dimensions", "expected"),
    [
        ({"width": 8.0}, {"width": 8.0, "depth": 6.0, "height": 3.2}),
        ({"depth": 6.0}, {"width": 8.0, "depth": 6.0, "height": 3.2}),
        ({"height": 2.8}, {"width": 7.0, "depth": 5.25, "height": 2.8}),
    ],
)
def test_one_dimension_uses_canonical_ratio(prompt_dimensions: dict, expected: dict) -> None:
    assert _dimensions(resolve_room_contract(None, prompt_dimensions=prompt_dimensions)) == expected


def test_two_horizontal_dimensions_derive_height() -> None:
    room = resolve_room_contract(None, prompt_dimensions={"width": 8.0, "depth": 8.0})

    assert _dimensions(room) == {"width": 8.0, "depth": 8.0, "height": 4.0}
    assert room["dimension_provenance"]["height"] == "derived_from_two_horizontal_dimensions"


@pytest.mark.parametrize(
    ("partial", "expected"),
    [
        (
            {"width": 8.0, "height": 2.8},
            {"width": 8.0, "depth": 5.0, "height": 2.8},
        ),
        (
            {"depth": 6.0, "height": 2.8},
            {"width": 7.0, "depth": 6.0, "height": 2.8},
        ),
    ],
)
def test_other_two_axis_combinations_fill_only_missing_fallback(
    partial: dict,
    expected: dict,
) -> None:
    room = resolve_room_contract(None, prompt_dimensions=partial)

    assert _dimensions(room) == expected
    missing = next(axis for axis in expected if axis not in partial)
    assert room["dimension_provenance"][missing] == "benchmark_fallback_for_partial_combination"


def test_all_explicit_dimensions_are_preserved() -> None:
    room = resolve_room_contract(
        {"dimensions": {"width": 8.0, "depth": 8.0, "height": 2.8}}
    )

    assert _dimensions(room) == {"width": 8.0, "depth": 8.0, "height": 2.8}
    assert room["explicit_dimensions"] == room["dimensions"]


def test_boundary_input_is_normalized_to_min_corner_room() -> None:
    room = resolve_room_contract(
        {"boundary": [[-4, -3], [4, -3], [4, 3], [-4, 3]], "height": 2.8}
    )

    assert room["boundary"] == [[0.0, 0.0], [8.0, 0.0], [8.0, 6.0], [0.0, 6.0]]
    assert _dimensions(room) == {"width": 8.0, "depth": 6.0, "height": 2.8}


def test_structured_and_prompt_dimensions_must_not_conflict() -> None:
    with pytest.raises(ValueError, match="conflicting explicit room width"):
        resolve_room_contract(
            {"dimensions": {"width": 8.0}},
            prompt_dimensions={"width": 7.0},
        )


def test_architecture_contract_and_scene_gate_use_resolved_room() -> None:
    room = resolve_room_contract(None, prompt_dimensions={"width": 8.0, "depth": 6.0})
    architecture = architecture_contract_for_room(room)
    scene = {
        "boundary": room["boundary"],
        "scene_height": room["height"],
    }

    assert architecture["wall_count"] == 4
    assert architecture["elements"] == ["floor", "walls", "ceiling"]
    require_scene_matches_architecture(scene, room)

    scene["scene_height"] = 3.0
    with pytest.raises(ValueError, match="conflicts with the resolved benchmark room"):
        require_scene_matches_architecture(scene, room)
