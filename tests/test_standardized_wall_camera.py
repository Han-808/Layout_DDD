from __future__ import annotations

import pytest

from benchmark.rendering.blender_worker import (
    STANDARDIZED_PERSPECTIVE_CAMERA_POLICY,
    standardized_perspective_camera_spec,
)


BOUNDARY = [[0.0, 0.0], [4.0, 0.0], [4.0, 6.0], [0.0, 6.0]]


def _camera(*walls: str) -> dict:
    return standardized_perspective_camera_spec(BOUNDARY, 3.0, walls)


def test_adjacent_open_walls_preserve_historical_southeast_pose() -> None:
    camera = _camera("north_wall", "west_wall")

    assert camera["policy_id"] == STANDARDIZED_PERSPECTIVE_CAMERA_POLICY
    assert camera["open_wall_ids"] == ["south_wall", "east_wall"]
    assert camera["selected_view"] == "southeast_open_corner"
    assert camera["camera_location"] == pytest.approx([7.1, -2.7, 7.2])
    assert camera["camera_target"] == pytest.approx([2.0, 3.0, 1.2])
    assert camera["fallback_reason"] is None


@pytest.mark.parametrize(
    ("active_walls", "selected_view", "expected_xy"),
    (
        (
            ("north_wall", "east_wall", "west_wall"),
            "south_open_side",
            (2.0, -4.65),
        ),
        (
            ("north_wall", "south_wall", "west_wall"),
            "east_open_side",
            (9.65, 3.0),
        ),
        (
            ("north_wall", "south_wall", "east_wall"),
            "west_open_side",
            (-5.65, 3.0),
        ),
    ),
)
def test_one_open_wall_selects_its_center(
    active_walls: tuple[str, ...],
    selected_view: str,
    expected_xy: tuple[float, float],
) -> None:
    camera = standardized_perspective_camera_spec(
        BOUNDARY,
        3.0,
        active_walls,
    )

    assert camera["selected_view"] == selected_view
    assert camera["camera_location"][:2] == pytest.approx(expected_xy)
    assert camera["fallback_reason"] is None


def test_opposite_open_walls_use_stable_southeast_axis_tie_break() -> None:
    camera = _camera("north_wall", "south_wall")

    assert camera["open_wall_ids"] == ["east_wall", "west_wall"]
    assert camera["selected_view"] == "east_open_side"
    assert camera["selection_reason"] == (
        "opposite_open_walls_prefer_historical_southeast_axis"
    )


def test_no_walls_preserves_historical_southeast_default() -> None:
    camera = _camera()

    assert camera["selected_view"] == "southeast_default"
    assert camera["camera_location"] == pytest.approx([7.1, -2.7, 7.2])


def test_four_walls_use_only_the_high_closed_fallback() -> None:
    camera = _camera(
        "north_wall",
        "south_wall",
        "east_wall",
        "west_wall",
    )

    assert camera["open_wall_ids"] == []
    assert camera["selected_view"] == "southeast_high_closed_fallback"
    assert camera["fallback_reason"] == "no_open_wall"
    assert camera["camera_location"] == pytest.approx([7.1, -2.7, 9.6])


def test_wall_order_does_not_change_camera_and_unknown_walls_fail_closed() -> None:
    expected = _camera("north_wall", "west_wall")
    reordered = _camera("west_wall", "north_wall", "west_wall")

    assert reordered == expected
    with pytest.raises(ValueError, match="unknown active wall ids"):
        _camera("north_wall", "door_wall")
