"""Native room contracts must hold on every Holodeck/SceneSmith import route."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from benchmark.api.evaluation import run_evaluate
from benchmark.api.generation import run_generate
from benchmark.evaluator.profile import L0
from benchmark.io_contracts import O1_OBJECT_STATE
from benchmark.nl_scene.generation_input import (
    build_direct_natural_language_generation_input,
)
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json, write_json


ROUTES = (
    "holodeck_raw",
    "scene_smith_raw",
    "holodeck_scene_state",
    "scene_smith_scene_state",
)
POLYGON_ROUTES = tuple(route for route in ROUTES if route != "scene_smith_raw")
RECTANGLE = [[0.0, 0.0], [4.0, 0.0], [4.0, 5.0], [0.0, 5.0]]


@pytest.mark.parametrize("route", ROUTES)
def test_matching_native_rooms_reach_the_same_full_evaluator(
    route: str, tmp_path: Path
) -> None:
    native = write_json(tmp_path / "native.json", _payload(route))
    original = native.read_bytes()
    result = _generate(route, native, tmp_path / "run")
    scene = read_json(result["generated_scene"])

    assert scene["boundary"] == RECTANGLE
    assert scene["scene_height"] == 3.0
    assert scene["objects"][0]["center"] == pytest.approx([2.0, 2.0, 0.5])
    assert scene["objects"][0]["asset_ref"]["asset_key"] == "chair-asset"
    assert scene["metadata"]["harness_compatibility"]["room_geometry_match"] == (
        "validated_against_benchmark"
    )
    assert native.read_bytes() == original
    assert Path(result["raw_native_artifact"]).read_bytes() == original
    report = run_evaluate(scene=scene, out=tmp_path / "evaluation_report.json")
    assert report["workflow"] == "canonical_l0_l4"
    assert report["request_id"] == "native-room-contract"
    assert report["layer_reports"][L0]["status"] == "passed"
    provenance_variant = deepcopy(scene)
    provenance_variant["metadata"]["harness_compatibility"]["room_geometry_match"] = (
        "audit-metadata-only"
    )
    assert run_evaluate(
        scene=provenance_variant, out=tmp_path / "provenance_variant_report.json"
    ) == report


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("dimension", ["width", "depth", "height"])
def test_native_room_conflicts_fail_before_asset_resolution(
    route: str, dimension: str, tmp_path: Path
) -> None:
    payload = _payload(route)
    if dimension == "height":
        _set_height(route, payload, 2.8)
    elif route == "scene_smith_raw":
        payload["room_geometry"]["length" if dimension == "width" else "width"] = 6.0
    else:
        boundary = deepcopy(RECTANGLE)
        axis = 0 if dimension == "width" else 1
        for point in boundary:
            if point[axis] > 0:
                point[axis] = 6.0
        _set_boundary(route, payload, boundary)
    native = write_json(tmp_path / "conflicting_native.json", payload)
    original = native.read_bytes()

    class UntouchedProvider:
        def resolve(self, *args, **kwargs):
            raise AssertionError("room conflicts must fail before exact asset lookup")

        def retrieve(self, *args, **kwargs):
            raise AssertionError("strict conversion must not retrieve")

    with pytest.raises(ArtifactValidationError, match="conflicts"):
        _generate(
            route, native, tmp_path / "run", {"asset_provider": UntouchedProvider()}
        )
    assert native.read_bytes() == original
    assert not (tmp_path / "run" / "generated_scene.json").exists()


@pytest.mark.parametrize("route", POLYGON_ROUTES)
def test_native_nonrectangular_boundaries_are_not_approximated(
    route: str, tmp_path: Path
) -> None:
    payload = _payload(route)
    # Same bounding rectangle as the benchmark, but different floor geometry.
    _set_boundary(route, payload, [[0, 0], [4, 0], [4, 2], [2, 2], [2, 5], [0, 5]])
    native = write_json(tmp_path / "polygon.json", payload)
    with pytest.raises(ArtifactValidationError, match="will not approximate or flatten"):
        _generate(route, native, tmp_path / "run")
    assert not (tmp_path / "run" / "generated_scene.json").exists()


@pytest.mark.parametrize("route", POLYGON_ROUTES)
def test_room_match_allows_only_origin_winding_and_cyclic_normalization(
    route: str, tmp_path: Path
) -> None:
    payload = _payload(route)
    ring = [[x + 10.0, y - 7.0] for x, y in RECTANGLE]
    ring = list(reversed(ring[2:] + ring[:2]))
    _set_boundary(route, payload, ring + [ring[0]])
    if route == "holodeck_raw":
        payload["objects"][0]["position"].update({"x": 12.0, "z": -5.0})
    else:
        payload["scene"]["object"][0]["transform"]["data"][12:15] = [12.0, -5.0, 0.5]
    native = write_json(tmp_path / "translated.json", payload)
    scene = read_json(_generate(route, native, tmp_path / "run")["generated_scene"])
    assert scene["objects"][0]["center"] == pytest.approx([2.0, 2.0, 0.5])
    assert {tuple(point) for point in scene["boundary"]} == {
        tuple(point) for point in RECTANGLE
    }


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("height", [0, -1, float("nan"), float("inf")])
def test_invalid_native_height_is_not_replaced_with_benchmark_height(
    route: str, height: float, tmp_path: Path
) -> None:
    payload = _payload(route)
    _set_height(route, payload, height)
    native = write_json(tmp_path / "invalid_height.json", payload)
    with pytest.raises(ArtifactValidationError, match="height.*(?:finite|positive)"):
        _generate(route, native, tmp_path / "run")


@pytest.mark.parametrize("dimension", ["length", "width"])
@pytest.mark.parametrize("value", [None, 0, -1, float("nan"), float("inf")])
def test_scenesmith_partial_or_invalid_room_dimensions_fail_closed(
    dimension: str, value: float | None, tmp_path: Path
) -> None:
    payload = _payload("scene_smith_raw")
    payload["room_geometry"][dimension] = value
    native = write_json(tmp_path / "invalid_room.json", payload)
    with pytest.raises(ArtifactValidationError, match="room_geometry"):
        _generate("scene_smith_raw", native, tmp_path / "run")


@pytest.mark.parametrize("route", ["holodeck_scene_state", "scene_smith_scene_state"])
def test_scenestate_does_not_drop_an_additional_selected_floor(
    route: str, tmp_path: Path
) -> None:
    payload = _payload(route)
    floors = payload["scene"]["arch"]["elements"]
    floors.append({**deepcopy(floors[0]), "id": "floor|room|extra"})
    native = write_json(tmp_path / "multiple_floors.json", payload)
    with pytest.raises(ArtifactValidationError, match="multiple floors"):
        _generate(route, native, tmp_path / "run", {"room_id": "room"})


@pytest.mark.parametrize("route", ROUTES)
def test_per_room_projection_is_explicit_and_still_checks_selected_geometry(
    route: str, tmp_path: Path
) -> None:
    payload = _payload(route)
    if route == "holodeck_raw":
        payload["rooms"].append({**deepcopy(payload["rooms"][0]), "id": "other"})
    elif route == "scene_smith_raw":
        payload = {"rooms": {"room": payload, "other": deepcopy(payload)}}
    else:
        elements = payload["scene"]["arch"]["elements"]
        elements.append({**deepcopy(elements[0]), "id": "floor|other", "roomId": "other"})
    native = write_json(tmp_path / "house.json", payload)
    with pytest.raises(ArtifactValidationError, match="multiple (?:rooms|floors)"):
        _generate(route, native, tmp_path / "whole_house")
    selected = _generate(route, native, tmp_path / "selected", {"room_id": "room"})
    assert read_json(selected["generated_scene"])["boundary"] == RECTANGLE

    # An explicitly selected room must match too; the other room is not a fallback.
    if route == "scene_smith_raw":
        payload["rooms"]["room"]["room_geometry"]["length"] = 6.0
    else:
        _set_boundary(route, payload, [[0, 0], [6, 0], [6, 5], [0, 5]])
    native = write_json(tmp_path / "mismatching_house.json", payload)
    with pytest.raises(ArtifactValidationError, match="conflicts"):
        _generate(route, native, tmp_path / "mismatching", {"room_id": "room"})


@pytest.mark.parametrize("route", POLYGON_ROUTES)
def test_inconsistent_wall_heights_are_not_hidden_by_the_maximum(
    route: str, tmp_path: Path
) -> None:
    payload = _payload(route)
    if route == "holodeck_raw":
        payload["walls"] = [{"roomId": "room", "height": 2.0}]
    else:
        payload["scene"]["arch"]["elements"].append(
            {"type": "Wall", "roomId": "room", "height": 2.0}
        )
    native = write_json(tmp_path / "wall_conflict.json", payload)
    with pytest.raises(ArtifactValidationError, match="wall heights conflict"):
        _generate(route, native, tmp_path / "run")


@pytest.mark.parametrize("route", ["holodeck_scene_state", "scene_smith_scene_state"])
@pytest.mark.parametrize("unit,scale", [(0, 1), (-1, 1), (1, 0), (-1, -1)])
def test_scenestate_units_must_be_positive(
    route: str, unit: float, scale: float, tmp_path: Path
) -> None:
    payload = _payload(route)
    payload["scene"]["unit"] = unit
    payload["scene"]["arch"]["scaleToMeters"] = scale
    native = write_json(tmp_path / "invalid_units.json", payload)
    with pytest.raises(ArtifactValidationError, match="must be positive"):
        _generate(route, native, tmp_path / "run")


def test_malformed_holodeck_room_entries_are_not_discarded(tmp_path: Path) -> None:
    payload = _payload("holodeck_raw")
    payload["rooms"].append(None)
    native = write_json(tmp_path / "invalid_rooms.json", payload)
    with pytest.raises(ArtifactValidationError, match="rooms entries must be objects"):
        _generate("holodeck_raw", native, tmp_path / "run")


def _generate(route: str, native: Path, out_dir: Path, config: dict | None = None):
    return run_generate(
        generation_input=build_direct_natural_language_generation_input(
            request_id="native-room-contract",
            instruction="Arrange a chair in the room.",
            scene_type="room",
            room={"boundary": deepcopy(RECTANGLE), "height": 3.0},
            evaluator_output_type=O1_OBJECT_STATE,
        ),
        adapter_name="holodeck" if route.startswith("holodeck") else "scene_smith",
        method_output=native,
        out_dir=out_dir,
        adapter_config=config,
    )


def _payload(route: str) -> dict:
    if route == "holodeck_raw":
        return {
            "wall_height": 3.0,
            "rooms": [{"id": "room", "floorPolygon": [
                {"x": x, "y": 0.0, "z": y} for x, y in RECTANGLE
            ]}],
            "walls": [],
            "objects": [{
                "id": "chair_1", "assetId": "chair-asset", "roomId": "room",
                "category": "chair", "position": {"x": 2, "y": 0.5, "z": 2},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "axisAlignedBoundingBox": {"size": {"x": 0.8, "y": 1, "z": 0.8}},
            }],
        }
    if route == "scene_smith_raw":
        return {
            "room_geometry": {"length": 4.0, "width": 5.0, "wall_height": 3.0},
            "objects": {"chair_1": {
                "object_id": "chair_1", "object_type": "furniture", "name": "chair",
                "asset_id": "chair-asset",
                "transform": {"translation": [0, -0.5, 0], "rotation_wxyz": [1, 0, 0, 0]},
                "bbox_min": [-0.4, -0.4, 0], "bbox_max": [0.4, 0.4, 1],
            }},
        }
    return {
        "format": "sceneState",
        "scene": {
            "unit": 1.0, "up": [0, 0, 1],
            "arch": {"coords2d": [0, 1], "scaleToMeters": 1.0, "elements": [
                {"id": "floor|room", "type": "Floor", "roomId": "room",
                 "points": [[x, y, 0] for x, y in RECTANGLE]},
                {"id": "wall|room", "type": "Wall", "roomId": "room", "height": 3.0},
            ]},
            "object": [{
                "id": "chair_1", "modelId": "synthetic.chair-asset", "category": "chair",
                "roomId": "room",
                "bbox_size": [0.8, 0.8, 1.0],
                "transform": {"data": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 2, 0.5, 1]},
            }],
        },
    }


def _set_boundary(route: str, payload: dict, boundary: list[list[float]]) -> None:
    if route == "holodeck_raw":
        payload["rooms"][0]["floorPolygon"] = [{"x": x, "y": 0, "z": y} for x, y in boundary]
    else:
        payload["scene"]["arch"]["elements"][0]["points"] = [[x, y, 0] for x, y in boundary]


def _set_height(route: str, payload: dict, height: float) -> None:
    if route == "holodeck_raw":
        payload["wall_height"] = height
    elif route == "scene_smith_raw":
        payload["room_geometry"]["wall_height"] = height
    else:
        payload["scene"]["arch"]["elements"][1]["height"] = height
