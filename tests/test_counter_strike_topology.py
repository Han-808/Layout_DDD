from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from benchmark.game_scene.counter_strike.loader import (
    CanonicalSceneImportTransform,
    CounterStrikeBenchmarkConfig,
    CounterStrikeCaseContract,
    load_counter_strike_benchmark_config,
)
from benchmark.game_scene.counter_strike.topology import (
    CounterStrikeTopologyError,
    _duplicate_cover_assembly_count,
    _engagement_region,
    analyze_counter_strike_static_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG = (
    ROOT / "configs" / "game" / "counter_strike" / "benchmark_v1.yaml"
)


@pytest.fixture
def benchmark_config() -> CounterStrikeBenchmarkConfig:
    return load_counter_strike_benchmark_config(BENCHMARK_CONFIG)


@pytest.fixture
def case_contract_factory(
    tmp_path: Path,
) -> Callable[..., CounterStrikeCaseContract]:
    def build(
        team_a: tuple[float, float],
        team_b: tuple[float, float],
    ) -> CounterStrikeCaseContract:
        canonical_spawns = {
            "team_a": {
                "points": [[team_a[0], team_a[1], 0.0]],
                "jitter_radius_m": 0.0,
            },
            "team_b": {
                "points": [[team_b[0], team_b[1], 0.0]],
                "jitter_radius_m": 0.0,
            },
        }
        return CounterStrikeCaseContract(
            path=tmp_path / "synthetic_case_contract.json",
            sha256="synthetic-case-contract",
            source_root=tmp_path,
            raw={"case_id": "synthetic_counter_strike_topology"},
            source_assertions=(),
            import_transform=CanonicalSceneImportTransform(
                source_up_axis="z",
                unit_scale=1.0,
                translation_applied=(0.0, 0.0, 0.0),
            ),
            canonical_team_spawns=canonical_spawns,
        )

    return build


def _object(
    object_id: str,
    *,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
) -> dict:
    return {
        "id": object_id,
        "category": "level_geometry",
        "center": list(center),
        "size": list(size),
        "rotation": [0.0, 0.0, 0.0],
        "metadata": {"interactive": False},
    }


def _arena(
    *,
    width: float = 24.0,
    depth: float = 12.0,
    obstacles: tuple[dict, ...] = (),
    include_floor: bool = True,
) -> dict:
    wall_thickness = 0.20
    wall_height = 3.0
    objects = [
        _object(
            "wall_west",
            center=(wall_thickness / 2.0, depth / 2.0, wall_height / 2.0),
            size=(wall_thickness, depth, wall_height),
        ),
        _object(
            "wall_east",
            center=(
                width - wall_thickness / 2.0,
                depth / 2.0,
                wall_height / 2.0,
            ),
            size=(wall_thickness, depth, wall_height),
        ),
        _object(
            "wall_south",
            center=(width / 2.0, wall_thickness / 2.0, wall_height / 2.0),
            size=(width, wall_thickness, wall_height),
        ),
        _object(
            "wall_north",
            center=(
                width / 2.0,
                depth - wall_thickness / 2.0,
                wall_height / 2.0,
            ),
            size=(width, wall_thickness, wall_height),
        ),
    ]
    if include_floor:
        objects.append(
            _object(
                "oversized_floor",
                center=(width / 2.0, depth / 2.0, 0.05),
                size=(width, depth, 0.10),
            )
        )
    objects.extend(deepcopy(list(obstacles)))
    return {
        "schema_version": "canonical_scene_v1",
        "scene_id": "synthetic_counter_strike_arena",
        "request_id": "synthetic_counter_strike_request",
        "scene_type": "counter_strike_static_arena",
        "boundary": [
            [0.0, 0.0],
            [width, 0.0],
            [width, depth],
            [0.0, depth],
        ],
        "scene_height": wall_height,
        "objects": objects,
        "metadata": {
            "coordinate_frame": {
                "origin": "room_min_corner_floor",
                "axes": "x_width_y_depth_z_up",
                "unit": "meter",
                "rotation_unit": "degree",
            }
        },
    }


def _multi_lane_obstacles() -> tuple[dict, ...]:
    # One authored island creates two spatially distinct passages between the
    # same endpoints; unlike repeated shortest-path penalties, this is a real
    # branch/loop in free-space topology.
    return (
        _object(
            "central_route_splitter",
            center=(12.0, 6.0, 1.5),
            size=(2.0, 6.0, 3.0),
        ),
    )


def _scores(
    scene: dict,
    *,
    contract: CounterStrikeCaseContract,
    config: CounterStrikeBenchmarkConfig,
) -> tuple[object, dict[str, dict]]:
    return analyze_counter_strike_static_geometry(
        scene,
        case_contract=contract,
        benchmark_config=config,
    )


def test_single_undivided_corridor_does_not_fabricate_four_routes(
    benchmark_config: CounterStrikeBenchmarkConfig,
    case_contract_factory: Callable[..., CounterStrikeCaseContract],
) -> None:
    topology, metrics = _scores(
        _arena(depth=5.0),
        contract=case_contract_factory((2.0, 2.5), (22.0, 2.5)),
        config=benchmark_config,
    )

    assert len(topology.routes) == 1
    assert metrics["route_structure"]["candidate_count"] == 1
    assert (
        metrics["route_structure"]["main_route_count"]
        + metrics["route_structure"]["flank_route_count"]
        + metrics["route_structure"]["alternate_route_count"]
        == 1
    )


def test_authored_multi_lane_layout_has_more_structural_routes_than_corridor(
    benchmark_config: CounterStrikeBenchmarkConfig,
    case_contract_factory: Callable[..., CounterStrikeCaseContract],
) -> None:
    contract = case_contract_factory((2.0, 6.0), (22.0, 6.0))
    corridor, corridor_metrics = _scores(
        _arena(),
        contract=contract,
        config=benchmark_config,
    )
    multi_lane, multi_lane_metrics = _scores(
        _arena(obstacles=_multi_lane_obstacles()),
        contract=contract,
        config=benchmark_config,
    )

    assert len(corridor.routes) == 1
    assert len(multi_lane.routes) > len(corridor.routes)
    assert (
        multi_lane_metrics["route_structure"]["candidate_count"]
        > corridor_metrics["route_structure"]["candidate_count"]
    )


def test_cross_spawn_exposure_is_charged_to_spawn_balance_alone(
    benchmark_config: CounterStrikeBenchmarkConfig,
    case_contract_factory: Callable[..., CounterStrikeCaseContract],
) -> None:
    """One topology fact may move one metric.

    An open arena leaves each spawn in line of sight of the other and an
    authored splitter breaks that line.  Whether a spawn can be shot into at
    spawn time is a fairness property, which is what spawn_balance measures.
    zone_clarity asks a different question -- whether each required role exists
    as its own region -- and both arenas answer it the same way, so scoring the
    exposure in both places charged one finding twice.
    """

    contract = case_contract_factory((2.0, 6.0), (22.0, 6.0))
    _, exposed = _scores(_arena(), contract=contract, config=benchmark_config)
    _, screened = _scores(
        _arena(obstacles=_multi_lane_obstacles()),
        contract=contract,
        config=benchmark_config,
    )

    assert (
        exposed["spawn_balance"]["components"]["initial_cross_spawn_los"] == 0.0
    )
    assert (
        screened["spawn_balance"]["components"]["initial_cross_spawn_los"]
        == 1.0
    )
    assert (
        screened["spawn_balance"]["score"] > exposed["spawn_balance"]["score"]
    )

    for metrics in (exposed, screened):
        zone = metrics["zone_clarity"]
        assert set(zone["components"]) == {
            "spawn_regions_separate",
            "preparation_regions_present",
            "main_engagement_region_present",
            "flank_region_present",
        }
        assert zone["score"] == pytest.approx(1.0)


def test_swapping_teams_preserves_symmetric_topology_scores(
    benchmark_config: CounterStrikeBenchmarkConfig,
    case_contract_factory: Callable[..., CounterStrikeCaseContract],
) -> None:
    scene = _arena(obstacles=_multi_lane_obstacles())
    _, forward = _scores(
        scene,
        contract=case_contract_factory((2.0, 6.0), (22.0, 6.0)),
        config=benchmark_config,
    )
    _, swapped = _scores(
        scene,
        contract=case_contract_factory((22.0, 6.0), (2.0, 6.0)),
        config=benchmark_config,
    )

    assert (
        swapped["route_structure"]["score"]
        == forward["route_structure"]["score"]
    )
    assert (
        swapped["cover_diversity"]["score"]
        == forward["cover_diversity"]["score"]
    )
    assert (
        swapped["spawn_balance"]["score"]
        == forward["spawn_balance"]["score"]
    )


def test_spawn_in_disconnected_component_fails_explicitly(
    benchmark_config: CounterStrikeBenchmarkConfig,
    case_contract_factory: Callable[..., CounterStrikeCaseContract],
) -> None:
    raw = deepcopy(benchmark_config.raw)
    raw["static_world"]["spawn_snap_radius_m"] = 0.25
    strict_config = CounterStrikeBenchmarkConfig(
        path=benchmark_config.path,
        sha256=benchmark_config.sha256,
        raw=raw,
    )
    full_height_splitter = _object(
        "disconnecting_wall",
        center=(8.0, 6.0, 1.5),
        size=(0.4, 12.0, 3.0),
    )

    with pytest.raises(CounterStrikeTopologyError) as caught:
        _scores(
            _arena(obstacles=(full_height_splitter,)),
            contract=case_contract_factory((2.0, 6.0), (20.0, 6.0)),
            config=strict_config,
        )

    assert caught.value.code in {"spawn_not_walkable", "spawn_teams_disconnected"}


def test_cover_forms_respond_to_varied_geometry_and_ignore_floor_and_large_walls(
    benchmark_config: CounterStrikeBenchmarkConfig,
    case_contract_factory: Callable[..., CounterStrikeCaseContract],
) -> None:
    contract = case_contract_factory((2.0, 6.0), (22.0, 6.0))
    baseline_topology, baseline_metrics = _scores(
        _arena(),
        contract=contract,
        config=benchmark_config,
    )
    covers = (
        _object(
            "low_compact_cover",
            center=(6.0, 3.0, 0.30),
            size=(0.8, 0.8, 0.60),
        ),
        _object(
            "waist_block_cover",
            center=(9.0, 9.0, 0.75),
            size=(1.5, 1.4, 1.50),
        ),
        _object(
            "standing_linear_cover",
            center=(15.0, 3.0, 1.10),
            size=(3.5, 0.5, 2.20),
        ),
        _object(
            "waist_wide_cover",
            center=(18.0, 9.0, 0.70),
            size=(2.5, 1.5, 1.40),
        ),
    )
    varied_topology, varied_metrics = _scores(
        _arena(obstacles=covers),
        contract=contract,
        config=benchmark_config,
    )

    baseline_ids = {
        item["object_id"] for item in baseline_topology.cover_candidates
    }
    varied_ids = {
        item["object_id"] for item in varied_topology.cover_candidates
    }
    structural_non_cover_ids = {
        "oversized_floor",
        "wall_west",
        "wall_east",
        "wall_south",
        "wall_north",
    }

    assert baseline_ids.isdisjoint(structural_non_cover_ids)
    assert varied_ids.isdisjoint(structural_non_cover_ids)
    assert varied_ids == {item["id"] for item in covers}
    assert (
        varied_metrics["cover_diversity"]["form_count"]
        > baseline_metrics["cover_diversity"]["form_count"]
    )
    assert varied_metrics["cover_diversity"]["forms"] == [
        "low_narrow_block",
        "standing_wide_linear",
        "waist_medium_block",
        "waist_wide_block",
    ]


def test_cover_mesh_fragmentation_does_not_create_extra_forms_or_assemblies(
    benchmark_config: CounterStrikeBenchmarkConfig,
    case_contract_factory: Callable[..., CounterStrikeCaseContract],
) -> None:
    contract = case_contract_factory((2.0, 6.0), (22.0, 6.0))
    whole = _object(
        "whole_cover",
        center=(10.0, 6.0, 0.75),
        size=(2.0, 2.0, 1.50),
    )
    fragments = (
        _object(
            "cover_left_fragment",
            center=(9.5, 6.0, 0.75),
            size=(1.0, 2.0, 1.50),
        ),
        _object(
            "cover_right_fragment",
            center=(10.5, 6.0, 0.75),
            size=(1.0, 2.0, 1.50),
        ),
    )

    whole_topology, whole_metrics = _scores(
        _arena(obstacles=(whole,)),
        contract=contract,
        config=benchmark_config,
    )
    split_topology, split_metrics = _scores(
        _arena(obstacles=fragments),
        contract=contract,
        config=benchmark_config,
    )

    assert len(whole_topology.cover_candidates) == 1
    assert len(split_topology.cover_candidates) == 1
    assert split_topology.cover_candidates[0]["member_count"] == 2
    assert (
        split_metrics["cover_diversity"]["forms"]
        == whole_metrics["cover_diversity"]["forms"]
    )
    assert (
        split_metrics["cover_diversity"]["components"]
        == whole_metrics["cover_diversity"]["components"]
    )
    assert (
        split_metrics["cover_diversity"]["score"]
        == whole_metrics["cover_diversity"]["score"]
    )


def test_cover_duplicate_check_does_not_confuse_concentric_different_scales() -> None:
    base = {
        "center_xy": [10.0, 10.0],
        "height_m": 2.0,
        "footprint_spans_m": [2.0, 2.0],
    }
    very_large = {
        "center_xy": [10.0, 10.0],
        "height_m": 2.0,
        "footprint_spans_m": [20.0, 20.0],
    }
    near_duplicate = {
        "center_xy": [10.1, 10.1],
        "height_m": 2.1,
        "footprint_spans_m": [2.1, 1.9],
    }

    assert (
        _duplicate_cover_assembly_count(
            [base, very_large],
            tolerance_m=0.50,
            scale_ratio_tolerance=0.20,
        )
        == 0
    )
    assert (
        _duplicate_cover_assembly_count(
            [base, near_duplicate],
            tolerance_m=0.50,
            scale_ratio_tolerance=0.20,
        )
        == 1
    )


def test_engagement_prefers_balanced_cell_with_strong_traffic_support() -> None:
    free = np.ones((3, 5), dtype=bool)
    traffic = np.zeros((3, 5), dtype=float)
    # The absolute maximum is substantially one-sided. A neighboring candidate
    # has 98% of peak traffic and equal travel distance, so it is the more
    # defensible static main-engagement proxy.
    traffic[1, 1] = 1.00
    traffic[1, 2] = 0.98
    distance_a = np.asarray(
        [
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    )
    distance_b = np.asarray(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [6.0, 5.0, 4.0, 3.0, 2.0],
        ]
    )

    engagement, region = _engagement_region(
        free,
        traffic,
        distance_a=distance_a,
        distance_b=distance_b,
        resolution=0.35,
        selection_objective="maximin_traffic_distance_balance",
    )

    assert engagement == (1, 2)
    assert region[engagement]


def test_engagement_does_not_hide_imbalance_at_unused_balanced_cell() -> None:
    free = np.ones((3, 5), dtype=bool)
    traffic = np.zeros((3, 5), dtype=float)
    traffic[1, 1] = 1.00
    # This perfectly balanced cell carries too little traffic to replace the
    # genuinely one-sided convergence.
    traffic[1, 2] = 0.35
    distance_a = np.asarray(
        [
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
        ]
    )
    distance_b = np.asarray(
        [
            [6.0, 5.0, 4.0, 3.0, 2.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [6.0, 5.0, 4.0, 3.0, 2.0],
        ]
    )

    engagement, _ = _engagement_region(
        free,
        traffic,
        distance_a=distance_a,
        distance_b=distance_b,
        resolution=0.35,
        selection_objective="maximin_traffic_distance_balance",
    )

    assert engagement == (1, 1)
