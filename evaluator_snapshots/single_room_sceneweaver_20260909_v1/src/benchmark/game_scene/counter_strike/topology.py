"""Deterministic static-topology analysis for Counter-Strike-like arenas.

The module intentionally evaluates only geometry that exists in the frozen
browser capture.  It does not simulate rounds, infer weapons, inspect bot AI,
or score player skill.  Spawn locations come from the benchmark-owned,
source-hash-verified case contract; all other evidence comes from the exported
canonical geometry.

The public entry point returns five pieces of evidence:

* a reusable single-floor walkability grid;
* inferred spatial-role masks used only as diagnostics for zone clarity;
* diverse source-to-source route candidates;
* spawn travel/exposure balance;
* geometry-derived cover-assembly proposals.

Zone clarity and cover diversity are deliberately deterministic
*components*. Their final scores also need the separately implemented visual
judgement. Route structure and spawn balance are fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import networkx as nx
from scipy import ndimage

from benchmark.evaluator.generic_validity.geometry import (
    get_room_boundary,
    get_footprint_corners_xy,
    normalize_objects,
    point_in_polygon_2d,
    point_segment_distance_2d,
    polygon_area,
    polygon_distance_2d,
)
from benchmark.evaluator.generic_validity.navigability import (
    compute_navigability_grid,
)

from .loader import CounterStrikeBenchmarkConfig, CounterStrikeCaseContract


COUNTER_STRIKE_TOPOLOGY_VERSION = "counter_strike_static_topology_v2"


class CounterStrikeTopologyError(RuntimeError):
    """Raised when trusted inputs cannot produce a meaningful topology."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"Counter-Strike topology failed [{code}]: {message}")


@dataclass(frozen=True)
class RouteCandidate:
    """One connected grid route between the two representative spawns."""

    route_id: str
    cells: tuple[tuple[int, int], ...]
    length_m: float
    detour_ratio: float
    maximum_overlap: float
    mean_separation_from_primary_m: float
    classification: str

    def to_dict(self, *, grid: "CounterStrikeTopology") -> dict[str, Any]:
        sampled = _sample_polyline(self.cells, maximum_points=80)
        return {
            "route_id": self.route_id,
            "classification": self.classification,
            "length_m": self.length_m,
            "detour_ratio": self.detour_ratio,
            "maximum_overlap": self.maximum_overlap,
            "mean_separation_from_primary_m": self.mean_separation_from_primary_m,
            "cell_count": len(self.cells),
            "polyline_world_xy": [
                list(grid.cell_to_world(cell)) for cell in sampled
            ],
        }


@dataclass
class CounterStrikeTopology:
    """In-memory topology plus arrays needed by metric and visualization code."""

    version: str
    grid: dict[str, Any]
    free: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    resolution: float
    team_a_cells: tuple[tuple[int, int], ...]
    team_b_cells: tuple[tuple[int, int], ...]
    team_a_representative: tuple[int, int]
    team_b_representative: tuple[int, int]
    distance_a: np.ndarray
    distance_b: np.ndarray
    traffic: np.ndarray
    main_engagement: np.ndarray
    team_a_spawn_zone: np.ndarray
    team_b_spawn_zone: np.ndarray
    team_a_preparation: np.ndarray
    team_b_preparation: np.ndarray
    flank_region: np.ndarray
    engagement_cell: tuple[int, int]
    primary_path: tuple[tuple[int, int], ...]
    routes: tuple[RouteCandidate, ...]
    cover_candidates: tuple[dict[str, Any], ...]

    def cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        return float(self.x_centers[col]), float(self.y_centers[row])

    def world_to_cell(self, point_xy: Iterable[float]) -> tuple[int, int]:
        point = tuple(float(value) for value in point_xy)
        col = int(np.argmin(np.abs(self.x_centers - point[0])))
        row = int(np.argmin(np.abs(self.y_centers - point[1])))
        return row, col

    def summary(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "grid": {
                "shape": list(self.free.shape),
                "resolution_m": self.resolution,
                "free_cell_count": int(np.sum(self.free)),
                "largest_component_cells": int(
                    self.grid["largest_component_cells"]
                ),
                "total_free_cells": int(self.grid["total_free_cells"]),
                "boundary_source": self.grid["boundary_source"],
            },
            "team_a_spawn_cells": [list(cell) for cell in self.team_a_cells],
            "team_b_spawn_cells": [list(cell) for cell in self.team_b_cells],
            "team_a_spawn_world_xy": [
                list(self.cell_to_world(cell)) for cell in self.team_a_cells
            ],
            "team_b_spawn_world_xy": [
                list(self.cell_to_world(cell)) for cell in self.team_b_cells
            ],
            "engagement_cell": list(self.engagement_cell),
            "engagement_world_xy": list(self.cell_to_world(self.engagement_cell)),
            "zone_cell_counts": {
                "team_a_spawn": int(np.sum(self.team_a_spawn_zone)),
                "team_b_spawn": int(np.sum(self.team_b_spawn_zone)),
                "team_a_preparation": int(np.sum(self.team_a_preparation)),
                "team_b_preparation": int(np.sum(self.team_b_preparation)),
                "main_engagement": int(np.sum(self.main_engagement)),
                "flank": int(np.sum(self.flank_region)),
            },
            "routes": [route.to_dict(grid=self) for route in self.routes],
            "cover_candidates": list(self.cover_candidates),
        }


def analyze_counter_strike_static_geometry(
    scene: dict[str, Any],
    *,
    case_contract: CounterStrikeCaseContract,
    benchmark_config: CounterStrikeBenchmarkConfig,
) -> tuple[CounterStrikeTopology, dict[str, dict[str, Any]]]:
    """Build one topology and return its four deterministic L4 results.

    The returned ``zone_clarity`` result is explicitly named
    ``deterministic_component``.  The caller must not publish it as the final
    hybrid metric without the perceptual component.
    """

    cfg = benchmark_config.raw
    topology = _build_topology(
        scene,
        case_contract=case_contract,
        config=cfg,
    )
    metrics = {
        "zone_clarity": _zone_clarity_component(topology),
        "route_structure": _route_structure(topology, cfg),
        "spawn_balance": _spawn_balance(topology, cfg),
        "cover_diversity": _cover_diversity(topology, cfg),
    }
    return topology, metrics


def _build_topology(
    scene: dict[str, Any],
    *,
    case_contract: CounterStrikeCaseContract,
    config: dict[str, Any],
) -> CounterStrikeTopology:
    static_cfg = config["static_world"]
    player_cfg = config["player_profile"]
    grid = compute_navigability_grid(
        scene,
        {
            "boundary_source": static_cfg["boundary_source"],
            "grid_resolution": static_cfg["grid_resolution_m"],
            "max_grid_cells": static_cfg["max_grid_cells"],
            "agent_radius": player_cfg["agent_radius_m"],
            "clearance_height": player_cfg["standing_height_m"],
            "step_over_height": player_cfg["step_over_height_m"],
            "connectivity": static_cfg["connectivity"],
        },
    )
    if grid.get("status") != "checked":
        raise CounterStrikeTopologyError(
            "navigability_unavailable",
            str(grid.get("reason") or "walkability grid was not checked"),
        )
    largest_label = int(grid["largest_component_id"])
    if largest_label < 0:
        raise CounterStrikeTopologyError(
            "no_walkable_component",
            "the exported static environment has no walkable component",
        )
    free = np.asarray(grid["free"], dtype=bool) & (
        np.asarray(grid["component_labels"], dtype=int) == largest_label
    )
    xs = np.asarray(grid["x_centers"], dtype=float)
    ys = np.asarray(grid["y_centers"], dtype=float)
    resolution = float(grid["grid_resolution"])
    snap_radius = float(static_cfg["spawn_snap_radius_m"])
    canonical_spawns = case_contract.canonical_team_spawns
    team_a = _snap_team_spawns(
        canonical_spawns["team_a"]["points"],
        free=free,
        xs=xs,
        ys=ys,
        maximum_distance_m=snap_radius
        + float(canonical_spawns["team_a"]["jitter_radius_m"]),
        team="team_a",
    )
    team_b = _snap_team_spawns(
        canonical_spawns["team_b"]["points"],
        free=free,
        xs=xs,
        ys=ys,
        maximum_distance_m=snap_radius
        + float(canonical_spawns["team_b"]["jitter_radius_m"]),
        team="team_b",
    )
    representative_a = _representative_cell(team_a, xs=xs, ys=ys)
    representative_b = _representative_cell(team_b, xs=xs, ys=ys)
    if representative_a == representative_b:
        raise CounterStrikeTopologyError(
            "spawn_teams_collapsed",
            "both teams snap to the same walkable cell",
        )

    connectivity = int(static_cfg["connectivity"])
    distance_a = _distance_map(free, team_a, resolution, connectivity)
    distance_b = _distance_map(free, team_b, resolution, connectivity)
    if not math.isfinite(float(distance_a[representative_b])):
        raise CounterStrikeTopologyError(
            "spawn_teams_disconnected",
            "team spawn regions are not connected through walkable space",
        )

    traffic = _spawn_pair_traffic(
        free,
        team_a,
        team_b,
        resolution=resolution,
        connectivity=connectivity,
    )
    engagement_cell, main_engagement = _engagement_region(
        free,
        traffic,
        distance_a=distance_a,
        distance_b=distance_b,
        resolution=resolution,
        selection_objective=str(
            config["l4_metrics"]["spawn_balance"][
                "engagement_selection"
            ]
        ),
    )
    # Team names are semantic labels, not geometry.  Use a canonical endpoint
    # order anywhere an otherwise equivalent shortest-path tie could choose a
    # different grid polyline.  This keeps route topology and downstream
    # engagement evidence invariant when the two team declarations are
    # exchanged.
    route_start, route_goal = sorted((representative_a, representative_b))
    primary_path = _weighted_shortest_path(
        free,
        route_start,
        route_goal,
        resolution=resolution,
        connectivity=connectivity,
    )
    if not primary_path:
        raise CounterStrikeTopologyError(
            "representative_route_missing",
            "representative team spawn cells have no connected route",
        )
    routes = _diverse_routes(
        free,
        route_start,
        route_goal,
        config=config["l4_metrics"]["route_structure"],
        resolution=resolution,
        connectivity=connectivity,
    )

    distance_to_primary = ndimage.distance_transform_edt(
        ~_path_mask(free.shape, primary_path)
    ) * resolution
    shortest_connection = _path_length(primary_path, resolution)
    spawn_cut = max(2.0, 0.12 * shortest_connection)
    prep_end = max(spawn_cut + 2.0, 0.30 * shortest_connection)
    team_a_spawn_zone = free & (distance_a <= spawn_cut) & (distance_a < distance_b)
    team_b_spawn_zone = free & (distance_b <= spawn_cut) & (distance_b < distance_a)
    team_a_preparation = (
        free
        & (distance_a > spawn_cut)
        & (distance_a <= prep_end)
        & (distance_a < distance_b)
    )
    team_b_preparation = (
        free
        & (distance_b > spawn_cut)
        & (distance_b <= prep_end)
        & (distance_b < distance_a)
    )
    travel_envelope = np.isfinite(distance_a) & np.isfinite(distance_b)
    travel_envelope &= (distance_a + distance_b) <= max(
        1.7 * shortest_connection,
        shortest_connection + 8.0,
    )
    flank_region = (
        free
        & travel_envelope
        & (distance_to_primary >= float(config["l4_metrics"]["route_structure"]["route_separation_m"]))
        & ~team_a_spawn_zone
        & ~team_b_spawn_zone
        & ~main_engagement
    )
    covers = _cover_candidates(
        scene,
        free=free,
        xs=xs,
        ys=ys,
        resolution=resolution,
        config=config["l4_metrics"]["cover_diversity"],
        player_height=float(player_cfg["standing_height_m"]),
        step_height=float(player_cfg["step_over_height_m"]),
        perimeter_band_m=float(static_cfg["perimeter_band_m"]),
    )
    return CounterStrikeTopology(
        version=COUNTER_STRIKE_TOPOLOGY_VERSION,
        grid=grid,
        free=free,
        x_centers=xs,
        y_centers=ys,
        resolution=resolution,
        team_a_cells=team_a,
        team_b_cells=team_b,
        team_a_representative=representative_a,
        team_b_representative=representative_b,
        distance_a=distance_a,
        distance_b=distance_b,
        traffic=traffic,
        main_engagement=main_engagement,
        team_a_spawn_zone=team_a_spawn_zone,
        team_b_spawn_zone=team_b_spawn_zone,
        team_a_preparation=team_a_preparation,
        team_b_preparation=team_b_preparation,
        flank_region=flank_region,
        engagement_cell=engagement_cell,
        primary_path=tuple(primary_path),
        routes=tuple(routes),
        cover_candidates=tuple(covers),
    )


def _snap_team_spawns(
    points: Iterable[Iterable[float]],
    *,
    free: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    maximum_distance_m: float,
    team: str,
) -> tuple[tuple[int, int], ...]:
    free_rows, free_cols = np.nonzero(free)
    if not len(free_rows):
        raise CounterStrikeTopologyError(
            "no_walkable_component",
            "cannot snap spawns without walkable cells",
        )
    free_xy = np.column_stack((xs[free_cols], ys[free_rows]))
    snapped: list[tuple[int, int]] = []
    for index, point in enumerate(points):
        values = list(point)
        if len(values) != 3:
            raise CounterStrikeTopologyError(
                "spawn_metadata_invalid",
                f"{team} spawn {index} is not a 3-vector",
            )
        xy = np.asarray(values[:2], dtype=float)
        distances = np.linalg.norm(free_xy - xy[None, :], axis=1)
        best = int(np.argmin(distances))
        if float(distances[best]) > maximum_distance_m:
            raise CounterStrikeTopologyError(
                "spawn_not_walkable",
                f"{team} spawn {index} is {distances[best]:.3f}m from the "
                f"nearest main-component cell (limit {maximum_distance_m:.3f}m)",
            )
        cell = int(free_rows[best]), int(free_cols[best])
        if cell not in snapped:
            snapped.append(cell)
    if not snapped:
        raise CounterStrikeTopologyError(
            "spawn_metadata_invalid",
            f"{team} has no distinct walkable spawn cells",
        )
    return tuple(snapped)


def _representative_cell(
    cells: Iterable[tuple[int, int]],
    *,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[int, int]:
    entries = list(cells)
    points = np.asarray([[xs[col], ys[row]] for row, col in entries], dtype=float)
    center = np.mean(points, axis=0)
    index = int(np.argmin(np.linalg.norm(points - center[None, :], axis=1)))
    return entries[index]


def _neighbors(
    cell: tuple[int, int],
    *,
    free: np.ndarray,
    connectivity: int,
) -> Iterable[tuple[tuple[int, int], float]]:
    row, col = cell
    cardinal = ((-1, 0), (1, 0), (0, -1), (0, 1))
    offsets = list(cardinal)
    if connectivity == 8:
        offsets.extend(((-1, -1), (-1, 1), (1, -1), (1, 1)))
    rows, cols = free.shape
    for dr, dc in offsets:
        nr, nc = row + dr, col + dc
        if nr < 0 or nr >= rows or nc < 0 or nc >= cols or not free[nr, nc]:
            continue
        if dr and dc:
            # Prevent a point-sized diagonal connection through two touching
            # obstacle corners; the configured player cylinder cannot use it.
            if not free[row + dr, col] or not free[row, col + dc]:
                continue
            step = math.sqrt(2.0)
        else:
            step = 1.0
        yield (nr, nc), step


def _distance_map(
    free: np.ndarray,
    sources: Iterable[tuple[int, int]],
    resolution: float,
    connectivity: int,
) -> np.ndarray:
    distances = np.full(free.shape, np.inf, dtype=float)
    queue: list[tuple[float, int, int]] = []
    for row, col in sources:
        distances[row, col] = 0.0
        heapq.heappush(queue, (0.0, row, col))
    while queue:
        distance, row, col = heapq.heappop(queue)
        if distance > distances[row, col] + 1.0e-12:
            continue
        for (nr, nc), step in _neighbors(
            (row, col),
            free=free,
            connectivity=connectivity,
        ):
            candidate = distance + step * resolution
            if candidate + 1.0e-12 < distances[nr, nc]:
                distances[nr, nc] = candidate
                heapq.heappush(queue, (candidate, nr, nc))
    return distances


def _weighted_shortest_path(
    free: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    resolution: float,
    connectivity: int,
    cell_cost: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    costs = (
        np.ones(free.shape, dtype=float)
        if cell_cost is None
        else np.asarray(cell_cost, dtype=float)
    )
    distance = np.full(free.shape, np.inf, dtype=float)
    predecessor_row = np.full(free.shape, -1, dtype=np.int32)
    predecessor_col = np.full(free.shape, -1, dtype=np.int32)
    distance[start] = 0.0
    queue: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]
    while queue:
        current, row, col = heapq.heappop(queue)
        if current > distance[row, col] + 1.0e-12:
            continue
        if (row, col) == goal:
            break
        for (nr, nc), step in _neighbors(
            (row, col),
            free=free,
            connectivity=connectivity,
        ):
            edge_cost = (
                0.5 * (float(costs[row, col]) + float(costs[nr, nc]))
                * step
                * resolution
            )
            candidate = current + edge_cost
            if candidate + 1.0e-12 < distance[nr, nc]:
                distance[nr, nc] = candidate
                predecessor_row[nr, nc] = row
                predecessor_col[nr, nc] = col
                heapq.heappush(queue, (candidate, nr, nc))
    if not math.isfinite(float(distance[goal])):
        return []
    path = [goal]
    current = goal
    while current != start:
        row = int(predecessor_row[current])
        col = int(predecessor_col[current])
        if row < 0 or col < 0:
            return []
        current = row, col
        path.append(current)
    path.reverse()
    return path


def _spawn_pair_traffic(
    free: np.ndarray,
    team_a: Iterable[tuple[int, int]],
    team_b: Iterable[tuple[int, int]],
    *,
    resolution: float,
    connectivity: int,
) -> np.ndarray:
    traffic = np.zeros(free.shape, dtype=float)
    pair_count = 0
    for source in team_a:
        for target in team_b:
            # The grid graph is undirected, but Dijkstra's deterministic
            # tie-breaking can select a different equally short polyline when
            # endpoints are reversed.  Traffic describes map geometry rather
            # than team identity, so canonicalize every unordered spawn pair.
            route_start, route_goal = sorted((source, target))
            path = _weighted_shortest_path(
                free,
                route_start,
                route_goal,
                resolution=resolution,
                connectivity=connectivity,
            )
            if not path:
                continue
            pair_count += 1
            start = int(0.22 * len(path))
            stop = max(start + 1, int(0.78 * len(path)))
            for row, col in path[start:stop]:
                traffic[row, col] += 1.0
    if pair_count <= 0:
        raise CounterStrikeTopologyError(
            "spawn_teams_disconnected",
            "no source-to-source spawn pair has a walkable route",
        )
    sigma_cells = max(1.0, 1.25 / resolution)
    return ndimage.gaussian_filter(traffic, sigma=sigma_cells) * free


def _engagement_region(
    free: np.ndarray,
    traffic: np.ndarray,
    *,
    distance_a: np.ndarray,
    distance_b: np.ndarray,
    resolution: float,
    selection_objective: str,
) -> tuple[tuple[int, int], np.ndarray]:
    finite = free & np.isfinite(distance_a) & np.isfinite(distance_b)
    if not np.any(finite) or float(np.max(traffic)) <= 0.0:
        raise CounterStrikeTopologyError(
            "engagement_region_unavailable",
            "spawn-pair route traffic did not define an engagement region",
        )
    if selection_objective != "maximin_traffic_distance_balance":
        raise CounterStrikeTopologyError(
            "engagement_selection_invalid",
            f"unsupported engagement selection {selection_objective!r}",
        )
    maximum = float(np.max(traffic))
    normalized_traffic = np.zeros(traffic.shape, dtype=float)
    normalized_traffic[finite] = traffic[finite] / maximum
    farther_distance = np.maximum(distance_a, distance_b)
    distance_balance = np.zeros(distance_a.shape, dtype=float)
    nonzero = finite & (farther_distance > 1.0e-12)
    distance_balance[nonzero] = (
        np.minimum(distance_a[nonzero], distance_b[nonzero])
        / farther_distance[nonzero]
    )

    # A hard near-peak traffic band is unstable when several equivalent
    # shortest routes exist: deterministic Dijkstra tie-breaking can put the
    # absolute traffic maximum on one side even though a strongly trafficked
    # central region is equally contestable.  The maximin objective keeps both
    # requirements explicit.  A balanced but unused cell cannot win, and an
    # arbitrary one-sided peak cannot erase a high-traffic balanced frontier.
    support = np.minimum(normalized_traffic, distance_balance)
    best_support = float(np.max(support[finite]))
    candidates = finite & (
        support >= best_support - max(1.0e-12, resolution * 1.0e-12)
    )
    # Resolve exact maximin ties by joint support and then by traffic.  The
    # final np.argmax is a stable row-major fallback only for exact geometric
    # equivalence.
    joint_support = normalized_traffic * distance_balance
    best_joint = float(np.max(joint_support[candidates]))
    candidates &= (
        joint_support >= best_joint - max(1.0e-12, resolution * 1.0e-12)
    )
    objective = np.where(candidates, normalized_traffic, -np.inf)
    flat = int(np.argmax(objective))
    engagement = tuple(int(value) for value in np.unravel_index(flat, free.shape))
    threshold = max(0.42 * maximum, float(traffic[engagement]) * 0.55)
    region = finite & (traffic >= threshold)
    # Keep the component containing the selected hotspot so unrelated blurred
    # traffic islands do not become one semantic zone.
    labels, _ = ndimage.label(region, structure=np.ones((3, 3), dtype=int))
    label = int(labels[engagement])
    if label > 0:
        region = labels == label
    if int(np.sum(region)) < 3:
        radius_cells = max(1, int(round(1.5 / resolution)))
        seed = np.zeros(free.shape, dtype=bool)
        seed[engagement] = True
        region = ndimage.binary_dilation(seed, iterations=radius_cells) & finite
    return engagement, region


def _diverse_routes(
    free: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    config: dict[str, Any],
    resolution: float,
    connectivity: int,
) -> list[RouteCandidate]:
    """Extract corridor routes from a thinned free-space skeleton.

    Earlier prototypes repeatedly penalized a normal grid shortest path.  That
    can manufacture many visually different polylines inside one undivided open
    hall, which is not the checklist's notion of multiple routes with distinct
    entrances/exits.  A topology-preserving skeleton instead exposes only
    branches and loops created by the authored obstacle layout.  Route count is
    therefore tied to static map structure rather than an optimizer's ability
    to draw parallel curves.
    """

    budget = int(config["candidate_path_budget"])
    maximum_overlap = float(config["max_route_overlap"])
    separation_m = float(config["route_separation_m"])
    skeleton = _zhang_suen_skeleton(free)
    graph = _skeleton_graph(skeleton, resolution=resolution)
    if not graph:
        return []
    start_skeleton = _nearest_active_cell(skeleton, start)
    goal_skeleton = _nearest_active_cell(skeleton, goal)
    if (
        start_skeleton is None
        or goal_skeleton is None
        or start_skeleton not in graph
        or goal_skeleton not in graph
        or not nx.has_path(graph, start_skeleton, goal_skeleton)
    ):
        return []
    start_connector = _weighted_shortest_path(
        free,
        start,
        start_skeleton,
        resolution=resolution,
        connectivity=connectivity,
    )
    goal_connector = _weighted_shortest_path(
        free,
        goal_skeleton,
        goal,
        resolution=resolution,
        connectivity=connectivity,
    )
    if not start_connector or not goal_connector:
        return []

    accepted_buffers: list[np.ndarray] = []
    primary_distance: np.ndarray | None = None
    raw: list[dict[str, Any]] = []
    try:
        path_iterator = nx.shortest_simple_paths(
            graph,
            start_skeleton,
            goal_skeleton,
            weight="weight",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    maximum_generated = max(100, budget * 25)
    for generated_index, skeleton_path in enumerate(path_iterator):
        if generated_index >= maximum_generated:
            break
        path = (
            start_connector[:-1]
            + [tuple(cell) for cell in skeleton_path]
            + goal_connector[1:]
        )
        path = _deduplicate_adjacent_cells(path)
        if not path:
            continue
        mask = _path_mask(free.shape, path)
        buffer_iterations = max(1, int(round(0.5 * separation_m / resolution)))
        buffered = ndimage.binary_dilation(
            mask,
            iterations=buffer_iterations,
        )
        overlaps = [
            _jaccard(buffered, previous) for previous in accepted_buffers
        ]
        overlap = max(overlaps, default=0.0)
        if primary_distance is None:
            primary_distance = (
                ndimage.distance_transform_edt(~mask) * resolution
            )
        mean_separation = float(
            np.mean([primary_distance[cell] for cell in path])
        )
        if not raw or overlap <= maximum_overlap:
            accepted_buffers.append(buffered)
            raw.append(
                {
                    "path": path,
                    "length_m": _path_length(path, resolution),
                    "maximum_overlap": overlap,
                    "mean_separation_m": mean_separation,
                }
            )
            if len(raw) >= budget:
                break

    if not raw:
        return []
    shortest = min(float(item["length_m"]) for item in raw)
    max_main = float(config["max_main_detour_ratio"])
    min_flank = float(config["min_flank_detour_ratio"])
    max_flank = float(config["max_flank_detour_ratio"])
    required_main = int(config["required_main_routes"])
    main_assigned = 0
    routes: list[RouteCandidate] = []
    for index, item in enumerate(sorted(raw, key=lambda value: value["length_m"])):
        ratio = float(item["length_m"]) / max(shortest, 1.0e-9)
        if ratio <= max_main and main_assigned < required_main:
            classification = "main"
            main_assigned += 1
        elif (
            min_flank <= ratio <= max_flank
            and float(item["mean_separation_m"]) >= separation_m
        ):
            classification = "flank"
        else:
            classification = "alternate"
        routes.append(
            RouteCandidate(
                route_id=f"route_{index:02d}",
                cells=tuple(item["path"]),
                length_m=float(item["length_m"]),
                detour_ratio=ratio,
                maximum_overlap=float(item["maximum_overlap"]),
                mean_separation_from_primary_m=float(item["mean_separation_m"]),
                classification=classification,
            )
        )
    return routes


def _zhang_suen_skeleton(mask: np.ndarray) -> np.ndarray:
    """Topology-preserving binary thinning with deterministic iteration order."""

    image = np.asarray(mask, dtype=bool).copy()
    # Boundary pixels are kept away from the padded-array edge; the arena's
    # blocking perimeter normally ensures this already, but the explicit pad
    # makes the operation well-defined for synthetic tests too.
    for _ in range(max(image.shape) * 2):
        changed = False
        for phase in (0, 1):
            padded = np.pad(image, 1, mode="constant", constant_values=False)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            count = sum(item.astype(np.uint8) for item in neighbors)
            transitions = sum(
                ((~neighbors[index]) & neighbors[(index + 1) % 8]).astype(
                    np.uint8
                )
                for index in range(8)
            )
            if phase == 0:
                preserve_a = ~(p2 & p4 & p6)
                preserve_b = ~(p4 & p6 & p8)
            else:
                preserve_a = ~(p2 & p4 & p8)
                preserve_b = ~(p2 & p6 & p8)
            remove = (
                image
                & (count >= 2)
                & (count <= 6)
                & (transitions == 1)
                & preserve_a
                & preserve_b
            )
            if np.any(remove):
                image[remove] = False
                changed = True
        if not changed:
            break
    return image


def _skeleton_graph(
    skeleton: np.ndarray,
    *,
    resolution: float,
) -> nx.Graph:
    graph = nx.Graph()
    rows, cols = np.nonzero(skeleton)
    for row, col in zip(rows.tolist(), cols.tolist()):
        graph.add_node((row, col))
    for row, col in list(graph.nodes):
        for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
            neighbor = row + dr, col + dc
            if neighbor not in graph:
                continue
            if dr and dc and (
                (row + dr, col) in graph or (row, col + dc) in graph
            ):
                # A diagonal beside an existing cardinal connection is only a
                # pixel-grid shortcut around the same bend.  Keeping it makes
                # every staircase corner a triangular graph cycle, so
                # shortest_simple_paths enumerates many near-identical
                # polylines before reaching a real obstacle-induced branch.
                # Preserve diagonal-only connections; remove only redundant
                # diagonals whose endpoints already share a cardinal bridge.
                continue
            graph.add_edge(
                (row, col),
                neighbor,
                weight=math.hypot(dr, dc) * resolution,
            )
    return graph


def _nearest_active_cell(
    mask: np.ndarray,
    source: tuple[int, int],
) -> tuple[int, int] | None:
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    delta_row = rows.astype(float) - source[0]
    delta_col = cols.astype(float) - source[1]
    index = int(np.argmin(delta_row**2 + delta_col**2))
    return int(rows[index]), int(cols[index])


def _deduplicate_adjacent_cells(
    path: Iterable[tuple[int, int]],
) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for cell in path:
        normalized = int(cell[0]), int(cell[1])
        if not result or normalized != result[-1]:
            result.append(normalized)
    return result


def _zone_clarity_component(
    topology: CounterStrikeTopology,
) -> dict[str, Any]:
    free_count = max(int(np.sum(topology.free)), 1)
    spawn_overlap = bool(
        np.any(topology.team_a_spawn_zone & topology.team_b_spawn_zone)
    )
    prep_a = int(np.sum(topology.team_a_preparation))
    prep_b = int(np.sum(topology.team_b_preparation))
    main_fraction = float(np.sum(topology.main_engagement)) / free_count
    flank_fraction = float(np.sum(topology.flank_region)) / free_count
    # Every check here asks whether a required role exists as its own region.
    # Whether a spawn is exposed to the opposing one at spawn time is a
    # different question -- it is about fairness, not about role legibility --
    # and `spawn_balance.initial_cross_spawn_los_invalid` already decides it.
    # Scoring it here as well charged one topology fact to two metrics.
    components = {
        "spawn_regions_separate": 0.0 if spawn_overlap else 1.0,
        "preparation_regions_present": min(
            1.0,
            min(prep_a, prep_b) / max(8.0, 0.01 * free_count),
        ),
        "main_engagement_region_present": _band_score(
            main_fraction,
            lower=0.003,
            preferred_lower=0.01,
            preferred_upper=0.15,
            upper=0.35,
        ),
        "flank_region_present": min(
            1.0,
            flank_fraction / 0.08,
        ),
    }
    score = float(np.mean(list(components.values())))
    return {
        "metric": "zone_clarity",
        "status": "checked_deterministic_component",
        "score": score,
        "score_definition": "mean of four frozen static-role separability checks",
        "components": components,
        "zone_cell_counts": topology.summary()["zone_cell_counts"],
        "requires_perceptual_component": True,
    }


def _route_structure(
    topology: CounterStrikeTopology,
    config: dict[str, Any],
) -> dict[str, Any]:
    metric_cfg = config["l4_metrics"]["route_structure"]
    main = [route for route in topology.routes if route.classification == "main"]
    flank = [route for route in topology.routes if route.classification == "flank"]
    required_main = int(metric_cfg["required_main_routes"])
    required_flank = int(metric_cfg["required_flank_routes"])
    main_coverage = min(1.0, len(main) / max(required_main, 1))
    flank_coverage = (
        1.0
        if required_flank <= 0
        else min(1.0, len(flank) / required_flank)
    )
    score = 0.5 * main_coverage + 0.5 * flank_coverage
    return {
        "metric": "route_structure",
        "status": "checked",
        "score": float(score),
        "verdict": (
            "valid"
            if len(main) >= required_main and len(flank) >= required_flank
            else "invalid"
        ),
        "score_definition": (
            "0.5 * required-main-route coverage + "
            "0.5 * required-flank-route coverage"
        ),
        "main_route_count": len(main),
        "flank_route_count": len(flank),
        "alternate_route_count": sum(
            route.classification == "alternate" for route in topology.routes
        ),
        "required_main_routes": required_main,
        "required_flank_routes": required_flank,
        "candidate_count": len(topology.routes),
        "routes": [route.to_dict(grid=topology) for route in topology.routes],
        "limitation": (
            "static grid corridor diversity; no claim about dynamic round flow"
        ),
    }


def _spawn_balance(
    topology: CounterStrikeTopology,
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = config["l4_metrics"]["spawn_balance"]
    target = topology.engagement_cell
    target_distance = _distance_map(
        topology.free,
        [target],
        topology.resolution,
        int(config["static_world"]["connectivity"]),
    )
    distances_a = [float(target_distance[cell]) for cell in topology.team_a_cells]
    distances_b = [float(target_distance[cell]) for cell in topology.team_b_cells]
    travel_a = float(target_distance[topology.team_a_representative])
    travel_b = float(target_distance[topology.team_b_representative])
    median_a = float(np.median(distances_a))
    median_b = float(np.median(distances_b))
    path_a = _weighted_shortest_path(
        topology.free,
        topology.team_a_representative,
        target,
        resolution=topology.resolution,
        connectivity=int(config["static_world"]["connectivity"]),
    )
    path_b = _weighted_shortest_path(
        topology.free,
        topology.team_b_representative,
        target,
        resolution=topology.resolution,
        connectivity=int(config["static_world"]["connectivity"]),
    )
    exposure_a = _path_exposure(
        topology.free,
        path_a,
        resolution=topology.resolution,
        ray_count=int(cfg["exposure_ray_count"]),
        ray_length_m=float(cfg["exposure_ray_length_m"]),
    )
    exposure_b = _path_exposure(
        topology.free,
        path_b,
        resolution=topology.resolution,
        ray_count=int(cfg["exposure_ray_count"]),
        ray_length_m=float(cfg["exposure_ray_length_m"]),
    )
    travel_score, travel_relative = _ratio_balance_score(
        travel_a,
        travel_b,
        float(cfg["travel_ratio_tolerance"]),
    )
    exposure_score, exposure_relative = _ratio_balance_score(
        exposure_a,
        exposure_b,
        float(cfg["exposure_ratio_tolerance"]),
    )
    contact_score, contact_relative = _ratio_balance_score(
        median_a,
        median_b,
        float(cfg["contact_distance_ratio_tolerance"]),
    )
    initial_los = _any_cross_team_los(
        topology.free,
        topology.team_a_cells,
        topology.team_b_cells,
    )
    components = {
        "representative_travel_balance": travel_score,
        "static_exposure_balance": exposure_score,
        "all_spawn_contact_distance_balance": contact_score,
        "initial_cross_spawn_los": (
            0.0
            if initial_los and cfg["initial_cross_spawn_los_invalid"]
            else 1.0
        ),
    }
    score = float(np.mean(list(components.values())))
    return {
        "metric": "spawn_balance",
        "status": "checked",
        "score": score,
        "verdict": "valid" if score >= 0.60 else "invalid",
        "score_definition": (
            "mean of representative travel, route exposure, all-spawn "
            "engagement distance, and initial-LOS checks"
        ),
        "components": components,
        "measurements": {
            "representative_travel_m": {
                "team_a": travel_a,
                "team_b": travel_b,
                "relative_difference": travel_relative,
            },
            "path_exposure": {
                "team_a": exposure_a,
                "team_b": exposure_b,
                "relative_difference": exposure_relative,
            },
            "median_spawn_to_engagement_m": {
                "team_a": median_a,
                "team_b": median_b,
                "relative_difference": contact_relative,
            },
            "initial_cross_spawn_line_of_sight": initial_los,
        },
        "engagement_world_xy": list(topology.cell_to_world(target)),
        "limitation": (
            "static path/openness potential only; weapons, timing, and player "
            "behaviour are out of scope"
        ),
    }


def _cover_diversity(
    topology: CounterStrikeTopology,
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = config["l4_metrics"]["cover_diversity"]
    candidates = list(topology.cover_candidates)
    forms = sorted({str(item["form"]) for item in candidates})
    height_bands = sorted(
        {str(item["height_band"]) for item in candidates}
    )
    width_bands = sorted(
        {str(item["width_band"]) for item in candidates}
    )
    arrangements = sorted(
        {str(item["arrangement_type"]) for item in candidates}
    )
    required_assemblies = int(cfg["min_cover_assemblies"])
    required_forms = int(cfg["min_cover_forms"])
    required_heights = int(cfg["min_height_bands"])
    required_widths = int(cfg["min_width_bands"])
    required_arrangements = int(cfg["min_arrangement_types"])
    area_m2 = float(np.sum(topology.free)) * topology.resolution**2
    density = 100.0 * len(candidates) / max(area_m2, 1.0e-9)
    duplicate_count = _duplicate_cover_assembly_count(
        candidates,
        tolerance_m=float(cfg["duplicate_position_tolerance_m"]),
        scale_ratio_tolerance=float(
            cfg["duplicate_scale_ratio_tolerance"]
        ),
    )
    uniqueness = (
        0.0
        if not candidates
        else max(0.0, 1.0 - duplicate_count / len(candidates))
    )
    components = {
        "usable_assembly_coverage": min(
            1.0,
            len(candidates) / max(required_assemblies, 1),
        ),
        "form_signature_coverage": min(
            1.0,
            len(forms) / max(required_forms, 1),
        ),
        "height_variety": min(
            1.0,
            len(height_bands) / max(required_heights, 1),
        ),
        "width_variety": min(
            1.0,
            len(width_bands) / max(required_widths, 1),
        ),
        "arrangement_variety": min(
            1.0,
            len(arrangements) / max(required_arrangements, 1),
        ),
        "position_uniqueness": uniqueness,
    }
    component_weights = {
        "usable_assembly_coverage": 0.20,
        "form_signature_coverage": 0.35,
        "height_variety": 0.15,
        "width_variety": 0.15,
        "arrangement_variety": 0.10,
        "position_uniqueness": 0.05,
    }
    score = float(
        sum(
            components[name] * component_weights[name]
            for name in components
        )
    )
    passes_contract = (
        len(candidates) >= required_assemblies
        and len(forms) >= required_forms
        and len(height_bands) >= required_heights
        and len(width_bands) >= required_widths
        and len(arrangements) >= required_arrangements
        and duplicate_count == 0
    )
    return {
        "metric": "cover_diversity",
        "status": "checked_deterministic_component",
        "score": score,
        "verdict": "valid" if passes_contract else "invalid",
        "score_definition": (
            "frozen weighted coverage of usable assemblies, rotation-invariant "
            "form signatures, height/width bands, arrangement types, and "
            "position uniqueness"
        ),
        "components": components,
        "component_weights": component_weights,
        "cover_candidate_count": len(candidates),
        "cover_assembly_count": len(candidates),
        "form_count": len(forms),
        "forms": forms,
        "required_forms": required_forms,
        "height_band_count": len(height_bands),
        "height_bands": height_bands,
        "required_height_bands": required_heights,
        "width_band_count": len(width_bands),
        "width_bands": width_bands,
        "required_width_bands": required_widths,
        "arrangement_type_count": len(arrangements),
        "arrangement_types": arrangements,
        "required_arrangement_types": required_arrangements,
        "duplicate_placement_count": duplicate_count,
        "density_per_100m2": density,
        "density_affects_score": False,
        "candidates": candidates,
        "requires_perceptual_component": True,
        "limitation": (
            "geometry proposes usable cover assemblies; the perceptual "
            "component validates meaningful visible form and arrangement "
            "differences"
        ),
    }


def _cover_candidates(
    scene: dict[str, Any],
    *,
    free: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    resolution: float,
    config: dict[str, Any],
    player_height: float,
    step_height: float,
    perimeter_band_m: float,
) -> list[dict[str, Any]]:
    objects, _ = normalize_objects(scene)
    adjacency = float(config["adjacency_distance_m"])
    minimum_component_height = float(config["minimum_component_height_m"])
    maximum_base_offset = float(config["maximum_base_offset_m"])
    minimum_occlusion_height = float(config["minimum_occlusion_height_m"])
    raw: list[dict[str, Any]] = []
    for obj in objects:
        footprint = get_footprint_corners_xy(obj)
        height = float(obj.top_z - obj.bottom_z)
        if (
            obj.bottom_z > step_height + maximum_base_offset
            or obj.top_z <= step_height
            or height < minimum_component_height
            or polygon_area(footprint) < 0.015
        ):
            continue
        adjacent_free = _adjacent_free_cell_count(
            [footprint],
            free=free,
            xs=xs,
            ys=ys,
            distance_m=adjacency,
        )
        if adjacent_free < max(2, int(round(0.5 / max(resolution, 1.0e-9)))):
            continue
        raw.append(
            {
                "object": obj,
                "object_id": str(obj.id),
                "footprint": footprint,
                "bottom_z": float(obj.bottom_z),
                "top_z": float(obj.top_z),
                "height_m": height,
                "area_m2": polygon_area(footprint),
            }
        )

    groups = _cover_assembly_groups(raw, config=config)
    boundary = np.asarray(get_room_boundary(scene), dtype=float)
    results: list[dict[str, Any]] = []
    for group_index, member_indices in enumerate(groups):
        members = [raw[index] for index in member_indices]
        polygons = [item["footprint"] for item in members]
        points = np.concatenate(polygons, axis=0)
        center, spans = _rotation_invariant_footprint_span(points)
        major, minor = float(spans[0]), float(spans[1])
        bottom = min(float(item["bottom_z"]) for item in members)
        top = max(float(item["top_z"]) for item in members)
        effective_height = top - bottom
        if effective_height < minimum_occlusion_height:
            continue
        adjacent_free = _adjacent_free_cell_count(
            polygons,
            free=free,
            xs=xs,
            ys=ys,
            distance_m=adjacency,
        )
        if adjacent_free < max(
            2,
            int(round(0.5 / max(resolution, 1.0e-9))),
        ):
            continue
        if _is_perimeter_shell_assembly(
            points,
            major_span_m=major,
            boundary=boundary,
            perimeter_band_m=perimeter_band_m,
        ):
            continue
        member_heights = np.asarray(
            [float(item["height_m"]) for item in members],
            dtype=float,
        )
        member_tops = np.asarray(
            [float(item["top_z"]) for item in members],
            dtype=float,
        )
        shape = _cover_assembly_shape(
            points,
            member_footprints=polygons,
            member_tops=member_tops,
            major_span_m=major,
            minor_span_m=minor,
            config=config,
        )
        height_band = _cover_height_band(
            effective_height,
            player_height=player_height,
        )
        width_band = _cover_width_band(major, minor)
        results.append(
            {
                "assembly_id": f"cover_assembly_{group_index:03d}",
                # object_id is retained for older visualization consumers, but
                # is explicitly the stable representative of an assembly.
                "object_id": min(str(item["object_id"]) for item in members),
                "member_object_ids": sorted(
                    str(item["object_id"]) for item in members
                ),
                "member_count": len(members),
                "form": f"{height_band}_{width_band}_{shape}",
                "height_band": height_band,
                "width_band": width_band,
                "assembly_shape": shape,
                "height_m": effective_height,
                "member_height_range_m": (
                    float(np.max(member_heights) - np.min(member_heights))
                    if len(member_heights)
                    else 0.0
                ),
                "footprint_spans_m": [major, minor],
                "footprint_area_sum_m2": float(
                    sum(float(item["area_m2"]) for item in members)
                ),
                "center_xy": [float(center[0]), float(center[1])],
                "adjacent_free_cell_count": adjacent_free,
            }
        )
    _assign_cover_arrangements(
        results,
        neighbor_distance_m=float(config["arrangement_neighbor_distance_m"]),
    )
    return sorted(results, key=lambda item: item["assembly_id"])


def _cover_assembly_groups(
    raw: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> list[list[int]]:
    """Group touching mesh fragments without using scene-specific names."""

    graph = nx.Graph()
    graph.add_nodes_from(range(len(raw)))
    merge_distance = float(config["assembly_merge_distance_m"])
    vertical_gap = float(config["assembly_vertical_gap_m"])
    for first in range(len(raw)):
        for second in range(first + 1, len(raw)):
            a = raw[first]
            b = raw[second]
            if (
                polygon_distance_2d(a["footprint"], b["footprint"])
                > merge_distance
            ):
                continue
            z_gap = max(
                0.0,
                max(float(a["bottom_z"]), float(b["bottom_z"]))
                - min(float(a["top_z"]), float(b["top_z"])),
            )
            if z_gap <= vertical_gap:
                graph.add_edge(first, second)
    return [
        sorted(int(index) for index in component)
        for component in nx.connected_components(graph)
    ]


def _rotation_invariant_footprint_span(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    entries = np.asarray(points, dtype=float)
    center = np.mean(entries, axis=0)
    centered = entries - center
    if len(entries) <= 1 or float(np.max(np.abs(centered))) <= 1.0e-12:
        return center, np.asarray([0.0, 0.0], dtype=float)
    covariance = centered.T @ centered / max(len(entries), 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    projections = centered @ vectors[:, order]
    spans = np.ptp(projections, axis=0)
    spans = np.sort(np.asarray(spans, dtype=float))[::-1]
    return center, spans


def _cover_assembly_shape(
    points: np.ndarray,
    *,
    member_footprints: list[np.ndarray],
    member_tops: np.ndarray,
    major_span_m: float,
    minor_span_m: float,
    config: dict[str, Any],
) -> str:
    aspect = major_span_m / max(minor_span_m, 1.0e-9)
    if (
        len(member_footprints) > 1
        and float(np.ptp(member_tops)) >= float(config["stepped_height_delta_m"])
    ):
        return "stepped"
    hull = _convex_hull_2d(np.asarray(points, dtype=float))
    hull_area = polygon_area(hull)
    covered_area = sum(polygon_area(item) for item in member_footprints)
    fill_ratio = min(1.0, covered_area / max(hull_area, 1.0e-9))
    if len(member_footprints) > 1 and fill_ratio < 0.80:
        return "compound"
    if aspect >= 3.0:
        return "linear"
    if major_span_m >= 4.0:
        return "slab"
    return "block"


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Return a deterministic monotonic-chain hull for a small point set."""

    unique = sorted(
        {
            (float(point[0]), float(point[1]))
            for point in np.asarray(points, dtype=float)
        }
    )
    if len(unique) <= 2:
        return np.asarray(unique, dtype=float)

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _cover_height_band(height_m: float, *, player_height: float) -> str:
    if height_m < 0.60 * player_height:
        return "low"
    if height_m < 0.95 * player_height:
        return "waist"
    if height_m < 1.50 * player_height:
        return "standing"
    return "tall"


def _cover_width_band(major_span_m: float, minor_span_m: float) -> str:
    if major_span_m <= 1.25:
        return "narrow"
    if major_span_m <= 2.0 and major_span_m / max(minor_span_m, 1.0e-9) < 3.0:
        return "medium"
    return "wide"


def _adjacent_free_cell_count(
    polygons: list[np.ndarray],
    *,
    free: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    distance_m: float,
) -> int:
    points = np.concatenate(polygons, axis=0)
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    col_start = max(
        0,
        int(np.searchsorted(xs, low[0] - distance_m, side="left")),
    )
    col_stop = min(
        len(xs),
        int(np.searchsorted(xs, high[0] + distance_m, side="right")),
    )
    row_start = max(
        0,
        int(np.searchsorted(ys, low[1] - distance_m, side="left")),
    )
    row_stop = min(
        len(ys),
        int(np.searchsorted(ys, high[1] + distance_m, side="right")),
    )
    count = 0
    for row in range(row_start, row_stop):
        for col in range(col_start, col_stop):
            if not free[row, col]:
                continue
            point = np.asarray([xs[col], ys[row]], dtype=float)
            if min(
                _point_to_polygon_edge_or_interior_distance(point, polygon)
                for polygon in polygons
            ) <= distance_m:
                count += 1
    return count


def _point_to_polygon_edge_or_interior_distance(
    point: np.ndarray,
    polygon: np.ndarray,
) -> float:
    if point_in_polygon_2d(point, polygon):
        return 0.0
    return min(
        point_segment_distance_2d(
            point,
            polygon[index],
            polygon[(index + 1) % len(polygon)],
        )
        for index in range(len(polygon))
    )


def _is_perimeter_shell_assembly(
    points: np.ndarray,
    *,
    major_span_m: float,
    boundary: np.ndarray,
    perimeter_band_m: float,
) -> bool:
    if len(boundary) < 3:
        return False
    boundary_spans = np.ptp(boundary, axis=0)
    structural_span = max(4.0, 0.25 * float(np.min(boundary_spans)))
    if major_span_m < structural_span:
        return False
    near = [
        min(
            point_segment_distance_2d(
                point,
                boundary[index],
                boundary[(index + 1) % len(boundary)],
            )
            for index in range(len(boundary))
        )
        <= perimeter_band_m
        for point in np.asarray(points, dtype=float)
    ]
    return bool(near) and float(np.mean(near)) >= 0.50


def _assign_cover_arrangements(
    candidates: list[dict[str, Any]],
    *,
    neighbor_distance_m: float,
) -> None:
    centers = np.asarray(
        [item["center_xy"] for item in candidates],
        dtype=float,
    )
    for index, candidate in enumerate(candidates):
        if len(candidates) <= 1:
            candidate["arrangement_type"] = "isolated"
            continue
        distances = np.linalg.norm(centers - centers[index], axis=1)
        neighbors = [
            other
            for other, distance in enumerate(distances.tolist())
            if other != index and distance <= neighbor_distance_m
        ]
        if not neighbors:
            arrangement = "isolated"
        elif len(neighbors) == 1:
            arrangement = "paired"
        else:
            vectors = centers[neighbors] - centers[index]
            covariance = vectors.T @ vectors / max(len(vectors), 1)
            values = np.sort(np.linalg.eigvalsh(covariance))[::-1]
            linearity = (
                math.inf
                if len(values) < 2 or values[1] <= 1.0e-12
                else float(values[0] / values[1])
            )
            arrangement = "row" if linearity >= 5.0 else "cluster"
        candidate["arrangement_type"] = arrangement


def _duplicate_cover_assembly_count(
    candidates: list[dict[str, Any]],
    *,
    tolerance_m: float,
    scale_ratio_tolerance: float,
) -> int:
    """Count near-identical colocated assemblies, not shared centroids.

    A large ring and a small block may legitimately have the same centroid.
    Conversely, a duplicated cover placement has both a near-identical center
    and near-identical rotation-invariant spans/height.
    """

    duplicates = 0
    for index, candidate in enumerate(candidates):
        center = np.asarray(candidate["center_xy"], dtype=float)
        spans = np.asarray(candidate["footprint_spans_m"], dtype=float)
        height = float(candidate["height_m"])
        if any(
            np.linalg.norm(
                center - np.asarray(previous["center_xy"], dtype=float)
            )
            <= tolerance_m
            and _relative_vector_difference(
                spans,
                np.asarray(previous["footprint_spans_m"], dtype=float),
            )
            <= scale_ratio_tolerance
            and abs(height - float(previous["height_m"]))
            / max(height, float(previous["height_m"]), 1.0e-9)
            <= scale_ratio_tolerance
            for previous in candidates[:index]
        ):
            duplicates += 1
    return duplicates


def _relative_vector_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    denominator = np.maximum(
        np.maximum(np.abs(first), np.abs(second)),
        1.0e-9,
    )
    return float(np.max(np.abs(first - second) / denominator))


def _path_exposure(
    free: np.ndarray,
    path: list[tuple[int, int]],
    *,
    resolution: float,
    ray_count: int,
    ray_length_m: float,
) -> float:
    if not path:
        return 1.0
    sampled = _sample_polyline(path, maximum_points=24)
    maximum_steps = max(1, int(round(ray_length_m / resolution)))
    values: list[float] = []
    for row, col in sampled:
        for ray in range(ray_count):
            angle = 2.0 * math.pi * ray / ray_count
            dr = math.sin(angle)
            dc = math.cos(angle)
            clear_steps = 0
            previous = (row, col)
            for step in range(1, maximum_steps + 1):
                nr = int(round(row + dr * step))
                nc = int(round(col + dc * step))
                if (nr, nc) == previous:
                    continue
                previous = (nr, nc)
                if (
                    nr < 0
                    or nr >= free.shape[0]
                    or nc < 0
                    or nc >= free.shape[1]
                    or not free[nr, nc]
                ):
                    break
                clear_steps += 1
            values.append(clear_steps / maximum_steps)
    return float(np.mean(values)) if values else 1.0


def _any_cross_team_los(
    free: np.ndarray,
    team_a: Iterable[tuple[int, int]],
    team_b: Iterable[tuple[int, int]],
) -> bool:
    return any(
        _grid_line_clear(free, source, target)
        for source in team_a
        for target in team_b
    )


def _grid_line_clear(
    free: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    row0, col0 = start
    row1, col1 = goal
    steps = max(abs(row1 - row0), abs(col1 - col0))
    if steps <= 0:
        return True
    for index in range(steps + 1):
        alpha = index / steps
        row = int(round(row0 + alpha * (row1 - row0)))
        col = int(round(col0 + alpha * (col1 - col0)))
        if not free[row, col]:
            return False
    return True


def _ratio_balance_score(
    first: float,
    second: float,
    tolerance: float,
) -> tuple[float, float]:
    denominator = max(abs(first), abs(second), 1.0e-9)
    relative = abs(first - second) / denominator
    if tolerance <= 0.0:
        return (1.0 if relative <= 1.0e-12 else 0.0), relative
    # Equal => 1; at the configured tolerance => 0.5; at twice the
    # tolerance => 0.  This keeps the frozen tolerance interpretable without
    # turning a one-percent crossing into a discontinuous pass/fail jump.
    return max(0.0, 1.0 - 0.5 * relative / tolerance), relative


def _path_mask(
    shape: tuple[int, int],
    path: Iterable[tuple[int, int]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for cell in path:
        mask[cell] = True
    return mask


def _path_length(path: Iterable[tuple[int, int]], resolution: float) -> float:
    entries = list(path)
    length = 0.0
    for previous, current in zip(entries, entries[1:]):
        dr = current[0] - previous[0]
        dc = current[1] - previous[1]
        length += math.hypot(dr, dc) * resolution
    return float(length)


def _sample_polyline(
    path: Iterable[tuple[int, int]],
    *,
    maximum_points: int,
) -> list[tuple[int, int]]:
    entries = list(path)
    if len(entries) <= maximum_points:
        return entries
    indices = np.linspace(0, len(entries) - 1, maximum_points).round().astype(int)
    return [entries[int(index)] for index in indices]


def _jaccard(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.sum(first | second))
    return 0.0 if union <= 0 else float(np.sum(first & second)) / union


def _point_aabb_distance(
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    return float(np.linalg.norm(delta))


def _band_score(
    value: float,
    *,
    lower: float,
    preferred_lower: float,
    preferred_upper: float,
    upper: float,
) -> float:
    if preferred_lower <= value <= preferred_upper:
        return 1.0
    if value <= lower or value >= upper:
        return 0.0
    if value < preferred_lower:
        return (value - lower) / max(preferred_lower - lower, 1.0e-9)
    return (upper - value) / max(upper - preferred_upper, 1.0e-9)


def strip_topology_arrays(value: Any) -> Any:
    """Convert metric/topology diagnostics to JSON-safe values.

    Public reports should use :meth:`CounterStrikeTopology.summary`; this helper
    exists for test fixtures and experiment diagnostics that contain NumPy
    scalars.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): strip_topology_arrays(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strip_topology_arrays(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value
