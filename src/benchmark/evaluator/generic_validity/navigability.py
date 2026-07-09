from __future__ import annotations

from collections import deque

import numpy as np

from benchmark.evaluator.generic_validity.geometry import (
    get_footprint_corners_xy,
    get_room_boundary,
    normalize_objects,
    point_in_polygon_2d,
    point_polygon_distance_2d,
)


def check_navigability(scene: dict, config: dict | None = None, navigability_cache: dict | None = None) -> dict:
    cache = navigability_cache or compute_navigability_grid(scene, config)
    if cache.get("status") != "checked":
        return {
            "metric": "navigability",
            "status": cache.get("status", "invalid_input"),
            "score": float(cache.get("score", 0.0)),
            "reason": cache.get("reason", "navigability grid could not be computed"),
        }
    total_free_cells = int(cache["total_free_cells"])
    largest_component_cells = int(cache["largest_component_cells"])
    resolution = float(cache["grid_resolution"])
    total_free_area = total_free_cells * resolution * resolution
    largest_area = largest_component_cells * resolution * resolution
    score = 0.0 if total_free_cells <= 0 else float(largest_component_cells) / float(total_free_cells)
    return {
        "metric": "navigability",
        "status": "checked",
        "score": float(score),
        "largest_connected_free_area": float(largest_area),
        "total_free_area": float(total_free_area),
        "num_components": int(cache["num_components"]),
        "grid_resolution": resolution,
        "blocking_object_count": int(cache["blocking_object_count"]),
        "non_blocking_object_count": int(cache["non_blocking_object_count"]),
    }


def compute_navigability_grid(scene: dict, config: dict | None = None) -> dict:
    cfg = config or {}
    boundary = np.asarray(get_room_boundary(scene), dtype=float)
    if boundary.ndim != 2 or len(boundary) < 3:
        return {"status": "invalid_input", "score": 0.0, "reason": "scene boundary is missing or invalid"}
    resolution = max(float(cfg.get("grid_resolution", 0.08)), 0.01)
    agent_radius = max(float(cfg.get("agent_radius", 0.25)), 0.0)
    connectivity = int(cfg.get("connectivity", 4))
    objects, object_errors = normalize_objects(scene)
    min_x, max_x = float(np.min(boundary[:, 0])), float(np.max(boundary[:, 0]))
    min_y, max_y = float(np.min(boundary[:, 1])), float(np.max(boundary[:, 1]))
    xs = np.arange(min_x + resolution / 2.0, max_x, resolution)
    ys = np.arange(min_y + resolution / 2.0, max_y, resolution)
    if len(xs) == 0 or len(ys) == 0:
        return {"status": "invalid_input", "score": 0.0, "reason": "room boundary AABB is empty"}

    inside_room = np.zeros((len(ys), len(xs)), dtype=bool)
    occupied = np.zeros_like(inside_room)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            inside_room[row, col] = point_in_polygon_2d([x, y], boundary)

    blocking_objects = []
    non_blocking_objects = []
    for obj in objects:
        if _is_blocking_object(obj, cfg):
            blocking_objects.append(obj)
        else:
            non_blocking_objects.append(obj)

    blocking_footprints = [(obj, get_footprint_corners_xy(obj)) for obj in blocking_objects]
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            if not inside_room[row, col]:
                continue
            point = np.array([x, y], dtype=float)
            for _, footprint in blocking_footprints:
                if point_in_polygon_2d(point, footprint) or point_polygon_distance_2d(point, footprint) <= agent_radius:
                    occupied[row, col] = True
                    break

    free = inside_room & ~occupied
    component_labels, component_sizes = _label_components(free, connectivity=connectivity)
    total_free_cells = int(np.sum(free))
    if component_sizes:
        largest_component_id, largest_component_cells = max(component_sizes.items(), key=lambda item: item[1])
    else:
        largest_component_id, largest_component_cells = -1, 0
    score = 0.0 if total_free_cells <= 0 else float(largest_component_cells) / float(total_free_cells)
    return {
        "status": "checked",
        "score": float(score),
        "x_centers": xs,
        "y_centers": ys,
        "inside_room": inside_room,
        "occupied": occupied,
        "free": free,
        "component_labels": component_labels,
        "component_sizes": component_sizes,
        "largest_component_id": int(largest_component_id),
        "largest_component_cells": int(largest_component_cells),
        "total_free_cells": total_free_cells,
        "num_components": len(component_sizes),
        "grid_resolution": resolution,
        "blocking_object_count": len(blocking_objects),
        "non_blocking_object_count": len(non_blocking_objects),
        "blocking_object_ids": [obj.id for obj in blocking_objects],
        "object_errors": object_errors,
    }


def _is_blocking_object(obj: object, config: dict) -> bool:
    clearance_height = float(config.get("clearance_height", 1.70))
    step_over_height = float(config.get("step_over_height", 0.15))
    return bool(obj.bottom_z < clearance_height and obj.top_z > step_over_height)


def _label_components(free: np.ndarray, *, connectivity: int) -> tuple[np.ndarray, dict[int, int]]:
    labels = np.full(free.shape, -1, dtype=int)
    sizes: dict[int, int] = {}
    next_label = 0
    neighbor_offsets = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        neighbor_offsets.extend([(1, 1), (1, -1), (-1, 1), (-1, -1)])
    rows, cols = free.shape
    for row in range(rows):
        for col in range(cols):
            if not free[row, col] or labels[row, col] >= 0:
                continue
            queue: deque[tuple[int, int]] = deque([(row, col)])
            labels[row, col] = next_label
            size = 0
            while queue:
                current_row, current_col = queue.popleft()
                size += 1
                for dr, dc in neighbor_offsets:
                    nr, nc = current_row + dr, current_col + dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        continue
                    if free[nr, nc] and labels[nr, nc] < 0:
                        labels[nr, nc] = next_label
                        queue.append((nr, nc))
            sizes[next_label] = size
            next_label += 1
    return labels, sizes
