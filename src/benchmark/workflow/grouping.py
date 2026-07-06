from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any

from benchmark.workflow.scoring import find_layout_object, visible_attachments, visible_relations


@dataclass(frozen=True)
class ResolvedGroupingConfig:
    enabled: bool
    num_renderable_objects: int
    room_width_m: float
    room_depth_m: float
    room_diagonal_m: float
    scene_object_extent_m: float
    median_object_footprint_diag_m: float
    effective_max_gap_m: float
    effective_min_gap_m: float
    effective_max_normalized_gap: float
    effective_max_group_diameter_m: float
    effective_max_objects_per_group: int
    effective_strong_link_max_group_diameter_m: float
    effective_strong_link_max_objects_per_group: int
    derived_support_enabled: bool
    derived_support_vertical_tolerance_m: float
    derived_support_min_xy_overlap_ratio: float


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    reason: str
    strength: str
    priority: int
    weight: float
    is_ground_truth_relation: bool
    derived_from: str


def build_object_groups(layout: dict, case: dict, config: dict | None = None) -> list[dict]:
    return build_object_grouping_report(layout, case, config)["object_groups"]


def build_object_grouping_report(layout: dict, case: dict, config: dict | None = None) -> dict:
    objects = _renderable_objects(layout)
    ids = [_object_id(obj) for obj in objects]
    resolved = resolve_grouping_config(layout, case, config)
    if not ids:
        return {
            "object_groups": [],
            "resolved_grouping_config": asdict(resolved),
            "omitted_edges": [],
            "cross_group_relations": [],
        }

    if not resolved.enabled:
        groups = [_group_record(index + 1, [object_id], {}, [], objects) for index, object_id in enumerate(ids)]
        _validate_group_partition(groups, ids)
        return {
            "object_groups": groups,
            "resolved_grouping_config": asdict(resolved),
            "omitted_edges": [],
            "cross_group_relations": [],
        }

    region_groups = _semantic_region_groups(objects, case)
    if region_groups:
        _validate_group_partition(region_groups, ids)
        return {
            "object_groups": region_groups,
            "resolved_grouping_config": {
                **asdict(resolved),
                "grouping_source": "semantic_region",
            },
            "omitted_edges": [],
            "cross_group_relations": [],
        }

    index_by_id = {object_id: index for index, object_id in enumerate(ids)}
    dsu = _DisjointSet(ids)
    reasons: dict[str, list[str]] = {object_id: [] for object_id in ids}
    formation_edges: list[dict] = []
    omitted_edges: list[dict] = []

    must_edges = _must_link_edges(layout, case)
    for edge in must_edges:
        if edge.source in index_by_id and edge.target in index_by_id:
            dsu.union(edge.source, edge.target)
            _record_reason(reasons, edge.source, edge.target, edge.reason)
            _record_formation_edge(formation_edges, edge)

    relation_edges = _relation_edges(layout, case)
    candidate_edges = (
        relation_edges
        + _layout_relation_edges(layout)
        + _derived_support_edges(objects, resolved)
        + _metadata_local_edges(objects, resolved)
        + _proximity_edges(objects, resolved)
    )
    candidate_edges.sort(key=lambda edge: (edge.priority, edge.weight, edge.source, edge.target))
    for edge in candidate_edges:
        if edge.source not in index_by_id or edge.target not in index_by_id:
            continue
        if dsu.find(edge.source) == dsu.find(edge.target):
            _record_reason(reasons, edge.source, edge.target, edge.reason)
            _record_formation_edge(formation_edges, edge)
            continue

        merged_ids = dsu.component(edge.source) + dsu.component(edge.target)
        merged_objects = [objects[index_by_id[item]] for item in merged_ids]
        limit_check = _group_limit_check(merged_objects, resolved, strength=edge.strength)
        if limit_check["within_limits"]:
            dsu.union(edge.source, edge.target)
            _record_reason(reasons, edge.source, edge.target, edge.reason)
            _record_formation_edge(formation_edges, edge)
        else:
            omitted_edges.append(_omitted_edge_record(edge, limit_check))

    components = dsu.components(ids)
    groups = []
    for group_index, object_ids in enumerate(components, start=1):
        groups.append(_group_record(group_index, object_ids, reasons, formation_edges, objects))
    _validate_group_partition(groups, ids)
    object_to_group = {
        object_id: group["group_id"]
        for group in groups
        for object_id in group.get("object_ids", [])
    }
    cross_group_relations = _cross_group_relations(relation_edges + _layout_relation_edges(layout), object_to_group)
    return {
        "object_groups": groups,
        "resolved_grouping_config": asdict(resolved),
        "omitted_edges": omitted_edges,
        "cross_group_relations": cross_group_relations,
    }


def resolve_grouping_config(layout: dict, case: dict, config: dict | None = None) -> ResolvedGroupingConfig:
    objects = _renderable_objects(layout)
    section = _grouping_section(config)
    enabled = bool(section.get("enabled", True))

    room_width, room_depth = _room_dimensions(case, objects)
    room_diagonal = math.hypot(room_width, room_depth)
    scene_extent = _footprint_diameter(objects)
    footprint_diags = [_footprint_diag(obj) for obj in objects if _footprint_diag(obj) > 0]
    median_diag = float(median(footprint_diags)) if footprint_diags else 1.0

    proximity = _subsection(section, "proximity")
    diameter = _subsection(section, "diameter")
    object_count = _subsection(section, "object_count")
    strong = _subsection(section, "strong_link_relaxation")
    derived_support = _subsection(section, "derived_support")

    max_group_diameter = _clamp(
        room_diagonal * float(diameter.get("ratio_of_room_diagonal", 0.35)),
        float(diameter.get("min_m", 2.5)),
        float(diameter.get("max_m", 8.0)),
    )
    object_cap = _clamp_int(
        math.ceil(math.sqrt(max(1, len(objects)))) + int(object_count.get("additive_margin", 1)),
        int(object_count.get("min_objects_per_group", 6)),
        int(object_count.get("max_objects_per_group", 12)),
    )
    strong_diameter = max_group_diameter * float(strong.get("max_group_diameter_multiplier", 1.25))
    strong_objects = max(object_cap, math.ceil(object_cap * float(strong.get("max_objects_multiplier", 1.5))))
    return ResolvedGroupingConfig(
        enabled=enabled,
        num_renderable_objects=len(objects),
        room_width_m=_round(room_width),
        room_depth_m=_round(room_depth),
        room_diagonal_m=_round(room_diagonal),
        scene_object_extent_m=_round(scene_extent),
        median_object_footprint_diag_m=_round(median_diag),
        effective_max_gap_m=float(proximity.get("max_gap_m", 1.25)),
        effective_min_gap_m=float(proximity.get("min_gap_m", 0.25)),
        effective_max_normalized_gap=float(proximity.get("max_normalized_gap", 0.75)),
        effective_max_group_diameter_m=_round(max_group_diameter),
        effective_max_objects_per_group=object_cap,
        effective_strong_link_max_group_diameter_m=_round(strong_diameter),
        effective_strong_link_max_objects_per_group=strong_objects,
        derived_support_enabled=bool(derived_support.get("enabled", True)),
        derived_support_vertical_tolerance_m=float(derived_support.get("vertical_tolerance_m", 0.08)),
        derived_support_min_xy_overlap_ratio=float(derived_support.get("min_xy_overlap_ratio", 0.15)),
    )


class _DisjointSet:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        keep, move = sorted([root_left, root_right])
        self.parent[move] = keep

    def component(self, value: str) -> list[str]:
        root = self.find(value)
        return sorted(item for item in self.parent if self.find(item) == root)

    def components(self, ordered_ids: list[str]) -> list[list[str]]:
        by_root: dict[str, list[str]] = {}
        for object_id in ordered_ids:
            by_root.setdefault(self.find(object_id), []).append(object_id)
        return sorted((items for items in by_root.values()), key=lambda items: ordered_ids.index(items[0]))


def _must_link_edges(layout: dict, case: dict) -> list[_Edge]:
    edges: list[_Edge] = []
    for obj in layout.get("objects", []):
        if not isinstance(obj, dict):
            continue
        parent = obj.get("support_parent")
        if isinstance(parent, str) and parent and parent != "floor":
            edges.append(_Edge(_object_id(obj), parent, "support_parent", "must_link", 0, 0.0, True, "layout.support_parent"))

    for attachment in visible_attachments(case):
        child_obj = find_layout_object(layout, str(attachment.get("child") or ""))
        parent_obj = find_layout_object(layout, str(attachment.get("parent") or ""))
        if child_obj and parent_obj:
            edges.append(
                _Edge(
                    _object_id(child_obj),
                    _object_id(parent_obj),
                    "attachment",
                    "must_link",
                    0,
                    0.0,
                    True,
                    "case.visible_attachments",
                )
            )
    return edges


def _relation_edges(layout: dict, case: dict) -> list[_Edge]:
    edges: list[_Edge] = []
    for relation in visible_relations(case):
        source = find_layout_object(layout, str(relation.get("subject") or relation.get("source") or ""))
        target = find_layout_object(layout, str(relation.get("object") or relation.get("target") or ""))
        if source and target:
            edges.append(
                _Edge(
                    _object_id(source),
                    _object_id(target),
                    "explicit_relation",
                    "strong_link",
                    1,
                    _floor_gap(source, target),
                    True,
                    "case.visible_relations",
                )
            )
    return edges


def _layout_relation_edges(layout: dict) -> list[_Edge]:
    edges: list[_Edge] = []
    for relation in layout.get("relations", []) if isinstance(layout.get("relations"), list) else []:
        if not isinstance(relation, dict):
            continue
        source = find_layout_object(layout, str(relation.get("source") or relation.get("subject") or ""))
        target = find_layout_object(layout, str(relation.get("target") or relation.get("object") or ""))
        if source and target:
            edges.append(
                _Edge(
                    _object_id(source),
                    _object_id(target),
                    "explicit_relation",
                    "strong_link",
                    1,
                    _floor_gap(source, target),
                    True,
                    "layout.relations",
                )
            )
    return edges


def _metadata_local_edges(objects: list[dict], resolved: ResolvedGroupingConfig) -> list[_Edge]:
    edges: list[_Edge] = []
    for index, obj_a in enumerate(objects):
        region_a = obj_a.get("region_id")
        if not isinstance(region_a, str) or not region_a:
            continue
        for obj_b in objects[index + 1 :]:
            if obj_b.get("region_id") != region_a:
                continue
            gap = _floor_gap(obj_a, obj_b)
            if gap <= _effective_gap_pair(obj_a, obj_b, resolved):
                edges.append(
                    _Edge(
                        _object_id(obj_a),
                        _object_id(obj_b),
                        "metadata_region",
                        "soft_link",
                        2,
                        gap,
                        False,
                        "layout.region_id+bbox_floor_gap",
                    )
                )
    return edges


def _proximity_edges(objects: list[dict], resolved: ResolvedGroupingConfig) -> list[_Edge]:
    edges: list[_Edge] = []
    for index, obj_a in enumerate(objects):
        for obj_b in objects[index + 1 :]:
            gap = _floor_gap(obj_a, obj_b)
            if gap <= _effective_gap_pair(obj_a, obj_b, resolved):
                edges.append(
                    _Edge(
                        _object_id(obj_a),
                        _object_id(obj_b),
                        "proximity",
                        "soft_link",
                        3,
                        gap,
                        False,
                        "bbox_floor_gap",
                    )
                )
    return edges


def _derived_support_edges(objects: list[dict], resolved: ResolvedGroupingConfig) -> list[_Edge]:
    if not resolved.derived_support_enabled:
        return []
    edges: list[_Edge] = []
    for child in objects:
        child_box = _footprint_box(child)
        child_bottom = float(child["center"][2]) - float(child["size"][2]) / 2.0
        child_area = _footprint_area(child)
        for parent in objects:
            if _object_id(child) == _object_id(parent):
                continue
            parent_top = float(parent["center"][2]) + float(parent["size"][2]) / 2.0
            if abs(child_bottom - parent_top) > resolved.derived_support_vertical_tolerance_m:
                continue
            overlap = _overlap_area(child_box, _footprint_box(parent))
            if overlap <= 0:
                continue
            overlap_ratio = overlap / max(1.0e-9, min(child_area, _footprint_area(parent)))
            if overlap_ratio < resolved.derived_support_min_xy_overlap_ratio:
                continue
            if child_area > _footprint_area(parent) * 1.25:
                continue
            edges.append(
                _Edge(
                    _object_id(child),
                    _object_id(parent),
                    "derived_support_geometry",
                    "strong_link",
                    1,
                    max(0.0, 1.0 - overlap_ratio),
                    False,
                    "bbox_vertical_contact+xy_overlap",
                )
            )
    return edges


def _effective_gap_pair(obj_a: dict, obj_b: dict, resolved: ResolvedGroupingConfig) -> float:
    local_scale = max(_mean_footprint_diag(obj_a, obj_b), 1.0e-6)
    return _clamp(
        resolved.effective_max_normalized_gap * local_scale,
        resolved.effective_min_gap_m,
        resolved.effective_max_gap_m,
    )


def _floor_gap(obj_a: dict, obj_b: dict) -> float:
    ax_min, ax_max, ay_min, ay_max = _footprint_box(obj_a)
    bx_min, bx_max, by_min, by_max = _footprint_box(obj_b)
    dx = max(0.0, bx_min - ax_max, ax_min - bx_max)
    dy = max(0.0, by_min - ay_max, ay_min - by_max)
    return math.hypot(dx, dy)


def _mean_footprint_diag(obj_a: dict, obj_b: dict) -> float:
    return (_footprint_diag(obj_a) + _footprint_diag(obj_b)) / 2.0


def _footprint_diag(obj: dict) -> float:
    width, depth, _ = [float(value) for value in obj["size"]]
    return math.hypot(width, depth)


def _group_limit_check(objects: list[dict], resolved: ResolvedGroupingConfig, *, strength: str) -> dict:
    if strength == "strong_link":
        max_objects = resolved.effective_strong_link_max_objects_per_group
        max_diameter = resolved.effective_strong_link_max_group_diameter_m
    else:
        max_objects = resolved.effective_max_objects_per_group
        max_diameter = resolved.effective_max_group_diameter_m
    diameter = _footprint_diameter(objects)
    exceeds_objects = len(objects) > max_objects
    exceeds_diameter = diameter > max_diameter
    return {
        "within_limits": not (exceeds_objects or exceeds_diameter),
        "objects": len(objects),
        "diameter_m": _round(diameter),
        "max_objects": max_objects,
        "max_diameter_m": _round(max_diameter),
        "would_exceed": {
            "objects": exceeds_objects,
            "diameter": exceeds_diameter,
        },
    }


def _footprint_diameter(objects: list[dict]) -> float:
    if not objects:
        return 0.0
    min_x = min(_footprint_box(obj)[0] for obj in objects)
    max_x = max(_footprint_box(obj)[1] for obj in objects)
    min_y = min(_footprint_box(obj)[2] for obj in objects)
    max_y = max(_footprint_box(obj)[3] for obj in objects)
    return math.hypot(max_x - min_x, max_y - min_y)


def _group_record(
    group_index: int,
    object_ids: list[str],
    reasons: dict[str, list[str]],
    formation_edges: list[dict],
    all_objects: list[dict],
) -> dict:
    object_set = set(object_ids)
    group_objects = [obj for obj in all_objects if _object_id(obj) in object_set]
    group_reasons = sorted({reason for object_id in object_ids for reason in reasons.get(object_id, [])})
    return {
        "group_id": f"group_{group_index:03d}",
        "group_source": "spatial_cluster",
        "region_id": None,
        "region_category": None,
        "object_ids": object_ids,
        "num_objects": len(object_ids),
        "group_footprint_diameter_m": _round(_footprint_diameter(group_objects)),
        "edge_reasons": group_reasons,
        "formation_edges": [
            edge
            for edge in formation_edges
            if edge["source"] in object_set and edge["target"] in object_set
        ],
    }


def _semantic_region_groups(objects: list[dict], case: dict) -> list[dict]:
    source_regions = _case_object_regions(case)
    if not source_regions:
        return []
    object_ids = [_object_id(obj) for obj in objects]
    assigned = [object_id for object_id in object_ids if object_id in source_regions]
    if len(assigned) < max(2, int(len(object_ids) * 0.5)):
        return []
    region_meta = _case_region_metadata(case)
    by_region: dict[str, list[str]] = {}
    unassigned = []
    for object_id in object_ids:
        region_id = source_regions.get(object_id)
        if region_id:
            by_region.setdefault(region_id, []).append(object_id)
        else:
            unassigned.append(object_id)
    groups = []
    for index, (region_id, ids) in enumerate(sorted(by_region.items()), start=1):
        group = _group_record(index, ids, {object_id: ["semantic_region"] for object_id in ids}, [], objects)
        meta = region_meta.get(region_id, {})
        group.update(
            {
                "group_source": "semantic_region",
                "region_id": region_id,
                "region_category": meta.get("label") or meta.get("category") or meta.get("name"),
            }
        )
        groups.append(group)
    if unassigned:
        group = _group_record(len(groups) + 1, unassigned, {object_id: ["region_unassigned"] for object_id in unassigned}, [], objects)
        group.update({"group_source": "semantic_region_unassigned", "region_id": None, "region_category": None})
        groups.append(group)
    return groups


def _case_object_regions(case: dict) -> dict[str, str]:
    regions = {}
    for obj in case.get("objects", []) if isinstance(case.get("objects"), list) else []:
        if not isinstance(obj, dict):
            continue
        object_id = obj.get("id")
        region_id = obj.get("source_region_id") or obj.get("region_id")
        if isinstance(object_id, str) and isinstance(region_id, str) and region_id:
            regions[object_id] = region_id
    return regions


def _case_region_metadata(case: dict) -> dict[str, dict]:
    room = case.get("room") if isinstance(case, dict) else {}
    candidates = []
    if isinstance(room, dict):
        if isinstance(room.get("regions"), list):
            candidates.extend(room["regions"])
        floor_plan = room.get("floor_plan")
        if isinstance(floor_plan, dict) and isinstance(floor_plan.get("regions"), list):
            candidates.extend(floor_plan["regions"])
    metadata = {}
    for region in candidates:
        if not isinstance(region, dict):
            continue
        region_id = region.get("id")
        if isinstance(region_id, str) and region_id:
            metadata[region_id] = region
    return metadata


def _cross_group_relations(explicit_edges: list[_Edge], object_to_group: dict[str, str]) -> list[dict]:
    records = []
    for edge in explicit_edges:
        source_group = object_to_group.get(edge.source)
        target_group = object_to_group.get(edge.target)
        if source_group and target_group and source_group != target_group:
            records.append(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "reason": edge.reason,
                    "source_group": source_group,
                    "target_group": target_group,
                    "status": "cross_group_due_to_limits",
                }
            )
    return _dedupe_records(records)


def _omitted_edge_record(edge: _Edge, limit_check: dict) -> dict:
    return {
        **_edge_record(edge),
        "rejected_reason": "group_limit",
        "would_exceed": limit_check["would_exceed"],
        "attempted_group_size": limit_check["objects"],
        "attempted_group_footprint_diameter_m": limit_check["diameter_m"],
        "max_objects": limit_check["max_objects"],
        "max_group_diameter_m": limit_check["max_diameter_m"],
    }


def _record_reason(reasons: dict[str, list[str]], source: str, target: str, reason: str) -> None:
    reasons.setdefault(source, []).append(reason)
    reasons.setdefault(target, []).append(reason)


def _record_formation_edge(edges: list[dict], edge: _Edge) -> None:
    record = _edge_record(edge)
    if record not in edges:
        edges.append(record)


def _edge_record(edge: _Edge) -> dict:
    source, target = sorted([edge.source, edge.target])
    return {
        "source": source,
        "target": target,
        "reason": edge.reason,
        "strength": edge.strength,
        "priority": edge.priority,
        "weight": _round(edge.weight),
        "is_ground_truth_relation": edge.is_ground_truth_relation,
        "derived_from": edge.derived_from,
    }


def _validate_group_partition(groups: list[dict], expected_ids: list[str]) -> None:
    assigned = [object_id for group in groups for object_id in group.get("object_ids", [])]
    if sorted(assigned) != sorted(expected_ids) or len(assigned) != len(set(assigned)):
        raise ValueError("Object grouping failed to assign every object exactly once.")


def _grouping_section(config: dict | None) -> dict:
    if not isinstance(config, dict):
        return {}
    section = config.get("grouping")
    return section if isinstance(section, dict) else config


def _subsection(section: dict, key: str) -> dict:
    value = section.get(key)
    return value if isinstance(value, dict) else {}


def _renderable_objects(layout: dict) -> list[dict]:
    objects = []
    for obj in layout.get("objects", []) if isinstance(layout.get("objects"), list) else []:
        if isinstance(obj, dict) and isinstance(obj.get("center"), list) and isinstance(obj.get("size"), list):
            try:
                [float(value) for value in obj["center"]]
                size = [float(value) for value in obj["size"]]
            except (TypeError, ValueError):
                continue
            if len(obj["center"]) == 3 and len(size) == 3 and all(value > 0 for value in size):
                objects.append(obj)
    return objects


def _room_dimensions(case: dict, objects: list[dict]) -> tuple[float, float]:
    room = case.get("room") if isinstance(case, dict) else {}
    if isinstance(room, dict):
        width = _positive_float(room.get("width") or room.get("room_width") or room.get("width_m"))
        depth = _positive_float(room.get("depth") or room.get("room_depth") or room.get("depth_m"))
        if width and depth:
            return width, depth
        polygon = room.get("floor_polygon") or room.get("boundary")
        dims = _polygon_dimensions(polygon)
        if dims:
            return dims
    if objects:
        min_x = min(_footprint_box(obj)[0] for obj in objects)
        max_x = max(_footprint_box(obj)[1] for obj in objects)
        min_y = min(_footprint_box(obj)[2] for obj in objects)
        max_y = max(_footprint_box(obj)[3] for obj in objects)
        return max(1.0, max_x - min_x), max(1.0, max_y - min_y)
    return 1.0, 1.0


def _polygon_dimensions(polygon: object) -> tuple[float, float] | None:
    if not isinstance(polygon, list) or len(polygon) < 2:
        return None
    points = [point for point in polygon if isinstance(point, list) and len(point) >= 2]
    if len(points) < 2:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError):
        return None
    return max(1.0e-6, max(xs) - min(xs)), max(1.0e-6, max(ys) - min(ys))


def _positive_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _footprint_box(obj: dict) -> tuple[float, float, float, float]:
    x, y, _ = [float(value) for value in obj["center"]]
    width, depth, _ = [float(value) for value in obj["size"]]
    return x - width / 2.0, x + width / 2.0, y - depth / 2.0, y + depth / 2.0


def _footprint_area(obj: dict) -> float:
    width, depth, _ = [float(value) for value in obj["size"]]
    return max(0.0, width * depth)


def _overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    overlap_x = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    overlap_y = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    return overlap_x * overlap_y


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))


def _round(value: float) -> float:
    return round(float(value), 6)


def _dedupe_records(records: list[dict]) -> list[dict]:
    unique = []
    for record in records:
        if record not in unique:
            unique.append(record)
    return unique


def _object_id(obj: dict) -> str:
    return str(obj.get("object_id") or obj.get("id") or "unknown")
