"""Deterministic SpatialLM wall-loop conversion for the additive benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping
import zipfile

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LinearRing, Polygon

from benchmark.non_rectangular import validate_room_layout
from benchmark.scene_generation.non_rectangular_multi_room.architecture import (
    build_polygon_architecture,
)


SPATIALLM_DATASET_ID = "manycore-research/SpatialLM-Dataset"
SPATIALLM_DATASET_URL = (
    "https://huggingface.co/datasets/manycore-research/SpatialLM-Dataset"
)
SPATIALLM_LICENSE = "cc-by-nc-4.0"
SPATIALLM_LAYOUT_PATH = "layout"
SPATIALLM_SPLIT_PATH = "split.csv"
COHORT_SCHEMA_VERSION = "non_rectangular_spatiallm_layout_cohort_v1"
PROVENANCE_SCHEMA_VERSION = "spatiallm_room_layout_provenance_v1"
CONVERSION_REPORT_SCHEMA_VERSION = "spatiallm_room_layout_conversion_report_v1"
GEOMETRY_TOLERANCE_M = 1.0e-6
ADJACENCY_THRESHOLD_M = 0.35

_MEMBER_RE = re.compile(r"^(scene_\d{6})_(\d{2})_(\d+)\.txt$")
_WALL_RE = re.compile(r"^wall_(\d+)=Wall\((.*)\)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{40}$")
_PALETTE = (
    "#88BDE6",
    "#FBB258",
    "#90CD97",
    "#F6AAC9",
    "#BFA554",
    "#B9A0D3",
    "#8DD3C7",
    "#FDB462",
    "#80B1D3",
    "#FB8072",
)


class SpatialLMConversionError(ValueError):
    """Raised when source layouts cannot be converted without repair."""


@dataclass(frozen=True, slots=True)
class _Member:
    scene_id: str
    source_room_id: int
    sample: int
    archive_path: Path
    member_name: str


@dataclass(frozen=True, slots=True)
class _Wall:
    source_wall_id: int
    start_xyz: tuple[float, float, float]
    end_xyz: tuple[float, float, float]
    height_m: float
    thickness_m: float


@dataclass(frozen=True, slots=True)
class _ParsedMember:
    walls: tuple[_Wall, ...]
    excluded_entity_counts: dict[str, int]


def convert_spatiallm_cohort(
    *,
    source_root: str | Path,
    output_root: str | Path,
    cohort_id: str,
    scene_ids: Iterable[str],
    dataset_revision: str,
    preview_root: str | Path | None = None,
) -> dict[str, Any]:
    """Convert selected scenes, validate them, and write a frozen cohort."""

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    preview = Path(preview_root).resolve() if preview_root is not None else None
    ordered_scene_ids = tuple(str(item) for item in scene_ids)
    _validate_request(
        source_root=source,
        output_root=output,
        cohort_id=cohort_id,
        scene_ids=ordered_scene_ids,
        dataset_revision=dataset_revision,
    )
    split_path = source / SPATIALLM_SPLIT_PATH
    split_sha256 = _sha256_file(split_path)
    split_metadata = _load_split_metadata(split_path, ordered_scene_ids)
    member_index = _index_members(source, ordered_scene_ids)
    archive_sha256: dict[Path, str] = {}

    bundles: list[dict[str, Any]] = []
    for scene_id in ordered_scene_ids:
        if scene_id not in member_index:
            raise SpatialLMConversionError(
                f"selected scene is absent from layout archives: {scene_id}"
            )
        scene_split = split_metadata.get(scene_id)
        if scene_split is None:
            raise SpatialLMConversionError(
                f"selected scene is absent from split.csv: {scene_id}"
            )
        if "train" in scene_split["split_labels"]:
            raise SpatialLMConversionError(
                f"selected scene contains train rows: {scene_id}"
            )
        archive_paths = sorted(
            {
                member.archive_path
                for members in member_index[scene_id].values()
                for member in members
            }
        )
        for archive_path in archive_paths:
            archive_sha256.setdefault(archive_path, _sha256_file(archive_path))
        bundle = _convert_scene(
            scene_id=scene_id,
            room_members=member_index[scene_id],
            split_metadata=scene_split,
            dataset_revision=dataset_revision,
            split_sha256=split_sha256,
            archive_sha256=archive_sha256,
        )
        bundles.append(bundle)

    manifest = _build_manifest(
        cohort_id=cohort_id,
        scene_ids=ordered_scene_ids,
        dataset_revision=dataset_revision,
        split_sha256=split_sha256,
        archive_sha256=archive_sha256,
        bundles=bundles,
    )
    _write_cohort(output, bundles=bundles, manifest=manifest)
    if preview is not None:
        _write_previews(preview, bundles=bundles)
    return manifest


def _validate_request(
    *,
    source_root: Path,
    output_root: Path,
    cohort_id: str,
    scene_ids: tuple[str, ...],
    dataset_revision: str,
) -> None:
    if not source_root.is_dir() or source_root.is_symlink():
        raise SpatialLMConversionError("source_root must be a real directory")
    split_path = source_root / SPATIALLM_SPLIT_PATH
    if not split_path.is_file() or split_path.is_symlink():
        raise SpatialLMConversionError("source_root must contain regular split.csv")
    if output_root.exists() or output_root.is_symlink():
        raise SpatialLMConversionError("output_root already exists")
    if not cohort_id or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", cohort_id):
        raise SpatialLMConversionError("cohort_id must be a stable ID")
    if not scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise SpatialLMConversionError("scene_ids must be non-empty and unique")
    if any(re.fullmatch(r"scene_\d{6}", item) is None for item in scene_ids):
        raise SpatialLMConversionError("scene IDs must match scene_XXXXXX")
    if _SHA256_RE.fullmatch(dataset_revision) is None:
        raise SpatialLMConversionError(
            "dataset_revision must be a lowercase 40-character commit SHA"
        )


def _load_split_metadata(
    split_path: Path,
    scene_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    selected = set(scene_ids)
    splits: dict[str, set[str]] = {scene_id: set() for scene_id in scene_ids}
    room_ids: dict[str, set[int]] = {scene_id: set() for scene_id in scene_ids}
    with split_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"scene_id", "room_id", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SpatialLMConversionError("split.csv lacks required columns")
        for row in reader:
            scene_id = str(row["scene_id"])
            if scene_id not in selected:
                continue
            splits[scene_id].add(str(row["split"]))
            room_ids[scene_id].add(int(row["room_id"]))
    return {
        scene_id: {
            "split_labels": sorted(splits[scene_id]),
            "source_room_ids": sorted(room_ids[scene_id]),
        }
        for scene_id in scene_ids
        if splits[scene_id]
    }


def _index_members(
    source_root: Path,
    scene_ids: tuple[str, ...],
) -> dict[str, dict[int, list[_Member]]]:
    selected = set(scene_ids)
    index: dict[str, dict[int, list[_Member]]] = {}
    archives = sorted(source_root.glob("chunk_*.zip"))
    if not archives:
        raise SpatialLMConversionError("source_root contains no chunk_*.zip")
    for archive_path in archives:
        if not archive_path.is_file() or archive_path.is_symlink():
            raise SpatialLMConversionError(
                f"layout archive must be a regular file: {archive_path.name}"
            )
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
        except (OSError, zipfile.BadZipFile) as exc:
            raise SpatialLMConversionError(
                f"cannot inspect layout archive: {archive_path.name}"
            ) from exc
        for member_name in names:
            match = _MEMBER_RE.fullmatch(Path(member_name).name)
            if match is None:
                continue
            scene_id, source_room_text, sample_text = match.groups()
            if scene_id not in selected:
                continue
            member = _Member(
                scene_id=scene_id,
                source_room_id=int(source_room_text),
                sample=int(sample_text),
                archive_path=archive_path,
                member_name=member_name,
            )
            index.setdefault(scene_id, {}).setdefault(
                member.source_room_id, []
            ).append(member)
    for rooms in index.values():
        for members in rooms.values():
            members.sort(key=lambda item: (item.sample, item.member_name))
    return index


def _convert_scene(
    *,
    scene_id: str,
    room_members: Mapping[int, list[_Member]],
    split_metadata: Mapping[str, Any],
    dataset_revision: str,
    split_sha256: str,
    archive_sha256: Mapping[Path, str],
) -> dict[str, Any]:
    source_room_ids = sorted(room_members)
    if source_room_ids != list(split_metadata["source_room_ids"]):
        raise SpatialLMConversionError(
            f"layout/split room coverage mismatch for {scene_id}: "
            f"layout={source_room_ids!r}, split={split_metadata['source_room_ids']!r}"
        )
    converted_rooms: list[dict[str, Any]] = []
    room_provenance: list[dict[str, Any]] = []
    reversed_room_ids: list[str] = []
    excluded_totals = {"door": 0, "window": 0, "bbox": 0, "other": 0}
    for source_room_id in source_room_ids:
        members = room_members[source_room_id]
        if len({item.sample for item in members}) != len(members):
            raise SpatialLMConversionError(
                f"duplicate source sample for {scene_id} room {source_room_id}"
            )
        parsed_samples: list[tuple[_Member, bytes, _ParsedMember]] = []
        for member in members:
            raw = _read_member(member)
            parsed_samples.append((member, raw, _parse_member(raw)))
        reference_signature = _wall_signature(parsed_samples[0][2].walls)
        if any(
            _wall_signature(parsed.walls) != reference_signature
            for _, _, parsed in parsed_samples[1:]
        ):
            raise SpatialLMConversionError(
                f"wall geometry differs across source samples: "
                f"{scene_id} room {source_room_id}"
            )
        selected_member, selected_raw, selected_parsed = parsed_samples[0]
        room_id = f"room_{source_room_id:03d}"
        converted, reversed_winding = _convert_room(
            room_id=room_id,
            walls=selected_parsed.walls,
        )
        converted_rooms.append(converted)
        if reversed_winding:
            reversed_room_ids.append(room_id)
        for key in excluded_totals:
            excluded_totals[key] += int(
                selected_parsed.excluded_entity_counts.get(key, 0)
            )
        room_provenance.append(
            {
                "room_id": room_id,
                "source_room_id": source_room_id,
                "selected_sample": selected_member.sample,
                "selected_member": selected_member.member_name,
                "selected_member_sha256": _sha256_bytes(selected_raw),
                "source_sample_count": len(parsed_samples),
                "source_samples": [
                    {
                        "sample": member.sample,
                        "archive": member.archive_path.name,
                        "member": member.member_name,
                        "member_sha256": _sha256_bytes(raw),
                    }
                    for member, raw, _ in parsed_samples
                ],
                "wall_geometry_identical_across_samples": True,
                "source_wall_count": len(selected_parsed.walls),
                "excluded_entity_counts_selected_sample": dict(
                    selected_parsed.excluded_entity_counts
                ),
                "polygon_winding_reversed": reversed_winding,
            }
        )

    room_order = [str(room["room_id"]) for room in converted_rooms]
    layout = {
        "schema_version": "non_rectangular_room_layout_v1",
        "layout_id": scene_id,
        "coordinate_frame": {
            "origin": "shared_scene_global",
            "axes": "x_width_y_depth_z_up",
            "handedness": "right_handed",
            "unit": "meter",
            "rotation_unit": "degree",
        },
        "geometry_conventions": {
            "polygon_winding": "counter_clockwise",
            "polygon_closure": "implicit_last_to_first",
            "wall_segment_order": "matches_floor_polygon_edges",
            "inward_normal": "left_of_directed_wall_segment",
        },
        "geometry_tolerance_m": GEOMETRY_TOLERANCE_M,
        "room_count": len(converted_rooms),
        "room_order": room_order,
        "rooms": converted_rooms,
    }
    validation = validate_room_layout(layout)
    geometry_report = _geometry_report(layout, validation=validation)
    if not geometry_report["adjacency"]["connected"]:
        raise SpatialLMConversionError(
            f"converted scene adjacency graph is disconnected: {scene_id}"
        )
    if geometry_report["concave_room_count"] < 1:
        raise SpatialLMConversionError(
            f"selected scene lacks a concave room: {scene_id}"
        )
    if geometry_report["minimum_edge_length_m"] < 0.0995:
        raise SpatialLMConversionError(
            f"selected scene contains a sub-0.10 m edge: {scene_id}"
        )

    layout_bytes = _json_bytes(layout)
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "layout_id": scene_id,
        "source": {
            "dataset_id": SPATIALLM_DATASET_ID,
            "dataset_url": SPATIALLM_DATASET_URL,
            "dataset_revision": dataset_revision,
            "license": SPATIALLM_LICENSE,
            "layout_path": SPATIALLM_LAYOUT_PATH,
            "split_path": SPATIALLM_SPLIT_PATH,
            "split_sha256": split_sha256,
            "scene_id": scene_id,
            "split_labels": list(split_metadata["split_labels"]),
            "archives": [
                {
                    "filename": path.name,
                    "sha256": archive_sha256[path],
                }
                for path in sorted(
                    {
                        member.archive_path
                        for members in room_members.values()
                        for member in members
                    }
                )
            ],
            "rooms": room_provenance,
        },
        "conversion": {
            "policy": "spatiallm_wall_loops_to_nonrect_room_layout_v1",
            "selected_sample_policy": "lowest_sample_index_per_room",
            "all_sample_wall_geometry_must_match": True,
            "coordinates_transformed": False,
            "coordinates_rounded": False,
            "geometry_repair_applied": False,
            "polygon_winding_normalized_to_ccw": True,
            "winding_reversed_room_ids": reversed_room_ids,
            "wall_height_preserved": True,
            "wall_thickness_preserved": True,
            "source_room_type_labels_read_for_output": False,
            "source_room_type_labels_in_output": False,
            "excluded_entities": ["door", "window", "bbox", "point_cloud"],
            "excluded_entity_counts_selected_samples": excluded_totals,
            "floor_z_policy": "require_source_wall_endpoints_at_zero",
            "floor_z_m": 0.0,
        },
        "output": {
            "room_layout_sha256": _sha256_bytes(layout_bytes),
        },
    }
    geometry_report["room_layout_sha256"] = _sha256_bytes(layout_bytes)
    return {
        "scene_id": scene_id,
        "layout": layout,
        "provenance": provenance,
        "geometry_report": geometry_report,
    }


def _read_member(member: _Member) -> bytes:
    try:
        with zipfile.ZipFile(member.archive_path) as archive:
            return archive.read(member.member_name)
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise SpatialLMConversionError(
            f"cannot read {member.archive_path.name}:{member.member_name}"
        ) from exc


def _parse_member(raw: bytes) -> _ParsedMember:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SpatialLMConversionError("layout member is not UTF-8") from exc
    walls: list[_Wall] = []
    excluded = {"door": 0, "window": 0, "bbox": 0, "other": 0}
    for line in text.splitlines():
        stripped = line.strip()
        match = _WALL_RE.fullmatch(stripped)
        if match is not None:
            values = match.group(2).split(",")
            if len(values) != 8:
                raise SpatialLMConversionError("Wall must contain 8 values")
            try:
                numbers = tuple(float(item) for item in values)
            except ValueError as exc:
                raise SpatialLMConversionError("Wall contains a non-number") from exc
            if any(not math.isfinite(item) for item in numbers):
                raise SpatialLMConversionError("Wall contains a non-finite number")
            wall = _Wall(
                source_wall_id=int(match.group(1)),
                start_xyz=(numbers[0], numbers[1], numbers[2]),
                end_xyz=(numbers[3], numbers[4], numbers[5]),
                height_m=numbers[6],
                thickness_m=numbers[7],
            )
            walls.append(wall)
            continue
        if stripped.startswith("door_"):
            excluded["door"] += 1
        elif stripped.startswith("window_"):
            excluded["window"] += 1
        elif stripped.startswith("bbox_"):
            excluded["bbox"] += 1
        elif stripped:
            excluded["other"] += 1
    walls.sort(key=lambda item: item.source_wall_id)
    if len(walls) < 3 or [item.source_wall_id for item in walls] != list(
        range(len(walls))
    ):
        raise SpatialLMConversionError("wall IDs must be contiguous from zero")
    for index, wall in enumerate(walls):
        next_wall = walls[(index + 1) % len(walls)]
        if wall.end_xyz != next_wall.start_xyz:
            raise SpatialLMConversionError("source wall chain is not exactly closed")
        if abs(wall.start_xyz[2]) > GEOMETRY_TOLERANCE_M or abs(
            wall.end_xyz[2]
        ) > GEOMETRY_TOLERANCE_M:
            raise SpatialLMConversionError("source wall endpoint z must be zero")
        if wall.height_m <= 0.0 or wall.thickness_m < 0.0:
            raise SpatialLMConversionError("source wall dimensions are invalid")
    return _ParsedMember(walls=tuple(walls), excluded_entity_counts=excluded)


def _wall_signature(walls: tuple[_Wall, ...]) -> tuple[Any, ...]:
    return tuple(
        (
            wall.source_wall_id,
            wall.start_xyz,
            wall.end_xyz,
            wall.height_m,
            wall.thickness_m,
        )
        for wall in walls
    )


def _convert_room(
    *,
    room_id: str,
    walls: tuple[_Wall, ...],
) -> tuple[dict[str, Any], bool]:
    raw_points = [
        (_clean_float(wall.start_xyz[0]), _clean_float(wall.start_xyz[1]))
        for wall in walls
    ]
    ring = LinearRing(raw_points)
    polygon = Polygon(raw_points)
    if not ring.is_simple or not polygon.is_valid or polygon.area <= 0.0:
        raise SpatialLMConversionError(f"source room polygon is invalid: {room_id}")
    reversed_winding = not ring.is_ccw
    if reversed_winding:
        points = [raw_points[0], *reversed(raw_points[1:])]
        ordered_walls = list(reversed(walls))
    else:
        points = raw_points
        ordered_walls = list(walls)
    segments: list[dict[str, Any]] = []
    for index, source_wall in enumerate(ordered_walls):
        start = points[index]
        end = points[(index + 1) % len(points)]
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= GEOMETRY_TOLERANCE_M:
            raise SpatialLMConversionError(f"source wall is degenerate: {room_id}")
        segments.append(
            {
                "wall_id": f"{room_id}.wall_{index:03d}",
                "start_xy": [start[0], start[1]],
                "end_xy": [end[0], end[1]],
                "inward_normal_xy": [
                    _clean_float(-dy / length),
                    _clean_float(dx / length),
                ],
                "height_m": _clean_float(source_wall.height_m),
                "thickness_m": _clean_float(source_wall.thickness_m),
            }
        )
    return {
        "room_id": room_id,
        "floor_z_m": 0.0,
        "floor_polygon_xy": [[x, y] for x, y in points],
        "wall_segments": segments,
    }, reversed_winding


def _geometry_report(
    layout: Mapping[str, Any],
    *,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    polygons = {
        str(room["room_id"]): Polygon(room["floor_polygon_xy"])
        for room in layout["rooms"]
    }
    adjacency_edges: list[dict[str, Any]] = []
    adjacency: dict[str, set[str]] = {room_id: set() for room_id in polygons}
    room_ids = list(layout["room_order"])
    for left_index, left_id in enumerate(room_ids):
        for right_id in room_ids[left_index + 1 :]:
            distance = float(polygons[left_id].distance(polygons[right_id]))
            if distance <= ADJACENCY_THRESHOLD_M + 1.0e-12:
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)
                adjacency_edges.append(
                    {
                        "room_a": left_id,
                        "room_b": right_id,
                        "polygon_distance_m": distance,
                    }
                )
    components: list[list[str]] = []
    remaining = set(room_ids)
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[str] = set()
        while stack:
            room_id = stack.pop()
            if room_id in component:
                continue
            component.add(room_id)
            stack.extend(adjacency[room_id] - component)
        remaining -= component
        components.append(sorted(component))

    concave_count = 0
    strong_concave_count = 0
    angled_count = 0
    edge_lengths: list[float] = []
    for room in layout["rooms"]:
        polygon = polygons[str(room["room_id"])]
        solidity = float(polygon.area / polygon.convex_hull.area)
        concave_count += int(solidity < 0.999)
        strong_concave_count += int(solidity < 0.90)
        for wall in room["wall_segments"]:
            start, end = wall["start_xy"], wall["end_xy"]
            dx = abs(float(end[0]) - float(start[0]))
            dy = abs(float(end[1]) - float(start[1]))
            length = math.hypot(dx, dy)
            edge_lengths.append(length)
            angled_count += int(length > 0.0 and min(dx, dy) / length > 5.0e-8)

    architecture = build_polygon_architecture(layout)
    shared_walls = [item for item in architecture["physical_walls"] if item["shared"]]
    wall_heights = [
        float(wall["height_m"])
        for room in layout["rooms"]
        for wall in room["wall_segments"]
    ]
    wall_thicknesses = [
        float(wall["thickness_m"])
        for room in layout["rooms"]
        for wall in room["wall_segments"]
    ]
    return {
        "schema_version": CONVERSION_REPORT_SCHEMA_VERSION,
        "layout_id": str(layout["layout_id"]),
        "valid": True,
        "room_count": int(validation["room_count"]),
        "wall_segment_count": int(validation["wall_segment_count"]),
        "total_floor_area_m2": float(validation["total_floor_area_m2"]),
        "minimum_edge_length_m": min(edge_lengths),
        "concave_room_count": concave_count,
        "strong_concave_room_count": strong_concave_count,
        "angled_wall_segment_count": angled_count,
        "adjacency": {
            "policy": "polygon_distance_leq_0.35m_v1",
            "threshold_m": ADJACENCY_THRESHOLD_M,
            "connected": len(components) == 1,
            "component_count": len(components),
            "components": components,
            "edges": adjacency_edges,
        },
        "architecture_check": {
            "compiled": True,
            "logical_wall_count": len(architecture["logical_walls"]),
            "physical_wall_count": len(architecture["physical_walls"]),
            "exact_shared_wall_count": len(shared_walls),
            "near_or_partial_wall_pairing_applied": False,
        },
        "wall_height_m": {
            "min": min(wall_heights),
            "max": max(wall_heights),
        },
        "wall_thickness_m": {
            "min": min(wall_thicknesses),
            "max": max(wall_thicknesses),
        },
        "coordinates_transformed": False,
        "coordinates_rounded": False,
        "geometry_repair_applied": False,
        "excluded_from_layout": [
            "door",
            "window",
            "bbox",
            "point_cloud",
            "room_type",
        ],
    }


def _build_manifest(
    *,
    cohort_id: str,
    scene_ids: tuple[str, ...],
    dataset_revision: str,
    split_sha256: str,
    archive_sha256: Mapping[Path, str],
    bundles: list[dict[str, Any]],
) -> dict[str, Any]:
    scenes: list[dict[str, Any]] = []
    total_rooms = 0
    total_walls = 0
    for bundle in bundles:
        scene_id = str(bundle["scene_id"])
        layout_bytes = _json_bytes(bundle["layout"])
        provenance_bytes = _json_bytes(bundle["provenance"])
        report_bytes = _json_bytes(bundle["geometry_report"])
        room_count = int(bundle["geometry_report"]["room_count"])
        wall_count = int(bundle["geometry_report"]["wall_segment_count"])
        total_rooms += room_count
        total_walls += wall_count
        scenes.append(
            {
                "scene_id": scene_id,
                "layout_id": scene_id,
                "room_count": room_count,
                "wall_segment_count": wall_count,
                "concave_room_count": int(
                    bundle["geometry_report"]["concave_room_count"]
                ),
                "angled_wall_segment_count": int(
                    bundle["geometry_report"]["angled_wall_segment_count"]
                ),
                "adjacency_connected": True,
                "room_layout_path": f"{scene_id}/room_layout.json",
                "room_layout_sha256": _sha256_bytes(layout_bytes),
                "source_provenance_path": f"{scene_id}/source_provenance.json",
                "source_provenance_sha256": _sha256_bytes(provenance_bytes),
                "geometry_validation_path": f"{scene_id}/geometry_validation.json",
                "geometry_validation_sha256": _sha256_bytes(report_bytes),
            }
        )
    manifest = {
        "schema_version": COHORT_SCHEMA_VERSION,
        "cohort_id": cohort_id,
        "source": {
            "dataset_id": SPATIALLM_DATASET_ID,
            "dataset_url": SPATIALLM_DATASET_URL,
            "dataset_revision": dataset_revision,
            "license": SPATIALLM_LICENSE,
            "layout_path": SPATIALLM_LAYOUT_PATH,
            "split_path": SPATIALLM_SPLIT_PATH,
            "split_sha256": split_sha256,
            "archives": [
                {"filename": path.name, "sha256": digest}
                for path, digest in sorted(
                    archive_sha256.items(), key=lambda item: item[0].name
                )
            ],
        },
        "selection": {
            "scene_order": list(scene_ids),
            "scene_count": len(scene_ids),
            "train_scenes_allowed": False,
            "source_room_type_labels_in_output": False,
        },
        "conversion_policy": {
            "coordinates_transformed": False,
            "coordinates_rounded": False,
            "geometry_repair_allowed": False,
            "winding_normalization_allowed": True,
            "near_or_partial_shared_wall_pairing_applied": False,
            "excluded_entities": [
                "door",
                "window",
                "bbox",
                "point_cloud",
                "room_type",
            ],
        },
        "totals": {
            "scene_count": len(scenes),
            "room_count": total_rooms,
            "wall_segment_count": total_walls,
        },
        "scenes": scenes,
    }
    canonical_without_identity = _json_bytes(manifest)
    manifest["manifest_sha256"] = _sha256_bytes(canonical_without_identity)
    return manifest


def _write_cohort(
    output_root: Path,
    *,
    bundles: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        for bundle in bundles:
            scene_root = temporary / str(bundle["scene_id"])
            scene_root.mkdir()
            _write_json(scene_root / "room_layout.json", bundle["layout"])
            _write_json(
                scene_root / "source_provenance.json", bundle["provenance"]
            )
            _write_json(
                scene_root / "geometry_validation.json",
                bundle["geometry_report"],
            )
        _write_json(temporary / "manifest_v1.json", manifest)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_previews(
    preview_root: Path,
    *,
    bundles: list[dict[str, Any]],
) -> None:
    if preview_root.exists() or preview_root.is_symlink():
        raise SpatialLMConversionError("preview_root already exists")
    preview_root.mkdir(parents=True)
    paths: list[Path] = []
    for bundle in bundles:
        path = preview_root / f"{bundle['scene_id']}.png"
        _render_layout(bundle["layout"], bundle["geometry_report"], path)
        paths.append(path)
    _render_contact_sheet(paths, preview_root / "selected10_contact_sheet.png")


def _render_layout(
    layout: Mapping[str, Any],
    report: Mapping[str, Any],
    path: Path,
) -> None:
    width, height = 900, 760
    image = Image.new("RGB", (width, height), "#FAFAF8")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _font(30)
    body_font = _font(18)
    label_font = _font(18)
    draw.text((34, 20), str(layout["layout_id"]), fill="#111111", font=title_font)
    draw.text(
        (34, 61),
        (
            f"R/W/C/A = {report['room_count']}/{report['wall_segment_count']}/"
            f"{report['concave_room_count']}/{report['angled_wall_segment_count']}  |  "
            f"adjacency connected={str(report['adjacency']['connected']).lower()}"
        ),
        fill="#444444",
        font=body_font,
    )
    rooms = list(layout["rooms"])
    all_points = [point for room in rooms for point in room["floor_polygon_xy"]]
    min_x = min(float(point[0]) for point in all_points)
    min_y = min(float(point[1]) for point in all_points)
    max_x = max(float(point[0]) for point in all_points)
    max_y = max(float(point[1]) for point in all_points)
    plot_left, plot_top, plot_right, plot_bottom = 50, 105, width - 50, height - 70
    scale = min(
        (plot_right - plot_left) / max(max_x - min_x, 1.0e-9),
        (plot_bottom - plot_top) / max(max_y - min_y, 1.0e-9),
    )
    used_width = (max_x - min_x) * scale
    used_height = (max_y - min_y) * scale
    offset_x = plot_left + ((plot_right - plot_left) - used_width) / 2.0
    offset_y = plot_top + ((plot_bottom - plot_top) - used_height) / 2.0

    def project(point: list[float]) -> tuple[float, float]:
        return (
            offset_x + (float(point[0]) - min_x) * scale,
            offset_y + (max_y - float(point[1])) * scale,
        )

    for index, room in enumerate(rooms):
        points = [project(point) for point in room["floor_polygon_xy"]]
        draw.polygon(points, fill=_PALETTE[index % len(_PALETTE)] + "66")
        draw.line(points + [points[0]], fill="#20242ACC", width=4, joint="curve")
        polygon = Polygon(room["floor_polygon_xy"])
        anchor = polygon.representative_point()
        cx, cy = project([anchor.x, anchor.y])
        label = str(room["room_id"])
        box = draw.textbbox((cx, cy), label, font=label_font, anchor="mm")
        draw.rounded_rectangle(
            (box[0] - 6, box[1] - 3, box[2] + 6, box[3] + 3),
            radius=5,
            fill="#FFFFFFDD",
        )
        draw.text((cx, cy), label, fill="#101010", font=label_font, anchor="mm")
    draw.text(
        (34, height - 44),
        (
            "global coordinates preserved | room types excluded | "
            f"exact shared walls={report['architecture_check']['exact_shared_wall_count']}"
        ),
        fill="#333333",
        font=body_font,
    )
    image.save(path, optimize=True)


def _render_contact_sheet(paths: list[Path], output: Path) -> None:
    columns, rows = 5, 2
    cell = (430, 365)
    sheet = Image.new("RGB", (columns * cell[0], rows * cell[1]), "white")
    for index, path in enumerate(paths):
        preview = Image.open(path).convert("RGB")
        preview.thumbnail((cell[0] - 12, cell[1] - 12), Image.Resampling.LANCZOS)
        x = (index % columns) * cell[0] + (cell[0] - preview.width) // 2
        y = (index // columns) * cell[1] + (cell[1] - preview.height) // 2
        sheet.paste(preview, (x, y))
    sheet.save(output, optimize=True)


def _font(size: int):
    for candidate in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ):
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_json_bytes(value))


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_float(value: float) -> float:
    return 0.0 if abs(float(value)) < 1.0e-15 else float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preview-root", type=Path)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--scene-id", action="append", required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    manifest = convert_spatiallm_cohort(
        source_root=args.source_root,
        output_root=args.output_root,
        preview_root=args.preview_root,
        cohort_id=args.cohort_id,
        scene_ids=args.scene_id,
        dataset_revision=args.dataset_revision,
    )
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADJACENCY_THRESHOLD_M",
    "COHORT_SCHEMA_VERSION",
    "CONVERSION_REPORT_SCHEMA_VERSION",
    "PROVENANCE_SCHEMA_VERSION",
    "SPATIALLM_DATASET_ID",
    "SpatialLMConversionError",
    "convert_spatiallm_cohort",
]
