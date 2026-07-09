"""Legend HSSD-HAB small-scene selector.

This selector belongs to the legacy HSSD input path. Current input should enter
through natural-language scene specs.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from benchmark.legend.hssd.hssd_hab_converter import convert_hssd_hab, iter_scene_instance_paths
from benchmark.utils.io import write_json


@dataclass(frozen=True)
class HSSDSmallSceneCandidate:
    scene_id: str
    path: Path
    object_count: int


def select_natural_small_hssd_scene(
    *,
    hssd_root: str | Path,
    min_objects: int = 6,
    max_objects: int = 20,
) -> HSSDSmallSceneCandidate:
    candidates = [
        candidate
        for candidate in iter_hssd_scene_candidates(hssd_root)
        if min_objects <= candidate.object_count <= max_objects
    ]
    if not candidates:
        raise ValueError(f"No HSSD scene has object_count in [{min_objects}, {max_objects}].")
    return sorted(candidates, key=lambda item: (item.object_count, item.scene_id, str(item.path)))[0]


def iter_hssd_scene_candidates(hssd_root: str | Path) -> Iterable[HSSDSmallSceneCandidate]:
    for path in iter_scene_instance_paths(hssd_root):
        try:
            scene = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_objects = scene.get("object_instances") or scene.get("objects") or []
        if not isinstance(raw_objects, list):
            continue
        scene_id = str(scene.get("scene_id") or scene.get("name") or Path(path).stem.replace(".scene_instance", ""))
        yield HSSDSmallSceneCandidate(scene_id=scene_id, path=Path(path), object_count=len(raw_objects))


def convert_selected_small_hssd_scene(
    *,
    hssd_root: str | Path,
    out_dir: str | Path,
    min_objects: int = 6,
    max_objects: int = 20,
    levels: list[str] | None = None,
    compact_object_ids: bool = False,
    preserve_raw_metadata: bool = False,
    bbox_from_scale: bool = False,
    include_estimated_relations: bool = True,
    input_representation_mode: str | None = None,
) -> tuple[HSSDSmallSceneCandidate, list[Path], Path]:
    selected = select_natural_small_hssd_scene(
        hssd_root=hssd_root,
        min_objects=min_objects,
        max_objects=max_objects,
    )
    selected_levels = levels or ["structured_basic"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    converted_paths = convert_hssd_hab(
        hssd_root=hssd_root,
        out_dir=out,
        scene_paths=[selected.path],
        levels=selected_levels,
        max_objects=None,
        compact_object_ids=compact_object_ids,
        preserve_raw_metadata=preserve_raw_metadata,
        bbox_from_scale=bbox_from_scale,
        include_estimated_relations=include_estimated_relations,
        input_representation_mode=input_representation_mode,
    )
    stable_paths = [_copy_to_stable_selected_name(path, out) for path in converted_paths]
    manifest_path = write_json(
        out / "selected_manifest.json",
        {
            "selection_policy": "natural_small_complete_scene",
            "min_objects": min_objects,
            "max_objects": max_objects,
            "selected": {
                "scene_id": selected.scene_id,
                "scene_instance": str(selected.path),
                "object_count": selected.object_count,
            },
            "levels": selected_levels,
            "compact_object_ids": compact_object_ids,
            "preserve_raw_metadata": preserve_raw_metadata,
            "bbox_from_scale": bbox_from_scale,
            "include_estimated_relations": include_estimated_relations,
            "input_representation_mode": input_representation_mode,
            "stable_case_paths": [str(path) for path in stable_paths],
            "truncated": False,
        },
    )
    return selected, stable_paths, manifest_path


def _copy_to_stable_selected_name(path: Path, out_dir: Path) -> Path:
    suffix = _level_suffix(path)
    stable_path = out_dir / f"selected_{suffix}.json"
    if path.resolve() != stable_path.resolve():
        shutil.copy2(path, stable_path)
    return stable_path


def _level_suffix(path: Path) -> str:
    name = path.name
    for suffix in ["prompt_only", "structured_basic", "structured_relation"]:
        if name.endswith(f"_{suffix}.json"):
            return suffix
    return path.stem
