from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


LOCAL_SCENE_SOURCE = "local_repo"
DEFAULT_SCENE_COLLECTION = "Scenes"
DEFAULT_SCENE_ROOT = Path(DEFAULT_SCENE_COLLECTION)


@dataclass(frozen=True)
class LocalScene:
    scene_id: str
    scene_type: str
    collection: str
    repo_path: str
    scene_json_path: str
    asset_count: int
    boundary: list | None
    scene_height: float | None
    metadata: dict[str, Any]

    def to_scene_ref(self) -> dict:
        ref = {
            "source": LOCAL_SCENE_SOURCE,
            "collection": self.collection,
            "scene_id": self.scene_id,
            "repo_path": self.repo_path,
            "scene_json_path": self.scene_json_path,
            "asset_count": self.asset_count,
        }
        if self.scene_type:
            ref["scene_type"] = self.scene_type
        if self.boundary is not None:
            ref["boundary"] = self.boundary
        if self.scene_height is not None:
            ref["scene_height"] = self.scene_height
        if self.metadata:
            ref["metadata"] = self.metadata
        return ref


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_local_scene_ref(scene_ref: object = None, *, scene_id: object = None, root: str | Path | None = None) -> dict | None:
    """Return a local-repo scene_ref when the requested scene JSON exists."""

    root_path = Path(root) if root is not None else project_root()
    if isinstance(scene_ref, dict) and scene_ref.get("source") not in {None, "", LOCAL_SCENE_SOURCE}:
        return None
    requested = _requested_scene_id(scene_ref, scene_id)
    scene = _resolve_requested_scene(root_path, requested, scene_ref)
    if scene is None:
        return None
    existing = dict(scene_ref) if isinstance(scene_ref, dict) else {}
    ref = scene.to_scene_ref()
    ref.update({key: value for key, value in existing.items() if value is not None})
    existing_metadata = existing.get("metadata") if isinstance(existing.get("metadata"), dict) else {}
    if scene.metadata or existing_metadata:
        ref["metadata"] = {**scene.metadata, **existing_metadata}
    return ref


def load_local_scene(scene_ref: object = None, *, scene_id: object = None, root: str | Path | None = None) -> dict | None:
    """Load a local scene JSON and attach its resolved scene_ref."""

    root_path = Path(root) if root is not None else project_root()
    ref = resolve_local_scene_ref(scene_ref, scene_id=scene_id, root=root_path)
    if not ref:
        return None
    scene_json_path = ref.get("scene_json_path")
    if not isinstance(scene_json_path, str) or not scene_json_path:
        return None
    loaded = _read_json(root_path / scene_json_path)
    if not isinstance(loaded, dict):
        return None
    loaded["scene_ref"] = ref
    source = loaded.get("source") if isinstance(loaded.get("source"), dict) else {}
    source["scene_ref"] = ref
    loaded["source"] = source
    return loaded


@lru_cache(maxsize=4)
def load_local_scene_index(root: Path | str | None = None) -> dict[str, LocalScene]:
    root_path = Path(root) if root is not None else project_root()
    scene_root = root_path / DEFAULT_SCENE_ROOT
    scenes: dict[str, LocalScene] = {}
    if not scene_root.exists():
        return scenes
    for path in sorted(scene_root.rglob("*.json")):
        if any(part.startswith(".") for part in path.relative_to(scene_root).parts):
            continue
        scene = _scene_from_file(path, root_path)
        if scene is None:
            continue
        scenes.setdefault(scene.scene_id, scene)
        scenes.setdefault(Path(scene.scene_json_path).stem, scene)
        scenes.setdefault(scene.scene_json_path, scene)
    return scenes


def _resolve_requested_scene(root: Path, requested: str, scene_ref: object) -> LocalScene | None:
    index = load_local_scene_index(root)
    if requested and requested in index:
        return index[requested]
    path = _requested_scene_path(scene_ref)
    if path:
        candidates = [path]
        if not path.is_absolute():
            candidates.append(root / path)
            candidates.append(root / DEFAULT_SCENE_ROOT / path)
        for candidate in candidates:
            scene = _scene_from_file(candidate, root)
            if scene is not None:
                return scene
    if requested:
        requested_path = Path(requested)
        if requested_path.suffix == ".json":
            for candidate in [requested_path, root / requested_path, root / DEFAULT_SCENE_ROOT / requested_path]:
                scene = _scene_from_file(candidate, root)
                if scene is not None:
                    return scene
    return None


def _scene_from_file(path: Path, root: Path) -> LocalScene | None:
    if not path.exists() or path.suffix.lower() != ".json":
        return None
    loaded = _read_json(path)
    if not isinstance(loaded, dict):
        return None
    objects = loaded.get("objects") if isinstance(loaded.get("objects"), list) else []
    assets = loaded.get("assets") if isinstance(loaded.get("assets"), list) else []
    scene_id = _first_nonempty_str(loaded.get("scene_id"), path.stem)
    scene_type = _first_nonempty_str(loaded.get("scene_type"), loaded.get("room_type"), loaded.get("type"))
    scene_height = _float_or_none(loaded.get("scene_height") or loaded.get("wall_height"))
    metadata = {
        key: loaded.get(key)
        for key in ["scene_type", "room_type", "dataset", "split", "generator", "source"]
        if loaded.get(key) is not None
    }
    return LocalScene(
        scene_id=scene_id,
        scene_type=scene_type,
        collection=DEFAULT_SCENE_COLLECTION,
        repo_path=_repo_relative(path.parent, root),
        scene_json_path=_repo_relative(path, root),
        asset_count=len(assets) if assets else len(objects),
        boundary=loaded.get("boundary") if isinstance(loaded.get("boundary"), list) else None,
        scene_height=scene_height,
        metadata=metadata,
    )


def _requested_scene_id(scene_ref: object, scene_id: object) -> str:
    if isinstance(scene_ref, dict):
        for key in ["scene_id", "id", "name", "scene_name"]:
            value = scene_ref.get(key)
            if isinstance(value, str) and value:
                return Path(value).stem if value.endswith(".json") else value
        path = _requested_scene_path(scene_ref)
        if path is not None:
            return path.stem if path.suffix else path.name
    if isinstance(scene_id, str) and scene_id:
        return Path(scene_id).stem if scene_id.endswith(".json") else scene_id
    return ""


def _requested_scene_path(scene_ref: object) -> Path | None:
    if not isinstance(scene_ref, dict):
        return None
    for key in ["scene_json_path", "repo_path", "path", "source_path"]:
        value = scene_ref.get(key)
        if isinstance(value, str) and value:
            path = Path(value)
            if path.suffix == ".json":
                return path
    return None


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_nonempty_str(*values: object, default: str = "") -> str:
    for value in values:
        if isinstance(value, str) and value:
            return value
        if value is not None and value != "" and not isinstance(value, (dict, list, tuple, set)):
            return str(value)
    return default


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()
