from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


LOCAL_ASSET_SOURCE = "local_repo"
DEFAULT_ASSET_COLLECTION = "imaginarium_assets"
DEFAULT_ASSET_ROOT = Path("Assets") / DEFAULT_ASSET_COLLECTION
ASSET_INFO_CSV = DEFAULT_ASSET_ROOT / "imaginarium_asset_info.csv"


@dataclass(frozen=True)
class LocalAsset:
    asset_id: str
    category: str
    repo_path: str
    mesh_path: str
    pointcloud_path: str
    metadata_path: str
    dimensions: list[float] | None
    metadata: dict[str, Any]

    def to_asset_ref(self) -> dict:
        ref = {
            "source": LOCAL_ASSET_SOURCE,
            "collection": DEFAULT_ASSET_COLLECTION,
            "asset_id": self.asset_id,
            "repo_path": self.repo_path,
            "mesh_path": self.mesh_path,
            "pointcloud_path": self.pointcloud_path,
            "metadata_path": self.metadata_path,
            "category": self.category,
        }
        if self.dimensions:
            ref["dimensions"] = list(self.dimensions)
        summary = {
            key: self.metadata.get(key)
            for key in ["name_en", "caption_en", "class_en", "retrieval_class_en", "license", "short_desc"]
            if self.metadata.get(key)
        }
        if summary:
            ref["metadata"] = summary
        return ref


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_local_asset_ref(asset_ref: object = None, *, asset_id: object = None, root: str | Path | None = None) -> dict | None:
    """Return a local-repo asset_ref if the requested asset exists."""

    root_path = Path(root) if root is not None else project_root()
    if isinstance(asset_ref, dict) and asset_ref.get("source") not in {None, "", LOCAL_ASSET_SOURCE}:
        return None
    requested = _requested_asset_id(asset_ref, asset_id)
    if not requested:
        return None
    asset = load_local_asset_index(root_path).get(requested)
    if asset is None:
        return None
    existing = dict(asset_ref) if isinstance(asset_ref, dict) else {}
    ref = asset.to_asset_ref()
    ref.update({key: value for key, value in existing.items() if value is not None})
    existing_metadata = existing.get("metadata") if isinstance(existing.get("metadata"), dict) else {}
    local_metadata = asset.to_asset_ref().get("metadata") if isinstance(asset.to_asset_ref().get("metadata"), dict) else {}
    if local_metadata or existing_metadata:
        ref["metadata"] = {**local_metadata, **existing_metadata}
    return ref


@lru_cache(maxsize=4)
def load_local_asset_index(root: Path | str | None = None) -> dict[str, LocalAsset]:
    root_path = Path(root) if root is not None else project_root()
    csv_path = root_path / ASSET_INFO_CSV
    asset_root = root_path / DEFAULT_ASSET_ROOT
    assets: dict[str, LocalAsset] = {}
    if not csv_path.exists() or not asset_root.exists():
        return assets
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = str(row.get("name_en") or "").strip()
            if not name:
                continue
            folder = asset_root / name
            mesh = folder / f"{name}.fbx"
            pointcloud = folder / f"{name}.ply"
            metadata_path = folder / f"{name}_metadata.json"
            if not folder.exists():
                continue
            file_metadata = _read_metadata(metadata_path)
            dimensions = _dimensions_from_metadata(file_metadata) or _parse_dimensions(row.get("bbx"))
            category = str(row.get("category") or row.get("retrieval_class_en") or row.get("class_en") or "object").strip() or "object"
            metadata = {key: value for key, value in row.items() if value not in {None, ""}}
            if file_metadata:
                metadata["file_metadata"] = file_metadata
            assets[name] = LocalAsset(
                asset_id=name,
                category=category,
                repo_path=_repo_relative(folder, root_path),
                mesh_path=_repo_relative(mesh, root_path) if mesh.exists() else "",
                pointcloud_path=_repo_relative(pointcloud, root_path) if pointcloud.exists() else "",
                metadata_path=_repo_relative(metadata_path, root_path) if metadata_path.exists() else "",
                dimensions=dimensions,
                metadata=metadata,
            )
    return assets


def _requested_asset_id(asset_ref: object, asset_id: object) -> str:
    if isinstance(asset_ref, dict):
        for key in ["asset_id", "template_id", "name", "repo_asset_id"]:
            value = asset_ref.get(key)
            if isinstance(value, str) and value:
                return Path(value).name
        repo_path = asset_ref.get("repo_path") or asset_ref.get("mesh_path") or asset_ref.get("pointcloud_path")
        if isinstance(repo_path, str) and repo_path:
            path = Path(repo_path)
            return path.parent.name if path.suffix else path.name
    if isinstance(asset_id, str) and asset_id:
        return asset_id
    return ""


def _read_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _dimensions_from_metadata(metadata: dict) -> list[float] | None:
    value = metadata.get("transformed_size") if isinstance(metadata, dict) else None
    return _parse_dimensions(value)


def _parse_dimensions(value: object) -> list[float] | None:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return None
    if len(parts) < 3:
        return None
    try:
        dims = [float(parts[index]) for index in range(3)]
    except (TypeError, ValueError):
        return None
    return dims if all(item > 0 for item in dims) else None


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()
