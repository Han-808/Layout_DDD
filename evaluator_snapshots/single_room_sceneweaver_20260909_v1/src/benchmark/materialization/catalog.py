from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.materialization.contracts import MaterializationError
from benchmark.materialization.geometry import finite_vec3


SUPPORTED_MESH_SUFFIXES = (".fbx", ".glb", ".gltf", ".obj")


@dataclass(frozen=True)
class FrozenCatalogAsset:
    asset_id: str
    category: str
    retrieval_category: str
    description: str
    short_description: str
    appearance_metadata: dict[str, str]
    canonical_bbox_center_m: tuple[float, float, float]
    canonical_bbox_size_m: tuple[float, float, float]
    mesh_path: Path
    metadata_path: Path | None
    hashes: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "category": self.category,
            "retrieval_category": self.retrieval_category,
            "description": self.description,
            "short_description": self.short_description,
            "appearance_metadata": dict(self.appearance_metadata),
            "canonical_bbox_center_m": list(self.canonical_bbox_center_m),
            "canonical_bbox_size_m": list(self.canonical_bbox_size_m),
            "mesh_path": self.mesh_path.as_posix(),
            "metadata_path": (
                self.metadata_path.as_posix() if self.metadata_path is not None else None
            ),
            "hashes": dict(self.hashes),
        }


class FrozenCatalog:
    """Strict, path-resolved view of one benchmark-supplied local catalog."""

    def __init__(
        self,
        *,
        asset_csv: str | Path,
        asset_root: str | Path,
        allowed_asset_ids: tuple[str, ...] | list[str] | set[str],
        snapshot_id: str,
    ) -> None:
        self.asset_csv = Path(asset_csv).expanduser().resolve()
        self.asset_root = Path(asset_root).expanduser().resolve()
        self.snapshot_id = str(snapshot_id or "").strip()
        self.allowed_asset_ids = frozenset(str(value) for value in allowed_asset_ids)
        if not self.snapshot_id:
            raise MaterializationError("frozen catalog snapshot_id must be non-empty")
        if not self.allowed_asset_ids:
            raise MaterializationError("frozen catalog allow-list must be non-empty")
        if not self.asset_csv.is_file():
            raise MaterializationError(
                f"frozen catalog metadata CSV does not exist: {self.asset_csv}"
            )
        if not self.asset_root.is_dir():
            raise MaterializationError(
                f"frozen catalog asset root does not exist: {self.asset_root}"
            )
        self._rows = self._load_rows()
        self.catalog_csv_sha256 = sha256_file(self.asset_csv)

    def resolve(self, asset_id: str) -> FrozenCatalogAsset:
        key = str(asset_id or "").strip()
        if not key:
            raise MaterializationError("asset_id must be non-empty")
        if key not in self.allowed_asset_ids:
            raise MaterializationError(
                f"asset_id {key!r} is outside frozen catalog snapshot {self.snapshot_id!r}"
            )
        row = self._rows.get(key)
        if row is None:
            raise MaterializationError(
                f"allow-listed asset_id {key!r} is missing from frozen catalog metadata"
            )
        asset_dir = (self.asset_root / key).resolve()
        try:
            asset_dir.relative_to(self.asset_root)
        except ValueError as exc:
            raise MaterializationError(f"asset_id {key!r} escapes the frozen asset root") from exc
        metadata_path = asset_dir / f"{key}_metadata.json"
        metadata = _read_metadata(metadata_path) if metadata_path.is_file() else {}
        bbox_size = _first_vec3(
            metadata.get("transformed_size"),
            row.get("bbx"),
            path=f"catalog[{key}].canonical_bbox_size_m",
            positive=True,
        )
        bbox_center = _first_vec3(
            metadata.get("transformed_bbox_center"),
            [0.0, 0.0, 0.0],
            path=f"catalog[{key}].canonical_bbox_center_m",
        )
        category = _first_text(
            row.get("category"),
            row.get("class_en"),
            row.get("retrieval_class_en"),
        )
        retrieval_category = _first_text(row.get("retrieval_class_en"), category)
        description = _first_text(
            row.get("caption_en"),
            row.get("short_desc"),
            category,
        )
        short_description = _first_text(row.get("short_desc"), description)
        appearance_metadata = {
            key: str(row.get(key) or "").strip()
            for key in (
                "picture",
                "source",
                "author_name",
                "author_link",
                "license",
                "state",
            )
            if str(row.get(key) or "").strip()
        }
        if not category or not description:
            raise MaterializationError(
                f"frozen catalog asset {key!r} lacks category or description"
            )
        mesh_path = _resolve_mesh(asset_dir, key)
        hashes = {
            "mesh_sha256": sha256_file(mesh_path),
            "catalog_csv_sha256": self.catalog_csv_sha256,
            "asset_tree_sha256": sha256_asset_tree(asset_dir),
        }
        if metadata_path.is_file():
            hashes["metadata_sha256"] = sha256_file(metadata_path)
        return FrozenCatalogAsset(
            asset_id=key,
            category=category,
            retrieval_category=retrieval_category or category,
            description=description,
            short_description=short_description or description,
            appearance_metadata=appearance_metadata,
            canonical_bbox_center_m=tuple(bbox_center),
            canonical_bbox_size_m=tuple(bbox_size),
            mesh_path=mesh_path,
            metadata_path=metadata_path if metadata_path.is_file() else None,
            hashes=hashes,
        )

    def _load_rows(self) -> dict[str, dict[str, str]]:
        rows: dict[str, dict[str, str]] = {}
        with self.asset_csv.open("r", encoding="utf-8-sig", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=2):
                key = str(row.get("name_en") or "").strip()
                if not key:
                    continue
                if key in rows:
                    raise MaterializationError(
                        f"frozen catalog metadata contains duplicate asset_id {key!r} "
                        f"at CSV row {index}"
                    )
                rows[key] = {str(name): str(value or "") for name, value in row.items()}
        return rows


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_asset_tree(asset_dir: str | Path) -> str:
    """Hash every regular file in one frozen asset dependency directory."""

    root = Path(asset_dir).expanduser().resolve()
    if not root.is_dir():
        raise MaterializationError(
            f"frozen asset dependency root does not exist: {root}"
        )
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise MaterializationError(
                f"frozen asset dependency tree contains a symbolic link: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise MaterializationError(
                f"frozen asset dependency tree contains a non-regular file: {path}"
            )
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not entries:
        raise MaterializationError(
            f"frozen asset dependency tree is empty: {root}"
        )
    return sha256_json(entries)


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"cannot read frozen asset metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MaterializationError(f"frozen asset metadata must be a JSON object: {path}")
    return value


def _resolve_mesh(asset_dir: Path, asset_id: str) -> Path:
    candidates = [
        asset_dir / f"{asset_id}{suffix}" for suffix in SUPPORTED_MESH_SUFFIXES
    ]
    existing = [path.resolve() for path in candidates if path.is_file()]
    if not existing:
        raise MaterializationError(
            f"frozen catalog asset {asset_id!r} has no supported local rigid mesh"
        )
    # The format order is part of the materializer revision and makes selection
    # deterministic when a catalog intentionally stores multiple encodings.
    return existing[0]


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _first_vec3(
    *values: Any,
    path: str,
    positive: bool = False,
) -> list[float]:
    for value in values:
        parsed: Any = value
        if isinstance(parsed, str):
            text = parsed.strip()
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = [part.strip() for part in text.split(",")]
        try:
            return finite_vec3(parsed, path, positive=positive)
        except MaterializationError:
            continue
    raise MaterializationError(f"{path} is missing or invalid")
