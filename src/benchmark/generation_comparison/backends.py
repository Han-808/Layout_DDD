"""Concrete loaders for immutable catalog snapshots.

Backends normalize already-selected records. They never search, rank, or
retrieve assets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.generation_comparison.catalog import CanonicalAssetCatalog
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


def load_3d_future_subset_catalog(
    index_path: str | Path,
    *,
    catalog_id: str,
    catalog_version: str,
    asset_root: str | Path | None = None,
    hash_local_meshes: bool = True,
) -> CanonicalAssetCatalog:
    """Normalize one explicitly frozen 3D-FUTURE subset into the common catalog.

    The source file is a selection manifest, not a retrieval index. Every entry
    must already identify a JID and bbox. ``asset_root`` only resolves the
    standard ``<jid>/raw_model.obj`` path when no mesh path is recorded.
    """

    source = Path(index_path).expanduser().resolve()
    loaded = read_json(source)
    records = loaded.get("assets") if isinstance(loaded, Mapping) else loaded
    if isinstance(records, Mapping):
        items = list(records.items())
    elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        items = [(str(index), item) for index, item in enumerate(records)]
    else:
        raise ArtifactValidationError(
            "3D-FUTURE subset manifest must be a list, mapping, or contain assets"
        )
    root = Path(asset_root).expanduser().resolve() if asset_root is not None else None
    canonical: list[dict[str, Any]] = []
    for index, (fallback_id, raw) in enumerate(items):
        if not isinstance(raw, Mapping):
            raise ArtifactValidationError(
                f"3D-FUTURE subset entry {index} must be an object"
            )
        item = dict(raw)
        jid = _first_text(
            item.get("asset_id"),
            item.get("jid"),
            item.get("uid"),
            fallback_id,
        )
        category = _first_text(
            item.get("category"), item.get("class"), item.get("object_type")
        )
        description = _first_text(
            item.get("description"), item.get("desc"), item.get("name"), category
        )
        bbox = _bbox_size(item)
        center = item.get("bbox_center_local") or item.get("bbox_center")
        if center is None:
            raise ArtifactValidationError(
                f"3D-FUTURE subset entry {jid!r} lacks bbox_center_local"
            )
        mesh = _first_text(
            item.get("mesh_uri"),
            item.get("mesh_path"),
            item.get("model_path"),
            item.get("path"),
        )
        if mesh and "://" not in mesh:
            mesh_path = Path(mesh).expanduser()
            if not mesh_path.is_absolute():
                mesh = (source.parent / mesh_path).resolve().as_posix()
        if not mesh and root is not None:
            mesh = (root / jid / "raw_model.obj").as_posix()
        record: dict[str, Any] = {
            "asset_id": jid,
            "source_db": "3d_future",
            "category": category,
            "description": description,
            "bbox_size_local": bbox,
            "bbox_center_local": center,
            "native_scale": item.get("native_scale", item.get("scale", 1.0)),
            "metadata": {
                **(
                    dict(item["metadata"])
                    if isinstance(item.get("metadata"), Mapping)
                    else {}
                ),
                "pilot_backend": "3d_future_subset_v1",
            },
        }
        if mesh:
            record["mesh_uri"] = mesh
        if item.get("canonical_front") is not None:
            record["canonical_front"] = item["canonical_front"]
        if item.get("physical_dimensions") is not None:
            record["physical_dimensions"] = item["physical_dimensions"]
        if isinstance(item.get("content"), Mapping):
            record["content"] = dict(item["content"])
        canonical.append(record)
    return CanonicalAssetCatalog.from_mapping(
        {
            "catalog_id": catalog_id,
            "catalog_version": catalog_version,
            "assets": canonical,
            "metadata": {
                "backend": "3d_future_subset_v1",
            },
        },
        hash_local_meshes=hash_local_meshes,
        base_dir=source.parent,
    )


def _bbox_size(item: Mapping[str, Any]) -> Any:
    for key in ("bbox_size_local", "bbox_size", "size"):
        if item.get(key) is not None:
            return item[key]
    metadata = item.get("assetMetadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    bbox = metadata.get("boundingBox")
    if isinstance(bbox, Mapping):
        return [bbox.get("x"), bbox.get("y"), bbox.get("z")]
    raise ArtifactValidationError(
        f"3D-FUTURE subset entry {_first_text(item.get('jid'), item.get('asset_id'))!r} "
        "lacks bbox_size_local"
    )


def _first_text(*values: Any) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


__all__ = ["load_3d_future_subset_catalog"]
