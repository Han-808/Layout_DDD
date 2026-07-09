from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


def resolve_asset_metadata(
    obj: dict,
    *,
    asset_csv_path: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> dict:
    """Return an object enriched with real asset metadata references.

    This resolver deliberately does not load mesh or point-cloud geometry. FBX
    and PLY files are preserved only as URI/path references.
    """

    if not isinstance(obj, dict):
        raise ValueError("object must be a mapping")
    enriched = deepcopy(obj)
    warnings: list[str] = []
    asset_ref = _dict(enriched.get("asset_ref"))
    asset_proxy = _dict(enriched.get("asset_proxy"))
    metadata = _dict(enriched.get("metadata"))
    jid = _string(enriched.get("jid") or asset_ref.get("asset_key"))
    csv_row = _asset_csv_row(asset_csv_path, jid) if jid and asset_csv_path else {}
    metadata_json = _read_asset_metadata_json(asset_root, jid, warnings) if jid and asset_root else {}

    if jid:
        enriched["jid"] = jid
        asset_ref.setdefault("source_db", "imaginarium")
        asset_ref.setdefault("asset_key", jid)

    _resolve_asset_file_uris(asset_ref, asset_root=asset_root, jid=jid, warnings=warnings)

    if metadata_json:
        _copy_if_present(asset_proxy, "bbox_center_local", metadata_json, "transformed_bbox_center")
        _copy_if_present(asset_proxy, "bbox_size", metadata_json, "transformed_size")
        _copy_if_present(asset_proxy, "point_count", metadata_json, "actual_points")
        _copy_if_present(asset_proxy, "has_color", metadata_json, "has_color")
        _copy_if_present(asset_proxy, "has_normal", metadata_json, "has_normal")
        _copy_if_present(metadata, "is_centered", metadata_json, "is_centered")
        _copy_if_present(metadata, "is_normalized", metadata_json, "is_normalized")
        _copy_if_present(metadata, "is_coordinate_transformed", metadata_json, "is_coordinate_transformed")
        asset_proxy.setdefault("type", "obb_from_metadata")

    size = (
        _size_or_none(enriched.get("size"))
        or _size_or_none(asset_proxy.get("bbox_size"))
        or _size_or_none(metadata_json.get("transformed_size"))
        or _size_or_none(csv_row.get("bbx"))
    )
    if size is None:
        identity = jid or enriched.get("id") or enriched.get("object_id") or "<unknown>"
        raise ValueError(f"Cannot resolve valid size for asset object {identity!r}. Provide size, asset_proxy.bbox_size, metadata transformed_size, or asset_info.csv bbx.")
    enriched["size"] = size
    if "bbox_size" not in asset_proxy:
        asset_proxy["bbox_size"] = size
        asset_proxy.setdefault("type", "obb_from_csv" if csv_row else "obb_from_object_size")

    desc = _first_text(enriched.get("desc"), csv_row.get("caption_en"), csv_row.get("short_desc"))
    if desc:
        enriched["desc"] = desc
    short_desc = _first_text(enriched.get("short_desc"), csv_row.get("short_desc"), enriched.get("desc"))
    if short_desc:
        enriched["short_desc"] = short_desc
    category = _first_text(enriched.get("category"), csv_row.get("class_en"), csv_row.get("retrieval_class_en"))
    if category:
        enriched["category"] = category
    retrieval_category = _first_text(enriched.get("retrieval_category"), csv_row.get("retrieval_class_en"), enriched.get("category"))
    if retrieval_category:
        enriched["retrieval_category"] = retrieval_category

    metadata.setdefault("interactive", False)
    enriched["asset_ref"] = asset_ref
    enriched["asset_proxy"] = asset_proxy
    enriched["metadata"] = metadata
    if warnings:
        enriched["asset_resolution"] = {"warnings": warnings}
    return enriched


def enrich_scene_assets(
    scene: dict,
    *,
    asset_csv_path: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> tuple[dict, dict]:
    if not isinstance(scene, dict):
        raise ValueError("scene must be a JSON object")
    enriched_scene = deepcopy(scene)
    key = "objects" if isinstance(enriched_scene.get("objects"), list) else "assets" if isinstance(enriched_scene.get("assets"), list) else None
    if key is None:
        return enriched_scene, {"object_count": 0, "enriched_count": 0, "warnings": ["scene has no objects/assets list"]}
    objects = []
    warnings: list[dict[str, Any]] = []
    enriched_count = 0
    for index, obj in enumerate(enriched_scene[key]):
        if not isinstance(obj, dict):
            continue
        try:
            resolved = resolve_asset_metadata(obj, asset_csv_path=asset_csv_path, asset_root=asset_root)
            enriched_count += 1
        except ValueError as exc:
            resolved = deepcopy(obj)
            warnings.append({"index": index, "object_id": obj.get("id") or obj.get("object_id"), "warning": str(exc)})
        if isinstance(resolved.get("asset_resolution"), dict):
            for warning in resolved["asset_resolution"].get("warnings", []):
                warnings.append({"index": index, "object_id": resolved.get("id") or resolved.get("object_id"), "warning": warning})
        objects.append(resolved)
    enriched_scene[key] = objects
    return enriched_scene, {"object_count": len(objects), "enriched_count": enriched_count, "warnings": warnings}


def _asset_csv_row(asset_csv_path: str | Path | None, jid: str) -> dict:
    if not asset_csv_path:
        return {}
    path = Path(asset_csv_path).expanduser()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if _string(row.get("name_en")) == jid:
                return dict(row)
    return {}


def _read_asset_metadata_json(asset_root: str | Path | None, jid: str, warnings: list[str]) -> dict:
    if not asset_root or not jid:
        return {}
    path = Path(asset_root).expanduser() / jid / f"{jid}_metadata.json"
    if not path.exists():
        warnings.append(f"metadata JSON not found for asset {jid!r}")
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"metadata JSON could not be read for asset {jid!r}: {exc}")
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _resolve_asset_file_uris(asset_ref: dict, *, asset_root: str | Path | None, jid: str | None, warnings: list[str]) -> None:
    if not asset_root or not jid:
        return
    root = Path(asset_root).expanduser()
    asset_dir = root / jid
    candidates = {
        "metadata_uri": asset_dir / f"{jid}_metadata.json",
        "mesh_uri": asset_dir / f"{jid}.fbx",
        "pointcloud_uri": asset_dir / f"{jid}.ply",
    }
    for key, path in candidates.items():
        if key in asset_ref and asset_ref.get(key):
            continue
        if path.exists():
            asset_ref[key] = _relative_uri(root, path)
        else:
            warnings.append(f"{key} target not found for asset {jid!r}")


def _relative_uri(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _size_or_none(value: object) -> list[float] | None:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = [part.strip() for part in text.split(",")]
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        size = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    return size if all(item > 0 for item in size) else None


def _copy_if_present(target: dict, target_key: str, source: dict, source_key: str) -> None:
    if source_key in source and source[source_key] is not None:
        target[target_key] = source[source_key]


def _dict(value: object) -> dict:
    return deepcopy(value) if isinstance(value, dict) else {}


def _first_text(*values: object) -> str | None:
    for value in values:
        text = _string(value)
        if text:
            return text
    return None


def _string(value: object) -> str:
    return str(value).strip() if value is not None else ""
