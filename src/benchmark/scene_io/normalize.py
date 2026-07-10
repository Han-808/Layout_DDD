from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from benchmark.scene_io.assets import enrich_object_with_asset_metadata
from benchmark.scene_io.validate import validate_generated_scene


def normalize_scene(
    scene: dict,
    *,
    asset_csv: str | Path | None = None,
    asset_root: str | Path | None = None,
    enrich_assets: bool = False,
) -> dict:
    if not isinstance(scene, dict):
        raise ValueError("scene must be a JSON object")
    normalized = deepcopy(scene)
    if "boundary" not in normalized and isinstance(normalized.get("room"), dict):
        normalized["boundary"] = normalized["room"].get("boundary")
    if "scene_height" not in normalized and isinstance(normalized.get("room"), dict):
        normalized["scene_height"] = normalized["room"].get("height")
    if "objects" not in normalized and isinstance(normalized.get("assets"), list):
        normalized["objects"] = [_legacy_asset_to_object(item, index) for index, item in enumerate(normalized["assets"]) if isinstance(item, dict)]
    normalized.pop("assets", None)
    normalized["objects"] = [
        normalize_object(obj, asset_csv=asset_csv, asset_root=asset_root, enrich_assets=enrich_assets)
        for obj in normalized.get("objects", [])
        if isinstance(obj, dict)
    ]
    validate_generated_scene(normalized)
    return normalized


def normalize_object(
    obj: dict,
    *,
    asset_csv: str | Path | None = None,
    asset_root: str | Path | None = None,
    enrich_assets: bool = False,
) -> dict:
    normalized = deepcopy(obj)
    if enrich_assets:
        normalized = enrich_object_with_asset_metadata(normalized, asset_csv_path=asset_csv, asset_root=asset_root)
    jid = normalized.get("jid")
    asset_ref = normalized.get("asset_ref") if isinstance(normalized.get("asset_ref"), dict) else {}
    if jid is None:
        jid = asset_ref.get("asset_key")
    if jid is not None:
        normalized["jid"] = str(jid)
    asset_ref = _canonical_asset_ref(asset_ref, jid)
    normalized["asset_ref"] = asset_ref
    asset_proxy = normalized.get("asset_proxy") if isinstance(normalized.get("asset_proxy"), dict) else {}
    if "bbox_size" not in asset_proxy and isinstance(normalized.get("size"), list):
        asset_proxy["bbox_size"] = normalized["size"]
    asset_proxy.setdefault("type", "obb_from_metadata_or_csv")
    asset_proxy.setdefault("bbox_center_local", [0, 0, 0])
    normalized["asset_proxy"] = asset_proxy
    metadata = normalized.get("metadata") if isinstance(normalized.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata["interactive"] = bool(normalized.get("interactive") or metadata.get("interactive") or False)
    normalized["metadata"] = metadata
    if "rotation" not in normalized:
        normalized["rotation"] = [0, 0, 0]
    return normalized


def _legacy_asset_to_object(asset: dict, index: int) -> dict:
    placement = asset.get("placement") if isinstance(asset.get("placement"), dict) else {}
    asset_ref = asset.get("asset_ref") if isinstance(asset.get("asset_ref"), dict) else {}
    jid = asset.get("jid") or asset_ref.get("asset_key") or asset_ref.get("asset_id") or asset.get("asset_id")
    center = asset.get("center") or asset.get("position") or placement.get("center") or placement.get("position")
    rotation = asset.get("rotation") or placement.get("rotation_euler_degrees")
    if rotation is None and placement.get("yaw_degrees") is not None:
        rotation = [0, 0, placement.get("yaw_degrees")]
    size = asset.get("size") or asset.get("dimensions") or asset_ref.get("dimensions")
    return {
        "id": str(asset.get("id") or asset.get("object_id") or f"obj_{index:03d}"),
        "jid": str(jid) if jid is not None else "",
        "category": asset.get("category") or asset_ref.get("category") or "object",
        "retrieval_category": asset.get("retrieval_category") or asset.get("category") or asset_ref.get("category") or "object",
        "desc": asset.get("desc") or asset.get("description") or asset_ref.get("caption_en") or "",
        "short_desc": asset.get("short_desc") or asset.get("desc") or asset.get("description") or "",
        "size": size,
        "center": center,
        "rotation": rotation or [0, 0, 0],
        "asset_ref": _canonical_asset_ref(asset_ref, jid),
        "asset_proxy": asset.get("asset_proxy") if isinstance(asset.get("asset_proxy"), dict) else {
            "type": "obb_from_metadata_or_csv",
            "bbox_center_local": [0, 0, 0],
            "bbox_size": size,
        },
        "metadata": asset.get("metadata") if isinstance(asset.get("metadata"), dict) else {"interactive": False},
    }


def _canonical_asset_ref(asset_ref: dict, jid: Any) -> dict:
    ref = dict(asset_ref) if isinstance(asset_ref, dict) else {}
    if "source" in ref and "source_db" not in ref:
        ref["source_db"] = ref.pop("source")
    ref.setdefault("source_db", "imaginarium")
    if jid is not None:
        ref.setdefault("asset_key", str(jid))
    if "metadata_path" in ref and "metadata_uri" not in ref:
        ref["metadata_uri"] = ref["metadata_path"]
    if "pointcloud_path" in ref and "pointcloud_uri" not in ref:
        ref["pointcloud_uri"] = ref["pointcloud_path"]
    if "mesh_path" in ref and "mesh_uri" not in ref:
        ref["mesh_uri"] = ref["mesh_path"]
    ref.setdefault("mesh_uri", None)
    ref.setdefault("pointcloud_uri", None)
    ref.setdefault("metadata_uri", None)
    return ref
