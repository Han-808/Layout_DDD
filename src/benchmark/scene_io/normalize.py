from __future__ import annotations

from copy import deepcopy
from pathlib import Path

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
        normalized["objects"] = normalized["assets"]
    if enrich_assets:
        normalized["objects"] = [
            normalize_object(obj, asset_csv=asset_csv, asset_root=asset_root, enrich_assets=True)
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
    if "center" not in normalized and isinstance(normalized.get("pose"), dict):
        normalized["center"] = normalized["pose"].get("center")
    if "rotation" not in normalized and isinstance(normalized.get("pose"), dict):
        normalized["rotation"] = normalized["pose"].get("rotation", [0, 0, 0])
    if "rotation" not in normalized:
        normalized["rotation"] = [0, 0, 0]
    if enrich_assets:
        normalized = enrich_object_with_asset_metadata(normalized, asset_csv_path=asset_csv, asset_root=asset_root)
    if "asset_ref" not in normalized:
        jid = normalized.get("jid")
        normalized["asset_ref"] = {"source_db": "imaginarium", "asset_key": jid} if jid else {}
    if "metadata" not in normalized or not isinstance(normalized.get("metadata"), dict):
        normalized["metadata"] = {}
    normalized["metadata"].setdefault("interactive", False)
    return normalized
