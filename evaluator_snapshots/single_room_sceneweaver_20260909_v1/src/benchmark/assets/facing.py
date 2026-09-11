"""Benchmark-owned facing convention for the local Imaginarium catalog.

The catalog does not require generators to repeat this information.  It is a
versioned normalization contract shared by generator prompts and evaluator
adapters.  Scene transforms remain entirely generator-owned.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Sequence


CATALOG_FACING_CONTRACT_VERSION = "imaginarium_catalog_facing_v1"
DEFAULT_DIRECTED_FUNCTIONAL_SIDE = "local_neg_y"
IMAGINARIUM_SOURCE_DATABASES = frozenset(
    {
        "imaginarium",
        "imaginarium_catalog",
        "frozen_imaginarium_catalog",
    }
)
MATERIALIZED_FIXED_CATALOG_SOURCE_DB = "fixed_catalog"
IMAGINARIUM_CATALOG_SNAPSHOT_PREFIX = "imaginarium_catalog_"
LOCAL_SIDE_XY = {
    "local_pos_x": (1.0, 0.0),
    "local_neg_x": (-1.0, 0.0),
    "local_pos_y": (0.0, 1.0),
    "local_neg_y": (0.0, -1.0),
}


def benchmark_catalog_facing_contract() -> dict[str, Any]:
    """Return the immutable generator/evaluator-facing contract payload."""

    return {
        "contract_version": CATALOG_FACING_CONTRACT_VERSION,
        "scope": {
            "source_db": sorted(IMAGINARIUM_SOURCE_DATABASES),
            "materialized_source_db": MATERIALIZED_FIXED_CATALOG_SOURCE_DB,
            "materialized_provenance_requirement": (
                "metadata.materialization.catalog_snapshot_id starts with "
                f"{IMAGINARIUM_CATALOG_SNAPSHOT_PREFIX!r}"
            ),
            "asset_policy": "directed_assets_only",
        },
        "default_directed_functional_side": (
            DEFAULT_DIRECTED_FUNCTIONAL_SIDE
        ),
        "non_directed_policy": "no_facing_constraint",
        "exception_policy": "benchmark_owned_explicit_asset_override",
        "yaw_semantics": (
            "rotation_euler_xyz_deg[2] rotates the canonical local side "
            "into the desired world heading"
        ),
        "cardinal_yaw_examples_deg": {
            "face_world_neg_y": 0.0,
            "face_world_pos_x": 90.0,
            "face_world_pos_y": 180.0,
            "face_world_neg_x": -90.0,
        },
    }


def yaw_degrees_for_world_heading(
    desired_world_heading_xy: Sequence[float],
    *,
    local_side_id: str = DEFAULT_DIRECTED_FUNCTIONAL_SIDE,
) -> float:
    """Rotate ``local_side_id`` onto a non-zero world-space XY heading."""

    if local_side_id not in LOCAL_SIDE_XY:
        raise ValueError(f"unsupported local side ID: {local_side_id!r}")
    if (
        not isinstance(desired_world_heading_xy, Sequence)
        or isinstance(desired_world_heading_xy, (str, bytes))
        or len(desired_world_heading_xy) != 2
    ):
        raise ValueError("desired_world_heading_xy must contain two numbers")
    try:
        world_x = float(desired_world_heading_xy[0])
        world_y = float(desired_world_heading_xy[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "desired_world_heading_xy must contain two finite numbers"
        ) from exc
    if (
        not math.isfinite(world_x)
        or not math.isfinite(world_y)
        or math.hypot(world_x, world_y) <= 1e-12
    ):
        raise ValueError(
            "desired_world_heading_xy must be finite and non-zero"
        )
    local_x, local_y = LOCAL_SIDE_XY[local_side_id]
    yaw = math.degrees(
        math.atan2(world_y, world_x) - math.atan2(local_y, local_x)
    )
    normalized = ((yaw + 180.0) % 360.0) - 180.0
    if math.isclose(normalized, -180.0, abs_tol=1e-9):
        normalized = 180.0
    if math.isclose(normalized, 0.0, abs_tol=1e-12):
        normalized = 0.0
    return float(normalized)


def normalize_catalog_facing_overrides(
    value: dict[str, Any] | None,
) -> dict[str, str | dict[str, str]]:
    """Validate benchmark-owned per-asset or per-role side overrides."""

    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("catalog_side_overrides must be a JSON object")
    normalized: dict[str, str | dict[str, str]] = {}
    for raw_asset_key, raw_override in value.items():
        asset_key = str(raw_asset_key).strip()
        if not asset_key:
            raise ValueError("catalog_side_overrides contains a blank asset key")
        if isinstance(raw_override, str):
            normalized[asset_key] = _validate_side_id(raw_override)
            continue
        if not isinstance(raw_override, dict) or not raw_override:
            raise ValueError(
                "catalog side override must be a side ID or a non-empty "
                "surface-role mapping"
            )
        role_map: dict[str, str] = {}
        for raw_role, raw_side in raw_override.items():
            role = str(raw_role).strip()
            if not role:
                raise ValueError(
                    "catalog side override contains a blank surface role"
                )
            role_map[role] = _validate_side_id(raw_side)
        normalized[asset_key] = role_map
    return normalized


def resolve_catalog_functional_side(
    object_record: dict[str, Any],
    *,
    surface_roles: Sequence[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Resolve a directed side for an in-scope catalog asset.

    ``None`` means the deterministic contract is not applicable or a set of
    explicit role overrides disagrees.  The evaluator must then retain its
    existing visual decoder path.
    """

    if not isinstance(object_record, dict):
        return None
    asset_ref = object_record.get("asset_ref")
    if not isinstance(asset_ref, dict):
        return None
    source_db = str(asset_ref.get("source_db") or "").strip().lower()
    asset_key = str(
        asset_ref.get("asset_key") or object_record.get("jid") or ""
    ).strip()
    source_resolution = _imaginarium_source_resolution(
        object_record,
        source_db=source_db,
    )
    if source_resolution is None or not asset_key:
        return None

    normalized_overrides = normalize_catalog_facing_overrides(overrides)
    role_values = [
        str(role).strip()
        for role in surface_roles or []
        if str(role).strip()
    ]
    selected_side = DEFAULT_DIRECTED_FUNCTIONAL_SIDE
    resolution_source = "catalog_default"
    override = normalized_overrides.get(asset_key)
    if isinstance(override, str):
        selected_side = override
        resolution_source = "explicit_asset_override"
    elif isinstance(override, dict):
        selected: list[str] = []
        for role in role_values:
            side = override.get(role, override.get("*"))
            if side is not None and side not in selected:
                selected.append(side)
        if not selected and "*" in override:
            selected.append(override["*"])
        if len(selected) > 1:
            return None
        if selected:
            selected_side = selected[0]
            resolution_source = "explicit_role_override"

    return {
        "contract_version": CATALOG_FACING_CONTRACT_VERSION,
        "source_db": source_db,
        "source_resolution": source_resolution,
        "asset_key": asset_key,
        "side_id": selected_side,
        "resolution_source": resolution_source,
        "surface_roles": role_values,
        "scene_access": "read_only",
        "transform_mutation": False,
    }


def catalog_facing_cache_manifest(
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the reproducibility payload included in detector manifests."""

    normalized = normalize_catalog_facing_overrides(overrides)
    return {
        "contract": benchmark_catalog_facing_contract(),
        "overrides": deepcopy(normalized),
    }


def _validate_side_id(value: Any) -> str:
    side_id = str(value or "").strip()
    if side_id not in LOCAL_SIDE_XY:
        raise ValueError(f"unsupported catalog functional side: {side_id!r}")
    return side_id


def _imaginarium_source_resolution(
    object_record: dict[str, Any],
    *,
    source_db: str,
) -> str | None:
    if source_db in IMAGINARIUM_SOURCE_DATABASES:
        return "canonical_asset_ref"
    if source_db != MATERIALIZED_FIXED_CATALOG_SOURCE_DB:
        return None
    metadata = object_record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    materialization = metadata.get("materialization")
    materialization = (
        materialization if isinstance(materialization, dict) else {}
    )
    snapshot_id = str(
        materialization.get("catalog_snapshot_id") or ""
    ).strip().lower()
    if snapshot_id.startswith(IMAGINARIUM_CATALOG_SNAPSHOT_PREFIX):
        return "materialized_catalog_snapshot_provenance"
    return None
