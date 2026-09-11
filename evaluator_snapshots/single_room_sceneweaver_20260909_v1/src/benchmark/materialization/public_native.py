from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping

from benchmark.materialization.contracts import MaterializationError
from benchmark.materialization.geometry import finite_vec3
from benchmark.materialization.native_registry import (
    NativeRegistryAuthority,
    write_benchmark_native_registry,
)
from benchmark.utils.io import read_json


PUBLIC_NATIVE_MAPPING_VERSION = "public_native_instance_mapping_v1"

_ROOT_FIELDS = frozenset({"schema_version", "instances"})
_REQUIRED_INSTANCE_FIELDS = frozenset(
    {
        "instance_id",
        "asset_id",
        "native_root_name",
        "center_m",
        "uniform_scale",
        "rotation_euler_xyz_deg",
    }
)
_ALLOWED_INSTANCE_FIELDS = _REQUIRED_INSTANCE_FIELDS | {"slot_id"}


def load_public_native_instance_mapping(
    path: str | Path,
) -> dict[str, Any]:
    """Load the small, unsigned identity/placement map a submitter may write.

    The public mapping deliberately contains no benchmark seal, evaluator ID,
    source hash, or geometry/material fingerprint. Those values are derived by
    the trusted read-only inspector.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise MaterializationError(
            f"public native instance mapping does not exist: {resolved}"
        )
    try:
        value = read_json(resolved)
    except Exception as exc:
        raise MaterializationError(
            f"public native instance mapping is not valid JSON: {exc}"
        ) from exc
    return validate_public_native_instance_mapping(value)


def validate_public_native_instance_mapping(
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MaterializationError(
            "public native instance mapping must be a JSON object"
        )
    result = deepcopy(dict(value))
    missing_root = sorted(_ROOT_FIELDS - set(result))
    extra_root = sorted(set(result) - _ROOT_FIELDS)
    if missing_root or extra_root:
        raise MaterializationError(
            "public native instance mapping has invalid root fields: "
            f"missing={missing_root}, extra={extra_root}"
        )
    if result.get("schema_version") != PUBLIC_NATIVE_MAPPING_VERSION:
        raise MaterializationError(
            "public native instance mapping schema_version must be "
            f"{PUBLIC_NATIVE_MAPPING_VERSION!r}"
        )
    instances = result.get("instances")
    if not isinstance(instances, list) or not instances:
        raise MaterializationError(
            "public native instance mapping instances must be a non-empty list"
        )

    normalized: list[dict[str, Any]] = []
    instance_ids: set[str] = set()
    root_names: set[str] = set()
    for index, raw in enumerate(instances):
        if not isinstance(raw, Mapping):
            raise MaterializationError(
                f"public native mapping instances[{index}] must be an object"
            )
        item = dict(raw)
        missing = sorted(_REQUIRED_INSTANCE_FIELDS - set(item))
        extra = sorted(set(item) - _ALLOWED_INSTANCE_FIELDS)
        if missing or extra:
            raise MaterializationError(
                "public native mapping instance has invalid fields at "
                f"index {index}: missing={missing}, extra={extra}"
            )
        for field in ("instance_id", "asset_id", "native_root_name"):
            text = str(item.get(field) or "").strip()
            if not text:
                raise MaterializationError(
                    f"public native mapping instances[{index}].{field} "
                    "must be non-empty"
                )
            item[field] = text
        if item["instance_id"] in instance_ids:
            raise MaterializationError(
                "public native instance mapping contains duplicate "
                f"instance_id {item['instance_id']!r}"
            )
        if item["native_root_name"] in root_names:
            raise MaterializationError(
                "public native instance mapping contains duplicate "
                f"native_root_name {item['native_root_name']!r}"
            )
        instance_ids.add(item["instance_id"])
        root_names.add(item["native_root_name"])

        item["center_m"] = finite_vec3(
            item.get("center_m"),
            f"public_native.instances[{index}].center_m",
        )
        item["rotation_euler_xyz_deg"] = finite_vec3(
            item.get("rotation_euler_xyz_deg"),
            (
                "public_native.instances"
                f"[{index}].rotation_euler_xyz_deg"
            ),
        )
        raw_scale = item.get("uniform_scale")
        if isinstance(raw_scale, bool):
            raise MaterializationError(
                f"public_native.instances[{index}].uniform_scale "
                "must be numeric"
            )
        try:
            scale = float(raw_scale)
        except (TypeError, ValueError) as exc:
            raise MaterializationError(
                f"public_native.instances[{index}].uniform_scale "
                "must be numeric"
            ) from exc
        if not math.isfinite(scale) or scale <= 0.0:
            raise MaterializationError(
                f"public_native.instances[{index}].uniform_scale must be "
                "finite and greater than zero"
            )
        item["uniform_scale"] = scale
        if "slot_id" in item:
            slot_id = str(item.get("slot_id") or "").strip()
            if not slot_id:
                raise MaterializationError(
                    f"public_native.instances[{index}].slot_id must be "
                    "non-empty when provided"
                )
            item["slot_id"] = slot_id
        normalized.append(item)

    normalized.sort(key=lambda item: item["instance_id"])
    return {
        "schema_version": PUBLIC_NATIVE_MAPPING_VERSION,
        "instances": normalized,
    }


def seal_inspected_public_native_registry(
    path: str | Path,
    *,
    authority: NativeRegistryAuthority,
    source_blend_path: str | Path,
    case_bundle_manifest_sha256: str,
    catalog_snapshot_id: str,
    public_mapping: Mapping[str, Any],
    inspection: Mapping[str, Any],
) -> Path:
    """Derive and seal the trusted registry after read-only inspection."""

    if inspection.get("status") != "passed":
        raise MaterializationError(
            "cannot seal a public native registry from a failed inspection"
        )
    mapping = validate_public_native_instance_mapping(public_mapping)
    observed_rows = {
        str(item.get("instance_id") or ""): item
        for item in inspection.get("instances", [])
        if isinstance(item, Mapping)
        and str(item.get("instance_id") or "").strip()
    }
    expected_ids = {
        str(item["instance_id"]) for item in mapping["instances"]
    }
    if set(observed_rows) != expected_ids:
        raise MaterializationError(
            "public native inspection instance set does not match the "
            "unsigned mapping"
        )

    registry_rows: list[dict[str, Any]] = []
    for item in mapping["instances"]:
        instance_id = str(item["instance_id"])
        observed = observed_rows[instance_id]
        geometry = _sha256_digest(
            observed.get("geometry_sha256"),
            f"{instance_id}.geometry_sha256",
        )
        material = _sha256_digest(
            observed.get("material_sha256"),
            f"{instance_id}.material_sha256",
        )
        root_name = str(observed.get("root_object_name") or "").strip()
        if root_name != item["native_root_name"]:
            raise MaterializationError(
                f"public native inspection root mismatch for {instance_id!r}"
            )
        asset_id = str(observed.get("asset_id") or "").strip()
        if asset_id != item["asset_id"]:
            raise MaterializationError(
                f"public native inspection asset mismatch for {instance_id!r}"
            )
        row = {
            "instance_id": instance_id,
            "evaluator_object_id": instance_id,
            "asset_id": asset_id,
            "native_root_name": root_name,
            "center_m": deepcopy(item["center_m"]),
            "uniform_scale": float(item["uniform_scale"]),
            "rotation_euler_xyz_deg": deepcopy(
                item["rotation_euler_xyz_deg"]
            ),
            "geometry_sha256": geometry,
            "material_sha256": material,
        }
        if item.get("slot_id") is not None:
            row["slot_id"] = str(item["slot_id"])
        registry_rows.append(row)

    return write_benchmark_native_registry(
        path,
        authority=authority,
        source_blend_path=source_blend_path,
        case_bundle_manifest_sha256=case_bundle_manifest_sha256,
        catalog_snapshot_id=catalog_snapshot_id,
        instances=registry_rows,
    )


def _sha256_digest(value: Any, label: str) -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise MaterializationError(
            f"trusted public native inspection did not derive {label}"
        )
    return digest
