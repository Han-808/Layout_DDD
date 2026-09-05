"""Plan and verify a frozen GLB bundle for selected Imaginarium assets."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json, write_json


BUNDLE_PLAN_SCHEMA_VERSION = "imaginarium_glb_bundle_plan_v1"
BUNDLE_REPORT_SCHEMA_VERSION = "imaginarium_glb_bundle_report_v1"
GEOMETRY_TOLERANCE_M = 1.0e-4
SAFE_ASSET_ID = re.compile(r"[A-Za-z0-9_.-]+")


def build_imaginarium_glb_bundle_plan(
    *,
    catalog_spec: Mapping[str, Any],
    asset_root: str | Path,
    bundle_root: str | Path,
) -> dict[str, Any]:
    """Write the deterministic conversion plan consumed by the Blender worker."""

    source_root = Path(asset_root).expanduser().resolve()
    target_root = Path(bundle_root).expanduser().resolve()
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"bundle output already exists; use a fresh attempt directory: {target_root}")
    assets = catalog_spec.get("assets")
    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes)):
        raise ArtifactValidationError("catalog spec assets must be a list")
    rows = []
    seen: set[str] = set()
    for index, item in enumerate(assets):
        if not isinstance(item, Mapping):
            raise ArtifactValidationError(f"catalog assets[{index}] must be an object")
        asset_id = str(item.get("asset_id") or "").strip()
        if not SAFE_ASSET_ID.fullmatch(asset_id):
            raise ArtifactValidationError(
                f"catalog asset_id {asset_id!r} is unsafe for exact GLB filenames"
            )
        if asset_id in seen:
            raise ArtifactValidationError(f"duplicate catalog asset_id {asset_id!r}")
        seen.add(asset_id)
        source = source_root / asset_id / f"{asset_id}.fbx"
        metadata = source_root / asset_id / f"{asset_id}_metadata.json"
        if not source.is_file():
            raise FileNotFoundError(f"Imaginarium FBX is missing: {source}")
        if not metadata.is_file():
            raise FileNotFoundError(f"Imaginarium metadata is missing: {metadata}")
        metadata_payload = read_json(metadata)
        if not isinstance(metadata_payload, Mapping):
            raise ArtifactValidationError(
                f"Imaginarium metadata must be an object: {metadata}"
            )
        bbox_size = _vector3(
            metadata_payload.get("transformed_size"),
            f"{asset_id}.transformed_size",
            positive=True,
        )
        bbox_center = _vector3(
            metadata_payload.get("transformed_bbox_center"),
            f"{asset_id}.transformed_bbox_center",
        )
        target = target_root / asset_id / f"{asset_id}.glb"
        rows.append(
            {
                "asset_id": asset_id,
                "source_fbx": source.as_posix(),
                "source_fbx_sha256": file_sha256(source),
                "source_metadata": metadata.as_posix(),
                "source_metadata_sha256": file_sha256(metadata),
                "expected_bbox_size": bbox_size,
                "expected_bbox_center": bbox_center,
                "target_glb": target.as_posix(),
            }
        )
    payload = {
        "schema_version": BUNDLE_PLAN_SCHEMA_VERSION,
        "catalog_id": str(catalog_spec.get("catalog_id") or ""),
        "catalog_version": str(catalog_spec.get("catalog_version") or ""),
        "source_db": str(catalog_spec.get("source_db") or "imaginarium"),
        "conversion_policy": {
            "format": "glb",
            "geometry_transform": "none",
            "scale_transform": "none",
            "origin_transform": "none",
            "verification": "fbx_import_vs_glb_reimport_vs_metadata_bbox",
            "geometry_tolerance_m": GEOMETRY_TOLERANCE_M,
            "xy_order_tolerance_m": GEOMETRY_TOLERANCE_M,
            "export_loose_geometry": True,
        },
        "asset_count": len(rows),
        "assets": rows,
    }
    plan_path = write_json(target_root / "bundle_plan.json", payload)
    return {**payload, "plan_path": plan_path.resolve().as_posix()}


def validate_imaginarium_glb_bundle(
    *,
    plan: Mapping[str, Any] | str | Path,
    report: Mapping[str, Any] | str | Path,
    expected_asset_root: str | Path | None = None,
    expected_bundle_root: str | Path | None = None,
) -> dict[str, Any]:
    plan_payload = _load_mapping(plan, "bundle plan")
    report_payload = _load_mapping(report, "bundle report")
    if plan_payload.get("schema_version") != BUNDLE_PLAN_SCHEMA_VERSION:
        raise ArtifactValidationError("unsupported Imaginarium bundle plan schema")
    if report_payload.get("schema_version") != BUNDLE_REPORT_SCHEMA_VERSION:
        raise ArtifactValidationError("unsupported Imaginarium bundle report schema")
    policy = plan_payload.get("conversion_policy")
    policy = policy if isinstance(policy, Mapping) else {}
    tolerance = policy.get("geometry_tolerance_m")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not math.isfinite(float(tolerance))
        or float(tolerance) != GEOMETRY_TOLERANCE_M
    ):
        raise ArtifactValidationError(
            "Imaginarium bundle plan must use the frozen geometry tolerance"
        )
    tolerance = float(tolerance)
    plan_file = (
        Path(plan).expanduser().resolve()
        if isinstance(plan, (str, Path))
        else None
    )
    bundle_root = (
        Path(expected_bundle_root).expanduser().resolve()
        if expected_bundle_root is not None
        else plan_file.parent
        if plan_file is not None
        else None
    )
    asset_root = (
        Path(expected_asset_root).expanduser().resolve()
        if expected_asset_root is not None
        else None
    )
    expected = {
        str(item["asset_id"]): item for item in plan_payload.get("assets", [])
    }
    observed = {
        str(item["asset_id"]): item for item in report_payload.get("assets", [])
    }
    errors = []
    if plan_file is not None:
        reported_plan = report_payload.get("plan")
        if not reported_plan or Path(str(reported_plan)).expanduser().resolve() != plan_file:
            errors.append({"code": "report_plan_mismatch"})
    if report_payload.get("geometry_tolerance_m") != tolerance:
        errors.append({"code": "report_tolerance_mismatch"})
    if set(expected) != set(observed):
        errors.append(
            {
                "code": "asset_inventory_mismatch",
                "missing": sorted(set(expected) - set(observed)),
                "unexpected": sorted(set(observed) - set(expected)),
            }
        )
    for asset_id in sorted(set(expected) & set(observed)):
        planned = expected[asset_id]
        actual = observed[asset_id]
        path = Path(str(planned["target_glb"]))
        source_fbx = Path(str(planned["source_fbx"]))
        source_metadata = Path(str(planned["source_metadata"]))
        if bundle_root is not None:
            required_target = bundle_root / asset_id / f"{asset_id}.glb"
            if path.expanduser().resolve() != required_target:
                errors.append({"code": "target_root_mismatch", "asset_id": asset_id})
        if asset_root is not None:
            required_source = asset_root / asset_id / f"{asset_id}.fbx"
            required_metadata = asset_root / asset_id / f"{asset_id}_metadata.json"
            if source_fbx.expanduser().resolve() != required_source:
                errors.append({"code": "source_root_mismatch", "asset_id": asset_id})
            if source_metadata.expanduser().resolve() != required_metadata:
                errors.append(
                    {"code": "source_metadata_root_mismatch", "asset_id": asset_id}
                )
        if not source_fbx.is_file() or file_sha256(source_fbx) != planned.get(
            "source_fbx_sha256"
        ):
            errors.append({"code": "source_hash_mismatch", "asset_id": asset_id})
        if not source_metadata.is_file() or file_sha256(
            source_metadata
        ) != planned.get("source_metadata_sha256"):
            errors.append(
                {"code": "source_metadata_hash_mismatch", "asset_id": asset_id}
            )
        if actual.get("status") != "passed":
            errors.append({"code": "conversion_failed", "asset_id": asset_id})
            continue
        if Path(str(actual.get("target_glb") or "")).expanduser().resolve() != path.expanduser().resolve():
            errors.append({"code": "reported_target_mismatch", "asset_id": asset_id})
        if not path.is_file():
            errors.append({"code": "glb_missing", "asset_id": asset_id})
            continue
        digest = file_sha256(path)
        if digest != actual.get("target_glb_sha256"):
            errors.append(
                {
                    "code": "glb_hash_mismatch",
                    "asset_id": asset_id,
                    "expected": actual.get("target_glb_sha256"),
                    "actual": digest,
                }
            )
        if actual.get("source_fbx_sha256") != planned.get("source_fbx_sha256"):
            errors.append({"code": "source_hash_mismatch", "asset_id": asset_id})
        if actual.get("source_metadata_sha256") != planned.get(
            "source_metadata_sha256"
        ):
            errors.append(
                {"code": "source_metadata_hash_mismatch", "asset_id": asset_id}
            )
        geometry_checks = {
            "source_size_vs_metadata": _close3(
                actual.get("source_bbox_size"),
                planned.get("expected_bbox_size"),
                tolerance,
            ),
            "source_center_vs_metadata": _close3(
                actual.get("source_bbox_center"),
                planned.get("expected_bbox_center"),
                tolerance,
            ),
            "roundtrip_size_vs_source": _close3(
                actual.get("roundtrip_bbox_size"),
                actual.get("source_bbox_size"),
                tolerance,
            ),
            "roundtrip_center_vs_source": _close3(
                actual.get("roundtrip_bbox_center"),
                actual.get("source_bbox_center"),
                tolerance,
            ),
            "roundtrip_xy_order_vs_metadata": _same_xy_order(
                actual.get("roundtrip_bbox_size"),
                planned.get("expected_bbox_size"),
                tolerance,
            ),
        }
        if not all(geometry_checks.values()):
            errors.append(
                {
                    "code": "geometry_unverified",
                    "asset_id": asset_id,
                    "checks": geometry_checks,
                }
            )
    return {
        "schema_version": "imaginarium_glb_bundle_validation_v1",
        "valid": not errors,
        "asset_count": len(expected),
        "errors": errors,
    }


def bundle_mesh_path(bundle_root: str | Path, asset_id: str) -> Path:
    root = Path(bundle_root).expanduser().resolve()
    nested = root / asset_id / f"{asset_id}.glb"
    flat = root / f"{asset_id}.glb"
    if nested.is_file():
        return nested
    if flat.is_file():
        return flat
    raise FileNotFoundError(f"frozen GLB is missing for asset {asset_id!r} under {root}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mapping(value: Mapping[str, Any] | str | Path, label: str) -> dict[str, Any]:
    loaded = read_json(value) if isinstance(value, (str, Path)) else value
    if not isinstance(loaded, Mapping):
        raise ArtifactValidationError(f"{label} must be a JSON object")
    return dict(loaded)


def _vector3(value: Any, path: str, *, positive: bool = False) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ArtifactValidationError(f"{path} must be a 3-vector")
    result = [float(component) for component in value]
    if positive and any(component <= 0.0 for component in result):
        raise ArtifactValidationError(f"{path} must be positive")
    return result


def _close3(left: Any, right: Any, tolerance: float) -> bool:
    try:
        first = _vector3(left, "reported_bbox")
        second = _vector3(right, "expected_bbox")
    except (ArtifactValidationError, TypeError, ValueError):
        return False
    return all(
        math.isfinite(first[index])
        and math.isfinite(second[index])
        and abs(first[index] - second[index]) <= tolerance
        for index in range(3)
    )


def _same_xy_order(left: Any, right: Any, tolerance: float = GEOMETRY_TOLERANCE_M) -> bool:
    try:
        first = _vector3(left, "reported_bbox")
        second = _vector3(right, "expected_bbox")
    except (ArtifactValidationError, TypeError, ValueError):
        return False
    left_delta = first[0] - first[1]
    right_delta = second[0] - second[1]
    # Near-square boxes have no numerically resolvable XY ordering. The same
    # frozen per-axis tolerance still applies in _close3; do not mistake
    # sub-tolerance floating-point sign changes for a 90-degree rotation.
    return (abs(left_delta) <= tolerance or abs(right_delta) <= tolerance
            or left_delta * right_delta > 0.0)


__all__ = [
    "BUNDLE_PLAN_SCHEMA_VERSION",
    "BUNDLE_REPORT_SCHEMA_VERSION",
    "GEOMETRY_TOLERANCE_M",
    "build_imaginarium_glb_bundle_plan",
    "bundle_mesh_path",
    "file_sha256",
    "validate_imaginarium_glb_bundle",
]
