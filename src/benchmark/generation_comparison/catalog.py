"""Immutable logical asset catalogs for controlled generation inputs."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from benchmark.generation_comparison.identity import canonical_json_sha256
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


CATALOG_SCHEMA_VERSION = "canonical_asset_catalog_v1"


@dataclass(frozen=True)
class CanonicalAssetCatalog:
    """A content-addressed catalog whose public representation cannot mutate it."""

    _json: str

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        hash_local_meshes: bool = False,
        base_dir: str | Path | None = None,
    ) -> "CanonicalAssetCatalog":
        normalized = validate_asset_catalog(
            value,
            hash_local_meshes=hash_local_meshes,
            base_dir=Path(base_dir) if base_dir is not None else None,
        )
        supplied = value.get("catalog_sha256")
        observed = canonical_json_sha256(_catalog_hash_payload(normalized))
        if supplied is not None and str(supplied) != observed:
            raise ArtifactValidationError(
                "asset catalog content does not match supplied catalog_sha256"
            )
        normalized["catalog_sha256"] = observed
        return cls(
            json.dumps(
                normalized,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )

    @property
    def catalog_id(self) -> str:
        return str(self.as_dict()["catalog_id"])

    @property
    def catalog_version(self) -> str:
        return str(self.as_dict()["catalog_version"])

    @property
    def sha256(self) -> str:
        return str(self.as_dict()["catalog_sha256"])

    @property
    def identity(self) -> dict[str, str]:
        return {
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "catalog_sha256": self.sha256,
        }

    @property
    def assets(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self.as_dict()["assets"])

    @property
    def asset_ids(self) -> tuple[str, ...]:
        return tuple(str(item["asset_id"]) for item in self.assets)

    def get(self, asset_id: str) -> dict[str, Any]:
        for item in self.assets:
            if item["asset_id"] == asset_id:
                return item
        raise ArtifactValidationError(
            f"asset {asset_id!r} is absent from catalog {self.catalog_id!r}"
        )

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self._json)


def load_asset_catalog(
    value: CanonicalAssetCatalog | Mapping[str, Any] | str | Path,
    *,
    hash_local_meshes: bool = False,
) -> CanonicalAssetCatalog:
    if isinstance(value, CanonicalAssetCatalog):
        return value
    base_dir: Path | None = None
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        loaded = read_json(path)
        if not isinstance(loaded, Mapping):
            raise ArtifactValidationError("asset catalog file must contain an object")
        value = loaded
        base_dir = path.parent
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("asset catalog must be an object or path")
    return CanonicalAssetCatalog.from_mapping(
        value,
        hash_local_meshes=hash_local_meshes,
        base_dir=base_dir,
    )


def validate_asset_catalog(
    value: Mapping[str, Any],
    *,
    hash_local_meshes: bool = False,
    base_dir: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("asset catalog must be an object")
    schema = str(value.get("schema_version") or CATALOG_SCHEMA_VERSION)
    if schema != CATALOG_SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"asset catalog schema_version must be {CATALOG_SCHEMA_VERSION!r}"
        )
    catalog_id = _required_text(value.get("catalog_id"), "catalog_id")
    catalog_version = _required_text(
        value.get("catalog_version"), "catalog_version"
    )
    linear_unit = str(value.get("linear_unit") or "meter").strip().lower()
    if linear_unit not in {"m", "meter", "meters", "metre", "metres"}:
        raise ArtifactValidationError("asset catalog linear_unit must be meter")
    raw_assets = value.get("assets")
    if not (
        isinstance(raw_assets, Sequence)
        and not isinstance(raw_assets, (str, bytes))
        and raw_assets
    ):
        raise ArtifactValidationError("asset catalog assets must be a non-empty list")
    assets = [
        _normalize_asset(
            item,
            index,
            hash_local_meshes=hash_local_meshes,
            base_dir=base_dir,
        )
        for index, item in enumerate(raw_assets)
    ]
    assets.sort(key=lambda item: str(item["asset_id"]))
    ids = [str(item["asset_id"]) for item in assets]
    if len(ids) != len(set(ids)):
        raise ArtifactValidationError("asset catalog asset_id values must be unique")
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ArtifactValidationError("asset catalog metadata must be an object")
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_id": catalog_id,
        "catalog_version": catalog_version,
        "linear_unit": "meter",
        "assets": assets,
        "metadata": dict(metadata),
    }


def physical_dimensions(asset: Mapping[str, Any]) -> list[float]:
    explicit = asset.get("physical_dimensions")
    if explicit is not None:
        return _positive_vector(explicit, "asset.physical_dimensions")
    size = _positive_vector(asset.get("bbox_size_local"), "asset.bbox_size_local")
    scale = _scale_vector(asset.get("native_scale", 1.0), "asset.native_scale")
    return [float(size[index] * scale[index]) for index in range(3)]


def converter_asset_manifest(catalog: CanonicalAssetCatalog) -> dict[str, Any]:
    """Expose the frozen snapshot to converters for exact ID dereference only."""

    records: dict[str, dict[str, Any]] = {}
    for asset in catalog.assets:
        record = dict(asset)
        metadata = dict(record.get("metadata") or {})
        metadata["comparison_catalog_provenance"] = {
            **catalog.identity,
            "asset_id": asset["asset_id"],
            "bbox_size_local": list(asset["bbox_size_local"]),
            "bbox_center_local": list(asset["bbox_center_local"]),
            "native_scale": list(asset["native_scale"]),
            "physical_dimensions": list(asset["physical_dimensions"]),
            "canonical_front": asset.get("canonical_front"),
        }
        record["metadata"] = metadata
        record["asset_key"] = asset["asset_id"]
        record["bbox_size"] = list(asset["bbox_size_local"])
        records[str(asset["asset_id"])] = record
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        **catalog.identity,
        "assets": records,
    }


def _normalize_asset(
    value: Any,
    index: int,
    *,
    hash_local_meshes: bool,
    base_dir: Path | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"assets[{index}] must be an object")
    asset_id = _required_text(value.get("asset_id"), f"assets[{index}].asset_id")
    source_db = _required_text(value.get("source_db"), f"assets[{index}].source_db")
    category = _required_text(value.get("category"), f"assets[{index}].category")
    description = _required_text(
        value.get("description"), f"assets[{index}].description"
    )
    bbox_size = _positive_vector(
        value.get("bbox_size_local"), f"assets[{index}].bbox_size_local"
    )
    bbox_center = _finite_vector(
        value.get("bbox_center_local"), f"assets[{index}].bbox_center_local"
    )
    native_scale = _scale_vector(
        value.get("native_scale", 1.0), f"assets[{index}].native_scale"
    )
    derived_physical = [
        float(bbox_size[axis] * native_scale[axis]) for axis in range(3)
    ]
    physical = physical_dimensions(
        {
            "bbox_size_local": bbox_size,
            "native_scale": native_scale,
            **(
                {"physical_dimensions": value["physical_dimensions"]}
                if value.get("physical_dimensions") is not None
                else {}
            ),
        }
    )
    if any(
        abs(physical[axis] - derived_physical[axis]) > 1.0e-9
        for axis in range(3)
    ):
        raise ArtifactValidationError(
            f"assets[{index}].physical_dimensions must equal "
            "bbox_size_local * native_scale in comparison v1"
        )
    result: dict[str, Any] = {
        "asset_id": asset_id,
        "source_db": source_db,
        "category": category,
        "description": description,
        "bbox_size_local": bbox_size,
        "bbox_center_local": bbox_center,
        "native_scale": native_scale,
        "physical_dimensions": physical,
    }
    mesh_uri = value.get("mesh_uri") or value.get("mesh_path")
    if mesh_uri is not None:
        result["mesh_uri"] = _required_text(mesh_uri, f"assets[{index}].mesh_uri")
    canonical_front = value.get("canonical_front")
    if canonical_front is not None:
        front = _finite_vector(canonical_front, f"assets[{index}].canonical_front")
        if math.sqrt(sum(component * component for component in front)) <= 1.0e-12:
            raise ArtifactValidationError(
                f"assets[{index}].canonical_front must be non-zero"
            )
        result["canonical_front"] = front
    content = value.get("content") or {}
    if not isinstance(content, Mapping):
        raise ArtifactValidationError(f"assets[{index}].content must be an object")
    content_record = dict(content)
    supplied_mesh_hash = content_record.get("mesh_sha256")
    if supplied_mesh_hash is not None:
        supplied_mesh_hash = str(supplied_mesh_hash).strip().lower()
        if len(supplied_mesh_hash) != 64 or any(
            character not in "0123456789abcdef" for character in supplied_mesh_hash
        ):
            raise ArtifactValidationError(
                f"assets[{index}].content.mesh_sha256 must be lowercase SHA-256"
            )
        content_record["mesh_sha256"] = supplied_mesh_hash
    if result.get("mesh_uri") and hash_local_meshes:
        mesh_path = _local_mesh_path(str(result["mesh_uri"]), base_dir=base_dir)
        if mesh_path is not None and mesh_path.is_file():
            observed_mesh_hash = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
            if supplied_mesh_hash is not None and supplied_mesh_hash != observed_mesh_hash:
                raise ArtifactValidationError(
                    f"assets[{index}].content.mesh_sha256 does not match local mesh"
                )
            content_record["mesh_sha256"] = observed_mesh_hash
            content_record["mesh_bytes"] = mesh_path.stat().st_size
    if content_record:
        result["content"] = content_record
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ArtifactValidationError(f"assets[{index}].metadata must be an object")
    result["metadata"] = dict(metadata)
    return result


def _catalog_hash_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = deepcopy(
        {key: item for key, item in value.items() if key != "catalog_sha256"}
    )
    for asset in payload.get("assets", []):
        if not isinstance(asset, dict):
            continue
        content = asset.get("content")
        content = content if isinstance(content, Mapping) else {}
        mesh_hash = content.get("mesh_sha256")
        if mesh_hash:
            # Local cache paths are materialization details. Once mesh bytes
            # have an explicit identity, the logical catalog hash follows the
            # content rather than a host-specific locator.
            asset.pop("mesh_uri", None)
    return payload


def _required_text(value: Any, path: str) -> str:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ArtifactValidationError(f"{path} must be a non-empty string")
    return str(value).strip()


def _positive_vector(value: Any, path: str) -> list[float]:
    vector = _finite_vector(value, path)
    if any(component <= 0.0 for component in vector):
        raise ArtifactValidationError(f"{path} values must be positive")
    return vector


def _finite_vector(value: Any, path: str) -> list[float]:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    ):
        raise ArtifactValidationError(f"{path} must be a finite 3-vector")
    result: list[float] = []
    for component in value:
        if isinstance(component, bool):
            raise ArtifactValidationError(f"{path} must be a finite 3-vector")
        try:
            number = float(component)
        except (TypeError, ValueError) as exc:
            raise ArtifactValidationError(f"{path} must be a finite 3-vector") from exc
        if not math.isfinite(number):
            raise ArtifactValidationError(f"{path} must be a finite 3-vector")
        result.append(0.0 if number == 0.0 else number)
    return result


def _scale_vector(value: Any, path: str) -> list[float]:
    if isinstance(value, bool):
        raise ArtifactValidationError(f"{path} must be positive")
    if isinstance(value, (int, float)):
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - guarded above
            raise ArtifactValidationError(f"{path} must be positive") from exc
        vector = [scalar, scalar, scalar]
    else:
        vector = _finite_vector(value, path)
    if any(not math.isfinite(component) or component <= 0.0 for component in vector):
        raise ArtifactValidationError(f"{path} values must be positive")
    return vector


def _local_mesh_path(value: str, *, base_dir: Path | None) -> Path | None:
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        return None
    path = Path(parsed.path if parsed.scheme == "file" else value).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "CanonicalAssetCatalog",
    "converter_asset_manifest",
    "load_asset_catalog",
    "physical_dimensions",
    "validate_asset_catalog",
]
