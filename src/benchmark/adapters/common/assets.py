"""Injected asset-database boundary used by harness converters."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from benchmark.adapters.common.geometry import category_from_identifier
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


MESH_EXTENSIONS = (".glb", ".gltf", ".obj", ".fbx")


class AssetProvider(Protocol):
    """External asset lookup interface; implementations remain harness-agnostic."""

    def resolve(
        self,
        asset_key: str,
        *,
        source_db: str | None = None,
        hint: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None: ...

    def retrieve(
        self,
        query: str,
        *,
        category: str | None = None,
        size: Sequence[float] | None = None,
        hint: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None: ...


class MappingAssetProvider:
    """Deterministic provider for JSON manifests and converter fixtures."""

    def __init__(
        self,
        records: Mapping[str, Any] | Sequence[Any],
        *,
        base_dir: Path | None = None,
    ) -> None:
        defaults: dict[str, Any] = {}
        if isinstance(records, Mapping) and isinstance(
            records.get("assets"), (Mapping, list)
        ):
            source_db = records.get("source_db") or records.get("asset_namespace")
            if source_db:
                defaults["source_db"] = str(source_db)
            records = records["assets"]
        self.base_dir = Path(base_dir) if base_dir is not None else None
        self._records: dict[str, dict[str, Any]] = {}
        if isinstance(records, Mapping):
            items = records.items()
        elif isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
            items = ((str(index), value) for index, value in enumerate(records))
        else:
            raise TypeError(
                "asset manifest must be an object, list, or contain an assets object/list"
            )
        for fallback_key, raw_record in items:
            if not isinstance(raw_record, Mapping):
                continue
            record = {**defaults, **dict(raw_record)}
            key = _first_text(
                record.get("asset_key"),
                record.get("asset_id"),
                record.get("jid"),
                record.get("uid"),
                fallback_key,
            )
            if not key:
                continue
            record.setdefault("asset_key", key)
            record = _resolve_record_paths(record, self.base_dir)
            for alias in _asset_aliases(key, record):
                self._records.setdefault(alias, record)

    def resolve(
        self,
        asset_key: str,
        *,
        source_db: str | None = None,
        hint: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        del source_db, hint
        for alias in _asset_aliases(str(asset_key), {}):
            if alias in self._records:
                return dict(self._records[alias])
        return None

    def retrieve(
        self,
        query: str,
        *,
        category: str | None = None,
        size: Sequence[float] | None = None,
        hint: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        del hint
        unique = {id(record): record for record in self._records.values()}.values()
        candidates = list(unique)
        if not candidates:
            return None
        target_category = str(category or query or "").strip().casefold()
        target_size = _size(size)

        def score(record: dict[str, Any]) -> tuple[float, str]:
            record_category = _first_text(
                record.get("category"), record.get("class"), record.get("object_type")
            ).casefold()
            description = _first_text(
                record.get("description"), record.get("desc"), record.get("name")
            ).casefold()
            semantic = 0.0
            if target_category:
                if record_category == target_category:
                    semantic = 0.0
                elif record_category and (
                    target_category in record_category
                    or record_category in target_category
                ):
                    semantic = 1.0
                elif target_category in description:
                    semantic = 2.0
                else:
                    semantic = 100.0
            shape = 0.0
            record_size = _record_bbox_size(record)
            if target_size and record_size:
                shape = sum(
                    abs(math.log(max(a, 1.0e-9) / max(b, 1.0e-9)))
                    for a, b in zip(record_size, target_size)
                )
            return semantic + shape, str(record.get("asset_key") or "")

        best = min(candidates, key=score)
        if target_category and score(best)[0] >= 100.0:
            return None
        return dict(best)


class DatasetRetrievalAssetProvider:
    """Bridge the repository's SharedRetrieverRuntime into AssetProvider."""

    def __init__(self, runtime: Any) -> None:
        if not callable(getattr(runtime, "retrieve", None)):
            raise TypeError("dataset retrieval runtime must implement retrieve()")
        self.runtime = runtime

    def resolve(
        self,
        asset_key: str,
        *,
        source_db: str | None = None,
        hint: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        del source_db, hint
        assets = getattr(self.runtime, "assets", None)
        if isinstance(assets, Mapping) and not assets:
            load_index = getattr(self.runtime, "_load_index", None)
            if callable(load_index):
                load_index()
                assets = getattr(self.runtime, "assets", None)
        if isinstance(assets, Mapping) and asset_key in assets:
            return self._normalize(dict(assets[asset_key]))
        return None

    def retrieve(
        self,
        query: str,
        *,
        category: str | None = None,
        size: Sequence[float] | None = None,
        hint: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any] | None:
        del category, hint
        result = self.runtime.retrieve(query, size_constraint=size)
        if not isinstance(result, Mapping):
            raise TypeError("dataset retrieval runtime must return a mapping")
        return self._normalize(dict(result))

    def _normalize(self, record: dict[str, Any]) -> dict[str, Any]:
        key = _first_text(
            record.get("asset_key"), record.get("jid"), record.get("asset_id")
        )
        if key:
            record.setdefault("asset_key", key)
        record.setdefault("source_db", self._asset_namespace())
        return record

    def _asset_namespace(self) -> str:
        composed = getattr(self.runtime, "composed", None)
        dataset = getattr(composed, "dataset", None)
        namespace = getattr(dataset, "asset_namespace", None)
        if namespace:
            return str(namespace)
        provenance = getattr(self.runtime, "public_provenance", None)
        if callable(provenance):
            value = provenance()
            if isinstance(value, Mapping) and value.get("asset_namespace"):
                return str(value["asset_namespace"])
        return "dataset_retrieval"


def load_asset_provider(
    config: Mapping[str, Any],
    *,
    source_path: Path | None = None,
) -> AssetProvider | None:
    provider = config.get("asset_provider") or config.get("retrieval_runtime")
    if provider is not None:
        if _is_dataset_retrieval_runtime(provider):
            return DatasetRetrievalAssetProvider(provider)
        if not callable(getattr(provider, "resolve", None)) and not callable(
            getattr(provider, "retrieve", None)
        ):
            raise TypeError("config.asset_provider must implement resolve() or retrieve()")
        return provider

    manifest = config.get("asset_manifest")
    if manifest is not None:
        base_dir = (
            Path(source_path).parent
            if source_path is not None and Path(source_path).is_file()
            else None
        )
        return MappingAssetProvider(manifest, base_dir=base_dir)

    manifest_path = config.get("asset_manifest_path")
    if manifest_path:
        path = Path(str(manifest_path)).expanduser()
        if not path.is_absolute() and source_path is not None:
            base = (
                Path(source_path)
                if Path(source_path).is_dir()
                else Path(source_path).parent
            )
            path = base / path
        if not path.is_file():
            raise FileNotFoundError(f"asset manifest not found: {path}")
        return MappingAssetProvider(read_json(path), base_dir=path.parent)
    return None


def resolve_asset_record(
    provider: AssetProvider | None,
    *,
    asset_key: str | None,
    source_db: str,
    category: str | None,
    description: str | None,
    size: Sequence[float] | None,
    hint: Mapping[str, Any] | None = None,
    native_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = dict(native_record or {})
    resolved: Mapping[str, Any] | None = None
    if (
        provider is not None
        and asset_key
        and callable(getattr(provider, "resolve", None))
    ):
        resolved = provider.resolve(asset_key, source_db=source_db, hint=hint)
    if provider is not None and resolved is None and callable(getattr(provider, "retrieve", None)):
        resolved = provider.retrieve(
            description or category or asset_key or "object",
            category=category,
            size=size,
            hint=hint,
        )
    if resolved is not None:
        if not isinstance(resolved, Mapping):
            raise TypeError("asset provider must return a mapping or None")
        record.update(dict(resolved))
    if asset_key:
        record.setdefault("asset_key", asset_key)
    record.setdefault("source_db", source_db)
    return record


def asset_fields(
    *,
    object_id: str,
    target_size: Sequence[float],
    record: Mapping[str, Any],
    fallback_category: str | None,
    fallback_description: str | None,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    size = _size(target_size)
    if size is None:
        raise ArtifactValidationError(f"object {object_id!r} has no positive target size")
    nested_ref = record.get("asset_ref")
    nested_ref = nested_ref if isinstance(nested_ref, Mapping) else {}
    key = _first_text(
        record.get("asset_key"),
        record.get("asset_id"),
        record.get("jid"),
        record.get("uid"),
        nested_ref.get("asset_key"),
        object_id,
    )
    source_db = _first_text(
        record.get("source_db"),
        nested_ref.get("source_db"),
        config.get("asset_source_db"),
        "external",
    )
    category = _first_text(
        record.get("category"),
        record.get("class"),
        record.get("object_type"),
        fallback_category,
        category_from_identifier(object_id),
    )
    description = _first_text(
        record.get("description"),
        record.get("desc"),
        record.get("caption"),
        fallback_description,
        category,
    )
    mesh_uri = _first_text(
        record.get("mesh_uri"),
        record.get("mesh_path"),
        record.get("file_path"),
        record.get("glb_path"),
        record.get("path"),
        nested_ref.get("mesh_uri"),
    )
    if not mesh_uri:
        mesh_uri = _mesh_from_root(config.get("asset_root"), key)
    if mesh_uri and Path(urlparse(mesh_uri).path).suffix.lower() not in MESH_EXTENSIONS:
        raise ArtifactValidationError(
            f"asset {key!r} mesh_uri must reference a supported triangle mesh "
            f"({', '.join(MESH_EXTENSIONS)}), got {mesh_uri!r}"
        )
    metadata_uri = _first_text(
        record.get("metadata_uri"),
        record.get("metadata_path"),
        nested_ref.get("metadata_uri"),
    )
    bbox_center = _record_bbox_center(record) or [0.0, 0.0, 0.0]
    asset_ref: dict[str, Any] = {"source_db": source_db, "asset_key": key}
    if mesh_uri:
        asset_ref["mesh_uri"] = mesh_uri
    if metadata_uri:
        asset_ref["metadata_uri"] = metadata_uri
    metadata = dict(
        record.get("metadata")
        if isinstance(record.get("metadata"), Mapping)
        else {}
    )
    metadata.setdefault("interactive", bool(record.get("interactive", False)))
    canonical_front = record.get("canonical_front") or record.get("front")
    if canonical_front is not None:
        metadata.setdefault("canonical_front", canonical_front)
    return {
        "jid": key,
        "category": category,
        "retrieval_category": _first_text(record.get("retrieval_category"), category),
        "description": description,
        "desc": description,
        "short_desc": _first_text(
            record.get("short_description"), record.get("short_desc"), description
        ),
        "geometry_provenance": "asset_mesh" if mesh_uri else "bbox_proxy",
        "asset_ref": asset_ref,
        "asset_proxy": {
            "type": "external_asset_bbox" if mesh_uri else "harness_layout_bbox",
            "bbox_center_local": bbox_center,
            "bbox_size": list(size),
        },
        "metadata": metadata,
    }


def record_bbox_size(record: Mapping[str, Any]) -> list[float] | None:
    return _record_bbox_size(record)


def record_bbox_center(record: Mapping[str, Any]) -> list[float] | None:
    return _record_bbox_center(record)


def _record_bbox_size(record: Mapping[str, Any]) -> list[float] | None:
    direct = _size(
        record.get("bbox_size")
        or record.get("size")
        or record.get("dimensions")
        or record.get("transformed_size")
    )
    if direct:
        return direct
    proxy = record.get("asset_proxy")
    if isinstance(proxy, Mapping):
        direct = _size(proxy.get("bbox_size"))
        if direct:
            return direct
    asset_metadata = record.get("assetMetadata")
    if isinstance(asset_metadata, Mapping):
        bbox = asset_metadata.get("boundingBox")
        if isinstance(bbox, Mapping):
            return _size([bbox.get("x"), bbox.get("y"), bbox.get("z")])
    return None


def _record_bbox_center(record: Mapping[str, Any]) -> list[float] | None:
    value = record.get("bbox_center_local") or record.get("bbox_center")
    if value is None and isinstance(record.get("asset_proxy"), Mapping):
        value = record["asset_proxy"].get("bbox_center_local")
    result = _numeric_vector(value)
    return result if result is not None else None


def _size(value: Any) -> list[float] | None:
    result = _numeric_vector(value)
    if result is None or any(item <= 0.0 for item in result):
        return None
    return result


def _numeric_vector(value: Any) -> list[float] | None:
    if isinstance(value, Mapping) and all(axis in value for axis in ("x", "y", "z")):
        value = [value["x"], value["y"], value["z"]]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return None
    try:
        result = [float(value[index]) for index in range(3)]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _asset_aliases(key: str, record: Mapping[str, Any]) -> set[str]:
    aliases = {str(key)}
    if "." in str(key):
        aliases.add(str(key).split(".", 1)[1])
    for field in ("asset_key", "asset_id", "jid", "uid", "modelId", "model_id"):
        value = _first_text(record.get(field))
        if value:
            aliases.add(value)
            if "." in value:
                aliases.add(value.split(".", 1)[1])
    nested_ref = record.get("asset_ref")
    if isinstance(nested_ref, Mapping):
        nested_key = _first_text(nested_ref.get("asset_key"))
        if nested_key:
            aliases.add(nested_key)
    return aliases


def _is_dataset_retrieval_runtime(value: Any) -> bool:
    return (
        callable(getattr(value, "retrieve", None))
        and callable(getattr(value, "retrieve_batch", None))
        and getattr(value, "composed", None) is not None
    )


def _resolve_record_paths(
    record: dict[str, Any], base_dir: Path | None
) -> dict[str, Any]:
    if base_dir is None:
        return record
    for key in (
        "mesh_uri",
        "mesh_path",
        "file_path",
        "glb_path",
        "path",
        "metadata_uri",
        "metadata_path",
    ):
        value = record.get(key)
        if not value:
            continue
        if "://" in str(value):
            continue
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            record[key] = (base_dir / path).resolve().as_posix()
    return record


def _mesh_from_root(root_value: Any, asset_key: str) -> str:
    if not root_value:
        return ""
    root = Path(str(root_value)).expanduser()
    candidates = [root / asset_key]
    for extension in MESH_EXTENSIONS:
        candidates.extend(
            (
                root / f"{asset_key}{extension}",
                root / asset_key / f"{asset_key}{extension}",
            )
        )
    for candidate in candidates:
        if candidate.is_file() and candidate.suffix.lower() in MESH_EXTENSIONS:
            return candidate.resolve().as_posix()
    return ""


def _first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


__all__ = [
    "AssetProvider",
    "DatasetRetrievalAssetProvider",
    "MappingAssetProvider",
    "asset_fields",
    "load_asset_provider",
    "record_bbox_center",
    "record_bbox_size",
    "resolve_asset_record",
]
