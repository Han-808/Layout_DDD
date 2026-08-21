"""Dataset descriptor for model-independent retrieval profiles v2."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from ._common import (
    RetrievalContractError,
    exact_keys,
    identifier,
    object_value,
    string_value,
)


_CANONICAL_FIELDS = (
    "asset_id",
    "short_description",
    "description",
    "category",
    "size_xyz_m",
)


@dataclass(frozen=True, slots=True)
class DatasetDescriptor:
    dataset_id: str
    asset_namespace: str
    metadata_collection_key: str
    order_key: str
    field_mapping: Mapping[str, str]

    @classmethod
    def parse(cls, value: Any, *, label: str) -> "DatasetDescriptor":
        raw = object_value(value, label=label)
        exact_keys(
            raw,
            label=label,
            required=(
                "dataset_id",
                "asset_namespace",
                "metadata_collection_key",
                "order_key",
                "field_mapping",
            ),
        )
        mapping = object_value(raw["field_mapping"], label=f"{label}.field_mapping")
        exact_keys(
            mapping,
            label=f"{label}.field_mapping",
            required=_CANONICAL_FIELDS,
        )
        normalized = {
            name: string_value(mapping[name], label=f"{label}.field_mapping.{name}")
            for name in _CANONICAL_FIELDS
        }
        if len(set(normalized.values())) != len(normalized):
            raise RetrievalContractError(f"{label}.field_mapping values must be unique")
        return cls(
            dataset_id=identifier(raw["dataset_id"], label=f"{label}.dataset_id"),
            asset_namespace=identifier(
                raw["asset_namespace"], label=f"{label}.asset_namespace"
            ),
            metadata_collection_key=string_value(
                raw["metadata_collection_key"],
                label=f"{label}.metadata_collection_key",
            ),
            order_key=string_value(raw["order_key"], label=f"{label}.order_key"),
            field_mapping=normalized,
        )

    def validate_and_normalize_assets(
        self, metadata: Mapping[str, Any]
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        assets_value = metadata.get(self.metadata_collection_key)
        order_value = metadata.get(self.order_key)
        if not isinstance(assets_value, dict) or not isinstance(order_value, list):
            raise RetrievalContractError(
                "index metadata must contain the configured asset object and order array"
            )
        order: list[str] = []
        seen: set[str] = set()
        for index, value in enumerate(order_value):
            asset_id = string_value(value, label=f"index order[{index}]")
            if asset_id in seen:
                raise RetrievalContractError(f"duplicate asset ID in index order: {asset_id}")
            seen.add(asset_id)
            order.append(asset_id)
        if set(assets_value) != seen:
            raise RetrievalContractError("asset-map keys and ordered asset IDs differ")

        normalized: dict[str, dict[str, Any]] = {}
        mapping_is_legacy_identity = self.field_mapping == {
            "asset_id": "jid",
            "short_description": "short_desc",
            "description": "description",
            "category": "category",
            "size_xyz_m": "size",
        }
        for asset_id in order:
            record = assets_value.get(asset_id)
            if not isinstance(record, dict):
                raise RetrievalContractError(f"asset {asset_id!r} must be an object")
            source_id = record.get(self.field_mapping["asset_id"])
            if source_id != asset_id:
                raise RetrievalContractError(
                    f"asset identity mismatch for {asset_id!r}: {source_id!r}"
                )
            short_desc = string_value(
                record.get(self.field_mapping["short_description"]),
                label=f"asset {asset_id}.short_description",
            )
            description = string_value(
                record.get(self.field_mapping["description"]),
                label=f"asset {asset_id}.description",
            )
            category = record.get(self.field_mapping["category"])
            if not isinstance(category, str):
                raise RetrievalContractError(f"asset {asset_id}.category must be a string")
            size = record.get(self.field_mapping["size_xyz_m"])
            if not isinstance(size, list) or len(size) != 3:
                raise RetrievalContractError(
                    f"asset {asset_id}.size_xyz_m must contain three numbers"
                )
            dimensions: list[float] = []
            for axis, child in enumerate(size):
                if isinstance(child, bool) or not isinstance(child, (int, float)):
                    raise RetrievalContractError(
                        f"asset {asset_id}.size_xyz_m[{axis}] must be numeric"
                    )
                number = float(child)
                if not math.isfinite(number) or number <= 0:
                    raise RetrievalContractError(
                        f"asset {asset_id}.size_xyz_m[{axis}] must be finite and positive"
                    )
                dimensions.append(number)
            if mapping_is_legacy_identity:
                # Preserve the current raw record exactly for byte-parity of
                # the existing Imaginarium/Qwen retrieval artifacts.
                canonical = dict(record)
            else:
                canonical = {
                    "jid": asset_id,
                    "short_desc": short_desc,
                    "size": dimensions,
                    "category": category,
                    "description": description,
                }
            normalized[asset_id] = canonical
        return normalized, order
