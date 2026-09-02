"""Versioned contracts for controlled cross-method generation experiments."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.generation_comparison.identity import (
    architecture_sha256,
    canonical_json_sha256,
    normalize_rectangular_architecture,
)
from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json


PROTOCOL_SCHEMA_VERSION = "generation_comparison_protocol_v1"
PROTOCOL_ID = "generation_comparison_v1"
PROTOCOL_VERSION = 1

NATIVE = "native"
SHARED_DB = "shared_db"
FROZEN_ASSETS = "frozen_assets"
PROTOCOL_MODES = {NATIVE, SHARED_DB, FROZEN_ASSETS}

INVENTORY_METHOD_NATIVE = "method_native"
INVENTORY_FROZEN = "frozen"
SCALE_METHOD_NATIVE = "method_native"
SCALE_FIXED_NATIVE = "fixed_native_scale"


@dataclass(frozen=True)
class ComparisonProtocol:
    """Immutable, normalized protocol snapshot backed by canonical JSON text."""

    _json: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ComparisonProtocol":
        normalized = validate_comparison_protocol(value)
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
    def mode(self) -> str:
        return str(self.as_dict()["mode"])

    @property
    def case_id(self) -> str:
        return str(self.as_dict()["case_id"])

    @property
    def architecture(self) -> dict[str, Any]:
        return dict(self.as_dict()["architecture"])

    @property
    def architecture_hash(self) -> str:
        return str(self.as_dict()["architecture_sha256"])

    @property
    def objects(self) -> list[dict[str, Any]]:
        return list(self.as_dict()["objects"])

    @property
    def inventory_policy(self) -> str:
        return str(self.as_dict()["object_inventory_policy"])

    @property
    def scale_policy(self) -> str:
        return str(self.as_dict()["scale_policy"])

    @property
    def catalog_identity(self) -> dict[str, str] | None:
        value = self.as_dict().get("assets")
        return dict(value) if isinstance(value, dict) else None

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.as_dict())

    @property
    def inventory_sha256(self) -> str | None:
        value = self.as_dict().get("object_inventory_sha256")
        return str(value) if value else None

    @property
    def binding_sha256(self) -> str | None:
        value = self.as_dict().get("asset_binding_sha256")
        return str(value) if value else None

    @property
    def bindings(self) -> dict[str, str]:
        return {
            str(item["slot_id"]): str(item["asset_id"])
            for item in self.objects
            if item.get("asset_id")
        }

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self._json)


def load_comparison_protocol(
    value: ComparisonProtocol | Mapping[str, Any] | str | Path,
) -> ComparisonProtocol:
    if isinstance(value, ComparisonProtocol):
        return value
    if isinstance(value, (str, Path)):
        loaded = read_json(value)
        if not isinstance(loaded, Mapping):
            raise ArtifactValidationError("comparison protocol file must contain an object")
        value = loaded
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("comparison protocol must be an object or path")
    return ComparisonProtocol.from_mapping(value)


def validate_comparison_protocol(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("comparison protocol must be an object")
    supplied_schema = value.get("schema_version")
    if supplied_schema is not None and supplied_schema != PROTOCOL_SCHEMA_VERSION:
        raise ArtifactValidationError(
            f"comparison schema_version must be {PROTOCOL_SCHEMA_VERSION!r}"
        )
    protocol_id = str(value.get("protocol_id") or "")
    if protocol_id != PROTOCOL_ID:
        raise ArtifactValidationError(
            f"comparison protocol_id must be {PROTOCOL_ID!r}"
        )
    try:
        protocol_version = int(value.get("protocol_version"))
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError("comparison protocol_version must be 1") from exc
    if protocol_version != PROTOCOL_VERSION:
        raise ArtifactValidationError(
            f"comparison protocol_version must be {PROTOCOL_VERSION}"
        )
    mode = str(value.get("mode") or "").strip().lower()
    if mode not in PROTOCOL_MODES:
        raise ArtifactValidationError(
            "comparison mode must be native, shared_db, or frozen_assets"
        )
    case_id = _required_text(value.get("case_id"), "comparison case_id")
    architecture_value = value.get("architecture")
    if not isinstance(architecture_value, Mapping):
        raise ArtifactValidationError("comparison architecture must be an object")
    architecture = normalize_rectangular_architecture(architecture_value)

    inventory_policy = str(value.get("object_inventory_policy") or "").strip()
    if inventory_policy not in {INVENTORY_METHOD_NATIVE, INVENTORY_FROZEN}:
        raise ArtifactValidationError(
            "object_inventory_policy must be method_native or frozen"
        )
    raw_objects = value.get("objects") or []
    if not (
        isinstance(raw_objects, Sequence)
        and not isinstance(raw_objects, (str, bytes))
    ):
        raise ArtifactValidationError("comparison objects must be a list")
    objects = [_normalize_slot(item, index) for index, item in enumerate(raw_objects)]
    slots = [str(item["slot_id"]) for item in objects]
    if len(slots) != len(set(slots)):
        raise ArtifactValidationError("comparison object slot_id values must be unique")
    if inventory_policy == INVENTORY_FROZEN and not objects:
        raise ArtifactValidationError("frozen object inventory must not be empty")

    asset_policy = str(value.get("asset_policy") or "").strip()
    scale_policy = str(value.get("scale_policy") or "").strip()
    retrieval_policy = str(value.get("retrieval_policy") or "").strip()
    assets = _normalize_catalog_identity(value.get("assets"))
    if mode == NATIVE:
        if inventory_policy != INVENTORY_METHOD_NATIVE:
            raise ArtifactValidationError(
                "native mode requires object_inventory_policy=method_native"
            )
        if assets is not None:
            raise ArtifactValidationError(
                "native mode must not bind a benchmark shared asset catalog"
            )
        if asset_policy != "native":
            raise ArtifactValidationError("native mode requires asset_policy=native")
        if retrieval_policy != "method_native":
            raise ArtifactValidationError(
                "native mode requires retrieval_policy=method_native"
            )
        if scale_policy != SCALE_METHOD_NATIVE:
            raise ArtifactValidationError(
                "native mode requires scale_policy=method_native"
            )
    elif mode == SHARED_DB:
        if asset_policy != "shared_catalog":
            raise ArtifactValidationError(
                "shared_db mode requires asset_policy=shared_catalog"
            )
        if retrieval_policy != "method_native_shared_catalog":
            raise ArtifactValidationError(
                "shared_db mode requires "
                "retrieval_policy=method_native_shared_catalog"
            )
        if scale_policy not in {SCALE_METHOD_NATIVE, SCALE_FIXED_NATIVE}:
            raise ArtifactValidationError(
                "shared_db scale_policy must be method_native or fixed_native_scale"
            )
        if assets is None:
            raise ArtifactValidationError("shared_db mode requires assets identity")
    else:
        if inventory_policy != INVENTORY_FROZEN:
            raise ArtifactValidationError(
                "frozen_assets mode requires object_inventory_policy=frozen"
            )
        if asset_policy != "frozen_exact":
            raise ArtifactValidationError(
                "frozen_assets mode requires asset_policy=frozen_exact"
            )
        if retrieval_policy != "disabled_exact_bindings":
            raise ArtifactValidationError(
                "frozen_assets mode requires retrieval_policy=disabled_exact_bindings"
            )
        if scale_policy != SCALE_FIXED_NATIVE:
            raise ArtifactValidationError(
                "frozen_assets mode requires scale_policy=fixed_native_scale"
            )
        if assets is None:
            raise ArtifactValidationError("frozen_assets mode requires assets identity")
        missing = [item["slot_id"] for item in objects if not item.get("asset_id")]
        if missing:
            raise ArtifactValidationError(
                "frozen_assets objects require exact asset_id bindings; "
                f"missing={missing}"
            )

    generation = value.get("generation") or {}
    evaluator = value.get("evaluator") or {}
    if not isinstance(generation, Mapping):
        raise ArtifactValidationError("comparison generation policy must be an object")
    if not isinstance(evaluator, Mapping):
        raise ArtifactValidationError("comparison evaluator policy must be an object")
    generation_policy = dict(generation)
    generation_policy.setdefault("budget_policy", "method_native_recorded")
    evaluator_policy = dict(evaluator)
    evaluator_policy.setdefault("policy", "same_canonical_run_evaluate")
    if evaluator_policy.get("policy") != "same_canonical_run_evaluate":
        raise ArtifactValidationError(
            "comparison evaluator.policy must be same_canonical_run_evaluate"
        )

    inventory_projection = [
        {
            "slot_id": item["slot_id"],
            "category": item["category"],
            "description": item["description"],
        }
        for item in sorted(objects, key=lambda item: str(item["slot_id"]))
    ]
    bindings = {
        str(item["slot_id"]): str(item["asset_id"])
        for item in sorted(objects, key=lambda item: str(item["slot_id"]))
        if item.get("asset_id")
    }
    normalized: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "case_id": case_id,
        "architecture": architecture,
        "architecture_sha256": architecture_sha256(architecture),
        "object_inventory_policy": inventory_policy,
        "objects": objects,
        "object_inventory_sha256": (
            canonical_json_sha256(inventory_projection)
            if inventory_policy == INVENTORY_FROZEN
            else None
        ),
        "asset_policy": asset_policy,
        "assets": assets,
        "asset_binding_sha256": (
            canonical_json_sha256(bindings) if bindings else None
        ),
        "scale_policy": scale_policy,
        "retrieval_policy": retrieval_policy,
        "generation": generation_policy,
        "evaluator": evaluator_policy,
    }
    return normalized


def _normalize_slot(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"comparison objects[{index}] must be an object")
    slot = {
        "slot_id": _required_text(
            value.get("slot_id"), f"comparison objects[{index}].slot_id"
        ),
        "category": _required_text(
            value.get("category"), f"comparison objects[{index}].category"
        ),
        "description": _required_text(
            value.get("description"), f"comparison objects[{index}].description"
        ),
    }
    if value.get("asset_id") is not None:
        slot["asset_id"] = _required_text(
            value.get("asset_id"), f"comparison objects[{index}].asset_id"
        )
    metadata = value.get("metadata") or {}
    if not isinstance(metadata, Mapping):
        raise ArtifactValidationError(
            f"comparison objects[{index}].metadata must be an object"
        )
    slot["metadata"] = dict(metadata)
    return slot


def _normalize_catalog_identity(value: Any) -> dict[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ArtifactValidationError("comparison assets must be an object")
    identity = {
        "catalog_id": _required_text(value.get("catalog_id"), "assets.catalog_id"),
        "catalog_version": _required_text(
            value.get("catalog_version"), "assets.catalog_version"
        ),
        "catalog_sha256": _required_text(
            value.get("catalog_sha256"), "assets.catalog_sha256"
        ),
    }
    digest = identity["catalog_sha256"]
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ArtifactValidationError("assets.catalog_sha256 must be lowercase SHA-256")
    return identity


def _required_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{path} must be a non-empty string")
    return value.strip()


__all__ = [
    "ComparisonProtocol",
    "FROZEN_ASSETS",
    "INVENTORY_FROZEN",
    "INVENTORY_METHOD_NATIVE",
    "NATIVE",
    "PROTOCOL_ID",
    "PROTOCOL_MODES",
    "PROTOCOL_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "SCALE_FIXED_NATIVE",
    "SCALE_METHOD_NATIVE",
    "SHARED_DB",
    "load_comparison_protocol",
    "validate_comparison_protocol",
]
