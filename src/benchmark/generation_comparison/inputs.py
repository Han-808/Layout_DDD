"""Build the public generator projection for one comparison case."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any

from benchmark.generation_comparison.catalog import CanonicalAssetCatalog
from benchmark.generation_comparison.protocol import (
    FROZEN_ASSETS,
    INVENTORY_FROZEN,
    NATIVE,
    ComparisonProtocol,
)
from benchmark.io_contracts import I2_NATURAL_LANGUAGE_STRUCTURE
from benchmark.nl_scene.converter import FINE_GRAINED
from benchmark.scene_io.validate import (
    ArtifactValidationError,
    validate_generation_input,
    validate_object_plan,
)


def build_controlled_generation_input(
    generation_input: Mapping[str, Any],
    *,
    protocol: ComparisonProtocol,
    catalog: CanonicalAssetCatalog | None,
    materialization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attach only public comparison controls to the existing input contract."""

    controlled = deepcopy(dict(generation_input))
    if protocol.mode == NATIVE:
        controlled["generation_comparison"] = _public_control(
            protocol=protocol,
            catalog=catalog,
            materialization=materialization,
        )
        validate_generation_input(controlled)
        return controlled

    if protocol.inventory_policy == INVENTORY_FROZEN:
        controlled["scene_request"]["structure"] = True
        plan = _object_plan(controlled, protocol)
        controlled["object_plan"] = plan
        contract = controlled["generation_contract"]
        contract["input_type"] = I2_NATURAL_LANGUAGE_STRUCTURE
        if protocol.mode == FROZEN_ASSETS:
            assert catalog is not None
            _apply_frozen_dimensions(plan, protocol=protocol, catalog=catalog)
            controlled["asset_selection"] = _asset_selection(
                controlled,
                protocol,
                catalog,
            )
            contract["input_mode"] = "structured_assets"
            contract["requires_asset_selection"] = True
            controlled.pop("generator_input", None)
        else:
            controlled.pop("asset_selection", None)
            contract["input_mode"] = "natural_language_structured"
            contract["requires_asset_selection"] = False
            controlled["generator_input"] = {
                "input_mode": "natural_language_structured",
                "request_id": str(controlled["request_id"]),
                "instruction": str(controlled["scene_request"]["instruction"]),
                "scene_type": str(controlled["scene_request"].get("scene_type") or "room"),
                "room": deepcopy(controlled["scene_request"].get("room")),
                "prompt_granularity": str(
                    controlled["scene_request"].get("prompt_granularity")
                    or FINE_GRAINED
                ),
                "object_plan": deepcopy(plan),
            }
    controlled["generation_comparison"] = _public_control(
        protocol=protocol,
        catalog=catalog,
        materialization=materialization,
    )
    validate_generation_input(controlled)
    return controlled


def _apply_frozen_dimensions(
    plan: dict[str, Any],
    *,
    protocol: ComparisonProtocol,
    catalog: CanonicalAssetCatalog,
) -> None:
    """Replace planning estimates with the exact selected physical dimensions."""

    bindings = protocol.bindings
    for item in plan.get("objects", []):
        if not isinstance(item, dict):
            continue
        slot_id = str(item.get("id") or "")
        asset_id = bindings.get(slot_id)
        if asset_id is None:
            raise ArtifactValidationError(
                f"frozen object plan slot {slot_id!r} has no exact asset binding"
            )
        asset = catalog.get(asset_id)
        item["estimated_size"] = list(asset["physical_dimensions"])
        metadata = item.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        metadata["comparison_scale_policy"] = "fixed_native_scale"
        item["metadata"] = metadata


def _public_control(
    *,
    protocol: ComparisonProtocol,
    catalog: CanonicalAssetCatalog | None,
    materialization: Mapping[str, Any] | None,
) -> dict[str, Any]:
    payload = protocol.as_dict()
    result: dict[str, Any] = {
        "schema_version": payload["schema_version"],
        "protocol_id": payload["protocol_id"],
        "protocol_version": payload["protocol_version"],
        "mode": payload["mode"],
        "case_id": payload["case_id"],
        "architecture": payload["architecture"],
        "architecture_sha256": payload["architecture_sha256"],
        "object_inventory_policy": payload["object_inventory_policy"],
        "objects": [
            {
                key: deepcopy(item[key])
                for key in ("slot_id", "category", "description", "asset_id")
                if key in item
            }
            for item in payload["objects"]
        ],
        "object_inventory_sha256": payload["object_inventory_sha256"],
        "asset_policy": payload["asset_policy"],
        "scale_policy": payload["scale_policy"],
        "retrieval_policy": payload["retrieval_policy"],
        "generation": payload["generation"],
    }
    if catalog is not None:
        result["catalog"] = catalog.identity
    if materialization is not None:
        result["method_materialization"] = {
            key: deepcopy(materialization[key])
            for key in (
                "adapter",
                "logical_to_native_slot",
                "comparison_control_path",
                "method_catalog_path",
                "materialized_catalog_sha256",
            )
            if key in materialization
        }
    return result


def _object_plan(
    generation_input: Mapping[str, Any],
    protocol: ComparisonProtocol,
) -> dict[str, Any]:
    supplied = generation_input.get("object_plan")
    if isinstance(supplied, Mapping):
        plan = deepcopy(dict(supplied))
        expected_slots = {str(item["slot_id"]) for item in protocol.objects}
        actual_slots = {
            str(item.get("id") or "")
            for item in plan.get("objects", [])
            if isinstance(item, Mapping)
        }
        if actual_slots != expected_slots:
            raise ArtifactValidationError(
                "public object_plan IDs must exactly match the frozen comparison "
                f"slots; missing={sorted(expected_slots - actual_slots)}, "
                f"unexpected={sorted(actual_slots - expected_slots)}"
            )
        for index, item in enumerate(plan.get("objects", [])):
            if not isinstance(item, dict):
                continue
            count = item.get("count", 1)
            if isinstance(count, bool) or int(count) != 1:
                raise ArtifactValidationError(
                    "controlled FrozenAssets object plans require one expanded "
                    f"instance per slot; object_plan.objects[{index}].count={count!r}"
                )
            item["count"] = 1
            metadata = item.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
            metadata["comparison_slot_id"] = str(item["id"])
            item["metadata"] = metadata
        validate_object_plan(plan)
        return plan
    request = generation_input["scene_request"]
    return {
        "request_id": str(generation_input["request_id"]),
        "scene_type": str(request.get("scene_type") or "room"),
        "scene_description": str(request.get("instruction") or ""),
        "objects": [
            {
                "id": str(item["slot_id"]),
                "category": str(item["category"]),
                "description": str(item["description"]),
                "metadata": {
                    "comparison_slot_id": str(item["slot_id"]),
                },
                "placement_intent": {
                    "absolute_relations": [],
                    "relative_relations": [],
                },
            }
            for item in protocol.objects
        ],
        "global_constraints": [],
        "relations": [],
    }


def _asset_selection(
    generation_input: Mapping[str, Any],
    protocol: ComparisonProtocol,
    catalog: CanonicalAssetCatalog,
) -> dict[str, Any]:
    public_objects = {
        str(item.get("id")): item
        for item in (generation_input.get("object_plan") or {}).get("objects", [])
        if isinstance(item, Mapping) and item.get("id") is not None
    }
    rows = []
    for slot in protocol.objects:
        asset = catalog.get(str(slot["asset_id"]))
        public_spec = public_objects.get(str(slot["slot_id"])) or slot
        selected = {
            "jid": asset["asset_id"],
            "category": asset["category"],
            "description": asset["description"],
            "size": asset["physical_dimensions"],
            "asset_ref": {
                "source_db": asset["source_db"],
                "asset_key": asset["asset_id"],
                **(
                    {"mesh_uri": asset["mesh_uri"]}
                    if asset.get("mesh_uri")
                    else {}
                ),
            },
            "asset_proxy": {
                "type": "external_asset_bbox",
                "bbox_size": asset["bbox_size_local"],
                "bbox_center_local": asset["bbox_center_local"],
            },
            "metadata": {
                "comparison_protocol": protocol.as_dict()["protocol_id"],
                "catalog_sha256": catalog.sha256,
                "canonical_front": asset.get("canonical_front"),
                "native_scale": asset["native_scale"],
                "comparison_catalog_provenance": {
                    **catalog.identity,
                    "asset_id": asset["asset_id"],
                    "bbox_size_local": asset["bbox_size_local"],
                    "bbox_center_local": asset["bbox_center_local"],
                    "native_scale": asset["native_scale"],
                    "physical_dimensions": asset["physical_dimensions"],
                    "canonical_front": asset.get("canonical_front"),
                },
            },
        }
        rows.append(
            {
                "object_id": slot["slot_id"],
                "object_spec": {
                    "category": public_spec["category"],
                    "description": public_spec["description"],
                    "count": 1,
                },
                "retrieval_query": None,
                "selected_asset": selected,
                "candidates": [],
                "selection_action": "select",
                "selection_decision": {
                    "action": "select",
                    "selected_jid": asset["asset_id"],
                    "reason": "exact frozen comparison binding",
                    "generation_request": None,
                },
                "selection_reason": "exact frozen comparison binding",
            }
        )
    return {"request_id": str(generation_input["request_id"]), "objects": rows}


__all__ = ["build_controlled_generation_input"]
