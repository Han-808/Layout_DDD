"""Fairness validation for controlled generation outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from benchmark.generation_comparison.catalog import (
    CanonicalAssetCatalog,
    physical_dimensions,
)
from benchmark.generation_comparison.identity import (
    architecture_sha256,
    canonical_json_sha256,
)
from benchmark.generation_comparison.native_identity import selected_iteration_objects
from benchmark.generation_comparison.protocol import (
    FROZEN_ASSETS,
    INVENTORY_FROZEN,
    NATIVE,
    SCALE_FIXED_NATIVE,
    SHARED_DB,
    ComparisonProtocol,
)


VALIDATION_SCHEMA_VERSION = "generation_comparison_validation_v1"


def validate_comparison_run(
    *,
    adapter_name: str,
    protocol: ComparisonProtocol,
    catalog: CanonicalAssetCatalog | None,
    canonical_scene: Mapping[str, Any],
    materialization: Mapping[str, Any] | None,
    native_selection: Mapping[str, Any] | None,
    method_input_architecture_sha256: str | None,
    eligibility: Mapping[str, Any],
    selected_iteration: int | None = None,
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    expected_architecture = protocol.architecture_hash
    output_architecture = architecture_sha256(
        {
            "room_model": "single_room",
            "boundary_model": "axis_aligned_rectangle",
            "room": {
                "boundary": canonical_scene.get("boundary"),
                "height": canonical_scene.get("scene_height"),
                "unit": "meter",
            },
        }
    )
    if method_input_architecture_sha256 != expected_architecture:
        violations.append(
            _violation(
                "architecture_mismatch",
                "method-native input architecture differs from comparison case",
                expected=expected_architecture,
                actual=method_input_architecture_sha256,
            )
        )
    if output_architecture != expected_architecture:
        violations.append(
            _violation(
                "architecture_mismatch",
                "canonical output architecture differs from comparison case",
                expected=expected_architecture,
                actual=output_architecture,
            )
        )
    if eligibility.get("eligible") is not True:
        violations.append(
            _violation(
                "unsupported_protocol_semantics",
                "method failed the pre-run eligibility gate",
                reasons=eligibility.get("reasons"),
            )
        )

    if protocol.mode != NATIVE:
        if catalog is None or protocol.catalog_identity != catalog.identity:
            violations.append(
                _violation(
                    "catalog_mismatch",
                    "active catalog identity differs from protocol",
                    expected=protocol.catalog_identity,
                    actual=catalog.identity if catalog is not None else None,
                )
            )
        if not isinstance(materialization, Mapping) or materialization.get(
            "materialized_catalog_sha256"
        ) != (catalog.sha256 if catalog is not None else None):
            violations.append(
                _violation(
                    "catalog_mismatch",
                    "method materialization does not prove the active catalog identity",
                )
            )

    objects = canonical_scene.get("objects")
    objects = list(objects) if isinstance(objects, Sequence) else []
    slot_map = (
        materialization.get("logical_to_native_slot")
        if isinstance(materialization, Mapping)
        else {}
    )
    slot_map = dict(slot_map) if isinstance(slot_map, Mapping) else {}
    native_to_logical = {str(native): str(logical) for logical, native in slot_map.items()}
    observed: dict[str, Mapping[str, Any]] = {}
    unexpected_native_ids: list[str] = []
    duplicate_slot_bindings: dict[str, list[str]] = {}
    for raw in objects:
        if not isinstance(raw, Mapping):
            continue
        native_id = str(raw.get("id") or "")
        logical = native_to_logical.get(native_id)
        if adapter_name == "catalog_placement":
            object_metadata = raw.get("metadata")
            object_metadata = (
                object_metadata if isinstance(object_metadata, Mapping) else {}
            )
            placement_metadata = object_metadata.get("catalog_placement")
            placement_metadata = (
                placement_metadata
                if isinstance(placement_metadata, Mapping)
                else {}
            )
            slot_id = placement_metadata.get("slot_id")
            logical = str(slot_id) if slot_id is not None else None
        if protocol.inventory_policy != INVENTORY_FROZEN:
            logical = native_id
        if logical is None:
            unexpected_native_ids.append(native_id)
        else:
            if logical in observed:
                duplicate_slot_bindings.setdefault(
                    logical,
                    [str(observed[logical].get("id") or "")],
                ).append(native_id)
            observed[logical] = raw

    if protocol.inventory_policy == INVENTORY_FROZEN:
        expected_slots = {str(item["slot_id"]) for item in protocol.objects}
        missing = sorted(expected_slots - set(observed))
        unexpected = sorted(unexpected_native_ids)
        if missing:
            violations.append(
                _violation(
                    "object_inventory_mismatch",
                    "expected comparison slots are missing",
                    missing_slot_ids=missing,
                )
            )
        if unexpected:
            violations.append(
                _violation(
                    "unexpected_object_insertion",
                    "generated output contains objects outside the frozen inventory",
                    native_object_ids=unexpected,
                )
            )
        if duplicate_slot_bindings:
            violations.append(
                _violation(
                    "object_inventory_mismatch",
                    "multiple generated objects bind the same frozen slot",
                    duplicate_slot_bindings=duplicate_slot_bindings,
                )
            )
        expected_categories = {
            str(item["slot_id"]): str(item["category"]) for item in protocol.objects
        }
        expected_descriptions = {
            str(item["slot_id"]): str(item["description"])
            for item in protocol.objects
        }
        changed_categories = {
            slot: {
                "expected": expected_categories[slot],
                "actual": str(obj.get("category") or ""),
            }
            for slot, obj in observed.items()
            if slot in expected_categories
            and str(obj.get("category") or "") != expected_categories[slot]
        }
        if changed_categories:
            violations.append(
                _violation(
                    "object_inventory_mismatch",
                    "generated object categories differ from frozen slots",
                    categories=changed_categories,
                )
            )
        changed_descriptions = {
            slot: {
                "expected": expected_descriptions[slot],
                "actual": str(obj.get("description") or ""),
            }
            for slot, obj in observed.items()
            if slot in expected_descriptions
            and str(obj.get("description") or "") != expected_descriptions[slot]
        }
        if changed_descriptions:
            violations.append(
                _violation(
                    "object_inventory_mismatch",
                    "generated object descriptions differ from frozen slots",
                    descriptions=changed_descriptions,
                )
            )

    native_objects = selected_iteration_objects(
        native_selection or {}, selected_iteration=selected_iteration
    )
    native_by_id = {
        str(item.get("native_object_id")): item
        for item in native_objects
        if isinstance(item, Mapping) and item.get("native_object_id")
    }
    canonical_native_ids = {
        str(obj.get("id")) for obj in objects if isinstance(obj, Mapping) and obj.get("id")
    }
    if set(native_by_id) != canonical_native_ids:
        violations.append(
            _violation(
                "native_canonical_inventory_mismatch",
                "native and canonical object identities differ across conversion",
                missing_from_canonical=sorted(set(native_by_id) - canonical_native_ids),
                missing_from_native=sorted(canonical_native_ids - set(native_by_id)),
            )
        )
    selected_assets: dict[str, str | None] = {}
    for logical, obj in observed.items():
        native_id = str(obj.get("id") or "")
        asset_ref = obj.get("asset_ref")
        asset_ref = asset_ref if isinstance(asset_ref, Mapping) else {}
        canonical_asset_id = _text(asset_ref.get("asset_key"), obj.get("jid"))
        native_row = native_by_id.get(native_id)
        native_asset_id = (
            _text(native_row.get("asset_id")) if isinstance(native_row, Mapping) else None
        )
        selected_assets[logical] = native_asset_id
        if not native_asset_id:
            violations.append(
                _violation(
                    "missing_native_asset_identity",
                    "selected asset ID is not persisted in the native artifact/binding",
                    slot_id=logical,
                    native_object_id=native_id,
                )
            )
        elif canonical_asset_id != native_asset_id:
            violations.append(
                _violation(
                    "asset_identity_conversion_mismatch",
                    "canonical conversion changed the persisted native asset ID",
                    slot_id=logical,
                    native_asset_id=native_asset_id,
                    canonical_asset_id=canonical_asset_id,
                )
            )

    if protocol.mode in {SHARED_DB, FROZEN_ASSETS} and catalog is not None:
        outside = {
            slot: asset_id
            for slot, asset_id in selected_assets.items()
            if asset_id is not None and asset_id not in set(catalog.asset_ids)
        }
        if outside:
            violations.append(
                _violation(
                    "catalog_mismatch",
                    "method selected assets outside the shared catalog",
                    selected_outside_catalog=outside,
                )
            )
        metadata_mismatches: dict[str, Any] = {}
        for slot, obj in observed.items():
            asset_id = selected_assets.get(slot)
            if not asset_id or asset_id not in set(catalog.asset_ids):
                continue
            asset = catalog.get(asset_id)
            asset_ref = obj.get("asset_ref")
            asset_ref = asset_ref if isinstance(asset_ref, Mapping) else {}
            object_metadata = obj.get("metadata")
            object_metadata = (
                object_metadata if isinstance(object_metadata, Mapping) else {}
            )
            provenance = object_metadata.get("comparison_catalog_provenance")
            asset_metadata = object_metadata.get("asset_metadata")
            asset_metadata = (
                asset_metadata if isinstance(asset_metadata, Mapping) else {}
            )
            if adapter_name == "catalog_placement":
                provenance = asset_metadata.get("comparison_catalog_provenance")
            provenance = provenance if isinstance(provenance, Mapping) else {}
            differences: dict[str, Any] = {}
            for field, actual in (
                ("source_db", asset_ref.get("source_db")),
                ("mesh_uri", asset_ref.get("mesh_uri")),
            ):
                expected = asset.get(field)
                if expected is not None and actual != expected:
                    differences[field] = {"expected": expected, "actual": actual}
            expected_front = asset.get("canonical_front")
            actual_front = object_metadata.get("canonical_front")
            if adapter_name == "catalog_placement":
                actual_front = asset_metadata.get("canonical_front")
            if expected_front is not None and actual_front != expected_front:
                differences["canonical_front"] = {
                    "expected": expected_front,
                    "actual": actual_front,
                }
            geometry_audit = object_metadata.get("geometry_audit")
            if adapter_name == "catalog_placement":
                geometry_audit = object_metadata.get("catalog_placement")
            geometry_audit = (
                geometry_audit if isinstance(geometry_audit, Mapping) else {}
            )
            audited_local_bbox = geometry_audit.get("asset_local_bbox_size")
            if adapter_name == "catalog_placement":
                audited_local_bbox = geometry_audit.get("catalog_bbox_size_m")
            if audited_local_bbox is not None and not _vectors_close(
                audited_local_bbox,
                asset["bbox_size_local"],
                tolerance=tolerance,
            ):
                differences["asset_local_bbox_size"] = {
                    "expected": asset["bbox_size_local"],
                    "actual": audited_local_bbox,
                }
            expected_provenance = {
                **catalog.identity,
                "asset_id": asset["asset_id"],
                "bbox_size_local": list(asset["bbox_size_local"]),
                "bbox_center_local": list(asset["bbox_center_local"]),
                "native_scale": list(asset["native_scale"]),
                "physical_dimensions": list(asset["physical_dimensions"]),
                "canonical_front": asset.get("canonical_front"),
            }
            if dict(provenance) != expected_provenance:
                differences["comparison_catalog_provenance"] = {
                    "expected": expected_provenance,
                    "actual": dict(provenance),
                }
            if differences:
                metadata_mismatches[slot] = differences
        if metadata_mismatches:
            violations.append(
                _violation(
                    "asset_metadata_mismatch",
                    "canonical asset metadata differs from the shared catalog snapshot",
                    objects=metadata_mismatches,
                )
            )

    if protocol.mode == FROZEN_ASSETS:
        replacements = {
            slot: {"expected": expected, "actual": selected_assets.get(slot)}
            for slot, expected in protocol.bindings.items()
            if selected_assets.get(slot) != expected
        }
        if replacements:
            violations.append(
                _violation(
                    "exact_asset_replacement",
                    "method changed one or more frozen exact asset bindings",
                    replacements=replacements,
                )
            )

    if protocol.scale_policy == SCALE_FIXED_NATIVE and catalog is not None:
        scale_changes: dict[str, Any] = {}
        for slot, obj in observed.items():
            asset_id = protocol.bindings.get(slot) or selected_assets.get(slot)
            if not asset_id or asset_id not in set(catalog.asset_ids):
                continue
            expected_size = physical_dimensions(catalog.get(asset_id))
            actual_size = obj.get("size")
            if not _vectors_close(actual_size, expected_size, tolerance=tolerance):
                scale_changes[slot] = {
                    "asset_id": asset_id,
                    "expected_physical_dimensions": expected_size,
                    "actual_evaluated_size": actual_size,
                }
            object_metadata = obj.get("metadata")
            object_metadata = (
                object_metadata if isinstance(object_metadata, Mapping) else {}
            )
            geometry_audit = object_metadata.get("geometry_audit")
            if adapter_name == "catalog_placement":
                geometry_audit = object_metadata.get("catalog_placement")
            geometry_audit = (
                geometry_audit if isinstance(geometry_audit, Mapping) else {}
            )
            audited_scale = geometry_audit.get("native_scale")
            if audited_scale is None:
                audited_scale = geometry_audit.get("applied_scale")
            if adapter_name == "catalog_placement":
                requested = geometry_audit.get("requested_uniform_scale")
                audited_scale = (
                    [requested, requested, requested]
                    if requested is not None
                    else None
                )
            if (
                adapter_name == "respace"
                and isinstance(audited_scale, Sequence)
                and not isinstance(audited_scale, (str, bytes))
                and len(audited_scale) == 3
            ):
                audited_scale = [
                    audited_scale[0],
                    audited_scale[2],
                    audited_scale[1],
                ]
            if audited_scale is not None and not _vectors_close(
                audited_scale,
                catalog.get(asset_id)["native_scale"],
                tolerance=tolerance,
            ):
                scale_changes.setdefault(
                    slot,
                    {
                        "asset_id": asset_id,
                        "expected_physical_dimensions": expected_size,
                        "actual_evaluated_size": actual_size,
                    },
                )["native_scale"] = {
                    "expected": catalog.get(asset_id)["native_scale"],
                    "actual": audited_scale,
                }
        if scale_changes:
            violations.append(
                _violation(
                    "scale_policy_violation",
                    "generated placed dimensions differ from fixed native asset dimensions",
                    objects=scale_changes,
                )
            )

    metadata = canonical_scene.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    compatibility = metadata.get("harness_compatibility")
    compatibility = compatibility if isinstance(compatibility, Mapping) else {}
    if (
        adapter_name == "catalog_placement"
        and metadata.get("asset_grounding")
        != "selected_frozen_catalog_exact_asset_id"
    ):
        violations.append(
            _violation(
                "converter_retrieval_policy_violation",
                "catalog_placement must use its frozen selected-asset input",
                actual=metadata.get("asset_grounding"),
            )
        )
    if (
        adapter_name != "catalog_placement"
        and compatibility.get("asset_resolution_policy") != "exact_only"
    ):
        violations.append(
            _violation(
                "converter_retrieval_policy_violation",
                "controlled conversion must use exact_only asset resolution",
                actual=compatibility.get("asset_resolution_policy"),
            )
        )

    observed_inventory = [
        {
            "slot_id": slot,
            "category": str(obj.get("category") or ""),
            "description": str(obj.get("description") or ""),
        }
        for slot, obj in sorted(observed.items())
    ]
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "method": adapter_name,
        "protocol_mode": protocol.mode,
        "eligible": eligibility.get("eligible") is True,
        "valid_comparison_run": not violations,
        "violations": violations,
        "architecture": {
            "comparison_case_sha256": expected_architecture,
            "method_input_sha256": method_input_architecture_sha256,
            "canonical_output_sha256": output_architecture,
        },
        "catalog": catalog.identity if catalog is not None else None,
        "object_inventory_sha256": protocol.inventory_sha256,
        "observed_object_inventory_sha256": (
            canonical_json_sha256(observed_inventory)
            if protocol.inventory_policy == INVENTORY_FROZEN
            else None
        ),
        "asset_binding_sha256": (
            canonical_json_sha256(selected_assets) if selected_assets else None
        ),
        "expected_asset_binding_sha256": protocol.binding_sha256,
        "selected_asset_ids": selected_assets,
        "selected_iteration": selected_iteration,
    }


def _violation(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def _vectors_close(value: Any, expected: Sequence[float], *, tolerance: float) -> bool:
    if not (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    ):
        return False
    try:
        actual = [float(component) for component in value]
    except (TypeError, ValueError):
        return False
    return all(
        math.isfinite(actual[index])
        and abs(actual[index] - float(expected[index])) <= tolerance
        for index in range(3)
    )


def _text(*values: Any) -> str | None:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


__all__ = ["VALIDATION_SCHEMA_VERSION", "validate_comparison_run"]
