"""Fail-closed method eligibility for generation comparison protocols."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmark.adapters import get_adapter
from benchmark.generation_comparison.catalog import CanonicalAssetCatalog
from benchmark.generation_comparison.materializers import SUPPORTED_METHODS
from benchmark.generation_comparison.protocol import (
    FROZEN_ASSETS,
    INVENTORY_FROZEN,
    NATIVE,
    SCALE_FIXED_NATIVE,
    SHARED_DB,
    ComparisonProtocol,
)


ELIGIBILITY_SCHEMA_VERSION = "generation_comparison_eligibility_v1"
ELIGIBLE = "ELIGIBLE"
INELIGIBLE = "INELIGIBLE"

# LayoutVLM's released scene config directly carries the fixed object instances,
# exact UIDs, local bbox, and placement scale. Other upstream integrations need
# their thin runner to attest the listed controls. An attestation is persisted;
# output validation still independently rejects drift.
BUILTIN_CONTROLS: dict[str, set[str]] = {
    "catalog_placement": {
        "fixed_object_inventory",
        "exact_asset_ids",
        "fixed_native_scale",
    },
    "layout_gpt": set(),
    "direct_layout": set(),
    "layout_vlm": {
        "fixed_object_inventory",
        "exact_asset_ids",
        "fixed_native_scale",
    },
    "respace": set(),
    "scene_weaver": set(),
}


def check_method_eligibility(
    *,
    adapter_name: str,
    protocol: ComparisonProtocol,
    catalog: CanonicalAssetCatalog | None,
    adapter_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[dict[str, Any]] = []
    if adapter_name not in SUPPORTED_METHODS:
        reasons.append(
            _reason(
                "unsupported_method",
                "no controlled-generation materializer is registered",
            )
        )
        return _report(adapter_name, protocol, reasons, {}, set())

    adapter = get_adapter(adapter_name)
    capabilities = adapter.capabilities
    if not bool(getattr(adapter, "executable_integration", False)):
        reasons.append(
            _reason(
                "not_executable",
                "adapter has evaluation compatibility but no executable integration",
            )
        )
    if "single_room" not in capabilities.room_models:
        reasons.append(
            _reason("unsupported_room_model", "adapter does not support single_room")
        )
    if "axis_aligned_rectangle" not in capabilities.boundary_models:
        reasons.append(
            _reason(
                "unsupported_boundary_model",
                "adapter does not support axis_aligned_rectangle",
            )
        )
    if capabilities.architecture_features:
        reasons.append(
            _reason(
                "unsupported_protocol_semantics",
                "v1 common track does not generate semantic walls/openings/topology",
                declared_features=list(capabilities.architecture_features),
            )
        )
    if protocol.mode != NATIVE and not capabilities.preserves_asset_identity:
        reasons.append(
            _reason(
                "cannot_preserve_asset_identity",
                "controlled protocols require exact native-to-canonical asset identity",
            )
        )

    if protocol.mode != NATIVE:
        if catalog is None:
            reasons.append(
                _reason("catalog_missing", "controlled protocol requires a catalog")
            )
        elif protocol.catalog_identity != catalog.identity:
            reasons.append(
                _reason(
                    "catalog_mismatch",
                    "protocol catalog identity does not match supplied snapshot",
                    expected=protocol.catalog_identity,
                    actual=catalog.identity,
                )
            )
        elif protocol.mode == FROZEN_ASSETS:
            missing = sorted(set(protocol.bindings.values()) - set(catalog.asset_ids))
            if missing:
                reasons.append(
                    _reason(
                        "frozen_asset_missing",
                        "frozen bindings reference assets outside the catalog",
                        asset_ids=missing,
                    )
                )

    configured = _configured_controls(adapter_config)
    builtin = set(BUILTIN_CONTROLS.get(adapter_name, set()))
    if (
        adapter_name in {"catalog_placement", "layout_vlm"}
        and catalog is not None
        and any(
            list(asset.get("native_scale") or []) != [1.0, 1.0, 1.0]
            for asset in catalog.assets
        )
    ):
        # These direct scene-table paths use unit-scale processed assets. A
        # wrapper may attest non-unit scale support, but v1 must not assume it.
        builtin.discard("fixed_native_scale")
    available = builtin | configured
    required: set[str] = set()
    if protocol.mode == SHARED_DB:
        required.add("shared_catalog")
        if protocol.inventory_policy == INVENTORY_FROZEN:
            required.add("fixed_object_inventory")
        if protocol.scale_policy == SCALE_FIXED_NATIVE:
            required.add("fixed_native_scale")
    elif protocol.mode == FROZEN_ASSETS:
        required.update(
            {
                "fixed_object_inventory",
                "exact_asset_ids",
                "fixed_native_scale",
            }
        )
        if adapter_name == "scene_weaver":
            required.update(
                {
                    "frozen_iteration_bindings",
                    "no_object_insertion_removal",
                }
            )
    missing_controls = sorted(required - available)
    code_by_control = {
        "shared_catalog": "cannot_accept_shared_catalog",
        "fixed_object_inventory": "cannot_accept_fixed_object_inventory",
        "exact_asset_ids": "cannot_accept_fixed_asset_ids",
        "fixed_native_scale": "incompatible_scale_policy",
        "frozen_iteration_bindings": "cannot_freeze_iterative_asset_bindings",
        "no_object_insertion_removal": "cannot_freeze_iterative_object_inventory",
    }
    for control in missing_controls:
        reasons.append(
            _reason(
                code_by_control[control],
                f"adapter/runner has not declared comparison control {control!r}",
                required_control=control,
            )
        )
    return _report(
        adapter_name,
        protocol,
        reasons,
        configured,
        required,
        builtin=builtin,
    )


def require_method_eligible(report: Mapping[str, Any]) -> None:
    if report.get("status") == ELIGIBLE:
        return
    codes = [
        str(item.get("code"))
        for item in report.get("reasons", [])
        if isinstance(item, Mapping)
    ]
    from benchmark.scene_io.validate import ArtifactValidationError

    raise ArtifactValidationError(
        f"comparison run is INELIGIBLE for {report.get('method')}: {codes}"
    )


def _configured_controls(config: Mapping[str, Any] | None) -> set[str]:
    value = (config or {}).get("comparison_support")
    if isinstance(value, Mapping):
        return {str(key) for key, enabled in value.items() if enabled is True}
    if isinstance(value, (list, tuple, set)):
        return {str(item) for item in value}
    return set()


def _reason(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


def _report(
    adapter_name: str,
    protocol: ComparisonProtocol,
    reasons: list[dict[str, Any]],
    configured: set[str] | dict[str, Any],
    required: set[str],
    *,
    builtin: set[str] | None = None,
) -> dict[str, Any]:
    configured_values = (
        set(configured) if not isinstance(configured, dict) else set(configured)
    )
    return {
        "schema_version": ELIGIBILITY_SCHEMA_VERSION,
        "method": adapter_name,
        "protocol_id": protocol.as_dict()["protocol_id"],
        "protocol_version": protocol.as_dict()["protocol_version"],
        "protocol_mode": protocol.mode,
        "status": ELIGIBLE if not reasons else INELIGIBLE,
        "eligible": not reasons,
        "required_controls": sorted(required),
        "builtin_controls": sorted(
            BUILTIN_CONTROLS.get(adapter_name, set()) if builtin is None else builtin
        ),
        "runner_declared_controls": sorted(configured_values),
        "reasons": reasons,
    }


__all__ = [
    "BUILTIN_CONTROLS",
    "ELIGIBILITY_SCHEMA_VERSION",
    "ELIGIBLE",
    "INELIGIBLE",
    "check_method_eligibility",
    "require_method_eligible",
]
