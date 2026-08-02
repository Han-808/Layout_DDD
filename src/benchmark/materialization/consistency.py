from __future__ import annotations

from copy import deepcopy
from typing import Any

from benchmark.materialization.contracts import CONSISTENCY_GATE_VERSION
from benchmark.materialization.geometry import nearly_equal


CONSISTENCY_TOLERANCE_M = 1.0e-5


def run_consistency_gate(
    *,
    plan: dict[str, Any],
    normalized_scene: dict[str, Any],
    instance_registry: dict[str, Any],
    blend_inspection: dict[str, Any],
    hashes: dict[str, str],
    tolerance_m: float = CONSISTENCY_TOLERANCE_M,
) -> dict[str, Any]:
    """Cross-check every trusted representation before metrics or rendering."""

    mismatches: list[dict[str, Any]] = []
    checks: dict[str, dict[str, Any]] = {}
    plan_rows = _indexed(plan.get("instances"), "instance_id", "plan", mismatches)
    registry_rows = _indexed(
        instance_registry.get("instances"),
        "instance_id",
        "instance_registry",
        mismatches,
    )
    scene_rows = _indexed(
        normalized_scene.get("objects"),
        "metadata.materialization.instance_id",
        "normalized_scene",
        mismatches,
    )
    blend_rows = _indexed(
        blend_inspection.get("instances"),
        "instance_id",
        "trusted_blend",
        mismatches,
    )
    identifier_sets = {
        "plan": sorted(plan_rows),
        "instance_registry": sorted(registry_rows),
        "normalized_scene": sorted(scene_rows),
        "trusted_blend": sorted(blend_rows),
    }
    identifiers_match = len({tuple(values) for values in identifier_sets.values()}) == 1
    checks["instance_id_sets"] = {
        "status": "passed" if identifiers_match else "failed",
        "observed": identifier_sets,
    }
    if not identifiers_match:
        mismatches.append(
            {
                "code": "instance_id_set_mismatch",
                "path": "instances",
                "observed": identifier_sets,
            }
        )

    for instance_id in sorted(set(plan_rows) | set(registry_rows) | set(scene_rows) | set(blend_rows)):
        expected = plan_rows.get(instance_id)
        registry = registry_rows.get(instance_id)
        scene = scene_rows.get(instance_id)
        observed = blend_rows.get(instance_id)
        if not all(isinstance(value, dict) for value in (expected, registry, scene, observed)):
            continue
        scene_materialization = _nested(scene, "metadata.materialization")
        comparisons = {
            "evaluator_object_id": (
                expected.get("evaluator_object_id"),
                registry.get("evaluator_object_id"),
                scene.get("id"),
                observed.get("evaluator_object_id"),
            ),
            "asset_id": (
                expected.get("asset_id"),
                registry.get("asset_id"),
                _nested(scene, "asset_ref.asset_key"),
                observed.get("asset_id"),
            ),
            "slot_id": (
                expected.get("slot_id"),
                registry.get("slot_id"),
                scene_materialization.get("slot_id"),
                observed.get("slot_id"),
            ),
            "center_m": (
                expected.get("center_m"),
                _nested(registry, "transform.center_m"),
                scene.get("center"),
                observed.get("center_m"),
            ),
            "rotation_euler_xyz_deg": (
                expected.get("rotation_euler_xyz_deg"),
                _nested(registry, "transform.rotation_euler_xyz_deg"),
                scene.get("rotation"),
                observed.get("rotation_euler_xyz_deg"),
            ),
            "target_size_m": (
                expected.get("target_size_m"),
                _nested(registry, "transform.target_size_m"),
                scene_materialization.get("target_size_m"),
                observed.get("target_size_m"),
            ),
            "uniform_scale": (
                expected.get("uniform_scale"),
                _nested(registry, "transform.uniform_scale"),
                scene_materialization.get("uniform_scale"),
                observed.get("uniform_scale"),
            ),
            "local_bbox_size_m": (
                expected.get("local_bbox_size_m"),
                _nested(registry, "local_bbox.size_m"),
                scene.get("size"),
                observed.get("local_bbox_size_m"),
            ),
            "world_bounds": (
                expected.get("world_bounds"),
                registry.get("world_bounds"),
                scene_materialization.get("world_bounds"),
                observed.get("world_bounds"),
            ),
            "category": (
                expected.get("category"),
                scene.get("category"),
            ),
            "retrieval_category": (
                expected.get("retrieval_category"),
                scene.get("retrieval_category"),
            ),
            "description": (
                expected.get("description"),
                scene.get("description"),
                scene.get("desc"),
            ),
            "short_description": (
                expected.get("short_description"),
                scene.get("short_desc"),
            ),
            "catalog_bbox_center_m": (
                expected.get("catalog_bbox_center_m"),
                _nested(registry, "canonical_bbox.center_m"),
                scene_materialization.get("catalog_bbox_center_m"),
            ),
            "catalog_bbox_size_m": (
                expected.get("catalog_bbox_size_m"),
                _nested(registry, "canonical_bbox.size_m"),
                scene_materialization.get("catalog_bbox_size_m"),
            ),
            "asset_hashes": (
                expected.get("asset_hashes"),
                registry.get("asset_hashes"),
            ),
            "appearance_metadata": (
                expected.get("appearance_metadata"),
                registry.get("appearance_metadata"),
                _nested(
                    scene,
                    "metadata.appearance_provenance.catalog_metadata",
                ),
            ),
            "geometry_sha256": (
                registry.get("geometry_sha256"),
                scene_materialization.get("geometry_sha256"),
                observed.get("geometry_sha256"),
            ),
            "material_sha256": (
                registry.get("material_sha256"),
                _nested(scene, "metadata.appearance_provenance.material_sha256"),
                scene_materialization.get("material_sha256"),
                observed.get("material_sha256"),
            ),
            "asset_assembly_sha256": (
                registry.get("asset_assembly_sha256"),
                _nested(
                    scene,
                    "metadata.appearance_provenance.asset_assembly_sha256",
                ),
                scene_materialization.get("asset_assembly_sha256"),
                observed.get("asset_assembly_sha256"),
            ),
            "render_enabled": (
                registry.get("render_enabled"),
                observed.get("render_enabled"),
            ),
        }
        for field, values in comparisons.items():
            if not _all_equal(values, tolerance=tolerance_m):
                mismatches.append(
                    {
                        "code": "representation_mismatch",
                        "instance_id": instance_id,
                        "path": field,
                        "values": deepcopy(list(values)),
                    }
                )

    technical = blend_inspection.get("technical_state")
    if not isinstance(technical, dict):
        mismatches.append(
            {
                "code": "missing_blend_technical_state",
                "path": "trusted_blend.technical_state",
            }
        )
        technical = {}
    prohibited = sorted(
        key
        for key, value in technical.items()
        if key.startswith(("hidden_", "disabled_", "extra_", "missing_", "unsupported_"))
        and bool(value)
    )
    render_enabled = bool(technical.get("all_instances_render_enabled", False))
    no_extra = int(technical.get("extra_renderable_instance_count", 0) or 0) == 0
    checks["trusted_blend_technical_state"] = {
        "status": "passed" if render_enabled and no_extra and not prohibited else "failed",
        "all_instances_render_enabled": render_enabled,
        "extra_renderable_instance_count": technical.get(
            "extra_renderable_instance_count"
        ),
        "prohibited_flags": prohibited,
    }
    if not render_enabled:
        mismatches.append(
            {
                "code": "render_disabled_instance",
                "path": "trusted_blend.technical_state.all_instances_render_enabled",
            }
        )
    if not no_extra:
        mismatches.append(
            {
                "code": "extra_renderable_instance",
                "path": "trusted_blend.technical_state.extra_renderable_instance_count",
            }
        )
    for flag in prohibited:
        mismatches.append(
            {
                "code": "unsupported_trusted_blend_state",
                "path": f"trusted_blend.technical_state.{flag}",
            }
        )

    required_hashes = {
        "source_artifact_sha256",
        "normalized_scene_sha256",
        "instance_registry_sha256",
        "trusted_render_source_sha256",
        "materialization_plan_sha256",
        "trusted_blend_inspection_sha256",
        "provenance_core_sha256",
        "adapter_contract_revision_sha256",
    }
    missing_hashes = sorted(required_hashes - set(hashes))
    invalid_hashes = sorted(
        key
        for key in required_hashes & set(hashes)
        if not _is_sha256(hashes.get(key))
    )
    checks["hash_coverage"] = {
        "status": (
            "passed"
            if not missing_hashes and not invalid_hashes
            else "failed"
        ),
        "required": sorted(required_hashes),
        "missing": missing_hashes,
        "invalid": invalid_hashes,
    }
    for key in missing_hashes:
        mismatches.append(
            {
                "code": "missing_required_hash",
                "path": f"hashes.{key}",
            }
        )
    for key in invalid_hashes:
        mismatches.append(
            {
                "code": "invalid_required_hash",
                "path": f"hashes.{key}",
            }
        )

    status = "passed" if not mismatches else "failed"
    return {
        "gate_version": CONSISTENCY_GATE_VERSION,
        "status": status,
        "tolerance_m": float(tolerance_m),
        "checks": checks,
        "mismatches": mismatches,
        "hashes": dict(hashes),
    }


def _indexed(
    value: Any,
    path: str,
    label: str,
    mismatches: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        mismatches.append(
            {
                "code": "instances_not_list",
                "path": f"{label}.instances",
            }
        )
        return {}
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            mismatches.append(
                {
                    "code": "instance_not_object",
                    "path": f"{label}.instances[{index}]",
                }
            )
            continue
        raw = _nested(item, path)
        key = str(raw or "").strip()
        if not key:
            mismatches.append(
                {
                    "code": "missing_instance_id",
                    "path": f"{label}.instances[{index}].{path}",
                }
            )
            continue
        if key in result:
            mismatches.append(
                {
                    "code": "duplicate_instance_id",
                    "path": f"{label}.instances[{index}].{path}",
                    "instance_id": key,
                }
            )
            continue
        result[key] = item
    return result


def _nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _all_equal(values: tuple[Any, ...], *, tolerance: float) -> bool:
    first = values[0]
    return all(nearly_equal(first, value, tolerance=tolerance) for value in values[1:])


def _is_sha256(value: Any) -> bool:
    text = str(value or "").lower()
    return len(text) == 64 and all(
        character in "0123456789abcdef" for character in text
    )
