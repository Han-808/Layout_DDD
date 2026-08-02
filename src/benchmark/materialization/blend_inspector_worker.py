"""Independent read-only inspector for sanitized and registered native blends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import subprocess
import sys
from array import array
from itertools import product
from pathlib import Path

import bpy
from mathutils import Euler, Vector


INSTANCE_ID_PROPERTY = "benchmark_instance_id"
EVALUATOR_ID_PROPERTY = "benchmark_evaluator_object_id"
CANONICAL_ID_PROPERTY = "benchmark_object_id"
ASSET_ID_PROPERTY = "benchmark_asset_id"
ROLE_PROPERTY = "benchmark_role"
TOLERANCE = 1.0e-5
ARCHITECTURE_ID_PROPERTY = "benchmark_architecture_id"
ARCHITECTURE_COLLECTION = "benchmark_architecture"
CANONICAL_WALL_IDS = (
    "north_wall",
    "south_wall",
    "east_wall",
    "west_wall",
)
RENDER_CAPABLE_OBJECT_TYPES = frozenset(
    {
        "MESH",
        "CURVE",
        "CURVES",
        "FONT",
        "SURFACE",
        "META",
        "VOLUME",
        "POINTCLOUD",
        "GREASEPENCIL",
        "GPENCIL",
    }
)


def main() -> None:
    args = _parse_args()
    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expected_path = (
        Path(args.expected_registry_json).expanduser().resolve()
        if args.expected_registry_json
        else None
    )
    catalog_path = (
        Path(args.catalog_plan_json).expanduser().resolve()
        if args.catalog_plan_json
        else None
    )
    expected_data = _load_json(expected_path) if expected_path else None
    catalog_data = _load_json(catalog_path) if catalog_path else None
    # A materialization plan is both a registry expectation and the frozen
    # catalog/transform authority for sanitized output.
    if catalog_data is None and isinstance(expected_data, dict) and expected_data.get(
        "schema_version"
    ) == "catalog_materialization_plan_v1":
        catalog_data = expected_data

    registry_records = _records(expected_data, "expected registry")
    catalog_records = _records(catalog_data, "catalog plan")
    expected_records = _merge_expected_records(
        registry_records,
        catalog_records,
        expected_data=expected_data,
        catalog_data=catalog_data,
    )
    report = _inspect(
        mode=args.mode,
        expected_records=expected_records,
        expected_data=expected_data,
        catalog_data=catalog_data,
        expected_path=expected_path,
        catalog_path=catalog_path,
    )
    _write_json(out_path, report)


def _parse_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", required=True)
    parser.add_argument(
        "--mode",
        choices=["sanitized", "registered_native"],
        required=True,
    )
    parser.add_argument("--expected-registry-json")
    parser.add_argument("--catalog-plan-json")
    return parser.parse_args(values)


def _inspect(
    *,
    mode: str,
    expected_records: dict[str, dict],
    expected_data: dict | None,
    catalog_data: dict | None,
    expected_path: Path | None,
    catalog_path: Path | None,
) -> dict:
    checks: list[dict] = []
    roots_by_id, root_resolution = _resolve_roots(expected_records)
    checks.extend(root_resolution)

    discovered_property_ids = {
        str(obj.get(INSTANCE_ID_PROPERTY))
        for obj in bpy.data.objects
        if obj.get(INSTANCE_ID_PROPERTY) is not None
        and _is_instance_root(obj)
    }
    if expected_records:
        _check(
            checks,
            "instance_id_set",
            set(roots_by_id) == set(expected_records),
            {
                "expected": sorted(expected_records),
                "resolved": sorted(roots_by_id),
                "property_roots": sorted(discovered_property_ids),
            },
            "instance_registry_mismatch",
        )

    assigned_objects: set = set()
    root_objects = set(roots_by_id.values())
    instances = []
    for instance_id in sorted(roots_by_id):
        root = roots_by_id[instance_id]
        expected = expected_records.get(instance_id, {})
        observed, descendants, instance_checks = _inspect_instance(
            root,
            instance_id=instance_id,
            expected=expected,
            mode=mode,
        )
        instances.append(observed)
        assigned_objects.update(descendants)
        checks.extend(instance_checks)

    nested_roots = [
        root.name
        for root in root_objects
        if any(ancestor in root_objects for ancestor in _ancestors(root))
    ]
    _check(
        checks,
        "instance_roots_not_nested",
        not nested_roots,
        {"nested_root_names": sorted(nested_roots)},
        "nested_instance_roots",
    )
    duplicate_root_names = [
        root.name
        for root in root_objects
        if list(roots_by_id.values()).count(root) > 1
    ]
    _check(
        checks,
        "instance_roots_not_merged",
        not duplicate_root_names,
        {"merged_root_names": sorted(set(duplicate_root_names))},
        "merged_instance_roots",
    )

    architecture_spec = _expected_architecture_spec(
        mode=mode,
        expected_data=expected_data,
        catalog_data=catalog_data,
    )
    architecture_validation = _validate_architecture_allowlist(
        boundary=architecture_spec.get("boundary"),
        scene_height=architecture_spec.get("scene_height"),
        architecture=architecture_spec.get("architecture"),
        required=bool(architecture_spec.get("required")),
    )
    _check(
        checks,
        "architecture_matches_materializer_allowlist",
        architecture_validation["passed"],
        {
            key: value
            for key, value in architecture_validation.items()
            if key != "passed"
        },
        "architecture_allowlist_mismatch",
    )
    architecture_objects = {
        bpy.data.objects[name]
        for name in architecture_validation["allowed_object_names"]
        if bpy.data.objects.get(name) is not None
    }
    visible_renderables = {
        obj
        for obj in bpy.data.objects
        if _is_render_capable_object(obj)
        and not _technically_hidden(obj)
    }
    # Preserve the previous stronger rule for MESH objects: even a hidden
    # unregistered mesh is not part of a trusted sanitized scene. Other
    # render-capable Blender types are rejected whenever technically visible.
    scene_renderables = visible_renderables | {
        obj for obj in bpy.data.objects if obj.type == "MESH"
    }
    unregistered = (
        scene_renderables - assigned_objects - architecture_objects
    )
    _check(
        checks,
        "no_unregistered_renderable_meshes",
        not unregistered,
        {
            "objects": sorted(
                f"{obj.name}:{obj.type}" for obj in unregistered
            )
        },
        "unregistered_renderable_mesh",
    )

    linked = sorted(
        obj.name
        for obj in bpy.data.objects
        if obj.library is not None
        or (obj.data is not None and obj.data.library is not None)
    )
    _check(
        checks,
        "no_linked_object_or_data_libraries",
        not bpy.data.libraries and not linked,
        {
            "library_count": len(bpy.data.libraries),
            "linked_object_names": linked,
        },
        "external_library_link",
    )

    restricted = assigned_objects | root_objects
    modifier_objects = sorted(obj.name for obj in restricted if obj.modifiers)
    constrained_objects = sorted(obj.name for obj in restricted if obj.constraints)
    animated_objects = sorted(
        obj.name for obj in restricted if obj.animation_data is not None
    )
    instanced_objects = sorted(
        obj.name
        for obj in restricted
        if getattr(obj, "instance_type", "NONE") != "NONE"
        or getattr(obj, "instance_collection", None) is not None
    )
    shape_key_objects = sorted(
        obj.name
        for obj in restricted
        if obj.type == "MESH" and getattr(obj.data, "shape_keys", None) is not None
    )
    disallowed_object_types = sorted(
        f"{obj.name}:{obj.type}"
        for obj in restricted
        if obj.type not in {"EMPTY", "MESH"}
    )
    animated_data = sorted(
        obj.name
        for obj in restricted
        if obj.data is not None
        and getattr(obj.data, "animation_data", None) is not None
    )
    animated_shape_keys = sorted(
        obj.name
        for obj in restricted
        if obj.type == "MESH"
        and getattr(obj.data, "shape_keys", None) is not None
        and getattr(obj.data.shape_keys, "animation_data", None) is not None
    )
    animated_materials = sorted(
        {
            material.name
            for obj in restricted
            if obj.type == "MESH"
            for material in obj.data.materials
            if material is not None
            and (
                getattr(material, "animation_data", None) is not None
                or (
                    material.node_tree is not None
                    and getattr(material.node_tree, "animation_data", None)
                    is not None
                )
            )
        }
    )
    driver_owners = _driver_owners()
    action_names = sorted(action.name for action in bpy.data.actions)
    _check(
        checks,
        "registered_assets_are_rigid",
        not (
            modifier_objects
            or constrained_objects
            or animated_objects
            or instanced_objects
            or shape_key_objects
            or disallowed_object_types
            or animated_data
            or animated_shape_keys
            or animated_materials
            or driver_owners
            or action_names
        ),
        {
            "modifier_objects": modifier_objects,
            "constrained_objects": constrained_objects,
            "animated_objects": animated_objects,
            "collection_instance_objects": instanced_objects,
            "shape_key_objects": shape_key_objects,
            "disallowed_object_types": disallowed_object_types,
            "animated_data_objects": animated_data,
            "animated_shape_key_objects": animated_shape_keys,
            "animated_materials_or_node_trees": animated_materials,
            "driver_owners": driver_owners,
            "actions": action_names,
        },
        "non_rigid_or_procedural_asset",
    )

    external_references = _external_references()
    _check(
        checks,
        "no_unpacked_external_references",
        not external_references,
        {"references": external_references},
        "external_file_reference",
    )

    hidden_registered = sorted(
        obj.name
        for obj in restricted
        if _technically_hidden(obj)
    )
    _check(
        checks,
        "registered_instances_render_enabled",
        not hidden_registered,
        {"hidden_or_disabled_object_names": hidden_registered},
        "registered_instance_hidden",
    )
    registered_render_state_mismatches = _object_render_state_mismatches(
        restricted
    )
    _check(
        checks,
        "registered_instances_have_canonical_render_state",
        not registered_render_state_mismatches,
        {"mismatches": registered_render_state_mismatches},
        "registered_asset_render_state_override",
    )

    appearance_state = _appearance_state()
    if mode == "sanitized":
        compositor_nodes = sum(
            _compositor_node_count(scene)
            for scene in bpy.data.scenes
        )
        _check(
            checks,
            "sanitized_scene_has_no_camera_light_or_compositor",
            not bpy.data.cameras
            and not bpy.data.lights
            and compositor_nodes == 0,
            {
                "camera_count": len(bpy.data.cameras),
                "light_count": len(bpy.data.lights),
                "compositor_node_count": compositor_nodes,
            },
            "submitted_appearance_state_in_sanitized_blend",
        )
        sanitized_render_state = _validate_sanitized_render_state()
        _check(
            checks,
            "sanitized_scene_render_state_is_canonical",
            sanitized_render_state["passed"],
            {
                key: value
                for key, value in sanitized_render_state.items()
                if key != "passed"
            },
            "sanitized_scene_render_state_override",
        )
        _check_sanitized_scene_provenance(
            checks,
            expected_data=expected_data,
            expected_path=expected_path,
        )
    else:
        _check(
            checks,
            "native_appearance_state_is_inventory_only",
            True,
            {
                "policy": "ignored_and_never_propagated",
                **appearance_state,
            },
        )

    passed = all(item["passed"] for item in checks)
    reason_codes = sorted(
        {
            reason
            for item in checks
            if not item["passed"]
            for reason in item.get("reason_codes", [])
        }
    )
    technical_state = {
        "all_instances_render_enabled": (
            not hidden_registered
            and len(roots_by_id) == len(expected_records or roots_by_id)
        ),
        "extra_renderable_instance_count": len(unregistered),
        "hidden_instance_count": len(hidden_registered),
        "missing_instance_count": max(
            0,
            len(expected_records) - len(roots_by_id),
        ),
        "unsupported_linked_library_count": len(bpy.data.libraries) + len(linked),
        "unsupported_external_reference_count": len(external_references),
        "unsupported_non_rigid_state_count": sum(
            len(values)
            for values in (
                modifier_objects,
                constrained_objects,
                animated_objects,
                instanced_objects,
                shape_key_objects,
                disallowed_object_types,
                animated_data,
                animated_shape_keys,
                animated_materials,
                driver_owners,
                action_names,
            )
        )
        + len(registered_render_state_mismatches),
        "camera_count": len(bpy.data.cameras),
        "light_count": len(bpy.data.lights),
    }
    catalog_placement = None
    if mode == "registered_native" and passed:
        catalog_placement = {
            "schema_version": "catalog_placement_v1",
            "instances": [
                _placement_instance(item)
                for item in instances
            ],
        }
    return {
        "backend": "blender_read_only_registered_scene_inspector_v1",
        "status": "passed" if passed else "failed",
        "mode": mode,
        "blender_version": bpy.app.version_string,
        "source_blend": (
            str(Path(bpy.data.filepath).resolve()) if bpy.data.filepath else None
        ),
        "source_scene_saved": False,
        "render_invocation_count": 0,
        "auto_execution_required": False,
        "expected_registry_path": (
            expected_path.as_posix() if expected_path is not None else None
        ),
        "catalog_plan_path": (
            catalog_path.as_posix() if catalog_path is not None else None
        ),
        "checks": checks,
        "reason_codes": reason_codes,
        "instances": instances,
        "catalog_placement": catalog_placement,
        "technical_state": technical_state,
        "unregistered_renderables": sorted(obj.name for obj in unregistered),
        "architecture_objects": sorted(obj.name for obj in architecture_objects),
        "scene_state": appearance_state,
    }


def _placement_instance(item: dict) -> dict:
    result = {
        "instance_id": item["instance_id"],
        "asset_id": item["asset_id"],
        "center_m": item["center_m"],
        "target_size_m": item["target_size_m"],
        "rotation_euler_xyz_deg": item["rotation_euler_xyz_deg"],
    }
    if item.get("slot_id") is not None:
        result["slot_id"] = item["slot_id"]
    return result


def _resolve_roots(
    expected_records: dict[str, dict],
) -> tuple[dict[str, object], list[dict]]:
    checks: list[dict] = []
    roots: dict[str, object] = {}
    property_roots = [
        obj
        for obj in bpy.data.objects
        if obj.get(INSTANCE_ID_PROPERTY) is not None and _is_instance_root(obj)
    ]
    if not expected_records:
        for root in property_roots:
            instance_id = str(root.get(INSTANCE_ID_PROPERTY))
            if instance_id in roots:
                _check(
                    checks,
                    f"root_resolution:{instance_id}",
                    False,
                    {"candidate_names": [roots[instance_id].name, root.name]},
                    "duplicate_instance_root",
                )
                continue
            roots[instance_id] = root
        return roots, checks

    for instance_id, expected in expected_records.items():
        root_name = _first_text(
            expected.get("root_object_name"),
            expected.get("blender_root_name"),
            expected.get("native_root_name"),
            expected.get("blend_object"),
            expected.get("root_name"),
            expected.get("object_name"),
        )
        candidates = []
        if root_name:
            named = bpy.data.objects.get(root_name)
            if named is not None:
                candidates.append(named)
        else:
            candidates = [
                root
                for root in property_roots
                if str(root.get(INSTANCE_ID_PROPERTY)) == instance_id
            ]
        unique = list(dict.fromkeys(candidates))
        _check(
            checks,
            f"root_resolution:{instance_id}",
            len(unique) == 1,
            {
                "requested_root_name": root_name or None,
                "candidate_names": [obj.name for obj in unique],
            },
            "missing_or_ambiguous_instance_root",
        )
        if len(unique) == 1:
            roots[instance_id] = unique[0]
    return roots, checks


def _inspect_instance(
    root,
    *,
    instance_id: str,
    expected: dict,
    mode: str,
) -> tuple[dict, set, list[dict]]:
    checks: list[dict] = []
    descendants = {root, *_descendants(root)}
    meshes = [obj for obj in descendants if obj.type == "MESH"]
    _check(
        checks,
        f"instance_has_mesh:{instance_id}",
        bool(meshes),
        {"root_object_name": root.name, "mesh_names": sorted(obj.name for obj in meshes)},
        "instance_has_no_mesh",
    )
    _check(
        checks,
        f"instance_root_not_parented:{instance_id}",
        root.parent is None,
        {"parent_name": root.parent.name if root.parent else None},
        "nested_or_parented_instance_root",
    )

    require_properties = mode == "sanitized"
    property_expectations = {
        INSTANCE_ID_PROPERTY: instance_id,
        EVALUATOR_ID_PROPERTY: expected.get("evaluator_object_id"),
        CANONICAL_ID_PROPERTY: expected.get("evaluator_object_id"),
        ASSET_ID_PROPERTY: expected.get("asset_id"),
    }
    property_mismatches = {}
    for key, expected_value in property_expectations.items():
        observed = root.get(key)
        if expected_value is None:
            continue
        if require_properties or observed is not None:
            if str(observed) != str(expected_value):
                property_mismatches[key] = {
                    "expected": expected_value,
                    "observed": observed,
                }
    if require_properties and root.get(ROLE_PROPERTY) != "instance_root":
        property_mismatches[ROLE_PROPERTY] = {
            "expected": "instance_root",
            "observed": root.get(ROLE_PROPERTY),
        }
    if require_properties:
        trusted_property_expectations = {
            "benchmark_slot_id": expected.get("slot_id"),
            "benchmark_catalog_snapshot_id": expected.get(
                "_catalog_snapshot_id"
            ),
            "benchmark_materialization_revision": expected.get(
                "_materialization_revision"
            ),
            "benchmark_adapter_contract_revision": expected.get(
                "_adapter_contract_revision"
            ),
            "benchmark_center_m": expected.get("center_m"),
            "benchmark_target_size_m": expected.get("target_size_m"),
            "benchmark_rotation_euler_xyz_deg": expected.get(
                "rotation_euler_xyz_deg"
            ),
            "benchmark_uniform_scale": expected.get("uniform_scale"),
            "benchmark_catalog_bbox_center_m": expected.get(
                "catalog_bbox_center_m"
            ),
            "benchmark_catalog_bbox_size_m": expected.get(
                "catalog_bbox_size_m"
            ),
            "benchmark_local_bbox_size_m": expected.get("local_bbox_size_m"),
            "benchmark_mesh_sha256": (
                expected.get("asset_hashes", {}).get("mesh_sha256")
                if isinstance(expected.get("asset_hashes"), dict)
                else None
            ),
            "benchmark_asset_tree_sha256": (
                expected.get("asset_hashes", {}).get("asset_tree_sha256")
                if isinstance(expected.get("asset_hashes"), dict)
                else None
            ),
        }
        for key, expected_value in trusted_property_expectations.items():
            observed = root.get(key)
            if key == "benchmark_slot_id" and expected_value is None:
                if observed is not None:
                    property_mismatches[key] = {
                        "expected": None,
                        "observed": observed,
                    }
                continue
            if expected_value is None:
                continue
            observed_value = _json_scalar_or_vector(observed)
            if not _close_json(observed_value, expected_value):
                property_mismatches[key] = {
                    "expected": expected_value,
                    "observed": observed_value,
                }
    _check(
        checks,
        f"trusted_identity_properties:{instance_id}",
        not property_mismatches,
        {"mismatches": property_mismatches},
        "instance_identity_property_mismatch",
    )
    descendant_mismatches = []
    if require_properties:
        for obj in descendants - {root}:
            if obj.get(ROLE_PROPERTY) != "asset_descendant":
                descendant_mismatches.append(
                    {
                        "object_name": obj.name,
                        "property": ROLE_PROPERTY,
                        "expected": "asset_descendant",
                        "observed": obj.get(ROLE_PROPERTY),
                    }
                )
            for key in (
                INSTANCE_ID_PROPERTY,
                EVALUATOR_ID_PROPERTY,
                CANONICAL_ID_PROPERTY,
                ASSET_ID_PROPERTY,
            ):
                expected_value = property_expectations.get(key)
                if expected_value is not None and str(obj.get(key)) != str(expected_value):
                    descendant_mismatches.append(
                        {
                            "object_name": obj.name,
                            "property": key,
                            "expected": expected_value,
                            "observed": obj.get(key),
                        }
                    )
    _check(
        checks,
        f"descendant_identity_properties:{instance_id}",
        not descendant_mismatches,
        {"mismatches": descendant_mismatches},
        "descendant_identity_property_mismatch",
    )

    matrix_validation = _validate_rigid_uniform_matrix(root.matrix_world)
    _check(
        checks,
        f"rigid_uniform_transform:{instance_id}",
        matrix_validation["valid"],
        matrix_validation,
        "invalid_rigid_uniform_transform",
    )

    bounds = _observed_bounds(root, meshes, matrix_validation)
    expected_mismatches = _compare_expected_geometry(
        root,
        bounds,
        expected,
        matrix_validation=matrix_validation,
    )
    _check(
        checks,
        f"catalog_transform_consistency:{instance_id}",
        not expected_mismatches,
        {"mismatches": expected_mismatches},
        "catalog_transform_mismatch",
    )

    geometry_fingerprint = _geometry_fingerprint(root, meshes)
    mesh_data_fingerprint = _geometry_fingerprint(
        root,
        meshes,
        include_object_transform=False,
        transform_vertex_coordinates=False,
    )
    mesh_assembly_fingerprint = _geometry_fingerprint(
        root,
        meshes,
        include_object_transform=True,
        transform_vertex_coordinates=False,
    )
    material_fingerprint = _material_fingerprint(meshes)
    asset_assembly_fingerprint = _asset_assembly_fingerprint(root, meshes)
    if mode == "registered_native":
        unsupported_material_state = _unsupported_native_material_state(
            meshes
        )
        _check(
            checks,
            f"native_material_state_supported:{instance_id}",
            not unsupported_material_state,
            {"unsupported_state": unsupported_material_state},
            "unsupported_native_material_state",
        )
        registry_record = expected.get("_registry_record")
        expected_geometry = _expected_fingerprint(
            registry_record if isinstance(registry_record, dict) else {},
            "geometry",
        )
        expected_material = _expected_fingerprint(
            registry_record if isinstance(registry_record, dict) else {},
            "material",
        )
        fingerprint_mismatches = {}
        for label, observed, wanted in (
            ("geometry_sha256", geometry_fingerprint, expected_geometry),
            ("material_sha256", material_fingerprint, expected_material),
        ):
            if (
                wanted is None
                or observed is None
                or str(observed).lower() != str(wanted).lower()
            ):
                fingerprint_mismatches[label] = {
                    "expected": wanted,
                    "observed": observed,
                    "expected_required": True,
                }
        _check(
            checks,
            f"benchmark_owned_fingerprints:{instance_id}",
            not fingerprint_mismatches,
            {"mismatches": fingerprint_mismatches},
            "registered_asset_fingerprint_mismatch",
        )
    exported_world_bounds = bounds.get("world_bounds_observed")
    expected_rotation = _vec3_or_none(expected.get("rotation_euler_xyz_deg"))
    if isinstance(exported_world_bounds, dict) and expected_rotation is not None:
        exported_world_bounds = json.loads(json.dumps(exported_world_bounds))
        exported_world_bounds["obb"][
            "rotation_euler_xyz_deg"
        ] = expected_rotation
    observed = {
        "instance_id": instance_id,
        "evaluator_object_id": _first_text(
            root.get(EVALUATOR_ID_PROPERTY),
            root.get(CANONICAL_ID_PROPERTY),
            expected.get("evaluator_object_id"),
        ),
        "asset_id": _first_text(
            root.get(ASSET_ID_PROPERTY),
            expected.get("asset_id"),
        ),
        "slot_id": (
            root.get("benchmark_slot_id")
            if root.get("benchmark_slot_id") is not None
            else expected.get("slot_id")
        ),
        "root_object_name": root.name,
        "root_object_type": root.type,
        "descendant_object_names": sorted(obj.name for obj in descendants - {root}),
        "mesh_object_names": sorted(obj.name for obj in meshes),
        "root_matrix_world": _matrix_rows(root.matrix_world),
        "root_location": [float(value) for value in root.matrix_world.translation],
        "center_m": bounds.get("center_m_observed"),
        "target_size_m": expected.get("target_size_m"),
        "rotation_euler_xyz_deg": expected.get("rotation_euler_xyz_deg"),
        "uniform_scale": matrix_validation.get("uniform_scale"),
        "local_bbox_size_m": bounds.get("local_bbox_size_m_observed"),
        "world_bounds": exported_world_bounds,
        "render_enabled": not any(
            _technically_hidden(obj) for obj in descendants
        ),
        "uniform_scale_observed": matrix_validation.get("uniform_scale"),
        "rotation_matrix_observed": matrix_validation.get("rotation_matrix"),
        **bounds,
        "geometry_sha256": geometry_fingerprint,
        "mesh_data_sha256": mesh_data_fingerprint,
        "mesh_assembly_sha256": mesh_assembly_fingerprint,
        "material_sha256": material_fingerprint,
        "asset_assembly_sha256": asset_assembly_fingerprint,
        "technical_visibility": {
            "hide_render": bool(root.hide_render),
            "hide_viewport": bool(root.hide_viewport),
            "hide_get": bool(root.hide_get()),
            "descendants_all_render_enabled": not any(
                _technically_hidden(obj) for obj in descendants
            ),
        },
        "custom_properties": _custom_properties(root),
    }
    return observed, descendants, checks


def _compare_expected_geometry(
    root,
    bounds: dict,
    expected: dict,
    *,
    matrix_validation: dict,
) -> list[dict]:
    mismatches: list[dict] = []
    expected_center = _vec3_or_none(expected.get("center_m"))
    expected_rotation = _vec3_or_none(expected.get("rotation_euler_xyz_deg"))
    catalog_center = _vec3_or_none(expected.get("catalog_bbox_center_m"))
    catalog_size = _vec3_or_none(expected.get("catalog_bbox_size_m"))
    local_size = _vec3_or_none(expected.get("local_bbox_size_m"))
    expected_scale = _number_or_none(expected.get("uniform_scale"))

    comparisons = (
        (
            "catalog_bbox_center_m",
            bounds.get("catalog_bbox_center_m_observed"),
            catalog_center,
        ),
        (
            "catalog_bbox_size_m",
            bounds.get("catalog_bbox_size_m_observed"),
            catalog_size,
        ),
        (
            "center_m",
            bounds.get("center_m_observed"),
            expected_center,
        ),
        (
            "local_bbox_size_m",
            bounds.get("local_bbox_size_m_observed"),
            local_size,
        ),
        (
            "uniform_scale",
            matrix_validation.get("uniform_scale"),
            expected_scale,
        ),
    )
    for field, observed, wanted in comparisons:
        if wanted is not None and not _close_json(observed, wanted):
            mismatches.append(
                {"field": field, "expected": wanted, "observed": observed}
            )
    if expected_rotation is not None:
        wanted_matrix = [
            [float(value) for value in row]
            for row in Euler(
                [math.radians(value) for value in expected_rotation],
                "XYZ",
            ).to_matrix()
        ]
        observed_rotation = matrix_validation.get("rotation_matrix")
        if not _close_json(observed_rotation, wanted_matrix):
            mismatches.append(
                {
                    "field": "rotation_matrix",
                    "expected": wanted_matrix,
                    "observed": observed_rotation,
                }
            )
    if (
        expected_center is not None
        and catalog_center is not None
        and expected_rotation is not None
        and expected_scale is not None
    ):
        rotation = Euler(
            [math.radians(value) for value in expected_rotation],
            "XYZ",
        ).to_matrix()
        expected_location = Vector(expected_center) - rotation @ (
            Vector(catalog_center) * expected_scale
        )
        observed_location = [float(value) for value in root.matrix_world.translation]
        if not _close_json(observed_location, list(expected_location)):
            mismatches.append(
                {
                    "field": "root_location",
                    "expected": list(expected_location),
                    "observed": observed_location,
                }
            )
    expected_world_bounds = expected.get("world_bounds")
    if isinstance(expected_world_bounds, dict):
        observed_world_bounds = bounds.get("world_bounds_observed")
        observed_comparable = _without_euler_provenance(observed_world_bounds)
        expected_comparable = _without_euler_provenance(expected_world_bounds)
        if not _close_json(observed_comparable, expected_comparable):
            mismatches.append(
                {
                    "field": "world_bounds",
                    "expected": expected_world_bounds,
                    "observed": observed_world_bounds,
                }
            )
    mesh_hash = (
        expected.get("asset_hashes", {}).get("mesh_sha256")
        if isinstance(expected.get("asset_hashes"), dict)
        else None
    )
    observed_mesh_hash = root.get("benchmark_mesh_sha256")
    if observed_mesh_hash is not None and mesh_hash is not None and str(
        observed_mesh_hash
    ).lower() != str(mesh_hash).lower():
        mismatches.append(
            {
                "field": "mesh_sha256",
                "expected": mesh_hash,
                "observed": observed_mesh_hash,
            }
        )
    return mismatches


def _observed_bounds(root, meshes: list, matrix_validation: dict) -> dict:
    if not meshes or not matrix_validation["invertible"]:
        return {
            "catalog_bbox_center_m_observed": None,
            "catalog_bbox_size_m_observed": None,
            "center_m_observed": None,
            "local_bbox_size_m_observed": None,
            "world_bounds_observed": None,
        }
    inverse = root.matrix_world.inverted()
    local_points = [
        inverse @ obj.matrix_world @ Vector(corner)
        for obj in meshes
        for corner in obj.bound_box
    ]
    minimum = Vector(
        [min(point[axis] for point in local_points) for axis in range(3)]
    )
    maximum = Vector(
        [max(point[axis] for point in local_points) for axis in range(3)]
    )
    local_center = (minimum + maximum) * 0.5
    catalog_size = maximum - minimum
    scale = float(matrix_validation["uniform_scale"])
    local_size = catalog_size * scale
    corners_world = []
    for signs in product((-1.0, 1.0), repeat=3):
        local = Vector(
            [
                local_center[axis]
                + signs[axis] * catalog_size[axis] * 0.5
                for axis in range(3)
            ]
        )
        corners_world.append(
            [float(value) for value in root.matrix_world @ local]
        )
    center_world = [
        float(value) for value in root.matrix_world @ local_center
    ]
    minimum_world = [
        min(corner[axis] for corner in corners_world) for axis in range(3)
    ]
    maximum_world = [
        max(corner[axis] for corner in corners_world) for axis in range(3)
    ]
    rotation = matrix_validation["rotation_matrix"]
    rotation_euler = [
        math.degrees(value)
        for value in root.matrix_world.to_3x3().normalized().to_euler("XYZ")
    ]
    world_bounds = {
        "obb": {
            "center_m": center_world,
            "local_size_m": [float(value) for value in local_size],
            "rotation_euler_xyz_deg": rotation_euler,
            "rotation_matrix": rotation,
            "corners_m": corners_world,
        },
        "aabb": {
            "min_m": minimum_world,
            "max_m": maximum_world,
            "size_m": [
                maximum_world[axis] - minimum_world[axis]
                for axis in range(3)
            ],
        },
    }
    return {
        "catalog_bbox_center_m_observed": [
            float(value) for value in local_center
        ],
        "catalog_bbox_size_m_observed": [
            float(value) for value in catalog_size
        ],
        "center_m_observed": center_world,
        "local_bbox_size_m_observed": [float(value) for value in local_size],
        "world_bounds_observed": world_bounds,
    }


def _validate_rigid_uniform_matrix(matrix) -> dict:
    values = [
        float(matrix[row][column])
        for row in range(4)
        for column in range(4)
    ]
    finite = all(math.isfinite(value) for value in values)
    affine = finite and _close_json(
        [values[12], values[13], values[14], values[15]],
        [0.0, 0.0, 0.0, 1.0],
    )
    columns = [
        Vector((matrix[0][column], matrix[1][column], matrix[2][column]))
        for column in range(3)
    ]
    scales = [float(column.length) for column in columns]
    invertible = finite and all(scale > TOLERANCE for scale in scales)
    uniform = (
        sum(scales) / 3.0 if invertible else None
    )
    uniform_scale = bool(
        invertible
        and max(scales) - min(scales)
        <= max(TOLERANCE, TOLERANCE * max(scales))
    )
    orthogonal = bool(
        invertible
        and all(
            abs(columns[left].dot(columns[right]))
            <= max(TOLERANCE, TOLERANCE * scales[left] * scales[right])
            for left, right in ((0, 1), (0, 2), (1, 2))
        )
    )
    determinant = float(matrix.to_3x3().determinant()) if finite else float("nan")
    positive_determinant = math.isfinite(determinant) and determinant > 0.0
    rotation_matrix = None
    if invertible:
        rotation_matrix = [
            [
                float(matrix[row][column]) / scales[column]
                for column in range(3)
            ]
            for row in range(3)
        ]
    return {
        "valid": bool(
            finite
            and affine
            and invertible
            and uniform_scale
            and orthogonal
            and positive_determinant
        ),
        "finite": finite,
        "affine": affine,
        "invertible": invertible,
        "uniform": uniform_scale,
        "orthogonal": orthogonal,
        "positive_determinant": positive_determinant,
        "determinant": determinant if math.isfinite(determinant) else None,
        "axis_scales": scales,
        "uniform_scale": uniform,
        "rotation_matrix": rotation_matrix,
    }


def _geometry_fingerprint(
    root,
    meshes: list,
    *,
    include_object_transform: bool = True,
    transform_vertex_coordinates: bool = True,
) -> str | None:
    if not meshes:
        return None
    inverse = root.matrix_world.inverted_safe()
    object_digests = []
    for obj in meshes:
        transform = inverse @ obj.matrix_world
        mesh = obj.data
        payload = {
            "vertices": [],
            "edges": [],
            "loops": [],
            "polygons": [],
            "uv_layers": [],
            "color_attributes": [],
            "attributes": [],
        }
        if include_object_transform:
            payload["transform"] = _matrix_rows(transform)
        for vertex in obj.data.vertices:
            point = (
                transform @ vertex.co
                if transform_vertex_coordinates
                else vertex.co
            )
            payload["vertices"].append(
                {
                    "co": [float(point[0]), float(point[1]), float(point[2])],
                    "normal": [
                        float(value)
                        for value in getattr(vertex, "normal", (0.0, 0.0, 0.0))
                    ],
                }
            )
        for edge in mesh.edges:
            payload["edges"].append(
                {
                    "vertices": [int(value) for value in edge.vertices],
                    "use_edge_sharp": bool(
                        getattr(edge, "use_edge_sharp", False)
                    ),
                    "use_seam": bool(getattr(edge, "use_seam", False)),
                }
            )
        for loop in mesh.loops:
            payload["loops"].append(
                {
                    "vertex_index": int(loop.vertex_index),
                    "edge_index": int(loop.edge_index),
                    "normal": [
                        float(value)
                        for value in getattr(loop, "normal", (0.0, 0.0, 0.0))
                    ],
                }
            )
        for polygon in mesh.polygons:
            payload["polygons"].append(
                {
                    "vertices": [int(value) for value in polygon.vertices],
                    "loop_start": int(polygon.loop_start),
                    "loop_total": int(polygon.loop_total),
                    "material_index": int(polygon.material_index),
                    "use_smooth": bool(polygon.use_smooth),
                    "normal": [
                        float(value)
                        for value in getattr(
                            polygon, "normal", (0.0, 0.0, 0.0)
                        )
                    ],
                }
            )
        for layer in mesh.uv_layers:
            payload["uv_layers"].append(
                {
                    "name": str(layer.name),
                    "active_render": bool(layer.active_render),
                    "uv": [
                        [float(value) for value in item.uv]
                        for item in layer.data
                    ],
                }
            )
        for attribute in getattr(mesh, "color_attributes", ()):
            payload["color_attributes"].append(
                _mesh_attribute_payload(attribute)
            )
        color_attribute_names = {
            item["name"] for item in payload["color_attributes"]
        }
        for attribute in getattr(mesh, "attributes", ()):
            if str(attribute.name) in color_attribute_names:
                continue
            payload["attributes"].append(
                _mesh_attribute_payload(attribute)
            )
        encoded = json.dumps(
            _quantized_fingerprint_payload(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object_digests.append(hashlib.sha256(encoded).hexdigest())
    combined = hashlib.sha256()
    for digest in sorted(object_digests):
        combined.update(bytes.fromhex(digest))
    return combined.hexdigest()


def _material_fingerprint(meshes: list) -> str:
    object_digests = []
    for obj in meshes:
        object_payload = _object_material_binding_payload(obj)
        object_digests.append(
            hashlib.sha256(
                json.dumps(
                    object_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
        )
    combined = hashlib.sha256()
    for digest in sorted(object_digests):
        combined.update(digest.encode("ascii"))
    return combined.hexdigest()


def _asset_assembly_fingerprint(root, meshes: list) -> str | None:
    """Bind the complete rigid hierarchy and each mesh's effective materials."""

    if not meshes:
        return None
    inverse = root.matrix_world.inverted_safe()
    child_digests = []
    members = set(meshes)
    members.update(_descendants(root))
    members.discard(root)
    if root in meshes:
        members.add(root)
    for obj in members:
        child_payload = {
            "object_type": str(obj.type),
            "root_relative_transform": _matrix_rows(
                inverse @ obj.matrix_world
            ),
            "object_shader_state": {
                "color": [
                    float(component)
                    for component in getattr(
                        obj,
                        "color",
                        (1.0, 1.0, 1.0, 1.0),
                    )
                ],
                "pass_index": int(getattr(obj, "pass_index", 0)),
                # Identity/provenance stamps are injected by the fresh
                # materializer but are not required on a registered-native
                # hierarchy; the signed registry is their trust authority.
                # Non-benchmark properties remain bound because they can be
                # consumed by shader/object attribute mechanisms.
                "custom_properties": {
                    key: value
                    for key, value in _custom_properties(obj).items()
                    if not key.startswith("benchmark_")
                },
                "instance_type": str(
                    getattr(obj, "instance_type", "NONE")
                ),
                "has_instance_collection": (
                    getattr(obj, "instance_collection", None) is not None
                ),
            },
            "local_geometry_topology_sha256": (
                _geometry_fingerprint(
                    root,
                    [obj],
                    include_object_transform=False,
                    transform_vertex_coordinates=False,
                )
                if obj.type == "MESH"
                else None
            ),
            "material_binding": (
                _object_material_binding_payload(obj)
                if obj.type == "MESH"
                else None
            ),
        }
        encoded = json.dumps(
            _quantized_fingerprint_payload(child_payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        child_digests.append(hashlib.sha256(encoded).hexdigest())
    combined = hashlib.sha256()
    for digest in sorted(child_digests):
        combined.update(bytes.fromhex(digest))
    return combined.hexdigest()


def _object_material_binding_payload(obj) -> dict:
    data_material_digests = []
    for material in obj.data.materials:
        if material is None:
            data_material_digests.append("none")
        else:
            data_material_digests.append(_material_signature(material))
    effective_material_slots = []
    for slot in obj.material_slots:
        material = slot.material
        effective_material_slots.append(
            {
                "link": str(slot.link),
                "material_sha256": (
                    _material_signature(material)
                    if material is not None
                    else "none"
                ),
            }
        )
    return {
        "data_material_slots": data_material_digests,
        "effective_material_slots": effective_material_slots,
        "polygon_material_indices": [
            int(polygon.material_index)
            for polygon in obj.data.polygons
        ],
    }


def _unsupported_native_material_state(meshes: list) -> list[dict]:
    """Reject material dependencies whose render semantics are not serialized."""

    issues: list[dict] = []
    seen_materials: set[int] = set()
    seen_trees: set[int] = set()
    for obj in sorted(meshes, key=lambda item: item.name):
        materials = [
            material
            for material in obj.data.materials
            if material is not None
        ]
        materials.extend(
            slot.material
            for slot in obj.material_slots
            if slot.material is not None
        )
        for material in materials:
            material_identity = _runtime_identity(material)
            if material_identity in seen_materials:
                continue
            seen_materials.add(material_identity)
            if material.use_nodes and material.node_tree is not None:
                _collect_unsupported_node_tree_state(
                    material.node_tree,
                    material_name=material.name,
                    issues=issues,
                    seen_trees=seen_trees,
                )
    return sorted(
        issues,
        key=lambda item: json.dumps(
            item,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _collect_unsupported_node_tree_state(
    node_tree,
    *,
    material_name: str,
    issues: list[dict],
    seen_trees: set[int],
) -> None:
    identity = _runtime_identity(node_tree)
    if identity in seen_trees:
        return
    seen_trees.add(identity)
    allowed_pointers = {
        "parent",
        "node_tree",
        "image",
        "color_ramp",
        "mapping",
        "color_mapping",
        "texture_mapping",
        "image_user",
    }
    allowed_collections = {
        "inputs",
        "outputs",
        "internal_links",
        # Blender 5.x stores collapsed/expanded socket-panel UI state here;
        # it has no shader execution semantics.
        "panel_states",
    }
    for node in getattr(node_tree, "nodes", ()):
        node_type = str(getattr(node, "bl_idname", ""))
        if node_type in {
            "ShaderNodeObjectInfo",
            "ShaderNodeScript",
            "ShaderNodeTexIES",
        }:
            issues.append(
                {
                    "material": material_name,
                    "node": str(node.name),
                    "node_type": node_type,
                    "state": "unsupported_implicit_or_external_shader_input",
                }
            )
        if node_type == "ShaderNodeAttribute":
            attribute_type = str(
                getattr(node, "attribute_type", "")
            ).upper()
            if attribute_type not in {"", "GEOMETRY"}:
                issues.append(
                    {
                        "material": material_name,
                        "node": str(node.name),
                        "node_type": node_type,
                        "state": "unsupported_attribute_source",
                        "attribute_type": attribute_type,
                    }
                )

        rna = getattr(node, "bl_rna", None)
        for prop in getattr(rna, "properties", ()):
            name = str(getattr(prop, "identifier", ""))
            prop_type = str(getattr(prop, "type", ""))
            if not name or name in {"rna_type", "id_data"}:
                continue
            try:
                raw = getattr(node, name)
            except Exception:
                issues.append(
                    {
                        "material": material_name,
                        "node": str(node.name),
                        "node_type": node_type,
                        "state": "unreadable_rna_property",
                        "property": name,
                    }
                )
                continue
            if (
                prop_type == "POINTER"
                and raw is not None
                and name not in allowed_pointers
            ):
                issues.append(
                    {
                        "material": material_name,
                        "node": str(node.name),
                        "node_type": node_type,
                        "state": "unsupported_pointer_property",
                        "property": name,
                        "pointer_type": str(
                            getattr(
                                getattr(raw, "bl_rna", None),
                                "identifier",
                                type(raw).__name__,
                            )
                        ),
                    }
                )
            elif (
                prop_type == "COLLECTION"
                and name not in allowed_collections
            ):
                try:
                    count = len(raw)
                except Exception:
                    count = -1
                if count:
                    issues.append(
                        {
                            "material": material_name,
                            "node": str(node.name),
                            "node_type": node_type,
                            "state": "unsupported_collection_property",
                            "property": name,
                            "count": count,
                        }
                    )

        image = getattr(node, "image", None)
        if image is not None:
            image_payload = _image_fingerprint_payload(image)
            image_source = str(getattr(image, "source", ""))
            if image_source not in {"FILE", "GENERATED"}:
                issues.append(
                    {
                        "material": material_name,
                        "node": str(node.name),
                        "node_type": node_type,
                        "state": "unsupported_dynamic_image_source",
                        "image": str(image.name),
                        "image_source": image_source,
                    }
                )
            if (
                image_source == "FILE"
                and not bool(getattr(image, "packed_files", ()))
            ):
                issues.append(
                    {
                        "material": material_name,
                        "node": str(node.name),
                        "node_type": node_type,
                        "state": "unpacked_image_dependency",
                        "image": str(image.name),
                    }
                )
            if image_payload.get("pixels_sha256") is None:
                issues.append(
                    {
                        "material": material_name,
                        "node": str(node.name),
                        "node_type": node_type,
                        "state": "image_pixels_not_canonicalizable",
                        "image": str(image.name),
                        "pixel_error": image_payload.get("pixel_error"),
                    }
                )
        nested_tree = getattr(node, "node_tree", None)
        if nested_tree is not None:
            _collect_unsupported_node_tree_state(
                nested_tree,
                material_name=material_name,
                issues=issues,
                seen_trees=seen_trees,
            )


def _material_signature(material) -> str:
    payload = {
        "diffuse_color": [float(value) for value in material.diffuse_color],
        "metallic": float(getattr(material, "metallic", 0.0)),
        "roughness": float(getattr(material, "roughness", 0.0)),
        "use_nodes": bool(material.use_nodes),
        "surface_render_method": str(
            getattr(material, "surface_render_method", "")
        ),
        "blend_method": str(getattr(material, "blend_method", "")),
        "alpha_threshold": float(
            getattr(material, "alpha_threshold", 0.0)
        ),
        "use_backface_culling": bool(
            getattr(material, "use_backface_culling", False)
        ),
        "properties": _rna_fingerprint_properties(
            material,
            excluded={
                "diffuse_color",
                "metallic",
                "roughness",
                "use_nodes",
                "surface_render_method",
                "blend_method",
                "alpha_threshold",
                "use_backface_culling",
                "node_tree",
            },
        ),
        "node_tree": None,
    }
    if material.use_nodes and material.node_tree is not None:
        payload["node_tree"] = _node_tree_fingerprint_payload(
            material.node_tree,
            active=[],
            memo={},
        )
    encoded = json.dumps(
        _quantized_fingerprint_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _node_tree_fingerprint_payload(
    node_tree,
    *,
    active: list[int],
    memo: dict[int, str],
) -> dict:
    """Serialize a shader node tree recursively without pointer-derived output."""

    identity = _runtime_identity(node_tree)
    if identity in active:
        return {
            "cycle_ref_depth": len(active) - active.index(identity),
            "tree_type": str(getattr(node_tree, "bl_idname", "")),
        }
    if identity in memo:
        return {"shared_node_tree_sha256": memo[identity]}

    active.append(identity)
    try:
        nodes = sorted(
            list(getattr(node_tree, "nodes", ())),
            key=lambda node: (
                str(getattr(node, "name", "")),
                str(getattr(node, "bl_idname", "")),
            ),
        )
        node_indices = {
            _runtime_identity(node): index
            for index, node in enumerate(nodes)
        }
        node_payloads = []
        for node in nodes:
            node_payload = {
                "name": str(getattr(node, "name", "")),
                "type": str(getattr(node, "bl_idname", "")),
                "mute": bool(getattr(node, "mute", False)),
                "label": str(getattr(node, "label", "")),
                "properties": _node_visual_properties(node),
                "inputs": [
                    _node_socket_payload(socket, index=index)
                    for index, socket in enumerate(
                        getattr(node, "inputs", ())
                    )
                ],
                "outputs": [
                    _node_socket_payload(socket, index=index)
                    for index, socket in enumerate(
                        getattr(node, "outputs", ())
                    )
                ],
            }
            image = getattr(node, "image", None)
            if image is not None:
                node_payload["image"] = _image_fingerprint_payload(image)
            nested_tree = getattr(node, "node_tree", None)
            if nested_tree is not None:
                node_payload["group_node_tree"] = (
                    _node_tree_fingerprint_payload(
                        nested_tree,
                        active=active,
                        memo=memo,
                    )
                )
            node_payloads.append(node_payload)

        links = []
        for link in getattr(node_tree, "links", ()):
            from_node = node_indices.get(_runtime_identity(link.from_node))
            to_node = node_indices.get(_runtime_identity(link.to_node))
            if from_node is None or to_node is None:
                continue
            links.append(
                [
                    int(from_node),
                    _socket_identifier(
                        link.from_node,
                        link.from_socket,
                        outputs=True,
                    ),
                    int(to_node),
                    _socket_identifier(
                        link.to_node,
                        link.to_socket,
                        outputs=False,
                    ),
                ]
            )
        links.sort(
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        payload = {
            "tree_type": str(getattr(node_tree, "bl_idname", "")),
            "properties": _rna_fingerprint_properties(
                node_tree,
                excluded={"nodes", "links", "interface"},
            ),
            "interface": _node_tree_interface_payload(node_tree),
            "nodes": node_payloads,
            "links": links,
        }
    finally:
        active.pop()

    encoded = json.dumps(
        _quantized_fingerprint_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    memo[identity] = hashlib.sha256(encoded).hexdigest()
    return payload


def _node_socket_payload(socket, *, index: int) -> dict:
    value = None
    if hasattr(socket, "default_value"):
        value = _fingerprint_scalar_or_vector(socket.default_value)
        if value is _UNSERIALIZABLE_FINGERPRINT_VALUE:
            value = None
    return {
        "index": int(index),
        "identifier": str(getattr(socket, "identifier", "")),
        "name": str(getattr(socket, "name", "")),
        "type": str(getattr(socket, "bl_idname", "")),
        "linked": bool(getattr(socket, "is_linked", False)),
        "value": value,
        "properties": _rna_fingerprint_properties(
            socket,
            excluded={
                "default_value",
                "identifier",
                "name",
                "is_linked",
                "links",
                "node",
            },
        ),
    }


def _socket_identifier(node, socket, *, outputs: bool) -> dict:
    sockets = list(
        getattr(node, "outputs" if outputs else "inputs", ())
    )
    try:
        index = sockets.index(socket)
    except ValueError:
        index = -1
    return {
        "index": int(index),
        "identifier": str(getattr(socket, "identifier", "")),
        "name": str(getattr(socket, "name", "")),
    }


def _node_tree_interface_payload(node_tree) -> list[dict]:
    interface = getattr(node_tree, "interface", None)
    items = getattr(interface, "items_tree", None)
    if items is not None:
        return [
            {
                "index": int(index),
                "item_type": str(getattr(item, "item_type", "")),
                "identifier": str(getattr(item, "identifier", "")),
                "name": str(getattr(item, "name", "")),
                "in_out": str(getattr(item, "in_out", "")),
                "socket_type": str(getattr(item, "socket_type", "")),
                "properties": _rna_fingerprint_properties(
                    item,
                    excluded={
                        "item_type",
                        "identifier",
                        "name",
                        "in_out",
                        "socket_type",
                        "parent",
                    },
                ),
            }
            for index, item in enumerate(items)
        ]
    result = []
    for in_out, sockets in (
        ("INPUT", getattr(node_tree, "inputs", ())),
        ("OUTPUT", getattr(node_tree, "outputs", ())),
    ):
        for index, socket in enumerate(sockets):
            result.append(
                {
                    "index": int(index),
                    "item_type": "SOCKET",
                    "identifier": str(
                        getattr(socket, "identifier", "")
                    ),
                    "name": str(getattr(socket, "name", "")),
                    "in_out": in_out,
                    "socket_type": str(
                        getattr(socket, "bl_socket_idname", "")
                    ),
                    "properties": _rna_fingerprint_properties(
                        socket,
                        excluded={"identifier", "name"},
                    ),
                }
            )
    return result


def _image_fingerprint_payload(image) -> dict:
    packed_hashes = []
    for packed in getattr(image, "packed_files", ()):
        packed_file = getattr(packed, "packed_file", None)
        if packed_file is None:
            continue
        packed_hashes.append(
            hashlib.sha256(bytes(packed_file.data)).hexdigest()
        )
    colorspace = getattr(image, "colorspace_settings", None)
    tiles = [
        {
            "number": int(getattr(tile, "number", 0)),
            "label": str(getattr(tile, "label", "")),
        }
        for tile in getattr(image, "tiles", ())
    ]
    pixels_sha256 = None
    pixel_error = None
    try:
        pixels = image.pixels
        pixel_count = len(pixels)
        pixel_values = array("f", [0.0]) * pixel_count
        if pixel_count:
            pixels.foreach_get(pixel_values)
        pixels_sha256 = hashlib.sha256(
            pixel_values.tobytes()
        ).hexdigest()
    except Exception as exc:
        pixel_count = None
        pixel_error = f"{type(exc).__name__}: {exc}"
    return {
        "source": str(getattr(image, "source", "")),
        "size": [int(value) for value in getattr(image, "size", ())],
        "packed": bool(getattr(image, "packed_files", ())),
        "packed_sha256": sorted(packed_hashes),
        "pixel_count": pixel_count,
        "pixels_sha256": pixels_sha256,
        "pixel_error": pixel_error,
        "colorspace": str(getattr(colorspace, "name", "")),
        "colorspace_is_data": bool(
            getattr(colorspace, "is_data", False)
        ),
        "alpha_mode": str(getattr(image, "alpha_mode", "")),
        "tiles": sorted(tiles, key=lambda item: item["number"]),
        "properties": _rna_fingerprint_properties(
            image,
            excluded={
                "name",
                "filepath",
                "filepath_raw",
                "packed_file",
                "packed_files",
                "pixels",
                "colorspace_settings",
                "tiles",
                "size",
                "source",
                "alpha_mode",
            },
        ),
    }


def _runtime_identity(value) -> int:
    pointer = getattr(value, "as_pointer", None)
    if callable(pointer):
        try:
            return int(pointer())
        except Exception:
            pass
    return id(value)


def _expected_architecture_spec(
    *,
    mode: str,
    expected_data: dict | None,
    catalog_data: dict | None,
) -> dict:
    if mode == "sanitized":
        for source in (catalog_data, expected_data):
            request = source.get("request") if isinstance(source, dict) else None
            if not isinstance(request, dict):
                continue
            if (
                isinstance(request.get("boundary"), list)
                and request.get("scene_height") is not None
                and isinstance(request.get("architecture"), dict)
            ):
                return {
                    "boundary": request["boundary"],
                    "scene_height": request["scene_height"],
                    "architecture": request["architecture"],
                    "source": "trusted_materialization_plan",
                    "required": True,
                }
        return {
            "boundary": None,
            "scene_height": None,
            "architecture": None,
            "source": "missing_trusted_materialization_plan",
            "required": True,
        }

    # Registered-native inspection may receive only a benchmark-owned instance
    # registry. Architecture is inventory-only for that source and is never
    # propagated, but if materializer provenance is present it still must
    # describe the exact materializer architecture rather than acting as a tag
    # based bypass.
    scene = bpy.context.scene
    raw_boundary = scene.get("benchmark_request_boundary")
    raw_architecture = scene.get("benchmark_architecture_contract")
    try:
        boundary = (
            json.loads(str(raw_boundary)) if raw_boundary is not None else None
        )
    except (TypeError, json.JSONDecodeError):
        boundary = None
    try:
        architecture = (
            json.loads(str(raw_architecture))
            if raw_architecture is not None
            else None
        )
    except (TypeError, json.JSONDecodeError):
        architecture = None
    try:
        scene_height = (
            float(scene.get("benchmark_scene_height"))
            if scene.get("benchmark_scene_height") is not None
            else None
        )
    except (TypeError, ValueError):
        scene_height = None
    return {
        "boundary": boundary,
        "scene_height": scene_height,
        "architecture": architecture,
        "source": (
            "source_materializer_provenance"
            if any(
                value is not None
                for value in (raw_boundary, raw_architecture, scene_height)
            )
            else "not_declared"
        ),
        "required": False,
    }


def _validate_architecture_allowlist(
    *,
    boundary,
    scene_height,
    architecture,
    required: bool = True,
) -> dict:
    tagged = [
        obj
        for obj in bpy.data.objects
        if obj.get(ARCHITECTURE_ID_PROPERTY) is not None
        or obj.get(ROLE_PROPERTY) == "architecture"
    ]
    declared = any(
        value is not None for value in (boundary, scene_height, architecture)
    )
    if not declared:
        mismatches = []
        if required:
            mismatches.append({"code": "missing_trusted_architecture_spec"})
        if tagged:
            mismatches.append(
                {
                    "code": "architecture_without_trusted_spec",
                    "objects": sorted(obj.name for obj in tagged),
                }
            )
        return {
            "passed": not mismatches,
            "expected_architecture_ids": [],
            "observed_architecture_ids": sorted(
                str(obj.get(ARCHITECTURE_ID_PROPERTY) or "")
                for obj in tagged
            ),
            "allowed_object_names": [],
            "mismatches": mismatches,
        }

    spec_errors: list[dict] = []
    resolved_boundary = _architecture_boundary(boundary, spec_errors)
    try:
        resolved_height = float(scene_height)
    except (TypeError, ValueError):
        resolved_height = None
    if (
        resolved_height is None
        or not math.isfinite(resolved_height)
        or resolved_height <= 0.0
    ):
        spec_errors.append(
            {
                "code": "invalid_architecture_scene_height",
                "observed": scene_height,
            }
        )
    active_wall_ids = _architecture_active_wall_ids(
        architecture,
        spec_errors,
    )
    expected_ids = ["floor", *active_wall_ids]
    by_id: dict[str, list] = {}
    malformed_tags = []
    for obj in tagged:
        architecture_id = str(
            obj.get(ARCHITECTURE_ID_PROPERTY) or ""
        ).strip()
        if not architecture_id:
            malformed_tags.append(
                {
                    "object_name": obj.name,
                    "code": "missing_architecture_id",
                }
            )
            continue
        by_id.setdefault(architecture_id, []).append(obj)

    mismatches = list(spec_errors)
    mismatches.extend(malformed_tags)
    observed_ids = sorted(by_id)
    duplicate_ids = {
        architecture_id: sorted(obj.name for obj in objects)
        for architecture_id, objects in by_id.items()
        if len(objects) != 1
    }
    if duplicate_ids:
        mismatches.append(
            {
                "code": "duplicate_architecture_id",
                "ids": duplicate_ids,
            }
        )
    unexpected_ids = sorted(set(by_id) - set(expected_ids))
    missing_ids = sorted(set(expected_ids) - set(by_id))
    if unexpected_ids:
        mismatches.append(
            {
                "code": "unexpected_architecture_id",
                "ids": unexpected_ids,
            }
        )
    if missing_ids:
        mismatches.append(
            {"code": "missing_architecture_id", "ids": missing_ids}
        )

    allowed_names: list[str] = []
    if not spec_errors and resolved_boundary is not None and resolved_height is not None:
        expected_geometry = _expected_architecture_geometry(
            resolved_boundary,
            resolved_height,
            active_wall_ids,
        )
        expected_materials = {
            architecture_id: _expected_architecture_material_signature(
                architecture_id
            )
            for architecture_id in expected_ids
        }
        for architecture_id in expected_ids:
            candidates = by_id.get(architecture_id, [])
            if len(candidates) != 1:
                continue
            obj = candidates[0]
            object_mismatches = _architecture_object_mismatches(
                obj,
                architecture_id=architecture_id,
                expected_geometry=expected_geometry[architecture_id],
                expected_material_signature=expected_materials[
                    architecture_id
                ],
            )
            if object_mismatches:
                mismatches.append(
                    {
                        "code": "architecture_object_mismatch",
                        "architecture_id": architecture_id,
                        "object_name": obj.name,
                        "mismatches": object_mismatches,
                    }
                )
            else:
                allowed_names.append(obj.name)
    return {
        "passed": not mismatches,
        "expected_architecture_ids": expected_ids,
        "observed_architecture_ids": observed_ids,
        "allowed_object_names": sorted(allowed_names),
        "mismatches": mismatches,
    }


def _architecture_boundary(value, errors: list[dict]) -> list[list[float]] | None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            not isinstance(point, (list, tuple)) or len(point) != 2
            for point in value
        )
    ):
        errors.append(
            {"code": "invalid_architecture_boundary", "observed": value}
        )
        return None
    try:
        boundary = [
            [float(point[0]), float(point[1])] for point in value
        ]
    except (TypeError, ValueError):
        errors.append(
            {"code": "invalid_architecture_boundary", "observed": value}
        )
        return None
    if not all(math.isfinite(component) for point in boundary for component in point):
        errors.append(
            {"code": "invalid_architecture_boundary", "observed": value}
        )
        return None
    xs = sorted({point[0] for point in boundary})
    ys = sorted({point[1] for point in boundary})
    if (
        len(xs) != 2
        or len(ys) != 2
        or set(map(tuple, boundary))
        != {
            (xs[0], ys[0]),
            (xs[1], ys[0]),
            (xs[1], ys[1]),
            (xs[0], ys[1]),
        }
    ):
        errors.append(
            {"code": "non_rectangular_architecture_boundary", "observed": value}
        )
        return None
    return boundary


def _architecture_active_wall_ids(
    architecture,
    errors: list[dict],
) -> list[str]:
    physical = (
        architecture.get("physical_walls")
        if isinstance(architecture, dict)
        else None
    )
    raw = (
        physical.get("active_wall_ids")
        if isinstance(physical, dict)
        else None
    )
    if not isinstance(raw, list):
        errors.append(
            {
                "code": "invalid_active_wall_ids",
                "observed": raw,
            }
        )
        return []
    values = [str(value) for value in raw]
    if (
        len(values) != len(set(values))
        or any(value not in CANONICAL_WALL_IDS for value in values)
    ):
        errors.append(
            {
                "code": "invalid_active_wall_ids",
                "observed": values,
            }
        )
        return []
    floor_z = architecture.get("floor_z", 0.0)
    try:
        floor_z_number = float(floor_z)
    except (TypeError, ValueError):
        floor_z_number = float("nan")
    if not math.isfinite(floor_z_number) or abs(floor_z_number) > TOLERANCE:
        errors.append(
            {"code": "invalid_architecture_floor_z", "observed": floor_z}
        )
    return [
        wall_id for wall_id in CANONICAL_WALL_IDS if wall_id in values
    ]


def _expected_architecture_geometry(
    boundary: list[list[float]],
    scene_height: float,
    active_wall_ids: list[str],
) -> dict[str, dict]:
    result = {
        "floor": {
            "object_name": "benchmark_floor",
            "location": [0.0, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "vertices": [[x, y, 0.0] for x, y in boundary],
            "faces": [[[x, y, 0.0] for x, y in boundary]],
            "edge_count": 4,
            "loop_count": 4,
            "uv_layers": [],
            "normals": [
                [
                    0.0,
                    0.0,
                    (
                        1.0
                        if sum(
                            boundary[index][0]
                            * boundary[(index + 1) % len(boundary)][1]
                            - boundary[(index + 1) % len(boundary)][0]
                            * boundary[index][1]
                            for index in range(len(boundary))
                        )
                        >= 0.0
                        else -1.0
                    ),
                ]
            ],
        }
    }
    thickness = 0.08
    for index, start in enumerate(boundary):
        end = boundary[(index + 1) % len(boundary)]
        wall_id = _wall_id_for_boundary_edge(boundary, index)
        if wall_id not in active_wall_ids:
            continue
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        unit = [dx / length, dy / length]
        perpendicular = [-unit[1], unit[0]]
        points = {}
        for endpoint_name, endpoint in (("start", start), ("end", end)):
            for side_name, side in (("minus", -1.0), ("plus", 1.0)):
                for height_name, z_value in (
                    ("bottom", 0.0),
                    ("top", scene_height),
                ):
                    points[(endpoint_name, side_name, height_name)] = [
                        endpoint[0]
                        + side * perpendicular[0] * thickness * 0.5,
                        endpoint[1]
                        + side * perpendicular[1] * thickness * 0.5,
                        z_value,
                    ]
        faces = [
            [
                points[(endpoint, side, "bottom")]
                for endpoint in ("start", "end")
                for side in ("minus", "plus")
            ],
            [
                points[(endpoint, side, "top")]
                for endpoint in ("start", "end")
                for side in ("minus", "plus")
            ],
            [
                points[("start", side, height)]
                for side in ("minus", "plus")
                for height in ("bottom", "top")
            ],
            [
                points[("end", side, height)]
                for side in ("minus", "plus")
                for height in ("bottom", "top")
            ],
            [
                points[(endpoint, "minus", height)]
                for endpoint in ("start", "end")
                for height in ("bottom", "top")
            ],
            [
                points[(endpoint, "plus", height)]
                for endpoint in ("start", "end")
                for height in ("bottom", "top")
            ],
        ]
        result[wall_id] = {
            "object_name": f"benchmark_{wall_id}",
            "location": [
                (start[0] + end[0]) * 0.5,
                (start[1] + end[1]) * 0.5,
                scene_height * 0.5,
            ],
            "rotation": [0.0, 0.0, math.atan2(dy, dx)],
            "scale": [1.0, 1.0, 1.0],
            "vertices": list(points.values()),
            "faces": faces,
            "edge_count": 12,
            "loop_count": 24,
            "uv_layers": [
                {
                    "name": "UVMap",
                    "active_render": True,
                    "active_clone": False,
                    "active": True,
                    "uv": [
                        [0.375, 0.0],
                        [0.625, 0.0],
                        [0.625, 0.25],
                        [0.375, 0.25],
                        [0.375, 0.25],
                        [0.625, 0.25],
                        [0.625, 0.5],
                        [0.375, 0.5],
                        [0.375, 0.5],
                        [0.625, 0.5],
                        [0.625, 0.75],
                        [0.375, 0.75],
                        [0.375, 0.75],
                        [0.625, 0.75],
                        [0.625, 1.0],
                        [0.375, 1.0],
                        [0.125, 0.5],
                        [0.375, 0.5],
                        [0.375, 0.75],
                        [0.125, 0.75],
                        [0.625, 0.5],
                        [0.875, 0.5],
                        [0.875, 0.75],
                        [0.625, 0.75],
                    ],
                }
            ],
            "normals": [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
                [-unit[0], -unit[1], 0.0],
                [unit[0], unit[1], 0.0],
                [-perpendicular[0], -perpendicular[1], 0.0],
                [perpendicular[0], perpendicular[1], 0.0],
            ],
        }
    return result


def _wall_id_for_boundary_edge(
    boundary: list[list[float]],
    index: int,
) -> str:
    midpoints = [
        (
            (point[0] + boundary[(offset + 1) % len(boundary)][0]) * 0.5,
            (point[1] + boundary[(offset + 1) % len(boundary)][1]) * 0.5,
        )
        for offset, point in enumerate(boundary)
    ]
    mapping = {
        max(range(4), key=lambda item: midpoints[item][1]): "north_wall",
        min(range(4), key=lambda item: midpoints[item][1]): "south_wall",
        max(range(4), key=lambda item: midpoints[item][0]): "east_wall",
        min(range(4), key=lambda item: midpoints[item][0]): "west_wall",
    }
    return mapping[index]


def _architecture_object_mismatches(
    obj,
    *,
    architecture_id: str,
    expected_geometry: dict,
    expected_material_signature: str,
) -> list[dict]:
    mismatches: list[dict] = []

    def mismatch(field: str, expected, observed) -> None:
        mismatches.append(
            {"field": field, "expected": expected, "observed": observed}
        )

    if obj.type != "MESH":
        mismatch("type", "MESH", obj.type)
        return mismatches
    if obj.name != expected_geometry["object_name"]:
        mismatch("name", expected_geometry["object_name"], obj.name)
    if obj.get(ROLE_PROPERTY) != "architecture":
        mismatch("benchmark_role", "architecture", obj.get(ROLE_PROPERTY))
    if str(obj.get(ARCHITECTURE_ID_PROPERTY) or "") != architecture_id:
        mismatch(
            ARCHITECTURE_ID_PROPERTY,
            architecture_id,
            obj.get(ARCHITECTURE_ID_PROPERTY),
        )
    custom_keys = sorted(str(key) for key in obj.keys())
    expected_custom_keys = sorted(
        [ROLE_PROPERTY, ARCHITECTURE_ID_PROPERTY]
    )
    if custom_keys != expected_custom_keys:
        mismatch("custom_properties", expected_custom_keys, custom_keys)
    if obj.parent is not None:
        mismatch("parent", None, obj.parent.name)
    collection_names = sorted(collection.name for collection in obj.users_collection)
    if collection_names != [ARCHITECTURE_COLLECTION]:
        mismatch(
            "collections",
            [ARCHITECTURE_COLLECTION],
            collection_names,
        )
    if _technically_hidden(obj):
        mismatch("technical_visibility", "render_enabled", "hidden_or_disabled")
    if (
        obj.library is not None
        or obj.data is None
        or obj.data.library is not None
    ):
        mismatch("library", None, "linked")
    if obj.modifiers:
        mismatch("modifiers", [], [modifier.name for modifier in obj.modifiers])
    if obj.constraints:
        mismatch(
            "constraints", [], [constraint.name for constraint in obj.constraints]
        )
    if (
        obj.animation_data is not None
        or getattr(obj.data, "animation_data", None) is not None
        or getattr(obj.data, "shape_keys", None) is not None
    ):
        mismatch("animation_or_shape_keys", False, True)
    if (
        getattr(obj, "instance_type", "NONE") != "NONE"
        or getattr(obj, "instance_collection", None) is not None
    ):
        mismatch("instancing", False, True)
    for attribute, expected_value in (
        ("visible_camera", True),
        ("visible_diffuse", True),
        ("visible_glossy", True),
        ("visible_transmission", True),
        ("visible_volume_scatter", True),
        ("visible_shadow", True),
        ("is_holdout", False),
        ("is_shadow_catcher", False),
    ):
        if hasattr(obj, attribute) and getattr(obj, attribute) != expected_value:
            mismatch(attribute, expected_value, getattr(obj, attribute))

    for field, observed, expected in (
        ("location", list(obj.location), expected_geometry["location"]),
        ("rotation", list(obj.rotation_euler), expected_geometry["rotation"]),
        ("scale", list(obj.scale), expected_geometry["scale"]),
    ):
        if not _close_json(observed, expected):
            mismatch(field, expected, [float(value) for value in observed])

    world_vertices = [
        [float(value) for value in obj.matrix_world @ vertex.co]
        for vertex in obj.data.vertices
    ]
    if _point_multiset(world_vertices) != _point_multiset(
        expected_geometry["vertices"]
    ):
        mismatch(
            "world_vertices",
            _point_multiset(expected_geometry["vertices"]),
            _point_multiset(world_vertices),
        )
    observed_faces = [
        [
            [float(value) for value in obj.matrix_world @ obj.data.vertices[index].co]
            for index in polygon.vertices
        ]
        for polygon in obj.data.polygons
    ]
    if _face_multiset(observed_faces) != _face_multiset(
        expected_geometry["faces"]
    ):
        mismatch(
            "world_faces",
            _face_multiset(expected_geometry["faces"]),
            _face_multiset(observed_faces),
        )
    if len(obj.data.edges) != expected_geometry["edge_count"]:
        mismatch(
            "edge_count",
            expected_geometry["edge_count"],
            len(obj.data.edges),
        )
    if len(obj.data.loops) != expected_geometry["loop_count"]:
        mismatch(
            "loop_count",
            expected_geometry["loop_count"],
            len(obj.data.loops),
        )
    normal_matrix = obj.matrix_world.to_3x3().inverted().transposed()
    observed_normals = [
        [float(value) for value in (normal_matrix @ polygon.normal).normalized()]
        for polygon in obj.data.polygons
    ]
    if _point_multiset(observed_normals) != _point_multiset(
        expected_geometry["normals"]
    ):
        mismatch(
            "world_face_normals",
            _point_multiset(expected_geometry["normals"]),
            _point_multiset(observed_normals),
        )
    observed_uv_layers = [
        {
            "name": str(layer.name),
            "active_render": bool(layer.active_render),
            "active_clone": bool(layer.active_clone),
            "active": bool(layer.active),
            "uv": [
                [float(component) for component in item.uv]
                for item in layer.data
            ],
        }
        for layer in obj.data.uv_layers
    ]
    color_attributes = list(getattr(obj.data, "color_attributes", ()))
    public_attributes = sorted(
        {
            (
                str(attribute.name),
                str(attribute.domain),
                str(attribute.data_type),
            )
            for attribute in obj.data.attributes
            if not str(attribute.name).startswith(".")
            and str(attribute.name) not in {"position", "sharp_face"}
        }
    )
    expected_public_attributes = sorted(
        {
            (layer["name"], "CORNER", "FLOAT2")
            for layer in expected_geometry["uv_layers"]
        }
    )
    if (
        not _close_json(observed_uv_layers, expected_geometry["uv_layers"])
        or color_attributes
        or public_attributes != expected_public_attributes
    ):
        mismatch(
            "mesh_visual_attributes",
            {
                "uv_layers": expected_geometry["uv_layers"],
                "color_attributes": [],
                "public_attributes": expected_public_attributes,
            },
            {
                "uv_layers": observed_uv_layers,
                "color_attributes": sorted(
                    str(attribute.name) for attribute in color_attributes
                ),
                "public_attributes": public_attributes,
            },
        )
    if (
        any(polygon.use_smooth for polygon in obj.data.polygons)
        or any(
            edge.use_seam or getattr(edge, "use_edge_sharp", False)
            for edge in obj.data.edges
        )
        or bool(getattr(obj.data, "has_custom_normals", False))
    ):
        mismatch(
            "mesh_shading_state",
            {
                "smooth_polygons": [],
                "seam_edges": [],
                "sharp_edges": [],
                "custom_normals": False,
            },
            {
                "smooth_polygons": [
                    int(polygon.index)
                    for polygon in obj.data.polygons
                    if polygon.use_smooth
                ],
                "seam_edges": [
                    int(edge.index)
                    for edge in obj.data.edges
                    if edge.use_seam
                ],
                "sharp_edges": [
                    int(edge.index)
                    for edge in obj.data.edges
                    if getattr(edge, "use_edge_sharp", False)
                ],
                "custom_normals": bool(
                    getattr(obj.data, "has_custom_normals", False)
                ),
            },
        )

    materials = [material for material in obj.data.materials]
    if len(materials) != 1 or materials[0] is None:
        mismatch(
            "material_slots",
            [architecture_id],
            [material.name if material is not None else None for material in materials],
        )
    else:
        material = materials[0]
        if material.name != architecture_id:
            mismatch("material_name", architecture_id, material.name)
        observed_signature = _material_signature(material)
        if observed_signature != expected_material_signature:
            mismatch(
                "material_signature",
                expected_material_signature,
                observed_signature,
            )
        if any(polygon.material_index != 0 for polygon in obj.data.polygons):
            mismatch(
                "polygon_material_indices",
                [0] * len(obj.data.polygons),
                [
                    int(polygon.material_index)
                    for polygon in obj.data.polygons
                ],
            )
    return mismatches


def _expected_architecture_material_signature(architecture_id: str) -> str:
    color = (
        (0.34, 0.36, 0.39, 1.0)
        if architecture_id == "floor"
        else (0.60, 0.62, 0.66, 1.0)
    )
    material = bpy.data.materials.new(
        f"__benchmark_expected_{architecture_id}"
    )
    try:
        material.diffuse_color = color
        material.roughness = 0.65
        material.use_nodes = True
        principled = next(
            (
                node
                for node in material.node_tree.nodes
                if node.type == "BSDF_PRINCIPLED"
            ),
            None,
        )
        if principled is None:
            raise RuntimeError(
                "Blender default material has no Principled BSDF node"
            )
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = 0.65
        return _material_signature(material)
    finally:
        bpy.data.materials.remove(material)


def _point_multiset(points) -> list[tuple[int, int, int]]:
    return sorted(
        tuple(int(round(float(value) / TOLERANCE)) for value in point)
        for point in points
    )


def _face_multiset(faces) -> list[tuple[tuple[int, int, int], ...]]:
    return sorted(
        tuple(sorted(_point_multiset(face)))
        for face in faces
    )


def _mesh_attribute_payload(attribute) -> dict:
    values = []
    for item in attribute.data:
        value = None
        for field in (
            "value",
            "vector",
            "color",
            "color_srgb",
            "uv",
            "byte_color",
        ):
            if hasattr(item, field):
                value = _json_scalar_or_vector(getattr(item, field))
                break
        values.append(value)
    return {
        "name": str(attribute.name),
        "domain": str(attribute.domain),
        "data_type": str(attribute.data_type),
        "values": values,
    }


def _node_visual_properties(node) -> dict:
    values = _rna_fingerprint_properties(
        node,
        excluded={
            "name",
            "label",
            "location",
            "width",
            "height",
            "dimensions",
            "select",
            "show_options",
            "show_preview",
            "show_texture",
            "hide",
            "parent",
            "color",
            "use_custom_color",
            "inputs",
            "outputs",
            "internal_links",
            "node_tree",
            "image",
            "color_ramp",
            "mapping",
            "color_mapping",
            "texture_mapping",
            "image_user",
            "mute",
        },
    )
    color_ramp = getattr(node, "color_ramp", None)
    if color_ramp is not None:
        values["color_ramp"] = _color_ramp_fingerprint_payload(color_ramp)
    curve_mapping = getattr(node, "mapping", None)
    if curve_mapping is not None and hasattr(curve_mapping, "curves"):
        values["curve_mapping"] = _curve_mapping_fingerprint_payload(
            curve_mapping
        )
    color_mapping = getattr(node, "color_mapping", None)
    if color_mapping is not None:
        values["color_mapping"] = _rna_fingerprint_properties(color_mapping)
    texture_mapping = getattr(node, "texture_mapping", None)
    if texture_mapping is not None:
        values["texture_mapping"] = _rna_fingerprint_properties(
            texture_mapping
        )
    image_user = getattr(node, "image_user", None)
    if image_user is not None:
        values["image_user"] = _rna_fingerprint_properties(image_user)
    return values


def _color_ramp_fingerprint_payload(color_ramp) -> dict:
    return {
        "properties": _rna_fingerprint_properties(
            color_ramp,
            excluded={"elements"},
        ),
        "elements": [
            {
                "position": float(element.position),
                "color": [
                    float(component) for component in element.color
                ],
            }
            for element in color_ramp.elements
        ],
    }


def _curve_mapping_fingerprint_payload(mapping) -> dict:
    return {
        "properties": _rna_fingerprint_properties(
            mapping,
            excluded={"curves"},
        ),
        "curves": [
            {
                "index": int(curve_index),
                "properties": _rna_fingerprint_properties(
                    curve,
                    excluded={"points"},
                ),
                "points": [
                    {
                        "index": int(point_index),
                        "location": [
                            float(component)
                            for component in point.location
                        ],
                        "handle_type": str(
                            getattr(point, "handle_type", "")
                        ),
                        "properties": _rna_fingerprint_properties(
                            point,
                            excluded={
                                "location",
                                "handle_type",
                                "select",
                            },
                        ),
                    }
                    for point_index, point in enumerate(curve.points)
                ],
            }
            for curve_index, curve in enumerate(mapping.curves)
        ],
    }


def _rna_fingerprint_properties(
    value,
    excluded: set[str] | None = None,
) -> dict:
    """Serialize writable primitive RNA state without address-bearing reprs."""

    skipped = {
        "rna_type",
        "name",
        "users",
        "use_fake_user",
        "is_embedded_data",
        "is_evaluated",
        "is_runtime_data",
        "original",
        "asset_data",
        "library",
        "override_library",
        "preview",
        "tag",
    }
    skipped.update(excluded or ())
    rna = getattr(value, "bl_rna", None)
    result = {}
    for prop in getattr(rna, "properties", ()):
        name = str(getattr(prop, "identifier", ""))
        if not name or name in skipped:
            continue
        if bool(getattr(prop, "is_readonly", False)):
            continue
        prop_type = str(getattr(prop, "type", ""))
        if prop_type not in {"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"}:
            continue
        try:
            raw = getattr(value, name)
        except Exception:
            continue
        serialized = _fingerprint_scalar_or_vector(raw)
        if serialized is _UNSERIALIZABLE_FINGERPRINT_VALUE:
            continue
        result[name] = serialized
    return result


_UNSERIALIZABLE_FINGERPRINT_VALUE = object()


def _fingerprint_scalar_or_vector(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (set, frozenset)):
        converted = [
            _fingerprint_scalar_or_vector(item)
            for item in value
        ]
        if any(
            item is _UNSERIALIZABLE_FINGERPRINT_VALUE
            for item in converted
        ):
            return _UNSERIALIZABLE_FINGERPRINT_VALUE
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, dict):
        converted = {}
        for key in sorted(value, key=lambda item: str(item)):
            item = _fingerprint_scalar_or_vector(value[key])
            if item is _UNSERIALIZABLE_FINGERPRINT_VALUE:
                return _UNSERIALIZABLE_FINGERPRINT_VALUE
            converted[str(key)] = item
        return converted
    if isinstance(value, (list, tuple)):
        converted = [
            _fingerprint_scalar_or_vector(item)
            for item in value
        ]
        if any(
            item is _UNSERIALIZABLE_FINGERPRINT_VALUE
            for item in converted
        ):
            return _UNSERIALIZABLE_FINGERPRINT_VALUE
        return converted
    try:
        sequence = list(value)
    except (TypeError, ValueError):
        return _UNSERIALIZABLE_FINGERPRINT_VALUE
    converted = [
        _fingerprint_scalar_or_vector(item)
        for item in sequence
    ]
    if any(
        item is _UNSERIALIZABLE_FINGERPRINT_VALUE
        for item in converted
    ):
        return _UNSERIALIZABLE_FINGERPRINT_VALUE
    return converted


def _quantized_fingerprint_payload(value):
    """Remove placement round-off while retaining geometry/material changes."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        rounded = round(value, 6)
        return 0.0 if rounded == 0.0 else rounded
    if isinstance(value, list):
        return [_quantized_fingerprint_payload(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _quantized_fingerprint_payload(item)
            for key, item in value.items()
        }
    return str(value)


def _check_sanitized_scene_provenance(
    checks: list[dict],
    *,
    expected_data: dict | None,
    expected_path: Path | None,
) -> None:
    scene = bpy.context.scene
    mismatches = {}
    if expected_path is not None:
        expected_hash = _sha256_file(expected_path)
        observed_hash = scene.get("benchmark_materialization_plan_sha256")
        if str(observed_hash) != expected_hash:
            mismatches["materialization_plan_sha256"] = {
                "expected": expected_hash,
                "observed": observed_hash,
            }
    if isinstance(expected_data, dict):
        fields = {
            "benchmark_materialization_revision": expected_data.get(
                "materialization_revision"
            ),
            "benchmark_adapter_contract_revision": expected_data.get(
                "adapter_contract_revision"
            ),
            "benchmark_catalog_snapshot_id": expected_data.get(
                "catalog_snapshot_id"
            ),
        }
        for prop, expected in fields.items():
            if expected is not None and str(scene.get(prop)) != str(expected):
                mismatches[prop] = {
                    "expected": expected,
                    "observed": scene.get(prop),
                }
        request = expected_data.get("request")
        if isinstance(request, dict):
            architecture = request.get("architecture")
            raw_architecture = scene.get("benchmark_architecture_contract")
            try:
                observed_architecture = json.loads(str(raw_architecture))
            except (TypeError, json.JSONDecodeError):
                observed_architecture = None
            if architecture is not None and observed_architecture != architecture:
                mismatches["architecture"] = {
                    "expected": architecture,
                    "observed": observed_architecture,
                }
    _check(
        checks,
        "sanitized_scene_provenance",
        not mismatches,
        {"mismatches": mismatches},
        "sanitized_scene_provenance_mismatch",
    )


def _external_references() -> list[dict]:
    references = []
    for image in bpy.data.images:
        if (
            image.source in {"FILE", "MOVIE", "SEQUENCE", "TILED"}
            and not image.packed_files
        ):
            references.append(
                {
                    "type": "image",
                    "name": image.name,
                    "filepath": image.filepath,
                }
            )
    for collection_name in ("movieclips", "sounds", "fonts", "volumes"):
        collection = getattr(bpy.data, collection_name, ())
        for item in collection:
            filepath = str(getattr(item, "filepath", "") or "")
            packed = bool(
                getattr(item, "packed_file", None)
                or getattr(item, "packed_files", None)
            )
            if filepath and not packed:
                references.append(
                    {
                        "type": collection_name.rstrip("s"),
                        "name": item.name,
                        "filepath": filepath,
                    }
                )
    return sorted(
        references,
        key=lambda item: (item["type"], item["name"], item["filepath"]),
    )


def _appearance_state() -> dict:
    scene = bpy.context.scene
    compositor = getattr(scene, "node_tree", None)
    if compositor is None:
        compositor = getattr(scene, "compositing_node_group", None)
    compositor_enabled = bool(getattr(scene, "use_nodes", False))
    return {
        "camera_objects": sorted(
            obj.name for obj in bpy.data.objects if obj.type == "CAMERA"
        ),
        "light_objects": sorted(
            obj.name for obj in bpy.data.objects if obj.type == "LIGHT"
        ),
        "world_name": scene.world.name if scene.world is not None else None,
        "world_uses_nodes": bool(scene.world and scene.world.use_nodes),
        "compositor_uses_nodes": compositor_enabled,
        "compositor_node_count": _compositor_node_count(scene),
        "sequencer_strip_count": _sequencer_strip_count(scene),
        "action_count": len(bpy.data.actions),
    }


def _compositor_node_count(scene) -> int:
    compositor = getattr(scene, "node_tree", None)
    if compositor is None:
        compositor = getattr(scene, "compositing_node_group", None)
    if not bool(getattr(scene, "use_nodes", False)) or compositor is None:
        return 0
    return len(compositor.nodes)


def _sequencer_strip_count(scene) -> int:
    editor = scene.sequence_editor
    if editor is None:
        return 0
    values = getattr(editor, "sequences_all", None)
    if values is None:
        values = getattr(editor, "strips", ())
    return len(values)


def _object_render_state_mismatches(objects) -> list[dict]:
    expected = {
        "visible_camera": True,
        "visible_diffuse": True,
        "visible_glossy": True,
        "visible_transmission": True,
        "visible_volume_scatter": True,
        "visible_shadow": True,
        "is_holdout": False,
        "is_shadow_catcher": False,
        "show_instancer_for_render": True,
    }
    mismatches = []
    for obj in sorted(objects, key=lambda item: item.name):
        for attribute, wanted in expected.items():
            if not hasattr(obj, attribute):
                mismatches.append(
                    {
                        "object_name": obj.name,
                        "attribute": attribute,
                        "expected": wanted,
                        "observed": "unsupported",
                    }
                )
                continue
            observed = getattr(obj, attribute)
            if observed != wanted:
                mismatches.append(
                    {
                        "object_name": obj.name,
                        "attribute": attribute,
                        "expected": wanted,
                        "observed": observed,
                    }
                )
    return mismatches


_FACTORY_RENDER_STATE_SNAPSHOT = None


def _render_state_snapshot(scene) -> dict:
    settings = {
        "scene": _rna_fingerprint_properties(scene),
        "render": _rna_fingerprint_properties(scene.render),
        "render_image_settings": _rna_fingerprint_properties(
            scene.render.image_settings
        ),
        "view_settings": _rna_fingerprint_properties(scene.view_settings),
        "display_settings": _rna_fingerprint_properties(
            scene.display_settings
        ),
        "sequencer_colorspace_settings": _rna_fingerprint_properties(
            scene.sequencer_colorspace_settings
        ),
        "display_shading": _rna_fingerprint_properties(
            scene.display.shading
        ),
    }
    for field in ("cycles", "eevee"):
        value = getattr(scene, field, None)
        if value is not None:
            settings[field] = _rna_fingerprint_properties(value)

    view_layers = []
    for view_layer in scene.view_layers:
        payload = {
            "properties": _rna_fingerprint_properties(view_layer),
            "material_override": (
                view_layer.material_override.name
                if view_layer.material_override is not None
                else None
            ),
            "world_override": (
                view_layer.world_override.name
                if getattr(view_layer, "world_override", None) is not None
                else None
            ),
        }
        for field in ("cycles", "eevee"):
            value = getattr(view_layer, field, None)
            if value is not None:
                payload[field] = _rna_fingerprint_properties(value)
        view_layers.append(payload)

    compositor = getattr(scene, "node_tree", None)
    if compositor is None:
        compositor = getattr(scene, "compositing_node_group", None)
    return _quantized_fingerprint_payload(
        {
            "settings": settings,
            "view_layers": view_layers,
            "world": (
                _world_render_state_payload(scene.world)
                if scene.world is not None
                else None
            ),
            "compositor": {
                "use_nodes": bool(getattr(scene, "use_nodes", False)),
                "node_count": (
                    len(compositor.nodes)
                    if compositor is not None
                    else 0
                ),
            },
            "sequencer_strip_count": _sequencer_strip_count(scene),
        }
    )


def _factory_render_state_snapshot() -> dict:
    global _FACTORY_RENDER_STATE_SNAPSHOT
    if _FACTORY_RENDER_STATE_SNAPSHOT is not None:
        return _FACTORY_RENDER_STATE_SNAPSHOT

    marker = "BENCHMARK_FACTORY_RENDER_STATE="
    worker_dir = str(Path(__file__).resolve().parent)
    expression = (
        "import bpy,json,sys;"
        f"sys.path.insert(0,{worker_dir!r});"
        "import blend_inspector_worker as worker;"
        f"print({marker!r}+json.dumps("
        "worker._render_state_snapshot(bpy.context.scene),"
        "sort_keys=True,separators=(',',':'),allow_nan=False))"
    )
    completed = subprocess.run(
        [
            str(bpy.app.binary_path),
            "--background",
            "--factory-startup",
            "--disable-autoexec",
            "--python-exit-code",
            "1",
            "--python-expr",
            expression,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout)[-2000:]
        raise RuntimeError(
            "cannot obtain factory render-state snapshot: "
            f"Blender exited with {completed.returncode}: {detail}"
        )
    encoded = next(
        (
            line[len(marker) :]
            for line in completed.stdout.splitlines()
            if line.startswith(marker)
        ),
        None,
    )
    if encoded is None:
        raise RuntimeError(
            "factory render-state subprocess returned no snapshot"
        )
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise RuntimeError("factory render-state snapshot is malformed")
    _FACTORY_RENDER_STATE_SNAPSHOT = value
    return value


def _validate_sanitized_render_state() -> dict:
    """Reject source state that can survive into standardized rendering."""

    scene = bpy.context.scene
    mismatches: list[dict] = []
    if len(bpy.data.scenes) != 1:
        mismatches.append(
            {
                "field": "scene_count",
                "expected": 1,
                "observed": len(bpy.data.scenes),
            }
        )
    if len(scene.view_layers) != 1:
        mismatches.append(
            {
                "field": "view_layer_count",
                "expected": 1,
                "observed": len(scene.view_layers),
            }
        )
    if len(bpy.data.worlds) != 1 or scene.world is None:
        mismatches.append(
            {
                "field": "world_assignment",
                "expected": {
                    "world_count": 1,
                    "active_world": True,
                },
                "observed": {
                    "world_count": len(bpy.data.worlds),
                    "active_world": scene.world is not None,
                },
            }
        )

    compositor_state = []
    sequencer_state = []
    for candidate in bpy.data.scenes:
        compositor = getattr(candidate, "node_tree", None)
        if compositor is None:
            compositor = getattr(candidate, "compositing_node_group", None)
        node_count = len(compositor.nodes) if compositor is not None else 0
        if node_count:
            compositor_state.append(
                {
                    "scene": candidate.name,
                    "use_nodes": bool(getattr(candidate, "use_nodes", False)),
                    "node_count": node_count,
                }
            )
        strip_count = _sequencer_strip_count(candidate)
        if strip_count:
            sequencer_state.append(
                {
                    "scene": candidate.name,
                    "strip_count": strip_count,
                }
            )
    if compositor_state:
        mismatches.append(
            {
                "field": "compositor",
                "expected": [],
                "observed": compositor_state,
            }
        )
    if sequencer_state:
        mismatches.append(
            {
                "field": "sequencer_strips",
                "expected": [],
                "observed": sequencer_state,
            }
        )

    view_layer_overrides = []
    layer_collection_overrides = []
    for view_layer in scene.view_layers:
        override = getattr(view_layer, "material_override", None)
        world_override = getattr(view_layer, "world_override", None)
        if override is not None or world_override is not None:
            view_layer_overrides.append(
                {
                    "view_layer": view_layer.name,
                    "material_override": (
                        override.name if override is not None else None
                    ),
                    "world_override": (
                        world_override.name
                        if world_override is not None
                        else None
                    ),
                }
            )
        layer_collection_overrides.extend(
            _layer_collection_render_overrides(
                view_layer.layer_collection,
                view_layer_name=view_layer.name,
            )
        )
    if view_layer_overrides:
        mismatches.append(
            {
                "field": "view_layer_overrides",
                "expected": [],
                "observed": view_layer_overrides,
            }
        )
    if layer_collection_overrides:
        mismatches.append(
            {
                "field": "layer_collection_overrides",
                "expected": [],
                "observed": layer_collection_overrides,
            }
        )

    collection_overrides = [
        {
            "collection": collection.name,
            "hide_render": bool(collection.hide_render),
            "hide_viewport": bool(collection.hide_viewport),
        }
        for collection in bpy.data.collections
        if collection.hide_render or collection.hide_viewport
    ]
    if collection_overrides:
        mismatches.append(
            {
                "field": "collection_visibility_overrides",
                "expected": [],
                "observed": collection_overrides,
            }
        )

    try:
        _compare_render_state_payload(
            mismatches,
            "factory_startup_render_state",
            _render_state_snapshot(scene),
            _factory_render_state_snapshot(),
        )
    except Exception as exc:
        mismatches.append(
            {
                "field": "canonical_render_state_reference",
                "expected": "available",
                "observed": f"{type(exc).__name__}: {exc}",
            }
        )

    return {
        "passed": not mismatches,
        "mismatches": mismatches,
    }


def _layer_collection_render_overrides(
    layer_collection,
    *,
    view_layer_name: str,
    path: tuple[str, ...] = (),
) -> list[dict]:
    current_path = (*path, layer_collection.collection.name)
    values = {
        "exclude": bool(layer_collection.exclude),
        "hide_viewport": bool(layer_collection.hide_viewport),
        "holdout": bool(layer_collection.holdout),
        "indirect_only": bool(layer_collection.indirect_only),
    }
    result = []
    if any(values.values()):
        result.append(
            {
                "view_layer": view_layer_name,
                "collection_path": list(current_path),
                **values,
            }
        )
    for child in layer_collection.children:
        result.extend(
            _layer_collection_render_overrides(
                child,
                view_layer_name=view_layer_name,
                path=current_path,
            )
        )
    return result


def _world_render_state_payload(world) -> dict:
    node_tree = getattr(world, "node_tree", None)
    return {
        "color": [float(value) for value in world.color],
        "use_nodes": bool(world.use_nodes),
        "properties": _rna_fingerprint_properties(
            world,
            excluded={"color", "use_nodes", "node_tree"},
        ),
        "node_tree": (
            _node_tree_fingerprint_payload(
                node_tree,
                active=[],
                memo={},
            )
            if node_tree is not None
            else None
        ),
    }


def _compare_render_state_payload(
    mismatches: list[dict],
    field: str,
    observed,
    expected,
) -> None:
    observed_payload = _quantized_fingerprint_payload(observed)
    expected_payload = _quantized_fingerprint_payload(expected)
    if observed_payload == expected_payload:
        return
    mismatches.append(
        {
            "field": field,
            "expected_sha256": _json_payload_sha256(expected_payload),
            "observed_sha256": _json_payload_sha256(observed_payload),
        }
    )


def _json_payload_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _records(value: dict | None, label: str) -> dict[str, dict]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    raw = value.get("instances")
    if raw is None:
        raw = value.get("registry")
    if raw is None:
        return {}
    records: dict[str, dict] = {}
    if isinstance(raw, list):
        iterable = [(None, item) for item in raw]
    elif isinstance(raw, dict):
        iterable = list(raw.items())
    else:
        raise ValueError(f"{label} instances must be an array or object")
    for index, (mapping_key, item) in enumerate(iterable):
        if not isinstance(item, dict):
            raise ValueError(f"{label} instance {index} must be an object")
        record = dict(item)
        instance_id = _first_text(record.get("instance_id"), mapping_key)
        if not instance_id:
            raise ValueError(f"{label} instance {index} is missing instance_id")
        if instance_id in records:
            raise ValueError(f"{label} contains duplicate instance_id {instance_id!r}")
        record["instance_id"] = instance_id
        records[instance_id] = record
    return records


def _merge_expected_records(
    registry: dict[str, dict],
    catalog: dict[str, dict],
    *,
    expected_data: dict | None,
    catalog_data: dict | None,
) -> dict[str, dict]:
    authoritative_ids = set(registry) if registry else set(catalog)
    records = {}
    for instance_id in sorted(authoritative_ids):
        record = dict(catalog.get(instance_id, {}))
        record.update(registry.get(instance_id, {}))
        record["instance_id"] = instance_id
        record["_registry_record"] = dict(registry.get(instance_id, {}))
        transform = record.get("transform")
        if isinstance(transform, dict):
            for key in (
                "center_m",
                "target_size_m",
                "rotation_euler_xyz_deg",
                "uniform_scale",
            ):
                if record.get(key) is None and transform.get(key) is not None:
                    record[key] = transform[key]
        canonical_bbox = record.get("canonical_bbox")
        if isinstance(canonical_bbox, dict):
            if record.get("catalog_bbox_center_m") is None:
                record["catalog_bbox_center_m"] = canonical_bbox.get("center_m")
            if record.get("catalog_bbox_size_m") is None:
                record["catalog_bbox_size_m"] = canonical_bbox.get("size_m")
        local_bbox = record.get("local_bbox")
        if (
            isinstance(local_bbox, dict)
            and record.get("local_bbox_size_m") is None
        ):
            record["local_bbox_size_m"] = local_bbox.get("size_m")
        source_size = _vec3_or_none(record.get("catalog_bbox_size_m"))
        target_size = _vec3_or_none(record.get("target_size_m"))
        if record.get("uniform_scale") is None and source_size and target_size:
            record["uniform_scale"] = min(
                target_size[index] / source_size[index]
                for index in range(3)
            )
        scale = _number_or_none(record.get("uniform_scale"))
        if (
            record.get("local_bbox_size_m") is None
            and source_size
            and scale is not None
        ):
            record["local_bbox_size_m"] = [
                component * scale for component in source_size
            ]
        records[instance_id] = record
    # Retain top-level provenance for property comparison without polluting the
    # public instance record schema.
    source = catalog_data or expected_data or {}
    for record in records.values():
        record["_catalog_snapshot_id"] = source.get("catalog_snapshot_id")
        record["_materialization_revision"] = source.get(
            "materialization_revision"
        )
        record["_adapter_contract_revision"] = source.get(
            "adapter_contract_revision"
        )
    return records


def _is_instance_root(obj) -> bool:
    if obj.get(ROLE_PROPERTY) == "instance_root":
        return True
    instance_id = obj.get(INSTANCE_ID_PROPERTY)
    if instance_id is None:
        return False
    return not any(
        ancestor.get(INSTANCE_ID_PROPERTY) == instance_id
        for ancestor in _ancestors(obj)
    )


def _descendants(root) -> list:
    result = []
    stack = list(root.children)
    while stack:
        obj = stack.pop()
        result.append(obj)
        stack.extend(obj.children)
    return result


def _ancestors(obj) -> list:
    result = []
    current = obj.parent
    while current is not None:
        result.append(current)
        current = current.parent
    return result


def _technically_hidden(obj) -> bool:
    if obj.hide_render or obj.hide_viewport or obj.hide_get():
        return True
    try:
        if not obj.visible_get(view_layer=bpy.context.view_layer):
            return True
    except (RuntimeError, TypeError):
        return True
    return any(
        collection.hide_render or collection.hide_viewport
        for collection in obj.users_collection
    ) or _excluded_from_any_view_layer(obj)


def _is_render_capable_object(obj) -> bool:
    return bool(
        obj.type in RENDER_CAPABLE_OBJECT_TYPES
        or getattr(obj, "is_instancer", False)
        or getattr(obj, "instance_type", "NONE") != "NONE"
        or getattr(obj, "instance_collection", None) is not None
    )


def _excluded_from_any_view_layer(obj) -> bool:
    collection_names = {collection.name for collection in obj.users_collection}

    def excluded(layer_collection, inherited: bool = False) -> bool:
        current = bool(
            inherited
            or layer_collection.exclude
            or layer_collection.hide_viewport
        )
        if (
            layer_collection.collection.name in collection_names
            and current
        ):
            return True
        return any(
            excluded(child, current)
            for child in layer_collection.children
        )

    return any(
        excluded(view_layer.layer_collection)
        for scene in bpy.data.scenes
        for view_layer in scene.view_layers
    )


def _driver_owners() -> list[str]:
    owners = []
    collections = (
        ("object", bpy.data.objects),
        ("mesh", bpy.data.meshes),
        ("material", bpy.data.materials),
        ("scene", bpy.data.scenes),
        ("world", bpy.data.worlds),
        ("camera", bpy.data.cameras),
        ("light", bpy.data.lights),
        ("collection", bpy.data.collections),
        ("node_group", bpy.data.node_groups),
    )
    for kind, values in collections:
        for value in values:
            animation = getattr(value, "animation_data", None)
            drivers = getattr(animation, "drivers", None)
            if drivers:
                owners.append(f"{kind}:{value.name}")
            node_tree = getattr(value, "node_tree", None)
            node_animation = (
                getattr(node_tree, "animation_data", None)
                if node_tree is not None
                else None
            )
            if getattr(node_animation, "drivers", None):
                owners.append(f"{kind}_node_tree:{value.name}")
    return sorted(set(owners))


def _custom_properties(obj) -> dict:
    return {
        str(key): _json_scalar_or_vector(obj[key])
        for key in sorted(obj.keys())
        if key != "_RNA_UI"
    }


def _json_scalar_or_vector(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_scalar_or_vector(item) for item in value]
    try:
        return [_json_scalar_or_vector(item) for item in value]
    except TypeError:
        return str(value)


def _matrix_rows(matrix) -> list[list[float]]:
    return [
        [float(matrix[row][column]) for column in range(4)]
        for row in range(4)
    ]


def _vec3_or_none(value: object) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _number_or_none(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _expected_fingerprint(expected: dict, kind: str) -> str | None:
    candidates = [
        expected.get(f"{kind}_sha256"),
        expected.get(f"{kind}_fingerprint_sha256"),
        expected.get(f"{kind}_fingerprint"),
    ]
    fingerprints = expected.get("fingerprints")
    if isinstance(fingerprints, dict):
        candidates.extend(
            [
                fingerprints.get(f"{kind}_sha256"),
                fingerprints.get(kind),
            ]
        )
    for candidate in candidates:
        text = str(candidate or "").strip().lower()
        if len(text) == 64 and all(
            character in "0123456789abcdef" for character in text
        ):
            return text
    return None


def _close(left: float, right: float) -> bool:
    tolerance = max(
        TOLERANCE,
        TOLERANCE * max(abs(float(left)), abs(float(right))),
    )
    return abs(float(left) - float(right)) <= tolerance


def _close_json(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and _close(
            float(left),
            float(right),
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _close_json(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _close_json(left[key], right[key]) for key in left
        )
    return left == right


def _without_euler_provenance(value: object) -> object:
    if not isinstance(value, dict):
        return value
    copied = json.loads(json.dumps(value))
    obb = copied.get("obb")
    if isinstance(obb, dict):
        obb.pop("rotation_euler_xyz_deg", None)
    return copied


def _first_text(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _check(
    checks: list[dict],
    check_id: str,
    passed: bool,
    detail: dict,
    reason_code: str | None = None,
) -> None:
    item = {
        "id": check_id,
        "passed": bool(passed),
        "detail": detail,
    }
    if not passed and reason_code:
        item["reason_codes"] = [reason_code]
    checks.append(item)


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must contain an object: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
