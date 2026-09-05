#!/usr/bin/env python3
"""Fail-closed SceneWeaver FrozenAssets bridge.

The released SceneWeaver initializer cannot ingest arbitrary exact mesh IDs.
This bridge therefore requires a separately versioned upstream-side plugin that
initializes the frozen GLBs and removes add/remove/replace/resize tools while
leaving SceneWeaver's native reflection/evaluation loop intact.  No fallback to
the released asset-generating initializer is allowed.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from _common import (
    file_sha256,
    read_mapping,
    required_model_identity,
    required_model_deployment_id,
    require_observed_model_match,
    verify_catalog_contract,
    verify_api_endpoint_contract,
    verify_model_contract,
    write_json,
    write_runner_report,
    write_text,
)


LAYOUT_PATTERN = re.compile(r"layout_([0-9]+)\.json$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", required=True, type=Path)
    parser.add_argument("--plugin-entrypoint", required=True, type=Path)
    parser.add_argument("--expected-plugin-sha256", required=True)
    parser.add_argument("--plugin-python", default=sys.executable)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--method-input", required=True, type=Path)
    parser.add_argument("--comparison-input", required=True, type=Path)
    parser.add_argument("--comparison-catalog", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--asset-bindings-output", required=True, type=Path)
    parser.add_argument("--runner-report", required=True, type=Path)
    parser.add_argument("--size-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()

    repo = args.repo_path.expanduser().resolve()
    plugin = args.plugin_entrypoint.expanduser().resolve()
    if not (repo / "Pipeline/main.py").is_file():
        raise FileNotFoundError("configured checkout is not the released SceneWeaver repo")
    if not plugin.is_file():
        raise FileNotFoundError(f"SceneWeaver frozen initializer plugin is missing: {plugin}")
    plugin_hash = file_sha256(plugin)
    expected_plugin_hash = str(args.expected_plugin_sha256).strip().lower()
    if (
        len(expected_plugin_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_plugin_hash)
        or plugin_hash != expected_plugin_hash
    ):
        raise RuntimeError(
            "SceneWeaver frozen plugin does not match expected SHA-256"
        )
    request = read_mapping(args.request, "SceneWeaver request")
    if request.get("feedback_source") != "native_sceneweaver_only":
        raise RuntimeError("SceneWeaver request permits non-native evaluator feedback")
    control = read_mapping(args.comparison_input, "comparison control")
    catalog = read_mapping(args.comparison_catalog, "SceneWeaver catalog")
    identity = required_model_identity()
    deployment_id = required_model_deployment_id()
    endpoint_sha256 = verify_api_endpoint_contract(
        os.environ.get("LAYOUT_DDD_API_BASE_URL", ""),
        completion_endpoint=False,
    )
    verify_model_contract(control, identity)
    verify_catalog_contract(control, catalog)
    args.output_root.mkdir(parents=True, exist_ok=True)
    plugin_report = args.output_root / "plugin_report.json"
    command = [
        str(args.plugin_python),
        plugin.as_posix(),
        "--repo-path",
        repo.as_posix(),
        "--request",
        args.request.resolve().as_posix(),
        "--method-input",
        args.method_input.resolve().as_posix(),
        "--comparison-input",
        args.comparison_input.resolve().as_posix(),
        "--comparison-catalog",
        args.comparison_catalog.resolve().as_posix(),
        "--output-root",
        args.output_root.resolve().as_posix(),
        "--plugin-report",
        plugin_report.resolve().as_posix(),
    ]
    completed = subprocess.run(
        command,
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    write_text(args.output_root / "plugin.stdout.txt", completed.stdout)
    write_text(args.output_root / "plugin.stderr.txt", completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"SceneWeaver frozen initializer plugin exited with {completed.returncode}"
        )
    if file_sha256(plugin) != plugin_hash:
        raise RuntimeError("SceneWeaver frozen plugin changed during execution")
    plugin_data = read_mapping(plugin_report, "SceneWeaver plugin report")
    observed_identities = _verify_plugin_report(
        plugin_data,
        identity,
        deployment_id,
        endpoint_sha256,
        expected_object_plan_sha256=str(
            request.get("public_object_plan_sha256") or ""
        ),
    )
    layouts = _layouts(args.output_root)
    if not layouts:
        raise FileNotFoundError("SceneWeaver plugin emitted no layout_N.json states")
    observation, bindings = _observe_trajectory(
        layouts=layouts,
        control=control,
        catalog=catalog,
        request=request,
        plugin_report=plugin_data,
        tolerance=args.size_tolerance,
    )
    if not observation["valid"]:
        raise RuntimeError(
            "SceneWeaver native trajectory violates frozen controls: "
            f"{observation['violations']}"
        )
    write_json(args.asset_bindings_output, {"asset_bindings": bindings})
    usage = plugin_data.get("resource_usage")
    usage = usage if isinstance(usage, dict) else {}
    write_runner_report(
        args.runner_report,
        adapter="scene_weaver",
        identity=identity,
        generation_calls=usage.get("generation_calls"),
        tokens=usage.get("tokens"),
        iteration_count=len(layouts),
        rendering_calls=usage.get("rendering_calls"),
        tool_calls=usage.get("tool_calls"),
        protocol_observation=observation,
        observed_model_identities=observed_identities,
        model_identity_evidence="observed_response",
        model_deployment_id=deployment_id,
        model_endpoint_sha256=endpoint_sha256,
        extra={
            "plugin_entrypoint": plugin.as_posix(),
            "plugin_entrypoint_sha256": plugin_hash,
            "plugin_report": plugin_report.resolve().as_posix(),
            "benchmark_feedback_used_by_native_loop": False,
            "asset_bindings": args.asset_bindings_output.resolve().as_posix(),
        },
    )


def _verify_plugin_report(
    value: dict[str, Any],
    expected_identity: dict[str, str],
    deployment_id: str,
    endpoint_sha256: str,
    expected_object_plan_sha256: str,
) -> list[dict[str, str]]:
    if value.get("benchmark_feedback_used_by_native_loop") is not False:
        raise RuntimeError("SceneWeaver plugin did not attest native-only reflection")
    if value.get("model_input_asset_locators_used") is not False:
        raise RuntimeError(
            "SceneWeaver plugin exposed host-local asset locators to model input"
        )
    if (
        len(expected_object_plan_sha256) != 64
        or value.get("public_object_plan_sha256")
        != expected_object_plan_sha256
        or value.get("model_input_public_object_plan_sha256")
        != expected_object_plan_sha256
    ):
        raise RuntimeError(
            "SceneWeaver plugin did not prove use of the exact public object plan"
        )
    controls = value.get("frozen_controls")
    controls = controls if isinstance(controls, dict) else {}
    required = (
        "fixed_object_inventory",
        "exact_asset_ids",
        "fixed_native_scale",
        "frozen_iteration_bindings",
        "no_object_insertion_removal",
        "retrieval_disabled",
        "asset_replacement_disabled",
        "resize_disabled",
        "full_precision_local_bbox_observed",
        "full_precision_native_pose_observed",
        "native_object_dimensions_observed",
        "released_object_dimensions_export_preserved",
        "catalog_resolution_outside_model_context",
        "bottom_center_origin_rebased",
    )
    missing = [name for name in required if controls.get(name) is not True]
    if missing:
        raise RuntimeError(f"SceneWeaver frozen plugin lacks controls: {missing}")
    if controls.get("orientation_basis_policy") != (
        "bake_catalog_front_to_sceneweaver_positive_x"
    ):
        raise RuntimeError("SceneWeaver plugin lacks the frozen orientation-basis policy")
    usage = value.get("resource_usage")
    usage = usage if isinstance(usage, dict) else {}
    if usage.get("model_identity_evidence") != "observed_response":
        raise RuntimeError(
            "SceneWeaver plugin must report observed response model identity"
        )
    identities = usage.get("model_identities") or [usage.get("model_identity")]
    identities = [item for item in identities if isinstance(item, dict)]
    observed = require_observed_model_match(expected_identity, identities)
    if usage.get("model_deployment_id") != deployment_id:
        raise RuntimeError(
            "SceneWeaver plugin model deployment differs from comparison policy"
        )
    if usage.get("model_endpoint_sha256") != endpoint_sha256:
        raise RuntimeError(
            "SceneWeaver plugin model endpoint differs from comparison policy"
        )
    observations = value.get("iteration_asset_observations")
    if not isinstance(observations, list) or not observations:
        raise RuntimeError(
            "SceneWeaver plugin must report per-iteration observed asset identities"
        )
    return observed


def _layouts(root: Path) -> list[tuple[int, Path]]:
    candidates = []
    for path in root.rglob("layout_*.json"):
        match = LAYOUT_PATTERN.search(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    candidates.sort()
    numbers = [number for number, _ in candidates]
    if len(numbers) != len(set(numbers)):
        raise RuntimeError("SceneWeaver plugin emitted ambiguous duplicate iterations")
    return candidates


def _observe_trajectory(
    *,
    layouts: list[tuple[int, Path]],
    control: dict[str, Any],
    catalog: dict[str, Any],
    request: dict[str, Any],
    plugin_report: dict[str, Any],
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    slot_map = catalog.get("logical_to_native_slot")
    frozen = catalog.get("frozen_asset_bindings")
    if not isinstance(slot_map, dict) or not isinstance(frozen, dict):
        raise RuntimeError("SceneWeaver materialization lacks frozen bindings")
    expected_ids = {str(value) for value in slot_map.values()}
    expected_bindings = {
        str(native_id): dict(asset)
        for native_id, asset in frozen.items()
        if isinstance(asset, dict)
    }
    if set(expected_bindings) != expected_ids:
        raise RuntimeError(
            "SceneWeaver frozen binding inventory differs from the native slot map: "
            f"missing={sorted(expected_ids - set(expected_bindings))}, "
            f"unexpected={sorted(set(expected_bindings) - expected_ids)}"
        )
    observed_by_iteration = _observed_iteration_assets(
        plugin_report,
        expected_iterations={iteration for iteration, _path in layouts},
    )
    room = request.get("benchmark_room")
    room = room if isinstance(room, dict) else {}
    try:
        expected_room = [float(value) for value in room.get("roomsize", [])]
        expected_height = float(room.get("height"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "SceneWeaver request lacks a numeric benchmark room contract"
        ) from exc
    if (
        len(expected_room) != 2
        or any(not math.isfinite(value) or value <= 0.0 for value in expected_room)
        or not math.isfinite(expected_height)
        or expected_height <= 0.0
        or room.get("unit") != "meter"
    ):
        raise RuntimeError(
            "SceneWeaver request benchmark room must be positive metre dimensions"
        )
    native_room = plugin_report.get("native_room_observation")
    native_room = native_room if isinstance(native_room, dict) else {}
    geometry_tolerance = _controlled_geometry_tolerance(control)
    violations = []
    if not _vector_close(native_room.get("roomsize"), expected_room, tolerance):
        violations.append("native_solver_roomsize_mismatch")
    try:
        observed_height = float(native_room.get("height"))
    except (TypeError, ValueError):
        observed_height = float("nan")
    if (
        not math.isfinite(observed_height)
        or abs(observed_height - expected_height) > tolerance
    ):
        violations.append("native_solver_room_height_mismatch")
    if native_room.get("unit") != "meter":
        violations.append("native_solver_room_unit_mismatch")
    rows = []
    stable_bindings: dict[str, dict[str, Any]] | None = None
    precise_rotations: dict[str, dict[str, list[float]]] = {
        object_id: {} for object_id in expected_ids
    }
    precise_local_bboxes: dict[str, dict[str, list[float]]] = {
        object_id: {} for object_id in expected_ids
    }
    precise_positions: dict[str, dict[str, list[float]]] = {
        object_id: {} for object_id in expected_ids
    }
    precise_dimensions: dict[str, dict[str, list[float]]] = {
        object_id: {} for object_id in expected_ids
    }
    for iteration, path in layouts:
        value = read_mapping(path, f"SceneWeaver layout_{iteration}")
        objects = value.get("objects")
        objects = objects if isinstance(objects, dict) else {}
        actual_ids = set(str(key) for key in objects)
        current = []
        if actual_ids != expected_ids:
            current.append("object_inventory_mismatch")
        roomsize = value.get("roomsize")
        if not _vector_close(roomsize, expected_room, tolerance):
            current.append("roomsize_mismatch")
        observed_assets = observed_by_iteration[iteration]
        if set(observed_assets) != expected_ids:
            current.append("observed_asset_inventory_mismatch")
        iteration_bindings: dict[str, dict[str, Any]] = {}
        for object_id in sorted(actual_ids & expected_ids):
            asset = expected_bindings[object_id]
            expected_size = asset["physical_dimensions"]
            native_asset = objects[object_id].get("asset_id") or objects[object_id].get("jid")
            if native_asset is not None and str(native_asset) != str(asset["asset_key"]):
                current.append(f"asset_replaced:{object_id}")
            observed = observed_assets.get(object_id)
            if not isinstance(observed, dict):
                current.append(f"asset_observation_missing:{object_id}")
                continue
            observed_asset_id = str(observed.get("asset_id") or "")
            if observed_asset_id != str(asset["asset_key"]):
                current.append(f"observed_asset_replaced:{object_id}")
            mesh_path = Path(str(observed.get("mesh_path") or "")).expanduser()
            observed_hash = str(observed.get("mesh_sha256") or "")
            expected_hash = str(asset.get("mesh_sha256") or "")
            if not mesh_path.is_file():
                current.append(f"observed_mesh_missing:{object_id}")
            elif file_sha256(mesh_path) != observed_hash:
                current.append(f"observed_mesh_hash_invalid:{object_id}")
            if not expected_hash or observed_hash != expected_hash:
                current.append(f"frozen_mesh_mismatch:{object_id}")
            observed_local_bbox = _finite_vector3(
                observed.get("canonical_local_bbox_size")
            )
            if (
                observed_local_bbox is None
                or any(value <= 0.0 for value in observed_local_bbox)
            ):
                current.append(f"canonical_local_bbox_missing:{object_id}")
                observed_local_bbox = None
            else:
                precise_local_bboxes[object_id][str(iteration)] = observed_local_bbox
                if not _vector_close(
                    observed_local_bbox,
                    expected_size,
                    geometry_tolerance,
                ):
                    current.append(f"canonical_local_bbox_mismatch:{object_id}")
            expected_basis = _orientation_basis(asset)
            observed_basis = observed.get("orientation_basis")
            if not _orientation_basis_matches(
                observed_basis,
                expected_basis,
                tolerance=tolerance,
            ):
                current.append(f"orientation_basis_mismatch:{object_id}")
            expected_anchor = _anchor_basis(asset)
            if not _anchor_basis_matches(
                observed.get("anchor_basis"),
                expected_anchor,
                tolerance=tolerance,
            ):
                current.append(f"anchor_basis_mismatch:{object_id}")
            precise_position = _finite_vector3(
                observed.get("full_precision_native_bottom_center")
            )
            if precise_position is None:
                current.append(f"full_precision_position_missing:{object_id}")
            else:
                precise_positions[object_id][str(iteration)] = precise_position
                if not _vector_close(
                    objects[object_id].get("location"),
                    [round(value, 2) for value in precise_position], tolerance,
                ):
                    current.append(f"released_location_quantization_mismatch:{object_id}")
            observed_dimensions = _finite_vector3(
                observed.get("full_precision_native_object_dimensions")
            )
            if observed_dimensions is None or any(value <= 0 for value in observed_dimensions):
                current.append(f"native_object_dimensions_missing:{object_id}")
                observed_dimensions = None
            else:
                precise_dimensions[object_id][str(iteration)] = observed_dimensions
            precise_rotation = _finite_vector3(
                observed.get("full_precision_native_euler_xyz")
            )
            if precise_rotation is None:
                current.append(f"full_precision_rotation_missing:{object_id}")
            else:
                precise_rotations[object_id][str(iteration)] = precise_rotation
                if not _vector_close(
                    objects[object_id].get("rotation"),
                    [round(value, 2) for value in precise_rotation],
                    tolerance,
                ):
                    current.append(f"released_rotation_quantization_mismatch:{object_id}")
                if observed_local_bbox is not None and observed_dimensions is not None:
                    expected_dimensions = _released_object_dimensions(
                        observed_local_bbox,
                        expected_basis["basis_yaw_degrees"],
                    )
                    if not _vector_close(observed_dimensions, expected_dimensions, geometry_tolerance):
                        current.append(f"frozen_object_dimensions_mismatch:{object_id}")
                    if not _vector_close(
                        objects[object_id].get("size"),
                        [round(value, 2) for value in observed_dimensions],
                        tolerance,
                    ):
                        current.append(f"released_object_dimensions_mismatch:{object_id}")
            iteration_bindings[object_id] = {
                **asset,
                "asset_key": observed_asset_id,
                "mesh_uri": asset.get("mesh_uri"),
                "mesh_sha256": expected_hash,
                "observed_runtime_mesh_path": mesh_path.resolve().as_posix(),
                "observed_runtime_mesh_sha256": observed_hash,
                "orientation_basis": expected_basis,
                "anchor_basis": expected_anchor,
                "canonical_local_bbox_size": list(expected_size),
                "identity_source": "sceneweaver_plugin_observed_asset",
            }
        if stable_bindings is None:
            stable_bindings = iteration_bindings
        elif iteration_bindings != stable_bindings:
            current.append("iteration_asset_binding_drift")
        violations.extend(f"iteration_{iteration}:{item}" for item in current)
        rows.append(
            {
                "iteration": iteration,
                "layout": path.resolve().as_posix(),
                "layout_sha256": file_sha256(path),
                "valid": not current,
                "violations": current,
            }
        )
    bindings = stable_bindings or {}
    for object_id, binding in bindings.items():
        binding["full_precision_native_euler_xyz_by_iteration"] = dict(
            precise_rotations.get(object_id) or {}
        )
        binding["full_precision_native_local_bbox_size_by_iteration"] = dict(
            precise_local_bboxes.get(object_id) or {}
        )
        binding["full_precision_native_bottom_center_by_iteration"] = dict(
            precise_positions.get(object_id) or {}
        )
        binding["full_precision_native_object_dimensions_by_iteration"] = dict(
            precise_dimensions.get(object_id) or {}
        )
    return (
        {
            "valid": not violations,
            "violations": violations,
            "iterations": rows,
            "benchmark_feedback_used_by_native_loop": False,
            "retrieval_calls": 0,
            "native_size_semantics": "released_object_dimensions_rounded_2dp",
            "asset_geometry_tolerance_m": geometry_tolerance,
            "orientation_basis_policy": (
                "bake_catalog_front_to_sceneweaver_positive_x"
            ),
        },
        bindings,
    )


def _observed_iteration_assets(
    plugin_report: dict[str, Any],
    *,
    expected_iterations: set[int],
) -> dict[int, dict[str, dict[str, Any]]]:
    rows = plugin_report.get("iteration_asset_observations")
    rows = rows if isinstance(rows, list) else []
    result: dict[int, dict[str, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"SceneWeaver iteration_asset_observations[{index}] is invalid"
            )
        iteration = row.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int):
            raise RuntimeError("SceneWeaver asset observation iteration must be integer")
        if iteration in result:
            raise RuntimeError(
                f"SceneWeaver plugin reports duplicate asset iteration {iteration}"
            )
        objects = row.get("objects")
        if not isinstance(objects, dict):
            raise RuntimeError(
                f"SceneWeaver asset observation {iteration} lacks objects mapping"
            )
        result[iteration] = {
            str(object_id): dict(value)
            for object_id, value in objects.items()
            if isinstance(value, dict)
        }
    if set(result) != expected_iterations:
        raise RuntimeError(
            "SceneWeaver plugin asset-observation iterations differ from layouts: "
            f"missing={sorted(expected_iterations - set(result))}, "
            f"unexpected={sorted(set(result) - expected_iterations)}"
        )
    return result


def _vector_close(value: Any, expected: list[float], tolerance: float) -> bool:
    if not isinstance(value, list) or len(value) != len(expected):
        return False
    try:
        return all(
            abs(float(value[index]) - expected[index]) <= tolerance
            for index in range(len(expected))
        )
    except (TypeError, ValueError):
        return False


def _finite_vector3(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(item) for item in result) else None


def _controlled_geometry_tolerance(control: dict[str, Any]) -> float:
    generation = control.get("generation")
    generation = generation if isinstance(generation, dict) else {}
    try:
        value = float(generation.get("asset_geometry_tolerance_m"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "SceneWeaver frozen control lacks asset_geometry_tolerance_m"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(
            "SceneWeaver asset_geometry_tolerance_m must be finite and positive"
        )
    return value


def _orientation_basis(asset: dict[str, Any]) -> dict[str, Any]:
    front = asset.get("canonical_front")
    if front is None:
        return {
            "policy": "bake_catalog_front_to_sceneweaver_positive_x",
            "canonical_front": None,
            "native_zero_front": [1.0, 0.0, 0.0],
            "basis_yaw_degrees": 0.0,
            "applied": False,
            "status": "canonical_front_unavailable",
        }
    if not isinstance(front, list) or len(front) != 3:
        raise RuntimeError("SceneWeaver frozen canonical front must be a 3-vector")
    values = [float(value) for value in front]
    if abs(values[2]) > 1.0e-6 or math.hypot(values[0], values[1]) <= 1.0e-12:
        raise RuntimeError("SceneWeaver frozen canonical front must be horizontal")
    yaw = -math.degrees(math.atan2(values[1], values[0]))
    return {
        "policy": "bake_catalog_front_to_sceneweaver_positive_x",
        "canonical_front": values,
        "native_zero_front": [1.0, 0.0, 0.0],
        "basis_yaw_degrees": yaw,
        "applied": True,
        "status": "validated",
    }


def _orientation_basis_matches(
    observed: Any,
    expected: dict[str, Any],
    *,
    tolerance: float,
) -> bool:
    if not isinstance(observed, dict):
        return False
    exact_fields = (
        "policy",
        "canonical_front",
        "native_zero_front",
        "applied",
        "status",
    )
    if any(observed.get(field) != expected.get(field) for field in exact_fields):
        return False
    try:
        return abs(
            float(observed.get("basis_yaw_degrees"))
            - float(expected["basis_yaw_degrees"])
        ) <= tolerance
    except (TypeError, ValueError):
        return False


def _anchor_basis(asset: dict[str, Any]) -> dict[str, Any]:
    size = [float(value) for value in asset.get("physical_dimensions", [])]
    center = [float(value) for value in asset.get("bbox_center_local", [])]
    scale = [float(value) for value in asset.get("native_scale", [])]
    if len(size) != 3 or len(center) != 3 or len(scale) != 3:
        raise RuntimeError("SceneWeaver frozen asset lacks bbox center/scale contract")
    scaled_center = [center[index] * scale[index] for index in range(3)]
    bottom_center = [
        scaled_center[0],
        scaled_center[1],
        scaled_center[2] - size[2] / 2.0,
    ]
    return {
        "policy": "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin",
        "canonical_bbox_center_local": center,
        "native_scale": scale,
        "physical_dimensions": size,
        "canonical_bottom_center_local": bottom_center,
        "native_origin_semantics": "bbox_bottom_center",
        "applied": True,
    }


def _anchor_basis_matches(
    observed: Any,
    expected: dict[str, Any],
    *,
    tolerance: float,
) -> bool:
    if not isinstance(observed, dict):
        return False
    for field in ("policy", "native_origin_semantics", "applied"):
        if observed.get(field) != expected[field]:
            return False
    return all(
        _vector_close(observed.get(field), expected[field], tolerance)
        for field in (
            "canonical_bbox_center_local",
            "native_scale",
            "physical_dimensions",
            "canonical_bottom_center_local",
        )
    )


def _released_object_dimensions(
    local_size: list[float],
    basis_yaw_degrees: float,
) -> list[float]:
    # The native exporter uses obj.dimensions: scaled local-axis dimensions.
    # The input basis is baked, but runtime rotation_euler is not.
    rotation = _euler_xyz_matrix([0.0, 0.0, math.radians(basis_yaw_degrees)])
    return [
        sum(abs(rotation[row][column]) * local_size[column] for column in range(3))
        for row in range(3)
    ]


def _euler_xyz_matrix(angles: list[float]) -> list[list[float]]:
    x, y, z = angles
    cx, cy, cz = math.cos(x), math.cos(y), math.cos(z)
    sx, sy, sz = math.sin(x), math.sin(y), math.sin(z)
    return [
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ]


if __name__ == "__main__":
    main()
