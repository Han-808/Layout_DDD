#!/usr/bin/env python3
"""Thin FrozenAssets bridge around LayoutVLM's released solver."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from _common import (
    observed_model_identity,
    public_object_plan,
    read_mapping,
    required_model_identity,
    required_model_deployment_id,
    require_observed_model_match,
    response_total_tokens,
    verify_catalog_contract,
    verify_api_endpoint_contract,
    verify_model_contract,
    write_json,
    write_runner_report,
)


class _CountingChat:
    def __init__(self, delegate: Any, identity: dict[str, str]) -> None:
        self.delegate = delegate
        self.identity = dict(identity)
        self.calls = 0
        self.successful_calls = 0
        self.observed_identities: list[dict[str, str]] = []
        self._tokens = 0
        self._token_reports = 0

    @property
    def tokens(self) -> int | None:
        return (
            self._tokens
            if self._token_reports == self.successful_calls == self.calls
            else None
        )

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        response = self.delegate.invoke(*args, **kwargs)
        self.successful_calls += 1
        observed = observed_model_identity(
            response,
            provider=self.identity["provider"],
        )
        if observed not in self.observed_identities:
            self.observed_identities.append(observed)
        tokens = response_total_tokens(response)
        if tokens is not None:
            self._tokens += tokens
            self._token_reports += 1
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--method-input", required=True, type=Path)
    parser.add_argument("--comparison-input", required=True, type=Path)
    parser.add_argument("--comparison-catalog", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runner-report", required=True, type=Path)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    repo = args.repo_path.expanduser().resolve()
    if not (repo / "src/layoutvlm/layoutvlm.py").is_file():
        raise FileNotFoundError("configured checkout is not the released LayoutVLM repo")
    request = read_mapping(args.request, "LayoutVLM scene config")
    method_input = read_mapping(args.method_input, "method input")
    control = read_mapping(args.comparison_input, "comparison control")
    catalog = read_mapping(args.comparison_catalog, "LayoutVLM catalog")
    identity = required_model_identity()
    deployment_id = required_model_deployment_id()
    verify_model_contract(control, identity)
    verify_catalog_contract(control, catalog)
    plan = public_object_plan(method_input)
    prepared = _prepare_task(request, plan, catalog, args.comparison_catalog)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = write_json(args.work_dir / "prepared_scene_config.json", prepared)

    api_key = os.environ.get("LAYOUT_DDD_API_KEY", "").strip()
    api_base = os.environ.get("LAYOUT_DDD_API_BASE_URL", "").strip()
    if not api_key or not api_base:
        raise RuntimeError(
            "LayoutVLM bridge requires LAYOUT_DDD_API_KEY and "
            "LAYOUT_DDD_API_BASE_URL"
        )
    endpoint_sha256 = verify_api_endpoint_contract(
        api_base,
        completion_endpoint=False,
    )
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = api_base
    os.environ["OPENAI_BASE_URL"] = api_base
    sys.path.insert(0, repo.as_posix())
    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        from langchain_openai import ChatOpenAI
        from src.layoutvlm.layoutvlm import LayoutVLM

        chat = _CountingChat(
            ChatOpenAI(model_name=identity["model_id"], max_tokens=2048),
            identity,
        )
        solver = LayoutVLM(
            mode="one_shot",
            save_dir=args.work_dir.resolve().as_posix(),
            asset_source="objaverse",
            gpt_4o_model_name=identity["model_id"],
        )
        precision_tracking = _install_exact_asset_size_literals(solver, prepared)
        # Released one-shot code primarily uses llm_slow; overriding all three
        # prevents a hidden gpt-4o-mini/grouping call if upstream routing changes.
        solver.llm_slow = chat
        solver.llm_slow_mini = chat
        solver.llm_slow_grouping = chat
        layout = solver.solve(prepared, MAX_ATTEMPTS=int(args.max_attempts))
    finally:
        os.chdir(old_cwd)
    if not isinstance(layout, dict):
        raise RuntimeError("LayoutVLM solver did not return its native layout mapping")
    solver_observation = _observe_solver_state(
        solver,
        prepared,
        tolerance=1.0e-6,
    )
    if not solver_observation["valid"]:
        raise RuntimeError(
            "LayoutVLM did not place the full frozen inventory without changing "
            f"asset size: {solver_observation['violations']}"
        )
    observed_identities = require_observed_model_match(
        identity,
        chat.observed_identities,
    )
    write_json(args.output, layout)
    expected_ids = sorted(prepared["assets"])
    actual_ids = sorted(str(key) for key in layout)
    violations = [] if actual_ids == expected_ids else ["object_inventory_mismatch"]
    write_runner_report(
        args.runner_report,
        adapter="layout_vlm",
        identity=identity,
        generation_calls=chat.calls,
        tokens=chat.tokens,
        protocol_observation={
            "valid": not violations,
            "violations": violations,
            "expected_object_ids": expected_ids,
            "actual_object_ids": actual_ids,
            "exact_assets_in_native_scene_config": True,
            "fixed_native_scale": True,
            "retrieval_calls": 0,
            "asset_bbox_axis_transform": "swap_xy",
            "native_zero_rotation_front": [1.0, 0.0, 0.0],
            "solver_state": solver_observation,
            "exact_asset_size_literal_shim": precision_tracking,
        },
        observed_model_identities=observed_identities,
        model_identity_evidence="observed_response",
        model_deployment_id=deployment_id,
        model_endpoint_sha256=endpoint_sha256,
        extra={"prepared_scene_config": prepared_path.resolve().as_posix()},
    )


def _prepare_task(
    request: dict[str, Any],
    plan: dict[str, Any],
    catalog: dict[str, Any],
    catalog_path: Path,
) -> dict[str, Any]:
    assets = catalog.get("frozen_assets")
    assets = assets if isinstance(assets, dict) else {}
    slot_map = catalog.get("logical_to_native_slot")
    slot_map = slot_map if isinstance(slot_map, dict) else {}
    public = {
        str(item.get("id")): item
        for item in plan.get("objects", [])
        if isinstance(item, dict)
    }
    prepared_assets = {}
    for logical_slot, native_id_value in slot_map.items():
        native_id = str(native_id_value)
        item = assets.get(native_id)
        if not isinstance(item, dict):
            raise RuntimeError(f"LayoutVLM materialization lacks slot {native_id!r}")
        if not native_id.endswith("-0"):
            raise RuntimeError(
                f"LayoutVLM controlled native ID must end in '-0': {native_id!r}"
            )
        asset_var_name = native_id[:-2]
        path_value = item.get("path")
        if not path_value:
            raise FileNotFoundError(f"LayoutVLM asset {item.get('uid')!r} has no mesh")
        mesh = Path(str(path_value))
        if not mesh.is_absolute():
            mesh = catalog_path.resolve().parent / mesh
        mesh = mesh.resolve()
        if mesh.suffix.lower() != ".glb" or not mesh.is_file():
            raise RuntimeError(
                f"LayoutVLM requires frozen GLB input for {item.get('uid')!r}: {mesh}"
            )
        public_item = public.get(str(logical_slot)) or {}
        metadata = public_item.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        support = str(metadata.get("support") or "floor")
        front = item.get("canonical_front")
        annotations = {
            "category": str(public_item.get("category") or item.get("category")),
            "description": str(
                public_item.get("description") or item.get("description")
            ),
            "onCeiling": support == "ceiling",
            "onFloor": support == "floor",
            "onWall": support == "wall",
            "onObject": support not in {"floor", "wall", "ceiling"},
            "frontView": front,
        }
        # LayoutVLM's released ``main.py::prepare_task_assets`` swaps the
        # source mesh X/Y bbox before constructing the solver asset. The
        # frozen bridge starts from an already prepared catalog record, so it
        # reproduces that native input transform explicitly.
        canonical_metadata = item["assetMetadata"]
        canonical_bbox = canonical_metadata.get("boundingBox")
        if not isinstance(canonical_bbox, Mapping):
            raise RuntimeError(
                f"LayoutVLM asset {item.get('uid')!r} lacks boundingBox metadata"
            )
        native_metadata = {
            **dict(canonical_metadata),
            "boundingBox": {
                "x": float(canonical_bbox["y"]),
                "y": float(canonical_bbox["x"]),
                "z": float(canonical_bbox["z"]),
            },
            "canonicalBoundingBoxBeforeLayoutVLMSwap": dict(canonical_bbox),
            "axisTransform": "swap_xy_for_layoutvlm_processed_positive_x_frame",
        }
        prepared_assets[native_id] = {
            "uid": item["uid"],
            "count": 1,
            "instance_var_name": asset_var_name,
            "asset_var_name": asset_var_name,
            "instance_idx": 0,
            "annotations": annotations,
            "category": annotations["category"],
            "description": annotations["description"],
            "path": mesh.as_posix(),
            "onCeiling": annotations["onCeiling"],
            "onFloor": annotations["onFloor"],
            "onWall": annotations["onWall"],
            "onObject": annotations["onObject"],
            "frontView": front,
            "assetMetadata": native_metadata,
        }
    return {
        "task_description": request["task_description"],
        "layout_criteria": _layout_criteria(request, plan),
        "boundary": request["boundary"],
        "assets": prepared_assets,
    }


def _layout_criteria(
    request: dict[str, Any], plan: dict[str, Any]
) -> str:
    public_constraints = {
        "objects": [
            {
                key: value
                for key, value in item.items()
                if key != "estimated_size"
            }
            for item in (plan.get("objects") or [])
            if isinstance(item, Mapping)
        ],
        "global_constraints": plan.get("global_constraints") or [],
        "zones": plan.get("zones") or [],
        "relations": plan.get("relations") or [],
        "object_placement_intents": {
            str(item.get("id")): item.get("placement_intent") or {}
            for item in plan.get("objects", [])
            if isinstance(item, dict) and item.get("id") is not None
        },
    }
    base = str(
        request.get("layout_criteria")
        or "Follow the public task description and object-plan relations."
    )
    return (
        base
        + "\nNATIVE FRAME RULE: LayoutVLM zero rotation faces native +X. "
        "For Imaginarium assets whose canonical functional front is local -Y, "
        "that local -Y side is native +X at zero; choose rotations in the "
        "LayoutVLM native frame. Exact size is supplied only by Assets.size."
        + "\nPUBLIC CONTROLLED OBJECT-PLAN CONSTRAINTS:\n"
        + json.dumps(public_constraints, ensure_ascii=False, separators=(",", ":"))
    )


def _observe_solver_state(
    solver: Any,
    prepared: Mapping[str, Any],
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Reject LayoutVLM's randomized fallback and model-mutated asset sizes."""

    sandbox = getattr(solver, "sandbox", None)
    if sandbox is None or not callable(getattr(sandbox, "export_layout", None)):
        raise RuntimeError("LayoutVLM solver did not expose its native sandbox state")
    placed = sandbox.export_layout(incomplete_scene=True, use_degree=True)
    placed = placed if isinstance(placed, Mapping) else {}
    assets = prepared.get("assets")
    assets = assets if isinstance(assets, Mapping) else {}
    expected_ids = {str(value) for value in assets}
    actual_ids = {str(value) for value in placed}
    violations: list[str] = []
    if actual_ids != expected_ids:
        violations.append("unplaced_or_unexpected_assets")
    size_observations: dict[str, Any] = {}
    local_vars = getattr(sandbox, "local_vars", {})
    local_vars = local_vars if isinstance(local_vars, Mapping) else {}
    for instance_id in sorted(expected_ids):
        item = assets[instance_id]
        item = item if isinstance(item, Mapping) else {}
        variable = str(item.get("asset_var_name") or "")
        parent = local_vars.get(variable)
        observed_size = getattr(parent, "size", None)
        metadata = item.get("assetMetadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        bbox = metadata.get("boundingBox")
        bbox = bbox if isinstance(bbox, Mapping) else {}
        expected_size = [bbox.get("x"), bbox.get("y"), bbox.get("z")]
        size_observations[instance_id] = {
            "asset_var_name": variable,
            "expected_native_bbox": expected_size,
            "observed_solver_size": list(observed_size) if isinstance(observed_size, (list, tuple)) else observed_size,
        }
        if not _vector_close(observed_size, expected_size, tolerance):
            violations.append(f"frozen_size_changed:{instance_id}")
    return {
        "valid": not violations,
        "violations": violations,
        "expected_object_ids": sorted(expected_ids),
        "placed_object_ids": sorted(actual_ids),
        "sizes": size_observations,
    }


def _install_exact_asset_size_literals(
    solver: Any,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    """Undo released ``{:.2f}`` bbox truncation without changing solver logic."""

    original = solver.get_task_program
    assets = prepared.get("assets")
    assets = assets if isinstance(assets, Mapping) else {}
    exact_by_variable: dict[str, list[float]] = {}
    variable_by_instance: dict[str, str] = {}
    for instance_id, item_value in assets.items():
        item = item_value if isinstance(item_value, Mapping) else {}
        variable = str(item.get("asset_var_name") or "")
        metadata = item.get("assetMetadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        bbox = metadata.get("boundingBox")
        bbox = bbox if isinstance(bbox, Mapping) else {}
        size = [float(bbox[axis]) for axis in ("x", "y", "z")]
        if not variable:
            raise RuntimeError(f"LayoutVLM asset {instance_id!r} lacks asset_var_name")
        previous = exact_by_variable.get(variable)
        if previous is not None and previous != size:
            raise RuntimeError(
                f"LayoutVLM variable {variable!r} has conflicting frozen sizes"
            )
        exact_by_variable[variable] = size
        variable_by_instance[str(instance_id)] = variable
    tracking: dict[str, Any] = {
        "policy": "replace_released_two_decimal_size_literals_with_exact_input",
        "calls": 0,
        "variables": sorted(exact_by_variable),
        "replacements": [],
    }

    def exact_task_program(*args: Any, **kwargs: Any) -> str:
        program = original(*args, **kwargs)
        if not isinstance(program, str):
            raise RuntimeError("LayoutVLM get_task_program returned non-text output")
        call_replacements: dict[str, int] = {}
        grouped_assets = args[0] if args else kwargs.get("grouped_assets")
        grouped_assets = (
            grouped_assets
            if isinstance(grouped_assets, (list, tuple, set))
            else []
        )
        required_variables = {
            variable_by_instance[str(instance_id)]
            for instance_id in grouped_assets
            if str(instance_id) in variable_by_instance
        }
        for variable, size in exact_by_variable.items():
            pattern = re.compile(
                rf"({re.escape(variable)}\s*=\s*Assets\([^\n]*?size=)\[[^\]]+\]"
            )
            replacement = rf"\g<1>{json.dumps(size, separators=(',', ':'))}"
            program, count = pattern.subn(replacement, program)
            if count > 1 or (variable in required_variables and count != 1):
                raise RuntimeError(
                    "LayoutVLM exact-size shim found an invalid number of "
                    f"definitions for {variable!r}; count={count}"
                )
            if count:
                call_replacements[variable] = count
        missing_grouped = sorted(required_variables - set(call_replacements))
        if missing_grouped:
            raise RuntimeError(
                "LayoutVLM exact-size shim omitted grouped asset variables: "
                f"{missing_grouped}"
            )
        tracking["calls"] += 1
        tracking["replacements"].append(call_replacements)
        return program

    solver.get_task_program = exact_task_program
    return tracking


def _vector_close(left: Any, right: Any, tolerance: float) -> bool:
    if not (
        isinstance(left, (list, tuple))
        and isinstance(right, (list, tuple))
        and len(left) == len(right) == 3
    ):
        return False
    try:
        return all(
            abs(float(left[index]) - float(right[index])) <= tolerance
            for index in range(3)
        )
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
