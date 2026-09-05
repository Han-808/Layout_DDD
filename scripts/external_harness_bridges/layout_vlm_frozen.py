#!/usr/bin/env python3
"""Thin FrozenAssets bridge around LayoutVLM's released solver."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from _common import (
    file_sha256,
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


CONSTRAINT_PROGRAM_POLICY_VERSION = "layoutvlm_constraint_dsl_v1"
_CONSTRAINT_METHODS = {
    "against_wall",
    "align_with",
    "distance_constraint",
    "on_top_of",
    "point_towards",
}
_RESERVED_DSL_NAMES = {"AssetInstance", "radians", "solver", "walls"}


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
    parser.add_argument("--prepared-scene-config-output", required=True, type=Path)
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
    prepared_path = write_json(
        args.prepared_scene_config_output,
        prepared,
    )
    prepared_scene_config_sha256 = file_sha256(prepared_path)

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
    # The pinned upstream constructor still creates three clients from the
    # process environment. They are replaced below before any invocation.
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
            ChatOpenAI(
                model_name=identity["model_id"],
                max_tokens=2048,
                openai_api_key=api_key,
                openai_api_base=api_base,
            ),
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
        constraint_guard = _install_constraint_program_guard(
            solver,
            prepared,
            report_path=args.work_dir / "constraint_program_guard.json",
        )
        # Untrusted model-authored constraint code is executed by the released
        # solver. The AST guard below restricts it to the advertised pose DSL;
        # remove model credentials from its process environment as defense in
        # depth before reaching either upstream exec boundary.
        for environment_name in (
            "LAYOUT_DDD_API_KEY",
            "LAYOUT_DDD_API_BASE_URL",
            "OPENAI_API_KEY",
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
        ):
            os.environ.pop(environment_name, None)
        layout = solver.solve(prepared, MAX_ATTEMPTS=int(args.max_attempts))
    finally:
        os.chdir(old_cwd)
    if file_sha256(prepared_path) != prepared_scene_config_sha256:
        raise RuntimeError("LayoutVLM prepared scene config changed during generation")
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
            "constraint_program_guard": constraint_guard,
            "prepared_scene_config_sha256": prepared_scene_config_sha256,
        },
        observed_model_identities=observed_identities,
        model_identity_evidence="observed_response",
        model_deployment_id=deployment_id,
        model_endpoint_sha256=endpoint_sha256,
        extra={
            "prepared_scene_config": prepared_path.resolve().as_posix(),
            "prepared_scene_config_sha256": prepared_scene_config_sha256,
        },
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
        if (
            not re.fullmatch(r"[A-Za-z_]\w*", asset_var_name)
            or asset_var_name.startswith("__")
            or asset_var_name in _RESERVED_DSL_NAMES
        ):
            raise RuntimeError(
                f"LayoutVLM controlled native ID has unsafe variable name: {native_id!r}"
            )
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


def _install_constraint_program_guard(
    solver: Any,
    prepared: Mapping[str, Any],
    *,
    report_path: Path,
) -> dict[str, Any]:
    """Validate final model-authored code immediately before upstream exec."""

    original_solve_single_group = solver._solve_single_group
    tracking: dict[str, Any] = {
        "schema_version": "layoutvlm_constraint_program_guard_report_v1",
        "policy_version": CONSTRAINT_PROGRAM_POLICY_VERSION,
        "accepted_program_count": 0,
        "rejected_program_count": 0,
        "programs": [],
    }
    write_json(report_path, tracking)

    def guarded_solve_single_group(*args: Any, **kwargs: Any) -> Any:
        group_assets = kwargs.get("group_assets")
        if group_assets is None and len(args) > 3:
            group_assets = args[3]
        if group_assets is None:
            raise RuntimeError(
                "LayoutVLM guard could not resolve the current asset group"
            )
        sandbox = solver.sandbox
        original_sanity_check = sandbox.sanity_check

        def guarded_sanity_check(*sanity_args: Any, **sanity_kwargs: Any) -> Any:
            sanity_group = sanity_kwargs.get("group_assets")
            if sanity_group is None and sanity_args:
                sanity_group = sanity_args[0]
            program = sanity_kwargs.get("entire_program")
            if program is None and len(sanity_args) > 1:
                program = sanity_args[1]
            try:
                audit = _validate_constraint_program(
                    program,
                    group_assets=sanity_group,
                    prepared=prepared,
                )
            except (TypeError, ValueError, SyntaxError) as exc:
                tracking["rejected_program_count"] += 1
                tracking["programs"].append(
                    {
                        "status": "rejected",
                        "source_sha256": _program_sha256(program),
                        "reason": str(exc),
                    }
                )
                write_json(report_path, tracking)
                raise RuntimeError(
                    "LayoutVLM model constraint program was rejected by the "
                    f"frozen DSL policy: {exc}"
                ) from None
            tracking["accepted_program_count"] += 1
            tracking["programs"].append({"status": "accepted", **audit})
            write_json(report_path, tracking)
            return original_sanity_check(*sanity_args, **sanity_kwargs)

        sandbox.sanity_check = guarded_sanity_check
        try:
            return original_solve_single_group(*args, **kwargs)
        finally:
            sandbox.sanity_check = original_sanity_check

    solver._solve_single_group = guarded_solve_single_group
    return tracking


def _validate_constraint_program(
    program: Any,
    *,
    group_assets: Any,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(program, str) or not program.strip():
        raise ValueError("constraint program must be non-empty text")
    encoded = program.encode("utf-8")
    if len(encoded) > 1024 * 1024:
        raise ValueError("constraint program exceeds the 1 MiB policy limit")
    try:
        tree = ast.parse(program, mode="exec")
    except SyntaxError as exc:
        raise SyntaxError("constraint program is not valid Python") from exc
    if len(tree.body) > 5000:
        raise ValueError("constraint program exceeds the statement policy limit")

    assets = prepared.get("assets")
    if not isinstance(assets, Mapping):
        raise ValueError("prepared LayoutVLM task lacks an asset table")
    all_refs = _known_asset_refs(assets, list(assets))
    current_ids = _string_set(group_assets, "current asset group")
    current_refs = _known_asset_refs(assets, current_ids)
    wall_count = _wall_count(prepared)
    assignments: set[tuple[tuple[str, int], str]] = set()
    constrained: set[tuple[str, int]] = set()
    constraint_count = 0

    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1:
                raise ValueError("pose assignment must have exactly one target")
            asset_ref, field = _pose_target(statement.targets[0])
            if asset_ref not in current_refs:
                raise ValueError("pose assignment target is not in the current group")
            key = (asset_ref, field)
            if key in assignments:
                raise ValueError("duplicate pose-field assignment is forbidden")
            _validate_vector_literal(
                statement.value,
                allow_radians=field == "rotation",
            )
            assignments.add(key)
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            first_ref = _validate_constraint_call(
                statement.value,
                current_refs=current_refs,
                all_refs=all_refs,
                wall_count=wall_count,
            )
            constrained.add(first_ref)
            constraint_count += 1
            continue
        raise ValueError(
            f"statement type {type(statement).__name__} is outside the frozen DSL"
        )

    missing_fields = [
        f"{name}[{index}].{field}"
        for name, index in sorted(current_refs)
        for field in ("position", "rotation")
        if ((name, index), field) not in assignments
    ]
    if missing_fields:
        raise ValueError(
            "every current asset requires explicit position and rotation; "
            f"missing={missing_fields}"
        )
    unconstrained = sorted(current_refs - constrained)
    if unconstrained:
        raise ValueError(
            "every current asset requires at least one advertised constraint; "
            f"missing={unconstrained}"
        )
    return {
        "source_sha256": hashlib.sha256(encoded).hexdigest(),
        "current_asset_refs": [
            f"{name}[{index}]" for name, index in sorted(current_refs)
        ],
        "pose_assignment_count": len(assignments),
        "constraint_count": constraint_count,
    }


def _known_asset_refs(
    assets: Mapping[str, Any],
    native_ids: Any,
) -> set[tuple[str, int]]:
    refs: set[tuple[str, int]] = set()
    for native_id in _string_set(native_ids, "asset IDs"):
        item = assets.get(native_id)
        if not isinstance(item, Mapping):
            raise ValueError(f"unknown LayoutVLM asset ID {native_id!r}")
        variable = str(item.get("asset_var_name") or "")
        if (
            not re.fullmatch(r"[A-Za-z_]\w*", variable)
            or variable.startswith("__")
            or variable in _RESERVED_DSL_NAMES
        ):
            raise ValueError(f"asset {native_id!r} has an invalid Python variable")
        index = item.get("instance_idx")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise ValueError(f"asset {native_id!r} has an invalid instance index")
        ref = (variable, index)
        if ref in refs:
            raise ValueError(f"duplicate LayoutVLM asset reference {ref!r}")
        refs.add(ref)
    return refs


def _string_set(value: Any, label: str) -> set[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (set, list, tuple, dict)):
        raise ValueError(f"{label} must be a finite collection")
    items = value.keys() if isinstance(value, dict) else value
    result = {str(item) for item in items}
    if len(result) != len(value):
        raise ValueError(f"{label} contains duplicate identifiers")
    return result


def _pose_target(value: ast.expr) -> tuple[tuple[str, int], str]:
    if not isinstance(value, ast.Attribute) or value.attr not in {
        "position",
        "rotation",
    }:
        raise ValueError("assignment target must be an asset position or rotation")
    return _asset_ref(value.value), value.attr


def _asset_ref(value: ast.expr) -> tuple[str, int]:
    if not isinstance(value, ast.Subscript):
        raise ValueError("asset reference must use a literal placement index")
    owner = value.value
    if isinstance(owner, ast.Attribute):
        if owner.attr != "placements" or not isinstance(owner.value, ast.Name):
            raise ValueError("unsupported asset attribute chain")
        name = owner.value.id
    elif isinstance(owner, ast.Name):
        name = owner.id
    else:
        raise ValueError("unsupported asset reference")
    index = _integer_index(value.slice)
    return name, index


def _integer_index(value: ast.expr) -> int:
    if (
        not isinstance(value, ast.Constant)
        or not isinstance(value.value, int)
        or isinstance(value.value, bool)
        or value.value < 0
    ):
        raise ValueError("placement index must be a non-negative integer literal")
    return int(value.value)


def _validate_vector_literal(value: ast.expr, *, allow_radians: bool) -> None:
    if not isinstance(value, (ast.List, ast.Tuple)) or len(value.elts) != 3:
        raise ValueError("pose value must be a three-element list or tuple")
    for element in value.elts:
        if allow_radians and _is_radians_literal(element):
            continue
        _numeric_literal(element)


def _is_radians_literal(value: ast.expr) -> bool:
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "radians"
        and len(value.args) == 1
        and not value.keywords
    ):
        return False
    _numeric_literal(value.args[0])
    return True


def _numeric_literal(value: ast.expr) -> float:
    sign = 1.0
    literal = value
    if isinstance(value, ast.UnaryOp) and isinstance(value.op, (ast.UAdd, ast.USub)):
        sign = -1.0 if isinstance(value.op, ast.USub) else 1.0
        literal = value.operand
    if (
        not isinstance(literal, ast.Constant)
        or isinstance(literal.value, bool)
        or not isinstance(literal.value, (int, float))
    ):
        raise ValueError("constraint parameters must be finite numeric literals")
    try:
        number = sign * float(literal.value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("constraint numeric literal is not finite") from exc
    if not math.isfinite(number):
        raise ValueError("constraint numeric literal is not finite")
    return number


def _validate_constraint_call(
    call: ast.Call,
    *,
    current_refs: set[tuple[str, int]],
    all_refs: set[tuple[str, int]],
    wall_count: int,
) -> tuple[str, int]:
    if not (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "solver"
        and call.func.attr in _CONSTRAINT_METHODS
    ):
        raise ValueError("only advertised solver constraint calls are allowed")
    if len(call.args) < 2:
        raise ValueError("solver constraint requires two object arguments")
    if any(isinstance(argument, ast.Starred) for argument in call.args):
        raise ValueError("starred constraint arguments are forbidden")
    if any(keyword.arg is None for keyword in call.keywords):
        raise ValueError("expanded constraint keyword arguments are forbidden")
    first = _asset_ref(call.args[0])
    if first not in current_refs:
        raise ValueError("constraint first argument must be a current-group asset")
    method = call.func.attr
    second = call.args[1]
    if method == "against_wall":
        _require_constraint_shape(call, positional=2, keywords=set())
        _validate_wall_ref(second, wall_count)
    elif method == "on_top_of":
        _require_constraint_shape(call, positional=2, keywords=set())
        _validate_known_asset_ref(second, all_refs, label="on_top_of")
    elif method in {"align_with", "point_towards"}:
        _validate_angle_constraint_shape(call)
        _validate_known_asset_or_fixed_point(
            second,
            all_refs,
            allow_fixed_point=method == "point_towards",
            label=method,
        )
    else:
        _validate_distance_constraint_shape(call)
        _validate_known_asset_or_fixed_point(
            second,
            all_refs,
            allow_fixed_point=True,
            label="distance_constraint",
        )
    return first


def _require_constraint_shape(
    call: ast.Call,
    *,
    positional: int,
    keywords: set[str],
) -> None:
    if len(call.args) != positional:
        raise ValueError("solver constraint has unsupported positional arity")
    actual_keywords = {str(keyword.arg) for keyword in call.keywords}
    if actual_keywords != keywords or len(actual_keywords) != len(call.keywords):
        raise ValueError("solver constraint has unsupported keyword arguments")


def _validate_angle_constraint_shape(call: ast.Call) -> None:
    if len(call.args) not in {2, 3}:
        raise ValueError("solver orientation constraint has unsupported arity")
    keywords = {str(keyword.arg) for keyword in call.keywords}
    if not keywords.issubset({"angle"}) or len(keywords) != len(call.keywords):
        raise ValueError("solver orientation constraint accepts only angle")
    if len(call.args) == 3 and "angle" in keywords:
        raise ValueError("solver orientation angle cannot be supplied twice")
    if len(call.args) == 3:
        _numeric_literal(call.args[2])
    for keyword in call.keywords:
        _numeric_literal(keyword.value)


def _validate_distance_constraint_shape(call: ast.Call) -> None:
    if not 2 <= len(call.args) <= 5:
        raise ValueError("distance_constraint has unsupported positional arity")
    parameter_names = ("min_distance", "max_distance", "weight")
    supplied_positionally = set(parameter_names[: len(call.args) - 2])
    supplied_by_keyword: set[str] = set()
    for offset, argument in enumerate(call.args[2:]):
        name = parameter_names[offset]
        if name in {"min_distance", "max_distance"} and _is_none_literal(argument):
            continue
        _numeric_literal(argument)
    for keyword in call.keywords:
        name = str(keyword.arg)
        if name not in parameter_names:
            raise ValueError("distance_constraint has unsupported keyword arguments")
        if name in supplied_by_keyword or name in supplied_positionally:
            raise ValueError(f"distance_constraint supplied {name} more than once")
        supplied_by_keyword.add(name)
        if name in {"min_distance", "max_distance"} and _is_none_literal(
            keyword.value
        ):
            continue
        _numeric_literal(keyword.value)


def _validate_known_asset_ref(
    value: ast.expr,
    all_refs: set[tuple[str, int]],
    *,
    label: str,
) -> None:
    if _asset_ref(value) not in all_refs:
        raise ValueError(f"{label} target must be a known asset")


def _validate_known_asset_or_fixed_point(
    value: ast.expr,
    all_refs: set[tuple[str, int]],
    *,
    allow_fixed_point: bool,
    label: str,
) -> None:
    if allow_fixed_point and _is_fixed_asset_instance(value):
        return
    _validate_known_asset_ref(value, all_refs, label=label)


def _validate_wall_ref(value: ast.expr, wall_count: int) -> None:
    if not (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Name)
        and value.value.id == "walls"
    ):
        raise ValueError("constraint target must be a known asset or wall")
    index = _integer_index(value.slice)
    if index >= wall_count:
        raise ValueError("wall index is outside the native room boundary")


def _is_fixed_asset_instance(value: ast.expr) -> bool:
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "AssetInstance"
        and not value.args
        and len(value.keywords) == 1
        and value.keywords[0].arg == "position"
    ):
        return False
    _validate_vector_literal(value.keywords[0].value, allow_radians=False)
    return True


def _is_none_literal(value: ast.expr) -> bool:
    return isinstance(value, ast.Constant) and value.value is None


def _wall_count(prepared: Mapping[str, Any]) -> int:
    boundary = prepared.get("boundary")
    boundary = boundary if isinstance(boundary, Mapping) else {}
    vertices = boundary.get("floor_vertices")
    if not isinstance(vertices, list) or len(vertices) < 3:
        raise ValueError("prepared LayoutVLM task lacks a valid wall boundary")
    return len(vertices)


def _program_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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
