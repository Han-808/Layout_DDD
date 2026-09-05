"""Opt-in, source-pinned native SceneWeaver FrozenAssets mutation controls.

Generation-side controls authorized for the frozen experiment, NOT conversion
or benchmark scoring. No upstream files are written. Only enumerated function
bodies are patched in memory; their native callers/decorators keep their identity.
This component does not launch, route models, or certify the complete native loop.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import CodeType

from scene_weaver_frozen_assets import validate_binding
from scene_weaver_frozen import _orientation_basis


POLICY = "sceneweaver_frozen_native_mutations_v1"
UPSTREAM_COMMIT = "7ae54b2ec3fc66147704faa7daf7b017ba8b1bd9"
GUARD_GLOBAL = "_layout_ddd_frozen_guard"
SOURCE_HASHES = {
    "infinigen/core/constraints/example_solver/solve.py": "5f740bd0c29b7147c741e10b337c51ba119cad0726849e398e3bb6c194f3f95e",
    "infinigen/core/constraints/example_solver/populate.py": "4866ad91bb0c73938e50a3d142ff4ad68723bdaafa365bc86cd56f3e358d4eff",
    "infinigen/core/constraints/example_solver/moves/addition.py": "cc2f4cb8197044aacb1f16ecdc805f4f9397dcadf32d49aabea82086ad05e87c",
    "infinigen/core/constraints/example_solver/moves/deletion.py": "693b06069af23a8c965c3f860a09510ec5774085e8946326ae9a69bd11467896",
    "infinigen_examples/generate_indoors.py": "8957b2b26c4564acd66e1f3a9f64449e8f958530b31d1dd395c0459bd7dda4df",
    "infinigen_examples/steps/evaluate.py": "7b343755203c2dc6a23a31453461b90d19cf465058bcda9f959dfd3dc3c37c0f",
}


class FrozenMutationError(RuntimeError):
    """A native generation proposal violates the frozen input; never repair it."""


def _vector(value):
    if not hasattr(value, "__len__") or len(value) != 3:
        raise FrozenMutationError("expected a finite three-vector")
    try:
        result = [float(v) for v in value]
    except (ValueError, TypeError) as exc:
        raise FrozenMutationError("expected a finite three-vector") from exc
    if not all(math.isfinite(v) for v in result):
        raise FrozenMutationError("expected a finite three-vector")
    return result


class FrozenMutationGuard:
    def __init__(self, bindings: dict, journal: Path, *, tolerance: float = 1e-4):
        if not bindings:
            raise ValueError("frozen mutation controls require exact slot bindings")
        self._bindings = {slot: validate_binding(slot, value, tolerance=tolerance)
                          for slot, value in bindings.items()}
        self.tolerance = tolerance
        self.journal = Path(journal)
        # A separate journal for every native worker/attempt; never truncate one.
        with self.journal.open("x", encoding="utf-8"):
            pass
        self.emit("policy", policy=POLICY, slot_ids=sorted(bindings),
                  scale_policy="fixed_native_scale", benchmark_feedback_used=False)

    def emit(self, event, **details):
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, **details}, sort_keys=True,
                                    allow_nan=False) + "\n")

    def reject(self, reason):
        self.emit("rejected_mutation", reason=reason)
        raise FrozenMutationError(reason)

    def _object(self, obj):
        slot = obj.get("frozen_slot_id")
        asset = self._bindings.get(slot)
        if asset is None or obj.get("frozen_asset_id") != asset["asset_key"]:
            self.reject("unregistered_or_replaced_frozen_asset")
        if obj.get("frozen_mesh_sha256") != asset["mesh_sha256"]:
            self.reject("frozen_mesh_identity_mismatch")
        dims = _vector(obj.dimensions)
        if any(abs(v - 1) > 1e-6 for v in _vector(obj.scale)):
            self.reject("native_object_scale_changed")
        expected = list(asset["physical_dimensions"])
        yaw = _orientation_basis(asset)["basis_yaw_degrees"]
        if round(yaw / 90) % 2:
            expected[0], expected[1] = expected[1], expected[0]
        if any(abs(dims[i] - expected[i]) > self.tolerance for i in range(3)):
            self.reject("native_physical_dimensions_changed")
        return slot, dims, expected

    def keep_size(self, obj, requested, context):
        """Validate a dimension echo and leave scale/mesh bytes untouched.

        The released layout serializer rounds object dimensions to two decimals.
        An exact or that rounded echo is allowed, including zero for a thin axis;
        it is never a new target bbox and is never applied to the object.
        """
        slot, dims, expected = self._object(obj)
        requested = _vector(requested)
        exact = all(abs(requested[i] - expected[i]) <= self.tolerance for i in range(3))
        rounded = any(all(abs(requested[i] - round(base[i], 2)) <= 1e-6
                          for i in range(3)) for base in (dims, expected))
        if any(v < 0 for v in requested) or not (exact or rounded):
            self.reject(f"scale_change_requested:{slot}:{context}")
        self.emit("resize_suppressed", slot_id=slot, context=context,
                  requested_size=requested, observed_size=dims,
                  accepted_representation="exact_dimensions" if exact else "released_2dp_echo")
        return obj

    def keep_population(self, obj, placeholder, context):
        slot, dims, _ = self._object(obj)
        other, target, _ = self._object(placeholder)
        if slot != other or any(abs(a-b) > self.tolerance for a, b in zip(dims, target)):
            self.reject("population_placeholder_asset_mismatch")
        self.emit("population_resize_suppressed", slot_id=slot, context=context,
                  observed_size=dims, placeholder_size=target)

    def validate_layout(self, state, layouts):
        """Validate the complete proposal BEFORE any pose is applied."""
        if not isinstance(layouts, dict) or set(layouts) != set(self._bindings):
            self.reject("layout_object_inventory_mismatch")
        current = {key for key, value in state.objs.items()
                   if getattr(value, "generator", None) is not None}
        if current != set(self._bindings):
            self.reject("native_state_object_inventory_mismatch")
        for key, info in layouts.items():
            if not isinstance(info, dict):
                self.reject("malformed_native_layout")
            slot, _, _ = self._object(state.objs[key].obj)
            if slot != key:
                self.reject("native_state_slot_identity_mismatch")
            _vector(info.get("location"))
            _vector(info.get("rotation"))
            self.keep_size(state.objs[key].obj, info.get("size"), "layout_preflight")

    def skip_deletion(self, state, name, reason):
        if name not in state.objs:
            self.reject("deletion_target_missing")
        objstate = state.objs[name]
        if name in self._bindings:
            slot, _, _ = self._object(objstate.obj)
            if slot != name:
                self.reject("deletion_slot_identity_mismatch")
        elif getattr(objstate, "generator", None) is not None:
            self.reject("unregistered_native_object")
        # Do not remove room/structural parents either: deleting their Blender
        # children could indirectly remove a frozen asset. No metric is changed.
        self.emit("automatic_deletion_suppressed", slot_id=name, reason=reason,
                  unresolved_native_condition_retained=True)
        return True

    def filter_moves(self, schedules):
        blocked = {"addition", "deletion", "resample_asset"}
        result = {key: value for key, value in schedules.items() if key not in blocked}
        self.emit("native_move_controls", disabled=sorted(set(schedules) & blocked),
                  retained=list(result), retained_weights_unchanged=True)
        return result


def _statements(code):
    return ast.parse(code.replace("GUARD", GUARD_GLOBAL)).body


def _named(tree, path):
    node = tree
    for name in path.split("."):
        matches = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name]
        if len(matches) != 1:
            raise RuntimeError(f"native mutation patch target not unique: {path}")
        node = matches[0]
    return node


def _assigns(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)


def _replace_resize(function, start_name, replacement):
    hits = 0
    # Search statement lists, not arbitrary textual matches or commented code.
    for node in list(ast.walk(function)):
        for field, body in ast.iter_fields(node):
            if field not in {"body", "orelse"} or not isinstance(body, list):
                continue
            starts = [i for i, item in enumerate(body) if _assigns(item, start_name)]
            for start in reversed(starts):
                stops = [i for i in range(start, len(body))
                         if isinstance(body[i], ast.Expr) and isinstance(body[i].value, ast.Call)
                         and ast.unparse(body[i].value.func) == "bpy.ops.object.transform_apply"]
                if not stops:
                    raise RuntimeError("native resize patch end not found")
                body[start:stops[0]+1] = _statements(replacement)
                hits += 1
    if hits != 1:
        raise RuntimeError(f"native resize patch must match one block, got {hits}")


def _guard_delete(function, slot, reason):
    hits = 0
    for node in list(ast.walk(function)):
        for field, body in ast.iter_fields(node):
            if field not in {"body", "orelse"} or not isinstance(body, list):
                continue
            for index in range(len(body)-1, -1, -1):
                item = body[index]
                if (isinstance(item, ast.Expr) and isinstance(item.value, ast.Call)
                        and ast.unparse(item.value.func) == "delete_obj_with_children"):
                    body[index:index] = _statements(
                        f"if GUARD.skip_deletion(state, {slot}, {reason!r}):\n    continue")
                    hits += 1
    if hits != 1:
        raise RuntimeError(f"native deletion patch must match one block, got {hits}")


def _patch_function(path, original):
    node = deepcopy(original)
    if path == "resize_obj":
        node.body = _statements("return GUARD.keep_size(obj, size, 'addition.resize_obj')")
    elif path == "Solver.update_graph":
        _replace_resize(node, "size", "GUARD.keep_size(obj, info['size'], 'Solver.update_graph')")
        loops = [i for i, child in enumerate(node.body) if isinstance(child, ast.For)
                 and ast.unparse(child.target) == "(name, info)"]
        if len(loops) != 1:
            raise RuntimeError("native layout-update loop not unique")
        node.body[loops[0]:loops[0]] = _statements("GUARD.validate_layout(self.state, layouts)")
    elif path == "Solver.delete_object":
        node.body[:0] = _statements("if GUARD.skip_deletion(self.state, name, 'native_cleanup'):\n    return")
    elif path == "Solver._configure_move_weights":
        returns = [item for item in node.body if isinstance(item, ast.Return)]
        if len(returns) != 1 or ast.unparse(returns[0].value) != "schedules":
            raise RuntimeError("native move-schedule return changed")
        returns[0].value = ast.parse(f"{GUARD_GLOBAL}.filter_moves(schedules)", mode="eval").body
    elif path in {"populate_state_placeholders_mid", "update_asset_location"}:
        _replace_resize(node, "scale_x", f"GUARD.keep_population(obj, placeholder, {path!r})")
    elif path == "compose_indoors":
        _guard_delete(node, "name", "native_support_cleanup")
    elif path == "del_top_collide_obj":
        _guard_delete(node, "max_key", "native_collision_cleanup")
        # With no legal deletion the native cleanup loop cannot make progress.
        # Stop that deletion/re-optimize cycle, NOT the native reflection loop.
        # The original collision measurements and unresolved state are retained.
        node.body[:0] = _statements("frozen_removed_any = False")
        pops = 0
        for block in list(ast.walk(node)):
            body = getattr(block, "body", None)
            if not isinstance(body, list):
                continue
            for i in range(len(body)-1, -1, -1):
                if ast.unparse(body[i]) == "state.objs.pop(max_key)":
                    body[i+1:i+1] = _statements("frozen_removed_any = True")
                    pops += 1
        stops = [item for item in node.body if _assigns(item, "stop") and isinstance(item.value, ast.Constant) and item.value.value is False]
        if pops != 1 or len(stops) != 1:
            raise RuntimeError("native collision cleanup termination changed")
        stops[0].value = ast.parse("not frozen_removed_any", mode="eval").body
    else:
        # Explicit tools / unexpected discrete mutations must fail before writes.
        node.body[:0] = _statements(f"GUARD.reject('forbidden_native_operation:{path}')")
    node.decorator_list = []  # Existing callable/decorators remain installed.
    return ast.fix_missing_locations(node)


def _function_code(node, filename):
    unit = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    compiled = compile(unit, filename, "exec", dont_inherit=True)
    return next(c for c in compiled.co_consts if isinstance(c, CodeType) and c.co_name == node.name)


def _code_identity(code):
    # Ignore only source locations/qualname: compiling a selected class method at
    # module scope changes those. Detect earlier monkey patches before replacing
    # an existing callable, including modifications to nested function constants.
    constants = tuple(_code_identity(v) if isinstance(v, CodeType) else v for v in code.co_consts)
    return (code.co_code, constants, code.co_names, code.co_varnames, code.co_argcount,
            code.co_posonlyargcount, code.co_kwonlyargcount, code.co_cellvars, code.co_freevars)


PATCH_TARGETS = {
    "infinigen/core/constraints/example_solver/solve.py": (
        "Solver.update_graph", "Solver.update_graph_size", "Solver.remove_object",
        "Solver.delete_object", "Solver._configure_move_weights"),
    "infinigen/core/constraints/example_solver/populate.py": (
        "populate_state_placeholders_mid", "update_asset_location"),
    "infinigen/core/constraints/example_solver/moves/addition.py": (
        "resize_obj", "Addition.apply", "Addition.apply_random", "Resample.apply"),
    "infinigen/core/constraints/example_solver/moves/deletion.py": ("Deletion.apply",),
    "infinigen_examples/generate_indoors.py": ("compose_indoors",),
    "infinigen_examples/steps/evaluate.py": ("del_top_collide_obj",),
}


def build_patch_plan(repo: Path):
    """Verify all pinned source bytes, then compile only the enumerated targets."""
    repo = Path(repo).resolve()
    plan = []
    for relative, targets in PATCH_TARGETS.items():
        source = (repo / relative).read_bytes()
        if hashlib.sha256(source).hexdigest() != SOURCE_HASHES[relative]:
            raise RuntimeError(f"SceneWeaver source hash mismatch: {relative}")
        tree = ast.parse(source, filename=str(repo / relative))
        for target in targets:
            original = _named(tree, target)
            patched = _patch_function(target, original)
            code = _function_code(patched, str(repo / relative))
            native = deepcopy(original)
            native.decorator_list = []
            before, after = ast.unparse(original), ast.unparse(patched)
            plan.append({"relative_path": relative, "target": target, "code": code,
                         "native_code": _function_code(native, str(repo / relative)),
                         "audit": {"relative_path": relative, "target": target,
                                   "source_sha256": SOURCE_HASHES[relative],
                                   "native_ast_sha256": hashlib.sha256(before.encode()).hexdigest(),
                                   "patched_ast_sha256": hashlib.sha256(after.encode()).hexdigest(),
                                   "native_body": before, "patched_body": after}})
    return plan


def install_mutation_guards(repo: Path, guard: FrozenMutationGuard, modules: dict):
    """Call in each native worker before generation, with actual loaded modules.

    modules maps each PATCH_TARGETS relative path to its imported module. Function
    code replacement preserves aliases and gin wrappers. A changed checkout or
    unexpected wrapper fails before ANY function is changed. Full source/patch
    provenance is journaled before activation; native files stay byte-identical.
    """
    repo = Path(repo).resolve()
    plan = build_patch_plan(repo)
    assignments = []
    for item in plan:
        module = modules[item["relative_path"]]
        if Path(module.__file__).resolve() != repo / item["relative_path"]:
            raise RuntimeError("mutation guard module comes from another checkout")
        function = module
        for part in item["target"].split("."):
            function = getattr(function, part)
        function = inspect.unwrap(function)
        if (function.__code__.co_freevars or function.__globals__ is not vars(module)
                or Path(function.__code__.co_filename).resolve() != repo / item["relative_path"]
                or GUARD_GLOBAL in function.__globals__
                or _code_identity(function.__code__) != _code_identity(item["native_code"])):
            raise RuntimeError("unexpected or already patched native function")
        assignments.append((function, item["code"]))
    guard.emit("native_patch_plan", policy=POLICY, upstream_commit=UPSTREAM_COMMIT,
               patches=[item["audit"] for item in plan])
    for function, code in assignments:
        function.__globals__[GUARD_GLOBAL] = guard
        function.__code__ = code
    guard.emit("native_patch_installed", target_count=len(plan), native_source_files_modified=False)
    return [item["audit"] for item in plan]
