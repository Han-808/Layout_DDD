"""Mutation boundary CI: no bpy, real upstream checkout, dataset, or API calls."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import FunctionType, ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def controls():
    path = Path(__file__).resolve().parents[1] / "scripts/external_harness_bridges/scene_weaver_frozen_mutations.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("frozen_mutation_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.pop(0)


class Object(dict):
    def __init__(self, slot, asset, dims):
        super().__init__(frozen_slot_id=slot, frozen_asset_id=asset["asset_key"],
                         frozen_mesh_sha256=asset["mesh_sha256"])
        self.dimensions = list(dims)
        self.scale = [1, 1, 1]


@pytest.fixture
def fixture(controls, tmp_path):
    mesh = tmp_path / "exact.glb"
    mesh.write_bytes(b"tiny identity fixture, not actual GLB geometry")
    binding = {"asset_key": "db.exact_rug", "mesh_uri": str(mesh),
               "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
               "bbox_size_local": [2.136, 1.738, 0.000076], "bbox_center_local": [0, 0, 0.000038],
               "physical_dimensions": [2.136, 1.738, 0.000076], "native_scale": [1, 1, 1],
               "canonical_front": [0, -1, 0]}
    guard = controls.FrozenMutationGuard({"rug": binding}, tmp_path / "journal.jsonl")
    # B_front bakes -Y into +X, swapping the local X/Y dimensions.
    obj = Object("rug", binding, [1.738, 2.136, 0.000076])
    state = SimpleNamespace(objs={"rug": SimpleNamespace(obj=obj, generator=object()),
                                 "newroom_0-0": SimpleNamespace(generator=None)})
    return SimpleNamespace(binding=binding, guard=guard, obj=obj, state=state, mesh=mesh)


@pytest.mark.parametrize("rounded", [False, True])
def test_native_size_echo_does_not_clamp_thin_mesh_or_change_scale(fixture, rounded):
    f = fixture
    before = deepcopy(vars(f.obj)), deepcopy(dict(f.obj)), f.mesh.read_bytes()
    requested = [round(v, 2) for v in f.obj.dimensions] if rounded else f.obj.dimensions[:]
    assert f.guard.keep_size(f.obj, requested, "position_update") is f.obj
    assert before == (vars(f.obj), dict(f.obj), f.mesh.read_bytes())
    event = json.loads(f.guard.journal.read_text().splitlines()[-1])
    assert event["accepted_representation"] == ("released_2dp_echo" if rounded else "exact_dimensions")
    assert f.obj.dimensions[2] < 0.01


@pytest.mark.parametrize("requested", [[1.74, 2.14, 0.01], [1, 2, 3], [-1, 0, 0], [1, 2],
                                       [float("nan"), 2, 0], None])
def test_resize_requests_rejected_without_mutation(controls, fixture, requested):
    before = deepcopy(vars(fixture.obj))
    with pytest.raises(controls.FrozenMutationError):
        fixture.guard.keep_size(fixture.obj, requested, "resize")
    assert vars(fixture.obj) == before


@pytest.mark.parametrize("violation", ["asset_id", "mesh_hash", "object_scale", "baked_size", "slot"])
def test_existing_asset_geometry_violation_is_not_normalized(controls, fixture, violation):
    obj = fixture.obj
    if violation == "asset_id":
        obj["frozen_asset_id"] = "replacement"
    elif violation == "mesh_hash":
        obj["frozen_mesh_sha256"] = "0" * 64
    elif violation == "slot":
        obj["frozen_slot_id"] = "missing"
    elif violation == "object_scale":
        obj.scale[0] = 2
    else:
        obj.dimensions[1] *= 2
    before = deepcopy(vars(obj)), deepcopy(dict(obj))
    with pytest.raises(controls.FrozenMutationError):
        fixture.guard.keep_size(obj, obj.dimensions, "update")
    assert before == (vars(obj), dict(obj))


def test_population_only_moves_pose_not_scale(fixture):
    copy = deepcopy(fixture.obj)
    fixture.guard.keep_population(copy, fixture.obj, "populate")
    assert copy.dimensions == fixture.obj.dimensions
    assert copy.scale == [1, 1, 1]


def test_initializer_uses_only_supplied_exact_factory_mapping(controls, fixture):
    f = fixture
    f.guard.configure_initialization({"rug": "fixed.RugFactory"})
    solver = SimpleNamespace()
    f.guard.fixed_retrieval(solver, {"rug": "1"}, {"rug": "fixed.RugFactory"})
    assert solver.LoadObjavCnts == solver.LoadObjavFiles == {}
    with pytest.raises(controls.FrozenMutationError):
        f.guard.fixed_retrieval(solver, {"rug": "1"}, {"rug": None})


def test_initial_failed_candidate_rollback_is_not_committed_object_deletion(controls, fixture):
    f = fixture
    candidate = f.state.objs.pop("rug")
    factory = SimpleNamespace(frozen_slot_id="rug")
    assert f.guard.begin_initialization_slot(factory, f.state) == "rug"
    f.state.objs["rug"] = candidate
    f.guard.finish_initialization_attempt("rug", False)
    assert f.guard.skip_deletion(f.state, "rug", "failed_initial_attempt") is False
    f.state.objs.pop("rug")  # Native rollback removes its unaccepted candidate.
    f.guard.begin_initialization_slot(factory, f.state)
    f.state.objs["rug"] = candidate
    f.guard.finish_initialization_attempt("rug", True)
    f.guard.assert_complete_initialization(f.state)
    assert f.guard.skip_deletion(f.state, "rug", "later_physics_cleanup") is True
    with pytest.raises(controls.FrozenMutationError):
        f.guard.begin_initialization_slot(factory, f.state)


def test_incomplete_initialization_cannot_be_exported(controls, fixture):
    fixture.state.objs.pop("rug")
    with pytest.raises(controls.FrozenMutationError, match="incomplete_frozen_initialization"):
        fixture.guard.assert_complete_initialization(fixture.state)


def test_population_cannot_exchange_slots(controls, fixture, tmp_path):
    guard = controls.FrozenMutationGuard({"rug": fixture.binding, "other": fixture.binding},
                                          tmp_path / "two_slots.jsonl")
    other = Object("other", fixture.binding, fixture.obj.dimensions)
    with pytest.raises(controls.FrozenMutationError, match="placeholder_asset_mismatch"):
        guard.keep_population(fixture.obj, other, "populate")


def _layout(f):
    return {"rug": {"location": [1, 2, 0], "rotation": [0.1, 0.2, 0.3],
                    "size": [round(v, 2) for v in f.obj.dimensions]}}


@pytest.mark.parametrize("violation", ["missing", "inserted", "replaced_slot", "bad_pose", "architecture"])
def test_whole_layout_preflight_rejects_before_pose_updates(controls, fixture, violation):
    layout = _layout(fixture)
    if violation == "missing":
        layout.clear()
    elif violation == "inserted":
        layout["new"] = deepcopy(layout["rug"])
    elif violation == "replaced_slot":
        layout["other"] = layout.pop("rug")
    elif violation == "architecture":
        layout["newroom_0-0"] = deepcopy(layout["rug"])
    else:
        layout["rug"]["rotation"][0] = float("inf")
    with pytest.raises(controls.FrozenMutationError):
        fixture.guard.validate_layout(fixture.state, layout)


def test_layout_preflight_and_cleanup_preserve_original_objects(fixture):
    fixture.guard.validate_layout(fixture.state, _layout(fixture))
    before = dict(fixture.state.objs)
    assert fixture.guard.skip_deletion(fixture.state, "rug", "no_support") is True
    assert fixture.guard.skip_deletion(fixture.state, "newroom_0-0", "parent_cleanup") is True
    assert fixture.state.objs == before
    last = json.loads(fixture.guard.journal.read_text().splitlines()[-1])
    assert last["unresolved_native_condition_retained"] is True


def test_only_discrete_inventory_moves_are_removed(fixture):
    schedules = {key: object() for key in ["translate", "rotate", "plane_change", "reinit_pose",
                                          "addition", "deletion", "resample_asset"]}
    result = fixture.guard.filter_moves(schedules)
    assert list(result) == ["translate", "rotate", "plane_change", "reinit_pose"]
    assert all(result[key] is schedules[key] for key in result)
    assert len(schedules) == 7


def test_journal_never_overwrites_an_earlier_attempt(controls, fixture):
    before = fixture.guard.journal.read_bytes()
    with pytest.raises(FileExistsError):
        controls.FrozenMutationGuard({"rug": fixture.binding}, fixture.guard.journal)
    assert fixture.guard.journal.read_bytes() == before


def _compile(controls, source, path, globals):
    node = controls._patch_function(path, ast.parse(source).body[0])
    return FunctionType(controls._function_code(node, "synthetic_native.py"), globals)


def test_update_patch_keeps_native_pose_and_relation_calls_but_not_resize(controls, fixture):
    # Synthetic shape fixture, not vendored native code or a fake generation run.
    source = '''def update_graph(self):
    layouts = self.proposal
    for name, info in layouts.items():
        obj = self.state.objs[name].obj
        self.pose_calls.append((name, info["location"], info["rotation"]))
        size = info["size"]
        self.resize_calls.append(size)
        bpy.ops.object.transform_apply(scale=True)
        self.relation_calls.append(name)
'''
    native = SimpleNamespace(state=fixture.state, proposal=_layout(fixture),
                             pose_calls=[], resize_calls=[], relation_calls=[])
    function = _compile(controls, source, "Solver.update_graph", {controls.GUARD_GLOBAL: fixture.guard})
    before = fixture.mesh.read_bytes()
    function(native)
    assert native.pose_calls == [("rug", [1, 2, 0], [0.1, 0.2, 0.3])]
    assert native.relation_calls == ["rug"]
    assert native.resize_calls == []
    assert fixture.mesh.read_bytes() == before
    native.proposal.clear()
    with pytest.raises(controls.FrozenMutationError, match="inventory_mismatch"):
        function(native)
    assert len(native.pose_calls) == 1


def test_collision_cleanup_stops_without_deleting_or_claiming_collision_fix(controls, fixture):
    source = '''def del_top_collide_obj(state, iter):
    stop = True
    collisions = measure_collisions(state)
    if not collisions:
        return stop
    for max_key in collisions:
        delete_obj_with_children(state, max_key)
        state.objs.pop(max_key)
    stop = False
    return stop
'''
    measurements = []
    def measure(state):
        measurements.append(list(state.objs))
        return ["rug"]
    function = _compile(controls, source, "del_top_collide_obj",
                        {controls.GUARD_GLOBAL: fixture.guard, "measure_collisions": measure})
    assert function(fixture.state, 0) is True  # no legal cleanup, not a zero-collision verdict
    assert measurements == [["rug", "newroom_0-0"]]
    assert "rug" in fixture.state.objs
    assert "native_collision_cleanup" in fixture.guard.journal.read_text()


@pytest.mark.parametrize("path", ["Addition.apply", "Addition.apply_random", "Resample.apply",
                                  "Deletion.apply", "Solver.remove_object", "Solver.update_graph_size"])
def test_forbidden_tool_or_discrete_path_fails_before_writes(controls, fixture, path):
    function = _compile(controls, "def apply(self):\n    self.mutated = True\n", path,
                        {controls.GUARD_GLOBAL: fixture.guard})
    obj = SimpleNamespace(mutated=False)
    with pytest.raises(controls.FrozenMutationError, match="forbidden_native_operation"):
        function(obj)
    assert obj.mutated is False


def test_changed_resize_shape_fails_closed(controls):
    with pytest.raises(RuntimeError, match="one block"):
        controls._patch_function("populate_state_placeholders_mid", ast.parse("def changed():\n    return\n").body[0])


def _synthetic_repo(controls, tmp_path, monkeypatch):
    source = "def resize_obj(obj, size, apply_transform=True):\n    obj.scale = size\n    return obj\n"
    path = tmp_path / "native.py"
    path.write_text(source)
    module = ModuleType("synthetic_native")
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec", dont_inherit=True), vars(module))
    monkeypatch.setattr(controls, "SOURCE_HASHES", {"native.py": hashlib.sha256(path.read_bytes()).hexdigest()})
    monkeypatch.setattr(controls, "PATCH_TARGETS", {"native.py": ("resize_obj",)})
    return path, module


def test_installer_preserves_callable_aliases_and_source_bytes(controls, fixture, tmp_path, monkeypatch):
    path, module = _synthetic_repo(controls, tmp_path, monkeypatch)
    alias = module.resize_obj
    before = path.read_bytes()
    audit = controls.install_mutation_guards(tmp_path, fixture.guard, {"native.py": module})
    assert alias is module.resize_obj
    assert alias(fixture.obj, fixture.obj.dimensions) is fixture.obj
    assert fixture.obj.scale == [1, 1, 1]
    assert path.read_bytes() == before
    assert audit[0]["native_ast_sha256"] != audit[0]["patched_ast_sha256"]
    assert "native_patch_installed" in fixture.guard.journal.read_text()
    with pytest.raises(RuntimeError, match="already patched"):
        controls.install_mutation_guards(tmp_path, fixture.guard, {"native.py": module})


@pytest.mark.parametrize("tamper", ["source", "callable", "module_path"])
def test_installer_is_atomic_and_refuses_other_code(controls, fixture, tmp_path, monkeypatch, tamper):
    path, module = _synthetic_repo(controls, tmp_path, monkeypatch)
    original = module.resize_obj
    if tamper == "source":
        path.write_text(path.read_text()+"# edited\n")
    elif tamper == "callable":
        exec(compile("def resize_obj(obj, size, apply_transform=True):\n    return None\n", str(path), "exec"), vars(module))
    else:
        module.__file__ = str(tmp_path / "different.py")
    with pytest.raises(RuntimeError):
        controls.install_mutation_guards(tmp_path, fixture.guard, {"native.py": module})
    assert controls.GUARD_GLOBAL not in vars(module)
    assert "native_patch_installed" not in fixture.guard.journal.read_text()
    if tamper != "callable":
        assert module.resize_obj is original
