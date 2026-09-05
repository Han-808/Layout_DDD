"""Plugin boundary tests: synthetic assets/upstream modules, never a real API loop."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def plugin(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/external_harness_bridges/scene_weaver_frozen_plugin.py"
    monkeypatch.syspath_prepend(str(path.parent))
    spec = importlib.util.spec_from_file_location("sceneweaver_plugin_ci", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inputs(plugin, tmp_path):
    mesh = tmp_path / "identity.glb"
    mesh.write_bytes(b"CI identity fixture; not real mesh geometry")
    asset = {"asset_key": "imaginarium.exact", "source_db": "imaginarium", "category": "chair",
        "description": "Approved chair", "mesh_uri": str(mesh), "mesh_sha256": plugin.digest(mesh),
        "bbox_size": [1, 2, 3], "bbox_center_local": [0, 0, 1.5],
        "native_scale": [1, 1, 1], "physical_dimensions": [1, 2, 3], "canonical_front": [0, -1, 0]}
    plan = {"objects": [{"id": "chair_a", "description": "Public facing instruction"}]}
    request = {"count": 1, "prompt": "Arrange the approved chairs", "public_object_plan": plan,
               "public_object_plan_sha256": plugin.logical_hash(plan), "feedback_source": "native_sceneweaver_only",
               "benchmark_room": {"roomsize": [6, 5], "height": 3, "unit": "meter"}}
    catalog = {"logical_to_native_slot": {"chair_a": "chair_a", "chair_b": "chair_b"},
        "catalog": {"catalog_id": "fixture", "catalog_sha256": "1"*64},
        "frozen_asset_bindings": {"chair_a": asset, "chair_b": deepcopy(asset)}}
    prepared = plugin.prepare_input(request, catalog)
    return SimpleNamespace(request=request, catalog=catalog, prepared=prepared, mesh=mesh)


def native_initial(prepared):
    return {"roomsize": prepared["public"]["roomsize"],
        "big_category_dict": {s: "1" for s in prepared["bindings"]},
        "name_mapping": prepared["mapping"], "small_category_list": [], "Placement_small": [],
        "Placement_big": {s: {"1": {"position": [1, 2, 0], "rotation": 5,
                                      "size": obj["size"]}}
            for s, obj in prepared["public"]["objects"].items()}}


def test_public_projection_and_duplicate_asset_instances(plugin, inputs):
    p = inputs.prepared
    assert p["public"]["objects"]["chair_a"]["size"] == [2, 1, 3]
    assert p["bindings"]["chair_a"]["asset_key"] == p["bindings"]["chair_b"]["asset_key"]
    assert p["mapping"]["chair_a"] != p["mapping"]["chair_b"]
    assert str(inputs.mesh) not in plugin.frozen_prompt(p)
    assert "Public facing instruction" in plugin.frozen_prompt(p)
    assert "bbox_size_local" not in inputs.catalog["frozen_asset_bindings"]["chair_a"]
    assert plugin.prepare_input(inputs.request, inputs.catalog) == p


@pytest.mark.parametrize("field", ["reference_annotation", "evaluator_config", "mesh_path", "cache_path"])
def test_private_public_plan_rejected(plugin, inputs, field):
    inputs.request["public_object_plan"][field] = "private"
    inputs.request["public_object_plan_sha256"] = plugin.logical_hash(inputs.request["public_object_plan"])
    with pytest.raises(ValueError, match="private"):
        plugin.prepare_input(inputs.request, inputs.catalog)


@pytest.mark.parametrize("change", ["inventory", "replacement", "room", "scale", "missing_position_z", "insertion"])
def test_initializer_rejects_without_repair(plugin, inputs, tmp_path, change):
    value = native_initial(inputs.prepared)
    if change == "inventory":
        value["big_category_dict"]["chair_a"] = "2"
    elif change == "replacement":
        value["name_mapping"] = {**value["name_mapping"], "chair_a": None}
    elif change == "room":
        value["roomsize"] = [9, 9]
    elif change == "scale":
        value["Placement_big"]["chair_a"]["1"]["size"] = [1, 1, 1]
    elif change == "missing_position_z":
        value["Placement_big"]["chair_a"]["1"]["position"] = [1, 2]
    else:
        value["Placement_big"]["extra"] = deepcopy(value["Placement_big"]["chair_a"])
    path = tmp_path / "native.json"
    plugin.save_new(path, value)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        plugin.validate_initial_output(path, inputs.prepared)
    assert path.read_bytes() == before


def test_valid_initializer_retains_native_pose(plugin, inputs, tmp_path):
    value = native_initial(inputs.prepared)
    path = tmp_path / "native.json"
    plugin.save_new(path, value)
    assert plugin.validate_initial_output(path, inputs.prepared) == value
    assert value["Placement_big"]["chair_a"]["1"]["rotation"] == 5


@pytest.mark.parametrize("vertices", [
    [[0, 0], [6, 0], [6, 5], [3, 5], [3, 2], [0, 2]],
    [[0, 0], [6, 0], [6, 6], [0, 6]],
    [[0, 0], [6, 5], [6, 0], [0, 5]],
])
def test_native_contour_is_not_replaced_with_a_bounding_box(plugin, vertices):
    with pytest.raises(ValueError):
        plugin.validate_rectangle(vertices, [6, 5])


def test_rectangular_origin_conversion_is_explicit(plugin):
    points = [[2, 3], [8, 3], [8, 8], [2, 8], [2, 3]]
    assert plugin.validate_rectangle(points, [6, 5], allow_origin_shift=True) == points[:-1]
    with pytest.raises(ValueError, match="origin"):
        plugin.validate_rectangle(points, [6, 5])


def test_archive_retains_backtracking_bytes_without_duplicate_iteration_files(plugin, tmp_path):
    work = tmp_path / "native"
    work.mkdir()
    file = work / "layout_0.json"
    file.write_bytes(b'{"native": "first"}\n')
    archive = plugin.ArtifactArchive(tmp_path / "archive")
    first = plugin.read_json(archive.capture(work, "attempt_0"))
    file.write_bytes(b'{"native": "second"}\n')
    second = plugin.read_json(archive.capture(work, "attempt_1"))
    archive.capture(work, "unchanged")
    assert (archive.root / first["files"][0]["blob"]).read_bytes() == b'{"native": "first"}\n'
    assert (archive.root / second["files"][0]["blob"]).read_bytes() == file.read_bytes()
    assert len(list((archive.root / "blobs").iterdir())) == 2
    assert list(archive.root.rglob("layout_*.json")) == []
    assert first["files"][0]["native_path"] == "layout_0.json"


def test_sdk_tool_response_is_saved_before_native_consumer(plugin, tmp_path):
    raw = {"model": "same-model", "usage": {"total_tokens": 22},
           "choices": [{"message": {"content": None, "tool_calls": [{"function": {"name": "init_gpt"}}]}}]}
    response = SimpleNamespace(model="same-model", model_dump=lambda **kw: raw)
    calls = []
    def create(**kw):
        calls.append(kw)
        return response
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    recorder = plugin.ObservedCompletions(client, tmp_path / "calls", {"provider": "test", "model_id": "same-model"})
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]}]
    assert recorder.create(model="same-model", messages=messages, tools=[], max_tokens=4096) is response
    assert plugin.read_json(tmp_path / "calls/call_00000/response.json") == raw
    assert calls[0]["messages"] == messages
    assert recorder.tokens == 22


@pytest.mark.parametrize("failure", ["identity", "provider_error", "locator"])
def test_route_failure_never_silently_falls_back(plugin, tmp_path, failure):
    calls = []
    def create(**kw):
        calls.append(kw)
        if failure == "provider_error":
            raise RuntimeError("token secret-ci-value at https://private.test/key")
        return SimpleNamespace(model="other", model_dump=lambda **kw: {"model": "other"})
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    recorder = plugin.ObservedCompletions(client, tmp_path / "calls", {"provider": "test", "model_id": "same"},
                                          forbidden_locators=("/assets/private.glb",), secret="secret-ci-value")
    with pytest.raises(RuntimeError):
        recorder.create(model="same", messages=[{"role": "user", "content": "/assets/private.glb" if failure == "locator" else "public"}])
    assert len(calls) == (0 if failure == "locator" else 1)
    for file in (tmp_path / "calls").rglob("*.json"):
        assert "secret-ci-value" not in file.read_text()
    if failure == "identity":
        assert (tmp_path / "calls/call_00000/response.json").is_file()


def test_worker_never_inherits_generation_credentials(plugin, monkeypatch):
    monkeypatch.setenv("LAYOUT_DDD_API_KEY", "secret")
    monkeypatch.setenv("SOME_AUTH_TOKEN", "secret")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = plugin.worker_environment()
    assert "LAYOUT_DDD_API_KEY" not in result and "SOME_AUTH_TOKEN" not in result
    assert result["CUDA_VISIBLE_DEVICES"] == "0"
    assert result["PYTHONDONTWRITEBYTECODE"] == "1"


def test_config_overlay_does_not_open_native_key_file(plugin):
    tree = ast.parse("class Config:\n    def __init__(self):\n        raise AssertionError('keyfile read')\nconfig = Config()\n")
    plugin.config_overlay(tree)
    namespace = {}
    exec(compile(tree, "native_config_fixture", "exec"), namespace)
    assert "config" not in namespace and "Config" in namespace


def test_headless_overlay_preserves_native_room_work(plugin):
    tree = ast.parse("def build_room_structure():\n    state, solver = generate_room()\n    for area in bpy.context.screen.areas:\n        override = area\n    return state, solver, override\n")
    plugin.room_ui_overlay(tree)
    ns = {"bpy": SimpleNamespace(app=SimpleNamespace(background=True), context=SimpleNamespace(screen=None)),
          "generate_room": lambda: ("native_state", "native_solver")}
    exec(compile(ast.fix_missing_locations(tree), "native_room_fixture", "exec"), ns)
    assert ns["build_room_structure"]() == ("native_state", "native_solver", {})


def test_room_height_is_bound_before_module_level_capture(plugin):
    tree = ast.parse("def global_params(unit=0.5, wall_height=('uniform', 2.7, 3.8)):\n    return wall_height\nWALL_HEIGHT = global_params()\n")
    plugin.height_overlay(tree, 3.1)
    ns = {}
    exec(compile(ast.fix_missing_locations(tree), "native_height_fixture", "exec"), ns)
    assert ns["WALL_HEIGHT"] == 3.1
    assert ns["global_params"].__defaults__[0] == 0.5


def test_transitive_helper_pins_match_checked_in_files(plugin):
    root = Path(plugin.__file__).parent
    assert len(plugin.HELPER_HASHES) == 4
    for name, expected in plugin.HELPER_HASHES.items():
        assert plugin.digest(root / name) == expected


def test_native_environment_failure_is_before_client_and_agent(plugin, inputs, tmp_path, monkeypatch):
    args = SimpleNamespace(repo_path=tmp_path, output_root=tmp_path / "output", worker_python=sys.executable,
                           plugin_report=tmp_path / "report.json")
    args.output_root.mkdir()
    monkeypatch.setenv("LAYOUT_DDD_API_KEY", "secret-ci")
    common = importlib.import_module("_common")
    monkeypatch.setattr(common, "required_model_identity", lambda: {"provider": "test", "model_id": "same"})
    monkeypatch.setattr(common, "required_model_deployment_id", lambda: "deployment")
    monkeypatch.setattr(common, "verify_api_endpoint_contract", lambda *a, **kw: "1"*64)
    fake_sdk = ModuleType("openai")
    fake_sdk.OpenAI = lambda **kw: pytest.fail("must not construct model client")
    monkeypatch.setitem(sys.modules, "openai", fake_sdk)
    fake_http = ModuleType("httpx")
    fake_http.Client = lambda **kw: pytest.fail("must not construct transport")
    monkeypatch.setitem(sys.modules, "httpx", fake_http)
    monkeypatch.setattr(plugin.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="before any model call"):
        plugin.run_driver(args, inputs.prepared)
    assert (args.output_root / "generation_asset_selection.json").is_file()
    assert not (args.output_root / "model_calls").exists()


def test_existing_report_rejected_before_any_runtime_import(plugin, tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("preserved existing artifact")
    with pytest.raises(FileExistsError):
        plugin.run_driver(SimpleNamespace(plugin_report=path), {})
    assert path.read_text() == "preserved existing artifact"


@pytest.mark.parametrize("last_worker_fails", [False, True])
def test_driver_routes_two_native_states_through_existing_bridge_and_converter(plugin, inputs, tmp_path, monkeypatch, last_worker_fails):
    """Mocked native driver/Blender, real plugin orchestration/bridge/converter.

    This tests routing, not actual native reasoning or mesh generation.
    """
    from benchmark.adapters.scene_weaver.converter import convert_scene_weaver
    from benchmark.nl_scene.generation_input import build_generation_input, build_scene_request
    from scene_weaver_frozen import _observe_trajectory, _layouts, _anchor_basis, _orientation_basis
    import _common as common
    args = SimpleNamespace(repo_path=tmp_path / "repo", output_root=tmp_path / "output", worker_python=sys.executable,
        comparison_input=tmp_path / "control.json", comparison_catalog=tmp_path / "catalog.json",
        request=tmp_path / "request.json", plugin_report=tmp_path / "report.json")
    args.output_root.mkdir()
    (args.repo_path / "Pipeline/config").mkdir(parents=True)
    plugin.save_new(args.repo_path / "Pipeline/config/config.json", {"llm": {"api_key": "/must-not-read-native-key",
        "max_tokens": 4096, "temperature": 0.7}})
    control = {"generation": {"asset_geometry_tolerance_m": 1e-4}}
    plugin.save_new(args.comparison_input, control)
    plugin.save_new(args.comparison_catalog, inputs.catalog)
    plugin.save_new(args.request, inputs.request)
    monkeypatch.setenv("LAYOUT_DDD_API_KEY", "secret-ci")
    monkeypatch.setenv("LAYOUT_DDD_API_BASE_URL", "https://configured.invalid/v1")
    monkeypatch.setattr(common, "required_model_identity", lambda: {"provider": "test", "model_id": "same"})
    monkeypatch.setattr(common, "required_model_deployment_id", lambda: "deployment")
    monkeypatch.setattr(common, "verify_api_endpoint_contract", lambda *a, **kw: "1"*64)
    monkeypatch.setattr(plugin, "verify_sources", lambda repo: [])
    calls, commands = [], []
    def response(**kw):
        calls.append(kw)
        return SimpleNamespace(model="same", model_dump=lambda **kw: {"model": "same", "usage": {"total_tokens": 5}})
    sdk = ModuleType("openai")
    sdk.OpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=response)),
                                            close=lambda: None, max_retries=2)
    http = ModuleType("httpx")
    http.Client = lambda **kw: SimpleNamespace()
    monkeypatch.setitem(sys.modules, "openai", sdk)
    monkeypatch.setitem(sys.modules, "httpx", http)
    gpt = ModuleType("gpt")
    gpt.GPT4 = type("GPT4", (), {})
    llm = ModuleType("app.llm")
    llm.MULTIMODAL_MODELS = []
    backend = ModuleType("app.tool.update_infinigen")
    prompt = ModuleType("app.prompt.gpt.init_gpt")
    for name in ("step_1_big_object_prompt_system", "step_3_class_name_prompt_system", "step_5_position_prompt_system"):
        setattr(prompt, name, "native prompt")
    native = ModuleType("app.agent.scenedesigner")
    native.ToolCollection = lambda *tools: tuple(tools)
    for name in ("AddRelationExecute", "UpdateLayoutExecute", "UpdateRotationExecute", "Terminate"):
        setattr(native, name, type(name, (), {}))
    initializer = SimpleNamespace(InitGPTExecute=type("InitGPTExecute", (), {}))
    def overlay(repo, module_name, relative, transform, audit):
        if module_name == "app.config":
            return SimpleNamespace(LLMSettings=lambda **kw: SimpleNamespace(**kw))
        return initializer
    monkeypatch.setattr(plugin, "load_overlay", overlay)
    for module in (gpt, llm, backend, prompt, native):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    run = args.output_root / "sceneweaver_native"
    native_bytes = {}
    def worker(command, **kwargs):
        assert "LAYOUT_DDD_API_KEY" not in kwargs["env"]
        commands.append(command)
        attempt = Path(command[command.index("--attempt-dir")+1])
        if "--worker-preflight" in command:
            assert calls == []
            plugin.save_new(attempt / "worker_preflight.json", {"status": "fixture-only"})
            return SimpleNamespace(returncode=0)
        record = plugin.read_json(run / "args.json")
        iteration = record["iter"]
        if iteration == 1 and last_worker_fails:
            return SimpleNamespace(returncode=1)
        objs, observations = {}, {}
        for slot, asset in inputs.prepared["bindings"].items():
            pose = [1.125 + iteration, 2.0, 0.0]
            rotation = [0, 0, 0.125 * iteration]
            size = inputs.prepared["public"]["objects"][slot]["size"]
            objs[slot] = {"location": [round(v, 2) for v in pose], "rotation": [round(v, 2) for v in rotation], "size": size}
            observations[slot] = {"asset_id": asset["asset_key"], "mesh_path": asset["mesh_uri"],
                "mesh_sha256": asset["mesh_sha256"], "canonical_local_bbox_size": asset["physical_dimensions"],
                "orientation_basis": _orientation_basis(asset), "anchor_basis": _anchor_basis(asset),
                "full_precision_native_bottom_center": pose, "full_precision_native_euler_xyz": rotation,
                "full_precision_native_object_dimensions": size}
        path = run / "record_scene" / f"layout_{iteration}.json"
        plugin.save_new(path, {"objects": objs, "structure": {}, "roomsize": [6, 5]})
        native_bytes[iteration] = path.read_bytes()
        plugin.save_new(run / "observations" / f"iteration_{iteration}.json", {"iteration": iteration,
            "objects": observations, "native_room_observation": {"roomsize": [6, 5], "height": 3, "unit": "meter"}})
        record["success"] = True
        (run / "args.json").write_text(json.dumps(record))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(plugin.subprocess, "run", worker)
    class FakeNativeAgent:
        max_steps = 15
        system_prompt = "native system"
        def run(self, public_prompt):
            assert "Public facing instruction" in public_prompt
            assert str(inputs.mesh) not in public_prompt
            assert len(self.available_tools0) == 1 and len(self.available_tools1) == 4
            model = gpt.GPT4()
            model.client.chat.completions.create(model=model.MODEL, messages=[{"role": "user", "content": public_prompt}])
            initial = run / "pipeline/init.json"
            plugin.save_new(initial, native_initial(inputs.prepared))
            assert backend.update_infinigen("init_gpt", 0, str(initial))
            if last_worker_fails:
                with pytest.raises(RuntimeError, match="native worker failed"):
                    backend.update_infinigen("update", 1, str(initial))
            else:
                assert backend.update_infinigen("update", 1, str(initial))
            return "native fixture run completed"
    native.SceneDesigner = FakeNativeAgent
    old_path = sys.path[:]
    try:
        if last_worker_fails:
            with pytest.raises(RuntimeError, match="earlier layouts are preserved"):
                plugin.run_driver(args, inputs.prepared)
        else:
            plugin.run_driver(args, inputs.prepared)
    finally:
        sys.path[:] = old_path
    assert len(commands) == 3 and len(calls) == 1
    if last_worker_fails:
        assert not args.plugin_report.exists()
        assert (run / "record_scene/layout_0.json").read_bytes() == native_bytes[0]
        assert list((args.output_root / "native_archive/blobs").iterdir())
        return
    report = plugin.read_json(args.plugin_report)
    assert report["resource_usage"]["tokens"] == 5
    layouts = _layouts(args.output_root)
    assert [i for i, _ in layouts] == [0, 1]
    valid, bindings = _observe_trajectory(layouts=layouts, control=control, catalog=inputs.catalog,
        request=inputs.request, plugin_report=report, tolerance=1e-6)
    assert valid["valid"] is True
    generation_input = build_generation_input(scene_request=build_scene_request(request_id="fixture",
        instruction="Public instruction", scene_type="living_room",
        room={"boundary": [[0, 0], [6, 0], [6, 5], [0, 5]], "height": 3, "unit": "meter"}, structure=False))
    class ExactProvider:
        def resolve(self, key, *, source_db=None, hint=None):
            assert key == "imaginarium.exact"
            return dict(inputs.prepared["bindings"]["chair_a"])
        def retrieve(self, *a, **kw):
            pytest.fail("strict conversion must not retrieve")
    for iteration, path in layouts:
        scene = convert_scene_weaver(path.parent, generation_input, {
            "selected_iteration": iteration, "rotation_unit": "radian", "asset_bindings": bindings,
            "sceneweaver_native_size_semantics": "released_object_dimensions_rounded_2dp",
            "sceneweaver_asset_geometry_tolerance_m": 1e-4,
            "sceneweaver_orientation_basis": "bake_catalog_front_to_sceneweaver_positive_x",
            "sceneweaver_anchor_basis": "rebase_catalog_bbox_bottom_center_to_sceneweaver_origin"}, ExactProvider())
        assert {obj["id"] for obj in scene["objects"]} == {"chair_a", "chair_b"}
        assert {obj["asset_ref"]["asset_key"] for obj in scene["objects"]} == {"imaginarium.exact"}
        assert path.read_bytes() == native_bytes[iteration]
