#!/usr/bin/env python3
"""SceneWeaver–FrozenAssets (restricted mutation set), pinned release launcher.

Runs the released SceneDesigner and its native initialization, optimization and
reflection. Only generation input, transport, frozen mutation controls and
observation are supplied here. No converter, evaluator or placement algorithm.
The separate worker uses the configured SceneWeaver Python environment.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace


UPSTREAM_COMMIT = "7ae54b2ec3fc66147704faa7daf7b017ba8b1bd9"
# This entrypoint is itself pinned by the existing bridge. Check its transitive
# helpers BEFORE importing them; updating a helper requires a new plugin pin.
HELPER_HASHES = {
    "_common.py": "f8ae27c903071d832a995c4e910fceb457242fa98c892e4f9044e1a181150c6f",
    "scene_weaver_frozen.py": "9228f57d5720515b88836e3b370e0bf0188060e74894ad352b8b6f5c60f7f42e",
    "scene_weaver_frozen_assets.py": "453bf81461136688182b91026eb04f0bc4ac9ca13f30766e1b89cd0e492c6ddd",
    "scene_weaver_frozen_mutations.py": "d2a8d13ebab871a14992e15da10e0c0904b53c31c262ab7b41e03ed4b4ab6455",
}
OVERLAY_HASHES = {
    "Pipeline/app/config.py": "a8ed428c0224c2f8581617e622fc3c7651b3846bfef0f2751acbed424c06213c",
    "Pipeline/app/tool/init_gpt.py": "3e59bf04c870a0318cf7829f4402da4b698b6a76715c549dce8827343819d77a",
    "infinigen_examples/steps/room_structure.py": "644a46753499179fd5874a5fa629b6eac08d9352aa88691ddbcc1b686bd13e29",
    "infinigen/core/constraints/example_solver/room/constants.py": "09aa578877a82c77e3bcbc492d5c4299283918b6925c9e0ef43a1ce80250425b",
}
ALLOWED_ACTIONS = {"init_gpt", "update", "add_relation", "finalize_scene"}
DISABLED_STAGES = ("floating_objs", "room_pillars", "room_stairs", "room_doors",
                   "room_windows", "skirting_floor", "skirting_ceiling")
PRIVATE_KEYS = {"reference_annotation", "evaluation_report", "evaluator_config",
                "hidden_annotations", "benchmark_score", "self_reflection",
                "mesh_uri", "mesh_path", "asset_path", "local_path", "asset_root",
                "path", "file_path", "cache_path", "metadata_path", "metadata_uri"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def logical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_sources(repo):
    helpers = {"_common.py", "scene_weaver_frozen.py", "scene_weaver_frozen_assets.py",
               "scene_weaver_frozen_mutations.py"}
    if set(HELPER_HASHES) != helpers:
        raise RuntimeError("plugin transitive source pins are incomplete")
    for name, expected in HELPER_HASHES.items():
        if digest(Path(__file__).parent / name) != expected:
            raise RuntimeError(f"plugin helper identity mismatch: {name}")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError("SceneWeaver upstream commit differs from qualified source contract")
    # No full diff content is logged (an operator's local file could hold secrets).
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"], text=True).strip():
        raise RuntimeError("SceneWeaver upstream must be a clean isolated checkout")
    for name, expected in OVERLAY_HASHES.items():
        if digest(repo / name) != expected:
            raise RuntimeError(f"native overlay source mismatch: {name}")
    from scene_weaver_frozen_mutations import build_patch_plan
    return build_patch_plan(repo)


def reject_private(value):
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(value):
            raise ValueError("public SceneWeaver input contains private fields or asset locators")
        for item in value.values():
            reject_private(item)
    elif isinstance(value, list):
        for item in value:
            reject_private(item)


def prepare_input(request, catalog):
    """Generation-input normalization only; never receives internal benchmark state."""
    from scene_weaver_frozen_assets import validate_binding
    from scene_weaver_frozen import _orientation_basis
    if request.get("feedback_source") != "native_sceneweaver_only" or request.get("count") != 1:
        raise ValueError("one native-only SceneWeaver trajectory is required")
    plan = request.get("public_object_plan")
    reject_private(plan)
    if not isinstance(plan, dict) or logical_hash(plan) != request.get("public_object_plan_sha256"):
        raise ValueError("public object plan identity mismatch")
    room = request["benchmark_room"]
    dims = [float(v) for v in room["roomsize"]]
    height = float(room["height"])
    if len(dims) != 2 or not all(math.isfinite(v) and v > 0 for v in dims + [height]) or room.get("unit") != "meter":
        raise ValueError("positive rectangular room in meters required")
    boundary = room.get("boundary")
    if boundary is not None:
        validate_rectangle(boundary, dims, allow_origin_shift=True)
    frozen = catalog["frozen_asset_bindings"]
    slot_map = catalog["logical_to_native_slot"]
    if not frozen or slot_map != {slot: slot for slot in frozen}:
        raise ValueError("SceneWeaver requires exact identity slot mapping")
    bindings, mapping, public_objects = {}, {}, {}
    for slot, record in frozen.items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", slot) or slot.startswith(("newroom", "window", "entrance")):
            raise ValueError("slot conflicts with the released native identifier/export contract")
        asset = deepcopy(record)
        if "bbox_size_local" in asset and asset["bbox_size_local"] != asset.get("bbox_size", asset["bbox_size_local"]):
            raise ValueError("conflicting materialized local bbox fields")
        asset["bbox_size_local"] = asset.get("bbox_size_local", asset.get("bbox_size"))
        bindings[slot] = validate_binding(slot, asset, tolerance=1e-4)
        mapping[slot] = f"layout_ddd_frozen.Frozen{hashlib.sha256(slot.encode()).hexdigest()[:24]}Factory"
        dimensions = list(bindings[slot]["physical_dimensions"])
        if round(_orientation_basis(asset)["basis_yaw_degrees"] / 90) % 2:
            dimensions[0], dimensions[1] = dimensions[1], dimensions[0]
        public_objects[slot] = {"asset_id": asset["asset_key"], "category": asset.get("category"),
                                "description": asset.get("description"), "size": dimensions}
    public = {"prompt": request["prompt"], "public_object_plan": plan,
              "roomsize": dims, "height": height, "objects": public_objects,
              "factory_mapping": mapping}
    reject_private(public)
    return {"bindings": bindings, "mapping": mapping, "public": public,
            "public_object_plan_sha256": logical_hash(plan)}


def validate_rectangle(vertices, dimensions, *, allow_origin_shift=False):
    points = [[float(v) for v in p] for p in vertices]
    if points and points[-1] == points[0]:
        points.pop()
    if len(points) != 4 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
        raise ValueError("one four-corner native rectangle required")
    origin = [min(p[i] for p in points) for i in range(2)]
    normalized = [[p[i]-origin[i] for i in range(2)] for p in points]
    expected = [[0, 0], [dimensions[0], 0], list(dimensions), [0, dimensions[1]]]
    if not allow_origin_shift and max(abs(v) for v in origin) > 1e-6:
        raise ValueError("native room origin differs from its contract")
    if any(min(math.dist(p, q) for q in expected) > 1e-6 for p in normalized) or len({tuple(p) for p in normalized}) != 4:
        raise ValueError("native boundary is not the frozen rectangle")
    for a, b in zip(normalized, normalized[1:]+normalized[:1]):
        if abs(a[0]-b[0]) > 1e-6 and abs(a[1]-b[1]) > 1e-6:
            raise ValueError("native room edges differ from axis-aligned contract")
    return points


def frozen_prompt(prepared):
    return ("\nFROZEN-ASSETS INPUT CONTRACT (overrides asset/count/size choices in examples):\n"
            "Every key in objects is a distinct native category/slot, count exactly 1. "
            "Use all and only these slots; the instance key in Placement is \"1\". "
            "Do not enlarge the room, retrieve, replace, add, remove or resize objects. "
            "Copy the provided factory_mapping exactly into Mapping results; never use null. "
            "Choose native relations, positions and rotations yourself. Position has THREE "
            "coordinates in meters, rotation is native degrees at initialization; copy each "
            "provided size exactly. Native +X is the facing basis when front is known. "
            "Keep supported objects in Placement with their native parent relation. "
            "The public object plan is part of the task, not evaluator feedback.\n"
            + json.dumps(prepared["public"], ensure_ascii=False, allow_nan=False))


def validate_initial_output(path, prepared):
    """Reject altered inventory/room/bindings before native execution, no repair."""
    value = read_json(path)
    slots = set(prepared["bindings"])
    counts = value.get("big_category_dict", {})
    if set(counts) != slots or any(str(v) != "1" for v in counts.values()):
        raise ValueError("native initializer changed frozen inventory")
    if value.get("name_mapping") != prepared["mapping"]:
        raise ValueError("native initializer changed exact asset factories")
    if value.get("roomsize") != prepared["public"]["roomsize"]:
        raise ValueError("native initializer changed frozen room")
    if value.get("small_category_list") or value.get("Placement_small"):
        raise ValueError("native initializer inserted additional inventory")
    placements = value.get("Placement_big", {})
    if set(placements) != slots:
        raise ValueError("native placements differ from frozen inventory")
    for slot, instances in placements.items():
        if set(instances) != {"1"}:
            raise ValueError("one placement per exact slot is required")
        pose = instances["1"]
        if pose.get("size") != prepared["public"]["objects"][slot]["size"]:
            raise ValueError("native initializer attempted asset scaling")
        pos = pose.get("position")
        if not isinstance(pos, list) or len(pos) != 3 or not all(math.isfinite(float(v)) for v in pos):
            raise ValueError("native initializer requires finite three-coordinate positions")
        if not math.isfinite(float(pose.get("rotation"))):
            raise ValueError("native initializer requires finite rotation")
    return value


class ArtifactArchive:
    """Byte-preserving content-addressed snapshots before native backtracking writes.

    Blob names deliberately are not layout_N.json, avoiding duplicate iteration
    discovery. Each manifest retains the exact original relative name and hash.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(exist_ok=False)
        (self.root / "blobs").mkdir()
        self.sequence = 0

    def capture(self, source, label):
        rows = []
        for path in sorted(Path(source).rglob("*")):
            if path.is_symlink():
                raise RuntimeError("native artifact symlinks are not accepted")
            if not path.is_file():
                continue
            sha = digest(path)
            target = self.root / "blobs" / sha
            if not target.exists():
                with path.open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            if digest(target) != sha or digest(path) != sha:
                raise RuntimeError("native artifact changed during preservation")
            rows.append({"native_path": path.relative_to(source).as_posix(), "sha256": sha,
                         "blob": target.relative_to(self.root).as_posix(), "bytes": path.stat().st_size})
        manifest = self.root / f"snapshot_{self.sequence:04d}.json"
        save_new(manifest, {"label": label, "files": rows})
        self.sequence += 1
        return manifest


class ObservedCompletions:
    """Preserve SDK response BEFORE native code consumes it, including tool calls."""
    def __init__(self, client, output, identity, *, forbidden_locators=(), secret=""):
        self.client, self.root, self.identity = client, Path(output), identity
        self.root.mkdir(exist_ok=False)
        self.forbidden_locators = tuple(forbidden_locators)
        self.secret = secret
        self.calls, self.identities, self.tokens = 0, [], 0

    def create(self, **kwargs):
        from _common import observed_model_identity, require_observed_model_match, redact_error_detail
        if kwargs.get("model") != self.identity["model_id"] or kwargs.get("stream"):
            raise RuntimeError("native call changed configured model/response contract")
        serialized = json.dumps(kwargs, ensure_ascii=False, allow_nan=False)
        if any(locator and locator in serialized for locator in self.forbidden_locators):
            raise RuntimeError("native model request contains host-local asset locator")
        reject_private(kwargs.get("messages"))
        call = self.root / f"call_{self.calls:05d}"
        call.mkdir()
        self.calls += 1
        save_new(call / "request.json", kwargs)
        started = time.monotonic()
        try:
            response = self.client.chat.completions.create(**kwargs)
            raw = response.model_dump(mode="json")
            save_new(call / "response.json", raw)
            observed = observed_model_identity(response, provider=self.identity["provider"])
            require_observed_model_match(self.identity, [observed])
            self.identities.append(observed)
            self.tokens += int((raw.get("usage") or {}).get("total_tokens") or 0)
            save_new(call / "result.json", {"status": "received", "runtime_seconds": time.monotonic()-started,
                                           "observed_identity": observed})
            return response
        except Exception as exc:
            save_new(call / "failure.json", {"runtime_seconds": time.monotonic()-started,
                "error": redact_error_detail(str(exc), secrets=(self.secret,), truncated=False)})
            # Native retry policy is retained, but credential-bearing exceptions
            # must not reach its verbose logger.
            raise RuntimeError(f"configured SceneWeaver model call failed; see {call.name}") from None


def load_overlay(repo, module_name, relative, transform, audit_dir):
    """Load an audited native module with a narrow initialization/UI overlay."""
    if module_name in sys.modules:
        raise RuntimeError(f"native module imported before overlay: {module_name}")
    path = repo / relative
    if digest(path) != OVERLAY_HASHES[relative]:
        raise RuntimeError("native overlay source drift")
    tree = ast.parse(path.read_text(), filename=str(path))
    before = ast.dump(tree, include_attributes=False)
    transform(tree)
    ast.fix_missing_locations(tree)
    save_new(Path(audit_dir) / f"{module_name}.json", {"source": relative, "sha256": digest(path),
                "native_ast_sha256": hashlib.sha256(before.encode()).hexdigest(),
                "patched_ast": ast.dump(tree, include_attributes=False)})
    module = ModuleType(module_name)
    module.__file__, module.__package__ = str(path), module_name.rpartition(".")[0]
    sys.modules[module_name] = module
    exec(compile(tree, str(path), "exec", dont_inherit=True), vars(module))
    return module


def config_overlay(tree):
    matches = [n for n in tree.body if isinstance(n, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == "config" for t in n.targets)]
    if len(matches) != 1 or ast.unparse(matches[0].value) != "Config()":
        raise RuntimeError("released Config singleton changed")
    tree.body.remove(matches[0])


def init_overlay(tree):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "InitGPTExecute")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
    block = next(n for n in method.body if isinstance(n, ast.Try))
    calls = [n for n in block.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and ast.unparse(n.value.func) == "os.system"]
    if len(calls) != 1 or "../run/roominfo.json" not in ast.unparse(calls[0]):
        raise RuntimeError("released initializer roominfo copy changed")
    block.body.remove(calls[0])


def room_ui_overlay(tree):
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_room_structure")
    loops = [n for n in node.body if isinstance(n, ast.For) and ast.unparse(n.iter) == "bpy.context.screen.areas"]
    if len(loops) != 1:
        raise RuntimeError("released room UI initialization changed")
    index = node.body.index(loops[0])
    node.body[index:index+1] = [ast.parse("override = {}").body[0],
        ast.If(test=ast.parse("not bpy.app.background and bpy.context.screen is not None", mode="eval").body,
               body=loops, orelse=[])]


def height_overlay(tree, height):
    """Bind public room height before native modules capture WALL_HEIGHT by value."""
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "global_params")
    defaults = dict(zip([a.arg for a in node.args.args][-len(node.args.defaults):], node.args.defaults))
    if ast.literal_eval(defaults["wall_height"]) != ("uniform", 2.7, 3.8):
        raise RuntimeError("native default room height changed")
    index = [a.arg for a in node.args.args].index("wall_height") - (len(node.args.args)-len(node.args.defaults))
    node.args.defaults[index] = ast.Constant(value=float(height))


def import_height_constants(repo, height, audit):
    # Importing example_solver executes its __init__ and imports the room
    # solidifier, which captures WALL_HEIGHT by value. A later gin override
    # alone would falsely report the requested height while building another.
    import importlib.abc
    import importlib.machinery
    import importlib.util
    name = "infinigen.core.constraints.example_solver.room.constants"
    relative = "infinigen/core/constraints/example_solver/room/constants.py"
    path = repo / relative
    if name in sys.modules:
        raise RuntimeError("room constants imported before frozen architecture binding")
    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            if digest(path) != OVERLAY_HASHES[relative]:
                raise RuntimeError("native room constants source drift")
            tree = ast.parse(path.read_text(), filename=str(path))
            height_overlay(tree, height)
            save_new(Path(audit) / "room_height_input.json", {"source": relative, "source_sha256": digest(path),
                "height_m": height, "patched_ast": ast.dump(tree, include_attributes=False)})
            return compile(ast.fix_missing_locations(tree), str(path), "exec", dont_inherit=True)
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == name:
                return importlib.util.spec_from_file_location(name, repo / relative, loader=Loader(name, str(repo / relative)))
            return None
    finder = Finder()
    sys.meta_path.insert(0, finder)
    try:
        return importlib.import_module(name)
    finally:
        sys.meta_path.remove(finder)


def run_driver(args, prepared):
    if Path(args.plugin_report).exists():
        raise FileExistsError("plugin report already exists; use a new run, not an overwrite")
    from _common import required_model_identity, required_model_deployment_id, verify_api_endpoint_contract
    from openai import OpenAI
    import httpx
    identity = required_model_identity()
    deployment = required_model_deployment_id()
    endpoint = verify_api_endpoint_contract(os.environ.get("LAYOUT_DDD_API_BASE_URL", ""), completion_endpoint=False)
    secret = os.environ.get("LAYOUT_DDD_API_KEY", "").strip()
    if not secret:
        raise RuntimeError("LAYOUT_DDD_API_KEY is required for generation (not for --preflight-only)")
    root = args.output_root.resolve()
    # Native code contains shell cp statements. Never interpolate unsafe paths.
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", str(root)):
        raise ValueError("released native file-copy commands require a shell-safe output path")
    run = root / "sceneweaver_native"
    run.mkdir(exist_ok=False)
    for name in ("pipeline", "args", "record_files", "record_scene", "observations"):
        (run / name).mkdir()
    audit = root / "plugin_audit"
    audit.mkdir(exist_ok=False)
    archive = ArtifactArchive(root / "native_archive")
    context = root / "frozen_worker_input.json"
    save_new(context, prepared)
    # Fixed IDs are durably saved before any model call or conversion.
    save_new(root / "generation_asset_selection.json", {"selection_policy": "provided_exact_ids",
        "bindings": {s: a["asset_key"] for s, a in prepared["bindings"].items()}, "retrieval_calls": 0})
    # Import and install the real worker boundary BEFORE the first paid call.
    # This is dependency qualification, not an optimizer/renderer smoke test.
    qualification = root / "native_environment_preflight"
    qualification.mkdir()
    child_env = worker_environment()
    with (qualification / "stdout.txt").open("x") as out, (qualification / "stderr.txt").open("x") as err:
        checked = subprocess.run([args.worker_python, str(Path(__file__).resolve()), "--worker",
            "--worker-preflight", "--repo-path", str(args.repo_path), "--worker-input", str(context),
            "--native-run", str(run), "--attempt-dir", str(qualification)], cwd=args.repo_path,
            env=child_env, stdout=out, stderr=err, check=False)
    if checked.returncode != 0 or not (qualification / "worker_preflight.json").is_file():
        raise RuntimeError("SceneWeaver native environment preflight failed before any model call")
    os.environ.update(save_dir=str(run), UserDemand=frozen_prompt(prepared), socket="False",
                      sceneweaver_dir=str(args.repo_path))
    sys.path[:0] = [str(args.repo_path / "Pipeline"), str(args.repo_path)]
    # Read the released non-secret budget fields without opening its api_key file.
    config_module = load_overlay(args.repo_path, "app.config", "Pipeline/app/config.py", config_overlay, audit)
    # Native app.logger writes beneath PROJECT_ROOT at import time. Relocate
    # only that runtime root; never create logs/config inside the pinned source.
    config_module.PROJECT_ROOT = root / "native_driver"
    native = read_json(args.repo_path / "Pipeline/config/config.json")["llm"]
    settings = config_module.LLMSettings(model=identity["model_id"], base_url="configured-route",
        api_key="injected-client", api_type="Openai", api_version="",
        max_tokens=native.get("max_tokens", 4096), max_input_tokens=native.get("max_input_tokens"),
        temperature=native.get("temperature", 1.0))
    config_module.config = SimpleNamespace(llm={"default": settings}, workspace_root=Path("native_scene"))
    client = OpenAI(api_key=secret, base_url=os.environ["LAYOUT_DDD_API_BASE_URL"],
                    http_client=httpx.Client(follow_redirects=False))
    observed = ObservedCompletions(client, root / "model_calls", identity,
        forbidden_locators=[a["mesh_uri"] for a in prepared["bindings"].values()], secret=secret)
    proxy = SimpleNamespace(chat=SimpleNamespace(completions=observed))
    gpt = importlib.import_module("gpt")

    def gpt_init(self, version="gpt-4-turbo", region="eastus2"):
        self.version, self.MODEL, self.client = version, identity["model_id"], proxy
    gpt.GPT4.__init__ = gpt_init
    llm = importlib.import_module("app.llm")
    llm.AzureOpenAI = lambda **kwargs: proxy
    llm.MULTIMODAL_MODELS = set(llm.MULTIMODAL_MODELS) | {identity["model_id"]}
    backend = importlib.import_module("app.tool.update_infinigen")
    attempts = root / "worker_attempts"
    attempts.mkdir()
    counter = 0
    last_attempt_success = False

    def update(action, iter, json_name, ideas=None, description=None, inplace=False, invisible=False):
        nonlocal counter, last_attempt_success
        last_attempt_success = False
        if action not in ALLOWED_ACTIONS or invisible:
            raise RuntimeError("native action not permitted in FrozenAssets track")
        if json_name:
            native_path = Path(json_name).resolve()
            if not native_path.is_relative_to(run) or not native_path.is_file():
                raise RuntimeError("native action input must be an emitted file in this run")
        archive.capture(run, f"before_{counter}_{action}_{iter}")
        if action == "init_gpt":
            validate_initial_output(json_name, prepared)
        record = {"iter": iter, "action": action, "json_name": json_name,
                  "ideas": ideas, "description": description, "inplace": inplace, "success": False}
        # args.json is a native working file. Its previous bytes were archived.
        (run / "args.json").write_text(json.dumps(record, indent=4))
        (run / "args" / f"args_{iter}.json").write_text(json.dumps(record, indent=4))
        attempt = attempts / f"attempt_{counter:04d}"
        attempt.mkdir()
        counter += 1
        command = [args.worker_python, str(Path(__file__).resolve()), "--worker",
                   "--repo-path", str(args.repo_path), "--worker-input", str(context),
                   "--native-run", str(run), "--attempt-dir", str(attempt)]
        started = time.monotonic()
        save_new(attempt / "launch.json", {"command": command, "cwd": str(args.repo_path),
                  "native_action": record, "start": datetime.now(timezone.utc).isoformat()})
        code = None
        try:
            with (attempt / "stdout.txt").open("x") as out, (attempt / "stderr.txt").open("x") as err:
                # Same outer process group: existing runner owns timeout/cancel
                # cleanup for driver AND native descendants. No detached server.
                child_env = worker_environment()
                code = subprocess.run(command, cwd=args.repo_path, env=child_env,
                                      stdout=out, stderr=err, check=False).returncode
        finally:
            snapshot = archive.capture(run, f"after_{counter-1}_{action}_{iter}")
            save_new(attempt / "result.json", {"return_code": code, "runtime_seconds": time.monotonic()-started,
                                               "native_snapshot": str(snapshot)})
        if code != 0 or read_json(run / "args.json").get("success") is not True:
            raise RuntimeError(f"SceneWeaver native worker failed: {attempt.name}")
        last_attempt_success = True
        return True

    backend.update_infinigen = update
    initializer = load_overlay(args.repo_path, "app.tool.init_gpt", "Pipeline/app/tool/init_gpt.py", init_overlay, audit)
    prompts = importlib.import_module("app.prompt.gpt.init_gpt")
    for name in ("step_1_big_object_prompt_system", "step_3_class_name_prompt_system", "step_5_position_prompt_system"):
        setattr(prompts, name, getattr(prompts, name) + frozen_prompt(prepared))
    native_agent = importlib.import_module("app.agent.scenedesigner")
    cls = native_agent.SceneDesigner
    cls.available_tools0 = native_agent.ToolCollection(initializer.InitGPTExecute())
    cls.available_tools1 = native_agent.ToolCollection(native_agent.AddRelationExecute(),
        native_agent.UpdateLayoutExecute(), native_agent.UpdateRotationExecute(), native_agent.Terminate())
    cls.available_tools2 = native_agent.ToolCollection(native_agent.Terminate())
    cls.system_prompt += frozen_prompt(prepared)
    save_new(audit / "effective_workflow.json", {"upstream_commit": UPSTREAM_COMMIT,
        "variant": "SceneWeaver–FrozenAssets (restricted mutation set)", "native_max_steps": cls.max_steps,
        "native_llm_max_tokens": settings.max_tokens, "native_llm_temperature": settings.temperature,
        "native_llm_max_input_tokens": settings.max_input_tokens, "sdk_max_retries": client.max_retries,
        "public_object_plan_sha256": prepared["public_object_plan_sha256"],
        "helper_hashes": HELPER_HASHES, "disabled_finalization_stages": DISABLED_STAGES})
    try:
        result = cls().run(frozen_prompt(prepared))
        save_new(audit / "native_run_return.json", {"result": result})
    finally:
        archive.capture(run, "native_driver_return_or_failure")
        client.close()
    if not last_attempt_success:
        raise RuntimeError("native loop ended after a failed action; earlier layouts are preserved, not promoted to success")
    verify_sources(args.repo_path)
    from scene_weaver_frozen import _layouts, _observe_trajectory, _verify_plugin_report
    layouts = _layouts(run)
    rows = [read_json(run / "observations" / f"iteration_{i}.json") for i, _ in layouts]
    if not rows or any(row["native_room_observation"] != rows[0]["native_room_observation"] for row in rows):
        raise RuntimeError("native trajectory lacks consistent observed room geometry")
    report = {"variant": "SceneWeaver–FrozenAssets (restricted mutation set)",
        "benchmark_feedback_used_by_native_loop": False, "model_input_asset_locators_used": False,
        "public_object_plan_sha256": prepared["public_object_plan_sha256"],
        "model_input_public_object_plan_sha256": prepared["public_object_plan_sha256"],
        "native_room_observation": rows[0]["native_room_observation"],
        "iteration_asset_observations": rows,
        "resource_usage": {"model_identity_evidence": "observed_response", "model_identities": observed.identities,
            "model_deployment_id": deployment, "model_endpoint_sha256": endpoint, "generation_calls": observed.calls,
            "tokens": observed.tokens, "native_worker_calls": counter,
            "rendering_calls": None, "rendered_selected_states": len(rows)},
        "frozen_controls": {name: True for name in (
            "fixed_object_inventory", "exact_asset_ids", "fixed_native_scale", "frozen_iteration_bindings",
            "no_object_insertion_removal", "retrieval_disabled", "asset_replacement_disabled", "resize_disabled",
            "full_precision_local_bbox_observed", "full_precision_native_pose_observed", "native_object_dimensions_observed",
            "released_object_dimensions_export_preserved", "catalog_resolution_outside_model_context", "bottom_center_origin_rebased")}}
    report["frozen_controls"]["orientation_basis_policy"] = "bake_catalog_front_to_sceneweaver_positive_x"
    _verify_plugin_report(report, identity, deployment, endpoint, prepared["public_object_plan_sha256"])
    observation, _ = _observe_trajectory(layouts=layouts, control=read_json(args.comparison_input),
        catalog=read_json(args.comparison_catalog), request=read_json(args.request), plugin_report=report, tolerance=1e-6)
    if not observation["valid"]:
        raise RuntimeError(f"native trajectory violated frozen contract: {observation['violations']}")
    save_new(args.plugin_report, report)


def observe_state(state, solver, iteration, prepared, guard):
    """Read actual placeholder and populated geometry, never substitute catalog geometry."""
    import bpy
    from mathutils import Matrix
    from scene_weaver_frozen_assets import _bounds, local_vertex_digest
    from scene_weaver_frozen import _orientation_basis, _anchor_basis
    from infinigen_examples.steps.tools import calc_position_bias
    guard.assert_complete_initialization(state)
    bpy.context.view_layer.update()
    observed = {}
    for slot, binding in prepared["bindings"].items():
        native = state.objs[slot]
        placeholder = native.obj
        obj = bpy.data.objects.get(getattr(native, "populate_obj", ""))
        if obj is None:
            raise RuntimeError("native state lacks populated frozen mesh")
        guard.keep_population(obj, placeholder, "iteration_observer")
        if digest(binding["mesh_uri"]) != binding["mesh_sha256"] or local_vertex_digest(obj) != obj.get("frozen_vertex_sha256"):
            raise RuntimeError("native workflow changed frozen source or mesh vertices")
        if max(abs(obj.matrix_world[i][j]-placeholder.matrix_world[i][j]) for i in range(4) for j in range(4)) > 1e-6:
            raise RuntimeError("native placeholder and populated mesh pose disagree")
        basis = _orientation_basis(binding)
        inverse = Matrix.Rotation(-math.radians(basis["basis_yaw_degrees"]), 4, "Z")
        size, _ = _bounds(inverse @ v.co for v in obj.data.vertices)
        anchor = placeholder.location + calc_position_bias(placeholder)
        observed[slot] = {"asset_id": obj["frozen_asset_id"], "mesh_path": binding["mesh_uri"],
            "mesh_sha256": obj["frozen_mesh_sha256"], "canonical_local_bbox_size": size,
            "orientation_basis": basis, "anchor_basis": _anchor_basis(binding),
            "full_precision_native_bottom_center": list(anchor),
            "full_precision_native_object_dimensions": list(placeholder.dimensions),
            "full_precision_native_euler_xyz": list(placeholder.rotation_euler),
            "native_local_vertex_sha256": local_vertex_digest(obj)}
    dimensions = [float(v) for v in solver.dimensions]
    if len(dimensions) < 3:
        raise RuntimeError("native solver does not expose observed room height")
    room = {"roomsize": dimensions[:2], "height": dimensions[2], "unit": "meter"}
    expected = prepared["public"]["roomsize"] + [prepared["public"]["height"]]
    if any(abs(dimensions[i]-expected[i]) > 1e-6 for i in range(3)):
        raise RuntimeError("native room dimensions differ from frozen architecture")
    # Verify the actual native solidifier input contour, not only the dimensions
    # copied into the released export header. No bounding-rectangle replacement.
    room_states = [obj for name, obj in state.objs.items() if name.startswith("newroom")]
    if len(room_states) != 1:
        raise RuntimeError("native state is not a single room")
    contour = getattr(room_states[0], "contour", None)
    if contour is None or contour.geom_type != "Polygon" or len(contour.interiors):
        raise RuntimeError("native room lacks an observable simple contour")
    vertices = validate_rectangle(list(contour.exterior.coords), dimensions[:2])
    room["native_contour_vertices"] = vertices
    return {"iteration": iteration, "objects": observed, "native_room_observation": room}


def worker_environment():
    result = {k: v for k, v in os.environ.items() if not any(
        token in k.upper() for token in ("API_KEY", "TOKEN", "SECRET", "PASSWORD"))}
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def run_worker(args, prepared):
    """Launch the released Blender generation entry, with no model client in worker."""
    from scene_weaver_frozen_assets import make_frozen_factory
    from scene_weaver_frozen_mutations import FrozenMutationGuard, install_mutation_guards, PATCH_TARGETS
    sys.path[:0] = [str(args.repo_path), str(args.repo_path / "Pipeline")]
    os.environ.update(save_dir=str(args.native_run), ROOM_INFO=str(args.native_run / "roominfo.json"))
    os.chdir(args.repo_path)
    import gin
    constants = import_height_constants(args.repo_path, prepared["public"]["height"], args.attempt_dir)
    gin.parse_config(f"global_params.wall_height = {prepared['public']['height']!r}")
    constants.initialize_constants()
    from infinigen.core.placement.factory import AssetFactory
    from infinigen.core.constraints import usage_lookup
    from infinigen.core import tags as t
    factories = {}
    module = ModuleType("infinigen.assets.objects.layout_ddd_frozen")
    for slot, binding in prepared["bindings"].items():
        factory = make_frozen_factory(AssetFactory, slot, binding)
        factories[slot] = factory
        setattr(module, prepared["mapping"][slot].rsplit(".", 1)[1], factory)
    sys.modules[module.__name__] = module
    native_lookup_init = usage_lookup.initialize_from_dict

    def initialize_lookup(mapping):
        mapping = {tag: list(values) for tag, values in mapping.items()}
        mapping.setdefault(t.Semantics.SingleGenerator, [])
        for tag in (t.Semantics.Object, t.Semantics.Furniture, t.Semantics.RealPlaceholder):
            mapping.setdefault(tag, []).extend(factories.values())
        native_lookup_init(mapping)
    usage_lookup.initialize_from_dict = initialize_lookup
    guard = FrozenMutationGuard(prepared["bindings"], args.attempt_dir / "mutation_journal.jsonl")
    guard.configure_initialization(prepared["mapping"])
    load_overlay(args.repo_path, "infinigen_examples.steps.room_structure",
                 "infinigen_examples/steps/room_structure.py", room_ui_overlay, args.attempt_dir)
    modules = {path: importlib.import_module(path[:-3].replace("/", ".")) for path in PATCH_TARGETS}
    install_mutation_guards(args.repo_path, guard, modules)
    from infinigen_examples.steps import record
    native_record = record.record_scene

    def record_scene(state, solver, terrain, house_bbox, solved_bbox, camera_rigs, iter, p, transparent=False):
        guard.assert_complete_initialization(state)
        native_record(state, solver, terrain, house_bbox, solved_bbox, camera_rigs, iter, p, transparent=transparent)
        observation = observe_state(state, solver, iter, prepared, guard)
        # Working observation can be overwritten by native backtracking only
        # after the driver archived the previous layout AND observation bytes.
        (args.native_run / "observations" / f"iteration_{iter}.json").write_text(json.dumps(observation, indent=2))
    record.record_scene = record_scene
    native = modules["infinigen_examples/generate_indoors.py"]
    if args.worker_preflight:
        save_new(args.attempt_dir / "worker_preflight.json", {"status": "NATIVE_IMPORT_AND_OVERLAY_PASS",
            "upstream_commit": UPSTREAM_COMMIT, "slots": len(factories), "mutation_targets": len(build_worker_targets()),
            "model_calls": 0, "geometry_generated": False, "native_loop_executed": False})
        return
    action = read_json(args.native_run / "args.json")
    if action["action"] not in ALLOWED_ACTIONS:
        raise ValueError("worker action is outside FrozenAssets protocol")
    overrides = ["compose_indoors.terrain_enabled=False", "compose_indoors.invisible_room_ceilings_enabled=True",
                 f"global_params.wall_height={prepared['public']['height']!r}"]
    overrides += [f"compose_indoors.{name}_enabled=False" for name in DISABLED_STAGES]
    output = args.attempt_dir / "native_output"
    output.mkdir()
    native.main(SimpleNamespace(**{k: action[k] for k in ("iter", "action", "json_name", "description", "inplace")},
        seed="0", configs=["fast_solve.gin", "overhead.gin", "studio.gin"], overrides=overrides,
        input_folder=None, output_folder=output, task=["coarse"], task_uniqname=None))
    verify_sources(args.repo_path)


def build_worker_targets():
    from scene_weaver_frozen_mutations import PATCH_TARGETS
    return [target for targets in PATCH_TARGETS.values() for target in targets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--method-input", type=Path)
    parser.add_argument("--comparison-input", type=Path)
    parser.add_argument("--comparison-catalog", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--plugin-report", type=Path)
    parser.add_argument("--worker-python", default=os.environ.get("LAYOUT_DDD_SCENEWEAVER_PYTHON", sys.executable))
    parser.add_argument("--preflight-only", action="store_true", help="No imports of upstream, model client, renderer or generation")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-preflight", action="store_true")
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--native-run", type=Path)
    parser.add_argument("--attempt-dir", type=Path)
    args = parser.parse_args()
    args.repo_path = args.repo_path.resolve()
    sys.dont_write_bytecode = True
    plan = verify_sources(args.repo_path)
    if args.worker:
        if args.preflight_only or not all((args.worker_input, args.native_run, args.attempt_dir)):
            parser.error("worker requires worker-input/native-run/attempt-dir")
        run_worker(args, read_json(args.worker_input))
        return
    if not all((args.request, args.method_input, args.comparison_input, args.comparison_catalog, args.output_root, args.plugin_report)):
        parser.error("driver requires request/method-input/comparison-input/comparison-catalog/output-root/plugin-report")
    prepared = prepare_input(read_json(args.request), read_json(args.comparison_catalog))
    from _common import verify_catalog_contract
    verify_catalog_contract(read_json(args.comparison_input), read_json(args.comparison_catalog))
    args.worker_python = shutil.which(args.worker_python) or args.worker_python
    if not Path(args.worker_python).is_file():
        raise FileNotFoundError("configured SceneWeaver worker executable missing")
    if args.preflight_only:
        save_new(args.plugin_report, {"status": "STATIC_PREFLIGHT_ONLY", "model_calls": 0,
            "upstream_imported": False, "upstream_commit": UPSTREAM_COMMIT,
            "plugin_entrypoint_sha256": digest(Path(__file__)),
            "input_file_sha256": {name: digest(getattr(args, name)) for name in
                                  ("request", "method_input", "comparison_input", "comparison_catalog")},
            "helper_hashes": HELPER_HASHES, "mutation_targets": len(plan),
            "slots": len(prepared["bindings"]), "public_input_sha256": logical_hash(prepared["public"]),
            "remaining": ["native_environment_qualification", "real_model_route_smoke", "full_native_loop_and_evaluator_smoke"]})
        return
    run_driver(args, prepared)


if __name__ == "__main__":
    main()
