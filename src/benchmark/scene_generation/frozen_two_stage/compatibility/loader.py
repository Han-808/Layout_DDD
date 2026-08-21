"""Safe loader for the existing frozen two-stage core bundle.

The migration contract is documented in
``docs/generation_transport_compatibility.md``.  The loader fixes the module
filename to ``generation_runner.py`` and creates a synthetic package so legacy
relative imports work without mutating ``sys.path``.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


_REQUIRED_CORE_ATTRIBUTES = (
    "DEFAULT_STAGE_A_PROMPT",
    "DEFAULT_STAGE_C_PROMPT",
    "RetrieverAdapter",
    "_load_briefs",
    "_load_model_config",
    "_runner_source_manifest",
    "initialize_run",
    "run_case",
    "utc_now",
    "write_json_exclusive",
)


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Credential-free model fields needed by static config validation."""

    key: str
    label: str
    endpoint: str
    configured_model: str
    wire_model: str
    timeout_seconds: float
    max_infrastructure_retries: int
    retry_delay_seconds: float
    temperature: float | None
    top_p: float | None
    top_k: int | None
    max_tokens: int
    repetition_penalty: float | None
    reasoning_effort: str | None
    preserved_thinking: bool | None
    strategy_type: str


@dataclass(frozen=True, slots=True)
class StaticCoreMetadata:
    """Core identity obtained from source text without executing the core."""

    root: Path
    runner_path: Path
    runner_sha256: str
    runner_version: str


@dataclass(frozen=True, slots=True)
class FrozenCoreRuntimeInputs:
    """Runtime objects loaded only for an actual generation run."""

    model: Any
    briefs: tuple[Mapping[str, Any], ...]
    retriever: Any


def load_frozen_core(core_root: str | Path) -> ModuleType:
    """Load one exact frozen core as an isolated synthetic package."""

    root = Path(core_root).expanduser().resolve()
    init_path = root / "__init__.py"
    runner_path = root / "generation_runner.py"
    if not root.is_dir() or not init_path.is_file() or not runner_path.is_file():
        raise FileNotFoundError(
            "frozen core must contain __init__.py and generation_runner.py: "
            f"{root}"
        )
    identity_hash = hashlib.sha256(str(root).encode("utf-8"))
    for source_path in sorted(root.glob("*.py")):
        identity_hash.update(b"\0")
        identity_hash.update(source_path.name.encode("utf-8"))
        identity_hash.update(b"\0")
        identity_hash.update(source_path.read_bytes())
    identity = identity_hash.hexdigest()[:24]
    package_name = f"_layout_ddd_frozen_core_{identity}"
    module_name = f"{package_name}.generation_runner"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return _validate_core(existing)

    package_spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(root)],
    )
    if package_spec is None or package_spec.loader is None:
        raise ImportError(f"cannot create frozen-core package spec: {root}")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[package_name] = package
    try:
        package_spec.loader.exec_module(package)
        module = importlib.import_module(module_name)
        return _validate_core(module)
    except Exception:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        raise


def _validate_core(module: ModuleType) -> ModuleType:
    missing = [
        attribute
        for attribute in _REQUIRED_CORE_ATTRIBUTES
        if not hasattr(module, attribute)
    ]
    if missing:
        raise TypeError(f"frozen core is missing required attributes: {missing}")
    return module


def inspect_core_metadata(core_root: str | Path) -> StaticCoreMetadata:
    """Parse the frozen runner AST without importing or executing it."""

    root = Path(core_root).expanduser().resolve()
    init_path = root / "__init__.py"
    runner_path = root / "generation_runner.py"
    if not root.is_dir() or not init_path.is_file() or not runner_path.is_file():
        raise FileNotFoundError(
            "frozen core must contain __init__.py and generation_runner.py: "
            f"{root}"
        )
    source = runner_path.read_bytes()
    try:
        tree = ast.parse(source, filename=str(runner_path))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid frozen-core Python source: {exc}") from exc
    runner_versions: list[str] = []
    functions: set[str] = set()
    classes: set[str] = set()
    assignments: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.add(target.id)
                    if target.id == "RUNNER_VERSION" and isinstance(
                        node.value, ast.Constant
                    ) and isinstance(node.value.value, str):
                        runner_versions.append(node.value.value)
    if len(runner_versions) != 1 or not runner_versions[0]:
        raise ValueError("frozen core must declare one literal RUNNER_VERSION")
    required_functions = {
        "_load_briefs",
        "_load_model_config",
        "_runner_source_manifest",
        "initialize_run",
        "run_case",
        "utc_now",
    }
    missing_functions = required_functions - functions
    if missing_functions:
        raise ValueError(
            f"frozen core source is missing functions: {sorted(missing_functions)}"
        )
    if "RetrieverAdapter" not in classes:
        raise ValueError("frozen core source is missing RetrieverAdapter")
    missing_prompts = {
        "DEFAULT_STAGE_A_PROMPT",
        "DEFAULT_STAGE_C_PROMPT",
    } - assignments
    if missing_prompts:
        raise ValueError(
            f"frozen core source is missing prompt paths: {sorted(missing_prompts)}"
        )
    return StaticCoreMetadata(
        root=root,
        runner_path=runner_path,
        runner_sha256=hashlib.sha256(source).hexdigest(),
        runner_version=runner_versions[0],
    )


def _load_plain_json(path: Path) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"duplicate model-config JSON key: {key}")
            value[key] = child
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite model-config number is not allowed: {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid model-config JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("model config must be a JSON object")
    return value


def inspect_model_metadata(models_path: str | Path, model_key: str) -> ModelMetadata:
    """Validate model metadata without reading a credential environment value."""

    path = Path(models_path).expanduser().resolve()
    value = _load_plain_json(path)
    if value.get("schema_version") != "hy34_model_transport_config_v1":
        raise ValueError("unsupported model config schema")
    endpoint = value.get("endpoint")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("model config endpoint must be non-empty")
    api_key_env = value.get("api_key_env")
    if not isinstance(api_key_env, str) or not api_key_env:
        raise ValueError("model config must name a credential environment variable")
    models = value.get("models")
    if not isinstance(models, dict) or model_key not in models:
        raise ValueError(f"unknown model key {model_key!r}")
    model = models[model_key]
    request = value.get("request")
    if not isinstance(model, dict) or not isinstance(request, dict):
        raise ValueError("model config has invalid models/request objects")
    label = model.get("label")
    configured_model = model.get("configured_model")
    wire_model = model.get("wire_model")
    if not isinstance(label, str) or not label:
        raise ValueError("model label must be non-empty")
    if not isinstance(wire_model, str) or not wire_model:
        raise ValueError("model wire_model must be non-empty")
    if not isinstance(configured_model, str) or not configured_model:
        raise ValueError("model configured_model must be non-empty")
    required_request_fields = frozenset(
        {
            "timeout_seconds",
            "max_infrastructure_retries",
            "retry_delay_seconds",
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "repetition_penalty",
            "reasoning_effort",
            "preserved_thinking",
            "strategy_type",
        }
    )
    missing_request_fields = required_request_fields - request.keys()
    if missing_request_fields:
        raise ValueError(
            "model request is missing fields: "
            f"{sorted(missing_request_fields)}"
        )
    retries = request.get("max_infrastructure_retries")
    delay = request.get("retry_delay_seconds")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ValueError("model max_infrastructure_retries must be non-negative")
    if (
        isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not math.isfinite(float(delay))
        or delay < 0
    ):
        raise ValueError("model retry_delay_seconds must be non-negative")
    timeout = request["timeout_seconds"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("model timeout_seconds must be finite and positive")
    max_tokens = request["max_tokens"]
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ValueError("model max_tokens must be a positive integer")
    for field_name in ("temperature", "top_p", "repetition_penalty"):
        field_value = request[field_name]
        if field_value is not None and (
            isinstance(field_value, bool)
            or not isinstance(field_value, (int, float))
            or not math.isfinite(float(field_value))
        ):
            raise ValueError(f"model {field_name} must be finite or null")
    top_k = request["top_k"]
    if top_k is not None and (
        isinstance(top_k, bool) or not isinstance(top_k, int)
    ):
        raise ValueError("model top_k must be an integer or null")
    effort = request["reasoning_effort"]
    if effort is not None and (not isinstance(effort, str) or not effort):
        raise ValueError("model reasoning_effort must be a string or null")
    preserved = request["preserved_thinking"]
    if preserved is not None and not isinstance(preserved, bool):
        raise ValueError("model preserved_thinking must be a boolean or null")
    strategy = request["strategy_type"]
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("model strategy_type must be non-empty")
    return ModelMetadata(
        key=model_key,
        label=label,
        endpoint=endpoint,
        configured_model=configured_model,
        wire_model=wire_model,
        timeout_seconds=float(timeout),
        max_infrastructure_retries=retries,
        retry_delay_seconds=float(delay),
        temperature=(
            None if request["temperature"] is None else float(request["temperature"])
        ),
        top_p=None if request["top_p"] is None else float(request["top_p"]),
        top_k=top_k,
        max_tokens=max_tokens,
        repetition_penalty=(
            None
            if request["repetition_penalty"] is None
            else float(request["repetition_penalty"])
        ),
        reasoning_effort=effort,
        preserved_thinking=preserved,
        strategy_type=strategy,
    )


def inspect_brief_ids(briefs_path: str | Path) -> tuple[str, ...]:
    """Read paired brief identities from strict JSON without executing core code."""

    path = Path(briefs_path).expanduser().resolve()
    value = _load_plain_json(path)
    if value.get("schema_version") != "hy34_paired_briefs_v1":
        raise ValueError("unsupported paired briefs schema")
    briefs = value.get("briefs")
    if not isinstance(briefs, list) or not briefs:
        raise ValueError("paired briefs must be a non-empty array")
    ids: list[str] = []
    for index, brief in enumerate(briefs):
        if not isinstance(brief, dict):
            raise ValueError(f"briefs[{index}] must be an object")
        brief_id = brief.get("brief_id")
        if not isinstance(brief_id, str) or not brief_id:
            raise ValueError(f"briefs[{index}].brief_id is invalid")
        ids.append(brief_id)
    if len(ids) != len(set(ids)):
        raise ValueError("paired briefs contain duplicate brief IDs")
    return tuple(ids)


def select_static_brief_ids(
    available_brief_ids: Sequence[str],
    ordered_brief_ids: Sequence[str],
) -> tuple[str, ...]:
    """Validate an ordered subset using only static brief identities."""

    available = tuple(available_brief_ids)
    requested = tuple(ordered_brief_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("ordered_brief_ids must not contain duplicates")
    available_set = set(available)
    missing = [brief_id for brief_id in requested if brief_id not in available_set]
    if missing:
        raise ValueError(f"unknown selected brief IDs: {missing}")
    return requested


def validate_runtime_consistency(
    *,
    core: ModuleType,
    static_core: StaticCoreMetadata,
    model: Any,
    static_model: ModelMetadata,
    briefs: Sequence[Mapping[str, Any]],
    static_brief_ids: Sequence[str],
) -> None:
    """Ensure credential-bearing runtime values equal the static trusted view."""

    if str(getattr(core, "RUNNER_VERSION", "")) != static_core.runner_version:
        raise ValueError("runtime core version differs from static trusted metadata")
    current_runner_hash = hashlib.sha256(
        static_core.runner_path.read_bytes()
    ).hexdigest()
    if current_runner_hash != static_core.runner_sha256:
        raise ValueError("runtime core source changed after static trust validation")
    expected_model = {
        "key": static_model.key,
        "label": static_model.label,
        "endpoint": static_model.endpoint,
        "configured_model": static_model.configured_model,
        "wire_model": static_model.wire_model,
        "timeout_seconds": static_model.timeout_seconds,
        "max_infrastructure_retries": static_model.max_infrastructure_retries,
        "retry_delay_seconds": static_model.retry_delay_seconds,
        "temperature": static_model.temperature,
        "top_p": static_model.top_p,
        "top_k": static_model.top_k,
        "max_tokens": static_model.max_tokens,
        "repetition_penalty": static_model.repetition_penalty,
        "reasoning_effort": static_model.reasoning_effort,
        "preserved_thinking": static_model.preserved_thinking,
        "strategy_type": static_model.strategy_type,
    }
    actual_model = {
        key: getattr(model, key, None) for key in expected_model
    }
    if actual_model != expected_model:
        mismatched = sorted(
            key for key in expected_model if actual_model[key] != expected_model[key]
        )
        raise ValueError(
            "runtime model differs from static trusted metadata in fields: "
            f"{mismatched}"
        )
    actual_brief_ids = tuple(str(brief.get("brief_id")) for brief in briefs)
    if actual_brief_ids != tuple(static_brief_ids):
        raise ValueError(
            "runtime brief order differs from static trusted metadata: "
            f"expected={tuple(static_brief_ids)}, actual={actual_brief_ids}"
        )


def load_selected_briefs(
    core: ModuleType,
    briefs_path: str | Path,
    ordered_brief_ids: Sequence[str],
) -> tuple[Mapping[str, Any], ...]:
    """Select frozen briefs in caller order and reject unknown/duplicate IDs."""

    briefs = core._load_briefs(Path(briefs_path))
    if not isinstance(briefs, list):
        raise TypeError("frozen core _load_briefs must return a list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for brief in briefs:
        if not isinstance(brief, Mapping):
            raise TypeError("frozen core returned a non-mapping brief")
        brief_id = brief.get("brief_id")
        if not isinstance(brief_id, str) or not brief_id:
            raise ValueError("frozen core returned a brief without brief_id")
        if brief_id in by_id:
            raise ValueError(f"frozen core returned duplicate brief {brief_id!r}")
        by_id[brief_id] = brief
    requested = tuple(ordered_brief_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("ordered_brief_ids must not contain duplicates")
    missing = [brief_id for brief_id in requested if brief_id not in by_id]
    if missing:
        raise ValueError(f"unknown selected brief IDs: {missing}")
    return tuple(by_id[brief_id] for brief_id in requested)


def load_runtime_inputs(
    core: ModuleType,
    *,
    models_path: str | Path,
    model_key: str,
    briefs_path: str | Path,
    ordered_brief_ids: Sequence[str],
    retriever_root: str | Path,
) -> FrozenCoreRuntimeInputs:
    """Load credential-bearing model and retriever only for a real run."""

    briefs = load_selected_briefs(core, briefs_path, ordered_brief_ids)
    model = core._load_model_config(Path(models_path), model_key)
    retriever = core.RetrieverAdapter(Path(retriever_root))
    return FrozenCoreRuntimeInputs(model=model, briefs=briefs, retriever=retriever)
