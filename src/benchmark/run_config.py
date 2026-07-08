from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.utils.io import load_json_schema, load_yaml, write_json


COMPONENT_DIRS = {
    "dataset": "datasets",
    "model": "models",
    "inference": "inference",
    "evaluation": "evaluation",
    "render": "render",
    "grouping": "grouping",
    "repair": "repair",
    "viewer": "viewer",
}
COMPONENT_KINDS = tuple(COMPONENT_DIRS)
MODEL_OVERRIDE_ALIASES = {"model_id": "model", "model_endpoint": "endpoint"}


@dataclass(frozen=True)
class ResolvedRunConfig:
    data: dict

    @property
    def model_name(self) -> str:
        return str(self.data["model"]["name"])

    @property
    def model_config(self) -> dict:
        return deepcopy(self.data["model_config"])

    @property
    def benchmark_config(self) -> dict:
        return deepcopy(self.data["benchmark_config"])

    @property
    def dataset_config(self) -> dict:
        return deepcopy(self.data["dataset"])


def load_resolved_run_config(
    project_root: str | Path,
    *,
    experiment_config_path: str | Path,
    experiment_name: str | None = None,
    model_overrides: dict | None = None,
) -> ResolvedRunConfig:
    root = Path(project_root)
    experiment_path = _resolve_path(root, experiment_config_path)
    experiment = _load_experiment_file(experiment_path, experiment_name)
    defaults = experiment.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError(f"Experiment config {experiment_path} requires a defaults mapping.")

    components: dict[str, dict] = {}
    refs: dict[str, str] = {"experiment": _relative(root, experiment_path)}
    for kind in COMPONENT_KINDS:
        ref = defaults.get(kind)
        if not ref:
            raise ValueError(f"Experiment config {experiment_path} requires defaults.{kind}.")
        component_path = _component_path(root, kind, ref)
        components[kind] = _component_body(load_yaml(component_path, default={}), kind)
        refs[kind] = _relative(root, component_path)

    for kind in COMPONENT_KINDS:
        if isinstance(experiment.get(kind), dict):
            components[kind] = _deep_merge(components[kind], experiment[kind])

    if isinstance(experiment.get("model_overrides"), dict):
        components["model"] = _apply_generation_overrides(components["model"], experiment["model_overrides"])
    if model_overrides:
        components["model"] = _apply_generation_overrides(components["model"], model_overrides)

    benchmark_config = _build_benchmark_config(root, components, refs)
    if experiment.get("max_repair_iterations") is not None:
        benchmark_config.setdefault("benchmark", {})["max_repair_iterations"] = int(experiment["max_repair_iterations"])
    model_config = _build_model_config(root, components)
    resolved_body = {
        "experiment_name": experiment.get("name") or experiment_name or experiment_path.stem,
        "description": experiment.get("description", ""),
        "out": experiment.get("out"),
        "max_repair_iterations": experiment.get("max_repair_iterations"),
        "judge_model": experiment.get("judge_model", components["model"].get("judge_model", "same")),
        "config_refs": refs,
        "dataset": components["dataset"],
        "model": _resolved_model_component(components["model"], components["inference"]),
        "inference": components["inference"],
        "evaluation": components["evaluation"],
        "render": components["render"],
        "grouping": components["grouping"],
        "repair": components["repair"],
        "viewer": components["viewer"],
        "model_config": model_config,
        "benchmark_config": benchmark_config,
    }
    config_hash = config_hash_for(resolved_body)
    resolved_body["config_hash"] = config_hash
    resolved_body["benchmark_config"]["config_refs"] = refs
    resolved_body["benchmark_config"]["config_hash"] = config_hash
    _validate_resolved(resolved_body)
    return ResolvedRunConfig(resolved_body)


def save_resolved_run_config(out_dir: str | Path, resolved: ResolvedRunConfig) -> tuple[Path, Path]:
    out = Path(out_dir)
    config_path = write_json(out / "resolved_run_config.json", resolved.data)
    hash_path = out / "config_hash.txt"
    hash_path.write_text(str(resolved.data["config_hash"]) + "\n", encoding="utf-8")
    return config_path, hash_path


def pipeline_resources_from_resolved(project_root: str | Path, resolved: ResolvedRunConfig) -> Any:
    from benchmark.pipeline import PipelineResources

    root = Path(project_root)
    return PipelineResources(
        model_config=resolved.model_config,
        benchmark_config=resolved.benchmark_config,
        layout_schema=load_json_schema(_legend_layout_schema_path(root)),
        scene_schema=load_json_schema(root / "schemas" / "scene.schema.json"),
        resolved_run_config=deepcopy(resolved.data),
    )


def _legend_layout_schema_path(root: Path) -> Path:
    legend_path = root / "legend" / "schemas" / "legend_layout.schema.json"
    return legend_path if legend_path.exists() else root / "schemas" / "layout.schema.json"


def config_hash_for(data: dict) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _load_experiment_file(path: Path, experiment_name: str | None) -> dict:
    loaded = load_yaml(path, default={})
    if not isinstance(loaded, dict):
        raise ValueError(f"Experiment config {path} must be a mapping.")
    if "experiments" in loaded:
        if not experiment_name:
            raise ValueError(f"Experiment file {path} contains multiple experiments; provide --experiment.")
        selected = loaded.get("experiments", {}).get(experiment_name)
        if not isinstance(selected, dict):
            raise ValueError(f"Unknown experiment '{experiment_name}' in {path}.")
        return {"name": experiment_name, **selected}
    return {"name": loaded.get("name") or path.stem, **loaded}


def _component_body(loaded: Any, kind: str) -> dict:
    if not isinstance(loaded, dict):
        raise ValueError(f"{kind} component must be a mapping.")
    return deepcopy(loaded)


def _component_path(root: Path, kind: str, ref: object) -> Path:
    text = str(ref)
    candidate = Path(text)
    if candidate.suffix in {".yaml", ".yml"} or "/" in text or "\\" in text:
        return _resolve_path(root, candidate)
    return root / "configs" / COMPONENT_DIRS[kind] / f"{text}.yaml"


def _resolve_path(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return path.as_posix()


def _build_model_config(root: Path, components: dict[str, dict]) -> dict:
    model = _resolved_model_component(components["model"], components["inference"])
    name = str(model.get("name") or model.get("model_name") or model.get("model") or model.get("provider"))
    if not name:
        raise ValueError("Model component requires name.")
    model_def = {
        key: value
        for key, value in model.items()
        if key not in {"generation", "judge", "model_name"}
    }
    model_def.setdefault("name", name)
    model_def.setdefault("provider", model.get("provider"))
    generation = model.get("generation")
    if isinstance(generation, dict):
        for key, value in generation.items():
            model_def[MODEL_OVERRIDE_ALIASES.get(key, key)] = value
    if "model_id" in model_def and "model" not in model_def:
        model_def["model"] = model_def["model_id"]
    api = load_yaml(root / "configs" / "model_config.yaml", default={}).get("api", {})
    return {
        "default_model": name,
        "api": api if isinstance(api, dict) else {},
        "judge": {"model": components["model"].get("judge_model", "same")},
        "models": {name: model_def},
    }


def _resolved_model_component(model: dict, inference: dict) -> dict:
    resolved = deepcopy(model)
    for key in ["endpoint", "runtime_profile", "context_limit_tokens"]:
        if key in inference and resolved.get(key) in {None, ""}:
            resolved[key] = inference[key]
        elif key in inference and key not in resolved:
            resolved[key] = inference[key]
    if "model_id" in resolved and "model" not in resolved:
        resolved["model"] = resolved["model_id"]
    return resolved


def _build_benchmark_config(root: Path, components: dict[str, dict], refs: dict[str, str]) -> dict:
    config = load_yaml(root / "configs" / "benchmark_config.yaml", default={})
    if not isinstance(config, dict):
        config = {}
    for kind in ["evaluation", "render", "grouping", "repair", "viewer"]:
        config = _deep_merge(config, _as_benchmark_patch(kind, components[kind]))
    config.setdefault("config_refs", refs)
    return config


def _as_benchmark_patch(kind: str, component: dict) -> dict:
    if any(key in component for key in ["benchmark", "evaluation", "evaluation_policy", "vlm_judge", "render", "grouping", "view_validation", "repair", "viewer"]):
        return deepcopy(component)
    if kind == "render":
        return {"render": deepcopy(component)}
    if kind == "grouping":
        return {"grouping": deepcopy(component)}
    if kind == "evaluation":
        return {"evaluation": deepcopy(component)}
    if kind == "repair":
        return {"repair": deepcopy(component)}
    if kind == "viewer":
        return {"viewer": deepcopy(component)}
    return deepcopy(component)


def _apply_generation_overrides(model: dict, overrides: dict) -> dict:
    resolved = deepcopy(model)
    generation = dict(resolved.get("generation") or {})
    for key, value in overrides.items():
        if value is None:
            continue
        config_key = MODEL_OVERRIDE_ALIASES.get(key, key)
        if config_key == "endpoint":
            resolved["endpoint"] = value
        elif config_key == "model":
            resolved["model"] = value
            resolved["model_id"] = value
        else:
            generation[config_key] = value
    resolved["generation"] = generation
    return resolved


def _deep_merge(base: dict, patch: dict) -> dict:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _validate_resolved(resolved: dict) -> None:
    dataset = resolved.get("dataset", {})
    if not dataset.get("source_type") and not dataset.get("adapter"):
        raise ValueError("Resolved dataset config requires source_type.")
    model = resolved.get("model", {})
    provider = model.get("provider")
    if not provider:
        raise ValueError("Resolved model config requires provider.")
    if provider in {"openai_compatible", "vllm"}:
        if not model.get("endpoint"):
            raise ValueError("OpenAI-compatible model config requires endpoint.")
        if not (model.get("model") or model.get("model_id")):
            raise ValueError("OpenAI-compatible model config requires model/model_id.")
