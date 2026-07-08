from __future__ import annotations

import shutil
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

from benchmark.data import iter_case_paths
from benchmark.data.scene_adapters import normalize_scene, scene_to_case, scene_to_layout
from benchmark.feedback import build_feedback
from benchmark.metrics.aggregate import aggregate_case_results, write_benchmark_metrics_outputs
from benchmark.models import create_model
from benchmark.input_modes import canonicalize_input_mode
from benchmark.utils.io import load_json_schema, load_yaml, read_json
from benchmark.utils.io import write_json
from benchmark.visualization import export_viewer_scene
from benchmark.workflow import BenchmarkAgent
from benchmark.workflow.artifacts import configured_max_repair_iterations, output_settings, save_viewer_scene
from benchmark.workflow.judge_evidence_selector import judge_generation_overrides
from benchmark.workflow.evaluate import evaluate_scene
from benchmark.workflow.state import BenchmarkState
from benchmark.workflow.vlm_judge import VLM_JUDGE_INPUT_JSON_ONLY, resolve_vlm_judge_input_mode


MODEL_OVERRIDE_KEYS = {
    "model_id": "model",
    "model_endpoint": "endpoint",
}
SAME_MODEL = "same"


@dataclass(frozen=True)
class PipelineResources:
    model_config: dict
    benchmark_config: dict
    layout_schema: dict
    scene_schema: dict | None = None
    resolved_run_config: dict | None = None


def load_pipeline_resources(project_root: str | Path) -> PipelineResources:
    root = Path(project_root)
    return PipelineResources(
        model_config=load_yaml(root / "configs" / "model_config.yaml", default={}),
        benchmark_config=load_yaml(root / "configs" / "benchmark_config.yaml", default={}),
        layout_schema=load_json_schema(_legend_layout_schema_path(root)),
        scene_schema=load_json_schema(root / "schemas" / "scene.schema.json"),
    )


def run_case_pipeline(
    *,
    case_path: str | Path,
    out_dir: str | Path,
    model_name: str = "mock",
    resources: PipelineResources,
    max_repair_iterations: int | None = None,
    mock_behavior: str | None = None,
    model_overrides: dict | None = None,
    judge_model_name: str | None = None,
    input_json: dict | None = None,
) -> BenchmarkState:
    model_config = deepcopy(resources.model_config)
    if mock_behavior:
        model_config.setdefault("models", {}).setdefault("mock", {})["behavior"] = mock_behavior
    apply_model_overrides(model_config, model_name, model_overrides)

    model = create_model(model_name, model_config)
    resolved_judge_model_name, judge_model = create_judge_model(
        model_name=model_name,
        model=model,
        model_config=model_config,
        judge_model_name=judge_model_name,
        benchmark_config=resources.benchmark_config,
    )
    repair_budget = (
        int(max_repair_iterations)
        if max_repair_iterations is not None
        else configured_max_repair_iterations(resources.benchmark_config)
    )
    benchmark_config = deepcopy(resources.benchmark_config)
    resolved_run_config = deepcopy(resources.resolved_run_config) if resources.resolved_run_config else None
    if resolved_run_config:
        benchmark_config.setdefault("config_refs", resolved_run_config.get("config_refs", {}))
        benchmark_config.setdefault("config_hash", resolved_run_config.get("config_hash", ""))

    return BenchmarkAgent().run(
        {
            "case_path": str(Path(case_path)),
            "out_dir": str(Path(out_dir)),
            **({"input_json": input_json} if input_json is not None else {}),
            "model": model,
            "model_name": model_name,
            "judge_model": judge_model,
            "judge_model_name": resolved_judge_model_name,
            "layout_schema": resources.layout_schema,
            "benchmark_config": benchmark_config,
            "resolved_run_config": resolved_run_config,
            "max_repair_iterations": repair_budget,
            "pipeline_mode": "generation",
            "generation_used": True,
        }
    )


def evaluate_scene_pipeline(
    *,
    scene_path: str | Path,
    out_dir: str | Path,
    resources: PipelineResources,
    judge_model_name: str | None = None,
    model_overrides: dict | None = None,
    vlm_judge_input_mode: str = VLM_JUDGE_INPUT_JSON_ONLY,
) -> BenchmarkState:
    """Evaluate a canonical scene JSON directly, without layout generation."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_input = read_json(scene_path)
    scene_schema = resources.scene_schema or _default_scene_schema()
    input_schema_type = _validate_scene_or_layout_or_raise(scene_input, scene_schema, resources.layout_schema, scene_path)

    scene = normalize_scene(scene_input)
    normalized_scene_path = output_dir / "normalized_scene.json"
    write_json(normalized_scene_path, scene)
    layout = scene_to_layout(scene)
    case = scene_to_case(scene)
    scene_schema_version = _scene_schema_version(scene_schema, scene)
    benchmark_config = _benchmark_config_with_vlm_judge_input_mode(resources.benchmark_config, vlm_judge_input_mode)
    resolved_mode = resolve_vlm_judge_input_mode(benchmark_config, vlm_judge_input_mode)
    model_config = deepcopy(resources.model_config)
    resolved_judge_model_name, judge_model = create_direct_judge_model(
        model_config=model_config,
        judge_model_name=judge_model_name,
        model_overrides=model_overrides,
        benchmark_config=benchmark_config,
    )
    resolved_run_config = deepcopy(resources.resolved_run_config) if resources.resolved_run_config else None
    if resolved_run_config:
        benchmark_config.setdefault("config_refs", resolved_run_config.get("config_refs", {}))
        benchmark_config.setdefault("config_hash", resolved_run_config.get("config_hash", ""))

    report, case_metrics = evaluate_scene(
        scene,
        case=case,
        out_dir=output_dir,
        model_name="scene_evaluation",
        benchmark_config=benchmark_config,
        layout_schema=resources.layout_schema,
        iteration=0,
        judge_model=judge_model,
        judge_model_name=resolved_judge_model_name,
        mode=resolved_mode,
    )
    report["pipeline_mode"] = "evaluation"
    report["generation_used"] = False
    report["scene_schema_version"] = scene_schema_version
    report["scene_schema_valid"] = input_schema_type == "scene"
    report["input_schema_type"] = input_schema_type
    report["input_schema_valid"] = True
    report["normalized_scene_path"] = "normalized_scene.json"
    report["evaluation_report_path"] = "evaluation_report.json"
    report["feedback_path"] = "feedback.json"
    report["case_metrics_path"] = "case_metrics.json"

    feedback = build_feedback(report, layout, case, benchmark_config=benchmark_config)
    feedback_issue_count = len(feedback.get("issues", [])) if isinstance(feedback.get("issues"), list) else 0
    feedback_action_count = len(feedback.get("suggested_actions", [])) if isinstance(feedback.get("suggested_actions"), list) else 0
    case_metrics.update(
        {
            "pipeline_mode": "evaluation",
            "generation_used": False,
            "scene_schema_version": scene_schema_version,
            "scene_schema_valid": input_schema_type == "scene",
            "input_schema_type": input_schema_type,
            "input_schema_valid": True,
            "normalized_scene_path": "normalized_scene.json",
            "evaluation_report_path": "evaluation_report.json",
            "feedback_path": "feedback.json",
            "case_metrics_path": "case_metrics.json",
            "feedback_issue_count": feedback_issue_count,
            "feedback_suggested_action_count": feedback_action_count,
        }
    )
    report["feedback_issue_count"] = feedback_issue_count
    report["feedback_suggested_action_count"] = feedback_action_count
    if isinstance(report.get("metrics"), dict):
        report["metrics"].update(case_metrics)

    evaluation_report_path = output_dir / "evaluation_report.json"
    case_metrics_path = output_dir / "case_metrics.json"
    write_json(evaluation_report_path, report)
    write_json(case_metrics_path, case_metrics)

    feedback_path = output_dir / "feedback.json"
    write_json(feedback_path, feedback)

    state: BenchmarkState = {
        "task_id": str(case.get("task_id") or case.get("case_id") or scene.get("scene_id")),
        "case_path": str(scene_path),
        "input_scene_path": str(scene_path),
        "out_dir": str(output_dir),
        "input_json": case,
        "current_scene": scene,
        "current_scene_path": str(normalized_scene_path),
        "normalized_scene_path": str(normalized_scene_path),
        "input_schema_type": input_schema_type,
        "current_layout": layout,
        "current_layout_path": "",
        "current_evaluation": report,
        "current_evaluation_path": str(evaluation_report_path),
        "case_metrics": case_metrics,
        "case_metrics_path": str(case_metrics_path),
        "current_case_metrics_path": str(output_dir / "case_metrics_iter_0.json"),
        "current_feedback": feedback,
        "current_feedback_path": str(feedback_path),
        "judge_model": judge_model,
        "judge_model_name": resolved_judge_model_name,
        "benchmark_config": benchmark_config,
        "layout_schema": resources.layout_schema,
        "resolved_run_config": resolved_run_config,
        "iteration": 0,
        "history": [],
        "evaluation_reports": [report],
        "pipeline_mode": "evaluation",
        "generation_used": False,
    }

    if save_viewer_scene(benchmark_config) and layout.get("objects"):
        viewer_scene = export_viewer_scene(case, layout, report, [], benchmark_config)
        viewer_scene["pipeline_mode"] = "evaluation"
        viewer_scene["generation_used"] = False
        viewer_scene["feedback"] = feedback
        viewer_scene["metrics"] = case_metrics
        viewer_scene["metrics_summary"] = case_metrics
        viewer_scene_path = output_dir / "viewer_scene.json"
        write_json(viewer_scene_path, viewer_scene)
        state["viewer_scene"] = viewer_scene
        state["viewer_scene_path"] = str(viewer_scene_path)

    return state


def run_benchmark_pipeline(
    *,
    cases_dir: str | Path,
    out_dir: str | Path,
    model_name: str = "mock",
    resources: PipelineResources,
    max_repair_iterations: int | None = None,
    mock_behavior: str | None = None,
    model_overrides: dict | None = None,
    judge_model_name: str | None = None,
    cases: list[tuple[Path, dict]] | None = None,
    input_modes: list[str] | None = None,
) -> tuple[dict, Path, Path]:
    root_out = Path(out_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    case_metrics = []
    discovered_cases = cases or [(Path(case_path), None) for case_path in iter_case_paths(cases_dir)]
    expanded_cases = _expand_case_modes(discovered_cases, input_modes)
    for case_path, input_json, mode in expanded_cases:
        case_out = root_out / _case_output_id(case_path, input_json=input_json, mode=mode)
        state = run_case_pipeline(
            case_path=case_path,
            out_dir=case_out,
            model_name=model_name,
            resources=resources,
            max_repair_iterations=max_repair_iterations,
            mock_behavior=mock_behavior,
            model_overrides=model_overrides,
            judge_model_name=judge_model_name,
            input_json=input_json,
        )
        case_metrics.append(state["case_metrics"])

    summary = aggregate_case_results(case_metrics)
    csv_path, summary_path = write_benchmark_metrics_outputs(case_metrics, root_out)

    outputs = output_settings(resources.benchmark_config)
    legacy_json_name = outputs.get("per_model_summary_json")
    legacy_csv_name = outputs.get("per_model_summary_csv")
    if legacy_json_name and legacy_json_name != "benchmark_summary.json":
        from benchmark.utils.io import write_json

        write_json(root_out / legacy_json_name, summary)
    if legacy_csv_name and legacy_csv_name != "benchmark_metrics.csv":
        (root_out / legacy_csv_name).write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    return summary, summary_path, csv_path


def apply_model_overrides(model_config: dict, model_name: str, model_overrides: dict | None) -> None:
    if not model_overrides:
        return
    selected = model_config.setdefault("models", {}).setdefault(model_name, {})
    for key, value in model_overrides.items():
        if value is None:
            continue
        config_key = MODEL_OVERRIDE_KEYS.get(key, key)
        selected[config_key] = value


def create_judge_model(
    *,
    model_name: str,
    model: object,
    model_config: dict,
    judge_model_name: str | None = None,
    benchmark_config: dict | None = None,
) -> tuple[str, object]:
    resolved = judge_model_name or _configured_judge_model_name(model_config)
    if resolved in {"", SAME_MODEL, "same_model"}:
        return model_name, _model_with_judge_generation_overrides(model, benchmark_config)
    return resolved, _model_with_judge_generation_overrides(create_model(resolved, model_config), benchmark_config)


def create_direct_judge_model(
    *,
    model_config: dict,
    judge_model_name: str | None = None,
    model_overrides: dict | None = None,
    benchmark_config: dict | None = None,
) -> tuple[str, object]:
    resolved = judge_model_name or _configured_judge_model_name(model_config)
    if resolved in {"", SAME_MODEL, "same_model"}:
        resolved = str(model_config.get("default_model") or "mock")
    apply_model_overrides(model_config, resolved, model_overrides)
    return resolved, _model_with_judge_generation_overrides(create_model(resolved, model_config), benchmark_config)


def _configured_judge_model_name(model_config: dict) -> str:
    judge_config = model_config.get("judge")
    if isinstance(judge_config, dict):
        return str(judge_config.get("model") or SAME_MODEL)
    return SAME_MODEL


def _model_with_judge_generation_overrides(model: object, benchmark_config: dict | None) -> object:
    runtime_profile = getattr(model, "runtime_profile", None)
    overrides = judge_generation_overrides(benchmark_config, runtime_profile)
    if not overrides:
        return model
    judge_model = copy(model)
    for key, value in overrides.items():
        if value is not None and hasattr(judge_model, key):
            setattr(judge_model, key, value)
    return judge_model


def _validate_scene_or_layout_or_raise(scene: object, scene_schema: dict, layout_schema: dict, scene_path: str | Path) -> str:
    scene_errors = _schema_errors(scene, scene_schema)
    layout_errors = _schema_errors(scene, layout_schema) if _looks_like_legend_layout(scene) else None
    if not scene_errors:
        if layout_errors == []:
            return "legend_layout"
        return "scene"
    if layout_errors is None:
        layout_errors = _schema_errors(scene, layout_schema)
    if not layout_errors:
        return "legend_layout"
    raise ValueError(
        f"Input JSON at {scene_path} must match schemas/scene.schema.json or legend/schemas/legend_layout.schema.json. "
        f"Scene errors: {_format_schema_errors(scene_errors)}; layout errors: {_format_schema_errors(layout_errors)}"
    )


def _looks_like_legend_layout(scene: object) -> bool:
    if not isinstance(scene, dict) or "assets" in scene or "scene_ref" in scene:
        return False
    objects = scene.get("objects")
    if not isinstance(objects, list) or not objects:
        return False
    if scene.get("scene_type") is not None or isinstance(scene.get("boundary"), list):
        return False
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        if obj.get("jid") is not None:
            return False
        if obj.get("object_id") is not None or obj.get("yaw") is not None:
            return True
    return isinstance(scene.get("coordinate_system"), dict)


def _schema_errors(instance: object, schema: dict) -> list:
    validator = Draft202012Validator(schema)
    return sorted(validator.iter_errors(instance), key=lambda item: list(item.path))


def _format_schema_errors(errors: list) -> str:
    details = []
    for error in errors[:10]:
        location = "$" + "".join(f"[{part!r}]" if isinstance(part, int) else f".{part}" for part in error.path)
        details.append(f"{location}: {error.message}")
    if len(errors) > 10:
        details.append(f"... {len(errors) - 10} additional schema errors")
    return "; ".join(details)


def _default_scene_schema() -> dict:
    return load_json_schema(Path(__file__).resolve().parents[2] / "schemas" / "scene.schema.json")


def _legend_layout_schema_path(root: Path) -> Path:
    legend_path = root / "legend" / "schemas" / "legend_layout.schema.json"
    return legend_path if legend_path.exists() else root / "schemas" / "layout.schema.json"


def _scene_schema_version(scene_schema: dict, scene: dict) -> str:
    for value in [
        scene.get("scene_schema_version") if isinstance(scene, dict) else None,
        scene_schema.get("version") if isinstance(scene_schema, dict) else None,
        scene_schema.get("$id") if isinstance(scene_schema, dict) else None,
    ]:
        if isinstance(value, str) and value:
            return value
    return ""


def _benchmark_config_with_vlm_judge_input_mode(benchmark_config: dict | None, mode: str | None) -> dict:
    patched = deepcopy(benchmark_config or {})
    if not mode:
        return patched
    patched["vlm_judge_input_mode"] = mode
    evaluation = dict(patched.get("evaluation") or {})
    evaluation["vlm_judge_input_mode"] = mode
    patched["evaluation"] = evaluation
    return patched


def copy_viewer_assets(out_dir: str | Path, project_root: str | Path) -> None:
    viewer_src = Path(project_root) / "web" / "viewer"
    for path in viewer_src.iterdir():
        if path.is_file():
            shutil.copy2(path, Path(out_dir) / path.name)


def _expand_case_modes(cases: list[tuple[Path, dict | None]], input_modes: list[str] | None) -> list[tuple[Path, dict | None, str | None]]:
    if not input_modes:
        return [(case_path, input_json, None) for case_path, input_json in cases]
    modes = [canonicalize_input_mode(mode) for mode in input_modes]
    expanded = []
    for case_path, input_json in cases:
        base = input_json if isinstance(input_json, dict) else read_json(case_path)
        if not isinstance(base, dict):
            raise ValueError(f"Case at {case_path} must be a JSON object.")
        for mode in modes:
            expanded.append((case_path, _case_with_input_mode(base, mode), mode))
    return expanded


def _case_with_input_mode(case: dict, mode: str) -> dict:
    patched = deepcopy(case)
    patched["scene_representation_mode"] = mode
    source = patched.get("source")
    if isinstance(source, dict):
        source = dict(source)
        source["input_representation_mode"] = mode
        source["scene_representation_mode"] = mode
        patched["source"] = source
    return patched


def _case_output_id(case_path: str | Path, *, input_json: dict | None = None, mode: str | None = None) -> str:
    path = Path(case_path)
    case = input_json
    if case is None:
        try:
            case = read_json(path)
        except (OSError, ValueError):
            case = {}
    base = str(case.get("case_id") or case.get("task_id") or path.stem) if isinstance(case, dict) else path.stem
    return f"{base}__{mode}" if mode else base
