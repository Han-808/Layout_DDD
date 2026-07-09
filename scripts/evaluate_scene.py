from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.nl_scene.dummy_evaluator import evaluate_scene as evaluate_dummy_scene
from benchmark.pipeline import PipelineResources, evaluate_scene_pipeline, load_pipeline_resources
from benchmark.utils.io import load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a canonical 3D scene JSON without running layout generation.")
    parser.add_argument("--scene", required=True, help="Path to canonical scene JSON.")
    parser.add_argument("--out", required=True, help="Output directory for full evaluation artifacts, or report path when --dummy is used.")
    parser.add_argument("--instruction", default=None, help="Optional original natural-language instruction for --dummy reports.")
    parser.add_argument("--instruction-file", default=None, help="Optional file containing the instruction for --dummy reports.")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for --dummy placeholder metrics.")
    parser.add_argument("--dummy", action="store_true", help="Use the lightweight dummy evaluator instead of the full VLM-as-judge pipeline.")
    parser.add_argument("--judge-model", "--judge_model", dest="judge_model", default=None, help="Model name from configs/model_config.yaml to use as judge.")
    parser.add_argument("--vlm-judge-input-mode", choices=["json_only", "json_plus_render"], default=None)
    parser.add_argument("--benchmark-config", "--benchmark_config", dest="benchmark_config", default=None, help="Optional benchmark config YAML overlay.")
    parser.add_argument("--model-endpoint", "--model_endpoint", dest="model_endpoint", default=None)
    parser.add_argument("--model-id", "--model_id", dest="model_id", default=None)
    parser.add_argument("--context-length", "--context_length", dest="context_length", type=int, default=None)
    parser.add_argument("--judge-max-tokens", "--judge_max_tokens", dest="judge_max_tokens", type=int, default=None)
    parser.add_argument("--response-format-json", "--response_format_json", dest="response_format_json", action="store_true", default=None)
    parser.add_argument("--no-response-format-json", "--no_response_format_json", dest="response_format_json", action="store_false")
    parser.add_argument("--no-render", action="store_true", help="Alias for --vlm-judge-input-mode json_only.")
    parser.add_argument("--json-plus-render", action="store_true", help="Alias for --vlm-judge-input-mode json_plus_render.")
    args = parser.parse_args()

    if args.dummy:
        report = evaluate_dummy_scene(_path_arg(args.scene).read_text(encoding="utf-8"), instruction=_instruction(args, parser), seed=args.seed, dummy=True)
        report_path = _dummy_report_path(args.out)
        write_json(report_path, report)
        print(f"evaluation_report: {report_path}")
        return

    try:
        resources = _with_benchmark_config_overlay(load_pipeline_resources(PROJECT_ROOT), args.benchmark_config)
        mode = _resolve_mode(args)
        state = evaluate_scene_pipeline(
            scene_path=_path_arg(args.scene),
            out_dir=_path_arg(args.out),
            resources=resources,
            judge_model_name=args.judge_model,
            model_overrides=_model_cli_overrides(args),
            vlm_judge_input_mode=mode,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print(f"evaluation_report: {state['current_evaluation_path']}")
    print(f"feedback: {state['current_feedback_path']}")
    print(f"case_metrics: {state['case_metrics_path']}")
    if state.get("viewer_scene_path"):
        print(f"viewer_scene: {state['viewer_scene_path']}")


def _instruction(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str | None:
    if args.instruction and args.instruction_file:
        parser.error("Use either --instruction or --instruction-file, not both.")
    if args.instruction_file:
        return _path_arg(args.instruction_file).read_text(encoding="utf-8").strip()
    return args.instruction.strip() if isinstance(args.instruction, str) and args.instruction.strip() else None


def _dummy_report_path(value: str) -> Path:
    path = _path_arg(value)
    return path if path.suffix == ".json" else path / "evaluation_report.json"


def _resolve_mode(args: argparse.Namespace) -> str:
    if args.vlm_judge_input_mode:
        return args.vlm_judge_input_mode
    if args.json_plus_render:
        return "json_plus_render"
    if args.no_render:
        return "json_only"
    return "json_only"


def _model_cli_overrides(args: argparse.Namespace) -> dict:
    return {
        "model_endpoint": args.model_endpoint,
        "model_id": args.model_id,
        "context_length": args.context_length,
        "judge_max_tokens": args.judge_max_tokens,
        "response_format_json": args.response_format_json,
    }


def _with_benchmark_config_overlay(resources: PipelineResources, config_path: str | None) -> PipelineResources:
    if not config_path:
        return resources
    path = _path_arg(config_path)
    if not path.exists():
        raise ValueError(f"Benchmark config overlay does not exist: {config_path}")
    overlay = load_yaml(path, default={})
    if not isinstance(overlay, dict):
        raise ValueError(f"Benchmark config overlay must be a YAML object: {config_path}")
    return PipelineResources(
        model_config=resources.model_config,
        benchmark_config=_deep_merge(deepcopy(resources.benchmark_config), overlay),
        layout_schema=resources.layout_schema,
        scene_schema=resources.scene_schema,
        resolved_run_config=resources.resolved_run_config,
    )


def _deep_merge(base: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(dict(base[key]), value)
        else:
            base[key] = value
    return base


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
