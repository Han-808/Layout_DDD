from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.datasets.hssd_hab_converter import convert_hssd_hab
from benchmark.input_modes import list_main_input_modes
from benchmark.pipeline import PipelineResources, load_pipeline_resources, run_case_pipeline
from benchmark.utils.io import read_json, write_json


HF_DATASET = "hssd/hssd-hab"
HF_REVISION = "main"
DEFAULT_SCENE_IDS = [
    "102343992",
    "102344022",
    "102344049",
    "102344094",
    "102344115",
    "102344193",
    "102344250",
    "102344280",
    "102344307",
    "102344328",
]


@dataclass(frozen=True)
class ModeSpec:
    level: str
    input_representation_mode: str
    case_suffix: str


MODE_SPECS = {
    "prompt_only": ModeSpec(
        level="prompt_only",
        input_representation_mode="prompt_only",
        case_suffix="prompt_only",
    ),
    "compact_objects": ModeSpec(
        level="structured_basic",
        input_representation_mode="compact_objects",
        case_suffix="structured_basic",
    ),
    "compact_objects_with_estimated_relations": ModeSpec(
        level="structured_relation",
        input_representation_mode="compact_objects_with_estimated_relations",
        case_suffix="structured_relation",
    ),
    "full_metadata_budgeted": ModeSpec(
        level="structured_relation",
        input_representation_mode="full_metadata_budgeted",
        case_suffix="structured_relation",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 10-scene HSSD-HAB validity sweep across four input representation modes "
            "with the Qwen3-VL OpenAI-compatible endpoint."
        )
    )
    parser.add_argument("--hssd-root", default=str(PROJECT_ROOT / "data" / "external" / "hssd-hab"))
    parser.add_argument("--cases-root", default=str(PROJECT_ROOT / "data" / "external" / "hssd-hab-converted" / "hssd_hab_10_4mode"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "hssd_hab_10_4mode_qwen_validity"))
    parser.add_argument("--scene-ids", nargs="*", default=DEFAULT_SCENE_IDS)
    parser.add_argument("--modes", nargs="*", choices=sorted(MODE_SPECS), default=list_main_input_modes())
    parser.add_argument("--model", default="qwen3vl_sglang_32b")
    parser.add_argument("--judge-model", default="same")
    parser.add_argument("--vlm-judge-input-mode", choices=["json_only", "json_plus_render"], default=None)
    parser.add_argument("--model-endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-32B-Instruct")
    parser.add_argument("--judge-model-endpoint", default=None)
    parser.add_argument("--judge-model-id", default=None)
    parser.add_argument("--judge-timeout-seconds", type=int, default=None)
    parser.add_argument("--judge-context-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--generation-max-tokens", type=int, default=None)
    parser.add_argument("--repair-max-tokens", type=int, default=None)
    parser.add_argument("--judge-max-tokens", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--prompt-safety-margin-tokens", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--max-repair-iterations", type=int, default=0)
    parser.add_argument("--max-objects", type=int, default=None)
    parser.add_argument("--no-download", action="store_true", help="Fail if a raw scene JSON is missing instead of downloading it.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare/print the 40 case paths without calling the model endpoint.")
    parser.add_argument(
        "--valid-source",
        choices=["overall_valid", "validity_gate"],
        default="overall_valid",
        help="Field used to collapse each run to valid/not_valid.",
    )
    args = parser.parse_args()

    hssd_root = Path(args.hssd_root)
    cases_root = Path(args.cases_root)
    out_root = Path(args.out)
    scene_ids = [scene_id.strip() for scene_id in args.scene_ids if scene_id.strip()]
    modes = [mode.strip() for mode in args.modes if mode.strip()]

    scene_paths = _ensure_scene_files(hssd_root, scene_ids, download=not args.no_download)
    case_paths = _prepare_cases(
        hssd_root=hssd_root,
        cases_root=cases_root,
        scene_paths=scene_paths,
        scene_ids=scene_ids,
        modes=modes,
        max_objects=args.max_objects,
    )

    if args.dry_run:
        for item in case_paths:
            print(f"{item['scene_id']}\t{item['mode']}\t{item['case_path']}")
        return

    resources = _with_vlm_judge_input_mode(load_pipeline_resources(PROJECT_ROOT), args.vlm_judge_input_mode)
    _apply_judge_model_overrides(
        resources.model_config,
        judge_model=args.judge_model,
        endpoint=args.judge_model_endpoint,
        model_id=args.judge_model_id,
        timeout_seconds=args.judge_timeout_seconds,
        context_length=args.judge_context_length,
        max_tokens=args.judge_max_tokens,
    )
    model_overrides = {
        "model_endpoint": args.model_endpoint,
        "model_id": args.model_id,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "generation_max_tokens": args.generation_max_tokens,
        "repair_max_tokens": args.repair_max_tokens,
        "judge_max_tokens": args.judge_max_tokens,
        "context_length": args.context_length,
        "prompt_safety_margin_tokens": args.prompt_safety_margin_tokens,
        "timeout_seconds": args.timeout_seconds,
        "response_format_json": True,
    }

    out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for item in case_paths:
        scene_id = item["scene_id"]
        mode = item["mode"]
        case_path = item["case_path"]
        case_out = out_root / mode / scene_id
        try:
            state = run_case_pipeline(
                case_path=case_path,
                out_dir=case_out,
                model_name=args.model,
                resources=resources,
                max_repair_iterations=args.max_repair_iterations,
                model_overrides=model_overrides,
                judge_model_name=args.judge_model,
            )
            status = _status_from_state(state, args.valid_source)
        except Exception as exc:
            case_out.mkdir(parents=True, exist_ok=True)
            (case_out / "task_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            status = "not_valid"
        results.append({"scene_id": scene_id, "mode": mode, "status": status})
        print(f"{scene_id}\t{mode}\t{status}", flush=True)
        _write_status_outputs(out_root, results)

    _write_status_outputs(out_root, results)


def _apply_judge_model_overrides(
    model_config: dict,
    *,
    judge_model: str,
    endpoint: str | None,
    model_id: str | None,
    timeout_seconds: int | None,
    context_length: int | None,
    max_tokens: int | None,
) -> None:
    if not judge_model or judge_model in {"same", "same_model"}:
        return
    overrides = {
        "endpoint": endpoint,
        "model": model_id,
        "timeout_seconds": timeout_seconds,
        "context_length": context_length,
        "max_tokens": max_tokens,
        "judge_max_tokens": max_tokens,
        "response_format_json": True if (endpoint or model_id or max_tokens) else None,
        "judge_evidence_budgeting": True if (endpoint or model_id) else None,
    }
    selected = model_config.setdefault("models", {}).setdefault(judge_model, {})
    selected.setdefault("provider", "openai_compatible")
    for key, value in overrides.items():
        if value is not None:
            selected[key] = value


def _with_vlm_judge_input_mode(resources: PipelineResources, mode: str | None) -> PipelineResources:
    if not mode:
        return resources
    benchmark_config = deepcopy(resources.benchmark_config)
    benchmark_config["vlm_judge_input_mode"] = mode
    evaluation = dict(benchmark_config.get("evaluation") or {})
    evaluation["vlm_judge_input_mode"] = mode
    benchmark_config["evaluation"] = evaluation
    return PipelineResources(
        model_config=resources.model_config,
        benchmark_config=benchmark_config,
        layout_schema=resources.layout_schema,
        scene_schema=resources.scene_schema,
        resolved_run_config=resources.resolved_run_config,
    )


def _ensure_scene_files(hssd_root: Path, scene_ids: list[str], *, download: bool) -> list[Path]:
    scene_dir = hssd_root / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for scene_id in scene_ids:
        path = scene_dir / f"{scene_id}.scene_instance.json"
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing HSSD-HAB scene file: {path}")
            _download_scene(scene_id, path)
        paths.append(path)
    return paths


def _download_scene(scene_id: str, target: Path) -> None:
    relative_path = f"scenes/{scene_id}.scene_instance.json"
    url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}/{quote(relative_path)}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response:
        target.write_bytes(response.read())


def _prepare_cases(
    *,
    hssd_root: Path,
    cases_root: Path,
    scene_paths: list[Path],
    scene_ids: list[str],
    modes: list[str],
    max_objects: int | None,
) -> list[dict]:
    case_paths = []
    for mode in modes:
        spec = MODE_SPECS[mode]
        mode_dir = cases_root / mode
        convert_hssd_hab(
            hssd_root=hssd_root,
            out_dir=mode_dir,
            scene_paths=[str(path) for path in scene_paths],
            levels=[spec.level],
            max_objects=max_objects,
            preserve_raw_metadata=True,
            bbox_from_scale=True,
            include_estimated_relations=True,
            input_representation_mode=spec.input_representation_mode,
        )
        for scene_id in scene_ids:
            case_paths.append(
                {
                    "scene_id": scene_id,
                    "mode": mode,
                    "case_path": mode_dir / f"{scene_id}_{spec.case_suffix}.json",
                }
            )
    return case_paths


def _status_from_state(state: dict, valid_source: str) -> str:
    if valid_source == "validity_gate":
        passed = bool((state.get("case_metrics") or {}).get("validity_gate"))
    else:
        result_path = state.get("per_case_result_path")
        result = read_json(result_path) if result_path else state.get("per_case_result", {})
        history = result.get("history") if isinstance(result, dict) else None
        final = history[-1] if isinstance(history, list) and history else {}
        passed = bool(final.get("overall_valid"))
    return "valid" if passed else "not_valid"


def _write_status_outputs(out_root: Path, results: list[dict]) -> None:
    lines = ["scene_id\tmode\tstatus"]
    lines.extend(f"{item['scene_id']}\t{item['mode']}\t{item['status']}" for item in results)
    (out_root / "validity_results.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    matrix: dict[str, dict[str, str]] = {}
    for item in results:
        matrix.setdefault(item["scene_id"], {})[item["mode"]] = item["status"]
    write_json(out_root / "validity_matrix.json", matrix)
    (out_root / "validity_results.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in results),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
