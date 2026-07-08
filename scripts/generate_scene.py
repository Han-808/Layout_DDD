from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.models import create_model
from benchmark.pipeline import apply_model_overrides, load_pipeline_resources
from benchmark.utils.io import read_json
from benchmark.workflow.generation import generate_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a canonical scene JSON from a benchmark case.")
    parser.add_argument("--case", required=True, help="Path to benchmark case JSON.")
    parser.add_argument("--out", required=True, help="Output directory for generated scene artifacts.")
    parser.add_argument("--model", default="mock", help="Model name from configs/model_config.yaml.")
    parser.add_argument("--model-endpoint", "--model_endpoint", dest="model_endpoint", default=None)
    parser.add_argument("--model-id", "--model_id", dest="model_id", default=None)
    parser.add_argument("--context-length", "--context_length", dest="context_length", type=int, default=None)
    args = parser.parse_args()

    resources = load_pipeline_resources(PROJECT_ROOT)
    model_config = resources.model_config
    apply_model_overrides(model_config, args.model, _model_cli_overrides(args))
    model = create_model(args.model, model_config)
    scene, metadata = generate_scene(
        read_json(_path_arg(args.case)),
        model=model,
        scene_schema=resources.scene_schema,
        legend_layout_schema=resources.layout_schema,
        benchmark_config=resources.benchmark_config,
        out_dir=_path_arg(args.out),
    )
    print(f"generated_scene: {metadata['generated_scene_path']}")
    print(f"legend_layout: {metadata['legend_layout_path']}")
    print(f"scene_id: {scene.get('scene_id', '')}")


def _model_cli_overrides(args: argparse.Namespace) -> dict:
    return {
        "model_endpoint": args.model_endpoint,
        "model_id": args.model_id,
        "context_length": args.context_length,
    }


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
