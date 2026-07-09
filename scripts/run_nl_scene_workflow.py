from __future__ import annotations

CURRENT_INPUT_CHAIN = "natural_language"
LEGEND_INPUT_CHAIN = False

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.nl_scene.workflow import run_nl_scene_workflow
from benchmark.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the current natural-language scene construction MVP through conversion and retrieval.")
    parser.add_argument("--instruction", default=None, help="Natural-language scene instruction.")
    parser.add_argument("--instruction-file", default=None, help="File containing the natural-language scene instruction.")
    parser.add_argument("--scene-type", default=None, help="Optional scene type hint, e.g. living room.")
    parser.add_argument("--room-json", default=None, help="Optional room JSON file.")
    parser.add_argument("--asset-index-path", required=True, help="AssetRetriever index path prefix.")
    parser.add_argument("--retrieval-k", type=int, default=1)
    parser.add_argument("--retriever-module-path", default=None)
    parser.add_argument("--use-vlm-selector", dest="use_vlm_selector", action="store_true", default=True)
    parser.add_argument("--no-vlm-selector", dest="use_vlm_selector", action="store_false")
    parser.add_argument("--model", default=None, help="OpenAI-compatible model id.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--generated-scene", default=None, help="Optional generated scene JSON to evaluate with the dummy evaluator.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    instruction = _instruction(args, parser)
    room = read_json(_path_arg(args.room_json)) if args.room_json else None
    model_config = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    result = run_nl_scene_workflow(
        instruction=instruction,
        scene_type=args.scene_type,
        room=room,
        asset_index_path=str(_path_arg(args.asset_index_path)),
        retrieval_k=args.retrieval_k,
        retriever_module_path=args.retriever_module_path,
        use_vlm_selector=args.use_vlm_selector,
        model_config={key: value for key, value in model_config.items() if value is not None},
        generated_scene_path=_path_arg(args.generated_scene) if args.generated_scene else None,
        out_dir=_path_arg(args.out_dir),
        seed=args.seed,
    )
    for name, path in result["artifacts"].items():
        print(f"{name}: {path}")


def _instruction(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.instruction and args.instruction_file:
        parser.error("Use either --instruction or --instruction-file, not both.")
    if args.instruction_file:
        return _path_arg(args.instruction_file).read_text(encoding="utf-8").strip()
    if args.instruction:
        return args.instruction.strip()
    parser.error("One of --instruction or --instruction-file is required.")
    raise AssertionError("unreachable")


def _path_arg(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


if __name__ == "__main__":
    main()
