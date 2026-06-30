from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.models import create_model
from benchmark.pipeline import apply_model_overrides, load_pipeline_resources


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check an OpenAI-compatible model endpoint.")
    parser.add_argument("--model", default="qwen3vl_sglang", help="Model name from configs/model_config.yaml.")
    parser.add_argument("--model_endpoint", default=None, help="Override selected OpenAI-compatible model endpoint.")
    parser.add_argument("--model_id", default=None, help="Override selected OpenAI-compatible model id.")
    parser.add_argument("--timeout_seconds", type=int, default=None, help="Override selected model timeout_seconds.")
    response_group = parser.add_mutually_exclusive_group()
    response_group.add_argument("--response_format_json", dest="response_format_json", action="store_true", default=None)
    response_group.add_argument("--no_response_format_json", dest="response_format_json", action="store_false")
    parser.add_argument("--multimodal", action="store_true", help="Also send a tiny PNG data URL through chat completions.")
    args = parser.parse_args()

    resources = load_pipeline_resources(PROJECT_ROOT)
    model_config = deepcopy(resources.model_config)
    apply_model_overrides(model_config, args.model, _model_overrides(args))
    model = create_model(args.model, model_config)
    if not hasattr(model, "health_check"):
        print(json.dumps({"ok": False, "error": f"Model '{args.model}' does not expose endpoint health_check()."}, indent=2))
        raise SystemExit(2)
    try:
        result = model.health_check(multimodal=args.multimodal)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "endpoint": getattr(model, "endpoint", None),
                    "model_id": getattr(model, "model_id", None),
                },
                indent=2,
            )
        )
        raise SystemExit(1) from exc
    result["ok"] = True
    result["last_request_metadata"] = getattr(model, "last_request_metadata", {})
    print(json.dumps(result, indent=2))


def _model_overrides(args: argparse.Namespace) -> dict:
    return {
        "endpoint": args.model_endpoint,
        "model_id": args.model_id,
        "timeout_seconds": args.timeout_seconds,
        "response_format_json": args.response_format_json,
    }


if __name__ == "__main__":
    main()
