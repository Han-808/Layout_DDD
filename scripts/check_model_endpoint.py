"""Smoke-check an OpenAI-compatible model endpoint.

Summary:
    Quick connectivity/format check for a local or remote OpenAI-compatible
    LLM/VLM endpoint before running a full generation or evaluation pipeline.

Input:
    - ``--endpoint`` base URL and ``--model`` id (required).
    - Optional ``--api-key-env``, ``--timeout-seconds``, JSON-response-format flags,
      and ``--multimodal`` to also send a tiny image.

Output:
    - Prints a JSON status result (reachability and a sample response) to stdout.

Function:
    Sends a minimal chat completion (optionally multimodal) via
    ``OpenAICompatibleModel`` and reports the outcome. Read-only; no files.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.models import OpenAICompatibleModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-check an OpenAI-compatible model endpoint.")
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible base URL, for example http://127.0.0.1:8298/v1.")
    parser.add_argument("--model", required=True, help="Served model id.")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument(
        "--min-request-interval-seconds",
        type=float,
        default=0.0,
        help="Minimum start-to-start interval shared by requests from this model client.",
    )
    parser.add_argument(
        "--max-tokens-field",
        choices=["max_tokens", "max_completion_tokens"],
        default="max_tokens",
    )
    parser.add_argument(
        "--send-temperature",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--api-key-env",
        default=None,
        help="Environment variable containing the API key. Official OpenAI endpoints default to OPENAI_API_KEY.",
    )
    response_group = parser.add_mutually_exclusive_group()
    response_group.add_argument("--response-format-json", dest="response_format_json", action="store_true", default=None)
    response_group.add_argument("--no-response-format-json", dest="response_format_json", action="store_false")
    parser.add_argument("--multimodal", action="store_true", help="Also send a tiny PNG data URL through chat completions.")
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Use this local PNG/JPEG/WebP for the multimodal check instead of the synthetic 1x1 PNG.",
    )
    args = parser.parse_args()
    if args.image_path is not None and not args.multimodal:
        parser.error("--image-path requires --multimodal")
    image_data_url = _image_data_url(args.image_path) if args.image_path is not None else None

    model = OpenAICompatibleModel(
        name="endpoint_health_check",
        endpoint=args.endpoint,
        model_id=args.model,
        api_key_env=args.api_key_env,
        max_tokens=args.max_tokens,
        max_tokens_field=args.max_tokens_field,
        send_temperature=args.send_temperature,
        timeout_seconds=args.timeout_seconds,
        response_format_json=bool(args.response_format_json),
        min_request_interval_seconds=args.min_request_interval_seconds,
    )
    try:
        result = model.health_check(
            multimodal=args.multimodal,
            image_data_url=image_data_url,
        )
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


def _image_data_url(path: Path) -> str:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"Image does not exist or is not a file: {path}")
    mime_type, _ = mimetypes.guess_type(resolved.name)
    if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise SystemExit(f"Unsupported smoke-test image type: {mime_type or 'unknown'}")
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


if __name__ == "__main__":
    main()
