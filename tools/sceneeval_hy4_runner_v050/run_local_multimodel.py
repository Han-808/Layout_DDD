#!/usr/bin/env python3
"""Run the exact SceneEval-100 prompt/input through local API models."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from sceneeval_hy4.artifacts import ArtifactError
from sceneeval_hy4.inputs import InputError
from sceneeval_local_multimodel import (
    DEFAULT_BASE_URL,
    DEFAULT_INPUT_PATH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_REASONING_EFFORT,
    MODEL_ORDER,
    REASONING_EFFORTS,
    LocalMultiModelError,
    run_ordered_suite,
    run_smoke_probe,
    validate_exact_input,
)
from sceneeval_hy4.transport import EndpointError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "SceneEval-100 local capture runner using the exact frozen input and "
            "system/user prompts"
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check-input",
        help="validate the exact approved 100-row input without an API call",
    )
    check.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_PATH)

    run = subparsers.add_parser(
        "run",
        help="run the fixed multi-model suite serially",
    )
    run.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_PATH)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--base-url", default=DEFAULT_BASE_URL)
    run.add_argument("--timeout", type=float, default=1800.0)
    run.add_argument("--max-retries", type=int, default=2)
    run.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    run.add_argument(
        "--reasoning-effort",
        choices=sorted(REASONING_EFFORTS),
        default=DEFAULT_REASONING_EFFORT,
    )
    run.add_argument(
        "--api-key-env",
        default="FORGEAX_API_KEY",
        help=(
            "read the key from this environment variable when present; otherwise "
            "prompt with hidden input"
        ),
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="resume each model without resending attempted/ambiguous requests",
    )

    probe = subparsers.add_parser(
        "smoke-probe",
        help="send one exact SceneEval case to each model in fixed order",
    )
    probe.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_PATH)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--scene-id", type=int, default=0)
    probe.add_argument("--base-url", default=DEFAULT_BASE_URL)
    probe.add_argument("--timeout", type=float, default=1800.0)
    probe.add_argument("--max-retries", type=int, default=2)
    probe.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    probe.add_argument(
        "--reasoning-effort",
        choices=sorted(REASONING_EFFORTS),
        default=DEFAULT_REASONING_EFFORT,
    )
    probe.add_argument(
        "--api-key-env",
        default="FORGEAX_API_KEY",
        help=(
            "read the key from this environment variable when present; otherwise "
            "prompt with hidden input"
        ),
    )
    return parser


def _runtime_key(env_name: str) -> str:
    value = os.environ.get(env_name)
    if value:
        return value
    value = getpass.getpass("ForgeAX API Key（隐藏输入，不写入产物）: ")
    if not value:
        raise LocalMultiModelError("API key must be non-empty")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check-input":
            batch = validate_exact_input(args.input_jsonl)
            print(
                json.dumps(
                    {
                        "valid": True,
                        "row_count": len(batch.scenes),
                        "first_id": batch.scenes[0].scene_id,
                        "last_id": batch.scenes[-1].scene_id,
                        "input_sha256": batch.sha256,
                        "model_order": list(MODEL_ORDER),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        key = _runtime_key(args.api_key_env)
        if args.command == "smoke-probe":
            results = run_smoke_probe(
                api_key=key,
                output_root=args.output_dir,
                scene_id=args.scene_id,
                input_jsonl=args.input_jsonl,
                base_url=args.base_url,
                timeout_seconds=args.timeout,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
            )
        else:
            results = run_ordered_suite(
                api_key=key,
                output_root=args.output_dir,
                input_jsonl=args.input_jsonl,
                base_url=args.base_url,
                timeout_seconds=args.timeout,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                resume=args.resume,
            )
        print(json.dumps({"results": results}, ensure_ascii=False, sort_keys=True))
        return 2 if any(item.get("stopped", item.get("stop_batch")) for item in results.values()) else 0
    except (
        ArtifactError,
        EndpointError,
        InputError,
        LocalMultiModelError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
