"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifacts import ArtifactError
from .client_config import ClientConfigError, load_run_config
from .constants import MODEL_ALIAS
from .inputs import InputError, load_human100_jsonl
from .runner import run_batch
from .transport import EndpointError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HY4-only online SceneEval-100 raw capture runner"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check-input",
        help="validate a sanitized 100-row id/description JSONL without an API call",
    )
    check.add_argument("--input-jsonl", type=Path, required=True)

    run = subparsers.add_parser(
        "run",
        help="capture HY4 using the frozen openai_clients.yaml",
    )
    run.add_argument("--input-jsonl", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip immutable completed attempts and continue with the first unattempted id; "
            "never resend an attempted or ambiguous request"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        batch = load_human100_jsonl(args.input_jsonl)
        if args.command == "check-input":
            print(
                json.dumps(
                    {
                        "valid": True,
                        "model": MODEL_ALIAS,
                        "row_count": len(batch.scenes),
                        "first_id": batch.scenes[0].scene_id,
                        "last_id": batch.scenes[-1].scene_id,
                        "input_sha256": batch.sha256,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0

        config = load_run_config()
        summary, stopped = run_batch(
            batch,
            args.output_dir,
            config,
            resume=args.resume,
        )
        print(json.dumps({"summary": summary}, ensure_ascii=False, sort_keys=True))
        return 2 if stopped else 0
    except (
        InputError,
        ArtifactError,
        ClientConfigError,
        EndpointError,
        OSError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
