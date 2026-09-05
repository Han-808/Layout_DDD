"""CLI for the shared retrieval runtime v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .factory import build_runtime


def _write(value: Mapping[str, Any], output: str) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output == "-":
        print(text, end="")
    else:
        Path(output).write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--resource-bindings", type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    gate = subparsers.add_parser("gate")
    gate.add_argument(
        "--strict",
        action="store_true",
        help="compatibility flag; official profiles are strict regardless",
    )
    gate.add_argument("--skip-golden", action="store_true")
    gate.add_argument("--output", default="-")
    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--input", type=Path, required=True)
    retrieve.add_argument("--output", default="-")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    runtime = build_runtime(
        catalog_path=args.catalog,
        retrieval_profile_id=args.profile,
        resource_bindings_path=args.resource_bindings,
    )
    if args.command == "gate":
        report = runtime.gate(
            strict=bool(args.strict),
            run_golden=not args.skip_golden,
        )
        _write(report, args.output)
        return 0 if report["status"] != "failed" else 2
    request = json.loads(args.input.read_text(encoding="utf-8"))
    gate = runtime.gate(run_golden=False)
    if gate["status"] == "failed":
        _write(gate, args.output)
        return 2
    _write(runtime.retrieve_batch(request), args.output)
    return 0
