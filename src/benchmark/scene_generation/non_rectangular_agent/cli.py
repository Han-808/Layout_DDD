"""CLI for the customized-FloorPlan shared-DB Agent track."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .cohort_runner import execute_agent_fullrun, prepare_agent_fullrun


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "resource-gate", "run"):
        command = commands.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
        if name in {"resource-gate", "run"}:
            command.add_argument("--resource-bindings", type=Path)
        if name == "run":
            command.add_argument("--output-base", type=Path, required=True)
            command.add_argument("--fresh", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        prepared = prepare_agent_fullrun(args.profile)
        if args.command == "check":
            result = prepared.public_dict()
        elif args.command == "resource-gate":
            result = execute_agent_fullrun(
                prepared,
                command="resource-gate",
                resource_bindings_path=args.resource_bindings,
            )
        else:
            result = execute_agent_fullrun(
                prepared,
                command="run",
                output_base=args.output_base,
                resource_bindings_path=args.resource_bindings,
                fresh=args.fresh,
            )
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
        }
        _emit(result)
        return 2
    _emit(result)
    return 0 if result.get("status") in {None, "ready", "complete"} else 2


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True), flush=True)


__all__ = ["build_parser", "main"]
