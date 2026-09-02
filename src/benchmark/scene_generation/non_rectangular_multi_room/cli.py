"""Standalone CLI for non-rectangular global multi-room generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from benchmark.scene_generation.campaign.execution import (
    gate_resources,
    resolve_bindings,
)
from benchmark.scene_generation.non_rectangular_multi_room.campaign import (
    preflight_non_rectangular_campaign,
    prepare_non_rectangular_campaign,
    run_prepared_non_rectangular_campaign,
)


def _common(parser: argparse.ArgumentParser, *, bindings: bool = False) -> None:
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--room-layout", type=Path, required=True)
    parser.add_argument("--room-program", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path)
    parser.add_argument("--retrieval-catalog", type=Path)
    parser.add_argument(
        "--contract-version",
        choices=("v1", "v2"),
        default="v1",
    )
    if bindings:
        parser.add_argument("--generation-bindings", type=Path)
        parser.add_argument("--resource-bindings", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check")
    _common(check)
    resolve = commands.add_parser("resolve")
    _common(resolve, bindings=True)
    gate = commands.add_parser("resource-gate")
    _common(gate)
    gate.add_argument("--resource-bindings", type=Path)
    preflight = commands.add_parser("preflight")
    _common(preflight, bindings=True)
    run = commands.add_parser("run")
    _common(run, bindings=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    return parser


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True), flush=True)


def _progress(value: Mapping[str, Any]) -> None:
    if value.get("event") in {
        "run_terminal",
        "run_resumed_terminal",
    }:
        _emit(value)


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        prepared = prepare_non_rectangular_campaign(
            args.campaign,
            room_layout_path=args.room_layout,
            room_program_path=args.room_program,
            profile_root=args.profile_root,
            retrieval_catalog_path=args.retrieval_catalog,
            contract_version=args.contract_version,
        )
        if args.command == "check":
            _emit({"valid": True, **prepared.public_dict()})
            return 0
        if args.command == "resolve":
            _, _, report = resolve_bindings(
                prepared,
                generation_bindings_path=args.generation_bindings,
                resource_bindings_path=args.resource_bindings,
            )
            _emit(report)
            return 0
        if args.command == "resource-gate":
            _, report = gate_resources(
                prepared,
                resource_bindings_path=args.resource_bindings,
            )
            _emit(report)
            return 0 if report.get("status") != "failed" else 2
        if args.command == "preflight":
            report, _ = preflight_non_rectangular_campaign(
                prepared,
                generation_bindings_path=args.generation_bindings,
                resource_bindings_path=args.resource_bindings,
            )
            _emit(report)
            return 0 if report.get("ok") else 2
        summary, stopped, preflight = run_prepared_non_rectangular_campaign(
            prepared,
            output_root=args.output_dir,
            generation_bindings_path=args.generation_bindings,
            resource_bindings_path=args.resource_bindings,
            progress=_progress,
            resume=args.resume,
        )
        _emit(
            {
                "schema_version": "non_rectangular_campaign_terminal_v1",
                "summary": summary,
                "stopped": stopped,
                "preflight": preflight,
            }
        )
        return 0 if summary.get("status") == "complete" else 2
    except Exception as exc:
        print(
            f"error: {type(exc).__name__}: non-rectangular campaign command failed",
            file=sys.stderr,
        )
        return 3


__all__ = ["build_parser", "main"]
