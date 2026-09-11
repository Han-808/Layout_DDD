"""One model-agnostic API2/API3 generation campaign entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

from benchmark.scene_generation.campaign.api import (
    preflight_generation_campaign,
    prepare_generation_campaign,
    resolve_generation_campaign,
    resource_gate_generation_campaign,
    run_generation_campaign,
)
from benchmark.scene_generation.campaign.multi_room_execution import (
    PreparedMultiRoomCampaign,
)


def _emit(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True), flush=True)


def _common(parser: argparse.ArgumentParser, *, bindings: bool = False) -> None:
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--profile-root", type=Path)
    parser.add_argument("--retrieval-catalog", type=Path)
    parser.add_argument("--trust-manifest", type=Path)
    parser.add_argument(
        "--floor-plan",
        type=Path,
        help="required only by an explicitly registered multi-room campaign",
    )
    if bindings:
        parser.add_argument("--generation-bindings", type=Path)
        parser.add_argument("--resource-bindings", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="validate public contracts only")
    _common(check)
    resolve = commands.add_parser(
        "resolve", help="resolve local bindings without reading credentials"
    )
    _common(resolve, bindings=True)
    gate = commands.add_parser(
        "resource-gate", help="run strict resource hashes and golden Top-1 gate"
    )
    _common(gate)
    gate.add_argument("--resource-bindings", type=Path)
    preflight = commands.add_parser(
        "preflight", help="gate resources, then verify the live route contract"
    )
    _common(preflight, bindings=True)
    run = commands.add_parser(
        "run", help="gate, preflight, and run the frozen generation workflow"
    )
    _common(run, bindings=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--resume",
        action="store_true",
        help="resume hash-verified terminal rooms in an additive multi-room run",
    )
    return parser


def _progress(record: Mapping[str, Any]) -> None:
    if record.get("event") in {
        "case_terminal",
        "room_terminal",
        "room_resumed_terminal",
        "run_terminal",
    }:
        _emit(record)


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        prepared = prepare_generation_campaign(
            args.campaign,
            floor_plan_path=args.floor_plan,
            profile_root=args.profile_root,
            retrieval_catalog_path=args.retrieval_catalog,
            trust_manifest=args.trust_manifest,
        )
        if args.command == "check":
            _emit({"valid": True, **prepared.public_dict()})
            return 0
        if args.command == "resolve":
            _, _, report = resolve_generation_campaign(
                prepared,
                generation_bindings_path=args.generation_bindings,
                resource_bindings_path=args.resource_bindings,
            )
            _emit(report)
            return 0
        if args.command == "resource-gate":
            _, report = resource_gate_generation_campaign(
                prepared,
                resource_bindings_path=args.resource_bindings,
            )
            _emit(report)
            return 0 if report.get("status") != "failed" else 2
        if args.command == "preflight":
            report, _ = preflight_generation_campaign(
                prepared,
                generation_bindings_path=args.generation_bindings,
                resource_bindings_path=args.resource_bindings,
            )
            _emit(report)
            return 0 if report.get("ok") else 2
        summary, stopped, preflight = run_generation_campaign(
            prepared,
            output_root=args.output_dir,
            generation_bindings_path=args.generation_bindings,
            resource_bindings_path=args.resource_bindings,
            progress=_progress,
            resume=args.resume,
        )
        _emit(
            {
                "schema_version": (
                    "generation_campaign_terminal_v3"
                    if isinstance(prepared, PreparedMultiRoomCampaign)
                    else "generation_campaign_terminal_v2"
                ),
                "preflight_ok": preflight["ok"],
                "preflight": preflight,
                "summary": summary,
            }
        )
        failed = summary.get("failed", summary.get("failed_rooms", 0))
        return 2 if stopped or failed else 0
    except Exception as exc:
        # This is a public terminal surface.  Arbitrary dependency exceptions
        # may contain endpoints, binding paths, credential names, headers, or
        # server bodies, so only a stable type/category is emitted here.
        print(
            f"error: {type(exc).__name__}: generation campaign command failed",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
