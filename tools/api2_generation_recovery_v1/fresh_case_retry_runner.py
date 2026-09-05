#!/usr/bin/env python3
"""Give selected API2 generation cases multiple fresh one-shot chances.

Each fresh chance is an independent audited recovery run. Infrastructure retries
inside a chance remain governed by the frozen adapter configuration; a terminal
model/schema failure advances to the next fresh chance for that same brief.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import recovery_runner as recovery


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def parse_ids(value: str) -> tuple[str, ...]:
    ids = tuple(item.strip() for item in value.split(",") if item.strip())
    if not ids:
        raise argparse.ArgumentTypeError("at least one brief ID is required")
    if len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("brief IDs must not contain duplicates")
    return ids


def check(
    adapter_name: str,
    brief_ids: tuple[str, ...],
    fresh_chances: int,
) -> dict[str, Any]:
    if fresh_chances != 3:
        raise ValueError("this retry runner requires exactly three fresh chances")
    base = recovery.check(adapter_name, brief_ids)
    return {
        **base,
        "fresh_chances_per_case": fresh_chances,
        "fresh_retry_on_terminal_schema_failure": True,
        "later_cases_continue_after_exhausted_case": True,
    }


def run(
    adapter_name: str,
    brief_ids: tuple[str, ...],
    output_root: Path,
    fresh_chances: int,
) -> dict[str, Any]:
    policy = check(adapter_name, brief_ids, fresh_chances)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    output_root.mkdir(parents=True)
    write_json_exclusive(
        output_root / "execution_policy.json",
        {
            "schema_version": "api2_fresh_case_retry_policy_v1",
            "adapter": adapter_name,
            "selected_brief_ids": list(brief_ids),
            "fresh_chances_per_case": fresh_chances,
            "fresh_retry_conditions": [
                "terminal_schema_or_validation_failure",
                "terminal_api_or_model_failure",
                "exhausted_infrastructure_retries",
            ],
            "infrastructure_retries_per_stage_within_each_chance": policy[
                "maximum_infrastructure_retries"
            ],
            "case_failure_policy": "record_and_continue_next_fresh_chance",
            "batch_failure_policy": "record_and_continue_next_brief",
            "source_outputs_modified": False,
        },
    )

    case_rows: list[dict[str, Any]] = []
    for brief_id in brief_ids:
        attempts: list[dict[str, Any]] = []
        selected_source: str | None = None
        for chance in range(1, fresh_chances + 1):
            chance_root = output_root / brief_id / f"chance_{chance:02d}"
            chance_summary = recovery.run_selected(
                adapter_name,
                (brief_id,),
                chance_root,
            )
            result = chance_summary["results"][0]
            eligible = bool(
                result.get("status") == "complete"
                and result.get("eligible_for_strict_one_shot_evaluation") is True
            )
            relative_source = chance_root.relative_to(output_root).as_posix()
            attempts.append(
                {
                    "chance": chance,
                    "source_run": relative_source,
                    "status": result.get("status"),
                    "eligible": eligible,
                    "reason": result.get("reason"),
                }
            )
            print(
                json.dumps(
                    {
                        "brief_id": brief_id,
                        "fresh_chance": chance,
                        "status": result.get("status"),
                        "eligible": eligible,
                        "will_retry_fresh": not eligible and chance < fresh_chances,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            if eligible:
                selected_source = relative_source
                break

        row = {
            "brief_id": brief_id,
            "status": "complete" if selected_source is not None else "failed",
            "eligible": selected_source is not None,
            "selected_source": selected_source,
            "fresh_chances_used": len(attempts),
            "attempts": attempts,
            "continued_to_next_brief": True,
        }
        write_json_exclusive(output_root / brief_id / "case.retry.result.json", row)
        case_rows.append(row)

    complete = sum(bool(row["eligible"]) for row in case_rows)
    summary = {
        "schema_version": "api2_fresh_case_retry_summary_v1",
        "adapter": adapter_name,
        "requested_briefs": len(brief_ids),
        "processed_briefs": len(case_rows),
        "complete": complete,
        "eligible": complete,
        "failed": len(case_rows) - complete,
        "fresh_chances_per_case": fresh_chances,
        "stopped_early": False,
        "results": case_rows,
        "source_outputs_modified": False,
    }
    write_json_exclusive(output_root / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--adapter", choices=tuple(recovery.ADAPTERS), required=True)
    parser.add_argument("--brief-ids", type=parse_ids, required=True)
    parser.add_argument("--fresh-chances", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "check":
            report = check(args.adapter, args.brief_ids, args.fresh_chances)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.output_dir is None:
            raise ValueError("--output-dir is required for run")
        summary = run(
            args.adapter,
            args.brief_ids,
            args.output_dir.expanduser().resolve(),
            args.fresh_chances,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["failed"] == 0 else 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
