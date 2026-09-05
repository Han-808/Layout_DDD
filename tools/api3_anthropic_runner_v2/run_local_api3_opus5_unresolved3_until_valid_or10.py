#!/usr/bin/env python3
"""Retry three unresolved Opus 5 briefs until one valid result or ten chances each."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNNER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_ROOT.parents[1]
ARTIFACT_ROOT = REPO_ROOT / "Support/artifacts/outputs/e2e_scenegen_repro"
RUN_ROOT = ARTIFACT_ROOT / "runs"
MODEL_KEY = "api3-claude-opus-5"
MAIN_NAME = "api3_claude_opus_5_paired10_v1"
RETRY_STEM = "api3_claude_opus_5_failed_cases"
TARGET_BRIEFS = ("brief_04", "brief_06", "brief_08")
MAX_ADDITIONAL_OPPORTUNITIES = 10
FIRST_RETRY_ORDINAL = 7
CONTROLLER_NAME = "api3_claude_opus_5_unresolved3_until_valid_or10_v1"
CONTROLLER_ROOT = RUN_ROOT / CONTROLLER_NAME


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def is_valid_result(path: Path) -> bool:
    if not path.is_file():
        return False
    value = load_json(path)
    return (
        value.get("status") == "complete"
        and value.get("eligible_for_strict_one_shot_evaluation") is True
    )


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_history() -> None:
    main_root = RUN_ROOT / MAIN_NAME
    main_summary = main_root / "summary.json"
    if not main_summary.is_file() or load_json(main_summary).get("processed_briefs") != 10:
        raise ValueError(f"main run is not terminal 10/10: {main_root}")
    for retry_number in range(1, 7):
        retry_root = RUN_ROOT / f"{RETRY_STEM}_retry{retry_number}_v1"
        if not (retry_root / "summary.json").is_file():
            raise ValueError(f"prior retry is not terminal: {retry_root}")

    recovered: set[str] = set()
    for brief_index in range(10):
        brief_id = f"brief_{brief_index:02d}"
        candidates = [main_root / brief_id / "case.result.json"]
        candidates.extend(
            RUN_ROOT
            / f"{RETRY_STEM}_retry{retry_number}_v1"
            / brief_id
            / "case.result.json"
            for retry_number in range(1, 7)
        )
        if any(is_valid_result(path) for path in candidates):
            recovered.add(brief_id)
    unresolved = tuple(
        f"brief_{brief_index:02d}"
        for brief_index in range(10)
        if f"brief_{brief_index:02d}" not in recovered
    )
    if unresolved != TARGET_BRIEFS:
        raise ValueError(
            f"refusing changed unresolved set: expected={TARGET_BRIEFS} actual={unresolved}"
        )


def opportunity_root(brief_id: str, opportunity: int) -> Path:
    retry_ordinal = FIRST_RETRY_ORDINAL + opportunity - 1
    return RUN_ROOT / f"api3_claude_opus_5_{brief_id}_retry{retry_ordinal:02d}_v1"


def main() -> int:
    if not os.environ.get("API3_API_KEY"):
        raise ValueError("required credential environment variable is not set: API3_API_KEY")
    validate_history()
    if CONTROLLER_ROOT.exists():
        raise ValueError(f"refusing existing controller output: {CONTROLLER_ROOT}")
    for brief_id in TARGET_BRIEFS:
        for opportunity in range(1, MAX_ADDITIONAL_OPPORTUNITIES + 1):
            output_root = opportunity_root(brief_id, opportunity)
            if output_root.exists():
                raise ValueError(f"refusing existing opportunity output: {output_root}")

    CONTROLLER_ROOT.mkdir(parents=False, exist_ok=False)
    state: dict[str, Any] = {
        "schema_version": "api3_opus5_until_valid_or10_v1",
        "model_key": MODEL_KEY,
        "target_briefs": list(TARGET_BRIEFS),
        "target_valid_per_brief": 1,
        "max_additional_opportunities_per_brief": MAX_ADDITIONAL_OPPORTUNITIES,
        "scene_worker_concurrency": 3,
        "started_at": utc_now(),
        "completed_at": None,
        "terminal": False,
        "briefs": {
            brief_id: {
                "opportunities_used": 0,
                "valid_count": 0,
                "valid_output": None,
                "attempts": [],
            }
            for brief_id in TARGET_BRIEFS
        },
    }
    write_json_atomic(CONTROLLER_ROOT / "progress.json", state)
    state_lock = threading.Lock()

    def run_brief_worker(brief_id: str) -> None:
        brief_state = state["briefs"][brief_id]
        for opportunity in range(1, MAX_ADDITIONAL_OPPORTUNITIES + 1):
            output_root = opportunity_root(brief_id, opportunity)
            retry_ordinal = FIRST_RETRY_ORDINAL + opportunity - 1
            print(
                f"starting model={MODEL_KEY} brief={brief_id} "
                f"opportunity={opportunity}/{MAX_ADDITIONAL_OPPORTUNITIES} "
                f"retry_ordinal={retry_ordinal}",
                flush=True,
            )
            completed = subprocess.run(
                [
                    str(RUNNER_ROOT / "run_generation.sh"),
                    "run",
                    "--model",
                    MODEL_KEY,
                    "--output-dir",
                    str(output_root),
                    "--retriever-root",
                    str(RUNNER_ROOT),
                    "--brief-id",
                    brief_id,
                ],
                check=False,
            )
            if completed.returncode not in {0, 2}:
                with state_lock:
                    brief_state["controller_error"] = {
                        "opportunity": opportunity,
                        "returncode": completed.returncode,
                    }
                    write_json_atomic(CONTROLLER_ROOT / "progress.json", state)
                return

            result_path = output_root / brief_id / "case.result.json"
            if not result_path.is_file():
                with state_lock:
                    brief_state["controller_error"] = {
                        "opportunity": opportunity,
                        "reason": "terminal case.result.json missing",
                    }
                    write_json_atomic(CONTROLLER_ROOT / "progress.json", state)
                return

            result = load_json(result_path)
            valid = is_valid_result(result_path)
            with state_lock:
                brief_state["opportunities_used"] += 1
                brief_state["attempts"].append(
                    {
                        "opportunity": opportunity,
                        "retry_ordinal": retry_ordinal,
                        "output_root": str(output_root),
                        "status": result.get("status"),
                        "eligible": result.get("eligible_for_strict_one_shot_evaluation"),
                        "reason": result.get("reason"),
                        "valid": valid,
                    }
                )
                if valid:
                    brief_state["valid_count"] = 1
                    brief_state["valid_output"] = str(output_root)
                write_json_atomic(CONTROLLER_ROOT / "progress.json", state)
            if valid:
                print(
                    f"target recovered model={MODEL_KEY} brief={brief_id} "
                    f"opportunity={opportunity}",
                    flush=True,
                )
                return

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="opus5-scene") as executor:
        futures = {
            executor.submit(run_brief_worker, brief_id): brief_id
            for brief_id in TARGET_BRIEFS
        }
        for future in as_completed(futures):
            brief_id = futures[future]
            try:
                future.result()
            except Exception as exc:
                with state_lock:
                    state["briefs"][brief_id]["controller_error"] = {
                        "error_type": type(exc).__name__,
                        "reason": str(exc),
                    }
                    write_json_atomic(CONTROLLER_ROOT / "progress.json", state)

    state["terminal"] = True
    state["completed_at"] = utc_now()
    state["all_targets_recovered"] = all(
        state["briefs"][brief_id]["valid_count"] >= 1 for brief_id in TARGET_BRIEFS
    )
    state["unresolved_briefs"] = [
        brief_id
        for brief_id in TARGET_BRIEFS
        if state["briefs"][brief_id]["valid_count"] < 1
    ]
    write_json_atomic(CONTROLLER_ROOT / "summary.json", state)
    print(
        json.dumps(
            {
                "all_targets_recovered": state["all_targets_recovered"],
                "unresolved_briefs": state["unresolved_briefs"],
                "opportunities_used": {
                    brief_id: state["briefs"][brief_id]["opportunities_used"]
                    for brief_id in TARGET_BRIEFS
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if state["all_targets_recovered"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"controller error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(3)
