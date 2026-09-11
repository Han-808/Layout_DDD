#!/usr/bin/env python3
"""Select a registered floor-plan evaluator; describe by default, run only with --run.

This repository-only entry point keeps different evaluator implementations in
separate Python processes. It never replaces an already running evaluator.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluator_release_tools import (  # noqa: E402
    EvaluatorReleaseError, REGISTRY_PATH, read_json, relative_file,
    verify_nonrect_core, verify_snapshot,
)


def resolve_release(repo: Path, mode: str) -> dict[str, Any]:
    registry = read_json(repo / REGISTRY_PATH)
    baseline_id = registry["current"][mode]
    baseline = registry["baselines"][baseline_id]
    if baseline["mode"] != mode:
        raise EvaluatorReleaseError("Registry mode mismatch")
    result: dict[str, Any] = {
        "mode": mode, "baseline_id": baseline_id, "versions": baseline["versions"],
        "execution_available": False, "limitations": baseline.get("limitations", []),
    }
    if mode == "single_room":
        result.update(verify_snapshot(repo, baseline_id))
        source_root = Path(result["source_root"])
        entry = relative_file(source_root, "scripts/run_camera_cal_scene_level.py")
        result.update(entrypoint=str(entry), source_root=str(source_root), execution_available=True)
    elif mode == "non_rectangular_multi_room":
        core = verify_nonrect_core(repo)
        if core["source_commit"] != baseline["code_identity"]["commit"]:
            raise EvaluatorReleaseError("Nonrect baseline/core mismatch")
        entry = relative_file(repo, "scripts/run_complicated_combined142_hardened.py")
        sealed = repo / "Support/artifacts/releases/complicated_eval_combined142_v1/run_combined.py"
        result.update(core_verification=core, entrypoint=str(entry), source_root=str(repo),
                      execution_available=sealed.is_file(),
                      missing_requirement=None if sealed.is_file() else "sealed_combined142_release")
    else:
        result["missing_requirement"] = "verified_rectangular_multi_room_historical_source"
    return result


def run_release(release: dict[str, Any], arguments: list[str]) -> int:
    if not release["execution_available"]:
        raise EvaluatorReleaseError(
            f"Cannot run {release['baseline_id']}: {release.get('missing_requirement')}. "
            "No other mode or current checkout will be substituted."
        )
    if not arguments:
        raise EvaluatorReleaseError("Pass explicit runner arguments after --; no evaluation started")
    environment = dict(os.environ)
    # A fresh interpreter is required: changing sys.path inside a process that
    # already imported benchmark would mix two implementations in sys.modules.
    environment["PYTHONPATH"] = str(Path(release["source_root"]) / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", release["entrypoint"], *arguments]
    print(f"Evaluator baseline: {release['baseline_id']}", flush=True)
    return subprocess.call(command, env=environment)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "single_room", "multi_room", "non_rectangular_multi_room"))
    parser.add_argument("--run", action="store_true", help="Explicitly invoke the selected runner")
    parser.add_argument("runner_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        release = resolve_release(ROOT, args.mode)
        arguments = args.runner_args[1:] if args.runner_args[:1] == ["--"] else args.runner_args
        if not args.run:
            if arguments:
                raise EvaluatorReleaseError("Runner arguments require --run")
            print(json.dumps(release, indent=2))
            return 0
        return run_release(release, arguments)
    except (EvaluatorReleaseError, KeyError, OSError, ValueError) as exc:
        print(f"Evaluator selection refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
