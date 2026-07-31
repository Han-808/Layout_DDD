#!/usr/bin/env python3
"""Prepare, render, and run the three-backend blind grouping experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.grouping_blind30_contracts import (
    ExperimentPaths,
    atomic_write_json,
    load_experiment_config,
    read_json,
)
from scripts.grouping_blind30_dataset import (
    materialize_all_cases,
    prepare_dataset,
)
from scripts.grouping_blind30_runtime import (
    render_all,
    run_grouping_backends,
)


STAGES = ("prepare", "render", "group", "review", "all")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs"
        / "experiments"
        / "grouping_blind30_gpt56_v1.yaml",
    )
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--blender-bin", type=Path, default=None)
    parser.add_argument("--asset-root", type=Path, default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument(
        "--allow-incomplete-review",
        action="store_true",
        help="Build a debugging review even when anonymous results failed.",
    )
    args = parser.parse_args()

    config, paths = load_experiment_config(
        args.config,
        repo_root=PROJECT_ROOT,
        output_override=args.output_root,
    )
    paths.output_root.mkdir(parents=True, exist_ok=True)
    dataset = prepare_dataset(config, paths, resume=args.resume)
    materialize_all_cases(
        config,
        paths,
        dataset,
        resume=args.resume,
    )
    if args.stage == "prepare":
        _print_summary(
            {
                "stage": "prepare",
                "dataset_manifest": str(paths.dataset_manifest),
                "case_count": len(dataset["cases"]),
                "output_root": str(paths.output_root),
            }
        )
        return

    render_failures: list[dict[str, Any]] = []
    if args.stage in {"render", "group", "all"}:
        render_failures = render_all(
            config,
            paths,
            dataset,
            resume=args.resume,
            continue_on_error=args.continue_on_error,
            blender_override=args.blender_bin,
            asset_override=args.asset_root,
        )
        if args.stage == "render":
            summary = write_run_summary(
                config,
                paths,
                dataset,
                render_failures=render_failures,
            )
            _print_summary(summary)
            if render_failures:
                raise SystemExit(1)
            return
        if render_failures and not args.continue_on_error:
            raise RuntimeError(
                f"{len(render_failures)} render cases failed; refusing to "
                "start grouping"
            )

    grouping_failures: list[dict[str, Any]] = []
    if args.stage in {"group", "all"}:
        grouping_failures = run_grouping_backends(
            config,
            paths,
            dataset,
            resume=args.resume,
            continue_on_error=args.continue_on_error,
            endpoint_override=args.endpoint,
            model_override=args.model,
            api_key_env_override=args.api_key_env,
        )

    if args.stage in {"review", "all"}:
        from scripts.build_grouping_blind30_review import build_review

        build_review(
            config=config,
            paths=paths,
            dataset=dataset,
            allow_incomplete=args.allow_incomplete_review,
        )

    summary = write_run_summary(
        config,
        paths,
        dataset,
        render_failures=render_failures,
        grouping_failures=grouping_failures,
    )
    _print_summary(summary)
    if render_failures or grouping_failures:
        raise SystemExit(1)


def write_run_summary(
    config: dict[str, Any],
    paths: ExperimentPaths,
    dataset: dict[str, Any],
    *,
    render_failures: list[dict[str, Any]] | None = None,
    grouping_failures: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    render_complete = 0
    backend_counts = {
        backend: {"resolved": 0, "failed": 0, "unresolved": 0}
        for backend in config["backends"]
    }
    for case in dataset["cases"]:
        case_root = paths.case_root(str(case["case_id"]))
        if (case_root / "render" / "experiment_render.json").is_file():
            render_complete += 1
        for backend in config["backends"]:
            result_path = (
                case_root / "grouping" / backend / "result.json"
            )
            if result_path.is_file():
                try:
                    record = read_json(result_path)
                except (OSError, ValueError, json.JSONDecodeError):
                    record = {}
                if record.get("status") == "complete":
                    backend_counts[backend]["resolved"] += 1
                    continue
            failure_path = (
                case_root / "grouping" / backend / "failure.json"
            )
            if failure_path.is_file():
                backend_counts[backend]["failed"] += 1
            else:
                backend_counts[backend]["unresolved"] += 1
    summary = {
        "experiment_id": config["_experiment_id"],
        "output_root": str(paths.output_root),
        "dataset_manifest": str(paths.dataset_manifest),
        "dataset_fingerprint": dataset["dataset_fingerprint"],
        "scene_count": len(dataset["cases"]),
        "render": {
            "resolved": render_complete,
            "failed": len(render_failures or []),
            "unresolved": len(dataset["cases"]) - render_complete,
        },
        "grouping": backend_counts,
        "failure_records": {
            "render": render_failures or [],
            "grouping": grouping_failures or [],
        },
        "review_index": str(paths.review_root / "index.html"),
        "review_server_command": (
            f"{paths.repo_root / '.venv/bin/python'} "
            f"{paths.repo_root / 'scripts/serve_grouping_blind30_review.py'} "
            f"--output-root {paths.output_root}"
        ),
    }
    atomic_write_json(paths.run_summary, summary)
    return summary


def _print_summary(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
