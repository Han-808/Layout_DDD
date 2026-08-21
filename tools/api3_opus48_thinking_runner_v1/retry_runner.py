#!/usr/bin/env python3
"""Retry selected schema-invalid Opus 4.8 High generation briefs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Iterable


RUNNER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_ROOT.parents[1]
ADAPTER_PATH = RUNNER_ROOT / "generation_runner.py"
SOURCE_FULL10_ROOT = (
    REPO_ROOT
    / "Support"
    / "artifacts"
    / "outputs"
    / "e2e_scenegen_repro"
    / "runs"
    / "api3_claude_opus_4_8_high_paired10_r1"
)
RETRY_BRIEF_IDS = (
    "brief_00",
    "brief_01",
    "brief_03",
    "brief_05",
    "brief_06",
    "brief_09",
)


def load_adapter() -> Any:
    spec = importlib.util.spec_from_file_location(
        "api3_opus48_high_retry_adapter", ADAPTER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load adapter: {ADAPTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selected_briefs(
    adapter: Any, brief_ids: tuple[str, ...]
) -> list[dict[str, Any]]:
    runner = adapter.configure_core()
    briefs = runner._load_briefs(adapter.CORE_ROOT / "briefs.json")
    by_id = {str(brief["brief_id"]): brief for brief in briefs}
    if not brief_ids:
        raise ValueError("at least one retry brief ID is required")
    if len(brief_ids) != len(set(brief_ids)):
        raise ValueError("retry brief IDs must not contain duplicates")
    outside_frozen_set = [
        brief_id for brief_id in brief_ids if brief_id not in RETRY_BRIEF_IDS
    ]
    if outside_frozen_set:
        raise ValueError(f"brief IDs are outside the frozen failure set: {outside_frozen_set}")
    missing = [brief_id for brief_id in brief_ids if brief_id not in by_id]
    if missing:
        raise ValueError(f"unknown retry brief IDs: {missing}")
    return [by_id[brief_id] for brief_id in brief_ids]


def validate_source_failures(brief_ids: tuple[str, ...]) -> None:
    if not SOURCE_FULL10_ROOT.is_dir():
        raise FileNotFoundError(f"source full10 output is missing: {SOURCE_FULL10_ROOT}")
    observed: list[str] = []
    for brief_id in brief_ids:
        result_path = SOURCE_FULL10_ROOT / brief_id / "case.result.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"source case result is missing: {result_path}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        status = str(result.get("status") or "")
        if "schema_invalid" not in status:
            raise ValueError(
                f"source {brief_id} is not a schema-invalid case: {status}"
            )
        observed.append(brief_id)
    if tuple(observed) != brief_ids:
        raise ValueError(f"unexpected source failure set: {observed}")


def check(brief_ids: tuple[str, ...]) -> dict[str, Any]:
    validate_source_failures(brief_ids)
    adapter = load_adapter()
    runner = adapter.configure_core()
    runner.RetrieverAdapter(adapter.CORE_ROOT)
    model = runner._load_model_config(adapter.MODELS_PATH, adapter.MODEL_KEY)
    briefs = selected_briefs(adapter, brief_ids)
    if model.reasoning_effort != "high":
        raise ValueError("retry runner requires reasoning_effort=high")
    if model.preserved_thinking is not True:
        raise ValueError("retry runner requires preserved_thinking=true")
    request = adapter._request_value(
        model=model,
        system_prompt="Return only the final answer.",
        user_value={"check": True},
    )
    if request.get("thinking") != {"type": "adaptive"}:
        raise ValueError("retry request does not enable adaptive thinking")
    if request.get("reasoning_effort") != "high":
        raise ValueError("retry request does not set reasoning_effort=high")
    return {
        "ok": True,
        "model_key": model.key,
        "selected_brief_ids": [brief["brief_id"] for brief in briefs],
        "thinking": request["thinking"],
        "reasoning_effort": model.reasoning_effort,
        "preserved_thinking": model.preserved_thinking,
        "max_tokens": model.max_tokens,
        "maximum_infrastructure_retries": model.max_infrastructure_retries,
        "maximum_attempts_per_stage": model.max_infrastructure_retries + 1,
    }


def run_retry(output_root: Path, brief_ids: tuple[str, ...]) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    validate_source_failures(brief_ids)
    adapter = load_adapter()
    runner = adapter.configure_core()
    retriever = runner.RetrieverAdapter(adapter.CORE_ROOT)
    model = runner._load_model_config(adapter.MODELS_PATH, adapter.MODEL_KEY)
    briefs = selected_briefs(adapter, brief_ids)
    route = adapter.provider_route()
    retry_policy = adapter.RetryPolicy(
        max_infrastructure_retries=model.max_infrastructure_retries,
        retryable_transport_statuses=frozenset({"transport_failure"}),
        retryable_http_statuses=runner.RETRYABLE_HTTP_STATUSES,
        retry_delay_seconds=model.retry_delay_seconds,
        retry_ambiguous_timeouts=False,
        continue_after_case_failure=True,
    )
    execution_policy = {
        "schema_version": "api3_opus48_high_selected_retry_policy_v1",
        "source_full10_root": str(SOURCE_FULL10_ROOT),
        "selected_brief_ids": list(brief_ids),
        "selection_rule": "schema_invalid_in_source_full10",
        "case_failure_policy": "record_and_continue_next_brief",
        "thinking": {"type": "adaptive"},
        "reasoning_effort": model.reasoning_effort,
        "preserved_thinking": model.preserved_thinking,
        "max_tokens": model.max_tokens,
        "maximum_infrastructure_retries": model.max_infrastructure_retries,
        "maximum_attempts_per_stage": model.max_infrastructure_retries + 1,
        "schema_invalid_retry_within_case": False,
        "preflight_requires_reasoning_signal": False,
        "preflight_records_reasoning_signal_diagnostic": True,
        "retry_runner_sha256": runner.sha256_file(Path(__file__).resolve()),
    }
    spec = adapter.GenerationRunSpec(
        provider_key=route.key,
        model_key=model.key,
        wire_model=model.wire_model,
        ordered_brief_ids=brief_ids,
        briefs_path=adapter.CORE_ROOT / "briefs.json",
        models_path=adapter.MODELS_PATH,
        output_root=output_root,
        retry_policy=retry_policy,
        execution_policy=execution_policy,
        summary_schema_version="api3_opus48_high_selected_retry_summary_v1",
        summary_extra={
            "source_full10_root": str(SOURCE_FULL10_ROOT),
            "thinking": {"type": "adaptive"},
            "reasoning_effort": model.reasoning_effort,
        },
        source_manifest=adapter._runner_source_manifest(),
    )

    def progress(record: dict[str, Any]) -> None:
        if record.get("event") != "case_terminal":
            return
        print(
            json.dumps(
                {
                    "brief_id": record["brief_id"],
                    "status": record["status"],
                    "eligible": record["eligible"],
                    "continued_after_case": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    summary, _ = adapter.FrozenTwoStageOrchestrator(runner, route).run(
        spec=spec,
        model=model,
        briefs=briefs,
        retriever=retriever,
        progress=progress,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--brief-id", action="append", dest="brief_ids")
    run = subparsers.add_parser("run")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--brief-id", action="append", dest="brief_ids")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    brief_ids = tuple(args.brief_ids or RETRY_BRIEF_IDS)
    try:
        if args.command == "check":
            print(json.dumps(check(brief_ids), ensure_ascii=False, sort_keys=True))
            return 0
        summary = run_retry(args.output_dir.expanduser().resolve(), brief_ids)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["failed"] == 0 else 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
