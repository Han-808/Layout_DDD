#!/usr/bin/env python3
"""Run selected API2 Kimi-K3 or GLM-5.3 briefs with timeout retries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Iterable


RUNNER_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RUNNER_ROOT.parents[1]
ADAPTERS = {
    "kimi": REPO_ROOT
    / "tools"
    / "api2_kimi_k3_runner_v1"
    / "kimi_k3_generation_runner.py",
    "glm": REPO_ROOT
    / "tools"
    / "api2_glm53_runner_v1"
    / "glm53_generation_runner.py",
}


def load_adapter(name: str) -> Any:
    path = ADAPTERS[name]
    spec = importlib.util.spec_from_file_location(
        f"api2_{name}_recovery_adapter", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def selected_briefs(adapter: Any, ids: tuple[str, ...]) -> list[dict[str, Any]]:
    runner = adapter.configure_core()
    briefs = runner._load_briefs(adapter.FROZEN_CORE_ROOT / "briefs.json")
    by_id = {str(brief["brief_id"]): brief for brief in briefs}
    if len(ids) != len(set(ids)):
        raise ValueError("brief IDs must not contain duplicates")
    missing = [brief_id for brief_id in ids if brief_id not in by_id]
    if missing:
        raise ValueError(f"unknown brief IDs: {missing}")
    return [by_id[brief_id] for brief_id in ids]


def check(adapter_name: str, ids: tuple[str, ...]) -> dict[str, Any]:
    adapter = load_adapter(adapter_name)
    runner = adapter.configure_core()
    model = runner._load_model_config(adapter.MODELS_PATH, adapter.MODEL_KEY)
    briefs = selected_briefs(adapter, ids)
    if model.max_infrastructure_retries != 3:
        raise ValueError("recovery runner requires exactly three retries")
    if not hasattr(runner, "RETRY_TRANSPORT_AMBIGUOUS"):
        raise RuntimeError("core runner lacks timeout-retry support")
    return {
        "ok": True,
        "adapter": adapter_name,
        "model": model.label,
        "brief_ids": [brief["brief_id"] for brief in briefs],
        "maximum_infrastructure_retries": model.max_infrastructure_retries,
        "maximum_attempts_per_stage": model.max_infrastructure_retries + 1,
        "retry_transport_ambiguous": True,
    }


def run_selected(
    adapter_name: str,
    ids: tuple[str, ...],
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    adapter = load_adapter(adapter_name)
    runner = adapter.configure_core()
    model = runner._load_model_config(adapter.MODELS_PATH, adapter.MODEL_KEY)
    if model.max_infrastructure_retries != 3:
        raise ValueError("recovery runner requires exactly three retries")
    briefs = selected_briefs(adapter, ids)
    retriever = runner.RetrieverAdapter(adapter.FROZEN_CORE_ROOT)
    briefs_path = adapter.FROZEN_CORE_ROOT / "briefs.json"
    route = adapter.provider_route()
    retry_policy = adapter.RetryPolicy(
        max_infrastructure_retries=model.max_infrastructure_retries,
        retryable_transport_statuses=frozenset(
            {"transport_failure", "transport_ambiguous"}
        ),
        retryable_http_statuses=runner.RETRYABLE_HTTP_STATUSES,
        retry_delay_seconds=model.retry_delay_seconds,
        retry_ambiguous_timeouts=True,
        continue_after_case_failure=True,
    )
    execution_policy = {
        "schema_version": "api2_selected_brief_timeout_recovery_v1",
        "adapter": adapter_name,
        "case_failure_policy": "record_and_continue_next_brief",
        "selected_brief_ids": list(ids),
        "maximum_infrastructure_retries": model.max_infrastructure_retries,
        "maximum_attempts_per_stage": model.max_infrastructure_retries + 1,
        "retryable_conditions": [
            "transport_failure",
            "transport_ambiguous",
            "http_429",
            "http_500",
        ],
        "timeout_retry_warning": (
            "An ambiguous timeout may have reached the provider; a retry can "
            "produce a duplicate upstream request."
        ),
    }
    spec = adapter.GenerationRunSpec(
        provider_key=route.key,
        model_key=model.key,
        wire_model=model.wire_model,
        ordered_brief_ids=ids,
        briefs_path=briefs_path,
        models_path=adapter.MODELS_PATH,
        output_root=output_root,
        retry_policy=retry_policy,
        execution_policy=execution_policy,
        summary_schema_version="api2_selected_brief_timeout_recovery_summary_v1",
        summary_extra={"adapter": adapter_name},
        source_manifest=adapter._source_manifest(),
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


def parse_ids(value: str) -> tuple[str, ...]:
    ids = tuple(item.strip() for item in value.split(",") if item.strip())
    if not ids:
        raise argparse.ArgumentTypeError("at least one brief ID is required")
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run"))
    parser.add_argument("--adapter", choices=tuple(ADAPTERS), required=True)
    parser.add_argument("--brief-ids", type=parse_ids, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "check":
            report = check(args.adapter, args.brief_ids)
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0
        if args.output_dir is None:
            raise ValueError("--output-dir is required for run")
        summary = run_selected(
            args.adapter,
            args.brief_ids,
            args.output_dir.expanduser().resolve(),
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 0 if summary["failed"] == 0 else 2
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
