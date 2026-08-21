"""Top-level run orchestration for the camera-cal scene evaluator.

The compatibility script owns dependency wiring and public symbols.  This
module contains only the historical ``main`` call sequence and standard
library data handling; it never imports the script or Evaluation Campaign.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RunIO:
    """Filesystem, clock, and canonical JSON helpers for one run."""

    atomic_write_json: Callable[[Path, Any], None]
    utc_now: Callable[[], str]
    monotonic: Callable[[], float]
    json_sha256: Callable[[Any], str]


@dataclass(frozen=True)
class RunPlanning:
    """CLI, discovery, plan, and endpoint-input helpers."""

    parse_args: Callable[[], Any]
    effective_model_route: Callable[[], dict[str, Any]]
    normalize_metric_selection: Callable[[Any], tuple[str, ...]]
    discover_cases: Callable[..., list[dict[str, Any]]]
    renderer_config_from_args: Callable[..., dict[str, Any]]
    resolved_control: Callable[[], Any]
    build_experiment_plan: Callable[..., dict[str, Any]]
    endpoint_preflight_image: Callable[[dict[str, Any]], Path]


@dataclass(frozen=True)
class RunExecution:
    """External execution and reporting helpers for one run."""

    progress_factory: Callable[..., Any]
    endpoint_preflight: Callable[..., dict[str, Any]]
    endpoint_preflight_error_type: type[BaseException]
    abort_signal_factory: Callable[[], Any]
    run_case: Callable[..., dict[str, Any]]
    record_case_failure: Callable[..., dict[str, Any]]
    run_cases_parallel: Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]]]]
    build_summary: Callable[..., dict[str, Any]]
    api_usage_summary: Callable[[list[dict[str, Any]]], dict[str, Any]]


@dataclass(frozen=True)
class RunConstants:
    """Frozen schema/layer values written by the historical runner."""

    runner_schema_version: str
    l1_layer: str
    l2_layer: str
    l3_layer: str
    l4_layer: str


@dataclass(frozen=True)
class RunOrchestratorDeps:
    """The four explicit dependency groups required by :func:`run_main`."""

    io: RunIO
    planning: RunPlanning
    execution: RunExecution
    constants: RunConstants


def run_main(*, deps: RunOrchestratorDeps) -> None:
    """Execute the complete camera-cal run orchestration sequence."""

    args = deps.planning.parse_args()
    route = deps.planning.effective_model_route()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    grouping_config_path = args.grouping_config.expanduser().resolve()
    blender_bin = args.blender_bin.expanduser().resolve()
    metrics = deps.planning.normalize_metric_selection(args.metric)
    cases = deps.planning.discover_cases(
        dataset_root,
        case_ids=args.case_id,
        max_cases=args.max_cases,
    )
    if not grouping_config_path.is_file():
        raise FileNotFoundError(
            f"grouping config does not exist: {grouping_config_path}"
        )
    if not blender_bin.is_file():
        raise FileNotFoundError(f"Blender executable does not exist: {blender_bin}")

    output_root.mkdir(parents=True, exist_ok=True)
    progress = deps.execution.progress_factory(
        output_root / "progress.jsonl",
        terminal=args.terminal_progress,
    )
    renderer_config = deps.planning.renderer_config_from_args(
        args,
        blender_bin=blender_bin,
    )
    control = deps.planning.resolved_control()
    experiment = deps.planning.build_experiment_plan(
        dataset_root=dataset_root,
        output_root=output_root,
        grouping_config_path=grouping_config_path,
        route=route,
        metrics=metrics,
        functional_group_local_granularity=(
            args.functional_group_local_granularity
        ),
        functional_group_local_evidence_policy=(
            args.functional_group_local_evidence_policy
        ),
        deduction_multiplier=args.deduction_multiplier,
        cases=cases,
        renderer_config=renderer_config,
        control=control.to_dict(),
        max_workers=args.max_workers,
        endpoint_preflight_attempts=args.endpoint_preflight_attempts,
        endpoint_preflight_timeout_seconds=(
            args.endpoint_preflight_timeout_seconds
        ),
        resume=args.resume,
        continue_on_error=args.continue_on_error,
        export_audit_graphs=args.export_audit_graphs,
        l3_only=args.l3_only,
    )
    deps.io.atomic_write_json(output_root / "experiment_plan.json", experiment)

    started = deps.io.monotonic()
    run_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_manifest = {
        "schema_version": deps.constants.runner_schema_version,
        "status": "endpoint_preflight",
        "started_at": deps.io.utc_now(),
        "completed_at": None,
        "elapsed_seconds": None,
        "experiment_plan_sha256": deps.io.json_sha256(experiment),
        "source_prompt_used": False,
        "layers_executed": [deps.constants.l3_layer]
        if args.l3_only
        else [deps.constants.l1_layer, deps.constants.l3_layer],
        "layers_not_executed": [
            deps.constants.l1_layer,
            deps.constants.l2_layer,
            deps.constants.l4_layer,
        ]
        if args.l3_only
        else [deps.constants.l2_layer, deps.constants.l4_layer],
        "recovery_mode": "l3_only" if args.l3_only else None,
        "cases": [],
        "progress_path": str(progress.path),
        "api_usage": deps.execution.api_usage_summary([]),
        "endpoint_preflight_path": str(
            (output_root / "endpoint_preflight.json").resolve()
        ),
    }
    deps.io.atomic_write_json(output_root / "run_manifest.json", run_manifest)
    preflight_image = deps.planning.endpoint_preflight_image(cases[0])
    progress.emit(
        "endpoint_preflight_started",
        attempts=args.endpoint_preflight_attempts,
        concurrency=min(args.max_workers, args.endpoint_preflight_attempts),
        model=route["model"],
    )
    try:
        endpoint_preflight = deps.execution.endpoint_preflight(
            endpoint=str(route["endpoint"]),
            model_id=str(route["model"]),
            api_key_env=str(route["api_key_env"]),
            image_path=preflight_image,
            attempts=args.endpoint_preflight_attempts,
            concurrency=min(
                args.max_workers,
                args.endpoint_preflight_attempts,
            ),
            timeout_seconds=args.endpoint_preflight_timeout_seconds,
            min_request_interval_seconds=float(
                route.get("min_request_interval_seconds") or 0.0
            ),
        )
    except deps.execution.endpoint_preflight_error_type as exc:
        endpoint_preflight = exc.report
        deps.io.atomic_write_json(
            output_root / "endpoint_preflight.json",
            endpoint_preflight,
        )
        elapsed = deps.io.monotonic() - started
        run_manifest.update(
            status="endpoint_preflight_failed",
            completed_at=deps.io.utc_now(),
            elapsed_seconds=elapsed,
            endpoint_preflight=deepcopy(endpoint_preflight),
        )
        deps.io.atomic_write_json(output_root / "run_manifest.json", run_manifest)
        progress.emit(
            "endpoint_preflight_failed",
            error_type=type(exc).__name__,
            fatal_route_configuration=bool(
                endpoint_preflight.get("fatal_route_configuration")
            ),
            completed_attempts=endpoint_preflight.get(
                "completed_attempts"
            ),
            attempts_required=endpoint_preflight.get(
                "attempts_required"
            ),
        )
        print(
            json.dumps(
                {
                    "status": "endpoint_preflight_failed",
                    "report_path": str(
                        (output_root / "endpoint_preflight.json").resolve()
                    ),
                    "fatal_route_configuration": endpoint_preflight.get(
                        "fatal_route_configuration"
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
        raise SystemExit(2) from exc
    deps.io.atomic_write_json(
        output_root / "endpoint_preflight.json",
        endpoint_preflight,
    )
    run_manifest.update(
        status="running",
        endpoint_preflight=deepcopy(endpoint_preflight),
    )
    deps.io.atomic_write_json(output_root / "run_manifest.json", run_manifest)
    progress.emit(
        "endpoint_preflight_completed",
        attempts=endpoint_preflight["attempts_required"],
        api_invocations=endpoint_preflight["api_invocations"],
        model=route["model"],
    )
    progress.emit(
        "run_started",
        case_count=len(cases),
        metrics=list(metrics),
        max_workers=args.max_workers,
        l3_only=args.l3_only,
        output_root=str(output_root),
    )

    model_route_abort_signal = deps.execution.abort_signal_factory()
    case_kwargs = {
        "dataset_root": dataset_root,
        "output_root": output_root,
        "grouping_config_path": grouping_config_path,
        "route": route,
        "metrics": metrics,
        "functional_group_local_granularity": (
            args.functional_group_local_granularity
        ),
        "functional_group_local_evidence_policy": (
            args.functional_group_local_evidence_policy
        ),
        "deduction_multiplier": args.deduction_multiplier,
        "renderer_config": renderer_config,
        "control_config": control.to_dict(),
        "resume": args.resume,
        "export_audit_graphs": args.export_audit_graphs,
        "l3_only": args.l3_only,
        "progress": progress,
        "model_route_abort_signal": model_route_abort_signal,
    }
    if args.max_workers == 1:
        for index, case in enumerate(cases, start=1):
            progress.emit(
                "case_started",
                case_id=str(case["case_id"]),
                case_index=index,
                case_count=len(cases),
            )
            try:
                record = deps.execution.run_case(case=case, **case_kwargs)
            except Exception as exc:
                failure = deps.execution.record_case_failure(
                    case=case,
                    output_root=output_root,
                    error=exc,
                )
                failures.append(failure)
                run_records.append(failure)
                progress.emit(
                    "case_failed",
                    case_id=str(case["case_id"]),
                    case_index=index,
                    case_count=len(cases),
                    error_type=failure["error_type"],
                )
                if not args.continue_on_error:
                    break
            else:
                run_records.append(record)
                progress.emit(
                    "case_completed",
                    case_id=str(case["case_id"]),
                    case_index=index,
                    case_count=len(cases),
                    status=record["status"],
                    elapsed_seconds=round(
                        float(record.get("elapsed_seconds") or 0.0),
                        3,
                    ),
                    api_usage=record.get("api_usage"),
                )
            if model_route_abort_signal.is_set():
                break
    else:
        for index, case in enumerate(cases, start=1):
            progress.emit(
                "case_queued",
                case_id=str(case["case_id"]),
                case_index=index,
                case_count=len(cases),
            )
        parallel_records, parallel_failures = deps.execution.run_cases_parallel(
            cases=cases,
            case_kwargs=case_kwargs,
            output_root=output_root,
            progress=progress,
            max_workers=args.max_workers,
            continue_on_error=args.continue_on_error,
            model_route_abort_signal=model_route_abort_signal,
        )
        run_records.extend(parallel_records)
        failures.extend(parallel_failures)

    if model_route_abort_signal.is_set():
        route_abort = model_route_abort_signal.report()
        failures.append(
            {
                "case_id": "__run__",
                "status": "failed",
                "reason": "permanent_model_route_configuration_failure",
                **route_abort,
            }
        )
        progress.emit(
            "run_model_route_aborted",
            error_type=route_abort.get("error_type"),
        )

    ordered_records = sorted(
        run_records,
        key=lambda item: str(item.get("case_id") or ""),
    )
    elapsed = deps.io.monotonic() - started
    summary = deps.execution.build_summary(
        case_records=ordered_records,
        metrics=metrics,
        elapsed_seconds=elapsed,
    )
    deps.io.atomic_write_json(output_root / "summary.json", summary)
    run_manifest.update(
        status="failed" if failures else "complete",
        completed_at=deps.io.utc_now(),
        elapsed_seconds=elapsed,
        cases=ordered_records,
        summary_path=str((output_root / "summary.json").resolve()),
        api_usage=deepcopy(summary["api_usage"]),
    )
    deps.io.atomic_write_json(output_root / "run_manifest.json", run_manifest)
    progress.emit(
        "run_completed",
        status=run_manifest["status"],
        elapsed_seconds=round(elapsed, 3),
        totals=summary["totals"],
        api_usage=summary["api_usage"],
    )
    print(json.dumps(summary["totals"], indent=2), flush=True)
    if failures:
        raise SystemExit(1)


__all__ = [
    "RunConstants",
    "RunExecution",
    "RunIO",
    "RunOrchestratorDeps",
    "RunPlanning",
    "run_main",
]
