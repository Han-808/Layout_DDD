#!/usr/bin/env python3
"""Run the promptless L1/L3 camera-cal experiment at scene level.

The frozen camera-cal cases contain generation prompts, but this experiment
does not evaluate prompt fidelity and never supplies those prompts to the
Judge. L1 remains scene-level deterministic evidence plus conditional VLM
adjudication. L3 runs the existing metric-specific scope policy, judges every
eligible group required by that metric, and preserves the evaluator's existing
scene-level aggregation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from benchmark.api.evaluation import run_evaluate  # noqa: E402
from benchmark.evaluator.generic_validity.mesh_geometry import (  # noqa: E402
    load_collision_geometry_manifest,
)
from benchmark.evaluator.profile import (  # noqa: E402
    DEFAULT_EVALUATION_PROFILE,
    L1,
    L2,
    L3,
    L4,
)
from benchmark.evaluator.scoring import (  # noqa: E402
    DEFAULT_DEDUCTION_MULTIPLIER,
    DEDUCTION_MULTIPLIER_METRICS,
    INTRINSIC_VALIDITY_PROFILE_ID,
    L3_METRIC_WEIGHTS,
    MIN_PUBLISHABLE_SCORE_COVERAGE,
)
from benchmark.evaluator.scene_quality.functional_ownership import (  # noqa: E402
    CROSS_METRIC_OWNERSHIP_AUDIT_VERSION,
    FUNCTIONAL_OWNERSHIP_LEDGER_VERSION,
)
from benchmark.evaluator.scene_quality.placement_checks import (  # noqa: E402
    PLACEMENT_CHECK_LEDGER_VERSION,
    PLACEMENT_CHECK_RESULT_VERSION,
)
from benchmark.models import (  # noqa: E402
    EndpointConfigurationError,
    EndpointStabilityPreflightError,
    OpenAICompatibleModel,
    run_endpoint_stability_preflight,
)
from benchmark.rendering import (  # noqa: E402
    CYCLES_DEVICES,
    RENDER_ENGINES,
    BlenderRenderer,
)
from benchmark.visual_judge import (  # noqa: E402
    CameraCandidatePreviewRenderer,
    CameraEvidenceProvider,
    CameraViewEvidenceRenderer,
    DeterministicLocalCameraSelector,
    FUNCTIONAL_PROBE_DEFAULT_UNITS,
    build_openai_compatible_camera_selector,
    build_openai_compatible_vlm_judge,
    resolve_vlm_evaluation_control,
)
from benchmark.visual_judge.l3_prompts import (  # noqa: E402
    L3_METRIC_PROMPT_VERSION,
)
from benchmark.visual_judge.functional_discovery import (  # noqa: E402
    FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION,
    FUNCTIONAL_DISCOVERY_SCHEMA_VERSION,
    FUNCTIONAL_RELATION_SCHEMA_VERSION,
)
from benchmark.visual_judge.placement_discovery import (  # noqa: E402
    PLACEMENT_DISCOVERY_SCHEMA_VERSION,
)
from benchmark.visual_judge.contracts import (  # noqa: E402
    response_schema_audit_from_exception,
)
from benchmark.visual_judge.graphs import (  # noqa: E402
    AUDIT_GRAPH_EXPORT_VERSION,
    export_case_audit_graphs,
)
from benchmark.camera_cal_scene_level import io as _runtime_io  # noqa: E402
from benchmark.camera_cal_scene_level import progress as _runtime_progress  # noqa: E402
from benchmark.camera_cal_scene_level import telemetry as _runtime_telemetry  # noqa: E402
from benchmark.camera_cal_scene_level import cli as _runtime_cli  # noqa: E402
from benchmark.camera_cal_scene_level import discovery as _runtime_discovery  # noqa: E402
from benchmark.camera_cal_scene_level import planning as _runtime_planning  # noqa: E402
from benchmark.camera_cal_scene_level import resume as _runtime_resume  # noqa: E402
from benchmark.camera_cal_scene_level import scheduling as _runtime_scheduling  # noqa: E402


RUNNER_SCHEMA_VERSION = "camera_cal_scene_level_runner_v9"
PLAN_SCHEMA_VERSION = "camera_cal_scene_level_plan_v2"
CASE_SCHEMA_VERSION = "camera_cal_scene_level_case_v5"
COMPARISON_SCHEMA_VERSION = "camera_cal_scene_comparison_v1"
SUMMARY_SCHEMA_VERSION = "camera_cal_scene_level_summary_v2"
PROGRESS_SCHEMA_VERSION = "camera_cal_scene_level_progress_v1"
API_CALL_SCHEMA_VERSION = "camera_cal_api_call_v1"
API_USAGE_SCHEMA_VERSION = "camera_cal_api_usage_v1"
GROUPING_COMPLETION_MAX_TOKENS = 3192
JUDGE_COMPLETION_MAX_TOKENS = 8192
CAMERA_SELECTOR_COMPLETION_MAX_TOKENS = 2048
L1_BINARY_FAILURE_POLICY = {
    "p0b_official_mode": False,
    "on_engineering_failure": "scene_unresolved_continue_l3_diagnostics",
    "binary_defects": "always_empty",
    "schema_repair_retry_count": 1,
}

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT / "Support" / "datasets" / "camera_cal_scenesets"
)
DEFAULT_GROUPING_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "grouping"
    / "vlm_visual_evidence_scope_v2.yaml"
)
DEFAULT_BLENDER_BIN = Path(
    os.environ.get(
        "BLENDER_BIN",
        "/Applications/Blender.app/Contents/MacOS/Blender",
    )
)

L1_METRICS = ("collision", "oob", "support")
ANNOTATED_L3_METRICS = (
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
    "functional_consistency",
    "semantic_placement_consistency",
)
CANONICAL_L3_METRICS = ANNOTATED_L3_METRICS
# Empty compatibility export: all annotated L3 metrics are benchmark metrics.
EXPERIMENTAL_L3_METRICS: tuple[str, ...] = ()
FUNCTIONAL_PROBE_IMPLEMENTATION_FILES = (
    "src/benchmark/evaluator/scene_quality/functional_acquisition.py",
    "src/benchmark/evaluator/scene_quality/functional_boundary_evidence.py",
    "src/benchmark/evaluator/scene_quality/cross_group_relations.py",
    "src/benchmark/evaluator/scene_quality/functional_geometry.py",
    "src/benchmark/evaluator/scene_quality/functional_checks.py",
    "src/benchmark/evaluator/scene_quality/functional_ownership.py",
    "src/benchmark/evaluator/scene_quality/functional_probe.py",
    "src/benchmark/evaluator/scene_quality/global_group_first.py",
    "src/benchmark/evaluator/scene_quality/group_scoped.py",
    "src/benchmark/evaluator/scene_quality/interfaces.py",
    "src/benchmark/evaluator/scene_quality/placement_checks.py",
    "src/benchmark/rendering/camera_pose.py",
    "src/benchmark/visual_judge/functional_discovery.py",
    "src/benchmark/visual_judge/functional_discovery_contract.py",
    "src/benchmark/visual_judge/functional_discovery_validation.py",
    "src/benchmark/visual_judge/functional_evidence.py",
    "src/benchmark/visual_judge/openai_camera_selector.py",
    "src/benchmark/visual_judge/openai_compatible.py",
    "src/benchmark/visual_judge/orchestration/controller.py",
    "src/benchmark/visual_judge/placement_discovery.py",
    "src/benchmark/visual_judge/graphs/evaluation.py",
    "src/benchmark/visual_judge/graphs/evaluation_projection.py",
    "src/benchmark/visual_judge/graphs/exporter.py",
    "src/benchmark/visual_judge/render_views.py",
    "src/benchmark/visual_judge/response_repair.py",
    "src/benchmark/visual_judge/usable_surface.py",
)

_CASE_ID_PATTERN = re.compile(r"[NS]\d{3}")
_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_prompt_tokens",
    "reasoning_tokens",
)


class ProgressReporter(_runtime_progress.ProgressReporter):
    """Compatibility facade over the extracted progress implementation."""

    def __init__(self, path: Path, *, terminal: bool = True) -> None:
        super().__init__(
            path,
            terminal=terminal,
            clock=lambda: utc_now(),
            formatter=lambda record: _format_progress_record(record),
        )


class ModelRouteAbortSignal(_runtime_scheduling.ModelRouteAbortSignal):
    """Compatibility facade over the extracted route circuit breaker."""

    def __init__(self) -> None:
        super().__init__(
            error_formatter=lambda error: _bounded_error(error),
            abort_error_factory=EndpointConfigurationError,
        )


class APICallTracker(_runtime_telemetry.APICallTracker):
    """Compatibility facade over the extracted API-call implementation."""

    def __init__(
        self,
        *,
        case_id: str,
        calls_path: Path,
        usage_path: Path,
        progress: ProgressReporter,
        model_route_abort_signal: ModelRouteAbortSignal | None = None,
    ) -> None:
        super().__init__(
            case_id=case_id,
            calls_path=calls_path,
            usage_path=usage_path,
            progress=progress,
            model_route_abort_signal=model_route_abort_signal,
            observed_model_factory=_ObservedChatModel,
            read_records=lambda path: read_api_call_records(path),
            write_json=lambda path, value: atomic_write_json(path, value),
            usage_summary=lambda records: api_usage_summary(records),
            clock=lambda: utc_now(),
            monotonic=lambda: time.monotonic(),
        )


class _ObservedChatModel:
    def __init__(
        self,
        model: Any,
        *,
        role: str,
        tracker: APICallTracker,
    ) -> None:
        self._model = model
        self._role = str(role)
        self._tracker = tracker

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def chat_messages(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> str:
        if self._tracker.model_route_abort_signal is not None:
            self._tracker.model_route_abort_signal.raise_if_set()
        call_type = str(kwargs.get("call_type") or "chat")
        call_number, call_id, started = self._tracker.begin_call(
            role=self._role,
            call_type=call_type,
            messages=messages,
        )
        try:
            result = self._model.chat_messages(messages, **kwargs)
        except Exception as exc:
            self._tracker.finish_call(
                call_number=call_number,
                call_id=call_id,
                role=self._role,
                call_type=call_type,
                started=started,
                request_metadata=_safe_request_metadata(self._model),
                error=exc,
            )
            if (
                isinstance(exc, EndpointConfigurationError)
                and self._tracker.model_route_abort_signal is not None
            ):
                self._tracker.model_route_abort_signal.trip(exc)
            raise
        self._tracker.finish_call(
            call_number=call_number,
            call_id=call_id,
            role=self._role,
            call_type=call_type,
            started=started,
            request_metadata=_safe_request_metadata(self._model),
            error=None,
        )
        return result


class _ObservedEvidenceProvider:
    def __init__(
        self,
        provider: Any,
        *,
        phase: str,
        case_id: str,
        progress: ProgressReporter,
    ) -> None:
        self._provider = provider
        self._phase = str(phase)
        self._case_id = str(case_id)
        self._progress = progress

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def __call__(self, request: dict[str, Any]) -> Any:
        return self._invoke(request)

    def provide_scene_quality_evidence(
        self,
        request: dict[str, Any],
    ) -> Any:
        return self._invoke(request)

    def _invoke(self, request: dict[str, Any]) -> Any:
        metric, group_id = _request_scope(request)
        started = time.monotonic()
        self._progress.emit(
            "evidence_render_started",
            case_id=self._case_id,
            phase=self._phase,
            metric=metric,
            group_id=group_id,
        )
        try:
            result = self._provider(request)
        except Exception as exc:
            self._progress.emit(
                "evidence_render_failed",
                case_id=self._case_id,
                phase=self._phase,
                metric=metric,
                group_id=group_id,
                duration_seconds=round(
                    max(0.0, time.monotonic() - started),
                    3,
                ),
                error_type=type(exc).__name__,
            )
            raise
        usage = getattr(self._provider, "last_call_usage", None)
        usage = usage if isinstance(usage, dict) else {}
        self._progress.emit(
            "evidence_render_completed",
            case_id=self._case_id,
            phase=self._phase,
            metric=metric,
            group_id=group_id,
            duration_seconds=round(
                max(0.0, time.monotonic() - started),
                3,
            ),
            evidence_count=_evidence_count(result),
            cache_hit=usage.get("cache_hit"),
            internal_selector_calls=usage.get("selector_calls"),
            camera_actions=usage.get("camera_actions"),
        )
        return result


class _ObservedRenderer:
    def __init__(
        self,
        renderer: Any,
        *,
        phase: str,
        case_id: str,
        progress: ProgressReporter,
    ) -> None:
        self._renderer = renderer
        self._phase = str(phase)
        self._case_id = str(case_id)
        self._progress = progress

    def __getattr__(self, name: str) -> Any:
        return getattr(self._renderer, name)

    def render(self, request: Any) -> Any:
        metric = str(getattr(request, "metric", "") or "unknown")
        context = getattr(request, "context", None)
        context = context if isinstance(context, dict) else {}
        group_scope = context.get("group_scope")
        group_id = (
            str(group_scope.get("group_id") or "scene")
            if isinstance(group_scope, dict)
            else "scene"
        )
        started = time.monotonic()
        self._progress.emit(
            "repair_render_started",
            case_id=self._case_id,
            phase=self._phase,
            metric=metric,
            group_id=group_id,
        )
        try:
            result = self._renderer.render(request)
        except Exception as exc:
            self._progress.emit(
                "repair_render_failed",
                case_id=self._case_id,
                phase=self._phase,
                metric=metric,
                group_id=group_id,
                duration_seconds=round(
                    max(0.0, time.monotonic() - started),
                    3,
                ),
                error_type=type(exc).__name__,
            )
            raise
        self._progress.emit(
            "repair_render_completed",
            case_id=self._case_id,
            phase=self._phase,
            metric=metric,
            group_id=group_id,
            duration_seconds=round(
                max(0.0, time.monotonic() - started),
                3,
            ),
            evidence_count=_evidence_count(result),
        )
        return result


def run_cases_parallel(
    *,
    cases: list[dict[str, Any]],
    case_kwargs: dict[str, Any],
    output_root: Path,
    progress: ProgressReporter,
    max_workers: int,
    continue_on_error: bool,
    model_route_abort_signal: ModelRouteAbortSignal | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return _runtime_scheduling.run_cases_parallel(
        cases=cases,
        case_kwargs=case_kwargs,
        output_root=output_root,
        progress=progress,
        max_workers=max_workers,
        continue_on_error=continue_on_error,
        run_case_fn=run_case,
        failure_recorder=record_case_failure,
        cancellation_recorder=record_case_cancellation,
        model_route_abort_signal=model_route_abort_signal,
    )


def main() -> None:
    args = parse_args()
    route = effective_model_route()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    grouping_config_path = args.grouping_config.expanduser().resolve()
    blender_bin = args.blender_bin.expanduser().resolve()
    metrics = normalize_metric_selection(args.metric)
    cases = discover_cases(
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
    progress = ProgressReporter(
        output_root / "progress.jsonl",
        terminal=args.terminal_progress,
    )
    renderer_config = renderer_config_from_args(args, blender_bin=blender_bin)
    control = resolved_control()
    experiment = build_experiment_plan(
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
    atomic_write_json(output_root / "experiment_plan.json", experiment)

    started = time.monotonic()
    run_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "endpoint_preflight",
        "started_at": utc_now(),
        "completed_at": None,
        "elapsed_seconds": None,
        "experiment_plan_sha256": json_sha256(experiment),
        "source_prompt_used": False,
        "layers_executed": [L3] if args.l3_only else [L1, L3],
        "layers_not_executed": [L1, L2, L4] if args.l3_only else [L2, L4],
        "recovery_mode": "l3_only" if args.l3_only else None,
        "cases": [],
        "progress_path": str(progress.path),
        "api_usage": api_usage_summary([]),
        "endpoint_preflight_path": str(
            (output_root / "endpoint_preflight.json").resolve()
        ),
    }
    atomic_write_json(output_root / "run_manifest.json", run_manifest)
    preflight_image = _endpoint_preflight_image(cases[0])
    progress.emit(
        "endpoint_preflight_started",
        attempts=args.endpoint_preflight_attempts,
        concurrency=min(args.max_workers, args.endpoint_preflight_attempts),
        model=route["model"],
    )
    try:
        endpoint_preflight = run_endpoint_stability_preflight(
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
    except EndpointStabilityPreflightError as exc:
        endpoint_preflight = exc.report
        atomic_write_json(
            output_root / "endpoint_preflight.json",
            endpoint_preflight,
        )
        elapsed = time.monotonic() - started
        run_manifest.update(
            status="endpoint_preflight_failed",
            completed_at=utc_now(),
            elapsed_seconds=elapsed,
            endpoint_preflight=deepcopy(endpoint_preflight),
        )
        atomic_write_json(output_root / "run_manifest.json", run_manifest)
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
    atomic_write_json(
        output_root / "endpoint_preflight.json",
        endpoint_preflight,
    )
    run_manifest.update(
        status="running",
        endpoint_preflight=deepcopy(endpoint_preflight),
    )
    atomic_write_json(output_root / "run_manifest.json", run_manifest)
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

    model_route_abort_signal = ModelRouteAbortSignal()
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
                record = run_case(case=case, **case_kwargs)
            except Exception as exc:
                failure = record_case_failure(
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
        parallel_records, parallel_failures = run_cases_parallel(
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
    elapsed = time.monotonic() - started
    summary = build_summary(
        case_records=ordered_records,
        metrics=metrics,
        elapsed_seconds=elapsed,
    )
    atomic_write_json(output_root / "summary.json", summary)
    run_manifest.update(
        status="failed" if failures else "complete",
        completed_at=utc_now(),
        elapsed_seconds=elapsed,
        cases=ordered_records,
        summary_path=str((output_root / "summary.json").resolve()),
        api_usage=deepcopy(summary["api_usage"]),
    )
    atomic_write_json(output_root / "run_manifest.json", run_manifest)
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


def _planning_dependencies() -> _runtime_planning.PlanningDependencies:
    return _runtime_planning.PlanningDependencies(
        utc_now=lambda: utc_now(),
        file_sha256=lambda path: file_sha256(path),
        safe_route_manifest=lambda route: safe_route_manifest(route),
        resolve_vlm_evaluation_control=resolve_vlm_evaluation_control,
        model_class=OpenAICompatibleModel,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _runtime_cli._parse_args_impl(
        argv,
        description=__doc__,
        default_dataset_root=DEFAULT_DATASET_ROOT,
        default_grouping_config=DEFAULT_GROUPING_CONFIG,
        default_blender_bin=DEFAULT_BLENDER_BIN,
        annotated_l3_metrics=ANNOTATED_L3_METRICS,
        render_engines=RENDER_ENGINES,
        cycles_devices=CYCLES_DEVICES,
        default_deduction_multiplier=DEFAULT_DEDUCTION_MULTIPLIER,
    )


def positive_int(value: str) -> int:
    return _runtime_cli.positive_int(value)


def positive_float(value: str) -> float:
    return _runtime_cli.positive_float(value)


def effective_model_route(
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> dict[str, Any]:
    return _runtime_planning.effective_model_route(environ)


def normalize_metric_selection(values: Iterable[str]) -> tuple[str, ...]:
    return _runtime_planning.normalize_metric_selection(values)


def discover_cases(
    dataset_root: Path,
    *,
    case_ids: Iterable[str] = (),
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    return _runtime_discovery._discover_cases_impl(
        dataset_root,
        case_ids=case_ids,
        max_cases=max_cases,
        read_json_fn=read_json,
        case_paths_fn=case_paths,
    )


def case_paths(
    case_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    return _runtime_discovery.case_paths(case_root, manifest)


def _endpoint_preflight_image(case: dict[str, Any]) -> Path:
    return _runtime_discovery._endpoint_preflight_image_impl(
        case,
        read_json_fn=read_json,
        case_paths_fn=case_paths,
    )


def renderer_config_from_args(
    args: argparse.Namespace,
    *,
    blender_bin: Path,
) -> dict[str, Any]:
    return _runtime_planning.renderer_config_from_args(
        args, blender_bin=blender_bin
    )


def resolved_control() -> Any:
    return _runtime_planning.resolved_control(
        dependencies=_planning_dependencies()
    )


def build_experiment_plan(
    *,
    dataset_root: Path,
    output_root: Path,
    grouping_config_path: Path,
    route: dict[str, Any],
    metrics: tuple[str, ...],
    functional_group_local_granularity: str,
    functional_group_local_evidence_policy: str = "shared_group_bank",
    deduction_multiplier: float = DEFAULT_DEDUCTION_MULTIPLIER,
    cases: list[dict[str, Any]],
    renderer_config: dict[str, Any],
    control: dict[str, Any],
    max_workers: int,
    endpoint_preflight_attempts: int,
    endpoint_preflight_timeout_seconds: int,
    resume: bool,
    continue_on_error: bool,
    export_audit_graphs: bool = False,
    l3_only: bool = False,
) -> dict[str, Any]:
    return _runtime_planning.build_experiment_plan(
        dataset_root=dataset_root,
        output_root=output_root,
        grouping_config_path=grouping_config_path,
        route=route,
        metrics=metrics,
        functional_group_local_granularity=functional_group_local_granularity,
        functional_group_local_evidence_policy=(
            functional_group_local_evidence_policy
        ),
        deduction_multiplier=deduction_multiplier,
        cases=cases,
        renderer_config=renderer_config,
        control=control,
        max_workers=max_workers,
        endpoint_preflight_attempts=endpoint_preflight_attempts,
        endpoint_preflight_timeout_seconds=(
            endpoint_preflight_timeout_seconds
        ),
        resume=resume,
        continue_on_error=continue_on_error,
        export_audit_graphs=export_audit_graphs,
        l3_only=l3_only,
        dependencies=_planning_dependencies(),
    )


def run_case(
    *,
    case: dict[str, Any],
    dataset_root: Path,
    output_root: Path,
    grouping_config_path: Path,
    route: dict[str, Any],
    metrics: tuple[str, ...],
    renderer_config: dict[str, Any],
    control_config: dict[str, Any],
    resume: bool,
    functional_group_local_granularity: str = "per_check",
    functional_group_local_evidence_policy: str = "shared_group_bank",
    deduction_multiplier: float = DEFAULT_DEDUCTION_MULTIPLIER,
    export_audit_graphs: bool = False,
    l3_only: bool = False,
    progress: ProgressReporter | None = None,
    model_route_abort_signal: ModelRouteAbortSignal | None = None,
) -> dict[str, Any]:
    del dataset_root
    case_id = str(case["case_id"])
    progress = progress or ProgressReporter(
        output_root / "progress.jsonl",
        terminal=False,
    )
    source_root = Path(str(case["case_root"])).resolve()
    case_manifest = read_json(source_root / "case_manifest.json")
    paths = case_paths(source_root, case_manifest)
    annotation = read_json(paths["annotation"])
    scene = read_json(paths["scene"])
    grouping_config = read_yaml_object(grouping_config_path)
    fingerprint = case_input_fingerprint(
        case=case,
        case_manifest=case_manifest,
        paths=paths,
        route=route,
        metrics=metrics,
        functional_group_local_granularity=(
            functional_group_local_granularity
        ),
        functional_group_local_evidence_policy=(
            functional_group_local_evidence_policy
        ),
        deduction_multiplier=deduction_multiplier,
        grouping_config=grouping_config,
        renderer_config=renderer_config,
        control_config=control_config,
        l3_only=l3_only,
    )
    case_out = output_root / "cases" / case_id
    existing_manifest_path = case_out / "case_run_manifest.json"
    if existing_manifest_path.is_file():
        existing = read_json(existing_manifest_path)
        if resume and resumable_case(
            existing,
            expected_fingerprint=fingerprint,
            case_out=case_out,
        ):
            audit_graph_export = _maybe_export_audit_graphs(
                enabled=export_audit_graphs,
                case_id=case_id,
                case_out=case_out,
                grouping_report=read_json(case_out / "grouping.json"),
                scene_quality_report=read_json(
                    case_out / "scene_quality_report.json"
                ),
                progress=progress,
            )
            api_calls_path = case_out / "api_calls.jsonl"
            api_usage = api_usage_summary(
                read_api_call_records(api_calls_path)
            )
            progress.emit(
                "case_resumed",
                case_id=case_id,
                elapsed_seconds=float(
                    existing.get("elapsed_seconds") or 0.0
                ),
                api_usage=api_usage,
            )
            return {
                "case_id": case_id,
                "status": "resumed",
                "input_fingerprint": fingerprint,
                "elapsed_seconds": float(existing.get("elapsed_seconds") or 0.0),
                "grouping_status": existing.get("grouping_status"),
                "l1_status": existing.get("l1_status"),
                "l3_status": existing.get("l3_status"),
                "final_decision_status": existing.get(
                    "final_decision_status"
                ),
                "l1_decision_status": existing.get(
                    "l1_decision_status",
                    existing.get("final_decision_status"),
                ),
                "l3_decision_status": existing.get(
                    "l3_decision_status",
                    "resolved"
                    if existing.get("l3_status") == "evaluated"
                    else None,
                ),
                "l3_unresolved_metrics": list(
                    existing.get("l3_unresolved_metrics") or []
                ),
                "l1_engineering_failure": bool(
                    existing.get("l1_engineering_failure")
                ),
                "l1_engineering_failure_count": int(
                    existing.get("l1_engineering_failure_count") or 0
                ),
                "binary_response_schema_validation": deepcopy(
                    existing.get("binary_response_schema_validation") or {}
                ),
                "scene_comparison_path": str(
                    (case_out / "scene_comparison.json").resolve()
                ),
                "scene_quality_report_path": str(
                    (case_out / "scene_quality_report.json").resolve()
                ),
                "l1_report_path": str((case_out / "l1_report.json").resolve()),
                "control_manifest_path": str(
                    (case_out / "control_manifest.json").resolve()
                ),
                "api_calls_path": str(api_calls_path.resolve()),
                "api_usage_path": str(
                    (case_out / "api_usage.json").resolve()
                ),
                "api_usage": api_usage,
                "audit_graph_export": audit_graph_export,
            }
        if resume:
            raise RuntimeError(
                f"{case_id} existing output fingerprint does not match; "
                "use a new --output-root"
            )
        raise FileExistsError(
            f"{case_id} output already exists; use --resume or a new --output-root"
        )

    case_out.mkdir(parents=True, exist_ok=True)
    api_tracker = APICallTracker(
        case_id=case_id,
        calls_path=case_out / "api_calls.jsonl",
        usage_path=case_out / "api_usage.json",
        progress=progress,
        model_route_abort_signal=model_route_abort_signal,
    )
    started = time.monotonic()
    case_run_manifest = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "elapsed_seconds": None,
        "input_fingerprint": fingerprint,
        "source_case_root": str(source_root),
        "source_case_read_only": True,
        "source_prompt_used": False,
        "model_route": safe_route_manifest(route),
        "selected_l3_metrics": list(metrics),
        "layers_executed": [L3] if l3_only else [L1, L3],
        "layers_not_executed": [L1, L2, L4] if l3_only else [L2, L4],
        "recovery_mode": "l3_only" if l3_only else None,
        "deduction_multiplier": deduction_multiplier,
        "l1_binary_failure_policy": deepcopy(
            L1_BINARY_FAILURE_POLICY
        ),
        "progress_path": str(progress.path),
        "api_calls_path": str(api_tracker.calls_path),
        "api_usage_path": str(api_tracker.usage_path),
        "api_usage": api_tracker.summary(),
        "audit_graph_export": {
            "enabled": bool(export_audit_graphs),
            "status": "pending" if export_audit_graphs else "disabled",
            "decision_authority": "none",
        },
    }
    atomic_write_json(existing_manifest_path, case_run_manifest)
    progress.emit(
        "case_setup_started",
        case_id=case_id,
        selected_l3_metrics=list(metrics),
    )

    collision_geometry = load_collision_geometry_manifest(
        paths["collision_geometry"]
    )
    collision_geometry["manifest_path"] = str(
        paths["collision_geometry"].resolve()
    )
    identity_legend = identity_legend_from_manifest(paths["render_manifest"])
    grouping_evidence = grouping_evidence_packet(
        paths=paths,
        identity_legend=identity_legend,
    )
    overview_evidence = {
        "global": [
            str(paths["perspective"].resolve()),
            str(paths["top"].resolve()),
        ]
    }

    judge_config = model_config(route, role="judge")
    selector_config = model_config(route, role="camera-selector")
    grouping_model = api_tracker.observe_model(
        build_grouping_model(route),
        role="grouping",
    )
    raw_judge = build_openai_compatible_vlm_judge(judge_config)
    if callable(
        getattr(getattr(raw_judge, "model", None), "chat_messages", None)
    ):
        raw_judge.model = api_tracker.observe_model(
            raw_judge.model,
            role="judge",
        )
    vlm_selector = build_openai_compatible_camera_selector(selector_config)
    if callable(
        getattr(getattr(vlm_selector, "model", None), "chat_messages", None)
    ):
        vlm_selector.model = api_tracker.observe_model(
            vlm_selector.model,
            role="camera_selector",
        )
    renderer = BlenderRenderer(**renderer_config)
    l1_provider = _ObservedEvidenceProvider(
        CameraEvidenceProvider(
            renderer=renderer,
            blend_file=paths["blend"],
            out_dir=case_out / "l1_camera",
            mode="auto",
            selector=None,
            max_views=2,
            max_steps=1,
            candidate_count=6,
            collision_overlay=True,
            collision_contour=True,
            collision_geometry=collision_geometry,
        ),
        phase="l1_initial_evidence",
        case_id=case_id,
        progress=progress,
    )
    l3_provider = _ObservedEvidenceProvider(
        CameraEvidenceProvider(
            renderer=renderer,
            blend_file=paths["blend"],
            out_dir=case_out / "l3_initial_camera",
            mode="visibility_ranked",
            selector=None,
            max_views=2,
            max_steps=1,
            candidate_count=6,
            collision_overlay=False,
            collision_contour=False,
            collision_geometry=collision_geometry,
            active_repair=False,
        ),
        phase="l3_initial_evidence",
        case_id=case_id,
        progress=progress,
    )
    functional_probe_provider = _ObservedEvidenceProvider(
        CameraEvidenceProvider(
            renderer=renderer,
            blend_file=paths["blend"],
            out_dir=case_out / "l3_functional_probes",
            mode="query_cov",
            selector=vlm_selector,
            max_views=1,
            max_steps=0,
            candidate_count=6,
            collision_overlay=False,
            collision_contour=False,
            collision_geometry=collision_geometry,
            active_repair=False,
            usable_surface_cache_dir=(
                output_root / "_usable_surface_cache"
            ),
        ),
        phase="l3_functional_probe",
        case_id=case_id,
        progress=progress,
    )
    deterministic_selector = DeterministicLocalCameraSelector(
        candidate_policy=l3_provider.candidate_policy
    )
    evidence_renderer = _ObservedRenderer(
        CameraViewEvidenceRenderer(
            renderer=renderer,
            blend_file=paths["blend"],
            out_dir=case_out / "repair_camera",
        ),
        phase="final_evidence",
        case_id=case_id,
        progress=progress,
    )
    preview_renderer = _ObservedRenderer(
        CameraCandidatePreviewRenderer(
            renderer=renderer,
            blend_file=paths["blend"],
            out_dir=case_out / "repair_camera",
        ),
        phase="candidate_preview",
        case_id=case_id,
        progress=progress,
    )

    progress.emit(
        "evaluation_started",
        case_id=case_id,
        layers=[L3] if l3_only else [L1, L3],
        metrics=list(metrics),
    )
    try:
        report = run_evaluate(
            scene=scene,
            out=case_out / "evaluation_report.json",
            scene_request=promptless_scene_request(scene, case),
            collision_geometry=collision_geometry,
            render_evidence=overview_evidence,
            grouping_visual_evidence=grouping_evidence,
            grouping_identity_legend=identity_legend,
            vlm_judge=raw_judge,
            grouping_model=grouping_model,
            evaluation_profile=(
                promptless_l3_only_profile()
                if l3_only
                else promptless_l1_l3_profile()
            ),
            scoring_profile_id=INTRINSIC_VALIDITY_PROFILE_ID,
            deduction_multiplier=deduction_multiplier,
            p0b_official_mode=L1_BINARY_FAILURE_POLICY["p0b_official_mode"],
            p0b_local_view_provider=l1_provider,
            l3_initial_evidence_provider=l3_provider,
            functional_evidence_planner=vlm_selector,
            functional_probe_evidence_provider=(
                functional_probe_provider
            ),
            camera_selector=vlm_selector,
            deterministic_camera_selector=deterministic_selector,
            vlm_camera_selector=vlm_selector,
            evidence_renderer=evidence_renderer,
            candidate_preview_renderer=preview_renderer,
            scene_quality_config=scene_quality_config(
                metrics,
                functional_group_local_granularity=(
                    functional_group_local_granularity
                ),
                functional_group_local_evidence_policy=(
                    functional_group_local_evidence_policy
                ),
            ),
            asset_policy=camera_cal_asset_policy(),
            specification_contract=None,
            authorized_deviations=[],
            vlm_evaluation_control=control_config,
        )
    except Exception as exc:
        api_usage = api_tracker.summary()
        atomic_write_json(api_tracker.usage_path, api_usage)
        progress.emit(
            "evaluation_failed",
            case_id=case_id,
            duration_seconds=round(
                max(0.0, time.monotonic() - started),
                3,
            ),
            error_type=type(exc).__name__,
            api_usage=api_usage,
        )
        raise
    api_usage = api_tracker.summary()
    progress.emit(
        "evaluation_completed",
        case_id=case_id,
        duration_seconds=round(
            max(0.0, time.monotonic() - started),
            3,
        ),
        api_usage=api_usage,
    )
    grouping_report = deepcopy(report["reports"]["object_grouping"])
    l1_report = deepcopy(report["layer_reports"][L1])
    l3_report = deepcopy(report["reports"]["scene_quality"])
    control_manifest = deepcopy(
        report["evaluation_config"]["vlm_evaluation_control"]
    )
    l1_failures = collect_l1_engineering_failures(l1_report)
    schema_validation = binary_schema_validation_summary(l1_report)
    l1_decision_status = (
        "not_executed"
        if l3_only
        else "resolved"
        if l1_report.get("status") == "evaluated" and not l1_failures
        else "infrastructure_failure"
        if l1_failures
        else "unresolved"
    )
    l3_resolution = l3_resolution_audit(
        l3_report,
        metrics=metrics,
    )
    l3_decision_status = str(l3_resolution["status"])
    final_decision_status = (
        l3_decision_status
        if l3_only
        else "resolved"
        if l1_decision_status == "resolved"
        and l3_decision_status == "resolved"
        else "infrastructure_failure"
        if "infrastructure_failure"
        in {l1_decision_status, l3_decision_status}
        else "unresolved"
    )
    diagnostic_reason = (
        "l1_engineering_failure"
        if l1_failures
        else "l3_infrastructure_failure"
        if l3_decision_status == "infrastructure_failure"
        else "l1_unresolved"
        if l1_decision_status == "unresolved"
        else "l3_unresolved"
        if l3_decision_status == "unresolved"
        else None
    )
    comparison = build_scene_comparison(
        case_id=case_id,
        annotation=annotation,
        scene_quality_report=l3_report,
        metrics=metrics,
    )
    comparison["diagnostic_only"] = (
        final_decision_status != "resolved"
    )
    comparison["diagnostic_reason"] = diagnostic_reason
    l1_diagnostics = {
        "policy": deepcopy(L1_BINARY_FAILURE_POLICY),
        "recovery_mode": "l3_only" if l3_only else None,
        "l1_executed": not l3_only,
        "final_decision_status": l1_decision_status,
        "l1_decision_status": l1_decision_status,
        "combined_final_decision_status": final_decision_status,
        "engineering_failure_count": len(l1_failures),
        "engineering_failures": l1_failures,
        "response_schema_validation": schema_validation,
        "l3_diagnostics_completed": True,
    }
    report["runner_outcome"] = {
        "final_decision_status": final_decision_status,
        "l1_decision_status": l1_decision_status,
        "l3_decision_status": l3_decision_status,
        "l3_resolution_audit": deepcopy(l3_resolution),
        "l1_engineering_failure": bool(l1_failures),
        "l3_results_are_diagnostic_only": (
            final_decision_status != "resolved"
        ),
        "l1_diagnostics_path": str(
            (case_out / "l1_diagnostics.json").resolve()
        ),
    }
    elapsed = time.monotonic() - started

    atomic_write_json(case_out / "evaluation_report.json", report)
    atomic_write_json(case_out / "grouping.json", grouping_report)
    atomic_write_json(case_out / "l1_report.json", l1_report)
    atomic_write_json(case_out / "l1_diagnostics.json", l1_diagnostics)
    atomic_write_json(case_out / "scene_quality_report.json", l3_report)
    atomic_write_json(case_out / "scene_comparison.json", comparison)
    atomic_write_json(case_out / "control_manifest.json", control_manifest)
    audit_graph_export = _maybe_export_audit_graphs(
        enabled=export_audit_graphs,
        case_id=case_id,
        case_out=case_out,
        grouping_report=grouping_report,
        scene_quality_report=l3_report,
        progress=progress,
    )
    case_run_manifest.update(
        status="complete",
        completed_at=utc_now(),
        elapsed_seconds=elapsed,
        grouping_status=grouping_report.get("status"),
        l1_status=l1_report.get("status"),
        l3_status=l3_report.get("status"),
        final_decision_status=final_decision_status,
        l1_decision_status=l1_decision_status,
        l3_decision_status=l3_decision_status,
        l3_unresolved_metrics=list(
            l3_resolution.get("unresolved_metrics") or []
        ),
        l3_infrastructure_failure_metrics=list(
            l3_resolution.get("infrastructure_failure_metrics") or []
        ),
        l3_partial_coverage_metrics=list(
            l3_resolution.get("partial_coverage_metrics") or []
        ),
        l3_below_coverage_threshold_metrics=list(
            l3_resolution.get("below_coverage_threshold_metrics") or []
        ),
        l1_engineering_failure=bool(l1_failures),
        l1_engineering_failure_count=len(l1_failures),
        binary_response_schema_validation=schema_validation,
        scoring_profile=deepcopy(report.get("scoring_profile")),
        canonical_object_denominator=deepcopy(
            report.get("canonical_object_denominator")
        ),
        benchmark_score=report.get("benchmark_score"),
        benchmark_score_100=report.get("benchmark_score_100"),
        benchmark_score_status=report.get("benchmark_score_status"),
        scoring_reliability=deepcopy(
            report.get("scoring_reliability")
        ),
        api_usage=api_usage,
        audit_graph_export=audit_graph_export,
        paths={
            "evaluation_report": str(
                (case_out / "evaluation_report.json").resolve()
            ),
            "grouping": str((case_out / "grouping.json").resolve()),
            "l1_report": str((case_out / "l1_report.json").resolve()),
            "l1_diagnostics": str(
                (case_out / "l1_diagnostics.json").resolve()
            ),
            "scene_quality_report": str(
                (case_out / "scene_quality_report.json").resolve()
            ),
            "scene_comparison": str(
                (case_out / "scene_comparison.json").resolve()
            ),
            "control_manifest": str(
                (case_out / "control_manifest.json").resolve()
            ),
            "api_calls": str(api_tracker.calls_path),
            "api_usage": str(api_tracker.usage_path),
            "audit_graph_manifest": (
                str((case_out / "audit_graphs" / "manifest.json").resolve())
                if export_audit_graphs
                else None
            ),
        },
    )
    atomic_write_json(existing_manifest_path, case_run_manifest)
    return {
        "case_id": case_id,
        "status": "complete",
        "input_fingerprint": fingerprint,
        "elapsed_seconds": elapsed,
        "grouping_status": grouping_report.get("status"),
        "l1_status": l1_report.get("status"),
        "l3_status": l3_report.get("status"),
        "final_decision_status": final_decision_status,
        "l1_decision_status": l1_decision_status,
        "l3_decision_status": l3_decision_status,
        "l3_unresolved_metrics": list(
            l3_resolution.get("unresolved_metrics") or []
        ),
        "l3_infrastructure_failure_metrics": list(
            l3_resolution.get("infrastructure_failure_metrics") or []
        ),
        "l3_partial_coverage_metrics": list(
            l3_resolution.get("partial_coverage_metrics") or []
        ),
        "l3_below_coverage_threshold_metrics": list(
            l3_resolution.get("below_coverage_threshold_metrics") or []
        ),
        "l1_engineering_failure": bool(l1_failures),
        "l1_engineering_failure_count": len(l1_failures),
        "binary_response_schema_validation": schema_validation,
        "scene_comparison_path": str(
            (case_out / "scene_comparison.json").resolve()
        ),
        "scene_quality_report_path": str(
            (case_out / "scene_quality_report.json").resolve()
        ),
        "l1_report_path": str((case_out / "l1_report.json").resolve()),
        "control_manifest_path": str(
            (case_out / "control_manifest.json").resolve()
        ),
        "api_calls_path": str(api_tracker.calls_path),
        "api_usage_path": str(api_tracker.usage_path),
        "api_usage": api_usage,
        "audit_graph_export": audit_graph_export,
    }


def _maybe_export_audit_graphs(
    *,
    enabled: bool,
    case_id: str,
    case_out: Path,
    grouping_report: dict[str, Any],
    scene_quality_report: dict[str, Any],
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "status": "disabled",
            "decision_authority": "none",
        }
    result = export_case_audit_graphs(
        case_id=case_id,
        grouping_report=grouping_report,
        scene_quality_report=scene_quality_report,
        output_dir=case_out / "audit_graphs",
    )
    if progress is not None:
        progress.emit(
            "audit_graph_export_completed",
            case_id=case_id,
            status=result["status"],
            relation_candidate_count=(
                (result.get("relation_candidate_graph") or {}).get(
                    "candidate_count"
                )
            ),
            evaluation_query_graph_count=len(
                result.get("evaluation_query_graphs") or []
            ),
            decision_authority="none",
        )
    return {
        "enabled": True,
        "status": result["status"],
        "schema_version": result["schema_version"],
        "decision_authority": "none",
        "manifest_path": str(
            (case_out / "audit_graphs" / "manifest.json").resolve()
        ),
        "relation_candidate_count": (
            (result.get("relation_candidate_graph") or {}).get(
                "candidate_count"
            )
        ),
        "evaluation_query_graph_count": len(
            result.get("evaluation_query_graphs") or []
        ),
        "error_type": result.get("error_type"),
        "error": result.get("error"),
    }


def model_config(route: dict[str, Any], *, role: str) -> dict[str, Any]:
    return _runtime_planning.model_config(route, role=role)


def build_grouping_model(route: dict[str, Any]) -> OpenAICompatibleModel:
    return _runtime_planning.build_grouping_model(
        route, dependencies=_planning_dependencies()
    )


def promptless_scene_request(
    scene: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    return {
        "request_id": scene.get("request_id"),
        "instruction": "",
        "scene_type": scene.get("scene_type") or case.get("scene_type"),
        "prompt_granularity": "fine_grained",
        "metadata": {
            "promptless_camera_cal": True,
            "generation_prompt_withheld_from_evaluator": True,
        },
    }


def promptless_l1_l3_profile() -> dict[str, Any]:
    profile = deepcopy(DEFAULT_EVALUATION_PROFILE)
    profile["layer_weights"] = {
        L1: 0.30,
        L2: 0.0,
        L3: 0.70,
        L4: 0.0,
    }
    profile[L2]["enabled"] = False
    for metric in profile[L2]["metrics"].values():
        metric["enabled"] = False
        metric["weight"] = 0.0
    return profile


def promptless_l3_only_profile() -> dict[str, Any]:
    """Return an audit-explicit recovery profile that executes only L3."""

    profile = promptless_l1_l3_profile()
    profile["layer_weights"] = {
        L1: 0.0,
        L2: 0.0,
        L3: 1.0,
        L4: 0.0,
    }
    profile[L1]["enabled"] = False
    for metric in profile[L1]["metrics"].values():
        metric["enabled"] = False
        metric["weight"] = 0.0
    return profile


def scene_quality_config(
    metrics: tuple[str, ...],
    *,
    functional_group_local_granularity: str = "per_check",
    functional_group_local_evidence_policy: str = "shared_group_bank",
) -> dict[str, Any]:
    if functional_group_local_granularity not in {
        "per_check",
        "batched",
    }:
        raise ValueError(
            "functional_group_local_granularity must be exactly "
            "'per_check' or 'batched'"
        )
    if functional_group_local_evidence_policy not in {
        "isolated_episode",
        "shared_group_bank",
    }:
        raise ValueError(
            "functional_group_local_evidence_policy must be exactly "
            "'isolated_episode' or 'shared_group_bank'"
        )
    if (
        functional_group_local_evidence_policy == "shared_group_bank"
        and functional_group_local_granularity != "per_check"
    ):
        raise ValueError(
            "shared_group_bank requires "
            "functional_group_local_granularity='per_check'"
        )
    selected = set(metrics)
    return {
        "enabled": True,
        "metrics": {
            metric: {
                "enabled": metric in selected,
                "weight": (
                    L3_METRIC_WEIGHTS[metric]
                    if metric in selected
                    else 0.0
                ),
                **(
                    {
                        "group_local_check_granularity": (
                            functional_group_local_granularity
                        ),
                        "group_local_evidence_policy": (
                            functional_group_local_evidence_policy
                        ),
                        "group_local_active_window_max_images": 6,
                    }
                    if metric == "functional_consistency"
                    else {}
                ),
                **(
                    {
                        "residual_global_review": {
                            "enabled": True,
                            "placement_weight": 0.20,
                            "image_budget": 3,
                            "allowed_check_types": [
                                "scene_zone",
                                "contextual_anchor",
                            ],
                        }
                    }
                    if metric == "semantic_placement_consistency"
                    else {}
                ),
            }
            for metric in ANNOTATED_L3_METRICS
        },
    }


def camera_cal_asset_policy() -> dict[str, Any]:
    return {
        "mode": "fixed_catalog_selection",
        "identity_owner": "benchmark",
        "category_selection_owner": "generator",
        "scale_owner": "generator",
        "appearance_owner": "generator",
        "arrangement_owner": "generator",
        "source": "camera_cal_experiment_protocol",
    }


def identity_legend_from_manifest(path: Path) -> dict[str, str]:
    return _runtime_discovery._identity_legend_from_manifest_impl(
        path, read_json_fn=read_json
    )


def grouping_evidence_packet(
    *,
    paths: dict[str, Path],
    identity_legend: dict[str, str],
) -> list[dict[str, Any]]:
    return _runtime_discovery.grouping_evidence_packet(
        paths=paths, identity_legend=identity_legend
    )


def case_input_fingerprint(
    *,
    case: dict[str, Any],
    case_manifest: dict[str, Any],
    paths: dict[str, Path],
    route: dict[str, Any],
    metrics: tuple[str, ...],
    functional_group_local_granularity: str,
    functional_group_local_evidence_policy: str = "shared_group_bank",
    deduction_multiplier: float = DEFAULT_DEDUCTION_MULTIPLIER,
    grouping_config: dict[str, Any],
    renderer_config: dict[str, Any],
    control_config: dict[str, Any],
    l3_only: bool = False,
) -> str:
    critical = case_manifest.get("critical_artifact_hashes")
    critical = critical if isinstance(critical, dict) else {}
    prompt_path = (
        PROJECT_ROOT / "src" / "benchmark" / "visual_judge" / "l3_prompts.py"
    )
    prompt_context_path = (
        PROJECT_ROOT
        / "src"
        / "benchmark"
        / "evaluator"
        / "scene_quality"
        / "prompt_context.py"
    )
    scoring_paths = (
        PROJECT_ROOT / "src" / "benchmark" / "evaluator" / "scoring.py",
        PROJECT_ROOT / "src" / "benchmark" / "scoring_profiles.py",
    )
    return json_sha256(
        {
            "runner_schema_version": RUNNER_SCHEMA_VERSION,
            "case_id": case["case_id"],
            "semantic_content_fingerprint": case.get(
                "semantic_content_fingerprint"
            ),
            "canonical_scene_sha256": file_sha256(paths["scene"]),
            "annotation_sha256": file_sha256(paths["annotation"]),
            "blend_sha256": critical.get("blend"),
            "evidence_sha256": {
                name: file_sha256(paths[name])
                for name in ("perspective", "top", "identity")
            },
            "collision_geometry_manifest_sha256": file_sha256(
                paths["collision_geometry"]
            ),
            "grouping_config": grouping_config,
            "model_route": safe_route_manifest(route),
            "model_completion_budgets": {
                "grouping": GROUPING_COMPLETION_MAX_TOKENS,
                "judge": JUDGE_COMPLETION_MAX_TOKENS,
                "camera_selector": CAMERA_SELECTOR_COMPLETION_MAX_TOKENS,
            },
            "selected_l3_metrics": list(metrics),
            "recovery_mode": "l3_only" if l3_only else None,
            "deduction_multiplier": deduction_multiplier,
            "source_prompt_used": False,
            "metric_scoped_public_context": {
                "default_fields": {
                    "style_consistency": ["room_type"],
                    "object_pairing_consistency": ["room_type"],
                },
                "full_generation_instruction_used": False,
            },
            "l3_metric_prompt_version": L3_METRIC_PROMPT_VERSION,
            "l3_prompt_source_sha256": file_sha256(prompt_path),
            "l3_prompt_context_source_sha256": file_sha256(
                prompt_context_path
            ),
            "scoring_implementation_sha256": {
                str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
                for path in scoring_paths
            },
            "functional_probe_implementation_sha256": {
                relative: file_sha256(PROJECT_ROOT / relative)
                for relative in FUNCTIONAL_PROBE_IMPLEMENTATION_FILES
            },
            "profile": (
                promptless_l3_only_profile()
                if l3_only
                else promptless_l1_l3_profile()
            ),
            "scene_quality_config": scene_quality_config(
                metrics,
                functional_group_local_granularity=(
                    functional_group_local_granularity
                ),
                functional_group_local_evidence_policy=(
                    functional_group_local_evidence_policy
                ),
            ),
            "asset_policy": camera_cal_asset_policy(),
            "l1_binary_failure_policy": deepcopy(
                L1_BINARY_FAILURE_POLICY
            ),
            "control": control_config,
            "renderer": {
                key: value
                for key, value in renderer_config.items()
                if key != "blender_bin"
            },
        }
    )


def resumable_case(
    manifest: dict[str, Any],
    *,
    expected_fingerprint: str,
    case_out: Path,
) -> bool:
    return _runtime_resume.resumable_case(
        manifest,
        expected_fingerprint=expected_fingerprint,
        case_out=case_out,
    )


def build_scene_comparison(
    *,
    case_id: str,
    annotation: dict[str, Any],
    scene_quality_report: dict[str, Any],
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    annotation_metrics = annotation.get("metrics")
    annotation_metrics = (
        annotation_metrics if isinstance(annotation_metrics, dict) else {}
    )
    report_metrics = scene_quality_report.get("metrics")
    report_metrics = report_metrics if isinstance(report_metrics, dict) else {}
    comparisons: dict[str, dict[str, Any]] = {}
    for metric in metrics:
        human = annotation_metrics.get(metric)
        human = human if isinstance(human, dict) else {}
        model = report_metrics.get(metric)
        model = model if isinstance(model, dict) else {}
        expected = (
            "invalid"
            if human.get("anomaly") is True
            else "valid"
            if human.get("anomaly") is False
            else "unresolved"
        )
        predicted = metric_prediction(model)
        unclear = human.get("unclear") is True
        evaluated = predicted in {"valid", "invalid"}
        included = bool(not unclear and expected in {"valid", "invalid"} and evaluated)
        human_object_ids = _ordered_object_ids(
            human.get("affected_object_ids")
        )
        model_object_ids = _model_anomaly_object_ids(model)
        anomaly_level = _anomaly_object_comparison(
            expected=expected,
            predicted=predicted,
            unclear=unclear,
            human_object_ids=human_object_ids,
            model_object_ids=model_object_ids,
        )
        comparisons[metric] = {
            "human": {
                "expected": expected,
                "anomaly": human.get("anomaly"),
                "unclear": unclear,
                "affected_object_ids": list(human_object_ids),
                "issue": human.get("issue"),
            },
            "model": {
                "prediction": predicted,
                "status": model.get("status"),
                "score": model.get("score"),
                "reason": model.get("reason"),
                "eligible_group_count": (
                    (model.get("coverage") or {}).get("eligible_count")
                    if isinstance(model.get("coverage"), dict)
                    else None
                ),
                "resolved_group_count": (
                    (model.get("coverage") or {}).get("resolved_count")
                    if isinstance(model.get("coverage"), dict)
                    else None
                ),
                "judge_call_count": model.get("judge_call_count"),
                "anomaly_object_ids": list(model_object_ids),
                "group_results": deepcopy(model.get("group_results") or []),
            },
            "included_in_accuracy": included,
            "matches": predicted == expected if included else None,
            "anomaly_level": anomaly_level,
        }
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "case_id": case_id,
        "source_prompt_used": False,
        "comparison_scope": "scene_level_metric_verdict",
        "comparison_scopes": [
            "scene_level_metric_verdict",
            "anomaly_object_attribution",
        ],
        "metrics": comparisons,
    }


def metric_prediction(report: dict[str, Any]) -> str:
    status = str(report.get("status") or "")
    if (
        status in {"failed", "error", "infrastructure_failure"}
        or report.get("terminal_state") == "infrastructure_failure"
    ):
        return "infrastructure_failure"
    if status != "evaluated":
        return "unresolved"
    judgement = report.get("judgement")
    verdict = (
        judgement.get("verdict")
        if isinstance(judgement, dict)
        else None
    )
    if verdict in {"valid", "invalid"}:
        return str(verdict)
    verdict_score = report.get("verdict_score")
    if verdict_score == 1.0:
        return "valid"
    if verdict_score == 0.0:
        return "invalid"
    score = report.get("score")
    if score == 1.0:
        return "valid"
    if score == 0.0:
        return "invalid"
    return "unresolved"


def _model_anomaly_object_ids(
    report: dict[str, Any],
) -> tuple[str, ...]:
    findings = report.get("final_object_findings")
    if isinstance(findings, list):
        values = [
            item.get("object_id")
            for item in findings
            if isinstance(item, dict)
        ]
        normalized = _ordered_object_ids(values)
        if normalized:
            return normalized
    claims = report.get("final_defect_claims")
    if not isinstance(claims, list):
        judgement = report.get("judgement")
        judgement = judgement if isinstance(judgement, dict) else {}
        claims = judgement.get("defects")
    return _ordered_object_ids(
        target_id
        for claim in claims or []
        if isinstance(claim, dict)
        for target_id in claim.get("target_ids") or []
    )


def _ordered_object_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Any = [value]
    else:
        values = value
    if not isinstance(values, (list, tuple, set)) and not hasattr(
        values,
        "__iter__",
    ):
        return ()
    return tuple(
        dict.fromkeys(
            str(item).strip()
            for item in values
            if isinstance(item, (str, int))
            and str(item).strip()
        )
    )


def _anomaly_object_comparison(
    *,
    expected: str,
    predicted: str,
    unclear: bool,
    human_object_ids: tuple[str, ...],
    model_object_ids: tuple[str, ...],
) -> dict[str, Any]:
    human_set = set(human_object_ids)
    model_set = set(model_object_ids)
    included = bool(
        not unclear
        and expected == "invalid"
        and predicted in {"valid", "invalid"}
        and bool(human_set)
    )
    true_positive = tuple(
        item for item in human_object_ids if item in model_set
    )
    false_negative = tuple(
        item for item in human_object_ids if item not in model_set
    )
    false_positive = tuple(
        item for item in model_object_ids if item not in human_set
    )
    precision = (
        len(true_positive) / len(model_set)
        if included and model_set
        else None
    )
    recall = (
        len(true_positive) / len(human_set)
        if included and human_set
        else None
    )
    return {
        "scope": "anomaly_object_attribution",
        "included_in_accuracy": included,
        "human_object_ids": list(human_object_ids),
        "model_object_ids": list(model_object_ids),
        "true_positive_object_ids": list(true_positive),
        "false_negative_object_ids": list(false_negative),
        "false_positive_object_ids": list(false_positive),
        "precision": precision,
        "recall": recall,
        "exact_match": (
            human_set == model_set if included else None
        ),
        "covered_any_human_anomaly": (
            bool(true_positive)
            if included and human_set
            else None
        ),
        "exclusion_reason": (
            None
            if included
            else "human_annotation_unclear"
            if unclear
            else "human_anomaly_missing_object_ids"
            if expected == "invalid" and not human_set
            else "no_human_anomaly_scope"
            if expected == "valid"
            else "scene_or_model_unresolved"
        ),
    }


def collect_l1_engineering_failures(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect fail-closed L1 transport/schema failures without hiding L3."""

    failures: list[dict[str, Any]] = []

    def visit(
        value: Any,
        *,
        path: tuple[str, ...],
        metric: str | None,
    ) -> None:
        if isinstance(value, dict):
            current_metric = (
                str(value["metric"])
                if value.get("metric") is not None
                else metric
            )
            if (
                value.get("route") == "vlm_adjudication_failed"
                or value.get("status") == "vlm_adjudication_failed"
            ):
                item = {
                    "path": ".".join(path),
                    "metric": current_metric,
                    "route": value.get("route"),
                    "status": value.get("status"),
                    "error": value.get("adjudication_error")
                    or (
                        value.get("evidence", {}).get("error")
                        if isinstance(value.get("evidence"), dict)
                        else None
                    ),
                }
                evidence = value.get("evidence")
                evidence = evidence if isinstance(evidence, dict) else {}
                audit = (
                    value.get("adjudication_failure_audit")
                    or evidence.get("adjudication_failure_audit")
                )
                if isinstance(audit, dict):
                    item["response_schema_validation"] = deepcopy(audit)
                failures.append(item)
            for key, child in value.items():
                visit(
                    child,
                    path=(*path, str(key)),
                    metric=current_metric,
                )
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(
                    child,
                    path=(*path, str(index)),
                    metric=metric,
                )

    visit(report, path=("l1",), metric=None)
    return failures


def binary_schema_validation_summary(
    report: dict[str, Any],
) -> dict[str, int]:
    """Count binary response attempts separately from logical Judge calls."""

    audits: list[dict[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in (
                "response_schema_validation",
                "adjudication_failure_audit",
            ):
                audit = value.get(key)
                if isinstance(audit, dict) and audit.get("policy") == (
                    "single_schema_repair_retry_v1"
                ):
                    audits.append(audit)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(report)
    return {
        "logical_binary_judge_calls": len(audits),
        "response_attempts": sum(
            int(item.get("attempt_count") or 0)
            for item in audits
        ),
        "schema_repair_retries": sum(
            int(item.get("repair_retry_count") or 0)
            for item in audits
        ),
        "schema_repair_recoveries": sum(
            item.get("recovered") is True for item in audits
        ),
        "schema_repair_failures": sum(
            item.get("repair_retry_count") == 1
            and item.get("recovered") is False
            for item in audits
        ),
    }


def record_case_cancellation(
    *,
    case: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    """Persist an explicit terminal record for a fail-fast cancellation."""

    case_id = str(case["case_id"])
    case_out = output_root / "cases" / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    cancelled_at = utc_now()
    record = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "cancelled",
        "reason": "cancelled_after_prior_case_failure",
        "cancelled_at": cancelled_at,
        "final_decision_status": "not_run",
        "api_usage": api_usage_summary([]),
    }
    atomic_write_json(case_out / "cancellation.json", record)
    manifest_path = case_out / "case_run_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        manifest.update(
            status="cancelled",
            completed_at=cancelled_at,
            final_decision_status="not_run",
            reason=record["reason"],
            api_usage=deepcopy(record["api_usage"]),
        )
    else:
        manifest = deepcopy(record)
        manifest["completed_at"] = cancelled_at
    atomic_write_json(manifest_path, manifest)
    return record


def record_case_failure(
    *,
    case: dict[str, Any],
    output_root: Path,
    error: Exception,
) -> dict[str, Any]:
    case_id = str(case["case_id"])
    case_out = output_root / "cases" / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    api_calls_path = case_out / "api_calls.jsonl"
    api_usage_path = case_out / "api_usage.json"
    api_usage = api_usage_summary(
        read_api_call_records(api_calls_path)
    )
    atomic_write_json(api_usage_path, api_usage)
    failure = {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "failed_at": utc_now(),
        "api_calls_path": str(api_calls_path.resolve()),
        "api_usage_path": str(api_usage_path.resolve()),
        "api_usage": api_usage,
    }
    schema_audit = response_schema_audit_from_exception(error)
    if schema_audit is not None:
        failure["response_schema_validation"] = schema_audit
    atomic_write_json(case_out / "failure.json", failure)
    manifest_path = case_out / "case_run_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        manifest.update(
            status="failed",
            completed_at=failure["failed_at"],
            final_decision_status="unresolved",
            error_type=failure["error_type"],
            error=failure["error"],
            api_usage=api_usage,
            api_calls_path=str(api_calls_path.resolve()),
            api_usage_path=str(api_usage_path.resolve()),
        )
        if schema_audit is not None:
            manifest["binary_response_schema_validation"] = schema_audit
    else:
        manifest = deepcopy(failure)
        manifest.update(
            completed_at=failure["failed_at"],
            final_decision_status="unresolved",
        )
        if schema_audit is not None:
            manifest["binary_response_schema_validation"] = schema_audit
    atomic_write_json(manifest_path, manifest)
    return failure


def l3_resolution_audit(
    scene_quality_report: dict[str, Any],
    *,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    """Resolve runner status without relabelling partial coverage as infra."""

    report_metrics = scene_quality_report.get("metrics")
    report_metrics = (
        report_metrics if isinstance(report_metrics, dict) else {}
    )
    unresolved: list[str] = []
    infrastructure_failures: list[str] = []
    partial_coverage: list[str] = []
    below_coverage_threshold: list[str] = []
    reasons: dict[str, list[str]] = {}
    coverage_warnings: dict[str, list[str]] = {}
    for metric in metrics:
        item = report_metrics.get(metric)
        metric_reasons: list[str] = []
        metric_coverage_warnings: list[str] = []
        if not isinstance(item, dict):
            metric_reasons.append("metric_report_missing")
            infrastructure_failures.append(metric)
        else:
            item_status = str(item.get("status") or "")
            terminal_state = str(item.get("terminal_state") or "")
            item_reason = str(item.get("reason") or "")
            coverage_threshold_failure = bool(
                item_status == "failed_coverage_threshold"
                or item_reason
                in {
                    "below_minimum_score_coverage",
                    "failed_coverage_threshold",
                }
            )
            explicit_infrastructure_failure = bool(
                item_status in {"error", "infrastructure_failure"}
                or (item_status == "failed" and not coverage_threshold_failure)
                or terminal_state == "infrastructure_failure"
                or bool(item.get("infrastructure_failures"))
            )
            if explicit_infrastructure_failure:
                infrastructure_failures.append(metric)
            score_publishable = _metric_score_is_publishable(item)
            if item_status not in {"evaluated", "partial"}:
                metric_reasons.append(
                    f"metric_status:{item_status or 'missing'}"
                )
            for field in (
                "coverage",
                "functional_check_coverage",
                "placement_check_coverage",
            ):
                coverage = item.get(field)
                if (
                    isinstance(coverage, dict)
                    and coverage.get("complete") is False
                ):
                    marker = f"{field}:incomplete"
                    if score_publishable:
                        metric_coverage_warnings.append(marker)
                    else:
                        metric_reasons.append(marker)
            if metric_coverage_warnings:
                partial_coverage.append(metric)
                coverage_warnings[metric] = list(
                    dict.fromkeys(metric_coverage_warnings)
                )
            elif (
                not explicit_infrastructure_failure
                and not score_publishable
                and any(
                    isinstance(item.get(field), dict)
                    and item[field].get("complete") is False
                    for field in (
                        "coverage",
                        "functional_check_coverage",
                        "placement_check_coverage",
                    )
                )
            ):
                below_coverage_threshold.append(metric)
                metric_reasons.append("coverage:below_publishable_threshold")
            elif (
                not explicit_infrastructure_failure
                and item_status in {"evaluated", "partial"}
                and not score_publishable
            ):
                metric_reasons.append("metric_score:unavailable")
            for field in (
                "functional_check_phase",
                "placement_check_phase",
                "group_phase",
                "cross_group_relation_phase",
            ):
                phase = item.get(field)
                if (
                    isinstance(phase, dict)
                    and str(phase.get("status") or "")
                    in {
                        "unresolved",
                        "terminal_contract_failure",
                        "infrastructure_failure",
                    }
                ):
                    phase_status = str(phase.get("status") or "")
                    metric_reasons.append(f"{field}:{phase_status}")
                    if phase_status in {
                        "terminal_contract_failure",
                        "infrastructure_failure",
                    }:
                        infrastructure_failures.append(metric)
        if metric_reasons:
            if metric not in infrastructure_failures:
                unresolved.append(metric)
            reasons[metric] = list(dict.fromkeys(metric_reasons))
    infrastructure_failures = list(dict.fromkeys(infrastructure_failures))
    unresolved = list(dict.fromkeys(unresolved))
    partial_coverage = list(dict.fromkeys(partial_coverage))
    below_coverage_threshold = list(
        dict.fromkeys(below_coverage_threshold)
    )
    return {
        "status": (
            "infrastructure_failure"
            if infrastructure_failures
            else "unresolved"
            if unresolved
            else "resolved"
        ),
        "unresolved_metrics": unresolved,
        "infrastructure_failure_metrics": infrastructure_failures,
        "partial_coverage_metrics": partial_coverage,
        "below_coverage_threshold_metrics": below_coverage_threshold,
        "reasons_by_metric": reasons,
        "coverage_warnings_by_metric": coverage_warnings,
        "minimum_publishable_coverage": MIN_PUBLISHABLE_SCORE_COVERAGE,
        "policy": "terminal_binary_or_scoreable_partial_coverage_v3",
    }


def _metric_score_is_publishable(item: dict[str, Any]) -> bool:
    """Treat the evaluator's published score and threshold audit as authority."""

    score = item.get("score")
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(float(score))
    ):
        return False
    coverage = item.get("coverage")
    coverage = coverage if isinstance(coverage, dict) else {}
    scoring = item.get("scoring")
    scoring = scoring if isinstance(scoring, dict) else {}
    projections = [
        coverage,
        coverage.get("score_projection"),
        scoring.get("coverage_projection"),
    ]
    explicit_thresholds = [
        projection.get("coverage_threshold_passed")
        for projection in projections
        if isinstance(projection, dict)
        and isinstance(projection.get("coverage_threshold_passed"), bool)
    ]
    if explicit_thresholds:
        return all(explicit_thresholds)
    return True


def build_summary(
    *,
    case_records: list[dict[str, Any]],
    metrics: tuple[str, ...],
    elapsed_seconds: float,
) -> dict[str, Any]:
    metric_summaries = {
        metric: empty_metric_summary(total=len(case_records))
        for metric in metrics
    }
    successful = 0
    cancelled = 0
    grouping_failures = 0
    final_unresolved = 0
    final_infrastructure_failure = 0
    l3_unresolved_cases = 0
    l3_infrastructure_failure_cases = 0
    l1_engineering_failure_cases = 0
    binary_logical_judge_calls = 0
    binary_response_attempts = 0
    binary_schema_repair_retries = 0
    binary_schema_repair_recoveries = 0
    binary_schema_repair_failures = 0
    total_judge_calls = 0
    total_selector_calls = 0
    all_api_call_records: list[dict[str, Any]] = []
    latencies: list[float] = []
    for record in case_records:
        api_calls_path = record.get("api_calls_path")
        if isinstance(api_calls_path, str) and api_calls_path:
            all_api_call_records.extend(
                read_api_call_records(Path(api_calls_path))
            )
        if record.get("status") == "cancelled":
            cancelled += 1
        if record.get("status") not in {"complete", "resumed"}:
            for summary in metric_summaries.values():
                summary["case_failures"] += 1
            continue
        successful += 1
        if record.get("final_decision_status") == "unresolved":
            final_unresolved += 1
        if record.get("final_decision_status") == "infrastructure_failure":
            final_infrastructure_failure += 1
        if record.get("l3_decision_status") == "unresolved":
            l3_unresolved_cases += 1
        if record.get("l3_decision_status") == "infrastructure_failure":
            l3_infrastructure_failure_cases += 1
        if record.get("l1_engineering_failure") is True:
            l1_engineering_failure_cases += 1
        binary_schema = record.get(
            "binary_response_schema_validation"
        )
        binary_schema = (
            binary_schema if isinstance(binary_schema, dict) else {}
        )
        binary_logical_judge_calls += int(
            binary_schema.get("logical_binary_judge_calls") or 0
        )
        binary_response_attempts += int(
            binary_schema.get("response_attempts") or 0
        )
        binary_schema_repair_retries += int(
            binary_schema.get("schema_repair_retries") or 0
        )
        binary_schema_repair_recoveries += int(
            binary_schema.get("schema_repair_recoveries") or 0
        )
        binary_schema_repair_failures += int(
            binary_schema.get("schema_repair_failures") or 0
        )
        latency = record.get("elapsed_seconds")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        comparison_path = Path(str(record["scene_comparison_path"]))
        comparison = read_json(comparison_path)
        if record.get("grouping_status") not in {None, "complete"}:
            grouping_failures += 1
        report = read_json(Path(str(record["scene_quality_report_path"])))
        report_metrics = report.get("metrics")
        report_metrics = report_metrics if isinstance(report_metrics, dict) else {}
        control = (
            read_json(Path(str(record["control_manifest_path"])))
            if record.get("control_manifest_path")
            else {}
        )
        telemetry = telemetry_by_metric(control)
        comparisons = comparison.get("metrics")
        comparisons = comparisons if isinstance(comparisons, dict) else {}
        for metric in metrics:
            item = comparisons.get(metric)
            item = item if isinstance(item, dict) else {}
            human = item.get("human")
            human = human if isinstance(human, dict) else {}
            model = item.get("model")
            model = model if isinstance(model, dict) else {}
            summary = metric_summaries[metric]
            if record.get("final_decision_status") != "resolved":
                summary["diagnostic_only_cases"] += 1
            expected = human.get("expected")
            predicted = model.get("prediction")
            if expected in {"valid", "invalid"}:
                summary["human_distribution"][expected] += 1
            if predicted in {"valid", "invalid"}:
                summary["predicted_distribution"][predicted] += 1
                summary["evaluated"] += 1
            elif predicted == "infrastructure_failure":
                summary["infrastructure_failure"] += 1
            else:
                summary["unresolved"] += 1
            if human.get("unclear") is True:
                summary["excluded_unclear"] += 1
            if item.get("included_in_accuracy") is True:
                if item.get("matches") is True:
                    summary["correct"] += 1
                else:
                    summary["incorrect"] += 1
            anomaly_level = item.get("anomaly_level")
            anomaly_level = (
                anomaly_level
                if isinstance(anomaly_level, dict)
                else {}
            )
            if anomaly_level.get("included_in_accuracy") is True:
                summary["anomaly_object_cases"] += 1
                if anomaly_level.get("exact_match") is True:
                    summary["anomaly_object_exact_correct"] += 1
                else:
                    summary["anomaly_object_exact_incorrect"] += 1
                summary["anomaly_object_true_positive"] += len(
                    anomaly_level.get("true_positive_object_ids") or []
                )
                summary["anomaly_object_false_negative"] += len(
                    anomaly_level.get("false_negative_object_ids") or []
                )
                summary["anomaly_object_false_positive"] += len(
                    anomaly_level.get("false_positive_object_ids") or []
                )
            metric_report = report_metrics.get(metric)
            metric_report = (
                metric_report if isinstance(metric_report, dict) else {}
            )
            failure_counts = metric_failure_counts(metric_report)
            summary["camera_render_failures"] += failure_counts[
                "camera_render_failures"
            ]
            summary["judge_failures"] += failure_counts["judge_failures"]
            metric_telemetry = telemetry.get(metric, {})
            for key in (
                "judge_calls",
                "vlm_selector_calls",
                "preview_image_count",
                "final_image_count",
                "evidence_repair_count",
                "evidence_recovery_count",
            ):
                summary[key] += int(metric_telemetry.get(key) or 0)
            summary["initial_image_count"] += initial_image_count(
                metric_report
            )
    for summary in metric_summaries.values():
        denominator = summary["correct"] + summary["incorrect"]
        summary["accuracy"] = (
            summary["correct"] / denominator if denominator else None
        )
        object_denominator = (
            summary["anomaly_object_exact_correct"]
            + summary["anomaly_object_exact_incorrect"]
        )
        summary["anomaly_object_exact_accuracy"] = (
            summary["anomaly_object_exact_correct"]
            / object_denominator
            if object_denominator
            else None
        )
        object_precision_denominator = (
            summary["anomaly_object_true_positive"]
            + summary["anomaly_object_false_positive"]
        )
        object_recall_denominator = (
            summary["anomaly_object_true_positive"]
            + summary["anomaly_object_false_negative"]
        )
        summary["anomaly_object_precision"] = (
            summary["anomaly_object_true_positive"]
            / object_precision_denominator
            if object_precision_denominator
            else None
        )
        summary["anomaly_object_recall"] = (
            summary["anomaly_object_true_positive"]
            / object_recall_denominator
            if object_recall_denominator
            else None
        )
        precision = summary["anomaly_object_precision"]
        recall = summary["anomaly_object_recall"]
        summary["anomaly_object_f1"] = (
            (
                2.0 * precision * recall / (precision + recall)
                if precision + recall > 0.0
                else 0.0
            )
            if isinstance(precision, float)
            and isinstance(recall, float)
            else None
        )
        total_judge_calls += summary["judge_calls"]
        total_selector_calls += summary["vlm_selector_calls"]
        summary["grouping_failures"] = grouping_failures
    api_usage = api_usage_summary(all_api_call_records)
    operation_calls = (
        api_usage.get("operation_calls")
        if isinstance(api_usage.get("operation_calls"), dict)
        else {}
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": (
            "complete"
            if successful == len(case_records)
            else "partial"
            if successful
            else "failed"
        ),
        "source_prompt_used": False,
        "comparison_scopes": [
            "scene_level_metric_verdict",
            "anomaly_object_attribution",
        ],
        "elapsed_seconds": elapsed_seconds,
        "average_case_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "totals": {
            "cases": len(case_records),
            "successful": successful,
            "failed": len(case_records) - successful - cancelled,
            "cancelled": cancelled,
            "grouping_failures": grouping_failures,
            "final_unresolved": final_unresolved,
            "final_infrastructure_failure": (
                final_infrastructure_failure
            ),
            "l3_unresolved_cases": l3_unresolved_cases,
            "l3_infrastructure_failure_cases": (
                l3_infrastructure_failure_cases
            ),
            "l1_engineering_failure_cases": (
                l1_engineering_failure_cases
            ),
            "binary_logical_judge_calls": binary_logical_judge_calls,
            "binary_response_attempts": (
                binary_response_attempts
            ),
            "binary_schema_repair_retries": (
                binary_schema_repair_retries
            ),
            "binary_schema_repair_recoveries": (
                binary_schema_repair_recoveries
            ),
            "binary_schema_repair_failures": (
                binary_schema_repair_failures
            ),
            "judge_calls": total_judge_calls,
            "vlm_camera_selector_calls": total_selector_calls,
            "functional_discovery_calls": int(
                operation_calls.get("functional_discovery") or 0
            ),
            "functional_affordance_calls": int(
                operation_calls.get("functional_affordance") or 0
            ),
            "functional_relation_calls": int(
                operation_calls.get("functional_relation") or 0
            ),
            "placement_discovery_calls": int(
                operation_calls.get("placement_discovery") or 0
            ),
            "usable_surface_decoder_calls": int(
                operation_calls.get("usable_surface_decoder") or 0
            ),
            "vlm_camera_selector_api_calls": int(
                operation_calls.get("camera_selector") or 0
            ),
            "judge_api_calls": int(
                operation_calls.get("judge") or 0
            ),
            "api_calls_number": api_usage["api_calls_number"],
            "tokens_usage": deepcopy(api_usage["tokens_usage"]),
        },
        "api_usage": api_usage,
        "metrics": metric_summaries,
    }


def empty_metric_summary(*, total: int) -> dict[str, Any]:
    return _runtime_telemetry.empty_metric_summary(total=total)


def telemetry_by_metric(control_manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    return _runtime_telemetry.telemetry_by_metric(control_manifest)


def initial_image_count(metric_report: dict[str, Any]) -> int:
    return _runtime_telemetry.initial_image_count(metric_report)


def metric_failure_counts(metric_report: dict[str, Any]) -> dict[str, int]:
    return _runtime_telemetry.metric_failure_counts(metric_report)


def read_api_call_records(path: Path) -> list[dict[str, Any]]:
    return _runtime_telemetry.read_api_call_records(path)


def api_usage_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return _runtime_telemetry.api_usage_summary(records)


def _api_usage_bucket(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return _runtime_telemetry.api_usage_bucket(records)


def _normalized_token_usage(value: Any) -> dict[str, int] | None:
    return _runtime_telemetry.normalized_token_usage(value)


def _safe_request_metadata(model: Any) -> dict[str, Any]:
    return _runtime_telemetry.safe_request_metadata(model)


def _message_image_count(messages: Any) -> int:
    return _runtime_telemetry.message_image_count(messages)


def _request_scope(request: Any) -> tuple[str, str]:
    return _runtime_telemetry.request_scope(request)


def _evidence_count(value: Any) -> int | None:
    return _runtime_telemetry.evidence_count(value)


def _nonnegative_int_or_none(value: Any) -> int | None:
    return _runtime_telemetry.nonnegative_int_or_none(value)


def _bounded_error(error: Exception) -> str:
    return _runtime_telemetry.bounded_error(error)


def _format_progress_record(record: dict[str, Any]) -> str:
    return _runtime_progress.format_progress_record(record)


def _progress_value(value: Any) -> str:
    return _runtime_progress.progress_value(value)


def safe_route_manifest(route: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "endpoint": route["endpoint"],
        "model": route["model"],
        "api_key_env": route["api_key_env"],
        "authorization_configured": bool(
            route.get("authorization_configured")
        ),
    }
    if "min_request_interval_seconds" in route:
        manifest["min_request_interval_seconds"] = float(
            route["min_request_interval_seconds"]
        )
    return manifest


def read_json(path: Path) -> dict[str, Any]:
    return _runtime_io.read_json(path)


def read_yaml_object(path: Path) -> dict[str, Any]:
    return _runtime_io.read_yaml_object(path)


def atomic_write_json(path: Path, value: Any) -> None:
    _runtime_io.atomic_write_json(path, value)


def file_sha256(path: Path) -> str:
    return _runtime_io.file_sha256(path)


def json_sha256(value: Any) -> str:
    return _runtime_io.json_sha256(value)


def utc_now() -> str:
    return _runtime_io.utc_now()


if __name__ == "__main__":
    main()
