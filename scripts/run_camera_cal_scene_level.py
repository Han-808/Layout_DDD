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
from benchmark.camera_cal_scene_level import comparison as _runtime_comparison  # noqa: E402
from benchmark.camera_cal_scene_level import provenance as _runtime_provenance  # noqa: E402
from benchmark.camera_cal_scene_level import reports as _runtime_reports  # noqa: E402
from benchmark.camera_cal_scene_level import adapters as _runtime_adapters  # noqa: E402
from benchmark.camera_cal_scene_level import case_runtime as _runtime_case  # noqa: E402


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


def _provenance_dependencies() -> _runtime_provenance.ProvenanceDependencies:
    return _runtime_provenance.ProvenanceDependencies(
        project_root=PROJECT_ROOT,
        runner_schema_version=RUNNER_SCHEMA_VERSION,
        l3_metric_prompt_version=L3_METRIC_PROMPT_VERSION,
        grouping_completion_max_tokens=GROUPING_COMPLETION_MAX_TOKENS,
        judge_completion_max_tokens=JUDGE_COMPLETION_MAX_TOKENS,
        camera_selector_completion_max_tokens=(
            CAMERA_SELECTOR_COMPLETION_MAX_TOKENS
        ),
        l1_binary_failure_policy=L1_BINARY_FAILURE_POLICY,
        functional_probe_implementation_files=(
            FUNCTIONAL_PROBE_IMPLEMENTATION_FILES
        ),
        prompt_path="src/benchmark/visual_judge/l3_prompts.py",
        prompt_context_path=(
            "src/benchmark/evaluator/scene_quality/prompt_context.py"
        ),
        scoring_implementation_paths=(
            "src/benchmark/evaluator/scoring.py",
            "src/benchmark/scoring_profiles.py",
        ),
        default_deduction_multiplier=DEFAULT_DEDUCTION_MULTIPLIER,
        file_sha256=lambda path: file_sha256(path),
        json_sha256=lambda value: json_sha256(value),
        promptless_l1_l3_profile=lambda: promptless_l1_l3_profile(),
        promptless_l3_only_profile=lambda: promptless_l3_only_profile(),
        scene_quality_config=lambda *args, **kwargs: scene_quality_config(
            *args, **kwargs
        ),
        camera_cal_asset_policy=lambda: camera_cal_asset_policy(),
    )


def _reports_dependencies() -> _runtime_reports.ReportsDependencies:
    return _runtime_reports.ReportsDependencies(
        read_json=lambda path: read_json(path),
        atomic_write_json=lambda path, value: atomic_write_json(path, value),
        utc_now=lambda: utc_now(),
        api_usage_summary=lambda records: api_usage_summary(records),
        read_api_call_records=lambda path: read_api_call_records(path),
        empty_metric_summary=lambda **kwargs: empty_metric_summary(**kwargs),
        telemetry_by_metric=lambda value: telemetry_by_metric(value),
        initial_image_count=lambda value: initial_image_count(value),
        metric_failure_counts=lambda value: metric_failure_counts(value),
        response_schema_audit_from_exception=(
            response_schema_audit_from_exception
        ),
        metric_score_is_publishable=lambda item: (
            _metric_score_is_publishable(item)
        ),
        case_schema_version=CASE_SCHEMA_VERSION,
        summary_schema_version=SUMMARY_SCHEMA_VERSION,
        minimum_publishable_coverage=MIN_PUBLISHABLE_SCORE_COVERAGE,
    )


def _adapter_factories() -> _runtime_adapters.AdapterFactories:
    """Read current runner globals for every adapter construction."""

    return _runtime_adapters.AdapterFactories(
        model_config=model_config,
        build_grouping_model=build_grouping_model,
        build_openai_compatible_vlm_judge=(
            build_openai_compatible_vlm_judge
        ),
        build_openai_compatible_camera_selector=(
            build_openai_compatible_camera_selector
        ),
        BlenderRenderer=BlenderRenderer,
        CameraEvidenceProvider=CameraEvidenceProvider,
        DeterministicLocalCameraSelector=DeterministicLocalCameraSelector,
        CameraViewEvidenceRenderer=CameraViewEvidenceRenderer,
        CameraCandidatePreviewRenderer=CameraCandidatePreviewRenderer,
        ObservedEvidenceProvider=_ObservedEvidenceProvider,
        ObservedRenderer=_ObservedRenderer,
    )


def _build_runtime_adapters(**kwargs: Any) -> _runtime_adapters.AdapterBundle:
    return _runtime_adapters.build_adapters(
        **kwargs,
        factories=_adapter_factories(),
    )


def _case_runtime_dependencies() -> _runtime_case.CaseRuntimeDeps:
    """Build fresh dependencies so historical monkeypatch points stay live."""

    return _runtime_case.CaseRuntimeDeps(
        io=_runtime_case.CaseRuntimeIO(
            read_json=read_json,
            read_yaml_object=read_yaml_object,
            atomic_write_json=atomic_write_json,
            utc_now=utc_now,
            monotonic=time.monotonic,
        ),
        resume=_runtime_case.CaseRuntimeResume(
            case_paths=case_paths,
            case_input_fingerprint=case_input_fingerprint,
            resumable_case=resumable_case,
        ),
        policy=_runtime_case.CaseRuntimePolicy(
            safe_route_manifest=safe_route_manifest,
            identity_legend_from_manifest=identity_legend_from_manifest,
            grouping_evidence_packet=grouping_evidence_packet,
            promptless_scene_request=promptless_scene_request,
            promptless_l1_l3_profile=promptless_l1_l3_profile,
            promptless_l3_only_profile=promptless_l3_only_profile,
            scene_quality_config=scene_quality_config,
            camera_cal_asset_policy=camera_cal_asset_policy,
            collect_l1_engineering_failures=collect_l1_engineering_failures,
            binary_schema_validation_summary=(
                binary_schema_validation_summary
            ),
            l3_resolution_audit=l3_resolution_audit,
            build_scene_comparison=build_scene_comparison,
            maybe_export_audit_graphs=_maybe_export_audit_graphs,
            case_schema_version=CASE_SCHEMA_VERSION,
            l1_layer=L1,
            l2_layer=L2,
            l3_layer=L3,
            l4_layer=L4,
            l1_binary_failure_policy=L1_BINARY_FAILURE_POLICY,
            scoring_profile_id=INTRINSIC_VALIDITY_PROFILE_ID,
        ),
        external=_runtime_case.CaseRuntimeExternal(
            api_call_tracker_factory=APICallTracker,
            api_usage_summary=api_usage_summary,
            read_api_call_records=read_api_call_records,
            load_collision_geometry_manifest=(
                load_collision_geometry_manifest
            ),
            adapter_builder=_build_runtime_adapters,
            run_evaluate=run_evaluate,
            progress_factory=ProgressReporter,
        ),
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
    return _runtime_case.run_case_impl(
        case=case,
        dataset_root=dataset_root,
        output_root=output_root,
        grouping_config_path=grouping_config_path,
        route=route,
        metrics=metrics,
        renderer_config=renderer_config,
        control_config=control_config,
        resume=resume,
        functional_group_local_granularity=(
            functional_group_local_granularity
        ),
        functional_group_local_evidence_policy=(
            functional_group_local_evidence_policy
        ),
        deduction_multiplier=deduction_multiplier,
        export_audit_graphs=export_audit_graphs,
        l3_only=l3_only,
        progress=progress,
        model_route_abort_signal=model_route_abort_signal,
        deps=_case_runtime_dependencies(),
    )


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
    return _runtime_provenance.case_input_fingerprint(
        dependencies=_provenance_dependencies(),
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
    return _runtime_comparison.build_scene_comparison(
        case_id=case_id,
        annotation=annotation,
        scene_quality_report=scene_quality_report,
        metrics=metrics,
    )


def metric_prediction(report: dict[str, Any]) -> str:
    return _runtime_comparison.metric_prediction(report)


def _model_anomaly_object_ids(
    report: dict[str, Any],
) -> tuple[str, ...]:
    return _runtime_comparison._model_anomaly_object_ids(report)


def _ordered_object_ids(value: Any) -> tuple[str, ...]:
    return _runtime_comparison._ordered_object_ids(value)


def _anomaly_object_comparison(
    *,
    expected: str,
    predicted: str,
    unclear: bool,
    human_object_ids: tuple[str, ...],
    model_object_ids: tuple[str, ...],
) -> dict[str, Any]:
    return _runtime_comparison._anomaly_object_comparison(
        expected=expected,
        predicted=predicted,
        unclear=unclear,
        human_object_ids=human_object_ids,
        model_object_ids=model_object_ids,
    )


def collect_l1_engineering_failures(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    return _runtime_reports.collect_l1_engineering_failures(report)


def binary_schema_validation_summary(
    report: dict[str, Any],
) -> dict[str, int]:
    return _runtime_reports.binary_schema_validation_summary(report)


def record_case_cancellation(
    *,
    case: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    return _runtime_reports.record_case_cancellation(
        case=case,
        output_root=output_root,
        dependencies=_reports_dependencies(),
    )


def record_case_failure(
    *,
    case: dict[str, Any],
    output_root: Path,
    error: Exception,
) -> dict[str, Any]:
    return _runtime_reports.record_case_failure(
        case=case,
        output_root=output_root,
        error=error,
        dependencies=_reports_dependencies(),
    )


def l3_resolution_audit(
    scene_quality_report: dict[str, Any],
    *,
    metrics: tuple[str, ...],
) -> dict[str, Any]:
    return _runtime_reports.l3_resolution_audit(
        scene_quality_report,
        metrics=metrics,
        dependencies=_reports_dependencies(),
    )


def _metric_score_is_publishable(item: dict[str, Any]) -> bool:
    return _runtime_reports._metric_score_is_publishable(item)


def build_summary(
    *,
    case_records: list[dict[str, Any]],
    metrics: tuple[str, ...],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return _runtime_reports.build_summary(
        case_records=case_records,
        metrics=metrics,
        elapsed_seconds=elapsed_seconds,
        dependencies=_reports_dependencies(),
    )


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
    return _runtime_provenance.safe_route_manifest(route)


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
