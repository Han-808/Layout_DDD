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
from benchmark.models import OpenAICompatibleModel  # noqa: E402
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


RUNNER_SCHEMA_VERSION = "camera_cal_scene_level_runner_v4"
PLAN_SCHEMA_VERSION = "camera_cal_scene_level_plan_v1"
CASE_SCHEMA_VERSION = "camera_cal_scene_level_case_v3"
COMPARISON_SCHEMA_VERSION = "camera_cal_scene_comparison_v1"
SUMMARY_SCHEMA_VERSION = "camera_cal_scene_level_summary_v2"
PROGRESS_SCHEMA_VERSION = "camera_cal_scene_level_progress_v1"
API_CALL_SCHEMA_VERSION = "camera_cal_api_call_v1"
API_USAGE_SCHEMA_VERSION = "camera_cal_api_usage_v1"
GROUPING_COMPLETION_MAX_TOKENS = 3192
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
CANONICAL_L3_METRICS = (
    "scale_consistency",
    "object_pairing_consistency",
    "style_consistency",
)
EXPERIMENTAL_L3_METRICS = (
    "functional_consistency",
    "semantic_placement_consistency",
)
FUNCTIONAL_PROBE_IMPLEMENTATION_FILES = (
    "src/benchmark/evaluator/scene_quality/functional_acquisition.py",
    "src/benchmark/evaluator/scene_quality/functional_boundary_evidence.py",
    "src/benchmark/evaluator/scene_quality/functional_geometry.py",
    "src/benchmark/evaluator/scene_quality/functional_probe.py",
    "src/benchmark/evaluator/scene_quality/global_group_first.py",
    "src/benchmark/rendering/camera_pose.py",
    "src/benchmark/visual_judge/functional_discovery.py",
    "src/benchmark/visual_judge/functional_evidence.py",
    "src/benchmark/visual_judge/openai_camera_selector.py",
    "src/benchmark/visual_judge/openai_compatible.py",
    "src/benchmark/visual_judge/placement_discovery.py",
    "src/benchmark/visual_judge/render_views.py",
    "src/benchmark/visual_judge/usable_surface.py",
)

_CASE_ID_PATTERN = re.compile(r"N\d{3}")
_ENV_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_prompt_tokens",
    "reasoning_tokens",
)


class ProgressReporter:
    """Persist concise progress events and optionally mirror them to stdout."""

    def __init__(self, path: Path, *, terminal: bool = True) -> None:
        self.path = path.expanduser().resolve()
        self.terminal = bool(terminal)
        self._lock = threading.Lock()

    def emit(
        self,
        event: str,
        *,
        case_id: str | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        record = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "timestamp": utc_now(),
            "event": str(event),
            "case_id": str(case_id) if case_id else None,
            "details": deepcopy(details),
        }
        encoded = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
            if self.terminal:
                print(_format_progress_record(record), flush=True)
        return record


class APICallTracker:
    """Record actual logical chat-completions calls and reported token usage."""

    def __init__(
        self,
        *,
        case_id: str,
        calls_path: Path,
        usage_path: Path,
        progress: ProgressReporter,
    ) -> None:
        self.case_id = str(case_id)
        self.calls_path = calls_path.expanduser().resolve()
        self.usage_path = usage_path.expanduser().resolve()
        self.progress = progress
        self._lock = threading.Lock()
        self._records = read_api_call_records(self.calls_path)
        self._next_call_number = len(self._records) + 1
        atomic_write_json(self.usage_path, api_usage_summary(self._records))

    def observe_model(self, model: Any, *, role: str) -> Any:
        if not callable(getattr(model, "chat_messages", None)):
            return model
        return _ObservedChatModel(model, role=role, tracker=self)

    def begin_call(
        self,
        *,
        role: str,
        call_type: str,
        messages: Any,
    ) -> tuple[int, str, float]:
        with self._lock:
            call_number = self._next_call_number
            self._next_call_number += 1
        call_id = f"{self.case_id}-{call_number:05d}"
        self.progress.emit(
            "api_call_started",
            case_id=self.case_id,
            api_call_number=call_number,
            api_call_id=call_id,
            role=role,
            call_type=call_type,
            image_count=_message_image_count(messages),
        )
        return call_number, call_id, time.monotonic()

    def finish_call(
        self,
        *,
        call_number: int,
        call_id: str,
        role: str,
        call_type: str,
        started: float,
        request_metadata: dict[str, Any],
        error: Exception | None,
    ) -> dict[str, Any]:
        duration = max(0.0, time.monotonic() - started)
        usage = _normalized_token_usage(request_metadata.get("usage"))
        record = {
            "schema_version": API_CALL_SCHEMA_VERSION,
            "api_call_number": int(call_number),
            "api_call_id": str(call_id),
            "case_id": self.case_id,
            "role": str(role),
            "call_type": str(call_type),
            "status": "failed" if error is not None else "complete",
            "completed_at": utc_now(),
            "duration_seconds": duration,
            "model": request_metadata.get("model"),
            "endpoint": request_metadata.get("endpoint"),
            "message_count": _nonnegative_int_or_none(
                request_metadata.get("message_count")
            ),
            "image_count": _nonnegative_int_or_none(
                request_metadata.get("image_count")
            ),
            "prompt_chars": _nonnegative_int_or_none(
                request_metadata.get("prompt_chars")
            ),
            "finish_reason": request_metadata.get("finish_reason"),
            "tokens_usage": usage,
            "error_type": type(error).__name__ if error is not None else None,
            "error": _bounded_error(error) if error is not None else None,
        }
        with self._lock:
            self._records.append(record)
            self.calls_path.parent.mkdir(parents=True, exist_ok=True)
            with self.calls_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            cumulative = api_usage_summary(self._records)
            atomic_write_json(self.usage_path, cumulative)
        self.progress.emit(
            (
                "api_call_failed"
                if error is not None
                else "api_call_completed"
            ),
            case_id=self.case_id,
            api_call_number=call_number,
            api_call_id=call_id,
            role=role,
            call_type=call_type,
            duration_seconds=round(duration, 3),
            tokens_usage=usage,
            cumulative_api_calls=cumulative["api_calls_number"],
            cumulative_tokens_usage=cumulative["tokens_usage"],
            error_type=record["error_type"],
        )
        return record

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return api_usage_summary(self._records)


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
        cases=cases,
        renderer_config=renderer_config,
        control=control.to_dict(),
        max_workers=args.max_workers,
        resume=args.resume,
        continue_on_error=args.continue_on_error,
    )
    atomic_write_json(output_root / "experiment_plan.json", experiment)

    started = time.monotonic()
    run_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    run_manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "status": "running",
        "started_at": utc_now(),
        "completed_at": None,
        "elapsed_seconds": None,
        "experiment_plan_sha256": json_sha256(experiment),
        "source_prompt_used": False,
        "layers_executed": [L1, L3],
        "layers_not_executed": [L2, L4],
        "cases": [],
        "progress_path": str(progress.path),
        "api_usage": api_usage_summary([]),
    }
    atomic_write_json(output_root / "run_manifest.json", run_manifest)
    progress.emit(
        "run_started",
        case_count=len(cases),
        metrics=list(metrics),
        max_workers=args.max_workers,
        output_root=str(output_root),
    )

    case_kwargs = {
        "dataset_root": dataset_root,
        "output_root": output_root,
        "grouping_config_path": grouping_config_path,
        "route": route,
        "metrics": metrics,
        "renderer_config": renderer_config,
        "control_config": control.to_dict(),
        "resume": args.resume,
        "progress": progress,
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
    else:
        for index, case in enumerate(cases, start=1):
            progress.emit(
                "case_queued",
                case_id=str(case["case_id"]),
                case_index=index,
                case_count=len(cases),
            )
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_case: dict[Future[dict[str, Any]], dict[str, Any]] = {
                executor.submit(run_case, case=case, **case_kwargs): case
                for case in cases
            }
            for completed, future in enumerate(
                as_completed(future_to_case),
                start=1,
            ):
                case = future_to_case[future]
                try:
                    record = future.result()
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
                        completed_count=completed,
                        case_count=len(cases),
                        error_type=failure["error_type"],
                    )
                    if not args.continue_on_error:
                        for pending in future_to_case:
                            pending.cancel()
                        break
                else:
                    run_records.append(record)
                    progress.emit(
                        "case_completed",
                        case_id=str(case["case_id"]),
                        completed_count=completed,
                        case_count=len(cases),
                        status=record["status"],
                        elapsed_seconds=round(
                            float(record.get("elapsed_seconds") or 0.0),
                            3,
                        ),
                        api_usage=record.get("api_usage"),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New or resumable run-level output directory.",
    )
    parser.add_argument(
        "--grouping-config",
        type=Path,
        default=DEFAULT_GROUPING_CONFIG,
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Repeat to select cases. Omit to run every ready case.",
    )
    parser.add_argument(
        "--metric",
        action="append",
        choices=ANNOTATED_L3_METRICS,
        default=[],
        help="Repeat to select L3 metrics. Omit to run all five annotations.",
    )
    parser.add_argument("--max-cases", type=positive_int, default=None)
    parser.add_argument("--max-workers", type=positive_int, default=1)
    parser.add_argument(
        "--terminal-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Mirror progress.jsonl events to stdout. Persistent progress "
            "events are always written."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--blender-bin",
        type=Path,
        default=DEFAULT_BLENDER_BIN,
    )
    parser.add_argument("--render-width", type=positive_int, default=768)
    parser.add_argument("--render-height", type=positive_int, default=768)
    parser.add_argument(
        "--render-engine",
        choices=RENDER_ENGINES,
        default="BLENDER_EEVEE_NEXT",
    )
    parser.add_argument(
        "--cycles-device",
        choices=CYCLES_DEVICES,
        default="CPU",
    )
    parser.add_argument("--cycles-samples", type=positive_int, default=16)
    parser.add_argument(
        "--cycles-denoising",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--preview-render-engine",
        choices=RENDER_ENGINES,
        default="BLENDER_EEVEE_NEXT",
    )
    parser.add_argument("--preview-width", type=positive_int, default=256)
    parser.add_argument("--preview-height", type=positive_int, default=256)
    parser.add_argument(
        "--preview-cycles-samples",
        type=positive_int,
        default=1,
    )
    parser.add_argument(
        "--blender-timeout-seconds",
        type=positive_int,
        default=900,
    )
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def effective_model_route(
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> dict[str, Any]:
    env = os.environ if environ is None else environ
    required = ("JUDGE_ENDPOINT", "JUDGE_MODEL", "JUDGE_API_KEY_ENV")
    missing = [name for name in required if not str(env.get(name) or "").strip()]
    if missing:
        raise RuntimeError(
            "explicit runtime model routing is required; missing "
            + ", ".join(missing)
        )
    endpoint = str(env["JUDGE_ENDPOINT"]).strip().rstrip("/")
    model = str(env["JUDGE_MODEL"]).strip()
    api_key_env = str(env["JUDGE_API_KEY_ENV"]).strip()
    if not _ENV_NAME_PATTERN.fullmatch(api_key_env):
        raise ValueError("JUDGE_API_KEY_ENV must name a valid environment variable")
    if endpoint in {
        "http://127.0.0.1:4000",
        "http://127.0.0.1:4000/v1",
        "http://localhost:4000",
        "http://localhost:4000/v1",
    }:
        raise RuntimeError(
            "port 4000 is the stale LiteLLM route; set JUDGE_ENDPOINT "
            "explicitly to the intended endpoint"
        )
    if not str(env.get(api_key_env) or ""):
        raise RuntimeError(
            f"required API credential is not available in this process: "
            f"{api_key_env}"
        )
    return {
        "endpoint": endpoint,
        "model": model,
        "api_key_env": api_key_env,
        "authorization_configured": True,
    }


def normalize_metric_selection(values: Iterable[str]) -> tuple[str, ...]:
    selected = list(dict.fromkeys(str(value) for value in values))
    if not selected:
        return ANNOTATED_L3_METRICS
    unknown = sorted(set(selected) - set(ANNOTATED_L3_METRICS))
    if unknown:
        raise ValueError(f"unknown L3 metrics: {unknown}")
    return tuple(
        metric for metric in ANNOTATED_L3_METRICS if metric in selected
    )


def discover_cases(
    dataset_root: Path,
    *,
    case_ids: Iterable[str] = (),
    max_cases: int | None = None,
) -> list[dict[str, Any]]:
    root = dataset_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"camera-cal dataset does not exist: {root}")
    selected_ids = list(dict.fromkeys(str(value) for value in case_ids))
    invalid_ids = [
        case_id
        for case_id in selected_ids
        if not _CASE_ID_PATTERN.fullmatch(case_id)
    ]
    if invalid_ids:
        raise ValueError(f"invalid camera-cal case IDs: {invalid_ids}")

    discovered: dict[str, dict[str, Any]] = {}
    for case_root in sorted(root.iterdir()):
        if not case_root.is_dir() or not _CASE_ID_PATTERN.fullmatch(case_root.name):
            continue
        manifest_path = case_root / "case_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = read_json(manifest_path)
        if manifest.get("status") != "ready":
            continue
        case_id = str(manifest.get("case_id") or case_root.name)
        required = case_paths(case_root, manifest)
        missing = [
            name
            for name, path in required.items()
            if name != "render_manifest" and not path.is_file()
        ]
        if missing:
            continue
        discovered[case_id] = {
            "case_id": case_id,
            "case_root": str(case_root),
            "scene_type": manifest.get("scene_type"),
            "object_count": manifest.get("object_count"),
            "semantic_content_fingerprint": manifest.get(
                "semantic_content_fingerprint"
            ),
        }
    if selected_ids:
        missing_ids = [case_id for case_id in selected_ids if case_id not in discovered]
        if missing_ids:
            raise ValueError(f"requested cases are not ready: {missing_ids}")
        cases = [discovered[case_id] for case_id in selected_ids]
    else:
        cases = [discovered[case_id] for case_id in sorted(discovered)]
    if max_cases is not None:
        cases = cases[:max_cases]
    if not cases:
        raise ValueError("no ready camera-cal cases were selected")
    return cases


def case_paths(
    case_root: Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    paths = manifest.get("paths")
    paths = paths if isinstance(paths, dict) else {}
    evidence = paths.get("evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "scene": case_root
        / str(paths.get("canonical_scene") or "scene/canonical_scene.json"),
        "blend": case_root / str(paths.get("blend") or "prepared/evaluation.blend"),
        "annotation": case_root
        / str(paths.get("annotation") or "annotation.json"),
        "perspective": case_root
        / str(evidence.get("perspective") or "evidence/standardized_perspective.png"),
        "top": case_root
        / str(evidence.get("top") or "evidence/standardized_top.png"),
        "identity": case_root
        / str(evidence.get("identity") or "evidence/standardized_identity_map.png"),
        "render_manifest": case_root / "evidence" / "prepared_render_manifest.json",
        "collision_geometry": (
            case_root / "evidence" / "collision_geometry_manifest.json"
        ),
    }


def renderer_config_from_args(
    args: argparse.Namespace,
    *,
    blender_bin: Path,
) -> dict[str, Any]:
    return {
        "blender_bin": str(blender_bin),
        "timeout_seconds": int(args.blender_timeout_seconds),
        "width": int(args.render_width),
        "height": int(args.render_height),
        "render_engine": str(args.render_engine),
        "cycles_device": str(args.cycles_device),
        "cycles_samples": int(args.cycles_samples),
        "cycles_denoising": bool(args.cycles_denoising),
        "preview_render_engine": str(args.preview_render_engine),
        "preview_width": int(args.preview_width),
        "preview_height": int(args.preview_height),
        "preview_cycles_samples": int(args.preview_cycles_samples),
    }


def resolved_control() -> Any:
    return resolve_vlm_evaluation_control(
        {
            "camera_acquisition": {
                "policy": "deterministic_then_vlm",
                "deterministic": {
                    "max_rounds": 1,
                    "candidate_budget": 6,
                    "max_selected_views": 2,
                },
                "vlm": {
                    "max_rounds": 1,
                    "selection_mode": "repair_plan",
                    "max_selected_views": 2,
                },
                "total": {
                    "max_evidence_rounds": 2,
                    "max_total_images": 6,
                    "max_selector_calls": 3,
                    "max_camera_actions": 2,
                },
            },
            "budgets": {
                "max_evidence_rounds": 2,
                "max_views_per_round": 2,
                "max_total_images": 6,
                "max_selector_calls": 3,
                "max_camera_actions": 2,
            },
        },
        existing_max_views=2,
        existing_max_steps=1,
        existing_selector_available=True,
        judge_max_images=5,
    )


def build_experiment_plan(
    *,
    dataset_root: Path,
    output_root: Path,
    grouping_config_path: Path,
    route: dict[str, Any],
    metrics: tuple[str, ...],
    cases: list[dict[str, Any]],
    renderer_config: dict[str, Any],
    control: dict[str, Any],
    max_workers: int,
    resume: bool,
    continue_on_error: bool,
) -> dict[str, Any]:
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "runner_schema_version": RUNNER_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "source_cases_read_only": True,
        "source_prompt_used": False,
        "prompt_policy": "metric_rubrics_only_no_generation_prompt",
        "l3_metric_prompt_version": L3_METRIC_PROMPT_VERSION,
        "layers": {
            L1: {
                "enabled": True,
                "scope": "scene_level",
                "metrics": list(L1_METRICS),
                "backend": "deterministic_evidence_plus_conditional_vlm",
                "binary_failure_policy": deepcopy(
                    L1_BINARY_FAILURE_POLICY
                ),
            },
            L2: {
                "enabled": False,
                "reason": "promptless_camera_cal_experiment",
            },
            L3: {
                "enabled": True,
                "metrics": list(metrics),
                "scope": "metric_policy_then_scene_level_aggregation",
                "functional_global_probe_policy": {
                    "enabled_when_metric_selected": (
                        "functional_consistency" in metrics
                    ),
                    "planner_input": (
                        "one_global_image_plus_id_category_groups_boundary"
                    ),
                    "discovery_schema_version": (
                        FUNCTIONAL_DISCOVERY_SCHEMA_VERSION
                    ),
                    "affordance_schema_version": (
                        FUNCTIONAL_AFFORDANCE_SCHEMA_VERSION
                    ),
                    "relation_schema_version": (
                        FUNCTIONAL_RELATION_SCHEMA_VERSION
                    ),
                    "discovery_outputs": [
                        "directed_surface_targets",
                        "within_group_correspondences",
                        "cross_group_correspondences",
                        "approach_clearance_targets",
                        "boundary_sensitive_targets",
                        "unusual_unconfirmed",
                    ],
                    "unusual_confirmation_scope": "group_local",
                    "usable_surface_decoder": {
                        "trusted_side_ids": [
                            "local_pos_x",
                            "local_neg_x",
                            "local_pos_y",
                            "local_neg_y",
                        ],
                        "decode_scope": (
                            "directed_or_uncertain_clearance_targets_"
                            "before_probe_budget"
                        ),
                        "freeform_pose": False,
                    },
                    "probe_kinds": [
                        "functional_frontage",
                        "functional_correspondence",
                        "approach_clearance",
                    ],
                    "max_probe_units": 4,
                    "candidate_count_by_probe_kind": {
                        "functional_frontage": 4,
                        "functional_correspondence": 4,
                        "approach_clearance": 4,
                    },
                    "selected_raw_views_per_unit": 1,
                    "preferred_lens_mm": 32.0,
                    "elevation_range_degrees": [8.0, 16.0],
                    "source_scene_modified": False,
                    "judge_presentation": "raw_rgb_only",
                },
                "placement_discovery_schema_version": (
                    PLACEMENT_DISCOVERY_SCHEMA_VERSION
                ),
            },
            L4: {"enabled": False},
        },
        "model_route": safe_route_manifest(route),
        "grouping": {
            "config_path": str(grouping_config_path),
            "config_sha256": file_sha256(grouping_config_path),
        },
        "renderer": deepcopy(renderer_config),
        "control": deepcopy(control),
        "observability": {
            "terminal_progress_default": True,
            "progress_jsonl": str(
                (output_root / "progress.jsonl").resolve()
            ),
            "case_api_calls_jsonl": "cases/<case_id>/api_calls.jsonl",
            "case_api_usage_json": "cases/<case_id>/api_usage.json",
            "api_call_definition": (
                "one logical OpenAI-compatible chat-completions invocation; "
                "transport retries inside that invocation are not counted "
                "separately"
            ),
            "token_usage_source": (
                "endpoint response usage fields only; never estimated"
            ),
        },
        "max_workers": max_workers,
        "resume": resume,
        "continue_on_error": continue_on_error,
        "case_count": len(cases),
        "cases": deepcopy(cases),
    }


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
    progress: ProgressReporter | None = None,
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
        grouping_config=grouping_config,
        renderer_config=renderer_config,
        control_config=control_config,
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
        "l1_binary_failure_policy": deepcopy(
            L1_BINARY_FAILURE_POLICY
        ),
        "progress_path": str(progress.path),
        "api_calls_path": str(api_tracker.calls_path),
        "api_usage_path": str(api_tracker.usage_path),
        "api_usage": api_tracker.summary(),
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
        layers=[L1, L3],
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
            evaluation_profile=promptless_l1_l3_profile(),
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
            scene_quality_config=scene_quality_config(metrics),
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
    final_decision_status = (
        "resolved"
        if l1_report.get("status") == "evaluated" and not l1_failures
        else "unresolved"
    )
    diagnostic_reason = (
        "l1_engineering_failure"
        if l1_failures
        else "l1_unresolved"
        if final_decision_status == "unresolved"
        else None
    )
    comparison = build_scene_comparison(
        case_id=case_id,
        annotation=annotation,
        scene_quality_report=l3_report,
        metrics=metrics,
    )
    comparison["diagnostic_only"] = (
        final_decision_status == "unresolved"
    )
    comparison["diagnostic_reason"] = diagnostic_reason
    l1_diagnostics = {
        "policy": deepcopy(L1_BINARY_FAILURE_POLICY),
        "final_decision_status": final_decision_status,
        "engineering_failure_count": len(l1_failures),
        "engineering_failures": l1_failures,
        "response_schema_validation": schema_validation,
        "l3_diagnostics_completed": True,
    }
    report["runner_outcome"] = {
        "final_decision_status": final_decision_status,
        "l1_engineering_failure": bool(l1_failures),
        "l3_results_are_diagnostic_only": (
            final_decision_status == "unresolved"
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
    case_run_manifest.update(
        status="complete",
        completed_at=utc_now(),
        elapsed_seconds=elapsed,
        grouping_status=grouping_report.get("status"),
        l1_status=l1_report.get("status"),
        l3_status=l3_report.get("status"),
        final_decision_status=final_decision_status,
        l1_engineering_failure=bool(l1_failures),
        l1_engineering_failure_count=len(l1_failures),
        binary_response_schema_validation=schema_validation,
        api_usage=api_usage,
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
    }


def model_config(route: dict[str, Any], *, role: str) -> dict[str, Any]:
    max_images = 8 if role == "camera-selector" else 5
    return {
        "name": f"camera-cal-{role}",
        "endpoint": route["endpoint"],
        "model": route["model"],
        "api_key_env": route["api_key_env"],
        "temperature": 0.0,
        "send_temperature": False,
        "max_tokens": 2048,
        "timeout_seconds": 3000,
        "response_format_json": False,
        "max_retries": 1,
        "retry_backoff_seconds": 1.0,
        "max_images": max_images,
        "max_preview_images": max_images,
        "max_context_chars": 120000,
        "require_api_key": True,
    }


def build_grouping_model(route: dict[str, Any]) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        name="camera-cal-grouping",
        endpoint=str(route["endpoint"]),
        model_id=str(route["model"]),
        api_key_env=str(route["api_key_env"]),
        temperature=0.0,
        max_tokens=GROUPING_COMPLETION_MAX_TOKENS,
        timeout_seconds=3000,
        response_format_json=False,
        max_retries=1,
        retry_backoff_seconds=1.0,
        send_temperature=False,
        require_api_key=True,
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
        L1: 7.0 / 15.0,
        L2: 0.0,
        L3: 8.0 / 15.0,
        L4: 0.0,
    }
    profile[L2]["enabled"] = False
    for metric in profile[L2]["metrics"].values():
        metric["enabled"] = False
        metric["weight"] = 0.0
    return profile


def scene_quality_config(metrics: tuple[str, ...]) -> dict[str, Any]:
    selected = set(metrics)
    return {
        "enabled": True,
        "metrics": {
            metric: {
                "enabled": metric in selected,
                "weight": 1.0 if metric in selected else 0.0,
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
    if not path.is_file():
        return {}
    manifest = read_json(path)
    legend = manifest.get("identity_legend")
    if not isinstance(legend, dict):
        return {}
    return {
        str(alias): str(object_id)
        for alias, object_id in legend.items()
        if str(alias).strip() and str(object_id).strip()
    }


def grouping_evidence_packet(
    *,
    paths: dict[str, Path],
    identity_legend: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "path": str(paths["perspective"].resolve()),
            "role": "global_perspective_rgb",
            "representation": "rgb",
            "view_id": "global_perspective",
            "camera_scope": "global",
        },
        {
            "path": str(paths["top"].resolve()),
            "role": "global_top_rgb",
            "representation": "rgb",
            "view_id": "global_top",
            "camera_scope": "global",
        },
        {
            "path": str(paths["identity"].resolve()),
            "role": "global_identity_overlay",
            "representation": "identity_map",
            "view_id": "global_identity",
            "camera_scope": "global",
            "identity_overlay": True,
            "identity_legend": deepcopy(identity_legend),
        },
    ]


def case_input_fingerprint(
    *,
    case: dict[str, Any],
    case_manifest: dict[str, Any],
    paths: dict[str, Path],
    route: dict[str, Any],
    metrics: tuple[str, ...],
    grouping_config: dict[str, Any],
    renderer_config: dict[str, Any],
    control_config: dict[str, Any],
) -> str:
    critical = case_manifest.get("critical_artifact_hashes")
    critical = critical if isinstance(critical, dict) else {}
    prompt_path = (
        PROJECT_ROOT / "src" / "benchmark" / "visual_judge" / "l3_prompts.py"
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
            "selected_l3_metrics": list(metrics),
            "source_prompt_used": False,
            "l3_metric_prompt_version": L3_METRIC_PROMPT_VERSION,
            "l3_prompt_source_sha256": file_sha256(prompt_path),
            "functional_probe_implementation_sha256": {
                relative: file_sha256(PROJECT_ROOT / relative)
                for relative in FUNCTIONAL_PROBE_IMPLEMENTATION_FILES
            },
            "profile": promptless_l1_l3_profile(),
            "scene_quality_config": scene_quality_config(metrics),
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
    return bool(
        manifest.get("status") == "complete"
        and manifest.get("input_fingerprint") == expected_fingerprint
        and (case_out / "evaluation_report.json").is_file()
        and (case_out / "grouping.json").is_file()
        and (case_out / "l1_report.json").is_file()
        and (case_out / "l1_diagnostics.json").is_file()
        and (case_out / "scene_quality_report.json").is_file()
        and (case_out / "scene_comparison.json").is_file()
        and (case_out / "control_manifest.json").is_file()
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
    if report.get("status") != "evaluated":
        return "unresolved"
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
        atomic_write_json(manifest_path, manifest)
    return failure


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
    grouping_failures = 0
    final_unresolved = 0
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
        if record.get("status") not in {"complete", "resumed"}:
            for summary in metric_summaries.values():
                summary["case_failures"] += 1
            continue
        successful += 1
        if record.get("final_decision_status") == "unresolved":
            final_unresolved += 1
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
            if record.get("final_decision_status") == "unresolved":
                summary["diagnostic_only_cases"] += 1
            expected = human.get("expected")
            predicted = model.get("prediction")
            if expected in {"valid", "invalid"}:
                summary["human_distribution"][expected] += 1
            if predicted in {"valid", "invalid"}:
                summary["predicted_distribution"][predicted] += 1
                summary["evaluated"] += 1
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
            "failed": len(case_records) - successful,
            "grouping_failures": grouping_failures,
            "final_unresolved": final_unresolved,
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
    return {
        "total": total,
        "evaluated": 0,
        "unresolved": 0,
        "excluded_unclear": 0,
        "correct": 0,
        "incorrect": 0,
        "accuracy": None,
        "accuracy_scope": "scene_level_metric_verdict",
        "anomaly_object_cases": 0,
        "anomaly_object_exact_correct": 0,
        "anomaly_object_exact_incorrect": 0,
        "anomaly_object_exact_accuracy": None,
        "anomaly_object_true_positive": 0,
        "anomaly_object_false_negative": 0,
        "anomaly_object_false_positive": 0,
        "anomaly_object_precision": None,
        "anomaly_object_recall": None,
        "anomaly_object_f1": None,
        "predicted_distribution": {"valid": 0, "invalid": 0},
        "human_distribution": {"valid": 0, "invalid": 0},
        "grouping_failures": 0,
        "diagnostic_only_cases": 0,
        "case_failures": 0,
        "camera_render_failures": 0,
        "judge_failures": 0,
        "judge_calls": 0,
        "vlm_selector_calls": 0,
        "initial_image_count": 0,
        "final_image_count": 0,
        "preview_image_count": 0,
        "evidence_repair_count": 0,
        "evidence_recovery_count": 0,
    }


def telemetry_by_metric(control_manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    integration = control_manifest.get("integration")
    integration = integration if isinstance(integration, dict) else {}
    runtime = integration.get("runtime")
    runtime = runtime if isinstance(runtime, dict) else {}
    calls = runtime.get("controlled_calls")
    calls = calls if isinstance(calls, list) else []
    result: dict[str, dict[str, int]] = {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        metric = str(call.get("metric") or "")
        if not metric:
            continue
        target = result.setdefault(
            metric,
            {
                "judge_calls": 0,
                "vlm_selector_calls": 0,
                "preview_image_count": 0,
                "final_image_count": 0,
                "evidence_repair_count": 0,
                "evidence_recovery_count": 0,
            },
        )
        audit = call.get("audit")
        audit = audit if isinstance(audit, dict) else {}
        telemetry = audit.get("experiment_telemetry")
        telemetry = telemetry if isinstance(telemetry, dict) else {}
        target["judge_calls"] += int(telemetry.get("judge_calls") or 0)
        target["vlm_selector_calls"] += int(
            telemetry.get("vlm_selector_calls") or 0
        )
        target["preview_image_count"] += int(
            telemetry.get("preview_render_count") or 0
        )
        target["final_image_count"] += int(
            telemetry.get("final_render_count") or 0
        )
        if int(audit.get("rounds_used") or 0) > 0:
            target["evidence_repair_count"] += 1
        evaluation = audit.get("evaluation")
        evaluation = evaluation if isinstance(evaluation, dict) else {}
        if evaluation.get("evidence_recovery_outcome") in {
            "recovered",
            "recovered_after_repair",
        }:
            target["evidence_recovery_count"] += 1
    return result


def initial_image_count(metric_report: dict[str, Any]) -> int:
    group_results = metric_report.get("group_results")
    if isinstance(group_results, list):
        return sum(
            len(item.get("evidence_paths") or [])
            for item in group_results
            if isinstance(item, dict)
        )
    paths = metric_report.get("evidence_paths")
    return len(paths) if isinstance(paths, list) else 0


def metric_failure_counts(metric_report: dict[str, Any]) -> dict[str, int]:
    camera_failures = 0
    judge_failures = 0
    group_results = metric_report.get("group_results")
    group_results = group_results if isinstance(group_results, list) else []
    for item in group_results:
        if not isinstance(item, dict):
            continue
        resolution = item.get("evidence_resolution")
        resolution = resolution if isinstance(resolution, dict) else {}
        if resolution.get("provider_status") == "failed":
            camera_failures += 1
        if item.get("reason") == "vlm_judge_failed":
            judge_failures += 1
    if metric_report.get("reason") == "vlm_judge_failed":
        judge_failures += 1
    return {
        "camera_render_failures": camera_failures,
        "judge_failures": judge_failures,
    }


def read_api_call_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def api_usage_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    safe_records = [
        item for item in records if isinstance(item, dict)
    ]
    overall = _api_usage_bucket(safe_records)
    roles = sorted(
        {
            str(item.get("role") or "unknown")
            for item in safe_records
        }
    )
    call_types = sorted(
        {
            str(item.get("call_type") or "chat")
            for item in safe_records
        }
    )
    functional_affordance_type = (
        "vlm_camera_pose.functional_discovery.affordance"
    )
    functional_relation_type = (
        "vlm_camera_pose.functional_discovery.relations"
    )
    placement_discovery_type = (
        "vlm_camera_pose.placement_discovery"
    )
    usable_surface_type = "vlm_camera_pose.usable_surface_decode"
    camera_selector_types = {
        call_type
        for call_type in call_types
        if call_type.startswith("camera_selector_")
        or call_type
        in {
            "vlm_camera_pose.active_fallback",
            "vlm_camera_pose.query_cov",
        }
    }
    return {
        "schema_version": API_USAGE_SCHEMA_VERSION,
        "api_call_definition": (
            "logical OpenAI-compatible chat-completions invocation"
        ),
        "transport_retries_counted_separately": False,
        "token_usage_source": "endpoint_response_usage",
        "token_usage_estimated": False,
        "operation_calls": {
            "functional_discovery": sum(
                str(item.get("call_type") or "chat")
                in {
                    functional_affordance_type,
                    functional_relation_type,
                }
                for item in safe_records
            ),
            "functional_affordance": sum(
                str(item.get("call_type") or "chat")
                == functional_affordance_type
                for item in safe_records
            ),
            "functional_relation": sum(
                str(item.get("call_type") or "chat")
                == functional_relation_type
                for item in safe_records
            ),
            "placement_discovery": sum(
                str(item.get("call_type") or "chat")
                == placement_discovery_type
                for item in safe_records
            ),
            "usable_surface_decoder": sum(
                str(item.get("call_type") or "chat")
                == usable_surface_type
                for item in safe_records
            ),
            "camera_selector": sum(
                str(item.get("call_type") or "chat")
                in camera_selector_types
                for item in safe_records
            ),
            "judge": sum(
                str(item.get("role") or "unknown") == "judge"
                for item in safe_records
            ),
        },
        **overall,
        "by_role": {
            role: _api_usage_bucket(
                [
                    item
                    for item in safe_records
                    if str(item.get("role") or "unknown") == role
                ]
            )
            for role in roles
        },
        "by_call_type": {
            call_type: _api_usage_bucket(
                [
                    item
                    for item in safe_records
                    if str(item.get("call_type") or "chat")
                    == call_type
                ]
            )
            for call_type in call_types
        },
    }


def _api_usage_bucket(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    api_calls = len(records)
    successful = sum(
        item.get("status") == "complete" for item in records
    )
    failed = sum(item.get("status") == "failed" for item in records)
    usage_records = [
        item["tokens_usage"]
        for item in records
        if isinstance(item.get("tokens_usage"), dict)
    ]
    token_totals: dict[str, int] = {}
    for field in _TOKEN_FIELDS:
        values = [
            usage[field]
            for usage in usage_records
            if _nonnegative_int_or_none(usage.get(field)) is not None
        ]
        if values:
            token_totals[field] = sum(int(value) for value in values)
    if not api_calls:
        coverage = "not_applicable"
    elif len(usage_records) == api_calls:
        coverage = "complete"
    elif usage_records:
        coverage = "partial"
    else:
        coverage = "unavailable"
    return {
        "api_calls_number": api_calls,
        "successful_api_calls": successful,
        "failed_api_calls": failed,
        "token_usage_reported_calls": len(usage_records),
        "token_usage_missing_calls": api_calls - len(usage_records),
        "token_usage_coverage": coverage,
        "tokens_usage": token_totals or None,
    }


def _normalized_token_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, int] = {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": (
            "completion_tokens",
            "output_tokens",
        ),
        "total_tokens": ("total_tokens",),
    }
    for target, candidates in aliases.items():
        for source in candidates:
            parsed = _nonnegative_int_or_none(value.get(source))
            if parsed is not None:
                result[target] = parsed
                break
    prompt_details = value.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached = _nonnegative_int_or_none(
            prompt_details.get("cached_tokens")
        )
        if cached is not None:
            result["cached_prompt_tokens"] = cached
    completion_details = value.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning = _nonnegative_int_or_none(
            completion_details.get("reasoning_tokens")
        )
        if reasoning is not None:
            result["reasoning_tokens"] = reasoning
    if (
        "total_tokens" not in result
        and "prompt_tokens" in result
        and "completion_tokens" in result
    ):
        result["total_tokens"] = (
            result["prompt_tokens"] + result["completion_tokens"]
        )
    return result or None


def _safe_request_metadata(model: Any) -> dict[str, Any]:
    value = getattr(model, "last_request_metadata", None)
    if not isinstance(value, dict):
        return {}
    allowed = {
        "endpoint",
        "model",
        "message_count",
        "image_count",
        "prompt_chars",
        "finish_reason",
        "usage",
    }
    return {
        key: deepcopy(item)
        for key, item in value.items()
        if key in allowed
    }


def _message_image_count(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            total += sum(
                isinstance(item, dict)
                and item.get("type") == "image_url"
                for item in content
            )
    return total


def _request_scope(request: Any) -> tuple[str, str]:
    if not isinstance(request, dict):
        return "unknown", "scene"
    metric = str(request.get("metric") or "unknown")
    group_scope = request.get("group_scope")
    if isinstance(group_scope, dict) and group_scope.get("group_id"):
        return metric, str(group_scope["group_id"])
    object_ids = request.get("object_ids")
    if isinstance(object_ids, list) and object_ids:
        return metric, "+".join(str(value) for value in object_ids)
    event = request.get("event")
    if isinstance(event, dict):
        event_ids = event.get("object_ids")
        if isinstance(event_ids, list) and event_ids:
            return metric, "+".join(str(value) for value in event_ids)
        pair = [
            event.get("object_a_id"),
            event.get("object_b_id"),
        ]
        pair = [str(value) for value in pair if value]
        if pair:
            return metric, "+".join(pair)
    return metric, "scene"


def _evidence_count(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, dict):
        for key in (
            "visual_evidence",
            "render_evidence_items",
            "render_evidence",
            "paths",
            "candidates",
        ):
            items = value.get(key)
            if isinstance(items, (list, tuple)):
                return len(items)
        return None
    for name in ("visual_evidence", "candidates"):
        items = getattr(value, name, None)
        if isinstance(items, (list, tuple)):
            return len(items)
    return None


def _nonnegative_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _bounded_error(error: Exception) -> str:
    value = str(error)
    return value if len(value) <= 1000 else value[:997] + "..."


def _format_progress_record(record: dict[str, Any]) -> str:
    timestamp = str(record.get("timestamp") or "")
    clock = timestamp[11:19] if len(timestamp) >= 19 else timestamp
    case_id = str(record.get("case_id") or "run")
    event = str(record.get("event") or "progress")
    details = record.get("details")
    details = details if isinstance(details, dict) else {}
    preferred = (
        "phase",
        "metric",
        "group_id",
        "role",
        "call_type",
        "status",
        "api_call_number",
        "cumulative_api_calls",
        "duration_seconds",
        "evidence_count",
        "error_type",
    )
    fragments: list[str] = []
    for key in preferred:
        value = details.get(key)
        if value is not None:
            fragments.append(f"{key}={_progress_value(value)}")
    if isinstance(details.get("tokens_usage"), dict):
        fragments.append(
            "tokens="
            + _progress_value(details["tokens_usage"])
        )
    suffix = " " + " ".join(fragments) if fragments else ""
    return f"[{clock}] [{case_id}] {event}{suffix}"


def _progress_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return str(value).replace("\n", " ")


def safe_route_manifest(route: dict[str, Any]) -> dict[str, Any]:
    return {
        "endpoint": route["endpoint"],
        "model": route["model"],
        "api_key_env": route["api_key_env"],
        "authorization_configured": bool(
            route.get("authorization_configured")
        ),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_yaml_object(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML object: {path}")
    return value


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    main()
