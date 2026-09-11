"""API-call accounting and report telemetry for camera-cal runs."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable

from benchmark.camera_cal_scene_level.io import atomic_write_json, utc_now
from benchmark.camera_cal_scene_level.progress import ProgressReporter


API_CALL_SCHEMA_VERSION = "camera_cal_api_call_v1"
API_USAGE_SCHEMA_VERSION = "camera_cal_api_usage_v1"
TOKEN_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_prompt_tokens",
    "reasoning_tokens",
)


def empty_metric_summary(*, total: int) -> dict[str, Any]:
    return {
        "total": total,
        "evaluated": 0,
        "unresolved": 0,
        "infrastructure_failure": 0,
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


def api_usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    safe_records = [item for item in records if isinstance(item, dict)]
    overall = api_usage_bucket(safe_records)
    roles = sorted({str(item.get("role") or "unknown") for item in safe_records})
    call_types = sorted(
        {str(item.get("call_type") or "chat") for item in safe_records}
    )
    affordance = "vlm_camera_pose.functional_discovery.affordance"
    relation = "vlm_camera_pose.functional_discovery.relations"
    placement = "vlm_camera_pose.placement_discovery"
    surface = "vlm_camera_pose.usable_surface_decode"
    selector_types = {
        call_type
        for call_type in call_types
        if call_type.startswith("camera_selector_")
        or call_type
        in {"vlm_camera_pose.active_fallback", "vlm_camera_pose.query_cov"}
    }

    def family(item: dict[str, Any], base: str) -> bool:
        actual = str(item.get("call_type") or "chat")
        return actual == base or actual.startswith(base + ".schema_repair")

    return {
        "schema_version": API_USAGE_SCHEMA_VERSION,
        "api_call_definition": "logical OpenAI-compatible chat-completions invocation",
        "transport_retries_counted_separately": False,
        "token_usage_source": "endpoint_response_usage",
        "token_usage_estimated": False,
        "operation_calls": {
            "functional_discovery": sum(
                family(item, affordance) or family(item, relation)
                for item in safe_records
            ),
            "functional_affordance": sum(
                family(item, affordance) for item in safe_records
            ),
            "functional_relation": sum(
                family(item, relation) for item in safe_records
            ),
            "placement_discovery": sum(
                str(item.get("call_type") or "chat") == placement
                for item in safe_records
            ),
            "usable_surface_decoder": sum(
                str(item.get("call_type") or "chat") == surface
                for item in safe_records
            ),
            "camera_selector": sum(
                str(item.get("call_type") or "chat") in selector_types
                for item in safe_records
            ),
            "judge": sum(
                str(item.get("role") or "unknown") == "judge"
                for item in safe_records
            ),
        },
        **overall,
        "by_role": {
            role: api_usage_bucket(
                [
                    item
                    for item in safe_records
                    if str(item.get("role") or "unknown") == role
                ]
            )
            for role in roles
        },
        "by_call_type": {
            call_type: api_usage_bucket(
                [
                    item
                    for item in safe_records
                    if str(item.get("call_type") or "chat") == call_type
                ]
            )
            for call_type in call_types
        },
    }


def api_usage_bucket(records: list[dict[str, Any]]) -> dict[str, Any]:
    api_calls = len(records)
    successful = sum(item.get("status") == "complete" for item in records)
    failed = sum(item.get("status") == "failed" for item in records)
    usage_records = [
        item["tokens_usage"]
        for item in records
        if isinstance(item.get("tokens_usage"), dict)
    ]
    token_totals: dict[str, int] = {}
    for field in TOKEN_FIELDS:
        values = [
            usage[field]
            for usage in usage_records
            if nonnegative_int_or_none(usage.get(field)) is not None
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


def normalized_token_usage(value: Any) -> dict[str, int] | None:
    if not isinstance(value, dict):
        return None
    result: dict[str, int] = {}
    aliases = {
        "prompt_tokens": ("prompt_tokens", "input_tokens"),
        "completion_tokens": ("completion_tokens", "output_tokens"),
        "total_tokens": ("total_tokens",),
    }
    for target, candidates in aliases.items():
        for source in candidates:
            parsed = nonnegative_int_or_none(value.get(source))
            if parsed is not None:
                result[target] = parsed
                break
    prompt_details = value.get("prompt_tokens_details")
    if isinstance(prompt_details, dict):
        cached = nonnegative_int_or_none(prompt_details.get("cached_tokens"))
        if cached is not None:
            result["cached_prompt_tokens"] = cached
    completion_details = value.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning = nonnegative_int_or_none(
            completion_details.get("reasoning_tokens")
        )
        if reasoning is not None:
            result["reasoning_tokens"] = reasoning
    if (
        "total_tokens" not in result
        and "prompt_tokens" in result
        and "completion_tokens" in result
    ):
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result or None


def safe_request_metadata(model: Any) -> dict[str, Any]:
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
    return {key: deepcopy(item) for key, item in value.items() if key in allowed}


def message_image_count(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    total = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            total += sum(
                isinstance(item, dict) and item.get("type") == "image_url"
                for item in content
            )
    return total


def request_scope(request: Any) -> tuple[str, str]:
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
        pair = [event.get("object_a_id"), event.get("object_b_id")]
        pair = [str(value) for value in pair if value]
        if pair:
            return metric, "+".join(pair)
    return metric, "scene"


def evidence_count(value: Any) -> int | None:
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


def nonnegative_int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def bounded_error(error: Exception) -> str:
    value = str(error)
    return value if len(value) <= 1000 else value[:997] + "..."


class APICallTracker:
    """Record logical chat-completions calls and reported token usage."""

    def __init__(
        self,
        *,
        case_id: str,
        calls_path: Path,
        usage_path: Path,
        progress: ProgressReporter,
        model_route_abort_signal: Any = None,
        observed_model_factory: Callable[..., Any] | None = None,
        read_records: Callable[[Path], list[dict[str, Any]]] = read_api_call_records,
        write_json: Callable[[Path, Any], None] = atomic_write_json,
        usage_summary: Callable[[list[dict[str, Any]]], dict[str, Any]] = api_usage_summary,
        clock: Callable[[], str] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.case_id = str(case_id)
        self.calls_path = calls_path.expanduser().resolve()
        self.usage_path = usage_path.expanduser().resolve()
        self.progress = progress
        self.model_route_abort_signal = model_route_abort_signal
        self._observed_model_factory = observed_model_factory
        self._read_records = read_records
        self._write_json = write_json
        self._usage_summary = usage_summary
        self._clock = clock
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._records = self._read_records(self.calls_path)
        self._next_call_number = len(self._records) + 1
        self._write_json(self.usage_path, self._usage_summary(self._records))

    def observe_model(self, model: Any, *, role: str) -> Any:
        if not callable(getattr(model, "chat_messages", None)):
            return model
        if self._observed_model_factory is None:
            raise RuntimeError("observed model factory is unavailable")
        return self._observed_model_factory(model, role=role, tracker=self)

    def begin_call(
        self, *, role: str, call_type: str, messages: Any
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
            image_count=message_image_count(messages),
        )
        return call_number, call_id, self._monotonic()

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
        duration = max(0.0, self._monotonic() - started)
        usage = normalized_token_usage(request_metadata.get("usage"))
        record = {
            "schema_version": API_CALL_SCHEMA_VERSION,
            "api_call_number": int(call_number),
            "api_call_id": str(call_id),
            "case_id": self.case_id,
            "role": str(role),
            "call_type": str(call_type),
            "status": "failed" if error is not None else "complete",
            "completed_at": self._clock(),
            "duration_seconds": duration,
            "model": request_metadata.get("model"),
            "endpoint": request_metadata.get("endpoint"),
            "message_count": nonnegative_int_or_none(
                request_metadata.get("message_count")
            ),
            "image_count": nonnegative_int_or_none(
                request_metadata.get("image_count")
            ),
            "prompt_chars": nonnegative_int_or_none(
                request_metadata.get("prompt_chars")
            ),
            "finish_reason": request_metadata.get("finish_reason"),
            "tokens_usage": usage,
            "error_type": type(error).__name__ if error is not None else None,
            "error": bounded_error(error) if error is not None else None,
        }
        with self._lock:
            self._records.append(record)
            self.calls_path.parent.mkdir(parents=True, exist_ok=True)
            with self.calls_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            cumulative = self._usage_summary(self._records)
            self._write_json(self.usage_path, cumulative)
        self.progress.emit(
            "api_call_failed" if error is not None else "api_call_completed",
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
            return self._usage_summary(self._records)
