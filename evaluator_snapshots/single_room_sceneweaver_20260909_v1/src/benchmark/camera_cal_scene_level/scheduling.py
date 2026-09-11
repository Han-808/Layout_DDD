"""Case scheduling and run-shared model-route abort mechanics."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import threading
from pathlib import Path
from typing import Any, Callable

from benchmark.camera_cal_scene_level.telemetry import bounded_error
from benchmark.models import EndpointConfigurationError


class ModelRouteAbortSignal:
    """Run-shared circuit breaker for permanent endpoint route failures."""

    def __init__(
        self,
        *,
        error_formatter: Callable[[Exception], str] = bounded_error,
        abort_error_factory: Callable[[str], Exception] = EndpointConfigurationError,
    ) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._error_formatter = error_formatter
        self._abort_error_factory = abort_error_factory
        self._error_type: str | None = None
        self._error: str | None = None

    def trip(self, error: Exception) -> None:
        with self._lock:
            if not self._event.is_set():
                self._error_type = type(error).__name__
                self._error = self._error_formatter(error)
                self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def raise_if_set(self) -> None:
        if self._event.is_set():
            raise self._abort_error_factory(
                "model route was disabled after a permanent upstream "
                "configuration failure"
            )

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "triggered": self._event.is_set(),
                "error_type": self._error_type,
                "error": self._error,
            }


def run_cases_parallel(
    *,
    cases: list[dict[str, Any]],
    case_kwargs: dict[str, Any],
    output_root: Path,
    progress: Any,
    max_workers: int,
    continue_on_error: bool,
    run_case_fn: Callable[..., dict[str, Any]],
    failure_recorder: Callable[..., dict[str, Any]],
    cancellation_recorder: Callable[..., dict[str, Any]],
    model_route_abort_signal: ModelRouteAbortSignal | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run cases concurrently without losing already-started final states."""

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    fail_fast_triggered = False
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case: dict[Future[dict[str, Any]], dict[str, Any]] = {
            executor.submit(run_case_fn, case=case, **case_kwargs): case
            for case in cases
        }
        for completed, future in enumerate(
            as_completed(future_to_case),
            start=1,
        ):
            case = future_to_case[future]
            if future.cancelled():
                cancellation = cancellation_recorder(
                    case=case,
                    output_root=output_root,
                )
                records.append(cancellation)
                progress.emit(
                    "case_cancelled",
                    case_id=str(case["case_id"]),
                    completed_count=completed,
                    case_count=len(cases),
                    reason=cancellation["reason"],
                )
                continue
            try:
                record = future.result()
            except Exception as exc:
                failure = failure_recorder(
                    case=case,
                    output_root=output_root,
                    error=exc,
                )
                failures.append(failure)
                records.append(failure)
                progress.emit(
                    "case_failed",
                    case_id=str(case["case_id"]),
                    completed_count=completed,
                    case_count=len(cases),
                    error_type=failure["error_type"],
                )
                if not continue_on_error and not fail_fast_triggered:
                    fail_fast_triggered = True
                    for pending in future_to_case:
                        if pending is not future:
                            pending.cancel()
            else:
                records.append(record)
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
            route_aborted = bool(
                model_route_abort_signal is not None
                and model_route_abort_signal.is_set()
            )
            if route_aborted and not fail_fast_triggered:
                fail_fast_triggered = True
                for pending in future_to_case:
                    if pending is not future:
                        pending.cancel()
    return records, failures
