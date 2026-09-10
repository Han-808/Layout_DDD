"""Opt-in instrumentation/checkpoint overlay, installed in an isolated worker.

Frozen metric functions are called unchanged. Failures never become verdicts.
The only retry override requires explicit original transient exception types.
"""
from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import functools
import math
from pathlib import Path
import re
import time
from typing import Any, Callable

from . import VERSION
from .recovery import Circuit, ChannelUnavailable, RetryBudgetExhausted, RoomBudget
from . import results
from . import reasons
from .safety import (Heartbeat, IntegrityError, Policy, atomic_json, best_effort_record,
                     check_storage, diagnostic_projection, digest, exception_record,
                     identity, read_json, redact, room_slot, stamp)


@dataclass
class RoomContext:
    root: Path
    key: str
    heartbeat: Heartbeat
    policy: Policy
    run_identity: str
    attempt: Path | None = None
    failures: dict[str, Any] = field(default_factory=dict)
    budget: RoomBudget | None = None
    circuit: Circuit | None = None
    guard_error: Exception | None = None
    renderer_origins: dict[str, Any] = field(default_factory=dict)
    materialization_identity: str | None = None
    validator_origins: dict[str, list[dict]] = field(default_factory=dict)

    def ensure_execution(self):
        # A different room's outage must not invalidate our already acquired
        # complete result. Only our own swallowed guard is sticky here.
        if self.guard_error is not None:
            raise self.guard_error
        if self.budget is not None:
            self.budget.ensure()

    def observe(self, name: str, raw: Any) -> None:
        self.heartbeat.update(self.key, stage=name)
        self.failures[name] = raw
        if self.attempt is not None:
            best_effort_record(self.attempt / "execution_diagnostics" / f"{name}.json", {
                "schema_version": VERSION, "stage": name, "observed_at": stamp(),
                "room": self.key, "result": raw,
            })


CURRENT: ContextVar[RoomContext | None] = ContextVar("nonrect_execution_context", default=None)
_ARTIFACT_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".exr", ".blend", ".json"}


def _key_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _key_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_key_value(item) for item in value]
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and value.startswith("/") and Path(value).suffix.lower() in _ARTIFACT_SUFFIXES:
        path = Path(value)
        if path.is_symlink() or not path.is_file():
            raise IntegrityError("checkpoint input artifact is missing or a symlink")
        return {"file_sha256": digest(path), "suffix": path.suffix.lower()}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("unsupported checkpoint input; do not guess its identity")


def artifact_hashes(value: Any) -> dict[str, str]:
    result = {}
    def walk(item):
        if isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
        elif isinstance(item, str) and item.startswith("/"):
            path = Path(item)
            if path.suffix.lower() in _ARTIFACT_SUFFIXES:
                if path.is_symlink() or not path.is_file():
                    raise IntegrityError("checkpoint output artifact is missing or a symlink")
                result[str(path)] = digest(path)
    walk(value)
    return result


def valid_metric(raw: Any, layer: str) -> bool:
    if not isinstance(raw, dict):
        return False
    if layer == "l3":
        metrics = raw.get("metrics")
        names = {"scale_consistency", "style_consistency", "object_pairing_consistency",
                 "functional_consistency", "semantic_placement_consistency"}
        return (isinstance(metrics, dict) and names <= metrics.keys()
                and all(valid_metric(metrics[name], "l3_metric") for name in names))
    score = raw.get("score")
    return (raw.get("status") == ("checked" if layer == "l1" else "evaluated")
            and not isinstance(score, bool) and isinstance(score, (int, float))
            and math.isfinite(score) and 0 <= score <= 1)


def checkpoint_call(ctx: RoomContext, name: str, call_identity: Any,
                    call: Callable[[], Any], complete: Callable[[Any], bool]) -> Any:
    """Only complete exact-context metric steps; first result, never best-of-N.

L3 is deliberately one coupled step: Functional/Placement ownership is NOT
split or reconstructed. Incomplete L3 is observed but not resumed piecemeal.
"""
    ctx.ensure_execution()
    ctx.heartbeat.update(ctx.key, stage=name)
    path = None
    if ctx.policy.checkpoint_enabled:
        try:
            key = identity({"run": ctx.run_identity, "name": name, "input": _key_value(call_identity)})
            path = ctx.root / "execution_checkpoints" / name / f"{key}.json"
        except (TypeError, ValueError, OSError, IntegrityError):
            # Exotic configured providers remain runnable, without unsafe caching.
            pass
        if path is not None and path.exists():
            cached = read_json(path)
            if cached.get("key") != key or identity(cached["value"]) != cached.get("value_sha256"):
                raise IntegrityError("metric checkpoint integrity mismatch")
            for filename, sha in cached.get("artifact_sha256", {}).items():
                p = Path(filename)
                if p.is_symlink() or not p.is_file() or digest(p) != sha:
                    raise IntegrityError("metric checkpoint evidence integrity mismatch")
            if not complete(cached["value"]):
                raise IntegrityError("checkpoint contains an incomplete metric")
            best_effort_record(ctx.attempt / "execution_diagnostics" / f"{name}_reuse.json", {
                "checkpoint": str(path), "value_sha256": cached["value_sha256"],
                "original_attempt": cached["attempt"], "reused_at": stamp(),
                "new_api_calls": 0, "scientific_result_unchanged": True,
            })
            value = deepcopy(cached["value"])
            ctx.observe(name, value)
            return value
    try:
        result = call()
        ctx.ensure_execution()
    except Exception as exc:
        ctx.observe(name, exception_record(exc))
        raise
    ctx.observe(name, result)
    if path is not None and complete(result):
        try:
            # Preserve exact returned JSON, not a lossy/sanitized scoring payload.
            # If known secrets appear, skip caching entirely. Diagnostic files are
            # independently sanitized. These private artifacts stay in outputs.
            import json
            encoded = json.dumps(result, allow_nan=False)
            if redact(encoded) == encoded:
                atomic_json(path, {"schema_version": VERSION, "key": key,
                                   "value": result, "value_sha256": identity(result),
                                   "artifact_sha256": artifact_hashes(result),
                                   "attempt": str(ctx.attempt), "created_at": stamp()})
        except (OSError, TypeError, ValueError, IntegrityError):
            best_effort_record(ctx.attempt / "execution_diagnostics" / f"{name}_checkpoint_skipped.json",
                               {"reason": "checkpoint_write_or_identity_unavailable"})
    return result


def transient_origin(raw: Any) -> bool:
    """Conservative typed evidence, not camera/renderer category keywords."""
    if isinstance(raw, (list, tuple)):
        return bool(raw) and all(transient_origin(item) for item in raw)
    causes = []
    forbidden = False
    wrappers = {"BlenderRenderError", "EvidenceRenderFailure", "NonRectangularMaterializationInfrastructureError"}
    def walk(value, chained=False):
        nonlocal forbidden
        if isinstance(value, dict):
            stop = str(value.get("stop_reason") or "").lower()
            if any(word in stop for word in ("contract", "integrity", "constraint", "budget", "exhausted")):
                forbidden = True
            error_type = value.get("error_type") or value.get("validation_error_type")
            error = str(value.get("error") or value.get("validation_error") or "")
            if not error_type and value.get("adjudication_error"):
                error = str(value["adjudication_error"])
                match = re.match(r"([A-Za-z]+Error):", error)
                error_type = match[1] if match else "UnknownError"
            if error and not error_type:
                error_type = "UnknownError"
            if error_type:
                has_nested_cause = any(isinstance(value.get(k), (dict, list)) and value[k]
                                       for k in ("cause", "exception_chain", "root_cause", "causes"))
                if error_type not in wrappers or not (chained or has_nested_cause):
                    causes.append((str(error_type), error, value))
            for key, child in value.items():
                if key == "exception_chain" and isinstance(child, list):
                    for index, item in enumerate(child):
                        walk(item, chained=any(isinstance(later, dict) and later.get("error_type")
                                               for later in child[index + 1:]))
                    continue
                if key not in {"raw_response", "request_metadata", "raw_request"}:
                    walk(child)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
    walk(raw)
    if forbidden or not causes:
        return False
    for kind, error, record in causes:
        if kind in {"EndpointConnectionError", "TimeoutError", "ConnectionError", "BrokenPipeError",
                    "TimeoutExpired", "ConnectionResetError", "ConnectionAbortedError",
                    "ConnectionRefusedError", "RemoteDisconnected", "IncompleteRead"}:
            continue
        if kind == "OSError":
            import errno
            if record.get("errno") in {errno.EAGAIN, errno.EBUSY, errno.ETIMEDOUT,
                                       errno.ECONNRESET, errno.ECONNABORTED, errno.ECONNREFUSED}:
                continue
        if kind == "EndpointHTTPError":
            status = re.search(r"\bHTTP\s+(\d{3})\b", error)
            if status and int(status[1]) in {429, 500, 502, 503, 504}:
                continue
        # Renderer errors frequently wrap deterministic contract failures: do
        # not expand their retry eligibility from a string-only nested record.
        return False
    return True


def _failed_scope(raw: Any, metric: str) -> Any:
    if not isinstance(raw, dict):
        return raw
    if metric in {"collision", "oob", "support"}:
        records = raw.get("pairs" if metric == "collision" else "objects") or []
        failed = [r for r in records if isinstance(r, dict)
                  and r.get("requires_vlm") and r.get("final_verdict") not in {"valid", "invalid"}]
    else:
        failed = raw.get("infrastructure_failures") or raw
    scope = {"failures": failed, "stop_reason": raw.get("stop_reason"),
             "evidence_control": raw.get("evidence_control")}
    if isinstance(failed, list) and not transient_origin(failed):
        scope["error_type"] = "UnknownError"
    return scope


def enrich_renderer_origins(raw: Any, ctx: RoomContext) -> Any:
    """Match exact typed messages observed in THIS attempt before core wrapping.

    This derived classification input is never returned as the metric report.
    An unrelated renderer failure cannot confer retry eligibility on a sibling.
    """
    if isinstance(raw, list):
        return [enrich_renderer_origins(item, ctx) for item in raw]
    if not isinstance(raw, dict):
        return raw
    value = {k: enrich_renderer_origins(v, ctx) for k, v in raw.items()
             if k not in {"raw_response", "raw_request", "request_metadata"}}
    kind = raw.get("error_type") or raw.get("validation_error_type")
    message = raw.get("error") or raw.get("validation_error") or raw.get("adjudication_error")
    if isinstance(message, str):
        for name in ("BlenderRenderError", "EvidenceRenderFailure"):
            if message.startswith(name + ": "):
                kind, message = name, message[len(name) + 2:]
                break
        if kind:
            found = ctx.renderer_origins.get(identity({"type": kind, "message": message}))
            if found is not None:
                value["cause"] = found
    return value


def identity_failure_origin(raw: Any) -> bool:
    if isinstance(raw, (list, tuple)):
        return any(identity_failure_origin(item) for item in raw)
    if isinstance(raw, dict):
        return (raw.get("error_type") in {"IntegrityError", "BlenderSourceSceneModifiedError"}
                or any(identity_failure_origin(v) for k, v in raw.items()
                       if k not in {"raw_response", "raw_request", "request_metadata"}))
    return False


def observe_renderer_exceptions(renderer_class) -> None:
    """Observe original render calls; do not retry, rebuild evidence, or repair."""
    def wrap(function):
        @functools.wraps(function)
        def observed(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                ctx = CURRENT.get()
                if ctx is not None:
                    key = identity({"type": type(exc).__name__, "message": str(exc)})
                    record = exception_record(exc)
                    ctx.renderer_origins[key] = record
                    best_effort_record((ctx.attempt or ctx.root) / "execution_diagnostics" /
                                       f"renderer_origin_{key}.json", record)
                raise
        return observed
    for name, function in vars(renderer_class).items():
        if name.startswith("render_") and callable(function):
            setattr(renderer_class, name, wrap(function))


def install(resilient, runtime, evaluator_module, *, output: Path, dataset: str,
            policy: Policy, heartbeat: Heartbeat, code_identity: str,
            reuse_root: Path | None = None, circuit: Circuit | None = None) -> None:
    """Install once per isolated worker, after the existing scheduler/selection."""
    if getattr(resilient, "_execution_hardening_installed", False):
        raise RuntimeError("execution overlay must be installed once per worker")
    resilient._execution_hardening_installed = True
    base = resilient._Coordinator
    original_classify = resilient.classify_failure
    original_build = runtime.DefaultNonRectangularRuntimeFactory.build

    def classify(exc, *, stage):
        classified = original_classify(exc, stage=stage)
        ctx = CURRENT.get()
        if ctx is not None:
            origin = exception_record(exc)
            best_effort_record((ctx.attempt or ctx.root) / "execution_diagnostics" / "exception.json",
                               {"stage": stage, **exception_record(exc)})
            if isinstance(exc, evaluator_module.NonRectangularRoomMetricIncomplete):
                raw = enrich_renderer_origins(ctx.failures.get(str(exc.metric_id)), ctx)
                origin = {"exception": origin, "metric_origin": raw}
                if raw is not None and transient_origin(_failed_scope(raw, str(exc.metric_id))):
                    from dataclasses import replace
                    classified = replace(classified, retryable=True)
                elif raw is not None and results.hard_origin(raw):
                    from dataclasses import replace
                    classified = replace(classified, retryable=False, category="room_hard_failure")
            elif any(t.__name__ in {"BlenderRenderError", "EvidenceRenderFailure",
                                   "NonRectangularMaterializationInfrastructureError"}
                     for t in type(exc).__mro__):
                # Wrapper categories are not evidence of a recoverable cause.
                from dataclasses import replace
                classified = replace(classified, retryable=transient_origin(exception_record(exc)))
            elif not classified.retryable and transient_origin(exception_record(exc)):
                from dataclasses import replace
                classified = replace(classified, retryable=True)
            if (stage == "evaluation" and isinstance(exc, results.LOCAL_HARD_ERRORS)
                    and classified.category == "unclassified_failure"):
                from dataclasses import replace
                classified = replace(classified, retryable=False, category="room_hard_failure")
            if identity_failure_origin(origin):
                from dataclasses import replace
                classified = replace(classified, retryable=False, category="execution_identity_drift")
            try:
                ctx.ensure_execution()
                if classified.retryable and ctx.budget is not None:
                    ctx.budget.failed_attempt(identity({"stage": stage, "category": classified.category,
                                                       "type": type(exc).__name__,
                                                       "metric": getattr(exc, "metric_id", None)}))
            except (RetryBudgetExhausted, ChannelUnavailable) as guard:
                from dataclasses import replace
                classified = replace(classified, retryable=False,
                    category="channel_unavailable" if isinstance(guard, ChannelUnavailable) else "execution_budget_exhausted",
                    error_type=type(guard).__name__)
            best_effort_record((ctx.attempt or ctx.root) / "execution_diagnostics" / "retry_decision.json",
                               {"stage": stage, "policy": VERSION,
                                "decision": classified.public_dict(),
                                "original_category": original_classify(exc, stage=stage).category})
            best_effort_record((ctx.attempt or ctx.root) / "execution_diagnostics/failure_reasons.json", {
                "attempt_root": str(ctx.attempt) if ctx.attempt else None,
                "details": reasons.describe(origin, ctx, stage=stage, metric=getattr(exc, "metric_id", None))})
        return classified

    class Coordinator(base):
        def _write_scene_summary(self, bundle):
            super()._write_scene_summary(bundle)
            results.scene_result(self._scene_root(bundle))

        def _aggregate_scene(self, bundle):
            try:
                return super()._aggregate_scene(bundle)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError, AssertionError,
                    ZeroDivisionError, RecursionError) as exc:
                # Local aggregation bugs do not destroy verified room reports.
                # Integrity/source checks and infrastructure exceptions escape.
                atomic_json(self._scene_root(bundle) / "execution_aggregation_skip.json", {
                    "schema_version": results.RESULT_VERSION, "status": "skipped_hard_failure",
                    "stage": "scene_aggregation", **exception_record(exc)})

        def _run_room_once(self, bundle, unit, retry_round):
            if circuit is not None:
                circuit.ensure()  # Do not materialize fresh rooms on a halted lane.
            root = self._room_root(bundle, unit)
            key = f"{bundle.model}/{bundle.scene_id}/{unit.room_id}"
            check_storage(output, policy.minimum_free_gib)
            ctx = RoomContext(root, key, heartbeat, policy, identity({
                "source": bundle.source_identity(room_id=unit.room_id),
                "runtime": self.evaluator_factory.identity(), "code": code_identity,
                "policy": asdict(policy), "version": VERSION,
            }))
            ctx.budget = RoomBudget(root / "execution_retry_budget.json", ctx.run_identity, policy)
            ctx.circuit = circuit
            token = CURRENT.set(ctx)
            heartbeat.update(key, stage="waiting_room_slot", waiting_since=time.time())
            try:
                with room_slot(output / "control/room_slots", slots=4,
                               timeout=policy.room_slot_timeout_seconds,
                               work={"dataset": dataset, "room": key}):
                    heartbeat.update(key, stage="materialization", started_at=time.time())
                    check_storage(output, policy.minimum_free_gib)
                    return super()._run_room_once(bundle, unit, retry_round)
            except Exception as exc:
                best_effort_record(root / "execution_diagnostics" / "coordinator_failure.json",
                                   exception_record(exc))
                raise
            finally:
                heartbeat.remove(key)
                CURRENT.reset(token)
                ctx.budget.finish()

        def _room_selected_report(self, bundle, unit):
            selected = super()._room_selected_report(bundle, unit)
            if selected is not None or reuse_root is None:
                return selected
            source_eval = reuse_root / "datasets" / dataset / "lanes" / self.output_root.parent.name / "evaluation"
            source_room = source_eval / "models" / bundle.model / "scenes" / bundle.scene_id / "rooms" / unit.room_id
            if not (source_room / "room_report_selected.json").is_file():
                return None
            source_run = read_json(source_eval / "run_manifest.json")
            if identity(source_run["identity"]) != source_run["identity_sha256"]:
                raise IntegrityError("source campaign identity mismatch")
            if source_run["identity"]["runtime"] != self.evaluator_factory.identity():
                raise IntegrityError("successful room reuse requires identical runtime")
            source_scene = next((s for s in source_run["identity"]["scenes"]
                                 if s["model"] == bundle.model and s["scene_id"] == bundle.scene_id), None)
            if source_scene is None or {k: v["sha256"] for k, v in source_scene["artifacts"].items()} != bundle.file_sha256:
                raise IntegrityError("successful room reuse input mismatch")
            pointer = read_json(source_room / "room_report_selected.json")
            report_path = Path(pointer["room_report_path"])
            if not report_path.resolve().is_relative_to(source_room.resolve()) or report_path.is_symlink():
                raise IntegrityError("source room report path escapes its room")
            if digest(report_path) != pointer["room_report_sha256"]:
                raise IntegrityError("source successful report hash mismatch")
            report = resilient.validate_complete_room_report(read_json(report_path), unit=unit)
            from benchmark.non_rectangular.materialization import verify_completed_nonrect_materialization
            materialization = read_json(source_room / "materialization_selected.json")
            verified = verify_completed_nonrect_materialization(
                Path(materialization["materialization_root"]),
                expected_identity_sha256=pointer["materialization_identity_sha256"])
            if identity(read_json(verified.canonical_scene_path)) != resilient.sha256_json(
                    resilient.project_room_unit_to_canonical_scene(unit)):
                raise IntegrityError("source successful room canonical scene mismatch")
            root = self._room_root(bundle, unit)
            atomic_json(root / "room_report_selected.json", {
                **pointer, "attempt": 0, "reused_read_only": True,
                "source_selection_sha256": digest(source_room / "room_report_selected.json"),
                "source_run_identity_sha256": source_run["identity_sha256"],
                "execution_hardening": VERSION, "original_revision_preserved": True,
            })
            return report

    def build(factory, context):
        ctx = CURRENT.get()
        if ctx is None:
            return original_build(factory, context)
        ctx.attempt = context.attempt_root
        ctx.materialization_identity = context.materialization.identity_sha256
        ctx.run_identity = identity({"room": ctx.run_identity,
                                     "materialization": context.materialization.identity_sha256})
        ctx.failures.clear()
        ctx.heartbeat.update(ctx.key, stage="runtime_and_global_evidence")
        try:
            ctx.ensure_execution()
            instance = original_build(factory, context)
        except Exception as exc:
            ctx.observe("runtime_build", exception_record(exc))
            raise
        original_diagnostic = instance._persist_metric_diagnostic
        def observe_metric(*, unit, metric, layer, raw):
            ctx.observe(metric, raw)
            return original_diagnostic(unit=unit, metric=metric, layer=layer, raw=raw)
        instance._persist_metric_diagnostic = observe_metric
        quality = instance.scene_quality_evaluator
        def scene_quality(*args, **kwargs):
            key = {"args": args, "kwargs": {k: v for k, v in kwargs.items()
                   if k not in {"vlm_judge", "camera_evidence_provider"}}}
            raw = checkpoint_call(ctx, "scene_quality", key,
                                  lambda: quality(*args, **kwargs), lambda r: valid_metric(r, "l3"))
            for name, report in (raw.get("metrics") or {}).items() if isinstance(raw, dict) else []:
                ctx.observe(name, report)
            return raw
        instance.scene_quality_evaluator = scene_quality
        original_evaluate = instance.evaluate
        def evaluate(*args, **kwargs):
            ctx.ensure_execution()
            try:
                result = original_evaluate(*args, **kwargs)
                ctx.ensure_execution()
                return result
            finally:
                try:
                    results.capture_metrics(ctx, context.unit, instance, evaluator_module,
                                            original_classify, enrich_renderer_origins)
                except Exception as exc:
                    best_effort_record(ctx.attempt / "execution_diagnostics/result_capture_failure.json",
                                       exception_record(exc))
        instance.evaluate = evaluate
        return instance

    def wrap_metric(function, name):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            ctx = CURRENT.get()
            if ctx is None:
                return function(*args, **kwargs)
            key = {"args": args, "kwargs": {k: v for k, v in kwargs.items()
                   if k not in {"vlm_judge", "local_view_provider"}}}
            return checkpoint_call(ctx, name, key, lambda: function(*args, **kwargs),
                                   lambda raw: valid_metric(raw, "l1"))
        return wrapped

    for name, attr in (("collision", "check_collision"), ("oob", "check_polygon_oob"), ("support", "check_support")):
        setattr(evaluator_module, attr, wrap_metric(getattr(evaluator_module, attr), name))
    runtime.DefaultNonRectangularRuntimeFactory.build = build
    if hasattr(runtime, "BlenderRenderer"):
        observe_renderer_exceptions(runtime.BlenderRenderer)
        reasons.install(CURRENT)
    resilient.classify_failure = classify
    resilient._Coordinator = Coordinator
    resilient.COORDINATOR_REVISION += "+" + VERSION
