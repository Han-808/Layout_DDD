"""Provider-isolated runner for the selected non-rectangular scene cohort."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Iterable, Mapping

from benchmark.scene_generation.campaign.execution import repository_root
from benchmark.scene_generation.non_rectangular_multi_room.campaign import (
    ActivatedNonRectangularCampaign,
    PreparedNonRectangularCampaign,
    activate_non_rectangular_campaign,
    prepare_non_rectangular_campaign,
    rebind_prepared_non_rectangular_campaign_inputs,
    run_activated_non_rectangular_campaign,
)


PROFILE_SCHEMA_VERSION = "non_rectangular_fullrun_profile_v1"
RUN_MANIFEST_SCHEMA_VERSION = "non_rectangular_fullrun_manifest_v1"
SUMMARY_SCHEMA_VERSION = "non_rectangular_fullrun_summary_v1"
_RETRYABLE_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_RETRYABLE_PREFLIGHT_CATEGORIES = frozenset(
    {"transport_or_http", "response_contract", "preflight_assertion"}
)
_SAFE_PREFLIGHT_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "preflight_contract_id",
        "transport_status",
        "http_status",
        "response_contract_valid",
        "model_identity_matches",
        "content_nonempty",
        "reasoning_signal",
        "reasoning_tokens",
        "ok",
        "failure_category",
        "transport_stage",
        "error_type",
    }
)


class NonRectangularFullrunError(RuntimeError):
    """Raised when a cohort/profile/output identity is unsafe."""


@dataclass(frozen=True, slots=True)
class FullrunModel:
    model_key: str
    campaign_id: str
    model_profile_id: str
    output_name: str


@dataclass(frozen=True, slots=True)
class FullrunScene:
    scene_id: str
    room_count: int
    room_layout_path: Path
    room_program_path: Path
    room_layout_sha256: str
    room_program_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedFullrun:
    repo_root: Path
    profile_path: Path
    profile_sha256: str
    fullrun_id: str
    provider_family: str
    contract_version: str
    layout_manifest_path: Path
    program_manifest_path: Path
    layout_manifest_sha256: str
    program_manifest_sha256: str
    models: tuple[FullrunModel, ...]
    scenes: tuple[FullrunScene, ...]
    campaigns: Mapping[tuple[str, str], PreparedNonRectangularCampaign]
    stage_a_timeout_seconds: float
    stage_c_timeout_seconds: float
    preflight_max_total_attempts: int
    preflight_retry_delay_seconds: float
    model_cooldown_seconds: float
    case_failure_delay_seconds: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "prepared_non_rectangular_fullrun_v1",
            "valid": True,
            "fullrun_id": self.fullrun_id,
            "provider_family": self.provider_family,
            "contract_version": self.contract_version,
            "profile_sha256": self.profile_sha256,
            "layout_manifest_sha256": self.layout_manifest_sha256,
            "program_manifest_sha256": self.program_manifest_sha256,
            "model_order": [model.model_key for model in self.models],
            "scene_order": [scene.scene_id for scene in self.scenes],
            "model_count": len(self.models),
            "scene_count": len(self.scenes),
            "case_count": len(self.models) * len(self.scenes),
            "room_count_per_model": sum(scene.room_count for scene in self.scenes),
            "stage_timeouts_seconds": {
                "stage_a": self.stage_a_timeout_seconds,
                "stage_c": self.stage_c_timeout_seconds,
            },
            "retry_policy": {
                "max_infrastructure_retries": 5,
                "retry_ambiguous_timeouts": False,
            },
            "credential_loaded": False,
            "network_used": False,
        }


def prepare_fullrun(profile_path: str | Path) -> PreparedFullrun:
    """Validate one provider profile and all 30 scene/model pairs offline."""

    root = repository_root().resolve()
    profile = _repo_file(root, profile_path, label="fullrun profile")
    raw = _load_json(profile)
    _exact(
        raw,
        {
            "schema_version",
            "fullrun_id",
            "provider_family",
            "contract_version",
            "layout_manifest",
            "program_manifest",
            "scene_order",
            "models",
            "stage_timeouts_seconds",
            "preflight_policy",
            "model_cooldown_seconds",
            "case_failure_delay_seconds",
        },
        label="fullrun profile",
    )
    if raw["schema_version"] != PROFILE_SCHEMA_VERSION:
        raise NonRectangularFullrunError("unsupported fullrun profile schema")
    fullrun_id = _text(raw["fullrun_id"], label="fullrun_id")
    provider = _text(raw["provider_family"], label="provider_family")
    if provider not in {"api2", "api3"}:
        raise NonRectangularFullrunError("provider_family must be api2 or api3")
    if raw["contract_version"] != "v2":
        raise NonRectangularFullrunError("fullrun must select contract v2")
    layout_manifest_path = _repo_file(
        root, raw["layout_manifest"], label="layout manifest"
    )
    program_manifest_path = _repo_file(
        root, raw["program_manifest"], label="program manifest"
    )
    layout_manifest = _load_json(layout_manifest_path)
    program_manifest = _load_json(program_manifest_path)
    _verify_self_hash(layout_manifest)
    _verify_self_hash(program_manifest)
    scene_order = _text_list(raw["scene_order"], label="scene_order")
    if len(scene_order) != 10 or len(set(scene_order)) != 10:
        raise NonRectangularFullrunError("scene_order must contain 10 unique scenes")
    if layout_manifest.get("selection", {}).get("scene_order") != scene_order:
        raise NonRectangularFullrunError("layout manifest scene order mismatch")
    layout_rows = _scene_rows(layout_manifest, label="layout manifest")
    program_rows = _scene_rows(program_manifest, label="program manifest")
    if list(layout_rows) != scene_order or list(program_rows) != scene_order:
        raise NonRectangularFullrunError("cohort manifests do not match scene_order")
    if layout_manifest.get("totals") != {
        "room_count": 50,
        "scene_count": 10,
        "wall_segment_count": 283,
    }:
        raise NonRectangularFullrunError("selected layout cohort totals drifted")

    scenes: list[FullrunScene] = []
    for scene_id in scene_order:
        layout_row = layout_rows[scene_id]
        program_row = program_rows[scene_id]
        layout_path = _repo_file(
            root,
            layout_manifest_path.parent / scene_id / "room_layout.json",
            label=f"{scene_id} room layout",
        )
        program_path = _repo_file(
            root,
            program_manifest_path.parent / scene_id / "room_program.json",
            label=f"{scene_id} room program",
        )
        layout_sha = _sha256_file(layout_path)
        program_sha = _sha256_file(program_path)
        if layout_sha != layout_row.get("room_layout_sha256"):
            raise NonRectangularFullrunError(f"{scene_id} layout hash drift")
        if program_sha != program_row.get("room_program_sha256"):
            raise NonRectangularFullrunError(f"{scene_id} program hash drift")
        room_count = layout_row.get("room_count")
        if isinstance(room_count, bool) or not isinstance(room_count, int):
            raise NonRectangularFullrunError(f"{scene_id} room count is invalid")
        scenes.append(
            FullrunScene(
                scene_id=scene_id,
                room_count=room_count,
                room_layout_path=layout_path,
                room_program_path=program_path,
                room_layout_sha256=layout_sha,
                room_program_sha256=program_sha,
            )
        )

    model_values = raw["models"]
    if not isinstance(model_values, list) or len(model_values) != 3:
        raise NonRectangularFullrunError("provider fullrun must contain 3 models")
    models: list[FullrunModel] = []
    for index, value in enumerate(model_values):
        if not isinstance(value, dict):
            raise NonRectangularFullrunError(f"models[{index}] must be an object")
        _exact(
            value,
            {"model_key", "campaign_id", "model_profile_id", "output_name"},
            label=f"models[{index}]",
        )
        model = FullrunModel(
            model_key=_text(value["model_key"], label="model_key"),
            campaign_id=_text(value["campaign_id"], label="campaign_id"),
            model_profile_id=_text(
                value["model_profile_id"], label="model_profile_id"
            ),
            output_name=_portable_name(value["output_name"], label="output_name"),
        )
        models.append(model)
    if len({model.model_key for model in models}) != 3 or len(
        {model.output_name for model in models}
    ) != 3:
        raise NonRectangularFullrunError("model identities must be unique")

    timeouts = raw["stage_timeouts_seconds"]
    if not isinstance(timeouts, dict):
        raise NonRectangularFullrunError("stage_timeouts_seconds must be an object")
    _exact(timeouts, {"stage_a", "stage_c"}, label="stage_timeouts_seconds")
    stage_a_timeout = _positive_number(timeouts["stage_a"], label="stage_a")
    stage_c_timeout = _positive_number(timeouts["stage_c"], label="stage_c")
    preflight_policy = raw["preflight_policy"]
    if not isinstance(preflight_policy, dict):
        raise NonRectangularFullrunError("preflight_policy must be an object")
    _exact(
        preflight_policy,
        {"max_total_attempts", "retry_delay_seconds"},
        label="preflight_policy",
    )
    attempts = preflight_policy["max_total_attempts"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        raise NonRectangularFullrunError("max_total_attempts must be positive")
    preflight_delay = _nonnegative_number(
        preflight_policy["retry_delay_seconds"], label="preflight retry delay"
    )
    model_cooldown = _nonnegative_number(
        raw["model_cooldown_seconds"], label="model cooldown"
    )
    failure_delay = _nonnegative_number(
        raw["case_failure_delay_seconds"], label="case failure delay"
    )

    campaigns: dict[tuple[str, str], PreparedNonRectangularCampaign] = {}
    for model in models:
        template: PreparedNonRectangularCampaign | None = None
        for scene_index, scene in enumerate(scenes):
            if scene_index == 0:
                prepared = prepare_non_rectangular_campaign(
                    model.campaign_id,
                    room_layout_path=scene.room_layout_path,
                    room_program_path=scene.room_program_path,
                    contract_version="v2",
                )
                template = prepared
            else:
                assert template is not None
                prepared = rebind_prepared_non_rectangular_campaign_inputs(
                    template,
                    room_layout_path=scene.room_layout_path,
                    room_program_path=scene.room_program_path,
                )
            if prepared.model.model_profile_id != model.model_profile_id:
                raise NonRectangularFullrunError("model profile identity mismatch")
            if not prepared.route.route_profile_id.startswith(f"{provider}-"):
                raise NonRectangularFullrunError("provider route isolation mismatch")
            options = prepared.model.request_options
            if options.request_timeout_seconds != stage_a_timeout:
                raise NonRectangularFullrunError("stage timeout profile mismatch")
            if prepared.execution.max_infrastructure_retries != 5 or (
                prepared.model.transport_policy.max_infrastructure_retries != 5
            ):
                raise NonRectangularFullrunError("retry5 profile mismatch")
            if prepared.execution.retry_ambiguous_timeouts:
                raise NonRectangularFullrunError("ambiguous timeout retry is forbidden")
            campaigns[(model.model_key, scene.scene_id)] = prepared

    return PreparedFullrun(
        repo_root=root,
        profile_path=profile,
        profile_sha256=_sha256_file(profile),
        fullrun_id=fullrun_id,
        provider_family=provider,
        contract_version="v2",
        layout_manifest_path=layout_manifest_path,
        program_manifest_path=program_manifest_path,
        layout_manifest_sha256=_sha256_file(layout_manifest_path),
        program_manifest_sha256=_sha256_file(program_manifest_path),
        models=tuple(models),
        scenes=tuple(scenes),
        campaigns=campaigns,
        stage_a_timeout_seconds=stage_a_timeout,
        stage_c_timeout_seconds=stage_c_timeout,
        preflight_max_total_attempts=attempts,
        preflight_retry_delay_seconds=preflight_delay,
        model_cooldown_seconds=model_cooldown,
        case_failure_delay_seconds=failure_delay,
    )


def execute_fullrun(
    prepared: PreparedFullrun,
    *,
    command: str,
    output_base: str | Path | None = None,
    generation_bindings_path: str | Path | None = None,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    fresh: bool = False,
    activate: Callable[..., tuple[dict[str, Any], ActivatedNonRectangularCampaign | None]] = activate_non_rectangular_campaign,
    run_case: Callable[..., dict[str, Any]] = run_activated_non_rectangular_campaign,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Check, preflight, or sequentially run one provider-isolated profile."""

    if command == "check":
        return prepared.public_dict()
    if command not in {"preflight", "run"}:
        raise NonRectangularFullrunError("unsupported fullrun command")
    if command == "preflight" and fresh:
        raise NonRectangularFullrunError("fresh is only valid for run")

    state: _CoordinatorState | None = None
    lock: _OutputLock | None = None
    if command == "run":
        if output_base is None:
            raise NonRectangularFullrunError("run requires output_base")
        destination = Path(output_base).expanduser().absolute()
        lock = _OutputLock(destination, fresh=fresh)
        lock.acquire()
        try:
            _validate_output_topology(prepared, destination, fresh=fresh)
            state = _CoordinatorState(prepared, destination)
            state.initialize_or_verify()
        except Exception:
            lock.release()
            raise

    model_results: list[dict[str, Any]] = []
    try:
        for model_index, model in enumerate(prepared.models):
            first = prepared.campaigns[(model.model_key, prepared.scenes[0].scene_id)]
            report, activated, attempts = _activate_with_retry(
                first,
                prepared=prepared,
                generation_bindings_path=generation_bindings_path,
                resource_bindings_path=resource_bindings_path,
                environ=environ,
                activate=activate,
                sleeper=sleeper,
            )
            safe_report = _sanitize_preflight(report)
            model_result: dict[str, Any] = {
                "model": model.model_key,
                "campaign_id": model.campaign_id,
                "preflight_attempts": attempts,
                "preflight_ok": activated is not None,
                "complete_cases": 0,
                "failed_cases": 0,
                "case_results": [],
            }
            if state is not None:
                state.write_preflight(model.output_name, safe_report)
                state.event(
                    "preflight_terminal",
                    model=model.model_key,
                    status="passed" if activated is not None else "failed",
                    attempts=attempts,
                    failure_category=safe_report.get("failure_category"),
                )
            if activated is None:
                model_result["failed_cases"] = len(prepared.scenes) if command == "run" else 0
                model_results.append(model_result)
                if state is not None:
                    state.write_summary(_summary(prepared, model_results, terminal=False))
            elif command == "preflight":
                model_results.append(model_result)
            else:
                assert state is not None
                for scene in prepared.scenes:
                    case = prepared.campaigns[(model.model_key, scene.scene_id)]
                    case_output = state.output_base / model.output_name / scene.scene_id
                    resume = case_output.exists()
                    state.event(
                        "case_starting",
                        model=model.model_key,
                        scene=scene.scene_id,
                        status="resume" if resume else "fresh",
                    )
                    try:
                        result = run_case(
                            case,
                            activated,
                            output_root=case_output,
                            resume=resume,
                            progress=lambda value, m=model, s=scene: state.event(
                                "case_progress",
                                model=m.model_key,
                                scene=s.scene_id,
                                status=str(value.get("status") or value.get("event") or "unknown"),
                            ),
                            configuration_identity_extra={
                                "fullrun_profile_sha256": prepared.profile_sha256,
                                "layout_manifest_sha256": prepared.layout_manifest_sha256,
                                "program_manifest_sha256": prepared.program_manifest_sha256,
                            },
                        )
                        status = str(result.get("status", "unknown"))
                        complete = status == "complete"
                        model_result["complete_cases" if complete else "failed_cases"] += 1
                        model_result["case_results"].append(
                            {"scene_id": scene.scene_id, "status": status}
                        )
                        state.event(
                            "case_terminal",
                            model=model.model_key,
                            scene=scene.scene_id,
                            status=status,
                        )
                        if not complete and prepared.case_failure_delay_seconds > 0:
                            sleeper(prepared.case_failure_delay_seconds)
                    except Exception as exc:
                        model_result["failed_cases"] += 1
                        model_result["case_results"].append(
                            {
                                "scene_id": scene.scene_id,
                                "status": "runner_error",
                                "error_type": type(exc).__name__,
                            }
                        )
                        state.event(
                            "case_terminal",
                            model=model.model_key,
                            scene=scene.scene_id,
                            status="runner_error",
                            error_type=type(exc).__name__,
                        )
                        if prepared.case_failure_delay_seconds > 0:
                            sleeper(prepared.case_failure_delay_seconds)
                    state.write_summary(_summary(prepared, model_results + [model_result], terminal=False))
                model_results.append(model_result)
            if model_index + 1 < len(prepared.models) and prepared.model_cooldown_seconds > 0:
                sleeper(prepared.model_cooldown_seconds)

        if command == "preflight":
            passed = sum(bool(item["preflight_ok"]) for item in model_results)
            result = {
                "schema_version": "non_rectangular_fullrun_preflight_summary_v1",
                "fullrun_id": prepared.fullrun_id,
                "provider_family": prepared.provider_family,
                "status": "complete" if passed == len(prepared.models) else "partial",
                "model_count": len(prepared.models),
                "passed_models": passed,
                "failed_models": len(prepared.models) - passed,
                "models": model_results,
            }
        else:
            result = _summary(prepared, model_results, terminal=True)
        if state is not None:
            state.write_summary(result)
            state.event("runner_terminal", status=result["status"])
        return result
    finally:
        if lock is not None:
            lock.release()


def _activate_with_retry(
    campaign: PreparedNonRectangularCampaign,
    *,
    prepared: PreparedFullrun,
    generation_bindings_path: str | Path | None,
    resource_bindings_path: str | Path | None,
    environ: Mapping[str, str] | None,
    activate: Callable[..., tuple[dict[str, Any], ActivatedNonRectangularCampaign | None]],
    sleeper: Callable[[float], None],
) -> tuple[dict[str, Any], ActivatedNonRectangularCampaign | None, int]:
    last: dict[str, Any] = {"ok": False, "failure_category": "not_attempted"}
    for attempt in range(1, prepared.preflight_max_total_attempts + 1):
        try:
            last, activated = activate(
                campaign,
                generation_bindings_path=generation_bindings_path,
                resource_bindings_path=resource_bindings_path,
                environ=environ,
                stage_c_timeout_seconds=prepared.stage_c_timeout_seconds,
            )
        except Exception as exc:
            return {
                "ok": False,
                "failure_category": "preflight_runner_error",
                "error_type": type(exc).__name__,
            }, None, attempt
        if activated is not None and last.get("ok") is True:
            return last, activated, attempt
        if not _preflight_retryable(last):
            return last, None, attempt
        if attempt < prepared.preflight_max_total_attempts:
            sleeper(prepared.preflight_retry_delay_seconds)
    return last, None, prepared.preflight_max_total_attempts


def _preflight_retryable(report: Mapping[str, Any]) -> bool:
    category = report.get("failure_category")
    if category not in _RETRYABLE_PREFLIGHT_CATEGORIES:
        return False
    status = report.get("http_status")
    if isinstance(status, int) and not isinstance(status, bool):
        if not 200 <= status < 300 and status not in _RETRYABLE_HTTP:
            return False
    return True


def _summary(
    prepared: PreparedFullrun,
    model_results: list[dict[str, Any]],
    *,
    terminal: bool,
) -> dict[str, Any]:
    complete = sum(int(item["complete_cases"]) for item in model_results)
    failed = sum(int(item["failed_cases"]) for item in model_results)
    expected = len(prepared.models) * len(prepared.scenes)
    if terminal:
        status = "complete" if complete == expected and failed == 0 else "partial"
    else:
        status = "running"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "fullrun_id": prepared.fullrun_id,
        "provider_family": prepared.provider_family,
        "status": status,
        "terminal": terminal,
        "expected_cases": expected,
        "complete_cases": complete,
        "failed_cases": failed,
        "remaining_cases": max(0, expected - complete - failed),
        "models": model_results,
    }


class _OutputLock:
    def __init__(self, output_base: Path, *, fresh: bool) -> None:
        self.output_base = output_base
        self.fresh = fresh
        self.handle: Any = None

    def acquire(self) -> None:
        if self.output_base.is_symlink():
            raise NonRectangularFullrunError("output base must not be a symlink")
        if self.fresh and self.output_base.exists():
            raise FileExistsError(f"fresh output already exists: {self.output_base}")
        self.output_base.mkdir(parents=True, exist_ok=not self.fresh)
        state = self.output_base / "_runner_state"
        state.mkdir(exist_ok=True)
        lock_path = state / "runner.lock"
        if lock_path.is_symlink():
            raise NonRectangularFullrunError("runner lock must not be a symlink")
        self.handle = lock_path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise NonRectangularFullrunError("another writer holds the runner lock") from exc

    def release(self) -> None:
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


class _CoordinatorState:
    def __init__(self, prepared: PreparedFullrun, output_base: Path) -> None:
        self.prepared = prepared
        self.output_base = output_base
        self.root = output_base / "_runner_state"
        self.manifest = self.root / "run_manifest.json"
        self.events = self.root / "events.jsonl"
        self.summary = self.root / "summary.json"
        self.preflight_root = self.root / "preflight"

    def initialize_or_verify(self) -> None:
        expected = _coordinator_manifest(self.prepared)
        if self.manifest.exists():
            if self.manifest.is_symlink() or _load_json(self.manifest) != expected:
                raise NonRectangularFullrunError("runner manifest identity mismatch")
        else:
            _write_json_exclusive(self.manifest, expected)
        self.preflight_root.mkdir(exist_ok=True)
        if self.preflight_root.is_symlink():
            raise NonRectangularFullrunError("preflight state must not be a symlink")

    def event(self, event: str, **values: Any) -> None:
        if self.events.is_symlink():
            raise NonRectangularFullrunError("event journal must not be a symlink")
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "provider_family": self.prepared.provider_family,
            **{key: value for key, value in values.items() if value is not None},
        }
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_preflight(self, output_name: str, report: Mapping[str, Any]) -> None:
        _write_json_atomic(self.preflight_root / f"{output_name}.json", dict(report))

    def write_summary(self, value: Mapping[str, Any]) -> None:
        _write_json_atomic(self.summary, dict(value))


def _coordinator_manifest(prepared: PreparedFullrun) -> dict[str, Any]:
    return {
        "schema_version": RUN_MANIFEST_SCHEMA_VERSION,
        "fullrun_id": prepared.fullrun_id,
        "provider_family": prepared.provider_family,
        "contract_version": prepared.contract_version,
        "profile_sha256": prepared.profile_sha256,
        "layout_manifest_sha256": prepared.layout_manifest_sha256,
        "program_manifest_sha256": prepared.program_manifest_sha256,
        "model_order": [
            {
                "model_key": model.model_key,
                "campaign_id": model.campaign_id,
                "model_profile_id": model.model_profile_id,
                "output_name": model.output_name,
            }
            for model in prepared.models
        ],
        "scene_order": [scene.scene_id for scene in prepared.scenes],
        "stage_timeouts_seconds": {
            "stage_a": prepared.stage_a_timeout_seconds,
            "stage_c": prepared.stage_c_timeout_seconds,
        },
        "retry_policy": {
            "max_infrastructure_retries": 5,
            "retry_ambiguous_timeouts": False,
        },
    }


def _validate_output_topology(
    prepared: PreparedFullrun, output_base: Path, *, fresh: bool
) -> None:
    if output_base.is_symlink() or not output_base.is_dir():
        raise NonRectangularFullrunError("output base must be a real directory")
    expected_models = {model.output_name for model in prepared.models}
    allowed_top = expected_models | {"_runner_state"}
    extras = {path.name for path in output_base.iterdir()} - allowed_top
    if extras:
        raise NonRectangularFullrunError("output base contains unknown entries")
    expected_scenes = {scene.scene_id for scene in prepared.scenes}
    for model in prepared.models:
        root = output_base / model.output_name
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise NonRectangularFullrunError("model output must be a real directory")
        extras = {path.name for path in root.iterdir()} - expected_scenes
        if extras:
            raise NonRectangularFullrunError("model output contains unknown scenes")
        for path in root.iterdir():
            if path.is_symlink() or not path.is_dir():
                raise NonRectangularFullrunError("scene output must be a real directory")
    if fresh and any((output_base / name).exists() for name in expected_models):
        raise NonRectangularFullrunError("fresh run found existing model output")


def _sanitize_preflight(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: report[key] for key in sorted(_SAFE_PREFLIGHT_FIELDS) if key in report}


def _scene_rows(manifest: Mapping[str, Any], *, label: str) -> dict[str, dict[str, Any]]:
    values = manifest.get("scenes")
    if not isinstance(values, list):
        raise NonRectangularFullrunError(f"{label} scenes must be an array")
    output: dict[str, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict):
            raise NonRectangularFullrunError(f"{label} scene must be an object")
        scene_id = _text(item.get("scene_id"), label="scene_id")
        if scene_id in output:
            raise NonRectangularFullrunError(f"{label} contains duplicate scene")
        output[scene_id] = item
    return output


def _verify_self_hash(value: Mapping[str, Any]) -> None:
    observed = value.get("manifest_sha256")
    if not isinstance(observed, str):
        raise NonRectangularFullrunError("cohort manifest lacks self hash")
    payload = dict(value)
    payload.pop("manifest_sha256", None)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != observed:
        raise NonRectangularFullrunError("cohort manifest self hash drift")


def _repo_file(root: Path, value: Any, *, label: str) -> Path:
    if isinstance(value, Path):
        candidate = value.expanduser().absolute()
    else:
        text = _text(value, label=label)
        relative = Path(text)
        if relative.is_absolute() or ".." in relative.parts:
            raise NonRectangularFullrunError(f"{label} must be repository-relative")
        candidate = (root / relative).absolute()
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise NonRectangularFullrunError(f"{label} escapes repository root") from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise NonRectangularFullrunError(f"{label} must be a regular file")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"invalid JSON number: {item}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise NonRectangularFullrunError(
            f"cannot load JSON artifact: {type(exc).__name__}"
        ) from exc
    if not isinstance(value, dict):
        raise NonRectangularFullrunError("JSON artifact must be an object")
    return value


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key")
        output[key] = value
    return output


def _exact(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise NonRectangularFullrunError(f"{label} keys are not exact")


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise NonRectangularFullrunError(f"{label} must be non-empty text")
    return value


def _portable_name(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise NonRectangularFullrunError(f"{label} must be one path component")
    return text


def _text_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise NonRectangularFullrunError(f"{label} must be a non-empty array")
    return [_text(item, label=label) for item in value]


def _positive_number(value: Any, *, label: str) -> float:
    result = _nonnegative_number(value, label=label)
    if result <= 0:
        raise NonRectangularFullrunError(f"{label} must be positive")
    return result


def _nonnegative_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NonRectangularFullrunError(f"{label} must be numeric")
    result = float(value)
    if result < 0 or result != result or result == float("inf"):
        raise NonRectangularFullrunError(f"{label} must be finite and non-negative")
    return result


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise NonRectangularFullrunError("state artifact must not be a symlink")
    data = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "preflight", "run"):
        command = commands.add_parser(name)
        command.add_argument("--profile", type=Path, required=True)
        if name in {"preflight", "run"}:
            command.add_argument("--generation-bindings", type=Path)
            command.add_argument("--resource-bindings", type=Path)
        if name == "run":
            command.add_argument("--output-base", type=Path, required=True)
            command.add_argument("--fresh", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        prepared = prepare_fullrun(args.profile)
        result = execute_fullrun(
            prepared,
            command=args.command,
            output_base=getattr(args, "output_base", None),
            generation_bindings_path=getattr(args, "generation_bindings", None),
            resource_bindings_path=getattr(args, "resource_bindings", None),
            fresh=bool(getattr(args, "fresh", False)),
        )
        print(json.dumps(result, sort_keys=True, ensure_ascii=False), flush=True)
        return 0 if result.get("status", "complete") == "complete" or result.get("valid") else 2
    except Exception as exc:
        print(
            f"error: {type(exc).__name__}: non-rectangular fullrun command failed",
            file=sys.stderr,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FullrunModel",
    "FullrunScene",
    "NonRectangularFullrunError",
    "PreparedFullrun",
    "build_parser",
    "execute_fullrun",
    "main",
    "prepare_fullrun",
]
