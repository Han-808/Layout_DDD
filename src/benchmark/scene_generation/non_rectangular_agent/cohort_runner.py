"""Sequential, provider-neutral runner for the complicated Agent track."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from benchmark.scene_generation.campaign.execution import repository_root

from .asset_db import (
    open_shared_asset_database,
    shared_database_static_contract,
)
from .profiles import AgentTrackProfile, load_agent_track_profile
from .provenance import agent_source_manifest
from .runtime import run_agent_episode
from .suite import AgentFloorPlanSuite, load_agent_floorplan_suite


FULLRUN_MANIFEST_SCHEMA_VERSION = "non_rectangular_agent_fullrun_manifest_v1"
FULLRUN_SUMMARY_SCHEMA_VERSION = "non_rectangular_agent_fullrun_summary_v1"


class AgentFullrunError(RuntimeError):
    """Raised when a profile, output tree, or writer identity is unsafe."""


@dataclass(frozen=True, slots=True)
class PreparedAgentFullrun:
    repo_root: Path
    profile: AgentTrackProfile
    profile_sha256: str
    suite: AgentFloorPlanSuite
    shared_database_contract: Mapping[str, Any]
    shared_database_contract_sha256: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "prepared_non_rectangular_agent_fullrun_v1",
            "valid": True,
            "fullrun_id": self.profile.fullrun_id,
            "track_id": self.profile.track_id,
            "participant_class": self.profile.participant_class,
            "comparison_unit": "general_purpose_tool_using_agent_system",
            "profile_sha256": self.profile_sha256,
            "agent_order": [agent.agent_id for agent in self.profile.agents],
            "agent_count": len(self.profile.agents),
            "scene_count": len(self.suite.cases),
            "room_count_per_agent": sum(
                case.room_count for case in self.suite.cases
            ),
            "case_count": len(self.profile.agents) * len(self.suite.cases),
            "suite": self.suite.public_dict(),
            "shared_asset_database": dict(self.shared_database_contract),
            "shared_database_contract_sha256": self.shared_database_contract_sha256,
            "tool_policy": self.profile.tool_policy.public_dict(),
            "credential_loaded": False,
            "network_used": False,
        }


def prepare_agent_fullrun(profile_path: str | Path) -> PreparedAgentFullrun:
    """Resolve suite/profile/DB identities without resources, credentials, or network."""

    root = repository_root().resolve()
    profile = load_agent_track_profile(profile_path, repo_root=root)
    suite = load_agent_floorplan_suite(profile.suite_root)
    if profile.track_id != suite.public_dict()["track_id"]:
        raise AgentFullrunError("profile/suite track identity mismatch")
    declared = _load_json(profile.shared_database_contract)
    expected = shared_database_static_contract(
        catalog_path=profile.retrieval_catalog,
        retrieval_profile_id=profile.retrieval_profile_id,
        max_top_k=profile.max_top_k,
    )
    if declared != expected:
        raise AgentFullrunError("shared database contract differs from retrieval catalog")
    return PreparedAgentFullrun(
        repo_root=root,
        profile=profile,
        profile_sha256=_sha256_file(profile.path),
        suite=suite,
        shared_database_contract=expected,
        shared_database_contract_sha256=_sha256_file(
            profile.shared_database_contract
        ),
    )


def execute_agent_fullrun(
    prepared: PreparedAgentFullrun,
    *,
    command: str,
    output_base: str | Path | None = None,
    resource_bindings_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    fresh: bool = False,
    database_factory: Callable[..., Any] = open_shared_asset_database,
    episode_runner: Callable[..., dict[str, Any]] = run_agent_episode,
) -> dict[str, Any]:
    if command == "check":
        return prepared.public_dict()
    if command not in {"resource-gate", "run"}:
        raise AgentFullrunError("unsupported Agent fullrun command")
    if command == "resource-gate" and fresh:
        raise AgentFullrunError("fresh is only valid for run")
    if command == "resource-gate":
        database = _open_database(
            prepared,
            resource_bindings_path=resource_bindings_path,
            environ=environ,
            database_factory=database_factory,
        )
        return {
            "schema_version": "non_rectangular_agent_resource_gate_v1",
            "status": "ready",
            "shared_asset_database": database.public_manifest(),
            "credential_loaded": False,
            "network_used": False,
        }
    if output_base is None:
        raise AgentFullrunError("run requires output_base")
    destination = Path(output_base).expanduser().absolute()
    with _OutputLock(destination, fresh=fresh):
        _validate_output_topology(prepared, destination, fresh=fresh)
        state = _FullrunState(prepared, destination)
        state.initialize_or_verify()
        database = _open_database(
            prepared,
            resource_bindings_path=resource_bindings_path,
            environ=environ,
            database_factory=database_factory,
        )
        agent_results: list[dict[str, Any]] = []
        for agent in prepared.profile.agents:
            result = {
                "agent_id": agent.agent_id,
                "model_id": agent.model_id,
                "complete_cases": 0,
                "failed_cases": 0,
                "cases": [],
            }
            for case in prepared.suite.cases:
                case_root = destination / agent.agent_id / case.scene_id
                resume = case_root.exists()
                state.event(
                    "case_starting",
                    agent_id=agent.agent_id,
                    model_id=agent.model_id,
                    layout_id=case.scene_id,
                    status="resume" if resume else "fresh",
                )
                try:
                    summary = episode_runner(
                        case=case,
                        asset_catalog=database,
                        agent_profile=agent,
                        tool_policy=prepared.profile.tool_policy,
                        output_root=case_root,
                        suite_identity=prepared.suite.public_dict(),
                        resume=resume,
                        environ=environ,
                    )
                    status = str(summary.get("status") or "unknown")
                    complete = status == "complete"
                    result["complete_cases" if complete else "failed_cases"] += 1
                    result["cases"].append(
                        {
                            "layout_id": case.scene_id,
                            "status": status,
                            "reason": summary.get("reason"),
                        }
                    )
                    state.event(
                        "case_terminal",
                        agent_id=agent.agent_id,
                        layout_id=case.scene_id,
                        status=status,
                        reason=summary.get("reason"),
                    )
                except Exception as exc:
                    result["failed_cases"] += 1
                    result["cases"].append(
                        {
                            "layout_id": case.scene_id,
                            "status": "runner_error",
                            "error_type": type(exc).__name__,
                        }
                    )
                    state.event(
                        "case_terminal",
                        agent_id=agent.agent_id,
                        layout_id=case.scene_id,
                        status="runner_error",
                        error_type=type(exc).__name__,
                    )
                state.write_summary(
                    _summary(prepared, agent_results + [result], terminal=False)
                )
            agent_results.append(result)
        summary = _summary(prepared, agent_results, terminal=True)
        state.write_summary(summary)
        state.event("runner_terminal", status=summary["status"])
        return summary


def _open_database(
    prepared: PreparedAgentFullrun,
    *,
    resource_bindings_path: str | Path | None,
    environ: Mapping[str, str] | None,
    database_factory: Callable[..., Any],
) -> Any:
    database = database_factory(
        catalog_path=prepared.profile.retrieval_catalog,
        resource_bindings_path=resource_bindings_path,
        retrieval_profile_id=prepared.profile.retrieval_profile_id,
        max_top_k=prepared.profile.max_top_k,
        environ=environ,
    )
    _verify_runtime_database(prepared, database)
    return database


def _verify_runtime_database(prepared: PreparedAgentFullrun, database: Any) -> None:
    manifest = database.public_manifest()
    for field in (
        "schema_version",
        "mode",
        "retrieval_profile_id",
        "dataset_id",
        "asset_namespace",
        "index_id",
        "encoder_id",
        "encoder_revision",
        "expected_asset_count",
        "expected_dimension",
        "max_top_k",
        "resource_content_sha256",
        "selection_policy",
        "per_scene_assets_prefrozen",
        "external_asset_sources_allowed",
    ):
        if manifest.get(field) != prepared.shared_database_contract.get(field):
            raise AgentFullrunError(f"runtime shared DB {field} drifted")
    if int(getattr(database, "asset_count")) != int(
        prepared.shared_database_contract["expected_asset_count"]
    ):
        raise AgentFullrunError("runtime shared DB asset count drifted")


def _summary(
    prepared: PreparedAgentFullrun,
    agent_results: list[dict[str, Any]],
    *,
    terminal: bool,
) -> dict[str, Any]:
    complete = sum(int(item["complete_cases"]) for item in agent_results)
    failed = sum(int(item["failed_cases"]) for item in agent_results)
    expected = len(prepared.profile.agents) * len(prepared.suite.cases)
    status = (
        "complete"
        if terminal and complete == expected and failed == 0
        else ("partial" if terminal else "running")
    )
    return {
        "schema_version": FULLRUN_SUMMARY_SCHEMA_VERSION,
        "fullrun_id": prepared.profile.fullrun_id,
        "track_id": prepared.profile.track_id,
        "status": status,
        "terminal": terminal,
        "expected_cases": expected,
        "complete_cases": complete,
        "failed_cases": failed,
        "remaining_cases": max(0, expected - complete - failed),
        "agents": agent_results,
    }


class _OutputLock:
    def __init__(self, output: Path, *, fresh: bool) -> None:
        self.output = output
        self.fresh = fresh
        self.handle: Any = None

    def __enter__(self) -> "_OutputLock":
        if self.output.is_symlink():
            raise AgentFullrunError("output base must not be a symlink")
        if self.fresh and self.output.exists():
            raise FileExistsError(f"fresh output already exists: {self.output}")
        self.output.mkdir(parents=True, exist_ok=not self.fresh)
        state = self.output / "_runner_state"
        state.mkdir(exist_ok=True)
        path = state / "runner.lock"
        if path.is_symlink():
            raise AgentFullrunError("runner lock must not be a symlink")
        self.handle = path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise AgentFullrunError("another writer holds the Agent runner lock") from exc
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class _FullrunState:
    def __init__(self, prepared: PreparedAgentFullrun, output: Path) -> None:
        self.prepared = prepared
        self.output = output
        self.root = output / "_runner_state"
        self.manifest = self.root / "run_manifest.json"
        self.events = self.root / "events.jsonl"
        self.summary = self.root / "summary.json"

    def initialize_or_verify(self) -> None:
        expected = _coordinator_manifest(self.prepared)
        if self.manifest.exists():
            if self.manifest.is_symlink() or _load_json(self.manifest) != expected:
                raise AgentFullrunError("Agent runner manifest identity mismatch")
        else:
            _write_json_exclusive(self.manifest, expected)

    def event(self, event: str, **fields: Any) -> None:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **{key: value for key, value in fields.items() if value is not None},
        }
        if self.events.is_symlink():
            raise AgentFullrunError("Agent runner event journal must not be a symlink")
        with self.events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def write_summary(self, value: Mapping[str, Any]) -> None:
        temporary = self.summary.with_name(f".{self.summary.name}.tmp")
        encoded = json.dumps(
            dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        ) + "\n"
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, self.summary)


def _coordinator_manifest(prepared: PreparedAgentFullrun) -> dict[str, Any]:
    return {
        "schema_version": FULLRUN_MANIFEST_SCHEMA_VERSION,
        "fullrun_id": prepared.profile.fullrun_id,
        "track_id": prepared.profile.track_id,
        "profile_sha256": prepared.profile_sha256,
        "suite": prepared.suite.public_dict(),
        "shared_database_contract_sha256": prepared.shared_database_contract_sha256,
        "shared_database_snapshot_id": prepared.shared_database_contract["snapshot_id"],
        "agent_order": [
            {
                "agent_id": agent.agent_id,
                "implementation": agent.implementation,
                "implementation_version": agent.implementation_version,
                "model_id": agent.model_id,
            }
            for agent in prepared.profile.agents
        ],
        "tool_policy": prepared.profile.tool_policy.public_dict(),
        "source_manifest_sha256": agent_source_manifest()["manifest_sha256"],
    }


def _validate_output_topology(
    prepared: PreparedAgentFullrun, output: Path, *, fresh: bool
) -> None:
    allowed = {agent.agent_id for agent in prepared.profile.agents} | {"_runner_state"}
    extras = {path.name for path in output.iterdir()} - allowed
    if extras:
        raise AgentFullrunError("Agent output root contains unknown entries")
    expected_scenes = set(prepared.suite.scene_order)
    for agent in prepared.profile.agents:
        root = output / agent.agent_id
        if not root.exists():
            continue
        if not root.is_dir() or root.is_symlink():
            raise AgentFullrunError("Agent output must be a real directory")
        if {path.name for path in root.iterdir()} - expected_scenes:
            raise AgentFullrunError("Agent output contains unknown layouts")
    if fresh and any((output / agent.agent_id).exists() for agent in prepared.profile.agents):
        raise AgentFullrunError("fresh run found existing Agent outputs")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentFullrunError(f"cannot load JSON artifact: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise AgentFullrunError("JSON artifact must be an object")
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "AgentFullrunError",
    "PreparedAgentFullrun",
    "execute_agent_fullrun",
    "prepare_agent_fullrun",
]
