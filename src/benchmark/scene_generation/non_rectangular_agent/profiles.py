"""Strict public profiles for Agent implementations and shared task policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

from .tool_server import AgentToolPolicy


TRACK_PROFILE_SCHEMA_VERSION = "non_rectangular_agent_track_profile_v1"
_PORTABLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_FORBIDDEN_ENVIRONMENT = frozenset(
    {
        "PYTHONPATH",
        "PYTHONHOME",
        "PWD",
        "OLDPWD",
        "DYLD_INSERT_LIBRARIES",
    }
)


class AgentProfileError(ValueError):
    """Raised when a public Agent-track profile is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class AgentBackendProfile:
    agent_id: str
    display_name: str
    implementation: str
    implementation_version: str
    model_id: str
    command: tuple[str, ...]
    prompt_transport: str
    isolation_mode: str
    timeout_seconds: float
    max_process_attempts: int
    retry_delay_seconds: float
    retryable_exit_codes: tuple[int, ...]
    pass_environment: tuple[str, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "implementation": self.implementation,
            "implementation_version": self.implementation_version,
            "model_id": self.model_id,
            "command": list(self.command),
            "prompt_transport": self.prompt_transport,
            "isolation_mode": self.isolation_mode,
            "timeout_seconds": self.timeout_seconds,
            "max_process_attempts": self.max_process_attempts,
            "retry_delay_seconds": self.retry_delay_seconds,
            "retryable_exit_codes": list(self.retryable_exit_codes),
            "pass_environment": list(self.pass_environment),
            "credential_values_present": False,
        }


@dataclass(frozen=True, slots=True)
class AgentTrackProfile:
    path: Path
    fullrun_id: str
    track_id: str
    suite_root: Path
    retrieval_catalog: Path
    shared_database_contract: Path
    retrieval_profile_id: str
    max_top_k: int
    tool_policy: AgentToolPolicy
    agents: tuple[AgentBackendProfile, ...]

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TRACK_PROFILE_SCHEMA_VERSION,
            "fullrun_id": self.fullrun_id,
            "track_id": self.track_id,
            "suite_root": self.suite_root.as_posix(),
            "retrieval_catalog": self.retrieval_catalog.as_posix(),
            "shared_database_contract": self.shared_database_contract.as_posix(),
            "retrieval_profile_id": self.retrieval_profile_id,
            "max_top_k": self.max_top_k,
            "tool_policy": self.tool_policy.public_dict(),
            "agents": [agent.public_dict() for agent in self.agents],
        }


def load_agent_track_profile(
    path: str | Path, *, repo_root: str | Path
) -> AgentTrackProfile:
    root = Path(repo_root).expanduser().resolve()
    profile_path = _repo_file(root, path, label="Agent-track profile")
    value = _load_json(profile_path)
    _exact(
        value,
        {
            "schema_version",
            "fullrun_id",
            "track_id",
            "suite_root",
            "retrieval_catalog",
            "shared_database_contract",
            "retrieval_profile_id",
            "max_top_k",
            "tool_policy",
            "agents",
        },
        label="Agent-track profile",
    )
    if value["schema_version"] != TRACK_PROFILE_SCHEMA_VERSION:
        raise AgentProfileError("unsupported Agent-track profile schema")
    fullrun_id = _portable(value["fullrun_id"], label="fullrun_id")
    track_id = _portable(value["track_id"], label="track_id")
    if track_id != "complicated_floorplan_agent_track_v1":
        raise AgentProfileError("profile selects the wrong Agent track")
    max_top_k = _positive_int(value["max_top_k"], label="max_top_k")
    tool_raw = value["tool_policy"]
    if not isinstance(tool_raw, dict):
        raise AgentProfileError("tool_policy must be an object")
    _exact(
        tool_raw,
        {
            "max_total_calls",
            "max_asset_searches",
            "max_asset_inspections",
            "max_submission_validations",
        },
        label="tool_policy",
    )
    tool_policy = AgentToolPolicy(
        max_total_calls=_positive_int(
            tool_raw["max_total_calls"], label="max_total_calls"
        ),
        max_asset_searches=_positive_int(
            tool_raw["max_asset_searches"], label="max_asset_searches"
        ),
        max_asset_inspections=_positive_int(
            tool_raw["max_asset_inspections"], label="max_asset_inspections"
        ),
        max_submission_validations=_positive_int(
            tool_raw["max_submission_validations"],
            label="max_submission_validations",
        ),
        max_top_k=max_top_k,
    )
    agent_values = value["agents"]
    if not isinstance(agent_values, list) or not agent_values:
        raise AgentProfileError("agents must be a non-empty array")
    agents = tuple(_parse_agent(item, index=index) for index, item in enumerate(agent_values))
    if len({agent.agent_id for agent in agents}) != len(agents):
        raise AgentProfileError("Agent profile IDs must be unique")
    return AgentTrackProfile(
        path=profile_path,
        fullrun_id=fullrun_id,
        track_id=track_id,
        suite_root=_repo_file(root, value["suite_root"], label="suite_root", directory=True),
        retrieval_catalog=_repo_file(
            root, value["retrieval_catalog"], label="retrieval_catalog"
        ),
        shared_database_contract=_repo_file(
            root,
            value["shared_database_contract"],
            label="shared_database_contract",
        ),
        retrieval_profile_id=_text(
            value["retrieval_profile_id"], label="retrieval_profile_id"
        ),
        max_top_k=max_top_k,
        tool_policy=tool_policy,
        agents=agents,
    )


def _parse_agent(value: Any, *, index: int) -> AgentBackendProfile:
    label = f"agents[{index}]"
    if not isinstance(value, dict):
        raise AgentProfileError(f"{label} must be an object")
    _exact(
        value,
        {
            "agent_id",
            "display_name",
            "implementation",
            "implementation_version",
            "model_id",
            "command",
            "prompt_transport",
            "isolation_mode",
            "timeout_seconds",
            "max_process_attempts",
            "retry_delay_seconds",
            "retryable_exit_codes",
            "pass_environment",
        },
        label=label,
    )
    command = _text_list(value["command"], label=f"{label}.command")
    if not command:
        raise AgentProfileError(f"{label}.command must be non-empty")
    prompt_transport = _text(
        value["prompt_transport"], label=f"{label}.prompt_transport"
    )
    if prompt_transport != "stdin":
        raise AgentProfileError("Agent command prompt_transport must be stdin")
    isolation_mode = _text(
        value["isolation_mode"], label=f"{label}.isolation_mode"
    )
    if isolation_mode != "backend_enforced_task_workspace_only":
        raise AgentProfileError(
            "Agent backend must attest task-workspace-only isolation"
        )
    retryable = value["retryable_exit_codes"]
    if not isinstance(retryable, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or not 1 <= item <= 255
        for item in retryable
    ):
        raise AgentProfileError(f"{label}.retryable_exit_codes is invalid")
    pass_environment = _text_list(
        value["pass_environment"], label=f"{label}.pass_environment"
    )
    if any(
        not _ENV_NAME.fullmatch(name)
        or name in _FORBIDDEN_ENVIRONMENT
        or name.startswith("LAYOUT_DDD_AGENT_TOOL_")
        for name in pass_environment
    ):
        raise AgentProfileError("pass_environment contains an unsafe name")
    return AgentBackendProfile(
        agent_id=_portable(value["agent_id"], label=f"{label}.agent_id"),
        display_name=_text(value["display_name"], label=f"{label}.display_name"),
        implementation=_text(
            value["implementation"], label=f"{label}.implementation"
        ),
        implementation_version=_text(
            value["implementation_version"],
            label=f"{label}.implementation_version",
        ),
        model_id=_text(value["model_id"], label=f"{label}.model_id"),
        command=tuple(command),
        prompt_transport=prompt_transport,
        isolation_mode=isolation_mode,
        timeout_seconds=_positive_number(
            value["timeout_seconds"], label=f"{label}.timeout_seconds"
        ),
        max_process_attempts=_positive_int(
            value["max_process_attempts"],
            label=f"{label}.max_process_attempts",
        ),
        retry_delay_seconds=_nonnegative_number(
            value["retry_delay_seconds"],
            label=f"{label}.retry_delay_seconds",
        ),
        retryable_exit_codes=tuple(dict.fromkeys(retryable)),
        pass_environment=tuple(dict.fromkeys(pass_environment)),
    )


def _repo_file(
    root: Path, value: Any, *, label: str, directory: bool = False
) -> Path:
    relative = value if isinstance(value, Path) else Path(_text(value, label=label))
    if ".." in relative.parts:
        raise AgentProfileError(f"{label} must not contain parent traversal")
    path = relative.resolve() if relative.is_absolute() else (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise AgentProfileError(f"{label} escapes repository root") from exc
    valid = path.is_dir() if directory else path.is_file()
    if not valid or path.is_symlink():
        kind = "directory" if directory else "file"
        raise AgentProfileError(
            f"{label} does not resolve to a real {kind}"
        )
    return path


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentProfileError(f"cannot load Agent profile: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise AgentProfileError("Agent profile root must be an object")
    return value


def _exact(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise AgentProfileError(f"{label} keys differ from the fixed contract")


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise AgentProfileError(f"{label} must be trimmed text")
    return value


def _portable(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    if not _PORTABLE.fullmatch(text):
        raise AgentProfileError(f"{label} must be a portable identifier")
    return text


def _text_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise AgentProfileError(f"{label} must be an array")
    return [_text(item, label=f"{label}[]") for item in value]


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise AgentProfileError(f"{label} must be a positive integer")
    return value


def _positive_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise AgentProfileError(f"{label} must be positive")
    return float(value)


def _nonnegative_number(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise AgentProfileError(f"{label} must be non-negative")
    return float(value)


__all__ = [
    "AgentBackendProfile",
    "AgentProfileError",
    "AgentTrackProfile",
    "TRACK_PROFILE_SCHEMA_VERSION",
    "load_agent_track_profile",
]
