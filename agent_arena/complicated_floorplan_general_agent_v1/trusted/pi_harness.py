#!/usr/bin/env python3
"""Build a deterministic Pi command and per-episode gateway model config."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping
from urllib.parse import urlsplit

from api_profiles import ModelProfile


ARENA_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = Path(__file__).resolve().parent / "pi_harness"
HARNESS_CONTRACT = HARNESS_ROOT / "harness.json"
SYSTEM_PROMPT = HARNESS_ROOT / "SYSTEM.md"
HARNESS_EXTENSION = HARNESS_ROOT / "sequential_tools_extension.ts"
PI_VERSION = "0.85.0"
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "1adcb01a1f558a1ab2a5728e6d0d70941571bfce0dd12e9939d0a88b2c06699f"
)
EXPECTED_RUNTIME_CONTENT_ROOT_SHA256 = (
    "018b0b92bcff8bf89370d5457c89a9c00ed286b54544a7ff9741658e5128dc0f"
)
PI_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a "
    "conversation between a user and an AI assistant, then produce a structured "
    "summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)
ROUTE_PREFLIGHT_SYSTEM_PROMPT = (
    "You are a SIEVE transport compatibility probe running in an isolated "
    "workspace. Follow the user request exactly. Use only the registered "
    "tools. Never inspect credentials, parent directories, or networks."
)
ROUTE_PREFLIGHT_TASK_PROMPT = (
    "Use the read tool exactly once on preflight_fixture.txt. After the tool "
    "succeeds, reply with exactly SIEVE_PREFLIGHT_OK and make no other tool calls.\n"
)
ROUTE_PREFLIGHT_FIXTURE = "SIEVE_ROUTE_PREFLIGHT_FIXTURE_V1\n"
PROVIDER_ID = "sieve-gateway"
DATABASE_EVENT_SCHEMA_VERSION = "non_rectangular_agent_tool_event_v3"
SUPPORTED_APIS = frozenset({"openai-completions", "openai-responses"})
THINKING_LEVELS = frozenset({"off", "minimal", "low", "medium", "high"})
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CLI_RELATIVE = Path(
    "node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js"
)
NODE_RELATIVE = Path("bin/node")
RUNTIME_MANIFEST_RELATIVE = Path("runtime.manifest.json")


class PiHarnessError(RuntimeError):
    """Raised when the pinned runtime or episode configuration is invalid."""


@dataclass(frozen=True)
class PiEpisodeConfig:
    runtime_root: Path
    workspace: Path
    gateway_base_url: str
    model_profile: ModelProfile
    experiment_id: str
    experiment_sha256: str
    profile_registry_sha256: str
    max_model_requests: int
    wall_clock_seconds: int


def verify_runtime(runtime_root: str | Path) -> dict[str, Any]:
    candidate = Path(runtime_root).expanduser().absolute()
    if not candidate.is_dir() or candidate.is_symlink():
        raise PiHarnessError("Pi runtime root must be a real directory")
    root = candidate.resolve(strict=True)
    node = _real_file(root / NODE_RELATIVE, "Node executable")
    cli = _real_file(root / CLI_RELATIVE, "Pi CLI")
    package = _real_file(
        root / "node_modules/@earendil-works/pi-coding-agent/package.json",
        "Pi package manifest",
    )
    value = json.loads(package.read_text(encoding="utf-8"))
    if value.get("name") != "@earendil-works/pi-coding-agent":
        raise PiHarnessError("Pi runtime package identity differs")
    if value.get("version") != PI_VERSION:
        raise PiHarnessError("Pi runtime version differs")
    manifest_path = _real_file(
        root / RUNTIME_MANIFEST_RELATIVE,
        "Pi runtime manifest",
    )
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise PiHarnessError("Pi runtime manifest hash differs from the arena pin")
    try:
        manifest = json.loads(
            manifest_bytes.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, ValueError) as exc:
        raise PiHarnessError("Pi runtime manifest is not strict JSON") from exc
    if manifest.get("schema_version") != "sieve_pi_runtime_manifest_v1":
        raise PiHarnessError("Pi runtime manifest schema differs")
    if manifest.get("platform") != "darwin-arm64":
        raise PiHarnessError("Pi runtime platform differs")
    if manifest.get("pi_version") != PI_VERSION:
        raise PiHarnessError("Pi runtime manifest version differs")
    observed = _runtime_tree_fingerprint(root)
    if observed["content_root_sha256"] != EXPECTED_RUNTIME_CONTENT_ROOT_SHA256:
        raise PiHarnessError("Pi runtime content root differs from the arena pin")
    for key in (
        "regular_file_count",
        "symlink_count",
        "entry_count",
        "content_root_sha256",
    ):
        if manifest.get(key) != observed[key]:
            raise PiHarnessError(f"Pi runtime fingerprint differs: {key}")
    selected_hashes = {
        "node_sha256": hashlib.sha256(node.read_bytes()).hexdigest(),
        "pi_cli_sha256": hashlib.sha256(cli.read_bytes()).hexdigest(),
        "pi_package_manifest_sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
    }
    for key, digest in selected_hashes.items():
        if manifest.get(key) != digest:
            raise PiHarnessError(f"Pi runtime fingerprint differs: {key}")
    return {
        "runtime_root": str(root),
        "node": str(node),
        "cli": str(cli),
        "pi_version": PI_VERSION,
        "content_root_sha256": observed["content_root_sha256"],
        "runtime_manifest_sha256": manifest_sha256,
        **selected_hashes,
    }


def prepare_episode(config: PiEpisodeConfig) -> dict[str, Any]:
    runtime = verify_runtime(config.runtime_root)
    harness = json.loads(HARNESS_CONTRACT.read_text(encoding="utf-8"))
    if (
        not isinstance(harness, dict)
        or harness.get("schema_version") != "sieve_pi_common_harness_v4"
        or harness.get("harness_id") != "sieve-pi-common-harness-v4"
    ):
        raise PiHarnessError("Pi harness contract identity differs")
    workspace_input = config.workspace.expanduser().absolute()
    if not workspace_input.is_dir() or workspace_input.is_symlink():
        raise PiHarnessError("episode workspace must be a real directory")
    workspace = workspace_input.resolve(strict=True)
    _assert_pristine_workspace(workspace)
    todo = _real_file(workspace / "TODO.md", "episode TODO")
    extension_path = _real_file(HARNESS_EXTENSION, "Pi harness extension")
    model_profile = config.model_profile
    if not isinstance(model_profile, ModelProfile):
        raise PiHarnessError("model_profile must be a frozen registered profile")
    api = _choice(model_profile.pi.api_protocol, SUPPORTED_APIS, "Pi API protocol")
    thinking = _choice(
        model_profile.pi.thinking_level, THINKING_LEVELS, "thinking level"
    )
    wire_model = _model_id(model_profile.client_wire_model)
    context_window = _positive_int(
        model_profile.pi.context_window, "context window"
    )
    max_tokens = _positive_int(
        model_profile.pi.maximum_output_tokens, "max tokens"
    )
    max_model_requests = _positive_int(
        config.max_model_requests,
        "maximum model requests",
    )
    wall_clock_seconds = _positive_int(
        config.wall_clock_seconds,
        "wall-clock seconds",
    )
    if max_tokens > context_window:
        raise PiHarnessError("max tokens cannot exceed the context window")
    base_url = _gateway_base_url(config.gateway_base_url)
    experiment_id = _portable_id(config.experiment_id, "experiment_id")
    experiment_sha256 = _sha256(config.experiment_sha256, "experiment_sha256")
    registry_sha256 = _sha256(
        config.profile_registry_sha256, "profile_registry_sha256"
    )

    agent_dir = workspace / ".home/.pi/agent"
    agent_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    models_path = agent_dir / "models.json"
    pi_model = {
        "compat": dict(model_profile.pi.compatibility),
        "contextWindow": context_window,
        "cost": {
            "cacheRead": 0,
            "cacheWrite": 0,
            "input": 0,
            "output": 0,
        },
        "id": wire_model,
        "input": ["text"],
        "maxTokens": max_tokens,
        "name": model_profile.display_label,
        "reasoning": model_profile.reasoning.style != "none",
    }
    if model_profile.reasoning.style != "none":
        pi_model["thinkingLevelMap"] = {
            "off": None,
            thinking: model_profile.reasoning.effort,
        }
    models = {
        "providers": {
            PROVIDER_ID: {
                "api": api,
                "apiKey": "$ARENA_MODEL_GATEWAY_TOKEN",
                "authHeader": True,
                "baseUrl": base_url,
                "models": [pi_model],
                "name": "SIEVE Scoped Model Gateway",
            }
        }
    }
    _write_json_exclusive(models_path, models, mode=0o400)

    # Pi has both an Agent-level auto-retry loop and a provider-SDK retry
    # layer.  Both must be disabled: the trusted gateway is the sole owner of
    # transport/HTTP retries and refuses ambiguous retries.  Otherwise Pi
    # could silently repeat a request after the gateway has already failed it
    # closed, invalidating both cost accounting and comparability.
    settings_path = agent_dir / "settings.json"
    pi_timeout_ms = min(2_147_483_647, wall_clock_seconds * 1000)
    settings = {
        "branchSummary": {"reserveTokens": 16384, "skipPrompt": False},
        "compaction": {
            "enabled": True,
            "keepRecentTokens": 20000,
            "reserveTokens": 16384,
        },
        "defaultProjectTrust": "never",
        "defaultTools": ["read", "write", "edit", "bash"],
        "enableAnalytics": False,
        "enableInstallTelemetry": False,
        "httpIdleTimeoutMs": pi_timeout_ms,
        "packages": [],
        "shellCommandPrefix": "unset ARENA_MODEL_GATEWAY_TOKEN",
        "retry": {
            "enabled": False,
            "maxRetries": 0,
            "baseDelayMs": 0,
            "provider": {
                "timeoutMs": pi_timeout_ms,
                "maxRetries": 0,
                "maxRetryDelayMs": 0,
            },
        },
        "transport": "sse",
    }
    _write_json_exclusive(settings_path, settings, mode=0o400)

    system_prompt = SYSTEM_PROMPT.read_text(encoding="utf-8")
    task_prompt = todo.read_text(encoding="utf-8")
    command = [
        str(runtime["node"]),
        str(runtime["cli"]),
        "--mode",
        "json",
        "--print",
        "--no-session",
        "--offline",
        "--no-context-files",
        "--no-skills",
        "--no-prompt-templates",
        "--no-extensions",
        "--extension",
        str(extension_path),
        "--no-themes",
        "--no-approve",
        "--provider",
        PROVIDER_ID,
        "--model",
        wire_model,
        "--thinking",
        thinking,
        "--tools",
        "read,write,edit,bash",
        "--system-prompt",
        system_prompt,
    ]
    task = json.loads((workspace / "task.json").read_text(encoding="utf-8"))
    if not isinstance(task, dict):
        raise PiHarnessError("episode task must be a JSON object")
    tool_policy = task.get("tool_policy")
    if not isinstance(tool_policy, dict):
        raise PiHarnessError("episode task lacks the frozen tool policy")
    launch_record = {
        "schema_version": "sieve_pi_episode_launch_record_v3",
        "harness_id": "sieve-pi-common-harness-v4",
        "harness_contract_sha256": hashlib.sha256(
            HARNESS_CONTRACT.read_bytes()
        ).hexdigest(),
        "runtime": {
            "package": "@earendil-works/pi-coding-agent",
            "version": PI_VERSION,
            "content_root_sha256": runtime["content_root_sha256"],
            "runtime_manifest_sha256": runtime["runtime_manifest_sha256"],
            "node_sha256": runtime["node_sha256"],
            "pi_cli_sha256": runtime["pi_cli_sha256"],
            "pi_package_manifest_sha256": runtime[
                "pi_package_manifest_sha256"
            ],
        },
        "prompts": {
            "source_system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "effective_pi_system_prompt_sha256": hashlib.sha256(
                _effective_pi_system_prompt(system_prompt, workspace).encode("utf-8")
            ).hexdigest(),
            "pi_compaction_system_prompt_sha256": hashlib.sha256(
                PI_SUMMARIZATION_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "provider_visible_episode_system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "provider_visible_compaction_system_prompt_sha256": hashlib.sha256(
                PI_SUMMARIZATION_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "pi_cwd_suffix_forwarded_to_provider": False,
            "task_prompt_sha256": hashlib.sha256(
                task_prompt.encode("utf-8")
            ).hexdigest(),
        },
        "experiment": {
            "experiment_id": experiment_id,
            "experiment_sha256": experiment_sha256,
            "profile_registry_sha256": registry_sha256,
        },
        "model": {
            "provider": PROVIDER_ID,
            "model_profile_id": model_profile.model_profile_id,
            "model_profile_sha256": hashlib.sha256(
                json.dumps(
                    model_profile.public_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "client_wire_model": wire_model,
            "upstream_wire_model": model_profile.upstream_wire_model,
            "api_family_id": model_profile.api_family_id,
            "route_profile_id": model_profile.route_profile_id,
            "api_protocol": api,
            "pi_thinking_level": thinking,
            "provider_reasoning_profile": model_profile.reasoning.public_dict(),
            "provider_reasoning_field_owner": "trusted_gateway_v2",
            "reasoning_replay_policy": (
                "preserve_for_provider_replay_but_exclude_from_public_logs_v1"
                if model_profile.reasoning.preserve_across_tool_turns
                else "not_required_v1"
            ),
            "pi_compatibility": dict(model_profile.pi.compatibility),
            "temperature": (
                model_profile.temperature
                if model_profile.temperature is not None
                else "provider_default_not_overridden"
            ),
            "context_window": context_window,
            "maximum_output_tokens": max_tokens,
            "retry_policy": model_profile.retry.public_dict(),
        },
        "limits": {
            "maximum_model_turns": max_model_requests,
            "maximum_model_requests": max_model_requests,
            "wall_clock_seconds": wall_clock_seconds,
            "maximum_concurrent_tool_calls": 1,
            "tool_policy": tool_policy,
        },
        "retry_ownership": {
            "pi_agent_auto_retry_enabled": False,
            "pi_provider_sdk_max_retries": 0,
            "trusted_gateway_is_only_retry_owner": True,
            "ambiguous_request_retry": False,
        },
        "pi_settings": {
            "settings_sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest(),
            "models_sha256": hashlib.sha256(models_path.read_bytes()).hexdigest(),
            "transport": "sse",
            "http_idle_timeout_ms": pi_timeout_ms,
            "compaction": settings["compaction"],
            "branch_summary": settings["branchSummary"],
            "shell_command_prefix": settings["shellCommandPrefix"],
        },
        "tooling": {
            "harness_extension_sha256": hashlib.sha256(
                extension_path.read_bytes()
            ).hexdigest(),
            "all_tools_execution_mode": "sequential",
            "bash_model_gateway_capability_inherited": False,
            "sieve_agent_tool_sha256": hashlib.sha256(
                (workspace / "sieve-agent-tool").read_bytes()
            ).hexdigest(),
            "public_tool_policy": tool_policy,
        },
        "database_snapshot": task.get("asset_database"),
        "starting_workspace_sha256": _workspace_input_root(workspace),
        "validator_policy": task.get("public_validation_policy"),
        "tool_transcript": {
            "database_event_schema": DATABASE_EVENT_SCHEMA_VERSION,
            "database_transcript_hash_chained_host_side": True,
            "pi_event_projection_schema": "sieve_pi_tool_transcript_v1",
            "pi_event_projection_hash_chained_host_side": True,
            "records_complete_public_results_when_process_stream_is_complete": True,
            "records_credentials_headers_endpoints_or_hidden_reasoning": False,
        },
    }
    return {
        "schema_version": "sieve_pi_episode_launch_material_v3",
        "command": command,
        "stdin_text": task_prompt,
        "models_path": str(models_path),
        "settings_path": str(settings_path),
        "harness_extension_path": str(extension_path),
        "effective_system_prompt_sha256": launch_record["prompts"][
            "effective_pi_system_prompt_sha256"
        ],
        "allowed_system_prompt_sha256s": allowed_pi_system_prompt_sha256s(
            workspace
        ),
        "provider_id": PROVIDER_ID,
        "wire_model": wire_model,
        "api": api,
        "thinking": thinking,
        "runtime": runtime,
        "launch_record": launch_record,
    }


def verify_prepared_episode_material(material: Mapping[str, Any]) -> None:
    """Recheck Agent-visible harness files before accepting an episode."""

    if not isinstance(material, Mapping):
        raise PiHarnessError("launch material is malformed")
    record = material.get("launch_record")
    if not isinstance(record, Mapping):
        raise PiHarnessError("launch record is missing")
    settings = record.get("pi_settings")
    tooling = record.get("tooling")
    if not isinstance(settings, Mapping) or not isinstance(tooling, Mapping):
        raise PiHarnessError("launch record hashes are missing")
    checks = (
        (
            material.get("models_path"),
            settings.get("models_sha256"),
            "models.json",
        ),
        (
            material.get("settings_path"),
            settings.get("settings_sha256"),
            "settings.json",
        ),
        (
            material.get("harness_extension_path"),
            tooling.get("harness_extension_sha256"),
            "harness extension",
        ),
    )
    for raw_path, expected, label in checks:
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            raise PiHarnessError(f"{label} verification material is malformed")
        path = _real_file(Path(raw_path), label)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise PiHarnessError(f"{label} changed during the episode")


def verify_existing_episode_launch(
    *,
    launch_record: Mapping[str, Any],
    workspace: str | Path,
    runtime_root: str | Path,
    model_profile: ModelProfile,
    experiment_id: str,
    experiment_sha256: str,
    profile_registry_sha256: str,
    max_model_requests: int,
    wall_clock_seconds: int,
) -> dict[str, Any]:
    """Rebuild the full pinned-Pi launch contract for safe resume.

    The gateway port is intentionally ephemeral and private.  It is validated
    as a loopback-only value in the sealed models.json; every other field is
    reconstructed from frozen inputs and compared exactly.
    """

    if not isinstance(launch_record, Mapping):
        raise PiHarnessError("existing launch record is malformed")
    runtime = verify_runtime(runtime_root)
    workspace_input = Path(workspace).expanduser().absolute()
    if not workspace_input.is_dir() or workspace_input.is_symlink():
        raise PiHarnessError("existing episode workspace must be a real directory")
    workspace_path = workspace_input.resolve(strict=True)
    model = model_profile
    api = _choice(model.pi.api_protocol, SUPPORTED_APIS, "Pi API protocol")
    thinking = _choice(model.pi.thinking_level, THINKING_LEVELS, "thinking level")
    wire_model = _model_id(model.client_wire_model)
    context_window = _positive_int(model.pi.context_window, "context window")
    maximum_output_tokens = _positive_int(
        model.pi.maximum_output_tokens, "max tokens"
    )
    maximum_requests = _positive_int(max_model_requests, "maximum model requests")
    wall_clock = _positive_int(wall_clock_seconds, "wall-clock seconds")
    experiment_name = _portable_id(experiment_id, "experiment_id")
    experiment_hash = _sha256(experiment_sha256, "experiment_sha256")
    registry_hash = _sha256(profile_registry_sha256, "profile_registry_sha256")
    task = _read_strict_json_file(workspace_path / "task.json", "episode task")
    tool_policy = task.get("tool_policy")
    if not isinstance(tool_policy, dict):
        raise PiHarnessError("episode task lacks the frozen tool policy")

    models_path = workspace_path / ".home/.pi/agent/models.json"
    settings_path = workspace_path / ".home/.pi/agent/settings.json"
    _require_file_mode(models_path, 0o400, "models.json")
    _require_file_mode(settings_path, 0o400, "settings.json")
    models = _read_strict_json_file(models_path, "models.json")
    settings = _read_strict_json_file(settings_path, "settings.json")
    providers = models.get("providers")
    if not isinstance(providers, dict) or set(providers) != {PROVIDER_ID}:
        raise PiHarnessError("models.json provider set differs")
    provider = providers.get(PROVIDER_ID)
    if not isinstance(provider, dict):
        raise PiHarnessError("models.json provider is malformed")
    observed_base_url = provider.get("baseUrl")
    if not isinstance(observed_base_url, str):
        raise PiHarnessError("models.json gateway URL is malformed")
    canonical_base_url = _gateway_base_url(observed_base_url)
    if observed_base_url != canonical_base_url:
        raise PiHarnessError("models.json gateway URL is not canonical")
    pi_model: dict[str, Any] = {
        "compat": dict(model.pi.compatibility),
        "contextWindow": context_window,
        "cost": {
            "cacheRead": 0,
            "cacheWrite": 0,
            "input": 0,
            "output": 0,
        },
        "id": wire_model,
        "input": ["text"],
        "maxTokens": maximum_output_tokens,
        "name": model.display_label,
        "reasoning": model.reasoning.style != "none",
    }
    if model.reasoning.style != "none":
        pi_model["thinkingLevelMap"] = {
            "off": None,
            thinking: model.reasoning.effort,
        }
    expected_models = {
        "providers": {
            PROVIDER_ID: {
                "api": api,
                "apiKey": "$ARENA_MODEL_GATEWAY_TOKEN",
                "authHeader": True,
                "baseUrl": canonical_base_url,
                "models": [pi_model],
                "name": "SIEVE Scoped Model Gateway",
            }
        }
    }
    if models != expected_models:
        raise PiHarnessError("existing models.json differs from the frozen launch")

    timeout_ms = min(2_147_483_647, wall_clock * 1000)
    expected_settings = {
        "branchSummary": {"reserveTokens": 16384, "skipPrompt": False},
        "compaction": {
            "enabled": True,
            "keepRecentTokens": 20000,
            "reserveTokens": 16384,
        },
        "defaultProjectTrust": "never",
        "defaultTools": ["read", "write", "edit", "bash"],
        "enableAnalytics": False,
        "enableInstallTelemetry": False,
        "httpIdleTimeoutMs": timeout_ms,
        "packages": [],
        "shellCommandPrefix": "unset ARENA_MODEL_GATEWAY_TOKEN",
        "retry": {
            "enabled": False,
            "maxRetries": 0,
            "baseDelayMs": 0,
            "provider": {
                "timeoutMs": timeout_ms,
                "maxRetries": 0,
                "maxRetryDelayMs": 0,
            },
        },
        "transport": "sse",
    }
    if settings != expected_settings:
        raise PiHarnessError("existing settings.json differs from the frozen launch")

    extension = _real_file(HARNESS_EXTENSION, "Pi harness extension")
    system_prompt = _real_file(SYSTEM_PROMPT, "Pi system prompt").read_text(
        encoding="utf-8"
    )
    task_prompt = _real_file(workspace_path / "TODO.md", "episode TODO").read_text(
        encoding="utf-8"
    )
    expected_record = {
        "schema_version": "sieve_pi_episode_launch_record_v3",
        "harness_id": "sieve-pi-common-harness-v4",
        "harness_contract_sha256": hashlib.sha256(
            HARNESS_CONTRACT.read_bytes()
        ).hexdigest(),
        "runtime": {
            "package": "@earendil-works/pi-coding-agent",
            "version": PI_VERSION,
            "content_root_sha256": runtime["content_root_sha256"],
            "runtime_manifest_sha256": runtime["runtime_manifest_sha256"],
            "node_sha256": runtime["node_sha256"],
            "pi_cli_sha256": runtime["pi_cli_sha256"],
            "pi_package_manifest_sha256": runtime[
                "pi_package_manifest_sha256"
            ],
        },
        "prompts": {
            "source_system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "effective_pi_system_prompt_sha256": hashlib.sha256(
                _effective_pi_system_prompt(system_prompt, workspace_path).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "pi_compaction_system_prompt_sha256": hashlib.sha256(
                PI_SUMMARIZATION_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "provider_visible_episode_system_prompt_sha256": hashlib.sha256(
                system_prompt.encode("utf-8")
            ).hexdigest(),
            "provider_visible_compaction_system_prompt_sha256": hashlib.sha256(
                PI_SUMMARIZATION_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "pi_cwd_suffix_forwarded_to_provider": False,
            "task_prompt_sha256": hashlib.sha256(
                task_prompt.encode("utf-8")
            ).hexdigest(),
        },
        "experiment": {
            "experiment_id": experiment_name,
            "experiment_sha256": experiment_hash,
            "profile_registry_sha256": registry_hash,
        },
        "model": {
            "provider": PROVIDER_ID,
            "model_profile_id": model.model_profile_id,
            "model_profile_sha256": hashlib.sha256(
                json.dumps(
                    model.public_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
            "client_wire_model": wire_model,
            "upstream_wire_model": model.upstream_wire_model,
            "api_family_id": model.api_family_id,
            "route_profile_id": model.route_profile_id,
            "api_protocol": api,
            "pi_thinking_level": thinking,
            "provider_reasoning_profile": model.reasoning.public_dict(),
            "provider_reasoning_field_owner": "trusted_gateway_v2",
            "reasoning_replay_policy": (
                "preserve_for_provider_replay_but_exclude_from_public_logs_v1"
                if model.reasoning.preserve_across_tool_turns
                else "not_required_v1"
            ),
            "pi_compatibility": dict(model.pi.compatibility),
            "temperature": (
                model.temperature
                if model.temperature is not None
                else "provider_default_not_overridden"
            ),
            "context_window": context_window,
            "maximum_output_tokens": maximum_output_tokens,
            "retry_policy": model.retry.public_dict(),
        },
        "limits": {
            "maximum_model_turns": maximum_requests,
            "maximum_model_requests": maximum_requests,
            "wall_clock_seconds": wall_clock,
            "maximum_concurrent_tool_calls": 1,
            "tool_policy": tool_policy,
        },
        "retry_ownership": {
            "pi_agent_auto_retry_enabled": False,
            "pi_provider_sdk_max_retries": 0,
            "trusted_gateway_is_only_retry_owner": True,
            "ambiguous_request_retry": False,
        },
        "pi_settings": {
            "settings_sha256": hashlib.sha256(settings_path.read_bytes()).hexdigest(),
            "models_sha256": hashlib.sha256(models_path.read_bytes()).hexdigest(),
            "transport": "sse",
            "http_idle_timeout_ms": timeout_ms,
            "compaction": settings["compaction"],
            "branch_summary": settings["branchSummary"],
            "shell_command_prefix": settings["shellCommandPrefix"],
        },
        "tooling": {
            "harness_extension_sha256": hashlib.sha256(
                extension.read_bytes()
            ).hexdigest(),
            "all_tools_execution_mode": "sequential",
            "bash_model_gateway_capability_inherited": False,
            "sieve_agent_tool_sha256": hashlib.sha256(
                (workspace_path / "sieve-agent-tool").read_bytes()
            ).hexdigest(),
            "public_tool_policy": tool_policy,
        },
        "database_snapshot": task.get("asset_database"),
        "starting_workspace_sha256": _workspace_input_root(workspace_path),
        "validator_policy": task.get("public_validation_policy"),
        "tool_transcript": {
            "database_event_schema": DATABASE_EVENT_SCHEMA_VERSION,
            "database_transcript_hash_chained_host_side": True,
            "pi_event_projection_schema": "sieve_pi_tool_transcript_v1",
            "pi_event_projection_hash_chained_host_side": True,
            "records_complete_public_results_when_process_stream_is_complete": True,
            "records_credentials_headers_endpoints_or_hidden_reasoning": False,
        },
    }
    if dict(launch_record) != expected_record:
        raise PiHarnessError("existing launch record differs from frozen inputs")
    verify_prepared_episode_material(
        {
            "launch_record": launch_record,
            "models_path": str(models_path),
            "settings_path": str(settings_path),
            "harness_extension_path": str(extension),
        }
    )
    return {
        "schema_version": "sieve_existing_pi_launch_verification_v1",
        "valid": True,
        "runtime_content_root_sha256": runtime["content_root_sha256"],
        "model_profile_id": model.model_profile_id,
        "experiment_id": experiment_name,
    }


def prepare_route_preflight(config: PiEpisodeConfig) -> dict[str, Any]:
    """Build a real pinned-Pi tool round trip using production serialization."""

    todo = _real_file(config.workspace / "TODO.md", "preflight TODO")
    if todo.read_text(encoding="utf-8") != ROUTE_PREFLIGHT_TASK_PROMPT:
        raise PiHarnessError("route preflight task prompt differs")
    material = prepare_episode(config)
    fixture_path = config.workspace / "preflight_fixture.txt"
    descriptor = os.open(
        fixture_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o400,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(ROUTE_PREFLIGHT_FIXTURE.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    fixture_path.chmod(0o400)

    command = list(material["command"])
    try:
        prompt_index = command.index("--system-prompt") + 1
    except (ValueError, IndexError) as exc:
        raise PiHarnessError("Pi preflight command lacks the system prompt") from exc
    command[prompt_index] = ROUTE_PREFLIGHT_SYSTEM_PROMPT
    workspace = config.workspace.expanduser().absolute().resolve(strict=True)
    binding = pi_system_prompt_binding(workspace, ROUTE_PREFLIGHT_SYSTEM_PROMPT)
    effective_hash = next(iter(binding))
    record = {
        "schema_version": "sieve_pi_route_preflight_material_v1",
        "harness_id": material["launch_record"]["harness_id"],
        "runtime": material["launch_record"]["runtime"],
        "model": material["launch_record"]["model"],
        "pi_settings": material["launch_record"]["pi_settings"],
        "tooling": material["launch_record"]["tooling"],
        "retry_ownership": material["launch_record"]["retry_ownership"],
        "prompts": {
            "source_system_prompt_sha256": hashlib.sha256(
                ROUTE_PREFLIGHT_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "effective_pi_system_prompt_sha256": effective_hash,
            "provider_visible_system_prompt_sha256": hashlib.sha256(
                ROUTE_PREFLIGHT_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "task_prompt_sha256": hashlib.sha256(
                ROUTE_PREFLIGHT_TASK_PROMPT.encode("utf-8")
            ).hexdigest(),
            "fixture_sha256": hashlib.sha256(
                ROUTE_PREFLIGHT_FIXTURE.encode("utf-8")
            ).hexdigest(),
            "pi_cwd_suffix_forwarded_to_provider": False,
        },
        "production_shape": {
            "same_pinned_runtime": True,
            "same_models_json": True,
            "same_settings_json": True,
            "same_extension": True,
            "same_four_tool_schemas": True,
            "same_api_protocol": True,
            "same_reasoning_configuration": True,
        },
    }
    return {
        **material,
        "command": command,
        "stdin_text": ROUTE_PREFLIGHT_TASK_PROMPT,
        "allowed_system_prompt_sha256s": (effective_hash,),
        "system_prompt_rewrites": binding,
        "preflight_record": record,
    }


def _workspace_input_root(workspace: Path) -> str:
    entries: dict[str, str] = {}
    for name in (
        "TODO.md",
        "database-interface.json",
        "floorplan.json",
        "room_program.json",
        "sieve-agent-tool",
        "submission.schema.json",
        "task.json",
    ):
        path = _real_file(workspace / name, f"workspace input {name}")
        entries[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def expected_pi_system_prompt_sha256(workspace: str | Path) -> str:
    """Hash the exact custom prompt Pi 0.85.0 sends for this workspace."""

    candidate = Path(workspace).expanduser().absolute()
    if not candidate.is_dir() or candidate.is_symlink():
        raise PiHarnessError("episode workspace must be a real directory")
    resolved = candidate.resolve(strict=True)
    source = _real_file(SYSTEM_PROMPT, "Pi system prompt").read_text(
        encoding="utf-8"
    )
    return hashlib.sha256(
        _effective_pi_system_prompt(source, resolved).encode("utf-8")
    ).hexdigest()


def pi_system_prompt_binding(
    workspace: str | Path, source_prompt: str
) -> dict[str, str]:
    """Bind Pi's exact cwd-suffixed prompt to fixed provider-visible bytes."""

    candidate = Path(workspace).expanduser().absolute()
    if not candidate.is_dir() or candidate.is_symlink():
        raise PiHarnessError("Pi prompt workspace must be a real directory")
    if not isinstance(source_prompt, str) or not source_prompt or "\x00" in source_prompt:
        raise PiHarnessError("Pi source system prompt is malformed")
    effective = _effective_pi_system_prompt(
        source_prompt,
        candidate.resolve(strict=True),
    )
    return {
        hashlib.sha256(effective.encode("utf-8")).hexdigest(): source_prompt,
    }


def allowed_pi_system_prompt_sha256s(workspace: str | Path) -> tuple[str, str]:
    """Exact prompts emitted by pinned Pi for task and compaction requests."""

    return (
        expected_pi_system_prompt_sha256(workspace),
        hashlib.sha256(PI_SUMMARIZATION_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
    )


def system_prompt_rewrite_map(workspace: str | Path) -> dict[str, str]:
    """Map verified Pi-local prompts to fixed model-visible prompt bytes.

    Pi appends the absolute cwd to a custom system prompt. Episode paths embed
    model/run identity, so forwarding that suffix would break cross-model
    prompt equality and disclose the experimental arm. The gateway validates
    Pi's exact local prompt hash before applying this canonical rewrite.
    """

    candidate = Path(workspace).expanduser().absolute()
    if not candidate.is_dir() or candidate.is_symlink():
        raise PiHarnessError("episode workspace must be a real directory")
    resolved = candidate.resolve(strict=True)
    source = _real_file(SYSTEM_PROMPT, "Pi system prompt").read_text(
        encoding="utf-8"
    )
    effective = _effective_pi_system_prompt(source, resolved)
    return {
        hashlib.sha256(effective.encode("utf-8")).hexdigest(): source,
        hashlib.sha256(
            PI_SUMMARIZATION_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(): PI_SUMMARIZATION_SYSTEM_PROMPT,
    }


def _effective_pi_system_prompt(source: str, workspace: Path) -> str:
    # Frozen Pi 0.85.0 buildSystemPrompt() appends exactly this suffix when a
    # custom prompt is supplied and context files/skills/append prompt are off.
    prompt_cwd = str(workspace).replace("\\", "/")
    return f"{source}\nCurrent working directory: {prompt_cwd}\n"


def _assert_pristine_workspace(workspace: Path) -> None:
    expected_files = {
        "TODO.md",
        "database-interface.json",
        "floorplan.json",
        "room_program.json",
        "sieve-agent-tool",
        "submission.schema.json",
        "task.json",
    }
    expected_entries = expected_files | {".home", ".tmp"}
    observed = {path.name for path in workspace.iterdir()}
    if observed != expected_entries:
        raise PiHarnessError("episode workspace is not pristine")
    for name in expected_files:
        _real_file(workspace / name, f"workspace input {name}")
    for name in (".home", ".tmp"):
        path = workspace / name
        if not path.is_dir() or path.is_symlink() or any(path.iterdir()):
            raise PiHarnessError(f"episode {name} must be a real empty directory")


def _gateway_base_url(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise PiHarnessError("gateway must be an uncredentialed loopback HTTP URL")
    path = parsed.path.rstrip("/")
    if path not in {"", "/v1"}:
        raise PiHarnessError("gateway base URL path must be empty or /v1")
    return f"http://127.0.0.1:{parsed.port}/v1"


def _model_id(value: str) -> str:
    if not isinstance(value, str) or MODEL_ID.fullmatch(value) is None:
        raise PiHarnessError("wire model is not a portable explicit identity")
    return value


def _portable_id(value: str, label: str) -> str:
    if not isinstance(value, str) or PORTABLE_ID.fullmatch(value) is None:
        raise PiHarnessError(f"{label} is not a portable identity")
    return value


def _sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise PiHarnessError(f"{label} must be lowercase SHA-256")
    return value


def _choice(value: str, choices: frozenset[str], label: str) -> str:
    if value not in choices:
        raise PiHarnessError(f"unsupported {label}")
    return value


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PiHarnessError(f"{label} must be a positive integer")
    return value


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _real_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise PiHarnessError(f"{label} is missing or linked")
    return path.resolve()


def _require_file_mode(path: Path, expected: int, label: str) -> Path:
    resolved = _real_file(path, label)
    observed = resolved.stat().st_mode & 0o777
    if observed != expected:
        raise PiHarnessError(
            f"{label} mode differs: expected {oct(expected)}, got {oct(observed)}"
        )
    return resolved


def _read_strict_json_file(path: Path, label: str) -> dict[str, Any]:
    resolved = _real_file(path, label)
    try:
        value = json.loads(
            resolved.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise PiHarnessError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PiHarnessError(f"{label} root must be an object")
    return value


def _write_json_exclusive(path: Path, value: dict[str, Any], *, mode: int) -> None:
    if path.exists() or path.is_symlink():
        raise PiHarnessError("refusing to overwrite Pi episode configuration")
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)


def _runtime_tree_fingerprint(root: Path) -> dict[str, Any]:
    entries: dict[str, dict[str, Any]] = {}
    regular_file_count = 0
    symlink_count = 0
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        if relative == RUNTIME_MANIFEST_RELATIVE.as_posix():
            continue
        if path.is_symlink():
            target = os.readlink(path)
            try:
                (path.parent / target).resolve(strict=True).relative_to(root)
            except (FileNotFoundError, ValueError) as exc:
                raise PiHarnessError(
                    f"Pi runtime link escapes or is broken: {relative}"
                ) from exc
            entries[relative] = {"target": target, "type": "symlink"}
            symlink_count += 1
        elif path.is_file():
            payload = path.read_bytes()
            entries[relative] = {
                "executable": bool(path.stat().st_mode & stat.S_IXUSR),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
                "type": "file",
            }
            regular_file_count += 1
        elif not path.is_dir():
            raise PiHarnessError(f"unsupported Pi runtime entry: {relative}")
    canonical = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return {
        "regular_file_count": regular_file_count,
        "symlink_count": symlink_count,
        "entry_count": len(entries),
        "content_root_sha256": hashlib.sha256(canonical).hexdigest(),
    }


__all__ = [
    "PiEpisodeConfig",
    "PiHarnessError",
    "allowed_pi_system_prompt_sha256s",
    "prepare_episode",
    "prepare_route_preflight",
    "pi_system_prompt_binding",
    "ROUTE_PREFLIGHT_FIXTURE",
    "ROUTE_PREFLIGHT_SYSTEM_PROMPT",
    "ROUTE_PREFLIGHT_TASK_PROMPT",
    "system_prompt_rewrite_map",
    "verify_existing_episode_launch",
    "verify_prepared_episode_material",
    "expected_pi_system_prompt_sha256",
    "verify_runtime",
]
