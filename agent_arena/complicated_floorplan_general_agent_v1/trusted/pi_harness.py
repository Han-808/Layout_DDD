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
from typing import Any
from urllib.parse import urlsplit


ARENA_ROOT = Path(__file__).resolve().parents[1]
HARNESS_ROOT = Path(__file__).resolve().parent / "pi_harness"
HARNESS_CONTRACT = HARNESS_ROOT / "harness.json"
SYSTEM_PROMPT = HARNESS_ROOT / "SYSTEM.md"
PI_VERSION = "0.85.0"
PROVIDER_ID = "sieve-gateway"
SUPPORTED_APIS = frozenset({"openai-completions", "openai-responses"})
THINKING_LEVELS = frozenset({"off", "minimal", "low", "medium", "high"})
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
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
    wire_model: str
    api: str
    thinking: str
    context_window: int
    max_tokens: int


def verify_runtime(runtime_root: str | Path) -> dict[str, Any]:
    root = Path(runtime_root).expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise PiHarnessError("Pi runtime root must be a real directory")
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "sieve_pi_runtime_manifest_v1":
        raise PiHarnessError("Pi runtime manifest schema differs")
    if manifest.get("platform") != "darwin-arm64":
        raise PiHarnessError("Pi runtime platform differs")
    if manifest.get("pi_version") != PI_VERSION:
        raise PiHarnessError("Pi runtime manifest version differs")
    observed = _runtime_tree_fingerprint(root)
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
    }


def prepare_episode(config: PiEpisodeConfig) -> dict[str, Any]:
    runtime = verify_runtime(config.runtime_root)
    workspace = config.workspace.expanduser().resolve(strict=True)
    if not workspace.is_dir() or workspace.is_symlink():
        raise PiHarnessError("episode workspace must be a real directory")
    todo = _real_file(workspace / "TODO.md", "episode TODO")
    api = _choice(config.api, SUPPORTED_APIS, "Pi API protocol")
    thinking = _choice(config.thinking, THINKING_LEVELS, "thinking level")
    wire_model = _model_id(config.wire_model)
    context_window = _positive_int(config.context_window, "context window")
    max_tokens = _positive_int(config.max_tokens, "max tokens")
    if max_tokens > context_window:
        raise PiHarnessError("max tokens cannot exceed the context window")
    base_url = _gateway_base_url(config.gateway_base_url)

    agent_dir = workspace / ".home/.pi/agent"
    agent_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    models_path = agent_dir / "models.json"
    models = {
        "providers": {
            PROVIDER_ID: {
                "api": api,
                "apiKey": "$ARENA_MODEL_GATEWAY_TOKEN",
                "authHeader": True,
                "baseUrl": base_url,
                "models": [
                    {
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
                        "name": wire_model,
                        "reasoning": thinking != "off",
                    }
                ],
                "name": "SIEVE Scoped Model Gateway",
            }
        }
    }
    _write_json_exclusive(models_path, models, mode=0o400)

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
        SYSTEM_PROMPT.read_text(encoding="utf-8"),
    ]
    return {
        "schema_version": "sieve_pi_episode_launch_material_v1",
        "command": command,
        "stdin_text": todo.read_text(encoding="utf-8"),
        "models_path": str(models_path),
        "provider_id": PROVIDER_ID,
        "wire_model": wire_model,
        "api": api,
        "thinking": thinking,
        "runtime": runtime,
    }


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


def _choice(value: str, choices: frozenset[str], label: str) -> str:
    if value not in choices:
        raise PiHarnessError(f"unsupported {label}")
    return value


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PiHarnessError(f"{label} must be a positive integer")
    return value


def _real_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise PiHarnessError(f"{label} is missing or linked")
    return path.resolve()


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
    "prepare_episode",
    "verify_runtime",
]
