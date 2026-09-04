"""Build, but never launch, the pinned Codex CLI arena command."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence
from urllib.parse import urlsplit


REASONING = frozenset({"minimal", "low", "medium", "high", "xhigh"})
MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")


class CodexAdapterError(ValueError):
    pass


def build_codex_command(
    *,
    executable: str | Path,
    workspace: str | Path,
    model_id: str,
    reasoning_effort: str,
    gateway_base_url: str,
) -> list[str]:
    binary = Path(executable).expanduser().resolve(strict=True)
    root = Path(workspace).expanduser().resolve(strict=True)
    if not binary.is_file():
        raise CodexAdapterError("Codex executable is not a file")
    if not root.is_dir() or root.is_symlink():
        raise CodexAdapterError("Agent workspace must be a real directory")
    if not MODEL_ID.fullmatch(model_id):
        raise CodexAdapterError("model_id is not portable")
    if reasoning_effort not in REASONING:
        raise CodexAdapterError("unsupported Codex reasoning effort")
    parsed = urlsplit(gateway_base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise CodexAdapterError("Codex gateway must be an exact loopback HTTP origin")
    origin = f"http://127.0.0.1:{parsed.port}"
    overrides: Sequence[str] = (
        'approval_policy="never"',
        'sandbox_mode="workspace-write"',
        "sandbox_workspace_write.network_access=true",
        'web_search="disabled"',
        "analytics.enabled=false",
        "feedback.enabled=false",
        "allow_login_shell=false",
        'history.persistence="none"',
        'model_provider="arena_proxy"',
        'model_providers.arena_proxy.name="SIEVE scoped Responses proxy"',
        f'model_providers.arena_proxy.base_url="{origin}"',
        'model_providers.arena_proxy.env_key="ARENA_MODEL_GATEWAY_TOKEN"',
        'model_providers.arena_proxy.wire_api="responses"',
        "model_providers.arena_proxy.request_max_retries=0",
        "model_providers.arena_proxy.stream_max_retries=0",
        f'model_reasoning_effort="{reasoning_effort}"',
    )
    command = [
        str(binary),
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "--cd",
        str(root),
        "--model",
        model_id,
        "--json",
    ]
    for override in overrides:
        command.extend(["--config", override])
    command.append("-")
    return command


__all__ = ["CodexAdapterError", "build_codex_command"]
