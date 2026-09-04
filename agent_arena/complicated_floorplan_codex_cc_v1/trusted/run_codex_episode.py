#!/usr/bin/env python3
"""Run one pinned Codex episode through the isolated arena and shared DB."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from arena import ArenaError, create_episode, sha256_file, write_json_exclusive
from database_host import EpisodeDatabase, collect_and_normalize
from isolated_exec import run_isolated
from model_gateway import ScopedModelGateway
from adapters.codex import build_codex_command


ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--codex-executable", required=True)
    parser.add_argument("--codex-runtime-root", required=True)
    parser.add_argument("--expected-codex-version", required=True)
    parser.add_argument("--expected-codex-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--reasoning-effort",
        required=True,
        choices=("minimal", "low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--resource-bindings", required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--upstream-key-env", required=True)
    parser.add_argument("--max-model-requests", required=True, type=int)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()

    if not ENV_NAME.fullmatch(args.upstream_key_env):
        raise ArenaError("upstream-key-env must be an environment-variable name")
    upstream_secret = os.environ.get(args.upstream_key_env, "")
    if not upstream_secret:
        raise ArenaError("trusted upstream credential is unavailable")
    if not SHA256.fullmatch(args.expected_codex_sha256):
        raise ArenaError("expected Codex SHA-256 is invalid")
    executable = Path(args.codex_executable).expanduser().resolve(strict=True)
    runtime_root = Path(args.codex_runtime_root).expanduser().resolve(strict=True)
    if sha256_file(executable) != args.expected_codex_sha256:
        raise ArenaError("Codex executable hash differs from the registered identity")
    version = _codex_version(executable)
    if version != args.expected_codex_version:
        raise ArenaError("Codex CLI version differs from the registered identity")

    episode = create_episode(
        agent_id=args.agent_id,
        scene_id=args.scene_id,
        run_id=args.run_id,
    )
    identity = {
        "schema_version": "sieve_coding_agent_identity_v1",
        "participant_class": "general_purpose_coding_agent",
        "agent_id": args.agent_id,
        "implementation": "codex_cli",
        "implementation_version": version,
        "implementation_sha256": args.expected_codex_sha256,
        "model_id": args.model_id,
        "reasoning_effort": args.reasoning_effort,
        "runtime_root_sha256_recorded": False,
        "host_auth_file_mounted": False,
        "host_credential_passed_to_agent": False,
        "model_gateway": "scoped_loopback_responses_proxy_v1",
        "upstream_base_url_sha256": hashlib.sha256(
            args.upstream_base_url.encode("utf-8")
        ).hexdigest(),
        "max_model_requests": args.max_model_requests,
        "timeout_seconds": args.timeout_seconds,
    }
    try:
        with EpisodeDatabase(
            episode=episode,
            resource_bindings=args.resource_bindings,
        ) as database_service:
            with ScopedModelGateway(
                upstream_base_url=args.upstream_base_url,
                upstream_secret=upstream_secret,
                fixed_model=args.model_id,
                endpoint="/responses",
                max_requests=args.max_model_requests,
            ) as gateway:
                command = build_codex_command(
                    executable=executable,
                    workspace=episode.workspace,
                    model_id=args.model_id,
                    reasoning_effort=args.reasoning_effort,
                    gateway_base_url=gateway.base_url,
                )
                identity["command_sha256"] = hashlib.sha256(
                    json.dumps(command, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
                write_json_exclusive(
                    episode.host / "agent_identity.json", identity, mode=0o444
                )
                todo = (episode.workspace / "TODO.md").read_text(encoding="utf-8")
                process = run_isolated(
                    workspace=episode.workspace,
                    runtime_root=runtime_root,
                    command=command,
                    tool_socket=database_service.socket_path,
                    tool_token=database_service.capability_token,
                    stdout_path=episode.host / "agent.stdout.jsonl",
                    stderr_path=episode.host / "agent.stderr.log",
                    stdin_text=todo,
                    timeout_seconds=args.timeout_seconds,
                    model_gateway=gateway.endpoint_address,
                    model_gateway_token=gateway.capability_token,
                    extra_environment={
                        "ARENA_AGENT_ID": args.agent_id,
                        "ARENA_MODEL_ID": args.model_id,
                        "ARENA_RUN_ID": args.run_id,
                    },
                )
                process_payload = process.public_dict()
                process_payload["model_request_count"] = gateway.request_count
                write_json_exclusive(
                    episode.host / "process_result.json", process_payload, mode=0o444
                )
                if process.returncode != 0 or process.timed_out:
                    return _fail(
                        episode.host,
                        reason=process.status,
                        process=process_payload,
                    )
                if not (episode.workspace / "final_submission.json").is_file():
                    return _fail(
                        episode.host,
                        reason="sealed_submission_missing",
                        process=process_payload,
                    )
                summary = collect_and_normalize(episode, database_service.database)
    finally:
        upstream_secret = ""
    print(
        json.dumps(
            {
                "status": "complete",
                "episode_root": str(episode.root),
                "agent_workspace": str(episode.workspace),
                "official_evaluation_connected": False,
                "summary": summary,
            },
            sort_keys=True,
        )
    )
    return 0


def _codex_version(executable: Path) -> str:
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": "/var/empty",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
    }
    result = subprocess.run(
        [str(executable), "--version"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    value = result.stdout.strip()
    if result.returncode != 0 or not value or "\n" in value:
        raise ArenaError("cannot establish Codex CLI version")
    return value


def _fail(host: Path, *, reason: str, process: dict[str, Any]) -> int:
    value = {
        "schema_version": "sieve_isolated_agent_episode_summary_v1",
        "status": "failed",
        "reason": reason,
        "process": process,
        "official_evaluation_connected": False,
    }
    write_json_exclusive(host / "summary.json", value, mode=0o444)
    print(json.dumps(value, sort_keys=True))
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "credentials_exposed": False,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(2)
