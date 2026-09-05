"""Thin, auditable execution boundary for external generation harnesses."""

from __future__ import annotations

import contextlib
import glob
import hashlib
import inspect
import io
import json
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.scene_io.validate import ArtifactValidationError
from benchmark.utils.io import read_json, write_json


EXECUTION_RESULT_SCHEMA_VERSION = "external_harness_execution_v1"
NATIVE_ARTIFACT_MANIFEST_SCHEMA_VERSION = "native_artifact_manifest_v1"
ENV_TOKEN = re.compile(r"^\{env:([A-Za-z_][A-Za-z0-9_]*)\}$")
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
URL_TOKEN = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)
SECRET_KEY = re.compile(
    r"(?:api[_-]?key|password|secret|token|credential)",
    re.IGNORECASE,
)
PRIVATE_LOCATOR_KEY = re.compile(r"(?:endpoint|base[_-]?url|^url$)", re.IGNORECASE)
SAFE_TOKEN_COUNT_KEYS = {
    "tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
}


class ExternalExecutionError(ArtifactValidationError):
    """Raised when an upstream harness cannot produce its expected artifact."""


def execute_external_harness(
    *,
    adapter_name: str,
    method_input_path: Path,
    native_input_path: Path,
    out_dir: Path,
    config: Mapping[str, Any],
    default_native_artifact: str | None = None,
    default_native_artifact_glob: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Execute one callback, configured artifact, or shell-free subprocess."""

    root = Path(out_dir)
    execution_dir = root / "execution"
    upstream_output_dir = root / "upstream_output"
    execution_dir.mkdir(parents=True, exist_ok=True)
    upstream_output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = execution_dir / "stdout.txt"
    stderr_path = execution_dir / "stderr.txt"
    result_path = execution_dir / "execution_result.json"
    runner_config_path = execution_dir / "runner_config.json"
    execution_config = _execution_config(config)
    runner = config.get("runner")
    raw_output = config.get("raw_output_path") or config.get(
        f"{adapter_name}_output_path"
    )
    command_configured = execution_config.get("command") is not None
    configured_modes = sum(
        (
            runner is not None,
            raw_output is not None,
            command_configured,
        )
    )
    if configured_modes != 1:
        raise ExternalExecutionError(
            f"{adapter_name} execution requires exactly one of config.runner, "
            "config.raw_output_path, or config.execution.command"
        )

    runner_kind = (
        "callback"
        if runner is not None
        else "configured_native_artifact"
        if raw_output is not None
        else "subprocess"
    )
    write_json(
        runner_config_path,
        {
            "schema_version": EXECUTION_RESULT_SCHEMA_VERSION,
            "adapter": adapter_name,
            "runner_kind": runner_kind,
            "runner": _callable_name(runner) if runner is not None else None,
            "execution": _sanitize_execution_config(execution_config),
            "configured_native_artifact": (
                str(raw_output) if raw_output is not None else None
            ),
        },
    )

    started_wall = _utc_now()
    started_monotonic = time.monotonic()
    stdout = ""
    stderr = ""
    return_code: int | None = None
    timed_out = False
    cancelled = False
    command_for_audit: list[str] | None = None
    cwd: Path | None = None
    repo_path: Path | None = None
    upstream_commit: str | None = None
    upstream_repo_clean_before: bool | None = None
    upstream_repo_clean_after: bool | None = None
    runner_provenance: dict[str, Any] = {"status": "NOT_DISCOVERED"}
    callback_metadata: dict[str, Any] = {}
    source_artifact: Path | None = None
    error: BaseException | None = None

    try:
        repo_path = _resolve_repo_path(execution_config, required=command_configured)
        upstream_commit = _discover_git_commit(repo_path)
        upstream_repo_clean_before = _git_repo_clean(repo_path)
        if runner is not None:
            if not callable(runner):
                raise ExternalExecutionError("config.runner must be callable")
            try:
                source_file = inspect.getsourcefile(runner)
            except TypeError:
                source_file = None
            runner_provenance = _runner_source_provenance(
                Path(source_file) if source_file else None,
                entrypoint=_callable_name(runner),
            )
            _verify_expected_execution_identity(
                execution_config,
                upstream_commit=upstream_commit,
                upstream_repo_clean=upstream_repo_clean_before,
                runner_provenance=runner_provenance,
            )
            captured_stdout = io.StringIO()
            captured_stderr = io.StringIO()
            try:
                with contextlib.redirect_stdout(
                    captured_stdout
                ), contextlib.redirect_stderr(captured_stderr):
                    callback_result = runner(
                        method_input_path=Path(method_input_path),
                        out_dir=root,
                        config=config,
                    )
            finally:
                stdout = captured_stdout.getvalue()
                stderr = captured_stderr.getvalue()
            source_artifact, callback_metadata = _callback_artifact(callback_result)
            if not source_artifact.is_absolute():
                source_artifact = root / source_artifact
            return_code = int(callback_metadata.get("return_code", 0))
            if return_code != 0:
                raise ExternalExecutionError(
                    f"{adapter_name} callback returned nonzero code {return_code}"
                )
        elif raw_output is not None:
            runner_provenance = {"status": "NOT_EXECUTED"}
            source_artifact = _resolve_configured_path(
                raw_output,
                base=Path.cwd(),
                label="configured native artifact",
            )
            return_code = 0
        else:
            command, command_for_audit, cwd, environment, entrypoint = _subprocess_request(
                adapter_name=adapter_name,
                method_input_path=Path(method_input_path),
                native_input_path=Path(native_input_path),
                upstream_output_dir=upstream_output_dir,
                repo_path=repo_path,
                execution_config=execution_config,
            )
            runner_provenance = _runner_source_provenance(
                entrypoint,
                entrypoint=entrypoint.as_posix() if entrypoint else None,
            )
            _verify_expected_execution_identity(
                execution_config,
                upstream_commit=upstream_commit,
                upstream_repo_clean=upstream_repo_clean_before,
                runner_provenance=runner_provenance,
            )
            timeout_seconds = _timeout_seconds(execution_config)
            try:
                completed = _run_external_process(
                    command,
                    cwd=cwd,
                    environment=environment,
                    timeout_seconds=timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout = _process_text(exc.stdout)
                stderr = _process_text(exc.stderr)
                raise ExternalExecutionError(
                    f"{adapter_name} upstream process timed out after "
                    f"{timeout_seconds:g} seconds"
                ) from exc
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            return_code = int(completed.returncode)
            if return_code != 0:
                raise ExternalExecutionError(
                    f"{adapter_name} upstream process exited with code {return_code}"
                )
            source_artifact = _resolve_expected_artifact(
                execution_config,
                upstream_output_dir=upstream_output_dir,
                variables=_template_variables(
                    adapter_name=adapter_name,
                    method_input_path=Path(method_input_path),
                    native_input_path=Path(native_input_path),
                    upstream_output_dir=upstream_output_dir,
                    repo_path=repo_path,
                    python_executable=_python_executable(execution_config),
                    execution_config=execution_config,
                ),
                default_native_artifact=default_native_artifact,
                default_native_artifact_glob=default_native_artifact_glob,
            )
    except (KeyboardInterrupt, SystemExit) as exc:
        # Cancellation is an operator stop, never an ordinary generator failure.
        cancelled = True
        stdout = _process_text(getattr(exc, "upstream_stdout", stdout))
        stderr = _process_text(getattr(exc, "upstream_stderr", stderr))
        return_code = getattr(exc, "upstream_return_code", return_code)
        error = exc
    except Exception as exc:
        error = exc

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    _finish_runner_source_provenance(runner_provenance)
    upstream_repo_clean_after = _git_repo_clean(repo_path)
    if error is None:
        try:
            _verify_expected_execution_identity(
                execution_config,
                upstream_commit=upstream_commit,
                upstream_repo_clean=upstream_repo_clean_after,
                runner_provenance=runner_provenance,
                require_unchanged=True,
            )
        except Exception as exc:
            error = exc
    runtime_seconds = time.monotonic() - started_monotonic
    base_result: dict[str, Any] = {
        "schema_version": EXECUTION_RESULT_SCHEMA_VERSION,
        "adapter": adapter_name,
        "runner_kind": runner_kind,
        "method_input_path": Path(method_input_path).resolve().as_posix(),
        "native_input_path": Path(native_input_path).resolve().as_posix(),
        "runner_config_path": runner_config_path.resolve().as_posix(),
        "command": command_for_audit,
        "cwd": cwd.resolve().as_posix() if cwd is not None else None,
        "environment_overrides": _sanitize(
            execution_config.get("environment") or {}
        ),
        "return_code": return_code,
        "stdout_path": stdout_path.resolve().as_posix(),
        "stderr_path": stderr_path.resolve().as_posix(),
        "started_at": started_wall,
        "ended_at": _utc_now(),
        "runtime_seconds": runtime_seconds,
        "timed_out": timed_out,
        "cancelled": cancelled,
        "upstream_repo": repo_path.as_posix() if repo_path is not None else None,
        "upstream_commit": upstream_commit,
        "upstream_repo_clean_before_execution": upstream_repo_clean_before,
        "upstream_repo_clean_after_execution": upstream_repo_clean_after,
        "pinned_upstream_identity_enforced": bool(
            execution_config.get("expected_upstream_commit")
        ),
        "runner_provenance": runner_provenance,
        "upstream_output_dir": upstream_output_dir.resolve().as_posix(),
        "source_native_artifact_path": (
            source_artifact.resolve().as_posix()
            if source_artifact is not None and source_artifact.exists()
            else str(source_artifact) if source_artifact is not None else None
        ),
        "callback_metadata": _sanitize(callback_metadata),
    }
    if error is not None:
        base_result["status"] = "cancelled" if cancelled else "failed"
        base_result["error"] = {
            "type": type(error).__name__,
            "message": redact_private_locators(str(error)),
        }
        write_json(result_path, base_result)
        if cancelled:
            raise error
        if isinstance(error, ExternalExecutionError):
            raise error
        raise ExternalExecutionError(
            f"{adapter_name} external execution failed: {error}"
        ) from error

    assert source_artifact is not None
    try:
        preserved, primary_manifest = preserve_native_artifact(
            source_artifact,
            root / "native_artifacts" / "primary",
        )
        auxiliary = _preserve_auxiliary_artifacts(
            execution_config,
            root=root,
            upstream_output_dir=upstream_output_dir,
            variables=_template_variables(
                adapter_name=adapter_name,
                method_input_path=Path(method_input_path),
                native_input_path=Path(native_input_path),
                upstream_output_dir=upstream_output_dir,
                repo_path=repo_path,
                python_executable=_python_executable(execution_config),
                execution_config=execution_config,
            ),
        )
    except Exception as exc:
        base_result["status"] = "failed"
        base_result["error"] = {
            "type": type(exc).__name__,
            "message": redact_private_locators(str(exc)),
        }
        write_json(result_path, base_result)
        if isinstance(exc, ExternalExecutionError):
            raise
        raise ExternalExecutionError(
            f"{adapter_name} native artifact preservation failed: {exc}"
        ) from exc

    native_manifest_path = write_json(
        execution_dir / "native_artifact_manifest.json",
        primary_manifest,
    )
    base_result.update(
        {
            "status": "completed",
            "preserved_native_artifact_path": preserved.resolve().as_posix(),
            "native_artifact_path": preserved.resolve().as_posix(),
            "native_artifact_kind": primary_manifest["artifact_kind"],
            "native_artifact_sha256": primary_manifest["sha256"],
            "native_artifact_hash_algorithm": "sha256",
            "native_artifact_hash": {
                "algorithm": "sha256",
                "value": primary_manifest["sha256"],
            },
            "native_artifact_manifest_path": native_manifest_path.resolve().as_posix(),
            "preserved_auxiliary_artifacts": auxiliary,
            "native_artifact_verified_after_conversion": False,
            "auxiliary_artifacts_verified_after_conversion": False,
        }
    )
    write_json(result_path, base_result)
    base_result["execution_result_path"] = result_path.resolve().as_posix()
    return preserved, base_result


def preserve_supplied_native_artifact(
    *,
    adapter_name: str,
    source_path: Path,
    method_input_path: Path,
    native_input_path: Path | None,
    out_dir: Path,
    config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Snapshot Mode-A input before the converter sees it."""

    root = Path(out_dir)
    execution_dir = root / "execution"
    execution_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = execution_dir / "stdout.txt"
    stderr_path = execution_dir / "stderr.txt"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    runner_config_path = write_json(
        execution_dir / "runner_config.json",
        {
            "schema_version": EXECUTION_RESULT_SCHEMA_VERSION,
            "adapter": adapter_name,
            "runner_kind": "offline_supplied_artifact",
            "execution": _sanitize(_execution_config(config)),
            "conversion_sidecars": _sanitize(
                {
                    key: config[key]
                    for key in (
                        "asset_manifest_path",
                        "asset_ids_path",
                        "asset_bindings_path",
                        "scene_config_path",
                    )
                    if config.get(key)
                }
            ),
        },
    )
    started = _utc_now()
    started_monotonic = time.monotonic()
    preserved, manifest = preserve_native_artifact(
        Path(source_path),
        root / "native_artifacts" / "primary",
    )
    auxiliary = _preserve_offline_auxiliary_artifacts(
        config,
        source_path=Path(source_path).expanduser().resolve(),
        root=root,
    )
    manifest_path = write_json(
        execution_dir / "native_artifact_manifest.json",
        manifest,
    )
    execution_config = _execution_config(config)
    repo = _resolve_repo_path(execution_config, required=False)
    result = {
        "schema_version": EXECUTION_RESULT_SCHEMA_VERSION,
        "adapter": adapter_name,
        "runner_kind": "offline_supplied_artifact",
        "status": "completed",
        "method_input_path": Path(method_input_path).resolve().as_posix(),
        "native_input_path": (
            Path(native_input_path).resolve().as_posix()
            if native_input_path is not None
            else None
        ),
        "runner_config_path": runner_config_path.resolve().as_posix(),
        "command": None,
        "cwd": None,
        "environment_overrides": {},
        "return_code": 0,
        "stdout_path": stdout_path.resolve().as_posix(),
        "stderr_path": stderr_path.resolve().as_posix(),
        "started_at": started,
        "ended_at": _utc_now(),
        "runtime_seconds": time.monotonic() - started_monotonic,
        "timed_out": False,
        "upstream_repo": repo.as_posix() if repo is not None else None,
        "upstream_commit": _discover_git_commit(repo),
        "runner_provenance": {"status": "NOT_EXECUTED"},
        "source_native_artifact_path": Path(source_path).resolve().as_posix(),
        "preserved_native_artifact_path": preserved.resolve().as_posix(),
        "native_artifact_path": preserved.resolve().as_posix(),
        "native_artifact_kind": manifest["artifact_kind"],
        "native_artifact_sha256": manifest["sha256"],
        "native_artifact_hash_algorithm": "sha256",
        "native_artifact_hash": {
            "algorithm": "sha256",
            "value": manifest["sha256"],
        },
        "native_artifact_manifest_path": manifest_path.resolve().as_posix(),
        "preserved_auxiliary_artifacts": auxiliary,
        "native_artifact_verified_after_conversion": False,
        "auxiliary_artifacts_verified_after_conversion": False,
    }
    result_path = write_json(execution_dir / "execution_result.json", result)
    result["execution_result_path"] = result_path.resolve().as_posix()
    return preserved, result


def preserve_native_artifact(
    source_path: Path,
    destination_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Copy one immutable audit artifact and verify byte/manifest identity."""

    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise ExternalExecutionError(f"expected native artifact is missing: {source}")
    if not source.is_file() and not source.is_dir():
        raise ExternalExecutionError(
            f"native artifact must be a regular file or directory: {source}"
        )
    destination_root = Path(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / source.name
    if destination.exists():
        raise ExternalExecutionError(
            f"refusing to overwrite preserved native artifact: {destination}"
        )
    if source.is_dir() and _is_relative_to(destination.resolve(), source):
        raise ExternalExecutionError(
            "native artifact destination cannot be inside the source directory"
        )
    source_digest, source_entries = artifact_sha256(source)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
        artifact_kind = "directory"
    else:
        shutil.copy2(source, destination)
        artifact_kind = "file"
    preserved_digest, preserved_entries = artifact_sha256(destination)
    if source_digest != preserved_digest or source_entries != preserved_entries:
        raise ExternalExecutionError(
            "preserved native artifact does not match the upstream source"
        )
    manifest = {
        "schema_version": NATIVE_ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "artifact_kind": artifact_kind,
        "source_path": source.as_posix(),
        "preserved_path": destination.resolve().as_posix(),
        "sha256": preserved_digest,
        "entries": preserved_entries,
    }
    return destination, manifest


def artifact_sha256(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash a file directly or a directory through a canonical entry manifest."""

    artifact = Path(path)
    if artifact.is_file():
        digest = _file_sha256(artifact)
        entries = [
            {
                "path": artifact.name,
                "type": "file",
                "bytes": artifact.stat().st_size,
                "sha256": digest,
            }
        ]
        return digest, entries
    if not artifact.is_dir():
        raise ExternalExecutionError(f"artifact does not exist: {artifact}")
    entries: list[dict[str, Any]] = []
    for candidate in sorted(artifact.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(artifact).as_posix()
        if candidate.is_symlink():
            target = os.readlink(candidate)
            entries.append(
                {
                    "path": relative,
                    "type": "symlink",
                    "target": target,
                    "sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
                }
            )
        elif candidate.is_dir():
            entries.append({"path": relative, "type": "directory"})
        elif candidate.is_file():
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": candidate.stat().st_size,
                    "sha256": _file_sha256(candidate),
                }
            )
        else:
            raise ExternalExecutionError(
                f"native artifact contains unsupported filesystem entry: {candidate}"
            )
    payload = json.dumps(
        entries,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), entries


def verify_preserved_native_artifact(
    metadata: dict[str, Any],
    *,
    canonical_scene_path: Path | None = None,
) -> None:
    """Fail if conversion changed the preserved native artifact."""

    path_value = metadata.get("preserved_native_artifact_path")
    expected = metadata.get("native_artifact_sha256")
    if not path_value or not expected:
        raise ExternalExecutionError(
            "execution metadata lacks preserved native artifact identity"
        )
    actual, _ = artifact_sha256(Path(str(path_value)))
    if actual != expected:
        raise ExternalExecutionError(
            "preserved native artifact changed during canonical conversion"
        )
    auxiliary = metadata.get("preserved_auxiliary_artifacts") or {}
    if not isinstance(auxiliary, Mapping):
        raise ExternalExecutionError(
            "execution metadata auxiliary artifact inventory is invalid"
        )
    for name, item in auxiliary.items():
        if not isinstance(item, Mapping) or not item.get("path") or not item.get(
            "sha256"
        ):
            raise ExternalExecutionError(
                f"execution metadata for auxiliary artifact {name!r} is invalid"
            )
        auxiliary_digest, _ = artifact_sha256(Path(str(item["path"])))
        if auxiliary_digest != item["sha256"]:
            raise ExternalExecutionError(
                f"preserved auxiliary artifact {name!r} changed during conversion"
            )
    metadata["native_artifact_verified_after_conversion"] = True
    metadata["auxiliary_artifacts_verified_after_conversion"] = True
    if canonical_scene_path is not None:
        metadata["canonical_scene_path"] = Path(canonical_scene_path).resolve().as_posix()
    result_path = metadata.get("execution_result_path")
    if result_path:
        persisted = read_json(Path(str(result_path)))
        if not isinstance(persisted, dict):
            raise ExternalExecutionError("execution result must be a JSON object")
        persisted.update(
            {
                "native_artifact_verified_after_conversion": True,
                "auxiliary_artifacts_verified_after_conversion": True,
                "canonical_scene_path": metadata.get("canonical_scene_path"),
            }
        )
        write_json(Path(str(result_path)), persisted)


def update_execution_result(
    metadata: dict[str, Any],
    values: Mapping[str, Any],
) -> None:
    """Update in-memory and persisted execution metadata with audit-only fields."""

    metadata.update(dict(values))
    result_path = metadata.get("execution_result_path")
    if not result_path:
        return
    persisted = read_json(Path(str(result_path)))
    if not isinstance(persisted, dict):
        raise ExternalExecutionError("execution result must be a JSON object")
    persisted.update(_sanitize(dict(values)))
    write_json(Path(str(result_path)), persisted)


def _execution_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("execution")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ExternalExecutionError("adapter_config.execution must be a JSON object")
    return dict(value)


def _subprocess_request(
    *,
    adapter_name: str,
    method_input_path: Path,
    native_input_path: Path,
    upstream_output_dir: Path,
    repo_path: Path | None,
    execution_config: Mapping[str, Any],
) -> tuple[list[str], list[str], Path, dict[str, str], Path | None]:
    assert repo_path is not None
    python_executable = _python_executable(execution_config)
    variables = _template_variables(
        adapter_name=adapter_name,
        method_input_path=method_input_path,
        native_input_path=native_input_path,
        upstream_output_dir=upstream_output_dir,
        repo_path=repo_path,
        python_executable=python_executable,
        execution_config=execution_config,
    )
    environment = _environment(execution_config, variables=variables)
    if execution_config.get("expected_upstream_commit") is not None:
        # Pinned checkouts are read-only execution inputs. Avoid normal Python
        # imports creating untracked __pycache__ entries that would obscure the
        # pre/post clean-worktree evidence.
        environment.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    command, audit_command = _expand_command(
        execution_config.get("command"),
        variables=variables,
        environment=environment,
    )
    cwd_value = execution_config.get("cwd")
    if cwd_value is None:
        cwd = repo_path
    else:
        cwd_text = _format_template(str(cwd_value), variables)
        cwd = Path(cwd_text).expanduser()
        if not cwd.is_absolute():
            cwd = repo_path / cwd
        cwd = cwd.resolve()
    if not cwd.is_dir():
        raise ExternalExecutionError(f"configured upstream cwd is missing: {cwd}")
    _require_executable(command[0], environment, cwd=cwd)
    entrypoint = _require_python_entrypoint(command, cwd=cwd)
    return command, audit_command, cwd, environment, entrypoint


def _template_variables(
    *,
    adapter_name: str,
    method_input_path: Path,
    native_input_path: Path,
    upstream_output_dir: Path,
    repo_path: Path | None,
    python_executable: str,
    execution_config: Mapping[str, Any],
) -> dict[str, str]:
    values = {
        "adapter": adapter_name,
        "method_input": method_input_path.resolve().as_posix(),
        "native_input": native_input_path.resolve().as_posix(),
        "output_dir": upstream_output_dir.resolve().as_posix(),
        "upstream_output_dir": upstream_output_dir.resolve().as_posix(),
        "repo_path": repo_path.as_posix() if repo_path is not None else "",
        "python_executable": python_executable,
    }
    extra = execution_config.get("template_variables")
    if extra is not None and not isinstance(extra, Mapping):
        raise ExternalExecutionError(
            "execution.template_variables must be a JSON object"
        )
    for key, value in dict(extra or {}).items():
        name = str(key)
        if not SAFE_NAME.fullmatch(name):
            raise ExternalExecutionError(
                f"invalid execution template variable name {name!r}"
            )
        values[name] = str(value)
    return values


def _expand_command(
    value: Any,
    *,
    variables: Mapping[str, str],
    environment: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    if isinstance(value, str):
        tokens = shlex.split(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        tokens = [str(token) for token in value]
    else:
        raise ExternalExecutionError(
            "execution.command must be a string or string list"
        )
    if not tokens:
        raise ExternalExecutionError("execution.command must not be empty")
    command: list[str] = []
    audit: list[str] = []
    for token in tokens:
        env_match = ENV_TOKEN.fullmatch(token)
        if env_match:
            name = env_match.group(1)
            if name not in environment:
                raise ExternalExecutionError(
                    f"execution command requires missing environment variable {name}"
                )
            command.append(str(environment[name]))
            audit.append(f"<redacted-env:{name}>")
            continue
        expanded = _format_template(token, variables)
        command.append(expanded)
        audit.append(expanded)
    return command, _redact_command(audit)


def _format_template(value: str, variables: Mapping[str, str]) -> str:
    try:
        return value.format_map(dict(variables))
    except KeyError as exc:
        raise ExternalExecutionError(
            f"unknown execution template variable {exc.args[0]!r}"
        ) from exc


def _environment(
    execution_config: Mapping[str, Any],
    *,
    variables: Mapping[str, str],
) -> dict[str, str]:
    inherit = execution_config.get("inherit_environment", True)
    if not isinstance(inherit, bool):
        raise ExternalExecutionError("execution.inherit_environment must be boolean")
    environment = dict(os.environ) if inherit else {}
    overrides = execution_config.get("environment") or {}
    if not isinstance(overrides, Mapping):
        raise ExternalExecutionError("execution.environment must be a JSON object")
    for key, value in overrides.items():
        name = str(key)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ExternalExecutionError(f"invalid environment variable name {name!r}")
        environment[name] = _format_template(str(value), variables)
    return environment


def _python_executable(execution_config: Mapping[str, Any]) -> str:
    value = str(execution_config.get("python_executable") or sys.executable)
    candidate = shutil.which(value)
    if candidate is None:
        path = Path(value).expanduser()
        if not path.is_file():
            raise ExternalExecutionError(f"configured Python executable is missing: {value}")
        candidate = path.resolve().as_posix()
    return candidate


def _require_executable(
    value: str, environment: Mapping[str, str], *, cwd: Path
) -> None:
    if os.sep in value or (os.altsep and os.altsep in value):
        path = Path(value)
        if not path.is_absolute():
            path = cwd / path
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ExternalExecutionError(f"configured executable is missing: {value}")
        return
    if shutil.which(value, path=environment.get("PATH")) is None:
        raise ExternalExecutionError(f"configured executable is missing: {value}")


def _require_python_entrypoint(command: Sequence[str], *, cwd: Path) -> Path | None:
    # Inspect only the script forms we can identify without importing upstream
    # modules or executing an interpreter. Unknown/module forms stay unaudited.
    if command[0].endswith(".py"):
        value = command[0]
    else:
        candidates = iter(command[1:])
        value = next(candidates, "")
        while value in {"-u", "-B", "-E", "-I", "-s", "-S", "-O", "-OO", "-q"}:
            value = next(candidates, "")
        if not value.endswith(".py"):
            return None
    entrypoint = Path(value).expanduser()
    if not entrypoint.is_absolute():
        entrypoint = cwd / entrypoint
    entrypoint = entrypoint.resolve()
    if not entrypoint.is_file():
        raise ExternalExecutionError(
            f"configured Python entrypoint is missing: {entrypoint}"
        )
    return entrypoint


def _runner_source_provenance(
    path: Path | None, *, entrypoint: str | None
) -> dict[str, Any]:
    """Fingerprint the real shim/callback, not a user assertion about its version.

    This is source identity only, not proof that declared comparison controls
    are implemented. Do not import upstream modules to discover their source.
    """

    result: dict[str, Any] = {
        "status": "NOT_DISCOVERED",
        "entrypoint": entrypoint,
        "scope": "entrypoint_file_only_not_transitive_dependencies",
        "control_verification": "NOT_VERIFIED",
    }
    if path is None or not path.is_file():
        return result
    source = path.resolve()
    result.update(
        {
            "status": "SOURCE_FINGERPRINTED",
            "source_path": source.as_posix(),
            "source_sha256": _file_sha256(source),
            "source_git_commit": _discover_git_commit(source.parent),
            "source_git_tracked": None,
            "source_git_modified": None,
            **bridge_bundle_identity(source),
        }
    )
    if result["source_git_commit"] is not None:
        try:
            tracked = subprocess.run(
                [
                    "git", "-C", source.parent.as_posix(), "ls-files",
                    "--error-unmatch", "--", source.name,
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if tracked.returncode in {0, 1}:
                result["source_git_tracked"] = tracked.returncode == 0
            if tracked.returncode == 0:
                changed = subprocess.run(
                    [
                        "git", "-C", source.parent.as_posix(), "diff",
                        "--quiet", "HEAD", "--", source.name,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if changed.returncode in {0, 1}:
                    result["source_git_modified"] = changed.returncode == 1
        except (OSError, subprocess.TimeoutExpired):
            pass  # Unavailable Git evidence remains null, never a clean claim.
    return result


def _verify_expected_execution_identity(
    execution_config: Mapping[str, Any],
    *,
    upstream_commit: str | None,
    upstream_repo_clean: bool | None,
    runner_provenance: Mapping[str, Any],
    require_unchanged: bool = False,
) -> None:
    expected_commit = execution_config.get("expected_upstream_commit")
    if expected_commit is not None and str(expected_commit) != str(upstream_commit):
        raise ExternalExecutionError(
            "configured upstream checkout does not match expected_upstream_commit"
        )
    if expected_commit is not None and upstream_repo_clean is not True:
        raise ExternalExecutionError(
            "configured pinned upstream checkout is not a clean Git worktree"
        )
    expected_source = execution_config.get("expected_entrypoint_sha256")
    if expected_source is not None:
        expected_source = str(expected_source).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_source):
            raise ExternalExecutionError(
                "execution.expected_entrypoint_sha256 must be lowercase SHA-256"
            )
        if runner_provenance.get("source_sha256") != expected_source:
            raise ExternalExecutionError(
                "configured runner entrypoint differs from expected_entrypoint_sha256"
            )
        if require_unchanged and runner_provenance.get(
            "source_unchanged_during_execution"
        ) is not True:
            raise ExternalExecutionError(
                "configured runner entrypoint changed during external execution"
            )
    expected_bundle = execution_config.get("expected_bridge_bundle_sha256")
    if expected_bundle is not None:
        expected_bundle = str(expected_bundle).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected_bundle):
            raise ExternalExecutionError(
                "execution.expected_bridge_bundle_sha256 must be lowercase SHA-256"
            )
        if runner_provenance.get("bridge_bundle_sha256") != expected_bundle:
            raise ExternalExecutionError(
                "configured bridge bundle differs from expected_bridge_bundle_sha256"
            )
        if require_unchanged and runner_provenance.get(
            "bridge_bundle_unchanged_during_execution"
        ) is not True:
            raise ExternalExecutionError(
                "configured bridge bundle changed during external execution"
            )


def _finish_runner_source_provenance(provenance: dict[str, Any]) -> None:
    value = provenance.get("source_path")
    if value is None:
        return
    path = Path(value)
    try:
        digest = _file_sha256(path) if path.is_file() else None
    except OSError as exc:
        # Do not hide the actual upstream result/failure if it made its source
        # unreadable. Preserve the unavailable audit evidence explicitly.
        provenance["source_sha256_after_execution"] = None
        provenance["source_unchanged_during_execution"] = None
        provenance["source_verification_error"] = type(exc).__name__
        return
    provenance["source_sha256_after_execution"] = digest
    provenance["source_unchanged_during_execution"] = (
        digest == provenance["source_sha256"]
    )
    try:
        current_bundle = bridge_bundle_identity(path)
    except OSError as exc:
        provenance["bridge_bundle_sha256_after_execution"] = None
        provenance["bridge_bundle_unchanged_during_execution"] = None
        provenance["bridge_bundle_verification_error"] = type(exc).__name__
        return
    provenance["bridge_bundle_sha256_after_execution"] = current_bundle.get(
        "bridge_bundle_sha256"
    )
    provenance["bridge_bundle_unchanged_during_execution"] = (
        current_bundle.get("bridge_bundle_sha256")
        == provenance.get("bridge_bundle_sha256")
    )


def bridge_bundle_identity(entrypoint: str | Path) -> dict[str, Any]:
    """Hash the bridge entrypoint and its sibling shared helper, if present."""

    source = Path(entrypoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"bridge entrypoint is missing: {source}")
    paths = [source]
    common = source.with_name("_common.py")
    if common.is_file() and common not in paths:
        paths.append(common)
    entries = [
        {"name": path.name, "sha256": _file_sha256(path)}
        for path in sorted(paths, key=lambda item: item.name)
    ]
    payload = json.dumps(
        entries,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "bridge_bundle_scope": "entrypoint_plus_sibling_common_if_present",
        "bridge_bundle_files": entries,
        "bridge_bundle_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _resolve_repo_path(
    execution_config: Mapping[str, Any],
    *,
    required: bool,
) -> Path | None:
    value = execution_config.get("repo_path")
    if value is None:
        if required:
            raise ExternalExecutionError("execution.repo_path is required")
        return None
    path = Path(str(value)).expanduser().resolve()
    if not path.is_dir():
        raise ExternalExecutionError(f"configured upstream repo is missing: {path}")
    return path


def _discover_git_commit(repo_path: Path | None) -> str | None:
    if repo_path is None:
        return None
    try:
        completed = subprocess.run(
            ["git", "-C", repo_path.as_posix(), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value if re.fullmatch(r"[0-9a-fA-F]{40}", value) else None


def _git_repo_clean(repo_path: Path | None) -> bool | None:
    if repo_path is None:
        return None
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                repo_path.as_posix(),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return not completed.stdout.strip()


def _resolve_expected_artifact(
    execution_config: Mapping[str, Any],
    *,
    upstream_output_dir: Path,
    variables: Mapping[str, str],
    default_native_artifact: str | None,
    default_native_artifact_glob: str | None,
) -> Path:
    path_value = execution_config.get("native_artifact")
    glob_value = execution_config.get("native_artifact_glob")
    if path_value is None and glob_value is None:
        path_value = default_native_artifact
        glob_value = default_native_artifact_glob
    if path_value is not None and glob_value is not None:
        raise ExternalExecutionError(
            "configure only one of native_artifact or native_artifact_glob"
        )
    if path_value is not None:
        return _resolve_configured_path(
            _format_template(str(path_value), variables),
            base=upstream_output_dir,
            label="expected native artifact",
        )
    if glob_value is None:
        raise ExternalExecutionError(
            "execution must declare native_artifact or native_artifact_glob"
        )
    pattern = _format_template(str(glob_value), variables)
    if not Path(pattern).is_absolute():
        pattern = (upstream_output_dir / pattern).as_posix()
    matches = sorted(Path(path) for path in glob.glob(pattern))
    if not matches:
        raise ExternalExecutionError(
            f"expected native artifact glob matched nothing: {pattern}"
        )
    if len(matches) != 1:
        raise ExternalExecutionError(
            f"expected native artifact glob is ambiguous ({len(matches)} matches): "
            f"{pattern}"
        )
    return matches[0].resolve()


def _resolve_configured_path(value: Any, *, base: Path, label: str) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.exists():
        raise ExternalExecutionError(f"{label} is missing: {path}")
    return path


def _preserve_auxiliary_artifacts(
    execution_config: Mapping[str, Any],
    *,
    root: Path,
    upstream_output_dir: Path,
    variables: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    configured = execution_config.get("auxiliary_artifacts") or {}
    if not isinstance(configured, Mapping):
        raise ExternalExecutionError(
            "execution.auxiliary_artifacts must be a JSON object"
        )
    result: dict[str, dict[str, Any]] = {}
    for name_value, path_value in configured.items():
        name = str(name_value)
        if not SAFE_NAME.fullmatch(name):
            raise ExternalExecutionError(f"invalid auxiliary artifact name {name!r}")
        source = _resolve_configured_path(
            _format_template(str(path_value), variables),
            base=upstream_output_dir,
            label=f"auxiliary artifact {name!r}",
        )
        preserved, manifest = preserve_native_artifact(
            source,
            root / "native_artifacts" / "auxiliary" / name,
        )
        result[name] = {
            "path": preserved.resolve().as_posix(),
            "sha256": manifest["sha256"],
            "kind": manifest["artifact_kind"],
        }
    return result


def _preserve_offline_auxiliary_artifacts(
    config: Mapping[str, Any],
    *,
    source_path: Path,
    root: Path,
) -> dict[str, dict[str, Any]]:
    execution_config = _execution_config(config)
    configured = execution_config.get("auxiliary_artifacts") or {}
    if not isinstance(configured, Mapping):
        raise ExternalExecutionError(
            "execution.auxiliary_artifacts must be a JSON object"
        )
    paths = {str(name): value for name, value in configured.items()}
    for config_key, name in (
        ("asset_manifest_path", "asset_manifest"),
        ("asset_ids_path", "asset_ids"),
        ("asset_bindings_path", "asset_bindings"),
        ("scene_config_path", "scene_config"),
    ):
        if config.get(config_key) and name not in paths:
            paths[name] = config[config_key]
    base = source_path if source_path.is_dir() else source_path.parent
    result: dict[str, dict[str, Any]] = {}
    for name, value in paths.items():
        if not SAFE_NAME.fullmatch(name):
            raise ExternalExecutionError(f"invalid auxiliary artifact name {name!r}")
        source = _resolve_configured_path(
            value,
            base=base,
            label=f"offline auxiliary artifact {name!r}",
        )
        preserved, manifest = preserve_native_artifact(
            source,
            root / "native_artifacts" / "auxiliary" / name,
        )
        result[name] = {
            "path": preserved.resolve().as_posix(),
            "sha256": manifest["sha256"],
            "kind": manifest["artifact_kind"],
        }
    return result


def _callback_artifact(value: Any) -> tuple[Path, dict[str, Any]]:
    if isinstance(value, (str, Path)):
        return Path(value), {}
    if not isinstance(value, Mapping):
        raise ExternalExecutionError(
            "config.runner must return a path or execution-result mapping"
        )
    path_value = value.get("native_artifact_path") or value.get("native_output_path")
    if path_value is None:
        raise ExternalExecutionError(
            "runner result mapping requires native_artifact_path"
        )
    return Path(str(path_value)), dict(value)


def _timeout_seconds(execution_config: Mapping[str, Any]) -> float:
    value = execution_config.get("timeout_seconds", 3600.0)
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ExternalExecutionError("execution.timeout_seconds must be numeric") from exc
    if timeout <= 0.0:
        raise ExternalExecutionError("execution.timeout_seconds must be positive")
    return timeout


def _run_external_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    """Run one harness command and reap its process tree on timeout or cancel.

    The real bridges can launch Blender or other worker processes. Killing only
    the immediate bridge would leave those workers running after the benchmark
    has already recorded a timeout. On POSIX, a fresh session gives the command
    its own process group, which is terminated as a unit.
    """

    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _terminate_external_process(process)
        raise subprocess.TimeoutExpired(
            list(command),
            timeout_seconds,
            output=stdout,
            stderr=stderr,
        ) from exc
    except BaseException as exc:
        # Catch only to clean up, preserving the original cancellation/control
        # exception. The caller persists these streams and re-raises it.
        stdout, stderr = _terminate_external_process(process)
        exc.upstream_stdout = stdout
        exc.upstream_stderr = stderr
        exc.upstream_return_code = process.returncode
        raise
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=int(process.returncode),
        stdout=stdout,
        stderr=stderr,
    )


def _terminate_external_process(
    process: subprocess.Popen[str],
) -> tuple[str, str]:
    """Bounded cleanup of the controlled session, including remaining workers."""

    def signal_tree(sig: int) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif process.poll() is None:
                process.kill()
        except ProcessLookupError:
            pass

    signal_tree(signal.SIGTERM)
    try:
        stdout, stderr = process.communicate(timeout=2.0)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        signal_tree(signal.SIGKILL)
        try:
            stdout, stderr = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired as exc:
            # Do not hang on an inherited pipe if a worker violated the session
            # boundary. The caller still receives the original cancellation.
            stdout, stderr = _process_text(exc.stdout), _process_text(exc.stderr)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
            process.wait(timeout=2.0)
    finally:
        # The leader may exit before descendants that have closed their pipes.
        signal_tree(signal.SIGKILL)
    return stdout or "", stderr or ""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    name = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{name}"


def _sanitize(value: Any, *, key: str = "") -> Any:
    if (
        key.casefold() in SAFE_TOKEN_COUNT_KEYS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    ):
        # Usage counts are audit data, not bearer credentials. Keep the narrow
        # numeric allowlist while all token-like strings remain redacted.
        return value
    if SECRET_KEY.search(key):
        return "<redacted>"
    if (
        PRIVATE_LOCATOR_KEY.search(key)
        and not key.casefold().endswith("_sha256")
        and isinstance(value, (str, Path))
    ):
        digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
        return f"<redacted-locator:sha256={digest}>"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if callable(value):
        return _callable_name(value)
    if isinstance(value, str):
        return redact_private_locators(value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return repr(value)


def _redact_command(command: list[str]) -> list[str]:
    result = list(command)
    for index, token in enumerate(result[:-1]):
        if SECRET_KEY.search(token):
            result[index + 1] = "<redacted>"
    for index, token in enumerate(result):
        if token.startswith(("http://", "https://")):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            result[index] = f"<redacted-locator:sha256={digest}>"
    return result


def redact_private_locators(value: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        digest = hashlib.sha256(match.group(0).encode("utf-8")).hexdigest()
        return f"<redacted-locator:sha256={digest}>"

    return URL_TOKEN.sub(replacement, str(value))


def _sanitize_execution_config(value: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize(dict(value))
    command = value.get("command")
    if isinstance(command, str):
        sanitized["command"] = _redact_command(shlex.split(command))
    elif isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        sanitized["command"] = _redact_command([str(token) for token in command])
    return sanitized


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


__all__ = [
    "EXECUTION_RESULT_SCHEMA_VERSION",
    "ExternalExecutionError",
    "artifact_sha256",
    "bridge_bundle_identity",
    "execute_external_harness",
    "preserve_native_artifact",
    "preserve_supplied_native_artifact",
    "redact_private_locators",
    "update_execution_result",
    "verify_preserved_native_artifact",
]
