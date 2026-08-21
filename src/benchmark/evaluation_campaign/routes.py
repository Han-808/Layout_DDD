"""Deployment binding resolution for API1 direct and API2 owned proxy routes."""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Protocol
import urllib.error
import urllib.request
from urllib.parse import urlparse

from benchmark.evaluation_campaign.config import (
    CampaignConfigError,
    JudgeProfile,
    LocalBinding,
    validate_judge_profile,
)


@dataclass(frozen=True)
class ResolvedJudgeRoute:
    profile: JudgeProfile
    deployment_kind: str
    binding_fingerprint_sha256: str
    route_fingerprint_sha256: str
    endpoint: str = field(repr=False)
    api_key_env: str = field(repr=False)
    secret_environment: Mapping[str, str] = field(repr=False)
    adapter_attestation_sha256: str = "0" * 64
    excluded_environment_names: tuple[str, ...] = field(
        default_factory=tuple, repr=False
    )

    @property
    def min_request_interval_seconds(self) -> float:
        value = self.profile.wire_policy.get("min_request_interval_seconds", 0.0)
        return float(value)

    def public_manifest(self) -> dict[str, Any]:
        return {
            "profile": self.profile.public_dict(),
            "deployment_kind": self.deployment_kind,
            "binding_configured": True,
            "binding_fingerprint_sha256": self.binding_fingerprint_sha256,
            "route_fingerprint_sha256": self.route_fingerprint_sha256,
            "adapter_attestation_sha256": self.adapter_attestation_sha256,
            "credential_configured": True,
        }

    def evaluator_environment(
        self, base: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        """Build the evaluator child environment without upstream proxy secrets."""

        result = _filtered_base_environment(base or os.environ)
        for name in self.excluded_environment_names:
            result.pop(name, None)
        result.update(self.secret_environment)
        result.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "JUDGE_ENDPOINT": self.endpoint,
                "JUDGE_MODEL": self.profile.model_alias,
                "JUDGE_API_KEY_ENV": self.api_key_env,
            }
        )
        interval = self.min_request_interval_seconds
        if interval > 0.0:
            result["JUDGE_MIN_REQUEST_INTERVAL_SECONDS"] = _number_text(interval)
        else:
            result.pop("JUDGE_MIN_REQUEST_INTERVAL_SECONDS", None)
        return result


class JudgeRouteSession(AbstractContextManager[ResolvedJudgeRoute]):
    """Context manager surface shared by direct and managed deployments."""

    def __enter__(self) -> ResolvedJudgeRoute:
        raise NotImplementedError

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        raise NotImplementedError


class OwnedProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def kill(self) -> None: ...


ProcessFactory = Callable[..., OwnedProcess]
ModelProbe = Callable[[str, str, str], None]
ReadinessProbe = Callable[[str, str, str], bool]
PortProbe = Callable[[str, int], bool]


def open_judge_route(
    profile: JudgeProfile,
    binding: LocalBinding,
    *,
    repo_root: Path,
    environ: Mapping[str, str] | None = None,
    process_factory: ProcessFactory = subprocess.Popen,
    model_probe: ModelProbe | None = None,
    readiness_probe: ReadinessProbe | None = None,
    port_probe: PortProbe | None = None,
    sleep: Callable[[float], None] = time.sleep,
    ownership_root: Path | None = None,
) -> JudgeRouteSession:
    validate_judge_profile(profile)
    if binding.binding_id != profile.binding_id or binding.adapter != profile.adapter:
        raise CampaignConfigError("judge profile and local binding do not match")
    env = dict(os.environ if environ is None else environ)
    if profile.adapter == "openai_compatible_direct_v1":
        return _DirectSession(
            profile,
            binding,
            environ=env,
            model_probe=model_probe or _default_model_probe,
        )
    return _ManagedProxySession(
        profile,
        binding,
        repo_root=repo_root,
        environ=env,
        process_factory=process_factory,
        readiness_probe=readiness_probe or _default_readiness_probe,
        port_probe=port_probe or _default_port_probe,
        sleep=sleep,
        ownership_root=ownership_root,
    )


class _DirectSession(JudgeRouteSession):
    def __init__(
        self,
        profile: JudgeProfile,
        binding: LocalBinding,
        *,
        environ: Mapping[str, str],
        model_probe: ModelProbe,
    ) -> None:
        self.profile = profile
        self.binding = binding
        self.environ = dict(environ)
        self.model_probe = model_probe
        self._resolved: ResolvedJudgeRoute | None = None

    def __enter__(self) -> ResolvedJudgeRoute:
        endpoint = _validate_direct_endpoint(str(self.binding.values["endpoint"]))
        credential_env = str(self.binding.values["credential_env"])
        credential = str(self.environ.get(credential_env) or "").strip()
        if not credential:
            raise CampaignConfigError("local direct binding credential is unavailable")
        if len(credential) > 8192 or any(
            ord(character) < 32 or ord(character) == 127
            for character in credential
        ):
            raise CampaignConfigError("local direct binding credential is not header-safe")
        self.model_probe(endpoint, self.profile.model_alias, credential)
        binding_fingerprint = _binding_fingerprint(self.binding)
        attestation_sha256 = _json_sha256(
            {"adapter": "direct", "profile": self.profile.fingerprint_sha256}
        )
        route_fingerprint = _route_fingerprint(
            self.profile, binding_fingerprint, attestation_sha256
        )
        self._resolved = ResolvedJudgeRoute(
            profile=self.profile,
            deployment_kind="remote_https_or_loopback_direct",
            binding_fingerprint_sha256=binding_fingerprint,
            route_fingerprint_sha256=route_fingerprint,
            endpoint=endpoint,
            api_key_env=credential_env,
            secret_environment={credential_env: credential},
            excluded_environment_names=(),
            adapter_attestation_sha256=attestation_sha256,
        )
        return self._resolved

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._resolved = None


class _ManagedProxySession(JudgeRouteSession):
    def __init__(
        self,
        profile: JudgeProfile,
        binding: LocalBinding,
        *,
        repo_root: Path,
        environ: Mapping[str, str],
        process_factory: ProcessFactory,
        readiness_probe: ReadinessProbe,
        port_probe: PortProbe,
        sleep: Callable[[float], None],
        ownership_root: Path | None,
    ) -> None:
        self.profile = profile
        self.binding = binding
        self.repo_root = repo_root.expanduser().resolve()
        self.environ = dict(environ)
        self.process_factory = process_factory
        self.readiness_probe = readiness_probe
        self.port_probe = port_probe
        self.sleep = sleep
        self._process: OwnedProcess | None = None
        self._ownership_root = (
            ownership_root.expanduser().resolve()
            if ownership_root is not None
            else Path(os.environ.get("TMPDIR") or "/tmp")
            / "layoutddd-evaluation-campaign-proxy"
        )
        self._lease_path = self._ownership_root / f"{profile.binding_id}.lease.json"
        self._ownership_token: str | None = None

    def __enter__(self) -> ResolvedJudgeRoute:
        try:
            return self._start()
        except BaseException:
            self._stop_owned_process()
            raise

    def _start(self) -> ResolvedJudgeRoute:
        values = self.binding.values
        launcher = _trusted_repo_file(
            self.repo_root,
            values["launcher_path"],
            expected_sha256=str(values["launcher_sha256"]),
            executable=False,
        )
        expected_launcher = (
            self.repo_root
            / "src/benchmark/evaluation_campaign/owned_proxy_launcher.py"
        ).resolve()
        if launcher != expected_launcher:
            raise CampaignConfigError(
                "managed proxy must use the reviewed owned_proxy_launcher.py contract"
            )
        adapter_profile = _trusted_repo_file(
            self.repo_root,
            values["adapter_profile_path"],
            expected_sha256=str(values["adapter_profile_sha256"]),
            executable=False,
        )
        adapter_attestation = _attest_adapter_profile(adapter_profile, self.profile)
        adapter_attestation_sha256 = _json_sha256(adapter_attestation)
        port = int(values["local_port"])
        if not 1 <= port <= 65535:
            raise CampaignConfigError("managed proxy local_port is invalid")
        host = "127.0.0.1"
        if self.port_probe(host, port):
            raise CampaignConfigError(
                f"managed proxy port is occupied; refusing replacement: {host}:{port}"
            )
        upstream_mapping = values["upstream_environment"]
        proxy_environment = _filtered_base_environment(self.environ)
        for target_name, source_name in upstream_mapping.items():
            secret_value = str(self.environ.get(str(source_name)) or "").strip()
            if not secret_value:
                raise CampaignConfigError(
                    f"managed proxy upstream environment is unavailable: {source_name}"
                )
            proxy_environment[str(target_name)] = secret_value
        master_env = str(values["local_master_key_env"])
        port_env = str(values["local_port_env"])
        master_key = secrets.token_hex(32)
        proxy_environment[master_env] = master_key
        proxy_environment[port_env] = str(port)
        proxy_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        self._assert_no_live_lease()
        ownership_token = secrets.token_hex(24)
        self._ownership_token = ownership_token
        child_argv = [
            sys.executable,
            str(launcher),
            "--config",
            str(adapter_profile),
            "--host",
            host,
            "--port",
            str(port),
            "--ownership-token",
            ownership_token,
        ]
        self._process = self.process_factory(
            child_argv,
            cwd=str(self.repo_root),
            env=proxy_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._write_lease(child_argv)
        endpoint = f"http://{host}:{port}/v1"
        timeout = float(values["startup_timeout_seconds"])
        deadline = time.monotonic() + timeout
        while True:
            if self._process.poll() is not None:
                self._stop_owned_process()
                raise CampaignConfigError("owned API2 proxy exited during startup")
            if self.readiness_probe(
                endpoint, master_key, self.profile.model_alias
            ):
                break
            if time.monotonic() >= deadline:
                self._stop_owned_process()
                raise CampaignConfigError("owned API2 proxy readiness timed out")
            self.sleep(min(1.0, max(0.0, deadline - time.monotonic())))

        binding_fingerprint = _binding_fingerprint(self.binding)
        route_fingerprint = _route_fingerprint(
            self.profile, binding_fingerprint, adapter_attestation_sha256
        )
        return ResolvedJudgeRoute(
            profile=self.profile,
            deployment_kind="owned_loopback_proxy",
            binding_fingerprint_sha256=binding_fingerprint,
            route_fingerprint_sha256=route_fingerprint,
            endpoint=endpoint,
            api_key_env=master_env,
            secret_environment={master_env: master_key},
            excluded_environment_names=tuple(
                dict.fromkeys(
                    [
                        *(str(key) for key in upstream_mapping),
                        *(str(name) for name in upstream_mapping.values()),
                    ]
                )
            ),
            adapter_attestation_sha256=adapter_attestation_sha256,
        )

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop_owned_process()

    def _stop_owned_process(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            self._remove_owned_lease()
            return
        process.terminate()
        try:
            process.wait(timeout=10.0)
        except (subprocess.TimeoutExpired, TimeoutError):
            process.kill()
            process.wait(timeout=10.0)
        self._remove_owned_lease()

    def _assert_no_live_lease(self) -> None:
        if not self._lease_path.is_file():
            return
        try:
            row = json.loads(self._lease_path.read_text(encoding="utf-8"))
            if set(row) != {
                "schema_version",
                "pid",
                "ownership_token",
                "argv",
                "argv_sha256",
                "launcher_sha256",
            } or row.get("schema_version") != "owned_evaluation_proxy_lease_v1":
                raise ValueError("lease schema mismatch")
            pid = int(row["pid"])
            token = str(row["ownership_token"])
            launcher_sha = str(row["launcher_sha256"])
            argv = row["argv"]
            if (
                not isinstance(argv, list)
                or _json_sha256(argv) != row["argv_sha256"]
            ):
                raise ValueError("lease argv identity mismatch")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise CampaignConfigError("owned proxy lease is invalid; manual review required") from exc
        command = _process_command(pid)
        command_identity = all(str(item) in command for item in argv[1:])
        if (
            command
            and command_identity
            and token in command
            and launcher_sha == str(self.binding.values["launcher_sha256"])
        ):
            raise CampaignConfigError("an attested owned proxy lease is still active")
        if command:
            raise CampaignConfigError(
                "proxy lease PID identity differs; refusing PID-reuse/orphan cleanup"
            )
        self._lease_path.unlink(missing_ok=True)

    def _write_lease(self, argv: list[str]) -> None:
        process = self._process
        if process is None or self._ownership_token is None:
            raise RuntimeError("owned process is unavailable")
        self._ownership_root.mkdir(parents=True, exist_ok=True)
        temporary = self._lease_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema_version": "owned_evaluation_proxy_lease_v1",
                    "pid": process.pid,
                    "ownership_token": self._ownership_token,
                    "argv": argv,
                    "argv_sha256": _json_sha256(argv),
                    "launcher_sha256": self.binding.values["launcher_sha256"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._lease_path)

    def _remove_owned_lease(self) -> None:
        if not self._lease_path.is_file():
            return
        try:
            row = json.loads(self._lease_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if row.get("ownership_token") == self._ownership_token:
            self._lease_path.unlink(missing_ok=True)


def _filtered_base_environment(value: Mapping[str, str]) -> dict[str, str]:
    """Use a small OS/runtime allowlist; bindings add only explicit secrets."""

    allowed = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NO_PROXY",
        "BLENDER_BIN",
    }
    return {str(key): str(child) for key, child in value.items() if str(key) in allowed}


def _validate_direct_endpoint(value: str) -> str:
    endpoint = value.strip().rstrip("/")
    parsed = urlparse(endpoint)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise CampaignConfigError("direct endpoint may not contain auth, query, or fragment")
    if parsed.scheme == "https" and parsed.hostname:
        return endpoint
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        return endpoint
    raise CampaignConfigError("direct endpoint must be HTTPS or loopback HTTP")


def _trusted_repo_file(
    repo_root: Path,
    value: Any,
    *,
    expected_sha256: str,
    executable: bool,
) -> Path:
    relative = Path(str(value))
    if relative.is_absolute() or ".." in relative.parts:
        raise CampaignConfigError("managed proxy path must be repository-relative")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise CampaignConfigError("managed proxy path escapes repository") from exc
    if not path.is_file() or path.is_symlink():
        raise CampaignConfigError(f"trusted managed proxy file is unavailable: {relative}")
    if _file_sha256(path) != expected_sha256:
        raise CampaignConfigError(f"trusted managed proxy file hash mismatch: {relative}")
    if executable and not os.access(path, os.X_OK):
        raise CampaignConfigError(f"managed proxy launcher is not executable: {relative}")
    return path


def _binding_fingerprint(binding: LocalBinding) -> str:
    # This digest commits to local deployment details without publishing them.
    return _json_sha256(
        {
            "binding_id": binding.binding_id,
            "adapter": binding.adapter,
            "deployment": dict(binding.values),
        }
    )


def _route_fingerprint(
    profile: JudgeProfile,
    binding_sha256: str,
    adapter_attestation_sha256: str,
) -> str:
    return _json_sha256(
        {
            "profile": profile.public_dict(),
            "binding_fingerprint_sha256": binding_sha256,
            "adapter_attestation_sha256": adapter_attestation_sha256,
        }
    )


def _attest_adapter_profile(path: Path, profile: JudgeProfile) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise CampaignConfigError("PyYAML is required for adapter attestation") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise CampaignConfigError("managed adapter profile cannot be parsed") from exc
    if not isinstance(value, dict) or not isinstance(value.get("model_list"), list):
        raise CampaignConfigError("managed adapter profile has no model_list")
    expected = profile.adapter_attestation
    if expected is None:
        raise CampaignConfigError("managed Judge profile has no adapter attestation")
    matches = [
        row
        for row in value["model_list"]
        if isinstance(row, dict) and row.get("model_name") == expected["model_name"]
    ]
    if len(matches) != 1:
        raise CampaignConfigError("managed adapter must expose exactly one attested alias")
    row = matches[0]
    params = row.get("litellm_params")
    model_info = row.get("model_info")
    settings = value.get("litellm_settings")
    if not isinstance(params, dict):
        raise CampaignConfigError("managed adapter model has no litellm_params")
    if not isinstance(model_info, dict) or not isinstance(settings, dict):
        raise CampaignConfigError("managed adapter model/settings attestation is incomplete")
    additional_drop_params = params.get("additional_drop_params")
    if not isinstance(additional_drop_params, list):
        additional_drop_params = []
    actual = {
        "schema_version": "litellm_model_entry_v1",
        "model_name": row.get("model_name"),
        "provider_model": params.get("model"),
        "base_model": model_info.get("base_model"),
        "reasoning_effort": params.get("reasoning_effort"),
        "additional_drop_params": sorted(str(item) for item in additional_drop_params),
        "drop_params": settings.get("drop_params"),
        "num_retries": settings.get("num_retries"),
        "request_timeout_seconds": settings.get("request_timeout"),
    }
    if actual != dict(expected):
        raise CampaignConfigError("managed adapter model/effort attestation mismatch")
    return {
        **actual,
        "adapter_profile_sha256": _file_sha256(path),
    }


def _process_command(pid: int) -> str:
    if pid < 1:
        return ""
    completed = subprocess.run(
        ["ps", "-o", "command=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _default_model_probe(endpoint: str, model_alias: str, credential: str) -> None:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {credential}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise CampaignConfigError("direct route model discovery failed") from exc
    rows = value.get("data") if isinstance(value, dict) else None
    aliases = {
        str(row.get("id"))
        for row in rows or []
        if isinstance(row, dict) and row.get("id") is not None
    }
    if model_alias not in aliases:
        raise CampaignConfigError("configured model alias is unavailable")


def _default_readiness_probe(
    endpoint: str, master_key: str, model_alias: str
) -> bool:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {master_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            if not 200 <= int(response.status) < 300:
                return False
            value = json.loads(response.read().decode("utf-8"))
            rows = value.get("data") if isinstance(value, dict) else None
            return model_alias in {
                str(row.get("id"))
                for row in rows or []
                if isinstance(row, dict) and row.get("id") is not None
            }
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return False


def _default_port_probe(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _number_text(value: float) -> str:
    if float(value).is_integer():
        return f"{value:.1f}"
    return format(value, ".15g")
