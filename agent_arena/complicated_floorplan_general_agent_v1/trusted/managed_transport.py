#!/usr/bin/env python3
"""Host-owned transport adapters for routes that are not direct OpenAI HTTP.

The TokenHub Hy4 route has only been validated through one pinned LiteLLM
Anthropic adapter.  This module verifies that runtime, starts a private
loopback proxy inside the API-family lock, and gives the ordinary SIEVE model
gateway only a short-lived local proxy key.  Provider credentials never enter
Pi, an episode gateway, or an Agent workspace.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import subprocess
import tempfile
import time
from typing import Any, Iterator, Mapping
from urllib.parse import urlsplit

from api_profiles import (
    ExperimentProfile,
    ProfileRegistry,
    RouteRuntimeBinding,
    RuntimeBindings,
)
from isolated_exec import (
    IsolationError,
    _OwnedProcessTracker,
    _process_snapshot,
    _terminate_owned_processes,
    _terminate_untracked_process_group,
)
from model_gateway import ANTHROPIC_REASONING_DETAIL_FORMAT
from tokenhub_identity_relay import (
    TokenHubIdentityRelay,
    TokenHubIdentityRelayError,
)


TRUSTED_ROOT = Path(__file__).resolve().parent
TOKENHUB_MANIFEST = TRUSTED_ROOT / "adapters/tokenhub_litellm_v1.json"
DIRECT_ADAPTER = "direct_http_v1"
TOKENHUB_ADAPTER = "managed_tokenhub_litellm_v1"
LOOPBACK_HOST = "127.0.0.1"
READY_TIMEOUT_SECONDS = 60.0
LISTENER_RELEASE_TIMEOUT_SECONDS = 5.0
MIN_ADAPTER_TIMEOUT_GRACE_SECONDS = 30.0
READY_BODY_LIMIT = 1024 * 1024
PRIVATE_FILE_MODE = 0o400
SEALED_RECORD_MODE = 0o444
TREE_HASH_POLICY = "sieve_runtime_tree_sha256_v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
TOKENHUB_MODEL_PROFILE_ID = "tokenhub-hy4-preview-agent-v1"
TOKENHUB_ROUTE_PROFILE_ID = "tokenhub-chat-reasoning-agent-v1"
TOKENHUB_RETRYABLE_HTTP = [408, 409, 425, 429, 500, 502, 503, 504]
# Executable bytecode is part of the runtime identity.  In particular, never
# omit ``__pycache__``/``*.pyc``: Python may execute a header-valid cache in
# preference to source.  Only non-executable Finder metadata is ignored.
IGNORED_TREE_COMPONENTS: frozenset[str] = frozenset()
IGNORED_TREE_SUFFIXES: frozenset[str] = frozenset()


class ManagedTransportError(RuntimeError):
    """Raised when a managed adapter cannot be proven safe and exact."""


@dataclass(frozen=True)
class RuntimeTransport:
    bindings: RuntimeBindings
    credential: str
    adapter_id: str
    public_record: Mapping[str, Any]
    _manager: "_TokenHubLiteLLM | None" = None

    def ensure_healthy(self) -> None:
        if self._manager is not None:
            self._manager.ensure_healthy()

    def recycle_after_ambiguous(self) -> bool:
        """Drain a managed intermediary before any later logical unit.

        Direct routes have no host-owned intermediary to recycle.  TokenHub
        does: closing the outer episode gateway does not reliably cancel an
        in-flight request inside LiteLLM, so the trusted supervisor must kill
        and re-create that process before another preflight/episode may start.
        """

        if self._manager is not None:
            self._manager.recycle_after_ambiguous()
            return True
        return False

    def post_send_ambiguity_detected(self) -> bool:
        """Expose only the managed relay's in-process ambiguity bit."""

        if self._manager is None:
            return False
        return self._manager.post_send_ambiguity_detected()

    def classify_outer_result(self, outer_request_complete: bool) -> bool:
        """Reconcile an outer result with the managed raw-provider boundary."""

        if self._manager is None:
            return False
        return self._manager.classify_outer_result(outer_request_complete)


@contextmanager
def prepare_runtime_transport(
    *,
    registry: ProfileRegistry,
    experiment: ExperimentProfile,
    bindings: RuntimeBindings,
    provider_credential: str,
    invocation_root: Path,
    tokenhub_runtime_root: Path | None,
) -> Iterator[RuntimeTransport]:
    """Select the exact transport; unknown or mixed adapters fail closed."""

    adapters = {
        registry.route(registry.model(model_id).route_profile_id).transport_adapter_id
        for model_id in experiment.model_profile_ids
    }
    if adapters == {DIRECT_ADAPTER}:
        record = {
            "schema_version": "sieve_runtime_transport_record_v1",
            "adapter_id": DIRECT_ADAPTER,
            "managed_process_started": False,
            "provider_credential_visible_to_agent": False,
            "provider_endpoint_request_response_or_hidden_reasoning_recorded": False,
        }
        _write_sealed_json(invocation_root / "transport_start.json", record)
        try:
            yield RuntimeTransport(
                bindings=bindings,
                credential=provider_credential,
                adapter_id=DIRECT_ADAPTER,
                public_record=record,
            )
        finally:
            _write_sealed_json(
                invocation_root / "transport_completion.json",
                {
                    "schema_version": "sieve_runtime_transport_completion_v2",
                    "adapter_id": DIRECT_ADAPTER,
                    "started": False,
                    "ready": True,
                    "ended": True,
                    "listener_released": True,
                    "recycle_count": 0,
                    "provider_credential_endpoint_request_response_or_hidden_reasoning_recorded": False,
                },
            )
        return
    if adapters != {TOKENHUB_ADAPTER} or experiment.api_family_id != "tokenhub":
        raise ManagedTransportError("unsupported or mixed route transport adapters")
    if tokenhub_runtime_root is None:
        raise ManagedTransportError(
            "TokenHub requires --tokenhub-litellm-runtime with the pinned snapshot"
        )
    manager = _TokenHubLiteLLM(
        registry=registry,
        experiment=experiment,
        bindings=bindings,
        provider_credential=provider_credential,
        invocation_root=invocation_root,
        runtime_root=tokenhub_runtime_root,
    )
    try:
        manager.start()
        yield manager.transport()
    finally:
        manager.close()


def transport_contracts_for_experiment(
    registry: ProfileRegistry, experiment: ExperimentProfile
) -> list[dict[str, Any]]:
    """Return controlled, endpoint-free adapter identities for plan hashing."""

    adapters = sorted(
        {
            registry.route(
                registry.model(model_id).route_profile_id
            ).transport_adapter_id
            for model_id in experiment.model_profile_ids
        }
    )
    records: list[dict[str, Any]] = []
    for adapter in adapters:
        records.append(_transport_contract(adapter))
    return records


def transport_contract_for_route(route_adapter_id: str) -> dict[str, Any]:
    """Return one endpoint-free adapter identity for episode hashing."""

    return _transport_contract(route_adapter_id)


def verify_transport_runtime_compatibility(
    *,
    registry: ProfileRegistry,
    experiment: ExperimentProfile,
    tokenhub_runtime_root: Path | None,
    pi_runtime_root: Path | None = None,
    bindings: RuntimeBindings | None = None,
    verify_managed_lifecycle: bool = False,
) -> dict[str, Any]:
    """Run the credential-free release gate for an experiment transport."""

    adapters = {
        registry.route(registry.model(model_id).route_profile_id).transport_adapter_id
        for model_id in experiment.model_profile_ids
    }
    if adapters == {DIRECT_ADAPTER}:
        return {
            "schema_version": "sieve_transport_runtime_compatibility_v1",
            "ok": True,
            "adapter_id": DIRECT_ADAPTER,
            "managed_process_required": False,
            "production_provider_credential_or_route_used": False,
        }
    if adapters != {TOKENHUB_ADAPTER} or experiment.api_family_id != "tokenhub":
        raise ManagedTransportError("unsupported or mixed route transport adapters")
    if tokenhub_runtime_root is None:
        raise ManagedTransportError(
            "TokenHub runtime verification requires --tokenhub-litellm-runtime"
        )
    if verify_managed_lifecycle and bindings is None:
        raise ManagedTransportError(
            "TokenHub lifecycle verification requires runtime bindings"
        )
    if verify_managed_lifecycle and pi_runtime_root is None:
        raise ManagedTransportError(
            "TokenHub signed-replay verification requires the pinned Pi runtime"
        )
    manifest = _read_manifest(TOKENHUB_MANIFEST)
    _validate_tokenhub_experiment_contract(registry, experiment, manifest)
    runtime_record = verify_tokenhub_runtime(
        tokenhub_runtime_root,
        manifest=manifest,
    )
    lifecycle: dict[str, Any] | None = None
    signed_reasoning_e2e: dict[str, Any] | None = None
    if verify_managed_lifecycle:
        if bindings is None:  # defensive duplicate of the pre-hash guard
            raise ManagedTransportError(
                "TokenHub lifecycle verification requires runtime bindings"
            )
        if pi_runtime_root is None:  # defensive duplicate of the pre-hash guard
            raise ManagedTransportError(
                "TokenHub signed-replay verification requires the pinned Pi runtime"
            )
        # Lazy import avoids making the production transport depend on the
        # release fixture during ordinary paid execution.
        from tokenhub_release_gate import verify_tokenhub_pi_litellm_roundtrip

        signed_reasoning_e2e = verify_tokenhub_pi_litellm_roundtrip(
            registry=registry,
            experiment=experiment,
            bindings=bindings,
            pi_runtime_root=pi_runtime_root,
            tokenhub_runtime_root=tokenhub_runtime_root,
        )
        lifecycle = {
            "started": True,
            "authenticated_model_inventory_exact": True,
            "raw_provider_identity_verified_before_litellm": (
                signed_reasoning_e2e.get(
                    "raw_provider_identity_verified_before_litellm"
                )
                is True
            ),
            "ended": signed_reasoning_e2e.get("managed_adapter_ended") is True,
            "listener_released": signed_reasoning_e2e.get(
                "managed_adapter_listener_released"
            )
            is True,
            "provider_identity_relay_listener_released": (
                signed_reasoning_e2e.get(
                    "provider_identity_relay_listener_released"
                )
                is True
            ),
            "upstream_model_request_sent": True,
            "provider_route_kind": "loopback_fixture_only",
        }
        if not all(
            lifecycle[name]
            for name in (
                "raw_provider_identity_verified_before_litellm",
                "ended",
                "listener_released",
                "provider_identity_relay_listener_released",
            )
        ):
            raise ManagedTransportError(
                "managed adapter lifecycle release gate failed"
            )
    return {
        "schema_version": "sieve_transport_runtime_compatibility_v1",
        "ok": True,
        "adapter_id": TOKENHUB_ADAPTER,
        "managed_process_required": True,
        "adapter_manifest_sha256": _sha256_file(TOKENHUB_MANIFEST),
        "runtime": runtime_record,
        "managed_lifecycle": lifecycle,
        "signed_reasoning_e2e": signed_reasoning_e2e,
        "production_provider_credential_or_route_used": False,
    }


def _transport_contract(adapter: str) -> dict[str, Any]:
    if adapter == DIRECT_ADAPTER:
        return {
            "adapter_id": DIRECT_ADAPTER,
            "managed_process_required": False,
        }
    if adapter == TOKENHUB_ADAPTER:
        manifest = _read_manifest(TOKENHUB_MANIFEST)
        return {
            "adapter_id": TOKENHUB_ADAPTER,
            "managed_process_required": True,
            "manifest_sha256": _sha256_file(TOKENHUB_MANIFEST),
            "runtime": dict(manifest["runtime"]),
            "proxy": dict(manifest["proxy"]),
            "provider_identity_guard": dict(
                manifest["provider_identity_guard"]
            ),
            "reasoning_replay_bridge": dict(
                manifest["reasoning_replay_bridge"]
            ),
            "provider_endpoint_recorded_or_hashed": False,
        }
    raise ManagedTransportError("experiment references an unknown adapter")


class _TokenHubLiteLLM:
    def __init__(
        self,
        *,
        registry: ProfileRegistry,
        experiment: ExperimentProfile,
        bindings: RuntimeBindings,
        provider_credential: str,
        invocation_root: Path,
        runtime_root: Path,
    ) -> None:
        self.registry = registry
        self.experiment = experiment
        self.bindings = bindings
        self._provider_credential = provider_credential
        self.invocation_root = invocation_root
        self.runtime_root = runtime_root
        self.manifest = _read_manifest(TOKENHUB_MANIFEST)
        self._process: subprocess.Popen[bytes] | None = None
        self._tracker: _OwnedProcessTracker | None = None
        self._temporary_root: Path | None = None
        self._port: int | None = None
        self._proxy_key = ""
        self._runtime_record: dict[str, Any] | None = None
        self._identity_relay: TokenHubIdentityRelay | None = None
        self._started = False
        self._closed = False
        self._ready = False
        self._recycle_count = 0
        self._runtime_reverification_count = 0

    def start(self) -> None:
        if self._started or self._closed:
            raise ManagedTransportError("managed adapter lifecycle is invalid")
        self._started = True
        _validate_tokenhub_experiment_contract(
            self.registry,
            self.experiment,
            self.manifest,
        )
        self._runtime_record = verify_tokenhub_runtime(
            self.runtime_root,
            manifest=self.manifest,
        )
        provider_base = self._provider_base()
        identity_guard = _object(
            self.manifest.get("provider_identity_guard"),
            "provider identity guard",
        )
        self._identity_relay = TokenHubIdentityRelay(
            provider_base_url=provider_base,
            provider_credential=self._provider_credential,
            expected_request_model=str(identity_guard["expected_request_model"]),
            accepted_response_models=tuple(
                str(value)
                for value in identity_guard["accepted_response_models"]
            ),
            request_timeout_seconds=float(
                self.manifest["proxy"]["request_timeout_seconds"]
            ),
            allow_insecure_provider=(
                self._allow_insecure_provider_for_release_gate()
            ),
        )
        self._identity_relay.start()
        temporary = Path(tempfile.mkdtemp(prefix="sieve-tokenhub-adapter-"))
        temporary.chmod(0o700)
        self._temporary_root = temporary
        home = temporary / "home"
        home.mkdir(mode=0o700)
        config = temporary / "litellm.yaml"
        _write_private(
            config,
            _render_tokenhub_config(
                self.manifest,
                self._identity_relay.base_url,
            ),
        )
        self._port = _select_loopback_port()
        self._proxy_key = "sk-sieve-local-" + secrets.token_hex(32)
        self._spawn_process()
        self._wait_until_ready()
        self._ready = True
        _write_sealed_json(
            self.invocation_root / "transport_start.json",
            self.public_record(),
        )

    def _spawn_process(self) -> None:
        if (
            self._temporary_root is None
            or self._port is None
            or not self._proxy_key
            or self._identity_relay is None
            or self._process is not None
            or self._tracker is not None
        ):
            raise ManagedTransportError("managed adapter spawn state is invalid")
        self._identity_relay.ensure_healthy()
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "HOME": str(self._temporary_root / "home"),
            "TMPDIR": str(self._temporary_root),
            "LANG": "en_US.UTF-8",
            "LC_ALL": "en_US.UTF-8",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            # LiteLLM receives only a short-lived loopback relay capability.
            # The provider credential never enters the child process.
            "TOKENHUB_API_KEY": self._identity_relay.capability,
            "LITELLM_MASTER_KEY": self._proxy_key,
            "LITELLM_MODE": "PRODUCTION",
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LITELLM_LOG": "ERROR",
            "LITELLM_TELEMETRY": "False",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        launcher = self.runtime_root / str(
            self.manifest["runtime"]["launcher_relative_path"]
        )
        try:
            self._process = subprocess.Popen(
                [
                    str(launcher),
                    "--config",
                    str(self._temporary_root / "litellm.yaml"),
                    "--host",
                    LOOPBACK_HOST,
                    "--port",
                    str(self._port),
                    "--telemetry",
                    "False",
                ],
                cwd=self.runtime_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            snapshot = _process_snapshot()
            self._tracker = _OwnedProcessTracker(self._process.pid, snapshot)
        except (OSError, IsolationError) as exc:
            try:
                self._terminate_untracked_spawn()
            finally:
                # The bare-PGID cleanup is valid only in this immediate
                # post-Popen/pre-tracker window.  Never retain state that
                # could cause a later close to reuse that unsafe path.
                self._process = None
                self._tracker = None
            raise ManagedTransportError("managed_adapter_process_start_failed") from exc
        finally:
            environment.clear()

    def recycle_after_ambiguous(self) -> None:
        """Kill, prove drained, and restart the nested proxy in place."""

        if not self._started or self._closed:
            raise ManagedTransportError("managed adapter recycle lifecycle is invalid")
        self._stop_process()
        observed_runtime = verify_tokenhub_runtime(
            self.runtime_root,
            manifest=self.manifest,
        )
        if observed_runtime != self._runtime_record:
            raise ManagedTransportError(
                "managed adapter runtime changed before restart"
            )
        self._runtime_reverification_count += 1
        try:
            self._spawn_process()
            self._wait_until_ready()
        except BaseException:
            # A failed replacement is never left listening or connected to a
            # raw provider.  Cleanup failure intentionally supersedes the
            # readiness error because the safety barrier is then unproven.
            self._stop_process()
            raise
        self._ready = True
        self._recycle_count += 1
        _write_sealed_json(
            self.invocation_root
            / f"transport_recycle_{self._recycle_count:04d}.json",
            {
                "schema_version": "sieve_runtime_transport_recycle_v1",
                "adapter_id": TOKENHUB_ADAPTER,
                "reason": "ambiguous_post_send_or_stream_state",
                "recycle_index": self._recycle_count,
                "old_process_tree_terminated": True,
                "old_listener_released_before_restart": True,
                "provider_identity_relay_upstreams_drained": True,
                "runtime_reverified_before_restart": True,
                "same_loopback_binding_and_capability_reused": True,
                "new_process_ready": True,
                "provider_credential_endpoint_request_response_or_hidden_reasoning_recorded": False,
            },
        )

    def transport(self) -> RuntimeTransport:
        self.ensure_healthy()
        if self._port is None or not self._proxy_key:
            raise ManagedTransportError("managed adapter did not publish a local route")
        local_routes: dict[str, RouteRuntimeBinding] = {}
        for route_id, binding in self.bindings.routes.items():
            route = self.registry.route(route_id)
            if route.transport_adapter_id != TOKENHUB_ADAPTER:
                raise ManagedTransportError("managed experiment contains a direct route")
            local_routes[route_id] = RouteRuntimeBinding(
                route_profile_id=route_id,
                binding_profile_id=binding.binding_profile_id,
                upstream_base_url=f"http://{LOOPBACK_HOST}:{self._port}/v1",
                allow_insecure_upstream=True,
                managed_adapter_id=TOKENHUB_ADAPTER,
            )
        return RuntimeTransport(
            bindings=RuntimeBindings(
                api_family_id=self.bindings.api_family_id,
                routes=local_routes,
                source_path=self.bindings.source_path,
                source_sha256=self.bindings.source_sha256,
            ),
            credential=self._proxy_key,
            adapter_id=TOKENHUB_ADAPTER,
            public_record=self.public_record(),
            _manager=self,
        )

    def ensure_healthy(self) -> None:
        if not self._ready or self._closed or self._process is None:
            raise ManagedTransportError("managed_adapter_not_ready")
        if self._process.poll() is not None:
            raise ManagedTransportError("managed_adapter_exited")
        if self._tracker is None:
            raise ManagedTransportError("managed_adapter_identity_missing")
        snapshot = _process_snapshot()
        self._tracker.refresh(snapshot)
        if self._process.pid not in self._tracker.live_pids(snapshot):
            raise ManagedTransportError("managed_adapter_identity_changed")
        if self._port is None or not self._proxy_key:
            raise ManagedTransportError("managed_adapter_local_capability_missing")
        if self._identity_relay is None:
            raise ManagedTransportError("provider_identity_relay_missing")
        try:
            self._identity_relay.ensure_healthy()
        except TokenHubIdentityRelayError as exc:
            raise ManagedTransportError("provider_identity_relay_unhealthy") from exc
        aliases = _read_model_aliases(self._port, self._proxy_key)
        if aliases != {str(self.manifest["proxy"]["model_alias"])}:
            raise ManagedTransportError("managed_adapter_model_inventory_mismatch")

    def post_send_ambiguity_detected(self) -> bool:
        if self._identity_relay is None:
            raise ManagedTransportError("provider_identity_relay_missing")
        return self._identity_relay.post_send_ambiguity_detected()

    def classify_outer_result(self, outer_request_complete: bool) -> bool:
        if self._identity_relay is None:
            raise ManagedTransportError("provider_identity_relay_missing")
        return self._identity_relay.classify_outer_result(
            outer_request_complete
        )

    def _stop_process(self) -> None:
        """Terminate the birth-bound process tree and prove its listener gone."""

        self._ready = False
        process = self._process
        tracker = self._tracker
        if process is not None:
            if tracker is None:
                raise ManagedTransportError("managed_adapter_identity_missing")
            else:
                snapshot = _process_snapshot()
                tracker.refresh(snapshot)
                if tracker.live_pids(snapshot):
                    _terminate_owned_processes(process, tracker)
                elif process.poll() is None:
                    raise ManagedTransportError(
                        "managed_adapter_identity_unprovable"
                    )
                if process.poll() is None:
                    raise ManagedTransportError("managed_adapter_did_not_exit")
        if self._port is not None:
            deadline = time.monotonic() + LISTENER_RELEASE_TIMEOUT_SECONDS
            while _loopback_listener_exists(self._port):
                if time.monotonic() >= deadline:
                    raise ManagedTransportError("managed_adapter_listener_remained")
                time.sleep(0.05)
        self._process = None
        self._tracker = None
        if self._identity_relay is not None:
            try:
                self._identity_relay.drain_ambiguous()
            except TokenHubIdentityRelayError as exc:
                raise ManagedTransportError(
                    "provider_identity_relay_drain_failed"
                ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error: Exception | None = None
        was_ready = self._ready
        identity_relay_port = (
            self._identity_relay.port
            if self._identity_relay is not None
            else None
        )
        try:
            self._stop_process()
        except (IsolationError, ManagedTransportError) as exc:
            error = exc
        finally:
            if self._identity_relay is not None:
                try:
                    self._identity_relay.close()
                except TokenHubIdentityRelayError as exc:
                    if error is None:
                        error = ManagedTransportError(
                            "provider_identity_relay_close_failed"
                        )
                        error.__cause__ = exc
            self._ready = False
            self._provider_credential = ""
            self._proxy_key = ""
            if self._temporary_root is not None:
                try:
                    _remove_private_tree(self._temporary_root)
                except ManagedTransportError as exc:
                    if error is None:
                        error = exc
            completion = {
                "schema_version": "sieve_runtime_transport_completion_v2",
                "adapter_id": TOKENHUB_ADAPTER,
                "started": self._started,
                "ready": was_ready,
                "ended": error is None,
                "listener_released": (
                    self._port is None or not _loopback_listener_exists(self._port)
                ),
                "recycle_count": self._recycle_count,
                "runtime_reverification_count": (
                    self._runtime_reverification_count
                ),
                "provider_identity_relay": (
                    self._identity_relay.public_record()
                    if self._identity_relay is not None
                    else None
                ),
                "provider_identity_relay_listener_released": (
                    identity_relay_port is None
                    or not _loopback_listener_exists(identity_relay_port)
                ),
                "provider_credential_endpoint_request_response_or_hidden_reasoning_recorded": False,
            }
            completion_path = self.invocation_root / "transport_completion.json"
            if not completion_path.exists() and not completion_path.is_symlink():
                _write_sealed_json(completion_path, completion)
        if error is not None:
            raise ManagedTransportError("managed_adapter_close_barrier_failed") from error

    def public_record(self) -> dict[str, Any]:
        if self._runtime_record is None:
            raise ManagedTransportError("managed adapter runtime is unverified")
        return {
            "schema_version": "sieve_runtime_transport_record_v1",
            "adapter_id": TOKENHUB_ADAPTER,
            "adapter_manifest_sha256": _sha256_file(TOKENHUB_MANIFEST),
            "runtime": dict(self._runtime_record),
            "proxy_contract": {
                "model_alias": self.manifest["proxy"]["model_alias"],
                "upstream_model": self.manifest["proxy"]["upstream_model"],
                "maximum_output_tokens": self.manifest["proxy"][
                    "maximum_output_tokens"
                ],
                "reasoning_effort": "high",
                "anthropic_thinking": dict(
                    self.manifest["proxy"]["fixed_thinking"]
                ),
                "adapter_retry_count": self.manifest["proxy"]["num_retries"],
                "request_timeout_seconds": self.manifest["proxy"][
                    "request_timeout_seconds"
                ],
                "ambiguous_request_recovery": self.manifest["proxy"][
                    "ambiguous_request_recovery"
                ],
                "loopback_only": True,
            },
            "reasoning_replay_bridge": dict(
                self.manifest["reasoning_replay_bridge"]
            ),
            "provider_identity_guard": dict(
                self.manifest["provider_identity_guard"]
            ),
            "provider_identity_relay": (
                self._identity_relay.public_record()
                if self._identity_relay is not None
                else None
            ),
            "provider_credential_forwarded_to_agent_or_episode_gateway": False,
            "provider_credential_forwarded_to_litellm": False,
            "provider_endpoint_recorded_or_hashed": False,
            "request_response_or_hidden_reasoning_recorded": False,
        }

    def _provider_base(self) -> str:
        if set(self.bindings.routes) != {"tokenhub-chat-reasoning-agent-v1"}:
            raise ManagedTransportError("TokenHub route set differs from the pinned adapter")
        value = self.bindings.for_route(
            "tokenhub-chat-reasoning-agent-v1"
        ).upstream_base_url
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ManagedTransportError("TokenHub provider base is not an HTTPS root")
        return value.rstrip("/")

    def _allow_insecure_provider_for_release_gate(self) -> bool:
        """Permit HTTP only in the explicit loopback-only release fixture."""

        return False

    def _wait_until_ready(self) -> None:
        if self._process is None or self._tracker is None or self._port is None:
            raise ManagedTransportError("managed adapter process is absent")
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise ManagedTransportError("managed_adapter_exited_before_ready")
            snapshot = _process_snapshot()
            self._tracker.refresh(snapshot)
            if self._process.pid not in self._tracker.live_pids(snapshot):
                raise ManagedTransportError("managed_adapter_identity_changed_before_ready")
            try:
                models = _read_model_aliases(self._port, self._proxy_key)
            except ManagedTransportError:
                time.sleep(0.1)
                continue
            expected = {str(self.manifest["proxy"]["model_alias"])}
            if models != expected:
                raise ManagedTransportError("managed_adapter_model_inventory_mismatch")
            return
        raise ManagedTransportError("managed_adapter_readiness_timeout")

    def _terminate_untracked_spawn(self) -> None:
        """Boundedly remove and prove empty the adapter's initial process group."""

        process = self._process
        if process is None:
            return
        try:
            _terminate_untracked_process_group(process)
        except IsolationError as exc:
            raise ManagedTransportError(
                "untracked managed adapter process group did not exit"
            ) from exc


def verify_tokenhub_runtime(
    runtime_root: Path, *, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Verify source, interpreter, entrypoint, lock, and installed dependency tree."""

    expected = dict(_read_manifest(TOKENHUB_MANIFEST) if manifest is None else manifest)
    raw_root = runtime_root.expanduser().absolute()
    if not raw_root.is_dir() or raw_root.is_symlink():
        raise ManagedTransportError("TokenHub LiteLLM runtime must be a real directory")
    root = raw_root.resolve(strict=True)
    if root != raw_root:
        raise ManagedTransportError("TokenHub LiteLLM runtime path contains a link")
    info = root.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise ManagedTransportError("TokenHub LiteLLM runtime ownership/mode is unsafe")
    runtime = _object(expected.get("runtime"), "runtime manifest")
    launcher = _runtime_file(root, runtime["launcher_relative_path"], executable=True)
    uv_lock = _runtime_file(root, runtime["uv_lock_relative_path"])
    python_link = root / str(runtime["python_relative_path"])
    if not python_link.exists() or not python_link.is_symlink():
        raise ManagedTransportError("TokenHub runtime Python is missing")
    python = python_link.resolve(strict=True)
    if not python.is_file():
        raise ManagedTransportError("TokenHub runtime Python target is invalid")
    python_bundle = python.parent.parent
    if not python_bundle.is_dir() or python_bundle.is_symlink():
        raise ManagedTransportError("TokenHub Python bundle root is invalid")
    checks = {
        "launcher_sha256": _sha256_file(launcher),
        "uv_lock_sha256": _sha256_file(uv_lock),
        "python_resolved_sha256": _sha256_file(python),
        "python_bundle_tree_sha256": runtime_tree_sha256(python_bundle),
        "source_tree_sha256": runtime_tree_sha256(
            root,
            excluded_top_level={".git", ".venv"},
        ),
        "venv_tree_sha256": runtime_tree_sha256(root / ".venv"),
    }
    for name, observed in checks.items():
        if observed != runtime.get(name):
            raise ManagedTransportError(f"TokenHub runtime pin mismatch: {name}")
    commit = _git_output(root, ["rev-parse", "HEAD"])
    if commit != runtime.get("source_git_commit"):
        raise ManagedTransportError("TokenHub runtime git commit differs")
    if _git_output(root, ["status", "--porcelain", "--untracked-files=all"]):
        raise ManagedTransportError("TokenHub runtime source tree is dirty")
    _verify_nonrelocatable_runtime_links(root, launcher)
    provenance = _verify_python_runtime_provenance(python_link, root)
    version = _python_metadata_version(python_link, "litellm")
    if version != runtime.get("litellm_version"):
        raise ManagedTransportError("TokenHub installed LiteLLM version differs")
    codec = _verify_tokenhub_reasoning_codec(python_link, root, expected)
    return {
        "tree_hash_policy": TREE_HASH_POLICY,
        "source_git_commit": commit,
        "litellm_version": version,
        **checks,
        "reasoning_codec": codec,
        "python_runtime_provenance": provenance,
        "runtime_path_recorded_or_hashed": False,
    }


def runtime_tree_sha256(
    root: Path, *, excluded_top_level: set[str] | None = None
) -> str:
    """Hash every executable/runtime byte except Finder metadata."""

    base = root.expanduser().absolute()
    if not base.is_dir() or base.is_symlink():
        raise ManagedTransportError("runtime tree root must be a real directory")
    excluded = set(excluded_top_level or set())
    digest = hashlib.sha256()
    pending = [base]
    records: list[Path] = []
    while pending:
        directory = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise ManagedTransportError("runtime tree cannot be enumerated") from exc
        for entry in entries:
            path = Path(entry.path)
            relative = path.relative_to(base)
            if relative.parts[0] in excluded:
                continue
            if any(part in IGNORED_TREE_COMPONENTS for part in relative.parts):
                continue
            if path.suffix in IGNORED_TREE_SUFFIXES or path.name == ".DS_Store":
                continue
            records.append(path)
            if entry.is_dir(follow_symlinks=False):
                pending.append(path)
    for path in sorted(records, key=lambda child: child.relative_to(base).as_posix()):
        relative = path.relative_to(base).as_posix().encode("utf-8")
        info = path.lstat()
        mode = stat.S_IMODE(info.st_mode)
        if stat.S_ISLNK(info.st_mode):
            kind = b"L"
            payload = os.readlink(path).encode("utf-8")
        elif stat.S_ISDIR(info.st_mode):
            kind = b"D"
            payload = b""
        elif stat.S_ISREG(info.st_mode):
            kind = b"F"
            file_digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            payload = file_digest.digest()
        else:
            raise ManagedTransportError("runtime tree contains a special file")
        digest.update(kind + b"\0" + relative + b"\0")
        digest.update(f"{mode:o}".encode("ascii") + b"\0" + payload + b"\n")
    return digest.hexdigest()


def _render_tokenhub_config(manifest: Mapping[str, Any], provider_base: str) -> bytes:
    proxy = _object(manifest.get("proxy"), "proxy manifest")
    # JSON strings are valid YAML scalars and prevent operator URL injection.
    alias = json.dumps(proxy["model_alias"])
    upstream = json.dumps(proxy["upstream_model"])
    base = json.dumps(provider_base)
    allowed = json.dumps(proxy["allowed_openai_params"])
    thinking = _object(proxy["fixed_thinking"], "fixed thinking")
    text = f"""model_list:
  - model_name: {alias}
    litellm_params:
      model: {upstream}
      api_key: os.environ/TOKENHUB_API_KEY
      api_base: {base}
      max_tokens: {int(proxy['maximum_output_tokens'])}
      allowed_openai_params: {allowed}
      thinking:
        type: {json.dumps(thinking['type'])}
        budget_tokens: {int(thinking['budget_tokens'])}
    model_info:
      base_model: {upstream}

litellm_settings:
  drop_params: true
  telemetry: false
  num_retries: {int(proxy['num_retries'])}
  request_timeout: {int(proxy['request_timeout_seconds'])}

general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
"""
    return text.encode("utf-8")


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ManagedTransportError("managed adapter manifest is missing or linked")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ManagedTransportError("managed adapter manifest is invalid") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "adapter_id",
        "route_profile_id",
        "provider_identity_guard",
        "reasoning_replay_bridge",
        "runtime",
        "proxy",
    }:
        raise ManagedTransportError("managed adapter manifest field set differs")
    if (
        value["schema_version"] != "sieve_managed_transport_adapter_v4"
        or value["adapter_id"] != TOKENHUB_ADAPTER
        or value["route_profile_id"] != "tokenhub-chat-reasoning-agent-v1"
    ):
        raise ManagedTransportError("managed adapter manifest identity differs")
    reasoning_bridge = _object(
        value["reasoning_replay_bridge"], "reasoning replay bridge"
    )
    if reasoning_bridge != {
        "anthropic_signed_thinking_required_for_tool_followup": True,
        "credential_free_real_pi_litellm_gate_required": True,
        "gateway_detail_format": ANTHROPIC_REASONING_DETAIL_FORMAT,
        "incoming_litellm_field": "delta.thinking_blocks",
        "multiple_normal_thinking_blocks": "fail_closed",
        "normal_thinking_block_policy": (
            "aggregate_exactly_one_complete_signed_block"
        ),
        "pi_replay_field": "assistant.reasoning_details",
        "redacted_thinking_block_policy": "preserve_discrete_ordered_blocks",
        "signature_fragment_policy": (
            "concatenate_in_stream_order_before_pi_delivery"
        ),
        "upstream_litellm_field": "assistant.thinking_blocks",
    }:
        raise ManagedTransportError("managed adapter reasoning bridge differs")
    provider_identity_guard = _object(
        value["provider_identity_guard"], "provider identity guard"
    )
    if provider_identity_guard != {
        "accepted_response_models": ["hy4-preview"],
        "expected_request_model": "hy4-preview",
        "maximum_identity_preface_bytes": 1048576,
        "mismatch_policy": "fail_closed_before_litellm_success_stream",
        "provider_credential_forwarded_to_litellm": False,
        "raw_response_identity_field": "message_start.message.model",
        "relay_contract": "host_loopback_before_litellm_v1",
    }:
        raise ManagedTransportError("managed adapter identity guard differs")
    runtime = _object(value["runtime"], "runtime manifest")
    if set(runtime) != {
        "source_git_commit",
        "litellm_version",
        "uv_lock_relative_path",
        "uv_lock_sha256",
        "launcher_relative_path",
        "launcher_sha256",
        "python_relative_path",
        "python_resolved_sha256",
        "python_bundle_tree_sha256",
        "source_tree_sha256",
        "venv_tree_sha256",
        "tree_hash_policy",
    } or runtime.get("tree_hash_policy") != TREE_HASH_POLICY:
        raise ManagedTransportError("managed adapter runtime contract differs")
    if (
        any(
            not isinstance(runtime.get(name), str)
            or SHA256.fullmatch(str(runtime[name])) is None
            for name in {
                "uv_lock_sha256",
                "launcher_sha256",
                "python_resolved_sha256",
                "python_bundle_tree_sha256",
                "source_tree_sha256",
                "venv_tree_sha256",
            }
        )
        or not isinstance(runtime.get("source_git_commit"), str)
        or GIT_COMMIT.fullmatch(str(runtime["source_git_commit"])) is None
    ):
        raise ManagedTransportError("managed adapter runtime hashes are malformed")
    proxy = _object(value["proxy"], "proxy manifest")
    if set(proxy) != {
        "ambiguous_request_recovery",
        "allowed_openai_params",
        "model_alias",
        "upstream_model",
        "maximum_output_tokens",
        "fixed_thinking",
        "drop_params",
        "telemetry",
        "num_retries",
        "request_timeout_seconds",
    } or proxy != {
        "ambiguous_request_recovery": (
            "kill_process_tree_release_listener_restart_before_next_unit"
        ),
        "allowed_openai_params": ["thinking"],
        "model_alias": "hy4-preview",
        "upstream_model": "anthropic/hy4-preview",
        "maximum_output_tokens": 65536,
        "fixed_thinking": {"type": "enabled", "budget_tokens": 4096},
        "drop_params": True,
        "telemetry": False,
        "num_retries": 0,
        "request_timeout_seconds": 1860,
    }:
        raise ManagedTransportError("managed adapter proxy contract differs")
    return value


def _validate_tokenhub_experiment_contract(
    registry: ProfileRegistry,
    experiment: ExperimentProfile,
    manifest: Mapping[str, Any],
) -> None:
    """Prove the nested proxy cannot win the ambiguous-timeout race.

    The scoped gateway is the sole request-retry owner.  Its post-send socket
    timeout must therefore fire before LiteLLM's own request timeout.  If the
    two values are equal, LiteLLM could turn an unknown post-send state into a
    local retryable HTTP response just as the outer gateway times out.
    """

    if (
        experiment.api_family_id != "tokenhub"
        or experiment.model_profile_ids != (TOKENHUB_MODEL_PROFILE_ID,)
    ):
        raise ManagedTransportError("managed TokenHub experiment family differs")
    proxy = _object(manifest.get("proxy"), "proxy manifest")
    identity_guard = _object(
        manifest.get("provider_identity_guard"),
        "provider identity guard",
    )
    adapter_timeout = float(proxy["request_timeout_seconds"])
    for model_id in experiment.model_profile_ids:
        model = registry.model(model_id)
        route = registry.route(model.route_profile_id)
        expected_route = {
            "route_profile_id": TOKENHUB_ROUTE_PROFILE_ID,
            "api_family_id": "tokenhub",
            "pi_api_protocol": "openai-completions",
            "client_path": "/v1/chat/completions",
            "upstream_path": "/chat/completions",
            "auth_strategy": "standard_bearer_v1",
            "transport_adapter_id": TOKENHUB_ADAPTER,
            "physical_attempt_identity_strategy": "none",
            "option_style": "top_level_reasoning_v1",
            "response_contract": "openai_chat_stream_tool_identity_v1",
        }
        expected_model = {
            "model_profile_id": TOKENHUB_MODEL_PROFILE_ID,
            "display_label": "hy4-preview (TokenHub)",
            "api_family_id": "tokenhub",
            "route_profile_id": TOKENHUB_ROUTE_PROFILE_ID,
            "client_wire_model": "hy4-preview",
            "upstream_wire_model": "hy4-preview",
            "request_timeout_seconds": 1800,
            "temperature": None,
            "reasoning": {
                "style": "top_level_reasoning",
                "effort": "high",
                "thinking_type": None,
                "preserve_across_tool_turns": True,
            },
            "pi": {
                "api_protocol": "openai-completions",
                "thinking_level": "high",
                "context_window": 131072,
                "maximum_output_tokens": 65536,
                "compatibility": {
                    "maxTokensField": "max_tokens",
                    "requiresAssistantAfterToolResult": False,
                    "requiresReasoningContentOnAssistantMessages": True,
                    "requiresThinkingAsText": False,
                    "requiresToolResultName": False,
                    "sendSessionAffinityHeaders": False,
                    "sessionAffinityFormat": "openai-nosession",
                    "supportsDeveloperRole": False,
                    "supportsFinishReason": True,
                    "supportsLongCacheRetention": False,
                    "supportsOpenAIGrammarTools": False,
                    "supportsReasoningEffort": False,
                    "supportsStore": False,
                    "supportsStrictMode": False,
                    "supportsUsageInStreaming": True,
                    "thinkingFormat": "openai",
                },
            },
            "retry": {
                "max_infrastructure_retries": 5,
                "retry_delay_seconds": 30,
                "retryable_http_statuses": TOKENHUB_RETRYABLE_HTTP,
                "retry_transport_failures": True,
                "retry_ambiguous_timeouts": False,
            },
            "auth_query_parameters": {},
            "api3_strategy_type": None,
            "user_agent_suffix": "sieve-pi-tokenhub-hy4-preview-v1",
            "response_identity": {
                "required": True,
                "accepted_models": ["hy4-preview"],
            },
        }
        if route.public_dict() != expected_route or model.public_dict() != expected_model:
            raise ManagedTransportError(
                "managed TokenHub model/route contract differs"
            )
        if (
            manifest.get("route_profile_id") != route.route_profile_id
            or proxy.get("model_alias") != model.client_wire_model
            or proxy.get("upstream_model")
            != f"anthropic/{model.upstream_wire_model}"
            or proxy.get("maximum_output_tokens")
            != model.pi.maximum_output_tokens
            or proxy.get("fixed_thinking")
            != {"type": "enabled", "budget_tokens": 4096}
            or identity_guard.get("expected_request_model")
            != model.upstream_wire_model
            or identity_guard.get("accepted_response_models")
            != list(model.accepted_response_models)
            or identity_guard.get("provider_credential_forwarded_to_litellm")
            is not False
        ):
            raise ManagedTransportError(
                "managed TokenHub manifest/profile contract differs"
            )
        if (
            adapter_timeout
            < float(model.request_timeout_seconds)
            + MIN_ADAPTER_TIMEOUT_GRACE_SECONDS
        ):
            raise ManagedTransportError(
                "managed adapter timeout must exceed the outer ambiguous timeout"
            )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManagedTransportError(f"{label} must be an object")
    return dict(value)


def _runtime_file(root: Path, relative: Any, *, executable: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ManagedTransportError("runtime manifest path is invalid")
    path = root / relative
    if not path.is_file() or path.is_symlink() or path.resolve() != path.absolute():
        raise ManagedTransportError("runtime manifest file is missing or linked")
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022:
        raise ManagedTransportError("runtime manifest file ownership/mode is unsafe")
    if executable and not os.access(path, os.X_OK):
        raise ManagedTransportError("runtime launcher is not executable")
    return path


def _git_output(root: Path, arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["/usr/bin/git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ManagedTransportError("managed adapter git verification failed") from exc
    return result.stdout.strip()


def _python_metadata_version(python: Path, distribution: str) -> str:
    try:
        result = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import importlib.metadata,sys;"
                    "sys.stdout.write(importlib.metadata.version(sys.argv[1]))"
                ),
                distribution,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
            env={
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ManagedTransportError("managed adapter distribution check failed") from exc
    return result.stdout.strip()


def _verify_tokenhub_reasoning_codec(
    python: Path, runtime_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove the pinned codec emits fixed Anthropic thinking for Hy4.

    TokenHub's Hy4 identifier is absent from LiteLLM's built-in reasoning map.
    The managed config therefore fixes an explicit ``thinking`` block and
    allows only that provider parameter.  The outer gateway still owns and
    audits ``reasoning_effort=high``; this offline check proves that LiteLLM
    drops that OpenAI field while preserving the exact Anthropic wire shape.
    """

    proxy = _object(manifest.get("proxy"), "proxy manifest")
    program = (
        "import json;"
        "from litellm.utils import get_optional_params;"
        "from litellm.llms.anthropic.chat.transformation import AnthropicConfig;"
        "v=get_optional_params("
        "model='hy4-preview',custom_llm_provider='anthropic',"
        "reasoning_effort='high',max_tokens=65536,drop_params=True,"
        "allowed_openai_params=['thinking'],"
        "thinking={'type':'enabled','budget_tokens':4096});"
        "d=AnthropicConfig().transform_request("
        "model='hy4-preview',messages=[{'role':'user','content':'x'}],"
        "optional_params=v,litellm_params={},headers={});"
        "print(json.dumps({'thinking':d.get('thinking'),"
        "'reasoning_effort_present':'reasoning_effort' in d,"
        "'max_tokens':d.get('max_tokens')},sort_keys=True,separators=(',',':')))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", program],
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
            cwd=runtime_root,
            env={
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "LITELLM_TELEMETRY": "False",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
        observed = json.loads(lines[-1]) if lines else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise ManagedTransportError(
            "managed adapter reasoning codec verification failed"
        ) from exc
    expected = {
        "thinking": dict(proxy["fixed_thinking"]),
        "reasoning_effort_present": False,
        "max_tokens": int(proxy["maximum_output_tokens"]),
    }
    if observed != expected:
        raise ManagedTransportError("managed adapter reasoning codec shape differs")
    return {
        "input_reasoning_effort": "high",
        "anthropic_thinking": dict(expected["thinking"]),
        "openai_reasoning_effort_forwarded": False,
        "maximum_output_tokens": expected["max_tokens"],
    }


def _verify_nonrelocatable_runtime_links(root: Path, launcher: Path) -> None:
    """Reject a copied editable runtime that would execute the original tree."""

    try:
        first_line = launcher.open("rb").readline(4096).decode("utf-8").rstrip("\r\n")
    except (OSError, UnicodeError) as exc:
        raise ManagedTransportError("managed adapter launcher shebang is invalid") from exc
    expected_shebang = f"#!{root / '.venv/bin/python3'}"
    if first_line != expected_shebang:
        raise ManagedTransportError("managed adapter launcher escapes supplied runtime")
    site_roots = sorted((root / ".venv/lib").glob("python*/site-packages"))
    if len(site_roots) != 1 or not site_roots[0].is_dir() or site_roots[0].is_symlink():
        raise ManagedTransportError("managed adapter site-packages root differs")
    pth_files = sorted(site_roots[0].glob("*.pth"))
    if not pth_files:
        raise ManagedTransportError("managed adapter editable path contract is missing")
    for path in pth_files:
        if not path.is_file() or path.is_symlink():
            raise ManagedTransportError("managed adapter .pth entry is linked")
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise ManagedTransportError("managed adapter .pth entry is invalid") from exc
        for line in lines:
            candidate = line.strip()
            if not candidate or candidate.startswith("#") or candidate.startswith("import"):
                continue
            if not candidate.startswith("/"):
                raise ManagedTransportError("managed adapter .pth path is not absolute")
            try:
                Path(candidate).resolve(strict=True).relative_to(root)
            except (OSError, ValueError) as exc:
                raise ManagedTransportError(
                    "managed adapter .pth path escapes supplied runtime"
                ) from exc


def _verify_python_runtime_provenance(
    venv_python: Path, runtime_root: Path
) -> dict[str, Any]:
    """Prove venv detection and imported LiteLLM stay inside the verified tree."""

    program = (
        "import importlib.metadata,json,litellm,os,sys;"
        "d=importlib.metadata.distribution('litellm');"
        "print(json.dumps({'prefix':os.path.realpath(sys.prefix),"
        "'litellm':os.path.realpath(litellm.__file__),"
        "'metadata':os.path.realpath(str(d.locate_file('')))},"
        "sort_keys=True,separators=(',',':')))"
    )
    try:
        result = subprocess.run(
            [str(venv_python), "-c", program],
            check=True,
            capture_output=True,
            text=True,
            timeout=30.0,
            cwd=runtime_root,
            env={
                "PATH": "/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "LITELLM_LOCAL_MODEL_COST_MAP": "True",
                "LITELLM_TELEMETRY": "False",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
        observed = json.loads(lines[-1]) if lines else None
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise ManagedTransportError(
            "managed adapter Python provenance verification failed"
        ) from exc
    if not isinstance(observed, dict) or set(observed) != {
        "prefix",
        "litellm",
        "metadata",
    }:
        raise ManagedTransportError("managed adapter Python provenance differs")
    expected_prefix = (runtime_root / ".venv").resolve(strict=True)
    try:
        if Path(str(observed["prefix"])).resolve(strict=True) != expected_prefix:
            raise ManagedTransportError("managed adapter venv prefix differs")
        Path(str(observed["litellm"])).resolve(strict=True).relative_to(runtime_root)
        Path(str(observed["metadata"])).resolve(strict=True).relative_to(
            expected_prefix
        )
    except (OSError, ValueError) as exc:
        raise ManagedTransportError(
            "managed adapter imported code escapes verified runtime"
        ) from exc
    return {
        "venv_prefix_verified": True,
        "litellm_source_within_runtime": True,
        "distribution_metadata_within_venv": True,
        "runtime_path_recorded_or_hashed": False,
    }


def _select_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((LOOPBACK_HOST, 0))
        port = int(listener.getsockname()[1])
    if not 1024 <= port <= 65535:
        raise ManagedTransportError("managed adapter selected an invalid port")
    return port


def _read_model_aliases(port: int, key: str) -> set[str]:
    connection = http.client.HTTPConnection(LOOPBACK_HOST, port, timeout=1.0)
    try:
        connection.request(
            "GET",
            "/v1/models",
            headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        )
        response = connection.getresponse()
        body = response.read(READY_BODY_LIMIT + 1)
    except (OSError, http.client.HTTPException) as exc:
        raise ManagedTransportError("managed adapter is not ready") from exc
    finally:
        connection.close()
    if response.status != 200 or len(body) > READY_BODY_LIMIT:
        raise ManagedTransportError("managed adapter readiness response failed")
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise ManagedTransportError("managed adapter readiness JSON failed") from exc
    if not isinstance(value, dict) or not isinstance(value.get("data"), list):
        raise ManagedTransportError("managed adapter readiness contract failed")
    aliases: set[str] = set()
    for item in value["data"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ManagedTransportError("managed adapter model entry failed")
        aliases.add(item["id"])
    return aliases


def _loopback_listener_exists(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        return probe.connect_ex((LOOPBACK_HOST, port)) == 0


def _write_private(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        PRIVATE_FILE_MODE,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
    finally:
        os.close(descriptor)


def _write_sealed_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, SEALED_RECORD_MODE)
    finally:
        os.close(descriptor)


def _read_private_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ManagedTransportError("managed adapter lifecycle record is missing")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise ManagedTransportError(
            "managed adapter lifecycle record is invalid"
        ) from exc
    if not isinstance(value, dict):
        raise ManagedTransportError("managed adapter lifecycle record is not an object")
    return value


def _remove_private_tree(root: Path) -> None:
    if not root.exists():
        return
    if not root.is_dir() or root.is_symlink():
        raise ManagedTransportError("managed adapter temporary root is unsafe")
    for directory, directories, files in os.walk(root, topdown=False, followlinks=False):
        current = Path(directory)
        for name in files:
            path = current / name
            if path.is_symlink() or not path.is_file():
                raise ManagedTransportError("managed adapter temporary file is unsafe")
            path.unlink()
        for name in directories:
            path = current / name
            if path.is_symlink() or not path.is_dir():
                raise ManagedTransportError("managed adapter temporary directory is unsafe")
            path.rmdir()
    root.rmdir()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
