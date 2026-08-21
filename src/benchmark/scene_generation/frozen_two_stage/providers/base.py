"""Instance-scoped contracts for generation provider routes.

See ``docs/generation_transport_compatibility.md`` for the boundary between
these transport adapters and the frozen Stage A -> retrieval -> Stage C graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable


CanonicalJSONBytes = Callable[[Any], bytes]


@runtime_checkable
class ProviderModel(Protocol):
    """Small structural model-config surface consumed by provider routes."""

    wire_model: str
    api_key: str
    max_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
    repetition_penalty: float | None
    reasoning_effort: str | None
    preserved_thinking: bool | None
    strategy_type: str


@dataclass(frozen=True)
class NormalizedResponse:
    """Provider-neutral response fields consumed by the frozen workflow."""

    content: bytes
    reasoning: bytes | None
    reasoning_content: bytes | None
    usage: Mapping[str, Any] | None

    def as_legacy_tuple(
        self,
    ) -> tuple[bytes, bytes | None, bytes | None, Mapping[str, Any] | None]:
        """Return the tuple shape expected by the legacy generation core."""

        return self.content, self.reasoning, self.reasoning_content, self.usage


@runtime_checkable
class RequestCodec(Protocol):
    """Encode one request and normalize one provider response envelope."""

    def request_value(
        self,
        *,
        model: ProviderModel,
        system_prompt: str,
        user_value: Mapping[str, Any],
        canonical_json_bytes: CanonicalJSONBytes,
    ) -> dict[str, Any]:
        """Build the JSON-compatible request value without sending it."""

    def extract_api_message(self, response_body: bytes) -> NormalizedResponse:
        """Validate and normalize a response envelope."""

    def public_dict(self) -> dict[str, Any]:
        """Return credential-free codec provenance."""


@runtime_checkable
class GatewayPolicy(Protocol):
    """Construct provider-specific request headers and authentication."""

    def request_headers(
        self, model: ProviderModel, session_id: str
    ) -> dict[str, str]:
        """Build request headers for one transport attempt."""

    def public_dict(self) -> dict[str, Any]:
        """Return credential-free gateway provenance."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _source_display_path(path: Path, module_name: str) -> str:
    parts = path.resolve().parts
    try:
        source_index = len(parts) - 1 - tuple(reversed(parts)).index("src")
    except ValueError:
        return module_name.replace(".", "/") + path.suffix
    return Path(*parts[source_index:]).as_posix()


def _module_source_record(module_name: str) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError(f"provider source module has no file: {module_name}")
    path = Path(module_file)
    data = path.read_bytes()
    return {
        "module": module_name,
        "path": _source_display_path(path, module_name),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


@dataclass(frozen=True)
class ProviderRoute:
    """Compose one wire codec with one gateway policy.

    The route is instance-scoped and performs no model-name inference.  A
    caller selects a route explicitly, then supplies model values as data.
    """

    codec: RequestCodec
    gateway: GatewayPolicy
    route_key: str = "composed-provider-route"
    provenance_modules: tuple[str, ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        """Return the explicit registry key consumed by the orchestrator."""

        return self.route_key

    def request_value(
        self,
        *,
        model: ProviderModel,
        system_prompt: str,
        user_value: Mapping[str, Any],
        canonical_json_bytes: CanonicalJSONBytes,
    ) -> dict[str, Any]:
        return self.codec.request_value(
            model=model,
            system_prompt=system_prompt,
            user_value=user_value,
            canonical_json_bytes=canonical_json_bytes,
        )

    def request_headers(
        self, model: ProviderModel, session_id: str
    ) -> dict[str, str]:
        return self.gateway.request_headers(model, session_id)

    def extract_api_message(self, response_body: bytes) -> NormalizedResponse:
        return self.codec.extract_api_message(response_body)

    def public_dict(self) -> dict[str, Any]:
        """Return stable route provenance without credentials or clock state."""

        return {
            "route_key": self.route_key,
            "codec": self.codec.public_dict(),
            "gateway": self.gateway.public_dict(),
        }

    def source_files(self) -> list[dict[str, Any]]:
        """Hash the actual modules composing this provider route."""

        modules: Sequence[str] = (
            __name__,
            *self.provenance_modules,
            type(self.codec).__module__,
            type(self.gateway).__module__,
        )
        unique_modules = tuple(dict.fromkeys(modules))
        return [_module_source_record(name) for name in unique_modules]

    def source_manifest(self) -> dict[str, Any]:
        """Return source and configuration provenance for legacy manifests."""

        payload = {
            "schema_version": "frozen_two_stage_provider_source_manifest_v1",
            "provider_route": self.public_dict(),
            "files": self.source_files(),
        }
        return {
            **payload,
            "manifest_sha256": hashlib.sha256(
                _canonical_json_bytes(payload)
            ).hexdigest(),
        }
