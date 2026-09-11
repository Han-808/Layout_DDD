"""Private route bindings selected without reading credential values."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.parse import urlsplit


BINDINGS_SCHEMA_VERSION = "generation_route_bindings_v2"
BINDINGS_ENV = "LAYOUT_DDD_GENERATION_BINDINGS"
DEFAULT_LOCAL_RELATIVE = Path(".runtime/generation_bindings.local.json")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate route-binding key: {key}")
        result[key] = value
    return result


def _exact(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing or extra:
        raise ValueError(
            f"{label} keys differ: missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _endpoint(value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("route endpoint must be a non-empty trimmed string")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("route endpoint must be an HTTP(S) URL with a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("route endpoint must not contain URL credentials")
    if parsed.fragment:
        raise ValueError("route endpoint must not contain a fragment")
    return value


@dataclass(frozen=True, slots=True)
class PrivateRouteBinding:
    route_profile_id: str
    endpoint: str = field(repr=False)
    credential_env: str = field(repr=False)
    binding_sha256: str

    def credential(self, environ: Mapping[str, str] | None = None) -> str:
        environment = os.environ if environ is None else environ
        value = environment.get(self.credential_env)
        if not isinstance(value, str) or not value:
            # Do not echo the machine-local environment-variable name.
            raise ValueError("the selected route credential is not available")
        if len(value) > 8192 or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise ValueError("the selected route credential is not header-safe")
        return value

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "generation_route_binding_identity_v2",
            "route_profile_id": self.route_profile_id,
            "binding_sha256": self.binding_sha256,
            "endpoint_bound": True,
            "credential_bound": True,
        }


@dataclass(frozen=True, slots=True)
class LocalRouteBindings:
    bindings: Mapping[str, PrivateRouteBinding] = field(repr=False)

    @classmethod
    def load(cls, path: str | Path) -> "LocalRouteBindings":
        selected = Path(path).expanduser()
        if selected.is_symlink():
            raise ValueError("route binding must be a regular non-symlink JSON file")
        candidate = selected.resolve()
        if not candidate.is_file():
            raise ValueError("route binding must be a regular JSON file")
        if candidate.stat().st_size > 200_000:
            raise ValueError("route binding exceeds 200000 bytes")
        raw = json.loads(
            candidate.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number is not allowed: {value}")
            ),
        )
        if not isinstance(raw, dict):
            raise ValueError("route binding must be a JSON object")
        _exact(raw, {"schema_version", "bindings"}, label="route binding")
        if raw["schema_version"] != BINDINGS_SCHEMA_VERSION:
            raise ValueError(f"unsupported route-binding schema: {raw['schema_version']!r}")
        values = raw["bindings"]
        if not isinstance(values, dict) or not values:
            raise ValueError("route bindings must be a non-empty object")
        bindings: dict[str, PrivateRouteBinding] = {}
        for route_profile_id, value in values.items():
            if not isinstance(route_profile_id, str) or not route_profile_id:
                raise ValueError("route binding ID must be a non-empty string")
            if not isinstance(value, dict):
                raise ValueError(f"binding {route_profile_id!r} must be an object")
            _exact(value, {"endpoint", "credential_env"}, label=f"binding {route_profile_id}")
            endpoint = _endpoint(value["endpoint"])
            credential_env = value["credential_env"]
            if not isinstance(credential_env, str) or not _ENV_NAME.fullmatch(credential_env):
                raise ValueError("credential_env must be a portable environment name")
            digest = hashlib.sha256(
                _canonical(
                    {
                        "route_profile_id": route_profile_id,
                        "endpoint": endpoint,
                        "credential_env": credential_env,
                    }
                )
            ).hexdigest()
            bindings[route_profile_id] = PrivateRouteBinding(
                route_profile_id=route_profile_id,
                endpoint=endpoint,
                credential_env=credential_env,
                binding_sha256=digest,
            )
        return cls(bindings=bindings)

    def require(self, route_profile_id: str) -> PrivateRouteBinding:
        try:
            return self.bindings[route_profile_id]
        except KeyError as exc:
            raise ValueError("selected route has no local binding") from exc

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": BINDINGS_SCHEMA_VERSION,
            "bound_route_profile_ids": sorted(self.bindings),
        }


def select_binding_path(
    *,
    repo_root: Path,
    explicit_path: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Select explicit > environment > ignored repository-local binding."""

    if explicit_path is not None:
        return Path(explicit_path).expanduser().absolute()
    environment = os.environ if environ is None else environ
    selected = environment.get(BINDINGS_ENV)
    if selected:
        return Path(selected).expanduser().absolute()
    return (repo_root / DEFAULT_LOCAL_RELATIVE).absolute()
