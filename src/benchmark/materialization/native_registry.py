from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from benchmark.materialization.catalog import sha256_file
from benchmark.materialization.contracts import MaterializationError
from benchmark.utils.io import write_json


AUTHORITY_SCHEME = "hmac-sha256"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True)
class NativeRegistryAuthority:
    """Benchmark-side signing capability for registered native placements.

    The secret belongs to the benchmark placement tool and is never serialized
    into a generator artifact or registry.
    """

    key_id: str
    _secret: bytes = field(repr=False, compare=False)

    @classmethod
    def from_secret(
        cls,
        *,
        key_id: str,
        secret: bytes,
    ) -> "NativeRegistryAuthority":
        normalized_key_id = str(key_id or "").strip()
        if not normalized_key_id:
            raise ValueError("native registry authority key_id must be non-empty")
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise ValueError(
                "native registry authority secret must contain at least 32 bytes"
            )
        return cls(key_id=normalized_key_id, _secret=bytes(secret))

    def seal(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        unsigned = dict(payload)
        if "authority" in unsigned:
            raise ValueError("unsigned native registry payload contains authority")
        encoded = _canonical_bytes(unsigned)
        return {
            **unsigned,
            "authority": {
                "scheme": AUTHORITY_SCHEME,
                "key_id": self.key_id,
                "payload_sha256": hashlib.sha256(encoded).hexdigest(),
                "signature": hmac.new(
                    self._secret,
                    encoded,
                    hashlib.sha256,
                ).hexdigest(),
            },
        }

    def verify(self, registry: Mapping[str, Any]) -> None:
        authority = registry.get("authority")
        if not isinstance(authority, dict):
            raise MaterializationError(
                "native placement registry has no benchmark authority seal"
            )
        if set(authority) != {
            "scheme",
            "key_id",
            "payload_sha256",
            "signature",
        }:
            raise MaterializationError(
                "native placement registry authority seal has invalid fields"
            )
        if (
            authority.get("scheme") != AUTHORITY_SCHEME
            or authority.get("key_id") != self.key_id
        ):
            raise MaterializationError(
                "native placement registry authority seal is not trusted"
            )
        unsigned = {
            str(key): value
            for key, value in registry.items()
            if key != "authority"
        }
        encoded = _canonical_bytes(unsigned)
        payload_hash = hashlib.sha256(encoded).hexdigest()
        signature = hmac.new(
            self._secret,
            encoded,
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(
            str(authority.get("payload_sha256") or ""),
            payload_hash,
        ) or not hmac.compare_digest(
            str(authority.get("signature") or ""),
            signature,
        ):
            raise MaterializationError(
                "native placement registry benchmark authority seal is invalid"
            )


def write_benchmark_native_registry(
    path: str | Path,
    *,
    authority: NativeRegistryAuthority,
    source_blend_path: str | Path,
    case_bundle_manifest_sha256: str,
    catalog_snapshot_id: str,
    instances: Iterable[Mapping[str, Any]],
) -> Path:
    """Placement-tool output: write and seal one native registry."""

    source = Path(source_blend_path).expanduser().resolve()
    if not source.is_file():
        raise MaterializationError(
            f"native placement source blend does not exist: {source}"
        )
    payload = {
        "schema_version": "benchmark_owned_native_registry_v1",
        "registry_revision": "registered_rigid_catalog_v1",
        "producer": "benchmark_placement_tool",
        "case_bundle_manifest_sha256": str(
            case_bundle_manifest_sha256 or ""
        ),
        "catalog_snapshot_id": str(catalog_snapshot_id or ""),
        "source_blend_sha256": sha256_file(source),
        "instances": [dict(item) for item in instances],
    }
    return write_json(
        Path(path).expanduser().resolve(),
        authority.seal(payload),
    )
