"""Read-only shared Imaginarium database exposed to every Agent identically."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from benchmark.scene_generation.retrieval import (
    RetrievalCatalog,
    SharedRetrieverRuntime,
    build_runtime,
)
from benchmark.scene_generation.retrieval.runtime import _compute_size_score


SHARED_DB_SCHEMA_VERSION = "non_rectangular_agent_shared_asset_db_v1"
SHARED_DB_MODE = "shared_database"
DEFAULT_RETRIEVAL_PROFILE_ID = (
    "imaginarium-qwen3-embedding-0.6b-stable-top1-v2"
)


class SharedAssetDatabaseError(RuntimeError):
    """Raised when DB identity, resources, or a bounded query is invalid."""


class SharedAssetDatabase:
    """A gated immutable index with deterministic bounded Top-K search."""

    def __init__(
        self,
        runtime: SharedRetrieverRuntime,
        *,
        max_top_k: int = 12,
        require_ready_gate: bool = True,
    ) -> None:
        if isinstance(max_top_k, bool) or not isinstance(max_top_k, int) or max_top_k < 1:
            raise ValueError("max_top_k must be a positive integer")
        self.runtime = runtime
        self.max_top_k = max_top_k
        gate = runtime.gate(strict=True, run_golden=True)
        if require_ready_gate and gate.get("status") != "ready":
            raise SharedAssetDatabaseError("shared DB strict resource gate failed")
        if runtime.embeddings is None or not runtime.assets or not runtime.jid_list:
            raise SharedAssetDatabaseError("shared DB gate did not load the index")
        if len(runtime.assets) != len(runtime.jid_list):
            raise SharedAssetDatabaseError("shared DB asset/order cardinality mismatch")
        if len(set(runtime.jid_list)) != len(runtime.jid_list):
            raise SharedAssetDatabaseError("shared DB asset order contains duplicates")
        expected_rows = int(runtime.composed.index.expected_rows)
        if len(runtime.jid_list) != expected_rows:
            raise SharedAssetDatabaseError("shared DB row count differs from descriptor")
        self.gate_report = gate
        self._manifest = _runtime_manifest(runtime, max_top_k=max_top_k)
        self.snapshot_id = str(self._manifest["snapshot_id"])

    @property
    def asset_count(self) -> int:
        return len(self.runtime.jid_list)

    def public_manifest(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._manifest))

    def resolve(self, asset_id: str) -> dict[str, Any]:
        key = str(asset_id or "").strip()
        if not key:
            raise SharedAssetDatabaseError("asset_id must be non-empty")
        value = self.runtime.assets.get(key)
        if not isinstance(value, Mapping):
            raise SharedAssetDatabaseError(
                f"asset {key!r} is outside shared DB snapshot {self.snapshot_id!r}"
            )
        return _public_asset(value, asset_id=key)

    def search(
        self,
        query: str,
        *,
        size_constraint: Sequence[float] | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        text = " ".join(str(query or "").split())
        if not text or len(text) > 1000:
            raise SharedAssetDatabaseError(
                "search query must contain 1-1000 normalized characters"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise SharedAssetDatabaseError("top_k must be an integer")
        if top_k < 1 or top_k > self.max_top_k:
            raise SharedAssetDatabaseError(
                f"top_k must be between 1 and {self.max_top_k}"
            )
        normalized_size = _optional_positive_vec3(size_constraint)
        embeddings = self.runtime.embeddings
        if embeddings is None:
            raise SharedAssetDatabaseError("shared DB index is unavailable")
        encoded = np.asarray(self.runtime.encode_query(text)).reshape(-1)
        if encoded.shape[0] != embeddings.shape[1]:
            raise SharedAssetDatabaseError("query/index dimension mismatch")
        norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(encoded) + 1e-8
        cosine_scores = embeddings @ encoded / norms
        tolerance = float(self.runtime.composed.profile.policy.size_tolerance)
        scored: list[tuple[float, int, str]] = []
        for index, asset_id in enumerate(self.runtime.jid_list):
            asset = self.runtime.assets[asset_id]
            score = float(cosine_scores[index]) + _compute_size_score(
                asset.get("size"), normalized_size, tolerance
            )
            scored.append((score, index, asset_id))
        # Python's stable sort preserves declared index order for score ties.
        scored.sort(key=lambda item: item[0], reverse=True)
        minimum = float(self.runtime.composed.profile.policy.min_score)
        eligible = [item for item in scored if item[0] >= minimum]
        chosen = (eligible if eligible else scored)[:top_k]
        results: list[dict[str, Any]] = []
        for rank, (score, index, asset_id) in enumerate(chosen, start=1):
            results.append(
                {
                    "rank": rank,
                    "score": score,
                    "index_row": index,
                    **_public_asset(self.runtime.assets[asset_id], asset_id=asset_id),
                }
            )
        return {
            "schema_version": "non_rectangular_agent_asset_search_results_v1",
            "catalog_snapshot_id": self.snapshot_id,
            "query": text,
            "size_constraint": normalized_size,
            "top_k_requested": top_k,
            "result_count": len(results),
            "results": results,
        }


def open_shared_asset_database(
    *,
    catalog_path: str | Path,
    resource_bindings_path: str | Path | None = None,
    retrieval_profile_id: str = DEFAULT_RETRIEVAL_PROFILE_ID,
    max_top_k: int = 12,
    environ: Mapping[str, str] | None = None,
    encoder: Callable[[str], np.ndarray] | None = None,
) -> SharedAssetDatabase:
    """Build and strictly gate the one shared DB before any Agent launches."""

    runtime = build_runtime(
        catalog_path=Path(catalog_path).expanduser().resolve(),
        retrieval_profile_id=retrieval_profile_id,
        resource_bindings_path=resource_bindings_path,
        environ=os.environ if environ is None else environ,
        encoder=encoder,
    )
    return SharedAssetDatabase(runtime, max_top_k=max_top_k)


def shared_database_static_contract(
    *,
    catalog_path: str | Path,
    retrieval_profile_id: str = DEFAULT_RETRIEVAL_PROFILE_ID,
    max_top_k: int = 12,
) -> dict[str, Any]:
    """Resolve the frozen DB descriptor without bindings, model load, or network."""

    path = Path(catalog_path).expanduser().resolve()
    catalog = RetrievalCatalog.load(path)
    composed = catalog.compose(retrieval_profile_id)
    payload = {
        "schema_version": SHARED_DB_SCHEMA_VERSION,
        "mode": SHARED_DB_MODE,
        "retrieval_profile_id": retrieval_profile_id,
        "dataset_id": composed.dataset.dataset_id,
        "asset_namespace": composed.dataset.asset_namespace,
        "index_id": composed.index.index_id,
        "encoder_id": composed.encoder.encoder_id,
        "encoder_revision": composed.encoder.revision,
        "expected_asset_count": int(composed.index.expected_rows),
        "expected_dimension": int(composed.index.expected_dimension),
        "max_top_k": int(max_top_k),
        "resource_content_sha256": {
            "retrieval_catalog": _sha256_file(path),
            "metadata_index": composed.index.metadata_file.sha256,
            "embedding_matrix": composed.index.matrix_file.sha256,
            "encoder_snapshot": composed.encoder.snapshot_manifest_sha256,
            "golden_suite": composed.profile.golden_suite.sha256,
        },
        "selection_policy": "agent_selects_from_deterministic_bounded_topk_v1",
        "per_scene_assets_prefrozen": False,
        "external_asset_sources_allowed": False,
    }
    digest = _sha256_mapping(payload)
    return {
        **payload,
        "snapshot_id": f"imaginarium-shared-agent-db-v1-{digest[:16]}",
        "snapshot_sha256": digest,
    }


def _runtime_manifest(
    runtime: SharedRetrieverRuntime, *, max_top_k: int
) -> dict[str, Any]:
    static = shared_database_static_contract(
        catalog_path=runtime.composed.catalog_path,
        retrieval_profile_id=runtime.profile_id,
        max_top_k=max_top_k,
    )
    asset_order_sha = hashlib.sha256(
        json.dumps(
            list(runtime.jid_list),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        **static,
        "asset_order_sha256": asset_order_sha,
        "runtime_provenance": runtime.public_provenance(),
    }
    # The public snapshot identity is fixed entirely by the declared catalog,
    # metadata/index/encoder/golden hashes and Agent query policy above. Runtime
    # provenance and the independently derived ordered-ID hash are attestations,
    # not a second snapshot identity that appears only after resources load.
    return payload


def _public_asset(value: Mapping[str, Any], *, asset_id: str) -> dict[str, Any]:
    size = _optional_positive_vec3(value.get("size"))
    if size is None:
        raise SharedAssetDatabaseError(f"asset {asset_id!r} lacks a positive bbox size")
    category = str(value.get("category") or "").strip()
    description = str(value.get("description") or "").strip()
    short_desc = str(value.get("short_desc") or "").strip()
    if not category:
        category = "uncategorized"
    if not description:
        description = short_desc or category
    if not short_desc:
        short_desc = description
    return {
        "asset_id": asset_id,
        "jid": asset_id,
        "category": category,
        "description": description,
        "short_desc": short_desc,
        "size": size,
        "bbox_center_local": [0.0, 0.0, 0.0],
    }


def _optional_positive_vec3(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise SharedAssetDatabaseError("size_constraint must be a positive 3-vector")
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise SharedAssetDatabaseError("size_constraint must contain numbers")
        number = float(item)
        if not math.isfinite(number) or number <= 0.0:
            raise SharedAssetDatabaseError(
                "size_constraint must contain positive finite numbers"
            )
        output.append(number)
    return output


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_mapping(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "DEFAULT_RETRIEVAL_PROFILE_ID",
    "SHARED_DB_MODE",
    "SHARED_DB_SCHEMA_VERSION",
    "SharedAssetDatabase",
    "SharedAssetDatabaseError",
    "open_shared_asset_database",
    "shared_database_static_contract",
]
