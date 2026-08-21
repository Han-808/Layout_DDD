"""Single shared deterministic cosine + size-soft-score Top-1 runtime v2."""

from __future__ import annotations

import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._common import RetrievalContractError, sha256_file
from .bindings import LocalResourceBindings
from .profile import ComposedRetrievalProfile
from .provenance import retrieval_source_manifest


# These constants and the implementation of ``_compute_size_score`` and
# ``retrieve`` intentionally preserve the frozen v1 runtime's numerical and
# stable-order semantics.
_SIZE_SCORE_CLOSE = 0.08
_SIZE_SCORE_MODERATE = 0.03
_SIZE_PENALTY_FAR = -0.02
_SIZE_PENALTY_VERY_FAR = -0.08


def _compute_size_score(
    asset_size: Sequence[float] | None,
    target_size: Sequence[float] | None,
    tolerance: float,
) -> float:
    if not asset_size or not target_size or len(asset_size) != 3 or len(target_size) != 3:
        return 0.0
    valid_diffs: list[float] = []
    for asset_dim, target_dim in zip(asset_size, target_size):
        if target_dim <= 0 or asset_dim <= 0:
            continue
        valid_diffs.append(abs(float(np.log(float(asset_dim) / float(target_dim)))))
    if not valid_diffs:
        return 0.0
    mean_diff = float(np.mean(valid_diffs))
    if mean_diff <= np.log(1 + tolerance):
        return _SIZE_SCORE_CLOSE
    if mean_diff <= np.log(1 + tolerance * 2):
        return _SIZE_SCORE_MODERATE
    if mean_diff <= np.log(1 + tolerance * 3):
        return _SIZE_PENALTY_FAR
    return _SIZE_PENALTY_VERY_FAR


class SharedRetrieverRuntime:
    """One path-independent runtime composed from four reviewed descriptors."""

    def __init__(
        self,
        composed: ComposedRetrievalProfile,
        bindings: LocalResourceBindings,
        *,
        encoder: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.composed = composed
        self.bindings = bindings
        self._encoder = encoder
        self._model: Any = None
        self.assets: dict[str, dict[str, Any]] = {}
        self.jid_list: list[str] = []
        self.embeddings: np.ndarray | None = None
        self._resource_observations: dict[str, Any] = {}

    @property
    def embedding_model_name(self) -> str:
        return self.composed.encoder.upstream_model_id

    @property
    def profile_id(self) -> str:
        return self.composed.profile.retrieval_profile_id

    def _load_index(self) -> None:
        index = self.composed.index
        metadata_path = self.bindings.require(index.metadata_file.resource_id)
        matrix_path = self.bindings.require(index.matrix_file.resource_id)
        self.assets, self.jid_list, self.embeddings = index.load_and_validate(
            dataset=self.composed.dataset,
            metadata_path=metadata_path,
            matrix_path=matrix_path,
        )
        self._resource_observations.update(
            {
                index.metadata_file.resource_id: index.metadata_file.sha256,
                index.matrix_file.resource_id: index.matrix_file.sha256,
            }
        )

    def _validate_encoder_snapshot(self) -> Path:
        descriptor = self.composed.encoder
        root = self.bindings.require(descriptor.model_resource_id)
        observation = descriptor.validate_snapshot(root)
        self._resource_observations[descriptor.model_resource_id] = observation[
            "snapshot_manifest_sha256"
        ]
        return root

    def _get_encoder(self) -> Callable[[str], np.ndarray]:
        if self._encoder is not None:
            return self._encoder
        if self._model is None:
            model_root = self.bindings.require(self.composed.encoder.model_resource_id)
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            torch = importlib.import_module("torch")
            torch.manual_seed(0)
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            torch.use_deterministic_algorithms(True)
            sentence_transformers = importlib.import_module("sentence_transformers")
            self._model = sentence_transformers.SentenceTransformer(
                str(model_root),
                device="cpu",
                local_files_only=True,
            )

        def encode(text: str) -> np.ndarray:
            value = self._model.encode(
                text,
                prompt_name=self.composed.encoder.prompt_name,
                convert_to_numpy=True,
            )
            return np.asarray(value)

        return encode

    def encode_query(self, description: str) -> np.ndarray:
        text = " ".join(str(description or "").split())
        if not text:
            raise RetrievalContractError("retrieval description must be non-empty")
        vector = np.asarray(self._get_encoder()(text)).reshape(-1)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise RetrievalContractError("query encoder returned a malformed vector")
        if vector.shape[0] != self.composed.encoder.expected_dimension:
            raise RetrievalContractError(
                f"query dimension {vector.shape[0]} differs from encoder contract "
                f"{self.composed.encoder.expected_dimension}"
            )
        return vector

    def retrieve(
        self,
        description: str,
        *,
        size_constraint: Sequence[float] | None,
    ) -> dict[str, Any]:
        if self.embeddings is None:
            self._load_index()
        assert self.embeddings is not None
        query = self.encode_query(description)
        if query.shape[0] != self.embeddings.shape[1]:
            raise RetrievalContractError(
                f"query dimension {query.shape[0]} != index dimension {self.embeddings.shape[1]}"
            )
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query) + 1e-8
        cosine_scores = self.embeddings @ query / norms
        tolerance = self.composed.profile.policy.size_tolerance
        scored: list[tuple[float, int, str]] = []
        for index, jid in enumerate(self.jid_list):
            asset = self.assets[jid]
            score = float(cosine_scores[index])
            score += _compute_size_score(asset.get("size"), size_constraint, tolerance)
            scored.append((score, index, jid))
        scored.sort(key=lambda item: item[0], reverse=True)
        min_score = self.composed.profile.policy.min_score
        eligible = [item for item in scored if item[0] >= min_score]
        score, index, jid = (eligible or scored)[:1][0]
        return {
            **self.assets[jid],
            "rank": 1,
            "score": score,
            "index_row": index,
        }

    def retrieve_batch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        rows = request.get("requests")
        if not isinstance(rows, list) or not rows:
            raise RetrievalContractError("batch request must contain a non-empty requests list")
        seen: set[str] = set()
        results: list[dict[str, Any]] = []
        for order, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise RetrievalContractError(f"requests[{order}] must be an object")
            slot_id = str(row.get("slot_id") or "").strip()
            if not slot_id or slot_id in seen:
                raise RetrievalContractError("slot_id must be non-empty and unique")
            seen.add(slot_id)
            result = self.retrieve(
                str(row.get("retrieval_query") or ""),
                size_constraint=row.get("size_constraint"),
            )
            results.append(
                {
                    "order": order,
                    "slot_id": slot_id,
                    "retrieval_query": row["retrieval_query"],
                    "size_constraint": row.get("size_constraint"),
                    "invocation_count": 1,
                    "rank1": result,
                    "accepted_as_frozen_outcome": True,
                }
            )
        return {
            "schema_version": "hy34_frozen_top1_results_v1",
            "total_invocations": len(results),
            "retry_count": 0,
            "asset_replacement_count": 0,
            "results": results,
        }

    def _load_golden(self) -> list[dict[str, Any]]:
        descriptor = self.composed.profile.golden_suite
        path = (self.composed.catalog_path.parent / descriptor.path).resolve()
        try:
            path.relative_to(self.composed.catalog_path.parent)
        except ValueError as exc:
            raise RetrievalContractError("golden suite escapes retrieval catalog root") from exc
        if not path.is_file() or sha256_file(path) != descriptor.sha256:
            raise RetrievalContractError("golden suite content hash differs")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise RetrievalContractError(f"invalid golden suite: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != "generation_retrieval_golden_v2":
            raise RetrievalContractError("unsupported golden suite schema")
        rows = value.get("queries")
        if not isinstance(rows, list) or not rows:
            raise RetrievalContractError("golden suite must contain queries")
        seen: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise RetrievalContractError(f"golden query {index} must be an object")
            required = {
                "id",
                "description",
                "size_constraint",
                "expected_asset_id",
                "expected_score",
            }
            if set(row) != required:
                raise RetrievalContractError(f"golden query {index} keys differ")
            query_id = str(row["id"] or "").strip()
            description = str(row["description"] or "").strip()
            asset_id = str(row["expected_asset_id"] or "").strip()
            score = row["expected_score"]
            if not query_id or query_id in seen or not description or not asset_id:
                raise RetrievalContractError(f"golden query {index} identity/text is invalid")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
                raise RetrievalContractError(f"golden query {index} score is invalid")
            if asset_id not in self.assets:
                raise RetrievalContractError(
                    f"golden query {query_id!r} references an unknown asset"
                )
            seen.add(query_id)
            normalized.append(dict(row))
        self._resource_observations[
            f"golden:{self.profile_id}"
        ] = descriptor.sha256
        return normalized

    def gate(self, *, strict: bool | None = None, run_golden: bool = True) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        golden_results: list[dict[str, Any]] = []
        effective_strict = self.composed.profile.official_strict if strict is None else bool(strict)
        if self.composed.profile.official_strict:
            effective_strict = True
        try:
            self._load_index()
            self._validate_encoder_snapshot()
            package_drift = self.composed.encoder.package_drift()
            for item in package_drift:
                warnings.append({"code": "package_version_drift", "details": item})
            if run_golden:
                for item in self._load_golden():
                    result = self.retrieve(
                        str(item["description"]),
                        size_constraint=item.get("size_constraint"),
                    )
                    delta = abs(float(result["score"]) - float(item["expected_score"]))
                    row = {
                        "id": item["id"],
                        "expected_asset_id": item["expected_asset_id"],
                        "actual_asset_id": result["jid"],
                        "expected_score": item["expected_score"],
                        "actual_score": result["score"],
                        "score_delta": delta,
                    }
                    golden_results.append(row)
                    if result["jid"] != item["expected_asset_id"]:
                        warnings.append({"code": "golden_top1_drift", "details": row})
                    elif delta > self.composed.profile.golden_suite.score_tolerance:
                        warnings.append({"code": "golden_score_drift", "details": row})
        except Exception as exc:
            errors.append(
                {
                    "code": "retrieval_resource_gate_failed",
                    "error_type": type(exc).__name__,
                    # Gate reports are public provenance.  Third-party loader
                    # exceptions can contain a local binding or cache path, so
                    # the raw exception text is deliberately not serialized.
                    "message": "retrieval resources failed contract validation",
                }
            )
        if effective_strict and warnings:
            errors.extend(
                {
                    "code": item["code"],
                    "message": "strict retrieval profile rejected a compatibility warning",
                    "details": item.get("details", {}),
                }
                for item in warnings
            )
        status = "failed" if errors else ("ready_with_warnings" if warnings else "ready")
        return {
            "schema_version": "generation_retrieval_gate_report_v2",
            "status": status,
            "strict": effective_strict,
            "errors": errors,
            "warnings": warnings,
            "observed": self.public_provenance(),
            "golden_results": golden_results,
        }

    def public_provenance(self) -> dict[str, Any]:
        return {
            "schema_version": "generation_retrieval_provenance_v2",
            "retrieval_profile_id": self.profile_id,
            "catalog_sha256": self.composed.catalog_sha256,
            "profile_sha256": self.composed.profile_sha256,
            "dataset_id": self.composed.dataset.dataset_id,
            "asset_namespace": self.composed.dataset.asset_namespace,
            "encoder_id": self.composed.encoder.encoder_id,
            "embedding_model": self.composed.encoder.upstream_model_id,
            "encoder_revision": self.composed.encoder.revision,
            "index_id": self.composed.index.index_id,
            "resource_content_sha256": dict(sorted(self._resource_observations.items())),
            "runtime_source_sha256": retrieval_source_manifest()["manifest_sha256"],
        }
