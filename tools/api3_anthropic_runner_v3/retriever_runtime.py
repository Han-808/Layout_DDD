#!/usr/bin/env python3
"""Frozen Imaginarium query-encoder runtime used by the HY3/HY4 runner.

The default gate is deliberately compatibility-oriented: corrupt or unusable
frozen inputs are fatal, while package-version and golden-score drift are
reported as warnings. ``--strict`` promotes warnings to a non-zero exit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


_SIZE_SCORE_CLOSE = 0.08
_SIZE_SCORE_MODERATE = 0.03
_SIZE_PENALTY_FAR = -0.02
_SIZE_PENALTY_VERY_FAR = -0.08


class FrozenRetrieverError(RuntimeError):
    """Raised when the frozen retrieval contract cannot be used safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compute_size_score(
    asset_size: Sequence[float] | None,
    target_size: Sequence[float] | None,
    tolerance: float,
) -> float:
    """Match ``benchmark.assets.retriever`` size-soft-score semantics."""

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


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _issue(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, "details": details}


@dataclass(frozen=True)
class RuntimeConfig:
    config_path: Path
    index_stem: Path
    model_path: Path
    golden_queries_path: Path | None
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "RuntimeConfig":
        config_path = Path(path).expanduser().resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != "hy34_frozen_retriever_runtime_v1":
            raise FrozenRetrieverError("unsupported retriever runtime config schema")
        index_stem = Path(str(raw.get("index_stem") or "")).expanduser()
        model_path = Path(str(raw.get("model_path") or "")).expanduser()
        golden = raw.get("golden_queries_path")
        return cls(
            config_path=config_path,
            index_stem=index_stem,
            model_path=model_path,
            golden_queries_path=Path(str(golden)).expanduser() if golden else None,
            raw=raw,
        )


class FrozenRetrieverRuntime:
    """Frozen matrix + frozen query encoder + exact Top-1 policy."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        encoder: Callable[[str], np.ndarray] | None = None,
    ) -> None:
        self.config = config
        self._encoder = encoder
        self._model: Any = None
        self.assets: dict[str, dict[str, Any]] = {}
        self.jid_list: list[str] = []
        self.embeddings: np.ndarray | None = None

    @property
    def index_json_path(self) -> Path:
        return self.config.index_stem.with_suffix(".json")

    @property
    def index_npy_path(self) -> Path:
        return self.config.index_stem.with_suffix(".npy")

    def load_index(self) -> None:
        data = json.loads(self.index_json_path.read_text(encoding="utf-8"))
        assets = data.get("assets")
        jid_list = data.get("jid_list")
        if not isinstance(assets, dict) or not isinstance(jid_list, list):
            raise FrozenRetrieverError("index JSON must contain assets and jid_list")
        self.assets = {str(key): dict(value) for key, value in assets.items()}
        self.jid_list = [str(value) for value in jid_list]
        self.embeddings = np.load(self.index_npy_path, allow_pickle=False)

    def _get_encoder(self) -> Callable[[str], np.ndarray]:
        if self._encoder is not None:
            return self._encoder
        if self._model is None:
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
                # A host process may have initialized its inter-op pool before
                # importing this module. CPU inference remains usable; the
                # golden Top-1 check records behavioral compatibility.
                pass
            torch.use_deterministic_algorithms(True)
            sentence_transformers = importlib.import_module("sentence_transformers")
            model_cls = sentence_transformers.SentenceTransformer
            self._model = model_cls(
                str(self.config.model_path),
                device=str(self.config.raw.get("device") or "cpu"),
                local_files_only=True,
            )

        def encode(text: str) -> np.ndarray:
            value = self._model.encode(
                text,
                prompt_name=str(self.config.raw.get("query_prompt_name") or "query"),
                convert_to_numpy=True,
            )
            return np.asarray(value)

        return encode

    def encode_query(self, description: str) -> np.ndarray:
        text = " ".join(str(description or "").split())
        if not text:
            raise FrozenRetrieverError("retrieval description must be non-empty")
        vector = np.asarray(self._get_encoder()(text)).reshape(-1)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise FrozenRetrieverError("query encoder returned a malformed vector")
        return vector

    def retrieve(
        self,
        description: str,
        *,
        size_constraint: Sequence[float] | None,
    ) -> dict[str, Any]:
        if self.embeddings is None:
            self.load_index()
        assert self.embeddings is not None
        query = self.encode_query(description)
        if query.shape[0] != self.embeddings.shape[1]:
            raise FrozenRetrieverError(
                f"query dimension {query.shape[0]} != index dimension {self.embeddings.shape[1]}"
            )
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query) + 1e-8
        cosine_scores = self.embeddings @ query / norms
        tolerance = float(self.config.raw["retrieval"]["size_tolerance"])
        scored: list[tuple[float, int, str]] = []
        for index, jid in enumerate(self.jid_list):
            asset = self.assets[jid]
            score = float(cosine_scores[index])
            score += _compute_size_score(asset.get("size"), size_constraint, tolerance)
            scored.append((score, index, jid))
        # Python's stable sort preserves frozen jid_list order for exact ties.
        scored.sort(key=lambda item: item[0], reverse=True)
        min_score = float(self.config.raw["retrieval"]["min_score"])
        eligible = [item for item in scored if item[0] >= min_score]
        score, index, jid = (eligible or scored)[:1][0]
        return {
            **self.assets[jid],
            "rank": 1,
            "score": score,
            "index_row": index,
        }

    def gate(self, *, strict: bool = False, run_golden: bool = True) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        observed: dict[str, Any] = {}

        expected_hashes = self.config.raw.get("index_sha256") or {}
        for label, path in (("json", self.index_json_path), ("npy", self.index_npy_path)):
            if not path.is_file():
                errors.append(_issue("index_missing", f"missing frozen index {label}", path=str(path)))
                continue
            actual = _sha256_file(path)
            observed[f"index_{label}_sha256"] = actual
            expected = str(expected_hashes.get(label) or "")
            if expected and actual != expected:
                errors.append(
                    _issue(
                        "index_hash_mismatch",
                        f"frozen index {label} hash does not match",
                        expected=expected,
                        actual=actual,
                    )
                )

        if not self.config.model_path.is_dir():
            errors.append(
                _issue("model_missing", "frozen query encoder directory is missing", path=str(self.config.model_path))
            )
        else:
            observed["model_path"] = str(self.config.model_path)
            observed["model_revision"] = self.config.model_path.name
            expected_revision = str(self.config.raw.get("model_revision") or "")
            if expected_revision and self.config.model_path.name != expected_revision:
                warnings.append(
                    _issue(
                        "model_revision_path_drift",
                        "model directory basename differs from the recorded revision",
                        expected=expected_revision,
                        actual=self.config.model_path.name,
                    )
                )

        for distribution, expected in (self.config.raw.get("package_expectations") or {}).items():
            actual = _package_version(str(distribution))
            observed[f"package:{distribution}"] = actual
            if actual != str(expected):
                warnings.append(
                    _issue(
                        "package_version_drift",
                        f"{distribution} differs from the reference runtime",
                        expected=str(expected),
                        actual=actual,
                    )
                )

        if not errors:
            try:
                self.load_index()
                assert self.embeddings is not None
                expected_shape = tuple(int(value) for value in self.config.raw["expected_embedding_shape"])
                observed["embedding_shape"] = list(self.embeddings.shape)
                observed["asset_count"] = len(self.assets)
                observed["jid_count"] = len(self.jid_list)
                if tuple(self.embeddings.shape) != expected_shape:
                    errors.append(
                        _issue(
                            "embedding_shape_mismatch",
                            "frozen matrix shape does not match the contract",
                            expected=list(expected_shape),
                            actual=list(self.embeddings.shape),
                        )
                    )
                if len(self.assets) != len(self.jid_list) or len(self.jid_list) != self.embeddings.shape[0]:
                    errors.append(
                        _issue(
                            "index_alignment_invalid",
                            "assets, jid_list, and matrix rows are not aligned",
                            assets=len(self.assets),
                            jid_list=len(self.jid_list),
                            rows=int(self.embeddings.shape[0]),
                        )
                    )
                if set(self.assets) != set(self.jid_list):
                    errors.append(_issue("index_identity_invalid", "assets and jid_list identities differ"))
                if not np.isfinite(self.embeddings).all():
                    errors.append(_issue("index_non_finite", "frozen matrix contains non-finite values"))
            except Exception as exc:  # gate must report rather than traceback
                errors.append(_issue("index_load_failed", str(exc), error_type=type(exc).__name__))

        golden_results: list[dict[str, Any]] = []
        if not errors and run_golden and self.config.golden_queries_path:
            try:
                queries = json.loads(self.config.golden_queries_path.read_text(encoding="utf-8"))["queries"]
                tolerance = float(self.config.raw.get("golden_score_tolerance") or 1e-5)
                for item in queries:
                    result = self.retrieve(
                        str(item["description"]),
                        size_constraint=item.get("size_constraint"),
                    )
                    score_delta = abs(float(result["score"]) - float(item["expected_score"]))
                    row = {
                        "id": item["id"],
                        "expected_jid": item["expected_jid"],
                        "actual_jid": result["jid"],
                        "expected_score": item["expected_score"],
                        "actual_score": result["score"],
                        "score_delta": score_delta,
                    }
                    golden_results.append(row)
                    if result["jid"] != item["expected_jid"]:
                        warnings.append(
                            _issue("golden_top1_drift", "golden query Top-1 changed", **row)
                        )
                    elif score_delta > tolerance:
                        warnings.append(
                            _issue("golden_score_drift", "golden query score drifted", **row)
                        )
            except Exception as exc:
                errors.append(_issue("query_encoder_smoke_failed", str(exc), error_type=type(exc).__name__))

        status = "failed" if errors or (strict and warnings) else ("ready_with_warnings" if warnings else "ready")
        return {
            "schema_version": "hy34_frozen_retriever_gate_report_v1",
            "status": status,
            "strict": strict,
            "errors": errors,
            "warnings": warnings,
            "observed": observed,
            "golden_results": golden_results,
        }


def retrieve_batch(runtime: FrozenRetrieverRuntime, request: Mapping[str, Any]) -> dict[str, Any]:
    rows = request.get("requests")
    if not isinstance(rows, list) or not rows:
        raise FrozenRetrieverError("batch request must contain a non-empty requests list")
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for order, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise FrozenRetrieverError(f"requests[{order}] must be an object")
        slot_id = str(row.get("slot_id") or "").strip()
        if not slot_id or slot_id in seen:
            raise FrozenRetrieverError("slot_id must be non-empty and unique")
        seen.add(slot_id)
        result = runtime.retrieve(
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


def _dump_json(value: Mapping[str, Any], output: str) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output == "-":
        sys.stdout.write(text)
    else:
        Path(output).write_text(text, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--strict", action="store_true")
    gate.add_argument("--skip-golden", action="store_true")
    gate.add_argument("--output", default="-")
    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--input", required=True)
    retrieve.add_argument("--output", default="-")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    runtime = FrozenRetrieverRuntime(RuntimeConfig.load(args.config))
    if args.command == "gate":
        report = runtime.gate(strict=bool(args.strict), run_golden=not args.skip_golden)
        _dump_json(report, args.output)
        return 0 if report["status"] != "failed" else 2
    request = json.loads(Path(args.input).read_text(encoding="utf-8"))
    gate = runtime.gate(strict=False, run_golden=False)
    if gate["status"] == "failed":
        raise FrozenRetrieverError(f"runtime gate failed: {gate['errors']}")
    _dump_json(retrieve_batch(runtime, request), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
