from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _runtime_module():
    from tools.hy34_retrieval_conditioned_runner_v1 import retriever_runtime

    return retriever_runtime


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, package_expectations: dict[str, str] | None = None):
    module = _runtime_module()
    stem = tmp_path / "frozen"
    assets = {
        "a": {"jid": "a", "short_desc": "asset a", "size": [1.0, 1.0, 1.0]},
        "b": {"jid": "b", "short_desc": "asset b", "size": [2.0, 2.0, 2.0]},
    }
    stem.with_suffix(".json").write_text(
        json.dumps({"assets": assets, "jid_list": ["a", "b"]}), encoding="utf-8"
    )
    np.save(stem.with_suffix(".npy"), np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    model_path = tmp_path / "model" / "revision-1"
    model_path.mkdir(parents=True)
    golden = tmp_path / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "queries": [
                    {
                        "id": "smoke",
                        "description": "choose a",
                        "size_constraint": [1.0, 1.0, 1.0],
                        "expected_jid": "a",
                        "expected_score": 1.08,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "runtime.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": "hy34_frozen_retriever_runtime_v1",
                "index_stem": str(stem),
                "index_sha256": {
                    "json": _sha256(stem.with_suffix(".json")),
                    "npy": _sha256(stem.with_suffix(".npy")),
                },
                "model_path": str(model_path),
                "model_revision": "revision-1",
                "device": "cpu",
                "query_prompt_name": "query",
                "expected_embedding_shape": [2, 2],
                "retrieval": {
                    "category_argument": None,
                    "size_tolerance": 0.5,
                    "top_k": 1,
                    "min_score": 0.3,
                },
                "golden_queries_path": str(golden),
                "golden_score_tolerance": 1e-6,
                "package_expectations": package_expectations or {},
            }
        ),
        encoding="utf-8",
    )
    runtime = module.FrozenRetrieverRuntime(
        module.RuntimeConfig.load(config_path), encoder=lambda _text: np.asarray([1.0, 0.0])
    )
    return module, runtime, stem


def test_gate_accepts_healthy_frozen_runtime(tmp_path: Path) -> None:
    _module, runtime, _stem = _fixture(tmp_path)
    report = runtime.gate()
    assert report["status"] == "ready"
    assert report["errors"] == []
    assert report["golden_results"][0]["actual_jid"] == "a"


def test_default_gate_warns_but_strict_gate_fails_on_package_drift(tmp_path: Path) -> None:
    _module, runtime, _stem = _fixture(
        tmp_path, package_expectations={"definitely-not-installed-distribution": "1.0"}
    )
    relaxed = runtime.gate(strict=False, run_golden=False)
    strict = runtime.gate(strict=True, run_golden=False)
    assert relaxed["status"] == "ready_with_warnings"
    assert relaxed["warnings"][0]["code"] == "package_version_drift"
    assert strict["status"] == "failed"


def test_default_gate_reports_golden_top1_drift_without_blocking(tmp_path: Path) -> None:
    _module, runtime, _stem = _fixture(tmp_path)
    runtime._encoder = lambda _text: np.asarray([0.0, 1.0])
    relaxed = runtime.gate(strict=False)
    strict = runtime.gate(strict=True)
    assert relaxed["status"] == "ready_with_warnings"
    assert {item["code"] for item in relaxed["warnings"]} == {"golden_top1_drift"}
    assert strict["status"] == "failed"


def test_gate_hard_fails_on_corrupt_index(tmp_path: Path) -> None:
    _module, runtime, stem = _fixture(tmp_path)
    stem.with_suffix(".npy").write_bytes(b"corrupt")
    report = runtime.gate()
    assert report["status"] == "failed"
    assert {item["code"] for item in report["errors"]} == {"index_hash_mismatch"}


def test_batch_is_exactly_one_top1_invocation_per_slot(tmp_path: Path) -> None:
    module, runtime, _stem = _fixture(tmp_path)
    result = module.retrieve_batch(
        runtime,
        {
            "requests": [
                {
                    "slot_id": "chair",
                    "retrieval_query": "choose a",
                    "size_constraint": [1.0, 1.0, 1.0],
                }
            ]
        },
    )
    assert result["total_invocations"] == 1
    assert result["retry_count"] == 0
    assert result["asset_replacement_count"] == 0
    assert result["results"][0]["invocation_count"] == 1
    assert result["results"][0]["rank1"]["jid"] == "a"
