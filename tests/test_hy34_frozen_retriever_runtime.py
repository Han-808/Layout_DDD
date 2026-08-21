from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.scene_generation.retrieval import (
    LocalResourceBindings,
    RetrievalCatalog,
    RetrievalContractError,
    build_runtime,
    select_binding_path,
)
from benchmark.scene_generation.retrieval.runtime import (
    _compute_size_score as _v2_size_score,
)
from tools.api3_anthropic_runner_v2 import retriever_runtime as frozen_v1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value) -> str:
    data = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(data).hexdigest()


def _fixture(
    root: Path,
    *,
    prefix: str = "alpha",
    custom_fields: bool = False,
    package_expectations: dict[str, str] | None = None,
    official_strict: bool = False,
):
    root.mkdir(parents=True, exist_ok=True)
    resources = root / "resources"
    resources.mkdir()
    model = resources / "model"
    model.mkdir()
    model_file = model / "model.bin"
    model_file.write_bytes(b"synthetic-model-v2")

    if custom_fields:
        assets = {
            "a": {
                "uuid": "a",
                "caption": "asset a",
                "long_caption": "asset a long",
                "kind": "seat",
                "extent": [1.0, 1.0, 1.0],
            },
            "b": {
                "uuid": "b",
                "caption": "asset b",
                "long_caption": "asset b long",
                "kind": "table",
                "extent": [2.0, 2.0, 2.0],
            },
        }
        field_mapping = {
            "asset_id": "uuid",
            "short_description": "caption",
            "description": "long_caption",
            "category": "kind",
            "size_xyz_m": "extent",
        }
    else:
        assets = {
            "a": {
                "jid": "a",
                "short_desc": "asset a",
                "description": "asset a long",
                "category": "seat",
                "size": [1.0, 1.0, 1.0],
            },
            "b": {
                "jid": "b",
                "short_desc": "asset b",
                "description": "asset b long",
                "category": "table",
                "size": [2.0, 2.0, 2.0],
            },
        }
        field_mapping = {
            "asset_id": "jid",
            "short_description": "short_desc",
            "description": "description",
            "category": "category",
            "size_xyz_m": "size",
        }
    metadata = resources / "index.json"
    metadata.write_text(
        json.dumps({"records": assets, "record_order": ["a", "b"]}),
        encoding="utf-8",
    )
    matrix = resources / "index.npy"
    np.save(matrix, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    golden = root / "golden.json"
    golden.write_text(
        json.dumps(
            {
                "schema_version": "generation_retrieval_golden_v2",
                "queries": [
                    {
                        "id": "smoke",
                        "description": "choose a",
                        "size_constraint": [1.0, 1.0, 1.0],
                        "expected_asset_id": "a",
                        "expected_score": 1.08,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    model_manifest = [
        {
            "path": "model.bin",
            "bytes": model_file.stat().st_size,
            "sha256": _sha256(model_file),
        }
    ]
    ids = {
        "dataset": f"{prefix}-dataset-v2",
        "encoder": f"{prefix}-encoder-v2",
        "index": f"{prefix}-index-v2",
        "profile": f"{prefix}-profile-v2",
        "metadata": f"asset-index-json:{prefix}-v2",
        "matrix": f"asset-index-matrix:{prefix}-v2",
        "model": f"encoder-snapshot:{prefix}-v2",
    }
    catalog = root / "profiles_v2.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": "generation_retrieval_catalog_v2",
                "dataset_descriptors": [
                    {
                        "dataset_id": ids["dataset"],
                        "asset_namespace": f"{prefix}-asset-v2",
                        "metadata_collection_key": "records",
                        "order_key": "record_order",
                        "field_mapping": field_mapping,
                    }
                ],
                "encoder_descriptors": [
                    {
                        "encoder_id": ids["encoder"],
                        "implementation": "sentence_transformers_v2",
                        "model_resource_id": ids["model"],
                        "upstream_model_id": f"Example/{prefix}-encoder",
                        "revision": f"{prefix}-revision-v2",
                        "expected_dimension": 2,
                        "query": {
                            "whitespace_normalization": "collapse_ascii_whitespace_v2",
                            "prompt_name": "query",
                        },
                        "required_files": model_manifest,
                        "snapshot_manifest_sha256": _canonical_hash(model_manifest),
                        "package_expectations": package_expectations or {},
                    }
                ],
                "index_descriptors": [
                    {
                        "index_id": ids["index"],
                        "dataset_id": ids["dataset"],
                        "encoder_id": ids["encoder"],
                        "implementation": "dense_numpy_cosine_v2",
                        "metadata_file": {
                            "resource_id": ids["metadata"],
                            "bytes": metadata.stat().st_size,
                            "sha256": _sha256(metadata),
                        },
                        "matrix_file": {
                            "resource_id": ids["matrix"],
                            "bytes": matrix.stat().st_size,
                            "sha256": _sha256(matrix),
                        },
                        "expected_rows": 2,
                        "expected_dimension": 2,
                        "dtype": "float32",
                        "order_semantics": "declared_stable_order_v2",
                    }
                ],
                "retrieval_profiles": [
                    {
                        "retrieval_profile_id": ids["profile"],
                        "dataset_id": ids["dataset"],
                        "encoder_id": ids["encoder"],
                        "index_id": ids["index"],
                        "official_strict": official_strict,
                        "policy": {
                            "algorithm": "cosine_log_size_stable_top1_v2",
                            "category_argument": None,
                            "size_tolerance": 0.5,
                            "top_k": 1,
                            "min_score": 0.3,
                            "tie_order": "declared_index_order_v2",
                        },
                        "golden_suite": {
                            "path": "golden.json",
                            "sha256": _sha256(golden),
                            "score_tolerance": 1e-6,
                        },
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    bindings = root / "bindings.json"
    bindings.write_text(
        json.dumps(
            {
                "schema_version": "generation_resource_bindings_v2",
                "bindings": {
                    ids["metadata"]: {"path": "resources/index.json"},
                    ids["matrix"]: {"path": "resources/index.npy"},
                    ids["model"]: {"path": "resources/model"},
                },
            }
        ),
        encoding="utf-8",
    )
    runtime = build_runtime(
        catalog_path=catalog,
        retrieval_profile_id=ids["profile"],
        resource_bindings_path=bindings,
        encoder=lambda _text: np.asarray([1.0, 0.0]),
    )
    return runtime, catalog, bindings, matrix, ids


def test_gate_accepts_healthy_v2_runtime(tmp_path: Path) -> None:
    runtime, _catalog, _bindings, _matrix, _ids = _fixture(tmp_path)
    report = runtime.gate()
    assert report["status"] == "ready"
    assert report["errors"] == []
    assert report["golden_results"][0]["actual_asset_id"] == "a"
    assert all("/" not in key for key in report["observed"]["resource_content_sha256"])


def test_package_drift_warns_for_nonofficial_and_fails_for_official(tmp_path: Path) -> None:
    runtime, *_ = _fixture(
        tmp_path / "relaxed",
        package_expectations={"definitely-not-installed-distribution": "1.0"},
    )
    assert runtime.gate(strict=False)["status"] == "ready_with_warnings"
    assert runtime.gate(strict=True)["status"] == "failed"
    strict_runtime, *_ = _fixture(
        tmp_path / "strict",
        prefix="strict",
        package_expectations={"definitely-not-installed-distribution": "1.0"},
        official_strict=True,
    )
    strict_report = strict_runtime.gate(strict=False)
    assert strict_report["status"] == "failed"
    assert strict_report["strict"] is True


def test_retrieval_cli_forwards_strict_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmark.scene_generation.retrieval import cli as retrieval_cli

    seen: list[bool | None] = []

    class Runtime:
        def gate(self, *, strict=None, run_golden=True):
            seen.append(strict)
            return {"status": "ready", "strict": strict}

    monkeypatch.setattr(retrieval_cli, "build_runtime", lambda **_kwargs: Runtime())
    output = tmp_path / "gate.json"
    assert retrieval_cli.main(
        [
            "--catalog",
            str(tmp_path / "catalog.json"),
            "--profile",
            "profile-v2",
            "gate",
            "--strict",
            "--output",
            str(output),
        ]
    ) == 0
    assert seen == [True]


def test_gate_hard_fails_on_corrupt_index(tmp_path: Path) -> None:
    runtime, _catalog, _bindings, matrix, _ids = _fixture(tmp_path)
    matrix.write_bytes(b"corrupt")
    report = runtime.gate()
    assert report["status"] == "failed"
    assert report["errors"][0]["code"] == "retrieval_resource_gate_failed"
    assert str(tmp_path) not in json.dumps(report)


def test_batch_is_exactly_one_stable_top1_invocation_per_slot(tmp_path: Path) -> None:
    runtime, *_ = _fixture(tmp_path)
    result = runtime.retrieve_batch(
        {
            "requests": [
                {
                    "slot_id": "chair",
                    "retrieval_query": "choose a",
                    "size_constraint": [1.0, 1.0, 1.0],
                }
            ]
        }
    )
    assert result["schema_version"] == "hy34_frozen_top1_results_v1"
    assert result["total_invocations"] == 1
    assert result["retry_count"] == 0
    assert result["asset_replacement_count"] == 0
    assert result["results"][0]["rank1"]["jid"] == "a"
    assert result["results"][0]["rank1"]["score"] == pytest.approx(1.08)
    assert result == {
        "schema_version": "hy34_frozen_top1_results_v1",
        "total_invocations": 1,
        "retry_count": 0,
        "asset_replacement_count": 0,
        "results": [
            {
                "order": 0,
                "slot_id": "chair",
                "retrieval_query": "choose a",
                "size_constraint": [1.0, 1.0, 1.0],
                "invocation_count": 1,
                "rank1": {
                    "jid": "a",
                    "short_desc": "asset a",
                    "description": "asset a long",
                    "category": "seat",
                    "size": [1.0, 1.0, 1.0],
                    "rank": 1,
                    "score": 1.0799999900000001,
                    "index_row": 0,
                },
                "accepted_as_frozen_outcome": True,
            }
        ],
    }


def test_arbitrary_dataset_field_mapping_and_encoder_profile(tmp_path: Path) -> None:
    runtime, *_ = _fixture(tmp_path, prefix="other", custom_fields=True)
    report = runtime.gate()
    assert report["status"] == "ready"
    result = runtime.retrieve("choose a", size_constraint=[1.0, 1.0, 1.0])
    assert result["jid"] == "a"
    assert result["short_desc"] == "asset a"
    assert runtime.embedding_model_name == "Example/other-encoder"


def test_relocation_changes_only_local_binding_paths(tmp_path: Path) -> None:
    first, *_ = _fixture(tmp_path / "checkout-a")
    second, *_ = _fixture(tmp_path / "checkout-b")
    request = {
        "requests": [
            {"slot_id": "x", "retrieval_query": "choose a", "size_constraint": None}
        ]
    }
    assert first.retrieve_batch(request) == second.retrieve_batch(request)
    assert first.public_provenance() == second.public_provenance()
    assert str(tmp_path) not in json.dumps(first.public_provenance())


def test_binding_precedence_is_explicit_then_env_then_ignored_local(tmp_path: Path) -> None:
    catalog = tmp_path / "configs" / "retrieval" / "profiles_v2.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("{}", encoding="utf-8")
    explicit = tmp_path / "explicit.json"
    env = tmp_path / "env.json"
    assert select_binding_path(
        catalog_path=catalog,
        explicit_path=explicit,
        environ={"LAYOUT_DDD_RETRIEVAL_BINDINGS": str(env)},
    ) == explicit.resolve()
    assert select_binding_path(
        catalog_path=catalog,
        environ={"LAYOUT_DDD_RETRIEVAL_BINDINGS": str(env)},
    ) == env.resolve()
    assert select_binding_path(catalog_path=catalog, environ={}) == (
        tmp_path / ".runtime" / "retrieval_bindings.local.json"
    ).resolve()


def test_catalog_rejects_cross_descriptor_dimension_drift(tmp_path: Path) -> None:
    _runtime, catalog, *_ = _fixture(tmp_path)
    value = json.loads(catalog.read_text())
    value["index_descriptors"][0]["expected_dimension"] = 3
    catalog.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RetrievalContractError, match="dimensions differ"):
        RetrievalCatalog.load(catalog)


def test_local_binding_public_form_never_contains_paths(tmp_path: Path) -> None:
    _runtime, _catalog, bindings, _matrix, ids = _fixture(tmp_path)
    loaded = LocalResourceBindings.load(bindings)
    public = loaded.public_dict()
    assert set(public["bound_resource_ids"]) == {
        ids["metadata"],
        ids["matrix"],
        ids["model"],
    }
    assert str(tmp_path) not in json.dumps(public)


def test_one_binding_registry_can_serve_two_profiles_without_extra_provenance(
    tmp_path: Path,
) -> None:
    _runtime, catalog, bindings, _matrix, ids = _fixture(tmp_path)
    catalog_value = json.loads(catalog.read_text(encoding="utf-8"))
    second = dict(catalog_value["retrieval_profiles"][0])
    second["retrieval_profile_id"] = "second-profile-v2"
    catalog_value["retrieval_profiles"].append(second)
    catalog.write_text(json.dumps(catalog_value, sort_keys=True), encoding="utf-8")
    binding_value = json.loads(bindings.read_text(encoding="utf-8"))
    binding_value["bindings"]["unused-resource:other-profile-v2"] = {
        "path": "resources/model"
    }
    bindings.write_text(json.dumps(binding_value, sort_keys=True), encoding="utf-8")

    runtimes = [
        build_runtime(
            catalog_path=catalog,
            retrieval_profile_id=profile_id,
            resource_bindings_path=bindings,
            encoder=lambda _text: np.asarray([1.0, 0.0]),
        )
        for profile_id in (ids["profile"], "second-profile-v2")
    ]
    for runtime in runtimes:
        assert runtime.gate()["status"] == "ready"
        observed = runtime.public_provenance()["resource_content_sha256"]
        assert "unused-resource:other-profile-v2" not in observed
        assert set(observed) == {
            ids["metadata"],
            ids["matrix"],
            ids["model"],
            f"golden:{runtime.profile_id}",
        }


@pytest.mark.parametrize(
    ("asset_size", "target_size"),
    [
        ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        ([1.5, 1.5, 1.5], [1.0, 1.0, 1.0]),
        ([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]),
        ([2.5, 2.5, 2.5], [1.0, 1.0, 1.0]),
        ([3.0, 3.0, 3.0], [1.0, 1.0, 1.0]),
        (None, [1.0, 1.0, 1.0]),
    ],
)
def test_v2_size_boundary_math_matches_frozen_v1(
    asset_size: list[float] | None,
    target_size: list[float] | None,
) -> None:
    assert _v2_size_score(asset_size, target_size, 0.5) == (
        frozen_v1._compute_size_score(asset_size, target_size, 0.5)
    )


@pytest.mark.parametrize(
    ("query_vector", "expected"),
    [
        (np.asarray([1.0, 1.0]), "a"),  # exact tie keeps declared order
        (np.asarray([-1.0, 0.0]), "b"),  # all below min: best-score fallback
    ],
)
def test_v2_top1_tie_and_min_fallback_match_frozen_v1(
    tmp_path: Path,
    query_vector: np.ndarray,
    expected: str,
) -> None:
    runtime, _catalog, _bindings, matrix, _ids = _fixture(tmp_path)
    metadata = matrix.with_name("index.json")
    source = json.loads(metadata.read_text(encoding="utf-8"))
    legacy_stem = tmp_path / "legacy_index"
    legacy_stem.with_suffix(".json").write_text(
        json.dumps(
            {
                "assets": source["records"],
                "jid_list": source["record_order"],
            }
        ),
        encoding="utf-8",
    )
    legacy_stem.with_suffix(".npy").write_bytes(matrix.read_bytes())
    legacy = frozen_v1.FrozenRetrieverRuntime(
        frozen_v1.RuntimeConfig(
            config_path=tmp_path / "legacy.json",
            index_stem=legacy_stem,
            model_path=tmp_path / "resources" / "model",
            golden_queries_path=None,
            raw={"retrieval": {"size_tolerance": 0.5, "min_score": 0.3}},
        ),
        encoder=lambda _text: query_vector,
    )
    runtime._encoder = lambda _text: query_vector
    old = legacy.retrieve("  choose\u2003a\n", size_constraint=None)
    new = runtime.retrieve("  choose\u2003a\n", size_constraint=None)
    assert old == new
    assert new["jid"] == expected


def test_unicode_and_ascii_whitespace_normalization_matches_frozen_v1(
    tmp_path: Path,
) -> None:
    runtime, *_ = _fixture(tmp_path)
    old_seen: list[str] = []
    new_seen: list[str] = []
    vector = np.asarray([1.0, 0.0])
    legacy = frozen_v1.FrozenRetrieverRuntime(
        frozen_v1.RuntimeConfig(
            config_path=tmp_path / "legacy.json",
            index_stem=tmp_path / "unused-index",
            model_path=tmp_path / "resources" / "model",
            golden_queries_path=None,
            raw={"retrieval": {"size_tolerance": 0.5, "min_score": 0.3}},
        ),
        encoder=lambda text: old_seen.append(text) or vector,
    )
    runtime._encoder = lambda text: new_seen.append(text) or vector
    source = "\t choose\u2003  asset\n a  "
    assert np.array_equal(legacy.encode_query(source), runtime.encode_query(source))
    assert old_seen == new_seen == ["choose asset a"]


def test_v1_cli_and_shell_replay_interfaces_remain_frozen(tmp_path: Path) -> None:
    model = tmp_path / "model-revision-v1"
    model.mkdir()
    stem = tmp_path / "index"
    metadata = {"assets": {"a": {"jid": "a"}}, "jid_list": ["a"]}
    stem.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(stem.with_suffix(".npy"), np.asarray([[1.0, 0.0]], dtype=np.float32))
    config = {
        "schema_version": "hy34_frozen_retriever_runtime_v1",
        "index_stem": str(stem),
        "model_path": str(model),
        "expected_embedding_shape": [1, 2],
        "index_sha256": {
            "json": _sha256(stem.with_suffix(".json")),
            "npy": _sha256(stem.with_suffix(".npy")),
        },
        "retrieval": {"size_tolerance": 0.5, "min_score": 0.3},
        "package_expectations": {},
    }
    config_path = tmp_path / "runtime.json"
    report_path = tmp_path / "gate.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert frozen_v1.main(
        [
            "--config",
            str(config_path),
            "gate",
            "--skip-golden",
            "--output",
            str(report_path),
        ]
    ) == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "ready"
    assert _sha256(Path("tools/api3_anthropic_runner_v2/retriever_runtime.py")) == (
        "2ace18795781dac83e884528eb4c343052abd50707f834e103669d873261fe20"
    )
    assert _sha256(Path("tools/api3_anthropic_runner_v2/run_retriever.sh")) == (
        "e274e1cc620e62090bcf9c1850b3d530ff88ebc36f803675c82197e529e22b22"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("category_argument", "chair", "category_argument=null"),
        ("size_tolerance", 0.25, "size_tolerance=0.5"),
        ("min_score", 0.2, "min_score=0.3"),
    ],
)
def test_profile_rejects_policy_values_not_executed_by_the_frozen_contract(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    _runtime, catalog, *_ = _fixture(tmp_path)
    raw = json.loads(catalog.read_text(encoding="utf-8"))
    raw["retrieval_profiles"][0]["policy"][field] = value
    catalog.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(RetrievalContractError, match=message):
        RetrievalCatalog.load(catalog)


def test_current_catalog_and_tracked_profile_are_path_and_secret_free() -> None:
    paths = [
        Path("configs/retrieval/profiles_v2.json"),
        Path("configs/generation/api2_kimi_k3_scene10_v2.json"),
        Path("configs/generation/api2_glm53_scene10_v2.json"),
        Path("configs/generation/api3_opus48_high_scene10_v2.json"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "/Users/" not in text
        assert "/hcccao/" not in text
        assert "endpoint" not in text.lower()
        assert "credential" not in text.lower()
        assert "api_key" not in text.lower()


def test_v1_pod_and_golden_remain_exact_frozen_compatibility_mirrors() -> None:
    expected = {
        "tools/api3_anthropic_runner_v2/retriever_runtime.pod.json": (
            "1814be3b7fae3a56b43a310f579a8b9de2dc0ed724892b7ffd77d748770f1831"
        ),
        "tools/api3_anthropic_runner_v2/retriever_golden_v1.json": (
            "28992804c7dd6653dfddbfe95f5d4560039290d16ee38a10384176137a5d28fb"
        ),
    }
    for path_text, digest in expected.items():
        assert _sha256(Path(path_text)) == digest
    current_core = Path(
        "tools/api3_anthropic_runner_v2/generation_runner.py"
    ).read_text(encoding="utf-8")
    assert "retriever_runtime.pod.json" not in current_core
    assert "retriever_golden_v1.json" not in current_core
