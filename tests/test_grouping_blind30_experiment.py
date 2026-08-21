from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import yaml

from scripts.build_grouping_blind30_review import build_review
from scripts.grouping_blind30_contracts import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentPaths,
    atomic_write_json,
    load_experiment_config,
    read_json,
)
from scripts.grouping_blind30_dataset import (
    materialize_case,
    prepare_dataset,
    stratified_random_sample,
)
from scripts.grouping_blind30_runtime import run_grouping_backends
from scripts.serve_grouping_blind30_review import (
    HUMAN_REVIEW_SCHEMA_VERSION,
    validate_review_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def _scene(
    scene_id: str,
    *,
    scene_type: str,
    object_count: int,
) -> dict:
    objects = []
    for index in range(object_count):
        objects.append(
            {
                "jid": f"asset_{index}",
                "short_desc": f"object {index}",
                "desc": f"object {index}",
                "center": [
                    0.7 + index % 5 * 1.3,
                    0.7 + index // 5 * 1.3,
                    0.5,
                ],
                "size": [0.8, 0.8, 1.0],
                "rotation": [0.0, 0.0, float(index * 7)],
            }
        )
    return {
        "scene_id": scene_id,
        "scene_type": scene_type,
        "boundary": [[0, 0], [8, 0], [8, 8], [0, 8]],
        "scene_height": 2.8,
        "objects": objects,
    }


def _write_scene(
    root: Path,
    scene_id: str,
    *,
    scene_type: str,
    object_count: int,
) -> Path:
    path = root / f"{scene_id}.json"
    path.write_text(
        json.dumps(
            _scene(
                scene_id,
                scene_type=scene_type,
                object_count=object_count,
            )
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.requires_local_data
def test_reference_config_is_fixed_30_scene_three_backend_experiment() -> None:
    config, paths = load_experiment_config(
        ROOT
        / "configs"
        / "experiments"
        / "grouping_blind30_gpt56_v1.yaml",
        repo_root=ROOT,
    )

    assert config["schema_version"] == EXPERIMENT_SCHEMA_VERSION
    assert config["sample"]["size"] == 30
    assert sum(
        item["sample_count"] for item in config["sample"]["strata"]
    ) == 30
    assert set(config["backends"]) == {"topology", "anchor", "vlm"}
    assert config["model"]["endpoint"] == "http://127.0.0.1:4010/v1"
    assert config["model"]["model"] == "gpt-5.6-sol"
    assert config["model"]["api_key_env"] == "LITELLM_MASTER_KEY"
    assert "OPENAPI_GPT_KEY" not in json.dumps(config)
    vlm_config_path = ROOT / config["backend_config_files"]["vlm"]
    vlm_config = yaml.safe_load(
        vlm_config_path.read_text(encoding="utf-8")
    )
    assert (
        vlm_config["grouping"]["vlm"]["response_format_json"] is False
    )
    assert paths.output_root.name == (
        "grouping_blind30_gpt56_20260730_r1"
    )


@pytest.mark.requires_local_data
def test_checked_in_frozen_sample_reproduces_exact_30_cases(
    tmp_path: Path,
) -> None:
    config, _ = load_experiment_config(
        ROOT
        / "configs"
        / "experiments"
        / "grouping_blind30_gpt56_v1.yaml",
        repo_root=ROOT,
    )
    paths = ExperimentPaths(
        repo_root=ROOT,
        output_root=tmp_path / "frozen_sample",
    )

    dataset = prepare_dataset(config, paths, resume=False)

    assert len(dataset["cases"]) == 30
    assert dataset["dataset_fingerprint"] == (
        "0d0c191932d9dda2344a7d0b66c2b58413bb2e03fd3822db8112124f0d39fac3"
    )
    assert {
        case["stratum"] for case in dataset["cases"]
    } == {
        "05-10_objects",
        "11-20_objects",
        "21-30_objects",
        "31-45_objects",
        "46plus_objects",
    }
    assert (
        paths.case_root("case_030")
        / "input"
        / "identity_map.png"
    ).is_file()


def test_stratified_sampling_is_randomized_reproducible_and_balanced(
    tmp_path: Path,
) -> None:
    for index in range(18):
        _write_scene(
            tmp_path,
            f"small_{index:02d}",
            scene_type=f"type_{index % 4}",
            object_count=5 + index % 3,
        )
    for index in range(18):
        _write_scene(
            tmp_path,
            f"large_{index:02d}",
            scene_type=f"type_{index % 5}",
            object_count=20 + index % 4,
        )
    strata = [
        {
            "name": "small",
            "min_objects": 5,
            "max_objects": 10,
            "sample_count": 4,
        },
        {
            "name": "large",
            "min_objects": 20,
            "max_objects": None,
            "sample_count": 4,
        },
    ]

    first = stratified_random_sample(
        tmp_path,
        sample_size=8,
        strata=strata,
        seed=17,
    )
    second = stratified_random_sample(
        tmp_path,
        sample_size=8,
        strata=strata,
        seed=17,
    )
    different = stratified_random_sample(
        tmp_path,
        sample_size=8,
        strata=strata,
        seed=19,
    )

    assert first == second
    assert first != different
    assert [item["stratum"] for item in first].count("small") == 4
    assert [item["stratum"] for item in first].count("large") == 4
    assert len({item["source_path"] for item in first}) == 8
    assert len({item["scene_type"] for item in first}) >= 4


class _GroupingModel:
    model_id = "fake-vlm"
    endpoint = "http://127.0.0.1:4010/v1"
    last_request_metadata = {
        "model": "fake-vlm",
        "image_count": 3,
    }

    def chat_messages(self, messages, **kwargs) -> str:
        del messages, kwargs
        return json.dumps(
            {
                "object_groups": [
                    {
                        "object_ids": [
                            "scene_object_000",
                            "scene_object_001",
                        ],
                        "label": "left local scope",
                        "anchor_object_id": None,
                        "reason": "The two objects share one local area.",
                    },
                    {
                        "object_ids": [
                            "scene_object_002",
                            "scene_object_003",
                        ],
                        "label": "right local scope",
                        "anchor_object_id": None,
                        "reason": "The two objects share another local area.",
                    },
                ],
                "reason": "Two complete local evidence scopes.",
            }
        )


def test_all_three_backends_feed_one_blind_review_without_method_leakage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _write_scene(
        tmp_path,
        "scene_test",
        scene_type="living room",
        object_count=4,
    )
    output_root = tmp_path / "output"
    paths = ExperimentPaths(repo_root=ROOT, output_root=output_root)
    case = {
        "case_id": "case_001",
        "source_scene_id": "scene_test",
        "source_path": str(source_path),
        "source_sha256": _sha256(source_path),
        "scene_type": "living room",
        "object_count": 4,
        "stratum": "test",
    }
    config = {
        "_experiment_id": "blind_test",
        "backends": ["topology", "anchor", "vlm"],
        "backend_config_files": {
            "topology": (
                "configs/grouping/topology_metadata_geometry_v2.yaml"
            ),
            "anchor": "configs/grouping/anchor_object_v1.yaml",
            "vlm": "configs/grouping/vlm_semantic_partition_v1.yaml",
        },
        "model": {
            "name": "fake",
            "endpoint": "http://127.0.0.1:4010/v1",
            "model": "fake-vlm",
            "api_key_env": "LITELLM_MASTER_KEY",
            "max_tokens": 2000,
            "context_length": 32000,
            "timeout_seconds": 10,
            "response_format_json": False,
            "max_retries": 0,
            "retry_backoff_seconds": 0.0,
            "max_tokens_field": "max_tokens",
            "send_temperature": False,
        },
    }
    materialize_case(config, paths, case, resume=False)
    input_manifest = read_json(
        paths.case_root("case_001")
        / "input"
        / "input_manifest.json"
    )
    render_root = paths.case_root("case_001") / "render"
    render_root.mkdir(parents=True)
    views = []
    for name, color in (
        ("top", (60, 80, 100)),
        ("perspective", (80, 100, 120)),
    ):
        image_path = render_root / f"standardized_{name}.png"
        Image.new("RGB", (96, 96), color).save(image_path)
        views.append({"name": name, "path": str(image_path)})
    atomic_write_json(
        render_root / "render_manifest.json",
        {"views": views},
    )
    atomic_write_json(
        paths.method_key,
        {
            "cases": {
                "case_001": {
                    "A": "anchor",
                    "B": "vlm",
                    "C": "topology",
                }
            }
        },
    )
    monkeypatch.setattr(
        "scripts.grouping_blind30_runtime._build_model",
        lambda value: _GroupingModel(),
    )

    failures = run_grouping_backends(
        config,
        paths,
        {"cases": [case]},
        resume=False,
        continue_on_error=False,
        endpoint_override=None,
        model_override=None,
        api_key_env_override=None,
    )

    assert failures == []
    for backend in ("topology", "anchor", "vlm"):
        result = read_json(
            paths.case_root("case_001")
            / "grouping"
            / backend
            / "result.json"
        )
        assert result["status"] == "complete"
        assigned = [
            object_id
            for group in result["result"]["object_groups"]
            for object_id in group["object_ids"]
        ]
        assert sorted(assigned) == [
            "scene_object_000",
            "scene_object_001",
            "scene_object_002",
            "scene_object_003",
        ]

    public = build_review(
        config=config,
        paths=paths,
        dataset={
            "dataset_fingerprint": "test",
            "cases": [case],
        },
        allow_incomplete=False,
    )
    public_text = json.dumps(public).lower()
    assert '"backend"' not in public_text
    assert '"policy_id"' not in public_text
    assert '"model"' not in public_text
    assert '"endpoint"' not in public_text
    assert "topology_metadata_geometry_v2" not in public_text
    assert "anchor_object_partition_v1" not in public_text
    assert "vlm_semantic_partition_v1" not in public_text
    assert [
        item["blind_result_id"]
        for item in public["cases"][0]["variants"]
    ] == ["A", "B", "C"]
    assert (paths.review_root / "index.html").is_file()


def test_review_backend_validates_blind_ids_and_values() -> None:
    review_data = {
        "experiment_id": "blind_test",
        "cases": [{"case_id": "case_001"}],
    }
    payload = {
        "schema_version": HUMAN_REVIEW_SCHEMA_VERSION,
        "experiment_id": "blind_test",
        "answers": {
            "case_001": {
                "reviewed": True,
                "best_result": "B",
                "notes": "B makes the cleanest split.",
                "variants": {
                    "A": {"quality": "incorrect", "notes": ""},
                    "B": {"quality": "correct", "notes": ""},
                    "C": {
                        "quality": "partially_correct",
                        "notes": "",
                    },
                },
            }
        },
    }

    assert (
        validate_review_payload(
            payload,
            review_data=review_data,
        )
        == payload
    )
    broken = json.loads(json.dumps(payload))
    broken["answers"]["case_001"]["best_result"] = "topology"
    with pytest.raises(ValueError, match="best_result"):
        validate_review_payload(broken, review_data=review_data)


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
