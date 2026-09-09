from __future__ import annotations

from copy import deepcopy
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import threading
from typing import Any, Mapping

import pytest

import benchmark.non_rectangular.resilient as nonrect_resilient_module
from benchmark.materialization.catalog import FrozenCatalog
from benchmark.evaluator.context_projection import (
    EVALUATOR_CONTEXT_PROJECTION_VERSION,
)
from benchmark.models.openai_compatible_model import (
    EndpointConfigurationError,
    EndpointConnectionError,
    EndpointHTTPError,
    EndpointMalformedResponseError,
    OpenAICompatibleModel,
)
from benchmark.non_rectangular.camera import NonRectangularCameraEvidenceExhausted
from benchmark.non_rectangular.evaluator import NonRectangularRoomMetricIncomplete
from benchmark.non_rectangular.materialization import (
    NONRECT_MATERIALIZATION_REVISION,
    NonRectangularMaterializationContractError,
    NonRectangularMaterializationInfrastructureError,
    build_nonrect_room_materialization_plan,
)
from benchmark.non_rectangular.preflight import (
    NonRectangularEvaluationInput,
    prepare_non_rectangular_evaluation,
)
from benchmark.non_rectangular.projection import (
    ROOM_CANONICAL_PROJECTION_VERSION,
)
from benchmark.non_rectangular.resilient import (
    NoAPIMockEvaluatorFactory,
    NoAPIMockMaterializer,
    ResilientCampaignConfig,
    ResilientCampaignError,
    RoomRuntimeContext,
    classify_failure,
    run_resilient_nonrect_campaign,
)
from benchmark.rendering.blender import BlenderRenderError
from benchmark.non_rectangular.room_unit import build_room_evaluation_units
from benchmark.non_rectangular.runtime import (
    APIUsageRecorder,
    DefaultNonRectangularRuntimeFactory,
    RecordingOpenAICompatibleModel,
)
from benchmark.non_rectangular.workflow import (
    L1_METRICS,
    ROOM_REPORT_SCHEMA_VERSION,
)
from benchmark.scene_generation.non_rectangular_multi_room.architecture import (
    build_polygon_architecture,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"


def _fixture(name: str) -> dict[str, Any]:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2) + "\n", encoding="utf-8")


def _catalog(tmp_path: Path, sizes: Mapping[str, list[float]]) -> tuple[Path, Path]:
    asset_root = tmp_path / "assets"
    asset_root.mkdir()
    csv_path = tmp_path / "assets.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["name_en", "category", "caption_en", "short_desc", "bbx"],
        )
        writer.writeheader()
        for asset_id, size in sizes.items():
            writer.writerow(
                {
                    "name_en": asset_id,
                    "category": asset_id,
                    "caption_en": f"fixture {asset_id}",
                    "short_desc": asset_id,
                    "bbx": json.dumps(size),
                }
            )
            directory = asset_root / asset_id
            directory.mkdir()
            (directory / f"{asset_id}.obj").write_text(
                "o fixture\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
                encoding="utf-8",
            )
            _write_json(
                directory / f"{asset_id}_metadata.json",
                {
                    "transformed_size": size,
                    "transformed_bbox_center": [0.0, 0.0, 0.0],
                },
            )
    return csv_path, asset_root


def _generation_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    layout = _fixture("simple_multi_room.json")
    program = _fixture("simple_multi_room_program.json")
    plan = _fixture("simple_multi_room_object_plan.json")
    scene = _fixture("simple_multi_room_scene.json")
    sizes: dict[str, list[float]] = {}
    selection_rooms = []
    for plan_room, scene_room in zip(plan["rooms"], scene["rooms"]):
        selected_objects = []
        by_slot = {item["id"]: item for item in plan_room["objects"]}
        for raw in scene_room["objects"]:
            slot_id = str(raw["slot_id"])
            asset_id = f"asset_{slot_id}"
            size = [float(value) for value in raw["size"]]
            sizes[asset_id] = size
            raw["jid"] = asset_id
            raw["asset_ref"] = {
                "source_db": "imaginarium",
                "asset_key": asset_id,
            }
            raw["geometry_provenance"] = "asset_mesh"
            raw["metadata"] = {"uniform_scale": 1.0}
            selected_objects.append(
                {
                    "slot_id": slot_id,
                    "retrieval_slot_id": f"{plan_room['room_id']}::{slot_id}",
                    "planned_object": deepcopy(by_slot[slot_id]),
                    "retrieval_query": {
                        "description": slot_id,
                        "category": None,
                        "size_constraint": None,
                        "top_k": 1,
                    },
                    "selected_asset": {
                        "jid": asset_id,
                        "category": slot_id,
                        "desc": slot_id,
                        "short_desc": slot_id,
                        "size": size,
                        "asset_ref": {
                            "source_db": "imaginarium",
                            "asset_key": asset_id,
                        },
                        "asset_proxy": {
                            "type": "canonical_catalog_bbox",
                            "bbox_center_local": [0.0, 0.0, 0.0],
                            "bbox_size": size,
                        },
                        "metadata": {},
                    },
                }
            )
        selection_rooms.append(
            {"room_id": plan_room["room_id"], "objects": selected_objects}
        )
    asset_selection = {
        "schema_version": "non_rectangular_asset_selection_v1",
        "layout_id": layout["layout_id"],
        "binding_policy": "room_id_double_colon_slot_id_v1",
        "rooms": selection_rooms,
    }
    model_root = tmp_path / "generation" / "gpt-5.6-sol"
    scene_root = model_root / "scene_fixture"
    _write_json(scene_root / "room_layout.json", layout)
    _write_json(scene_root / "room_program.json", program)
    _write_json(scene_root / "stage_a/object_plan.json", plan)
    _write_json(scene_root / "retrieval/asset_selection.json", asset_selection)
    _write_json(scene_root / "generated_scene.json", scene)
    architecture = build_polygon_architecture(layout)
    _write_json(scene_root / "compiled_architecture.json", architecture)
    preflight = prepare_non_rectangular_evaluation(
        NonRectangularEvaluationInput.from_artifacts(
            room_layout=layout,
            room_program=program,
            object_plan=plan,
            generated_scene=scene,
        )
    )
    _write_json(scene_root / "evaluation_preflight.json", preflight.public_dict())
    csv_path, asset_root = _catalog(tmp_path, sizes)
    return model_root, csv_path, asset_root


def _config(
    tmp_path: Path,
    *,
    model_root: Path,
    csv_path: Path,
    asset_root: Path,
    output_name: str = "evaluation",
    resume: bool = False,
    recover_interrupted: bool = False,
    max_workers: int = 2,
) -> ResilientCampaignConfig:
    return ResilientCampaignConfig.create(
        model_roots={"gpt-5.6-sol": model_root},
        output_root=tmp_path / output_name,
        asset_csv=csv_path,
        asset_root=asset_root,
        catalog_snapshot_id="fixture-catalog-v1",
        blender_bin="/usr/bin/true",
        max_workers=max_workers,
        max_materialization_attempts=3,
        max_room_attempts=3,
        resume=resume,
        recover_interrupted=recover_interrupted,
    )


def test_no_api_generation_materialization_evaluation_and_resume(tmp_path: Path) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    source_hashes_before = {
        path.relative_to(model_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in model_root.rglob("*")
        if path.is_file()
    }
    config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
    )
    first = run_resilient_nonrect_campaign(
        config,
        evaluator_factory=NoAPIMockEvaluatorFactory(),
        materializer_backend=NoAPIMockMaterializer(),
    )

    assert first.status == "complete"
    assert first.room_count == first.complete_room_count == 2
    assert source_hashes_before == {
        path.relative_to(model_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in model_root.rglob("*")
        if path.is_file()
    }
    scene_root = (
        config.output_root
        / "models/gpt-5.6-sol/scenes/scene_fixture"
    )
    run_manifest = json.loads(
        (config.output_root / "run_manifest.json").read_text(encoding="utf-8")
    )
    materialization_identity = run_manifest["identity"]["materialization"]
    assert materialization_identity["materialization_revision"] == (
        NONRECT_MATERIALIZATION_REVISION
    )
    assert materialization_identity["room_canonical_projection_version"] == (
        ROOM_CANONICAL_PROJECTION_VERSION
    )
    assert materialization_identity["evaluator_context_projection_version"] == (
        EVALUATOR_CONTEXT_PROJECTION_VERSION
    )
    scene_report = json.loads(
        (scene_root / "evaluation_report.json").read_text(encoding="utf-8")
    )
    assert scene_report["terminal_status"] == "complete"
    assert scene_report["evaluation_mode"] == "non_rectangular_multi_room"
    for room_id in ("room_000", "room_001"):
        room_root = scene_root / "rooms" / room_id
        materialization = next(
            (room_root / "materialization_attempts").glob(
                "attempt_*/materialization"
            )
        )
        architecture = json.loads(
            (materialization / "architecture_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert architecture["coordinates_transformed"] is False
        assert architecture["adjacent_room_objects_included"] is False
        assert architecture["ceiling_included"] is False
        materialization_manifest = json.loads(
            (materialization / "materialization_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert materialization_manifest["materialization_revision"] == (
            NONRECT_MATERIALIZATION_REVISION
        )
        canonical_scene = json.loads(
            (materialization / "canonical_room_scene.json").read_text(
                encoding="utf-8"
            )
        )
        assert canonical_scene["metadata"]["projection_version"] == (
            ROOM_CANONICAL_PROJECTION_VERSION
        )
        assert canonical_scene["metadata"][
            "evaluator_context_projection_version"
        ] == EVALUATOR_CONTEXT_PROJECTION_VERSION
        mock_camera = next(
            (room_root / "evaluation_attempts").glob(
                "attempt_*/mock_camera/evidence_manifest.json"
            )
        )
        camera_manifest = json.loads(mock_camera.read_text(encoding="utf-8"))
        assert camera_manifest["global_coordinates_preserved"] is True
        assert camera_manifest["adjacent_room_objects_included"] is False
        assert camera_manifest["wall_ids"]

    resumed = run_resilient_nonrect_campaign(
        _config(
            tmp_path,
            model_root=model_root,
            csv_path=csv_path,
            asset_root=asset_root,
            resume=True,
        ),
        evaluator_factory=NoAPIMockEvaluatorFactory(),
        materializer_backend=NoAPIMockMaterializer(),
    )
    assert resumed.status == "complete"
    for room_id in ("room_000", "room_001"):
        room_root = scene_root / "rooms" / room_id
        assert len(list((room_root / "materialization_attempts").glob("attempt_*"))) == 1
        assert len(list((room_root / "evaluation_attempts").glob("attempt_*"))) == 1


def test_resume_refuses_projection_cache_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
    )
    result = run_resilient_nonrect_campaign(
        config,
        evaluator_factory=NoAPIMockEvaluatorFactory(),
        materializer_backend=NoAPIMockMaterializer(),
    )
    assert result.status == "complete"
    run_manifest = config.output_root / "run_manifest.json"
    manifest_before = run_manifest.read_bytes()

    monkeypatch.setattr(
        nonrect_resilient_module,
        "ROOM_CANONICAL_PROJECTION_VERSION",
        "non_rectangular_room_canonical_projection_v1",
    )
    with pytest.raises(
        ResilientCampaignError,
        match="resume refused because input/config identity drifted",
    ):
        run_resilient_nonrect_campaign(
            _config(
                tmp_path,
                model_root=model_root,
                csv_path=csv_path,
                asset_root=asset_root,
                resume=True,
            ),
            evaluator_factory=NoAPIMockEvaluatorFactory(),
            materializer_backend=NoAPIMockMaterializer(),
        )

    assert run_manifest.read_bytes() == manifest_before


def test_no_api_whole_workflow_on_existing_completed_generated_scene(
    tmp_path: Path,
) -> None:
    source = (
        ROOT
        / "Support/artifacts/outputs/e2e_multi_room/"
        "nonrect_spatiallm_selected10_api2_gpt56_kimi_retry5_v2_r1/"
        "gpt-5.6-sol/scene_011568"
    )
    asset_csv = Path(
        "/Users/han_mohan/Desktop/Layout_DDD/Support/Assets/"
        "imaginarium_asset_info.csv"
    )
    asset_root = Path(
        "/Users/han_mohan/Desktop/Layout_DDD/Support/Assets/imaginarium_assets"
    )
    if not source.is_dir() or not asset_csv.is_file() or not asset_root.is_dir():
        pytest.skip("completed local nonrect generation scene/catalog unavailable")
    model_root = tmp_path / "completed-model"
    copied = model_root / source.name
    for relative in (
        "room_layout.json",
        "room_program.json",
        "stage_a/object_plan.json",
        "retrieval/asset_selection.json",
        "generated_scene.json",
        "compiled_architecture.json",
        "evaluation_preflight.json",
    ):
        destination = copied / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
    config = ResilientCampaignConfig.create(
        model_roots={"gpt-5.6-sol": model_root},
        output_root=tmp_path / "completed-evaluation",
        asset_csv=asset_csv,
        asset_root=asset_root,
        catalog_snapshot_id="imaginarium-assets-v1",
        blender_bin="/usr/bin/true",
        max_workers=3,
    )
    result = run_resilient_nonrect_campaign(
        config,
        evaluator_factory=NoAPIMockEvaluatorFactory(),
        materializer_backend=NoAPIMockMaterializer(),
    )
    assert result.status == "complete"
    assert result.scene_count == 1
    assert result.room_count == result.complete_room_count == 9


class _ScriptedEvaluatorFactory(NoAPIMockEvaluatorFactory):
    def __init__(self, failures: Mapping[str, list[BaseException]]) -> None:
        self.failures = {key: list(value) for key, value in failures.items()}
        self.lock = threading.Lock()

    def identity(self) -> Mapping[str, Any]:
        return {
            "provider": "mock",
            "model": "scripted-no-api",
            "config_sha256": hashlib.sha256(b"scripted-no-api-v1").hexdigest(),
        }

    def build(self, context: RoomRuntimeContext):
        delegate = super().build(context)
        owner = self

        class Evaluator:
            def evaluate(self, unit):
                with owner.lock:
                    queue = owner.failures.get(unit.room_id, [])
                    failure = queue.pop(0) if queue else None
                if failure is not None:
                    raise failure
                return delegate.evaluate(unit)

        return Evaluator()


def test_retryable_room_failure_retries_after_initial_pass(tmp_path: Path) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
        output_name="retry",
    )
    result = run_resilient_nonrect_campaign(
        config,
        evaluator_factory=_ScriptedEvaluatorFactory(
            {"room_000": [TimeoutError("transient")]}
        ),
        materializer_backend=NoAPIMockMaterializer(),
    )
    assert result.status == "complete"
    room_root = (
        config.output_root
        / "models/gpt-5.6-sol/scenes/scene_fixture/rooms/room_000"
    )
    assert len(list((room_root / "evaluation_attempts").glob("attempt_*"))) == 2
    events = [
        json.loads(line)
        for line in (config.output_root / "events.jsonl").read_text().splitlines()
    ]
    failed = [item for item in events if item["event"] == "evaluation_failed"]
    assert failed[0]["category"] == "transport_or_timeout"
    assert failed[0]["retryable"] is True


def test_semantic_room_failure_is_not_retried(tmp_path: Path) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
        output_name="semantic",
    )
    result = run_resilient_nonrect_campaign(
        config,
        evaluator_factory=_ScriptedEvaluatorFactory(
            {"room_000": [ValueError("schema mismatch")]}
        ),
        materializer_backend=NoAPIMockMaterializer(),
    )
    assert result.status == "failed"
    room_root = (
        config.output_root
        / "models/gpt-5.6-sol/scenes/scene_fixture/rooms/room_000"
    )
    assert len(list((room_root / "evaluation_attempts").glob("attempt_*"))) == 1
    summary = json.loads((room_root / "summary.json").read_text())
    assert summary["latest_failure"]["retryable"] is False
    assert "schema mismatch" not in json.dumps(summary)


def test_room_retry_taxonomy_is_explicit_infrastructure_only() -> None:
    missing = classify_failure(
        NonRectangularRoomMetricIncomplete("collision incomplete"),
        stage="evaluation",
    )
    assert missing.category == "metric_normalization_failure"
    assert missing.retryable is False

    detailed = classify_failure(
        NonRectangularRoomMetricIncomplete(
            "collision incomplete",
            metric_id="collision",
            source_status="requires_vlm",
            failure_category="judge_response_contract_failure",
            event_keys=("collision:a:b",),
            fallback_rejection_reasons=(
                "collision:a:b:judge_response_contract_failure",
            ),
        ),
        stage="evaluation",
    )
    assert detailed.public_dict() == {
        "stage": "evaluation",
        "category": "judge_response_contract_failure",
        "error_type": "NonRectangularRoomMetricIncomplete",
        "retryable": False,
        "metric_id": "collision",
        "source_status": "requires_vlm",
        "event_keys": ["collision:a:b"],
        "fallback_rejection_reasons": [
            "collision:a:b:judge_response_contract_failure"
        ],
    }

    exhausted = classify_failure(
        NonRectangularCameraEvidenceExhausted("bounded camera exhausted"),
        stage="evaluation",
    )
    assert exhausted.category == "evidence_exhaustion_unclosed"
    assert exhausted.retryable is False

    malformed = classify_failure(
        EndpointMalformedResponseError("malformed"),
        stage="evaluation",
    )
    assert malformed.category == "api_response_contract"
    assert malformed.retryable is False

    unknown = classify_failure(RuntimeError("unknown"), stage="evaluation")
    assert unknown.category == "unclassified_failure"
    assert unknown.retryable is False

    for status in (429, 500, 502, 503, 504):
        http = classify_failure(
            EndpointHTTPError(f"Model endpoint returned HTTP {status}: transient"),
            stage="evaluation",
        )
        assert http.category == "retryable_http"
        assert http.retryable is True

    bad_request = classify_failure(
        EndpointHTTPError("Model endpoint returned HTTP 400: bad request"),
        stage="evaluation",
    )
    assert bad_request.category == "nonretryable_http"
    assert bad_request.retryable is False

    connection = classify_failure(
        EndpointConnectionError("connection reset"),
        stage="evaluation",
    )
    assert connection.category == "transport_or_timeout"
    assert connection.retryable is True

    renderer = classify_failure(
        BlenderRenderError("worker crashed"),
        stage="evaluation",
    )
    assert renderer.category == "service_or_renderer"
    assert renderer.retryable is True


def test_missing_grounded_metric_fails_once_and_later_room_continues(
    tmp_path: Path,
) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
        output_name="missing-grounded-continues",
    )
    result = run_resilient_nonrect_campaign(
        config,
        evaluator_factory=_ScriptedEvaluatorFactory(
            {
                "room_000": [
                    NonRectangularRoomMetricIncomplete("collision incomplete")
                ]
                * 5
            }
        ),
        materializer_backend=NoAPIMockMaterializer(),
    )

    assert result.status == "failed"
    scene_root = config.output_root / "models/gpt-5.6-sol/scenes/scene_fixture"
    failed_root = scene_root / "rooms/room_000"
    assert len(list((failed_root / "evaluation_attempts").glob("attempt_*"))) == 1
    failed_summary = json.loads((failed_root / "summary.json").read_text())
    assert failed_summary["status"] == "failed_nonretryable"
    assert failed_summary["latest_failure"] == {
        "stage": "evaluation",
        "category": "metric_normalization_failure",
        "error_type": "NonRectangularRoomMetricIncomplete",
        "retryable": False,
    }

    later_root = scene_root / "rooms/room_001"
    later_summary = json.loads((later_root / "summary.json").read_text())
    assert later_summary["status"] == "complete"
    assert (later_root / "room_report_selected.json").is_file()
    state = json.loads((config.output_root / "current_state.json").read_text())
    assert state["current_stage"] == "terminal"
    assert state["global_retry_round"] == 0


def test_metric_failure_identity_is_sanitized_in_attempt_and_event(
    tmp_path: Path,
) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
        output_name="metric-failure-identity",
    )
    failure = NonRectangularRoomMetricIncomplete(
        "secret raw response detail",
        metric_id="oob",
        source_status="requires_vlm",
        failure_category="judge_response_contract_failure",
        event_keys=("oob:fixture_object",),
        fallback_rejection_reasons=(
            "oob:fixture_object:judge_response_contract_failure",
        ),
    )
    result = run_resilient_nonrect_campaign(
        config,
        evaluator_factory=_ScriptedEvaluatorFactory(
            {"room_000": [failure]}
        ),
        materializer_backend=NoAPIMockMaterializer(),
    )

    assert result.status == "failed"
    attempt_root = (
        config.output_root
        / "models/gpt-5.6-sol/scenes/scene_fixture/rooms/room_000/"
        "evaluation_attempts/attempt_001"
    )
    manifest = json.loads((attempt_root / "attempt_manifest.json").read_text())
    assert manifest["failure"] == {
        "stage": "evaluation",
        "category": "judge_response_contract_failure",
        "error_type": "NonRectangularRoomMetricIncomplete",
        "retryable": False,
        "metric_id": "oob",
        "source_status": "requires_vlm",
        "event_keys": ["oob:fixture_object"],
        "fallback_rejection_reasons": [
            "oob:fixture_object:judge_response_contract_failure"
        ],
    }
    serialized = json.dumps(manifest)
    assert "secret raw response detail" not in serialized
    events = [
        json.loads(line)
        for line in (config.output_root / "events.jsonl").read_text().splitlines()
    ]
    event = next(item for item in events if item["event"] == "evaluation_failed")
    assert event["metric_id"] == "oob"
    assert event["event_keys"] == ["oob:fixture_object"]
    assert "secret raw response detail" not in json.dumps(event)


def test_retryable_room_failure_stops_at_bounded_room_attempts(
    tmp_path: Path,
) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
        output_name="retry-exhausted",
    )
    result = run_resilient_nonrect_campaign(
        config,
        evaluator_factory=_ScriptedEvaluatorFactory(
            {"room_000": [TimeoutError("transient")] * 5}
        ),
        materializer_backend=NoAPIMockMaterializer(),
    )
    assert result.status == "failed"
    room_root = (
        config.output_root
        / "models/gpt-5.6-sol/scenes/scene_fixture/rooms/room_000"
    )
    assert len(list((room_root / "evaluation_attempts").glob("attempt_*"))) == 3
    summary = json.loads((room_root / "summary.json").read_text())
    assert summary["status"] == "failed_retry_exhausted"
    assert summary["retry_exhausted"] is True
    assert summary["evaluation_retry_count"] == 2


def test_semantic_scene_preflight_failure_does_not_block_later_scene(
    tmp_path: Path,
) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    bad = model_root / "scene_bad"
    shutil.copytree(model_root / "scene_fixture", bad)
    generated = json.loads((bad / "generated_scene.json").read_text())
    generated["rooms"][0]["objects"][0].pop("id")
    _write_json(bad / "generated_scene.json", generated)
    config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
        output_name="scene-preflight-failure",
    )
    result = run_resilient_nonrect_campaign(
        config,
        evaluator_factory=NoAPIMockEvaluatorFactory(),
        materializer_backend=NoAPIMockMaterializer(),
    )
    assert result.status == "failed"
    assert result.complete_room_count == 2
    good = config.output_root / "models/gpt-5.6-sol/scenes/scene_fixture/summary.json"
    bad_summary = config.output_root / "models/gpt-5.6-sol/scenes/scene_bad/summary.json"
    assert json.loads(good.read_text())["status"] == "complete"
    rejected = json.loads(bad_summary.read_text())
    assert rejected["status"] == "nonretryable_generation_preflight_failure"
    assert rejected["failure"]["retryable"] is False


class _TransientMaterializer(NoAPIMockMaterializer):
    def __init__(self) -> None:
        self.failed = False
        self.lock = threading.Lock()

    def materialize(self, **kwargs):
        plan = json.loads(Path(kwargs["plan_path"]).read_text())
        if plan["request"]["room_id"] == "room_000":
            with self.lock:
                if not self.failed:
                    self.failed = True
                    raise NonRectangularMaterializationInfrastructureError(
                        "blender_process_crash", "transient"
                    )
        return super().materialize(**kwargs)


def test_backend_contract_failure_is_not_wrapped_or_retried(tmp_path: Path) -> None:
    class RejectOneRoom(NoAPIMockMaterializer):
        def materialize(self, **kwargs):
            plan = json.loads(Path(kwargs["plan_path"]).read_text())
            if plan["request"]["room_id"] == "room_000":
                raise NonRectangularMaterializationContractError("inspection rejected geometry")
            return super().materialize(**kwargs)

    model_root, csv_path, asset_root = _generation_root(tmp_path)
    config = _config(tmp_path, model_root=model_root, csv_path=csv_path,
                     asset_root=asset_root, output_name="contract-failure")
    result = run_resilient_nonrect_campaign(
        config, evaluator_factory=NoAPIMockEvaluatorFactory(),
        materializer_backend=RejectOneRoom(),
    )
    assert result.complete_room_count == 1  # Other room still progresses.
    room = config.output_root / "models/gpt-5.6-sol/scenes/scene_fixture/rooms/room_000"
    attempts = list((room / "materialization_attempts").glob("attempt_*"))
    assert len(attempts) == 1  # No global retry despite budget=3.
    failure = json.loads((attempts[0] / "attempt_manifest.json").read_text())["failure"]
    assert failure["category"] == "semantic_or_contract"
    assert failure["error_type"] == "NonRectangularMaterializationContractError"
    assert failure["retryable"] is False


def test_materialization_infrastructure_failure_has_separate_retry_budget(
    tmp_path: Path,
) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
        output_name="materialization-retry",
    )
    result = run_resilient_nonrect_campaign(
        config,
        evaluator_factory=NoAPIMockEvaluatorFactory(),
        materializer_backend=_TransientMaterializer(),
    )
    assert result.status == "complete"
    room_root = (
        config.output_root
        / "models/gpt-5.6-sol/scenes/scene_fixture/rooms/room_000"
    )
    assert len(list((room_root / "materialization_attempts").glob("attempt_*"))) == 2
    assert len(list((room_root / "evaluation_attempts").glob("attempt_*"))) == 1


def test_resume_rejects_config_drift(tmp_path: Path) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
        output_name="drift",
    )
    run_resilient_nonrect_campaign(
        config,
        evaluator_factory=NoAPIMockEvaluatorFactory(),
        materializer_backend=NoAPIMockMaterializer(),
    )
    with pytest.raises(ResilientCampaignError, match="identity drifted"):
        run_resilient_nonrect_campaign(
            _config(
                tmp_path,
                model_root=model_root,
                csv_path=csv_path,
                asset_root=asset_root,
                output_name="drift",
                resume=True,
                max_workers=1,
            ),
            evaluator_factory=NoAPIMockEvaluatorFactory(),
            materializer_backend=NoAPIMockMaterializer(),
        )


def test_duplicate_writer_lock_fails_before_campaign_writes(tmp_path: Path) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
        output_name="locked",
    )
    config.output_root.mkdir()
    lock_path = config.output_root / ".runner.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ResilientCampaignError, match="another writer"):
            run_resilient_nonrect_campaign(
                config,
                evaluator_factory=NoAPIMockEvaluatorFactory(),
                materializer_backend=NoAPIMockMaterializer(),
            )
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    assert not (config.output_root / "run_manifest.json").exists()


def test_ambiguous_interrupted_attempt_requires_explicit_recovery(
    tmp_path: Path,
) -> None:
    model_root, csv_path, asset_root = _generation_root(tmp_path)
    base_config = _config(
        tmp_path,
        model_root=model_root,
        csv_path=csv_path,
        asset_root=asset_root,
        output_name="interrupted",
    )
    run_resilient_nonrect_campaign(
        base_config,
        evaluator_factory=NoAPIMockEvaluatorFactory(),
        materializer_backend=NoAPIMockMaterializer(),
    )
    room_root = (
        base_config.output_root
        / "models/gpt-5.6-sol/scenes/scene_fixture/rooms/room_000"
    )
    (room_root / "room_report_selected.json").unlink()
    shutil.rmtree(room_root / "evaluation_attempts/attempt_001")
    (room_root / "evaluation_attempts/attempt_001").mkdir()

    failed_closed = run_resilient_nonrect_campaign(
        _config(
            tmp_path,
            model_root=model_root,
            csv_path=csv_path,
            asset_root=asset_root,
            output_name="interrupted",
            resume=True,
        ),
        evaluator_factory=NoAPIMockEvaluatorFactory(),
        materializer_backend=NoAPIMockMaterializer(),
    )
    assert failed_closed.status == "failed"
    assert not (room_root / "evaluation_attempts/attempt_002").exists()

    recovered = run_resilient_nonrect_campaign(
        _config(
            tmp_path,
            model_root=model_root,
            csv_path=csv_path,
            asset_root=asset_root,
            output_name="interrupted",
            resume=True,
            recover_interrupted=True,
        ),
        evaluator_factory=NoAPIMockEvaluatorFactory(),
        materializer_backend=NoAPIMockMaterializer(),
    )
    assert recovered.status == "complete"
    assert (room_root / "evaluation_attempts/attempt_002/attempt_manifest.json").is_file()


def test_l_shape_plan_preserves_polygon_wall_order_and_global_coordinates(
    tmp_path: Path,
) -> None:
    layout = _fixture("l_shape_single.json")
    program = {
        "schema_version": "non_rectangular_room_program_v1",
        "layout_id": layout["layout_id"],
        "target_total_instances": {"min": 1, "max": 1},
        "program_order": ["living_01"],
        "programs": [{"program_id": "living_01", "room_type": "living_room"}],
    }
    plan = {
        "schema_version": "non_rectangular_multi_room_object_plan_v2",
        "layout_id": layout["layout_id"],
        "room_order": ["room_000"],
        "rooms": [
            {
                "room_id": "room_000",
                "program_id": "living_01",
                "room_type": "living_room",
                "objects": [
                    {
                        "id": "sofa_01",
                        "category": "sofa",
                        "description": "sofa",
                        "count": 1,
                        "estimated_size": [1.0, 1.0, 1.0],
                        "support": "floor",
                        "facing_target": "room_interior",
                        "placement_hints": [],
                        "retrieval_query": "sofa",
                    }
                ],
            }
        ],
    }
    scene = {
        "schema_version": "non_rectangular_multi_room_scene_v1",
        "layout_id": layout["layout_id"],
        "coordinate_frame": layout["coordinate_frame"],
        "room_order": ["room_000"],
        "rooms": [
            {
                "room_id": "room_000",
                "program_id": "living_01",
                "room_type": "living_room",
                "objects": [
                    {
                        "id": "room_000__sofa_01__1",
                        "slot_id": "sofa_01",
                        "jid": "asset_sofa",
                        "category": "sofa",
                        "description": "sofa",
                        "size": [1.0, 1.0, 1.0],
                        "center": [3.25, 0.75, 0.5],
                        "rotation": [0.0, 0.0, 0.0],
                        "geometry_provenance": "asset_mesh",
                        "asset_ref": {
                            "source_db": "imaginarium",
                            "asset_key": "asset_sofa",
                        },
                        "asset_proxy": {
                            "type": "canonical_catalog_bbox",
                            "bbox_center_local": [0.0, 0.0, 0.0],
                            "bbox_size": [1.0, 1.0, 1.0],
                        },
                        "metadata": {
                            "uniform_scale": 1.0,
                            "agent_intended_task_slot": {
                                "placement_hints": ["private wall affinity"],
                                "retrieval_query": "private sofa retrieval",
                            },
                        },
                    }
                ],
            }
        ],
    }
    evaluation_input = NonRectangularEvaluationInput.from_artifacts(
        room_layout=layout,
        room_program=program,
        object_plan=plan,
        generated_scene=scene,
    )
    unit = build_room_evaluation_units(
        prepare_non_rectangular_evaluation(evaluation_input)
    )[0]
    selection = {
        "schema_version": "non_rectangular_asset_selection_v1",
        "layout_id": layout["layout_id"],
        "binding_policy": "room_id_double_colon_slot_id_v1",
        "rooms": [
            {
                "room_id": "room_000",
                "objects": [
                    {
                        "slot_id": "sofa_01",
                        "selected_asset": {
                            "jid": "asset_sofa",
                            "asset_ref": {
                                "source_db": "imaginarium",
                                "asset_key": "asset_sofa",
                            },
                            "asset_proxy": {
                                "type": "canonical_catalog_bbox",
                                "bbox_center_local": [0.0, 0.0, 0.0],
                                "bbox_size": [1.0, 1.0, 1.0],
                            },
                        },
                    }
                ],
            }
        ],
    }
    csv_path, asset_root = _catalog(tmp_path, {"asset_sofa": [1.0, 1.0, 1.0]})
    catalog = FrozenCatalog(
        asset_csv=csv_path,
        asset_root=asset_root,
        allowed_asset_ids={"asset_sofa"},
        snapshot_id="fixture-catalog-v1",
    )
    materialization_plan, canonical, _, architecture = (
        build_nonrect_room_materialization_plan(
            unit,
            room_layout=layout,
            asset_selection=selection,
            catalog=catalog,
            compiled_architecture=build_polygon_architecture(layout),
        )
    )
    assert materialization_plan["request"]["boundary"] == layout["rooms"][0][
        "floor_polygon_xy"
    ]
    assert [item["wall_id"] for item in materialization_plan["request"]["architecture"]["wall_segments"]] == [
        item["wall_id"] for item in layout["rooms"][0]["wall_segments"]
    ]
    assert materialization_plan["instances"][0]["center_m"] == [3.25, 0.75, 0.5]
    assert canonical["objects"][0]["center"] == [3.25, 0.75, 0.5]
    assert "agent_intended_task_slot" not in json.dumps(canonical, sort_keys=True)
    assert "agent_intended_task_slot" in scene["rooms"][0]["objects"][0]["metadata"]
    assert architecture["adjacent_room_objects_included"] is False
    assert architecture["ceiling_included"] is False


def test_rectangular_materializer_import_and_signature_are_unchanged() -> None:
    import inspect

    from benchmark.multi_room_evaluation.materializer import (
        materialize_multi_room_evaluation_dataset,
    )

    assert list(inspect.signature(materialize_multi_room_evaluation_dataset).parameters) == [
        "inventory",
        "output_root",
        "renderer",
        "asset_root",
        "require_complete",
        "dataset_id",
        "materialization_config",
    ]
    assert "evaluation_mode" not in inspect.signature(
        materialize_multi_room_evaluation_dataset
    ).parameters


def test_runtime_identity_does_not_expose_endpoint_or_key_environment() -> None:
    factory = DefaultNonRectangularRuntimeFactory(
        {
            "provider": "api2",
            "judge": {
                "endpoint": "http://127.0.0.1:4999/v1",
                "model": "judge-model",
                "api_key_env": "PRIVATE_TEST_KEY",
            },
        }
    )
    identity = factory.identity()
    serialized = json.dumps(identity)
    assert set(identity) == {"provider", "model", "config_sha256"}
    assert "4999" not in serialized
    assert "PRIVATE_TEST_KEY" not in serialized


@pytest.mark.parametrize("camera, expected", [
    ({}, 1),
    ({"collision_final_view_count": None}, None),
    ({"mode": "bbox_track"}, None),
    ({"metric_modes": {"collision": "bbox_track"}}, None),
    ({"collision_final_bundle": False}, 1),
    ({"collision_final_view_count": 2}, 2),
])
def test_nonrect_runtime_enables_collision_final_budget_only_on_l1(
    tmp_path, monkeypatch, camera, expected
):
    from types import SimpleNamespace
    import benchmark.non_rectangular.runtime as runtime

    providers = []

    def capture_provider(**kwargs):
        providers.append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(runtime, "_model_from_config", lambda *a, **kw: object())
    monkeypatch.setattr(runtime, "OpenAICompatibleVLMJudge", lambda *a, **kw: object())
    monkeypatch.setattr(runtime, "BlenderRenderer", lambda **kw: object())
    monkeypatch.setattr(runtime, "CameraEvidenceProvider", capture_provider)
    monkeypatch.setattr(runtime, "project_room_unit_to_canonical_scene", lambda unit: {})
    monkeypatch.setattr(runtime, "_render_nonrect_global_evidence", lambda **kw: ([], {}))
    monkeypatch.setattr(runtime, "CanonicalNonRectangularRoomEvaluator", lambda **kw: kw)
    factory = DefaultNonRectangularRuntimeFactory({
        "judge": {"model": "fixture-no-network"}, "camera": camera,
    })
    factory.build(SimpleNamespace(
        attempt_root=tmp_path,
        materialization=SimpleNamespace(blend_path=tmp_path / "scene.blend"),
        unit=object(),
    ))
    local, l3 = providers
    assert local["collision_final_view_count"] == expected
    assert local["collision_final_bundle"] is camera.get("collision_final_bundle", expected == 1)
    assert "collision_final_view_count" not in l3
    assert "collision_final_bundle" not in l3
    assert local["max_views"] == l3["max_views"] == 4
    assert local["candidate_count"] == l3["candidate_count"] == 6


def test_runtime_uses_one_combined_exact_request_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = APIUsageRecorder()
    model = RecordingOpenAICompatibleModel(
        recorder=recorder,
        exact_max_retries=5,
        exact_retry_delay_seconds=0.0,
        name="fixture",
        endpoint="http://127.0.0.1:4999/v1",
        model_id="fixture-model",
        require_api_key=False,
        max_retries=0,
    )
    calls = 0

    def fake_chat(self, *args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        if calls < 3:
            raise EndpointMalformedResponseError("malformed")
        self.last_request_metadata = {"usage": {"total_tokens": 17}}
        return "ok"

    monkeypatch.setattr(OpenAICompatibleModel, "chat_messages", fake_chat)
    assert model.chat_messages([]) == "ok"
    assert calls == 3
    assert recorder.totals() == {
        "logical_calls": 1,
        "failed_calls": 0,
        "usage_missing_calls": 0,
        "tokens": 17,
    }

    calls = 0

    def configuration_failure(self, *args, **kwargs):
        nonlocal calls
        del self, args, kwargs
        calls += 1
        raise EndpointConfigurationError("bad route")

    monkeypatch.setattr(
        OpenAICompatibleModel,
        "chat_messages",
        configuration_failure,
    )
    with pytest.raises(EndpointConfigurationError):
        model.chat_messages([])
    assert calls == 1
    assert recorder.totals()["failed_calls"] == 1
