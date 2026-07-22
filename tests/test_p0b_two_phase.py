from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.visual_judge.p0b import build_p0b_local_evidence_request
from scripts.run_p0b_two_phase import (
    SCHEMA_VERSION,
    _evidence_hashes,
    _file_sha256,
    _judge_resume_contract,
    _judgement_ready,
    _packet_ready,
    _validate_packet,
)


def _packet(image: Path) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": "case_001",
        "arm": "fixed_global",
        "camera_mode": "global_only",
        "resolved_camera_mode": "global_only",
        "metric_camera_modes": {},
        "evidence_style": "raw",
        "metric": "collision",
        "event_id": "a|b",
        "gt_label": "invalid",
        "source": {"event": {}, "detector_evidence": {}},
        "scene": {"objects": []},
        "frozen_event_packet_sha256": "event",
        "frozen_scene_sha256": "scene",
        "frozen_source_report_sha256": "report",
        "frozen_gt_sha256": "gt",
        "overview_render_evidence": [str(image)],
        "local_render_evidence_items": [],
        "frozen_evidence_sha256": _evidence_hashes([], [str(image)]),
    }


def test_prepared_packet_detects_image_content_drift(tmp_path: Path) -> None:
    image = tmp_path / "overview.png"
    image.write_bytes(b"first")
    packet = _packet(image)

    _validate_packet(packet, tmp_path / "packet.json")
    image.write_bytes(b"changed")

    with pytest.raises(ValueError, match="evidence file drift"):
        _validate_packet(packet, tmp_path / "packet.json")


def test_offline_camera_request_matches_p0b_context_contract() -> None:
    scene = {
        "boundary": [[0, 0], [5, 0], [5, 4], [0, 4]],
        "objects": [
            {"id": "chair", "category": "chair"},
            {"id": "table", "category": "table"},
        ],
    }
    request = build_p0b_local_evidence_request(
        metric="collision",
        event={"object_a": "chair", "object_b": "table"},
        prompt="Put the chair beside the table.",
        relationships=[{"subject": "chair", "predicate": "beside", "object": "table"}],
        scene=scene,
        detector_evidence={"normalized_overlap": 0.2},
    )

    assert request["object_ids"] == ["chair", "table"]
    assert request["access"] == "read_only_evidence_request"
    assert request["detector_evidence"] == {"normalized_overlap": 0.2}
    request["scene"]["objects"].clear()
    assert len(scene["objects"]) == 2


def test_judgement_resume_requires_same_packet_and_judge_and_retries_errors(
    tmp_path: Path,
) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text('{"packet":1}', encoding="utf-8")
    result_path = tmp_path / "result.json"
    identity = {"model": "model-a", "endpoint": "http://127.0.0.1:8298/v1"}
    judge_contract = {"schema_version": "test", "sha256": "judge-a"}
    result = {
        "predicted_label": "valid",
        "prepared_evidence_packet_sha256": _file_sha256(packet_path),
        "final_judge_model": identity,
        "final_judge_contract": judge_contract,
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")

    assert _judgement_ready(
        result_path,
        packet_path,
        expected_judge_identity=identity,
        expected_judge_contract=judge_contract,
    )
    assert not _judgement_ready(
        result_path,
        packet_path,
        expected_judge_identity={**identity, "model": "model-b"},
        expected_judge_contract=judge_contract,
    )

    result["error"] = "EndpointConnectionError"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert not _judgement_ready(
        result_path,
        packet_path,
        expected_judge_identity=identity,
        expected_judge_contract=judge_contract,
    )

    result.pop("error")
    result_path.write_text(json.dumps(result), encoding="utf-8")
    packet_path.write_text('{"packet":2}', encoding="utf-8")
    assert not _judgement_ready(
        result_path,
        packet_path,
        expected_judge_identity=identity,
        expected_judge_contract=judge_contract,
    )


def test_preparation_resume_requires_current_expected_contract(tmp_path: Path) -> None:
    image = tmp_path / "overview.png"
    image.write_bytes(b"png")
    packet_path = tmp_path / "packet.json"
    contract = {"schema_version": "prepare-v1", "source_sha256": "source-a"}
    packet = {**_packet(image), "preparation_contract": contract}
    packet_path.write_text(json.dumps(packet), encoding="utf-8")

    assert _packet_ready(packet_path, expected_contract=contract)
    assert not _packet_ready(
        packet_path,
        expected_contract={**contract, "source_sha256": "source-b"},
    )


def test_judge_resume_contract_covers_effective_config_without_key_value() -> None:
    config = {
        "endpoint": "http://127.0.0.1:8298/v1",
        "model": "model-a",
        "api_key_env": "TEST_API_KEY",
        "max_tokens": 2048,
        "response_format_json": True,
    }

    first = _judge_resume_contract(config)
    second = _judge_resume_contract({**config, "max_tokens": 4096})
    third = _judge_resume_contract({**config, "response_format_json": False})

    assert first != second
    assert first != third
    assert first["effective_config"]["api_key_env"] == "TEST_API_KEY"
    assert "api_key" not in first["effective_config"]


@pytest.mark.parametrize(
    "launcher",
    [
        "run_qwen235b_fp8_p0b_two_phase.sh",
        "run_qwen235b_fp8_p0b_fixed_deterministic.sh",
    ],
)
def test_qwen235b_launcher_does_not_reuse_stdin_for_models_json(launcher: str) -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "mnet" / launcher
    ).read_text(encoding="utf-8")

    assert "</tmp/qwen235b_models.json <<'PY'" not in script
    assert '"$SERVED_MODEL" /tmp/qwen235b_models.json <<\'PY\'' in script
