import ast
import json
from pathlib import Path

from benchmark.visual_judge.render_views import _collision_geometry_contract
from scripts.run_p0b_visual_config_experiment import (
    CANDIDATE_SCHEMA_VERSION,
    DETERMINISTIC_VARIANTS,
    SELECTION_SCHEMA_VERSION,
    VISUAL_CONFIG_ARMS,
    _candidate_packet_ready,
    _compose_items,
    _file_sha256,
    _finalized_packet_ready,
    _selection_ready,
    _selector_resume_contract,
    _selector_preview_visibility_warning,
)
from scripts.run_p0b_two_phase import SCHEMA_VERSION, _evidence_hashes


def _items(tmp_path: Path) -> list[dict]:
    values = [
        ("global", "metric_highlighted_global", "global"),
        ("raw_a", "metric_local_rgb", "a"),
        ("highlight_a", "metric_local_highlight", "a"),
        ("raw_b", "metric_local_rgb", "b"),
        ("highlight_b", "metric_local_highlight", "b"),
    ]
    result = []
    for name, role, view_id in values:
        path = tmp_path / f"{name}.png"
        path.write_bytes(b"png")
        result.append({"path": str(path), "role": role, "view_id": view_id})
    return result


def test_visual_config_arm_set_excludes_active_camera_adjustment() -> None:
    assert len(VISUAL_CONFIG_ARMS) == 9
    assert "fixed_global" in VISUAL_CONFIG_ARMS
    assert "deterministic_metric_local" in VISUAL_CONFIG_ARMS
    assert "vlm_select_from_candidates" in VISUAL_CONFIG_ARMS
    assert "active_metric_local" not in VISUAL_CONFIG_ARMS


def test_selection_arm_builds_dedicated_camera_selector_transport() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_p0b_visual_config_experiment.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
    }

    assert "build_openai_compatible_camera_selector" in called_names
    assert "build_openai_compatible_vlm_judge" not in called_names


def test_presence_matrix_and_compact_budget_keep_raw_highlight_pairs(tmp_path: Path) -> None:
    items = _items(tmp_path)

    full = _compose_items(items, **DETERMINISTIC_VARIANTS["deterministic_metric_local"])
    compact = _compose_items(
        items,
        **DETERMINISTIC_VARIANTS["budget_global_first_compact"],
    )
    local_first = _compose_items(items, **DETERMINISTIC_VARIANTS["order_local_first_full"])

    assert [item["role"] for item in full] == [
        "metric_highlighted_global",
        "metric_local_rgb",
        "metric_local_highlight",
        "metric_local_rgb",
        "metric_local_highlight",
    ]
    assert [item["view_id"] for item in compact] == ["global", "a", "a"]
    assert len(compact) == 3
    assert local_first[-1]["role"] == "metric_highlighted_global"


def test_raw_presence_arm_never_keeps_highlight_or_global(tmp_path: Path) -> None:
    selected = _compose_items(items=_items(tmp_path), **DETERMINISTIC_VARIANTS["presence_local_raw"])

    assert [item["role"] for item in selected] == ["metric_local_rgb", "metric_local_rgb"]


def test_selector_preview_visibility_is_advisory_not_a_routing_gate() -> None:
    warning = _selector_preview_visibility_warning(
        {},
        {"targets": [{"id": "obj_001", "required_for_visibility": True}]},
    )

    assert warning is not None
    assert "selector previews do not visibly expose" in warning


def test_selector_resume_is_invalidated_by_selector_identity_or_packet_drift(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate_packet.json"
    candidate.write_text('{"candidate":1}', encoding="utf-8")
    identity = {"model": "selector-a", "endpoint": "http://127.0.0.1:8298/v1"}
    selector_contract = {"schema_version": "selector-v1", "sha256": "selector-a"}
    decision = tmp_path / "selection_decision.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": SELECTION_SCHEMA_VERSION,
                "selected_view_ids": ["view-a"],
                "candidate_packet_sha256": _file_sha256(candidate),
                "pose_selector_model": identity,
                "selector_contract": selector_contract,
            }
        ),
        encoding="utf-8",
    )

    assert _selection_ready(decision, candidate, identity, selector_contract)
    assert not _selection_ready(
        decision,
        candidate,
        {**identity, "model": "selector-b"},
        selector_contract,
    )
    candidate.write_text('{"candidate":2}', encoding="utf-8")
    assert not _selection_ready(decision, candidate, identity, selector_contract)


def test_candidate_packet_resume_requires_current_preparation_contract(
    tmp_path: Path,
) -> None:
    preview = tmp_path / "preview.png"
    preview.write_bytes(b"png")
    contract = {"schema_version": "candidate-v1", "base_sha256": "base-a"}
    packet_path = tmp_path / "candidate_packet.json"
    packet_path.write_text(
        json.dumps(
            {
                "schema_version": CANDIDATE_SCHEMA_VERSION,
                "selector_request": {
                    "allow_adjustment": False,
                    "candidates": [
                        {"id": "view-a", "image_path": str(preview)}
                    ],
                },
                "candidate_count": 1,
                "candidate_preview_sha256": [
                    {
                        "id": "view-a",
                        "path": str(preview),
                        "sha256": _file_sha256(preview),
                    }
                ],
                "preparation_contract": contract,
            }
        ),
        encoding="utf-8",
    )

    assert _candidate_packet_ready(packet_path, expected_contract=contract)
    assert not _candidate_packet_ready(
        packet_path,
        expected_contract={**contract, "base_sha256": "base-b"},
    )


def test_selector_contract_covers_full_effective_config() -> None:
    config = {
        "endpoint": "http://127.0.0.1:8298/v1",
        "model": "selector-a",
        "max_tokens": 2048,
        "max_images": 6,
    }

    first = _selector_resume_contract(config)
    assert first != _selector_resume_contract({**config, "max_tokens": 4096})
    assert first != _selector_resume_contract({**config, "max_images": 4})


def test_finalize_resume_requires_full_contract_and_current_evidence(
    tmp_path: Path,
) -> None:
    image = tmp_path / "local.png"
    image.write_bytes(b"first")
    item = {"path": str(image), "role": "metric_local_rgb"}
    contract = {
        "schema_version": "finalize-v1",
        "base_packet_sha256": "base-a",
        "candidate_packet_sha256": "candidate-a",
        "selector_decision_sha256": "decision-a",
    }
    output = tmp_path / "finalized.json"
    output.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": "case-a",
                "arm": "vlm_select_from_candidates",
                "metric": "collision",
                "event_id": "a|b",
                "gt_label": "invalid",
                "source": {"event": {}, "detector_evidence": {}},
                "scene": {"objects": []},
                "frozen_event_packet_sha256": "event",
                "frozen_scene_sha256": "scene",
                "frozen_source_report_sha256": "report",
                "frozen_gt_sha256": "gt",
                "overview_render_evidence": [],
                "local_render_evidence_items": [item],
                "frozen_evidence_sha256": _evidence_hashes([item], []),
                "finalization_contract": contract,
            }
        ),
        encoding="utf-8",
    )

    assert _finalized_packet_ready(output, expected_contract=contract)
    assert not _finalized_packet_ready(
        output,
        expected_contract={**contract, "base_packet_sha256": "base-b"},
    )
    image.write_bytes(b"changed")
    assert not _finalized_packet_ready(output, expected_contract=contract)


def test_collision_geometry_contract_hashes_mesh_content_not_source_uri(
    tmp_path: Path,
) -> None:
    mesh = tmp_path / "object.ply"
    mesh.write_bytes(b"mesh-a")
    manifest = {
        "schema_version": "collision_geometry_v1",
        "objects": {
            "obj_001": {
                "geometry_path": str(mesh),
                "source_uri": "/private/assets/original.fbx",
                "complete": True,
            }
        },
    }

    first = _collision_geometry_contract(manifest)
    source_only = _collision_geometry_contract(
        {
            **manifest,
            "objects": {
                "obj_001": {
                    **manifest["objects"]["obj_001"],
                    "source_uri": "/different/private/path.fbx",
                }
            },
        }
    )
    mesh.write_bytes(b"mesh-b")
    changed_mesh = _collision_geometry_contract(manifest)

    assert first is not None
    assert first == source_only
    assert first != changed_mesh
