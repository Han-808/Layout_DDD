from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from benchmark.scene_generation.non_rectangular_agent.asset_db import (
    shared_database_static_contract,
)
from benchmark.scene_generation.non_rectangular_agent.cohort_runner import (
    execute_agent_fullrun,
    prepare_agent_fullrun,
)
from benchmark.scene_generation.non_rectangular_agent.contracts import (
    AGENT_SUBMISSION_SCHEMA_PATH,
    AgentSubmissionError,
    validate_agent_submission,
)
from benchmark.scene_generation.non_rectangular_agent.external import (
    AgentProcessResult,
)
from benchmark.scene_generation.non_rectangular_agent.profiles import (
    AgentBackendProfile,
)
from benchmark.scene_generation.non_rectangular_agent.runtime import (
    run_agent_episode,
)
from benchmark.scene_generation.non_rectangular_agent.suite import (
    AgentFloorPlanCase,
    load_agent_floorplan_suite,
)
from benchmark.scene_generation.non_rectangular_agent.tool_client import call_tool
from benchmark.scene_generation.non_rectangular_agent.tool_server import (
    AgentToolError,
    AgentToolPolicy,
    AgentToolServer,
    AgentToolSession,
    validate_task_submission_constraints,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"
SUITE = (
    ROOT
    / "configs/generation_extensions/non_rectangular_agent_v1/suites/"
    "complicated_floorplan_selected10_v1"
)
PROFILE = (
    ROOT
    / "configs/generation_extensions/non_rectangular_agent_v1/agent_track.example.json"
)


def _fixture(name: str) -> dict[str, Any]:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


class FakeCatalog:
    snapshot_id = "fixture-shared-db-v1"

    def __init__(self) -> None:
        plan = _fixture("simple_multi_room_object_plan_v2.json")
        self.assets = {
            f"asset_{item['id']}": {
                "asset_id": f"asset_{item['id']}",
                "jid": f"asset_{item['id']}",
                "category": f"catalog_{item['category']}",
                "description": f"Catalog-owned {item['description']}",
                "short_desc": f"Catalog-owned {item['description']}",
                "size": item["estimated_size"],
                "bbox_center_local": [0.0, 0.0, 0.0],
            }
            for room in plan["rooms"]
            for item in room["objects"]
        }

    @property
    def asset_count(self) -> int:
        return len(self.assets)

    def resolve(self, asset_id: str) -> dict[str, Any]:
        return deepcopy(self.assets[asset_id])

    def search(self, query: str, *, size_constraint: Any, top_k: int) -> dict[str, Any]:
        rows = list(self.assets.values())[:top_k]
        return {
            "schema_version": "non_rectangular_agent_asset_search_results_v1",
            "catalog_snapshot_id": self.snapshot_id,
            "query": query,
            "size_constraint": size_constraint,
            "top_k_requested": top_k,
            "result_count": len(rows),
            "results": rows,
        }

    def public_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "fixture_shared_db_v1",
            "snapshot_id": self.snapshot_id,
            "snapshot_sha256": "f" * 64,
            "expected_asset_count": len(self.assets),
        }


def _submission() -> dict[str, Any]:
    plan = _fixture("simple_multi_room_object_plan_v2.json")
    bindings = [
        {
            "room_id": room["room_id"],
            "slot_id": item["id"],
            "asset_id": f"asset_{item['id']}",
        }
        for room in plan["rooms"]
        for item in room["objects"]
    ]
    placement = {
        "schema_version": "non_rectangular_global_catalog_placement_v1",
        "layout_id": "fixture_simple_multi_room",
        "coordinate_frame": "shared_scene_global_x_width_y_depth_z_up_meters",
        "room_order": ["room_000", "room_001"],
        "rooms": [
            {
                "room_id": "room_000",
                "program_id": "kitchen_01",
                "room_type": "kitchen",
                "instances": [
                    {
                        "instance_id": "fixture.global_000",
                        "asset_id": "asset_counter",
                        "slot_id": "counter",
                        "center_m": [1.0, 1.0, 0.45],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    },
                    {
                        "instance_id": "fixture.global_001",
                        "asset_id": "asset_stool",
                        "slot_id": "stool",
                        "center_m": [2.0, 1.0, 0.45],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    },
                ],
            },
            {
                "room_id": "room_001",
                "program_id": "living_room_01",
                "room_type": "living_room",
                "instances": [
                    {
                        "instance_id": "fixture.global_002",
                        "asset_id": "asset_sofa",
                        "slot_id": "sofa",
                        "center_m": [5.7, 1.6, 0.4],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 90.0],
                    },
                    {
                        "instance_id": "fixture.global_003",
                        "asset_id": "asset_coffee_table",
                        "slot_id": "coffee_table",
                        "center_m": [4.2, 0.6, 0.2],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    },
                ],
            },
        ],
    }
    return {
        "schema_version": "non_rectangular_agent_submission_v1",
        "layout_id": "fixture_simple_multi_room",
        "object_plan": plan,
        "asset_bindings": bindings,
        "global_placement": placement,
    }


def _case() -> AgentFloorPlanCase:
    layout = _fixture("simple_multi_room.json")
    program = _fixture("simple_multi_room_program.json")
    return AgentFloorPlanCase(
        scene_id="fixture_simple_multi_room",
        room_count=2,
        wall_segment_count=10,
        room_layout_path=FIXTURES / "simple_multi_room.json",
        room_program_path=FIXTURES / "simple_multi_room_program.json",
        room_layout=layout,
        room_program=program,
        room_layout_sha256=hashlib.sha256(
            (FIXTURES / "simple_multi_room.json").read_bytes()
        ).hexdigest(),
        room_program_sha256=hashlib.sha256(
            (FIXTURES / "simple_multi_room_program.json").read_bytes()
        ).hexdigest(),
    )


def _agent() -> AgentBackendProfile:
    return AgentBackendProfile(
        agent_id="fixture-agent",
        display_name="Fixture Agent",
        implementation="fixture-runtime",
        implementation_version="1.0",
        model_id="fixture-model",
        command=("fixture-agent",),
        prompt_transport="stdin",
        isolation_mode="backend_enforced_task_workspace_only",
        timeout_seconds=60.0,
        max_process_attempts=2,
        retry_delay_seconds=0.0,
        retryable_exit_codes=(75,),
        pass_environment=(),
    )


def test_approved_agent_suite_and_shared_db_contract_are_frozen() -> None:
    suite = load_agent_floorplan_suite(SUITE)
    prepared = prepare_agent_fullrun(PROFILE)
    database = shared_database_static_contract(
        catalog_path=ROOT / "configs/retrieval/profiles_v2.json"
    )

    assert suite.public_dict()["track_type"] == "agent_only"
    assert suite.public_dict()["asset_access_mode"] == "shared_database"
    assert prepared.public_dict()["participant_class"] == (
        "general_purpose_tool_using_agent"
    )
    assert prepared.public_dict()["comparison_unit"] == (
        "general_purpose_tool_using_agent_system"
    )
    assert suite.public_dict()["scene_count"] == 10
    assert suite.public_dict()["room_count"] == 42
    assert suite.public_dict()["wall_segment_count"] == 314
    assert suite.public_dict()["aggregate_target_total_instances"] == {
        "min": 719,
        "max": 891,
    }
    assert prepared.public_dict()["room_count_per_agent"] == 42
    assert database["expected_asset_count"] == 2043
    assert database["per_scene_assets_prefrozen"] is False
    assert json.loads(AGENT_SUBMISSION_SCHEMA_PATH.read_text(encoding="utf-8")) == (
        json.loads(
            (ROOT / "schemas/non_rectangular_agent/submission_v1.schema.json").read_text(
                encoding="utf-8"
            )
        )
    )


def test_agent_submission_uses_authoritative_shared_db_assets() -> None:
    value = validate_agent_submission(
        _submission(),
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
        asset_catalog=FakeCatalog(),
    )

    assert value.public_dict()["planned_instance_count"] == 4
    selected = value.asset_selection["rooms"][0]["objects"][0]["selected_asset"]
    assert selected["jid"] == "asset_counter"
    assert selected["category"] == "catalog_counter"
    assert selected["size"] == [1.2, 0.6, 0.9]
    assert value.asset_selection["binding_policy"] == (
        "agent_selected_from_frozen_shared_database_v1"
    )

    invalid = _submission()
    invalid["asset_bindings"][0]["asset_id"] = "outside_db"
    invalid["global_placement"]["rooms"][0]["instances"][0]["asset_id"] = (
        "outside_db"
    )
    with pytest.raises(AgentSubmissionError, match="unavailable"):
        validate_agent_submission(
            invalid,
            room_layout=_fixture("simple_multi_room.json"),
            room_program=_fixture("simple_multi_room_program.json"),
            asset_catalog=FakeCatalog(),
        )


def test_task_constraints_enforce_per_room_ranges_and_unit_scale() -> None:
    validated = validate_agent_submission(
        _submission(),
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
        asset_catalog=FakeCatalog(),
    )
    task = {
        "target_total_instances": {"min": 4, "max": 4},
        "complexity_contract": {
            "room_instance_ranges": [
                {"room_id": "room_000", "min": 2, "max": 2},
                {"room_id": "room_001", "min": 2, "max": 2},
            ]
        },
        "geometry_contract": {
            "uniform_scale": {"policy": "exact", "value": 1.0}
        },
    }
    report = validate_task_submission_constraints(validated, task_payload=task)
    assert report["valid"] is True
    assert report["room_instance_ranges"][0]["actual"] == 2

    invalid_count = deepcopy(task)
    invalid_count["complexity_contract"]["room_instance_ranges"][0] = {
        "room_id": "room_000",
        "min": 1,
        "max": 1,
    }
    invalid_count["target_total_instances"] = {"min": 3, "max": 3}
    with pytest.raises(AgentToolError, match="outside"):
        validate_task_submission_constraints(validated, task_payload=invalid_count)

    scaled = _submission()
    scaled["global_placement"]["rooms"][0]["instances"][0]["uniform_scale"] = 0.9
    scaled_validated = validate_agent_submission(
        scaled,
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
        asset_catalog=FakeCatalog(),
    )
    with pytest.raises(AgentToolError, match="uniform_scale"):
        validate_task_submission_constraints(scaled_validated, task_payload=task)


def test_local_tool_server_seals_only_valid_submission(tmp_path: Path) -> None:
    (tmp_path / "submission.json").write_text(
        json.dumps(_submission()), encoding="utf-8"
    )
    session = AgentToolSession(
        workspace=tmp_path,
        room_layout=_fixture("simple_multi_room.json"),
        room_program=_fixture("simple_multi_room_program.json"),
        asset_catalog=FakeCatalog(),
        task_payload={"layout_id": "fixture_simple_multi_room"},
        policy=AgentToolPolicy(max_total_calls=8, max_top_k=4),
        seal_record_path=tmp_path / "trusted_submission_seal.json",
    )
    with AgentToolServer(session) as server:
        task = call_tool(
            "get_task",
            {},
            socket_path=str(server.socket_path),
            token=server.token,
        )
        result = call_tool(
            "finalize_submission",
            {"submission_path": "submission.json"},
            socket_path=str(server.socket_path),
            token=server.token,
        )

    assert task["ok"] is True
    assert result["ok"] is True
    assert (tmp_path / "final_submission.json").is_file()
    assert (tmp_path / "finalization.json").is_file()
    trusted_seal = json.loads(
        (tmp_path / "trusted_submission_seal.json").read_text()
    )
    assert trusted_seal["schema_version"] == "sieve_trusted_submission_seal_v1"
    assert (
        trusted_seal["finalization"]["submission_sha256"]
        == json.loads((tmp_path / "finalization.json").read_text())["submission_sha256"]
    )
    events = [
        json.loads(line)
        for line in (tmp_path / "tool_events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 2
    assert events[0]["schema_version"] == "non_rectangular_agent_tool_event_v2"
    assert events[0]["previous_event_sha256"] is None
    assert events[1]["previous_event_sha256"] == events[0]["event_sha256"]
    assert events[1]["result"]["valid"] is True
    assert len(events[1]["result_sha256"]) == 64
    assert len(json.loads((tmp_path / "finalization.json").read_text())["submission_sha256"]) == 64


def test_tool_budget_is_cumulative_across_resume(tmp_path: Path) -> None:
    policy = AgentToolPolicy(max_total_calls=2, max_top_k=4)
    kwargs = {
        "workspace": tmp_path,
        "room_layout": _fixture("simple_multi_room.json"),
        "room_program": _fixture("simple_multi_room_program.json"),
        "asset_catalog": FakeCatalog(),
        "task_payload": {"layout_id": "fixture_simple_multi_room"},
        "policy": policy,
    }
    first = AgentToolSession(**kwargs)
    first.dispatch("get_task", {})
    first.dispatch("get_task", {})
    resumed = AgentToolSession(**kwargs)

    assert resumed.counts()["total"] == 2
    with pytest.raises(AgentToolError, match="exhausted total"):
        resumed.dispatch("get_task", {})


def test_agent_episode_materializes_existing_canonical_contract_and_resumes(
    tmp_path: Path,
) -> None:
    def fake_process(**kwargs: Any) -> AgentProcessResult:
        artifacts = kwargs["artifacts"]
        server = kwargs["tool_server"]
        (artifacts.workspace / "submission.json").write_text(
            json.dumps(_submission()), encoding="utf-8"
        )
        server.session.dispatch(
            "finalize_submission", {"submission_path": "submission.json"}
        )
        return AgentProcessResult(
            status="complete",
            attempts=1,
            returncode=0,
            timed_out=False,
            final_submission_sealed=True,
        )

    output = tmp_path / "episode"
    first = run_agent_episode(
        case=_case(),
        asset_catalog=FakeCatalog(),
        agent_profile=_agent(),
        tool_policy=AgentToolPolicy(max_total_calls=8, max_top_k=4),
        output_root=output,
        suite_identity={"suite": "fixture"},
        process_runner=fake_process,
    )
    second = run_agent_episode(
        case=_case(),
        asset_catalog=FakeCatalog(),
        agent_profile=_agent(),
        tool_policy=AgentToolPolicy(max_total_calls=8, max_top_k=4),
        output_root=output,
        suite_identity={"suite": "fixture"},
        resume=True,
        process_runner=fake_process,
    )

    assert first == second
    assert first["status"] == "complete"
    assert first["planned_instance_count"] == 4
    assert first["generated_object_count"] == 4
    scene = json.loads(
        (output / "normalized/generated_scene.json").read_text(encoding="utf-8")
    )
    assert scene["schema_version"] == "non_rectangular_multi_room_scene_v1"
    assert scene["rooms"][0]["objects"][0]["category"] == "catalog_counter"
    assert scene["rooms"][0]["objects"][0]["metadata"][
        "agent_intended_task_slot"
    ]["category"] == "counter"
    assert scene["provenance"]["generation_mode"] == (
        "non_rectangular_agent_shared_db_v1"
    )


def test_external_agent_retries_only_declared_infrastructure_exit(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fixture_agent.py"
    submission_json = json.dumps(_submission())
    script.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "import subprocess",
                "import sys",
                "marker = Path('first_attempt.marker')",
                "if not marker.exists():",
                "    marker.write_text('retry', encoding='utf-8')",
                "    raise SystemExit(75)",
                f"Path('submission.json').write_text({submission_json!r}, encoding='utf-8')",
                "result = subprocess.run(["
                "'./layout-ddd-agent-tool', 'finalize-submission', "
                "'submission.json'], check=False)",
                "raise SystemExit(result.returncode)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    profile = _agent()
    profile = AgentBackendProfile(
        agent_id=profile.agent_id,
        display_name=profile.display_name,
        implementation=profile.implementation,
        implementation_version=profile.implementation_version,
        model_id=profile.model_id,
        command=(sys.executable, str(script)),
        prompt_transport=profile.prompt_transport,
        isolation_mode=profile.isolation_mode,
        timeout_seconds=profile.timeout_seconds,
        max_process_attempts=2,
        retry_delay_seconds=0.0,
        retryable_exit_codes=(75,),
        pass_environment=(),
    )

    summary = run_agent_episode(
        case=_case(),
        asset_catalog=FakeCatalog(),
        agent_profile=profile,
        tool_policy=AgentToolPolicy(max_total_calls=8, max_top_k=4),
        output_root=tmp_path / "external_episode",
        suite_identity={"suite": "fixture"},
    )

    assert summary["status"] == "complete"
    assert summary["process"]["attempts"] == 2
    attempts = tmp_path / "external_episode/agent/attempts"
    first = json.loads((attempts / "attempt_001/result.json").read_text())
    second = json.loads((attempts / "attempt_002/result.json").read_text())
    assert first["status"] == "retryable_infrastructure_failure"
    assert second["status"] == "complete"


def test_agent_fullrun_continues_all_ten_layouts_and_resumes(tmp_path: Path) -> None:
    prepared = prepare_agent_fullrun(PROFILE)

    class GateDatabase:
        asset_count = 2043
        snapshot_id = prepared.shared_database_contract["snapshot_id"]

        def public_manifest(self) -> dict[str, Any]:
            return {
                **prepared.shared_database_contract,
                "asset_order_sha256": "a" * 64,
                "runtime_provenance": {},
            }

    calls: list[tuple[str, bool]] = []

    def fake_episode(**kwargs: Any) -> dict[str, Any]:
        output = Path(kwargs["output_root"])
        output.mkdir(parents=True, exist_ok=bool(kwargs["resume"]))
        calls.append((kwargs["case"].scene_id, bool(kwargs["resume"])))
        return {"status": "complete", "reason": None}

    output = tmp_path / "fullrun"
    first = execute_agent_fullrun(
        prepared,
        command="run",
        output_base=output,
        fresh=True,
        database_factory=lambda **kwargs: GateDatabase(),
        episode_runner=fake_episode,
    )
    second = execute_agent_fullrun(
        prepared,
        command="run",
        output_base=output,
        database_factory=lambda **kwargs: GateDatabase(),
        episode_runner=fake_episode,
    )

    assert first["status"] == "complete"
    assert first["complete_cases"] == 10
    assert second["status"] == "complete"
    assert len(calls) == 20
    assert all(resume is False for _, resume in calls[:10])
    assert all(resume is True for _, resume in calls[10:])


def test_ambiguous_agent_timeout_is_terminal_without_blind_retry(
    tmp_path: Path,
) -> None:
    base = _agent()
    profile = AgentBackendProfile(
        agent_id=base.agent_id,
        display_name=base.display_name,
        implementation=base.implementation,
        implementation_version=base.implementation_version,
        model_id=base.model_id,
        command=(sys.executable, "-c", "import time; time.sleep(30)"),
        prompt_transport=base.prompt_transport,
        isolation_mode=base.isolation_mode,
        timeout_seconds=0.1,
        max_process_attempts=3,
        retry_delay_seconds=0.0,
        retryable_exit_codes=(75,),
        pass_environment=(),
    )

    summary = run_agent_episode(
        case=_case(),
        asset_catalog=FakeCatalog(),
        agent_profile=profile,
        tool_policy=AgentToolPolicy(max_total_calls=8, max_top_k=4),
        output_root=tmp_path / "timeout_episode",
        suite_identity={"suite": "fixture"},
    )

    assert summary["status"] == "failed"
    assert summary["reason"] == "ambiguous_timeout"
    assert summary["process"]["attempts"] == 1
    attempts = list((tmp_path / "timeout_episode/agent/attempts").iterdir())
    assert len(attempts) == 1
