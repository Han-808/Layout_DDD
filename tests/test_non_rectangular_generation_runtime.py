from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from benchmark.scene_generation.non_rectangular_multi_room import (
    run_non_rectangular_generation,
    run_non_rectangular_generation_v2,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/non_rectangular"


def _fixture(name: str) -> dict:
    value = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _placement() -> dict[str, Any]:
    return {
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
                        "asset_id": "asset_000",
                        "slot_id": "counter",
                        "center_m": [1.0, 1.0, 0.45],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    },
                    {
                        "instance_id": "fixture.global_001",
                        "asset_id": "asset_001",
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
                        "asset_id": "asset_002",
                        "slot_id": "sofa",
                        "center_m": [5.7, 1.6, 0.4],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 90.0],
                    },
                    {
                        "instance_id": "fixture.global_003",
                        "asset_id": "asset_003",
                        "slot_id": "coffee_table",
                        "center_m": [4.2, 0.6, 0.2],
                        "uniform_scale": 1.0,
                        "rotation_euler_xyz_deg": [0.0, 0.0, 0.0],
                    },
                ],
            },
        ],
    }


@dataclass
class _StageResult:
    status: str
    content: bytes | None
    reason: str | None = None


class _FakeCore:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def call_model_stage(self, **kwargs: Any) -> _StageResult:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected model stage call")
        value = self.responses.pop(0)
        return _StageResult(
            status="captured",
            content=json.dumps(value).encode("utf-8"),
        )

    @staticmethod
    def loads_strict(value: str) -> Any:
        return json.loads(value)

    @staticmethod
    def build_retrieval_request(plan: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": "hy34_frozen_top1_requests_v1",
            "retrieval_policy": {
                "category_argument": None,
                "size_constraint_used": False,
                "top_k": 1,
                "min_score": 0.3,
                "query_rewrite_allowed": False,
                "retry_allowed": False,
                "asset_replacement_allowed": False,
            },
            "requests": [
                {
                    "slot_id": item["id"],
                    "retrieval_query": item["metadata"]["retrieval_query"],
                    "estimated_size": item["estimated_size"],
                    "size_constraint": None,
                }
                for item in plan["objects"]
            ],
        }

    @staticmethod
    def write_exclusive(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(data)

    @classmethod
    def write_json_exclusive(cls, path: Path, value: Any) -> None:
        cls.write_exclusive(
            path,
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )

    @staticmethod
    def sha256_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def retrieve(self, request: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append(request)
        rows = []
        for order, item in enumerate(request["requests"]):
            rows.append(
                {
                    "order": order,
                    "slot_id": item["slot_id"],
                    "retrieval_query": item["retrieval_query"],
                    "size_constraint": None,
                    "invocation_count": 1,
                    "rank1": {
                        "rank": 1,
                        "jid": f"asset_{order:03d}",
                        "category": "fixture_asset",
                        "description": "Fixture asset.",
                        "short_desc": "Fixture.",
                        "size": item["estimated_size"],
                        "score": 0.9,
                        "index_row": order,
                    },
                    "accepted_as_frozen_outcome": True,
                }
            )
        return {
            "schema_version": "hy34_frozen_top1_results_v1",
            "total_invocations": len(rows),
            "retry_count": 0,
            "asset_replacement_count": 0,
            "results": rows,
        }


class _RetryPolicy:
    max_infrastructure_retries = 2
    retry_delay_seconds = 0.0

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
        }


def _run(
    tmp_path: Path,
    *,
    program: dict | None = None,
    core: _FakeCore | None = None,
    retriever: _FakeRetriever | None = None,
    resume: bool = False,
    contract_version: str = "v1",
):
    plan_fixture = (
        "simple_multi_room_object_plan_v2.json"
        if contract_version == "v2"
        else "simple_multi_room_object_plan.json"
    )
    actual_core = core or _FakeCore(
        [_fixture(plan_fixture), _placement()]
    )
    actual_retriever = retriever or _FakeRetriever()
    runner = (
        run_non_rectangular_generation_v2
        if contract_version == "v2"
        else run_non_rectangular_generation
    )
    summary = runner(
        core=actual_core,
        provider_route=SimpleNamespace(key="route_fixture"),
        model=SimpleNamespace(
            key="model_fixture",
            label="Model Fixture",
            wire_model="wire-fixture",
        ),
        retriever=actual_retriever,
        retry_policy=_RetryPolicy(),
        room_layout=_fixture("simple_multi_room.json"),
        room_program=program or _fixture("simple_multi_room_program.json"),
        output_root=tmp_path / "output",
        campaign_id="nonrect_fixture_campaign",
        workflow_profile_id="nonrect_global_workflow_v1",
        retrieval_profile_id="retrieval_fixture",
        resume=resume,
    )
    return summary, actual_core, actual_retriever, tmp_path / "output"


def test_runtime_calls_exactly_one_global_stage_a_and_stage_c(tmp_path: Path) -> None:
    summary, core, retriever, output = _run(tmp_path)

    assert summary["status"] == "complete"
    assert [call["stage"] for call in core.calls] == [
        "stage_a_non_rectangular_global_object_plan",
        "stage_c_non_rectangular_global_placement",
    ]
    assert len(retriever.calls) == 1
    assert len(retriever.calls[0]["requests"]) == 4
    assert core.calls[0]["user_value"]["room_layout"]["room_count"] == 2
    assert len(core.calls[1]["user_value"]["object_plan"]["rooms"]) == 2
    assert "local_to_global_offset_m" not in json.dumps(
        core.calls[1]["user_value"]
    )
    scene = json.loads((output / "generated_scene.json").read_text())
    assert scene["rooms"][1]["objects"][0]["center"] == [5.7, 1.6, 0.4]
    assert (output / "compiled_architecture.json").is_file()
    assert (output / "evaluation_preflight.json").is_file()
    assert summary["generation_mode"] == "non_rectangular_multi_room_global_v1"
    assert "object_plan_contract_version" not in summary


def test_v2_runtime_calls_simplified_stage_a_and_global_stage_c_once(
    tmp_path: Path,
) -> None:
    summary, core, retriever, output = _run(
        tmp_path,
        contract_version="v2",
    )

    assert summary["status"] == "complete"
    assert summary["generation_mode"] == "non_rectangular_multi_room_global_v2"
    assert summary["object_plan_contract_version"] == "v2"
    assert [call["stage"] for call in core.calls] == [
        "stage_a_non_rectangular_global_object_plan_v2",
        "stage_c_non_rectangular_global_placement_v2",
    ]
    stage_a_user = core.calls[0]["user_value"]
    assert stage_a_user["schema_version"] == "non_rectangular_stage_a_brief_v2"
    assert stage_a_user["generation_contract"]["output_schema_version"] == (
        "non_rectangular_multi_room_object_plan_v2"
    )
    assert stage_a_user["generation_contract"]["catalog_facing_prior"] == (
        "directed_local_neg_y"
    )
    assert len(retriever.calls) == 1
    stage_c_user = core.calls[1]["user_value"]
    assert stage_c_user["schema_version"] == "non_rectangular_stage_c_input_v2"
    assert stage_c_user["object_plan"]["schema_version"].endswith(
        "object_plan_v2"
    )
    assert stage_c_user["generation_contract"]["catalog_facing_prior"] == (
        "directed_local_neg_y"
    )
    plan = json.loads((output / "stage_a/object_plan.json").read_text())
    assert "metadata" not in plan["rooms"][0]["objects"][0]
    scene = json.loads((output / "generated_scene.json").read_text())
    assert scene["provenance"]["generation_mode"] == (
        "non_rectangular_multi_room_global_v2"
    )


def test_v2_terminal_resume_makes_no_external_calls(tmp_path: Path) -> None:
    first, _, _, _ = _run(tmp_path, contract_version="v2")
    core = _FakeCore([])
    retriever = _FakeRetriever()

    resumed, core, retriever, _ = _run(
        tmp_path,
        core=core,
        retriever=retriever,
        resume=True,
        contract_version="v2",
    )

    assert resumed == first
    assert resumed["object_plan_contract_version"] == "v2"
    assert core.calls == []
    assert retriever.calls == []


def test_v2_mapping_cutoff_stops_before_retrieval(tmp_path: Path) -> None:
    plan = _fixture("simple_multi_room_object_plan_v2.json")
    plan["rooms"][0].pop("program_id")
    plan["rooms"][0].pop("room_type")
    core = _FakeCore([plan])
    retriever = _FakeRetriever()

    summary, core, retriever, output = _run(
        tmp_path,
        core=core,
        retriever=retriever,
        contract_version="v2",
    )

    assert summary["status"] == "program_mapping_contract_failed"
    assert summary["generation_mode"] == "non_rectangular_multi_room_global_v2"
    assert [call["stage"] for call in core.calls] == [
        "stage_a_non_rectangular_global_object_plan_v2"
    ]
    assert retriever.calls == []
    assert not (output / "stage_c/placement_first_emission.json").exists()


def test_count_gate_failure_calls_no_retrieval_or_stage_c(tmp_path: Path) -> None:
    program = _fixture("simple_multi_room_program.json")
    program["target_total_instances"] = {"min": 7, "max": 8}
    core = _FakeCore([_fixture("simple_multi_room_object_plan.json")])
    retriever = _FakeRetriever()

    summary, core, retriever, output = _run(
        tmp_path,
        program=program,
        core=core,
        retriever=retriever,
    )

    assert summary["status"] == "object_count_contract_failed"
    assert [call["stage"] for call in core.calls] == [
        "stage_a_non_rectangular_global_object_plan"
    ]
    assert retriever.calls == []
    assert not (output / "generated_scene.json").exists()
    assert not (output / "stage_c/placement_first_emission.json").exists()


def test_terminal_resume_makes_no_external_calls(tmp_path: Path) -> None:
    first, _, _, _ = _run(tmp_path)
    core = _FakeCore([])
    retriever = _FakeRetriever()

    resumed, core, retriever, _ = _run(
        tmp_path,
        core=core,
        retriever=retriever,
        resume=True,
    )

    assert resumed == first
    assert core.calls == []
    assert retriever.calls == []


def test_half_room_invalid_mapping_stops_after_stage_a_with_terminal_zero(
    tmp_path: Path,
) -> None:
    plan = _fixture("simple_multi_room_object_plan.json")
    plan["rooms"][0].pop("program_id")
    plan["rooms"][0].pop("room_type")
    core = _FakeCore([plan])
    retriever = _FakeRetriever()

    summary, core, retriever, output = _run(
        tmp_path,
        core=core,
        retriever=retriever,
    )

    assert summary["status"] == "program_mapping_contract_failed"
    assert summary["reason"] == "program_mapping_contract_failed"
    assert summary["program_mapping"]["coverage_compliance"][
        "terminal_case_score"
    ] == 0.0
    assert [call["stage"] for call in core.calls] == [
        "stage_a_non_rectangular_global_object_plan"
    ]
    assert retriever.calls == []
    assert not (output / "generated_scene.json").exists()
    assert not (output / "stage_c/placement_first_emission.json").exists()


def test_mapping_terminal_zero_resume_makes_no_external_calls(
    tmp_path: Path,
) -> None:
    plan = _fixture("simple_multi_room_object_plan.json")
    plan["rooms"][0].pop("program_id")
    plan["rooms"][0].pop("room_type")
    first, _, _, _ = _run(tmp_path, core=_FakeCore([plan]))
    core = _FakeCore([])
    retriever = _FakeRetriever()

    resumed, core, retriever, _ = _run(
        tmp_path,
        core=core,
        retriever=retriever,
        resume=True,
    )

    assert resumed == first
    assert resumed["status"] == "program_mapping_contract_failed"
    assert core.calls == []
    assert retriever.calls == []


def test_resume_rebuilds_deterministic_summary_without_resending(tmp_path: Path) -> None:
    first, _, _, output = _run(tmp_path)
    (output / "summary.json").unlink()
    core = _FakeCore([])
    retriever = _FakeRetriever()

    resumed, core, retriever, _ = _run(
        tmp_path,
        core=core,
        retriever=retriever,
        resume=True,
    )

    assert resumed == first
    assert resumed["stage_calls"] == {
        "stage_a": 1,
        "retrieval_batch": 1,
        "stage_c": 1,
    }
    assert core.calls == []
    assert retriever.calls == []
