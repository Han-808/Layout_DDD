"""Trusted bridge from one isolated workspace to the real shared asset DB."""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
from pathlib import Path
import sys
from typing import Any

from arena import (
    ARENA_ROOT,
    ArenaError,
    Episode,
    read_json,
    verify_episode_inputs,
    write_json_exclusive,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from benchmark.scene_generation.non_rectangular_agent.asset_db import (  # noqa: E402
    open_shared_asset_database,
)
from benchmark.scene_generation.non_rectangular_agent.contracts import (  # noqa: E402
    validate_agent_submission,
)
from benchmark.scene_generation.non_rectangular_agent.normalization import (  # noqa: E402
    materialize_agent_scene,
)
from benchmark.scene_generation.non_rectangular_agent.tool_server import (  # noqa: E402
    AgentToolPolicy,
    AgentToolServer,
    AgentToolSession,
    validate_task_submission_constraints,
)
from benchmark.scene_generation.non_rectangular_multi_room.architecture import (  # noqa: E402
    build_polygon_architecture,
)


class EpisodeDatabase(AbstractContextManager["EpisodeDatabase"]):
    """Own the real DB and one authenticated, per-episode tool service."""

    def __init__(self, *, episode: Episode, resource_bindings: str | Path) -> None:
        verify_episode_inputs(episode)
        arena = read_json(ARENA_ROOT / "arena.json")
        policy_raw = arena["tool_policy"]
        self.episode = episode
        self.database = open_shared_asset_database(
            catalog_path=REPOSITORY_ROOT / "configs/retrieval/profiles_v2.json",
            resource_bindings_path=Path(resource_bindings).expanduser().resolve(),
            retrieval_profile_id=(
                "imaginarium-qwen3-embedding-0.6b-stable-top1-v2"
            ),
            max_top_k=int(policy_raw["max_top_k"]),
        )
        expected = read_json(ARENA_ROOT / "fixed_suite/shared_database_contract.json")
        observed = self.database.public_manifest()
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ArenaError(f"runtime shared DB differs from frozen contract: {key}")
        self.policy = AgentToolPolicy(
            max_total_calls=int(policy_raw["max_total_calls"]),
            max_asset_searches=int(policy_raw["max_asset_searches"]),
            max_asset_inspections=int(policy_raw["max_asset_inspections"]),
            max_submission_validations=int(
                policy_raw["max_submission_validations"]
            ),
            max_top_k=int(policy_raw["max_top_k"]),
        )
        floorplan = read_json(episode.workspace / "floorplan.json")
        room_program = read_json(episode.workspace / "room_program.json")
        task = read_json(episode.workspace / "task.json")
        self.session = AgentToolSession(
            workspace=episode.workspace,
            room_layout=floorplan,
            room_program=room_program,
            asset_catalog=self.database,
            task_payload=task,
            policy=self.policy,
            audit_path=episode.host / "tool_events.jsonl",
            seal_record_path=episode.host / "submission_seal.json",
        )
        self.server = AgentToolServer(self.session)

    @property
    def socket_path(self) -> Path:
        return self.server.socket_path

    @property
    def capability_token(self) -> str:
        return self.server.token

    def __enter__(self) -> "EpisodeDatabase":
        self.server.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.server.close()


def collect_and_normalize(episode: Episode, database: Any) -> dict[str, Any]:
    """Revalidate outside the sandbox and publish evaluator-compatible artifacts."""

    verify_episode_inputs(episode)
    final_submission = episode.workspace / "final_submission.json"
    finalization = episode.workspace / "finalization.json"
    if not final_submission.is_file() or final_submission.is_symlink():
        raise ArenaError("Agent did not produce a real sealed final_submission.json")
    if not finalization.is_file() or finalization.is_symlink():
        raise ArenaError("Agent did not produce a real finalization.json")
    trusted_seal_path = episode.host / "submission_seal.json"
    if not trusted_seal_path.is_file() or trusted_seal_path.is_symlink():
        raise ArenaError("trusted submission seal is missing or linked")
    trusted_seal = read_json(trusted_seal_path)
    if trusted_seal.get("schema_version") != "sieve_trusted_submission_seal_v1":
        raise ArenaError("trusted submission seal schema differs")
    sealed_finalization = trusted_seal.get("finalization")
    if not isinstance(sealed_finalization, dict):
        raise ArenaError("trusted submission seal payload is malformed")
    workspace_finalization = read_json(finalization)
    workspace_finalization.pop("tool_counts", None)
    if workspace_finalization != sealed_finalization:
        raise ArenaError("workspace finalization differs from trusted seal")
    submission_sha256 = hashlib.sha256(final_submission.read_bytes()).hexdigest()
    if sealed_finalization.get("submission_sha256") != submission_sha256:
        raise ArenaError("final submission differs from trusted seal")
    floorplan = read_json(episode.workspace / "floorplan.json")
    room_program = read_json(episode.workspace / "room_program.json")
    submission = read_json(final_submission)
    validated = validate_agent_submission(
        submission,
        room_layout=floorplan,
        room_program=room_program,
        asset_catalog=database,
    )
    task_constraints = validate_task_submission_constraints(
        validated,
        task_payload=read_json(episode.workspace / "task.json"),
    )
    scene, preflight = materialize_agent_scene(
        room_layout=floorplan,
        room_program=room_program,
        validated=validated,
        generation_mode="non_rectangular_agent_shared_db_isolated_v1",
    )
    architecture = build_polygon_architecture(floorplan)
    outputs = {
        "received_submission.json": submission,
        "object_plan.json": validated.object_plan,
        "asset_selection.json": validated.asset_selection,
        "global_placement.json": validated.global_placement,
        "generated_scene.json": scene,
        "compiled_architecture.json": architecture,
        "evaluation_preflight.json": preflight,
        "submission_validation.json": validated.public_dict(),
        "task_constraint_validation.json": task_constraints,
    }
    for name, value in outputs.items():
        _write_host_json(episode.host / name, value)
    summary = {
        "schema_version": "sieve_isolated_agent_episode_summary_v1",
        "status": "complete",
        "scene_id": episode.case.scene_id,
        "database_snapshot_id": database.snapshot_id,
        "planned_instance_count": validated.public_dict()["planned_instance_count"],
        "room_count": validated.public_dict()["room_count"],
        "submission_sha256": submission_sha256,
        "official_evaluation_connected": False,
        "normalized_artifacts": sorted(outputs),
    }
    _write_host_json(episode.host / "summary.json", summary)
    return summary


def _write_host_json(path: Path, value: Any) -> None:
    if not isinstance(value, dict):
        raise ArenaError(f"trusted JSON root must be an object: {path.name}")
    write_json_exclusive(path, value, mode=0o444)


__all__ = ["EpisodeDatabase", "collect_and_normalize"]
