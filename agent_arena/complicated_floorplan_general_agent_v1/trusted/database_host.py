"""Trusted bridge from one isolated workspace to the real shared asset DB."""

from __future__ import annotations

from contextlib import AbstractContextManager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any

from arena import (
    ARENA_ROOT,
    ArenaError,
    Episode,
    canonical_task,
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
    AgentToolShutdownError,
    validate_task_submission_constraints,
    verify_tool_event_journal,
)
from benchmark.scene_generation.non_rectangular_multi_room.architecture import (  # noqa: E402
    build_polygon_architecture,
)


NORMALIZED_ARTIFACT_NAMES = (
    "asset_selection.json",
    "compiled_architecture.json",
    "evaluation_preflight.json",
    "generated_scene.json",
    "global_placement.json",
    "object_plan.json",
    "received_submission.json",
    "submission_validation.json",
    "task_constraint_validation.json",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def arena_tool_policy() -> AgentToolPolicy:
    arena = read_json(ARENA_ROOT / "arena.json")
    policy_raw = arena["tool_policy"]
    return AgentToolPolicy(
        max_total_calls=int(policy_raw["max_total_calls"]),
        max_asset_searches=int(policy_raw["max_asset_searches"]),
        max_asset_inspections=int(policy_raw["max_asset_inspections"]),
        max_submission_validations=int(policy_raw["max_submission_validations"]),
        max_top_k=int(policy_raw["max_top_k"]),
    )


class EpisodeDatabase(AbstractContextManager["EpisodeDatabase"]):
    """Own the real DB and one authenticated, per-episode tool service."""

    def __init__(self, *, episode: Episode, resource_bindings: str | Path) -> None:
        verify_episode_inputs(episode)
        self.episode = episode
        self.policy = arena_tool_policy()
        self.database = open_shared_asset_database(
            catalog_path=REPOSITORY_ROOT / "configs/retrieval/profiles_v2.json",
            resource_bindings_path=Path(resource_bindings).expanduser().resolve(),
            retrieval_profile_id=(
                "imaginarium-qwen3-embedding-0.6b-stable-top1-v2"
            ),
            max_top_k=self.policy.max_top_k,
        )
        expected = read_json(ARENA_ROOT / "fixed_suite/shared_database_contract.json")
        observed = self.database.public_manifest()
        for key, value in expected.items():
            if observed.get(key) != value:
                raise ArenaError(f"runtime shared DB differs from frozen contract: {key}")
        # Never serve contract data back out of an Agent-writable directory.
        floorplan = read_json(episode.case.floorplan)
        room_program = read_json(episode.case.room_program)
        task = canonical_task(episode.case)
        self.session = AgentToolSession(
            workspace=episode.workspace,
            room_layout=floorplan,
            room_program=room_program,
            asset_catalog=self.database,
            task_payload=task,
            policy=self.policy,
            audit_path=episode.host / "tool_events.jsonl",
            seal_record_path=episode.host / "submission_seal.json",
            sealed_submission_path=episode.host / "sealed_submission.json",
        )
        self.server = AgentToolServer(self.session)
        self._entered = False
        self._closed = False
        self._close_verification: dict[str, Any] | None = None

    @property
    def socket_path(self) -> Path:
        return self.server.socket_path

    @property
    def capability_token(self) -> str:
        return self.server.token

    def __enter__(self) -> "EpisodeDatabase":
        if self._entered:
            raise AgentToolShutdownError("episode database cannot be entered twice")
        self._entered = True
        self.server.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._close_verification = self.server.close()
        self._closed = True

    def verify_closed_success(self) -> dict[str, Any]:
        """Require a drained service and one successful finalize call."""

        if not self._entered or not self._closed or self._close_verification is None:
            raise AgentToolShutdownError(
                "episode database is not proven closed and drained"
            )
        observed = verify_tool_event_journal(
            self.episode.host / "tool_events.jsonl",
            policy=self.policy,
            require_finalized=True,
            expected_mode=0o444,
        )
        if observed != self._close_verification:
            raise AgentToolShutdownError("tool audit changed after server shutdown")
        path = self.episode.host / "tool_audit_verification.json"
        if path.exists() or path.is_symlink():
            _require_host_mode(path, 0o444, "stored tool audit verification")
            if read_json(path) != observed:
                raise AgentToolShutdownError("stored tool audit verification differs")
        else:
            write_json_exclusive(path, observed, mode=0o444)
        return observed


def collect_and_normalize(
    episode: Episode,
    episode_database: EpisodeDatabase,
) -> dict[str, Any]:
    """Revalidate outside the sandbox and publish evaluator-compatible artifacts."""

    verify_episode_inputs(episode)
    tool_audit = episode_database.verify_closed_success()
    database = episode_database.database
    final_submission = episode.workspace / "final_submission.json"
    finalization = episode.workspace / "finalization.json"
    if not final_submission.is_file() or final_submission.is_symlink():
        raise ArenaError("Agent did not produce a real sealed final_submission.json")
    if not finalization.is_file() or finalization.is_symlink():
        raise ArenaError("Agent did not produce a real finalization.json")
    trusted_seal_path = episode.host / "submission_seal.json"
    if not trusted_seal_path.is_file() or trusted_seal_path.is_symlink():
        raise ArenaError("trusted submission seal is missing or linked")
    _require_host_mode(trusted_seal_path, 0o400, "trusted submission seal")
    trusted_seal = read_json(trusted_seal_path)
    if set(trusted_seal) != {
        "schema_version",
        "finalization",
        "sealed_submission",
    } or trusted_seal.get("schema_version") != "sieve_trusted_submission_seal_v2":
        raise ArenaError("trusted submission seal schema differs")
    sealed_finalization = trusted_seal.get("finalization")
    if not isinstance(sealed_finalization, dict):
        raise ArenaError("trusted submission seal payload is malformed")
    workspace_finalization = read_json(finalization)
    workspace_finalization.pop("tool_counts", None)
    if workspace_finalization != sealed_finalization:
        raise ArenaError("workspace finalization differs from trusted seal")
    sealed_submission_path = episode.host / "sealed_submission.json"
    _require_host_mode(sealed_submission_path, 0o400, "host-sealed submission")
    sealed_bytes = sealed_submission_path.read_bytes()
    submission_sha256 = hashlib.sha256(sealed_bytes).hexdigest()
    if sealed_finalization.get("submission_sha256") != submission_sha256:
        raise ArenaError("host-sealed submission differs from trusted seal")
    sealed_record = trusted_seal.get("sealed_submission")
    if sealed_record != {
        "path": "sealed_submission.json",
        "sha256": submission_sha256,
        "size_bytes": len(sealed_bytes),
        "mode": "0o400",
    }:
        raise ArenaError("trusted sealed-submission record differs")
    if final_submission.read_bytes() != sealed_bytes:
        raise ArenaError("workspace final submission differs from host-sealed bytes")
    floorplan = read_json(episode.case.floorplan)
    room_program = read_json(episode.case.room_program)
    submission = _read_json_bytes(sealed_bytes, "sealed_submission.json")
    validated = validate_agent_submission(
        submission,
        room_layout=floorplan,
        room_program=room_program,
        asset_catalog=database,
    )
    task_constraints = validate_task_submission_constraints(
        validated,
        task_payload=canonical_task(episode.case),
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
    artifact_manifest = [
        _artifact_record(episode.host / name, name)
        for name in NORMALIZED_ARTIFACT_NAMES
    ]
    summary = {
        "schema_version": "sieve_isolated_agent_episode_summary_v3",
        "status": "complete",
        "scene_id": episode.case.scene_id,
        "database_snapshot_id": database.snapshot_id,
        "planned_instance_count": validated.public_dict()["planned_instance_count"],
        "room_count": validated.public_dict()["room_count"],
        "submission_sha256": submission_sha256,
        "tool_event_journal_sha256": tool_audit["journal_sha256"],
        "official_evaluation_connected": False,
        "normalized_artifacts": list(NORMALIZED_ARTIFACT_NAMES),
        "normalized_artifact_manifest": artifact_manifest,
    }
    _write_host_json(episode.host / "summary.json", summary)
    return summary


def _write_host_json(path: Path, value: Any) -> None:
    if not isinstance(value, dict):
        raise ArenaError(f"trusted JSON root must be an object: {path.name}")
    write_json_exclusive(path, value, mode=0o444)


def _require_host_mode(path: Path, expected: int, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ArenaError(f"{label} must be a real file")
    if stat.S_IMODE(path.stat().st_mode) != expected:
        raise ArenaError(f"{label} mode differs")


def _read_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {item}")
            ),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise ArenaError(f"{label} is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ArenaError(f"{label} root must be an object")
    return value


def verify_normalized_artifact_manifest(
    host_directory: str | Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Bind every evaluator-facing normalized artifact to size and SHA-256."""

    host = Path(host_directory).expanduser().absolute()
    if not host.is_dir() or host.is_symlink():
        raise ArenaError("normalized artifact host directory is invalid")
    names = summary.get("normalized_artifacts")
    manifest = summary.get("normalized_artifact_manifest")
    if names != list(NORMALIZED_ARTIFACT_NAMES):
        raise ArenaError("normalized artifact name set differs")
    if not isinstance(manifest, list) or len(manifest) != len(
        NORMALIZED_ARTIFACT_NAMES
    ):
        raise ArenaError("normalized artifact manifest is malformed")
    expected_names = list(NORMALIZED_ARTIFACT_NAMES)
    observed_names: list[str] = []
    total_bytes = 0
    for expected_name, record in zip(expected_names, manifest, strict=True):
        if not isinstance(record, dict) or record.get("path") != expected_name:
            raise ArenaError("normalized artifact manifest ordering differs")
        size = record.get("size_bytes")
        digest = record.get("sha256")
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or not isinstance(digest, str)
            or SHA256.fullmatch(digest) is None
        ):
            raise ArenaError("normalized artifact manifest record is invalid")
        path = host / expected_name
        if not path.is_file() or path.is_symlink():
            raise ArenaError("normalized artifact is missing or linked")
        file_stat = path.stat()
        if stat.S_IMODE(file_stat.st_mode) != 0o444:
            raise ArenaError("normalized artifact is not host-sealed")
        if file_stat.st_size != size or _sha256_file(path) != digest:
            raise ArenaError("normalized artifact content differs from manifest")
        observed_names.append(expected_name)
        total_bytes += size
    return {
        "schema_version": "sieve_normalized_artifact_verification_v1",
        "artifact_count": len(observed_names),
        "total_bytes": total_bytes,
        "verified": True,
    }


def _artifact_record(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ArenaError("normalized artifact is missing or linked")
    file_stat = path.stat()
    if stat.S_IMODE(file_stat.st_mode) != 0o444:
        raise ArenaError("normalized artifact is not host-sealed")
    return {
        "path": name,
        "size_bytes": file_stat.st_size,
        "sha256": _sha256_file(path),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except BaseException:
        # fdopen owns and closes the descriptor once constructed.
        raise
    return digest.hexdigest()


__all__ = [
    "EpisodeDatabase",
    "NORMALIZED_ARTIFACT_NAMES",
    "arena_tool_policy",
    "collect_and_normalize",
    "verify_normalized_artifact_manifest",
]
