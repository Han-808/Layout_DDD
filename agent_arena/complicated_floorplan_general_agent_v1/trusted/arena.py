"""Trusted, stdlib-only helpers for the standalone SIEVE Agent arena."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


ARENA_ROOT = Path(__file__).resolve().parents[1]
PORTABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
WORKSPACE_FILES = frozenset(
    {
        "TODO.md",
        "database-interface.json",
        "floorplan.json",
        "room_program.json",
        "sieve-agent-tool",
        "submission.schema.json",
        "task.json",
    }
)


class ArenaError(RuntimeError):
    """Raised when a frozen arena or episode violates its trust contract."""


@dataclass(frozen=True)
class FixedCase:
    scene_id: str
    room_count: int
    wall_segment_count: int
    target_min: int
    target_max: int
    floorplan: Path
    room_program: Path
    floorplan_sha256: str
    room_program_sha256: str


@dataclass(frozen=True)
class Episode:
    root: Path
    workspace: Path
    host: Path
    case: FixedCase


def load_arena() -> dict[str, Any]:
    value = read_json(ARENA_ROOT / "arena.json")
    if value.get("schema_version") != "sieve_general_agent_arena_v1":
        raise ArenaError("unsupported arena schema")
    if value.get("participant_class") != "general_purpose_tool_using_agent":
        raise ArenaError("arena participant class drifted")
    if value.get("track_id") != "complicated_floorplan_agent_track_v1":
        raise ArenaError("arena track identity drifted")
    selection = _mapping(value.get("entrant_selection"), "entrant_selection")
    if selection.get("status") != "deferred" or selection.get(
        "registered_entrants"
    ) != []:
        raise ArenaError("base arena must not preselect Agent entrants")
    integrity = _mapping(value.get("integrity"), "integrity")
    if integrity != {
        "current_lock": "arena.lock.v3.json",
        "predecessor_lock": "arena.lock.v2.json",
    }:
        raise ArenaError("arena integrity-chain declaration drifted")
    return value


def verify_fixed_suite() -> dict[str, Any]:
    arena = load_arena()
    suite = _mapping(arena.get("suite"), "arena.suite")
    scene_order = _text_list(suite.get("scene_order"), "arena scene order")
    if len(scene_order) != 10 or len(set(scene_order)) != 10:
        raise ArenaError("arena must contain ten unique scenes")

    proposal = read_json(ARENA_ROOT / "fixed_suite/proposal_manifest.json")
    design = _mapping(proposal.get("experimental_design"), "experimental_design")
    access = _mapping(design.get("asset_access"), "asset_access")
    if proposal.get("status") != "human_approved":
        raise ArenaError("fixed suite is not human-approved")
    if proposal.get("generation_authorized") is not True:
        raise ArenaError("fixed suite is not generation-authorized")
    if design.get("track_type") != "agent_only":
        raise ArenaError("fixed suite is not Agent-only")
    if access.get("mode") != "shared_database":
        raise ArenaError("fixed suite does not select the shared database")

    layout_manifest = read_json(ARENA_ROOT / str(suite["layout_manifest_path"]))
    program_manifest = read_json(ARENA_ROOT / str(suite["program_manifest_path"]))
    if _text_list(
        _mapping(layout_manifest.get("selection"), "layout selection").get(
            "scene_order"
        ),
        "layout scene order",
    ) != scene_order:
        raise ArenaError("layout scene order differs from arena")
    if _text_list(program_manifest.get("scene_order"), "program scene order") != scene_order:
        raise ArenaError("program scene order differs from arena")

    layout_rows = _rows_by_scene(layout_manifest.get("scenes"), "layout scenes")
    program_rows = _rows_by_scene(program_manifest.get("scenes"), "program scenes")
    cases: list[FixedCase] = []
    for scene_id in scene_order:
        layout = layout_rows.get(scene_id)
        program = program_rows.get(scene_id)
        if layout is None or program is None:
            raise ArenaError(f"fixed scene is missing: {scene_id}")
        floorplan = ARENA_ROOT / "fixed_suite/layouts" / scene_id / "room_layout.json"
        room_program = ARENA_ROOT / "fixed_suite/programs" / scene_id / "room_program.json"
        expected_layout_hash = str(layout.get("room_layout_sha256") or "")
        expected_program_hash = str(program.get("room_program_sha256") or "")
        if sha256_file(floorplan) != expected_layout_hash:
            raise ArenaError(f"FloorPlan hash drifted: {scene_id}")
        if sha256_file(room_program) != expected_program_hash:
            raise ArenaError(f"room-program hash drifted: {scene_id}")
        target = _mapping(program.get("target_total_instances"), "target range")
        cases.append(
            FixedCase(
                scene_id=scene_id,
                room_count=_positive_int(layout.get("room_count"), "room_count"),
                wall_segment_count=_positive_int(
                    layout.get("wall_segment_count"), "wall_segment_count"
                ),
                target_min=_positive_int(target.get("min"), "target min"),
                target_max=_positive_int(target.get("max"), "target max"),
                floorplan=floorplan,
                room_program=room_program,
                floorplan_sha256=expected_layout_hash,
                room_program_sha256=expected_program_hash,
            )
        )
    if sum(case.room_count for case in cases) != int(suite.get("room_count", -1)):
        raise ArenaError("fixed suite room total drifted")
    if sum(case.wall_segment_count for case in cases) != int(
        suite.get("wall_segment_count", -1)
    ):
        raise ArenaError("fixed suite wall total drifted")
    aggregate = _mapping(
        suite.get("aggregate_target_total_instances"),
        "aggregate target range",
    )
    if {
        "min": sum(case.target_min for case in cases),
        "max": sum(case.target_max for case in cases),
    } != aggregate:
        raise ArenaError("fixed suite aggregate target range drifted")

    database = read_json(ARENA_ROOT / "fixed_suite/shared_database_contract.json")
    public_database = _mapping(arena.get("database"), "arena.database")
    for key in ("snapshot_id", "snapshot_sha256", "expected_asset_count"):
        if database.get(key) != public_database.get(key):
            raise ArenaError(f"shared database identity drifted: {key}")
    if database.get("external_asset_sources_allowed") is not False:
        raise ArenaError("shared database unexpectedly allows external assets")
    return {
        "schema_version": "sieve_agent_arena_fixed_suite_verification_v1",
        "valid": True,
        "scene_count": len(cases),
        "room_count": sum(case.room_count for case in cases),
        "wall_segment_count": sum(case.wall_segment_count for case in cases),
        "aggregate_target_total_instances": {
            "min": sum(case.target_min for case in cases),
            "max": sum(case.target_max for case in cases),
        },
        "database_snapshot_id": database["snapshot_id"],
        "cases": [case.scene_id for case in cases],
    }


def fixed_case(scene_id: str) -> FixedCase:
    verify_fixed_suite()
    arena = load_arena()
    suite = _mapping(arena["suite"], "arena.suite")
    layouts = _rows_by_scene(
        read_json(ARENA_ROOT / str(suite["layout_manifest_path"]))["scenes"],
        "layout scenes",
    )
    programs = _rows_by_scene(
        read_json(ARENA_ROOT / str(suite["program_manifest_path"]))["scenes"],
        "program scenes",
    )
    if scene_id not in layouts or scene_id not in programs:
        raise ArenaError(f"unknown fixed scene: {scene_id}")
    layout = layouts[scene_id]
    program = programs[scene_id]
    target = _mapping(program["target_total_instances"], "target range")
    floorplan = ARENA_ROOT / "fixed_suite/layouts" / scene_id / "room_layout.json"
    room_program = ARENA_ROOT / "fixed_suite/programs" / scene_id / "room_program.json"
    return FixedCase(
        scene_id=scene_id,
        room_count=_positive_int(layout["room_count"], "room_count"),
        wall_segment_count=_positive_int(
            layout["wall_segment_count"], "wall_segment_count"
        ),
        target_min=_positive_int(target["min"], "target min"),
        target_max=_positive_int(target["max"], "target max"),
        floorplan=floorplan,
        room_program=room_program,
        floorplan_sha256=str(layout["room_layout_sha256"]),
        room_program_sha256=str(program["room_program_sha256"]),
    )


def create_episode(*, agent_id: str, scene_id: str, run_id: str) -> Episode:
    from verify_arena import verify_lock

    verify_lock()
    agent_id = portable_id(agent_id, "agent_id")
    scene_id = portable_id(scene_id, "scene_id")
    run_id = portable_id(run_id, "run_id")
    case = fixed_case(scene_id)
    episode_root = ARENA_ROOT / "episodes" / agent_id / scene_id / run_id
    if episode_root.exists() or episode_root.is_symlink():
        raise ArenaError(f"episode already exists: {episode_root}")
    workspace = episode_root / "workspace"
    host = episode_root / "host"
    workspace.mkdir(parents=True, mode=0o700)
    host.mkdir(mode=0o700)
    (workspace / ".home").mkdir(mode=0o700)
    (workspace / ".tmp").mkdir(mode=0o700)

    arena = load_arena()
    database = _mapping(arena["database"], "arena.database")
    replacements = {
        "{{ARENA_ID}}": str(arena["arena_id"]),
        "{{SCENE_ID}}": case.scene_id,
        "{{ROOM_COUNT}}": str(case.room_count),
        "{{WALL_SEGMENT_COUNT}}": str(case.wall_segment_count),
        "{{TARGET_MIN}}": str(case.target_min),
        "{{TARGET_MAX}}": str(case.target_max),
        "{{DATABASE_SNAPSHOT_ID}}": str(database["snapshot_id"]),
    }
    todo = (ARENA_ROOT / "TODO.md").read_text(encoding="utf-8")
    for before, after in replacements.items():
        todo = todo.replace(before, after)
    if "{{" in todo or "}}" in todo:
        raise ArenaError("TODO template contains unresolved fields")

    write_text_exclusive(workspace / "TODO.md", todo, mode=0o444)
    copy_exclusive(case.floorplan, workspace / "floorplan.json", mode=0o444)
    copy_exclusive(case.room_program, workspace / "room_program.json", mode=0o444)
    copy_exclusive(
        ARENA_ROOT / "public/database-interface.json",
        workspace / "database-interface.json",
        mode=0o444,
    )
    copy_exclusive(
        ARENA_ROOT / "public/submission.schema.json",
        workspace / "submission.schema.json",
        mode=0o444,
    )
    copy_exclusive(
        ARENA_ROOT / "public/sieve-agent-tool",
        workspace / "sieve-agent-tool",
        mode=0o555,
    )
    task = {
        "schema_version": "sieve_isolated_agent_task_v1",
        "arena_id": arena["arena_id"],
        "track_id": arena["track_id"],
        "participant_class": arena["participant_class"],
        "layout_id": case.scene_id,
        "room_count": case.room_count,
        "wall_segment_count": case.wall_segment_count,
        "target_total_instances": {"min": case.target_min, "max": case.target_max},
        "authoritative_inputs": {
            "floorplan": "floorplan.json",
            "floorplan_sha256": case.floorplan_sha256,
            "room_program": "room_program.json",
            "room_program_sha256": case.room_program_sha256,
            "architecture_mutable": False,
        },
        "asset_database": database,
        "tool_policy": arena["tool_policy"],
        "database_interface": "database-interface.json",
        "final_submission": {
            "draft_path": "submission.json",
            "seal_command": "./sieve-agent-tool finalize-submission submission.json",
            "schema": "submission.schema.json",
            "schema_version": "non_rectangular_agent_submission_v1",
        },
        "evaluation_scope": arena["evaluation"],
        "isolation": {
            "workspace_only": True,
            "network_default": "deny",
            "model_gateway_scoped": True,
        },
    }
    write_json_exclusive(workspace / "task.json", task, mode=0o444)

    observed = {
        path.name: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "mode": oct(path.stat().st_mode & 0o777),
        }
        for path in sorted(workspace.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    if set(observed) != WORKSPACE_FILES:
        raise ArenaError("materialized public workspace file set drifted")
    manifest = {
        "schema_version": "sieve_isolated_agent_episode_manifest_v1",
        "arena_id": arena["arena_id"],
        "agent_id": agent_id,
        "scene_id": scene_id,
        "run_id": run_id,
        "workspace_relative_path": "workspace",
        "workspace_files": observed,
        "host_home_mounted": False,
        "repository_mounted": False,
        "other_episodes_mounted": False,
        "network_default": "deny",
        "model_gateway_required": True,
        "database_snapshot_id": database["snapshot_id"],
    }
    write_json_exclusive(host / "episode_manifest.json", manifest, mode=0o444)
    return Episode(root=episode_root, workspace=workspace, host=host, case=case)


def verify_episode_inputs(episode: Episode) -> dict[str, Any]:
    manifest = read_json(episode.host / "episode_manifest.json")
    expected = _mapping(manifest.get("workspace_files"), "workspace_files")
    for name, metadata in expected.items():
        if name not in WORKSPACE_FILES:
            raise ArenaError(f"unexpected controlled workspace file: {name}")
        path = episode.workspace / name
        item = _mapping(metadata, f"workspace_files.{name}")
        if sha256_file(path) != item.get("sha256"):
            raise ArenaError(f"Agent changed an authoritative workspace file: {name}")
    return {"valid": True, "verified_files": len(expected)}


def portable_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not PORTABLE_ID.fullmatch(value):
        raise ArenaError(f"{label} must be a portable identifier")
    return value


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ArenaError(f"required JSON is missing or linked: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArenaError(f"cannot read JSON {path.name}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ArenaError(f"JSON root must be an object: {path.name}")
    return value


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ArenaError(f"controlled file is missing or linked: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_exclusive(path: Path, value: Mapping[str, Any], *, mode: int) -> None:
    encoded = json.dumps(
        dict(value), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    write_bytes_exclusive(path, encoded.encode("utf-8"), mode=mode)


def write_text_exclusive(path: Path, value: str, *, mode: int) -> None:
    write_bytes_exclusive(path, value.encode("utf-8"), mode=mode)


def write_bytes_exclusive(path: Path, value: bytes, *, mode: int) -> None:
    if path.exists() or path.is_symlink():
        raise ArenaError(f"refusing to overwrite episode file: {path.name}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)


def copy_exclusive(source: Path, target: Path, *, mode: int) -> None:
    if not source.is_file() or source.is_symlink():
        raise ArenaError(f"copy source is missing or linked: {source}")
    write_bytes_exclusive(target, source.read_bytes(), mode=mode)


def _rows_by_scene(value: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ArenaError(f"{label} must be an array")
    output: dict[str, dict[str, Any]] = {}
    for item in value:
        row = _mapping(item, f"{label}[]")
        scene_id = str(row.get("scene_id") or "")
        if not scene_id or scene_id in output:
            raise ArenaError(f"{label} has an invalid scene identity")
        output[scene_id] = row
    return output


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArenaError(f"{label} must be an object")
    return dict(value)


def _text_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ArenaError(f"{label} must be a non-empty string array")
    return list(value)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ArenaError(f"{label} must be a positive integer")
    return value


__all__ = [
    "ARENA_ROOT",
    "ArenaError",
    "Episode",
    "FixedCase",
    "WORKSPACE_FILES",
    "create_episode",
    "fixed_case",
    "load_arena",
    "portable_id",
    "read_json",
    "sha256_file",
    "verify_episode_inputs",
    "verify_fixed_suite",
    "write_json_exclusive",
]
